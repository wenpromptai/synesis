"""Tests for SEC EDGAR XBRL mixin — filter_and_limit_facts, company facts, frames."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from synesis.providers.sec_edgar._xbrl import _filter_and_limit_facts
from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.providers.sec_edgar.models import (
    CompanyFacts,
    XBRLFact,
    XBRLFrame,
    XBRLFrameEntry,
)


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------


def _make_fact(concept: str, period_end: str, value: float) -> XBRLFact:
    """Helper to create XBRLFact with minimal boilerplate."""
    return XBRLFact(
        concept=concept,
        label=concept,
        unit="USD",
        period_end=date.fromisoformat(period_end),
        value=value,
        form="10-Q",
        frame=f"CY{period_end[:4]}Q1",
        filed=date.fromisoformat(period_end),
    )


SAMPLE_COMPANY_FACTS_JSON = {
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {
                "label": "Net Income (Loss)",
                "units": {
                    "USD": [
                        {
                            "end": "2025-12-31",
                            "val": 30000000000,
                            "filed": "2026-01-30",
                            "form": "10-Q",
                            "frame": "CY2025Q4",
                        },
                        {
                            "end": "2025-09-30",
                            "val": 25000000000,
                            "filed": "2025-10-28",
                            "form": "10-Q",
                            "frame": "CY2025Q3",
                        },
                    ]
                },
            },
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        {
                            "end": "2025-12-31",
                            "val": 95000000000,
                            "filed": "2026-01-30",
                            "form": "10-Q",
                            "frame": "CY2025Q4",
                        },
                    ]
                },
            },
        }
    },
}

SAMPLE_XBRL_FRAME_JSON = {
    "taxonomy": "us-gaap",
    "tag": "NetIncomeLoss",
    "ccp": "CY2025Q3I",
    "uom": "USD",
    "data": [
        {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "val": 25000000000,
            "accn": "0000320193-25-000010",
            "end": "2025-09-30",
        },
        {
            "cik": 789019,
            "entityName": "Microsoft Corporation",
            "val": 22000000000,
            "accn": "0000789019-25-000020",
            "end": "2025-09-30",
        },
        {
            "cik": 1652044,
            "entityName": "Alphabet Inc.",
            "val": -5000000000,
            "accn": "0001652044-25-000030",
            "end": "2025-09-30",
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_redis():
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    return redis


@pytest.fixture()
def client(mock_redis):
    return SECEdgarClient(redis=mock_redis)


# ---------------------------------------------------------------------------
# _filter_and_limit_facts
# ---------------------------------------------------------------------------


class TestFilterAndLimitFacts:
    def test_alias_expansion(self):
        """Requesting 'Revenues' also returns ASC 606 alias tag."""
        facts = [
            _make_fact("Revenues", "2024-12-31", 90e9),
            _make_fact("RevenueFromContractWithCustomerExcludingAssessedTax", "2025-03-31", 95e9),
            _make_fact("NetIncomeLoss", "2025-03-31", 20e9),
        ]
        result = _filter_and_limit_facts(facts, concepts=["Revenues"], limit=10)

        concept_names = {f.concept for f in result}
        assert "Revenues" in concept_names
        assert "RevenueFromContractWithCustomerExcludingAssessedTax" in concept_names
        # NetIncomeLoss should be excluded
        assert "NetIncomeLoss" not in concept_names

    def test_per_concept_limit(self):
        """Limit=2 returns at most 2 facts per concept."""
        facts = [
            _make_fact("NetIncomeLoss", "2025-12-31", 30e9),
            _make_fact("NetIncomeLoss", "2025-09-30", 25e9),
            _make_fact("NetIncomeLoss", "2025-06-30", 22e9),
            _make_fact("Revenues", "2025-12-31", 95e9),
            _make_fact("Revenues", "2025-09-30", 85e9),
            _make_fact("Revenues", "2025-06-30", 80e9),
        ]
        result = _filter_and_limit_facts(facts, concepts=None, limit=2)

        # 2 per concept => 4 total
        from collections import Counter

        counts = Counter(f.concept for f in result)
        assert counts["NetIncomeLoss"] == 2
        assert counts["Revenues"] == 2

    def test_empty_facts_list(self):
        """Empty facts list returns empty result."""
        result = _filter_and_limit_facts([], concepts=["Revenues"], limit=10)
        assert result == []

    def test_no_concepts_filter_returns_all(self):
        """When concepts is None, all facts are returned (subject to per-concept limit)."""
        facts = [
            _make_fact("NetIncomeLoss", "2025-12-31", 30e9),
            _make_fact("Revenues", "2025-12-31", 95e9),
            _make_fact("Assets", "2025-12-31", 400e9),
        ]
        result = _filter_and_limit_facts(facts, concepts=None, limit=10)

        concept_names = {f.concept for f in result}
        assert concept_names == {"NetIncomeLoss", "Revenues", "Assets"}

    def test_results_sorted_by_period_descending(self):
        """Result is sorted by period_end descending."""
        facts = [
            _make_fact("NetIncomeLoss", "2024-06-30", 18e9),
            _make_fact("NetIncomeLoss", "2025-12-31", 30e9),
            _make_fact("NetIncomeLoss", "2025-06-30", 22e9),
        ]
        result = _filter_and_limit_facts(facts, concepts=None, limit=10)

        dates = [f.period_end for f in result]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# get_company_facts
# ---------------------------------------------------------------------------


class TestCompanyFacts:
    async def test_happy_path(self, client: SECEdgarClient, mock_redis):
        """Test get_company_facts returns parsed CompanyFacts from HTTP response."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_COMPANY_FACTS_JSON)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            result = await client.get_company_facts("AAPL")

        assert result is not None
        assert isinstance(result, CompanyFacts)
        assert result.ticker == "AAPL"
        assert result.cik == "0000320193"
        assert result.entity_name == "Apple Inc."
        assert len(result.facts) > 0
        # Should have both concepts
        concept_names = {f.concept for f in result.facts}
        assert "NetIncomeLoss" in concept_names
        assert "Revenues" in concept_names
        # Cache should be set
        mock_redis.set.assert_called()

    async def test_cache_hit_with_concept_filtering(self, client: SECEdgarClient, mock_redis):
        """Test cached CompanyFacts is filtered by requested concepts."""
        client._cik_map = {"AAPL": "0000320193"}

        # Prepare cached CompanyFacts with multiple concepts
        cached_facts = CompanyFacts(
            ticker="AAPL",
            cik="0000320193",
            entity_name="Apple Inc.",
            facts=[
                _make_fact("NetIncomeLoss", "2025-12-31", 30e9),
                _make_fact("Revenues", "2025-12-31", 95e9),
                _make_fact("Assets", "2025-12-31", 400e9),
            ],
            concept_count=3,
        )
        mock_redis.get.return_value = orjson.dumps(cached_facts.model_dump(mode="json"))

        result = await client.get_company_facts("AAPL", concepts=["NetIncomeLoss"])

        assert result is not None
        # Only NetIncomeLoss should be in result
        concept_names = {f.concept for f in result.facts}
        assert concept_names == {"NetIncomeLoss"}

    async def test_404_returns_none(self, client: SECEdgarClient, mock_redis):
        """Test 404 from SEC returns None."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock(
            side_effect=__import__("httpx").HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )
        )

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            result = await client.get_company_facts("AAPL")

        assert result is None

    async def test_no_cik_returns_none(self, client: SECEdgarClient, mock_redis):
        """Test unknown ticker returns None."""
        cached_map = {"AAPL": "0000320193"}
        mock_redis.get.side_effect = [orjson.dumps(cached_map), None]

        result = await client.get_company_facts("ZZZZ")
        assert result is None


# ---------------------------------------------------------------------------
# get_xbrl_frame
# ---------------------------------------------------------------------------


class TestXBRLFrame:
    async def test_happy_path(self, client: SECEdgarClient, mock_redis):
        """Test get_xbrl_frame returns parsed frame, sorted by abs value, limited."""
        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_XBRL_FRAME_JSON)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            result = await client.get_xbrl_frame(
                taxonomy="us-gaap",
                tag="NetIncomeLoss",
                unit="USD",
                period="CY2025Q3I",
                limit=2,
            )

        assert result is not None
        assert isinstance(result, XBRLFrame)
        assert result.taxonomy == "us-gaap"
        assert result.tag == "NetIncomeLoss"
        assert result.unit == "USD"
        assert result.period == "CY2025Q3I"
        # Limit applied
        assert len(result.entries) == 2
        # Sorted by abs(value) descending — Apple (25B) > MSFT (22B) > Alphabet (-5B abs)
        assert result.entries[0].entity_name == "Apple Inc."
        assert result.entries[1].entity_name == "Microsoft Corporation"
        # entry_count reflects total before limit
        assert result.entry_count == 3
        # Cache should be set
        mock_redis.set.assert_called()

    async def test_404_returns_none(self, client: SECEdgarClient, mock_redis):
        """Test 404 from SEC returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock(
            side_effect=__import__("httpx").HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )
        )

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            result = await client.get_xbrl_frame(
                taxonomy="us-gaap",
                tag="NonExistent",
                unit="USD",
                period="CY2025Q3I",
            )

        assert result is None

    async def test_cache_hit(self, client: SECEdgarClient, mock_redis):
        """Test frame returned from cache with limit applied."""
        cached_frame = XBRLFrame(
            taxonomy="us-gaap",
            tag="NetIncomeLoss",
            unit="USD",
            period="CY2025Q3I",
            entries=[
                XBRLFrameEntry(
                    cik=320193,
                    entity_name="Apple",
                    value=25e9,
                    accession_number="a1",
                    end="2025-09-30",
                ),
                XBRLFrameEntry(
                    cik=789019,
                    entity_name="MSFT",
                    value=22e9,
                    accession_number="a2",
                    end="2025-09-30",
                ),
                XBRLFrameEntry(
                    cik=1652044,
                    entity_name="GOOG",
                    value=18e9,
                    accession_number="a3",
                    end="2025-09-30",
                ),
            ],
            entry_count=3,
        )
        mock_redis.get.return_value = orjson.dumps(cached_frame.model_dump(mode="json"))

        result = await client.get_xbrl_frame(
            taxonomy="us-gaap",
            tag="NetIncomeLoss",
            unit="USD",
            period="CY2025Q3I",
            limit=1,
        )

        assert result is not None
        assert len(result.entries) == 1
        assert result.entries[0].entity_name == "Apple"
