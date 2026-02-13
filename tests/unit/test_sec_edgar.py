"""Tests for SEC EDGAR provider."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.providers.sec_edgar.models import EarningsRelease, InsiderTransaction


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_CIK_MAP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
    "2": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
}

SAMPLE_SUBMISSIONS = {
    "cik": "320193",
    "entityType": "operating",
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "4", "8-K"],
            "filingDate": ["2026-02-10", "2026-01-30", "2026-01-20", "2025-12-15"],
            "accessionNumber": [
                "0000320193-26-000010",
                "0000320193-26-000009",
                "0000320193-26-000008",
                "0000320193-25-000007",
            ],
            "primaryDocument": ["doc1.htm", "doc2.htm", "xslF345X05/doc3.xml", "doc4.htm"],
            "items": ["2.02", "", "", "5.02"],
            "acceptanceDateTime": [
                "2026-02-10T16:30:00.000Z",
                "2026-01-30T08:00:00.000Z",
                "2026-01-20T10:15:00.000Z",
                "2025-12-15T12:00:00.000Z",
            ],
        }
    },
}

SAMPLE_FORM4_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>John Doe</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate>
        <value>2026-01-15</value>
      </transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares>
          <value>5000</value>
        </transactionShares>
        <transactionPricePerShare>
          <value>185.50</value>
        </transactionPricePerShare>
        <transactionAcquiredDisposedCode>
          <value>D</value>
        </transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction>
          <value>45000</value>
        </sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate>
        <value>2026-01-10</value>
      </transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares>
          <value>1000</value>
        </transactionShares>
        <transactionPricePerShare>
          <value>180.00</value>
        </transactionPricePerShare>
        <transactionAcquiredDisposedCode>
          <value>A</value>
        </transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction>
          <value>50000</value>
        </sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


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
# CIK Mapping
# ---------------------------------------------------------------------------


class TestCIKMapping:
    async def test_load_cik_mapping(self, client: SECEdgarClient, mock_redis):
        """Test CIK mapping is loaded and cached."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps(SAMPLE_CIK_MAP)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            cik_map = await client._load_cik_mapping()

        assert "AAPL" in cik_map
        assert cik_map["AAPL"] == "0000320193"
        assert cik_map["MSFT"] == "0000789019"
        # Verify Redis cache was set
        mock_redis.set.assert_called()

    async def test_cik_mapping_from_cache(self, client: SECEdgarClient, mock_redis):
        """Test CIK mapping loaded from Redis cache."""
        cached_map = {"AAPL": "0000320193", "MSFT": "0000789019"}
        mock_redis.get.return_value = orjson.dumps(cached_map)

        cik_map = await client._load_cik_mapping()
        assert cik_map == cached_map

    async def test_get_cik_known_ticker(self, client: SECEdgarClient, mock_redis):
        """Test CIK lookup for a known ticker."""
        client._cik_map = {"AAPL": "0000320193", "TSLA": "0001318605"}
        cik = await client._get_cik("AAPL")
        assert cik == "0000320193"

    async def test_get_cik_unknown_ticker(self, client: SECEdgarClient, mock_redis):
        """Test CIK lookup for unknown ticker returns None."""
        cached_map = {"AAPL": "0000320193"}
        mock_redis.get.return_value = orjson.dumps(cached_map)

        cik = await client._get_cik("ZZZZ")
        assert cik is None


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------


class TestGetFilings:
    async def test_get_filings(self, client: SECEdgarClient, mock_redis):
        """Test fetching filings from SEC API."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_SUBMISSIONS)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            filings = await client.get_filings("AAPL", limit=10)

        assert len(filings) == 4
        assert filings[0].form == "8-K"
        assert filings[0].ticker == "AAPL"
        assert filings[0].filed_date == date(2026, 2, 10)
        assert filings[0].items == "2.02"

    async def test_get_filings_filtered(self, client: SECEdgarClient, mock_redis):
        """Test filings filtered by form type."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_SUBMISSIONS)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            filings = await client.get_filings("AAPL", form_types=["8-K"])

        assert len(filings) == 2
        assert all(f.form == "8-K" for f in filings)

    async def test_get_filings_from_cache(self, client: SECEdgarClient, mock_redis):
        """Test filings loaded from cache."""
        cached_filings = [
            {
                "ticker": "AAPL",
                "form": "10-K",
                "filed_date": "2026-01-15",
                "accepted_datetime": "2026-01-15T08:00:00",
                "accession_number": "0000320193-26-000001",
                "primary_document": "doc.htm",
                "items": "",
                "url": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019326000001/doc.htm",
            }
        ]
        mock_redis.get.return_value = orjson.dumps(cached_filings)
        client._cik_map = {"AAPL": "0000320193"}

        filings = await client.get_filings("AAPL")
        assert len(filings) == 1
        assert filings[0].form == "10-K"

    async def test_get_filings_unknown_ticker(self, client: SECEdgarClient, mock_redis):
        """Test filings for unknown ticker returns empty."""
        cached_map = {"AAPL": "0000320193"}
        mock_redis.get.side_effect = [orjson.dumps(cached_map), None]

        filings = await client.get_filings("ZZZZ")
        assert filings == []


# ---------------------------------------------------------------------------
# Form 4 XML Parsing
# ---------------------------------------------------------------------------


class TestForm4Parsing:
    def test_parse_form4_xml(self):
        """Test Form 4 XML parsing produces correct transactions."""
        transactions = SECEdgarClient._parse_form4_xml(
            SAMPLE_FORM4_XML,
            ticker="AAPL",
            filing_date=date(2026, 1, 20),
            filing_url="https://www.sec.gov/example",
        )

        assert len(transactions) == 2

        # First transaction: sale
        sell = transactions[0]
        assert sell.owner_name == "John Doe"
        assert sell.owner_relationship == "Officer (CFO)"
        assert sell.transaction_code == "S"
        assert sell.shares == 5000.0
        assert sell.price_per_share == 185.50
        assert sell.shares_after == 45000.0
        assert sell.acquired_or_disposed == "D"
        assert sell.transaction_date == date(2026, 1, 15)

        # Second transaction: purchase
        buy = transactions[1]
        assert buy.transaction_code == "P"
        assert buy.shares == 1000.0
        assert buy.price_per_share == 180.00
        assert buy.acquired_or_disposed == "A"

    def test_parse_form4_xml_invalid(self):
        """Test Form 4 parsing with invalid XML returns empty."""
        result = SECEdgarClient._parse_form4_xml(
            "not xml at all",
            ticker="AAPL",
            filing_date=date(2026, 1, 1),
            filing_url="https://example.com",
        )
        assert result == []

    def test_parse_form4_xml_with_namespace(self):
        """Test Form 4 XML parsing with namespace prefix."""
        xml_ns = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument xmlns="http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany">
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Jane Smith</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>CTO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-01-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2000</value></transactionShares>
        <transactionPricePerShare><value>190.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>10000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        txns = SECEdgarClient._parse_form4_xml(
            xml_ns, ticker="AAPL", filing_date=date(2026, 1, 20), filing_url="https://example.com"
        )
        assert len(txns) == 1
        assert txns[0].owner_name == "Jane Smith"
        assert txns[0].owner_relationship == "Officer (CTO)"
        assert txns[0].transaction_code == "P"

    def test_parse_form4_xml_missing_transaction_amounts(self):
        """Test that transaction without transactionAmounts is skipped."""
        xml_no_amounts = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Bob Jones</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-01-15</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <!-- No transactionAmounts element -->
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>5000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        txns = SECEdgarClient._parse_form4_xml(
            xml_no_amounts,
            ticker="AAPL",
            filing_date=date(2026, 1, 20),
            filing_url="https://example.com",
        )
        assert len(txns) == 0

    def test_parse_form4_xml_director_relationship(self):
        """Test director relationship is correctly identified."""
        xml_director = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Director Person</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>0</isOfficer>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-01-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>200.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>1500</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        txns = SECEdgarClient._parse_form4_xml(
            xml_director,
            ticker="AAPL",
            filing_date=date(2026, 1, 20),
            filing_url="https://example.com",
        )
        assert len(txns) == 1
        assert txns[0].owner_relationship == "Director"

    def test_parse_form4_xml_ten_percent_owner(self):
        """Test 10% owner relationship is correctly identified."""
        xml_owner = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Big Fund LLC</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>0</isOfficer>
      <isTenPercentOwner>1</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-01-15</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100000</value></transactionShares>
        <transactionPricePerShare><value>195.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>900000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        txns = SECEdgarClient._parse_form4_xml(
            xml_owner,
            ticker="AAPL",
            filing_date=date(2026, 1, 20),
            filing_url="https://example.com",
        )
        assert len(txns) == 1
        assert txns[0].owner_relationship == "10% Owner"
        assert txns[0].owner_name == "Big Fund LLC"


# ---------------------------------------------------------------------------
# Insider Sentiment
# ---------------------------------------------------------------------------


class TestInsiderSentiment:
    async def test_sentiment_with_transactions(self, client: SECEdgarClient, mock_redis):
        """Test insider sentiment computation."""
        # Mock get_insider_transactions to return controlled data
        with patch.object(client, "get_insider_transactions") as mock_txns:
            mock_txns.return_value = [
                InsiderTransaction(
                    ticker="AAPL",
                    owner_name="Buyer",
                    owner_relationship="Director",
                    transaction_date=date(2026, 1, 15),
                    transaction_code="P",
                    shares=1000,
                    price_per_share=180.0,
                    shares_after=5000,
                    acquired_or_disposed="A",
                    filing_date=date(2026, 1, 16),
                    filing_url="https://example.com",
                ),
                InsiderTransaction(
                    ticker="AAPL",
                    owner_name="Seller",
                    owner_relationship="Officer (CEO)",
                    transaction_date=date(2026, 1, 10),
                    transaction_code="S",
                    shares=500,
                    price_per_share=185.0,
                    shares_after=10000,
                    acquired_or_disposed="D",
                    filing_date=date(2026, 1, 11),
                    filing_url="https://example.com",
                ),
            ]

            sentiment = await client.get_insider_sentiment("AAPL")

        assert sentiment is not None
        assert sentiment["ticker"] == "AAPL"
        assert sentiment["buy_count"] == 1
        assert sentiment["sell_count"] == 1
        # Buy: 1000 * 180 = 180,000; Sell: 500 * 185 = 92,500
        # MSPR = (180000 - 92500) / (180000 + 92500) ≈ 0.3211
        assert sentiment["mspr"] == pytest.approx(0.3211, abs=0.001)

    async def test_sentiment_no_transactions(self, client: SECEdgarClient, mock_redis):
        """Test sentiment returns None when no transactions."""
        with patch.object(client, "get_insider_transactions") as mock_txns:
            mock_txns.return_value = []
            sentiment = await client.get_insider_sentiment("AAPL")

        assert sentiment is None


# ---------------------------------------------------------------------------
# Historical EPS (XBRL)
# ---------------------------------------------------------------------------

SAMPLE_XBRL_EPS = {
    "cik": 320193,
    "taxonomy": "us-gaap",
    "tag": "EarningsPerShareBasic",
    "units": {
        "USD/shares": [
            {
                "end": "2025-12-31",
                "val": 2.40,
                "filed": "2026-01-30",
                "form": "10-Q",
                "frame": "CY2025Q4",
            },
            {
                "end": "2025-09-30",
                "val": 2.18,
                "filed": "2025-10-28",
                "form": "10-Q",
                "frame": "CY2025Q3",
            },
            {
                "end": "2025-06-30",
                "val": 1.40,
                "filed": "2025-07-30",
                "form": "10-K",
                "frame": "CY2025Q2",
            },
            {
                "end": "2025-03-31",
                "val": 1.52,
                "filed": "2025-04-28",
                "form": "10-Q",
                "frame": "CY2025Q1",
            },
            {
                "end": "2024-12-31",
                "val": 2.10,
                "filed": "2025-01-28",
                "form": "10-Q",
                "frame": "CY2024Q4",
            },
            # Annual entry (no Q) should be filtered out
            {
                "end": "2024-12-31",
                "val": 7.20,
                "filed": "2025-02-15",
                "form": "10-K",
                "frame": "CY2024",
            },
        ]
    },
}

SAMPLE_XBRL_REVENUE = {
    "cik": 320193,
    "taxonomy": "us-gaap",
    "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "units": {
        "USD": [
            {
                "end": "2025-12-31",
                "val": 94_836_000_000,
                "filed": "2026-01-30",
                "form": "10-Q",
                "frame": "CY2025Q4",
            },
            {
                "end": "2025-09-30",
                "val": 85_777_000_000,
                "filed": "2025-10-28",
                "form": "10-Q",
                "frame": "CY2025Q3",
            },
            # Annual entry should be filtered out
            {
                "end": "2024-12-31",
                "val": 350_000_000_000,
                "filed": "2025-02-15",
                "form": "10-K",
                "frame": "CY2024",
            },
        ]
    },
}


class TestHistoricalEPS:
    async def test_get_historical_eps(self, client: SECEdgarClient, mock_redis):
        """Test fetching historical EPS from XBRL API."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_XBRL_EPS)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            eps = await client.get_historical_eps("AAPL", limit=4)

        assert len(eps) == 4
        # Should be sorted descending by period
        assert eps[0]["period"] == "2025-12-31"
        assert eps[0]["actual"] == 2.40
        assert eps[0]["frame"] == "CY2025Q4"
        assert eps[3]["period"] == "2025-03-31"
        # Annual entry (CY2024) should be excluded
        assert all("Q" in e["frame"] for e in eps)

    async def test_get_historical_eps_cached(self, client: SECEdgarClient, mock_redis):
        """Test EPS is returned from cache without HTTP call."""
        cached = [
            {
                "period": "2025-12-31",
                "actual": 2.40,
                "filed": "2026-01-30",
                "form": "10-Q",
                "frame": "CY2025Q4",
            },
            {
                "period": "2025-09-30",
                "actual": 2.18,
                "filed": "2025-10-28",
                "form": "10-Q",
                "frame": "CY2025Q3",
            },
        ]
        mock_redis.get.return_value = orjson.dumps(cached)
        client._cik_map = {"AAPL": "0000320193"}

        eps = await client.get_historical_eps("AAPL", limit=4)
        assert len(eps) == 2
        assert eps[0]["actual"] == 2.40

    async def test_get_historical_eps_no_cik(self, client: SECEdgarClient, mock_redis):
        """Test unknown ticker returns empty list."""
        cached_map = {"AAPL": "0000320193"}
        mock_redis.get.side_effect = [orjson.dumps(cached_map), None]

        eps = await client.get_historical_eps("ZZZZ")
        assert eps == []

    async def test_get_historical_eps_404(self, client: SECEdgarClient, mock_redis):
        """Test 404 from SEC returns empty list."""
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
            eps = await client.get_historical_eps("AAPL")

        assert eps == []


# ---------------------------------------------------------------------------
# Historical Revenue (XBRL)
# ---------------------------------------------------------------------------


class TestHistoricalRevenue:
    async def test_get_historical_revenue(self, client: SECEdgarClient, mock_redis):
        """Test fetching historical revenue from XBRL API."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_XBRL_REVENUE)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            rev = await client.get_historical_revenue("AAPL", limit=4)

        assert len(rev) == 2
        assert rev[0]["period"] == "2025-12-31"
        assert rev[0]["actual"] == 94_836_000_000
        # Annual entry should be excluded
        assert all("Q" in e["frame"] for e in rev)

    async def test_get_historical_revenue_fallback(self, client: SECEdgarClient, mock_redis):
        """Test revenue falls back to Revenues concept when primary returns empty."""
        client._cik_map = {"AAPL": "0000320193"}

        # First call returns no quarterly data (empty units)
        empty_response = MagicMock()
        empty_response.content = orjson.dumps({"units": {"USD": []}})
        empty_response.raise_for_status = MagicMock()

        # Second call (fallback) returns actual data
        fallback_data = {
            "units": {
                "USD": [
                    {
                        "end": "2025-12-31",
                        "val": 94_836_000_000,
                        "filed": "2026-01-30",
                        "form": "10-Q",
                        "frame": "CY2025Q4",
                    },
                ]
            }
        }
        fallback_response = MagicMock()
        fallback_response.content = orjson.dumps(fallback_data)
        fallback_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=[empty_response, fallback_response])
            # Cache miss for both calls
            mock_redis.get.return_value = None
            rev = await client.get_historical_revenue("AAPL", limit=4)

        assert len(rev) == 1
        assert rev[0]["actual"] == 94_836_000_000


# ---------------------------------------------------------------------------
# HTML to Text (static helper)
# ---------------------------------------------------------------------------


class TestHtmlToText:
    def test_strips_tags(self):
        html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"
        result = SECEdgarClient._html_to_text(html)
        assert "Title" in result
        assert "Hello world" in result
        assert "<" not in result

    def test_collapses_whitespace(self):
        html = "<p>  lots   of    spaces  </p>"
        result = SECEdgarClient._html_to_text(html)
        assert result == "lots of spaces"

    def test_empty_html(self):
        assert SECEdgarClient._html_to_text("") == ""

    def test_plain_text_passthrough(self):
        assert SECEdgarClient._html_to_text("no tags here") == "no tags here"


# ---------------------------------------------------------------------------
# Filing Content
# ---------------------------------------------------------------------------

SAMPLE_FILING_HTML = """
<html><body>
<h1>Apple Inc. Reports Q1 2026 Results</h1>
<p>Revenue of $95B, up 8% YoY</p>
<table><tr><td>EPS</td><td>$2.40</td></tr></table>
</body></html>
"""


class TestFilingContent:
    async def test_content_via_crawler(self, client: SECEdgarClient, mock_redis):
        """Test filing content fetched via Crawl4AI."""
        mock_crawler = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Apple Q1 Results\nRevenue $95B"
        mock_crawler.crawl_sec_filing = AsyncMock(return_value=mock_result)

        content = await client.get_filing_content(
            "https://sec.gov/filing.htm", crawler=mock_crawler
        )

        assert content == "# Apple Q1 Results\nRevenue $95B"
        mock_crawler.crawl_sec_filing.assert_awaited_once()
        # Verify cached
        mock_redis.set.assert_called()

    async def test_content_fallback_html(self, client: SECEdgarClient, mock_redis):
        """Test fallback to HTML stripping when crawler fails."""
        mock_crawler = AsyncMock()
        mock_crawler.crawl_sec_filing = AsyncMock(side_effect=Exception("crawl4ai down"))

        mock_response = MagicMock()
        mock_response.text = SAMPLE_FILING_HTML
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            content = await client.get_filing_content(
                "https://sec.gov/filing.htm", crawler=mock_crawler
            )

        assert content is not None
        assert "Apple Inc." in content
        assert "Revenue" in content
        assert "<" not in content

    async def test_content_no_crawler(self, client: SECEdgarClient, mock_redis):
        """Test content fetched without crawler (HTML fallback only)."""
        mock_response = MagicMock()
        mock_response.text = "<p>Earnings report</p>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            content = await client.get_filing_content("https://sec.gov/filing.htm")

        assert content == "Earnings report"

    async def test_content_from_cache(self, client: SECEdgarClient, mock_redis):
        """Test content returned from Redis cache."""
        mock_redis.get.return_value = b"cached markdown content"

        content = await client.get_filing_content("https://sec.gov/filing.htm")
        assert content == "cached markdown content"

    async def test_content_fetch_failure(self, client: SECEdgarClient, mock_redis):
        """Test content returns None when both crawler and HTTP fail."""
        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=Exception("network error"))
            content = await client.get_filing_content("https://sec.gov/filing.htm")

        assert content is None


# ---------------------------------------------------------------------------
# Earnings Releases
# ---------------------------------------------------------------------------


class TestEarningsReleases:
    async def test_get_earnings_releases(self, client: SECEdgarClient, mock_redis):
        """Test earnings releases filters to Item 2.02 and populates content."""
        with (
            patch.object(client, "get_filings") as mock_filings,
            patch.object(client, "get_filing_content") as mock_content,
        ):
            mock_filings.return_value = [
                MagicMock(
                    ticker="AAPL",
                    form="8-K",
                    filed_date=date(2026, 2, 10),
                    accepted_datetime="2026-02-10T16:30:00",
                    accession_number="0000320193-26-000010",
                    url="https://sec.gov/filing1.htm",
                    items="2.02",
                ),
                MagicMock(
                    ticker="AAPL",
                    form="8-K",
                    filed_date=date(2025, 12, 15),
                    accepted_datetime="2025-12-15T12:00:00",
                    accession_number="0000320193-25-000007",
                    url="https://sec.gov/filing2.htm",
                    items="5.02",  # Not earnings
                ),
                MagicMock(
                    ticker="AAPL",
                    form="8-K",
                    filed_date=date(2025, 10, 28),
                    accepted_datetime="2025-10-28T16:30:00",
                    accession_number="0000320193-25-000005",
                    url="https://sec.gov/filing3.htm",
                    items="2.02",
                ),
            ]
            mock_content.return_value = "# Earnings Press Release"

            releases = await client.get_earnings_releases("AAPL", limit=5)

        assert len(releases) == 2
        assert all(isinstance(r, EarningsRelease) for r in releases)
        assert all("2.02" in r.items for r in releases)
        assert all(r.content == "# Earnings Press Release" for r in releases)
        assert mock_content.await_count == 2

    async def test_get_earnings_releases_empty(self, client: SECEdgarClient, mock_redis):
        """Test returns empty when no Item 2.02 filings exist."""
        with patch.object(client, "get_filings") as mock_filings:
            mock_filings.return_value = [
                MagicMock(items="5.02", form="8-K"),
            ]
            releases = await client.get_earnings_releases("AAPL")

        assert releases == []

    async def test_get_earnings_releases_respects_limit(self, client: SECEdgarClient, mock_redis):
        """Test limit parameter is respected."""
        with (
            patch.object(client, "get_filings") as mock_filings,
            patch.object(client, "get_filing_content") as mock_content,
        ):
            mock_filings.return_value = [
                MagicMock(
                    ticker="AAPL",
                    form="8-K",
                    filed_date=date(2026, 2, 10),
                    accepted_datetime="2026-02-10T16:30:00",
                    accession_number=f"acc-{i}",
                    url=f"https://sec.gov/filing{i}.htm",
                    items="2.02",
                )
                for i in range(10)
            ]
            mock_content.return_value = "content"

            releases = await client.get_earnings_releases("AAPL", limit=3)

        assert len(releases) == 3
        assert mock_content.await_count == 3
