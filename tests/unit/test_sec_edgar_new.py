"""Tests for SEC EDGAR new features — derivatives, Form 3 holdings, owner country, 8-K events."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.providers.sec_edgar.models import (
    DerivativeTransaction,
    EventFiling8K,
    InsiderTransaction,
)


# ---------------------------------------------------------------------------
# Sample XML Data
# ---------------------------------------------------------------------------

SAMPLE_DERIVATIVE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Jane Smith</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Stock Option (Right to Buy)</value></securityTitle>
      <transactionDate><value>2026-01-15</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <conversionOrExercisePrice><value>150.00</value></conversionOrExercisePrice>
      <underlyingSecurity>
        <underlyingSecurityShares>10000</underlyingSecurityShares>
      </underlyingSecurity>
      <expirationDate><value>2028-06-30</value></expirationDate>
      <transactionAmounts>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""

SAMPLE_DERIVATIVE_NO_EXERCISE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Bob Jones</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Restricted Stock Unit</value></securityTitle>
      <transactionDate><value>2026-02-01</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <!-- No conversionOrExercisePrice element -->
      <underlyingSecurity>
        <underlyingSecurityShares>5000</underlyingSecurityShares>
      </underlyingSecurity>
      <!-- No expirationDate element -->
      <transactionAmounts>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""

SAMPLE_FORM3_HOLDING_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>New Director</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeHolding>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction>
          <value>25000</value>
        </sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeHolding>
    <nonDerivativeHolding>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction>
          <value>0</value>
        </sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeHolding>
  </nonDerivativeTable>
</ownershipDocument>"""

SAMPLE_FORM4_WITH_COUNTRY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Foreign Insider</rptOwnerName>
      <rptOwnerCountry>GB</rptOwnerCountry>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>VP International</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-03-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>2000</value></transactionShares>
        <transactionPricePerShare><value>200.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>12000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

# Submissions data for 8-K event tests
SAMPLE_8K_SUBMISSIONS = {
    "cik": "320193",
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["8-K", "8-K", "8-K", "10-Q"],
            "filingDate": ["2026-02-10", "2026-01-20", "2025-12-15", "2025-11-01"],
            "accessionNumber": [
                "0000320193-26-000010",
                "0000320193-26-000008",
                "0000320193-25-000007",
                "0000320193-25-000006",
            ],
            "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm", "doc4.htm"],
            "items": ["2.02,9.01", "5.02", "", ""],
            "acceptanceDateTime": [
                "2026-02-10T16:30:00.000Z",
                "2026-01-20T10:00:00.000Z",
                "2025-12-15T12:00:00.000Z",
                "2025-11-01T08:00:00.000Z",
            ],
            "reportDate": ["", "", "", ""],
        }
    },
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
# Derivative Transaction Parsing
# ---------------------------------------------------------------------------


class TestDerivativeTransactionParsing:
    def test_valid_derivative_xml(self):
        """Valid derivative XML extracts exercise price, expiration, underlying shares."""
        txns = SECEdgarClient._parse_derivative_xml(
            SAMPLE_DERIVATIVE_XML,
            ticker="AAPL",
            filing_date=date(2026, 1, 20),
            filing_url="https://www.sec.gov/example",
        )

        assert len(txns) == 1
        txn = txns[0]
        assert isinstance(txn, DerivativeTransaction)
        assert txn.ticker == "AAPL"
        assert txn.owner_name == "Jane Smith"
        assert txn.owner_relationship == "Officer (CEO)"
        assert txn.transaction_code == "M"
        assert txn.security_title == "Stock Option (Right to Buy)"
        assert txn.exercise_price == 150.00
        assert txn.underlying_shares == 10000.0
        assert txn.expiration_date == date(2028, 6, 30)
        assert txn.acquired_or_disposed == "A"
        assert txn.transaction_date == date(2026, 1, 15)
        assert txn.filing_date == date(2026, 1, 20)

    def test_missing_exercise_price(self):
        """Derivative with no exercise price element returns None for that field."""
        txns = SECEdgarClient._parse_derivative_xml(
            SAMPLE_DERIVATIVE_NO_EXERCISE_XML,
            ticker="AAPL",
            filing_date=date(2026, 2, 5),
            filing_url="https://www.sec.gov/example",
        )

        assert len(txns) == 1
        txn = txns[0]
        assert txn.exercise_price is None
        assert txn.expiration_date is None
        assert txn.security_title == "Restricted Stock Unit"
        assert txn.underlying_shares == 5000.0
        assert txn.owner_relationship == "Director"

    def test_invalid_xml_returns_empty(self):
        """Invalid XML returns empty list without raising."""
        result = SECEdgarClient._parse_derivative_xml(
            "not xml at all",
            ticker="AAPL",
            filing_date=date(2026, 1, 1),
            filing_url="https://example.com",
        )
        assert result == []

    def test_no_owner_returns_empty(self):
        """XML with no reportingOwner returns empty list."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Option</value></securityTitle>
      <transactionDate><value>2026-01-15</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""
        result = SECEdgarClient._parse_derivative_xml(
            xml,
            ticker="AAPL",
            filing_date=date(2026, 1, 1),
            filing_url="https://example.com",
        )
        assert result == []


# ---------------------------------------------------------------------------
# Form 3 Holding Parsing
# ---------------------------------------------------------------------------


class TestForm3HoldingParsing:
    def test_holdings_with_shares(self):
        """Holdings with nonzero shares produce InsiderTransaction with code J."""
        txns = SECEdgarClient._parse_form4_xml(
            SAMPLE_FORM3_HOLDING_XML,
            ticker="TSLA",
            filing_date=date(2026, 3, 1),
            filing_url="https://www.sec.gov/form3",
        )

        # Only the holding with 25000 shares should be included (0 shares skipped)
        assert len(txns) == 1
        txn = txns[0]
        assert isinstance(txn, InsiderTransaction)
        assert txn.transaction_code == "J"
        assert txn.shares == 25000.0
        assert txn.shares_after == 25000.0
        assert txn.owner_name == "New Director"
        assert txn.owner_relationship == "Director"
        assert txn.acquired_or_disposed == "A"
        # Form 3 holdings use filing date as transaction date
        assert txn.transaction_date == date(2026, 3, 1)

    def test_zero_shares_holding_skipped(self):
        """Holdings with zero shares are excluded."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Empty Holder</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeHolding>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction>
          <value>0</value>
        </sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeHolding>
  </nonDerivativeTable>
</ownershipDocument>"""
        txns = SECEdgarClient._parse_form4_xml(
            xml,
            ticker="AAPL",
            filing_date=date(2026, 1, 1),
            filing_url="https://example.com",
        )
        assert txns == []


# ---------------------------------------------------------------------------
# Owner Country
# ---------------------------------------------------------------------------


class TestOwnerCountry:
    def test_country_extracted_from_form4(self):
        """Form 4 XML with rptOwnerCountry extracts it correctly."""
        txns = SECEdgarClient._parse_form4_xml(
            SAMPLE_FORM4_WITH_COUNTRY_XML,
            ticker="AAPL",
            filing_date=date(2026, 3, 20),
            filing_url="https://www.sec.gov/form4",
        )

        assert len(txns) == 1
        assert txns[0].owner_country == "GB"
        assert txns[0].owner_name == "Foreign Insider"
        assert txns[0].owner_relationship == "Officer (VP International)"

    def test_country_empty_when_missing(self):
        """Form 4 XML without rptOwnerCountry defaults to empty string."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>Domestic Insider</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-01-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>100.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>1500</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        txns = SECEdgarClient._parse_form4_xml(
            xml,
            ticker="AAPL",
            filing_date=date(2026, 1, 20),
            filing_url="https://example.com",
        )
        assert len(txns) == 1
        assert txns[0].owner_country == ""

    def test_country_on_derivative_transaction(self):
        """Derivative parser also extracts owner country."""
        txns = SECEdgarClient._parse_derivative_xml(
            SAMPLE_DERIVATIVE_XML.replace(
                "<rptOwnerName>Jane Smith</rptOwnerName>",
                "<rptOwnerName>Jane Smith</rptOwnerName>\n      <rptOwnerCountry>CA</rptOwnerCountry>",
            ),
            ticker="AAPL",
            filing_date=date(2026, 1, 20),
            filing_url="https://www.sec.gov/example",
        )
        assert len(txns) == 1
        assert txns[0].owner_country == "CA"


# ---------------------------------------------------------------------------
# 8-K Events
# ---------------------------------------------------------------------------


class TestGet8KEvents:
    async def test_multi_item_split(self, client: SECEdgarClient, mock_redis):
        """Multi-item 8-K filing '2.02,9.01' is correctly split into list."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_8K_SUBMISSIONS)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            events = await client.get_8k_events("AAPL", limit=10)

        assert len(events) == 3
        assert all(isinstance(e, EventFiling8K) for e in events)

        # First event has multi-item
        assert events[0].items == ["2.02", "9.01"]
        assert len(events[0].item_descriptions) == 2
        assert "Results of Operations" in events[0].item_descriptions[0]
        assert "Financial Statements" in events[0].item_descriptions[1]

        # Second event: single item
        assert events[1].items == ["5.02"]

    async def test_item_filter(self, client: SECEdgarClient, mock_redis):
        """Item filter only returns 8-K filings with matching items."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_8K_SUBMISSIONS)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            events = await client.get_8k_events("AAPL", items=["2.02"])

        # Only the first filing has "2.02"
        assert len(events) == 1
        assert "2.02" in events[0].items

    async def test_empty_items_string(self, client: SECEdgarClient, mock_redis):
        """Filing with empty items string is included when no filter applied."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_8K_SUBMISSIONS)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            events = await client.get_8k_events("AAPL", limit=10)

        # Third 8-K has empty items string — should be included with items=[""]
        empty_item_events = [e for e in events if e.items == [""]]
        assert len(empty_item_events) == 1

    async def test_item_filter_excludes_empty_items(self, client: SECEdgarClient, mock_redis):
        """Filing with empty items is excluded when item filter is applied."""
        client._cik_map = {"AAPL": "0000320193"}

        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_8K_SUBMISSIONS)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            events = await client.get_8k_events("AAPL", items=["5.02"])

        # Only the second 8-K has "5.02"
        assert len(events) == 1
        assert events[0].items == ["5.02"]
        assert events[0].item_descriptions == ["Departure/Appointment of Officers"]
