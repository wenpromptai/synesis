"""Tests for SEC EDGAR 13F-HR methods (get_13f_filings, get_13f_holdings, compare_13f_quarters)."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from synesis.providers.sec_edgar.client import SECEdgarClient
from synesis.providers.sec_edgar.models import Filing13F, Holding13F, SECFiling


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_13F_SUBMISSIONS = {
    "cik": "1536411",
    "name": "Duquesne Family Office",
    "filings": {
        "recent": {
            "form": ["13F-HR", "13F-HR/A", "13F-HR", "10-K"],
            "filingDate": ["2026-02-14", "2025-12-01", "2025-11-14", "2025-10-01"],
            "accessionNumber": [
                "0001536411-26-000001",
                "0001536411-25-000003",
                "0001536411-25-000002",
                "0001536411-25-000001",
            ],
            "primaryDocument": ["primary_doc.xml", "primary_doc.xml", "primary_doc.xml", "doc.htm"],
            "reportDate": ["2025-12-31", "2025-09-30", "2025-09-30", ""],
            "acceptanceDateTime": [
                "2026-02-14T10:00:00.000Z",
                "2025-12-01T09:00:00.000Z",
                "2025-11-14T08:00:00.000Z",
                "2025-10-01T12:00:00.000Z",
            ],
        }
    },
}

SAMPLE_13F_INDEX = {
    "directory": {
        "item": [
            {"name": "primary_doc.xml", "type": "primary_doc.xml"},
            {"name": "infotable.xml", "type": "infotable.xml"},
            {"name": "R9999.htm", "type": "R9999.htm"},
        ]
    }
}

SAMPLE_13F_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>500000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>2500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>300000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>800000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>67066G104</cusip>
    <value>200000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>1500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>
"""

SAMPLE_13F_XML_PREV = """\
<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>400000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>2000000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>300000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>800000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
  <infoTable>
    <nameOfIssuer>TESLA INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>88160R101</cusip>
    <value>100000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def client(mock_redis: AsyncMock) -> SECEdgarClient:
    return SECEdgarClient(redis=mock_redis)


def _make_response(
    content: bytes | str, status_code: int = 200, text: str | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content if isinstance(content, bytes) else content.encode()
    resp.text = text or (content if isinstance(content, str) else content.decode())
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Tests: get_13f_filings
# ---------------------------------------------------------------------------


class TestGet13FFilings:
    @pytest.mark.asyncio
    async def test_get_13f_filings(self, client: SECEdgarClient) -> None:
        resp = _make_response(orjson.dumps(SAMPLE_13F_SUBMISSIONS))

        with patch.object(client, "_fetch", AsyncMock(return_value=resp)):
            filings = await client.get_13f_filings("1536411", limit=2)

        assert len(filings) == 2
        assert all(f.form == "13F-HR" for f in filings)
        assert filings[0].filed_date == date(2026, 2, 14)
        assert filings[0].accession_number == "0001536411-26-000001"
        assert filings[0].items == "2025-12-31"  # reportDate stored in items
        assert filings[1].filed_date == date(2025, 11, 14)

    @pytest.mark.asyncio
    async def test_get_13f_filings_from_cache(
        self, client: SECEdgarClient, mock_redis: AsyncMock
    ) -> None:
        cached_filing = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2026, 2, 14),
            accepted_datetime=datetime(2026, 2, 14, 10, 0),
            accession_number="0001536411-26-000001",
            primary_document="primary_doc.xml",
            items="2025-12-31",
            url="https://www.sec.gov/test",
        )
        mock_redis.get.return_value = orjson.dumps([cached_filing.model_dump(mode="json")])

        filings = await client.get_13f_filings("1536411")

        assert len(filings) == 1
        assert filings[0].accession_number == "0001536411-26-000001"

    @pytest.mark.asyncio
    async def test_get_13f_filings_no_13f(self, client: SECEdgarClient) -> None:
        data = {
            "cik": "1536411",
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K"],
                    "filingDate": ["2026-02-14", "2026-01-10"],
                    "accessionNumber": ["acc1", "acc2"],
                    "primaryDocument": ["d1.htm", "d2.htm"],
                    "reportDate": ["", ""],
                    "acceptanceDateTime": ["2026-02-14T10:00:00Z", "2026-01-10T08:00:00Z"],
                }
            },
        }
        resp = _make_response(orjson.dumps(data))

        with patch.object(client, "_fetch", AsyncMock(return_value=resp)):
            filings = await client.get_13f_filings("1536411")

        assert len(filings) == 0

    @pytest.mark.asyncio
    async def test_get_13f_filings_fetch_failure(self, client: SECEdgarClient) -> None:
        with patch.object(client, "_fetch", AsyncMock(side_effect=Exception("timeout"))):
            filings = await client.get_13f_filings("1536411")

        assert filings == []

    @pytest.mark.asyncio
    async def test_get_13f_filings_url_construction(self, client: SECEdgarClient) -> None:
        resp = _make_response(orjson.dumps(SAMPLE_13F_SUBMISSIONS))

        with patch.object(client, "_fetch", AsyncMock(return_value=resp)):
            filings = await client.get_13f_filings("1536411", limit=1)

        assert len(filings) == 1
        assert "0001536411" in filings[0].url
        assert "000153641126000001" in filings[0].url


# ---------------------------------------------------------------------------
# Tests: _parse_13f_xml
# ---------------------------------------------------------------------------


class TestParse13FXML:
    def test_parse_valid_xml(self) -> None:
        holdings = SECEdgarClient._parse_13f_xml(SAMPLE_13F_XML)

        assert len(holdings) == 3
        assert holdings[0].name_of_issuer == "APPLE INC"
        assert holdings[0].cusip == "037833100"
        assert holdings[0].value_thousands == 500000
        assert holdings[0].shares == 2500000
        assert holdings[1].name_of_issuer == "MICROSOFT CORP"
        assert holdings[2].name_of_issuer == "NVIDIA CORP"

    def test_parse_empty_xml(self) -> None:
        xml = '<?xml version="1.0"?><informationTable></informationTable>'
        holdings = SECEdgarClient._parse_13f_xml(xml)
        assert len(holdings) == 0

    def test_parse_invalid_xml(self) -> None:
        holdings = SECEdgarClient._parse_13f_xml("not xml at all")
        assert len(holdings) == 0

    def test_parse_missing_cusip(self) -> None:
        xml = """\
<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <value>500000</value>
  </infoTable>
</informationTable>
"""
        holdings = SECEdgarClient._parse_13f_xml(xml)
        assert len(holdings) == 0

    def test_parse_no_namespace(self) -> None:
        xml = """\
<?xml version="1.0"?>
<informationTable>
  <infoTable>
    <nameOfIssuer>GOOGLE INC</nameOfIssuer>
    <titleOfClass>CL A</titleOfClass>
    <cusip>02079K107</cusip>
    <value>100000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>50000</sshPrnamt>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
  </infoTable>
</informationTable>
"""
        holdings = SECEdgarClient._parse_13f_xml(xml)
        assert len(holdings) == 1
        assert holdings[0].name_of_issuer == "GOOGLE INC"
        assert holdings[0].shares == 50000


# ---------------------------------------------------------------------------
# Tests: get_13f_holdings
# ---------------------------------------------------------------------------


class TestGet13FHoldings:
    @pytest.mark.asyncio
    async def test_get_13f_holdings(self, client: SECEdgarClient) -> None:
        filing = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2026, 2, 14),
            accepted_datetime=datetime(2026, 2, 14, 10, 0),
            accession_number="0001536411-26-000001",
            primary_document="primary_doc.xml",
            items="2025-12-31",
            url="",
        )
        index_resp = _make_response(orjson.dumps(SAMPLE_13F_INDEX))
        xml_resp = _make_response(SAMPLE_13F_XML, text=SAMPLE_13F_XML)

        with patch.object(client, "_fetch", AsyncMock(side_effect=[index_resp, xml_resp])):
            result = await client.get_13f_holdings(filing, "1536411")

        assert result is not None
        assert isinstance(result, Filing13F)
        assert len(result.holdings) == 3
        # Sorted by value descending
        assert result.holdings[0].name_of_issuer == "APPLE INC"
        assert result.holdings[0].value_thousands == 500000
        assert result.total_value_thousands == 1000000
        assert result.report_date == date(2025, 12, 31)

    @pytest.mark.asyncio
    async def test_get_13f_holdings_cached(
        self, client: SECEdgarClient, mock_redis: AsyncMock
    ) -> None:
        filing = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2026, 2, 14),
            accepted_datetime=datetime(2026, 2, 14, 10, 0),
            accession_number="0001536411-26-000001",
            primary_document="",
            items="2025-12-31",
            url="",
        )
        cached = Filing13F(
            cik="1536411",
            fund_name="Test Fund",
            filed_date=date(2026, 2, 14),
            report_date=date(2025, 12, 31),
            accession_number="0001536411-26-000001",
            url="",
            holdings=[
                Holding13F(
                    name_of_issuer="AAPL",
                    title_of_class="COM",
                    cusip="037833100",
                    value_thousands=500,
                    shares=100,
                    investment_discretion="SOLE",
                )
            ],
            total_value_thousands=500,
        )
        mock_redis.get.return_value = orjson.dumps(cached.model_dump(mode="json"))

        result = await client.get_13f_holdings(filing, "1536411")

        assert result is not None
        assert result.total_value_thousands == 500

    @pytest.mark.asyncio
    async def test_get_13f_holdings_no_infotable(self, client: SECEdgarClient) -> None:
        filing = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2026, 2, 14),
            accepted_datetime=datetime(2026, 2, 14, 10, 0),
            accession_number="0001536411-26-000001",
            primary_document="",
            items="",
            url="",
        )
        index_data = {"directory": {"item": [{"name": "primary_doc.xml"}]}}
        index_resp = _make_response(orjson.dumps(index_data))

        with patch.object(client, "_fetch", AsyncMock(return_value=index_resp)):
            result = await client.get_13f_holdings(filing, "1536411")

        assert result is None


# ---------------------------------------------------------------------------
# Tests: compare_13f_quarters
# ---------------------------------------------------------------------------


class TestCompare13FQuarters:
    @pytest.mark.asyncio
    async def test_compare_detects_changes(self, client: SECEdgarClient) -> None:
        filing_curr = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2026, 2, 14),
            accepted_datetime=datetime.min,
            accession_number="acc-curr",
            primary_document="",
            items="2025-12-31",
            url="",
        )
        filing_prev = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2025, 11, 14),
            accepted_datetime=datetime.min,
            accession_number="acc-prev",
            primary_document="",
            items="2025-09-30",
            url="",
        )

        curr_holdings = Filing13F(
            cik="1536411",
            fund_name="",
            filed_date=date(2026, 2, 14),
            report_date=date(2025, 12, 31),
            accession_number="acc-curr",
            url="",
            holdings=[
                Holding13F(
                    name_of_issuer="APPLE INC",
                    title_of_class="COM",
                    cusip="037833100",
                    value_thousands=500,
                    shares=2500,
                    investment_discretion="SOLE",
                ),
                Holding13F(
                    name_of_issuer="NVIDIA CORP",
                    title_of_class="COM",
                    cusip="67066G104",
                    value_thousands=200,
                    shares=1500,
                    investment_discretion="SOLE",
                ),
            ],
            total_value_thousands=700,
        )
        prev_holdings = Filing13F(
            cik="1536411",
            fund_name="",
            filed_date=date(2025, 11, 14),
            report_date=date(2025, 9, 30),
            accession_number="acc-prev",
            url="",
            holdings=[
                Holding13F(
                    name_of_issuer="APPLE INC",
                    title_of_class="COM",
                    cusip="037833100",
                    value_thousands=400,
                    shares=2000,
                    investment_discretion="SOLE",
                ),
                Holding13F(
                    name_of_issuer="TESLA INC",
                    title_of_class="COM",
                    cusip="88160R101",
                    value_thousands=100,
                    shares=500,
                    investment_discretion="SOLE",
                ),
            ],
            total_value_thousands=500,
        )

        with (
            patch.object(
                client, "get_13f_filings", AsyncMock(return_value=[filing_curr, filing_prev])
            ),
            patch.object(
                client, "get_13f_holdings", AsyncMock(side_effect=[curr_holdings, prev_holdings])
            ),
        ):
            result = await client.compare_13f_quarters("1536411", "Duquesne Family Office")

        assert result is not None
        assert result["fund_name"] == "Duquesne Family Office"
        assert result["total_value_current"] == 700
        assert result["total_value_previous"] == 500

        # NVIDIA is new (not in prev)
        new_cusips = [p["cusip"] for p in result["new_positions"]]
        assert "67066G104" in new_cusips

        # TESLA is exited (not in curr)
        exited_cusips = [p["cusip"] for p in result["exited_positions"]]
        assert "88160R101" in exited_cusips

        # APPLE increased 25% (2000->2500)
        increased_cusips = [p["cusip"] for p in result["increased"]]
        assert "037833100" in increased_cusips

    @pytest.mark.asyncio
    async def test_compare_not_enough_filings(self, client: SECEdgarClient) -> None:
        with patch.object(client, "get_13f_filings", AsyncMock(return_value=[])):
            result = await client.compare_13f_quarters("1536411", "Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_compare_cached(self, client: SECEdgarClient, mock_redis: AsyncMock) -> None:
        cached = {"fund_name": "Test", "new_positions": [], "exited_positions": []}
        mock_redis.get.return_value = orjson.dumps(cached)

        result = await client.compare_13f_quarters("1536411", "Test")

        assert result is not None
        assert result["fund_name"] == "Test"

    @pytest.mark.asyncio
    async def test_compare_no_change(self, client: SECEdgarClient) -> None:
        """When holdings are identical, no increases/decreases/new/exited."""
        filing_curr = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2026, 2, 14),
            accepted_datetime=datetime.min,
            accession_number="acc-curr",
            primary_document="",
            items="2025-12-31",
            url="",
        )
        filing_prev = SECFiling(
            ticker="",
            form="13F-HR",
            filed_date=date(2025, 11, 14),
            accepted_datetime=datetime.min,
            accession_number="acc-prev",
            primary_document="",
            items="2025-09-30",
            url="",
        )

        same_holding = Holding13F(
            name_of_issuer="APPLE INC",
            title_of_class="COM",
            cusip="037833100",
            value_thousands=500,
            shares=2500,
            investment_discretion="SOLE",
        )
        holdings = Filing13F(
            cik="1536411",
            fund_name="",
            filed_date=date(2026, 2, 14),
            report_date=date(2025, 12, 31),
            accession_number="acc",
            url="",
            holdings=[same_holding],
            total_value_thousands=500,
        )

        with (
            patch.object(
                client, "get_13f_filings", AsyncMock(return_value=[filing_curr, filing_prev])
            ),
            patch.object(client, "get_13f_holdings", AsyncMock(return_value=holdings)),
        ):
            result = await client.compare_13f_quarters("1536411", "Test")

        assert result is not None
        assert len(result["new_positions"]) == 0
        assert len(result["exited_positions"]) == 0
        assert len(result["increased"]) == 0
        assert len(result["decreased"]) == 0
