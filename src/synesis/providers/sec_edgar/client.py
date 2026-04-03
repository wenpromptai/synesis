"""SEC EDGAR API client.

Free API — no key required, just a User-Agent header.
- Company info + filings: https://data.sec.gov/submissions/CIK{cik}.json
- Full-text search: https://efts.sec.gov/LATEST/search-index
- XBRL Company Facts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
- XBRL Frames: https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json
- Form 3/4/5 XML: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}
- Filing feed: https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom
"""

from __future__ import annotations

from synesis.providers.sec_edgar._13f import ThirteenFMixin
from synesis.providers.sec_edgar._base import SECEdgarBase
from synesis.providers.sec_edgar._feeds import FeedsMixin
from synesis.providers.sec_edgar._filings import FilingsMixin
from synesis.providers.sec_edgar._insiders import InsidersMixin
from synesis.providers.sec_edgar._ownership import OwnershipMixin
from synesis.providers.sec_edgar._xbrl import XBRLMixin

# Re-export for backward compatibility (tests import from client.py)
from synesis.providers.sec_edgar._base import _SECRateLimiter  # noqa: F401


class SECEdgarClient(
    FilingsMixin,
    InsidersMixin,
    XBRLMixin,
    ThirteenFMixin,
    OwnershipMixin,
    FeedsMixin,
    SECEdgarBase,
):
    """Client for the SEC EDGAR API.

    Provides access to SEC filings, insider transactions (Form 3/4/5),
    XBRL financial data, 13F holdings, ownership/governance filings,
    real-time filing feeds, and full-text filing search.
    Uses Redis caching to minimize requests.

    Usage:
        client = SECEdgarClient(redis=redis_client)
        filings = await client.get_filings("AAPL", form_types=["8-K", "10-K"])
        info = await client.get_company_info("AAPL")
        facts = await client.get_company_facts("AAPL")
        feed = await client.get_filing_feed(form_type="8-K")
        await client.close()
    """
