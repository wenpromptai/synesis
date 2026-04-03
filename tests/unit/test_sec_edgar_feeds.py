"""Tests for SEC EDGAR feeds mixin — Atom feed parsing."""

from __future__ import annotations


from synesis.providers.sec_edgar._feeds import FeedsMixin
from synesis.providers.sec_edgar.models import FilingFeedEntry


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>SEC EDGAR Filings</title>
  <entry>
    <title>8-K - Apple Inc. (0000320193)</title>
    <link href="https://www.sec.gov/Archives/edgar/data/320193/000032019326000010/0000320193-26-000010-index.htm" rel="alternate" type="text/html"/>
    <summary>Filed 8-K for Apple Inc.</summary>
    <updated>2026-02-10T16:30:00-05:00</updated>
    <category term="8-K" label="form type"/>
  </entry>
  <entry>
    <title>4 - John Doe (0001234567)</title>
    <link href="https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/0001234567-26-000001-index.htm" rel="alternate" type="text/html"/>
    <summary>Filed Form 4 for insider transaction</summary>
    <updated>2026-02-09T12:00:00-05:00</updated>
    <category term="4" label="form type"/>
  </entry>
</feed>"""

SAMPLE_ATOM_FEED_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>SEC EDGAR Filings</title>
</feed>"""

SAMPLE_ATOM_FEED_MISSING_LINK = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>SC 13D - Activist Fund LP (0009876543)</title>
    <summary>Schedule 13D filing</summary>
    <updated>2026-03-01T10:00:00-05:00</updated>
    <category term="SC 13D" label="form type"/>
  </entry>
</feed>"""


# ---------------------------------------------------------------------------
# TestParseAtomFeed
# ---------------------------------------------------------------------------


class TestParseAtomFeed:
    def test_valid_atom_feed(self):
        """Valid Atom XML with entries parses correctly."""
        entries = FeedsMixin._parse_atom_feed(SAMPLE_ATOM_FEED)

        assert len(entries) == 2
        assert all(isinstance(e, FilingFeedEntry) for e in entries)

        # First entry
        assert entries[0].title == "8-K - Apple Inc. (0000320193)"
        assert "320193" in entries[0].link
        assert entries[0].summary == "Filed 8-K for Apple Inc."
        assert "2026-02-10" in entries[0].updated
        assert entries[0].category == "8-K"

        # Second entry
        assert entries[1].title == "4 - John Doe (0001234567)"
        assert entries[1].category == "4"

    def test_empty_feed(self):
        """Feed with no entries returns empty list."""
        entries = FeedsMixin._parse_atom_feed(SAMPLE_ATOM_FEED_EMPTY)
        assert entries == []

    def test_invalid_xml_returns_empty(self):
        """Invalid XML returns empty list instead of raising."""
        entries = FeedsMixin._parse_atom_feed("this is not xml at all <><>")
        assert entries == []

    def test_entry_missing_link(self):
        """Entry with title but no link element is still included."""
        entries = FeedsMixin._parse_atom_feed(SAMPLE_ATOM_FEED_MISSING_LINK)

        assert len(entries) == 1
        assert entries[0].title == "SC 13D - Activist Fund LP (0009876543)"
        assert entries[0].link == ""
        assert entries[0].category == "SC 13D"

    def test_entry_with_empty_title_and_link_skipped(self):
        """Entry with neither title nor link is excluded."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <summary>Has summary but no title or link</summary>
    <updated>2026-01-01T00:00:00Z</updated>
  </entry>
</feed>"""
        entries = FeedsMixin._parse_atom_feed(xml)
        assert entries == []
