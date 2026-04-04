"""Integration tests for Google News RSS ingestion.

Tests real HTTP fetches from Google News RSS feeds. No API key required.
Run with: pytest tests/integration/test_google_rss.py -m integration -v
"""

from __future__ import annotations

import pytest

from synesis.ingestion.google_rss import parse_feed_xml
from synesis.processing.news.classifier import NewsClassifier
from synesis.processing.news.models import SourcePlatform, UnifiedMessage


@pytest.mark.integration
class TestGoogleRSSIntegration:
    """Integration tests that fetch real Google News RSS feeds."""

    @pytest.mark.asyncio
    async def test_fetch_and_parse_real_feed(self) -> None:
        """Fetch a real Google News RSS feed and verify parsing works."""
        import httpx

        url = (
            "https://news.google.com/rss/search?"
            'q="data+center"+OR+semiconductor+OR+GPU+when:7d'
            "&hl=en-US&gl=US&ceid=US:en"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        items = parse_feed_xml(resp.content)

        # Should have items (Google News typically returns 50-100)
        assert len(items) > 0

        # Verify item structure
        first = items[0]
        assert first.guid  # non-empty
        assert first.title  # non-empty, source name stripped
        assert first.link  # Google redirect URL
        assert first.pub_date.year >= 2025
        assert first.source_name  # publisher name

        # Title should NOT end with " - SourceName"
        assert not first.title.endswith(f" - {first.source_name}")

    @pytest.mark.asyncio
    async def test_items_classify_through_stage1(self) -> None:
        """Verify real RSS items can be classified by Stage 1."""
        import httpx

        url = (
            "https://news.google.com/rss/search?"
            'q="AI"+"billion"+OR+"million"+acquisition+OR+investment+when:3d'
            "&hl=en-US&gl=US&ceid=US:en"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        items = parse_feed_xml(resp.content)
        assert len(items) > 0

        classifier = NewsClassifier()
        classified = 0

        for item in items[:10]:
            message = UnifiedMessage(
                external_id=item.guid,
                source_platform=SourcePlatform.google_rss,
                source_account=item.source_name,
                text=item.title,
                timestamp=item.pub_date,
                raw={},
            )
            result = await classifier.classify(message)

            # Every item should produce a valid classification
            assert result.impact_score >= 0
            assert result.urgency is not None
            classified += 1

        assert classified == min(10, len(items))
