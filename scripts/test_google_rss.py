"""End-to-end test: Google News RSS → full NewsProcessor pipeline → Discord.

Runs real RSS items through the complete pipeline:
  1. Fetch RSS feeds
  2. NewsProcessor.process_message() for each item:
     - Semantic dedup (Model2Vec via Redis)
     - Stage 1 classification (impact scoring + ticker matching)
     - Stage 1 Discord notification (for normal/high/critical)
     - Stage 2 skipped (would need LLM API keys)
  3. Run the same items again to verify dedup catches them

Requires Redis running locally (for dedup storage).

Usage:
    uv run python scripts/test_google_rss.py
"""

import asyncio

import httpx

from synesis.agent.pydantic_runner import emit_stage1_notification
from synesis.config import get_settings
from synesis.core.processor import NewsProcessor, ProcessingResult
from synesis.ingestion.google_rss import RSSItem, parse_feed_xml
from synesis.processing.news.models import SourcePlatform, UnifiedMessage, UrgencyLevel
from synesis.storage.redis import close_redis, init_redis


def _make_message(item: RSSItem) -> UnifiedMessage:
    """Convert a parsed RSS item to a UnifiedMessage."""
    return UnifiedMessage(
        external_id=item.guid,
        source_platform=SourcePlatform.google_rss,
        source_account=item.source_name,
        text=item.title,
        timestamp=item.pub_date,
        raw={
            "guid": item.guid,
            "link": item.link,
            "resolved_url": item.link,
            "source_name": item.source_name,
            "source_url": item.source_url,
        },
    )


def _print_result(result: ProcessingResult) -> None:
    """Print a single processing result."""
    msg = result.message
    if result.is_duplicate:
        print(f"  {'DEDUP':<10} {'---':>5} {'-':<20} {msg.source_account[:24]:<25} {msg.text[:55]}")
        return
    if result.extraction is None:
        return

    ext = result.extraction
    tickers = ", ".join(ext.matched_tickers) if ext.matched_tickers else "-"
    print(
        f"  {ext.urgency.value:<10} {ext.impact_score:>5} "
        f"{tickers:<20} {msg.source_account[:24]:<25} {msg.text[:55]}"
    )


async def main() -> None:
    settings = get_settings()

    # 1. Connect to Redis
    print("Connecting to Redis...")
    redis = await init_redis(settings.redis_url)
    print(f"  Redis connected at {settings.redis_url}\n")

    # 2. Fetch RSS feeds
    feeds = settings.rss_feeds
    print(f"Fetching {len(feeds)} feed(s)...")
    all_items = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for url in feeds:
            print(f"  {url[:90]}...")
            resp = await client.get(url)
            resp.raise_for_status()
            items = parse_feed_xml(resp.content)
            all_items.extend(items)
            print(f"    -> {len(items)} items")

    print(f"\nTotal: {len(all_items)} items\n")

    # 3. Initialize NewsProcessor (loads Model2Vec for dedup + classifier)
    print("Initializing NewsProcessor (loading Model2Vec for dedup)...")
    processor = NewsProcessor(redis=redis)
    await processor.initialize()
    print("  Ready.\n")

    # 4. Process all items through full pipeline (pass 1)
    print("=" * 120)
    print("PASS 1: Processing all items through full pipeline")
    print("=" * 120)
    print(f"  {'URGENCY':<10} {'SCORE':>5} {'TICKERS':<20} {'SOURCE':<25} TITLE")
    print(f"  {'-' * 115}")

    counts: dict[str, int] = {"critical": 0, "high": 0, "normal": 0, "low": 0, "dedup": 0}
    discord_sent = False

    for item in all_items:
        message = _make_message(item)

        # Full pipeline: dedup -> Stage 1 -> Stage 2 gate
        # Only send ONE Discord notification as proof-of-life, not all of them
        callback = emit_stage1_notification if not discord_sent else None
        result = await processor.process_message(
            message,
            on_stage1_complete=callback,
        )

        if result.is_duplicate:
            counts["dedup"] += 1
        elif result.extraction:
            counts[result.extraction.urgency.value] += 1
            if not discord_sent and result.extraction.urgency != UrgencyLevel.low:
                discord_sent = True
                print(f"  >>> Discord notification sent for: {message.text[:70]}")

        _print_result(result)

    print("\n--- Pass 1 Summary ---")
    for label, count in counts.items():
        pct = count / len(all_items) * 100 if all_items else 0
        print(f"  {label:<10}: {count:>3} ({pct:.1f}%)")
    print(f"  Discord test notification: {'sent' if discord_sent else 'none (no normal+ items)'}")

    # 5. Process same items again (pass 2) — should all be caught by dedup
    print(f"\n{'=' * 120}")
    print("PASS 2: Re-processing same items (should all be deduped)")
    print("=" * 120)

    dedup_count = 0
    pass2_total = min(10, len(all_items))  # test first 10 to keep it quick

    for item in all_items[:pass2_total]:
        message = _make_message(item)
        result = await processor.process_message(message)

        if result.is_duplicate:
            dedup_count += 1
        _print_result(result)

    print("\n--- Pass 2 Summary ---")
    print(f"  Tested: {pass2_total} items")
    print(f"  Deduped: {dedup_count}/{pass2_total}")
    if dedup_count == pass2_total:
        print("  All items correctly caught by semantic dedup!")
    else:
        print(f"  WARNING: {pass2_total - dedup_count} items were NOT deduped")

    # Cleanup
    await processor.close()
    await close_redis()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
