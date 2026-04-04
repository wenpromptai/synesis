"""End-to-end test: Google News RSS freshness filter + semantic dedup.

Verifies the RSS poller's two-layer dedup:
  1. Freshness filter — articles older than 15 min are skipped
  2. Semantic dedup — Model2Vec cosine similarity catches duplicates
     (including rotating GUIDs from Google News)

Flow:
  1. Fetch real RSS feeds
  2. Show freshness distribution (how many fresh vs stale)
  3. Pass 1 — process all items through semantic dedup (cache embeddings)
  4. Pass 2 — re-process same items (should all be caught by dedup)
  5. Pass 3 — re-process with randomised GUIDs (simulates Google rotating them)

Requires Redis running locally.

Usage:
    uv run python scripts/test_google_rss.py
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import httpx

from synesis.config import get_settings
from synesis.ingestion.google_rss import _FRESHNESS_MINUTES, RSSItem, parse_feed_xml
from synesis.processing.news.deduplication import create_deduplicator
from synesis.processing.news.models import SourcePlatform, UnifiedMessage
from synesis.storage.redis import close_redis, init_redis


def _make_message(item: RSSItem, *, rotate_guid: bool = False) -> UnifiedMessage:
    """Convert a parsed RSS item to a UnifiedMessage.

    When rotate_guid=True, replaces the GUID with a random UUID to simulate
    Google News rotating GUIDs between poll cycles.
    """
    gid = str(uuid.uuid4()) if rotate_guid else item.guid
    return UnifiedMessage(
        external_id=gid,
        source_platform=SourcePlatform.google_rss,
        source_account=item.source_name,
        text=item.title,
        timestamp=item.pub_date,
        raw={
            "guid": gid,
            "link": item.link,
            "resolved_url": item.link,
            "source_name": item.source_name,
            "source_url": item.source_url,
        },
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
    all_items: list[RSSItem] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for url in feeds:
            print(f"  {url[:90]}...")
            resp = await client.get(url)
            resp.raise_for_status()
            items = parse_feed_xml(resp.content)
            all_items.extend(items)
            print(f"    -> {len(items)} items")

    print(f"\nTotal: {len(all_items)} items\n")
    if not all_items:
        print("No items fetched, exiting.")
        await close_redis()
        return

    # 3. Freshness filter check
    now = datetime.now(UTC)
    cutoff = now - timedelta(minutes=_FRESHNESS_MINUTES)
    fresh_items = [i for i in all_items if i.pub_date >= cutoff]
    stale_items = [i for i in all_items if i.pub_date < cutoff]

    print("=" * 100)
    print(f"FRESHNESS CHECK (cutoff = {_FRESHNESS_MINUTES} min)")
    print("=" * 100)
    print(f"  Now (UTC):   {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cutoff:      {cutoff.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Fresh items: {len(fresh_items)} (would be processed)")
    print(f"  Stale items: {len(stale_items)} (would be skipped)")

    if fresh_items:
        print("\n  Fresh articles:")
        for item in fresh_items:
            age_min = (now - item.pub_date).total_seconds() / 60
            print(f"    age={age_min:>4.0f}m  {item.source_name[:20]:<22} {item.title[:55]}")
    else:
        print("\n  No fresh articles right now (all >15 min old) — this is normal")
        print("  When breaking news drops, fresh articles will appear")

    # 4. Create deduplicator (loads Model2Vec)
    print(f"\n{'=' * 100}")
    print("Creating MessageDeduplicator (loading Model2Vec)...")
    deduplicator = await create_deduplicator(redis)
    print("  Ready.\n")

    # Use a subset for dedup testing (fresh or first 20 if none fresh)
    test_items = fresh_items if fresh_items else all_items[:20]
    test_label = "fresh" if fresh_items else "first 20 (all stale, for dedup testing only)"
    print(f"Testing dedup with {len(test_items)} {test_label} items\n")

    # ── Pass 1: cache embeddings ──────────────────────────────────────
    print("=" * 100)
    print(f"PASS 1: Processing {len(test_items)} items (caching embeddings)")
    print("=" * 100)

    pass1_new = 0
    for item in test_items:
        msg = _make_message(item)
        result = await deduplicator.process_message(msg)
        status = "DEDUP" if result.is_duplicate else "NEW"
        sim = f"{result.similarity:.3f}" if result.similarity else "-"
        print(f"  {status:<6} sim={sim:<6} {msg.source_account[:20]:<22} {msg.text[:55]}")
        if not result.is_duplicate:
            pass1_new += 1

    print(f"\n  Pass 1: {pass1_new} new, {len(test_items) - pass1_new} deduped\n")

    # ── Pass 2: same items again ──────────────────────────────────────
    print("=" * 100)
    print(f"PASS 2: Re-processing same {len(test_items)} items (same GUIDs)")
    print("=" * 100)

    pass2_dedup = 0
    for item in test_items:
        msg = _make_message(item)
        result = await deduplicator.process_message(msg)
        if result.is_duplicate:
            pass2_dedup += 1
        status = "DEDUP" if result.is_duplicate else "MISSED"
        sim = f"{result.similarity:.3f}" if result.similarity else "-"
        print(f"  {status:<6} sim={sim:<6} {msg.source_account[:20]:<22} {msg.text[:55]}")

    print(f"\n  Pass 2: {pass2_dedup}/{len(test_items)} caught by dedup")
    if pass2_dedup == len(test_items):
        print("  All items correctly deduped!")
    else:
        print(f"  WARNING: {len(test_items) - pass2_dedup} items missed dedup")

    # ── Pass 3: rotated GUIDs (simulates Google News rotating GUIDs) ──
    print(f"\n{'=' * 100}")
    print(f"PASS 3: Re-processing {len(test_items)} items with ROTATED GUIDs")
    print("=" * 100)

    pass3_dedup = 0
    for item in test_items:
        msg = _make_message(item, rotate_guid=True)
        result = await deduplicator.process_message(msg)
        if result.is_duplicate:
            pass3_dedup += 1
        status = "DEDUP" if result.is_duplicate else "MISSED"
        sim = f"{result.similarity:.3f}" if result.similarity else "-"
        print(f"  {status:<6} sim={sim:<6} {msg.source_account[:20]:<22} {msg.text[:55]}")

    print(f"\n  Pass 3: {pass3_dedup}/{len(test_items)} caught by semantic dedup")
    if pass3_dedup == len(test_items):
        print("  All rotating-GUID duplicates caught by semantic dedup!")
    else:
        missed = len(test_items) - pass3_dedup
        print(f"  WARNING: {missed} items with rotated GUIDs were NOT caught")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print("=" * 100)
    print(f"  Freshness filter:       {len(fresh_items)} fresh / {len(stale_items)} stale")
    print(f"  Pass 1 (cache):         {pass1_new} new items cached")
    print(f"  Pass 2 (same GUID):     {pass2_dedup}/{len(test_items)} deduped")
    print(f"  Pass 3 (rotated GUID):  {pass3_dedup}/{len(test_items)} deduped")

    all_ok = pass2_dedup == len(test_items) and pass3_dedup == len(test_items)
    print(f"\n  {'ALL PASSED' if all_ok else 'SOME FAILURES — check above'}")

    await close_redis()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
