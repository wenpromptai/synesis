"""Event deduplication and merge logic."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from synesis.core.logging import get_logger
from synesis.processing.events.models import CalendarEvent

if TYPE_CHECKING:
    from synesis.storage.database import Database

logger = get_logger(__name__)

# Words to strip when normalizing titles
_STRIP_WORDS = {"the", "a", "an", "of", "for", "and", "in", "on", "at", "to"}


def normalize_title(title: str) -> str:
    """Normalize an event title for hashing.

    Lowercases, strips articles/prepositions, collapses whitespace.
    """
    title = title.lower().strip()
    # Remove punctuation except hyphens
    title = re.sub(r"[^\w\s-]", "", title)
    words = [w for w in title.split() if w not in _STRIP_WORDS]
    return " ".join(words)


def hash_title(title: str) -> str:
    """Create a SHA256 hash of a normalized title."""
    normalized = normalize_title(title)
    return hashlib.sha256(normalized.encode()).hexdigest()


async def deduplicate_and_store(
    events: list[CalendarEvent],
    db: Database,
) -> int:
    """Deduplicate events and upsert into the database.

    Exact dedup is handled by the DB unique constraint (title_hash, event_date).
    Near-dedup: before inserting, check for events within +/-5 days with
    overlapping tickers. If a near-match is found, adopt its title_hash and date
    so the UPSERT merges into the existing row.

    Returns the number of events upserted.
    """
    stored = 0

    for event in events:
        title_h = hash_title(event.title)

        # Near-dedup: check for similar events nearby
        if event.tickers:
            nearby = await db.get_nearby_events(
                event.event_date, days_range=5, tickers=event.tickers
            )
            for row in nearby:
                if row["title_hash"] == title_h:
                    # Same hash, same date = exact dedup handled by UPSERT constraint
                    continue
                # Near match: same tickers within 5 days, check title similarity
                existing_norm = normalize_title(row["title"])
                new_norm = normalize_title(event.title)
                if _titles_overlap(existing_norm, new_norm):
                    logger.debug(
                        "Near-dedup: merging into existing event",
                        existing_id=row["id"],
                        existing_title=row["title"],
                        new_title=event.title,
                    )
                    # Merge: adopt existing title_hash AND date so UPSERT updates the row
                    title_h = row["title_hash"]
                    event = event.model_copy(update={"event_date": row["event_date"]})
                    break

        result = await db.upsert_calendar_event(event, title_h)
        if result is not None:
            stored += 1

    logger.info("Dedup complete", input_events=len(events), stored=stored)
    return stored


def _titles_overlap(a: str, b: str) -> bool:
    """Check if two normalized titles are similar enough to merge.

    Uses word-level Jaccard similarity with a threshold of 0.5.
    """
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return False
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) >= 0.5
