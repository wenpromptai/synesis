"""Message deduplication using SemHash with Model2Vec embeddings.

This module provides semantic deduplication of incoming messages:
- Uses Model2Vec for fast embedding generation (~5ms per message)
- Stores embeddings in Redis with 60-min TTL
- Uses cosine similarity with 0.75 threshold
- Store-first pattern to prevent race conditions

References:
- SemHash: https://minishlab.github.io/semhash-blogpost/
- Model2Vec: https://github.com/MinishLab/model2vec
"""

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
from model2vec import StaticModel
from redis.asyncio import Redis
from redis.exceptions import RedisError

from synesis.core.logging import get_logger
from synesis.processing.news.models import UnifiedMessage

logger = get_logger(__name__)

# Model2Vec model for embeddings (256-dim, very fast)
DEFAULT_MODEL = "minishlab/potion-base-8M"
SIMILARITY_THRESHOLD = 0.75
REDIS_TTL_SECONDS = 60 * 60  # 60 minutes
REDIS_KEY_PREFIX = "dedup:emb:"


@dataclass
class DeduplicationResult:
    """Result of deduplication check."""

    is_duplicate: bool
    duplicate_of: str | None = None  # external_id of original message
    similarity: float | None = None
    processing_time_ms: float = 0.0


@dataclass
class MessageDeduplicator:
    """Semantic deduplication using Model2Vec embeddings stored in Redis.

    Uses cosine similarity to detect near-duplicate messages within
    a sliding 60-minute window. Uses store-first pattern to prevent
    race conditions when similar messages arrive simultaneously.
    """

    redis: Redis
    similarity_threshold: float = SIMILARITY_THRESHOLD
    ttl_seconds: int = REDIS_TTL_SECONDS
    model_name: str = DEFAULT_MODEL

    _model: StaticModel | None = field(default=None, init=False, repr=False)
    _embedding_dim: int = field(default=256, init=False, repr=False)

    async def initialize(self) -> None:
        """Load the Model2Vec model.

        Raises:
            RuntimeError: If model fails to load (network, disk, or model error)
        """
        if self._model is None:
            logger.info("Loading Model2Vec model", model=self.model_name)
            try:
                self._model = StaticModel.from_pretrained(self.model_name)
                # Get embedding dimension from model
                test_emb = self._model.encode(["test"])
                self._embedding_dim = test_emb.shape[1]
                logger.info(
                    "Model loaded",
                    model=self.model_name,
                    embedding_dim=self._embedding_dim,
                )
            except OSError as e:
                logger.error(
                    "Failed to load model - check disk space and permissions",
                    model=self.model_name,
                    error=str(e),
                )
                raise RuntimeError(f"Model loading failed: {e}") from e
            except Exception as e:
                logger.error(
                    "Failed to load model - check network and model name",
                    model=self.model_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise RuntimeError(f"Model loading failed: {e}") from e

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        Raises:
            RuntimeError: If model not initialized or encoding fails
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        try:
            # Model2Vec encode returns shape (1, dim) for single text
            embedding = self._model.encode([text])[0]
            result: np.ndarray = embedding.astype(np.float32)
            return result
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _make_redis_key(self, external_id: str, platform: str) -> str:
        """Create Redis key for storing embedding."""
        return f"{REDIS_KEY_PREFIX}{platform}:{external_id}"

    def _make_hash_key(self, text: str) -> str:
        """Create a short hash of text for exact match check."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def _check_duplicate_with_embedding(
        self, message: UnifiedMessage, embedding: np.ndarray
    ) -> DeduplicationResult:
        """Check if a message is a duplicate using pre-computed embedding.

        Args:
            message: The message to check
            embedding: Pre-computed embedding for the message

        Returns:
            DeduplicationResult with duplicate status and details
        """
        start_time = time.perf_counter()

        # Get all recent embeddings from Redis
        pattern = f"{REDIS_KEY_PREFIX}*"
        best_match: tuple[str | None, float] = (None, 0.0)
        embeddings_checked = 0
        errors_encountered = 0
        expected_bytes = self._embedding_dim * 4  # float32 = 4 bytes

        try:
            async for key in self.redis.scan_iter(match=pattern, count=100):
                # Skip if it's the same message
                key_str = key.decode() if isinstance(key, bytes) else key
                if message.external_id in key_str:
                    continue

                try:
                    # Get stored embedding
                    stored_data = await self.redis.get(key)
                    if stored_data is None:
                        continue

                    # Validate data size before deserialization
                    if len(stored_data) != expected_bytes:
                        logger.warning(
                            "Corrupted embedding data - skipping",
                            key=key_str,
                            expected_bytes=expected_bytes,
                            actual_bytes=len(stored_data),
                        )
                        errors_encountered += 1
                        continue

                    embeddings_checked += 1

                    # Deserialize embedding
                    stored_embedding = np.frombuffer(stored_data, dtype=np.float32)

                    # Validate no NaN/Inf values
                    if not np.isfinite(stored_embedding).all():
                        logger.warning(
                            "Embedding contains NaN/Inf - skipping",
                            key=key_str,
                        )
                        errors_encountered += 1
                        continue

                    # Compute similarity
                    similarity = self._cosine_similarity(embedding, stored_embedding)

                    if similarity > best_match[1]:
                        # Extract external_id from key
                        # Key format: dedup:emb:{platform}:{external_id}
                        parts = key_str.split(":")
                        if len(parts) >= 4:
                            original_id = parts[-1]
                            best_match = (original_id, similarity)

                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to process embedding - skipping",
                        key=key_str,
                        error=str(e),
                    )
                    errors_encountered += 1
                    continue

        except RedisError as e:
            logger.error(
                "Redis error during duplicate check - treating as unique",
                message_id=message.external_id,
                error=str(e),
                embeddings_checked=embeddings_checked,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return DeduplicationResult(
                is_duplicate=False,
                processing_time_ms=elapsed_ms,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Check if it's a duplicate
        is_duplicate = best_match[1] >= self.similarity_threshold

        if errors_encountered > 0:
            logger.warning(
                "Dedup check completed with errors",
                message_id=message.external_id,
                errors_encountered=errors_encountered,
                embeddings_checked=embeddings_checked,
            )

        logger.debug(
            "Dedup check complete",
            message_id=message.external_id,
            embeddings_checked=embeddings_checked,
            best_similarity=f"{best_match[1]:.3f}" if best_match[1] > 0 else "none",
            is_duplicate=is_duplicate,
            threshold=self.similarity_threshold,
        )

        return DeduplicationResult(
            is_duplicate=is_duplicate,
            duplicate_of=best_match[0] if is_duplicate else None,
            similarity=best_match[1] if best_match[1] > 0 else None,
            processing_time_ms=elapsed_ms,
        )

    async def check_duplicate(self, message: UnifiedMessage) -> DeduplicationResult:
        """Check if a message is a duplicate of a recent message.

        Args:
            message: The message to check

        Returns:
            DeduplicationResult with duplicate status and details
        """
        embedding = self._get_embedding(message.text)
        return await self._check_duplicate_with_embedding(message, embedding)

    async def _store_embedding(self, message: UnifiedMessage, embedding: np.ndarray) -> bool:
        """Store pre-computed embedding in Redis atomically.

        Uses SET with NX (only set if not exists) for atomic claim to prevent
        race conditions when similar messages arrive simultaneously.

        Args:
            message: The message being stored
            embedding: Pre-computed embedding

        Returns:
            True if stored successfully (we claimed the key), False otherwise
        """
        key = self._make_redis_key(message.external_id, message.source_platform.value)

        try:
            # Use SET with NX for atomic "claim" - prevents race conditions
            was_set = await self.redis.set(
                key,
                embedding.tobytes(),
                nx=True,  # Only set if key doesn't exist
                ex=self.ttl_seconds,
            )
            if was_set:
                logger.debug(
                    "Stored embedding",
                    key=key,
                    ttl_seconds=self.ttl_seconds,
                )
                return True
            else:
                logger.debug(
                    "Embedding already exists - likely duplicate",
                    key=key,
                    message_id=message.external_id,
                )
                return False
        except RedisError as e:
            logger.error(
                "Failed to store embedding - duplicates may not be detected",
                message_id=message.external_id,
                key=key,
                error=str(e),
            )
            return False

    async def store_message(self, message: UnifiedMessage) -> None:
        """Store message embedding in Redis for future duplicate checks.

        Args:
            message: The message to store
        """
        embedding = self._get_embedding(message.text)
        await self._store_embedding(message, embedding)

    async def process_message(self, message: UnifiedMessage) -> DeduplicationResult:
        """Store embedding first, then check for duplicates.

        Store-first pattern prevents race conditions when similar messages
        arrive simultaneously. Whoever stores first wins.

        Args:
            message: The message to process

        Returns:
            DeduplicationResult with duplicate status
        """
        start_time = time.perf_counter()

        # Generate embedding ONCE (used for both store and check)
        try:
            embedding = self._get_embedding(message.text)
        except RuntimeError as e:
            logger.error(
                "Failed to generate embedding - treating as unique",
                message_id=message.external_id,
                error=str(e),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return DeduplicationResult(
                is_duplicate=False,
                processing_time_ms=elapsed_ms,
            )

        # 1. Store embedding FIRST (atomic claim)
        stored = await self._store_embedding(message, embedding)
        if not stored:
            logger.warning(
                "Failed to store embedding - continuing with duplicate check",
                message_id=message.external_id,
            )

        # 2. Then check for duplicates (excluding self)
        result = await self._check_duplicate_with_embedding(message, embedding)

        if result.is_duplicate:
            logger.info(
                "Duplicate detected",
                message_id=message.external_id,
                duplicate_of=result.duplicate_of,
                similarity=f"{result.similarity:.3f}" if result.similarity else None,
            )
        else:
            # Log best match even when not duplicate (diagnostic)
            logger.debug(
                "Unique message",
                message_id=message.external_id,
                best_similarity=f"{result.similarity:.3f}" if result.similarity else "none",
                threshold=self.similarity_threshold,
            )

        return result

    async def cleanup(self) -> None:
        """Cleanup resources. Redis handles TTL automatically."""
        pass


async def create_deduplicator(
    redis: Redis,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    ttl_seconds: int = REDIS_TTL_SECONDS,
) -> MessageDeduplicator:
    """Create and initialize a MessageDeduplicator.

    Args:
        redis: Redis client
        similarity_threshold: Cosine similarity threshold for duplicates (0.75 default)
        ttl_seconds: TTL for stored embeddings (60 min default)

    Returns:
        Initialized MessageDeduplicator

    Raises:
        RuntimeError: If model initialization fails
    """
    deduplicator = MessageDeduplicator(
        redis=redis,
        similarity_threshold=similarity_threshold,
        ttl_seconds=ttl_seconds,
    )
    await deduplicator.initialize()
    return deduplicator
