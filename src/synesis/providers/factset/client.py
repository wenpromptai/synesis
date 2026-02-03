"""FactSet SQL Server database client.

Provides connection pooling and query execution for the FactSet database.
Uses pymssql for SQL Server connectivity.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pymssql

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class FactSetClient:
    """SQL Server client for FactSet database access.

    Manages connection pooling and provides async query execution.
    Uses synchronous pymssql under the hood with asyncio.to_thread.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        """Initialize FactSet client.

        Args:
            host: SQL Server host. If None, reads from settings.
            port: SQL Server port. If None, reads from settings.
            database: Database name. If None, reads from settings.
            user: Database user. If None, reads from settings.
            password: Database password. If None, reads from settings.
        """
        from synesis.config import get_settings

        settings = get_settings()

        self._host = host or settings.sqlserver_host
        self._port = port or settings.sqlserver_port
        self._database = database or settings.sqlserver_database
        self._user = user or settings.sqlserver_user
        self._password = password or (
            settings.sqlserver_password.get_secret_value() if settings.sqlserver_password else None
        )

        self._connection: pymssql.Connection | None = None
        self._lock = asyncio.Lock()

    def _get_connection(self) -> pymssql.Connection:
        """Get or create a database connection (synchronous)."""
        if self._connection is None:
            if not self._host:
                raise ValueError("SQLSERVER_HOST is not configured")
            if not self._database:
                raise ValueError("SQLSERVER_DATABASE is not configured")
            if not self._user:
                raise ValueError("SQLSERVER_USER is not configured")
            if not self._password:
                raise ValueError("SQLSERVER_PASSWORD is not configured")

            try:
                self._connection = pymssql.connect(
                    server=self._host,
                    port=str(self._port),
                    database=self._database,
                    user=self._user,
                    password=self._password,
                    as_dict=True,
                    login_timeout=30,
                    timeout=60,
                )
                logger.debug(f"Connected to FactSet database at {self._host}")
            except pymssql.OperationalError as e:
                logger.error(f"Failed to connect to FactSet database: {e}")
                raise ConnectionError(f"FactSet database unavailable: {e}") from e
            except pymssql.InterfaceError as e:
                logger.error(f"FactSet database interface error: {e}")
                raise ConnectionError(f"FactSet database connection failed: {e}") from e
        return self._connection

    def _execute_query_sync(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query synchronously and return results as dicts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or {})
            # With as_dict=True, fetchall returns list of dicts
            rows: list[dict[str, Any]] = cursor.fetchall()  # type: ignore[assignment]
            # Convert datetime objects to date objects where appropriate
            results: list[dict[str, Any]] = []
            for row in rows:
                converted_row: dict[str, Any] = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        converted_row[key] = value.date()
                    else:
                        converted_row[key] = value
                results.append(converted_row)
            return results
        finally:
            cursor.close()

    def _execute_scalar_sync(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a query synchronously and return a single value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or {})
            # With as_dict=True, fetchone returns dict
            row: dict[str, Any] | None = cursor.fetchone()  # type: ignore[assignment]
            if row:
                # Return the first column value
                first_value = next(iter(row.values()))
                if isinstance(first_value, datetime):
                    return first_value.date()
                return first_value
            return None
        finally:
            cursor.close()

    async def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        Args:
            query: SQL query with %(param)s placeholders
            params: Query parameters

        Returns:
            List of row dicts
        """
        async with self._lock:
            return await asyncio.to_thread(self._execute_query_sync, query, params)

    async def execute_scalar(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a query and return a single scalar value.

        Args:
            query: SQL query with %(param)s placeholders
            params: Query parameters

        Returns:
            Single value from first row, first column
        """
        async with self._lock:
            return await asyncio.to_thread(self._execute_scalar_sync, query, params)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None, None]:
        """Context manager for transaction handling.

        Note: Since we use a single connection with as_dict=True and
        autocommit behavior, this is primarily for grouping operations.
        """
        async with self._lock:
            try:
                yield
            except Exception:
                if self._connection:
                    self._connection.rollback()
                raise

    async def close(self) -> None:
        """Close the database connection."""
        async with self._lock:
            if self._connection:
                try:
                    self._connection.close()
                except Exception as e:
                    logger.warning(f"Error closing FactSet connection: {e}")
                finally:
                    self._connection = None
                    logger.debug("FactSet database connection closed")

    async def health_check(self) -> bool:
        """Check if database connection is healthy.

        Returns:
            True if connection is working, False otherwise
        """
        try:
            result = await self.execute_scalar("SELECT 1")
            return bool(result == 1)
        except Exception as e:
            logger.error(f"FactSet health check failed: {e}")
            return False


# Cache for global max date (refreshed periodically)
_max_price_date_cache: tuple[date | None, float] = (None, 0.0)
_CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours


async def get_cached_max_price_date(client: FactSetClient) -> date | None:
    """Get cached global max price date.

    This is expensive to compute (~14ms) but rarely changes (once per day).
    Cache with 4-hour TTL to avoid repeated queries.

    Args:
        client: FactSet client instance

    Returns:
        Most recent price date in the database
    """
    global _max_price_date_cache
    import time

    cached_date, cached_time = _max_price_date_cache
    current_time = time.time()

    if cached_date is not None and (current_time - cached_time) < _CACHE_TTL_SECONDS:
        return cached_date

    # Refresh cache
    from synesis.providers.factset.queries import GLOBAL_MAX_PRICE_DATE

    result = await client.execute_scalar(GLOBAL_MAX_PRICE_DATE)
    max_date: date | None = result if isinstance(result, date) else None
    if max_date:
        _max_price_date_cache = (max_date, current_time)
        logger.debug(f"Refreshed max price date cache: {max_date}")
    return max_date


def clear_max_price_date_cache() -> None:
    """Clear the max price date cache (for testing)."""
    global _max_price_date_cache
    _max_price_date_cache = (None, 0.0)


@lru_cache(maxsize=1)
def get_factset_client() -> FactSetClient:
    """Get singleton FactSet client instance.

    Returns:
        Cached FactSet client
    """
    return FactSetClient()
