"""Crawler provider implementations.

This module exports crawler implementations for fetching and parsing web content:
- Crawl4AICrawlerProvider: Uses crawl4ai service for AI-friendly web crawling
"""

from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider

__all__ = [
    "Crawl4AICrawlerProvider",
]
