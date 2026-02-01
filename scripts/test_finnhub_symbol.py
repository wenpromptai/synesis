"""Test Finnhub symbol search and ticker verification."""

import asyncio
from synesis.config import get_settings
from synesis.ingestion.finnhub import FinnhubService
from redis.asyncio import Redis


async def main():
    settings = get_settings()

    if not settings.finnhub_api_key:
        print("ERROR: FINNHUB_API_KEY not set")
        return

    redis = Redis.from_url(settings.redis_url)
    api_key = settings.finnhub_api_key.get_secret_value()

    service = FinnhubService(api_key=api_key, redis=redis)

    # Test cases
    test_tickers = [
        "AAPL",  # Valid - Apple
        "MSFT",  # Valid - Microsoft
        "GME",  # Valid - GameStop (meme stock)
        "DJT",  # Valid - Trump Media
        "HOOD",  # Valid - Robinhood
        "YOLO",  # Could be ETF or slang
        "WSB",  # Invalid - not a ticker
        "DD",  # Invalid - not a ticker
        "FAKE",  # Invalid
    ]

    print("=" * 60)
    print("Finnhub Symbol Search & Verification Test")
    print("=" * 60)

    for ticker in test_tickers:
        print(f"\n{ticker}:")

        # Test search
        results = await service.search_symbol(ticker)
        if results:
            print(f"  Search results: {len(results)} matches")
            for r in results[:3]:
                print(f"    - {r['symbol']}: {r['description']} ({r['type']})")
        else:
            print("  Search results: None")

        # Test verification
        is_valid, company = await service.verify_ticker(ticker)
        print(f"  Verified: {is_valid}")
        if company:
            print(f"  Company: {company}")

    await service.close()
    await redis.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
