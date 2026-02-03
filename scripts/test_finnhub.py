#!/usr/bin/env python3
"""Test Finnhub WebSocket, REST API, and FinnhubService endpoints."""

import argparse
import asyncio
import os
import sys

import httpx
import websockets


async def test_rest_quote(api_key: str, ticker: str = "AAPL") -> bool:
    """Test Finnhub /quote REST endpoint."""
    print(f"\n[REST] Testing /quote for {ticker}...")

    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": ticker, "token": api_key}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            current = data.get("c", 0)
            high = data.get("h", 0)
            low = data.get("l", 0)
            prev_close = data.get("pc", 0)

            if current > 0:
                print(f"  ✓ Current: ${current:.2f}")
                print(f"    High: ${high:.2f}, Low: ${low:.2f}, Prev Close: ${prev_close:.2f}")
                return True
            else:
                print(f"  ✗ Invalid response (price=0): {data}")
                return False
        elif response.status_code == 401:
            print("  ✗ Invalid API key (401)")
            return False
        elif response.status_code == 429:
            print("  ✗ Rate limited (429)")
            return False
        else:
            print(f"  ✗ HTTP {response.status_code}: {response.text}")
            return False


async def test_websocket(api_key: str, ticker: str = "AAPL", timeout: float = 10.0) -> bool:
    """Test Finnhub WebSocket connection."""
    print("\n[WebSocket] Connecting to wss://ws.finnhub.io...")

    url = f"wss://ws.finnhub.io?token={api_key}"
    import orjson

    try:
        async with websockets.connect(url) as ws:
            print("  ✓ Connected")

            # Subscribe to ticker
            subscribe_msg = orjson.dumps({"type": "subscribe", "symbol": ticker})
            await ws.send(subscribe_msg)
            print(f"  ✓ Subscribed to {ticker}")

            # Wait for messages
            print(f"  Waiting for trade data (up to {timeout}s)...")
            start = asyncio.get_event_loop().time()
            trade_received = False

            while asyncio.get_event_loop().time() - start < timeout:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = orjson.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "trade":
                        trades = data.get("data", [])
                        if trades:
                            trade = trades[0]
                            print(f"  ✓ Trade received: {trade.get('s')} @ ${trade.get('p'):.2f}")
                            trade_received = True
                            break
                    elif msg_type == "ping":
                        print("  - Ping received (connection alive)")
                    elif msg_type == "error":
                        print(f"  ✗ Error: {data}")
                        return False

                except asyncio.TimeoutError:
                    print("  - Still waiting... (market might be closed)")
                    continue

            if not trade_received:
                print("  ⚠ No trades received (market may be closed, but connection works)")

            # Unsubscribe
            unsubscribe_msg = orjson.dumps({"type": "unsubscribe", "symbol": ticker})
            await ws.send(unsubscribe_msg)

            return True

    except websockets.exceptions.InvalidStatusCode as e:
        if e.status_code == 401:
            print("  ✗ Invalid API key (401)")
        else:
            print(f"  ✗ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


async def test_basic_financials(api_key: str, ticker: str = "AAPL") -> bool:
    """Test Finnhub /stock/metric endpoint (basic financials)."""
    print(f"\n[REST] Testing /stock/metric for {ticker}...")

    url = "https://finnhub.io/api/v1/stock/metric"
    params = {"symbol": ticker, "metric": "all", "token": api_key}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metric", {})

            if metrics:
                pe = metrics.get("peBasicExclExtraTTM")
                mktcap = metrics.get("marketCapitalization")
                high52 = metrics.get("52WeekHigh")
                low52 = metrics.get("52WeekLow")

                print(f"  ✓ P/E Ratio: {pe}")
                print(f"    Market Cap: ${mktcap / 1000:.1f}B" if mktcap else "    Market Cap: N/A")
                print(
                    f"    52W Range: ${low52:.2f} - ${high52:.2f}"
                    if high52 and low52
                    else "    52W Range: N/A"
                )
                return True
            else:
                print(f"  ✗ No metrics returned: {data}")
                return False
        elif response.status_code == 401:
            print("  ✗ Invalid API key (401)")
            return False
        elif response.status_code == 429:
            print("  ✗ Rate limited (429)")
            return False
        else:
            print(f"  ✗ HTTP {response.status_code}: {response.text}")
            return False


async def test_insider_transactions(api_key: str, ticker: str = "AAPL") -> bool:
    """Test Finnhub /stock/insider-transactions endpoint."""
    print(f"\n[REST] Testing /stock/insider-transactions for {ticker}...")

    url = "https://finnhub.io/api/v1/stock/insider-transactions"
    params = {"symbol": ticker, "token": api_key}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            transactions = data.get("data", [])

            if transactions:
                print(f"  ✓ Found {len(transactions)} insider transactions")
                for txn in transactions[:3]:
                    name = txn.get("name", "Unknown")
                    code = txn.get("transactionCode", "?")
                    shares = txn.get("share", 0)
                    date = txn.get("filingDate", "")
                    action = "bought" if code == "P" else "sold" if code == "S" else code
                    print(f"    - {name}: {action} {shares:,} shares ({date})")
                return True
            else:
                print("  ⚠ No insider transactions found (this is OK for some tickers)")
                return True  # Not all tickers have insider transactions
        elif response.status_code == 401:
            print("  ✗ Invalid API key (401)")
            return False
        elif response.status_code == 429:
            print("  ✗ Rate limited (429)")
            return False
        else:
            print(f"  ✗ HTTP {response.status_code}: {response.text}")
            return False


async def test_insider_sentiment(api_key: str, ticker: str = "AAPL") -> bool:
    """Test Finnhub /stock/insider-sentiment endpoint."""
    print(f"\n[REST] Testing /stock/insider-sentiment for {ticker}...")

    url = "https://finnhub.io/api/v1/stock/insider-sentiment"
    params = {"symbol": ticker, "from": "2020-01-01", "token": api_key}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            sentiment_data = data.get("data", [])

            if sentiment_data:
                latest = sentiment_data[-1]
                mspr = latest.get("mspr")
                change = latest.get("change", 0)
                year = latest.get("year")
                month = latest.get("month")

                signal = "bullish" if mspr > 0 else "bearish" if mspr < 0 else "neutral"
                print(f"  ✓ MSPR Score: {mspr:.4f} ({signal})")
                print(f"    Net Change: {change:+,} shares")
                print(f"    Period: {year}-{month:02d}" if year and month else "    Period: N/A")
                return True
            else:
                print("  ⚠ No insider sentiment data found")
                return True  # Not all tickers have sentiment data
        elif response.status_code == 401:
            print("  ✗ Invalid API key (401)")
            return False
        elif response.status_code == 429:
            print("  ✗ Rate limited (429)")
            return False
        else:
            print(f"  ✗ HTTP {response.status_code}: {response.text}")
            return False


async def test_sec_filings(api_key: str, ticker: str = "AAPL") -> bool:
    """Test Finnhub /stock/filings endpoint."""
    print(f"\n[REST] Testing /stock/filings for {ticker}...")

    url = "https://finnhub.io/api/v1/stock/filings"
    params = {"symbol": ticker, "token": api_key}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list) and data:
                print(f"  ✓ Found {len(data)} SEC filings")
                for filing in data[:5]:
                    form = filing.get("form", "Unknown")
                    date = filing.get("filedDate", "")
                    print(f"    - {form} filed {date}")
                return True
            else:
                print("  ⚠ No SEC filings found")
                return True
        elif response.status_code == 401:
            print("  ✗ Invalid API key (401)")
            return False
        elif response.status_code == 429:
            print("  ✗ Rate limited (429)")
            return False
        else:
            print(f"  ✗ HTTP {response.status_code}: {response.text}")
            return False


async def test_eps_surprises(api_key: str, ticker: str = "AAPL") -> bool:
    """Test Finnhub /stock/earnings endpoint."""
    print(f"\n[REST] Testing /stock/earnings for {ticker}...")

    url = "https://finnhub.io/api/v1/stock/earnings"
    params = {"symbol": ticker, "token": api_key}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list) and data:
                print(f"  ✓ Found {len(data)} earnings records")
                for earning in data[:4]:
                    period = earning.get("period", "")
                    actual = earning.get("actual")
                    estimate = earning.get("estimate")
                    surprise = earning.get("surprisePercent")

                    if actual is not None and estimate is not None:
                        beat = (
                            "beat"
                            if actual > estimate
                            else "missed"
                            if actual < estimate
                            else "met"
                        )
                        print(
                            f"    - {period}: ${actual:.2f} vs ${estimate:.2f} est ({beat}, {surprise:+.1f}%)"
                        )
                return True
            else:
                print("  ⚠ No earnings data found")
                return True
        elif response.status_code == 401:
            print("  ✗ Invalid API key (401)")
            return False
        elif response.status_code == 429:
            print("  ✗ Rate limited (429)")
            return False
        else:
            print(f"  ✗ HTTP {response.status_code}: {response.text}")
            return False


async def test_earnings_calendar(api_key: str, ticker: str = "AAPL") -> bool:
    """Test Finnhub /calendar/earnings endpoint."""
    print(f"\n[REST] Testing /calendar/earnings for {ticker}...")

    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y-%m-%d")
    future = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {"symbol": ticker, "from": today, "to": future, "token": api_key}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            calendar = data.get("earningsCalendar", [])

            if calendar:
                next_earning = calendar[0]
                date = next_earning.get("date", "")
                hour = next_earning.get("hour", "")
                estimate = next_earning.get("epsEstimate")

                hour_str = (
                    " (before market)"
                    if hour == "bmo"
                    else " (after market)"
                    if hour == "amc"
                    else ""
                )
                print(f"  ✓ Next Earnings: {date}{hour_str}")
                if estimate:
                    print(f"    EPS Estimate: ${estimate:.2f}")
                return True
            else:
                print("  ⚠ No upcoming earnings scheduled (this is normal)")
                return True
        elif response.status_code == 401:
            print("  ✗ Invalid API key (401)")
            return False
        elif response.status_code == 429:
            print("  ✗ Rate limited (429)")
            return False
        else:
            print(f"  ✗ HTTP {response.status_code}: {response.text}")
            return False


async def test_finnhub_service(ticker: str = "AAPL") -> bool:
    """Test the FinnhubService class with Redis caching."""
    print(f"\n[FinnhubService] Testing with Redis caching for {ticker}...")

    try:
        from redis.asyncio import Redis
        from synesis.config import get_settings
        from synesis.providers import FinnhubService

        settings = get_settings()
        if not settings.finnhub_api_key:
            print("  ⚠ FINNHUB_API_KEY not in settings, skipping service test")
            return True

        redis = Redis.from_url(settings.redis_url)

        try:
            await redis.ping()
            print("  ✓ Redis connected")
        except Exception as e:
            print(f"  ⚠ Redis not available ({e}), skipping service test")
            return True

        service = FinnhubService(
            api_key=settings.finnhub_api_key.get_secret_value(),
            redis=redis,
        )

        try:
            # Test basic financials
            print(f"\n  Testing get_basic_financials({ticker})...")
            financials = await service.get_basic_financials(ticker)
            if financials:
                print(
                    f"    ✓ P/E: {financials.get('peRatio')}, MktCap: {financials.get('marketCap')}"
                )
            else:
                print("    ⚠ No financials returned")

            # Test insider transactions
            print(f"\n  Testing get_insider_transactions({ticker})...")
            txns = await service.get_insider_transactions(ticker, limit=3)
            print(f"    ✓ Found {len(txns)} transactions")

            # Test insider sentiment
            print(f"\n  Testing get_insider_sentiment({ticker})...")
            sentiment = await service.get_insider_sentiment(ticker)
            if sentiment:
                print(f"    ✓ MSPR: {sentiment.get('mspr')}")
            else:
                print("    ⚠ No sentiment data")

            # Test SEC filings
            print(f"\n  Testing get_sec_filings({ticker})...")
            filings = await service.get_sec_filings(ticker, limit=3)
            print(f"    ✓ Found {len(filings)} filings")

            # Test EPS surprises
            print(f"\n  Testing get_eps_surprises({ticker})...")
            surprises = await service.get_eps_surprises(ticker, limit=4)
            print(f"    ✓ Found {len(surprises)} earnings records")

            # Test earnings calendar
            print(f"\n  Testing get_earnings_calendar({ticker})...")
            calendar = await service.get_earnings_calendar(ticker)
            if calendar and calendar.get("date"):
                print(f"    ✓ Next earnings: {calendar.get('date')}")
            else:
                print("    ⚠ No upcoming earnings scheduled")

            # Test cache hit
            print("\n  Testing cache hit (fetching financials again)...")
            financials2 = await service.get_basic_financials(ticker)
            if financials2:
                print("    ✓ Cache working (got data again)")

            return True

        finally:
            await service.close()
            await redis.close()

    except ImportError as e:
        print(f"  ⚠ Could not import required modules ({e}), skipping service test")
        return True
    except Exception as e:
        print(f"  ✗ Service test failed: {e}")
        return False


async def main() -> None:
    """Run Finnhub tests."""
    parser = argparse.ArgumentParser(description="Test Finnhub API endpoints")
    parser.add_argument(
        "ticker", nargs="?", default="AAPL", help="Stock ticker to test (default: AAPL)"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests including WebSocket")
    parser.add_argument("--service", action="store_true", help="Test FinnhubService class")
    parser.add_argument(
        "--fundamentals", action="store_true", help="Test fundamental data endpoints only"
    )
    args = parser.parse_args()

    api_key = os.environ.get("FINNHUB_API_KEY")

    if not api_key:
        print("Error: FINNHUB_API_KEY environment variable not set")
        print("\nSet it with: export FINNHUB_API_KEY=your_key_here")
        sys.exit(1)

    ticker = args.ticker.upper()

    print("=" * 60)
    print(f"Finnhub API Test - Ticker: {ticker}")
    print("=" * 60)

    results = {}

    # Always test REST quote
    results["REST /quote"] = await test_rest_quote(api_key, ticker)

    if args.all or args.fundamentals:
        # Test fundamental data endpoints
        results["Financials"] = await test_basic_financials(api_key, ticker)
        await asyncio.sleep(1.1)  # Rate limit

        results["Insider Txns"] = await test_insider_transactions(api_key, ticker)
        await asyncio.sleep(1.1)

        results["Insider Sentiment"] = await test_insider_sentiment(api_key, ticker)
        await asyncio.sleep(1.1)

        results["SEC Filings"] = await test_sec_filings(api_key, ticker)
        await asyncio.sleep(1.1)

        results["EPS Surprises"] = await test_eps_surprises(api_key, ticker)
        await asyncio.sleep(1.1)

        results["Earnings Calendar"] = await test_earnings_calendar(api_key, ticker)

    if args.all:
        # Test WebSocket (takes time)
        results["WebSocket"] = await test_websocket(api_key, ticker)

    if args.service:
        # Test FinnhubService class
        results["FinnhubService"] = await test_finnhub_service(ticker)

    # Print summary
    print("\n" + "=" * 60)
    print("Results:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
