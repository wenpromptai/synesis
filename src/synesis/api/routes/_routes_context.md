# API Routes Reference

Base URL: `http://localhost:7337/api/v1`

All endpoints are rate-limited per IP via slowapi.

| Route Group | Limit | Source |
|-------------|-------|--------|
| `/system/*` | 60/min | Local |
| `/fh` REST (`/`, `/{ticker}`) | 60/min | Finnhub free tier (60/min) |
| `/fh/ticker/verify`, `/fh/ticker/search` | 120/min | Local Redis/memory cache |
| `/fh` WS reads, subscriptions | 120/min | Local Redis cache |
| `/fh` subscribe/unsubscribe | 60/min | Local WebSocket mgmt |
| `/yf/*` (quote, history, FX, expirations) | 30/min | yfinance (no official limit) |
| `/yf/options/{ticker}/chain` | 10/min | Heavy (chain + optional Greeks) |
| `/yf/options/{ticker}/snapshot` | 10/min | Heavy (quote + history + chain) |
| `/watchlist` reads | 60/min | Local Redis/PG |
| `/watchlist` writes (add, delete, cleanup) | 10/min | Local Redis/PG |
| `/earnings/*` | 30/min | NASDAQ (free, ~200/min) |
| `/sec_edgar` (filings, insiders, sentiment, search) | 60/min | SEC EDGAR (10 req/sec) |
| `/sec_edgar/earnings`, `/sec_edgar/earnings/latest` | 10/min | SEC + Crawl4AI |
| `/fred` search, observations, release series/dates | 30/min | FRED API (120 req/min) |
| `/fred` series info, single release | 60/min | FRED API (120 req/min) |
| `/market/brief` | 5/min | Local (triggers background job) |

---

## System (`/system`)

### GET `/system/status`
Agent and ingestion status. No params.
```
curl localhost:7337/api/v1/system/status
```
```json
{"telegram": false, "agent_running": true}
```

### GET `/system/config`
Current runtime config. No params.
```
curl localhost:7337/api/v1/system/config
```
```json
{"env": "development", "llm_provider": "openai", "telegram_enabled": false}
```

---

## Finnhub (`/fh`)

Finnhub ticker verification, symbol search, and real-time prices. Ticker endpoints use a bulk US symbol list cached in Redis/memory (no per-request Finnhub call). Price REST endpoints hit Finnhub API directly; WebSocket endpoints read from a local cache fed by Finnhub's streaming WebSocket.

### GET `/fh/ticker/verify/{ticker}`
Check whether a ticker exists on a major US exchange. Uses bulk symbol list cached in Redis (24h) and memory — no Finnhub API call per request. Returns `valid` (bool) and `company_name`.

| Param | Type | Description |
|-------|------|-------------|
| `ticker` | path | Ticker symbol to verify (e.g. `AAPL`, `TSLA`) |

```
curl localhost:7337/api/v1/fh/ticker/verify/AAPL
```
```json
{"valid": true, "company_name": "APPLE INC"}
```

```
curl localhost:7337/api/v1/fh/ticker/verify/ZZZZ
```
```json
{"valid": false, "company_name": null}
```

### GET `/fh/ticker/search?q=`
Search for stock symbols matching a query. Results filtered to common stocks, ETFs, ADRs, REITs.

| Param | Type | Description |
|-------|------|-------------|
| `q` | query | **required**. Search text (ticker or company name) |

```
curl "localhost:7337/api/v1/fh/ticker/search?q=apple"
```
```json
{"results": [{"symbol": "AAPL", "description": "APPLE INC", "type": "Common Stock"}], "count": 1}
```

### GET `/fh/{ticker}`
Full quote from Finnhub REST API. Path param: any stock ticker.
```
curl localhost:7337/api/v1/fh/AAPL
```
```json
{"ticker": "AAPL", "current": 264.72, "change": 0.54, "percent_change": 0.2044, "high": 266.53, "low": 260.2, "open": 262.41, "previous_close": 264.18, "timestamp": 1772485200}
```

### GET `/fh?tickers={csv}`
Batch quotes. Query param `tickers`: comma-separated ticker symbols.
```
curl "localhost:7337/api/v1/fh?tickers=AAPL,TSLA"
curl "localhost:7337/api/v1/fh?tickers=AAPL,NVDA,MSFT"
```
```json
{
    "quotes": {
        "AAPL": {"current": 264.72, "change": 0.54, "percent_change": 0.2044, "high": 266.53, "low": 260.2, "open": 262.41, "previous_close": 264.18, "timestamp": 1772485200},
        "NVDA": {"current": 182.48, "change": 5.29, "percent_change": 2.9855, "high": 183.46, "low": 174.64, "open": 175.01, "previous_close": 177.19, "timestamp": 1772485200},
        "MSFT": {"current": 398.55, "change": 5.81, "percent_change": 1.4794, "high": 401.19, "low": 390.63, "open": 392.855, "previous_close": 392.74, "timestamp": 1772485200}
    },
    "found": 3,
    "missing": []
}
```

### GET `/fh/subscriptions`
List current WebSocket subscriptions. No params.
```
curl localhost:7337/api/v1/fh/subscriptions
```
```json
{"subscribed_tickers": [], "count": 0, "ws_connected": true, "max_symbols": 50}
```

### POST `/fh/subscribe`
Subscribe tickers to WebSocket stream. Body: `{"tickers": ["AAPL", "TSLA"]}`. Max 50 symbols total.
```
curl -X POST localhost:7337/api/v1/fh/subscribe \
  -H "Content-Type: application/json" \
  -d '{"tickers":["AAPL","TSLA"]}'
```
```json
{"subscribed": ["AAPL", "TSLA"], "total": 2}
```

### POST `/fh/unsubscribe`
Unsubscribe tickers from WebSocket stream. Body: `{"tickers": ["TSLA"]}`.
```
curl -X POST localhost:7337/api/v1/fh/unsubscribe \
  -H "Content-Type: application/json" \
  -d '{"tickers":["TSLA"]}'
```
```json
{"unsubscribed": ["TSLA"], "total": 1}
```

### GET `/fh/ws/prices?tickers={csv}`
Batch cached prices from WebSocket stream. Tickers must be subscribed first. Query param `tickers`: comma-separated.
```
curl "localhost:7337/api/v1/fh/ws/prices?tickers=AAPL,TSLA"
```
```json
{"prices": {"AAPL": 264.72}, "found": 1, "missing": ["TSLA"]}
```

### GET `/fh/ws/prices/{ticker}`
Single ticker from WebSocket cache. Must be subscribed. Returns 404 if not subscribed.
```
curl localhost:7337/api/v1/fh/ws/prices/AAPL
```
```json
{"ticker": "AAPL", "price": 264.72}
```

---

## yfinance (`/yf`)

Free data from Yahoo Finance. ~15 min delayed for US equities during market hours. FX/crypto near real-time. No API key. Works with stocks, ETFs, indices, crypto (`BTC-USD`), FX pairs (`EURUSD=X`).

### GET `/yf/quote/{ticker}`
Snapshot quote. Path param: any ticker — stocks, ETFs, crypto.

| Param | Type | Description |
|-------|------|-------------|
| `ticker` | path | Ticker symbol (e.g. `AAPL`, `SPY`, `BTC-USD`) |

**Stock:**
```
curl localhost:7337/api/v1/yf/quote/AAPL
```
```json
{"ticker": "AAPL", "name": "Apple Inc.", "currency": "USD", "exchange": "NMS", "last": 264.72, "prev_close": 264.18, "open": 262.44, "high": 266.53, "low": 260.2, "volume": 41576035, "market_cap": 3890834833408.0, "avg_50d": 265.3786, "avg_200d": 242.9163}
```

**ETF:**
```
curl localhost:7337/api/v1/yf/quote/SPY
```
```json
{"ticker": "SPY", "name": "State Street SPDR S&P 500 ETF T", "currency": "USD", "exchange": "PCX", "last": 686.38, "prev_close": 685.99, "open": 678.7, "high": 688.62, "low": 678.031, "volume": 83702843, "market_cap": 629947236352.0, "avg_50d": 687.7826, "avg_200d": 654.1827}
```

**Crypto:**
```
curl localhost:7337/api/v1/yf/quote/BTC-USD
```
```json
{"ticker": "BTC-USD", "name": "Bitcoin USD", "currency": "USD", "exchange": "CCC", "last": 66607.36, "prev_close": 68804.586, "open": 68804.586, "high": 69158.99, "low": 66607.36, "volume": 55707901952, "market_cap": 1331960217600.0, "avg_50d": 77712.336, "avg_200d": 97086.04}
```

### GET `/yf/history/{ticker}?period=&interval=`
OHLCV history bars.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ticker` | path | — | Ticker symbol |
| `period` | query | `1mo` | `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max` |
| `interval` | query | `1d` | `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo` |

**Max period per interval:**

| Interval | Max Period | Example Bar Count |
|----------|-----------|-------------------|
| `1m`     | `7d`      | ~389/day |
| `2m`     | `60d`     | ~195/day |
| `5m`     | `60d`     | ~78/day |
| `15m`    | `60d`     | ~26/day |
| `30m`    | `60d`     | ~13/day |
| `60m`    | `730d`    | ~7/day |
| `90m`    | `60d`     | ~5/day |
| `1h`     | `730d`    | ~7/day |
| `1d`     | `max`     | ~251/year |
| `5d`     | `max`     | ~52/year |
| `1wk`    | `max`     | ~52/year |
| `1mo`    | `max`     | 12/year |
| `3mo`    | `max`     | 4/year |

**Daily bars (default):**
```
curl "localhost:7337/api/v1/yf/history/AAPL?period=5d&interval=1d"
```
```json
{
    "ticker": "AAPL", "period": "5d", "interval": "1d",
    "bars": [
        {"date": "2026-02-24", "open": 267.86, "high": 274.89, "low": 267.71, "close": 272.14, "volume": 47014600},
        {"date": "2026-02-25", "open": 271.78, "high": 274.94, "low": 271.05, "close": 274.23, "volume": 33714300}
    ],
    "count": 5
}
```

**Intraday 5-minute bars:**
```
curl "localhost:7337/api/v1/yf/history/AAPL?period=1d&interval=5m"
```
```json
{
    "ticker": "AAPL", "period": "1d", "interval": "5m",
    "bars": [{"date": "2026-03-02", "open": 262.44, "high": 263.12, "low": 260.2, "close": 262.15, "volume": 3605326}],
    "count": 78
}
```

**Intraday 1-minute bars:**
```
curl "localhost:7337/api/v1/yf/history/AAPL?period=1d&interval=1m"
```
```json
{"ticker": "AAPL", "period": "1d", "interval": "1m", "bars": [{"date": "2026-03-02", "open": 262.44, "high": 262.46, "low": 260.2, "close": 260.65, "volume": 3173412}], "count": 389}
```

**Hourly bars:**
```
curl "localhost:7337/api/v1/yf/history/AAPL?period=5d&interval=1h"
```
```json
{"ticker": "AAPL", "period": "5d", "interval": "1h", "bars": [{"date": "2026-02-24", "open": 268.0, "high": 274.89, "low": 267.74, "close": 273.67, "volume": 8122809}], "count": 35}
```

**Weekly bars:**
```
curl "localhost:7337/api/v1/yf/history/TSLA?period=6mo&interval=1wk"
```
```json
{"ticker": "TSLA", "period": "6mo", "interval": "1wk", "bars": [{"date": "2025-09-01", "open": 328.23, "high": 355.87, "low": 325.6, "close": 350.84, "volume": 316826100}], "count": 27}
```

**Monthly bars (max history):**
```
curl "localhost:7337/api/v1/yf/history/MSFT?period=max&interval=1mo"
```
```json
{"ticker": "MSFT", "period": "max", "interval": "1mo", "bars": [{"date": "1986-03-01", "open": 0.054, "high": 0.063, "low": 0.054, "close": 0.058, "volume": 1857052800}], "count": 481}
```

**Quarterly bars (max history):**
```
curl "localhost:7337/api/v1/yf/history/AAPL?period=max&interval=3mo"
```
```json
{"ticker": "AAPL", "period": "max", "interval": "3mo", "bars": [{"date": "1984-12-01", "open": 0.1, "high": 0.11, "low": 0.08, "close": 0.08, "volume": 11099804800}], "count": 166}
```

### GET `/yf/fx/{pair}`
FX spot rate. Path param: Yahoo Finance FX pair format.

| Param | Type | Description |
|-------|------|-------------|
| `pair` | path | FX pair (e.g. `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`, `SGDUSD=X`). The `=X` suffix is required. URL-encode as `%3DX`. |

```
curl "localhost:7337/api/v1/yf/fx/EURUSD%3DX"
```
```json
{"pair": "EURUSD=X", "rate": 1.1626556, "bid": 1.1799409, "ask": 1.1792454}
```

```
curl "localhost:7337/api/v1/yf/fx/GBPUSD%3DX"
```
```json
{"pair": "GBPUSD=X", "rate": 1.329363, "bid": 1.3460037, "ask": 1.345913}
```

```
curl "localhost:7337/api/v1/yf/fx/USDJPY%3DX"
```
```json
{"pair": "USDJPY=X", "rate": 157.763, "bid": 156.035, "ask": 156.07}
```

### GET `/yf/options/{ticker}/expirations`
List available options expiration dates. No query params.

| Param | Type | Description |
|-------|------|-------------|
| `ticker` | path | Stock ticker (must have listed options) |

```
curl localhost:7337/api/v1/yf/options/AAPL/expirations
```
```json
{
    "ticker": "AAPL",
    "expirations": ["2026-03-04", "2026-03-06", "2026-03-09", "2026-03-11", "2026-03-13", "2026-03-20", "2026-03-27", "2026-04-02", "2026-04-10", "2026-04-17", "2026-05-15", "2026-06-18", "2026-07-17", "2026-08-21", "2026-09-18", "2026-10-16", "2026-11-20", "2026-12-18", "2027-01-15", "2027-03-19", "2027-06-17", "2027-12-17", "2028-01-21", "2028-03-17", "2028-12-15"],
    "count": 25
}
```

### GET `/yf/options/{ticker}/chain?expiration=&greeks=`
Full options chain (calls + puts).

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ticker` | path | — | Stock ticker |
| `expiration` | query | **required** | Expiration date `YYYY-MM-DD` (must be from `/expirations`) |
| `greeks` | query | `false` | Set `true` to compute Black-Scholes Greeks (delta, gamma, theta, vega, rho) |

**Without Greeks:**
```
curl "localhost:7337/api/v1/yf/options/AAPL/chain?expiration=2026-03-06&greeks=false"
```
```json
{
    "ticker": "AAPL", "expiration": "2026-03-06",
    "calls": [{"contract_symbol": "AAPL260306C00165000", "strike": 165.0, "last_price": 99.65, "bid": 0.0, "ask": 0.0, "volume": 1, "open_interest": 0, "implied_volatility": 0.00001, "in_the_money": true, "greeks": null}],
    "puts": [{"contract_symbol": "AAPL260306P00165000", "strike": 165.0, "last_price": 0.01, "bid": 0.0, "ask": 0.01, "volume": 1, "open_interest": 100, "implied_volatility": 0.5, "in_the_money": false, "greeks": null}]
}
```

**With Greeks:**
```
curl "localhost:7337/api/v1/yf/options/AAPL/chain?expiration=2026-04-17&greeks=true"
```
```json
{
    "ticker": "AAPL", "expiration": "2026-04-17",
    "calls": [{
        "contract_symbol": "AAPL260417C00125000", "strike": 125.0, "last_price": 123.33, "bid": 140.0, "ask": 143.05, "volume": 1, "open_interest": 6, "implied_volatility": 1.3005, "in_the_money": true,
        "greeks": {"delta": 0.970285, "gamma": 0.000558, "theta": -0.106384, "vega": 0.062745, "rho": 0.141431, "implied_volatility": 1.300541}
    }],
    "puts": [{
        "contract_symbol": "AAPL260417P00120000", "strike": 120.0, "last_price": 0.03, "bid": 0.0, "ask": 0.0, "volume": 2, "open_interest": 0, "implied_volatility": 0.500005, "in_the_money": false,
        "greeks": {"delta": -0.000002, "gamma": 0.0, "theta": -0.000005, "vega": 0.000008, "rho": -0.000001, "implied_volatility": 0.500005}
    }]
}
```

### GET `/yf/options/{ticker}/snapshot?greeks=`
Pre-computed options snapshot: spot price, 30d realized vol, nearest valid expiry (skips <7 DTE), ATM ±10 strikes per side.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `ticker` | path | — | Stock ticker |
| `greeks` | query | `true` | Compute Black-Scholes Greeks |

```
curl "localhost:7337/api/v1/yf/options/AAPL/snapshot"
```
```json
{
    "ticker": "AAPL", "spot": 264.72, "realized_vol_30d": 0.3245, "expiration": "2026-03-20", "days_to_expiry": 15,
    "calls": [{"contract_symbol": "AAPL260320C00265000", "strike": 265.0, "last_price": 5.50, "bid": 5.30, "ask": 5.70, "volume": 1200, "open_interest": 8500, "implied_volatility": 0.28, "in_the_money": false, "greeks": {"delta": 0.48, "gamma": 0.02, "theta": -0.15, "vega": 0.35, "rho": 0.05, "implied_volatility": 0.28}}],
    "puts": [{"contract_symbol": "AAPL260320P00265000", "strike": 265.0, "last_price": 5.80, "bid": 5.60, "ask": 6.00, "volume": 900, "open_interest": 7200, "implied_volatility": 0.29, "in_the_money": true, "greeks": {"delta": -0.52, "gamma": 0.02, "theta": -0.14, "vega": 0.35, "rho": -0.05, "implied_volatility": 0.29}}]
}
```

---

## Watchlist (`/watchlist`)

CRUD for tracked tickers (Redis + PostgreSQL). Tickers auto-expire after `ttl_days` (default 7).

### GET `/watchlist/`
List all watched tickers. No params. Returns `[]` if empty.
```
curl localhost:7337/api/v1/watchlist/
```
```json
["AAPL", "TSLA", "NVDA"]
```

### GET `/watchlist/stats`
Summary stats. No params.
```
curl localhost:7337/api/v1/watchlist/stats
```
```json
{"total_tickers": 0, "sources": {}, "ttl_days": 7}
```

### GET `/watchlist/detailed`
All tickers with full metadata. No params.
```
curl localhost:7337/api/v1/watchlist/detailed
```
```json
[{"ticker": "AAPL", "source": "api", "added_at": "2026-03-03T09:25:51Z", "last_seen_at": "2026-03-03T09:25:51Z", "mention_count": 1}]
```

### GET `/watchlist/{ticker}`
Single ticker metadata. Returns 404 if not on watchlist.

| Param | Type | Description |
|-------|------|-------------|
| `ticker` | path | Ticker symbol |

```
curl localhost:7337/api/v1/watchlist/AAPL
```
```json
{"ticker": "AAPL", "source": "api", "added_at": "2026-03-03T09:25:51Z", "last_seen_at": "2026-03-03T09:25:51Z", "mention_count": 1}
```

### POST `/watchlist/`
Add a ticker. Body: `{"ticker": "AAPL", "source": "api"}`. `source` defaults to `"api"`.
```
curl -X POST localhost:7337/api/v1/watchlist/ \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","source":"api"}'
```
```json
{"ticker": "AAPL", "is_new": true}
```

### DELETE `/watchlist/{ticker}`
Remove a ticker. Returns `204 No Content` on success, 404 if not found.
```
curl -X DELETE localhost:7337/api/v1/watchlist/AAPL
```

### POST `/watchlist/cleanup`
Remove expired tickers (older than `ttl_days`). Returns list of removed tickers.
```
curl -X POST localhost:7337/api/v1/watchlist/cleanup
```
```json
["STALE_TICKER_1"]
```

---

## Earnings (`/earnings`)

NASDAQ earnings calendar. Free, no API key.

### GET `/earnings/calendar?date=`
All earnings for a given date. Defaults to today if no date given.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `date` | query | today | Date in `YYYY-MM-DD` format |

```
curl "localhost:7337/api/v1/earnings/calendar?date=2026-03-03"
```
```json
{
    "date": "2026-03-03",
    "earnings": [
        {"ticker": "CRWD", "company_name": "CrowdStrike Holdings, Inc.", "earnings_date": "2026-03-03", "time": "after-hours", "eps_forecast": null, "num_estimates": 14, "market_cap": 93775577711.0, "fiscal_quarter": "Jan/2026"},
        {"ticker": "TGT", "company_name": "Target Corporation", "earnings_date": "2026-03-03", "time": "pre-market", "eps_forecast": null, "num_estimates": 13, "market_cap": 51524842304.0, "fiscal_quarter": "Jan/2026"}
    ],
    "count": 83
}
```

```
curl localhost:7337/api/v1/earnings/calendar
```
Returns today's earnings (same format).

### GET `/earnings/upcoming?days=`
Upcoming earnings for **all watchlist tickers**. Empty if watchlist is empty.

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `days` | query | `14` | 1-90 | Days to look ahead |

```
curl "localhost:7337/api/v1/earnings/upcoming?days=30"
```
```json
{"tickers_checked": 3, "earnings": [{"ticker": "AAPL", "company_name": "Apple Inc.", "earnings_date": "2026-04-24", "time": "after-hours", "eps_forecast": null, "num_estimates": 10, "market_cap": 3890834833408.0, "fiscal_quarter": "Mar/2026"}], "count": 1}
```

### GET `/earnings/upcoming/{ticker}?days=`
Next earnings for a **specific** ticker.

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `ticker` | path | — | — | Ticker symbol |
| `days` | query | `14` | 1-90 | Days to look ahead |

```
curl "localhost:7337/api/v1/earnings/upcoming/AAPL?days=90"
```
```json
{"ticker": "AAPL", "next_earnings": null, "all_in_range": []}
```

If found:
```json
{"ticker": "AAPL", "next_earnings": {"ticker": "AAPL", "company_name": "Apple Inc.", "earnings_date": "2026-04-24", "time": "after-hours", "eps_forecast": null, "num_estimates": 10, "market_cap": 3890834833408.0, "fiscal_quarter": "Mar/2026"}, "all_in_range": [...]}
```

---

## SEC EDGAR (`/sec_edgar`)

SEC filings, insider transactions, sentiment, full-text search. Free, no API key.

### GET `/sec_edgar/filings?ticker=&forms=&limit=`
Recent SEC filings. Filter by form type.

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `ticker` | query | **required** | — | Stock ticker |
| `forms` | query | all forms | — | Comma-separated form types: `4`, `8-K`, `10-K`, `10-Q`, `SC 13G`, etc. |
| `limit` | query | `10` | 1-50 | Max results |

**All recent filings:**
```
curl "localhost:7337/api/v1/sec_edgar/filings?ticker=AAPL&limit=2"
```
```json
{
    "ticker": "AAPL",
    "filings": [
        {"ticker": "AAPL", "form": "4", "filed_date": "2026-02-26", "accepted_datetime": "2026-02-26T18:34:19Z", "accession_number": "0001059235-26-000004", "primary_document": "xslF345X05/wk-form4_1772148856.xml", "items": "", "url": "https://www.sec.gov/Archives/edgar/data/0000320193/000105923526000004/xslF345X05/wk-form4_1772148856.xml"}
    ],
    "count": 2
}
```

**Filtered by form type (10-K and 10-Q only):**
```
curl "localhost:7337/api/v1/sec_edgar/filings?ticker=AAPL&forms=10-K,10-Q&limit=2"
```
```json
{
    "ticker": "AAPL",
    "filings": [
        {"ticker": "AAPL", "form": "10-Q", "filed_date": "2026-01-30", "accepted_datetime": "2026-01-30T06:01:32Z", "accession_number": "0000320193-26-000006", "primary_document": "aapl-20251227.htm", "items": "", "url": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019326000006/aapl-20251227.htm"},
        {"ticker": "AAPL", "form": "10-K", "filed_date": "2025-10-31", "accepted_datetime": "2025-10-31T06:01:26Z", "accession_number": "0000320193-25-000079", "primary_document": "aapl-20250927.htm", "items": "", "url": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019325000079/aapl-20250927.htm"}
    ],
    "count": 2
}
```

### GET `/sec_edgar/insiders?ticker=&limit=`
Recent insider transactions from Form 4 (buys and sells).

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `ticker` | query | **required** | — | Stock ticker |
| `limit` | query | `10` | 1-50 | Max results |

```
curl "localhost:7337/api/v1/sec_edgar/insiders?ticker=NVDA&limit=2"
```
```json
{
    "ticker": "NVDA",
    "transactions": [
        {"ticker": "NVDA", "owner_name": "Kress Colette", "owner_relationship": "Officer (EVP & Chief Financial Officer)", "transaction_date": "2026-02-04", "transaction_code": "S", "shares": 1287.0, "price_per_share": 172.543, "shares_after": 873125.0, "acquired_or_disposed": "D", "filing_date": "2026-02-06", "filing_url": "https://www.sec.gov/Archives/edgar/data/0001045810/000158867026000004/xslF345X05/wk-form4_1770415598.xml"}
    ],
    "count": 2
}
```

Note: Returns empty for tickers where recent Form 4s are option exercises/awards rather than open-market buys/sells (e.g. AAPL).

### GET `/sec_edgar/insiders/sells?ticker=&min_value=&limit=`
Insider sells only, with minimum transaction value filter.

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `ticker` | query | **required** | — | Stock ticker |
| `min_value` | query | `0` | >= 0 | Minimum `shares * price_per_share` |
| `limit` | query | `10` | 1-50 | Max results |

**All sells:**
```
curl "localhost:7337/api/v1/sec_edgar/insiders/sells?ticker=NVDA&limit=2"
```

**Sells over $100k:**
```
curl "localhost:7337/api/v1/sec_edgar/insiders/sells?ticker=NVDA&min_value=100000&limit=3"
```
```json
{
    "ticker": "NVDA",
    "sells": [
        {"ticker": "NVDA", "owner_name": "Kress Colette", "owner_relationship": "Officer (EVP & Chief Financial Officer)", "transaction_date": "2026-02-04", "transaction_code": "S", "shares": 1287.0, "price_per_share": 172.543, "shares_after": 873125.0, "acquired_or_disposed": "D", "filing_date": "2026-02-06", "filing_url": "..."},
        {"ticker": "NVDA", "owner_name": "Kress Colette", "owner_relationship": "Officer (EVP & Chief Financial Officer)", "transaction_date": "2026-02-04", "transaction_code": "S", "shares": 3524.0, "price_per_share": 173.406, "shares_after": 869601.0, "acquired_or_disposed": "D", "filing_date": "2026-02-06", "filing_url": "..."},
        {"ticker": "NVDA", "owner_name": "Kress Colette", "owner_relationship": "Officer (EVP & Chief Financial Officer)", "transaction_date": "2026-02-04", "transaction_code": "S", "shares": 6076.0, "price_per_share": 174.5304, "shares_after": 863525.0, "acquired_or_disposed": "D", "filing_date": "2026-02-06", "filing_url": "..."}
    ],
    "count": 3
}
```

### GET `/sec_edgar/sentiment?ticker=`
Computed insider sentiment from Form 4 data. `mspr` = net buy/sell ratio (-1.0 = all sells, +1.0 = all buys). Returns **404 if no insider buy/sell transactions found** for the ticker.

| Param | Type | Description |
|-------|------|-------------|
| `ticker` | query | **required**. Stock ticker |

**Ticker with insider data:**
```
curl "localhost:7337/api/v1/sec_edgar/sentiment?ticker=NVDA"
```
```json
{"ticker": "NVDA", "mspr": -1.0, "change": -20, "buy_count": 0, "sell_count": 20, "total_buy_value": 0.0, "total_sell_value": 7658827.11}
```

**Ticker without insider buy/sell data (404):**
```
curl "localhost:7337/api/v1/sec_edgar/sentiment?ticker=AAPL"
```
```json
{"detail": "No insider data for AAPL"}
```

### GET `/sec_edgar/search?query=&forms=&date_from=&date_to=&limit=`
Full-text search across SEC filings.

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `query` | query | **required** | min 1 char | Search text |
| `forms` | query | all | — | Comma-separated form types: `10-K`, `8-K`, etc. |
| `date_from` | query | none | — | Start date `YYYY-MM-DD` |
| `date_to` | query | none | — | End date `YYYY-MM-DD` |
| `limit` | query | `10` | 1-50 | Max results |

**Basic search:**
```
curl "localhost:7337/api/v1/sec_edgar/search?query=artificial+intelligence&limit=2"
```
```json
{
    "query": "artificial intelligence",
    "results": [{"entity": "Xiao-I Corp  (AIXI)  (CIK 0001935172)", "filed": "2023-04-28", "form": null, "url": "https://www.sec.gov/Archives/edgar/data/...", "description": ""}],
    "count": 2
}
```

**With form filter and date range:**
```
curl "localhost:7337/api/v1/sec_edgar/search?query=revenue+growth&forms=10-K&date_from=2025-01-01&date_to=2025-12-31&limit=2"
```
```json
{
    "query": "revenue growth",
    "results": [{"entity": "KULICKE & SOFFA INDUSTRIES INC  (KLIC)  (CIK 0000056978)", "filed": "2025-11-20", "form": null, "url": "...", "description": ""}],
    "count": 2
}
```

### GET `/sec_edgar/earnings?ticker=&limit=`
Earnings press releases (8-K Item 2.02) with full crawled HTML content. Requires Crawl4AI Docker service running.

| Param | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `ticker` | query | **required** | — | Stock ticker |
| `limit` | query | `5` | 1-20 | Max results |

```
curl "localhost:7337/api/v1/sec_edgar/earnings?ticker=AAPL&limit=1"
```

### GET `/sec_edgar/earnings/latest?ticker=`
Most recent earnings press release with full content. Returns 404 if none found. Requires Crawl4AI.

| Param | Type | Description |
|-------|------|-------------|
| `ticker` | query | **required**. Stock ticker |

```
curl "localhost:7337/api/v1/sec_edgar/earnings/latest?ticker=AAPL"
```

---

## FRED (`/fred`)

Federal Reserve Economic Data. Free API key required (register at https://fredaccount.stlouisfed.org). ~800k+ series covering macro, rates, employment, housing, inflation, trade.

### GET `/fred/search?q=&limit=&filter_variable=&filter_value=`
Search for FRED series by keyword.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | query | **required** | Search text |
| `limit` | query | `20` | Max results (1-1000) |
| `filter_variable` | query | none | Filter by: `frequency`, `units`, or `seasonal_adjustment` |
| `filter_value` | query | none | Value for filter_variable |

```
curl "localhost:7337/api/v1/fred/search?q=consumer+price+index&limit=2"
```
```json
{
    "query": "consumer price index",
    "results": [
        {"id": "CPIAUCSL", "title": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average", "frequency": "Monthly", "units": "Index 1982-1984=100", "seasonal_adjustment": "Seasonally Adjusted", "popularity": 95, "observation_start": "1947-01-01", "observation_end": "2026-01-01"},
        {"id": "CPILFESL", "title": "Consumer Price Index for All Items Less Food and Energy", "frequency": "Monthly", "units": "Index 1982-1984=100", "seasonal_adjustment": "Seasonally Adjusted", "popularity": 82}
    ],
    "count": 2
}
```

**With filter:**
```
curl "localhost:7337/api/v1/fred/search?q=GDP&filter_variable=frequency&filter_value=Quarterly&limit=2"
```

### GET `/fred/series/{series_id}`
Get metadata for a FRED series. Returns 404 if not found.

| Param | Type | Description |
|-------|------|-------------|
| `series_id` | path | FRED series ID (e.g., `CPIAUCSL`, `GDP`, `UNRATE`, `DFF`) |

```
curl localhost:7337/api/v1/fred/series/GDP
```
```json
{"id": "GDP", "title": "Gross Domestic Product", "frequency": "Quarterly", "units": "Billions of Dollars", "seasonal_adjustment": "Seasonally Adjusted Annual Rate", "last_updated": "2026-01-30 07:46:02-06", "popularity": 93, "observation_start": "1947-01-01", "observation_end": "2025-10-01"}
```

### GET `/fred/series/{series_id}/observations?start=&end=&frequency=&units=&sort_order=&limit=`
Get time-series data points for a FRED series.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `series_id` | path | — | FRED series ID |
| `start` | query | none | Start date `YYYY-MM-DD` |
| `end` | query | none | End date `YYYY-MM-DD` |
| `frequency` | query | none | Aggregation: `d`, `w`, `bw`, `m`, `q`, `sa`, `a` |
| `units` | query | none | Transform: `lin`, `chg`, `ch1`, `pch`, `pc1`, `pca`, `cch`, `cca`, `log` |
| `sort_order` | query | `asc` | `asc` or `desc` |
| `limit` | query | `100000` | Max observations (1-100000) |

**Recent GDP (last 4 quarters):**
```
curl "localhost:7337/api/v1/fred/series/GDP/observations?sort_order=desc&limit=4"
```
```json
{
    "series_id": "GDP", "title": "Gross Domestic Product", "units": "Billions of Dollars", "frequency": "Quarterly", "count": 4,
    "observations": [
        {"date": "2025-10-01", "value": 29719.921},
        {"date": "2025-07-01", "value": 29374.391},
        {"date": "2025-04-01", "value": 29087.492},
        {"date": "2025-01-01", "value": 28876.235}
    ]
}
```

**CPI with percent change transformation:**
```
curl "localhost:7337/api/v1/fred/series/CPIAUCSL/observations?units=pch&start=2025-01-01&limit=6"
```

**Fed Funds Rate (daily):**
```
curl "localhost:7337/api/v1/fred/series/DFF/observations?start=2026-02-01&limit=5"
```

### GET `/fred/releases?limit=&offset=&order_by=&sort_order=`
List all FRED releases (e.g., "Consumer Price Index", "Employment Situation").

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | query | `100` | Max results (1-1000) |
| `offset` | query | `0` | Pagination offset |
| `order_by` | query | `release_id` | `release_id` or `name` |
| `sort_order` | query | `asc` | `asc` or `desc` |

```
curl "localhost:7337/api/v1/fred/releases?limit=3"
```
```json
{
    "releases": [
        {"id": 9, "name": "Advance Monthly Sales for Retail and Food Services", "press_release": true, "link": "http://www.census.gov/retail/"},
        {"id": 10, "name": "Consumer Price Index", "press_release": true, "link": "http://www.bls.gov/cpi/"},
        {"id": 11, "name": "Employment Cost Index", "press_release": true, "link": "http://www.bls.gov/eci/"}
    ],
    "count": 3,
    "total": 300,
    "offset": 0
}
```

### GET `/fred/releases/{release_id}`
Get a single FRED release. Returns 404 if not found.

| Param | Type | Description |
|-------|------|-------------|
| `release_id` | path | FRED release ID |

```
curl localhost:7337/api/v1/fred/releases/10
```
```json
{"id": 10, "name": "Consumer Price Index", "press_release": true, "link": "http://www.bls.gov/cpi/"}
```

### GET `/fred/releases/{release_id}/series?limit=&offset=`
Get all series within a FRED release.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `release_id` | path | — | FRED release ID |
| `limit` | query | `100` | Max results (1-1000) |
| `offset` | query | `0` | Pagination offset |

```
curl "localhost:7337/api/v1/fred/releases/10/series?limit=3"
```
```json
{
    "release_id": 10,
    "series": [
        {"id": "CPIAUCSL", "title": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average", "frequency": "Monthly", "units": "Index 1982-1984=100"},
        {"id": "CPILFESL", "title": "Consumer Price Index for All Items Less Food and Energy in U.S. City Average", "frequency": "Monthly", "units": "Index 1982-1984=100"},
        {"id": "CPIUFDSL", "title": "Consumer Price Index for All Urban Consumers: Food in U.S. City Average", "frequency": "Monthly", "units": "Index 1982-1984=100"}
    ],
    "count": 3
}
```

### GET `/fred/releases/{release_id}/dates?include_future=&limit=`
Get scheduled dates for a FRED release (useful for knowing when data drops).

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `release_id` | path | — | FRED release ID |
| `include_future` | query | `true` | Include future scheduled dates |
| `limit` | query | `100` | Max results (1-1000) |

```
curl "localhost:7337/api/v1/fred/releases/10/dates?limit=3"
```
```json
{
    "release_id": 10,
    "dates": [
        {"release_id": 10, "release_name": "Consumer Price Index", "date": "2026-03-12"},
        {"release_id": 10, "release_name": "Consumer Price Index", "date": "2026-02-12"},
        {"release_id": 10, "release_name": "Consumer Price Index", "date": "2026-01-15"}
    ],
    "count": 3
}
```

### Common Series Quick Reference

| Indicator | Series ID | Example |
|-----------|-----------|---------|
| CPI (headline inflation) | `CPIAUCSL` | `/fred/series/CPIAUCSL/observations?units=pch&sort_order=desc&limit=6` |
| Core CPI (ex food/energy) | `CPILFESL` | `/fred/series/CPILFESL/observations?sort_order=desc&limit=6` |
| NFP (nonfarm payrolls) | `PAYEMS` | `/fred/series/PAYEMS/observations?units=chg&sort_order=desc&limit=6` |
| Unemployment Rate | `UNRATE` | `/fred/series/UNRATE/observations?sort_order=desc&limit=6` |
| Fed Funds Rate | `DFF` | `/fred/series/DFF/observations?start=2026-02-01&limit=5` |
| GDP | `GDP` | `/fred/series/GDP/observations?sort_order=desc&limit=4` |
| 10Y-2Y Spread | `T10Y2Y` | `/fred/series/T10Y2Y/observations?sort_order=desc&limit=5` |
| Consumer Sentiment | `UMCSENT` | `/fred/series/UMCSENT/observations?sort_order=desc&limit=6` |
| Retail Sales | `RSAFS` | `/fred/series/RSAFS/observations?sort_order=desc&limit=6` |

**Key release IDs** (for `/fred/releases/{id}/dates`): CPI = `10`, Employment Situation = `50`, GDP = `53`, PPI = `46`

**Useful `units` transforms:** `lin` (default), `chg` (change), `pch` (% change), `pc1` (% change from year ago), `log` (natural log)

---

## Market Brief (`/market`)

Daily market brief: benchmarks, sectors, top movers, and LLM-powered analysis. Scheduled at 10:30am ET daily. Can also be triggered manually.

### POST `/market/brief`
Manually trigger the daily market brief. Runs in background — returns immediately.

```
curl -X POST localhost:7337/api/v1/market/brief
```
```json
{"status": "triggered", "message": "Market brief job started in background"}
```

Returns `503` if market brief is not configured (requires Redis).
