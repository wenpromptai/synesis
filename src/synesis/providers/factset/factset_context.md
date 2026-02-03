# FactSet Provider Documentation

## All Functions at a Glance

```python
from datetime import date
from synesis.providers.factset import FactSetProvider

p = FactSetProvider()

# ══════════════════════════════════════════════════════════════════════════════
# TICKER RESOLUTION (3 functions)
# ══════════════════════════════════════════════════════════════════════════════

# 1. get_fsym_id - Get internal FactSet ID
fsym_id = await p.get_fsym_id("AAPL-US")
# → "MH33D6-R"

# 2. resolve_ticker - Get full security info
sec = await p.resolve_ticker("AAPL-US")
# → FactSetSecurity(ticker="AAPL-US", name="Apple Inc.", currency="USD", ...)

# 3. search_securities - Find tickers by name
results = await p.search_securities("DBS Bank", limit=10)
# → [FactSetSecurity(ticker="D05-SG", name="DBS Group Holdings Ltd", ...), ...]

# ══════════════════════════════════════════════════════════════════════════════
# PRICE DATA (5 functions)
# ══════════════════════════════════════════════════════════════════════════════

# 4. get_price - Single date price (default: latest)
price = await p.get_price("AAPL-US")
price = await p.get_price("AAPL-US", date(2024, 1, 2))
# → FactSetPrice(close=185.64, open=187.15, high=188.44, low=183.88, ...)

# 5. get_price_history - Date range prices
prices = await p.get_price_history("AAPL-US", date(2024, 1, 1), date(2024, 1, 31))
# → [FactSetPrice(...), FactSetPrice(...), ...] ordered by date ASC

# 6. get_latest_prices - Batch latest prices
prices = await p.get_latest_prices(["AAPL-US", "MSFT-US", "NVDA-US"])
# → {"AAPL-US": FactSetPrice(...), "MSFT-US": FactSetPrice(...), ...}

# 7. get_adjusted_price_history - Split-adjusted prices
adj_prices = await p.get_adjusted_price_history("AAPL-US", date(2014, 1, 1), date(2014, 1, 31))
# → [FactSetPrice(close=19.75, is_adjusted=True), ...] (adjusted for 7:1 and 4:1 splits)

# 8. get_adjustment_factors - Raw split/dividend factors
factors = await p.get_adjustment_factors("AAPL-US", date(2014, 1, 1), date(2025, 1, 1))
# → {date(2014, 6, 9): 0.142857, date(2020, 8, 31): 0.25, ...}

# ══════════════════════════════════════════════════════════════════════════════
# CORPORATE ACTIONS (3 functions)
# ══════════════════════════════════════════════════════════════════════════════

# 9. get_corporate_actions - All events (dividends, splits, etc.)
actions = await p.get_corporate_actions("AAPL-US", limit=20)
# → [FactSetCorporateAction(event_type="dividend", ...), ...] most recent first

# 10. get_dividends - Dividend history only
divs = await p.get_dividends("AAPL-US", limit=8)
# → [FactSetCorporateAction(dividend_amount=0.25, effective_date=...), ...]

# 11. get_splits - Split history only (20-year lookback)
splits = await p.get_splits("AAPL-US", limit=10)
# → [FactSetCorporateAction(split_factor=4.0, ...), FactSetCorporateAction(split_factor=7.0, ...)]

# ══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTALS (2 functions)
# ══════════════════════════════════════════════════════════════════════════════

# 12. get_fundamentals - EPS, margins, ratios
annual = await p.get_fundamentals("AAPL-US", "annual", limit=4)
quarterly = await p.get_fundamentals("AAPL-US", "quarterly", limit=8)
ltm = await p.get_fundamentals("AAPL-US", "ltm", limit=4)
# → [FactSetFundamentals(eps_diluted=7.46, net_margin=26.9, ...), ...]

# 13. get_company_profile - Business description
profile = await p.get_company_profile("AAPL-US")
# → "Designs, manufactures smartphones, personal computers, tablets..."

# ══════════════════════════════════════════════════════════════════════════════
# SHARES & MARKET CAP (2 functions)
# ══════════════════════════════════════════════════════════════════════════════

# 14. get_shares_outstanding - Current share count
shares = await p.get_shares_outstanding("AAPL-US")
# → FactSetSharesOutstanding(shares_outstanding=14_681_140_000, ...)

# 15. get_market_cap - shares × price
mcap = await p.get_market_cap("AAPL-US")
# → 3_964_054_611_400  (≈$3.96 trillion)

# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP (1 function)
# ══════════════════════════════════════════════════════════════════════════════

# 16. close - Close database connection
await p.close()
```

---

## Quick Start

```python
from synesis.providers.factset import FactSetProvider

async def main():
    p = FactSetProvider()

    # Get latest price for Apple
    price = await p.get_price("AAPL-US")
    print(f"AAPL-US: ${price.close}")

    await p.close()
```

---

## Ticker Resolution

### Regional Suffixes

FactSet uses **regional suffixes** to identify securities. The format is `TICKER-REGION`.

| Region | Suffix | Example |
|--------|--------|---------|
| United States | `-US` | `AAPL-US`, `MSFT-US` |
| Singapore | `-SG` | `D05-SG` (DBS Bank) |
| Hong Kong | `-HK` | `0700-HK` (Tencent) |
| Japan | `-JP` | `7203-JP` (Toyota) |
| United Kingdom | `-GB` | `SHEL-GB` (Shell) |
| Australia | `-AU` | `BHP-AU` |
| Canada | `-CA` | `TD-CA` |
| Germany | `-DE` | `SAP-DE` |
| China | `-CN` | `600519-CN` (Moutai) |
| Korea | `-KR` | `005930-KR` (Samsung) |
| Taiwan | `-TW` | `2330-TW` (TSMC) |

### Auto-Suffix Behavior

**US stocks only**: If no suffix is provided, `-US` is automatically added.

```python
# These work for US stocks:
await p.get_price("AAPL")      # Auto-adds -US
await p.get_price("AAPL-US")   # Explicit (RECOMMENDED)

# For non-US stocks, YOU MUST provide the suffix:
await p.get_price("D05-SG")    # Singapore - DBS Bank
await p.get_price("0700-HK")   # Hong Kong - Tencent
await p.get_price("D05")       # FAILS - tries D05-US which doesn't exist
```

> **Best Practice**: Always use explicit suffixes (e.g., `AAPL-US`) even for US stocks.
> This makes code self-documenting and avoids surprises.

### Finding Tickers: Use Search

If you don't know the ticker or region, use `search_securities()`:

```python
# Search finds stocks across all regions
results = await p.search_securities("DBS Bank", limit=10)
for r in results:
    print(f"{r.ticker}: {r.name} ({r.exchange_code}, {r.currency})")

# Output:
# D05-SG: DBS Group Holdings Ltd (SES, SGD)
```

### Common Workflow: Unknown Ticker

```python
async def find_and_get_price(company_name: str, region: str = None):
    """Find a company's ticker, then get its price."""
    p = FactSetProvider()

    # Step 1: Search for the company
    results = await p.search_securities(company_name, limit=20)

    # Step 2: Filter by region if specified
    if region:
        suffix = f"-{region.upper()}"
        results = [r for r in results if r.ticker and r.ticker.endswith(suffix)]

    if not results:
        print(f"No results for '{company_name}' in region {region}")
        return None

    # Step 3: Use the first match (or let user choose)
    ticker = results[0].ticker
    print(f"Found: {ticker} - {results[0].name}")

    # Step 4: Get price with explicit ticker
    price = await p.get_price(ticker)
    await p.close()
    return price

# Examples:
await find_and_get_price("DBS", region="SG")      # D05-SG
await find_and_get_price("OCBC", region="SG")     # O39-SG
await find_and_get_price("Apple", region="US")    # AAPL-US
await find_and_get_price("Samsung", region="KR")  # 005930-KR
```

---

## API Reference

### 1. Ticker/Security Resolution

#### `get_fsym_id(ticker)` → `str | None`
Get FactSet's internal security ID. Cached for performance.

```python
fsym_id = await p.get_fsym_id("AAPL-US")
# Returns: "MH33D6-R"
```

#### `resolve_ticker(ticker)` → `FactSetSecurity | None`
Get full security details including name, exchange, currency, country, sector.

```python
sec = await p.resolve_ticker("AAPL-US")
print(f"""
Ticker:   {sec.ticker}        # AAPL-US
Name:     {sec.name}          # Apple Inc.
Exchange: {sec.exchange_code} # USA
Currency: {sec.currency}      # USD
Country:  {sec.country}       # US
Sector:   {sec.sector}        # 1300
""")
```

#### `search_securities(query, limit=20)` → `list[FactSetSecurity]`
Search by company name or ticker. Works across all regions.

```python
# Find all Apple-related securities
results = await p.search_securities("Apple", limit=10)

# Find Singapore banks
results = await p.search_securities("Bank", limit=50)
sg_banks = [r for r in results if r.ticker and r.ticker.endswith("-SG")]
```

---

### 2. Price Data

#### `get_price(ticker, price_date=None)` → `FactSetPrice | None`
Get OHLCV for a single date. If no date, returns latest available.

```python
# Latest price
latest = await p.get_price("AAPL-US")
print(f"Date: {latest.price_date}, Close: ${latest.close}")

# Specific date
jan2 = await p.get_price("AAPL-US", date(2024, 1, 2))
print(f"Open: {jan2.open}, High: {jan2.high}, Low: {jan2.low}, Close: {jan2.close}")

# Pre-calculated returns (from FactSet)
print(f"1-day: {latest.one_day_pct}%, YTD: {latest.ytd_pct}%")
```

#### `get_price_history(ticker, start_date, end_date=None)` → `list[FactSetPrice]`
Get price history for date range. Ordered by date ascending.

```python
# January 2024 prices
prices = await p.get_price_history("AAPL-US", date(2024, 1, 1), date(2024, 1, 31))
for p in prices:
    print(f"{p.price_date}: ${p.close}")
```

#### `get_latest_prices(tickers)` → `dict[str, FactSetPrice]`
Batch lookup for multiple tickers.

```python
prices = await p.get_latest_prices(["AAPL-US", "MSFT-US", "NVDA-US"])
for ticker, price in prices.items():
    print(f"{ticker}: ${price.close}")
```

#### `get_adjusted_price_history(ticker, start_date, end_date=None)` → `list[FactSetPrice]`
Get split-adjusted prices for accurate historical comparisons.

```python
# Pre-2014 AAPL prices adjusted for 7:1 (2014) and 4:1 (2020) splits
adj = await p.get_adjusted_price_history("AAPL-US", date(2014, 1, 1), date(2014, 1, 31))

# Raw price was ~$550, adjusted is ~$19.75 ($550 / 28)
print(f"Adjusted close: ${adj[0].close:.2f}")
print(f"is_adjusted: {adj[0].is_adjusted}")  # True
```

---

### 3. Corporate Actions

#### `get_corporate_actions(ticker, start_date=None, limit=20)` → `list[FactSetCorporateAction]`
Get dividends, splits, and other actions. Most recent first.

```python
actions = await p.get_corporate_actions("AAPL-US", limit=20)
for a in actions:
    print(f"{a.effective_date}: {a.event_type} ({a.event_code})")
```

#### `get_dividends(ticker, limit=10)` → `list[FactSetCorporateAction]`
Get dividend history only.

```python
divs = await p.get_dividends("AAPL-US", limit=8)  # ~2 years quarterly
for d in divs:
    print(f"{d.effective_date}: ${d.dividend_amount} ({d.dividend_type})")
```

#### `get_splits(ticker, limit=10)` → `list[FactSetCorporateAction]`
Get stock split history (lookback: 20 years).

```python
splits = await p.get_splits("AAPL-US")
for s in splits:
    print(f"{s.effective_date}: {s.split_factor}:1 split")
    # 2020-08-31: 4.0:1 split (Forward Split)
    # 2014-06-09: 7.0:1 split (Bonus Issue: 6 new for 1 existing)
```

**Note on split_factor:**
- **Forward Split (FSP)**: `split_factor = new_term / old_term` (4:1 → 4.0)
- **Bonus Issue (BNS)**: `split_factor = (old_term + new_term) / old_term` (6 new for 1 → 7.0)

---

### 4. Adjustment Factors

#### `get_adjustment_factors(ticker, start_date, end_date)` → `dict[date, float]`
Get raw adjustment factors for manual price adjustment.

```python
factors = await p.get_adjustment_factors("AAPL-US", date(2014, 1, 1), date(2025, 1, 1))

# Split dates have factors:
# 2014-06-09: 0.142857 (1/7 for 7:1 split)
# 2020-08-31: 0.25 (1/4 for 4:1 split)
```

**Usage**: Multiply historical price by cumulative factor of all splits AFTER that date.

---

### 5. Fundamentals

#### `get_fundamentals(ticker, period_type="annual", limit=4)` → `list[FactSetFundamentals]`
Get EPS, margins, ratios. Period types: `annual`, `quarterly`, `ltm`.

```python
# Last 4 years annual
annual = await p.get_fundamentals("AAPL-US", "annual", limit=4)
for f in annual:
    print(f"""
FY{f.fiscal_year}:
  EPS:        ${f.eps_diluted:.2f}
  Net Margin: {f.net_margin:.1f}%
  ROE:        {f.roe:.1f}%
""")

# Last 8 quarters
quarterly = await p.get_fundamentals("AAPL-US", "quarterly", limit=8)
```

#### `get_company_profile(ticker)` → `str | None`
Get company business description.

```python
profile = await p.get_company_profile("AAPL-US")
# "Designs, manufactures smartphones, personal computers, tablets..."
```

---

### 6. Shares & Market Cap

#### `get_shares_outstanding(ticker)` → `FactSetSharesOutstanding | None`
Get current shares outstanding (adjusted, actual count not millions).

```python
shares = await p.get_shares_outstanding("AAPL-US")
print(f"Shares: {shares.shares_outstanding:,.0f}")  # ~15 billion
print(f"Report date: {shares.report_date}")
print(f"Has ADR: {shares.has_adr}")
```

#### `get_market_cap(ticker)` → `float | None`
Calculate market cap = shares × latest price.

```python
mcap = await p.get_market_cap("AAPL-US")
print(f"Market Cap: ${mcap:,.0f}")  # ~$3.9 trillion
```

---

## Complete Example

```python
import asyncio
from datetime import date
from synesis.providers.factset import FactSetProvider

async def analyze_stock(ticker: str):
    p = FactSetProvider()

    # 1. Resolve ticker and get basic info
    sec = await p.resolve_ticker(ticker)
    if not sec:
        print(f"Ticker {ticker} not found")
        return

    print(f"=== {sec.name} ({sec.ticker}) ===")
    print(f"Exchange: {sec.exchange_code}, Currency: {sec.currency}")

    # 2. Get latest price
    price = await p.get_price(ticker)
    print(f"\nLatest Price: ${price.close:.2f} ({price.price_date})")
    print(f"YTD Return: {price.ytd_pct:.1f}%")

    # 3. Get market cap
    mcap = await p.get_market_cap(ticker)
    print(f"Market Cap: ${mcap/1e12:.2f}T")

    # 4. Get fundamentals
    fundamentals = await p.get_fundamentals(ticker, "annual", limit=1)
    if fundamentals:
        f = fundamentals[0]
        print(f"\nFY{f.fiscal_year} Fundamentals:")
        print(f"  EPS: ${f.eps_diluted:.2f}")
        print(f"  Net Margin: {f.net_margin:.1f}%")

    # 5. Get recent dividends
    divs = await p.get_dividends(ticker, limit=4)
    if divs:
        annual_div = sum(d.dividend_amount or 0 for d in divs)
        yield_pct = (annual_div / price.close) * 100
        print(f"\nDividend Yield: {yield_pct:.2f}%")

    await p.close()

# Run for US stock (always use explicit suffix)
asyncio.run(analyze_stock("AAPL-US"))

# Run for Singapore stock
asyncio.run(analyze_stock("D05-SG"))
```

---

## Data Models

### FactSetPrice
```python
class FactSetPrice:
    fsym_id: str
    price_date: date
    open: float | None
    high: float | None
    low: float | None
    close: float
    volume: float | None
    is_adjusted: bool  # True if split-adjusted

    # Pre-calculated returns from FactSet
    one_day_pct: float | None
    wtd_pct: float | None    # Week-to-date
    mtd_pct: float | None    # Month-to-date
    qtd_pct: float | None    # Quarter-to-date
    ytd_pct: float | None    # Year-to-date
    one_mth_pct: float | None
    three_mth_pct: float | None
    six_mth_pct: float | None
    one_yr_pct: float | None
    # ... more return periods
```

### FactSetSecurity
```python
class FactSetSecurity:
    fsym_id: str              # Internal ID (e.g., "MH33D6-R")
    fsym_security_id: str     # Security-level ID
    ticker: str               # With region (e.g., "AAPL-US")
    name: str                 # Company name
    exchange_code: str        # Exchange (e.g., "NAS", "SES")
    security_type: str        # Type (e.g., "SHARE")
    currency: str             # Trading currency
    country: str | None       # Country code
    sector: str | None        # GICS sector code
    industry: str | None      # Industry code
```

### FactSetCorporateAction
```python
class FactSetCorporateAction:
    fsym_id: str
    event_type: str           # "dividend", "split", "rights", "other"
    event_code: str           # "DVC", "FSP", "RSP", "BNS"
    effective_date: date

    # Dividend fields
    dividend_amount: float | None
    dividend_currency: str | None
    dividend_type: str | None

    # Split fields
    split_factor: float | None  # Effective multiplier (e.g., 7.0 for 7:1)
    split_from: float | None    # Original term
    split_to: float | None      # New term
```

### FactSetFundamentals
```python
class FactSetFundamentals:
    fsym_id: str
    period_end: date
    fiscal_year: int | None
    period_type: str          # "annual", "quarterly", "ltm"

    eps_diluted: float | None
    bps: float | None         # Book value per share
    dps: float | None         # Dividends per share
    roe: float | None
    roa: float | None
    net_margin: float | None
    gross_margin: float | None
    operating_margin: float | None
    debt_to_equity: float | None
    debt_to_assets: float | None
    ev_to_ebitda: float | None
    price_to_book: float | None
    price_to_sales: float | None
```

---

## Error Handling

All methods return `None` (or empty list) when data is not found:

```python
# Invalid ticker
price = await p.get_price("INVALID123")  # Returns None

# No data for date
price = await p.get_price("AAPL-US", date(1900, 1, 1))  # Returns None

# Check before using
price = await p.get_price("AAPL-US")
if price:
    print(f"${price.close}")
else:
    print("No price data available")
```

---

## Performance Notes

1. **Caching**: `get_fsym_id()` and `resolve_ticker()` cache results in memory
2. **Batch operations**: Use `get_latest_prices()` for multiple tickers
3. **Date filtering**: All queries use date filters to avoid full table scans
4. **Connection**: Single connection per `FactSetProvider` instance. Call `close()` when done.
