"""Market-related constants — benchmark/sector tickers, labels, day names."""

BENCHMARK_TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP", "^VIX"]

SECTOR_TICKERS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]

SECTOR_LABELS: dict[str, str] = {
    "XLK": "Tech",
    "XLV": "Health",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLC": "Comms",
    "XLY": "Cons Disc",
    "XLP": "Cons Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
