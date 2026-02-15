"""Test FactSet ticker_region format for non-US tickers (SG, HK, etc).

Run: uv run python scripts/test_factset_regions.py
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from synesis.providers.factset.client import FactSetClient  # noqa: E402


async def main() -> None:
    client = FactSetClient()

    # Test tickers: US + SG + HK
    test_tickers = [
        ("AAPL", "US - Apple"),
        ("D05", "SG - DBS"),
        ("C38", "SG - CapitaLand"),
        ("Z74", "SG - SingTel"),
        ("0700", "HK - Tencent"),
        ("9988", "HK - Alibaba"),
        ("VOD", "UK - Vodafone"),
        ("NESN", "CH - Nestle"),
    ]

    print("=" * 80)
    print("TESTING ticker_region FORMAT IN FACTSET")
    print("=" * 80)

    # Query 1: Check exact ticker_region format for known tickers
    print("\n1. EXACT MATCH TEST (ticker_region)")
    print("-" * 80)

    for ticker, desc in test_tickers:
        # Try different region suffixes
        for suffix in ["-US", "-SG", "-HK", "-CH", "-UK", ""]:
            full_ticker = f"{ticker}{suffix}" if suffix else ticker
            query = """
                SELECT TOP 1 ticker_region, proper_name, fref_security_type
                FROM sym_v1.sym_ticker_region t WITH (NOLOCK)
                JOIN fgp_v1.fgp_sec_coverage s WITH (NOLOCK) ON t.fsym_id = s.fsym_id
                WHERE t.ticker_region = %(ticker)s
            """
            rows = await client.execute_query(query, {"ticker": full_ticker})
            if rows:
                row = rows[0]
                print(f"  {desc:20} → {row['ticker_region']:15} | {row['proper_name'][:40]}")
                break

    # Query 2: Find all regions for a bare ticker
    print("\n2. ALL REGIONS FOR BARE TICKERS")
    print("-" * 80)

    for ticker, desc in test_tickers:
        query = """
            SELECT ticker_region, proper_name, fref_security_type
            FROM sym_v1.sym_ticker_region t WITH (NOLOCK)
            JOIN fgp_v1.fgp_sec_coverage s WITH (NOLOCK) ON t.fsym_id = s.fsym_id
            WHERE t.ticker_region LIKE %(pattern)s
            ORDER BY ticker_region
        """
        rows = await client.execute_query(query, {"pattern": f"{ticker}-%"})
        if rows:
            print(f"\n  {desc} ({ticker}):")
            for row in rows[:5]:  # Show max 5
                print(f"    {row['ticker_region']:15} | {row['proper_name'][:40]}")

    # Query 3: Sample SG tickers to confirm format
    print("\n3. SAMPLE SG TICKERS (ticker_region LIKE '%-SG')")
    print("-" * 80)

    query = """
        SELECT TOP 20 ticker_region, proper_name, fref_security_type
        FROM sym_v1.sym_ticker_region t WITH (NOLOCK)
        JOIN fgp_v1.fgp_sec_coverage s WITH (NOLOCK) ON t.fsym_id = s.fsym_id
        WHERE t.ticker_region LIKE '%-SG'
          AND s.fref_security_type IN ('SHARE', 'ADR', 'GDR', 'NVDR', 'ETF_ETF')
        ORDER BY t.ticker_region
    """
    rows = await client.execute_query(query)
    for row in rows:
        print(f"  {row['ticker_region']:15} | {row['proper_name'][:40]}")

    # Query 4: Sample HK tickers
    print("\n4. SAMPLE HK TICKERS (ticker_region LIKE '%-HK')")
    print("-" * 80)

    query = """
        SELECT TOP 20 ticker_region, proper_name, fref_security_type
        FROM sym_v1.sym_ticker_region t WITH (NOLOCK)
        JOIN fgp_v1.fgp_sec_coverage s WITH (NOLOCK) ON t.fsym_id = s.fsym_id
        WHERE t.ticker_region LIKE '%-HK'
          AND s.fref_security_type IN ('SHARE', 'ADR', 'GDR', 'NVDR', 'ETF_ETF')
        ORDER BY t.ticker_region
    """
    rows = await client.execute_query(query)
    for row in rows:
        print(f"  {row['ticker_region']:15} | {row['proper_name'][:40]}")

    # Query 5: Count by region
    print("\n5. TICKER COUNT BY REGION (top 20)")
    print("-" * 80)

    query = """
        SELECT
            RIGHT(ticker_region, CHARINDEX('-', REVERSE(ticker_region)) - 1) as region,
            COUNT(*) as count
        FROM sym_v1.sym_ticker_region t WITH (NOLOCK)
        JOIN fgp_v1.fgp_sec_coverage s WITH (NOLOCK) ON t.fsym_id = s.fsym_id
        WHERE s.fref_security_type IN ('SHARE', 'ADR', 'GDR', 'NVDR', 'ETF_ETF', 'REIT', 'MLP')
          AND t.ticker_region LIKE '%-%'
        GROUP BY RIGHT(ticker_region, CHARINDEX('-', REVERSE(ticker_region)) - 1)
        ORDER BY count DESC
    """
    rows = await client.execute_query(query)
    for row in rows[:20]:
        print(f"  {row['region']:10} | {row['count']:,} tickers")

    await client.close()
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
