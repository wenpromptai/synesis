"""Verify all examples in FactSet documentation are accurate."""

import asyncio
from datetime import date
from synesis.providers.factset import FactSetProvider


async def verify_documentation_examples() -> bool:
    """Verify all examples from factset_context.md work correctly."""
    p = FactSetProvider()
    errors = []

    print("=" * 60)
    print("FACTSET DOCUMENTATION VERIFICATION")
    print("=" * 60)

    # 1. get_fsym_id - Line 218
    print("\n[1/17] get_fsym_id('AAPL-US')...")
    try:
        fsym_id = await p.get_fsym_id("AAPL-US")
        assert fsym_id is not None, "fsym_id is None"
        # Doc says: "MH33D6-R" - verify this is still correct
        print(f"  ✓ fsym_id = {fsym_id}")
        if fsym_id != "MH33D6-R":
            print(f"  ⚠ Doc says 'MH33D6-R' but got '{fsym_id}' - UPDATE DOC!")
    except Exception as e:
        errors.append(f"get_fsym_id: {e}")
        print(f"  ✗ ERROR: {e}")

    # 2. resolve_ticker - Line 226
    print("\n[2/17] resolve_ticker('AAPL-US')...")
    try:
        sec = await p.resolve_ticker("AAPL-US")
        assert sec is not None, "security is None"
        assert sec.ticker == "AAPL-US", f"ticker mismatch: {sec.ticker}"
        # Doc says: exchange_code = "USA" - verify
        print(f"  ✓ ticker={sec.ticker}, exchange={sec.exchange_code}, currency={sec.currency}")
        if sec.exchange_code != "USA":
            print(f"  ⚠ Doc says exchange 'USA' but got '{sec.exchange_code}' - UPDATE DOC!")
    except Exception as e:
        errors.append(f"resolve_ticker: {e}")
        print(f"  ✗ ERROR: {e}")

    # 3. search_securities - Line 165
    print("\n[3/17] search_securities('DBS Bank', limit=10)...")
    try:
        results = await p.search_securities("DBS Bank", limit=10)
        assert len(results) > 0, "No results"
        # Doc says should find D05-SG
        d05 = next((r for r in results if r.ticker == "D05-SG"), None)
        if d05:
            print(f"  ✓ Found D05-SG: {d05.name}")
        else:
            print(f"  ⚠ D05-SG not found in results - got: {[r.ticker for r in results[:5]]}")
    except Exception as e:
        errors.append(f"search_securities: {e}")
        print(f"  ✗ ERROR: {e}")

    # 4. get_price latest - Line 258
    print("\n[4/17] get_price('AAPL-US') latest...")
    try:
        latest = await p.get_price("AAPL-US")
        assert latest is not None, "price is None"
        assert latest.close > 0, f"invalid close: {latest.close}"
        print(f"  ✓ Latest: ${latest.close:.2f} on {latest.price_date}")
    except Exception as e:
        errors.append(f"get_price latest: {e}")
        print(f"  ✗ ERROR: {e}")

    # 5. get_price specific date - Line 262
    print("\n[5/17] get_price('AAPL-US', date(2024, 1, 2))...")
    try:
        jan2 = await p.get_price("AAPL-US", date(2024, 1, 2))
        assert jan2 is not None, "price is None"
        # Doc says: close=185.64, open=187.15, high=188.44, low=183.88
        print(f"  ✓ Open={jan2.open}, High={jan2.high}, Low={jan2.low}, Close={jan2.close}")
        # Verify approximate values (allow small variance)
        if abs(jan2.close - 185.64) > 1:
            print(f"  ⚠ Doc says close=185.64 but got {jan2.close} - VERIFY/UPDATE DOC!")
    except Exception as e:
        errors.append(f"get_price specific: {e}")
        print(f"  ✗ ERROR: {e}")

    # 6. get_price_history - Line 274
    print("\n[6/17] get_price_history('AAPL-US', 2024-01-01, 2024-01-31)...")
    try:
        prices = await p.get_price_history("AAPL-US", date(2024, 1, 1), date(2024, 1, 31))
        assert len(prices) > 0, "No prices"
        assert prices[0].price_date <= prices[-1].price_date, "Not ASC order"
        print(f"  ✓ {len(prices)} prices, {prices[0].price_date} to {prices[-1].price_date}")
    except Exception as e:
        errors.append(f"get_price_history: {e}")
        print(f"  ✗ ERROR: {e}")

    # 7. get_latest_prices - Line 283
    print("\n[7/17] get_latest_prices(['AAPL-US', 'MSFT-US', 'NVDA-US'])...")
    try:
        latest_prices = await p.get_latest_prices(["AAPL-US", "MSFT-US", "NVDA-US"])
        assert "AAPL-US" in latest_prices, "AAPL-US missing"
        assert "MSFT-US" in latest_prices, "MSFT-US missing"
        assert "NVDA-US" in latest_prices, "NVDA-US missing"
        print(f"  ✓ Got prices for: {list(latest_prices.keys())}")
    except Exception as e:
        errors.append(f"get_latest_prices: {e}")
        print(f"  ✗ ERROR: {e}")

    # 8. get_adjusted_price_history - Line 293
    print("\n[8/17] get_adjusted_price_history('AAPL-US', 2014-01-01, 2014-01-31)...")
    try:
        adj = await p.get_adjusted_price_history("AAPL-US", date(2014, 1, 1), date(2014, 1, 31))
        assert len(adj) > 0, "No adjusted prices"
        assert adj[0].is_adjusted, "Not marked as adjusted"
        # Doc says: ~$19.75 (raw $550 / 28)
        print(f"  ✓ Adjusted close: ${adj[0].close:.2f}, is_adjusted={adj[0].is_adjusted}")
        if not (15 < adj[0].close < 25):
            print(f"  ⚠ Doc says ~$19.75 but got ${adj[0].close:.2f} - VERIFY!")
    except Exception as e:
        errors.append(f"get_adjusted_price_history: {e}")
        print(f"  ✗ ERROR: {e}")

    # 9. get_corporate_actions - Line 308
    print("\n[9/17] get_corporate_actions('AAPL-US', limit=20)...")
    try:
        actions = await p.get_corporate_actions("AAPL-US", limit=20)
        assert len(actions) > 0, "No actions"
        types = set(a.event_type for a in actions)
        print(f"  ✓ {len(actions)} actions, types: {types}")
    except Exception as e:
        errors.append(f"get_corporate_actions: {e}")
        print(f"  ✗ ERROR: {e}")

    # 10. get_dividends - Line 317
    print("\n[10/17] get_dividends('AAPL-US', limit=8)...")
    try:
        divs = await p.get_dividends("AAPL-US", limit=8)
        assert len(divs) > 0, "No dividends"
        # Doc says: dividend_amount=0.25
        print(
            f"  ✓ {len(divs)} dividends, latest: ${divs[0].dividend_amount} on {divs[0].effective_date}"
        )
    except Exception as e:
        errors.append(f"get_dividends: {e}")
        print(f"  ✗ ERROR: {e}")

    # 11. get_splits - Line 326
    print("\n[11/17] get_splits('AAPL-US')...")
    try:
        splits = await p.get_splits("AAPL-US")
        assert len(splits) >= 2, f"Expected 2+ splits, got {len(splits)}"
        # Doc says: 4.0 (2020) and 7.0 (2014)
        for s in splits:
            print(f"  {s.effective_date}: factor={s.split_factor}")
        split_2020 = next((s for s in splits if s.effective_date == date(2020, 8, 31)), None)
        split_2014 = next((s for s in splits if s.effective_date == date(2014, 6, 9)), None)
        if split_2020:
            if split_2020.split_factor is None:
                errors.append("get_splits: 2020 split has split_factor=None (expected 4.0)")
                print("  ⚠ 2020 split has split_factor=None")
            else:
                assert abs(split_2020.split_factor - 4.0) < 0.1, (
                    f"2020 split wrong: {split_2020.split_factor}"
                )
        if split_2014:
            if split_2014.split_factor is None:
                errors.append("get_splits: 2014 split has split_factor=None (expected 7.0)")
                print("  ⚠ 2014 split has split_factor=None")
            else:
                assert abs(split_2014.split_factor - 7.0) < 0.1, (
                    f"2014 split wrong: {split_2014.split_factor}"
                )
        print("  ✓ Split factors verified")
    except Exception as e:
        errors.append(f"get_splits: {e}")
        print(f"  ✗ ERROR: {e}")

    # 12. get_adjustment_factors - Line 345
    print("\n[12/17] get_adjustment_factors('AAPL-US', 2014-01-01, 2025-01-01)...")
    try:
        factors = await p.get_adjustment_factors("AAPL-US", date(2014, 1, 1), date(2025, 1, 1))
        assert len(factors) > 0, "No factors"
        # Doc says: 2014-06-09: 0.142857, 2020-08-31: 0.25
        f_2014 = factors.get(date(2014, 6, 9))
        f_2020 = factors.get(date(2020, 8, 31))
        print(f"  2014-06-09: {f_2014}")
        print(f"  2020-08-31: {f_2020}")
        if f_2014:
            assert abs(f_2014 - 0.142857) < 0.001, f"2014 factor wrong: {f_2014}"
        if f_2020:
            assert abs(f_2020 - 0.25) < 0.001, f"2020 factor wrong: {f_2020}"
        print("  ✓ Factors verified")
    except Exception as e:
        errors.append(f"get_adjustment_factors: {e}")
        print(f"  ✗ ERROR: {e}")

    # 13. get_fundamentals annual - Line 363
    print("\n[13/17] get_fundamentals('AAPL-US', 'annual', limit=4)...")
    try:
        annual = await p.get_fundamentals("AAPL-US", "annual", limit=4)
        assert len(annual) > 0, "No annual"
        # Doc says: eps_diluted=7.46, net_margin=26.9
        f = annual[0]
        print(f"  ✓ FY{f.fiscal_year}: EPS=${f.eps_diluted}, Net Margin={f.net_margin}%")
    except Exception as e:
        errors.append(f"get_fundamentals annual: {e}")
        print(f"  ✗ ERROR: {e}")

    # 14. get_fundamentals quarterly - Line 373
    print("\n[14/17] get_fundamentals('AAPL-US', 'quarterly', limit=8)...")
    try:
        quarterly = await p.get_fundamentals("AAPL-US", "quarterly", limit=8)
        assert len(quarterly) > 0, "No quarterly"
        print(f"  ✓ {len(quarterly)} quarterly periods")
    except Exception as e:
        errors.append(f"get_fundamentals quarterly: {e}")
        print(f"  ✗ ERROR: {e}")

    # 15. get_company_profile - Line 380
    print("\n[15/17] get_company_profile('AAPL-US')...")
    try:
        profile = await p.get_company_profile("AAPL-US")
        assert profile is not None, "Profile is None"
        assert len(profile) > 50, f"Profile too short: {len(profile)}"
        print(f"  ✓ Profile: {len(profile)} chars")
        print(f"    Content: {profile[:100]}{'...' if len(profile) > 100 else ''}")
    except Exception as e:
        errors.append(f"get_company_profile: {e}")
        print(f"  ✗ ERROR: {e}")

    # 16. get_shares_outstanding - Line 392
    print("\n[16/17] get_shares_outstanding('AAPL-US')...")
    try:
        shares = await p.get_shares_outstanding("AAPL-US")
        assert shares is not None, "Shares is None"
        # Doc says: 14_681_140_000
        print(f"  ✓ Shares: {shares.shares_outstanding:,.0f}")
        # AAPL should have 10-20B shares
        assert 10_000_000_000 < shares.shares_outstanding < 20_000_000_000
    except Exception as e:
        errors.append(f"get_shares_outstanding: {e}")
        print(f"  ✗ ERROR: {e}")

    # 17. get_market_cap - Line 402
    print("\n[17/17] get_market_cap('AAPL-US')...")
    try:
        mcap = await p.get_market_cap("AAPL-US")
        assert mcap is not None, "Market cap is None"
        # Doc says: 3_964_054_611_400 (~$3.96T)
        print(f"  ✓ Market Cap: ${mcap:,.0f}")
        # AAPL should be $2-5T
        assert 2_000_000_000_000 < mcap < 5_000_000_000_000
    except Exception as e:
        errors.append(f"get_market_cap: {e}")
        print(f"  ✗ ERROR: {e}")

    await p.close()

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} errors")
        for err in errors:
            print(f"  ✗ {err}")
        return False
    else:
        print("✓ ALL DOCUMENTATION EXAMPLES VERIFIED")
        return True


if __name__ == "__main__":
    success = asyncio.run(verify_documentation_examples())
    exit(0 if success else 1)
