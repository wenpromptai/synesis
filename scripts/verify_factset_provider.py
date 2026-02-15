#!/usr/bin/env python3
"""Comprehensive verification of all 16 FactSet provider endpoints.

Systematically verifies:
1. Functional correctness - Returns data, handles edge cases
2. Mathematical correctness - Calculations are accurate
3. Logical correctness - Queries, filters, and transformations are correct

Usage:
    uv run python scripts/verify_factset_provider.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date
from typing import Any

from synesis.providers.factset import FactSetProvider


class VerificationResult:
    """Container for verification results."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = False
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.details: dict[str, Any] = {}

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def mark_passed(self) -> None:
        if not self.errors:
            self.passed = True


async def verify_get_fsym_id(p: FactSetProvider) -> VerificationResult:
    """[1/16] Verify get_fsym_id() - ticker to fsym_id resolution."""
    result = VerificationResult("get_fsym_id")

    try:
        # Test 1: Valid ticker
        fsym_id = await p.get_fsym_id("AAPL")
        if fsym_id is None:
            result.add_error("AAPL fsym_id is None")
            return result
        result.details["aapl_fsym_id"] = fsym_id

        # Test 2: Normalization - AAPL and AAPL-US should return same id
        fsym_id_normalized = await p.get_fsym_id("AAPL-US")
        if fsym_id != fsym_id_normalized:
            result.add_error(f"Normalization mismatch: {fsym_id} != {fsym_id_normalized}")

        # Test 3: Invalid ticker should return None
        invalid_fsym_id = await p.get_fsym_id("INVALID123XYZ")
        if invalid_fsym_id is not None:
            result.add_error(f"Invalid ticker should return None, got: {invalid_fsym_id}")

        # Test 4: Test caching (second call should use cache)
        fsym_id_cached = await p.get_fsym_id("AAPL")
        if fsym_id != fsym_id_cached:
            result.add_error("Caching inconsistency")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_resolve_ticker(p: FactSetProvider) -> VerificationResult:
    """[2/16] Verify resolve_ticker() - full security info retrieval."""
    result = VerificationResult("resolve_ticker")

    try:
        # Test 1: Resolve AAPL
        sec = await p.resolve_ticker("AAPL")
        if sec is None:
            result.add_error("AAPL security is None")
            return result

        # Test 2: Verify expected fields
        if sec.ticker != "AAPL-US":
            result.add_error(f"Expected ticker AAPL-US, got {sec.ticker}")

        if sec.currency != "USD":
            result.add_error(f"Expected USD currency, got {sec.currency}")

        if sec.exchange_code not in ("NAS", "NGS"):
            result.add_warning(f"Unexpected exchange code: {sec.exchange_code}")

        if not sec.name:
            result.add_error("Security name is empty")

        if sec.security_type != "SHARE":
            result.add_warning(f"Unexpected security type: {sec.security_type}")

        result.details["security"] = {
            "fsym_id": sec.fsym_id,
            "ticker": sec.ticker,
            "name": sec.name,
            "exchange": sec.exchange_code,
            "currency": sec.currency,
            "country": sec.country,
            "sector": sec.sector,
        }

        # Test 3: Invalid ticker
        invalid_sec = await p.resolve_ticker("INVALID123XYZ")
        if invalid_sec is not None:
            result.add_error("Invalid ticker should return None")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_search_securities(p: FactSetProvider) -> VerificationResult:
    """[3/16] Verify search_securities() - LIKE pattern search."""
    result = VerificationResult("search_securities")

    try:
        # Test 1: Search for "Apple"
        results = await p.search_securities("Apple", limit=10)
        if len(results) == 0:
            result.add_error("No results for 'Apple' search")
            return result

        # Test 2: AAPL should be in results
        apple_found = any("AAPL" in (r.ticker or "") for r in results)
        if not apple_found:
            result.add_warning("AAPL not found in Apple search results")

        result.details["result_count"] = len(results)
        result.details["tickers_found"] = [r.ticker for r in results[:5]]

        # Test 3: Verify limit is respected
        limited = await p.search_securities("Inc", limit=5)
        if len(limited) > 5:
            result.add_error(f"Limit not respected: got {len(limited)} > 5")

        # Test 4: Empty search should still return results
        empty_results = await p.search_securities("XYZNONSENSE12345", limit=5)
        # This should be empty
        if len(empty_results) > 0:
            result.add_warning("Unexpected results for nonsense query")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_price(p: FactSetProvider) -> VerificationResult:
    """[4/16] Verify get_price() - single date price lookup."""
    result = VerificationResult("get_price")

    try:
        # Test 1: Get price for specific date
        test_date = date(2024, 1, 2)  # First trading day of 2024
        price = await p.get_price("AAPL", test_date)
        if price is None:
            result.add_error(f"Price is None for {test_date}")
            return result

        # Test 2: Verify OHLCV makes sense
        if price.close <= 0:
            result.add_error(f"Invalid close price: {price.close}")

        if price.open is not None and price.open <= 0:
            result.add_error(f"Invalid open price: {price.open}")

        if price.high is not None and price.low is not None:
            if price.high < price.low:
                result.add_error(f"High < Low: {price.high} < {price.low}")
            if price.high < price.close or price.low > price.close:
                result.add_warning("Close outside high-low range")

        result.details["specific_date"] = {
            "date": str(test_date),
            "open": price.open,
            "high": price.high,
            "low": price.low,
            "close": price.close,
            "volume": price.volume,
        }

        # Test 3: Latest price (None date)
        latest = await p.get_price("AAPL")
        if latest is None:
            result.add_error("Latest price is None")
        else:
            result.details["latest_price"] = {
                "date": str(latest.price_date),
                "close": latest.close,
            }

        # Test 4: Pre-calculated returns exist
        if price.one_day_pct is None:
            result.add_warning("one_day_pct is None")
        if price.ytd_pct is None:
            result.add_warning("ytd_pct is None")

        # Test 5: Invalid ticker
        invalid_price = await p.get_price("INVALID123XYZ")
        if invalid_price is not None:
            result.add_error("Invalid ticker should return None")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_price_history(p: FactSetProvider) -> VerificationResult:
    """[5/16] Verify get_price_history() - date range price retrieval."""
    result = VerificationResult("get_price_history")

    try:
        # Test 1: Get January 2024 prices
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)
        prices = await p.get_price_history("AAPL", start, end)

        if len(prices) == 0:
            result.add_error("No prices returned")
            return result

        result.details["price_count"] = len(prices)
        result.details["date_range"] = f"{prices[0].price_date} to {prices[-1].price_date}"

        # Test 2: Verify ascending order
        for i in range(1, len(prices)):
            if prices[i].price_date < prices[i - 1].price_date:
                result.add_error(
                    f"Not ascending order at {i}: {prices[i - 1].price_date} > {prices[i].price_date}"
                )
                break

        # Test 3: No duplicates
        dates = [p.price_date for p in prices]
        if len(dates) != len(set(dates)):
            result.add_error("Duplicate dates found")

        # Test 4: All prices valid
        invalid_prices = [p for p in prices if p.close <= 0]
        if invalid_prices:
            result.add_error(f"Found {len(invalid_prices)} prices with close <= 0")

        # Test 5: Approximate trading days in January (should be ~21-22)
        if len(prices) < 15 or len(prices) > 25:
            result.add_warning(f"Unexpected number of trading days: {len(prices)}")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_latest_prices(p: FactSetProvider) -> VerificationResult:
    """[6/16] Verify get_latest_prices() - batch price lookup."""
    result = VerificationResult("get_latest_prices")

    try:
        # Test 1: Get prices for multiple tickers
        tickers = ["AAPL", "MSFT", "NVDA"]
        prices = await p.get_latest_prices(tickers)

        if "AAPL" not in prices:
            result.add_error("AAPL missing from results")
        if "MSFT" not in prices:
            result.add_error("MSFT missing from results")
        if "NVDA" not in prices:
            result.add_error("NVDA missing from results")

        result.details["found_tickers"] = list(prices.keys())

        # Test 2: All on same date
        dates = set(p.price_date for p in prices.values())
        if len(dates) != 1:
            result.add_error(f"Multiple dates in batch: {dates}")
        else:
            result.details["batch_date"] = str(list(dates)[0])

        # Test 3: All prices valid
        for ticker, price in prices.items():
            if price.close <= 0:
                result.add_error(f"{ticker} has invalid close: {price.close}")

        # Test 4: Empty list returns empty dict
        empty = await p.get_latest_prices([])
        if empty != {}:
            result.add_error("Empty input should return empty dict")

        # Test 5: Mix of valid and invalid tickers
        mixed = await p.get_latest_prices(["AAPL", "INVALID123XYZ"])
        if "AAPL" not in mixed:
            result.add_error("Valid ticker missing when mixed with invalid")
        if "INVALID123XYZ" in mixed:
            result.add_error("Invalid ticker should not be in results")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_adjusted_price_history(p: FactSetProvider) -> VerificationResult:
    """[7/16] Verify get_adjusted_price_history() - split-adjusted prices."""
    result = VerificationResult("get_adjusted_price_history")

    try:
        # Test 1: Get pre-2014 split prices (AAPL 7:1 split on 2014-06-09)
        # and post-2020 split prices (AAPL 4:1 split on 2020-08-31)
        pre_2014 = await p.get_adjusted_price_history("AAPL", date(2014, 1, 2), date(2014, 1, 3))
        if len(pre_2014) == 0:
            result.add_error("No pre-2014 adjusted prices returned")
            return result

        # Pre-2014 prices should be adjusted for both 7:1 and 4:1 splits
        # Raw price around $550, adjusted should be ~$550 / 28 ≈ $19.64
        adjusted_close = pre_2014[0].close
        if adjusted_close < 15 or adjusted_close > 25:
            result.add_error(f"Pre-2014 adjusted price looks wrong: ${adjusted_close:.2f}")

        result.details["pre_2014_adjusted"] = {
            "date": str(pre_2014[0].price_date),
            "adjusted_close": round(adjusted_close, 2),
        }

        # Test 2: Get price between splits (2015 - only 4:1 needs adjustment)
        between_splits = await p.get_adjusted_price_history(
            "AAPL", date(2015, 1, 2), date(2015, 1, 5)
        )
        if len(between_splits) == 0:
            result.add_error("No between-splits prices returned")
        else:
            # Should be adjusted only for 4:1 split
            # Raw around $110, adjusted = $110 / 4 ≈ $27.50
            mid_adjusted = between_splits[0].close
            if mid_adjusted < 20 or mid_adjusted > 40:
                result.add_warning(f"Between-splits adjusted price: ${mid_adjusted:.2f}")
            result.details["between_splits_adjusted"] = {
                "date": str(between_splits[0].price_date),
                "adjusted_close": round(mid_adjusted, 2),
            }

        # Test 3: Post-split prices (2025) should have factor ~1.0
        recent = await p.get_adjusted_price_history("AAPL", date(2025, 1, 2), date(2025, 1, 3))
        if len(recent) > 0:
            # Get raw price for comparison
            raw = await p.get_price("AAPL", recent[0].price_date)
            if raw:
                diff = abs(recent[0].close - raw.close)
                # Post-split should be nearly identical (maybe small dividend adjustment)
                if diff > 1.0:
                    result.add_warning(f"Post-split adjustment diff too large: {diff:.2f}")
                result.details["post_split"] = {
                    "date": str(recent[0].price_date),
                    "raw_close": round(raw.close, 2),
                    "adjusted_close": round(recent[0].close, 2),
                }

        # Test 4: Verify is_adjusted flag is set
        if not pre_2014[0].is_adjusted:
            result.add_error("is_adjusted flag not set on adjusted prices")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_adjustment_factors(p: FactSetProvider) -> VerificationResult:
    """[8/16] Verify get_adjustment_factors() - split/dividend factors."""
    result = VerificationResult("get_adjustment_factors")

    try:
        # Test 1: Get factors for AAPL including known split dates
        factors = await p.get_adjustment_factors("AAPL", date(2014, 1, 1), date(2025, 1, 1))

        if len(factors) == 0:
            result.add_error("No adjustment factors returned")
            return result

        result.details["factor_count"] = len(factors)

        # Test 2: Check 2014 split factor (7:1 split = 0.142857)
        split_2014_date = date(2014, 6, 9)
        split_2014 = factors.get(split_2014_date)
        if split_2014 is None:
            result.add_warning(f"2014 split factor not found at {split_2014_date}")
        else:
            expected_2014 = 1 / 7  # 0.142857
            if abs(split_2014 - expected_2014) > 0.001:
                result.add_error(
                    f"2014 split factor wrong: {split_2014} (expected {expected_2014})"
                )
            result.details["split_2014"] = {
                "date": str(split_2014_date),
                "factor": round(split_2014, 6),
            }

        # Test 3: Check 2020 split factor (4:1 split = 0.25)
        split_2020_date = date(2020, 8, 31)
        split_2020 = factors.get(split_2020_date)
        if split_2020 is None:
            result.add_warning(f"2020 split factor not found at {split_2020_date}")
        else:
            expected_2020 = 0.25
            if abs(split_2020 - expected_2020) > 0.001:
                result.add_error(
                    f"2020 split factor wrong: {split_2020} (expected {expected_2020})"
                )
            result.details["split_2020"] = {
                "date": str(split_2020_date),
                "factor": round(split_2020, 6),
            }

        # Test 4: Factors should be in date order
        factor_dates = list(factors.keys())
        sorted_dates = sorted(factor_dates)
        if factor_dates != sorted_dates:
            result.add_warning("Factors not in date order")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_corporate_actions(p: FactSetProvider) -> VerificationResult:
    """[9/16] Verify get_corporate_actions() - all corporate events."""
    result = VerificationResult("get_corporate_actions")

    try:
        # Test 1: Get actions for AAPL
        actions = await p.get_corporate_actions("AAPL", limit=50)

        if len(actions) == 0:
            result.add_error("No corporate actions returned")
            return result

        result.details["action_count"] = len(actions)

        # Test 2: Verify event types are mapped correctly
        valid_types = {"dividend", "split", "rights", "other"}
        event_types = set(a.event_type for a in actions)
        invalid_types = event_types - valid_types
        if invalid_types:
            result.add_error(f"Unknown event types: {invalid_types}")

        result.details["event_types_found"] = list(event_types)

        # Test 3: Most recent first (DESC order)
        for i in range(1, len(actions)):
            if actions[i].effective_date > actions[i - 1].effective_date:
                result.add_error("Actions not in DESC date order")
                break

        # Test 4: Should have some dividends (AAPL pays quarterly)
        dividends = [a for a in actions if a.event_type == "dividend"]
        if len(dividends) == 0:
            result.add_warning("No dividends found in corporate actions")
        else:
            result.details["dividend_count"] = len(dividends)

        # Test 5: Limit is respected
        limited = await p.get_corporate_actions("AAPL", limit=5)
        if len(limited) > 5:
            result.add_error(f"Limit not respected: {len(limited)} > 5")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_dividends(p: FactSetProvider) -> VerificationResult:
    """[10/16] Verify get_dividends() - dividend history."""
    result = VerificationResult("get_dividends")

    try:
        # Test 1: Get dividends for AAPL
        divs = await p.get_dividends("AAPL", limit=10)

        if len(divs) == 0:
            result.add_error("No dividends returned")
            return result

        result.details["dividend_count"] = len(divs)

        # Test 2: All should be dividend type
        non_dividends = [d for d in divs if d.event_type != "dividend"]
        if non_dividends:
            result.add_error(f"Non-dividend events found: {len(non_dividends)}")

        # Test 3: All should have DVC event code
        non_dvc = [d for d in divs if d.event_code != "DVC"]
        if non_dvc:
            result.add_error(f"Non-DVC event codes found: {[d.event_code for d in non_dvc]}")

        # Test 4: All should have dividend amount
        missing_amounts = [d for d in divs if d.dividend_amount is None]
        if missing_amounts:
            result.add_error(f"{len(missing_amounts)} dividends missing amount")

        # Test 5: AAPL dividends should be reasonable (~$0.20-$0.30 range)
        for d in divs:
            if d.dividend_amount is not None:
                if d.dividend_amount < 0.01:
                    result.add_warning(f"Suspiciously small dividend: ${d.dividend_amount}")
                if d.dividend_amount > 10.0:
                    result.add_warning(f"Suspiciously large dividend: ${d.dividend_amount}")

        result.details["latest_dividend"] = {
            "date": str(divs[0].effective_date),
            "amount": divs[0].dividend_amount,
            "currency": divs[0].dividend_currency,
        }

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_splits(p: FactSetProvider) -> VerificationResult:
    """[11/16] Verify get_splits() - CRITICAL split factor calculation."""
    result = VerificationResult("get_splits")

    try:
        # Test 1: Get splits for AAPL (should have at least 2: 2014 and 2020)
        splits = await p.get_splits("AAPL", limit=10)

        if len(splits) < 2:
            result.add_error(f"Expected at least 2 splits, got {len(splits)}")
            return result

        result.details["split_count"] = len(splits)

        # Test 2: All should be split type
        non_splits = [s for s in splits if s.event_type != "split"]
        if non_splits:
            result.add_error(f"Non-split events found: {len(non_splits)}")

        # Test 3: Find and verify specific splits
        split_2014 = None
        split_2020 = None
        for s in splits:
            if s.effective_date == date(2014, 6, 9):
                split_2014 = s
            if s.effective_date == date(2020, 8, 31):
                split_2020 = s

        if split_2014 is None:
            result.add_warning("2014 split not found")
        else:
            result.details["split_2014"] = {
                "date": str(split_2014.effective_date),
                "split_to": split_2014.split_to,
                "split_from": split_2014.split_from,
                "split_factor": split_2014.split_factor,
            }
            # Verify split_factor = split_to / split_from
            # For 7:1 split: to=7, from=1, factor=7.0
            if split_2014.split_factor is not None:
                if abs(split_2014.split_factor - 7.0) > 0.1:
                    result.add_error(
                        f"2014 split_factor wrong: {split_2014.split_factor} (expected 7.0)"
                    )

        if split_2020 is None:
            result.add_warning("2020 split not found")
        else:
            result.details["split_2020"] = {
                "date": str(split_2020.effective_date),
                "split_to": split_2020.split_to,
                "split_from": split_2020.split_from,
                "split_factor": split_2020.split_factor,
            }
            # For 4:1 split: to=4, from=1, factor=4.0
            if split_2020.split_factor is not None:
                if abs(split_2020.split_factor - 4.0) > 0.1:
                    result.add_error(
                        f"2020 split_factor wrong: {split_2020.split_factor} (expected 4.0)"
                    )

        # Test 4: Verify split_factor calculation makes sense
        # Note: For BNS (Bonus Issue), factor = (from + to) / from
        # For FSP/RSP, factor = to / from
        for s in splits:
            if s.split_to and s.split_from and s.split_from > 0 and s.split_factor:
                if s.event_code == "BNS":
                    expected = (s.split_from + s.split_to) / s.split_from
                else:
                    expected = s.split_to / s.split_from
                if abs(s.split_factor - expected) > 0.001:
                    result.add_error(
                        f"Split factor mismatch at {s.effective_date}: "
                        f"{s.split_factor} != expected {expected}"
                    )

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_fundamentals(p: FactSetProvider) -> VerificationResult:
    """[12/16] Verify get_fundamentals() - EPS, margins, ratios."""
    result = VerificationResult("get_fundamentals")

    try:
        # Test 1: Annual fundamentals
        annual = await p.get_fundamentals("AAPL", "annual", limit=4)

        if len(annual) == 0:
            result.add_error("No annual fundamentals returned")
            return result

        result.details["annual_count"] = len(annual)

        # Test 2: All should have 'annual' period type
        non_annual = [f for f in annual if f.period_type != "annual"]
        if non_annual:
            result.add_error(f"Non-annual periods found: {[f.period_type for f in non_annual]}")

        # Test 3: Most recent first (DESC order)
        for i in range(1, len(annual)):
            if annual[i].period_end > annual[i - 1].period_end:
                result.add_error("Annual fundamentals not in DESC order")
                break

        # Test 4: AAPL EPS should be reasonable (~$6/year)
        for f in annual:
            if f.eps_diluted is not None:
                if f.eps_diluted < 0.5 or f.eps_diluted > 20.0:
                    result.add_warning(f"Suspicious EPS for {f.period_end}: ${f.eps_diluted}")

        result.details["latest_annual"] = {
            "period_end": str(annual[0].period_end),
            "fiscal_year": annual[0].fiscal_year,
            "eps_diluted": annual[0].eps_diluted,
            "net_margin": annual[0].net_margin,
        }

        # Test 5: Quarterly fundamentals
        quarterly = await p.get_fundamentals("AAPL", "quarterly", limit=8)
        if len(quarterly) == 0:
            result.add_error("No quarterly fundamentals returned")
        else:
            result.details["quarterly_count"] = len(quarterly)
            non_quarterly = [f for f in quarterly if f.period_type != "quarterly"]
            if non_quarterly:
                result.add_error(f"Non-quarterly periods: {[f.period_type for f in non_quarterly]}")

        # Test 6: LTM fundamentals
        ltm = await p.get_fundamentals("AAPL", "ltm", limit=4)
        if len(ltm) == 0:
            result.add_warning("No LTM fundamentals returned")
        else:
            result.details["ltm_count"] = len(ltm)

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_company_profile(p: FactSetProvider) -> VerificationResult:
    """[13/16] Verify get_company_profile() - company description."""
    result = VerificationResult("get_company_profile")

    try:
        # Test 1: Get AAPL profile
        profile = await p.get_company_profile("AAPL")

        if profile is None:
            result.add_error("Profile is None")
            return result

        # Test 2: Should be substantial text (at least 50 chars)
        if len(profile) < 50:
            result.add_error(f"Profile too short: {len(profile)} chars")

        result.details["profile_length"] = len(profile)
        result.details["profile_preview"] = profile[:150] + "..." if len(profile) > 150 else profile

        # Test 3: Should mention Apple or related terms
        lower_profile = profile.lower()
        if "apple" not in lower_profile and "iphone" not in lower_profile:
            result.add_warning("Profile doesn't mention Apple or iPhone")

        # Test 4: Invalid ticker
        invalid_profile = await p.get_company_profile("INVALID123XYZ")
        if invalid_profile is not None:
            result.add_warning("Invalid ticker returned a profile")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_shares_outstanding(p: FactSetProvider) -> VerificationResult:
    """[14/16] Verify get_shares_outstanding() - CRITICAL unit conversion."""
    result = VerificationResult("get_shares_outstanding")

    try:
        # Test 1: Get AAPL shares
        shares = await p.get_shares_outstanding("AAPL")

        if shares is None:
            result.add_error("Shares is None")
            return result

        result.details["shares_outstanding"] = f"{shares.shares_outstanding:,.0f}"
        result.details["shares_outstanding_raw"] = (
            f"{shares.shares_outstanding_raw:,.0f}" if shares.shares_outstanding_raw else None
        )
        result.details["report_date"] = str(shares.report_date)

        # Test 2: AAPL has ~15 billion shares outstanding
        if shares.shares_outstanding < 10_000_000_000:
            result.add_error(f"Shares too low: {shares.shares_outstanding:,.0f}")
        if shares.shares_outstanding > 20_000_000_000:
            result.add_error(f"Shares too high: {shares.shares_outstanding:,.0f}")

        # Test 3: Verify unit conversion (raw in millions → actual)
        if shares.shares_outstanding_raw is not None:
            expected = shares.shares_outstanding_raw * 1_000_000
            if abs(shares.shares_outstanding - expected) > 1:
                result.add_error(
                    f"Unit conversion error: {shares.shares_outstanding} != "
                    f"{shares.shares_outstanding_raw} * 1,000,000"
                )

        # Test 4: Invalid ticker
        invalid_shares = await p.get_shares_outstanding("INVALID123XYZ")
        if invalid_shares is not None:
            result.add_warning("Invalid ticker returned shares")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_get_market_cap(p: FactSetProvider) -> VerificationResult:
    """[15/16] Verify get_market_cap() - CRITICAL calculation verification."""
    result = VerificationResult("get_market_cap")

    try:
        # Test 1: Get AAPL market cap
        mcap = await p.get_market_cap("AAPL")

        if mcap is None:
            result.add_error("Market cap is None")
            return result

        result.details["market_cap"] = f"${mcap:,.0f}"

        # Test 2: AAPL market cap should be ~$3T (reasonable range: $2T-$5T)
        if mcap < 2_000_000_000_000:
            result.add_error(f"Market cap too low: ${mcap:,.0f}")
        if mcap > 5_000_000_000_000:
            result.add_error(f"Market cap too high: ${mcap:,.0f}")

        # Test 3: Verify calculation = shares × price
        shares = await p.get_shares_outstanding("AAPL")
        price = await p.get_price("AAPL")

        if shares and price:
            expected_mcap = shares.shares_outstanding * price.close
            diff = abs(mcap - expected_mcap)
            if diff > 1:  # Should be exact
                result.add_error(f"Market cap calculation mismatch: {mcap} != {expected_mcap}")

            result.details["calculation"] = {
                "shares": f"{shares.shares_outstanding:,.0f}",
                "price": f"${price.close:.2f}",
                "expected_mcap": f"${expected_mcap:,.0f}",
            }

        # Test 4: Invalid ticker
        invalid_mcap = await p.get_market_cap("INVALID123XYZ")
        if invalid_mcap is not None:
            result.add_warning("Invalid ticker returned market cap")

        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception: {e}")

    return result


async def verify_close(p: FactSetProvider) -> VerificationResult:
    """[16/16] Verify close() - connection cleanup."""
    result = VerificationResult("close")

    try:
        # Test 1: Close should not error
        await p.close()
        result.details["closed"] = True
        result.mark_passed()

    except Exception as e:
        result.add_error(f"Exception on close: {e}")

    return result


async def run_all_verifications() -> bool:
    """Run all verification tests and report results."""
    print("=" * 70)
    print("FACTSET PROVIDER COMPREHENSIVE VERIFICATION")
    print("=" * 70)
    print()

    p = FactSetProvider()
    results: list[VerificationResult] = []

    # Define all verification functions
    verifications = [
        ("1/16", verify_get_fsym_id),
        ("2/16", verify_resolve_ticker),
        ("3/16", verify_search_securities),
        ("4/16", verify_get_price),
        ("5/16", verify_get_price_history),
        ("6/16", verify_get_latest_prices),
        ("7/16", verify_get_adjusted_price_history),
        ("8/16", verify_get_adjustment_factors),
        ("9/16", verify_get_corporate_actions),
        ("10/16", verify_get_dividends),
        ("11/16", verify_get_splits),
        ("12/16", verify_get_fundamentals),
        ("13/16", verify_get_company_profile),
        ("14/16", verify_get_shares_outstanding),
        ("15/16", verify_get_market_cap),
        ("16/16", verify_close),
    ]

    for label, verify_func in verifications:
        print(f"[{label}] {verify_func.__name__.replace('verify_', '')}...", end=" ", flush=True)

        try:
            vr = await verify_func(p)
            results.append(vr)

            if vr.passed:
                print("✓ PASSED")
                if vr.warnings:
                    for w in vr.warnings:
                        print(f"    ⚠️  WARNING: {w}")
            else:
                print("✗ FAILED")
                for e in vr.errors:
                    print(f"    ✗ ERROR: {e}")

            # Show key details
            if vr.details:
                for key, value in vr.details.items():
                    if isinstance(value, dict):
                        print(f"    {key}:")
                        for k, v in value.items():
                            print(f"      {k}: {v}")
                    else:
                        print(f"    {key}: {value}")
            print()

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            failed_result = VerificationResult(verify_func.__name__)
            failed_result.add_error(f"Unhandled exception: {e}")
            results.append(failed_result)
            print()

    # Summary
    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    warnings = sum(len(r.warnings) for r in results)

    if failed == 0:
        print(f"✓ ALL {passed}/16 ENDPOINTS VERIFIED SUCCESSFULLY")
        if warnings > 0:
            print(f"  ({warnings} warnings)")
    else:
        print(f"✗ VERIFICATION FAILED: {failed} errors, {passed} passed")
        print()
        print("Failed endpoints:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")
                for err in r.errors:
                    print(f"      {err}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_verifications())
    sys.exit(0 if success else 1)
