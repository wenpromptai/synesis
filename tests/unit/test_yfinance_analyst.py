"""Tests for yfinance analyst ratings models and parsing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from synesis.providers.yfinance.models import (
    AnalystPriceTargets,
    AnalystRatings,
    RecommendationTrend,
    UpgradeDowngrade,
)


# ---------------------------------------------------------------------------
# RecommendationTrend
# ---------------------------------------------------------------------------


def test_recommendation_trend_full() -> None:
    trend = RecommendationTrend(
        period="0m",
        strong_buy=6,
        buy=25,
        hold=15,
        sell=1,
        strong_sell=1,
    )
    assert trend.period == "0m"
    assert trend.strong_buy == 6
    assert trend.buy == 25
    assert trend.hold == 15
    assert trend.sell == 1
    assert trend.strong_sell == 1


def test_recommendation_trend_defaults() -> None:
    trend = RecommendationTrend(period="-1m")
    assert trend.strong_buy == 0
    assert trend.buy == 0
    assert trend.hold == 0
    assert trend.sell == 0
    assert trend.strong_sell == 0


# ---------------------------------------------------------------------------
# UpgradeDowngrade
# ---------------------------------------------------------------------------


def test_upgrade_downgrade_full() -> None:
    dt = datetime(2026, 3, 23, 14, 2, 48, tzinfo=timezone.utc)
    ud = UpgradeDowngrade(
        date=dt,
        firm="B of A Securities",
        to_grade="Buy",
        from_grade="Buy",
        action="main",
        price_target_action="Lowers",
        current_price_target=320.0,
        prior_price_target=325.0,
    )
    assert ud.date == dt
    assert ud.firm == "B of A Securities"
    assert ud.to_grade == "Buy"
    assert ud.from_grade == "Buy"
    assert ud.action == "main"
    assert ud.price_target_action == "Lowers"
    assert ud.current_price_target == pytest.approx(320.0)
    assert ud.prior_price_target == pytest.approx(325.0)


def test_upgrade_downgrade_no_price_target() -> None:
    """Firms that don't publish price targets should have None."""
    ud = UpgradeDowngrade(
        date=datetime(2026, 1, 30, 13, 51, 44, tzinfo=timezone.utc),
        firm="Needham",
        to_grade="Hold",
        from_grade="Hold",
        action="reit",
    )
    assert ud.price_target_action is None
    assert ud.current_price_target is None
    assert ud.prior_price_target is None


# ---------------------------------------------------------------------------
# AnalystPriceTargets
# ---------------------------------------------------------------------------


def test_analyst_price_targets_full() -> None:
    apt = AnalystPriceTargets(
        current=260.48,
        high=350.0,
        low=205.0,
        mean=296.33,
        median=300.0,
    )
    assert apt.current == pytest.approx(260.48)
    assert apt.high == pytest.approx(350.0)
    assert apt.low == pytest.approx(205.0)
    assert apt.mean == pytest.approx(296.33)
    assert apt.median == pytest.approx(300.0)


def test_analyst_price_targets_all_none() -> None:
    apt = AnalystPriceTargets()
    assert apt.current is None
    assert apt.high is None
    assert apt.low is None
    assert apt.mean is None
    assert apt.median is None


# ---------------------------------------------------------------------------
# AnalystRatings (container)
# ---------------------------------------------------------------------------


def test_analyst_ratings_full() -> None:
    ratings = AnalystRatings(
        ticker="AAPL",
        recommendations=[
            RecommendationTrend(period="0m", strong_buy=6, buy=25, hold=15, sell=1, strong_sell=1),
            RecommendationTrend(period="-1m", strong_buy=5, buy=25, hold=16, sell=1, strong_sell=1),
        ],
        upgrades_downgrades=[
            UpgradeDowngrade(
                date=datetime(2026, 3, 23, 14, 2, 48, tzinfo=timezone.utc),
                firm="B of A Securities",
                to_grade="Buy",
                from_grade="Buy",
                action="main",
                price_target_action="Lowers",
                current_price_target=320.0,
                prior_price_target=325.0,
            ),
        ],
        price_targets=AnalystPriceTargets(
            current=260.48, high=350.0, low=205.0, mean=296.33, median=300.0
        ),
    )
    assert ratings.ticker == "AAPL"
    assert len(ratings.recommendations) == 2
    assert len(ratings.upgrades_downgrades) == 1
    assert ratings.price_targets is not None
    assert ratings.price_targets.mean == pytest.approx(296.33)


def test_analyst_ratings_empty_ticker() -> None:
    """A ticker with no analyst coverage should return empty lists."""
    ratings = AnalystRatings(ticker="UNKNOWN")
    assert ratings.ticker == "UNKNOWN"
    assert ratings.recommendations == []
    assert ratings.upgrades_downgrades == []
    assert ratings.price_targets is None


def test_analyst_ratings_serialization_roundtrip() -> None:
    """Model should survive JSON serialization and deserialization."""
    original = AnalystRatings(
        ticker="TSLA",
        recommendations=[
            RecommendationTrend(period="0m", strong_buy=10, buy=15, hold=12, sell=5, strong_sell=2),
        ],
        upgrades_downgrades=[
            UpgradeDowngrade(
                date=datetime(2026, 1, 30, 20, 47, 29, tzinfo=timezone.utc),
                firm="Maxim Group",
                to_grade="Buy",
                from_grade="Hold",
                action="up",
                price_target_action="Announces",
                current_price_target=300.0,
            ),
        ],
        price_targets=AnalystPriceTargets(
            current=180.0, high=400.0, low=100.0, mean=250.0, median=240.0
        ),
    )
    dumped = original.model_dump(mode="json")
    restored = AnalystRatings.model_validate(dumped)

    assert restored.ticker == "TSLA"
    assert len(restored.recommendations) == 1
    assert restored.recommendations[0].strong_buy == 10
    assert len(restored.upgrades_downgrades) == 1
    assert restored.upgrades_downgrades[0].firm == "Maxim Group"
    assert restored.upgrades_downgrades[0].current_price_target == pytest.approx(300.0)
    assert restored.upgrades_downgrades[0].prior_price_target is None
    assert restored.price_targets is not None
    assert restored.price_targets.median == pytest.approx(240.0)
