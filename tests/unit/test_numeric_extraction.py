"""Tests for impact scoring.

Tests Stage 1 features:
- Impact scoring (compute_impact_score)
"""

from synesis.processing.news.impact_scorer import (
    ImpactResult,
    compute_impact_score,
    _max_dollar_amount,
    _parse_dollar,
    _resolve_source,
    _text_pattern_score,
    _content_type_score,
    _magnitude_score,
    _suppressor_score,
    _DOLLAR_RE,
    CRITICAL_THRESHOLD,
    HIGH_THRESHOLD,
    NORMAL_THRESHOLD,
    _TEXT_PATTERN_CAP,
    _CONTENT_TYPE_CAP,
    _MAGNITUDE_CAP,
)
from synesis.processing.news import (
    LightClassification,
    UrgencyLevel,
)


class TestUrgencyLevel:
    """Tests for UrgencyLevel enum."""

    def test_urgency_level_values(self) -> None:
        assert UrgencyLevel.critical.value == "critical"
        assert UrgencyLevel.high.value == "high"
        assert UrgencyLevel.normal.value == "normal"
        assert UrgencyLevel.low.value == "low"


class TestLightClassificationModel:
    """Tests for the new lightweight LightClassification (no LLM fields)."""

    def test_default_values(self) -> None:
        """LightClassification has sensible defaults."""
        lc = LightClassification()
        assert lc.matched_tickers == []
        assert lc.impact_score == 0
        assert lc.impact_reasons == []
        assert lc.urgency == UrgencyLevel.normal

    def test_with_tickers_and_score(self) -> None:
        """LightClassification with tickers and impact score."""
        lc = LightClassification(
            matched_tickers=["NVDA", "MRVL"],
            impact_score=57,
            impact_reasons=["source:DeItaone:wire_relay:+15", "wire_prefix:+15", "mna:+15"],
            urgency=UrgencyLevel.critical,
        )
        assert lc.matched_tickers == ["NVDA", "MRVL"]
        assert lc.impact_score == 57
        assert lc.urgency == UrgencyLevel.critical

    def test_serialization(self) -> None:
        """JSON serialization of new LightClassification."""
        lc = LightClassification(
            matched_tickers=["AAPL", "~OPENAI"],
            impact_score=43,
            impact_reasons=["just_in:+10", "mna:+15"],
            urgency=UrgencyLevel.high,
        )
        data = lc.model_dump(mode="json")
        assert data["matched_tickers"] == ["AAPL", "~OPENAI"]
        assert data["impact_score"] == 43
        assert data["urgency"] == "high"


class TestImpactScoring:
    """Tests for compute_impact_score (replaces classify_urgency_by_rules)."""

    # --- Fast-track: Econ data releases ---

    def test_econ_t1_release_critical(self) -> None:
        """Tier 1 econ release from known source → critical."""
        r = compute_impact_score(
            "US CPI (YOY) (MAR) ACTUAL: 2.5% VS 1.9% PREVIOUS; EST 2.6%", "FirstSquawk"
        )
        assert r.urgency == UrgencyLevel.critical

    def test_econ_t1_gdp_release(self) -> None:
        """GDP release with actual/est → critical."""
        r = compute_impact_score(
            "US GDP (QOQ) (Q4) ACTUAL: 2.3% VS 3.1% PREVIOUS; EST 2.6%", "FirstSquawk"
        )
        assert r.urgency == UrgencyLevel.critical

    def test_econ_t2_release_high(self) -> None:
        """Tier 2 econ release from known source → at least high."""
        r = compute_impact_score(
            "US INITIAL JOBLESS CLAIMS ACTUAL: 202K VS 210K PREVIOUS; EST 212K", "FirstSquawk"
        )
        assert r.urgency in (UrgencyLevel.high, UrgencyLevel.critical)

    def test_econ_commentary_not_fast_tracked(self) -> None:
        """Commentary about CPI (no ACTUAL/EST) should NOT fast-track."""
        r = compute_impact_score(
            "FED'S WILLIAMS: WAR COULD INCREASE INFLATION AND CPI", "financialjuice"
        )
        assert r.urgency != UrgencyLevel.critical

    def test_econ_unknown_source_not_fast_tracked(self) -> None:
        """Econ release from unknown source should NOT fast-track."""
        r = compute_impact_score("US CPI ACTUAL: 2.5% VS 2.6% EST", "unknown_channel")
        assert r.urgency != UrgencyLevel.critical

    # --- Fast-track: M&A with large dollar amounts ---

    def test_mna_billion_dollar_critical(self) -> None:
        """M&A with $1B+ dollar amount → critical."""
        r = compute_impact_score("*NVIDIA INVESTS $2B IN MARVELL TECHNOLOGY", "DeItaone")
        assert r.urgency == UrgencyLevel.critical

    def test_mna_raises_critical(self) -> None:
        """Fundraising with large dollar amount → critical."""
        r = compute_impact_score(
            "JUST IN: OPENAI RAISES $122,000,000,000 AT $852 BILLION VALUATION.", "WatcherGuru"
        )
        assert r.urgency == UrgencyLevel.critical

    def test_mna_stake_before_dollar(self) -> None:
        """$X STAKE pattern (dollar before keyword) → critical."""
        r = compute_impact_score(
            "SPACEX IN TALKS WITH SAUDI PIF FOR $5 BLN ANCHOR STAKE IN 2026 IPO", "FirstSquawk"
        )
        assert r.urgency == UrgencyLevel.critical

    def test_mna_without_dollar_not_critical(self) -> None:
        """M&A without dollar amount should NOT fast-track to critical."""
        r = compute_impact_score("OPENAI HAS ACQUIRED TECH TALK SHOW TBPN", "FirstSquawk")
        assert r.urgency != UrgencyLevel.critical

    # --- Fast-track: Breaking from wire sources ---

    def test_breaking_wire_high(self) -> None:
        """BREAKING from wire source → at least high."""
        r = compute_impact_score(
            "⚠ BREAKING: IRAN: DRAFTING PROTOCOL WITH OMAN FOR HORMUZ STRAIT", "financialjuice"
        )
        assert r.urgency in (UrgencyLevel.high, UrgencyLevel.critical)

    def test_breaking_non_wire_not_fast_tracked(self) -> None:
        """BREAKING from non-wire source does NOT fast-track."""
        r = compute_impact_score("BREAKING: TRUMP REMOVES BONDI AS AG", "unusual_whales")
        # unusual_whales is "curated" not in WIRE_RELAY_SOURCES, so no fast-track
        assert r.urgency not in (UrgencyLevel.high, UrgencyLevel.critical)

    # --- Scoring: Source reliability ---

    def test_wire_source_boosts_score(self) -> None:
        """Wire source (DeItaone +15) boosts score significantly."""
        r_wire = compute_impact_score("*US MORTGAGE RATES RISE FOR FIFTH WEEK", "DeItaone")
        r_unknown = compute_impact_score("*US MORTGAGE RATES RISE FOR FIFTH WEEK", "unknown_acct")
        assert r_wire.score > r_unknown.score

    def test_sensational_source_zero_weight(self) -> None:
        """Sensational sources get zero weight (no boost)."""
        r = compute_impact_score("JUST IN: SILVER CRASHES UNDER $70", "WatcherGuru")
        assert r.components["source"] == 0

    # --- Scoring: Suppressors ---

    def test_rt_suppressed(self) -> None:
        """RT / retweet is suppressed."""
        r = compute_impact_score("RT @analyst: Great insight on markets", "unknown")
        assert r.urgency == UrgencyLevel.low

    def test_opinion_suppressed(self) -> None:
        """Opinion markers reduce score."""
        r = compute_impact_score("IMO the market is overreacting to CPI data", "unknown")
        assert r.components["suppressor"] < 0

    def test_digest_suppressed(self) -> None:
        """News digests are suppressed."""
        r = compute_impact_score("🌅 Market News Digest\nTop Stories", "unknown")
        assert r.components["suppressor"] < 0

    def test_promo_suppressed(self) -> None:
        """Promotional content is heavily suppressed."""
        r = compute_impact_score("SUBSCRIBE to our channel for market alerts!", "unknown")
        assert r.urgency == UrgencyLevel.low

    # --- Scoring: Dollar magnitude ---

    def test_dollar_billion_magnitude(self) -> None:
        """$1B+ dollar amounts contribute to magnitude score."""
        r = compute_impact_score("COMPANY INVESTS $10B IN NEW PROJECT", "FirstSquawk")
        assert r.components["magnitude"] >= 12

    def test_dollar_no_suffix_ignored(self) -> None:
        """Dollar amounts without magnitude suffix are ignored (avoids $141/BBL)."""
        r = compute_impact_score("OIL REACHES $141.37 PER BARREL", "FirstSquawk")
        assert r.components["magnitude"] == 0

    # --- Scoring: Content type ---

    def test_geopolitical_scored(self) -> None:
        """Geopolitical escalation detected."""
        r = compute_impact_score("IRAN LAUNCHES MISSILE STRIKES ON ISRAEL", "FirstSquawk")
        assert r.components["content_type"] > 0

    def test_tariff_scored(self) -> None:
        """Tariff news detected."""
        r = compute_impact_score("TRUMP IMPOSES 25% TARIFFS ON CHINA", "FirstSquawk")
        assert r.components["content_type"] > 0


class TestResolveSource:
    """Tests for _resolve_source — maps channel names to wire sources."""

    def test_direct_match(self) -> None:
        """Source account directly in SOURCE_RELIABILITY."""
        assert _resolve_source("FirstSquawk", "some text") == "FirstSquawk"

    def test_fallback_to_xcom_url(self) -> None:
        """Unknown channel but x.com URL contains known source."""
        text = "Headlines via https://x.com/DeItaone/status/123456"
        assert _resolve_source("marketfeed", text) == "DeItaone"

    def test_fallback_to_twitter_url(self) -> None:
        """twitter.com URL also works."""
        text = "Via https://twitter.com/FirstSquawk/status/789"
        assert _resolve_source("some_channel", text) == "FirstSquawk"

    def test_unknown_source_no_url(self) -> None:
        """Unknown channel, no URL → returns source_account unchanged."""
        assert _resolve_source("random_channel", "just some text") == "random_channel"

    def test_url_with_unknown_handle(self) -> None:
        """URL present but handle not in SOURCE_RELIABILITY → returns source_account."""
        text = "https://x.com/nobody_known/status/111"
        assert _resolve_source("my_channel", text) == "my_channel"

    def test_telegram_channel_direct(self) -> None:
        """Telegram channels listed directly in SOURCE_RELIABILITY."""
        assert _resolve_source("disclosetv", "any text") == "disclosetv"


class TestParseDollar:
    """Tests for _parse_dollar — extracts dollar amounts from regex matches."""

    def _parse(self, text: str) -> float:
        m = _DOLLAR_RE.search(text)
        assert m is not None, f"No dollar match in: {text}"
        return _parse_dollar(m)

    def test_billion_bln(self) -> None:
        assert self._parse("$5 BLN") == 5e9

    def test_billion_full(self) -> None:
        assert self._parse("$777 BILLION") == 777e9

    def test_billion_short(self) -> None:
        assert self._parse("$110B") == 110e9

    def test_million_full(self) -> None:
        assert self._parse("$500 MILLION") == 500e6

    def test_million_short(self) -> None:
        assert self._parse("$500M") == 500e6

    def test_trillion(self) -> None:
        assert self._parse("$1.5 TRILLION") == 1.5e12

    def test_thousand(self) -> None:
        assert self._parse("$10K") == 10e3

    def test_comma_separated(self) -> None:
        assert self._parse("$122,000 MILLION") == 122_000e6


class TestMaxDollarAmount:
    """Tests for _max_dollar_amount — finds largest dollar amount in text."""

    def test_multiple_amounts_returns_largest(self) -> None:
        text = "Raised $500M in round A and $2B in round B"
        assert _max_dollar_amount(text) == 2e9

    def test_no_amounts_returns_zero(self) -> None:
        assert _max_dollar_amount("No dollar amounts here") == 0.0

    def test_price_without_suffix_ignored(self) -> None:
        """$141.37/BBL has no valid suffix → 0."""
        assert _max_dollar_amount("Oil at $141.37/BBL") == 0.0

    def test_single_amount(self) -> None:
        assert _max_dollar_amount("Deal worth $10B") == 10e9


class TestThresholdBoundaries:
    """Tests for score-to-urgency mapping at exact boundaries."""

    def _make_result(self, score: int) -> ImpactResult:
        """Create an ImpactResult at an exact score to test threshold mapping."""
        # Create an ImpactResult at an exact score to test threshold mapping
        if score >= CRITICAL_THRESHOLD:
            urgency = UrgencyLevel.critical
        elif score >= HIGH_THRESHOLD:
            urgency = UrgencyLevel.high
        elif score >= NORMAL_THRESHOLD:
            urgency = UrgencyLevel.normal
        else:
            urgency = UrgencyLevel.low
        return ImpactResult(score=score, urgency=urgency)

    def test_critical_at_55(self) -> None:
        assert self._make_result(55).urgency == UrgencyLevel.critical

    def test_high_at_54(self) -> None:
        assert self._make_result(54).urgency == UrgencyLevel.high

    def test_high_at_32(self) -> None:
        assert self._make_result(32).urgency == UrgencyLevel.high

    def test_normal_at_31(self) -> None:
        assert self._make_result(31).urgency == UrgencyLevel.normal

    def test_normal_at_15(self) -> None:
        assert self._make_result(15).urgency == UrgencyLevel.normal

    def test_low_at_14(self) -> None:
        assert self._make_result(14).urgency == UrgencyLevel.low

    def test_low_at_0(self) -> None:
        assert self._make_result(0).urgency == UrgencyLevel.low

    def test_threshold_constants(self) -> None:
        """Verify threshold constants match expected values."""
        assert CRITICAL_THRESHOLD == 55
        assert HIGH_THRESHOLD == 32
        assert NORMAL_THRESHOLD == 15


class TestComponentCaps:
    """Tests for per-component score caps."""

    def test_text_pattern_capped_at_35(self) -> None:
        """Stacking many text patterns should cap at 35."""
        # Wire prefix + BREAKING + JUST IN + ALERT + EXCLUSIVE + SUPERLATIVE + SURPRISE + EXTREME
        text = "*BREAKING: JUST IN - ALERT: EXCLUSIVE SCOOP: SURPRISE CRASH — WORST SINCE 2008"
        score, _ = _text_pattern_score(text)
        assert score == _TEXT_PATTERN_CAP
        assert score == 35

    def test_content_type_capped_at_20(self) -> None:
        """Stacking many content types should cap at 20."""
        # M&A + geopolitical + tariff + bankruptcy — triggers multiple patterns
        text = "SANCTIONS ON ACQUISITION TARGET AMID TRADE WAR AND BANKRUPTCY"
        score, _ = _content_type_score(text)
        assert score <= _CONTENT_TYPE_CAP

    def test_magnitude_capped_at_20(self) -> None:
        """Large dollar + large pct should cap at 20."""
        text = "$500B WORTH OF STOCK SURGED 15%"
        score, _ = _magnitude_score(text)
        assert score <= _MAGNITUDE_CAP

    def test_suppressor_uncapped(self) -> None:
        """Suppressors have no cap — can go deeply negative."""
        text = (
            "RT @promo: SUBSCRIBE for GIVEAWAY! Follow us for faster headlines! YESTERDAY'S RECAP"
        )
        score, _ = _suppressor_score(text)
        assert score < -20  # Multiple suppressors stack


class TestScoreClamping:
    """Tests for total score clamping to [0, 100]."""

    def test_heavy_suppression_floors_at_zero(self) -> None:
        """Spam text should score 0, not negative."""
        r = compute_impact_score(
            "RT @scammer: SUBSCRIBE NOW! GIVEAWAY AIRDROP! Follow us for faster headlines!",
            "unknown",
        )
        assert r.score == 0

    def test_score_never_exceeds_100(self) -> None:
        """Even max stacking should not exceed 100."""
        r = compute_impact_score(
            "*BREAKING: JUST IN - ALERT: EXCLUSIVE: SURPRISE CRASH — WORST SINCE 2008! "
            "IRAN LAUNCHES MISSILE STRIKES, TARIFFS IMPOSED, BANKRUPTCY FILED! "
            "US CPI (YOY) ACTUAL: 10% — $500 TRILLION WIPED OUT, SURGED 50%",
            "FirstSquawk",
        )
        assert r.score <= 100


class TestFastTrackUpgradeOnly:
    """Tests that fast-track rules only upgrade, never downgrade."""

    def test_high_score_with_high_fast_track_stays_critical(self) -> None:
        """If additive score is already critical, fast-track high doesn't downgrade."""
        # This message scores very high additively AND triggers BREAKING wire fast-track (high)
        r = compute_impact_score(
            "*BREAKING: JUST IN - ALERT: IRAN LAUNCHES MISSILE STRIKES "
            "US CPI ACTUAL: 10% VS 2% PREVIOUS — $100B WIPED OUT, SURGED 50%",
            "FirstSquawk",
        )
        # Should be critical from additive score, not downgraded by fast-track:high
        assert r.urgency == UrgencyLevel.critical

    def test_fast_track_upgrades_low_score(self) -> None:
        """Fast-track can upgrade a message that scores low additively."""
        # T1 econ release from wire source — fast-tracks to critical even if text is short
        r = compute_impact_score("US CPI ACTUAL: 2.5% VS 2.6% EST", "FirstSquawk")
        assert r.urgency == UrgencyLevel.critical


class TestTextPatternScoring:
    """Tests for individual text pattern signals."""

    def test_wire_prefix_star(self) -> None:
        """Leading * gives +15."""
        score, reasons = _text_pattern_score("*SOME HEADLINE")
        assert score >= 15
        assert any("wire_prefix" in r for r in reasons)

    def test_breaking(self) -> None:
        """BREAKING gives +12."""
        score, reasons = _text_pattern_score("BREAKING: Major news")
        assert score >= 12
        assert any("breaking" in r for r in reasons)

    def test_just_in(self) -> None:
        """JUST IN gives +10."""
        score, reasons = _text_pattern_score("JUST IN: Something happened")
        assert score >= 10
        assert any("just_in" in r for r in reasons)

    def test_flash_alert(self) -> None:
        """ALERT: gives +12."""
        score, reasons = _text_pattern_score("ALERT: Market crash incoming")
        assert score >= 12
        assert any("flash_alert" in r for r in reasons)

    def test_superlative(self) -> None:
        """HIGHEST SINCE gives +10."""
        score, reasons = _text_pattern_score("PRICES HIT HIGHEST SINCE 2008")
        assert score >= 10
        assert any("superlative" in r for r in reasons)

    def test_surprise(self) -> None:
        """SURPRISE gives +10."""
        score, reasons = _text_pattern_score("SURPRISE FED CUT")
        assert score >= 10
        assert any("surprise" in r for r in reasons)

    def test_plain_text_zero(self) -> None:
        """Plain text with no signals → 0."""
        score, _ = _text_pattern_score("The weather is nice today")
        assert score == 0


class TestPercentageMoveScoring:
    """Tests for percentage move detection in magnitude scoring."""

    def test_large_move_10pct(self) -> None:
        """10%+ move with action verb → +10."""
        score, reasons = _magnitude_score("OIL PRICES SURGED NEARLY 10%")
        assert any("pct_move" in r for r in reasons)
        assert score >= 10

    def test_medium_move_3pct(self) -> None:
        """2-5% move → +6."""
        score, reasons = _magnitude_score("STOCK FALLS 3%")
        assert any("pct_move" in r and "+6" in r for r in reasons)

    def test_small_move_1pct_ignored(self) -> None:
        """<2% move → no pct_move score."""
        score, reasons = _magnitude_score("STOCK ROSE 1% TODAY")
        assert not any("pct_move" in r for r in reasons)

    def test_pct_without_verb_ignored(self) -> None:
        """Percentage without action verb → no match."""
        score, reasons = _magnitude_score("CPI AT 2.5%")
        assert not any("pct_move" in r for r in reasons)
