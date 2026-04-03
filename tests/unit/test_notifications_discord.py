"""Tests for Discord notification formatting and webhook routing."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from synesis.notifications.discord import (
    COLOR_CRITICAL,
    COLOR_NEUTRAL,
    COLOR_URGENT,
    format_stage1_embed,
    format_stage2_embed,
    send_discord,
)
from synesis.processing.news import (
    LightClassification,
    MarketEvaluation,
    SmartAnalysis,
    SourcePlatform,
    UnifiedMessage,
    UrgencyLevel,
)
from synesis.processing.news.models import ETFImpact


def _make_message(text: str = "Test message", account: str = "@test") -> UnifiedMessage:
    return UnifiedMessage(
        external_id="test_123",
        source_platform=SourcePlatform.telegram,
        source_account=account,
        text=text,
        timestamp=datetime.now(timezone.utc),
    )


def _make_extraction(**kwargs: object) -> LightClassification:
    defaults: dict = {
        "matched_tickers": ["AAPL"],
        "impact_score": 60,
        "impact_reasons": ["wire_prefix:+15"],
        "urgency": UrgencyLevel.high,
    }
    defaults.update(kwargs)
    return LightClassification(**defaults)


def _make_analysis(**kwargs: object) -> SmartAnalysis:
    defaults: dict = {
        "macro_impact": [ETFImpact(ticker="SPY", sentiment_score=0.7)],
        "primary_thesis": "Dovish pivot supports risk assets",
        "market_evaluations": [],
    }
    defaults.update(kwargs)
    return SmartAnalysis(**defaults)


# ---------------------------------------------------------------------------
# Stage 1 embed formatting
# ---------------------------------------------------------------------------


class TestFormatStage1Embed:
    """Tests for format_stage1_embed."""

    def test_critical_urgency_color_and_title(self) -> None:
        embed = format_stage1_embed(
            _make_message(), _make_extraction(urgency=UrgencyLevel.critical)
        )
        assert embed[0]["color"] == COLOR_CRITICAL
        assert "1st Pass" in embed[0]["title"]

    def test_high_urgency_color_and_title(self) -> None:
        embed = format_stage1_embed(_make_message(), _make_extraction(urgency=UrgencyLevel.high))
        assert embed[0]["color"] == COLOR_URGENT
        assert "1st Pass" in embed[0]["title"]

    def test_tickers_in_fields(self) -> None:
        embed = format_stage1_embed(
            _make_message(),
            _make_extraction(matched_tickers=["NVDA", "MRVL"]),
        )
        tickers_field = next(f for f in embed[0]["fields"] if f["name"] == "Tickers")
        assert "NVDA" in tickers_field["value"]
        assert "MRVL" in tickers_field["value"]

    def test_impact_in_fields(self) -> None:
        embed = format_stage1_embed(
            _make_message(),
            _make_extraction(impact_score=75),
        )
        impact_field = next(f for f in embed[0]["fields"] if f["name"] == "Impact")
        assert "75/100" in impact_field["value"]

    def test_source_account_in_author(self) -> None:
        embed = format_stage1_embed(_make_message(account="@DeItaone"), _make_extraction())
        assert embed[0]["author"]["name"] == "@DeItaone"

    def test_footer_is_stage1(self) -> None:
        embed = format_stage1_embed(_make_message(), _make_extraction())
        assert "Stage 1" in embed[0]["footer"]["text"]


# ---------------------------------------------------------------------------
# Stage 2 embed formatting
# ---------------------------------------------------------------------------


class TestFormatStage2Embed:
    """Tests for format_stage2_embed."""

    def test_default_color_and_title(self) -> None:
        embed = format_stage2_embed(_make_message(), _make_analysis())
        assert embed[0]["color"] == COLOR_NEUTRAL
        assert "Analysis" in embed[0]["title"]

    def test_macro_etf_in_fields(self) -> None:
        analysis = _make_analysis(
            macro_impact=[
                ETFImpact(ticker="SPY", sentiment_score=0.7),
                ETFImpact(ticker="GLD", sentiment_score=-0.3),
            ]
        )
        embed = format_stage2_embed(_make_message(), analysis)
        macro_field = next(f for f in embed[0]["fields"] if f["name"] == "Macro Impact")
        assert "`SPY`" in macro_field["value"]
        assert "`GLD`" in macro_field["value"]

    def test_sector_etf_in_fields(self) -> None:
        analysis = _make_analysis(
            sector_impact=[
                ETFImpact(ticker="XLK", sentiment_score=-0.5),
                ETFImpact(ticker="XLF", sentiment_score=0.3),
            ]
        )
        embed = format_stage2_embed(_make_message(), analysis)
        sector_field = next(f for f in embed[0]["fields"] if f["name"] == "Sector Impact")
        assert "`XLK`" in sector_field["value"]

    def test_polymarket_field_when_relevant(self) -> None:
        analysis = _make_analysis(
            market_evaluations=[
                MarketEvaluation(
                    market_id="mkt_1",
                    market_question="Will Fed cut rates?",
                    is_relevant=True,
                    relevance_reasoning="Direct",
                    current_price=0.35,
                    estimated_fair_price=0.55,
                    edge=0.20,
                    verdict="undervalued",
                    confidence=0.7,
                    reasoning="Likely",
                    recommended_side="yes",
                ),
            ]
        )
        embed = format_stage2_embed(_make_message(), analysis)
        poly_field = next((f for f in embed[0]["fields"] if f["name"] == "Polymarket"), None)
        assert poly_field is not None
        assert "Will Fed cut rates?" in poly_field["value"]

    def test_footer_is_stage2(self) -> None:
        embed = format_stage2_embed(_make_message(), _make_analysis())
        assert "Stage 2" in embed[0]["footer"]["text"]


# ---------------------------------------------------------------------------
# send_discord webhook routing
# ---------------------------------------------------------------------------


class TestSendDiscordWebhookRouting:
    """Tests for send_discord with webhook_url_override."""

    @pytest.mark.asyncio
    async def test_override_uses_provided_url(self) -> None:
        """webhook_url_override should be used instead of default."""
        override = SecretStr("https://discord.com/api/webhooks/override/token")

        with patch("synesis.notifications.discord.httpx.AsyncClient") as mock_client_cls:
            mock_response = AsyncMock()
            mock_response.status_code = 204
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await send_discord([{"title": "test"}], webhook_url_override=override)

        assert result is True
        mock_client.post.assert_called_once()
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "https://discord.com/api/webhooks/override/token"

    @pytest.mark.asyncio
    async def test_no_override_uses_default(self) -> None:
        """Without override, falls back to settings.discord_webhook_url."""
        with (
            patch("synesis.notifications.discord.httpx.AsyncClient") as mock_client_cls,
            patch("synesis.notifications.discord.get_settings") as mock_settings,
        ):
            mock_settings.return_value.discord_webhook_url = SecretStr(
                "https://discord.com/api/webhooks/default/token"
            )
            mock_response = AsyncMock()
            mock_response.status_code = 204
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await send_discord([{"title": "test"}])

        assert result is True
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "https://discord.com/api/webhooks/default/token"

    @pytest.mark.asyncio
    async def test_no_webhook_configured_returns_false(self) -> None:
        """Returns False when no webhook URL is configured."""
        with patch("synesis.notifications.discord.get_settings") as mock_settings:
            mock_settings.return_value.discord_webhook_url = None
            result = await send_discord([{"title": "test"}])

        assert result is False


# ---------------------------------------------------------------------------
# Dispatcher routing (Stage 2 → discord2 webhook)
# ---------------------------------------------------------------------------


class TestDispatcherStage2Routing:
    """Tests for dispatcher routing Stage 2 to separate webhook."""

    @pytest.mark.asyncio
    async def test_stage2_uses_discord2_webhook(self) -> None:
        """emit_stage2 passes discord2_webhook_url to send_discord."""
        discord2_url = SecretStr("https://discord.com/api/webhooks/stage2/token")

        with (
            patch("synesis.notifications.dispatcher.get_settings") as mock_settings,
            patch(
                "synesis.notifications.discord.send_discord", new_callable=AsyncMock
            ) as mock_send,
        ):
            mock_settings.return_value.notification_channel = "discord"
            mock_settings.return_value.discord2_webhook_url = discord2_url
            mock_settings.return_value.discord_webhook_url = SecretStr("https://default/token")
            mock_send.return_value = True

            from synesis.notifications.dispatcher import emit_stage2

            await emit_stage2(_make_message(), _make_analysis())

        mock_send.assert_called_once()
        assert mock_send.call_args.kwargs["webhook_url_override"] == discord2_url

    @pytest.mark.asyncio
    async def test_stage2_falls_back_to_default_webhook(self) -> None:
        """emit_stage2 falls back to discord_webhook_url when discord2 not set."""
        default_url = SecretStr("https://discord.com/api/webhooks/default/token")

        with (
            patch("synesis.notifications.dispatcher.get_settings") as mock_settings,
            patch(
                "synesis.notifications.discord.send_discord", new_callable=AsyncMock
            ) as mock_send,
        ):
            mock_settings.return_value.notification_channel = "discord"
            mock_settings.return_value.discord2_webhook_url = None
            mock_settings.return_value.discord_webhook_url = default_url
            mock_send.return_value = True

            from synesis.notifications.dispatcher import emit_stage2

            await emit_stage2(_make_message(), _make_analysis())

        mock_send.assert_called_once()
        assert mock_send.call_args.kwargs["webhook_url_override"] == default_url

    @pytest.mark.asyncio
    async def test_stage1_uses_default_webhook(self) -> None:
        """emit_stage1 uses default webhook (no override)."""
        with (
            patch("synesis.notifications.dispatcher.get_settings") as mock_settings,
            patch(
                "synesis.notifications.discord.send_discord", new_callable=AsyncMock
            ) as mock_send,
        ):
            mock_settings.return_value.notification_channel = "discord"
            mock_send.return_value = True

            from synesis.notifications.dispatcher import emit_stage1

            await emit_stage1(_make_message(), _make_extraction())

        mock_send.assert_called_once()
        # Stage 1 should NOT pass webhook_url_override
        assert "webhook_url_override" not in mock_send.call_args.kwargs
