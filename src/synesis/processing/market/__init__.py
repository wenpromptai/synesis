"""Market movers pipeline — daily market snapshot + top movers to Discord."""

from synesis.processing.market.job import market_movers_job

__all__ = ["market_movers_job"]
