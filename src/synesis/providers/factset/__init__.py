"""FactSet provider implementations.

This module exports the FactSet data provider for SQL Server database access:
- FactSetProvider: Historical prices, fundamentals, corporate actions, shares outstanding

Data is sourced from FactSet's SQL Server database with schemas:
- fgp_v1: Global prices, security coverage, corporate actions, adjustment factors, shares
- ff_v3: Fundamentals (basic, derived - annual/quarterly/LTM)
- sym_v1: Ticker mappings and sector classification
"""

from synesis.providers.factset.client import FactSetClient
from synesis.providers.factset.models import (
    FactSetCorporateAction,
    FactSetFundamentals,
    FactSetPrice,
    FactSetSecurity,
    FactSetSharesOutstanding,
)
from synesis.providers.factset.provider import FactSetProvider

__all__ = [
    # Main provider
    "FactSetProvider",
    # Database client
    "FactSetClient",
    # Models
    "FactSetPrice",
    "FactSetSecurity",
    "FactSetFundamentals",
    "FactSetCorporateAction",
    "FactSetSharesOutstanding",
]
