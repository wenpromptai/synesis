"""Deterministic scoring for USCompanyAnalyst.

All functions are pure — no I/O, no LLM. They take pre-fetched data
and return numeric scores or structured red flags.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from synesis.processing.intelligence.models import RedFlag


def compute_piotroski_f(
    net_income_current: float | None,
    operating_cf_current: float | None,
    roa_current: float | None,
    roa_previous: float | None,
    long_term_debt_current: float | None,
    long_term_debt_previous: float | None,
    current_ratio_current: float | None,
    current_ratio_previous: float | None,
    shares_outstanding_current: float | None,
    shares_outstanding_previous: float | None,
    gross_margin_current: float | None,
    gross_margin_previous: float | None,
    asset_turnover_current: float | None,
    asset_turnover_previous: float | None,
) -> int | None:
    """Compute Piotroski F-Score (0-9) from financial data.

    Returns None if all inputs are None (insufficient data).
    """
    criteria: list[bool | None] = [
        # Profitability (4 criteria)
        net_income_current > 0 if net_income_current is not None else None,
        operating_cf_current > 0 if operating_cf_current is not None else None,
        (roa_current > roa_previous)
        if roa_current is not None and roa_previous is not None
        else None,
        (operating_cf_current > net_income_current)
        if operating_cf_current is not None and net_income_current is not None
        else None,
        # Leverage / Liquidity (3 criteria)
        (long_term_debt_current < long_term_debt_previous)
        if long_term_debt_current is not None and long_term_debt_previous is not None
        else None,
        (current_ratio_current > current_ratio_previous)
        if current_ratio_current is not None and current_ratio_previous is not None
        else None,
        (shares_outstanding_current <= shares_outstanding_previous)
        if shares_outstanding_current is not None and shares_outstanding_previous is not None
        else None,
        # Operating efficiency (2 criteria)
        (gross_margin_current > gross_margin_previous)
        if gross_margin_current is not None and gross_margin_previous is not None
        else None,
        (asset_turnover_current > asset_turnover_previous)
        if asset_turnover_current is not None and asset_turnover_previous is not None
        else None,
    ]
    valid = [c for c in criteria if c is not None]
    if not valid:
        return None
    return sum(valid)


def compute_beneish_m(
    dsri: float | None,
    gmi: float | None,
    aqi: float | None,
    sgi: float | None,
    depi: float | None,
    sgai: float | None,
    lvgi: float | None,
    tata: float | None,
) -> float | None:
    """Compute Beneish M-Score for earnings manipulation detection.

    M > -1.78 suggests likely manipulation.
    Requires 2 years of financial data to compute input indices.
    Returns None if any component is None.
    """
    components = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
    if any(c is None for c in components):
        return None

    return (
        -4.84
        + 0.920 * dsri  # type: ignore[operator]
        + 0.528 * gmi  # type: ignore[operator]
        + 0.404 * aqi  # type: ignore[operator]
        + 0.892 * sgi  # type: ignore[operator]
        + 0.115 * depi  # type: ignore[operator]
        - 0.172 * sgai  # type: ignore[operator]
        + 4.679 * tata  # type: ignore[operator]
        - 0.327 * lvgi  # type: ignore[operator]
    )


def detect_insider_cluster(
    transactions: list[dict[str, Any]],
    window_days: int = 14,
    min_insiders: int = 3,
) -> bool:
    """Detect if 3+ unique insiders traded in the same direction within a window."""
    if len(transactions) < min_insiders:
        return False

    dated: list[tuple[date, str]] = []
    for t in transactions:
        try:
            d = date.fromisoformat(str(t["transaction_date"]))
            dated.append((d, t["owner_name"]))
        except (KeyError, ValueError):
            continue

    dated.sort(key=lambda x: x[0])
    window = timedelta(days=window_days)

    for i, (d_start, _) in enumerate(dated):
        names_in_window: set[str] = set()
        for d_other, name in dated[i:]:
            if d_other - d_start <= window:
                names_in_window.add(name)
        if len(names_in_window) >= min_insiders:
            return True

    return False


def detect_red_flags(
    late_filings: list[dict[str, Any]],
    insider_transactions: list[dict[str, Any]],
    financial_data: dict[str, Any],
) -> list[RedFlag]:
    """Detect red flags from multiple data sources."""
    flags: list[RedFlag] = []

    # Late filing alerts
    for lf in late_filings:
        flags.append(
            RedFlag(
                category="disclosure",
                flag="late_filing",
                severity="critical",
                evidence=f"{lf.get('form_type', 'NT')} filed on {lf.get('filed_date', 'unknown')}",
            )
        )

    # Insider selling cluster
    sells = [t for t in insider_transactions if t.get("transaction_code") == "S"]
    if detect_insider_cluster(sells):
        names = sorted({t["owner_name"] for t in sells})
        flags.append(
            RedFlag(
                category="governance",
                flag="insider_selling_cluster",
                severity="critical",
                evidence=f"{len(names)} insiders selling: {', '.join(names[:5])}",
            )
        )

    # Cash flow divergence: positive net income but negative operating CF
    net_income = financial_data.get("net_income")
    operating_cf = financial_data.get("operating_cf")
    if net_income is not None and operating_cf is not None and net_income > 0 and operating_cf < 0:
        flags.append(
            RedFlag(
                category="financial",
                flag="cash_flow_divergence",
                severity="warning",
                evidence=f"Net income ${net_income:,.0f} but operating CF ${operating_cf:,.0f}",
            )
        )

    return flags
