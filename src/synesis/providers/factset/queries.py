"""SQL query constants for FactSet database.

All queries use WITH (NOLOCK) hint for read-only access.
Performance notes:
- Avoid ORDER BY DESC on large tables without date filters
- Cache global max dates (price_date, etc.) with 4-hour TTL
- Use date range filters instead of TOP N DESC patterns
"""

# =============================================================================
# TICKER RESOLUTION
# =============================================================================

SECURITY_BY_TICKER = """
SELECT
    t.fsym_id,
    t.ticker_region,
    s.proper_name,
    s.fref_exchange_code,
    s.fref_security_type,
    s.currency,
    s.fsym_security_id
FROM sym_v1.sym_ticker_region t WITH (NOLOCK)
JOIN fgp_v1.fgp_sec_coverage s WITH (NOLOCK) ON t.fsym_id = s.fsym_id
WHERE t.ticker_region = %(ticker)s
"""

SECURITY_BY_FSYM_ID = """
SELECT
    s.fsym_id,
    t.ticker_region,
    s.proper_name,
    s.fref_exchange_code,
    s.fref_security_type,
    s.currency,
    s.fsym_security_id
FROM fgp_v1.fgp_sec_coverage s WITH (NOLOCK)
LEFT JOIN sym_v1.sym_ticker_region t WITH (NOLOCK) ON s.fsym_id = t.fsym_id
WHERE s.fsym_id = %(fsym_id)s
"""

SEARCH_SECURITIES = """
SELECT TOP %(limit)s
    s.fsym_id,
    t.ticker_region,
    s.proper_name,
    s.fref_exchange_code,
    s.fref_security_type,
    s.currency,
    s.fsym_security_id
FROM fgp_v1.fgp_sec_coverage s WITH (NOLOCK)
LEFT JOIN sym_v1.sym_ticker_region t WITH (NOLOCK) ON s.fsym_id = t.fsym_id
WHERE s.proper_name LIKE %(query)s
   OR t.ticker_region LIKE %(query)s
ORDER BY s.proper_name
"""

# Get entity info (country, sector, industry)
# Note: Uses fsym_security_id to join to ff_sec_entity (not fsym_id)
# sector_code and industry_code are codes (not descriptions)
ENTITY_INFO = """
SELECT
    e.iso_country,
    es.sector_code,
    es.industry_code
FROM ff_v3.ff_sec_entity se WITH (NOLOCK)
JOIN sym_v1.sym_entity e WITH (NOLOCK) ON se.factset_entity_id = e.factset_entity_id
LEFT JOIN sym_v1.sym_entity_sector es WITH (NOLOCK) ON e.factset_entity_id = es.factset_entity_id
WHERE se.fsym_id = %(fsym_security_id)s
"""

# =============================================================================
# PRICE DATA
# =============================================================================

# Get global max price date (cache this with 4-hour TTL)
GLOBAL_MAX_PRICE_DATE = """
SELECT MAX(price_date) as max_date
FROM fgp_v1.fgp_global_prices WITH (NOLOCK)
"""

# Get security-specific latest price (for when global max date doesn't have data)
LATEST_PRICE_FOR_SECURITY = """
SELECT TOP 1
    fsym_id,
    price_date,
    price_open,
    price_high,
    price_low,
    price AS price_close,
    volume,
    one_day_pct,
    wtd_pct,
    mtd_pct,
    qtd_pct,
    ytd_pct,
    one_mth_pct,
    three_mth_pct,
    six_mth_pct,
    one_yr_pct,
    two_yr_pct,
    three_yr_pct,
    five_yr_pct,
    ten_yr_pct
FROM fgp_v1.fgp_global_prices WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
ORDER BY price_date DESC
"""

# Get price for specific date (fast: ~26ms)
# Note: Column names match actual FactSet schema (one_mth_pct, three_mth_pct, etc.)
PRICE_BY_DATE = """
SELECT
    fsym_id,
    price_date,
    price_open,
    price_high,
    price_low,
    price AS price_close,
    volume,
    one_day_pct,
    wtd_pct,
    mtd_pct,
    qtd_pct,
    ytd_pct,
    one_mth_pct,
    three_mth_pct,
    six_mth_pct,
    one_yr_pct,
    two_yr_pct,
    three_yr_pct,
    five_yr_pct,
    ten_yr_pct
FROM fgp_v1.fgp_global_prices WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s AND price_date = %(price_date)s
"""

# Get price history for date range
PRICE_HISTORY = """
SELECT
    fsym_id,
    price_date,
    price_open,
    price_high,
    price_low,
    price AS price_close,
    volume,
    one_day_pct,
    wtd_pct,
    mtd_pct,
    qtd_pct,
    ytd_pct,
    one_mth_pct,
    three_mth_pct,
    six_mth_pct,
    one_yr_pct,
    two_yr_pct,
    three_yr_pct,
    five_yr_pct,
    ten_yr_pct
FROM fgp_v1.fgp_global_prices WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
  AND price_date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY price_date
"""

# Get latest prices for multiple securities (batch)
PRICES_BATCH_BY_DATE = """
SELECT
    fsym_id,
    price_date,
    price_open,
    price_high,
    price_low,
    price AS price_close,
    volume,
    one_day_pct,
    ytd_pct,
    one_yr_pct
FROM fgp_v1.fgp_global_prices WITH (NOLOCK)
WHERE fsym_id IN ({fsym_ids})
  AND price_date = %(price_date)s
"""

# =============================================================================
# FUNDAMENTALS
# =============================================================================

# Get max fundamental date for a security (use for date filtering)
FUNDAMENTALS_MAX_DATE_ANNUAL = """
SELECT MAX(date) as max_date
FROM ff_v3.ff_basic_der_af WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
"""

FUNDAMENTALS_MAX_DATE_QUARTERLY = """
SELECT MAX(date) as max_date
FROM ff_v3.ff_basic_der_qf WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
"""

FUNDAMENTALS_MAX_DATE_LTM = """
SELECT MAX(date) as max_date
FROM ff_v3.ff_basic_der_ltm WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
"""

# Annual fundamentals with date filter
# Note: Column mappings based on actual FactSet schema:
#   ff_fyr = fiscal year, ff_eps_dil = diluted EPS, ff_bps = book value per share
#   ff_dps = dividends per share, ff_roe = return on equity, ff_net_mgn = net margin
#   ff_debt_assets = debt to assets, ff_debt_eq = debt to equity
#   ff_entrpr_val_ebitda_oper = EV/EBITDA
FUNDAMENTALS_ANNUAL = """
SELECT
    b.fsym_id,
    b.date AS period_end,
    b.ff_fyr AS fiscal_year,
    'annual' AS period_type,
    d.ff_eps_dil AS eps_diluted,
    b.ff_bps AS bps,
    b.ff_dps AS dps,
    d.ff_roe AS roe,
    d.ff_net_mgn AS net_margin,
    d.ff_debt_assets AS debt_to_assets,
    a.ff_debt_eq AS debt_to_equity,
    a.ff_entrpr_val_ebitda_oper AS ev_to_ebitda
FROM ff_v3.ff_basic_af b WITH (NOLOCK)
LEFT JOIN ff_v3.ff_basic_der_af d WITH (NOLOCK)
    ON b.fsym_id = d.fsym_id AND b.date = d.date
LEFT JOIN ff_v3.ff_advanced_der_af a WITH (NOLOCK)
    ON b.fsym_id = a.fsym_id AND b.date = a.date
WHERE b.fsym_id = %(fsym_id)s
  AND b.date >= %(start_date)s
ORDER BY b.date DESC
"""

# Quarterly fundamentals with date filter
FUNDAMENTALS_QUARTERLY = """
SELECT
    b.fsym_id,
    b.date AS period_end,
    b.ff_fyr AS fiscal_year,
    'quarterly' AS period_type,
    d.ff_eps_dil AS eps_diluted,
    b.ff_bps AS bps,
    b.ff_dps AS dps,
    d.ff_roe AS roe,
    d.ff_net_mgn AS net_margin,
    d.ff_debt_assets AS debt_to_assets,
    a.ff_debt_eq AS debt_to_equity,
    a.ff_entrpr_val_ebitda_oper AS ev_to_ebitda
FROM ff_v3.ff_basic_qf b WITH (NOLOCK)
LEFT JOIN ff_v3.ff_basic_der_qf d WITH (NOLOCK)
    ON b.fsym_id = d.fsym_id AND b.date = d.date
LEFT JOIN ff_v3.ff_advanced_der_qf a WITH (NOLOCK)
    ON b.fsym_id = a.fsym_id AND b.date = a.date
WHERE b.fsym_id = %(fsym_id)s
  AND b.date >= %(start_date)s
ORDER BY b.date DESC
"""

# LTM (Last Twelve Months) fundamentals with date filter
FUNDAMENTALS_LTM = """
SELECT
    b.fsym_id,
    b.date AS period_end,
    NULL AS fiscal_year,
    'ltm' AS period_type,
    d.ff_eps_dil AS eps_diluted,
    b.ff_dps AS dps,
    d.ff_net_mgn AS net_margin
FROM ff_v3.ff_basic_ltm b WITH (NOLOCK)
LEFT JOIN ff_v3.ff_basic_der_ltm d WITH (NOLOCK)
    ON b.fsym_id = d.fsym_id AND b.date = d.date
WHERE b.fsym_id = %(fsym_id)s
  AND b.date >= %(start_date)s
ORDER BY b.date DESC
"""

# Company profile/description
# Note: Uses fsym_security_id to find entity (not fsym_id)
COMPANY_PROFILE = """
SELECT entity_profile AS entity_description
FROM ff_v3.ff_entity_profiles WITH (NOLOCK)
WHERE factset_entity_id = (
    SELECT factset_entity_id
    FROM ff_v3.ff_sec_entity WITH (NOLOCK)
    WHERE fsym_id = %(fsym_security_id)s
)
"""

# =============================================================================
# CORPORATE ACTIONS
# =============================================================================

# Get max effective date for corporate actions
CORPORATE_ACTIONS_MAX_DATE = """
SELECT MAX(effective_date) as max_date
FROM fgp_v1.fgp_ca_events WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
"""

# Corporate actions with date filter
# Note: Column mappings based on actual FactSet schema:
#   ca_event_type_code = event type, amt_gross_trading_adj = dividend amount
#   trading_currency = currency, div_type_code = dividend type
#   price_adj_factor = adjustment factor, dist_new_term/dist_old_term = split ratio
CORPORATE_ACTIONS = """
SELECT
    fsym_id,
    ca_event_type_code AS ca_event_type,
    effective_date,
    record_date,
    pay_date,
    amt_gross_trading_adj AS gross_dvd_cash,
    trading_currency AS gross_dvd_cash_currency,
    div_type_code AS dvd_type_desc,
    price_adj_factor AS adj_factor,
    dist_new_term AS adj_shares_to,
    dist_old_term AS adj_shares_from,
    short_desc
FROM fgp_v1.fgp_ca_events WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
  AND effective_date >= %(start_date)s
ORDER BY effective_date DESC
"""

# Dividends only
DIVIDENDS = """
SELECT
    fsym_id,
    ca_event_type_code AS ca_event_type,
    effective_date,
    record_date,
    pay_date,
    amt_gross_trading_adj AS gross_dvd_cash,
    trading_currency AS gross_dvd_cash_currency,
    div_type_code AS dvd_type_desc
FROM fgp_v1.fgp_ca_events WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
  AND ca_event_type_code = 'DVC'
  AND effective_date >= %(start_date)s
ORDER BY effective_date DESC
"""

# Splits only (FSP = Forward Split, RSP = Reverse Split, BNS = Bonus Issue)
SPLITS = """
SELECT
    fsym_id,
    ca_event_type_code AS ca_event_type,
    effective_date,
    price_adj_factor AS adj_factor,
    dist_new_term AS adj_shares_to,
    dist_old_term AS adj_shares_from,
    short_desc
FROM fgp_v1.fgp_ca_events WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
  AND ca_event_type_code IN ('FSP', 'RSP', 'BNS')
  AND effective_date >= %(start_date)s
ORDER BY effective_date DESC
"""

# =============================================================================
# ADJUSTMENT FACTORS
# =============================================================================

ADJUSTMENT_FACTORS = """
SELECT
    effective_date,
    adj_factor_combined,
    div_spl_spin_adj_factor
FROM fgp_v1.fgp_ca_adj_factors WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
  AND effective_date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY effective_date
"""

# All adjustment factors for a security (for historical price adjustment)
# Gets all factors up to a given date to enable forward-fill logic
# adj_factor_combined is populated on split dates with cumulative factor
# div_spl_spin_adj_factor tracks dividend adjustments since last split
ADJUSTMENT_FACTORS_FOR_HISTORY = """
SELECT
    effective_date,
    adj_factor_combined,
    div_spl_spin_adj_factor
FROM fgp_v1.fgp_ca_adj_factors WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
  AND effective_date <= %(end_date)s
ORDER BY effective_date
"""

# =============================================================================
# SHARES OUTSTANDING
# =============================================================================

# Current shares outstanding (uses fsym_security_id, not fsym_id)
# Note: Column mappings based on actual FactSet schema:
#   one_adr_eq = ADR ratio (how many ordinary shares = 1 ADR)
#   hasadr_flag = whether security has ADR
SHARES_OUTSTANDING_CURRENT = """
SELECT TOP 1
    fsym_security_id,
    report_date,
    adj_shares_outstanding,
    one_adr_eq AS adr_share_ratio,
    hasadr_flag
FROM fgp_v1.fgp_shares_sec_curr WITH (NOLOCK)
WHERE fsym_security_id = %(fsym_security_id)s
ORDER BY report_date DESC
"""

# Historical shares outstanding
# Note: Need to check if fgp_shares_sec_hist has same columns
SHARES_OUTSTANDING_HISTORY = """
SELECT
    fsym_security_id,
    report_date,
    adj_shares_outstanding
FROM fgp_v1.fgp_shares_sec_hist WITH (NOLOCK)
WHERE fsym_security_id = %(fsym_security_id)s
  AND report_date BETWEEN %(start_date)s AND %(end_date)s
ORDER BY report_date DESC
"""

# Get fsym_security_id from fsym_id
GET_SECURITY_ID = """
SELECT fsym_security_id
FROM fgp_v1.fgp_sec_coverage WITH (NOLOCK)
WHERE fsym_id = %(fsym_id)s
"""
