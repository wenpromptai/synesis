"""News categorizer - hybrid rule-based + LLM fallback.

Categorizes news messages into:
- breaking: Unexpected events (*BREAKING, JUST IN, sudden announcements)
- economic_calendar: Scheduled releases (CPI, NFP, FOMC, GDP, earnings dates)
- other: Analysis, commentary, opinions, general news

Rules are checked first (fast, no LLM cost).
If ambiguous, the LLM classifier determines the category.
"""

import re

from synesis.core.logging import get_logger
from synesis.processing.news.models import NewsCategory, UrgencyLevel

logger = get_logger(__name__)

# =============================================================================
# Rule-based patterns
# =============================================================================

# Breaking news indicators (case-insensitive)
BREAKING_PATTERNS = [
    r"\*+\s*BREAKING",  # *BREAKING, ** BREAKING
    r"^BREAKING:",  # BREAKING: at start
    r"JUST IN:",
    r"JUST IN -",
    r"FLASH:",
    r"ALERT:",
    r"URGENT:",
    r"\*+[A-Z\s]{3,100}\*+",  # *ALL CAPS HEADLINE* pattern (length-limited)
    r"^\*[A-Z]",  # Starts with *CAPS (DeItaone style)
]

# Economic calendar indicators (case-insensitive)
ECONOMIC_CALENDAR_PATTERNS = [
    # Fed / Monetary Policy
    r"\bFOMC\b",
    r"\bFED\s+(FUNDS|RATE|DECISION|MEETING|MINUTES|CHAIR|POWELL)\b",
    r"\bRATE\s+(CUT|HIKE|DECISION|UNCHANGED)\b",
    r"\bBASIS\s+POINTS?\b",
    r"\b\d+\s*BPS\b",  # 25bps, 50 bps
    # Economic Data Releases
    r"\bCPI\b",  # Consumer Price Index
    r"\bPPI\b",  # Producer Price Index
    r"\bPCE\b",  # Personal Consumption Expenditures
    r"\bNFP\b",  # Non-Farm Payrolls
    r"\bNON-?FARM\s+PAYROLLS?\b",
    r"\bJOBS\s+REPORT\b",
    r"\bUNEMPLOYMENT\s+RATE\b",
    r"\bJOBLESS\s+CLAIMS\b",
    r"\bGDP\b",
    r"\bRETAIL\s+SALES\b",
    r"\bHOUSING\s+STARTS\b",
    r"\bISM\b",  # ISM Manufacturing/Services
    r"\bPMI\b",  # Purchasing Managers Index
    r"\bDURABLE\s+GOODS\b",
    r"\bCONSUMER\s+(CONFIDENCE|SENTIMENT)\b",
    # Scheduled corporate events
    r"\bEARNINGS\s+(RELEASE|REPORT|CALL|BEAT|MISS)\b",
    r"\bQ[1-4]\s+(EARNINGS|RESULTS)\b",
    r"\bGUIDANCE\b",
    # Time indicators for scheduled releases
    r"\bRELEASED?\s+AT\b",
    r"\bSCHEDULED\s+(FOR|AT)\b",
    r"\bDUE\s+(AT|TODAY|TOMORROW)\b",
    r"\bEXPECTED\s+AT\b",
]

# Compile patterns for efficiency
_breaking_regex = re.compile("|".join(BREAKING_PATTERNS), re.IGNORECASE)
_calendar_regex = re.compile("|".join(ECONOMIC_CALENDAR_PATTERNS), re.IGNORECASE)


def categorize_by_rules(text: str) -> NewsCategory | None:
    """Categorize news using rule-based pattern matching.

    Args:
        text: The message text to categorize

    Returns:
        NewsCategory if confidently matched, None if ambiguous (needs LLM)
    """
    # Check for breaking news patterns
    is_breaking = bool(_breaking_regex.search(text))

    # Check for economic calendar patterns
    is_calendar = bool(_calendar_regex.search(text))

    # Decision logic
    if is_breaking and not is_calendar:
        logger.debug("Rule-based: breaking", text_preview=text[:50])
        return NewsCategory.breaking

    if is_calendar and not is_breaking:
        logger.debug("Rule-based: economic_calendar", text_preview=text[:50])
        return NewsCategory.economic_calendar

    if is_breaking and is_calendar:
        # Both patterns matched - likely breaking news about scheduled event
        # e.g., "*FED CUTS RATES BY 25BPS" - this is breaking news about calendar event
        # Prioritize breaking since it's time-sensitive
        logger.debug("Rule-based: breaking (calendar event announced)", text_preview=text[:50])
        return NewsCategory.breaking

    # Neither pattern matched strongly - could be analysis or other
    # Check if it looks like general commentary
    if _looks_like_analysis(text):
        logger.debug("Rule-based: other (analysis)", text_preview=text[:50])
        return NewsCategory.other

    # Ambiguous - let LLM decide
    logger.debug("Rule-based: ambiguous, needs LLM", text_preview=text[:50])
    return None


def _looks_like_analysis(text: str) -> bool:
    """Check if text looks like analysis/commentary rather than news."""
    analysis_indicators = [
        r"\bI\s+THINK\b",
        r"\bIN\s+MY\s+(VIEW|OPINION)\b",
        r"\bMY\s+TAKE\b",
        r"\bANALYSIS\b",
        r"\bCOMMENTARY\b",
        r"\bOPINION\b",
        r"\bTHREAD\b",
        r"^RT\s+@",  # Retweet
        r"\bIMO\b",
        r"\bIMHO\b",
    ]
    pattern = re.compile("|".join(analysis_indicators), re.IGNORECASE)
    return bool(pattern.search(text))


def categorize_news(text: str, llm_category: NewsCategory | None = None) -> NewsCategory:
    """Categorize news using hybrid approach.

    1. Try rule-based categorization first (fast, free)
    2. If ambiguous and LLM category provided, use that
    3. Otherwise default to "other"

    Args:
        text: The message text
        llm_category: Optional category from LLM classification

    Returns:
        NewsCategory
    """
    # Try rules first
    rule_category = categorize_by_rules(text)

    if rule_category is not None:
        return rule_category

    # Rules ambiguous - use LLM category if available
    if llm_category is not None:
        logger.debug("Using LLM category", category=llm_category.value)
        return llm_category

    # Default to other
    return NewsCategory.other


# =============================================================================
# Urgency Classification Rules
# =============================================================================

# Critical urgency - act IMMEDIATELY
CRITICAL_PATTERNS = [
    r"\*+\s*(BREAKING|FLASH|ALERT)",  # Breaking markers with asterisks
    r"^(BREAKING|FLASH|ALERT)[:\s]",  # Breaking markers at start of message
    r"(RATE|FED|FOMC).*(DECISION|CUT|HIKE)",  # Fed decisions
    r"\bEMERGENCY\b",
    r"\bSURPRISE\b",
    r"\bUNEXPECTED\b",
]

# High urgency - act FAST
HIGH_PATTERNS = [
    r"\b(CPI|PPI|PCE|NFP|GDP)\b.*\d",  # Economic data WITH numbers
    r"\bEARNINGS\b.*(BEAT|MISS)",  # Earnings with outcome
    r"\d+\s*BPS",  # Basis points moves
]

# Low urgency - background noise, opinions, and promotional content
LOW_PATTERNS = [
    # Opinions and analysis
    r"\bOPINION\b",
    r"\bANALYSIS\b",
    r"\bCOMMENTARY\b",
    r"^RT\s+@",  # Retweets
    r"\bTHREAD\b",
    r"\bIMO\b",
    r"\bI\s+THINK\b",
    # Promotional/spam content
    r"\bBOOST\s+(THIS|THE|OUR)\b",  # Telegram boost requests
    r"\bSUBSCRIBE\b",
    r"\bFOLLOW\s+(US|ME|FOR)\b",
    r"\bJOIN\s+(US|OUR|THE)\b",
    r"\bSIGN\s*UP\b",
    r"\bREFERRAL\b",
    r"\bAFFILIATE\b",
    r"\bPROMO(TION|TIONAL)?\b",
    r"\bGIVEAWAY\b",
    r"\bAIRDROP\b",
    r"\bFREE\s+(MONEY|CRYPTO|TOKEN|NFT)\b",
    r"\bLIMITED\s+(TIME|OFFER)\b",
    r"\bACT\s+(NOW|FAST)\b",
    r"\bDON'?T\s+MISS\b",
    r"\bLAST\s+CHANCE\b",
    r"\bEXCLUSIVE\s+(OFFER|DEAL|ACCESS)\b",
    r"\bUSE\s+(MY\s+)?CODE\b",  # Referral codes
    r"\bDISCOUNT\s+CODE\b",
    r"\bt\.me/\w+\?boost\b",  # Telegram boost links
    r"\bbit\.ly/",  # URL shorteners often used in spam
    r"\btinyurl\.com/",
]

# Compile patterns for efficiency
_critical_regex = re.compile("|".join(CRITICAL_PATTERNS), re.IGNORECASE)
_high_regex = re.compile("|".join(HIGH_PATTERNS), re.IGNORECASE)
_low_regex = re.compile("|".join(LOW_PATTERNS), re.IGNORECASE)


def classify_urgency_by_rules(text: str) -> UrgencyLevel | None:
    """Classify urgency using rule-based pattern matching.

    Returns:
        UrgencyLevel if confidently matched, None if ambiguous (needs LLM)
    """
    # Check critical patterns first (highest priority)
    if _critical_regex.search(text):
        logger.debug("Rule-based urgency: critical", text_preview=text[:50])
        return UrgencyLevel.critical

    # Check high patterns
    if _high_regex.search(text):
        logger.debug("Rule-based urgency: high", text_preview=text[:50])
        return UrgencyLevel.high

    # Check low patterns
    if _low_regex.search(text):
        logger.debug("Rule-based urgency: low", text_preview=text[:50])
        return UrgencyLevel.low

    # Ambiguous - let LLM decide
    logger.debug("Rule-based urgency: ambiguous, needs LLM", text_preview=text[:50])
    return None
