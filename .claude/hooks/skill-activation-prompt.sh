#!/bin/bash
# ============================================================================
# SKILL ACTIVATION PROMPT HOOK (v2.1)
# ============================================================================
# Trigger: Runs on every user prompt submission
# Purpose: Suggests relevant skills based on keywords AND intent patterns
#
# Dependencies: NONE - pure bash, works on any Unix/Linux/macOS
#
# Supports:
# - Keywords matching (simple word matching)
# - Intent patterns (regex matching for user intent)
# - Enforcement levels (suggest, warn, block)
# ============================================================================

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
SKILL_RULES="$PROJECT_DIR/.claude/skills/skill-rules.json"

# Read stdin (Claude Code pipes the prompt JSON here)
INPUT=$(cat 2>/dev/null || true)

# Quick exits
[ -z "$INPUT" ] && exit 0
[ ! -f "$SKILL_RULES" ] && exit 0

# Extract prompt text from JSON
# Input format: {"prompt": "the user's prompt text"}
PROMPT=$(echo "$INPUT" | sed -n 's/.*"prompt"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -z "$PROMPT" ] && exit 0

# Lowercase for matching
PROMPT_LOWER=$(echo "$PROMPT" | tr 'A-Z' 'a-z')

# Read skill-rules.json
RULES=$(cat "$SKILL_RULES")

# Initialize results
SUGGESTIONS=""
WARNINGS=""
BLOCKS=""

# Helper: Check if string contains substring
contains() {
  echo "$1" | grep -qi "$2" 2>/dev/null
}

# Parse skills from JSON (v2.0 format)
# We look for skill blocks and extract their config
SKILL_NAMES=$(echo "$RULES" | grep -o '"[a-zA-Z0-9_-]*"[[:space:]]*:[[:space:]]*{' | grep -v '"version"\|"description"\|"_template"\|"skills"' | sed 's/"//g; s/:.*//g')

for SKILL_NAME in $SKILL_NAMES; do
  # Extract the skill block (this is a simplified extraction)
  # In production, you'd want to use jq, but we're keeping it bash-only
  
  MATCHED_KEYWORDS=""
  MATCHED_INTENTS=""
  ENFORCEMENT="suggest"
  
  # Extract keywords for this skill
  # Look for the keywords array after the skill name
  SKILL_BLOCK=$(echo "$RULES" | tr '\n' ' ' | sed -n "s/.*\"$SKILL_NAME\"[^{]*{\\([^}]*\\)}.*/\\1/p")
  
  # Extract enforcement level if present
  if echo "$SKILL_BLOCK" | grep -q '"enforcement"' 2>/dev/null; then
    ENFORCEMENT=$(echo "$SKILL_BLOCK" | sed -n 's/.*"enforcement"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  fi
  
  # Extract keywords array
  KEYWORDS=$(echo "$SKILL_BLOCK" | sed -n 's/.*"keywords"[^[]*\[\([^]]*\)\].*/\1/p' | tr ',' '\n' | tr -d '"' | tr -d ' ' | grep -v '^$')
  
  # Match keywords
  for KW in $KEYWORDS; do
    KW_LOWER=$(echo "$KW" | tr 'A-Z' 'a-z')
    if contains "$PROMPT_LOWER" "$KW_LOWER"; then
      [ -z "$MATCHED_KEYWORDS" ] && MATCHED_KEYWORDS="$KW" || MATCHED_KEYWORDS="$MATCHED_KEYWORDS, $KW"
    fi
  done
  
  # Extract and match intent patterns
  # v2.1 FIX: Use while read loop instead of for loop to preserve patterns with spaces
  MATCHED_INTENTS=$(echo "$SKILL_BLOCK" | sed -n 's/.*"intentPatterns"[^[]*\[\([^]]*\)\].*/\1/p' | tr ',' '\n' | tr -d '"' | grep -v '^$' | while IFS= read -r PATTERN; do
    PATTERN=$(echo "$PATTERN" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -z "$PATTERN" ] && continue

    if echo "$PROMPT_LOWER" | grep -Eq "$PATTERN" 2>/dev/null; then
      echo "intent"
      break
    fi
  done)
  
  # If we have matches, categorize by enforcement level
  if [ -n "$MATCHED_KEYWORDS" ] || [ -n "$MATCHED_INTENTS" ]; then
    MATCH_INFO="$SKILL_NAME"
    [ -n "$MATCHED_KEYWORDS" ] && MATCH_INFO="$MATCH_INFO (keywords: $MATCHED_KEYWORDS)"
    
    case "$ENFORCEMENT" in
      "block")
        [ -z "$BLOCKS" ] && BLOCKS="$MATCH_INFO" || BLOCKS="$BLOCKS\n$MATCH_INFO"
        ;;
      "warn")
        [ -z "$WARNINGS" ] && WARNINGS="$MATCH_INFO" || WARNINGS="$WARNINGS\n$MATCH_INFO"
        ;;
      *)
        [ -z "$SUGGESTIONS" ] && SUGGESTIONS="$MATCH_INFO" || SUGGESTIONS="$SUGGESTIONS\n$MATCH_INFO"
        ;;
    esac
  fi
done

# Output results
if [ -n "$BLOCKS" ] || [ -n "$WARNINGS" ] || [ -n "$SUGGESTIONS" ]; then
  echo ""
  echo "🎯 SKILL ACTIVATION"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  if [ -n "$BLOCKS" ]; then
    echo "🛑 REQUIRED (read before proceeding):"
    echo -e "$BLOCKS" | while read -r line; do
      echo "   • $line"
    done
    echo ""
  fi
  
  if [ -n "$WARNINGS" ]; then
    echo "⚠️  RECOMMENDED:"
    echo -e "$WARNINGS" | while read -r line; do
      echo "   • $line"
    done
    echo ""
  fi
  
  if [ -n "$SUGGESTIONS" ]; then
    echo "💡 SUGGESTED:"
    echo -e "$SUGGESTIONS" | while read -r line; do
      echo "   • $line"
    done
    echo ""
  fi
  
  echo "📂 See .claude/skills/ for detailed guidelines."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
fi

exit 0
