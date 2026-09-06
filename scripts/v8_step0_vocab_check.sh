#!/usr/bin/env bash
# =============================================================================
# v8_step0_vocab_check.sh
#
# STEP 0 — VOCAB VERIFICATION.
#
# Rule out a vocab-mismatch problem BEFORE assuming a training-data problem:
# this has been the actual root cause in prior similar issues on this project.
#
# Checks, precisely and without assumption:
#    1. ॅ (U+0945 CANDRA E, dependent vowel sign used in मॅन/कॅन/लॅन/व्हॅन)
#    2. ॲ (U+0972 CANDRA A, independent vowel)
#    3. Total token count in vocab_extended.txt (must equal checkpoint
#       text_embed rows - 1).
#
# Prints the EXACT grep output for both characters. Writes a state file so
# downstream scripts know whether embedding surgery is required:
#   /tmp/v8_vocab_state.txt   contains: missing_char_codes=<comma-sep, or empty>
#
# Usage (server):
#   bash scripts/v8_step0_vocab_check.sh
#
# Exit codes:
#   0 = both present, no embedding surgery needed
#   2 = at least one character MISSING -> embedding surgery required (Step 3.3)
# =============================================================================
set -euo pipefail

VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"
STATE_FILE="/tmp/v8_vocab_state.txt"

if [[ ! -f "$VOCAB" ]]; then
    echo "[ERROR] vocab not found: $VOCAB" >&2
    exit 1
fi

echo "============================================================"
echo " STEP 0 — Vocab verification for Chandra_Vyanjan_Focus_v8"
echo "============================================================"
echo "Vocab file: $VOCAB"
echo ""

echo "[1] Checking ॅ (U+0945 CANDRA E vowel sign)..."
echo "    grep -n \"ॅ\" $VOCAB"
GREP_CHANDRA_E=$(grep -n "ॅ" "$VOCAB" || true)
if [[ -n "$GREP_CHANDRA_E" ]]; then
    echo "    FOUND ->"
    echo "$GREP_CHANDRA_E" | sed 's/^/      /'
else
    echo "    NOT FOUND -> ॅ is MISSING from the vocab"
fi
echo ""

echo "[2] Checking ॲ (U+0972 CANDRA A independent vowel)..."
echo "    grep -n \"ॲ\" $VOCAB"
GREP_CHANDRA_A=$(grep -n "ॲ" "$VOCAB" || true)
if [[ -n "$GREP_CHANDRA_A" ]]; then
    echo "    FOUND ->"
    echo "$GREP_CHANDRA_A" | sed 's/^/      /'
else
    echo "    NOT FOUND -> ॲ is MISSING from the vocab"
fi
echo ""

echo "[3] Total token count:"
VOCAB_COUNT=$(grep -vc '^\s*$' "$VOCAB")
echo "    vocab_extended.txt tokens = $VOCAB_COUNT  (checkpoint text_embed rows must be $((VOCAB_COUNT + 1)))"

# ---------------------------------------------------------------------------
# Determine missing char codes (code points only, machine-readable state)
# ---------------------------------------------------------------------------
MISSING=""
if [[ -z "$GREP_CHANDRA_E" ]]; then
    MISSING="0945"
fi
if [[ -z "$GREP_CHANDRA_A" ]]; then
    if [[ -n "$MISSING" ]]; then MISSING="$MISSING,0972"; else MISSING="0972"; fi
fi

echo "0945|$GREP_CHANDRA_E" > "$STATE_FILE"
echo "0972|$GREP_CHANDRA_A" >> "$STATE_FILE"
echo "TOKENS|$VOCAB_COUNT" >> "$STATE_FILE"
echo "MISSING|$MISSING" >> "$STATE_FILE"
echo ""
echo "[STATE] $STATE_FILE"
cat "$STATE_FILE"

echo ""
if [[ -n "$MISSING" ]]; then
    echo "RESULT  : CHANDRA ॅ/ॲ vocabulary is INCOMPLETE."
    echo "          Missing char U+$(echo "$MISSING" | tr ',' ' ' | tr ' ' $'\n' | sed 's/^/U+/' | paste -sd, -)"
    echo "          -> Run bash scripts/v8_prepare_pretrain.sh (embedding surgery enabled)."
    exit 2
else
    echo "RESULT  : Both ॅ (U+0945) and ॲ (U+0972) are PRESENT."
    echo "          No embedding surgery needed; this is a training-data coverage problem."
    echo "          -> Run bash scripts/v8_filter_dataset.sh next."
    exit 0
fi