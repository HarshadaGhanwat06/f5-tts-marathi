#!/usr/bin/env bash
#
# fix_emotion_selection.sh
#
# Fixes a bug where the chosen emotion always falls back to NEUTRAL.
#
# Root cause: in emotion_layer/synthesize.py, synthesize_tagged_speech()
# calls preprocess_english_words() on the WHOLE tagged text BEFORE parsing
# tags. preprocess_english_words() transliterates every [a-zA-Z]+ token,
# which includes the emotion tag itself, e.g.:
#     [ANGRY]  ->  [अँग्री]
#     [DISGUST]->  [डिसगस्ट]
# The Devanagari tag then does not match the English keys in
# emotion_mapping.json, so the fallback to NEUTRAL kicks in.
#
# Fix: parse the tags FIRST (so they stay English and match the mapping),
# then transliterate ONLY the spoken text of each segment.
#
# This script:
#   1. Backs up synthesize.py.
#   2. Applies the reorder fix.
#   3. Verifies the change.
#   4. Restarts the FastAPI (8000) and UI (7861) to reload.
#
# Usage:
#   bash scripts/fix_emotion_selection.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EMO_DIR="$REPO_ROOT/emotion_layer"
SYNTH="$EMO_DIR/synthesize.py"
BACKUP_DIR="$REPO_ROOT/deploy_backups/emotion_fix_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"

API_PORT=8000
UI_PORT=7861
PYTHON="$REPO_ROOT/f5tts/bin/python"

echo "============================================================"
echo " Fix: chosen emotion falling back to NEUTRAL"
echo "============================================================"

if [[ ! -f "$SYNTH" ]]; then
    echo "[ERROR] synthesize.py not found: $SYNTH"
    exit 1
fi

mkdir -p "$BACKUP_DIR"
mkdir -p "$LOG_DIR"

echo ""
echo "[1] Backing up synthesize.py -> $BACKUP_DIR"
cp -p "$SYNTH" "$BACKUP_DIR/synthesize.py"

echo ""
echo "[2] Applying reorder fix (parse tags first, transliterate text only)..."
python3 - "$SYNTH" <<'EOF'
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# The current (buggy) lines:
old_block = """    tagged_text = preprocess_english_words(tagged_text)  # <-- add this first
    segments = parse_tagged_text(tagged_text)"""

# The corrected lines: parse tags first (keep tags English), then
# transliterate only the spoken text of each segment.
new_block = """    # Parse emotion tags FIRST so the tag stays English (e.g. [ANGRY]) and
    # matches emotion_mapping.json keys. Transliterate ONLY the spoken text.
    segments = parse_tagged_text(tagged_text)
    for _seg in segments:
        _seg["text"] = preprocess_english_words(_seg["text"])"""

if old_block not in content:
    print(f"[ERROR] Expected block not found in {path}:")
    print(repr(old_block))
    sys.exit(2)

content = content.replace(old_block, new_block, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("[OK] synthesize.py updated.")
EOF

echo ""
echo "[VERIFY] Relevant section of synthesize_tagged_speech:"
grep -n "parse_tagged_text\|preprocess_english_words\|segments = \|_seg\[\"text\"\]" "$SYNTH"

echo ""
echo "[3] Restarting FastAPI (port $API_PORT) to reload..."
OLD_API_PID=$(ss -ltnp 2>/dev/null | grep ":$API_PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -z "$OLD_API_PID" ]]; then
    OLD_API_PID=$(pgrep -f "uvicorn emotion_layer.api:app.*$API_PORT" | head -1 || true)
fi
if [[ -n "$OLD_API_PID" ]]; then
    echo "  Stopping API PID $OLD_API_PID..."
    kill "$OLD_API_PID" 2>/dev/null || true
    for i in $(seq 1 20); do kill -0 "$OLD_API_PID" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$OLD_API_PID" 2>/dev/null; then kill -9 "$OLD_API_PID" 2>/dev/null || true; sleep 1; fi
fi
sleep 2
cd "$REPO_ROOT"
nohup "$PYTHON" -m uvicorn emotion_layer.api:app --host 0.0.0.0 --port $API_PORT > "$LOG_DIR/api_emofix.log" 2>&1 &
disown
echo "  New API PID: $!"

echo ""
echo "[4] Restarting UI (port $UI_PORT)..."
OLD_UI_PID=$(ss -ltnp 2>/dev/null | grep ":$UI_PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -z "$OLD_UI_PID" ]]; then
    OLD_UI_PID=$(pgrep -f "emotion_layer/ui.py.*$UI_PORT" | head -1 || true)
fi
if [[ -n "$OLD_UI_PID" ]]; then
    echo "  Stopping UI PID $OLD_UI_PID..."
    kill "$OLD_UI_PID" 2>/dev/null || true
    for i in $(seq 1 20); do kill -0 "$OLD_UI_PID" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$OLD_UI_PID" 2>/dev/null; then kill -9 "$OLD_UI_PID" 2>/dev/null || true; sleep 1; fi
fi
sleep 2
cd "$REPO_ROOT"
nohup "$PYTHON" emotion_layer/ui.py --port $UI_PORT > "$LOG_DIR/ui_emofix.log" 2>&1 &
disown
echo "  New UI PID: $!"

echo ""
echo "============================================================"
echo " FIX APPLIED"
echo "============================================================"
echo "Backup   : $BACKUP_DIR/synthesize.py"
echo "API log  : $LOG_DIR/api_emofix.log"
echo "UI log   : $LOG_DIR/ui_emofix.log"
echo ""
echo "Validation: after services come up, call the API with each "
echo "emotion (HAPPY/SAD/ANGRY/FEAR/DISGUST/NEUTRAL) and confirm the "
echo "'emotion_tag' in /synthesize response headers (x-actual-emotion) "
echo "is NOT NEUTRAL for an explicitly chosen emotion."
echo ""
echo "Rollback:"
echo "  cp $BACKUP_DIR/synthesize.py $REPO_ROOT/emotion_layer/synthesize.py"
echo "  # then restart API (port 8000) and UI (port 7861)"
