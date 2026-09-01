#!/usr/bin/env bash
#
# fix_emotion_tag_leak.sh
#
# Fixes the bug where the emotion tag text (e.g. [डिसगस्ट], [अँग्री]) is read
# aloud as part of the synthesized speech instead of being treated as a tag.
#
# Root cause: tag_parser.py uses re.split(r'\[(\w+)\]', ...) with the default
# Python \w, which does not reliably match Devanagari/Marathi characters. When
# an emotion tag is written in Devanagari, the tag is NOT stripped and leaks
# into the synthesized text.
#
# This script:
#   1. Backs up tag_parser.py.
#   2. Shows the current (broken) behavior with Devanagari tags.
#   3. Replaces the regex so tags matching ANY characters between [ and ]
#      are recognized (English + Devanagari).
#   4. Re-tests to confirm the tag is stripped.
#   5. Restarts the FastAPI (8000) and UI (7861) to reload the module.
#
# Usage:
#   bash scripts/fix_emotion_tag_leak.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BACK_SOURCE[0]:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EMO_DIR="$REPO_ROOT/emotion_layer"
TAG_PARSER="$EMO_DIR/tag_parser.py"
BACKUP_DIR="$REPO_ROOT/deploy_backups/tag_parser_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"

API_PORT=8000
UI_PORT=7861
PYTHON="$REPO_ROOT/f5tts/bin/python"

echo "============================================================"
echo " Fix: emotion tag text leaking into synthesized speech"
echo "============================================================"

if [[ ! -f "$TAG_PARSER" ]]; then
    echo "[ERROR] tag_parser.py not found: $TAG_PARSER"
    exit 1
fi

mkdir -p "$BACKUP_DIR"
mkdir -p "$LOG_DIR"

echo ""
echo "[1] Backing up tag_parser.py -> $BACKUP_DIR"
cp -p "$TAG_PARSER" "$BACKUP_DIR/tag_parser.py"

echo ""
echo "[2] Current behavior with Devanagari tag (should be BROKEN):"
python3 - <<'EOF'
import sys
sys.path.insert(0, "/root/f5-tts-marathi/emotion_layer")
try:
    from tag_parser import parse_tagged_text
except Exception as e:
    print(f"  (import failed: {e})")
    sys.exit(0)
for s in ["[अँग्री] मी ऑफिसमध्ये आहे, तुम्ही लंच सुरू करा.",
          "[डिसगस्ट] मी ऑफिसमध्ये आहे, तुम्ही लंच सुरू करा.",
          "[HAPPY] नमस्कार"]:
    print(f"  input : {s!r}")
    print(f"  output: {parse_tagged_text(s)}")
EOF

echo ""
echo "[3] Applying regex fix to $TAG_PARSER ..."
# Replace the brittle \w+ tag regex with [^\]]+ (matches any char except ']',
# so both English and Devanagari emotion tags are recognized).
# Do the replacement with Python to avoid bash/perl escaping issues.
python3 - "$TAG_PARSER" <<'EOF'
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = "parts = re.split(r'\\[(\\w+)\\]', text)"
new = "parts = re.split(r'\\[([^\\]]+)\\]', text)"

if old not in content:
    print(f"[ERROR] Could not find expected line in {path}: {old!r}")
    sys.exit(1)

content = content.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("[OK] Regex updated.")
EOF

echo ""
echo "[VERIFY] The tag-parsing line is now:"
grep -n "re.split" "$TAG_PARSER"

echo ""
echo "[4] Verifying the fix (Devanagari tag should now be stripped):"
python3 - <<'EOF'
import sys, importlib, importlib.util
p = "/root/f5-tts-marathi/emotion_layer/tag_parser.py"
spec = importlib.util.spec_from_file_location("tag_parser_fixed", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
for s in ["[अँग्री] मी ऑफिसमध्ये आहे, तुम्ही लंच सुरू करा.",
          "[डिसगस्ट] मी ऑफिसमध्ये आहे, तुम्ही लंच सुरू करा.",
          "[HAPPY] नमस्कार",
          "[SAD] आज मी दुःखी आहे"]:
    print(f"  input : {s!r}")
    print(f"  output: {m.parse_tagged_text(s)}")
EOF

echo ""
echo "[5] Restarting FastAPI (port $API_PORT) to reload the module..."
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
nohup "$PYTHON" -m uvicorn emotion_layer.api:app --host 0.0.0.0 --port $API_PORT > "$LOG_DIR/api_fixtag.log" 2>&1 &
disown
echo "  New API PID: $!"

echo ""
echo "[6] Restarting UI (port $UI_PORT)..."
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
nohup "$PYTHON" emotion_layer/ui.py --port $UI_PORT > "$LOG_DIR/ui_fixtag.log" 2>&1 &
disown
echo "  New UI PID: $!"

echo ""
echo "============================================================"
echo " FIX APPLIED"
echo "============================================================"
echo "Backup                : $BACKUP_DIR/tag_parser.py"
echo "API log               : $LOG_DIR/api_fixtag.log"
echo "UI log                : $LOG_DIR/ui_fixtag.log"
echo ""
echo "NOTE: With Devanagari tags like [अँग्री] / [डिसगस्ट], the tag will now"
echo "be stripped from the spoken text. However, emotion_mapping.json keys are"
echo "English (NEUTRAL/HAPPY/SAD/...), so a Devanagari tag falls back to"
echo "NEUTRAL. If you want Devanagari tags to actually drive emotion, those"
echo "need a separate mapping addition (not part of this fix)."
echo ""
echo "Rollback:"
echo "  cp $BACKUP_DIR/tag_parser.py $REPO_ROOT/emotion_layer/tag_parser.py"
echo "  # then restart API (port 8000) and UI (port 7861)"
