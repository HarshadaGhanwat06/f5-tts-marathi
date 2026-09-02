#!/usr/bin/env bash
#
# deploy_cartesia_v6.sh
#
# Deploy Cartesia_Rasa_Combined_v6 to the live services: FastAPI on port 8000
# and Gradio UI on port 7861.
#
# New checkpoint:
#   /root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_v6/model_last.pt
# Vocabulary (unchanged, 138 tokens):
#   /root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt
#
# Steps:
#   1. Backup production files (synthesize.py, api.py).
#   2. Force emotion_layer/synthesize.py to the v6 checkpoint + extended vocab.
#   3. Force the model identifier in emotion_layer/api.py to v6.
#   4. Restart FastAPI (8000) + Gradio UI (7861).
#   5. Validate health, model-info, and synthesis.
#   6. Print rollback procedure.
#
# Usage:
#   bash scripts/deploy_cartesia_v6.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EMO_DIR="$REPO_ROOT/emotion_layer"
SYNTH="$EMO_DIR/synthesize.py"
API="$EMO_DIR/api.py"
UI="$EMO_DIR/ui.py"

NEW_CKPT="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_v6/model_last.pt"
NEW_VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"

BACKUP_DIR="$REPO_ROOT/deploy_backups/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
API_PORT=8000
UI_PORT=7861

PYTHON="$REPO_ROOT/f5tts/bin/python"

echo "============================================================"
echo " F5-TTS Marathi Deployment - Cartesia_Rasa_Combined_v6"
echo "============================================================"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo ""
echo "[PRE-FLIGHT] Checking files..."
for f in "$NEW_CKPT" "$NEW_VOCAB" "$SYNTH" "$API" "$UI"; do
    if [[ ! -e "$f" ]]; then
        echo "[ERROR] Required file not found: $f"
        exit 1
    fi
done
NEW_VOCAB_COUNT=$(wc -l < "$NEW_VOCAB")
echo "  New checkpoint exists   : $NEW_CKPT"
echo "  New vocabulary exists   : $NEW_VOCAB ($NEW_VOCAB_COUNT entries)"

mkdir -p "$BACKUP_DIR"
mkdir -p "$LOG_DIR"
echo "  Backup dir              : $BACKUP_DIR"

# ---------------------------------------------------------------------------
# 1. Backup current production files
# ---------------------------------------------------------------------------
echo ""
echo "[1] Backing up production files..."
cp -p "$SYNTH" "$BACKUP_DIR/synthesize.py"
cp -p "$API"   "$BACKUP_DIR/api.py"
echo "  Backed up to $BACKUP_DIR"

# ---------------------------------------------------------------------------
# 2 & 3. Force synthesize.py / api.py to v6 paths
# ---------------------------------------------------------------------------
echo ""
echo "[2] Updating checkpoint & vocabulary in synthesize.py..."

# Force checkpoint path to v6 regardless of what is hardcoded
perl -pi -e 's{/root/f5-tts-marathi/f5tts/lib/python3\.12/ckpts/(Rasa_Marathi_Emotion_Female|Cartesia_Rasa_Combined_3h|Cartesia_Rasa_Combined_v2|Cartesia_Rasa_Combined_v3|Cartesia_Rasa_Combined_v4|Cartesia_Rasa_Combined_v5)/model_last\.pt}{/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_v6/model_last.pt}g' "$SYNTH"

# Force the extended vocab in synthesize.py (138 entries).
perl -pi -e 's{/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab(\.txt|_extended\.txt)}{/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt}g' "$SYNTH"

echo "[3] Updating model identifier in api.py..."
perl -pi -e 's{(Rasa_Marathi_Emotion_Female|Cartesia_Rasa_Combined_3h|Cartesia_Rasa_Combined_v2|Cartesia_Rasa_Combined_v3|Cartesia_Rasa_Combined_v4|Cartesia_Rasa_Combined_v5)/model_last\.pt}{Cartesia_Rasa_Combined_v6/model_last.pt}g' "$API"

# Verify edits
echo ""
echo "[VERIFY] synthesize.py model/vocab lines:"
grep -nE "ckpt_path\s*=|vocab_path\s*=|model_path\s*=" "$SYNTH" | head -20 || true
echo ""
echo "[VERIFY] api.py model identifier:"
grep -n "Cartesia_Rasa_Combined_v6\|model_identifier" "$API" | head -10 || true

# ---------------------------------------------------------------------------
# 4. Restart FastAPI on port 8000
# ---------------------------------------------------------------------------
echo ""
echo "[4] Restarting FastAPI on port $API_PORT..."
OLD_API_PID=$(ss -ltnp 2>/dev/null | grep ":$API_PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -z "$OLD_API_PID" ]]; then
    OLD_API_PID=$(pgrep -f "uvicorn emotion_layer.api:app.*$API_PORT" | head -1 || true)
fi
if [[ -n "$OLD_API_PID" ]]; then
    echo "  Stopping existing API process PID $OLD_API_PID..."
    kill "$OLD_API_PID" 2>/dev/null || true
    for i in $(seq 1 20); do
        if ! kill -0 "$OLD_API_PID" 2>/dev/null; then break; fi
        sleep 0.5
    done
    if kill -0 "$OLD_API_PID" 2>/dev/null; then
        kill -9 "$OLD_API_PID" 2>/dev/null || true
        sleep 1
    fi
else
    echo "  No existing API process found on port $API_PORT."
fi
sleep 2

cd "$REPO_ROOT"
nohup "$PYTHON" -m uvicorn emotion_layer.api:app --host 0.0.0.0 --port $API_PORT \
    > "$LOG_DIR/api_deploy_v6.log" 2>&1 &
disown
NEW_API_PID=$!
echo "  Started new API process PID $NEW_API_PID (log: $LOG_DIR/api_deploy_v6.log)"

# ---------------------------------------------------------------------------
# 5. Restart Gradio UI on port 7861
# ---------------------------------------------------------------------------
echo ""
echo "[5] Restarting Gradio UI on port $UI_PORT..."
OLD_UI_PID=$(ss -ltnp 2>/dev/null | grep ":$UI_PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -z "$OLD_UI_PID" ]]; then
    OLD_UI_PID=$(pgrep -f "emotion_layer/ui.py.*$UI_PORT" | head -1 || true)
fi
if [[ -n "$OLD_UI_PID" ]]; then
    echo "  Stopping existing UI process PID $OLD_UI_PID..."
    kill "$OLD_UI_PID" 2>/dev/null || true
    for i in $(seq 1 20); do
        if ! kill -0 "$OLD_UI_PID" 2>/dev/null; then break; fi
        sleep 0.5
    done
    if kill -0 "$OLD_UI_PID" 2>/dev/null; then
        kill -9 "$OLD_UI_PID" 2>/dev/null || true
        sleep 1
    fi
else
    echo "  No existing UI process found on port $UI_PORT."
fi
sleep 2

cd "$REPO_ROOT"
nohup "$PYTHON" emotion_layer/ui.py --port $UI_PORT \
    > "$LOG_DIR/ui_deploy_v6.log" 2>&1 &
disown
NEW_UI_PID=$!
echo "  Started new UI process PID $NEW_UI_PID (log: $LOG_DIR/ui_deploy_v6.log)"

# ---------------------------------------------------------------------------
# 6. Wait for health + validate
# ---------------------------------------------------------------------------
echo ""
echo "[6] Waiting for services (model ~5GB load may take a while)..."
API_HEALTHY=""
for i in $(seq 1 120); do
    if curl -s -m 3 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
        API_HEALTHY="yes"; break
    fi
    sleep 5
done
if [[ "$API_HEALTHY" == "yes" ]]; then
    echo "  Port $API_PORT /health is responding."
else
    echo "[ERROR] Port $API_PORT /health did NOT respond within the timeout."
fi

echo ""
echo "[LOG] API startup (model/vocab/errors):"
grep -E "Loading|checkpoint|vocab|Error|Traceback|error" "$LOG_DIR/api_deploy_v6.log" | tail -30 || true

echo ""
echo "[7] Querying /model-info..."
curl -s -m 5 "http://127.0.0.1:$API_PORT/model-info" || echo "(model-info not responding yet)"
echo ""

# ---------------------------------------------------------------------------
# 8. Synthesis validation
# ---------------------------------------------------------------------------
echo ""
echo "[8] Running synthesis validation..."
TEST_CASES=(
    "NEUTRAL|मी आज नवीन कॅमेरा वापरला."
    "NEUTRAL|तिने काळी बॅग घेतली."
    "NEUTRAL|घरात गॅस संपला आहे."
    "NEUTRAL|फोनमध्ये नवीन ॲप इन्स्टॉल केलं."
    "NEUTRAL|मला सकाळी कॉफी आवडते."
    "NEUTRAL|आज ऑफिसची महत्त्वाची मीटिंग आहे."
    "NEUTRAL|त्याने अंतराळात रॉकेट पाहिलं."
    "NEUTRAL|मी क्रिकेटचा बॉल विकत घेतला."
    "NEUTRAL|हे डॉक्टर रुग्णालयात तपासणी करतात."
    "NEUTRAL|काँग्रेसची सभा शहरात भरली आहे."
    "NEUTRAL|तिने कँ आणि कॉं असा उच्चार केला."
    "NEUTRAL|वा! किती सुंदर ड्रॉइंग केलीये. हा स्केच मला ब्यूटिफुल वाटला!"
    "NEUTRAL|thank you"
    "SAD|आज खूप दिवस कठीण गेला."
    "HAPPY|आज खूप आनंदी दिवस आहे."
    "ANGRY|हे काय करतोयस तू!"
    "FEAR|मला खूप भीती वाटत आहे."
    "DISGUST|हे खरंच घृणास्पद आहे."
)
mkdir -p "$REPO_ROOT/validation_output"
i=0
for tc in "${TEST_CASES[@]}"; do
    i=$((i+1))
    emotion="${tc%%|*}"; text="${tc#*|}"
    out_file="$REPO_ROOT/validation_output/test_v6_${i}.wav"
    resp=$(curl -s -m 300 -X POST "http://127.0.0.1:$API_PORT/synthesize" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"gen_text":sys.argv[1],"emotion_tag":sys.argv[2]}))' "$text" "$emotion")" \
        || echo "__CURL_FAIL__")
    if [[ "$resp" == "__CURL_FAIL__" || -z "$resp" ]]; then
        echo "  [Test $i] $emotion -> FAILED (no response)"
        continue
    fi
    base64str=$(echo "$resp" | python3 -c '
import sys,json
try:
    d=json.load(sys.stdin); print(d.get("audio_base64",""))
except Exception:
    print("__PARSE_ERR__")
' 2>/dev/null)
    if [[ -z "$base64str" || "$base64str" == "__PARSE_ERR__" ]]; then
        echo "  [Test $i] $emotion -> FAILED (parse error)"
        echo "    resp: $(echo "$resp" | head -c 300)"
        echo "    log:  $(tail -3 "$LOG_DIR/api_deploy_v6.log")"
        continue
    fi
    echo "$base64str" | base64 -d > "$out_file" 2>/dev/null
    if [[ -s "$out_file" ]]; then
        echo "  [Test $i] $emotion -> OK $out_file ($(stat -c %s "$out_file") bytes)"
    else
        echo "  [Test $i] $emotion -> FAILED (no audio written)"
        echo "    log:  $(tail -3 "$LOG_DIR/api_deploy_v6.log")"
    fi
done

# ---------------------------------------------------------------------------
# 9. UI check
# ---------------------------------------------------------------------------
echo ""
echo "[9] Checking UI on port $UI_PORT..."
if curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$UI_PORT/" >/dev/null 2>&1; then
    echo "  UI responding (HTTP code above)."
else
    echo "[ERROR] UI on port $UI_PORT not responding."
fi

# ---------------------------------------------------------------------------
# 10. Report + rollback
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " DEPLOYMENT REPORT"
echo "============================================================"
echo "1.  Model path served       : $NEW_CKPT"
echo "2.  Vocabulary path served  : $NEW_VOCAB ($NEW_VOCAB_COUNT entries)"
echo "3.  Port 8000 healthy       : $API_HEALTHY"
echo "4.  Test synthesis          : see [8] output above"
echo "5.  Rollback:"
echo "      cp $BACKUP_DIR/synthesize.py $SYNTH"
echo "      cp $BACKUP_DIR/api.py $API"
echo "      kill \$(ss -ltnp | grep ':$API_PORT ' | grep -oP 'pid=\\K[0-9]+' | head -1); sleep 3"
echo "      cd $REPO_ROOT && nohup $PYTHON -m uvicorn emotion_layer.api:app --host 0.0.0.0 --port $API_PORT > $LOG_DIR/api_rollback_v6.log 2>&1 & disown"
echo "      kill \$(ss -ltnp | grep ':$UI_PORT ' | grep -oP 'pid=\\K[0-9]+' | head -1); sleep 3"
echo "      cd $REPO_ROOT && nohup $PYTHON emotion_layer/ui.py --port $UI_PORT > $LOG_DIR/ui_rollback_v6.log 2>&1 & disown"
echo "============================================================"
echo ""
echo "IMPORTANT: Confirm the API startup log loaded Cartesia_Rasa_Combined_v6"
echo "model_last.pt + vocab_extended.txt before declaring success."
echo "Prior checkpoints are untouched and available for rollback."
