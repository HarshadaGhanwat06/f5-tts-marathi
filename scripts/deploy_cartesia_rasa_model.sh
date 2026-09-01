#!/usr/bin/env bash
#
# deploy_cartesia_rasa_model.sh
#
# Deploy the newly fine-tuned Marathi female F5-TTS model trained on the
# combined 3-hour Rasa + 200 Cartesia/Arushi dataset.
#
# New checkpoint:
#   /root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_3h/model_last.pt
# New vocabulary (extended, Chandrabindu):
#   /root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt
#
# The script:
#   1. Backs up the current production files (synthesize.py, api.py).
#   2. Swaps the model + vocab paths in emotion_layer/synthesize.py.
#   3. Updates the model identifier in emotion_layer/api.py.
#   4. Restarts the FastAPI service on port 8000.
#   5. Restarts the Gradio UI on port 7861.
#   6. Validates health, model loading, and synthesis.
#   7. Prints a rollback procedure.
#
# The old checkpoint (Rasa_Marathi_Emotion_Female/model_last.pt) is NOT
# deleted and remains as the rollback option.
#
# Usage:
#   bash scripts/deploy_cartesia_rasa_model.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EMO_DIR="$REPO_ROOT/emotion_layer"
SYNTH="$EMO_DIR/synthesize.py"
API="$EMO_DIR/api.py"
UI="$EMO_DIR/ui.py"

NEW_CKPT="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_3h/model_last.pt"
NEW_VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"
OLD_CKPT="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female/model_last.pt"
OLD_VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab.txt"

BACKUP_DIR="$REPO_ROOT/deploy_backups/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
API_PORT=8000
UI_PORT=7861

PYTHON="$REPO_ROOT/f5tts/bin/python"

echo "============================================================"
echo " F5-TTS Marathi Deployment - Cartesia_Rasa_Combined_3h"
echo "============================================================"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo ""
echo "[PRE-FLIGHT] Checking files..."

for f in "$NEW_CKPT" "$NEW_VOCAB" "$OLD_CKPT" "$SYNTH" "$API" "$UI"; do
    if [[ ! -e "$f" ]]; then
        echo "[ERROR] Required file not found: $f"
        exit 1
    fi
done

echo "  New checkpoint exists   : $NEW_CKPT"
echo "  New vocabulary exists   : $NEW_VOCAB"
echo "  Old checkpoint exists   : $OLD_CKPT (rollback kept)"
NEW_VOCAB_COUNT=$(wc -l < "$NEW_VOCAB")
echo "  New vocab entries       : $NEW_VOCAB_COUNT"

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
echo "  Backed up synthesize.py and api.py to $BACKUP_DIR"

# ---------------------------------------------------------------------------
# 2 & 3. Swap paths in synthesize.py and api.py
# ---------------------------------------------------------------------------
echo ""
echo "[2] Updating model & vocabulary paths in synthesize.py..."

# Old -> new swap. Use perl for exact literal replacement.
perl -pi -e 's{/root/f5-tts-marathi/f5tts/lib/python3\.12/ckpts/Rasa_Marathi_Emotion_Female/model_last\.pt}{/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_3h/model_last.pt}g' "$SYNTH"
perl -pi -e 's{/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab\.txt}{/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt}g' "$SYNTH"

echo "[3] Updating model identifier in api.py..."
perl -pi -e 's{Rasa_Marathi_Emotion_Female/model_last\.pt}{Cartesia_Rasa_Combined_3h/model_last.pt}g' "$API"

# Verify the changes took effect
echo ""
echo "[VERIFY] Confirming edits in synthesize.py..."
grep -n "ckpt_path =" "$SYNTH"
grep -n "vocab_path =" "$SYNTH"
echo ""
echo "[VERIFY] Confirming edits in api.py..."
grep -n "model_identifier" "$API"

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
        if ! kill -0 "$OLD_API_PID" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done
    # Force kill if needed
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
    > "$LOG_DIR/api_deploy.log" 2>&1 &
disown
NEW_API_PID=$!
echo "  Started new API process PID $NEW_API_PID"
echo "  Log file: $LOG_DIR/api_deploy.log"

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
        if ! kill -0 "$OLD_UI_PID" 2>/dev/null; then
            break
        fi
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
    > "$LOG_DIR/ui_deploy.log" 2>&1 &
disown
NEW_UI_PID=$!
echo "  Started new UI process PID $NEW_UI_PID"
echo "  Log file: $LOG_DIR/ui_deploy.log"

# ---------------------------------------------------------------------------
# 6 & 7. Wait for startup and validate
# ---------------------------------------------------------------------------
echo ""
echo "[6] Waiting for services to come up (model is ~5GB, may take a while)..."

# Wait for API health
API_HEALTHY=""
for i in $(seq 1 120); do
    if curl -s -m 3 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
        API_HEALTHY="yes"
        break
    fi
    sleep 5
done

if [[ "$API_HEALTHY" == "yes" ]]; then
    echo "  Port $API_PORT /health is responding."
else
    echo "[WARN] Port $API_PORT /health did not respond within the timeout."
    echo "       Check $LOG_DIR/api_deploy.log below."
fi

# Check API startup log for the actual loaded paths
echo ""
echo "[LOG] API startup log (model loading evidence):"
grep -E "Loading|checkpoint|vocab|Error|Traceback|error" "$LOG_DIR/api_deploy.log" | tail -30 || true

# ---------------------------------------------------------------------------
# Model load validation via /model-info
# ---------------------------------------------------------------------------
echo ""
echo "[7] Querying /model-info..."
curl -s -m 5 "http://127.0.0.1:$API_PORT/model-info" || echo "(model-info not responding yet)"
echo ""

# ---------------------------------------------------------------------------
# 8. Synthesis validation
# ---------------------------------------------------------------------------
echo ""
echo "[8] Running synthesis validation (this may take time for model load)..."
TEST_CASES=(
    "NEUTRAL|वा! किती सुंदर ड्रॉइंग केलीये. हा स्केच मला खूप ब्यूटिफुल वाटला!"
    "NEUTRAL|आज ऑफिसमध्ये माझी एक महत्त्वाची मीटिंग आहे."
    "NEUTRAL|हा प्रोजेक्ट खरंच खूप इंटरेस्टिंग वाटतोय."
    "NEUTRAL|आज तुमचा दिवस कसा होता?"
    "SAD|आज खूप दिवस कठीण गेला."
    "HAPPY|आज खूप आनंदी दिवस आहे."
)

mkdir -p "$REPO_ROOT/validation_output"
i=0
for tc in "${TEST_CASES[@]}"; do
    i=$((i+1))
    emotion="${tc%%|*}"
    text="${tc#*|}"
    out_file="$REPO_ROOT/validation_output/test_${i}.wav"
    echo ""
    echo "  [Test $i] emotion=$emotion"
    echo "    text: $text"

    resp=$(curl -s -m 300 -X POST "http://127.0.0.1:$API_PORT/synthesize" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"gen_text":sys.argv[1],"emotion_tag":sys.argv[2]}))' "$text" "$emotion")" \
        || echo "__CURL_FAIL__")

    if [[ "$resp" == "__CURL_FAIL__" || -z "$resp" ]]; then
        echo "    RESULT: FAILED (no response from API)"
        echo "    --- recent API log ---"
        tail -20 "$LOG_DIR/api_deploy.log"
        continue
    fi

    # Decode base64 audio from response
    base64str=$(echo "$resp" | python3 -c '
import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get("audio_base64",""))
except Exception as e:
    print("", file=sys.stderr)
    print("__PARSE_ERR__")
' 2>/dev/null)

    if [[ "$base64str" == "__PARSE_ERR__" || -z "$base64str" ]]; then
        echo "    RESULT: FAILED (could not parse audio response)"
        echo "    --- API response ---"
        echo "$resp" | head -c 500
        echo ""
        echo "    --- recent API log ---"
        tail -20 "$LOG_DIR/api_deploy.log"
        continue
    fi

    echo "$base64str" | base64 -d > "$out_file" 2>/dev/null
    if [[ -f "$out_file" && -s "$out_file" ]]; then
        size=$(stat -c %s "$out_file")
        echo "    RESULT: OK -> $out_file ($size bytes)"
    else
        echo "    RESULT: FAILED (audio file not written)"
        echo "    --- recent API log ---"
        tail -20 "$LOG_DIR/api_deploy.log"
    fi
done

# ---------------------------------------------------------------------------
# 9. UI startup check
# ---------------------------------------------------------------------------
echo ""
echo "[9] Checking UI on port $UI_PORT..."
if curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$UI_PORT/" >/dev/null 2>&1; then
    echo "  UI on port $UI_PORT is responding."
else
    echo "  UI on port $UI_PORT not yet responding (may still be loading)."
fi
echo ""
echo "[LOG] UI startup log:"
grep -E "Running|Error|Traceback|error|localhost|http" "$LOG_DIR/ui_deploy.log" | tail -20 || true

# ---------------------------------------------------------------------------
# 10. Final report + rollback
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " DEPLOYMENT REPORT"
echo "============================================================"
echo "1.  Model path served       : $NEW_CKPT"
echo "2.  Vocabulary path served  : $NEW_VOCAB ($NEW_VOCAB_COUNT entries)"
echo "3.  Process/service         : uvicorn (port 8000), ui.py gradio (port 7861)"
echo "4.  Port 8000 healthy       : $API_HEALTHY"
echo "5.  Port 7861 healthy       : see [9] output above"
echo "6.  Test synthesis          : see [8] output above"
echo "7.  Errors/warnings         : see logs below"
echo "8.  Rollback procedure      :"
echo "      cp $BACKUP_DIR/synthesize.py $SYNTH"
echo "      cp $BACKUP_DIR/api.py $API"
echo "      # restart API:"
echo "      kill \$(ss -ltnp | grep ':$API_PORT ' | grep -oP 'pid=\\K[0-9]+' | head -1)"
echo "      sleep 3"
echo "      cd $REPO_ROOT && nohup $PYTHON -m uvicorn emotion_layer.api:app --host 0.0.0.0 --port $API_PORT > $LOG_DIR/api_rollback.log 2>&1 & disown"
echo "      # restart UI:"
echo "      kill \$(ss -ltnp | grep ':$UI_PORT ' | grep -oP 'pid=\\K[0-9]+' | head -1)"
echo "      sleep 3"
echo "      cd $REPO_ROOT && nohup $PYTHON emotion_layer/ui.py --port $UI_PORT > $LOG_DIR/ui_rollback.log 2>&1 & disown"
echo "============================================================"
echo ""
echo "[INFO] Backup of production files: $BACKUP_DIR"
echo "[INFO] Deployment logs: $LOG_DIR/api_deploy.log and $LOG_DIR/ui_deploy.log"
echo "[INFO] Validation audio: $REPO_ROOT/validation_output/"

echo ""
echo "IMPORTANT: Confirm the API startup log actually loaded the NEW "
echo "checkpoint and vocab_extended.txt before declaring success."
echo "The old checkpoint is untouched and available for rollback."
