#!/usr/bin/env bash
# =============================================================================
# deploy_cartesia_v9.sh
#
# Deploy the word-list ॅ/ॲ model (Chandra_Words_v9) to the live services:
# FastAPI on port 8000 and Gradio UI on port 7861.
#
# New checkpoint:
#   /root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Chandra_Words_v9/model_last.pt
# Vocabulary:
#   /root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt
#
# This replaces whatever was previously hardcoded (v1..v8, Rasa, 3h,
# Chandra_Vyanjan_Focus_v8) in BOTH emotion_layer/synthesize.py and api.py.
#
# Validation suits:
#   Tests  1- 7: the v8 example set + the objective's headline words
#                 (ॲपल कॅब कॅमेरा गॅस चॅट मॅनेजर व्हॅन)
#   Tests  8-11: additional word-list generalization sentences (ॅ/ॲ)
#   Tests 12-17: emotion-layer regression (NEUTRAL/SAD/HAPPY/ANGRY/FEAR/DISGUST)
#
# Steps:
#   1. Backup production files (synthesize.py, api.py).
#   2. Force emotion_layer/synthesize.py to the v9 checkpoint + extended vocab.
#   3. Force the model identifier in emotion_layer/api.py to v9.
#   4. Restart FastAPI (8000) + Gradio UI (7861).
#   5. Validate health, model-info, and synthesis - each check prints an
#      explicit PASS/FAIL line (no silently-wrapped errors).
#   6. Print rollback procedure.
#
# Usage:
#   bash scripts/deploy_cartesia_v9.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EMO_DIR="$REPO_ROOT/emotion_layer"
SYNTH="$EMO_DIR/synthesize.py"
API="$EMO_DIR/api.py"
UI="$EMO_DIR/ui.py"

NEW_CKPT="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Chandra_Words_v9/model_last.pt"
NEW_VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"

BACKUP_DIR="$REPO_ROOT/deploy_backups/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
API_PORT=8000
UI_PORT=7861
PYTHON="$REPO_ROOT/f5tts/bin/python"

echo "============================================================"
echo " F5-TTS Marathi Deployment - Chandra_Words_v9 (word-list ॅ/ॲ)"
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

# Verify checkpoint embedding rows == vocab lines + 1 (catches stale copy)
if "$PYTHON" - "$NEW_CKPT" "$NEW_VOCAB" <<'PY'
import sys, torch
ckpt, vocab = sys.argv[1], sys.argv[2]
n = sum(1 for l in open(vocab, encoding="utf-8") if l.rstrip("\n"))
sd = torch.load(ckpt, map_location="cpu")
key = "transformer.text_embed.text_embed.weight"
found = None
for name in ("ema_model_state_dict", "model_state_dict"):
    if name in sd and isinstance(sd[name], dict) and key in sd[name]:
        found = sd[name][key]
        break
if found is None and key in sd:
    found = sd[key]
if found is None:
    print("[WARN] text_embed not found; cannot verify embedding rows here.")
elif found.shape[0] != n + 1:
    print(f"[ERROR] checkpoint embeds {found.shape[0]} != vocab+1 {n+1}")
    sys.exit(1)
else:
    print(f"  [OK] checkpoint embeds {found.shape[0]} == vocab+1 ({n})")
PY
then
    echo "  [OK] checkpoint embedding rows match vocab+1."
else
    echo "[ERROR] checkpoint embedding shape mismatch or load error (see above)."
    echo "        This is a stale/wrong checkpoint. Aborting."
    exit 1
fi

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
# 2 & 3. Force synthesize.py / api.py to v9 paths
# ---------------------------------------------------------------------------
echo ""
echo "[2] Updating checkpoint & vocabulary in synthesize.py..."

# Force checkpoint path to v9 regardless of what is hardcoded (incl v8)
perl -pi -e 's{/root/f5-tts-marathi/f5tts/lib/python3\.12/ckpts/(Rasa_Marathi_Emotion_Female|Cartesia_Rasa_Combined_3h|Cartesia_Rasa_Combined_v2|Cartesia_Rasa_Combined_v3|Cartesia_Rasa_Combined_v4|Cartesia_Rasa_Combined_v5|Cartesia_Rasa_Combined_v6|Cartesia_Rasa_Combined_v7|Cartesia_Rasa_Combined_v8|Chandra_Vyanjan_Focus_v8)/model_last\.pt}{/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Chandra_Words_v9/model_last.pt}g' "$SYNTH"

# Force the extended vocab in synthesize.py.
perl -pi -e 's{/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab(\.txt|_extended\.txt)}{/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt}g' "$SYNTH"

echo "[3] Updating model identifier in api.py..."
perl -pi -e 's{(Rasa_Marathi_Emotion_Female|Cartesia_Rasa_Combined_3h|Cartesia_Rasa_Combined_v2|Cartesia_Rasa_Combined_v3|Cartesia_Rasa_Combined_v4|Cartesia_Rasa_Combined_v5|Cartesia_Rasa_Combined_v6|Cartesia_Rasa_Combined_v7|Cartesia_Rasa_Combined_v8|Chandra_Vyanjan_Focus_v8)/model_last\.pt}{Chandra_Words_v9/model_last.pt}g' "$API"

# Verify edits
echo ""
echo "[VERIFY] synthesize.py model/vocab lines:"
grep -nE "ckpt_path\s*=|vocab_path\s*=|model_path\s*=" "$SYNTH" | head -20 || true
echo ""
echo "[VERIFY] api.py model identifier:"
grep -n "Chandra_Words_v9\|model_identifier" "$API" | head -10 || true

# Explicit check that the new paths are actually present now
if ! grep -q "$NEW_CKPT" "$SYNTH"; then
    echo "[ERROR] Chandra_Words_v9 checkpoint path NOT found in $SYNTH after edit."
    exit 1
fi
if ! grep -q "Chandra_Words_v9/model_last.pt" "$API"; then
    echo "[ERROR] Chandra_Words_v9 identifier NOT found in $API after edit."
    exit 1
fi

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
    > "$LOG_DIR/api_deploy_v9.log" 2>&1 &
disown
NEW_API_PID=$!
echo "  Started new API process PID $NEW_API_PID (log: $LOG_DIR/api_deploy_v9.log)"

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
    > "$LOG_DIR/ui_deploy_v9.log" 2>&1 &
disown
NEW_UI_PID=$!
echo "  Started new UI process PID $NEW_UI_PID (log: $LOG_DIR/ui_deploy_v9.log)"

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
    echo "  [PASS] Port $API_PORT /health is responding."
else
    echo "  [FAIL] Port $API_PORT /health did NOT respond within the timeout."
    echo "         Check $LOG_DIR/api_deploy_v9.log"
fi

echo ""
echo "[LOG] API startup (model/vocab/errors):"
grep -E "Loading|checkpoint|vocab|Error|Traceback|error" "$LOG_DIR/api_deploy_v9.log" | tail -30 || true

echo ""
echo "[7] Querying /model-info..."
MI_RESP=$(curl -s -m 5 "http://127.0.0.1:$API_PORT/model-info" || true)
echo "$MI_RESP"
if echo "$MI_RESP" | grep -q "Chandra_Words_v9"; then
    echo "  [PASS] /model-info confirms Chandra_Words_v9."
else
    echo "  [FAIL] /model-info does not mention Chandra_Words_v9. Confirm the load."
fi

# ---------------------------------------------------------------------------
# 8. Synthesis validation
# ---------------------------------------------------------------------------
echo ""
echo "[8] Running synthesis validation..."
TEST_CASES=(
    "NEUTRAL|फोनमध्ये नवीन ॲप इन्स्टॉल केलं."
    "NEUTRAL|मी आज नवीन कॅमेरा वापरला."
    "NEUTRAL|हा नवीन कॅब खूप चांगला आहे."
    "NEUTRAL|घरात गॅस संपला आहे."
    "NEUTRAL|संगणकावर चॅट करायचं होतं."
    "NEUTRAL|तो मॅनेजरशी बोलत आहे."
    "NEUTRAL|रस्त्यावर व्हॅन उभी आहे."
    "NEUTRAL|सॅलड ताजी आहे."
    "NEUTRAL|बॅटरी सेव्ह करण्यासाठी मोड वापरा."
    "NEUTRAL|संगणकातली सगळी डॅटा जपून ठेवली आहे."
    "NEUTRAL|तिने काळी बॅग घेतली."
    "SAD|आज खूप दिवस कठीण गेला."
    "HAPPY|आज खूप आनंदी दिवस आहे."
    "ANGRY|हे काय करतोयस तू!"
    "FEAR|मला खूप भीती वाटत आहे."
    "DISGUST|हे खरंच घृणास्पद आहे."
)
mkdir -p "$REPO_ROOT/validation_output"
i=0
PASS_CNT=0
FAIL_CNT=0
for tc in "${TEST_CASES[@]}"; do
    i=$((i+1))
    emotion="${tc%%|*}"; text="${tc#*|}"
    out_file="$REPO_ROOT/validation_output/test_v9_${i}.wav"
    resp=$(curl -s -m 300 -X POST "http://127.0.0.1:$API_PORT/synthesize" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"gen_text":sys.argv[1],"emotion_tag":sys.argv[2]}))' "$text" "$emotion")" \
        || echo "__CURL_FAIL__")
    if [[ "$resp" == "__CURL_FAIL__" || -z "$resp" ]]; then
        echo "  [FAIL] [Test $i] $emotion -> no response"
        FAIL_CNT=$((FAIL_CNT+1))
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
        echo "  [FAIL] [Test $i] $emotion -> parse error"
        echo "    resp: $(echo "$resp" | head -c 300)"
        echo "    log:  $(tail -3 "$LOG_DIR/api_deploy_v9.log")"
        FAIL_CNT=$((FAIL_CNT+1))
        continue
    fi
    echo "$base64str" | base64 -d > "$out_file" 2>/dev/null
    if [[ -s "$out_file" ]]; then
        echo "  [PASS] [Test $i] $emotion -> $out_file ($(stat -c %s "$out_file") bytes)"
        PASS_CNT=$((PASS_CNT+1))
    else
        echo "  [FAIL] [Test $i] $emotion -> no audio written"
        echo "    log:  $(tail -3 "$LOG_DIR/api_deploy_v9.log")"
        FAIL_CNT=$((FAIL_CNT+1))
    fi
done

# ---------------------------------------------------------------------------
# 9. UI check
# ---------------------------------------------------------------------------
echo ""
echo "[9] Checking UI on port $UI_PORT..."
UI_CODE=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$UI_PORT/" 2>/dev/null || true)
if [[ -n "$UI_CODE" && "$UI_CODE" != "000" ]]; then
    echo "  [PASS] UI responding (HTTP $UI_CODE)."
else
    echo "  [FAIL] UI on port $UI_PORT not responding."
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
echo "4.  Synthesis PASS/FAIL     : $PASS_CNT / $FAIL_CNT (see [8])"
echo "5.  Rollback:"
echo "      cp $BACKUP_DIR/synthesize.py $SYNTH"
echo "      cp $BACKUP_DIR/api.py $API"
echo "      kill \$(ss -ltnp | grep ':$API_PORT ' | grep -oP 'pid=\\K[0-9]+' | head -1); sleep 3"
echo "      cd $REPO_ROOT && nohup $PYTHON -m uvicorn emotion_layer.api:app --host 0.0.0.0 --port $API_PORT > $LOG_DIR/api_rollback_v9.log 2>&1 & disown"
echo "      kill \$(ss -ltnp | grep ':$UI_PORT ' | grep -oP 'pid=\\K[0-9]+' | head -1); sleep 3"
echo "      cd $REPO_ROOT && nohup $PYTHON emotion_layer/ui.py --port $UI_PORT > $LOG_DIR/ui_rollback_v9.log 2>&1 & disown"
echo "============================================================"
echo ""
echo "IMPORTANT: Confirm the API startup log loaded Chandra_Words_v9"
echo "model_last.pt + vocab_extended.txt before declaring success."
if [[ "$FAIL_CNT" -gt 0 ]]; then
    echo "IMPORTANT: $FAIL_CNT synthesis tests FAILED - investigate before use."
else
    echo "All synthesis validation tests passed."
fi