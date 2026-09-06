#!/usr/bin/env bash
# =============================================================================
# validate_v9.sh
#
# STEP 5 - VALIDATION for Chandra_Words_v9 (word-list ॅ/ॲ fix).
#
# PART A - training-loss trend from logs/training_v9.log:
#   prints per-epoch average loss; FLAGS if loss is RISING instead of falling
#   (the failure mode seen in the earlier 53-char vocab expansion).
#
# PART B - synthesis checks, requires the v9 model to be LIVE (run
#   deploy_cartesia_v9.sh first):
#    B.1  headline words from the objective (ॲपल कॅब कॅमेरा गॅस चॅट
#         मॅनेजर व्हॅन) - the ॅ/ॲ sounds targeted by v9
#    B.2  word-list generalization + consonant-syllable carrier check
#    B.3  emotion-layer regression (NEUTRAL/SAD/HAPPY/ANGRY/FEAR/DISGUST) to
#         confirm no catastrophic forgetting - v9 fine-tuned on a narrow set.
#
# Every synthesis check prints an explicit [PASS]/[FAIL] line.
# Output: WAVs in validation_output/test_v9_step5_*.wav
#
# Usage (server):
#   bash scripts/validate_v9.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG="$REPO_ROOT/logs/training_v9.log"
API_PORT=8000
OUT_DIR="$REPO_ROOT/validation_output"

echo "============================================================"
echo " STEP 5 - v9 Validation (word-list ॅ/ॲ fix)"
echo "============================================================"

# ---------------------------------------------------------------------------
# PART A - loss trend
# ---------------------------------------------------------------------------
echo ""
echo "--- PART A: training-loss trend (from $LOG) ---"
if [[ ! -f "$LOG" ]]; then
    echo "[FAIL] Training log not found: $LOG"
    echo "       Training must complete before validation."
    exit 1
fi

grep -nE "Epoch [0-9]+/[0-9]+:|loss=" "$LOG" | tail -40 || true

echo ""
echo "[A] Per-epoch avg loss (approx, from logged update points):"

"$REPO_ROOT/f5tts/bin/python3" - "$LOG" <<'PY'
import sys, re
log = sys.argv[1]
lines = []
with open(log, encoding="utf-8", errors="replace") as f:
    for ln in f:
        m = re.search(r"Epoch (\d+)/(\d+):.*?loss=([0-9.]+).*?update=(\d+)", ln)
        if m:
            lines.append((int(m.group(1)), int(m.group(4)), float(m.group(3))))
if not lines:
    print("  (no per-epoch loss lines found yet; training may still be running)")
    sys.exit(0)
from collections import defaultdict
agg = defaultdict(list)
for ep, upd, loss in lines:
    agg[ep].append(loss)
print(f"  epochs seen: {sorted(agg)}")
means = []
for ep in sorted(agg):
    m = sum(agg[ep]) / len(agg[ep])
    means.append((ep, m))
    print(f"  Epoch {ep:>3}: avg loss = {m:.4f}  (from {len(agg[ep])} logs)")
if len(means) >= 2:
    last = means[-1][1]; prev = means[-2][1]
    if last > prev * 1.15:
        print("  [FLAG] loss is RISING in the final epoch vs previous.")
        print("  This mimics the under-trained-character instability seen on this")
        print("  project; do NOT deploy as-is without human review.")
    else:
        print("  [OK] final-epoch loss is not rising.")
else:
    print("  (need >=2 epochs to judge the trend)")
PY

# ---------------------------------------------------------------------------
# PART B - synthesis (v9 must be live)
# ---------------------------------------------------------------------------
echo ""
echo "--- PART B: synthesis checks (requires v9 deployed on port $API_PORT) ---"
health=$(curl -s -m 5 "http://127.0.0.1:$API_PORT/model-info" || true)
if [[ -z "$health" ]]; then
    echo "[FAIL] API not reachable. Run bash scripts/deploy_cartesia_v9.sh first."
    exit 1
fi
echo "  /model-info: $health"
if ! echo "$health" | grep -q "Chandra_Words_v9"; then
    echo "[WARN] /model-info does not mention Chandra_Words_v9."
    echo "       Confirm you actually deployed v9 before trusting these tests."
fi

mkdir -p "$OUT_DIR"
B_TESTS=(
    "NEUTRAL|फोनमध्ये नवीन ॲप इन्स्टॉल केलं."
    "NEUTRAL|मला आज ॲपल आवडतं."
    "NEUTRAL|हा नवीन कॅब खूप चांगला आहे."
    "NEUTRAL|मी आज नवीन कॅमेरा वापरला."
    "NEUTRAL|घरात गॅस संपला आहे."
    "NEUTRAL|संगणकावर चॅट करायचं होतं."
    "NEUTRAL|तो मॅनेजरशी बोलत आहे."
    "NEUTRAL|रस्त्यावर व्हॅन उभी आहे."
    "NEUTRAL|सोनूने 'कॅ' असं नीट उच्चारलं."
    "NEUTRAL|आईने 'ट्रॅक्टर' हा शब्द वाचला."
    "SAD|आज खूप दिवस कठीण गेला."
    "HAPPY|आज खूप आनंदी दिवस आहे."
    "ANGRY|हे काय करतोयस तू!"
    "FEAR|मला खूप भीती वाटत आहे."
    "DISGUST|हे खरंच घृणास्पद आहे."
)
i=0
PASS_CNT=0
FAIL_CNT=0
for tc in "${B_TESTS[@]}"; do
    i=$((i+1))
    emotion="${tc%%|*}"; text="${tc#*|}"
    out_file="$OUT_DIR/test_v9_step5_${i}.wav"
    resp=$(curl -s -m 300 -X POST "http://127.0.0.1:$API_PORT/synthesize" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"gen_text":sys.argv[1],"emotion_tag":sys.argv[2]}))' "$text" "$emotion")" \
        || echo "__CURL_FAIL__")
    base64str=$(echo "$resp" | python3 -c '
import sys,json
try:
    d=json.load(sys.stdin); print(d.get("audio_base64",""))
except Exception:
    print("__PARSE_ERR__")
' 2>/dev/null)
    if [[ -z "$base64str" || "$base64str" == "__PARSE_ERR__" ]]; then
        echo "  [FAIL] [Test $i - $emotion] no/parse-error audio"
        echo "    resp: $(echo "$resp" | head -c 200)"
        FAIL_CNT=$((FAIL_CNT+1))
        continue
    fi
    echo "$base64str" | base64 -d > "$out_file" 2>/dev/null
    if [[ -s "$out_file" ]]; then
        tag="generalization"
        [[ "$i" -le 8 ]] && tag="objective/word-list"
        [[ "$i" -ge 11 ]] && tag="emotion regression"
        echo "  [PASS] [Test $i - $emotion] $tag -> $out_file"
        PASS_CNT=$((PASS_CNT+1))
    else
        echo "  [FAIL] [Test $i - $emotion] no audio written"
        FAIL_CNT=$((FAIL_CNT+1))
    fi
done

echo ""
echo "============================================================"
echo " VALIDATION COMPLETE"
echo "============================================================"
echo "Synthesis PASS/FAIL : $PASS_CNT / $FAIL_CNT"
echo "1. Listen to validation_output/test_v9_step5_*.wav"
echo "   Tests 1-8  : must sound like correct ॅ (mod/curr) pronunciation."
echo "   Tests 9-10 : consonant-syllable / consonant-cluster focus."
echo "   Tests 11+  : emotion regression (no forgetting)."
echo "2. If ॅ/ॲ sound right AND emotion still works -> v9 is final."
echo "3. If anything regressed -> rollback: deploy_cartesia_v9.sh prints the steps."
echo "============================================================"
if [[ "$FAIL_CNT" -gt 0 ]]; then
    echo "RESULT: $FAIL_CNT synthesis checks FAILED - investigate before use."
    exit 2
else
    echo "RESULT: all synthesis checks passed."
    exit 0
fi