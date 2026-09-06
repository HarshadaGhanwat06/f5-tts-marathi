#!/usr/bin/env bash
# =============================================================================
# validate_v8.sh
#
# STEP 5 — VALIDATION for Chandra_Vyanjan_Focus_v8 (ॅ/ॲ fix).
#
# PART A - training-loss trend from logs/training_v8.log:
#   prints per-epoch average loss; FLAGS if loss is RISING instead of falling
#   (the failure mode seen in the earlier 53-char vocab expansion).
#
# PART B - synthesis checks, requires the v8 model to be LIVE (run
#   deploy_cartesia_v8.sh first):
#    B.1  the 4 original example words  (मॅन कॅन लॅन व्हॅन)
#    B.2  5 ADDITIONAL ॅ/ॲ words NOT among the examples (generalization)
#    B.3  re-run the emotion-layer regression set (all original cases) to
#         confirm no catastrophic forgetting - this dataset is narrow, so
#         regression is a real risk.
#
# Output: WAVs in validation_output/test_v8_step5_*.wav
#
# Usage (server):
#   bash scripts/validate_v8.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG="$REPO_ROOT/logs/training_v8.log"
API_PORT=8000
OUT_DIR="$REPO_ROOT/validation_output"

echo "============================================================"
echo " STEP 5 - v8 Validation (ॅ/ॲ fix)"
echo "============================================================"

# ---------------------------------------------------------------------------
# PART A - loss trend
# ---------------------------------------------------------------------------
echo ""
echo "--- PART A: training-loss trend (from $LOG) ---"
if [[ ! -f "$LOG" ]]; then
    echo "[ERROR] Training log not found: $LOG"
    echo "        Training must complete before validation."
    exit 1
fi

grep -nE "Epoch [0-9]+/[0-9]+:|loss=" "$LOG" | tail -40 || true

echo ""
echo "[A] Per-epoch avg loss (approx, from final step of each epoch):"

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
# epoch-average over its update points
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
# PART B - synthesis (v8 must be live)
# ---------------------------------------------------------------------------
echo ""
echo "--- PART B: synthesis checks (requires v8 deployed on port $API_PORT) ---"
health=$(curl -s -m 5 "http://127.0.0.1:$API_PORT/model-info" || true)
if [[ -z "$health" ]]; then
    echo "[ERROR] API not reachable. Run bash scripts/deploy_cartesia_v8.sh first."
    exit 1
fi
echo "  /model-info: $health"
if ! echo "$health" | grep -q "Chandra_Vyanjan_Focus_v8"; then
    echo "[WARN] /model-info does not mention Chandra_Vyanjan_Focus_v8."
    echo "       Confirm you actually deployed v8 before trusting these tests."
fi

mkdir -p "$OUT_DIR"
B_TESTS=(
    "NEUTRAL|मी नवीन मॅन भेटलो."
    "NEUTRAL|हा कॅन खूप मोठा आहे."
    "NEUTRAL|लॅन कनेक्शन चालू आहे."
    "NEUTRAL|रस्त्यावर व्हॅन उभी आहे."
    "NEUTRAL|फोनमध्ये नवीन ॲप इन्स्टॉल केलं."
    "NEUTRAL|मी आज नवीन कॅमेरा वापरला."
    "NEUTRAL|तिने काळी बॅग घेतली."
    "NEUTRAL|घरात गॅस संपला आहे."
    "NEUTRAL|टेबलावर लॅम्प ठेवला."
    "SAD|आज खूप दिवस कठीण गेला."
    "HAPPY|आज खूप आनंदी दिवस आहे."
    "ANGRY|हे काय करतोयस तू!"
    "FEAR|मला खूप भीती वाटत आहे."
    "DISGUST|हे खरंच घृणास्पद आहे."
)
i=0
for tc in "${B_TESTS[@]}"; do
    i=$((i+1))
    emotion="${tc%%|*}"; text="${tc#*|}"
    out_file="$OUT_DIR/test_v8_step5_${i}.wav"
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
        echo "  [Test $i - $emotion] -> FAILED (no/parse-error audio)"
        echo "    resp: $(echo "$resp" | head -c 200)"
        continue
    fi
    echo "$base64str" | base64 -d > "$out_file" 2>/dev/null
    if [[ -s "$out_file" ]]; then
        tag="generalization"
        [[ "$i" -le 4 ]] && tag="ORIGINAL EXAMPLE"
        [[ "$i" -ge 10 ]] && tag="emotion regression"
        echo "  [Test $i - $emotion] $tag -> OK $out_file"
    else
        echo "  [Test $i - $emotion] -> FAILED (no audio written)"
    fi
done

echo ""
echo "============================================================"
echo " VALIDATION COMPLETE"
echo "============================================================"
echo "1. Listen to validation_output/test_v8_step5_*.wav"
echo "   Tests 1-4  : must sound like correct ॅ (mod/curr) pronunciation."
echo "   Tests 5-9  : generalization on NEW ॅ/ॲ words (kamerA/bAg/gAs/...)."
echo "   Tests 10+  : emotion regression (no forgetting)."
echo "2. If ॅ/ॲ sound right AND emotion still works -> v8 is final."
echo "3. If anything regressed -> rollback: deploy_cartesia_v8.sh prints the steps."
echo "============================================================"