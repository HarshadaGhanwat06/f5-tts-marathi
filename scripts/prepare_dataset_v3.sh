#!/usr/bin/env bash
# =============================================================================
# prepare_dataset_v3.sh
#
# Build the PREPROCESSED dataset (raw.arrow + duration.json) that finetune_cli's
# load_dataset() expects for --dataset_name Cartesia_Rasa_Combined_v3:
#
#   f5tts/data/Cartesia_Rasa_Combined_v3_custom/
#
# The loader resolves rel_data_path = ../data/{dataset_name}_{tokenizer} and
# then reads {rel_data_path}/raw.arrow (+ duration.json). This script:
#   1. Renders an F5-TTS metadata CSV (audio_file|text) with ABSOLUTE wav paths
#      from cartesia_rasa_3h_combined/metadata.csv (the combined 2663 set).
#   2. Runs prepare_csv_wavs.py (fine-tune mode) into the expected out_dir.
#   3. Sanity reports row count + sample rate.
#
# Run on server:
#   bash scripts/prepare_dataset_v3.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
PREPARE="/root/f5-tts-marathi/f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py"

SRC_META="/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined/metadata.csv"
SRC_WAVS="/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined"
OUT_DIR="/root/f5-tts-marathi/f5tts/data/Cartesia_Rasa_Combined_v3_custom"
TMP_CSV="/tmp/cartesia_v3_f5tts_meta.csv"

echo "============================================================"
echo " Prepare preprocessed dataset for Cartesia_Rasa_Combined_v3"
echo "============================================================"
echo "Source metadata : $SRC_META"
echo "WAV root        : $SRC_WAVS"
echo "Output dir      : $OUT_DIR"

# ---------------------------------------------------------------------------
# 1. Render audio_file|text metadata (audio_file ABSOLUTE)
# ---------------------------------------------------------------------------
echo ""
echo "[1] Rendering F5-TTS metadata (audio_file|text, absolute paths)..."
"$PYTHON" - "$SRC_META" "$SRC_WAVS" "$TMP_CSV" <<'PY'
import sys, os, csv
src, wroot, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = 0
with open(src, encoding="utf-8", newline="") as f, \
     open(out, "w", encoding="utf-8") as g:
    reader = csv.reader(f)
    header = next(reader, None)
    # combined metadata.csv is COMMA-delimited: id,text,audio_file,duration_seconds,source
    g.write("audio_file|text\n")
    for row in reader:
        if not row or len(row) < 3:
            continue
        # text = col index 1, audio_file = col index 2 (relative "wavs/xxx.wav")
        text = (row[1] or "").strip()
        relaudio = (row[2] or "").strip()
        if not text or not relaudio:
            continue
        abs_audio = os.path.join(wroot, relaudio)
        # pipe in text would break F5-TTS | delim; escape if present
        if "|" in text:
            print("[WARN] text contains |, skipping:", text[:40], file=sys.stderr)
            continue
        g.write(f"{abs_audio}|{text}\n")
        n += 1
print(f"Wrote {n} rows to {out}")
PY

echo ""
echo "[2] Combined metadata header:"
head -1 "$SRC_META"
echo ""
echo "    First 3 rendered rows (absolute_path|text):"
head -3 "$TMP_CSV"

# ---------------------------------------------------------------------------
# 2. Run prepare_csv_wavs
# ---------------------------------------------------------------------------
echo ""
echo "[3] Running prepare_csv_wavs.py (fine-tune mode)..."
"$PYTHON" "$PREPARE" "$TMP_CSV" "$OUT_DIR" --workers 32

echo ""
echo "[4] Output dir contents:"
ls -la "$OUT_DIR"

echo ""
echo "[5] Row count check:"
"$PYTHON" - "$OUT_DIR" <<'PY'
import sys, json, os
d = sys.argv[1]
j = os.path.join(d, "duration.json")
if os.path.exists(j):
    x = json.load(open(j, encoding="utf-8"))
    durs = x.get("duration", {})
    print("duration.json entries:", len(durs))
    if durs:
        vs = list(durs.values())
        print("total duration (s):", round(sum(vs), 1), "| avg:", round(sum(vs)/len(vs), 2))
else:
    print("duration.json NOT FOUND")
print("raw.arrow exists:", os.path.exists(os.path.join(d, "raw.arrow")))
PY

echo ""
echo "[INFO] Done. Now run: bash scripts/run_train_4epoch_v3.sh"
