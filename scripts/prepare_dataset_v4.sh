#!/usr/bin/env bash
# =============================================================================
# prepare_dataset_v4.sh
#
# Build the PREPROCESSED dataset for Cartesia_Rasa_Combined_v4:
#
#   = the complete v3 source  (cartesia_rasa_3h_combined, 2663 rows)
#   + the NEW targeted candra_e dataset (~200 rows, ॅ/ॲ focused)
#
# Output:
#   f5tts/data/Cartesia_Rasa_Combined_v4_custom/   (raw.arrow + duration.json)
#   symlink at f5tts/lib/python3.12/data/Cartesia_Rasa_Combined_v4_custom
#   (the loader resolves {dataset}_{tokenizer} -> that symlink)
#
# Run on server (after create_candra_e_dataset.sh --full + ear check):
#   bash scripts/prepare_dataset_v4.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
PREPARE="/root/f5-tts-marathi/f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py"

COMBINED_META="/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined/metadata.csv"
COMBINED_WAVS="/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined"
CANDRA_META="/root/f5-tts-marathi/cartesia_ws/cartesia_candra_e/metadata.csv"
CANDRA_WAVS="/root/f5-tts-marathi/cartesia_ws/cartesia_candra_e/output"

OUT_DIR="/root/f5-tts-marathi/f5tts/data/Cartesia_Rasa_Combined_v4_custom"
TMP_CSV="/tmp/cartesia_v4_f5tts_meta.csv"

echo "============================================================"
echo " Prepare preprocessed dataset for Cartesia_Rasa_Combined_v4"
echo "============================================================"
echo "Combined source  : $COMBINED_META"
echo "Candra_e source  : $CANDRA_META"
echo "Output dir       : $OUT_DIR"

if [[ ! -f "$CANDRA_META" ]]; then
    echo "[ERROR] candra_e metadata not found: $CANDRA_META"
    echo "        Run bash scripts/create_candra_e_dataset.sh --full first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Render audio_file|text metadata (absolute wav paths, both sources)
# ---------------------------------------------------------------------------
echo ""
echo "[1] Rendering F5-TTS metadata (audio_file|text, absolute paths)..."
"$PYTHON" - "$COMBINED_META" "$COMBINED_WAVS" "$CANDRA_META" "$CANDRA_WAVS" "$TMP_CSV" <<'PY'
import sys, os, csv

m1, w1, m2, w2, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
n = 0
skipped_pipe = 0

def write_rows(meta, wroot, g):
    global n, skipped_pipe
    with open(meta, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # both metas are comma-delimited: id,text,audio_file,duration_seconds[,source...]
        for row in reader:
            if not row or len(row) < 3:
                continue
            text = (row[1] or "").strip()
            relaudio = (row[2] or "").strip()
            if not text or not relaudio:
                continue
            if "|" in text:  # pipe would break the | delimiter
                skipped_pipe += 1
                print("[WARN] text contains |, skipping:", text[:40], file=sys.stderr)
                continue
            g.write(f"{os.path.join(wroot, relaudio)}|{text}\n")
            n += 1

with open(out, "w", encoding="utf-8") as g:
    g.write("audio_file|text\n")
    write_rows(m1, w1, g)
    write_rows(m2, w2, g)

print(f"Wrote {n} rows to {out} (skipped {skipped_pipe} with '|')")
PY

echo ""
echo "[2] Row counts / headers:"
echo "  combined  : $(wc -l < "$COMBINED_META") lines (incl header)"
echo "  candra_e  : $(wc -l < "$CANDRA_META") lines (incl header)"
echo "  rendered  : $(wc -l < "$TMP_CSV") lines (incl header)"
echo ""
echo "    First 3 rendered rows (combined):"
head -3 "$TMP_CSV"
echo "    Last 3 rendered rows (candra_e):"
tail -3 "$TMP_CSV"

# ---------------------------------------------------------------------------
# 3. Run prepare_csv_wavs
# ---------------------------------------------------------------------------
echo ""
echo "[3] Running prepare_csv_wavs.py ..."
"$PYTHON" "$PREPARE" "$TMP_CSV" "$OUT_DIR" --workers 32

echo ""
echo "[4] Output dir contents:"
ls -la "$OUT_DIR"

# ---------------------------------------------------------------------------
# 4b. Symlink into the loader's data root
# ---------------------------------------------------------------------------
echo ""
echo "[4b] Registering symlink for loader at lib/python3.12/data/..."
LINK_ROOT="/root/f5-tts-marathi/f5tts/lib/python3.12/data"
LINK_NAME="Cartesia_Rasa_Combined_v4_custom"
mkdir -p "$LINK_ROOT"
if [[ -L "$LINK_ROOT/$LINK_NAME" || -e "$LINK_ROOT/$LINK_NAME" ]]; then
    rm -rf "$LINK_ROOT/$LINK_NAME"
fi
ln -s "$OUT_DIR" "$LINK_ROOT/$LINK_NAME"
ls -la "$LINK_ROOT/$LINK_NAME"

# ---------------------------------------------------------------------------
# 5. Sanity report
# ---------------------------------------------------------------------------
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
        print("+ candra_e focus   :", "yes")
else:
    print("duration.json NOT FOUND")
print("raw.arrow exists:", os.path.exists(os.path.join(d, "raw.arrow")))
PY

echo ""
echo "[INFO] Done. Now run: bash scripts/prepare_pretrain_v4.sh"
echo "                  bash scripts/run_train_4epoch_v4.sh"