#!/usr/bin/env bash
# =============================================================================
# prepare_dataset_v8.sh
#
# STEP 2 — DATA PREP for Chandra_Vyanjan_Focus_v8.
#
# Runs prepare_csv_wavs.py on the filtered [ॅॲ] metadata produced by
# v8_filter_dataset.sh, exactly as done for every prior dataset version
# (v2..v7): same invocation, same --workers 32, same output layout.
#
# Output:
#   f5tts/data/Chandra_Vyanjan_Focus_v8/        (raw.arrow + duration.json)
#   symlink f5tts/lib/python3.12/data/Chandra_Vyanjan_Focus_v8_custom ->
#           f5tts/data/Chandra_Vyanjan_Focus_v8
#
# Post-condition (verified by this script): the preprocessed dir contains ONLY
# raw.arrow, duration.json, vocab.txt - NO nested duplicate folder. This exact
# stale-nested-folder bug has silently used the wrong dataset before on this
# project.
#
# Usage (server, after v8_filter_dataset.sh):
#   bash scripts/prepare_dataset_v8.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
PREPARE="/root/f5-tts-marathi/f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py"
META="$REPO_ROOT/cartesia_ws/chandra_vyanjan_focus/train_metadata.csv"
OUT_DIR="/root/f5-tts-marathi/f5tts/data/Chandra_Vyanjan_Focus_v8"

echo "============================================================"
echo " Prepare preprocessed dataset : Chandra_Vyanjan_Focus_v8"
echo "============================================================"
echo "Source metadata : $META"
echo "Output dir      : $OUT_DIR"

if [[ ! -f "$META" ]]; then
    echo "[ERROR] Filtered metadata not found: $META"
    echo "        Run bash scripts/v8_filter_dataset.sh first."
    exit 1
fi

echo ""
echo "[1] Rows to prepare: $(wc -l < "$META") (incl header)"
echo "    First 3:"
head -3 "$META"

# ---------------------------------------------------------------------------
# Clean any stale previous preprocessed dir (avoids nested/stale duplicate)
# ---------------------------------------------------------------------------
if [[ -d "$OUT_DIR" ]]; then
    echo ""
    echo "[WARN] Output dir exists; removing stale copy: $OUT_DIR"
    rm -rf "$OUT_DIR"
fi

echo ""
echo "[2] Running prepare_csv_wavs.py ..."
"$PYTHON" "$PREPARE" "$META" "$OUT_DIR" --workers 32

echo ""
echo "[3] Output dir contents (must be only raw.arrow/duration.json/vocab.txt):"
find "$OUT_DIR" -maxdepth 1 -type f | sort
echo ""
echo "    Nested dirs (must be NONE):"
NESTED=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d)
if [[ -n "$NESTED" ]]; then
    echo "$NESTED"
    echo "[ERROR] Nested folder(s) found. The loader may use a stale/wrong dataset."
    echo "        Aborting. Delete $OUT_DIR and rerun."
    exit 1
else
    echo "    (none - OK)"
fi

# ---------------------------------------------------------------------------
# Symlink for the loader (same name convention as prior versions: {name}_custom)
# ---------------------------------------------------------------------------
echo ""
echo "[4] Registering symlink for loader at lib/python3.12/data/..."
LINK_ROOT="/root/f5-tts-marathi/f5tts/lib/python3.12/data"
LINK_NAME="Chandra_Vyanjan_Focus_v8_custom"
mkdir -p "$LINK_ROOT"
if [[ -L "$LINK_ROOT/$LINK_NAME" || -e "$LINK_ROOT/$LINK_NAME" ]]; then
    rm -rf "$LINK_ROOT/$LINK_NAME"
fi
ln -s "$OUT_DIR" "$LINK_ROOT/$LINK_NAME"
ls -la "$LINK_ROOT/$LINK_NAME"

# ---------------------------------------------------------------------------
# Sanity report (sample count + total hours from duration.json)
# ---------------------------------------------------------------------------
echo ""
echo "[5] Dataset sanity report:"
"$PYTHON" - "$OUT_DIR" <<'PY'
import sys, json, os
d = sys.argv[1]
j = os.path.join(d, "duration.json")
if os.path.exists(j):
    x = json.load(open(j, encoding="utf-8"))
    durs = x.get("duration", [])
    if isinstance(durs, dict):
        vs = list(durs.values())
    elif isinstance(durs, list):
        vs = durs
    else:
        vs = []
    print("  duration.json entries :", len(vs))
    if vs:
        print("  total duration hrs    :", round(sum(vs)/3600, 3))
        print("  avg duration s        :", round(sum(vs)/len(vs), 2))
else:
    print("  duration.json NOT FOUND")
print("  raw.arrow exists      :", os.path.exists(os.path.join(d, "raw.arrow")))
print("  vocab.txt exists      :", os.path.exists(os.path.join(d, "vocab.txt")))
PY

echo ""
echo "[INFO] Done. Next: bash scripts/v8_prepare_pretrain.sh"