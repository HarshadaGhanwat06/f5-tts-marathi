#!/usr/bin/env bash
# =============================================================================
# prepare_dataset_v9.sh
#
# STEP 2 - DATA PREP for Chandra_Words_v9.
#
# Renders the Cartesia output (cartesia_ws/chandra_words/metadata.csv, comma
# delimited: id,text,audio_file,duration_seconds,source) into the F5-TTS pipe
# format (audio_file|text with ABSOLUTE wav paths), then runs the same
# prepare_csv_wavs.py invocation used by every prior version (--workers 32).
#
# Output:
#   cartesia_ws/chandra_words/train_metadata.csv     (pipe format, absolute)
#   f5tts/data/Chandra_Words_v9/                     (raw.arrow + duration.json)
#   symlink f5tts/lib/python3.12/data/Chandra_Words_v9_custom ->
#           f5tts/data/Chandra_Words_v9
#
# Post-condition (verified): the preprocessed dir contains ONLY raw.arrow,
# duration.json, vocab.txt - NO nested duplicate folder (that stale bug has
# silently used the wrong dataset before on this project).
#
# Gate: < 200 rows -> STOP (exit 3). FORCE=1 overrides.
#
# Usage (server, after create_chandra_words_dataset.sh --full + ear check):
#   bash scripts/prepare_dataset_v9.sh
#
# Then: bash scripts/prepare_pretrain_v9.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
PREPARE="/root/f5-tts-marathi/f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py"
META="/root/f5-tts-marathi/cartesia_ws/chandra_words/metadata.csv"
WAVROOT="/root/f5-tts-marathi/cartesia_ws/chandra_words/output"
TRAIN_CSV="/root/f5-tts-marathi/cartesia_ws/chandra_words/train_metadata.csv"
OUT_DIR="/root/f5-tts-marathi/f5tts/data/Chandra_Words_v9"

echo "============================================================"
echo " Prepare preprocessed dataset : Chandra_Words_v9 (ॅ/ॲ words)"
echo "============================================================"
echo "Source metadata : $META"

if [[ ! -f "$META" ]]; then
    echo "[ERROR] Cartesia metadata not found: $META"
    echo "        Run bash scripts/create_chandra_words_dataset.sh --full first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Render pipe-format metadata with absolute wav paths
# ---------------------------------------------------------------------------
echo ""
echo "[1] Rendering audio_file|text metadata (absolute paths)..."
"$PYTHON" - "$META" "$WAVROOT" "$TRAIN_CSV" <<'PY'
import sys, os, csv

meta, wroot, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = 0
skipped = 0
with open(out, "w", encoding="utf-8") as g:
    g.write("audio_file|text\n")
    with open(meta, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            rel = (row.get("audio_file") or "").strip()
            if not text or not rel:
                continue
            if "|" in text:
                skipped += 1
                print("[WARN] text contains '|', skipping:", text[:40], file=sys.stderr)
                continue
            abs_wav = rel if os.path.isabs(rel) else os.path.join(wroot, rel)
            g.write(f"{abs_wav}|{text}\n")
            n += 1
print(f"Rendered {n} rows (skipped {skipped}) -> {out}")
PY

ROWS=$(($(wc -l < "$TRAIN_CSV") - 1))
echo "    Rows (excl header): $ROWS"
echo "    First 3:"
head -3 "$TRAIN_CSV"

if [[ "$ROWS" -lt 200 ]]; then
    echo ""
    echo "[WARNING] Only $ROWS rows in train_metadata.csv. Too small for a"
    echo "          focused fine-tune without overfitting risk."
    if [[ "${FORCE:-0}" == "1" ]]; then
        echo "[FORCE] FORCE=1 set; proceeding."
    else
        echo ""
        echo "[STOP] Not proceeding. Check cartesia_ws/chandra_words/ (were all"
        echo "       samples synthesized?). Re-run create_chandra_words_dataset.sh"
        echo "       --full, then this script."
        exit 3
    fi
fi

# ---------------------------------------------------------------------------
# 2. Clean stale preprocessed dir, then run prepare_csv_wavs
# ---------------------------------------------------------------------------
if [[ -d "$OUT_DIR" ]]; then
    echo ""
    echo "[WARN] Output dir exists; removing stale copy: $OUT_DIR"
    rm -rf "$OUT_DIR"
fi

echo ""
echo "[2] Running prepare_csv_wavs.py ..."
"$PYTHON" "$PREPARE" "$TRAIN_CSV" "$OUT_DIR" --workers 32

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
# 3. Symlink for the loader (same {name}_custom convention)
# ---------------------------------------------------------------------------
echo ""
echo "[4] Registering symlink for loader at lib/python3.12/data/..."
LINK_ROOT="/root/f5-tts-marathi/f5tts/lib/python3.12/data"
LINK_NAME="Chandra_Words_v9_custom"
mkdir -p "$LINK_ROOT"
if [[ -L "$LINK_ROOT/$LINK_NAME" || -e "$LINK_ROOT/$LINK_NAME" ]]; then
    rm -rf "$LINK_ROOT/$LINK_NAME"
fi
ln -s "$OUT_DIR" "$LINK_ROOT/$LINK_NAME"
ls -la "$LINK_ROOT/$LINK_NAME"

# ---------------------------------------------------------------------------
# 4. Sanity report
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
echo "[INFO] Done. Next: bash scripts/prepare_pretrain_v9.sh"