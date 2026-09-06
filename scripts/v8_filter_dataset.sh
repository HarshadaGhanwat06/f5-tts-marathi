#!/usr/bin/env bash
# =============================================================================
# v8_filter_dataset.sh
#
# STEP 1 — Build the FILTERED training set.
#
# Extracts every sentence from the v7 training corpus that contains ॅ (U+0945)
# or ॲ (U+0972) anywhere in the text. Generalizes beyond any fixed example
# words because it matches the CHARACTER CLASS [ॅॲ].
#
# Authoritative text source: Cartesia_Rasa_Combined_v7's preprocessed
# raw.arrow (the exact texts that fed the v7 run). Audio paths are resolved
# back to absolute paths by matching each kept text against the source
# metadata CSVs, exactly mirroring the absolute-path convention used in all
# prior successful preps on this project.
#
# Output:
#   cartesia_ws/chandra_vyanjan_focus/train_metadata.csv
#      lines: <absolute-wav-path>|<text>     (header included)
#   cartesia_ws/chandra_vyanjan_focus/filter_report.txt
#
# Gate: prints the resulting row count. If fewer than 50 it STOPS (exit 3)
# with an explicit warning instead of silently proceeding with an
# under-represented dataset (overfitting/instability risk, as seen with the
# earlier 53-character vocab expansion). Re-run with FORCE=1 to override.
#
# Usage (server):
#   bash scripts/v8_filter_dataset.sh
#   FORCE=1 bash scripts/v8_filter_dataset.sh      # proceed even if < 50
#
# Exit codes:
#   0 = CSV written (>= 50 rows, or FORCE=1)
#   3 = < 50 rows and FORCE not set (stopped as instructed)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
ARROW="/root/f5-tts-marathi/f5tts/data/Cartesia_Rasa_Combined_v7_custom/raw.arrow"
OUT_DIR="$REPO_ROOT/cartesia_ws/chandra_vyanjan_focus"
OUT_CSV="$OUT_DIR/train_metadata.csv"
REPORT="$OUT_DIR/filter_report.txt"

# Source metadata CSVs + their wav root (same joining as prepare_dataset_v5.sh)
declare -a SRC_CSV=()
declare -a SRC_WAVROOT=()
declare -a SRC_LABEL=()
add_src() { SRC_CSV+=("$1"); SRC_WAVROOT+=("$2"); SRC_LABEL+=("$3"); }

add_src "/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined/metadata.csv" \
        "/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined" "combined"
add_src "/root/f5-tts-marathi/cartesia_ws/cartesia_candra_e/metadata.csv" \
        "/root/f5-tts-marathi/cartesia_ws/cartesia_candra_e/output" "candra_e"
add_src "/root/f5-tts-marathi/cartesia_ws/prono_vowel_dataset/metadata.csv" \
        "/root/f5-tts-marathi/cartesia_ws/prono_vowel_dataset/output" "prono_vowel"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " STEP 1 — Filter v7 corpus for [ॅॲ] (Chandra vowel-sign)"
echo "============================================================"

if [[ ! -f "$ARROW" ]]; then
    echo "[ERROR] v7 preprocessed dataset not found: $ARROW"
    echo "        Expected at f5tts/data/Cartesia_Rasa_Combined_v7_custom/raw.arrow"
    exit 1
fi

# Build source CSV list as a |-joined string for python (robust to spaces)
SRC_SPEC=""
for i in "${!SRC_CSV[@]}"; do
    SRC_SPEC+="${SRC_CSV[$i]}|${SRC_WAVROOT[$i]}|${SRC_LABEL[$i]}|"
done

"$PYTHON" - "$ARROW" "$OUT_CSV" "$REPORT" "$SRC_SPEC" <<'PY'
import sys, os, re, csv

arrow_path, out_csv, report_path, src_spec = sys.argv[1:5]

# ------------------------------------------------------------------
# 1. Read the exact texts that fed Cartesia_Rasa_Combined_v7.
#    F5-TTS writes raw.arrow via df.to_parquet() (misleading .arrow
#    extension), so try parquet first, then arrow-IPC, then feather.
# ------------------------------------------------------------------
import pyarrow

def load_table(path: str):
    """Return a pyarrow Table from a file written in any of: parquet,
    arrow-IPC (file/stream), or feather. Handles the .arrow-is-parquet
    quirk used by prepare_csv_wavs.py."""
    tried = []
    # 1) parquet
    try:
        import pyarrow.parquet as pq
        tried.append("parquet")
        return pq.read_table(path)
    except Exception as e:
        parquet_err = e
    # 2) arrow-IPC file
    try:
        import pyarrow.ipc as ipc
        with open(path, "rb") as f:
            tried.append("arrow-ipc-file")
            return ipc.open_file(f).read_all()
    except Exception as e:
        ipc_err = e
    # 3) arrow-IPC stream
    try:
        import pyarrow.ipc as ipc
        with open(path, "rb") as f:
            tried.append("arrow-ipc-stream")
            return ipc.open_stream(f).read_all()
    except Exception as e:
        stream_err = e
    # 4) feather
    try:
        import pyarrow.feather as ft
        tried.append("feather")
        return ft.read_table(path)
    except Exception as e:
        feather_err = e
    print(f"  [ERROR] Could not read {path} as {', '.join(tried)}")
    print("  parquet :", str(parquet_err)[:120])
    print("  ipc-file:", str(ipc_err)[:120])
    print("  ipc-strm:", str(stream_err)[:120])
    print("  feather :", str(feather_err)[:120])
    raise SystemExit(1)

texts = []
tab = load_table(arrow_path)
print("raw.arrow read successfully (format auto-detected).")
cols = tab.column_names
print("raw.arrow columns:", cols)
# text column is usually "text"; fall back otherwise
tcol = "text" if "text" in cols else cols[0]
raw = tab.column(tcol).to_pylist()
for t in raw:
    if t:
        texts.append(str(t))
print(f"v7 arrow total texts      : {len(texts)}")

# ------------------------------------------------------------------
# 2. Filter by [ॅॲ]
# ------------------------------------------------------------------
pat = re.compile(r"[ॅॲ]")
kept = []
seen = set()
for t in texts:
    if pat.search(t) and t not in seen:
        seen.add(t)
        kept.append(t)
print(f"texts containing ॅ/ॲ     : {len(kept)}")

# ------------------------------------------------------------------
# 3. Resolve each kept text back to an absolute wav path.
#    Build a text->(path,label) map from all source CSVs.
# ------------------------------------------------------------------
sources = []
parts = src_spec.rstrip("|").split("|")
i = 0
while i + 2 < len(parts):
    meta, wroot, label = parts[i], parts[i + 1], parts[i + 2]
    sources.append((meta, wroot, label))
    i += 3

text_to_path = {}
per_label = {}
for meta, wroot, label in sources:
    if not os.path.isfile(meta):
        print(f"  [WARN] source CSV missing, skipping: {label} ({meta})")
        continue
    n = 0
    with open(meta, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            text = (row[1] or "").strip()
            rel = (row[2] or "").strip()
            if not text or not rel:
                continue
            full = os.path.join(wroot, rel)
            if text not in text_to_path:
                text_to_path[text] = (full, label)
                n += 1
    per_label[label] = n
    print(f"  indexed {label}: {n} texts")

# ------------------------------------------------------------------
# 4. Emit filtered rows (absolute wav paths, audio_file|text format)
# ------------------------------------------------------------------
matched = 0
unmatched = 0
missing_wav = 0
with open(out_csv, "w", encoding="utf-8", newline="") as g:
    g.write("audio_file|text\n")
    for t in kept:
        if t not in text_to_path:
            unmatched += 1
            continue
        p, label = text_to_path[t]
        if not os.path.isfile(p):
            missing_wav += 1
            continue
        g.write(f"{p}|{t}\n")
        matched += 1

print("")
print("Filter report:")
print(f"  kept texts with ॅ/ॲ     : {len(kept)}")
print(f"  matched to wav          : {matched}")
print(f"  unmatched in sources    : {unmatched}")
print(f"  wav missing on disk     : {missing_wav}")
print(f"  class matched           : [ॅ ॲ] (any position, generalizes beyond examples)")

with open(report_path, "w", encoding="utf-8") as rf:
    rf.write(f"arrow_total={len(texts)}\n")
    rf.write(f"chandra_rows={len(kept)}\n")
    rf.write(f"matched_to_wav={matched}\n")
    rf.write(f"unmatched={unmatched}\n")
    rf.write(f"wav_missing={missing_wav}\n")

print("")
print(f"CSV written: {out_csv} ({matched + 1} lines incl header)")
PY

EXIT=$?
if [[ $EXIT -ne 0 ]]; then exit $EXIT; fi

# ---------------------------------------------------------------------------
# Gate: < 50 rows -> stop and ask, unless FORCE=1
# ---------------------------------------------------------------------------
ROWS=$(($(wc -l < "$OUT_CSV") - 1))
echo ""
echo "============================================================"
echo " FILTERED ROW COUNT : $ROWS"
echo "============================================================"
head -6 "$OUT_CSV"
echo "  ..."

if [[ "$ROWS" -lt 50 ]]; then
    echo ""
    echo "[WARNING] Only $ROWS samples contain ॅ/ॲ. Too few risks overfitting/"
    echo "          instability (same failure mode as the earlier 53-char vocab"
    echo "          expansion)."
    if [[ "${FORCE:-0}" == "1" ]]; then
        echo "[FORCE] FORCE=1 set; proceeding with $ROWS samples."
    else
        echo ""
        echo "[STOP] Not proceeding. Options:"
        echo "   a) proceed anyway  ->  FORCE=1 bash scripts/v8_filter_dataset.sh"
        echo "   b) broaden filter  ->  discuss row count with the pipeline owner"
        echo "   c) generate more ॅ/ॲ samples and regenerate v7-adjacent corpus, then re-run"
        exit 3
    fi
fi

echo ""
echo "[INFO] Done. Next: bash scripts/prepare_dataset_v8.sh"