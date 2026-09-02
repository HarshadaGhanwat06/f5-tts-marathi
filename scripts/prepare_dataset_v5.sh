#!/usr/bin/env bash
# =============================================================================
# prepare_dataset_v5.sh
#
# Build the PREPROCESSED fine-tuning dataset for Cartesia_Rasa_Combined_v5:
#
#   = the complete v3 source            (cartesia_rasa_3h_combined, 2663 rows)  REQUIRED
#   + the candra_e targeted set         (cartesia_candra_e, ~210 rows)           OPTIONAL*
#   + the pronunciation vowel/matra set (prono_vowel_dataset, 528 rows)          REQUIRED
#
#   * candra_e is included automatically IF present; it is skipped (with a
#     warning) when it was never generated.
#
# Output:
#   f5tts/data/Cartesia_Rasa_Combined_v5_custom/   (raw.arrow + duration.json)
#   symlink at f5tts/lib/python3.12/data/Cartesia_Rasa_Combined_v5_custom
#
# Run on server (after create_candra_e_dataset.sh --full AND
# create_prono_vowel_dataset.sh --full + ear check):
#   bash scripts/prepare_dataset_v5.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
PREPARE="/root/f5-tts-marathi/f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py"

# Source triples:  metadata.csv, wav root
COMBINED_META="/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined/metadata.csv"
COMBINED_WAVS="/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined"
CANDRA_META="/root/f5-tts-marathi/cartesia_ws/cartesia_candra_e/metadata.csv"
CANDRA_WAVS="/root/f5-tts-marathi/cartesia_ws/cartesia_candra_e/output"
PRONO_META="/root/f5-tts-marathi/cartesia_ws/prono_vowel_dataset/metadata.csv"
PRONO_WAVS="/root/f5-tts-marathi/cartesia_ws/prono_vowel_dataset/output"

OUT_DIR="/root/f5-tts-marathi/f5tts/data/Cartesia_Rasa_Combined_v5_custom"
TMP_CSV="/tmp/cartesia_v5_f5tts_meta.csv"

echo "============================================================"
echo " Prepare preprocessed dataset for Cartesia_Rasa_Combined_v5"
echo "============================================================"
echo "Combined (v3 source) : $COMBINED_META"
echo "Candra_e set         : $CANDRA_META"
echo "Prono vowel set      : $PRONO_META"
echo "Output dir           : $OUT_DIR"

for f in "$COMBINED_META" "$PRONO_META"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Not found: $f"
        echo "        Run the matching create_*.sh --full scripts first."
        exit 1
    fi
done
if [[ ! -f "$CANDRA_META" ]]; then
    echo "[WARN] candra_e metadata NOT found: $CANDRA_META"
    echo "       Skipping candra_e source (v5 = combined + prono_vowel only)."
fi

# ---------------------------------------------------------------------------
# 1. Render audio_file|text metadata (absolute wav paths, all three sources)
# ---------------------------------------------------------------------------
echo ""
echo "[1] Rendering F5-TTS metadata (audio_file|text, absolute paths)..."
"$PYTHON" - "$COMBINED_META" "$COMBINED_WAVS" "$CANDRA_META" "$CANDRA_WAVS" "$PRONO_META" "$PRONO_WAVS" "$TMP_CSV" <<'PY'
import sys, os, csv

sources = [
    (sys.argv[1], sys.argv[2], "combined"),
    (sys.argv[3], sys.argv[4], "candra_e"),
    (sys.argv[5], sys.argv[6], "prono_vowel"),
]
out = sys.argv[7]

n = 0
skipped_pipe = 0
counts = {}

with open(out, "w", encoding="utf-8") as g:
    g.write("audio_file|text\n")
    for meta, wroot, label in sources:
        if not os.path.isfile(meta):
            print(f"[WARN] skipping missing source: {label} ({meta})")
            continue
        cnt = 0
        with open(meta, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # all metas are comma-delimited: id,text,audio_file,duration_seconds[,source]
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
                cnt += 1
                n += 1
        counts[label] = cnt

print("Rows per source:", counts)
print(f"Wrote {n} rows to {out} (skipped {skipped_pipe} with '|')")
PY

echo ""
echo "[2] Row counts / headers:"
for label in "combined" "candra_e" "prono_vowel"; do
    case "$label" in
        combined) f="$COMBINED_META" ;;
        candra_e) f="$CANDRA_META" ;;
        prono_vowel) f="$PRONO_META" ;;
    esac
    if [[ -f "$f" ]]; then
        echo "  $label: $(wc -l < "$f") lines (incl header)"
    else
        echo "  $label: SKIPPED (not present)"
    fi
done
echo "  rendered : $(wc -l < "$TMP_CSV") lines (incl header)"
echo ""
echo "    First 3 rendered rows (combined):"
head -3 "$TMP_CSV"
echo "    Last 3 rendered rows (prono_vowel):"
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
LINK_NAME="Cartesia_Rasa_Combined_v5_custom"
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
else:
    print("duration.json NOT FOUND")
print("raw.arrow exists:", os.path.exists(os.path.join(d, "raw.arrow")))
PY

echo ""
echo "[INFO] Done. Now run:"
echo "  bash scripts/prepare_pretrain_v5.sh   (v4 trained -> warm start, or use v3)"
echo "  bash scripts/run_train_4epoch_v5.sh   (bash scripts/run_train_4epoch_v4.sh DATASET_NAME=Cartesia_Rasa_Combined_v5 ...)"