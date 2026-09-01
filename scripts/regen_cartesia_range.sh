#!/usr/bin/env bash
# =============================================================================
# regen_cartesia_range.sh
#
# Delete and regenerate a range of Cartesia dataset samples with a fixed speed
# (speaking_rate). Use this after changing voice/speed/params so existing audio
# and its metadata rows are fully replaced instead of skipped/duplicated.
#
# Usage (run on the SERVER from the repo root):
#   bash scripts/regen_cartesia_range.sh --start 201 --limit 5
#
# Optional env:
#   CARTESIA_SPEED   - Arushi synthesis speed multiplier (speaking_rate),
#                      default 0.8. This is passed through to the generator.
#
# Steps performed for the requested line range:
#   1. Delete the matching output/<NNNN>.wav files.
#   2. Remove the matching id rows from metadata.csv and failed_samples.csv
#      (so regeneration does not create duplicate/ghost rows).
#   3. Re-run the generator on that range with --overwrite semantics (the wavs
#      are already gone, so nothing is skipped) using the updated speaking_rate.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PROJECT_DIR="$REPO_ROOT/cartesia_ws"
OUTPUT_DIR="$PROJECT_DIR/output"
METADATA_CSV="$PROJECT_DIR/cartesia_dataset/metadata.csv"
FAILED_CSV="$PROJECT_DIR/cartesia_dataset/failed_samples.csv"

CARTESIA_SPEED="${CARTESIA_SPEED:-0.8}"

START_LINE=""
LIMIT=""

usage() {
    echo "Usage: $0 --start <LINE> [--limit <N>]"
    echo "  --start LINE  first sentence line number (>= 1). REQUIRED."
    echo "  --limit N     number of lines to regenerate (default: to end of file)."
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)
            START_LINE="$2"; shift 2 ;;
        --limit)
            LIMIT="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$START_LINE" ]]; then
    echo "[ERROR] --start is required."
    usage
fi

if [[ ! -f "$PROJECT_DIR/cartesia_dataset/sentences.txt" ]]; then
    echo "[ERROR] sentences.txt not found: $PROJECT_DIR/cartesia_dataset/sentences.txt"
    exit 1
fi
TOTAL_LINES="$(wc -l < "$PROJECT_DIR/cartesia_dataset/sentences.txt")"

# Compute inclusive end line
if [[ -n "$LIMIT" ]]; then
    END_LINE=$((START_LINE + LIMIT - 1))
else
    END_LINE="$TOTAL_LINES"
fi
if (( END_LINE > TOTAL_LINES )); then
    END_LINE="$TOTAL_LINES"
fi

echo "[INFO] Regenerating lines $START_LINE..$END_LINE (speaking_rate=$CARTESIA_SPEED)"
echo "[INFO] Output dir  : $OUTPUT_DIR"
echo "[INFO] metadata    : $METADATA_CSV"
echo "[INFO] failed       : $FAILED_CSV"

# ---------------------------------------------------------------------------
# 1. Delete the WAVs in range
# ---------------------------------------------------------------------------
for (( i = START_LINE; i <= END_LINE; i++ )); do
    fid="$(printf '%04d' "$i")"
    wav="$OUTPUT_DIR/$fid.wav"
    if [[ -f "$wav" ]]; then
        rm -f "$wav"
        echo "[INFO] Deleted $wav"
    fi
done

# ---------------------------------------------------------------------------
# 2. Remove matching rows from metadata.csv and failed_samples.csv
# ---------------------------------------------------------------------------
# Build a set of ids to drop: 0201..0205
ids_to_drop=""
for (( i = START_LINE; i <= END_LINE; i++ )); do
    fid="$(printf '%04d' "$i")"
    ids_to_drop="$ids_to_drop $fid"
done
ids_to_drop="$(echo "$ids_to_drop" | sed 's/^ //')"

for csv in "$METADATA_CSV" "$FAILED_CSV"; do
    if [[ -f "$csv" ]]; then
        grep -vE "^($(echo "$ids_to_drop" | sed 's/ /|/g'))[,|]" "$csv" > "$csv.tmp" \
            || true
        mv "$csv.tmp" "$csv"
        echo "[INFO] Removed rows ${ids_to_drop// /,} from $(basename "$csv")"
    fi
done

# ---------------------------------------------------------------------------
# 3. Regenerate with the fixed speaking_rate
# ---------------------------------------------------------------------------
export CARTESIA_SPEED
LIMIT_OPT=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_OPT="--limit $LIMIT"
fi
bash "$SCRIPT_DIR/run_cartesia_dataset.sh" --start "$START_LINE" $LIMIT_OPT

echo
echo "[INFO] Done regenerating lines $START_LINE..$END_LINE."
