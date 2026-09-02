#!/usr/bin/env bash
# =============================================================================
# prepare_pretrain_v5.sh
#
# Warm-start for the v5 fine-tune from the most recent TRAINED checkpoint
# (v4 if present, else v3). Continuing from trained weights keeps the
# accumulated pronunciation gains while the new prono_vowel set
# (ॅ/ॉ/ँ/ॉं) adds systematic acoustic-exposure samples.
#
# finetune_cli requires the pretrain basename to be "model_extended.pt", so we
# copy it to:
#   ckpts/Rasa_Marathi_Emotion_Female_v5/model_extended.pt
#
# Optional override:
#   CARTESIA_PRETRAIN=/path/to/trained/model_last.pt bash scripts/prepare_pretrain_v5.sh
#
# Run on server (after prepare_dataset_v5.sh):
#   bash scripts/prepare_pretrain_v5.sh
# =============================================================================
set -euo pipefail

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
CKPT_OUT="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female_v5/model_extended.pt"
VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"

CANDIDATES=(
    "/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_v4/model_last.pt"
    "/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Cartesia_Rasa_Combined_v3/model_last.pt"
)

CKPT_IN=""
if [[ -n "${CARTESIA_PRETRAIN:-}" ]]; then
    CKPT_IN="$CARTESIA_PRETRAIN"
elif [[ -f "${CANDIDATES[0]}" ]]; then
    CKPT_IN="${CANDIDATES[0]}"
elif [[ -f "${CANDIDATES[1]}" ]]; then
    CKPT_IN="${CANDIDATES[1]}"
fi

if [[ -z "$CKPT_IN" ]]; then
    echo "[ERROR] No trained checkpoint found. Run training for v3/v4 first,"
    echo "        or set CARTESIA_PRETRAIN=/path/to/trained/model_last.pt"
    exit 1
fi
if [[ ! -f "$VOCAB" ]]; then
    echo "[ERROR] vocab not found: $VOCAB"
    exit 1
fi

echo "============================================================"
echo " Warm-start pretrain for v5 (from trained checkpoint)"
echo "============================================================"
echo "Input  (trained)  : $CKPT_IN"
echo "Output (v5 pretrain): $CKPT_OUT"
echo "Vocab             : $VOCAB"

mkdir -p "$(dirname "$CKPT_OUT")"
cp -f "$CKPT_IN" "$CKPT_OUT"
echo "Copied $CKPT_IN -> $CKPT_OUT"

# ---------------------------------------------------------------------------
# Verify embedding rows == vocab lines + 1
# ---------------------------------------------------------------------------
"$PYTHON" - "$CKPT_OUT" "$VOCAB" <<'PY'
import sys, torch

ckpt_out, vocab_path = sys.argv[1], sys.argv[2]

vocab_size = 0
with open(vocab_path, encoding="utf-8") as f:
    for line in f:
        if line.rstrip("\n"):
            vocab_size += 1
padded = vocab_size + 1
print(f"Vocab lines      : {vocab_size} (model text_num_embeds expected = {padded})")

sd = torch.load(ckpt_out, map_location="cpu")
if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
    state = sd["model_state_dict"]
else:
    state = sd

key = "transformer.text_embed.text_embed.weight"
if key not in state:
    print("ERROR: key not found:", key)
    for k in state:
        if "text_embed" in k and "weight" in k:
            print("   ", k)
    sys.exit(1)

rows = state[key].shape[0]
print(f"Embedding rows   : {rows}")
print(f"Keys preserved   : {len(state)}")

if rows != padded:
    print(f"[ERROR] expected {padded} rows but checkpoint has {rows}. Aborting.")
    sys.exit(1)

print(f"[OK] v5 pretrain ready: {rows} text embeddings, matches expanded vocab.")
print("     Starting fine-tune from carried-forward trained weights.")
PY

echo ""
echo "[INFO] Done. Now run: bash scripts/run_train_4epoch_v5.sh"