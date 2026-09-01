#!/usr/bin/env bash
#
# prepare_pretrain_v3.py (embedded)
#
# Resize the text-embedding of a pretrain checkpoint to match the EXPANDED
# vocab (138 tokens -> model text_num_embeds = 139) so fine-tuning can start
# from the existing pretrain despite the added ॅ/Devanagari/Latin characters.
#
# Only the key transformer.text_embed.text_embed.weight is resized
# (old_rows -> new_rows). Existing rows are copied in place (their token IDs
# are unchanged because the vocab was appended), new rows are randomly
# initialized.
#
# Run on server:
#   bash scripts/prepare_pretrain_v3.sh
#
set -euo pipefail

PYTHON="/root/f5-tts-marathi/f5tts/bin/python3"
CKPT_IN="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female/model_extended.pt"
CKPT_OUT="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female/model_extended_v3.pt"
VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"

"$PYTHON" - "$CKPT_IN" "$CKPT_OUT" "$VOCAB" <<'PY'
import sys, torch

ckpt_in, ckpt_out, vocab_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Count vocab lines -> model embeds = tokens + 1 blank
vocab_size = 0
with open(vocab_path, encoding="utf-8") as f:
    for line in f:
        if line.rstrip("\n"):
            vocab_size += 1
new_num = vocab_size + 1

print(f"Vocab lines        : {vocab_size} (model text_num_embeds = {new_num})")
print("Loading:", ckpt_in)
sd = torch.load(ckpt_in, map_location="cpu")

# Some checkpoints wrap under 'model_state_dict'
if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
    state = sd["model_state_dict"]
    wrapped = True
else:
    state = sd
    wrapped = False

key = "transformer.text_embed.text_embed.weight"
if key not in state:
    print("ERROR: key not found:", key)
    print("Available keys containing 'text_embed.weight':")
    for k in state:
        if "text_embed" in k and "weight" in k:
            print("   ", k)
    sys.exit(1)

w = state[key]
old_num, dim = w.shape[0], w.shape[1]
print(f"Current embedding   : {old_num} x {dim}")
print(f"Target embedding    : {new_num} x {dim}")

if new_num == old_num:
    print("Already matches; no change needed.")
    torch.save(sd, ckpt_out)
    print("Saved (unchanged):", ckpt_out)
    sys.exit(0)
if new_num < old_num:
    print(f"ERROR: target ({new_num}) < current ({old_num}); refusing to shrink.")
    sys.exit(1)

# Copy existing rows, random-init the new tail rows
new_w = torch.empty(new_num, dim, dtype=w.dtype)
new_w[:old_num] = w
# init new rows (std matching nn.Embedding default)
torch.nn.init.normal_(new_w[old_num:], mean=0.0, std=1.0)
state[key] = new_w

torch.save(sd, ckpt_out)
print("")
print(f"[OK] Resized {old_num} -> {new_num} rows.")
print("Saved:", ckpt_out)
print("Remaining keys preserved:", len(state))
PY
