#!/usr/bin/env bash
#
# expand_vocab.sh [--apply]
#
# Append every character found in the training transcripts that is currently
# missing from vocab_extended.txt. Appending at the END preserves the existing
# token IDs, adding new ones afterward (the model's text_num_embeds is derived
# from the vocab file size, so retraining adapts automatically).
#
# Without --apply it only prints what WOULD be added (dry run).
# With --apply  it appends the missing characters (sorted by code point) to
#               vocab_extended.txt.
#
# Run on server:
#   bash scripts/expand_vocab.sh          # dry run
#   bash scripts/expand_vocab.sh --apply  # apply
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"
APPLY="${1:-}"

python3 - "$VOCAB" "$APPLY" <<'PY'
import sys, os, unicodedata

vocab_path = sys.argv[1]
apply = len(sys.argv) > 2 and sys.argv[2] == "--apply"

transcripts = [
    "/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined/metadata.csv",
    "/root/f5-tts-marathi/cartesia_ws/cartesia_dataset/metadata.csv",
    "/root/f5-tts-marathi/rasa_female_emotions/rasa_train_metadata.csv",
]

chars = set()
seen = set()
for p in transcripts:
    if p in seen:
        continue
    seen.add(p)
    if not os.path.exists(p):
        continue
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            text = parts[-1] if len(parts) > 1 else line
            for ch in text:
                if ch.strip() != "":
                    chars.add(ch)

vocab_tokens = set()
with open(vocab_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            vocab_tokens.add(line)

missing = sorted([ch for ch in (chars - vocab_tokens)], key=lambda c: ord(c))

print("Transcripts scanned:", len(seen))
print("Current vocab size  :", len(vocab_tokens))
print("Missing characters  :", len(missing))

for ch in missing:
    print(f"   {ch!r} U+{ord(ch):04X} {unicodedata.name(ch,'?')} cat={unicodedata.category(ch)}")

if not missing:
    print("\nNothing to add. Vocab is complete for these transcripts.")
    sys.exit(0)

if not apply:
    print("\n[DRY RUN] Run with --apply to append these to the vocab.")
    sys.exit(0)

# Read existing tokens IN ORDER preserving current file layout
existing = []
with open(vocab_path, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if line:
            existing.append(line)

# New token list = old tokens (a-z added if any) + missing, appended
# Preserve order for existing tokens; append missing sorted by code point.
new_tokens = list(existing)
seen_new = set(new_tokens)
added = 0
for ch in missing:
    if ch not in seen_new:
        new_tokens.append(ch)
        seen_new.add(ch)
        added += 1

with open(vocab_path, "w", encoding="utf-8", newline="\n") as f:
    for t in new_tokens:
        f.write(t + "\n")

print(f"\n[APPLIED] vocab_extended.txt updated: {len(existing)} -> {len(new_tokens)} tokens (+{added}).")
PY
