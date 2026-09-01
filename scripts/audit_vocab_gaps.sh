#!/usr/bin/env bash
#
# audit_vocab_gaps.py (embedded)
#
# Find every Unicode character that appears in the training transcripts but is
# MISSING from the tokenizer vocabulary. Run this BEFORE expanding the vocab.
#
# Run on server:
#   bash scripts/audit_vocab_gaps.sh [--apply]
#
# With --apply it appends the missing characters to vocab_extended.txt
# (newline-separated, matching the existing format), sorted by code point.
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

# Gather every transcript file used in training + the full corpus
transcripts = [
    "/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined/metadata.csv",
    "/root/f5-tts-marathi/cartesia_ws/cartesia_dataset/metadata.csv",
    "/root/f5-tts-marathi/rasa_female_emotions/rasa_train_metadata.csv",
]

categories = set()
chars = set()
files_seen = set()

def collect(path):
    if path in files_seen:
        return
    files_seen.add(path)
    if not os.path.exists(path):
        print("  [WARN] missing transcript file:", path)
        return
    delim = "\t"
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # multi-column: text is last column (metadata.csv has 6 cols, rasa has 2)
            parts = line.split("|")
            text = parts[-1] if len(parts) > 1 else line
            for ch in text:
                if ch.strip() == "":      # skip whitespace
                    continue
                chars.add(ch)
                cat = unicodedata.category(ch)
                categories.add(cat)

for p in transcripts:
    collect(p)

# Load current vocab tokens (each line = 1 token)
vocab_tokens = set()
if os.path.exists(vocab_path):
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                vocab_tokens.add(line)

print("Transcript files scanned:", sorted(files_seen))
print("Unique characters in transcripts:", len(chars))
print("Categories present:", sorted(categories))
print("Current vocab tokens:", len(vocab_tokens))

missing = sorted(ch for ch in chars - vocab_tokens, key=lambda c: ord(c))
print("\nMISSING from vocab:", len(missing))
for ch in missing:
    print(f"   {ch!r} U+{ord(ch):04X} {unicodedata.name(ch,'?')} cat={unicodedata.category(ch)}")
PY
