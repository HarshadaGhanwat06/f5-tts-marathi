import json
import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(
    "extracted/IISc_SYSPIN_Data/"
    "IISc_SYSPINProject_Marathi_Female_Spk001_HC"
)

WAV_DIR = ROOT / "wav"

JSON_FILE = (
    ROOT /
    "IISc_SYSPINProject_Marathi_Female_Spk001_HC_Transcripts.json"
)

VOCAB_FILE = Path(
    "../../models/openbible-marathi/vocab.txt"
)

OUTPUT = Path("metadata_all.csv")

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

transcripts = data["Transcripts"]

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = {
        line.rstrip("\n\r")
        for line in f
    }

# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

DEVANAGARI_DIGITS = str.maketrans(
    "०१२३४५६७८९",
    "0123456789"
)

QUOTE_MAP = str.maketrans({
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
})

REMOVE_CHARS = "{}|/"

def normalize(text):

    # Devanagari digits → ASCII
    text = text.translate(DEVANAGARI_DIGITS)

    # Smart quotes → normal quotes
    text = text.translate(QUOTE_MAP)

    # Normalize unsupported Marathi/Devanagari characters
    replacements = {
        "ॅ": "े",
        "ँ": "ं",
        "ऑ": "ओ",
        "ॲ": "अ",
        "ङ": "न",
        "ॊ": "ो",
        ";": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove dataset formatting artifacts
    for char in "{}|/":
        text = text.replace(char, "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
# ---------------------------------------------------------
# Process
# ---------------------------------------------------------

rows = []
remaining_oov = Counter()

missing_audio = []
empty_text = []

for key, item in transcripts.items():

    text = normalize(item["Transcript"])

    wav = WAV_DIR / f"{key}.wav"

    if not wav.exists():
        missing_audio.append(key)
        continue

    if not text:
        empty_text.append(key)
        continue

    for char in text:
        if char not in vocab:
            remaining_oov[char] += 1

    rows.append([
        str(wav.resolve()),
        text
    ])

# ---------------------------------------------------------
# Write metadata
# ---------------------------------------------------------

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(
        f,
        delimiter="|",
        quoting=csv.QUOTE_MINIMAL
    )

    for row in rows:
        writer.writerow(row)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("=" * 70)
print("SYSPIN PREPARATION")
print("=" * 70)

print(f"Input transcripts : {len(transcripts):,}")
print(f"Valid rows        : {len(rows):,}")
print(f"Missing audio     : {len(missing_audio):,}")
print(f"Empty text        : {len(empty_text):,}")

print("\nRemaining OOV:")
print("-" * 70)

if remaining_oov:

    for char, count in remaining_oov.most_common():
        print(
            f"{repr(char):8} "
            f"U+{ord(char):04X} "
            f"{count:,}"
        )

else:
    print("✅ ZERO OOV AFTER NORMALIZATION")

print("\nOutput:")
print(OUTPUT.resolve())

print("=" * 70)
