import json
from pathlib import Path
from collections import Counter

VOCAB_PATH = Path("models/openbible-marathi/vocab.txt")

JSON_PATH = Path(
    "datasets/syspin_marathi_female/extracted/"
    "IISc_SYSPIN_Data/"
    "IISc_SYSPINProject_Marathi_Female_Spk001_HC/"
    "IISc_SYSPINProject_Marathi_Female_Spk001_HC_Transcripts.json"
)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = {
        line.rstrip("\n\r")
        for line in f
    }

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

transcripts = data["Transcripts"]

char_counts = Counter()

for item in transcripts.values():
    char_counts.update(item["Transcript"])

missing = {
    char: count
    for char, count in char_counts.items()
    if char not in vocab
}

print("=" * 70)
print("SYSPIN MARATHI FULL-CORPUS VOCABULARY CHECK")
print("=" * 70)

print(f"Vocabulary entries : {len(vocab)}")
print(f"Transcripts checked: {len(transcripts):,}")
print(f"Dataset characters : {len(char_counts)}")
print(f"OOV characters     : {len(missing)}")

if missing:

    print("\n❌ OOV CHARACTERS")
    print("-" * 70)

    for char, count in sorted(
        missing.items(),
        key=lambda x: -x[1]
    ):
        print(
            f"{repr(char):8} "
            f"Unicode: U+{ord(char):04X} "
            f"Count: {count:,}"
        )

else:

    print("\n✅ ZERO OOV CHARACTERS")

print("\n" + "=" * 70)
