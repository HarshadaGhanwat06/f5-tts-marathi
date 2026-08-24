import json
from pathlib import Path
from collections import Counter

JSON_FILE = Path(
    "datasets/syspin_marathi_female/extracted/"
    "IISc_SYSPIN_Data/"
    "IISc_SYSPINProject_Marathi_Female_Spk001_HC/"
    "IISc_SYSPINProject_Marathi_Female_Spk001_HC_Transcripts.json"
)

# Change this to the actual vocab file you are currently using.
VOCAB_FILE = Path("PATH_TO_YOUR_MARATHI_VOCAB")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

transcripts = data["Transcripts"]

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = set(line.rstrip("\n") for line in f)

dataset_chars = Counter()

for item in transcripts.values():
    text = item["Transcript"]
    dataset_chars.update(text)

oov = {
    char: count
    for char, count in dataset_chars.items()
    if char not in vocab
}

print("=" * 70)
print("SYSPIN MARATHI FULL-CORPUS OOV CHECK")
print("=" * 70)

print(f"Transcripts checked : {len(transcripts):,}")
print(f"Vocab size          : {len(vocab)}")
print(f"Dataset characters  : {len(dataset_chars)}")
print(f"OOV characters      : {len(oov)}")

if oov:
    print("\nOOV CHARACTERS")
    print("-" * 70)

    for char, count in sorted(oov.items(), key=lambda x: -x[1]):
        print(f"{repr(char):10} : {count:,}")

else:
    print("\n✅ ZERO OOV CHARACTERS")

print("\n" + "=" * 70)
