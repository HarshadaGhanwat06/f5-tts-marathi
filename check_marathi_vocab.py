from pathlib import Path

VOCAB_PATH = "models/openbible-marathi/vocab.txt"

REF_TEXT = "मी ठीक आहे तुम्ही कसे आहात? मला मदत करा आपण पुन्हा भेटू"

GEN_TEXT = "आपण भारतात राहत असूनही इथल्या वेगवेगळ्या प्रांतात बोलल्या जाणाऱ्या भाषांची माहिती आपल्याला नसते."

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = {line.rstrip("\n\r") for line in f}

print("=" * 70)
print("MARATHI VOCABULARY CHECK")
print("=" * 70)

print(f"Vocabulary entries: {len(vocab)}")


def check_text(name, text):
    chars = set(text)

    missing = sorted(
        c for c in chars
        if c not in vocab
    )

    print(f"\n{name}")
    print("-" * 70)
    print(f"Text: {text}")
    print(f"Unique characters: {len(chars)}")

    if missing:
        print(f"\n❌ OOV characters ({len(missing)}):")

        for c in missing:
            print(
                f"  {repr(c)}  Unicode: U+{ord(c):04X}"
            )
    else:
        print("\n✅ No OOV characters")


check_text("REFERENCE TEXT", REF_TEXT)
check_text("GENERATION TEXT", GEN_TEXT)
