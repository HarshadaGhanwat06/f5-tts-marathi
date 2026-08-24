import json
import wave
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

# ---------------------------------------------------------
# Load JSON
# ---------------------------------------------------------

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

transcripts = data["Transcripts"]
speaker = data["SpeakersMetaData"]

print("=" * 70)
print("SYSPIN MARATHI FEMALE DATASET AUDIT")
print("=" * 70)

print("\nSPEAKER")
print("-" * 70)

for key, value in speaker.items():
    print(f"{key:20}: {value}")

# ---------------------------------------------------------
# WAV files
# ---------------------------------------------------------

wav_files = list(WAV_DIR.glob("*.wav"))
wav_map = {p.stem: p for p in wav_files}

print("\nDATASET")
print("-" * 70)

print(f"Transcript entries       : {len(transcripts):,}")
print(f"WAV files                : {len(wav_files):,}")

# ---------------------------------------------------------
# Audit
# ---------------------------------------------------------

characters = Counter()
durations = []

missing_wav = []
empty_transcript = []
wav_without_transcript = []

sample_rates = Counter()
channels = Counter()
sample_widths = Counter()

for key, item in transcripts.items():

    wav_path = wav_map.get(key)

    if wav_path is None:
        missing_wav.append(key)
        continue

    text = item.get("Transcript", "").strip()

    if not text:
        empty_transcript.append(key)

    characters.update(text)

    try:
        with wave.open(str(wav_path), "rb") as wf:

            rate = wf.getframerate()
            channel_count = wf.getnchannels()
            sample_width = wf.getsampwidth()

            sample_rates[rate] += 1
            channels[channel_count] += 1
            sample_widths[sample_width] += 1

            duration = wf.getnframes() / rate
            durations.append(duration)

    except Exception as e:
        print(f"ERROR reading {wav_path}: {e}")

# WAVs that have no transcript
transcript_keys = set(transcripts.keys())

for stem in wav_map:
    if stem not in transcript_keys:
        wav_without_transcript.append(stem)

# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------

print("\nMATCHING")
print("-" * 70)

print(f"Missing WAVs             : {len(missing_wav):,}")
print(f"WAV without transcript  : {len(wav_without_transcript):,}")
print(f"Empty transcripts        : {len(empty_transcript):,}")

# ---------------------------------------------------------
# Audio statistics
# ---------------------------------------------------------

print("\nAUDIO")
print("-" * 70)

if durations:

    total_duration = sum(durations)

    print(f"Audited WAVs             : {len(durations):,}")
    print(f"Total duration           : {total_duration / 3600:.2f} hours")
    print(f"Minimum duration         : {min(durations):.2f} sec")
    print(f"Maximum duration         : {max(durations):.2f} sec")
    print(
        f"Mean duration            : "
        f"{total_duration / len(durations):.2f} sec"
    )

print(f"\nSample rates             : {dict(sample_rates)}")
print(f"Channels                 : {dict(channels)}")
print(f"Sample widths            : {dict(sample_widths)}")

# ---------------------------------------------------------
# Text statistics
# ---------------------------------------------------------

print("\nTEXT")
print("-" * 70)

print(f"Unique characters        : {len(characters)}")

print("\nCharacters:")
print("".join(sorted(characters)))

# ---------------------------------------------------------
# Domain statistics
# ---------------------------------------------------------

domains = Counter(
    item.get("Domain", "UNKNOWN")
    for item in transcripts.values()
)

print("\nDOMAINS")
print("-" * 70)

for domain, count in domains.most_common():
    print(f"{domain:20}: {count:,}")

# ---------------------------------------------------------
# Save reports
# ---------------------------------------------------------

with open("syspin_characters.txt", "w", encoding="utf-8") as f:
    f.write("".join(sorted(characters)))

with open("missing_wavs.txt", "w", encoding="utf-8") as f:
    for key in missing_wav:
        f.write(key + "\n")

with open("wav_without_transcript.txt", "w", encoding="utf-8") as f:
    for key in wav_without_transcript:
        f.write(key + "\n")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)