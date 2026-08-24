from datasets import load_dataset
import soundfile as sf
import numpy as np

TARGET_SPEAKER = "SPEAKER_00"
OUTPUT_PATH = "audio/training_speaker00.wav"

print("=" * 70)
print("Downloading ONE Marathi training audio sample")
print("=" * 70)

print(f"[INFO] Target speaker: {TARGET_SPEAKER}")
print("[INFO] Dataset: davidguzmanr/open-bible-resources")
print("[INFO] Config: Marathi")
print("[INFO] Split: train")
print("[INFO] Streaming: enabled")
print("[INFO] Only ONE matching audio file will be saved.")

ds = load_dataset(
    "davidguzmanr/open-bible-resources",
    "Marathi",
    split="train",
    streaming=True,
)

for i, sample in enumerate(ds):

    speaker = sample["speaker_id"]

    if speaker != TARGET_SPEAKER:
        continue

    print("\n[FOUND]")
    print(f"Speaker ID : {speaker}")
    print(f"Text       : {sample['text']}")
    print(f"Duration   : {sample['duration_seconds']:.2f} sec")

    audio = sample["audio"]

    waveform = np.asarray(audio["array"])
    sample_rate = audio["sampling_rate"]

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples    : {len(waveform)}")

    sf.write(
        OUTPUT_PATH,
        waveform,
        sample_rate,
    )

    print("\n[SUCCESS]")
    print(f"Saved ONLY ONE audio file:")
    print(f"  {OUTPUT_PATH}")

    break

else:
    print(f"[ERROR] Could not find {TARGET_SPEAKER}")
