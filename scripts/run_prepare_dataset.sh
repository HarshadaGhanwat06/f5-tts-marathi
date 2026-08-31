#!/usr/bin/env bash
#
# run_prepare_dataset.sh
#
# Prepares the combined fine-tuning dataset for Marathi F5-TTS:
#
#   ~3 hours representative Rasa female Marathi audio
#   + all ~200 Cartesia code-mixed samples
#   -> cartesia_rasa_3h_combined/
#
# IMPORTANT: The Syspin dataset is NOT used anywhere in this workflow.
#
# This is dataset PREPARATION ONLY. It does NOT start fine-tuning, does NOT
# modify the original Rasa dataset, the original Cartesia dataset, the existing
# checkpoint, or any training configuration.
#
# Usage:
#   ./scripts/run_prepare_dataset.sh
#
# Required environment variable:
#   CARTESIA_VOICE_ID   - Cartesia voice ID (used only for reference/logging)
#
# Configurable via env (defaults below):
#   RASA_AUDIO_DIR       - Rasa audio dir (default: /root/f5-tts-marathi/rasa_female_emotions/wavs_female/train)
#   RASA_METADATA_CSV    - Rasa metadata CSV (default: <rasa dir>/rasa_train_metadata.csv)
#   CARTESIA_OUTPUT_DIR  - Cartesia output dir (default: /root/f5-tts-marathi/cartesia_ws/output)
#   CARTESIA_SENTENCES   - Cartesia sentences.txt (default: /root/f5-tts-marathi/cartesia_ws/cartesia_dataset/sentences.txt)
#   TARGET_SECONDS       - Target Rasa duration (default: 10800)
#   MIN_SAMPLES_PER_CAT  - Minimum samples per category (default: 3)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths (server layout)
RASA_AUDIO_DIR="${RASA_AUDIO_DIR:-/root/f5-tts-marathi/rasa_female_emotions/wavs_female/train}"
RASA_METADATA_CSV="${RASA_METADATA_CSV:-/root/f5-tts-marathi/rasa_female_emotions/rasa_train_metadata.csv}"
CARTESIA_OUTPUT_DIR="${CARTESIA_OUTPUT_DIR:-/root/f5-tts-marathi/cartesia_ws/output}"
CARTESIA_SENTENCES="${CARTESIA_SENTENCES:-/root/f5-tts-marathi/cartesia_ws/cartesia_dataset/sentences.txt}"
TARGET_SECONDS="${TARGET_SECONDS:-10800}"
MIN_SAMPLES_PER_CAT="${MIN_SAMPLES_PER_CAT:-3}"

OUT_DIR="/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined"

echo "[INFO] Combined dataset dir  : $OUT_DIR"
echo "[INFO] Rasa audio dir        : $RASA_AUDIO_DIR"
echo "[INFO] Rasa metadata CSV     : $RASA_METADATA_CSV"
echo "[INFO] Cartesia output dir   : $CARTESIA_OUTPUT_DIR"
echo "[INFO] Cartesia sentences    : $CARTESIA_SENTENCES"
echo "[INFO] Target Rasa duration  : $TARGET_SECONDS seconds"
echo ""

# Basic pre-checks (non-fatal; the Python script reports detailed errors)
for p in "$RASA_AUDIO_DIR" "$RASA_METADATA_CSV" "$CARTESIA_OUTPUT_DIR" "$CARTESIA_SENTENCES"; do
    if [[ ! -e "$p" ]]; then
        echo "[WARN] Path not found: $p"
    fi
done

# ---------------------------------------------------------------------------
# Generate the Python preparation script
# ---------------------------------------------------------------------------
cat > "$SCRIPT_DIR/_prepare_combined_dataset.py" <<'PY_EOF'
#!/usr/bin/env python3
"""
Prepare the combined fine-tuning dataset for Marathi F5-TTS.

~3 hours representative Rasa female Marathi audio + all ~200 Cartesia
code-mixed samples -> cartesia_rasa_3h_combined/.

Syspin data is NOT used anywhere. This script only PREPARES the dataset;
it does not start training or modify any source dataset.
"""

import csv
import os
import random
import shutil
import statistics
import sys
import wave

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
RASA_AUDIO_DIR = os.environ.get(
    "RASA_AUDIO_DIR",
    "/root/f5-tts-marathi/rasa_female_emotions/wavs_female/train")
RASA_METADATA_CSV = os.environ.get(
    "RASA_METADATA_CSV",
    "/root/f5-tts-marathi/rasa_female_emotions/rasa_train_metadata.csv")
CARTESIA_OUTPUT_DIR = os.environ.get(
    "CARTESIA_OUTPUT_DIR",
    "/root/f5-tts-marathi/cartesia_ws/output")
CARTESIA_SENTENCES = os.environ.get(
    "CARTESIA_SENTENCES",
    "/root/f5-tts-marathi/cartesia_ws/cartesia_dataset/sentences.txt")
TARGET_SECONDS = int(os.environ.get("TARGET_SECONDS", "10800"))
MIN_SAMPLES_PER_CAT = int(os.environ.get("MIN_SAMPLES_PER_CAT", "3"))

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined")
WAVS_DIR = os.path.join(OUT_DIR, "wavs")
METADATA_CSV = os.path.join(OUT_DIR, "metadata.csv")
SUMMARY_TXT = os.path.join(OUT_DIR, "dataset_summary.txt")
SELECTION_LOG = os.path.join(OUT_DIR, "selection_log.csv")

CONTENT_CATS = {"NAMES", "WIKI", "CONV", "BOOK", "NEWS", "INDIC",
                "ALEXA", "BB", "UMANG", "DIGI"}
EMOTION_CATS = {"ANGER", "HAPPY", "FEAR", "SAD", "DISGUST", "SURPRISE"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------
def wav_duration(path: str):
    """Return (duration_seconds, sample_rate_hz, channels) or None if invalid."""
    try:
        with wave.open(path, "rb") as wf:
            fr = wf.getframerate()
            n = wf.getnframes()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            if fr <= 0 or n <= 0 or ch <= 0 or sw <= 0:
                return None
            return (n / float(fr), fr, ch)
    except Exception:
        return None


def file_ok(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    if os.path.getsize(path) <= 0:
        return False
    return True


def category_from_name(fname: str):
    """Extract category token from MAR_F_<CATEGORY>_<NNNN>.wav"""
    base = os.path.splitext(os.path.basename(fname))[0]  # MAR_F_ANGER_00123
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Phase 1 — load & inspect Rasa metadata
# ---------------------------------------------------------------------------
def load_rasa():
    if not os.path.isfile(RASA_METADATA_CSV):
        raise SystemExit(f"[ERROR] Rasa metadata not found: {RASA_METADATA_CSV}")

    samples = []
    with open(RASA_METADATA_CSV, "r", encoding="utf-8-sig", newline="") as f:
        # The file uses '|' as delimiter
        reader = csv.DictReader(f, delimiter="|")
        if not reader.fieldnames:
            raise SystemExit("[ERROR] Rasa CSV has no header.")
        # Find columns flexibly
        audio_col = None
        text_col = None
        for fn in reader.fieldnames:
            low = fn.strip().lower()
            if "audio" in low or "file" in low or "path" in low or "wav" in low:
                audio_col = fn
            if "text" in low or "transcript" in low or "sentence" in low:
                text_col = fn
            if audio_col and text_col:
                break
        if not audio_col or not text_col:
            raise SystemExit(
                f"[ERROR] Could not find audio/text columns in CSV header: "
                f"{reader.fieldnames}")

        for row in reader:
            raw_audio = (row.get(audio_col) or "").strip()
            raw_text = (row.get(text_col) or "").strip()
            if not raw_audio:
                continue
            fname = os.path.basename(raw_audio)
            # Remap to actual audio dir
            actual_path = os.path.join(RASA_AUDIO_DIR, fname)
            cat = category_from_name(fname)
            samples.append({
                "orig_path": raw_audio,
                "basename": fname,
                "path": actual_path,
                "text": raw_text,
                "category": cat,
                "duration": None,
                "sample_rate": None,
                "channels": None,
            })
    return samples, audio_col, text_col


# ---------------------------------------------------------------------------
# Phase 3/4 — validate by reading durations
# ---------------------------------------------------------------------------
def inspect_wavs(samples):
    """Fill durations; mark validity. Returns (valid, invalid)."""
    valid = []
    invalid = []
    for s in samples:
        if not file_ok(s["path"]):
            s["invalid_reason"] = "missing_or_empty"
            invalid.append(s)
            continue
        d = wav_duration(s["path"])
        if d is None:
            s["invalid_reason"] = "unreadable_or_bad_wav"
            invalid.append(s)
            continue
        dur, sr, ch = d
        s["duration"] = dur
        s["sample_rate"] = sr
        s["channels"] = ch
        if not s["text"]:
            s["invalid_reason"] = "empty_transcript"
            invalid.append(s)
            continue
        valid.append(s)
    return valid, invalid


# ---------------------------------------------------------------------------
# Phase 2 — stratified selection toward ~TARGET_SECONDS
# ---------------------------------------------------------------------------
def select_rasa(valid, target_seconds):
    random.seed(42)  # reproducible

    cats = {}
    for s in valid:
        cats.setdefault(s["category"], []).append(s)
    for c in cats:
        random.shuffle(cats[c])

    # Category order balanced between content and emotion, keep all.
    cat_names = list(cats.keys())
    cat_names.sort(key=lambda c: (
        0 if c in CONTENT_CATS else 1, len(cats[c]) * -1))

    selected = []
    total = 0.0
    min_per_cat = MIN_SAMPLES_PER_CAT

    # Pass 1: guarantee minimum per category (for diversity)
    for c in cat_names:
        bucket = cats[c]
        n_min = min(min_per_cat, len(bucket))
        for s in bucket[:n_min]:
            if total >= target_seconds:
                break
            selected.append(s)
            total += s["duration"]
            bucket.remove(s)
        if total >= target_seconds:
            break

    # Pass 2: fill remaining, interleaving categories to preserve diversity.
    # Keep the category with the largest remaining duration in front so the
    # distribution roughly mirrors the source dataset.
    while total < target_seconds:
        leftovers = [c for c in cat_names if cats[c]]
        if not leftovers:
            break
        best = max(leftovers, key=lambda c: sum(x["duration"] for x in cats[c]))
        s = cats[best].pop(0)
        new_total = total + s["duration"]
        if new_total <= target_seconds + 15:
            selected.append(s)
            total = new_total
            continue
        # This sample would overshoot; only take it if we cannot otherwise
        # reach the target range with the remaining pool.
        remaining = sum(x["duration"] for c in leftovers for x in cats[c])
        if total + remaining < target_seconds - 300:
            selected.append(s)
            total = new_total
    return selected, total


# ---------------------------------------------------------------------------
# Phase 4 — validate Cartesia
# ---------------------------------------------------------------------------
def load_cartesia():
    with open(CARTESIA_SENTENCES, "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]

    valid = []
    invalid = []
    for i, text in enumerate(sentences, start=1):
        fname = f"{i:04d}.wav"
        path = os.path.join(CARTESIA_OUTPUT_DIR, fname)
        rec = {"fname": fname,
               "text": text,
               "path": path,
               "duration": None,
               "sample_rate": None,
               "channels": None}
        if not file_ok(path):
            rec["invalid_reason"] = "missing_or_empty"
            invalid.append(rec)
            continue
        d = wav_duration(path)
        if d is None:
            rec["invalid_reason"] = "unreadable_or_bad_wav"
            invalid.append(rec)
            continue
        rec["duration"], rec["sample_rate"], rec["channels"] = d
        if not rec["text"]:
            rec["invalid_reason"] = "empty_transcript"
            invalid.append(rec)
            continue
        valid.append(rec)
    return valid, invalid


# ---------------------------------------------------------------------------
def main():
    log("=" * 60)
    log("PHASE 1: Inspecting Rasa dataset")
    log("=" * 60)
    rasa, audio_col, text_col = load_rasa()
    log(f"Total Rasa raw samples      : {len(rasa)}")
    log(f"CSV delimiter/columns       : '|', audio='{audio_col}', text='{text_col}'")

    log("")
    log("PHASE 3+4: Validating all samples (reading durations)...")
    rasa_valid, rasa_invalid = inspect_wavs(rasa)
    log(f"Rasa valid   : {len(rasa_valid)}")
    log(f"Rasa invalid : {len(rasa_invalid)}")

    # Duration distribution
    if rasa_valid:
        ds = [s["duration"] for s in rasa_valid]
        log(f"Rasa total duration        : {sum(ds)/3600:.2f} hours "
            f"({sum(ds):.0f} s)")
        log(f"Rasa avg/min/max(duration) : {statistics.mean(ds):.2f} / "
            f"{min(ds):.2f} / {max(ds):.2f} s")
    log(f"Rasa categories found       : {len(set(s['category'] for s in rasa_valid))}")
    cats_counts = {}
    for s in rasa_valid:
        cats_counts[s["category"]] = cats_counts.get(s["category"], 0) + 1
    for c in sorted(cats_counts):
        log(f"  - {c}: {cats_counts[c]}")
    speakers_cats = set(s["category"] for s in rasa_valid)
    log(f"Speakers (approx, from category tokens): {len(speakers_cats)}")

    # ---------------------------------------------------------------
    log("")
    log("=" * 60)
    log(f"PHASE 2: Selecting ~{TARGET_SECONDS}s ({TARGET_SECONDS/3600:.2f}h) Rasa audio")
    log("=" * 60)
    chosen, chosen_total = select_rasa(rasa_valid, TARGET_SECONDS)
    log(f"Selected Rasa samples      : {len(chosen)}")
    log(f"Selected Rasa duration     : {chosen_total:.0f} s "
        f"({chosen_total/3600:.2f} h)")
    sel_by_cat = {}
    for s in chosen:
        sel_by_cat[s["category"]] = sel_by_cat.get(s["category"], 0) + 1
    log("Selection by category:")
    for c in sorted(sel_by_cat):
        log(f"  - {c}: {sel_by_cat[c]}")

    # ---------------------------------------------------------------
    log("")
    log("=" * 60)
    log("PHASE 4: Validating Cartesia dataset")
    log("=" * 60)
    cart_valid, cart_invalid = load_cartesia()
    log(f"Cartesia samples found     : {len(cart_valid) + len(cart_invalid)}")
    log(f"Cartesia valid             : {len(cart_valid)}")
    log(f"Cartesia invalid           : {len(cart_invalid)}")
    if cart_valid:
        cds = [c["duration"] for c in cart_valid]
        log(f"Cartesia total duration    : {sum(cds):.0f} s ({sum(cds)/60:.2f} min)")
        log(f"Cartesia avg duration      : {statistics.mean(cds):.2f} s")
        log(f"Cartesia min duration      : {min(cds):.2f} s")
        log(f"Cartesia max duration      : {max(cds):.2f} s")
    for inv in cart_invalid:
        log(f"  INVALID: {inv['fname']} ({inv.get('invalid_reason')})")
    for inv in rasa_invalid[:20]:
        log(f"  RASA INVALID: {inv['basename']} ({inv.get('invalid_reason')})")
    if len(rasa_invalid) > 20:
        log(f"  ... and {len(rasa_invalid)-20} more invalid Rasa samples")

    # ---------------------------------------------------------------
    log("")
    log("=" * 60)
    log("PHASE 5+6: Copying audio & building metadata")
    log("=" * 60)
    os.makedirs(WAVS_DIR, exist_ok=True)

    meta_rows = []
    selection_rows = []

    # Rasa
    for seq, s in enumerate(chosen, start=1):
        out_name = f"rasa_{seq:06d}.wav"
        out_path = os.path.join(WAVS_DIR, out_name)
        shutil.copyfile(s["path"], out_path)
        reason = f"category={s['category']}, duration={s['duration']:.2f}s"
        meta_rows.append({
            "id": out_name[:-4],
            "text": s["text"],
            "audio_file": f"wavs/{out_name}",
            "duration_seconds": round(s["duration"], 6),
            "source": "rasa",
        })
        selection_rows.append({
            "source_file": s["basename"],
            "output_file": out_name,
            "text": s["text"],
            "duration_seconds": round(s["duration"], 6),
            "selection_reason": reason,
        })

    rasa_dur = sum(r["duration_seconds"] for r in meta_rows)

    # Cartesia
    cart_meta_rows = []
    for seq, c in enumerate(cart_valid, start=1):
        out_name = f"cartesia_{seq:06d}.wav"
        out_path = os.path.join(WAVS_DIR, out_name)
        shutil.copyfile(c["path"], out_path)
        cart_meta_rows.append({
            "id": out_name[:-4],
            "text": c["text"],
            "audio_file": f"wavs/{out_name}",
            "duration_seconds": round(c["duration"], 6),
            "source": "cartesia",
        })
    cart_dur = sum(r["duration_seconds"] for r in cart_meta_rows)

    all_meta = meta_rows + cart_meta_rows

    # Write metadata.csv
    with open(METADATA_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "text", "audio_file", "duration_seconds", "source"])
        writer.writeheader()
        writer.writerows(all_meta)
    log(f"metadata.csv written       : {len(all_meta)} rows -> {METADATA_CSV}")

    # Write selection_log.csv
    with open(SELECTION_LOG, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_file", "output_file", "text", "duration_seconds",
            "selection_reason"])
        writer.writeheader()
        writer.writerows(selection_rows)
    log(f"selection_log.csv written  : {len(selection_rows)} rows -> {SELECTION_LOG}")

    # ---------------------------------------------------------------
    log("")
    log("=" * 60)
    log("PHASE 7: Dataset summary")
    log("=" * 60)
    combined_dur = rasa_dur + cart_dur
    rasa_pct = (rasa_dur / combined_dur * 100) if combined_dur else 0.0
    cart_pct = (cart_dur / combined_dur * 100) if combined_dur else 0.0

    summary = []
    summary.append("Dataset: cartesia_rasa_3h_combined")
    summary.append("")
    summary.append("Rasa:")
    summary.append(f"  Source      : {RASA_METADATA_CSV}")
    summary.append(f"  Selected    : {len(chosen)} samples")
    summary.append(f"  Duration    : {rasa_dur:.0f} s ({rasa_dur/3600:.2f} h)")
    summary.append(f"  Speakers    : {len(set(s['category'] for s in chosen))} "
                   "(by filename category token)")
    summary.append(f"  Emotions    : "
                   f"{sum(1 for s in chosen if s['category'] in EMOTION_CATS)} "
                   "samples with emotion label")
    summary.append("")
    summary.append("Cartesia:")
    summary.append(f"  Source      : {CARTESIA_OUTPUT_DIR}+{CARTESIA_SENTENCES}")
    summary.append(f"  Samples     : {len(cart_valid)}")
    summary.append(f"  Duration    : {cart_dur:.0f} s ({cart_dur/60:.2f} min)")
    if cart_valid:
        summary.append(f"  Avg duration: {statistics.mean(c['duration'] for c in cart_valid):.2f} s")
    summary.append("")
    summary.append("Combined:")
    summary.append(f"  Total samples : {len(all_meta)}")
    summary.append(f"  Total duration: {combined_dur:.0f} s ({combined_dur/3600:.2f} h)")
    summary.append(f"  Rasa duration : {rasa_dur:.0f} s")
    summary.append(f"  Cartesia dur  : {cart_dur:.0f} s")
    summary.append(f"  Rasa %        : {rasa_pct:.1f}%")
    summary.append(f"  Cartesia %    : {cart_pct:.1f}%")
    summary.append("")
    summary.append("Selection methodology:")
    summary.append(
        "  - Rasa: metadata CSV (| delimited, audio_file|text) is the source of "
        "truth.")
    summary.append(
        "  - Durations computed by reading WAV headers.")
    summary.append(
        "  - All samples validated (exists, readable, duration>0, transcript "
        "non-empty).")
    summary.append(
        "  - Invalid/missing samples excluded and reported.")
    summary.append(
        "  - Stratified/random (seed=42) selection across the filename category "
        "tokens")
    summary.append(
        "    (content + emotion categories) to preserve diversity.")
    summary.append(
        "  - Selection targets ~10800 s; guaranteed a minimum of "
        f"{MIN_SAMPLES_PER_CAT} per category.")
    summary.append(
        "  - Audio copied under unique names rasa_NNNNNN.wav / "
        "cartesia_NNNNNN.wav; originals untouched.")
    summary.append("")
    summary.append(
        "Note: Syspin data was NOT used anywhere in this dataset.")
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    log(f"dataset_summary.txt written -> {SUMMARY_TXT}")

    # ---------------------------------------------------------------
    log("")
    log("=" * 60)
    log("FINAL VALIDATION")
    log("=" * 60)
    failures = 0

    # every metadata row has an audio file & file exists & readable
    for r in all_meta:
        p = os.path.join(OUT_DIR, r["audio_file"])
        if not file_ok(p):
            log(f"  FAIL: metadata row {r['id']} missing audio {p}")
            failures += 1
        else:
            d = wav_duration(p)
            if d is None:
                log(f"  FAIL: unreadable audio {r['id']}")
                failures += 1

    # duplicates check
    names = [r["audio_file"] for r in all_meta]
    if len(names) != len(set(names)):
        log("  FAIL: duplicate output filenames detected")
        failures += 1

    # empty transcript check
    for r in all_meta:
        if not r["text"].strip():
            log(f"  FAIL: empty transcript for {r['id']}")
            failures += 1

    # count actual files
    actual_files = set(os.listdir(WAVS_DIR))
    for r in all_meta:
        fname = os.path.basename(r["audio_file"])
        if fname not in actual_files:
            log(f"  FAIL: {fname} not in wavs dir")
            failures += 1

    log(f"Validation failures      : {failures}")
    log("")
    log("Composition checked:")
    log(f"  Rasa    = {len(chosen)} samples, {rasa_dur:.0f} s "
        f"(≈ {rasa_dur/3600:.2f} h)")
    log(f"  Cartesia= {len(cart_valid)} samples, {cart_dur:.0f} s")
    log(f"  Combined= {len(all_meta)} samples, {combined_dur:.0f} s")

    # ---------------------------------------------------------------
    log("")
    log("=" * 60)
    log("FINAL REPORT")
    log("=" * 60)
    log(f"1.  Rasa samples available       : {len(rasa_valid)}")
    log(f"2.  Rasa samples selected        : {len(chosen)}")
    log(f"3.  Rasa duration selected       : {rasa_dur:.0f} s ({rasa_dur/3600:.2f} h)")
    log(f"4.  Cartesia samples available   : {len(cart_valid)+len(cart_invalid)}")
    log(f"5.  Cartesia samples included    : {len(cart_valid)}")
    log(f"6.  Cartesia total duration      : {cart_dur:.0f} s")
    log(f"7.  Combined sample count        : {len(all_meta)}")
    log(f"8.  Combined total duration      : {combined_dur:.0f} s")
    log(f"9.  Rasa/Cartesia % by duration  : {rasa_pct:.1f}% / {cart_pct:.1f}%")
    log(f"10. Validation failures          : {failures}")
    log(f"11. Final dataset path           : {OUT_DIR}")
    log(f"12. Command used                 : bash scripts/run_prepare_dataset.sh")
    log("=" * 60)

    if failures:
        log(f"[ERROR] Dataset has {failures} validation failures. Review above.")
        return 1
    if len(cart_invalid) > 0:
        log(f"[WARN] {len(cart_invalid)} Cartesia samples were invalid/missing "
             f"and NOT fabricated. Review before fine-tuning.")
    log("[INFO] Dataset preparation complete. No fine-tuning was started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
PY_EOF

chmod +x "$SCRIPT_DIR/_prepare_combined_dataset.py"

echo ""
echo "[INFO] Running dataset preparation..."
echo ""
python3 "$SCRIPT_DIR/_prepare_combined_dataset.py"
EXIT_CODE=$?

echo ""
echo "[INFO] Preparation script finished with exit code: $EXIT_CODE"
# The generated script is not a source artifact; keep it for reproducibility.
exit $EXIT_CODE
