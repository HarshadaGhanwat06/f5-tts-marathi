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
#   RASA_METADATA_CSV    - Rasa metadata CSV (default: /root/f5-tts-marathi/rasa_female_emotions/rasa_train_metadata.csv)
#   CARTESIA_OUTPUT_DIR  - Cartesia output dir (default: /root/f5-tts-marathi/cartesia_ws/output)
#   CARTESIA_SENTENCES   - Cartesia sentences.txt (default: /root/f5-tts-marathi/cartesia_ws/cartesia_dataset/sentences.txt)
#   TARGET_SECONDS       - Target Rasa duration (default: 10800)
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

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/root/f5-tts-marathi/cartesia_ws/cartesia_rasa_3h_combined")
WAVS_DIR = os.path.join(OUT_DIR, "wavs")
METADATA_CSV = os.path.join(OUT_DIR, "metadata.csv")
SUMMARY_TXT = os.path.join(OUT_DIR, "dataset_summary.txt")
SELECTION_LOG = os.path.join(OUT_DIR, "selection_log.csv")

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
# Phase 2 — proportional stratified duration-based selection.
#
# Selects ~TARGET_SECONDS of Rasa audio such that each category's selected
# duration mirrors its proportion of the original valid dataset. Selection is
# based on AUDIO DURATION, not sample count. Small categories are still
# represented, and emotional categories keep their original weight. No
# category is starved to feed WIKI.
# ---------------------------------------------------------------------------
def select_rasa(valid, target_seconds):
    seed = 42  # reproducible
    random.seed(seed)

    # Group valid samples by category.
    cats = {}
    for s in valid:
        cats.setdefault(s["category"], []).append(s)
    for c in cats:
        random.shuffle(cats[c])

    cat_names = sorted(cats.keys())
    total_valid_dur = sum(s["duration"] for s in valid)
    if total_valid_dur <= 0:
        return [], 0.0

    # Proportion per category = category duration / total valid duration.
    cat_dur = {c: sum(s["duration"] for s in cats[c]) for c in cat_names}

    # Allocate target duration per category proportionally.
    targets = {c: (cat_dur[c] / total_valid_dur) * target_seconds
               for c in cat_names}

    selected = []
    selected_dur_by_cat = {c: 0.0 for c in cat_names}
    buckets = {c: list(cats[c]) for c in cat_names}

    # Per-category round-robin pass: fill each category toward its target
    # duration, moving back and forth so no category is starved.
    while True:
        progressed = False
        for c in cat_names:
            if not buckets[c]:
                continue
            # Stop filling a category if it has reached its target.
            if selected_dur_by_cat[c] >= targets[c]:
                continue
            s = buckets[c][0]
            new_cat_total = selected_dur_by_cat[c] + s["duration"]
            # Take the sample unless it overshoots this category's target.
            if new_cat_total <= targets[c] + 2.0:
                buckets[c].pop(0)
                selected.append(s)
                selected_dur_by_cat[c] = new_cat_total
                progressed = True

        # Update running total.
        total = sum(selected_dur_by_cat.values())

        # Stop when we reach the target range, or nothing progressed, or
        # everything is exhausted.
        if total >= target_seconds - 5:
            break
        if not progressed:
            break

    total = sum(selected_dur_by_cat.values())

    # Final adjustment: if we are meaningfully below target but there is
    # unused audio available, fill the shortfall proportionally across
    # categories using their remaining audio, respecting each category's
    # original proportion as much as possible.
    if total < target_seconds - 60:
        remaining = {c: sum(x["duration"] for x in buckets[c]) for c in cat_names}
        total_remaining = sum(remaining.values())
        if total_remaining > 0:
            # Desired fill per category proportional to original distribution.
            round_robin = sorted(cat_names,
                                 key=lambda c: -remaining[c])
            while total < target_seconds and total_remaining > 0:
                progressed = False
                for c in round_robin:
                    if not buckets[c]:
                        continue
                    s = buckets[c][0]
                    new_total = total + s["duration"]
                    # Avoid severe overshoot; drop to next category if too big.
                    if new_total <= target_seconds + 10:
                        buckets[c].pop(0)
                        selected.append(s)
                        selected_dur_by_cat[c] += s["duration"]
                        remaining[c] -= s["duration"]
                        total_remaining -= s["duration"]
                        total = new_total
                        progressed = True
                    if total >= target_seconds - 5:
                        break
                if not progressed:
                    break

    total = sum(selected_dur_by_cat.values())
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
    log(f"PHASE 2: Selecting ~{TARGET_SECONDS}s ({TARGET_SECONDS/3600:.2f}h) "
        f"Rasa audio (proportional stratified)")
    log("=" * 60)
    seed = 42
    chosen, chosen_total = select_rasa(rasa_valid, TARGET_SECONDS)
    log(f"Selected Rasa samples      : {len(chosen)}")
    log(f"Selected Rasa duration     : {chosen_total:.0f} s "
        f"({chosen_total/3600:.2f} h)")
    log(f"Selection seed             : {seed}")

    # Original distribution by sample count & by duration
    orig_counts = {}
    orig_durs = {}
    for s in rasa_valid:
        orig_counts[s["category"]] = orig_counts.get(s["category"], 0) + 1
        orig_durs[s["category"]] = orig_durs.get(s["category"], 0.0) + s["duration"]

    sel_counts = {}
    sel_durs = {}
    for s in chosen:
        sel_counts[s["category"]] = sel_counts.get(s["category"], 0) + 1
        sel_durs[s["category"]] = sel_durs.get(s["category"], 0.0) + s["duration"]

    total_valid_dur = sum(orig_durs.values()) or 1.0
    sel_dur_total = sum(sel_durs.values()) or 1.0

    log("")
    log("Original category distribution:")
    for c in sorted(orig_counts):
        log(f"  {c}: {orig_counts[c]} samples, {orig_durs[c]:.0f}s "
            f"({orig_durs[c]/total_valid_dur*100:.1f}%)")

    log("")
    log("Selected category distribution:")
    for c in sorted(sel_counts):
        log(f"  {c}: {sel_counts[c]} samples, {sel_durs[c]:.0f}s "
            f"({sel_durs[c]/sel_dur_total*100:.1f}%)")

    log("")
    log("Category | Original % | Selected % | Selected Duration")
    for c in sorted(set(orig_counts) | set(sel_counts)):
        op = (orig_durs.get(c, 0.0) / total_valid_dur * 100)
        sp = (sel_durs.get(c, 0.0) / sel_dur_total * 100)
        log(f"  {c:<10} | {op:9.1f}% | {sp:10.1f}% | {sel_durs.get(c,0.0):8.0f}s")

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
            "category": s["category"],
            "selection_seed": seed,
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
            "selection_reason", "category", "selection_seed"])
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
        "  - Proportional stratified selection by AUDIO DURATION (seed=42):")
    summary.append(
        "      category target duration = (category duration / total valid ")
    summary.append(
        "                                duration) x target seconds.")
    summary.append(
        "  - Each category is filled toward its proportional target; the ")
    summary.append(
        "    original dataset distribution remains the primary guide so no")
    summary.append(
        "    single category (e.g. WIKI) dominates the subset.")
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
