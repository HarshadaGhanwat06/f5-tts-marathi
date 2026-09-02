#!/usr/bin/env bash
# =============================================================================
# create_prono_vowel_dataset.sh
#
# Generate a PRONUNCIATION-FOCUSED Marathi dataset for fine-tuning the acoustic
# pronunciation of vowel/matra patterns:
#
#   1. ॅ   (CANDRA E matra, dependent form of ॲ)   -> कॅ
#   2. ॉ   (CANDRA O matra, dependent form of ऑ)   -> कॉ
#   3. ँ   (candrabindu anusvara)                  -> कँ
#   4. ॉं  (CANDRA O + candrabindu)                -> कॉं
#
# NOTES on orthography:
#   - ॲ/ऑ are the INDEPENDENT vowel glyphs and cannot attach to a consonant;
#     the requested outputs in the spec (कॅ कॉ कँ कॉं) use the dependent
#     matra forms ॅ/ॉ.  Those exact exemplars are what this script emits.
#   - All outputs are Devanagari only (no Latin/IPA/roman), no punctuation,
#     one utterance per line, no duplicates.
#
# For EVERY consonant the same systematic structure is generated (16 lines):
#   isolated  :  कॅ
#   preceded  :  का कॅ
#   followed  :  कॅ का
#   repeated  :  कॅ कॅ
# (…and the same for कॉ / कँ / कॉं), then repeated for ख, ग, ...
#
# Consonant set (33, exactly as specified, incl. ङ ञ ण):
#   क ख ग घ ङ / च छ ज झ ञ / ट ठ ड ढ ण / त थ द ध न /
#   प फ ब भ म / य र ल व / श ष स ह
#
# Total: 33 x 4 patterns x 4 contexts = 528 unique samples.
# Balance: every consonant gets exactly 16 samples (4 per pattern x 4 contexts).
#
# The transcribed lines are synthesized with Cartesia (Arushi voice), so the
# transcripts above are EXACTLY what is sent to the TTS.
#
# Outputs (server):
#   cartesia_ws/prono_vowel_dataset/sentences.txt   (528 lines, one per line)
#   cartesia_ws/prono_vowel_dataset/output/0001.wav... (per line)
#   cartesia_ws/prono_vowel_dataset/metadata.csv     (+ failed_samples.csv)
#   cartesia_ws/generate_prono_vowel.py              (generated runner)
#
# Usage (run on the SERVER):
#   export CARTESIA_VOICE_ID=95d51f79-c397-46f9-b49a-23763d3eaa2d
#   bash scripts/create_prono_vowel_dataset.sh           # first 5 (test)
#   bash scripts/create_prono_vowel_dataset.sh --full    # all 528
#   bash scripts/create_prono_vowel_dataset.sh --start 20 --limit 10
#   bash scripts/create_prono_vowel_dataset.sh --rebuild-metadata  # reread WAVs -> metadata.csv
#
# Then: bash scripts/prepare_dataset_v5.sh   (mix into fine-tuning dataset)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="$REPO_ROOT/cartesia_ws/prono_vowel_dataset"
OUTPUT_DIR="$DATASET_DIR/output"
SENTENCES_FILE="$DATASET_DIR/sentences.txt"

WS_URL="${CARTESIA_WS_URL:-wss://vectortts.markytics.ai/tts/ws/}"
MODEL_ID="${CARTESIA_MODEL_ID:-sonic-3}"
LANGUAGE_CODE="${CARTESIA_LANGUAGE_CODE:-en}"
SAMPLE_RATE_HZ="${CARTESIA_SAMPLE_RATE:-16000}"
CLIENT_NAME="${CARTESIA_CLIENT_NAME:-f5tts-cartesia-pronov}"
SPEED="${CARTESIA_SPEED:-0.8}"
VOICE_ID="${CARTESIA_VOICE_ID:-}"

if [[ -z "$VOICE_ID" ]]; then
    echo "[ERROR] CARTESIA_VOICE_ID is not set." >&2
    echo "        export CARTESIA_VOICE_ID=95d51f79-c397-46f9-b49a-23763d3eaa2d" >&2
    exit 1
fi

mkdir -p "$DATASET_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Pronunciation-focused vowel/matra dataset (ॅ ॉ ँ ॉं)"
echo "============================================================"
echo "Dataset dir  : $DATASET_DIR"
echo "Voice        : $VOICE_ID"
echo "speaking_rate: $SPEED"

# ---------------------------------------------------------------------------
# 1. Render sentences.txt (systematic, balanced, 528 lines)
# ---------------------------------------------------------------------------
python3 - "$SENTENCES_FILE" <<'PY'
import sys

out_path = sys.argv[1]

# Exactly the consonants from the spec (includes ङ ञ ण).
consonants = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"

# 4 patterns. ॅ/ॉ are the dependent-matra forms of ॲ/ऑ (per the spec's own
# examples कॅ कॉ कँ कॉं).
patterns = [
    ("candra_e",            "\u0945"),          # ॅ
    ("candra_o",            "\u0949"),          # ॉ
    ("candrabindu",         "\u0901"),          # ँ
    ("candra_o_anusvara",   "\u0949\u0901"),    # ॉं
]

# 4 contexts (isolated / preceded / followed / repeated).
# "का" is the normal Marathi syllable used in the spec's examples.
def contexts(target):
    return [
        target,          # isolated : कॅ
        f"का {target}",  # preceded : का कॅ
        f"{target} का",  # followed : कॅ का
        f"{target} {target}",  # repeated : कॅ कॅ
    ]

lines = []
seen = set()
per_consonant = []
expected_per_consonant = len(patterns) * 4  # 16

for c in consonants:
    cons_lines = []
    for name, mat in patterns:
        target = c + mat
        for ctx in contexts(target):
            if ctx not in seen:
                seen.add(ctx)
                lines.append(ctx)
                cons_lines.append(ctx)
    per_consonant.append((c, len(cons_lines)))

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    for s in lines:
        f.write(s + "\n")

print(f"Consonants              : {len(consonants)}")
print(f"Patterns per consonant  : {len(patterns)} patterns x 4 contexts")
print(f"Total unique lines      : {len(lines)}")
print(f"Expected                : {len(consonants) * len(patterns) * 4}")
print("Per-consonant counts    :",
      {c: n for c, n in per_consonant})
print("Written                 :", out_path)
PY

echo ""
echo "[1] sentences.txt (first 8):"
head -8 "$SENTENCES_FILE"
echo "    ..."
echo "    Sample around कॉं block:"
grep -n "कॉं" "$SENTENCES_FILE" | head -4 || true
TOTAL_LINES="$(wc -l < "$SENTENCES_FILE")"
echo "    Total lines: $TOTAL_LINES"

# ---------------------------------------------------------------------------
# 2. Check websockets dependency (non-fatal)
# ---------------------------------------------------------------------------
if ! python3 -c "import websockets" 2>/dev/null; then
    echo "[WARN] 'websockets' missing. Installing..."
    if [[ -f "$REPO_ROOT/cartesia_ws/requirements.txt" ]]; then
        python3 -m pip install --quiet -r "$REPO_ROOT/cartesia_ws/requirements.txt"
    else
        python3 -m pip install --quiet websockets
    fi
fi

# ---------------------------------------------------------------------------
# 3. Generate the dedicated runner (same WS pipeline as the other runners)
# ---------------------------------------------------------------------------
cat > "$REPO_ROOT/cartesia_ws/generate_prono_vowel.py" <<'PY_EOF'
#!/usr/bin/env python3
"""
Pronunciation vowel/matra dataset generation (ॅ/ॉ/ँ/ॉं).

Same WebSocket synthesis pipeline as generate_dataset.py / generate_candra_e.py
but pointed at cartesia_ws/prono_vowel_dataset.

Usage:
    python generate_prono_vowel.py [--start N] [--limit M] [--overwrite]
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import statistics
import sys
import uuid
import wave

import websockets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "prono_vowel_dataset")
SENTENCES_FILE = os.path.join(DATASET_DIR, "sentences.txt")
OUTPUT_DIR = os.path.join(DATASET_DIR, "output")
METADATA_CSV = os.path.join(DATASET_DIR, "metadata.csv")
FAILED_CSV = os.path.join(DATASET_DIR, "failed_samples.csv")

WS_URL = os.environ.get("CARTESIA_WS_URL", "wss://vectortts.markytics.ai/tts/ws/")
MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3")
LANGUAGE_CODE = os.environ.get("CARTESIA_LANGUAGE_CODE", "en")
SAMPLE_RATE_HZ = int(os.environ.get("CARTESIA_SAMPLE_RATE", "16000"))
CLIENT_NAME = os.environ.get("CARTESIA_CLIENT_NAME", "f5tts-cartesia-pronov")
SPEED = float(os.environ.get("CARTESIA_SPEED", "0.8"))
VOICE_ID = os.environ.get("CARTESIA_VOICE_ID")
if not VOICE_ID:
    raise SystemExit("[ERROR] CARTESIA_VOICE_ID environment variable is not set.")

MAX_RETRIES = 3
RECV_TIMEOUT = 60


async def synthesize_text(text: str, output_path: str) -> dict:
    call_id = str(uuid.uuid4())
    query = (
        f"?tts_engine=cartesia"
        f"&client_name={CLIENT_NAME}"
        f"&call_id={call_id}"
        f"&voice_name={VOICE_ID}"
        f"&model_id={MODEL_ID}"
        f"&language_code={LANGUAGE_CODE}"
        f"&sample_rate_hz={SAMPLE_RATE_HZ}"
        f"&generator_sample_hz={SAMPLE_RATE_HZ}"
        f"&speaking_rate={SPEED}"
    )
    uri = f"{WS_URL}{query}"
    audio_chunks = []
    total_bytes = 0
    stream_msg_count = 0
    metadata = None
    received_complete = False

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"text": text}))
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
            except asyncio.TimeoutError:
                raise RuntimeError("Timed out waiting for server frames")
            frame = json.loads(raw)
            status = frame.get("status")
            if status == "metadata":
                metadata = frame
            elif status == "streaming":
                audio_b64 = frame.get("audio")
                if not audio_b64:
                    raise RuntimeError(f"streaming message missing audio: {frame!r}")
                chunk = base64.b64decode(audio_b64)
                audio_chunks.append(chunk)
                total_bytes += len(chunk)
                stream_msg_count += 1
            elif status == "complete":
                received_complete = True
                break
            elif status == "error":
                raise RuntimeError(frame.get("message") or "server error")

    if not received_complete:
        raise RuntimeError("WebSocket closed before 'complete'")
    if stream_msg_count == 0:
        raise RuntimeError("No audio chunks received")
    if metadata is None:
        raise RuntimeError("Missing audio metadata")

    sample_rate = metadata.get("sample_rate_hz")
    channels = metadata.get("channels")
    if sample_rate is None or channels is None:
        raise RuntimeError("Incomplete metadata")

    codec = (metadata.get("audio_codec") or metadata.get("encoding") or
             "LINEAR16").upper()
    if "LINEAR16" not in codec and "PCM" not in codec:
        raise RuntimeError(f"Unsupported codec: {codec}")

    audio = b"".join(audio_chunks)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio)

    duration = len(audio) / (int(channels) * 2 * int(sample_rate))
    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(channels),
        "duration_seconds": round(duration, 6),
        "total_bytes": total_bytes,
    }


async def process_sample(idx: int, total: int, text: str, overwrite: bool) -> dict:
    fid = f"{idx:04d}"
    out_wav = os.path.join(OUTPUT_DIR, f"{fid}.wav")

    if not overwrite and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 0:
        try:
            with wave.open(out_wav, "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate())
                sr = wf.getframerate()
            print(f"[{idx}/{total}] SKIP (exists)")
            return {"id": fid, "text": text, "audio_file": f"{fid}.wav",
                    "duration_seconds": round(dur, 6), "sample_rate_hz": sr,
                    "total_bytes": os.path.getsize(out_wav), "status": "skip"}
        except Exception:
            print(f"[{idx}/{total}] existing file invalid, regenerating")

    print(f"[{idx}/{total}] {text}")
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            meta = await synthesize_text(text, out_wav)
            if not os.path.isfile(out_wav) or os.path.getsize(out_wav) <= 0:
                raise RuntimeError("WAV validation failed")
            print(f"[{idx}/{total}] OK {meta['duration_seconds']:.2f}s")
            meta.update({"id": fid, "text": text,
                         "audio_file": f"{fid}.wav", "status": "success"})
            return meta
        except Exception as exc:
            last_error = exc
            print(f"[{idx}/{total}] attempt {attempt} failed: {exc}")
            if os.path.exists(out_wav):
                try:
                    os.remove(out_wav)
                except Exception:
                    pass
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.0)

    print(f"[{idx}/{total}] FAILED: {text} ({last_error})")
    return {"id": fid, "text": text, "audio_file": f"{fid}.wav",
            "duration_seconds": None, "sample_rate_hz": None,
            "total_bytes": 0, "status": "failed", "error": str(last_error)}


async def main() -> int:
    parser = argparse.ArgumentParser(description="ॅ/ॉ/ँ/ॉं pronunciation dataset")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-metadata", action="store_true",
                        help="Rewrite metadata.csv from sentences.txt + existing "
                             "WAVs (backfills rows lost by an interrupted run).")
    args = parser.parse_args()

    if not os.path.isfile(SENTENCES_FILE):
        print(f"[ERROR] sentences.txt not found: {SENTENCES_FILE}", file=sys.stderr)
        return 1

    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    total = len(sentences)
    print(f"Total sentences: {total}")

    start = max(1, args.start)
    if start > total:
        print(f"[ERROR] --start {args.start} beyond {total}", file=sys.stderr)
        return 1
    end = total if args.limit is None else min(total, start + args.limit - 1)
    print(f"Processing {start}..{end}")

    # ------------------------------------------------------------------
    # --rebuild-metadata mode: reconstruct metadata.csv from existing WAVs
    # ------------------------------------------------------------------
    if args.rebuild_metadata:
        rows = []
        missing = 0
        for idx, text in enumerate(sentences, start=1):
            fid = f"{idx:04d}"
            wav = os.path.join(OUTPUT_DIR, f"{fid}.wav")
            if not os.path.isfile(wav) or os.path.getsize(wav) <= 0:
                missing += 1
                continue
            try:
                with wave.open(wav, "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    n = wf.getnframes()
                dur = n / float(sr) if sr else 0.0
                rows.append({"id": fid, "text": text,
                             "audio_file": f"{fid}.wav",
                             "duration_seconds": round(dur, 6),
                             "source": "prono_vowel"})
            except Exception as exc:
                print(f"[REBUILD] corrupt wav {fid}: {exc}")
                missing += 1
        with open(METADATA_CSV, "w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=[
                "id", "text", "audio_file", "duration_seconds", "source"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[REBUILD] metadata.csv rewritten: {len(rows)} rows "
              f"(expected {total}; missing/corrupt: {missing}) -> {METADATA_CSV}")
        return 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    durations = []
    n_success = n_skip = n_failed = 0
    failed_rows = []

    for idx in range(start, end + 1):
        res = await process_sample(idx, total, sentences[idx - 1], args.overwrite)
        results.append(res)
        if res["status"] == "success":
            n_success += 1
            durations.append(res["duration_seconds"] or 0.0)
        elif res["status"] == "skip":
            n_skip += 1
            durations.append(res["duration_seconds"] or 0.0)
        else:
            n_failed += 1
            failed_rows.append({"id": res["id"], "text": res["text"],
                                "error": res.get("error", "")})

    # Backfill skipped rows too, so an interrupted run never drops rows from
    # metadata.csv. Dedupe against ids already present.
    existing_ids = set()
    if os.path.isfile(METADATA_CSV):
        with open(METADATA_CSV, "r", encoding="utf-8", newline="") as cf:
            for row in csv.DictReader(cf):
                existing_ids.add(row.get("id", ""))
    new_rows = [r for r in results if r["status"] in ("success", "skip")
                and r["id"] not in existing_ids]
    if new_rows:
        write_header = not os.path.isfile(METADATA_CSV)
        with open(METADATA_CSV, "a", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=[
                "id", "text", "audio_file", "duration_seconds", "source"])
            if write_header:
                writer.writeheader()
            for r in new_rows:
                writer.writerow({"id": r["id"], "text": r["text"],
                                 "audio_file": r["audio_file"],
                                 "duration_seconds": r["duration_seconds"],
                                 "source": "prono_vowel"})
        print(f"[INFO] metadata.csv updated (+{len(new_rows)} backfilled): {METADATA_CSV}")

    with open(FAILED_CSV, "w", encoding="utf-8", newline="") as ff:
        writer = csv.DictWriter(ff, fieldnames=["id", "text", "error"])
        writer.writeheader()
        for fr in failed_rows:
            writer.writerow(fr)

    print("=" * 60)
    print(f"Successful : {n_success}")
    print(f"Skipped    : {n_skip}")
    print(f"Failed     : {n_failed}")
    if durations:
        print(f"Total audio: {sum(durations)/60:.2f} min | "
              f"avg {statistics.mean(durations):.2f}s")
    print("=" * 60)
    return 2 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
PY_EOF
chmod +x "$REPO_ROOT/cartesia_ws/generate_prono_vowel.py"
echo "[INFO] Created: $REPO_ROOT/cartesia_ws/generate_prono_vowel.py"

# ---------------------------------------------------------------------------
# 4. Run mode
# ---------------------------------------------------------------------------
cd "$REPO_ROOT/cartesia_ws"
export CARTESIA_WS_URL CARTESIA_MODEL_ID CARTESIA_LANGUAGE_CODE
export CARTESIA_SAMPLE_RATE CARTESIA_CLIENT_NAME CARTESIA_SPEED CARTESIA_VOICE_ID

MODE="${1:-test}"
case "$MODE" in
    --full)
        echo "[INFO] Generating ALL prono_vowel sentences (528)."
        python3 generate_prono_vowel.py
        ;;
    --start)
        START_LINE="${2:?Usage: --start <LINE> [--limit <N>]}"
        LIMIT_OPT=""
        if [[ "${3:-}" == "--limit" ]]; then
            LIMIT_OPT="--limit ${4:?--limit requires a number}"
        fi
        echo "[INFO] Generating starting at line $START_LINE $LIMIT_OPT"
        python3 generate_prono_vowel.py --start "$START_LINE" $LIMIT_OPT
        ;;
    --rebuild-metadata)
        echo "[INFO] Rebuilding metadata.csv from existing WAVs (no synthesis)."
        python3 generate_prono_vowel.py --rebuild-metadata
        ;;
    *)
        echo "[INFO] Running 5-SAMPLE TEST mode."
        python3 generate_prono_vowel.py --start 1 --limit 5
        ;;
esac
EXIT_CODE=$?
echo ""
echo "[INFO] create_prono_vowel_dataset.sh finished with exit code $EXIT_CODE"
exit $EXIT_CODE