#!/usr/bin/env bash
#
# run_cartesia_dataset.sh
#
# Batch Cartesia synthesis for a Marathi dataset (Ubuntu).
#
# Generates the batch script `generate_dataset.py` (which reuses the already
# working WebSocket logic from websocket_test.py) and then runs the required
# first test on 5 sentences only.
#
# NOTE: This script is self-contained and does NOT re-run the earlier
#       run_cartesia_ws_test.sh script. It creates a fresh Python file.
#
# Usage:
#   ./scripts/run_cartesia_dataset.sh                  # first 5 sentences
#   ./scripts/run_cartesia_dataset.sh --full           # all sentences
#   ./scripts/run_cartesia_dataset.sh --start 201 --limit 5   # lines 201-205
#   ./scripts/run_cartesia_dataset.sh --validate
#
# Default behaviour: synthesizes the first 5 sentences (test mode).
# --start <LINE> [--limit <N>] starts at a given line using the TRUE line
#   number for filenames (e.g. line 201 -> 0201.wav).
#
# Required environment variable:
#   CARTESIA_VOICE_ID   - Cartesia voice ID (Arushi voice)
#
# Optional environment variables:
#   CARTESIA_WS_URL        - override endpoint (default internal endpoint)
#   CARTESIA_MODEL_ID      - override model (default: sonic-3)
#   CARTESIA_LANGUAGE_CODE - override language (default: en)
#   CARTESIA_SAMPLE_RATE   - override sample rate (default: 16000)
#   CARTESIA_CLIENT_NAME   - override client name (default: f5tts-cartesia-batch)
#   CARTESIA_SPEED         - Arushi synthesis speed multiplier (speaking_rate, default: 0.8)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WS_URL="${CARTESIA_WS_URL:-wss://vectortts.markytics.ai/tts/ws/}"
MODEL_ID="${CARTESIA_MODEL_ID:-sonic-3}"
LANGUAGE_CODE="${CARTESIA_LANGUAGE_CODE:-en}"
SAMPLE_RATE_HZ="${CARTESIA_SAMPLE_RATE:-16000}"
GENERATOR_SAMPLE_HZ="${SAMPLE_RATE_HZ}"
CLIENT_NAME="${CARTESIA_CLIENT_NAME:-f5tts-cartesia-batch}"
SPEED="${CARTESIA_SPEED:-0.8}"

# Voice ID required; from environment only.
if [[ -z "${CARTESIA_VOICE_ID:-}" ]]; then
    echo "[ERROR] CARTESIA_VOICE_ID environment variable is not set." >&2
    echo "        Export it before running, e.g.:" >&2
    echo "          export CARTESIA_VOICE_ID=95d51f79-c397-46f9-b49a-23763d3eaa2d" >&2
    exit 1
fi

TEST_DIR="$REPO_ROOT/cartesia_ws"
DATASET_DIR="$TEST_DIR/cartesia_dataset"
OUTPUT_DIR="$TEST_DIR/output"

echo "[INFO] Project root       : $REPO_ROOT"
echo "[INFO] Cartesia WS dir    : $TEST_DIR"
echo "[INFO] Dataset dir        : $DATASET_DIR"
echo "[INFO] Output dir         : $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
mkdir -p "$TEST_DIR"
mkdir -p "$DATASET_DIR"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Generate the batch Python script
# ---------------------------------------------------------------------------
cat > "$TEST_DIR/generate_dataset.py" <<'PY_EOF'
#!/usr/bin/env python3
"""
Batch Cartesia synthesis for a Marathi dataset.

Reuses the working WebSocket logic from websocket_test.py. Reads sentences.txt
line by line, synthesizes exactly one WAV per non-empty line, writes
output/0001.wav ... output/NNNN.wav, and records metadata.csv and
failed_samples.csv.

Usage:
    python generate_dataset.py [--limit N] [--overwrite]
    python generate_dataset.py --validate
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


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCES_FILE = os.path.join(BASE_DIR, "cartesia_dataset", "sentences.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
METADATA_CSV = os.path.join(BASE_DIR, "cartesia_dataset", "metadata.csv")
FAILED_CSV = os.path.join(BASE_DIR, "cartesia_dataset", "failed_samples.csv")

WS_URL = os.environ.get(
    "CARTESIA_WS_URL", "wss://vectortts.markytics.ai/tts/ws/")
MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3")
LANGUAGE_CODE = os.environ.get("CARTESIA_LANGUAGE_CODE", "en")
SAMPLE_RATE_HZ = int(os.environ.get("CARTESIA_SAMPLE_RATE", "16000"))
GENERATOR_SAMPLE_HZ = int(os.environ.get(
    "CARTESIA_GENERATOR_SAMPLE_RATE", str(SAMPLE_RATE_HZ)))
CLIENT_NAME = os.environ.get("CARTESIA_CLIENT_NAME", "f5tts-cartesia-batch")

SPEED = float(os.environ.get("CARTESIA_SPEED", "0.8"))

VOICE_ID = os.environ.get("CARTESIA_VOICE_ID")
if not VOICE_ID:
    raise SystemExit(
        "[ERROR] CARTESIA_VOICE_ID environment variable is not set.")

MAX_RETRIES = 3
RECV_TIMEOUT = 60


def log(msg: str) -> None:
    print(msg, flush=True)


def log_recv(msg: str) -> None:
    print(f"[RECV] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Reusable single-transcript synthesis (port of websocket_test.py logic)
# ---------------------------------------------------------------------------
async def synthesize_text(text: str, output_path: str) -> dict:
    """
    Synthesize a single transcript and write exactly one WAV.

    Returns a metadata dict:
        sample_rate_hz, channels, duration_seconds, total_bytes, audio_codec

    Raises an exception on any failure (no output WAV is written).
    """
    call_id = str(uuid.uuid4())

    query = (
        f"?tts_engine=cartesia"
        f"&client_name={CLIENT_NAME}"
        f"&call_id={call_id}"
        f"&voice_name={VOICE_ID}"
        f"&model_id={MODEL_ID}"
        f"&language_code={LANGUAGE_CODE}"
        f"&sample_rate_hz={SAMPLE_RATE_HZ}"
        f"&generator_sample_hz={GENERATOR_SAMPLE_HZ}"
        f"&speaking_rate={SPEED}"
    )
    uri = f"{WS_URL}{query}"
    audio_chunks = []
    total_bytes = 0
    stream_msg_count = 0
    metadata = None
    received_complete = False
    server_error = None

    async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"text": text}))

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
            except asyncio.TimeoutError:
                raise RuntimeError("Timed out waiting for server frames")

            try:
                frame = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid JSON received: {raw!r} ({exc})")

            if not isinstance(frame, dict) or "status" not in frame:
                raise RuntimeError(f"Message missing 'status' field: {frame!r}")

            status = frame["status"]

            if status == "processing":
                pass

            elif status == "metadata":
                metadata = frame
                log_recv(f"status=metadata encoding={frame.get('encoding')} "
                         f"channels={frame.get('channels')} "
                         f"sample_rate_hz={frame.get('sample_rate_hz')} "
                         f"audio_codec={frame.get('audio_codec')}")

            elif status == "synthesizing":
                pass

            elif status == "streaming":
                audio_b64 = frame.get("audio")
                if not isinstance(audio_b64, str) or not audio_b64:
                    raise RuntimeError(
                        f"streaming message missing/invalid base64 'audio': {frame!r}")
                try:
                    chunk = base64.b64decode(audio_b64)
                except Exception as exc:
                    raise RuntimeError(f"Failed to base64-decode audio: {exc}")
                audio_chunks.append(chunk)
                total_bytes += len(chunk)
                stream_msg_count += 1

            elif status == "complete":
                received_complete = True
                break

            elif status == "error":
                server_error = frame.get("message") or "unknown server error"
                raise RuntimeError(f"Server error: {server_error}")

            else:
                log_recv(f"status=unknown{status} frame={frame!r}")

    if not received_complete:
        raise RuntimeError("WebSocket closed before 'complete' was received")

    if stream_msg_count == 0:
        raise RuntimeError("Received zero streaming audio chunks; nothing to save")

    if metadata is None:
        raise RuntimeError(
            "Missing audio metadata; cannot determine sample rate/channels")

    sample_rate = metadata.get("sample_rate_hz")
    channels = metadata.get("channels")
    if sample_rate is None or channels is None:
        raise RuntimeError(f"Metadata incomplete for WAV writing: {metadata!r}")

    codec = (metadata.get("audio_codec") or
             metadata.get("encoding") or "LINEAR16").upper()
    if "LINEAR16" not in codec and "PCM" not in codec:
        raise RuntimeError(f"Unsupported codec for WAV writing: {codec}")

    sampwidth = 2
    audio = b"".join(audio_chunks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(sampwidth)
        wf.setframerate(int(sample_rate))
        wf.writeframes(audio)

    duration = len(audio) / (int(channels) * sampwidth * int(sample_rate))

    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(channels),
        "duration_seconds": round(duration, 6),
        "total_bytes": total_bytes,
        "audio_codec": f"{sampwidth * 8}-bit PCM",
    }


# ---------------------------------------------------------------------------
# WAV validation
# ---------------------------------------------------------------------------
def validate_wav(path: str, meta: dict) -> bool:
    """Return True if the WAV file passes validation."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) <= 0:
        return False
    try:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.getnframes()
    except Exception:
        return False
    if channels <= 0 or sampwidth <= 0 or framerate <= 0:
        return False
    if frames <= 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Single sample processing with retries
# ---------------------------------------------------------------------------
async def process_sample(idx: int, total: int, text: str, overwrite: bool) -> dict:
    """Synthesize one sentence. Returns a result dict."""
    file_id = f"{idx:04d}"
    out_wav = os.path.join(OUTPUT_DIR, f"{file_id}.wav")

    # Resume support
    if not overwrite and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 0:
        log(f"[{idx}/{total}] SKIP - already exists")
        try:
            with wave.open(out_wav, "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate()) \
                    if wf.getframerate() else 0.0
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sz = os.path.getsize(out_wav)
            return {
                "id": file_id, "text": text, "audio_file": f"{file_id}.wav",
                "duration_seconds": round(dur, 6), "sample_rate_hz": sr,
                "total_bytes": sz, "status": "skip",
            }
        except Exception:
            log(f"[{idx}/{total}] Existing file invalid; will regenerate: {out_wav}")

    log(f"[{idx}/{total}] Synthesizing:")
    log(text)
    log("")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"[{idx}/{total}] Receiving audio chunks... (attempt {attempt})")
            meta = await synthesize_text(text, out_wav)
            if not validate_wav(out_wav, meta):
                raise RuntimeError(f"WAV validation failed for {out_wav}")

            log(f"[{idx}/{total}] Complete")
            log(f"[{idx}/{total}] Saved: output/{file_id}.wav")
            log(f"[{idx}/{total}] Duration: {meta['duration_seconds']:.2f} sec")
            log(f"[{idx}/{total}] Bytes: {meta['total_bytes']}")

            meta.update({
                "id": file_id, "text": text, "audio_file": f"{file_id}.wav",
                "status": "success",
            })
            return meta

        except Exception as exc:
            last_error = exc
            log(f"[{idx}/{total}] Attempt {attempt} failed: {exc}")
            if os.path.exists(out_wav):
                try:
                    os.remove(out_wav)
                except Exception:
                    pass
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.0)

    # Failed after all retries
    log(f"[{idx}/{total}] FAILED")
    log(f"Text: {text}")
    log(f"Reason: {last_error}")
    return {
        "id": file_id, "text": text, "audio_file": f"{file_id}.wav",
        "duration_seconds": None, "sample_rate_hz": None,
        "total_bytes": 0, "status": "failed", "error": str(last_error),
    }


# ---------------------------------------------------------------------------
# Dataset validation (Phase 4)
# ---------------------------------------------------------------------------
def validate_dataset(sentences: list) -> int:
    """Validate the generated dataset (audio + text). Returns exit code."""
    issues = []

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Text validation
    # ------------------------------------------------------------
    print("\n[Text Validation]")
    text_issues = []

    if not sentences:
        text_issues.append("sentences.txt contains no non-empty lines")

    # Empty transcripts
    empties = [i + 1 for i, s in enumerate(sentences) if s == ""]
    if empties:
        text_issues.append(f"Empty transcripts at lines: {empties[:20]}{'...' if len(empties) > 20 else ''}")

    # Duplicate transcripts
    seen = {}
    dups = []
    for i, s in enumerate(sentences, start=1):
        if s in seen:
            dups.append((i, s))
        else:
            seen[s] = i
    if dups:
        msg = "Duplicate transcripts: " + ", ".join(
            f"line {i} -> first at line {seen[s]}" for i, s in dups[:20])
        if len(dups) > 20:
            msg += f" (and {len(dups) - 20} more)"
        text_issues.append(msg)

    # Invalid characters: anything not valid Devanagari/whitespace/punct
    import unicodedata
    invalid_chars = {}
    for i, s in enumerate(sentences, start=1):
        for ch in s:
            if ch.isspace():
                continue
            name = unicodedata.name(ch, "")
            is_marathi = (0x0900 <= ord(ch) <= 0x097F)
            is_ascii_print = (0x20 <= ord(ch) <= 0x7E)
            is_general_punct = (0x2000 <= ord(ch) <= 0x206F)
            if not (is_marathi or is_ascii_print or is_general_punct):
                invalid_chars.setdefault(ord(ch), []).append(i)
    if invalid_chars:
        details = ", ".join(
            f"U+{cp:04X} ({chr(cp)}) at lines {chrs[:10]}{'...' if len(chrs) > 10 else ''}"
            for cp, chrs in list(invalid_chars.items())[:20])
        text_issues.append(f"Invalid characters found: {details}")

    # ------------------------------------------------------------
    # 2. Audio / metadata consistency
    # ------------------------------------------------------------
    print("\n[Audio Validation]")

    # Load metadata.csv if present
    metadata_rows = {}
    expected_ids = {f"{i:04d}" for i in range(1, len(sentences) + 1)}
    if os.path.isfile(METADATA_CSV):
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_rows[row["id"]] = row
    else:
        text_issues.append(f"metadata.csv not found: {METADATA_CSV}")

    # Missing audio files
    missing_audio = [f"{i:04d}.wav" for i in range(1, len(sentences) + 1)
                     if not os.path.isfile(os.path.join(OUTPUT_DIR, f"{i:04d}.wav"))]
    if missing_audio:
        audio_issues = missing_audio
    else:
        audio_issues = []

    # Missing metadata rows
    missing_meta = sorted(expected_ids - set(metadata_rows.keys()))
    if missing_meta:
        text_issues.append(f"Missing metadata rows for ids: {missing_meta[:20]}{'...' if len(missing_meta) > 20 else ''}")

    audio_problems = []
    durations = []
    leading_tail_probs = 0
    for i in range(1, len(sentences) + 1):
        fid = f"{i:04d}"
        wav_path = os.path.join(OUTPUT_DIR, f"{fid}.wav")
        if not os.path.isfile(wav_path):
            continue

        try:
            with wave.open(wav_path, "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
            dur = nframes / float(framerate) if framerate else 0.0
            durations.append(dur)
            size = os.path.getsize(wav_path)

            if size <= 0:
                audio_problems.append(f"{fid}: empty file (size 0)")
            if channels <= 0 or sampwidth <= 0 or framerate <= 0:
                audio_problems.append(
                    f"{fid}: invalid WAV props (ch={channels} sw={sampwidth} sr={framerate})")
            if dur <= 0:
                audio_problems.append(f"{fid}: zero duration")

            # Rough silence detection (first/last 5% RMS)
            if sampwidth == 2 and framerate > 0 and nframes > 0:
                with wave.open(wav_path, "rb") as wf:
                    raw = wf.readframes(nframes)
                import array
                samples = array.array("h")
                samples.frombytes(raw[: min(len(raw), 2 * 160000)])  # 10s cap
                n = len(samples)
                if n > 0:
                    window = max(1, n // 20)
                    head = samples[:window]
                    tail = samples[-window:] if window > 0 else samples
                    def rms(a):
                        if len(a) == 0:
                            return 0.0
                        s = sum((x / 32768.0) ** 2 for x in a)
                        return (s / len(a)) ** 0.5
                    h = rms(head)
                    t = rms(tail)
                    if h < 0.001:
                        audio_problems.append(f"{fid}: excessive leading silence")
                        leading_tail_probs += 1
                    if t < 0.001:
                        audio_problems.append(f"{fid}: excessive trailing silence")
                        leading_tail_probs += 1

        except Exception as exc:
            audio_problems.append(f"{fid}: cannot open WAV ({exc})")

    # Abnormally short/long
    short = [f"{i:04d}" for i, d in enumerate(durations, start=1) if d < 5.0]
    long_ = [f"{i:04d}" for i, d in enumerate(durations, start=1) if d > 12.0]
    if short:
        audio_problems.append(
            f"{len(short)} samples < 5.0s (e.g. {short[:10]}{'...' if len(short) > 10 else ''})")
    if long_:
        audio_problems.append(
            f"{len(long_)} samples > 12.0s (e.g. {long_[:10]}{'...' if len(long_) > 10 else ''})")

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------
    if text_issues:
        print("\n  TEXT ISSUES:")
        for t in text_issues:
            print(f"    - {t}")

    if audio_issues:
        print("\n  MISSING AUDIO FILES:")
        print(f"    - {len(audio_issues)} files missing: {audio_issues[:20]}{'...' if len(audio_issues) > 20 else ''}")

    if audio_problems:
        print("\n  AUDIO ISSUES:")
        for a in audio_problems:
            print(f"    - {a}")

    total_dur = sum(durations)
    print("\n" + "-" * 60)
    print(f"WAV files present   : {len(durations)}")
    print(f"Missing WAV files   : {len(audio_issues)}")
    print(f"Text issues         : {len(text_issues)}")
    print(f"Audio issues        : {len(audio_problems)}")
    if durations:
        print(f"Total duration      : {total_dur/60:.2f} minutes")
        print(f"Min duration        : {min(durations):.2f} s")
        print(f"Max duration        : {max(durations):.2f} s")
        print(f"Avg duration        : {statistics.mean(durations):.2f} s")
        # Sample rate / channels consistency
        srs = set()
        chs = set()
        for i in range(1, len(sentences) + 1):
            p = os.path.join(OUTPUT_DIR, f"{i:04d}.wav")
            if os.path.isfile(p):
                try:
                    with wave.open(p, "rb") as wf:
                        srs.add(wf.getframerate())
                        chs.add(wf.getnchannels())
                except Exception:
                    pass
        print(f"Sample rates present: {sorted(srs)}")
        print(f"Channels present    : {sorted(chs)}")
    print("\n" + "=" * 60)

    # Targets to manually confirm via listening
    print("\n[Manual Check Required - listen to these]")
    print("  Confirm correct pronunciation of target words embedded in Marathi:")
    print("  office, drawing, meeting, project, computer")
    print("  (e.g. ड्रॉइंग, ब्यूटिफुल, ऑफिस, मीटिंग, प्रोजेक्ट)")

    n_issues = len(audio_issues) + len(text_issues) + len(audio_problems)
    print("\nVALIDATION RESULT:", "PASS (no issues)" if n_issues == 0 else f"FAIL ({n_issues} issue(s))")
    return 0 if n_issues == 0 else 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> int:
    parser = argparse.ArgumentParser(description="Cartesia batch synthesis")
    parser.add_argument("--start", type=int, default=1,
                        help="1-based line number to START from (default: 1). "
                             "Filenames use the true line number.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of sentences to synthesize from --start "
                             "(default: all remaining)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate existing WAV files")
    parser.add_argument("--validate", action="store_true",
                        help="Validate the generated dataset without synthesizing")
    args = parser.parse_args()

    if not os.path.isfile(SENTENCES_FILE):
        print(f"[ERROR] sentences.txt not found: {SENTENCES_FILE}", file=sys.stderr)
        return 1

    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    total = len(sentences)
    print(f"Total non-empty sentences found: {total}")
    if total != 200:
        print(f"[WARN] Expected 200 sentences but found {total}.", file=sys.stderr)

    # ------------------------------------------------------------------
    # --validate mode : check the generated dataset, no synthesis
    # ------------------------------------------------------------------
    if args.validate:
        return await validate_dataset(sentences)

    # Determine the slice to process, preserving TRUE line numbers.
    start = max(1, args.start)
    if start > total:
        print(f"[ERROR] --start {args.start} is beyond the {total} sentences.",
              file=sys.stderr)
        return 1
    if args.limit is not None:
        end = min(total, start + args.limit - 1)
    else:
        end = total
    print(f"Processing lines {start}..{end} (of {total})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    total_duration = 0.0
    durations = []
    n_success = 0
    n_skip = 0
    n_failed = 0
    failed_rows = []

    run_total = len(sentences)

    for idx in range(start, end + 1):
        text = sentences[idx - 1]
        result = await process_sample(idx, total, text, args.overwrite)
        results.append(result)

        if result["status"] == "success":
            n_success += 1
            d = result["duration_seconds"] or 0.0
            total_duration += d
            durations.append(d)
        elif result["status"] == "skip":
            n_skip += 1
            d = result["duration_seconds"] or 0.0
            total_duration += d
            durations.append(d)
        else:
            n_failed += 1
            failed_rows.append({
                "id": result["id"],
                "text": result["text"],
                "error": result.get("error", ""),
            })

    # ------------------------------------------------------------------
    # Write metadata.csv
    # ------------------------------------------------------------------
    new_rows = [r for r in results if r["status"] == "success"]
    if new_rows:
        write_header = not os.path.isfile(METADATA_CSV)
        with open(METADATA_CSV, "a", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(
                cf, fieldnames=[
                    "id", "text", "audio_file", "duration_seconds",
                    "sample_rate_hz", "total_bytes", "status"])
            if write_header:
                writer.writeheader()
            for r in new_rows:
                writer.writerow({
                    "id": r["id"],
                    "text": r["text"],
                    "audio_file": r["audio_file"],
                    "duration_seconds": r["duration_seconds"],
                    "sample_rate_hz": r["sample_rate_hz"],
                    "total_bytes": r["total_bytes"],
                    "status": "success",
                })
        print(f"\n[INFO] metadata.csv updated: {METADATA_CSV}")

    # ------------------------------------------------------------------
    # Write failed_samples.csv
    # ------------------------------------------------------------------
    with open(FAILED_CSV, "w", encoding="utf-8", newline="") as ff:
        writer = csv.DictWriter(ff, fieldnames=["id", "text", "error"])
        writer.writeheader()
        for fr in failed_rows:
            writer.writerow(fr)
    print(f"[INFO] failed_samples.csv written: {FAILED_CSV}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"Input sentences: {total}")
    print(f"Successful     : {n_success}")
    print(f"Skipped        : {n_skip}")
    print(f"Failed         : {n_failed}")
    if durations:
        print(f"Total audio duration: {total_duration/60:.2f} minutes")
        print(f"Average duration    : {statistics.mean(durations):.2f} seconds")
        print(f"Minimum duration    : {min(durations):.2f} seconds")
        print(f"Maximum duration    : {max(durations):.2f} seconds")

        out_of_range = [d for d in durations if d < 5.0 or d > 12.0]
        if out_of_range:
            print("\n[WARN] Samples outside expected range (5-12s):")
            for i, d in zip(
                    [r["id"] for r in results if r["status"] in ("success", "skip")],
                    durations):
                if d < 5.0 or d > 12.0:
                    print(f"  {i}: {d:.2f} sec")
        else:
            print("\nAll generated durations are within expected range (5-12s).")
    else:
        print("No audio generated.")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    if n_failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    asyncio.run(main())
PY_EOF

chmod +x "$TEST_DIR/generate_dataset.py"

echo ""
echo "[INFO] Created: $TEST_DIR/generate_dataset.py"
echo ""

# ---------------------------------------------------------------------------
# Generate the dataset-combination Python script (Phase 5)
# ---------------------------------------------------------------------------
cat > "$TEST_DIR/combine_dataset.py" <<'COMBINE_EOF'
#!/usr/bin/env python3
"""
Phase 5 - Combine the existing Marathi dataset with the Cartesia code-mixed
dataset into a single reproducible F5-TTS fine-tuning dataset.

Existing Marathi dataset  (majority)  : datasets/syspin_marathi_female/metadata_5h.csv
Cartesia code-mixed dataset (targeted): cartesia_ws/cartesia_dataset/metadata.csv

Outputs:
  datasets/combined/metadata_combined.csv       (F5-TTS format  audio_file|text)
  datasets/combined/metadata_cartesia_f5tts.csv (Cartesia component only)
  datasets/combined/dataset_composition.json    (version + counts + provenance)
  datasets/combined/cartesia/<NNNN>.wav         (resampled Cartesia audio copies)

No existing Marathi sample is modified or lost.
"""

import csv
import json
import os
import re
import sys
import wave
from collections import Counter
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np
    import librosa
    AUDIO_OK = True
except Exception as e:  # pragma: no cover
    AUDIO_OK = False
    print(f"[WARN] audio libs unavailable: {e}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

SYSPIN_META = os.path.join(
    REPO_ROOT, "datasets", "syspin_marathi_female", "metadata_5h.csv")
CARTESIA_META = os.path.join(
    BASE_DIR, "cartesia_dataset", "metadata.csv")
CARTESIA_AUDIO_DIR = os.path.join(BASE_DIR, "output")

VOCAB_FILE = os.path.join(
    REPO_ROOT, "models", "openbible-marathi", "vocab.txt")

COMBINED_DIR = os.path.join(REPO_ROOT, "datasets", "combined")
CARTESIA_COPY_DIR = os.path.join(COMBINED_DIR, "cartesia")
COMBINED_META = os.path.join(COMBINED_DIR, "metadata_combined.csv")
CARTESIA_META_F5TTS = os.path.join(COMBINED_DIR, "metadata_cartesia_f5tts.csv")
COMPOSITION_JSON = os.path.join(COMBINED_DIR, "dataset_composition.json")

DATASET_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# Reuse the same normalization as prepare_syspin.py for consistency
# ---------------------------------------------------------------------------
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
REPLACEMENTS = {
    "ॅ": "े", "ँ": "ं", "ऑ": "ओ", "ॲ": "अ",
    "ङ": "न", "ॊ": "ो", ";": "",
}


def normalize(text: str) -> str:
    text = text.translate(DEVANAGARI_DIGITS)
    text = text.translate(QUOTE_MAP)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    for char in "{}|/":
        text = text.replace(char, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def read_f5tts_metadata(path: str) -> list:
    """Read F5-TTS metadata: audio_file|text (pipe-delimited, no header)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            # split only on the first pipe (text may contain '|' removed, but be safe)
            parts = line.split("|", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed line in {path}: {line!r}")
            audio_file, text = parts
            rows.append({"audio_file": audio_file.strip(), "text": text.strip()})
    return rows


def read_cartesia_metadata(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def detect_syspin_audio_format(rows) -> dict:
    """Detect the dominant sample rate / channels from existing SYSPIN audio."""
    sr_counter = Counter()
    ch_counter = Counter()
    count = 0
    for r in rows:
        p = r["audio_file"]
        if count >= 20:
            break
        if not os.path.isfile(p):
            continue
        try:
            with wave.open(p, "rb") as wf:
                sr_counter[wf.getframerate()] += 1
                ch_counter[wf.getnchannels()] += 1
                count += 1
        except Exception:
            continue
    if not sr_counter:
        raise RuntimeError(
            "Could not read any SYSPIN WAV to detect audio format. "
            "Check that the extracted audio exists.")
    return {
        "sample_rate": sr_counter.most_common(1)[0][0],
        "channels": ch_counter.most_common(1)[0][0],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    # 1. Load existing SYSPIN subset
    if not os.path.isfile(SYSPIN_META):
        print(f"[ERROR] Existing Marathi metadata not found: {SYSPIN_META}", file=sys.stderr)
        return 1
    syspin_rows = read_f5tts_metadata(SYSPIN_META)
    print(f"Existing Marathi dataset (metadata_5h.csv): {len(syspin_rows)} rows")

    # 2. Load Cartesia code-mixed dataset
    if not os.path.isfile(CARTESIA_META):
        print(f"[ERROR] Cartesia metadata not found: {CARTESIA_META}", file=sys.stderr)
        return 1
    cart_rows = read_cartesia_metadata(CARTESIA_META)
    print(f"Cartesia code-mixed dataset (metadata.csv): {len(cart_rows)} rows")

    # 3. Audio format consistency
    print("\n[Audio format]")
    target_fmt = detect_syspin_audio_format(syspin_rows)
    print(f"  Existing (SYSPIN) dominant format: {target_fmt['sample_rate']} Hz, "
          f"{target_fmt['channels']} ch")

    os.makedirs(CARTESIA_COPY_DIR, exist_ok=True)

    # 4. Build Cartesia F5-TTS rows (normalize text, resample audio, abs paths)
    print("\n[Building Cartesia component]")
    cart_f5tts_rows = []
    missing = []
    oov_counter = Counter()

    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        vocab = set(line.rstrip("\n\r") for line in f)

    for row in cart_rows:
        fid = row.get("id", "").strip()
        text = row.get("text", "").strip()
        norm_text = normalize(text)

        src_wav = os.path.join(CARTESIA_AUDIO_DIR, f"{fid}.wav")
        dst_wav = os.path.join(CARTESIA_COPY_DIR, f"{fid}.wav")

        if not os.path.isfile(src_wav):
            missing.append(fid)
            continue

        # Resample Cartesia audio to the SYSPIN sample rate for consistency
        if AUDIO_OK:
            try:
                y, sr = librosa.load(src_wav, sr=None, mono=True)
                if sr != target_fmt["sample_rate"]:
                    y = librosa.resample(
                        y, orig_sr=sr, target_sr=target_fmt["sample_rate"])
                sf.write(dst_wav, y, target_fmt["sample_rate"], subtype="PCM_16")
            except Exception as exc:
                print(f"[WARN] Could not resample {fid}.wav: {exc}; keeping original")
                # fall back to a direct copy of original
                import shutil
                shutil.copyfile(src_wav, dst_wav)
        else:
            import shutil
            shutil.copyfile(src_wav, dst_wav)

        for ch in norm_text:
            if ch not in vocab:
                oov_counter[ch] += 1

        cart_f5tts_rows.append({
            "id": fid,
            "audio_file": os.path.abspath(dst_wav),
            "text": norm_text,
        })

    print(f"  Cartesia rows included  : {len(cart_f5tts_rows)}")
    print(f"  Cartesia rows missing   : {len(missing)}")
    if missing:
        print(f"  Missing wav ids: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    # 5. OOV check on Cartesia text
    print("\n[OOV check against vocab]")
    if oov_counter:
        print(f"  OOV characters in Cartesia text: {len(oov_counter)}")
        for ch, c in oov_counter.most_common():
            print(f"    {repr(ch)} U+{ord(ch):04X} : {c}")
    else:
        print("  ZERO OOV characters in Cartesia text")

    # 6. Write component CSV (F5-TTS format)
    with open(CARTESIA_META_F5TTS, "w", encoding="utf-8", newline="") as f:
        for r in cart_f5tts_rows:
            f.write(f"{r['audio_file']}|{r['text']}\n")
    print(f"\n[INFO] Wrote Cartesia F5-TTS component: {CARTESIA_META_F5TTS}")

    # 7. Build combined metadata (existing + cartesia), no loss of existing rows
    combined_rows = []
    for r in syspin_rows:
        r2 = dict(r)
        # Existing rows keep their absolute path and original (already-normalized) text
        combined_rows.append({"audio_file": r2["audio_file"], "text": r2["text"]})
    for r in cart_f5tts_rows:
        combined_rows.append({"audio_file": r["audio_file"], "text": r["text"]})

    with open(COMBINED_META, "w", encoding="utf-8", newline="") as f:
        for r in combined_rows:
            f.write(f"{r['audio_file']}|{r['text']}\n")
    print(f"[INFO] Wrote combined metadata: {COMBINED_META}")

    # 8. Composition report + versioning
    composition = {
        "dataset_version": DATASET_VERSION,
        "created_note": "Existing Marathi majority + targeted Cartesia code-mixed.",
        "existing_marathi": {
            "source": "datasets/syspin_marathi_female/metadata_5h.csv",
            "count": len(syspin_rows),
        },
        "cartesia_code_mixed": {
            "source": "cartesia_ws/cartesia_dataset/metadata.csv",
            "requested": len(cart_rows),
            "included": len(cart_f5tts_rows),
            "missing_audio": len(missing),
        },
        "combined": {
            "count": len(combined_rows),
            "existing_percent": round(100 * len(syspin_rows) / len(combined_rows), 2)
                if combined_rows else 0.0,
            "cartesia_percent": round(100 * len(cart_f5tts_rows) / len(combined_rows), 2)
                if combined_rows else 0.0,
        },
        "audio_format": {
            "sample_rate_hz": target_fmt["sample_rate"],
            "channels": target_fmt["channels"],
            "note": "Cartesia audio resampled to match the dominant existing format.",
        },
        "oov_characters": len(oov_counter),
    }

    with open(COMPOSITION_JSON, "w", encoding="utf-8") as f:
        json.dump(composition, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote composition report: {COMPOSITION_JSON}")

    # 9. Summary
    print("\n" + "=" * 60)
    print("COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"Existing Marathi : {len(syspin_rows)}")
    print(f"Cartesia code-mix: {len(cart_f5tts_rows)}")
    print(f"Combined total   : {len(combined_rows)}")
    if combined_rows:
        print(f"Existing ratio   : {composition['combined']['existing_percent']}%")
        print(f"Cartesia ratio   : {composition['combined']['cartesia_percent']}%")
    print(f"OOV characters   : {len(oov_counter)}")
    print(f"Audio format     : {target_fmt['sample_rate']} Hz, "
          f"{target_fmt['channels']} ch")
    print(f"Dataset version  : {DATASET_VERSION}")
    print("=" * 60)
    print("\nNext step: prepare the combined dataset with F5-TTS")
    print("  python f5tts/lib/python3.12/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py"
          "  datasets/combined/metadata_combined.csv  f5tts/data/SYSPIN_Marathi_Combined_5h  --workers 32")
    print("\nExisting Marathi samples are preserved (no loss).")
    return 0 if not missing else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
COMBINE_EOF

chmod +x "$TEST_DIR/combine_dataset.py"
echo "[INFO] Created: $TEST_DIR/combine_dataset.py"
echo ""

# ---------------------------------------------------------------------------
# Check websockets dependency (non-fatal; error surfaces on run)
# ---------------------------------------------------------------------------
if ! python3 -c "import websockets" 2>/dev/null; then
    echo "[WARN] 'websockets' not found. Installing from requirements.txt..."
    python3 -m pip install --quiet -r "$TEST_DIR/requirements.txt"
fi

# ---------------------------------------------------------------------------
# Determine run mode
# ---------------------------------------------------------------------------
cd "$TEST_DIR"

MODE="${1:-test}"
case "$MODE" in
    --full)
        echo "[INFO] Running FULL generation (all sentences)."
        python3 generate_dataset.py
        ;;
    --validate)
        echo "[INFO] Running DATASET VALIDATION only (no synthesis)."
        python3 generate_dataset.py --validate
        ;;
    --combine)
        echo "[INFO] Running DATASET COMBINATION (SYSPIN + Cartesia)."
        python3 combine_dataset.py
        ;;
    --start)
        # Usage: bash scripts/run_cartesia_dataset.sh --start <LINE> [--limit <N>]
        START_LINE="${2:?Usage: --start <LINE> [--limit <N>]}"
        LIMIT_OPT=""
        if [[ "${3:-}" == "--limit" ]]; then
            LIMIT_OPT="--limit ${4:?--limit requires a number}"
        fi
        echo "[INFO] Running generation starting at line $START_LINE ${LIMIT_OPT:+with $LIMIT_OPT}."
        echo "[INFO] speaking_rate = $SPEED | voice = $CARTESIA_VOICE_ID"
        python3 generate_dataset.py --start "$START_LINE" $LIMIT_OPT
        ;;
    *)
        echo "[INFO] Running 5-SAMPLE TEST mode."
        echo "[INFO] This synthesizes only the first 5 sentences."
        echo ""
        python3 generate_dataset.py --start 1 --limit 5
        ;;
esac

EXIT_CODE=$?
echo ""
echo "[INFO] Script finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
