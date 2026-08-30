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
#   ./scripts/run_cartesia_dataset.sh [--full]
#
# Default behaviour: synthesizes the first 5 sentences (test mode).
# Add --full to process all sentences instead (only after 5-sample test review).
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
# Main
# ---------------------------------------------------------------------------
async def main() -> int:
    parser = argparse.ArgumentParser(description="Cartesia batch synthesis")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only synthesize the first N sentences (test mode)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate existing WAV files")
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

    if args.limit is not None:
        sentences = sentences[:args.limit]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    total_duration = 0.0
    durations = []
    n_success = 0
    n_skip = 0
    n_failed = 0
    failed_rows = []

    run_total = len(sentences)

    for idx, text in enumerate(sentences, start=1):
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
# Check websockets dependency (non-fatal; error surfaces on run)
# ---------------------------------------------------------------------------
if ! python3 -c "import websockets" 2>/dev/null; then
    echo "[WARN] 'websockets' not found. Installing from requirements.txt..."
    python3 -m pip install --quiet -r "$TEST_DIR/requirements.txt"
fi

# ---------------------------------------------------------------------------
# Determine run mode (default: 5-sample test)
# ---------------------------------------------------------------------------
cd "$TEST_DIR"

if [[ "${1:-}" == "--full" ]]; then
    echo "[INFO] Running FULL generation (all sentences)."
    python3 generate_dataset.py
else
    echo "[INFO] Running 5-SAMPLE TEST mode."
    echo "[INFO] This synthesizes only the first 5 sentences."
    echo ""
    python3 generate_dataset.py --limit 5
fi

EXIT_CODE=$?
echo ""
echo "[INFO] Script finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
