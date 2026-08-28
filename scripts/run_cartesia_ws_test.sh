#!/usr/bin/env bash
#
# run_cartesia_ws_test.sh
#
# Minimal Cartesia TTS WebSocket end-to-end test (Ubuntu).
#
# This script:
#   1. Creates the isolated cartesia_ws/ test dir (websocket_test.py + requirements.txt).
#   2. Installs/verifies dependencies.
#   3. Runs the test with a generated unique call_id.
#
# Scope: proves the smallest possible path only:
#   text -> Internal TTS WebSocket -> tts_engine=cartesia -> sonic-3
#        -> JSON streaming messages -> base64 audio -> WAV file.
#
# It does NOT touch emotion_layer/ or the existing F5-TTS pipeline.
#
# Usage:
#   ./scripts/run_cartesia_ws_test.sh
#
# Required environment variables:
#   CARTESIA_VOICE_ID   - Cartesia voice ID (e.g. 95d51f79-...-23763d3eaa2d)
#   TEXT_TO_SYNTHESIZE  - (optional) override the default test sentence
#
# Optional environment variables:
#   CARTESIA_WS_URL        - override endpoint (default: internal company endpoint)
#   CARTESIA_MODEL_ID      - override model (default: sonic-3)
#   CARTESIA_LANGUAGE_CODE - override language (default: en)
#   CARTESIA_SAMPLE_RATE   - override sample rate (default: 16000)
#   CARTESIA_CLIENT_NAME   - override client name (default: f5tts-cartesia-test)
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (with env var overrides)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WS_URL="${CARTESIA_WS_URL:-wss://vectortts.markytics.ai/tts/ws/}"
MODEL_ID="${CARTESIA_MODEL_ID:-sonic-3}"
LANGUAGE_CODE="${CARTESIA_LANGUAGE_CODE:-en}"
SAMPLE_RATE_HZ="${CARTESIA_SAMPLE_RATE:-16000}"
GENERATOR_SAMPLE_HZ="${SAMPLE_RATE_HZ}"
CLIENT_NAME="${CARTESIA_CLIENT_NAME:-f5tts-cartesia-test}"

DEFAULT_TEXT="hi drawing khup sundar ahe."
TEXT_TO_SYNTHESIZE="${TEXT_TO_SYNTHESIZE:-$DEFAULT_TEXT}"

# Voice ID is required and comes from the environment ONLY (no hardcoding).
if [[ -z "${CARTESIA_VOICE_ID:-}" ]]; then
    echo "[ERROR] CARTESIA_VOICE_ID environment variable is not set." >&2
    echo "        Export it before running, e.g.:" >&2
    echo "          export CARTESIA_VOICE_ID=95d51f79-c397-46f9-b49a-23763d3eaa2d" >&2
    exit 1
fi

TEST_DIR="$REPO_ROOT/cartesia_ws"
OUTPUT_DIR="$TEST_DIR/output"

# ---------------------------------------------------------------------------
# 1. Create project structure
# ---------------------------------------------------------------------------
echo "[INFO] Creating project structure at $TEST_DIR"
mkdir -p "$TEST_DIR"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 2. Write requirements.txt (websockets client only; aiohttp already present)
# ---------------------------------------------------------------------------
cat > "$TEST_DIR/requirements.txt" <<'REQ_EOF'
# Minimal deps for the Cartesia TTS WebSocket test.
# aiohttp is already in the project environment and provides a WebSocket client,
# so we only add a pure client if needed. Here we use the 'websockets' library
# as a self-contained standard WebSocket client.
websockets>=12.0
REQ_EOF

# ---------------------------------------------------------------------------
# 3. Write websocket_test.py
# ---------------------------------------------------------------------------
cat > "$TEST_DIR/websocket_test.py" <<'PY_EOF'
#!/usr/bin/env python3
"""
Minimal Cartesia TTS WebSocket client test.

Connects to the internal TTS WebSocket endpoint with tts_engine=cartesia,
sends one text synthesis request, receives JSON frames, decodes base64 audio
chunks from 'streaming' messages, combines them, and writes a WAV file.

Scope-restricted: this is a standalone proof-of-path and does NOT integrate
with or modify the existing F5-TTS pipeline.
"""

import argparse
import base64
import json
import os
import sys
import uuid
import wave

import asyncio

import websockets


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_recv(msg: str) -> None:
    print(f"[RECV] {msg}")


async def main() -> int:
    # ------------------------------------------------------------------
    # Config from environment / CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Cartesia TTS WebSocket test")
    parser.add_argument("--url", default=os.environ.get(
        "CARTESIA_WS_URL", "wss://vectortts.markytics.ai/tts/ws/"))
    parser.add_argument("--voice-id", required=True,
                        help="Cartesia voice ID (from CARTESIA_VOICE_ID)")
    parser.add_argument("--client-name", default=os.environ.get(
        "CARTESIA_CLIENT_NAME", "f5tts-cartesia-test"))
    parser.add_argument("--call-id", default=None,
                        help="Unique call id; generated if not provided")
    parser.add_argument("--model-id", default=os.environ.get(
        "CARTESIA_MODEL_ID", "sonic-3"))
    parser.add_argument("--language-code", default=os.environ.get(
        "CARTESIA_LANGUAGE_CODE", "en"))
    parser.add_argument("--sample-rate-hz", type=int, default=int(os.environ.get(
        "CARTESIA_SAMPLE_RATE", "16000")))
    parser.add_argument("--generator-sample-hz", type=int, default=None)
    parser.add_argument("--text", default="hi drawing khup sundar ahe.",
                        help="Text to synthesize")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the output WAV (default: ./output)")
    args = parser.parse_args()

    if args.generator_sample_hz is None:
        args.generator_sample_hz = args.sample_rate_hz

    if args.call_id is None:
        args.call_id = str(uuid.uuid4())

    out_dir = os.path.abspath(args.out_dir or os.path.join(
        os.path.dirname(__file__), "output"))
    os.makedirs(out_dir, exist_ok=True)

    # Build the WebSocket URI with query parameters.
    query = (
        f"?tts_engine=cartesia"
        f"&client_name={args.client_name}"
        f"&call_id={args.call_id}"
        f"&voice_name={args.voice_id}"
        f"&model_id={args.model_id}"
        f"&language_code={args.language_code}"
        f"&sample_rate_hz={args.sample_rate_hz}"
        f"&generator_sample_hz={args.generator_sample_hz}"
    )
    uri = f"{args.url}{query}"

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    audio_chunks = []          # raw decoded bytes
    total_bytes = 0            # decoded audio byte count
    stream_msg_count = 0
    metadata = None            # last metadata frame
    received_complete = False
    received_error = None
    text_accepted = False

    log_info(f"Connecting to Cartesia TTS WebSocket: {args.url}")
    log_info(f"call_id: {args.call_id}")
    log_info(f"voice: {args.voice_id} | model: {args.model_id} | lang: {args.language_code}")
    log_info(f"sample_rate: {args.sample_rate_hz} | generator: {args.generator_sample_hz}")

    try:
        async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
            log_info("WebSocket connected")

            request_message = json.dumps({"text": args.text})
            log_info(f"Sending text: {args.text}")
            await ws.send(request_message)

            # Continue receiving until complete or error.
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    log_info("[WARN] Timed out waiting for server frames")
                    if received_complete:
                        break
                    raise RuntimeError("Timed out waiting for audio; connection not complete")

                try:
                    frame = json.loads(raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Invalid JSON received from server: {raw!r} ({exc})")

                if not isinstance(frame, dict) or "status" not in frame:
                    raise RuntimeError(f"Message missing 'status' field: {frame!r}")

                status = frame["status"]

                if status == "processing":
                    text_accepted = True
                    log_recv(f"status=processing message={frame.get('message')} "
                             f"utterance_id={frame.get('utterance_id')}")

                elif status == "metadata":
                    metadata = frame
                    log_recv(f"status=metadata encoding={frame.get('encoding')} "
                             f"channels={frame.get('channels')} "
                             f"sample_rate_hz={frame.get('sample_rate_hz')} "
                             f"audio_codec={frame.get('audio_codec')}")

                elif status == "synthesizing":
                    log_recv(f"status=synthesizing text_preview={frame.get('text_preview')}")

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
                    log_recv(f"status=streaming chunk_size={frame.get('chunk_size')} "
                             f"decoded_bytes={len(chunk)} total={total_bytes}")

                elif status == "complete":
                    received_complete = True
                    log_recv(f"status=complete total_bytes={frame.get('total_bytes')} "
                             f"duration={frame.get('duration_seconds')}")
                    break

                elif status == "error":
                    received_error = frame.get("message") or "unknown error"
                    log_info(f"[ERROR] Server reported error: {received_error}")
                    break

                else:
                    log_recv(f"status=unknown{status} frame={frame!r}")

            # ------------------------------------------------------------------
            # Validation & WAV saving
            # ------------------------------------------------------------------
            if received_error is not None:
                print("[ERROR] Aborting due to server error.", file=sys.stderr)
                return 1

            if not received_complete:
                raise RuntimeError("WebSocket closed before 'complete' was received")

            if stream_msg_count == 0:
                raise RuntimeError("Received zero streaming audio chunks; nothing to save")

            log_info(f"Total audio bytes received: {total_bytes}")

            audio = b"".join(audio_chunks)

            # Determine WAV parameters from server metadata.
            if metadata is None:
                raise RuntimeError(
                    "Missing audio metadata; cannot determine sample rate/channels")

            sample_rate = metadata.get("sample_rate_hz")
            channels = metadata.get("channels")
            if sample_rate is None or channels is None:
                raise RuntimeError(
                    f"Metadata incomplete for WAV writing: {metadata!r}")

            # Default codec: LINEAR16 (PCM s16). Only support PCM s16 for now.
            codec = (metadata.get("audio_codec") or
                     metadata.get("encoding") or "LINEAR16").upper()
            if "LINEAR16" not in codec and "PCM" not in codec:
                raise RuntimeError(
                    f"Unsupported codec for WAV writing: {codec}. ",
                    "Only LINEAR16/PCM s16 supported in this test.")

            sampwidth = 2  # 16-bit

            out_wav = os.path.join(out_dir, f"output_{args.call_id}.wav")
            with wave.open(out_wav, "wb") as wf:
                wf.setnchannels(int(channels))
                wf.setsampwidth(sampwidth)
                wf.setframerate(int(sample_rate))
                wf.writeframes(audio)

            log_info(f"Saved WAV: {out_wav}")

            # ------------------------------------------------------------------
            # Print verification summary
            # ------------------------------------------------------------------
            frames = len(audio) // (int(channels) * sampwidth)
            duration = frames / float(sample_rate) if sample_rate else 0.0
            filesize = os.path.getsize(out_wav)

            print("\n" + "=" * 60)
            print("VERIFICATION SUMMARY")
            print("=" * 60)
            print(f"WebSocket connected     : yes")
            print(f"Text accepted           : {'yes' if text_accepted else 'unknown'}")
            print(f"Audio chunks received   : {stream_msg_count}")
            print(f"Total audio bytes       : {total_bytes}")
            print(f"Complete received       : {received_complete}")
            print(f"sample_rate             : {sample_rate}")
            print(f"channels                : {channels}")
            print(f"sample width / encoding : {sampwidth * 8}-bit ({codec})")
            print(f"duration                : {duration:.3f} s")
            print(f"file size               : {filesize} bytes")
            print(f"Output WAV              : {out_wav}")
            print("=" * 60)

            # ffprobe optional validation
            ffprobe = os.popen("command -v ffprobe 2>/dev/null").read().strip()
            if ffprobe:
                print("\n[INFO] ffprobe validation:")
                os.system(
                    f'"{ffprobe}" -v error -show_entries '
                    f'format=duration,size:stream=codec_name,sample_rate,channels '
                    f'-of default=noprint_wrappers=1 "{out_wav}"'
                )
            else:
                print("\n[INFO] 'ffprobe' not found; skipping ffprobe validation.")

            log_info("WebSocket closed")
            return 0

    except websockets.exceptions.InvalidURI as exc:
        print(f"[ERROR] Invalid WebSocket URI: {exc}", file=sys.stderr)
        return 1
    except websockets.exceptions.InvalidHandshake as exc:
        print(f"[ERROR] WebSocket handshake failed: {exc}", file=sys.stderr)
        return 1
    except (ConnectionError, OSError) as exc:
        print(f"[ERROR] Connection/DNS failure: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
PY_EOF

chmod +x "$TEST_DIR/websocket_test.py"

echo "[INFO] Wrote:"
echo "        $TEST_DIR/websocket_test.py"
echo "        $TEST_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 4. Install/verify dependencies.
#    Prefer the existing project env. Check if 'websockets' is importable
#    before installing anything.
# ---------------------------------------------------------------------------
echo "[INFO] Checking for an available WebSocket client library..."

python3 - <<'EOF'
import importlib.util
for name in ("websockets", "aiohttp"):
    print(f"{name}: {'present' if importlib.util.find_spec(name) else 'missing'}")
EOF

if python3 -c "import websockets" 2>/dev/null; then
    echo "[INFO] 'websockets' already available; skipping install."
else
    echo "[INFO] Installing 'websockets'..."
    python3 -m pip install --quiet -r "$TEST_DIR/requirements.txt"
fi

# ---------------------------------------------------------------------------
# 5. Run the test
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Running Cartesia TTS WebSocket test..."
echo "[INFO] Text to synthesize: $TEXT_TO_SYNTHESIZE"
cd "$TEST_DIR"

python3 websocket_test.py \
    --voice-id "$CARTESIA_VOICE_ID" \
    --text "$TEXT_TO_SYNTHESIZE" \
    --call-id "$(uuidgen || cat /proc/sys/kernel/random/uuid)" \
    --url "$WS_URL" \
    --model-id "$MODEL_ID" \
    --language-code "$LANGUAGE_CODE" \
    --sample-rate-hz "$SAMPLE_RATE_HZ" \
    --generator-sample-hz "$GENERATOR_SAMPLE_HZ" \
    --client-name "$CLIENT_NAME"

EXIT_CODE=$?
echo "[INFO] Test exit code: $EXIT_CODE"
exit $EXIT_CODE
