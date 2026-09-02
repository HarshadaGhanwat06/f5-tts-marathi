#!/usr/bin/env bash
# =============================================================================
# create_candra_e_dataset.sh
#
# Build a TARGETED dataset of sentences heavy in ॅ (CANDRA E, U+0945) and ॲ
# (CANDRA A, U+0972) sounds, then synthesize them with Cartesia (Arushi).
#
# Why: the v3 model now pronounces English transliterations correctly but the
# ॅ sound is still wrong. The token exists in the vocab (138 tokens) yet the
# model has too few FOCUSED examples to learn the grapheme -> /a/ (as in "cat")
# mapping. This dataset directly fixes that imbalance.
#
# Coverage (~210 sentences):
#   A. ~60 natural curated sentences with real ॅ/ॲ loanwords in Marathi
#      (कॅमेरा, बॅग, हॅलो, गॅस, सॅम्पल, मॅप, पॅकेट, टॅक्स, डॅटा, ॲप, ॲक्शन, ...)
#   B. ~150 carrier sentences over ALL 30 consonant + ॅ combos (5 each)
#      (कॅ खॅ गॅ घॅ चॅ छॅ जॅ झॅ टॅ ठॅ डॅ ढॅ तॅ थॅ दॅ धॅ नॅ पॅ फॅ बॅ भॅ मॅ
#       यॅ रॅ लॅ वॅ शॅ सॅ हॅ ळॅ) so the model sees the matra in every place.
#
# Outputs (server):
#   cartesia_ws/cartesia_candra_e/sentences.txt      (one sentence per line)
#   cartesia_ws/cartesia_candra_e/output/0001.wav..  (per line)
#   cartesia_ws/cartesia_candra_e/metadata.csv       (+ failed_samples.csv)
#   cartesia_ws/generate_candra_e.py                 (generated runner)
#
# Usage (run on the SERVER):
#   export CARTESIA_VOICE_ID=95d51f79-c397-46f9-b49a-23763d3eaa2d
#   bash scripts/create_candra_e_dataset.sh          # first 5 samples (test)
#   bash scripts/create_candra_e_dataset.sh --full   # all samples
#   bash scripts/create_candra_e_dataset.sh --start 20 --limit 10
#
# Then: bash scripts/prepare_dataset_v4.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="$REPO_ROOT/cartesia_ws/cartesia_candra_e"
OUTPUT_DIR="$DATASET_DIR/output"
SENTENCES_FILE="$DATASET_DIR/sentences.txt"

# Cartesia synthesis params (same defaults as the main cartesia runner)
WS_URL="${CARTESIA_WS_URL:-wss://vectortts.markytics.ai/tts/ws/}"
MODEL_ID="${CARTESIA_MODEL_ID:-sonic-3}"
LANGUAGE_CODE="${CARTESIA_LANGUAGE_CODE:-en}"
SAMPLE_RATE_HZ="${CARTESIA_SAMPLE_RATE:-16000}"
CLIENT_NAME="${CARTESIA_CLIENT_NAME:-f5tts-cartesia-candrae}"
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
echo " Targeted ॅ / ॲ (CANDRA E/A) dataset generation"
echo "============================================================"
echo "Dataset dir : $DATASET_DIR"
echo "Voice       : $VOICE_ID"
echo "speaking_rate: $SPEED"

# ---------------------------------------------------------------------------
# 1. Render sentences.txt (curated + carrier coverage)
# ---------------------------------------------------------------------------
python3 - "$SENTENCES_FILE" <<'PY'
import sys

out_path = sys.argv[1]

# A. Natural sentences with real ॅ / ॲ loanwords (natural Marathi context)
real = [
    "मी आज नवीन कॅमेरा वापरला.",
    "तिने काळी बॅग घेतली.",
    "तिने हॅलो म्हणत फोन उचलला.",
    "घरात गॅस संपला आहे.",
    "कामाचा नमुना म्हणून हा सॅम्पल पाठवा.",
    "मला नवीन मॅप लागेल.",
    "तो पॅकेट कुणाचं आहे.",
    "आयकर भरताना टॅक्स मोजायचा.",
    "संगणकातली सगळी डॅटा जपून ठेवली आहे.",
    "फोनमध्ये नवीन ॲप इन्स्टॉल केलं.",
    "कारला नवीन बॅटरी लागली.",
    "त्याने मॅनेजरकडे कागद दिले.",
    "आम्ही बॅडमिंटन खेळलो.",
    "हा जॅकेट हिवाळ्यासाठी आहे.",
    "तिने सॅन्डविच खाल्ला.",
    "मला टॅक्सीची वेळ चुकली.",
    "त्याची फॅक्ट्री शहराबाहेर आहे.",
    "शाळेत प्रत्येक वर्गाला कॅप्टन असतो.",
    "वाढदिवसाला कॅप आणून दिलीस.",
    "त्याने हॅट घातली.",
    "मला लॅपटॉपची स्क्रीन बदलायची आहे.",
    "गावात नवीन गॅरेज उघडलं.",
    "त्याने टॅबवरून बातमी वाचली.",
    "स्टेजवर सुंदर पॅटर्न छापलेला होता.",
    "बागेतली गॅलरी खूप सुंदर आहे.",
    "त्याने सकाळी पॅनकेक खाल्ला.",
    "संगणकाची मॅथ्स मजा येते.",
    "तिने सॅलरी बँकेत जमा केली.",
    "मी रॅम वाढवायला सांगितलं.",
    "क्रिकेटसाठी बॅट आणि हेल्मेट हवं.",
    "तिने शॅम्पू विकत घेतला.",
    "सकाळी कॅश काढली.",
    "ही वॅन जुन्या बाजाराकडे जाते.",
    "ड्रॉइंगरूममध्ये वॅलीचा फोटो आहे.",
    "त्याने चॅम्पियन ट्रॉफी जिंकली.",
    "संगणकावर चॅट करायचं होतं.",
    "मुलांनी फॅमिली फोटो काढला.",
    "खोलीत ॲक्शन सिनेमा चालू आहे.",
    "पेपरमध्ये हे ॲड वाचलं.",
    "सकाळी ॲलार्म लावला नाही.",
    "घराचा ॲड्रेस लिहून द्या.",
    "मी कॅमेरा बरोबर लॅपटॉप घेऊन आलो.",
    "गॅस आणि बॅटरी दोन्ही बदलायच्या आहेत.",
    "हा नवीन सॅम्पल जरूर पाठवा.",
    "बॅगेत कॅमेरा आणि हॅट आहे.",
    "मॅनेजरने सगळी डॅटा तपासली.",
    "त्याचं ॲक्शन आणि ॲड दोन्ही छान आहे.",
    "कॅफेमध्ये गॅस स्टोव्ह आहे.",
    "हॅलो पाटील साहेब, सकाळचा नमस्कार.",
    "पॅकेट उघडून पाहा.",
    "टॅक्सीचं ॲप खूप उपयोगी आहे.",
    "डॅटा ॲनालिसिस कडे लक्ष द्या.",
    "सॅम्पल फाईल रिसीव्ह करा.",
    "मॅनेजमेंटची मीटिंग आज आहे.",
    "कॅशलेस व्यवहाराची सवय लावा.",
    "बॅटरी सेव्ह करण्यासाठी मोड वापरा.",
    "फॅक्टरीत काम सुरू आहे.",
    "वॅनमध्ये सामान भरलं.",
    "मला टॅबमधून इमेल पाठवायचा आहे.",
    "हे लॅपटॉप बॅटरीवर चालतं.",
]

# B. Carrier sentences covering EVERY consonant + ॅ combo
consonants = "कखगघचछजझटठडढतथदधनपफबभमयरलवशसहळ"
carriers = [
    "सोनूने '{SYL}' असं नीट उच्चारलं.",
    "आईने '{SYL}' हा शब्द वाचला.",
    "गुरुजींनी '{SYL}' असा आवाज काढला.",
    "बाळाने '{SYL}' म्हटलं.",
    "त्यांनी '{SYL}' असं म्हणून दाखवलं.",
]

sentences = list(real)
seen = set(real)
for c in consonants:
    syl = c + "\u0945"
    for tpl in carriers:
        s = tpl.replace("{SYL}", syl)
        if s not in seen:
            sentences.append(s)
            seen.add(s)

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    for s in sentences:
        f.write(s + "\n")

print(f"Written {len(sentences)} sentences -> {out_path}")
print(f"  curated natural : {len(real)}")
print(f"  consonant combos: {len(sentences) - len(real)}")
PY

echo ""
echo "[1] sentences.txt (first 5):"
head -5 "$SENTENCES_FILE"
echo "    ..."
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
# 3. Generate the dedicated runner (adapted from the main cartesia runner)
# ---------------------------------------------------------------------------
cat > "$REPO_ROOT/cartesia_ws/generate_candra_e.py" <<'PY_EOF'
#!/usr/bin/env python3
"""
Targeted ॅ/ॲ (CANDRA E/A) dataset generation.

Same WebSocket logic as generate_dataset.py but pointed at the
cartesia_ws/cartesia_candra_e dataset so ~200 extra focused samples can be
generated without touching the main dataset/metadata.

Usage:
    python generate_candra_e.py [--start N] [--limit M] [--overwrite]
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
DATASET_DIR = os.path.join(BASE_DIR, "cartesia_candra_e")
SENTENCES_FILE = os.path.join(DATASET_DIR, "sentences.txt")
OUTPUT_DIR = os.path.join(DATASET_DIR, "output")
METADATA_CSV = os.path.join(DATASET_DIR, "metadata.csv")
FAILED_CSV = os.path.join(DATASET_DIR, "failed_samples.csv")

WS_URL = os.environ.get("CARTESIA_WS_URL", "wss://vectortts.markytics.ai/tts/ws/")
MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3")
LANGUAGE_CODE = os.environ.get("CARTESIA_LANGUAGE_CODE", "en")
SAMPLE_RATE_HZ = int(os.environ.get("CARTESIA_SAMPLE_RATE", "16000"))
CLIENT_NAME = os.environ.get("CARTESIA_CLIENT_NAME", "f5tts-cartesia-candrae")
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
    parser = argparse.ArgumentParser(description="ॅ/ॲ targeted generation")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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

    new_rows = [r for r in results if r["status"] == "success"]
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
                                 "source": "candra_e"})
        print(f"[INFO] metadata.csv updated: {METADATA_CSV}")

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
chmod +x "$REPO_ROOT/cartesia_ws/generate_candra_e.py"
echo "[INFO] Created: $REPO_ROOT/cartesia_ws/generate_candra_e.py"

# ---------------------------------------------------------------------------
# 4. Run mode
# ---------------------------------------------------------------------------
cd "$REPO_ROOT/cartesia_ws"
export CARTESIA_WS_URL CARTESIA_MODEL_ID CARTESIA_LANGUAGE_CODE
export CARTESIA_SAMPLE_RATE CARTESIA_CLIENT_NAME CARTESIA_SPEED CARTESIA_VOICE_ID

MODE="${1:-test}"
case "$MODE" in
    --full)
        echo "[INFO] Generating ALL candra_e sentences."
        python3 generate_candra_e.py
        ;;
    --start)
        START_LINE="${2:?Usage: --start <LINE> [--limit <N>]}"
        LIMIT_OPT=""
        if [[ "${3:-}" == "--limit" ]]; then
            LIMIT_OPT="--limit ${4:?--limit requires a number}"
        fi
        echo "[INFO] Generating starting at line $START_LINE $LIMIT_OPT"
        python3 generate_candra_e.py --start "$START_LINE" $LIMIT_OPT
        ;;
    *)
        echo "[INFO] Running 5-SAMPLE TEST mode."
        python3 generate_candra_e.py --start 1 --limit 5
        ;;
esac
EXIT_CODE=$?
echo ""
echo "[INFO] create_candra_e_dataset.sh finished with exit code $EXIT_CODE"
exit $EXIT_CODE