#!/usr/bin/env bash
# =============================================================================
# create_chandra_words_dataset.sh
#
# Build a WORDS-FOCUSED ॅ (U+0945 CANDRA E) / ॲ (U+0972 CANDRA A) dataset for
# the v9 fine-tune, then synthesize it with Cartesia (Arushi voice).
#
# Why: the v8 run fixed ॅ/ॲ from the sentences that happened to contain them
# in the existing corpus. v9 drives the fix harder: it embeds a curated
# Devenagari WORD LIST (real Marathi loanwords that contain ॅ/ॲ, plus the
# full consonant+ॅ syllable set for complete coverage) and derives EXACTLY
# ONE natural sentence per word.
#
#   - one sentence per word  (>= 200 total; the list below is ~300+
#     real words + 30 consonant syllables, so "340-ish" comes naturally
#     and MORE is fine - the count is printed and gated, not hard-capped)
#   - every derived sentence contains at least one of ॅ / ॲ by construction
#   - synthesized with Cartesia Arushi VOICE 95d51f79-c397-46f9-b49a-23763d3eaa2d
#     at speaking_rate=0.8 (SAME pipeline as candra_e / prono_vowel)
#
# Outputs (server):
#   cartesia_ws/chandra_words/words.txt     (one word per line, for review)
#   cartesia_ws/chandra_words/sentences.txt (one sentence per line)
#   cartesia_ws/chandra_words/output/0001.wav..  (per line)
#   cartesia_ws/chandra_words/metadata.csv  (+ failed_samples.csv)
#   cartesia_ws/generate_chandra_words.py   (generated runner)
#
# Usage (run on the SERVER):
#   export CARTESIA_VOICE_ID=95d51f79-c397-46f9-b49a-23763d3eaa2d
#   bash scripts/create_chandra_words_dataset.sh           # first 5 (test)
#   bash scripts/create_chandra_words_dataset.sh --full    # all sentences
#   bash scripts/create_chandra_words_dataset.sh --start 20 --limit 10
#   bash scripts/create_chandra_words_dataset.sh --rebuild-metadata
#
# Gate: < 200 sentences -> STOP (exit 3), because such a small focused set
# risks the overfitting/instability failure mode this project already hit.
# FORCE=1 overrides.
#
# Then: bash scripts/prepare_dataset_v9.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_DIR="$REPO_ROOT/cartesia_ws/chandra_words"
OUTPUT_DIR="$DATASET_DIR/output"
WORDS_FILE="$DATASET_DIR/words.txt"
SENTENCES_FILE="$DATASET_DIR/sentences.txt"

# Cartesia synthesis params (same defaults as every prior targeted set)
WS_URL="${CARTESIA_WS_URL:-wss://vectortts.markytics.ai/tts/ws/}"
MODEL_ID="${CARTESIA_MODEL_ID:-sonic-3}"
LANGUAGE_CODE="${CARTESIA_LANGUAGE_CODE:-en}"
SAMPLE_RATE_HZ="${CARTESIA_SAMPLE_RATE:-16000}"
CLIENT_NAME="${CARTESIA_CLIENT_NAME:-f5tts-cartesia-chandra}"
SPEED="${CARTESIA_SPEED:-0.8}"          # speaking_rate=0.8 (v8 spec)
VOICE_ID="${CARTESIA_VOICE_ID:-}"

if [[ -z "$VOICE_ID" ]]; then
    echo "[ERROR] CARTESIA_VOICE_ID is not set." >&2
    echo "        export CARTESIA_VOICE_ID=95d51f79-c397-46f9-b49a-23763d3eaa2d" >&2
    exit 1
fi

mkdir -p "$DATASET_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Word-list ॅ/ॲ (CANDRA E/A) dataset generation - v9"
echo "============================================================"
echo "Dataset dir  : $DATASET_DIR"
echo "Voice        : $VOICE_ID"
echo "speaking_rate: $SPEED"

# ---------------------------------------------------------------------------
# 1. Render words.txt + sentences.txt (words embedded below, one line per word)
# ---------------------------------------------------------------------------
python3 - "$WORDS_FILE" "$SENTENCES_FILE" <<'PY'
import sys

words_out, sentences_out = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------------------
# Curated REAL Marathi loanwords containing ॅ (CANDRA E) or ॲ (CANDRA A),
# grouped by onset. Only in-use words (not invented spellings).
# ---------------------------------------------------------------------------
curated = {
    "ka": [
        "कॅब", "कॅबिनेट", "कॅमेरा", "कॅप", "कॅश", "कॅशियर", "कॅप्टन",
        "कॅटलॉग", "कॅटेगरी", "कॅलेंडर", "कॅफे", "कॅम्प", "कॅम्पस",
        "कॅम्पेन", "कॅरेक्टर", "कॅरम", "कॅडेट", "कॅलरी", "कॅबिन",
        "कॅप्सूल", "कॅनडा", "कॅमेरामॅन", "कॅनव्हास", "कॅटरिंग",
        "कॅपिटल", "कॅलिफोर्निया", "कॅशबॅक", "कॅट", "कॅन्सर",
        "कॅम्पिंग", "कॅरियर", "कॅफीन",
    ],
    "ga": [
        "गॅस", "गॅरेज", "गॅलरी", "गॅजेट", "गॅलन", "गॅप", "गॅस्ट्रिक", "गॅंबल",
    ],
    "cha": [
        "चॅट", "चॅम्पियन", "चॅप्टर", "चॅनल", "चॅलेंज", "चॅरिटी",
        "चॅटिंग", "चॅटबॉट", "चॅक", "चॅरेज", "चॅंबर",
    ],
    "ja": [
        "जॅकेट", "जॅम", "जॅक", "जॅग्वार", "जॅम्बो", "जॅझ",
    ],
    "ta": [
        "टॅक्स", "टॅक्सी", "टॅब", "टॅबलेट", "टॅप", "टॅग", "टॅंक", "टॅलेंट",
        "टॅली", "टॅटू", "टॅरिफ", "टॅक्टिक्स", "टॅंकर", "टॅंगो", "टॅक्सेबल",
        "टॅपिंग", "टॅब्लॉइड", "टॅक्सेशन", "टॅको", "टॅलक",
    ],
    "da": [
        "डॅटा", "डॅम", "डॅन्स", "डॅन्सर", "डॅशबोर्ड", "डॅमेज",
        "डॅटाबेस", "डॅडी", "डॅटाशीट", "डॅम्प", "डॅश",
    ],
    "tha": [
        "थॅंक्स",
    ],
    "na": [
        "नॅशनल", "नॅचरल", "नॅम", "नॅपकिन", "नॅव्ही", "नॅरेटर",
        "नॅनो", "नॅमप्लेट", "नॅरेटिव्ह",
    ],
    "pa": [
        "पॅकेट", "पॅन", "पॅटर्न", "पॅकेज", "पॅड", "पॅनकेक", "पॅशन",
        "पॅच", "पॅराच्युट", "पॅरलल", "पॅंट", "पॅनल", "पॅगोडा",
        "पॅरिस", "पॅसिफिक", "पॅडल", "पॅनलिस्ट", "पॅरामीटर",
    ],
    "fa": [
        "फॅक्ट्री", "फॅमिली", "फॅन", "फॅशन", "फॅक्ट", "फॅंटसी",
        "फॅट", "फॅन्सी", "फॅब्रिक",
    ],
    "ba": [
        "बॅग", "बॅटरी", "बॅडमिंटन", "बॅट", "बॅक", "बॅंक", "बॅंड",
        "बॅज", "बॅगेज", "बॅकअप", "बॅलन्स", "बॅल्कनी", "बॅरल",
        "बॅरियर", "बॅलट", "बॅंकॉक", "बॅंडेज", "बॅकबोन",
        "बॅकग्राऊंड", "बॅटन", "बॅकपॅक",
    ],
    "ma": [
        "मॅप", "मॅनेजर", "मॅनेजमेंट", "मॅथ्स", "मॅक्सिमम", "मॅजिक",
        "मॅन्युअल", "मॅरेथॉन", "मॅडम", "मॅम", "मॅच", "मॅंडेट", "मॅन",
        "मॅटर", "मॅट्रिक", "मॅकेनिकल", "मॅकेनिक", "मॅनेज", "मॅग्नेट",
    ],
    "ra": [
        "रॅम", "रॅक", "रॅप", "रॅकेट", "रॅली", "रॅगी", "रॅबिट",
        "रॅंच", "रॅंडम", "रॅट", "रॅम्प",
    ],
    "la": [
        "लॅपटॉप", "लॅम्प", "लॅब", "लॅन", "लॅप", "लॅंड",
        "लॅंडलाइन", "लॅमिनेट",
    ],
    "vha": [
        "व्हॅन", "व्हॅली", "व्हॅक्यूम", "व्हॅनिला", "व्हॅलिड",
        "व्हॅक्सिन", "व्हॅक्स",
    ],
    "sha": [
        "शॅम्पू", "शॅल", "शॅक",
    ],
    "sa": [
        "सॅम्पल", "सॅलड", "सॅलरी", "सॅन्डविच", "सॅंडल", "सॅक",
        "सॅफारी", "सॅंड", "सॅंपलिंग", "सॅंडबॉक्स",
    ],
    "ha": [
        "हॅलो", "हॅट", "हॅमर", "हॅक", "हॅकर", "हॅपी",
        "हॅम्बर्गर", "हॅम्स्टर",
    ],
}

# Consonant-cluster ॅ words (ट्रॅ-, ब्लॅ-, फ्लॅ-, स्टॅ-, स्कॅ-, स्पॅ-, स्नॅ-,
# प्रॅ-, प्लॅ-, ग्लॅ-, ग्रॅ-) - all in normal Marathi use.
cluster_prefix = [
    "ट्रॅक्टर", "ट्रॅफिक", "ट्रॅक", "ट्रॅकिंग",
    "ब्लॅक", "ब्लॅकबोर्ड", "ब्लॅकमेल",
    "फ्लॅट", "फ्लॅग", "फ्लॅश",
    "स्टॅम्प", "स्टॅंड", "स्टॅटस", "स्टॅट्स", "स्टॅटिक",
    "स्कॅन", "स्कॅनर", "स्कॅनिंग",
    "स्पॅम", "स्पॅन",
    "स्नॅक", "स्नॅक्स", "स्नॅपशॉट",
    "प्रॅक्टिस", "प्रॅक्टिकल",
    "प्लॅन", "प्लॅटफॉर्म", "प्लॅस्टिक",
    "ग्लॅमर", "ग्रॅंड",
]

# Independent ॲ (U+0972 CANDRA A) words - appear naturally at word start.
candra_a = [
    "ॲप", "ॲपल", "ॲक्शन", "ॲड", "ॲलार्म", "ॲड्रेस", "ॲडव्हान्स",
    "ॲडीशन", "ॲक्सेसरी", "ॲक्सिडेंट", "ॲटलस", "ॲटम", "ॲथलेटिक",
    "ॲडमिशन", "ॲम्बुलन्स", "ॲनिमेशन", "ॲन्टीना", "ॲपॉईंटमेंट",
    "ॲमाऊंट", "ॲल्बम", "ॲल्युमिनियम", "ॲव्होकॅडो", "ॲटॅक",
    "ॲकाउंट", "ॲक्ट", "ॲक्टर", "ॲक्टिव्ह", "ॲसिड", "ॲक्सेलरेटर",
    "ॲक्सेस", "ॲमेझॉन", "ॲडमिन", "ॲस्पिरिन", "ॲलर्जी",
]

# ---------------------------------------------------------------------------
# Flatten, dedupe (order preserved), then add the 30 consonant+ॅ syllables.
# The syllables guarantee mathematical coverage of every consonant+ॅ combo.
# ---------------------------------------------------------------------------
syllables = [c + "\u0945" for c in "कखगघचछजझटठडढतथदधनपफबभमयरलवशसहळ"]

words = []
seen = set()
group_counts = {}
for gname, wlist in list(curated.items()) + [("cluster", cluster_prefix),
                                              ("candra_a", candra_a)]:
    n = 0
    for w in wlist:
        if w not in seen:
            seen.add(w)
            words.append(w)
            n += 1
    group_counts[gname] = n

syllable_count = 0
for s in syllables:
    if s not in seen:
        seen.add(s)
        words.append(s)
        syllable_count += 1

# ---------------------------------------------------------------------------
# Templates. Real words get a natural frame; syllables get v8-style
# pronunciation carriers. All frames keep the ॅ/ॲ inside the utterance.
# ---------------------------------------------------------------------------
real_templates = [
    "{w} खूप महत्त्वाचं आहे.",
    "आज {w} बद्दल बोलूया.",
    "हे {w} कुठे मिळतं?",
    "मला {w} आवडतं.",
    "मी {w} पाहिलं.",
    "तिने {w} उदाहरण दिलं.",
    "{w} का महत्त्वाचं आहे?",
    "प्रत्येकाला {w} माहीत आहे.",
    "आपण {w} कडे लक्ष देऊ.",
    "{w} या विषयावर चर्चा करूया.",
    "तो {w} शब्द वापरतो.",
    "{w} ची माहिती शोधूया.",
]
syllable_carriers = [
    "सोनूने '{syl}' असं नीट उच्चारलं.",
    "आईने '{syl}' हा शब्द वाचला.",
    "गुरुजींनी '{syl}' असा आवाज काढला.",
    "बाळाने '{syl}' म्हटलं.",
    "त्यांनी '{syl}' असं म्हणून दाखवलं.",
]

real_words = words[: len(words) - syllable_count]
sentences = []
for idx, w in enumerate(real_words):
    tpl = real_templates[idx % len(real_templates)]
    sentences.append(tpl.replace("{w}", w))
for idx, s in enumerate(syllables):
    tpl = syllable_carriers[idx % len(syllable_carriers)]
    sentences.append(tpl.replace("{syl}", s))

# ---------------------------------------------------------------------------
# Verify every sentence actually contains ॅ (U+0945) or ॲ (U+0972).
# ---------------------------------------------------------------------------
def has_candra(t: str) -> bool:
    return "\u0945" in t or "\u0972" in t

bad = [s for s in sentences if not has_candra(s)]
if bad:
    print("ERROR: %d generated sentences lack ॅ/ॲ - bug in templates/words." % len(bad))
    for b in bad[:5]:
        print("  ", repr(b))
    raise SystemExit(1)

with open(words_out, "w", encoding="utf-8", newline="\n") as f:
    for w in words:
        f.write(w + "\n")
with open(sentences_out, "w", encoding="utf-8", newline="\n") as f:
    for s in sentences:
        f.write(s + "\n")

n_candra_e = sum(1 for s in sentences if "\u0945" in s)
n_candra_a = sum(1 for s in sentences if "\u0972" in s)
print("REAL words (curated clusters incl.) : %d" % len(real_words))
print("Consonant+ॅ syllables              : %d" % syllable_count)
print("TOTAL words / sentences            : %d" % len(sentences))
print("Per-group counts                   :", group_counts)
print("Sentences containing ॅ             : %d" % n_candra_e)
print("Sentences containing ॲ             : %d" % n_candra_a)
print("Sentences missing both             : %d (must be 0)" % len(bad))
print("Written words.txt / sentences.txt")
PY

echo ""
echo "[1] words.txt (first 8):"
head -8 "$WORDS_FILE"
echo "    ..."
echo "    words total  : $(wc -l < "$WORDS_FILE")"
echo "    sentences    : $(wc -l < "$SENTENCES_FILE")"
echo ""
echo "    sentences.txt (first 5):"
head -5 "$SENTENCES_FILE"
echo "    ..."

TOTAL_SENTENCES="$(wc -l < "$SENTENCES_FILE")"
if [[ "$TOTAL_SENTENCES" -lt 200 ]]; then
    echo ""
    echo "[WARNING] Only $TOTAL_SENTENCES sentences. Too few risks overfitting/"
    echo "          instability (same failure mode as the earlier 53-char vocab"
    echo "          expansion)."
    if [[ "${FORCE:-0}" == "1" ]]; then
        echo "[FORCE] FORCE=1 set; proceeding with $TOTAL_SENTENCES sentences."
    else
        echo ""
        echo "[STOP] Not proceeding. Options:"
        echo "   a) proceed anyway  ->  FORCE=1 bash scripts/create_chandra_words_dataset.sh"
        echo "   b) review the embedded word list (scripts/create_chandra_words_dataset.sh)"
        echo "      and widen it, then re-run"
        exit 3
    fi
fi

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
# 3. Generate the dedicated runner (same WS pipeline as candra_e / prono)
# ---------------------------------------------------------------------------
cat > "$REPO_ROOT/cartesia_ws/generate_chandra_words.py" <<'PY_EOF'
#!/usr/bin/env python3
"""
Word-list ॅ/ॲ (CANDRA E/A) dataset generation - v9.

Same WebSocket synthesis pipeline as generate_candra_e.py / generate_prono_vowel.py
but pointed at cartesia_ws/chandra_words.

Usage:
    python generate_chandra_words.py [--start N] [--limit M] [--overwrite]
    python generate_chandra_words.py --rebuild-metadata
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
DATASET_DIR = os.path.join(BASE_DIR, "chandra_words")
SENTENCES_FILE = os.path.join(DATASET_DIR, "sentences.txt")
OUTPUT_DIR = os.path.join(DATASET_DIR, "output")
METADATA_CSV = os.path.join(DATASET_DIR, "metadata.csv")
FAILED_CSV = os.path.join(DATASET_DIR, "failed_samples.csv")

WS_URL = os.environ.get("CARTESIA_WS_URL", "wss://vectortts.markytics.ai/tts/ws/")
MODEL_ID = os.environ.get("CARTESIA_MODEL_ID", "sonic-3")
LANGUAGE_CODE = os.environ.get("CARTESIA_LANGUAGE_CODE", "en")
SAMPLE_RATE_HZ = int(os.environ.get("CARTESIA_SAMPLE_RATE", "16000"))
CLIENT_NAME = os.environ.get("CARTESIA_CLIENT_NAME", "f5tts-cartesia-chandra")
SPEED = float(os.environ.get("CARTESIA_SPEED", "0.8"))  # speaking_rate=0.8
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
    parser = argparse.ArgumentParser(description="v9 word-list ॅ/ॲ generation")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-metadata", action="store_true",
                        help="Rewrite metadata.csv from sentences.txt + existing WAVs.")
    args = parser.parse_args()

    if not os.path.isfile(SENTENCES_FILE):
        print(f"[ERROR] sentences.txt not found: {SENTENCES_FILE}", file=sys.stderr)
        return 1

    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    total = len(sentences)
    print(f"Total sentences: {total}")

    # ------------------------------------------------------------------
    # --rebuild-metadata mode
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
                    n = wf.getnframes()
                dur = n / float(sr) if sr else 0.0
                rows.append({"id": fid, "text": text,
                             "audio_file": f"{fid}.wav",
                             "duration_seconds": round(dur, 6),
                             "source": "chandra_words"})
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
        return 2 if missing > 0 else 0

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

    # Backfill skipped rows so an interrupted run never drops rows from metadata.
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
                                 "source": "chandra_words"})
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
chmod +x "$REPO_ROOT/cartesia_ws/generate_chandra_words.py"
echo "[INFO] Created: $REPO_ROOT/cartesia_ws/generate_chandra_words.py"

# ---------------------------------------------------------------------------
# 4. Run mode
# ---------------------------------------------------------------------------
cd "$REPO_ROOT/cartesia_ws"
export CARTESIA_WS_URL CARTESIA_MODEL_ID CARTESIA_LANGUAGE_CODE
export CARTESIA_SAMPLE_RATE CARTESIA_CLIENT_NAME CARTESIA_SPEED CARTESIA_VOICE_ID

MODE="${1:-test}"
case "$MODE" in
    --full)
        echo "[INFO] Generating ALL chandra_words sentences ($TOTAL_SENTENCES)."
        python3 generate_chandra_words.py
        ;;
    --start)
        START_LINE="${2:?Usage: --start <LINE> [--limit <N>]}"
        LIMIT_OPT=""
        if [[ "${3:-}" == "--limit" ]]; then
            LIMIT_OPT="--limit ${4:?--limit requires a number}"
        fi
        echo "[INFO] Generating starting at line $START_LINE $LIMIT_OPT"
        python3 generate_chandra_words.py --start "$START_LINE" $LIMIT_OPT
        ;;
    --rebuild-metadata)
        echo "[INFO] Rebuilding metadata.csv from existing WAVs (no synthesis)."
        python3 generate_chandra_words.py --rebuild-metadata
        ;;
    *)
        echo "[INFO] Running 5-SAMPLE TEST mode."
        python3 generate_chandra_words.py --start 1 --limit 5
        ;;
esac
EXIT_CODE=$?
echo ""
echo "[INFO] create_chandra_words_dataset.sh finished with exit code $EXIT_CODE"
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[INFO] Synthesize the full set with: bash scripts/create_chandra_words_dataset.sh --full"
    echo "       (retry a failed batch with --start N --limit M, then --rebuild-metadata)"
fi
exit $EXIT_CODE