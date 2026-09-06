#!/usr/bin/env bash
# =============================================================================
# run_train_v9.sh
#
# STEP 3 (pre-flight) + STEP 4 (launch) for Chandra_Words_v9.
#
# CHANDRA FOCUSED WORDS training (v9) - one sentence per ॅ/ॲ (U+0945/U+0972)
# word from the word-list dataset Chandra_Words_v9 (see create_chandra_words
# _dataset.sh). Fine-tuned on top of the v7 trained checkpoint (the current
# latest/live checkpoint). Naming conventions are left unchanged.
#
# Word-list ॅ/ॲ targeted fine-tune on top of the v7 trained checkpoint:
#   - warm-start: ckpts/Rasa_Marathi_Emotion_Female_v9/model_extended.pt
#                 (prepare_pretrain_v9.sh output, based on v7's model_last.pt)
#   - fine-tuned on: Chandra_Words_v9 (one sentence per ॅ/ॲ word, word-list
#                    driven; see create_chandra_words_dataset.sh)
#   - 10 epochs, learning_rate 1e-5, batch 4 (sample) - same schedule shape
#     as the previous run, recomputed with THIS dataset's step count.
#
# save_per_updates / last_per_updates / warmup are proportional to this
# dataset's per-epoch step count (steps/epoch = samples / 4).
#
# STEP 3 pre-flight (mandatory, aborts with a nonzero exit on any failure):
#   1. disk space: df -h /root, require >= 10 GB free
#   2. GPU: nvidia-smi (shared box; verifies free VRAM >= 6 GB)
#   3. embedding shape already verified by prepare_pretrain_v9.sh (step 3.3)
#   4. stale copy cleanup: rm -f ckpts/Chandra_Words_v9/pretrained_*.pt
#
# Usage (server, after prepare_pretrain_v9.sh):
#   bash scripts/run_train_v9.sh
#
# Monitor:
#   tail -f /root/f5-tts-marathi/logs/training_v9.log
# Trainer should confirm in the log: text_num_embeds = 139 (or vocab+1).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
mkdir -p logs
LOG=logs/training_v9.log

DATASET_NAME="${DATASET_NAME:-Chandra_Words_v9}"
PRETRAIN="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female_v9/model_extended.pt"
VOCAB="/root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"
PREP_DIR="/root/f5-tts-marathi/f5tts/data/Chandra_Words_v9"
META="$REPO_ROOT/cartesia_ws/chandra_words/train_metadata.csv"
ACCELERATE="$REPO_ROOT/f5tts/bin/accelerate"
FINETUNE_CLI="$REPO_ROOT/f5tts/lib/python3.12/site-packages/f5_tts/train/finetune_cli.py"

echo "============================================================"
echo " Chandra_Words_v9 - 10 epoch fine-tune (word-list ॅ/ॲ fix)"
echo " CHANDRA FOCUSED WORDS training - warm start from v7"
echo " Pretrain base: v7 (Cartesia_Rasa_Combined_v7/model_last.pt)"
echo "============================================================"

# ---------------------------------------------------------------------------
# STEP 3 - PRE-FLIGHT SAFETY CHECKS (abort on failure, nonzero exit)
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 3: pre-flight safety checks ---"

for f in "$PRETRAIN" "$VOCAB" "$ACCELERATE" "$FINETUNE_CLI"; do
    if [[ ! -e "$f" ]]; then
        echo "[ERROR] Required file not found: $f"
        exit 1
    fi
done
if [[ ! -d "$PREP_DIR" ]]; then
    echo "[ERROR] Preprocessed dataset dir missing: $PREP_DIR"
    echo "        Run bash scripts/prepare_dataset_v9.sh first."
    exit 1
fi

# 3.0 - confirm the pretrain really is v7-based (name + embedding shape sanity)
if "$REPO_ROOT/f5tts/bin/python3" - "$PRETRAIN" "$VOCAB" <<'PY'
import sys, torch
ckpt, vocab = sys.argv[1], sys.argv[2]
n = sum(1 for l in open(vocab, encoding="utf-8") if l.rstrip("\n"))
sd = torch.load(ckpt, map_location="cpu")
key = "transformer.text_embed.text_embed.weight"
for name in ("ema_model_state_dict", "model_state_dict"):
    if name in sd and isinstance(sd[name], dict) and key in sd[name]:
        rows = sd[name][key].shape[0]
        break
else:
    rows = sd[key].shape[0] if key in sd else -1
print(f"  pretrain text_embed rows = {rows}; vocab+1 = {n+1}")
if rows != n + 1:
    print("[ERROR] pretrain embedding rows != vocab+1; wrong/stale pretrain.")
    sys.exit(1)
PY
then
    : # python heredoc passed
else
    echo "[ERROR] Pretrain shape check failed (see above)."
    exit 1
fi

# 3.1 - disk space
echo ""
echo "[3.1] Disk space (df -h /root):"
df -h /root
FREE_GB=$(df -BG --output=avail /root | awk 'NR==2 {gsub("G","",$1); print $1+0}')
echo "      Free: ${FREE_GB} GB"
if [[ "$FREE_GB" -lt 10 ]]; then
    echo "[ERROR] Free disk < 10 GB. Checkpoints are ~5GB each; disk-full mid-save"
    echo "        has caused checkpoint corruption multiple times on this project."
    echo "        Free space up before launching."
    exit 1
fi

# 3.2 - GPU
echo ""
echo "[3.2] GPU (nvidia-smi):"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader || {
    echo "[ERROR] nvidia-smi failed. Is the driver healthy / GPU free?"
    exit 1
}
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [[ -z "$FREE_MB" || "$FREE_MB" -lt 6144 ]]; then
    echo "      Free VRAM: ${FREE_MB} MB"
    echo "[ERROR] Less than 6 GB free VRAM on GPU. This box is shared with other"
    echo "        processes - free up VRAM before launching."
    exit 1
fi
echo "      Free VRAM: ${FREE_MB} MB (>= 6 GB OK)"

# 3.4 - stale pretrained_*.pt cleanup in the new ckpt dir
CKPT_DIR="/root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/$DATASET_NAME"
echo ""
echo "[3.4] Cleaning stale pretrained_*.pt in $CKPT_DIR ..."
if [[ -d "$CKPT_DIR" ]]; then
    find "$CKPT_DIR" -maxdepth 1 -name 'pretrained_*.pt' -print -delete
fi
mkdir -p "$CKPT_DIR"

# ---------------------------------------------------------------------------
# Compute proportional save / last / warmup rates from this dataset's size
# ---------------------------------------------------------------------------
echo ""
echo "--- dataset size -> proportional step counts ---"
SAMPLES=$("$REPO_ROOT/f5tts/bin/python3" - "$PREP_DIR" <<'PY'
import sys, json, os
d = sys.argv[1]
j = os.path.join(d, "duration.json")
try:
    x = json.load(open(j, encoding="utf-8"))
    durs = x.get("duration", [])
    if isinstance(durs, dict): n = len(durs)
    else: n = len(durs)
    print(n)
except Exception:
    print(0)
PY
)
if [[ -z "$SAMPLES" || "$SAMPLES" -eq 0 ]]; then
    SAMPLES=$(($(wc -l < "$META") - 1))
fi
BATCH=4
STEPS_PER_EPOCH=$(( (SAMPLES + BATCH - 1) / BATCH ))
TOTAL_UPDATES=$(( STEPS_PER_EPOCH * 10 ))

SAVE_PER=$STEPS_PER_EPOCH
LAST_PER=$STEPS_PER_EPOCH
WARMUP=$(( TOTAL_UPDATES / 20 ))
[[ "$WARMUP" -lt 10 ]] && WARMUP=10

echo "  samples          : $SAMPLES"
echo "  batch            : $BATCH (sample)"
echo "  steps/epoch      : $STEPS_PER_EPOCH"
echo "  total updates    : $TOTAL_UPDATES (10 epochs)"
echo "  num_warmup_updates: $WARMUP"
echo "  save_per_updates : $SAVE_PER"
echo "  last_per_updates : $LAST_PER"
echo "  keep_last_n      : 1"

echo ""
echo "[INFO] Launching fine-tune in background."
echo "[INFO] dataset_name  : $DATASET_NAME"
echo "[INFO] CHANDRA FOCUSED WORDS training (v9) - warm start from v7"
echo "[INFO] pretrain      : $PRETRAIN (v7 warm start)"
echo "[INFO] tokenizer     : custom ($VOCAB)"
echo "[INFO] log           : $LOG"

nohup "$ACCELERATE" launch \
    "$FINETUNE_CLI" \
    --exp_name F5TTS_v1_Base \
    --dataset_name "$DATASET_NAME" \
    --finetune \
    --pretrain "$PRETRAIN" \
    --tokenizer custom \
    --tokenizer_path "$VOCAB" \
    --batch_size_per_gpu "$BATCH" \
    --batch_size_type sample \
    --epochs 10 \
    --num_warmup_updates "$WARMUP" \
    --save_per_updates "$SAVE_PER" \
    --last_per_updates "$LAST_PER" \
    --keep_last_n_checkpoints 1 \
    --learning_rate 1e-5 \
    --log_samples \
    > "$LOG" 2>&1 &

disown
echo "[INFO] Launched (PID $!). Monitor: tail -f $LOG"
echo "[INFO] Verify in the log shortly after start:"
echo "        1. tokenizer_path  -> $VOCAB"
echo "        2. text_num_embeds -> matches the vocab lines + 1"
echo "        3. dataset rows    -> $SAMPLES"
echo "[INFO] After training completes -> bash scripts/deploy_cartesia_v9.sh"