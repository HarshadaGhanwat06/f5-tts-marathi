#!/usr/bin/env bash
# =============================================================================
# run_train_4epoch_v3.sh
#
# Re-fine-tune on the COMPLETE combined dataset (1630 Rasa + 1033 Cartesia)
# AFTER expanding vocab_extended.txt to 138 tokens (added ॅ and other missing
# Devanagari/Latin characters). 4 epochs.
#
# The model's text_num_embeds is derived from the vocab file size, so the newly
# added tokens automatically get fresh embeddings during training.
#
# NOTE: Uses a NEW dataset/checkpoint name (Cartesia_Rasa_Combined_v3) so the
# result lands in ckpts/Cartesia_Rasa_Combined_v3/ instead of overwriting the
# working v2 model.
#
# Run on server (after bash scripts/expand_vocab.sh --apply):
#   bash scripts/run_train_4epoch_v3.sh
#
# Monitor:
#   tail -f /root/f5-tts-marathi/logs/training_cartesia_v3.log
# =============================================================================
set -euo pipefail

REPO_ROOT=/root/f5-tts-marathi
cd "$REPO_ROOT"
mkdir -p logs
LOG=logs/training_cartesia_v3.log

# Default train dataset name; allow override.
DATASET_NAME="${DATASET_NAME:-Cartesia_Rasa_Combined_v3}"

echo "[INFO] Starting 4-epoch fine-tune on complete dataset."
echo "[INFO] dataset_name = $DATASET_NAME -> ckpts/$DATASET_NAME/"
echo "[INFO] vocab         : /root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"
echo "[INFO] Log           : $LOG"

nohup accelerate launch \
    f5tts/lib/python3.12/site-packages/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name "$DATASET_NAME" \
    --finetune \
    --pretrain /root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female/model_extended.pt \
    --tokenizer custom \
    --tokenizer_path /root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt \
    --batch_size_per_gpu 4 \
    --batch_size_type sample \
    --epochs 4 \
    --num_warmup_updates 100 \
    --save_per_updates 300 \
    --last_per_updates 300 \
    --keep_last_n_checkpoints 1 \
    --learning_rate 1e-5 \
    --log_samples \
    > "$LOG" 2>&1 &

disown
echo "[INFO] Launched training in background (PID $!)."
