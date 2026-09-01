#!/usr/bin/env bash
# =============================================================================
# run_train_4epoch.sh
#
# Fine-tune Marathi female F5-TTS on the COMPLETE combined dataset
# (1630 Rasa ~3h + 1033 Cartesia code-mixed ~88min) for 4 epochs.
#
# This is the SAME recipe used to build Cartesia_Rasa_Combined_3h, but with
# --epochs 4 and now that Cartesia dataset is fully generated (all 1033
# samples) the dataset_name config picks up the complete metadata.
#
# Prereqs (already done):
#   bash scripts/run_prepare_dataset.sh   (rebuild combined dataset)
#
# Run on the SERVER:
#   bash scripts/run_train_4epoch.sh
#
# Monitor:
#   tail -f /root/f5-tts-marathi/logs/training_cartesia_full_4epochs.log
# =============================================================================
set -euo pipefail

REPO_ROOT=/root/f5-tts-marathi
cd "$REPO_ROOT"

LOG=logs/training_cartesia_full_4epochs.log
mkdir -p logs

echo "[INFO] Starting 4-epoch fine-tune on complete Cartesia+Rasa dataset."
echo "[INFO] Log: $LOG"

nohup accelerate launch \
    f5tts/lib/python3.12/site-packages/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name Cartesia_Rasa_Combined_3h \
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
