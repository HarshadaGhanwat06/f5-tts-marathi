#!/usr/bin/env bash
# =============================================================================
# run_train_4epoch_v5.sh
#
# Fine-tune on the pronunciation-focused v5 dataset:
#   Cartesia_Rasa_Combined_v5 = v3 set (2663) + candra_e (~210) + prono vowel
#                               (528 ॅ/ॉ/ँ/ॉं samples).
# 4 epochs, warm-started from the most recent trained checkpoint (v4 or v3).
#
# Dataset:
#   preprocessed  -> f5tts/data/Cartesia_Rasa_Combined_v5_custom (v5 prep)
#   checkpoint    -> ckpts/Cartesia_Rasa_Combined_v5/
# Pretrain (warm start, 139 embeddings):
#   ckpts/Rasa_Marathi_Emotion_Female_v5/model_extended.pt (v5 prepare_pretrain)
# Vocab (extended, 138 tokens):
#   f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt
#
# Run on server (after prepare_dataset_v5.sh + prepare_pretrain_v5.sh):
#   bash scripts/run_train_4epoch_v5.sh
#
# Monitor:
#   tail -f /root/f5-tts-marathi/logs/training_cartesia_v5.log
# =============================================================================
set -euo pipefail

REPO_ROOT=/root/f5-tts-marathi
cd "$REPO_ROOT"
mkdir -p logs
LOG=logs/training_cartesia_v5.log

DATASET_NAME="${DATASET_NAME:-Cartesia_Rasa_Combined_v5}"

echo "[INFO] Starting 4-epoch fine-tune (ॅ/ॉ/ँ/ॉं pronunciation focus)."
echo "[INFO] dataset_name = $DATASET_NAME -> ckpts/$DATASET_NAME/"
echo "[INFO] pretrain     : /root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female_v5/model_extended.pt (warm start)"
echo "[INFO] vocab        : /root/f5-tts-marathi/f5tts/data/Rasa_Marathi_Emotion_Female/vocab_extended.txt"
echo "[INFO] Log          : $LOG"

ACCELERATE="$REPO_ROOT/f5tts/bin/accelerate"
FINETUNE_CLI="$REPO_ROOT/f5tts/lib/python3.12/site-packages/f5_tts/train/finetune_cli.py"

nohup "$ACCELERATE" launch \
    "$FINETUNE_CLI" \
    --exp_name F5TTS_v1_Base \
    --dataset_name "$DATASET_NAME" \
    --finetune \
    --pretrain /root/f5-tts-marathi/f5tts/lib/python3.12/ckpts/Rasa_Marathi_Emotion_Female_v5/model_extended.pt \
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
echo "[INFO] Confirm in the log: 'text_num_embeds' = 139 and dataset rows >= 3400."