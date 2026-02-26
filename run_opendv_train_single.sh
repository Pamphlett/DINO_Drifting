#!/usr/bin/env bash
set -euo pipefail

OPENDV_ROOT="/cpfs/pengyu/OpenDV-YouTube"
OPENDV_LANG_ROOT="/cpfs/pengyu/OpenDV-YouTube-Language"
LANG_CACHE_TRAIN="${OPENDV_LANG_ROOT}/mini_train_cache.json"
LANG_CACHE_VAL="${OPENDV_LANG_ROOT}/mini_val_cache.json"

DST_PATH="/mnt/hdd1/pengyu/logs/dino_foresight_lowres_opendv_single"
DATALOADER_LOG_PATH="${DST_PATH}/dataloader_debug.log"
DDP_STAGE_LOG_PATH="${DST_PATH}/ddp_stage.log"

echo "Starting single-GPU training on OpenDV (no language)..."

python /cpfs/pengyu/DINO-Foresight/train.py \
  --dataset opendv \
  --opendv_root "${OPENDV_ROOT}" \
  --opendv_lang_root "${OPENDV_LANG_ROOT}" \
  --opendv_use_lang_annos \
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}" \
  --opendv_lang_cache_val "${LANG_CACHE_VAL}" \
  --num_gpus 1 \
  --precision 16-mixed --num_workers 6 --num_workers_val 6 --dataloader_timeout 120 \
  --dataloader_log_path "${DATALOADER_LOG_PATH}" --dataloader_log_every 500 \
  --ddp_stage_log_path "${DDP_STAGE_LOG_PATH}" \
  --batch_size 16 --sequence_length 5 --img_size 196,392 \
  --hidden_dim 768 --heads 8 --layers 8 --dropout 0.1 \
  --single_step_sample_train --lr_base 8e-5 --loss_type SmoothL1 \
  --masking simple_replace --seperable_attention --random_horizontal_flip --random_crop --use_fc_bias \
  --dinov2_variant vitb14_reg --d_layers 5,11 --train_mask_mode full_mask \
  --max_epochs 10 --save_every_n_steps 10000 --num_sanity_val_steps 5 \
  --limit_val_batches 1.0 --val_check_interval 1.0 \
  --dst_path "${DST_PATH}" \
  --wandb_project dino-foresight --wandb_name opendv_mini_b16_1gpu_nolang \
  --wandb_mode online
