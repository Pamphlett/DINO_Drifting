#!/usr/bin/env bash
set -euo pipefail

: "${DINO_REPO:=/cpfs/pengyu/.cache/torch/hub/facebookresearch_dinov2_main}"

: "${WANDB_PROJECT:=dino-foresight}"
: "${WANDB_NAME:=rgb_decoder}"
: "${WANDB_MODE:=online}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
DST_ROOT="./logs/rgb_decoder"
DST_PATH="${DST_ROOT}/${RUN_TS}"

mkdir -p "${DST_PATH}"

echo "RGB decoder logs/checkpoints will be saved to: ${DST_PATH}"

CUDA_VISIBLE_DEVICES=1,2,3,4 python train_rgb_decoder.py \
  --train_root /mnt/hdd1/pengyu/OpenDV-YouTube/full_images \
  --val_root /mnt/hdd1/pengyu/OpenDV-YouTube/val_images \
  --img_size 224,448 \
  --dinov2_variant vitb14_reg \
  --d_layers 2,5,8,11 \
  --batch_size 8 \
  --num_workers 8 \
  --num_gpus 4 \
  --find_unused_parameters \
  --precision 16-mixed \
  --dst_path "${DST_PATH}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --wandb_mode "${WANDB_MODE}"
