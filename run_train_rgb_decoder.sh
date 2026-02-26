#!/usr/bin/env bash
set -euo pipefail

: "${DINO_REPO:=/cpfs/pengyu/.cache/torch/hub/facebookresearch_dinov2_main}"

: "${WANDB_PROJECT:=dino-foresight}"
: "${WANDB_NAME:=rgb_decoder}"
: "${WANDB_MODE:=online}"

CUDA_VISIBLE_DEVICES=1,2,3,4 python train_rgb_decoder.py \
  --train_root /mnt/hdd1/pengyu/OpenDV-YouTube/full_images \
  --val_root /mnt/hdd1/pengyu/OpenDV-YouTube/val_images \
  --img_size 196,392 \
  --dinov2_variant vitb14_reg \
  --d_layers 2,5,8,11 \
  --batch_size 8 \
  --num_workers 8 \
  --num_gpus 4 \
  --find_unused_parameters \
  --precision 16-mixed \
  --dst_path ./logs/rgb_decoder \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --wandb_mode "${WANDB_MODE}"
