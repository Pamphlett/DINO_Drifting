#!/usr/bin/env bash
set -euo pipefail

OPENDV_ROOT="/mnt/hdd1/pengyu/OpenDV-YouTube"
OPENDV_LANG_ROOT="/mnt/hdd1/pengyu/OpenDV-YouTube-Language"
LANG_CACHE_TRAIN="${OPENDV_LANG_ROOT}/mini_train_cache.json"
LANG_CACHE_VAL="${OPENDV_LANG_ROOT}/mini_val_cache.json"
CLIP_CACHE_DIR="/home/ruofei/.cache/huggingface/hub"

echo "Rebuilding offline CLIP features (train)..."
python build_opendv_lang_features.py \
  --opendv_root "${OPENDV_ROOT}" \
  --lang_cache "${LANG_CACHE_TRAIN}" \
  --split train \
  --feature_name lang_clip_{start}_{end}.pt \
  --clip_cache_dir "${CLIP_CACHE_DIR}" \
  --clip_local_files_only \
  --clip_max_length 77 \
  --batch_size 64 \
  --device cuda

echo "Rebuilding offline CLIP features (val)..."
python build_opendv_lang_features.py \
  --opendv_root "${OPENDV_ROOT}" \
  --lang_cache "${LANG_CACHE_VAL}" \
  --split val \
  --feature_name lang_clip_{start}_{end}.pt \
  --clip_cache_dir "${CLIP_CACHE_DIR}" \
  --clip_local_files_only \
  --clip_max_length 77 \
  --batch_size 64 \
  --device cuda

echo "Starting single-GPU training with online DINO + offline CLIP..."
python train.py \
  --dataset opendv \
  --opendv_root "${OPENDV_ROOT}" \
  --opendv_lang_root "${OPENDV_LANG_ROOT}" \
  --opendv_use_lang_annos \
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}" \
  --opendv_lang_cache_val "${LANG_CACHE_VAL}" \
  --opendv_use_lang_features \
  --opendv_lang_feat_name lang_clip_{start}_{end}.pt \
  --use_language_condition \
  --use_precomputed_text \
  --clip_max_length 77 \
  --num_gpus 1 --precision 16-mixed --num_workers 4 --num_workers_val 2 \
  --batch_size 4 --sequence_length 5 --img_size 224,448 \
  --hidden_dim 1152 --heads 8 --layers 12 --dropout 0.1 \
  --single_step_sample_train --lr_base 8e-5 --loss_type SmoothL1 \
  --masking simple_replace --seperable_attention --random_horizontal_flip --random_crop --use_fc_bias \
  --dinov2_variant vitb14_reg --d_layers 2,5,8,11 --train_mask_mode full_mask \
  --max_epochs 10 \
  --dst_path /logdir/dino_foresight_lowres_opendv_lang \
  --wandb_project dino-foresight --wandb_name opendv_b4_langclip_1gpu_e10 \
  --wandb_mode online
