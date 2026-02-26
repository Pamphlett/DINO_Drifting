#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-/home/ruofei/pengyu/DINO-Foresight/dino-foresight/u7fvr3vo/checkpoints/last.ckpt}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/mnt/hdd1/pengyu/logs/dino_foresight_lowres_opendv_lang/pred_feats}"
OPENDV_ROOT="${OPENDV_ROOT:-/mnt/hdd1/pengyu/OpenDV-YouTube}"
OPENDV_LANG_ROOT="${OPENDV_LANG_ROOT:-/mnt/hdd1/pengyu/OpenDV-YouTube-Language}"
LANG_CACHE_TRAIN="${LANG_CACHE_TRAIN:-${OPENDV_LANG_ROOT}/mini_train_cache.json}"
LANG_CACHE_VAL="${LANG_CACHE_VAL:-${OPENDV_LANG_ROOT}/mini_val_cache.json}"
DEFAULT_LANG_FEAT_NAME="lang_clip_{start}_{end}.pt"
OPENDV_LANG_FEAT_NAME="${OPENDV_LANG_FEAT_NAME:-$DEFAULT_LANG_FEAT_NAME}"

SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-5}"
IMG_SIZE="${IMG_SIZE:-196,392}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
UNROLL_STEPS="${UNROLL_STEPS:-1}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
MAX_BATCHES="${MAX_BATCHES:-}"
CUDA_VISIBLE_DEVICES=2

EXTRA_ARGS=()
if [[ -n "${MAX_BATCHES}" ]]; then
  EXTRA_ARGS+=(--max_batches "${MAX_BATCHES}")
fi

rm -rf "${OUT_ROOT_BASE}/train" "${OUT_ROOT_BASE}/val"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python save_predicted_dino_features.py \
  --ckpt "${CKPT}" \
  --out_root "${OUT_ROOT_BASE}/train" \
  --dataset opendv \
  --opendv_root "${OPENDV_ROOT}" \
  --opendv_lang_root "${OPENDV_LANG_ROOT}" \
  --opendv_use_lang_annos \
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}" \
  --opendv_lang_cache_val "${LANG_CACHE_VAL}" \
  --opendv_use_lang_features \
  --opendv_lang_feat_name "${OPENDV_LANG_FEAT_NAME}" \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --subset train \
  --unroll_steps "${UNROLL_STEPS}" \
  --save_dtype "${SAVE_DTYPE}" \
  "${EXTRA_ARGS[@]}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python save_predicted_dino_features.py \
  --ckpt "${CKPT}" \
  --out_root "${OUT_ROOT_BASE}/val" \
  --dataset opendv \
  --opendv_root "${OPENDV_ROOT}" \
  --opendv_lang_root "${OPENDV_LANG_ROOT}" \
  --opendv_use_lang_annos \
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}" \
  --opendv_lang_cache_val "${LANG_CACHE_VAL}" \
  --opendv_use_lang_features \
  --opendv_lang_feat_name "${OPENDV_LANG_FEAT_NAME}" \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --subset val \
  --unroll_steps "${UNROLL_STEPS}" \
  --save_dtype "${SAVE_DTYPE}" \
  "${EXTRA_ARGS[@]}"
