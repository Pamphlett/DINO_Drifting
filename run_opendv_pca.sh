#!/usr/bin/env bash
set -euo pipefail

OPENDV_ROOT="${OPENDV_ROOT:-/cpfs/pengyu/OpenDV-YouTube}"
OPENDV_LANG_ROOT="${OPENDV_LANG_ROOT:-/cpfs/pengyu/OpenDV-YouTube-Language}"
LANG_CACHE_TRAIN="${LANG_CACHE_TRAIN:-${OPENDV_LANG_ROOT}/mini_train_cache.json}"
LANG_CACHE_VAL="${LANG_CACHE_VAL:-${OPENDV_LANG_ROOT}/mini_val_cache.json}"

# Sampling strategy per clip
OPENDV_CLIP_SAMPLE="${OPENDV_CLIP_SAMPLE:-8}"   # uniformly sample K frames per clip
OPENDV_MAX_CLIPS="${OPENDV_MAX_CLIPS:-}"        # optional cap on clips
OPENDV_CLIP_SAMPLE_MODE="${OPENDV_CLIP_SAMPLE_MODE:-uniform}"

# Model/feature settings
FEATURE_EXTRACTOR="${FEATURE_EXTRACTOR:-dinov2}"
DLAYERS="${DLAYERS:-2,5,8,11}"
IMG_SIZE="${IMG_SIZE:-224,448}"
N_COMPONENTS="${N_COMPONENTS:-1152}"

# Runtime settings
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-123}"
INSPECT_TOKENS="${INSPECT_TOKENS:-0}"  # set to 1 to print token diagnostics once
INSPECT_TOKENS_ONLY="${INSPECT_TOKENS_ONLY:-0}"  # set to 1 to inspect one batch and skip save

echo "Running OpenDV PCA with per-clip uniform sampling..."

CMD=(python pca.py
  --dataset opendv
  --opendv_root "${OPENDV_ROOT}"
  --opendv_lang_root "${OPENDV_LANG_ROOT}"
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}"
  --opendv_lang_cache_val "${LANG_CACHE_VAL}"
  --opendv_clip_sample "${OPENDV_CLIP_SAMPLE}"
  --opendv_clip_sample_mode "${OPENDV_CLIP_SAMPLE_MODE}"
  --feature_extractor "${FEATURE_EXTRACTOR}"
  --dlayers "${DLAYERS}"
  --img_size "${IMG_SIZE}"
  --n_components "${N_COMPONENTS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --incremental
  --skip_eval
)

if [[ -n "${OPENDV_MAX_CLIPS}" ]]; then
  CMD+=(--opendv_max_clips "${OPENDV_MAX_CLIPS}")
fi

if [[ "${INSPECT_TOKENS}" != "0" ]]; then
  CMD+=(--inspect_tokens)
fi

if [[ "${INSPECT_TOKENS_ONLY}" != "0" ]]; then
  CMD+=(--inspect_tokens_only)
fi

echo "${CMD[@]}"
exec "${CMD[@]}"
