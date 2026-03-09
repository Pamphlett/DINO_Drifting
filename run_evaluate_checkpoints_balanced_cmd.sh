#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/path/to/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output}"
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
OPENDV_LANG_ROOT="${OPENDV_LANG_ROOT:-/path/to/OpenDV-YouTube-Language}"
LANG_CACHE_TRAIN="${LANG_CACHE_TRAIN:-${OPENDV_LANG_ROOT}/mini_train_cache.json}"
LANG_CACHE_VAL="${LANG_CACHE_VAL:-${OPENDV_LANG_ROOT}/mini_val_cache.json}"
RGB_DECODER_PATH="${RGB_DECODER_PATH:-}"
EVAL_SUBSET_INDEX_FILE="${EVAL_SUBSET_INDEX_FILE:-${OUTPUT_DIR}/eval_subset_indices.json}"

DATA_SPLIT="${DATA_SPLIT:-val}"
EVAL_SUBSET_SIZE="${EVAL_SUBSET_SIZE:-100}"
EVAL_SUBSET_SEED="${EVAL_SUBSET_SEED:-123}"
QUAL_NUM_SAMPLES="${QUAL_NUM_SAMPLES:-6}"
QUAL_SEED="${QUAL_SEED:-7}"
NUM_SAMPLES_PER_INPUT="${NUM_SAMPLES_PER_INPUT:-1}"
MAX_CKPTS_TO_COMPARE="${MAX_CKPTS_TO_COMPARE:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-}"

# Balanced command groups: keep both straight and turning clips in the eval subset.
STRAIGHT_GROUP="${STRAIGHT_GROUP:-straight=go straight,straight}"
TURN_GROUP="${TURN_GROUP:-turn=turn left,turn right,turn,left,right}"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${LANG_CACHE_VAL}" && "${DATA_SPLIT}" == "val" ]]; then
  echo "Validation cache not found: ${LANG_CACHE_VAL}" >&2
  exit 1
fi

if [[ ! -f "${LANG_CACHE_TRAIN}" && "${DATA_SPLIT}" == "train" ]]; then
  echo "Train cache not found: ${LANG_CACHE_TRAIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python evaluate_checkpoints.py
  --ckpt_dir "${CKPT_DIR}"
  --data_root "${DATA_ROOT}"
  --dataset opendv
  --data_split "${DATA_SPLIT}"
  --opendv_lang_root "${OPENDV_LANG_ROOT}"
  --opendv_use_lang_annos true
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}"
  --opendv_lang_cache_val "${LANG_CACHE_VAL}"
  --eval_text_field cmd
  --eval_cmd_groups "${STRAIGHT_GROUP}" "${TURN_GROUP}"
  --eval_subset_size "${EVAL_SUBSET_SIZE}"
  --eval_subset_seed "${EVAL_SUBSET_SEED}"
  --eval_subset_index_file "${EVAL_SUBSET_INDEX_FILE}"
  --qual_num_samples "${QUAL_NUM_SAMPLES}"
  --qual_seed "${QUAL_SEED}"
  --num_samples_per_input "${NUM_SAMPLES_PER_INPUT}"
  --max_ckpts_to_compare "${MAX_CKPTS_TO_COMPARE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${RGB_DECODER_PATH}" ]]; then
  CMD+=(--rgb_decoder_path "${RGB_DECODER_PATH}")
fi

if [[ -n "${DEVICE}" ]]; then
  CMD+=(--device "${DEVICE}")
fi

echo "Running balanced checkpoint evaluation..."
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
