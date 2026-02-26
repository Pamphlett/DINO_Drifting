#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ckpt_path> [eval_modality]"
  echo "  eval_modality: segm | depth | surface_normals (optional)"
  exit 1
fi

CKPT_PATH="$1"
EVAL_MODALITY="${2:-}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

# ---- GPU visible devices ----
export CUDA_VISIBLE_DEVICES=1,2,3,4
NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

OPENDV_ROOT="/pfs/pengyu/OpenDV-YouTube"
OPENDV_LANG_ROOT="/pfs/pengyu/OpenDV-YouTube-Language"
LANG_CACHE_TRAIN="${OPENDV_LANG_ROOT}/mini_train_cache.json"
LANG_CACHE_VAL="${OPENDV_LANG_ROOT}/mini_val_cache.json"

MASTER_ADDR="127.0.0.1"
MASTER_PORT="29512"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
CKPT_TAG="$(basename "${CKPT_PATH}" .ckpt)"
DST_ROOT="/cpfs/pengyu/DINO-Foresight/logs/dino_foresight_lowres_opendv_eval_noallgather"
DST_PATH="${DST_ROOT}/${RUN_TS}_${CKPT_TAG}"
DATALOADER_LOG_PATH="${DST_PATH}/dataloader_debug.log"
DDP_STAGE_LOG_PATH="${DST_PATH}/ddp_stage.log"
LOG_PATH="${DST_PATH}/eval.log"

: "${NCCL_ASYNC_ERROR_HANDLING:=1}"
: "${TORCH_NCCL_BLOCKING_WAIT:=1}"
: "${DINO_REPO:=/cpfs/pengyu/.cache/torch/hub/facebookresearch_dinov2_main}"
: "${HF_HOME:=/cpfs/pengyu/hfcaches}"
: "${CLIP_CACHE_DIR:=${HF_HOME}}"
: "${CLIP_LOCAL_ONLY:=1}"
export NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_BLOCKING_WAIT
export DINO_REPO
export HF_HOME
export CLIP_CACHE_DIR
export CLIP_LOCAL_ONLY

mkdir -p "${DST_PATH}"

# Write all stdout/stderr to text log while keeping terminal output.
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Starting ${NGPU}-GPU OpenDV evaluation (no cross-rank negative gather)..."
echo "Checkpoint: ${CKPT_PATH}"
echo "Eval outputs will be saved to: ${DST_PATH}"
echo "Eval log: ${LOG_PATH}"
echo "DINO_REPO=${DINO_REPO}"
echo "HF_HOME=${HF_HOME}"

EXTRA_ARGS=()
if [[ -n "${EVAL_MODALITY}" ]]; then
  EXTRA_ARGS+=(--eval_modality "${EVAL_MODALITY}")
fi

torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" /cpfs/pengyu/DINO-Foresight/train.py \
  --dataset opendv \
  --opendv_root "${OPENDV_ROOT}" \
  --opendv_lang_root "${OPENDV_LANG_ROOT}" \
  --opendv_use_lang_annos \
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}" \
  --opendv_lang_cache_val "${LANG_CACHE_VAL}" \
  --num_gpus "${NGPU}" --strategy ddp --ddp_timeout 120 --find_unused_parameters \
  --precision 16-mixed --num_workers 4 --num_workers_val 4 --dataloader_timeout 120 \
  --dataloader_log_path "${DATALOADER_LOG_PATH}" --dataloader_log_every 500 \
  --ddp_stage_log_path "${DDP_STAGE_LOG_PATH}" \
  --batch_size 4 --accum_iter 1 \
  --sequence_length 5 --img_size 224,448 \
  --hidden_dim 768 --heads 8 --layers 8 --dropout 0.1 \
  --single_step_sample_train \
  --use_drifting_loss --noise_dim 256 \
  --drift_temperatures 0.02,0.05,0.2 \
  --drift_step_size 0.1 \
  --drift_anchor_weight 0.3 \
  --drift_v_clip 20 \
  --drift_log_interval 20 --drift_antisymmetry_interval 400 \
  --drift_metric_token_cap 256 --drift_diversity_k 2 \
  --lr_base 1e-5 --warmup_p 0.05 --gclip 1.0 \
  --masking simple_replace --use_fc_bias \
  --dinov2_variant vitb14_reg --d_layers 2,5,8,11 --train_mask_mode full_mask \
  --pca_ckpt "/cpfs/pengyu/DINO-Foresight/dinov2_pca_224_l[2,_5,_8,_11]_1152.pth" \
  --max_epochs 1 --num_sanity_val_steps 0 --limit_val_batches 1.0 --val_check_interval 1.0 \
  --eval_ckpt_only \
  --ckpt "${CKPT_PATH}" \
  --dst_path "${DST_PATH}" \
  --wandb_mode disabled \
  --use_csv_logger --csv_logger_name csv_eval \
  "${EXTRA_ARGS[@]}"

echo "Evaluation completed."
echo "Check results under: ${DST_PATH}"
