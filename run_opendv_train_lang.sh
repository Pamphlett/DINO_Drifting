#!/usr/bin/env bash
set -euo pipefail

# ---- GPU visible devices ----
export CUDA_VISIBLE_DEVICES=1,2,3,4
NGPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

OPENDV_ROOT="/pfs/pengyu/OpenDV-YouTube"
OPENDV_LANG_ROOT="/pfs/pengyu/OpenDV-YouTube-Language"
LANG_CACHE_TRAIN="${OPENDV_LANG_ROOT}/mini_train_cache.json"
LANG_CACHE_VAL="${OPENDV_LANG_ROOT}/mini_val_cache.json"

MASTER_ADDR="127.0.0.1"
MASTER_PORT="29502"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
DST_ROOT="/cpfs/pengyu/DINO-Foresight/logs/dino_foresight_lowres_opendv"
DST_PATH="${DST_ROOT}/${RUN_TS}"
DATALOADER_LOG_PATH="${DST_PATH}/dataloader_debug.log"
DDP_STAGE_LOG_PATH="${DST_PATH}/ddp_stage.log"

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

CLIP_LOCAL_ONLY_ARGS=()
if [[ "${CLIP_LOCAL_ONLY}" == "1" || "${CLIP_LOCAL_ONLY,,}" == "true" ]]; then
  CLIP_LOCAL_ONLY_ARGS+=(--clip_local_files_only)
fi

echo "Starting ${NGPU}-GPU training on OpenDV (with language condition)..."
echo "Logs/checkpoints will be saved to: ${DST_PATH}"
echo "DINO_REPO=${DINO_REPO}"
echo "HF_HOME=${HF_HOME}"
echo "CLIP_CACHE_DIR=${CLIP_CACHE_DIR}"

# Uses language annotation clips + precomputed text tokens from each clip folder.
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" /cpfs/pengyu/DINO-Foresight/train.py \
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
  --clip_cache_dir "${CLIP_CACHE_DIR}" \
  "${CLIP_LOCAL_ONLY_ARGS[@]}" \
  --clip_max_length 77 \
  --num_gpus "${NGPU}" --strategy ddp --ddp_timeout 120 --find_unused_parameters \
  --precision 16-mixed --num_workers 6 --num_workers_val 6 --dataloader_timeout 120 \
  --dataloader_log_path "${DATALOADER_LOG_PATH}" --dataloader_log_every 500 \
  --ddp_stage_log_path "${DDP_STAGE_LOG_PATH}" \
  --batch_size 16 --sequence_length 5 --img_size 224,448 \
  --hidden_dim 1152 --heads 8 --layers 12 --dropout 0.1 \
  --single_step_sample_train --lr_base 8e-5 --use_drifting_loss --noise_dim 256 --drift_temperatures 0.02,0.05,0.2 \
  --masking simple_replace --random_horizontal_flip --random_crop --use_fc_bias --gclip 2.0 \
  --dinov2_variant vitb14_reg --d_layers 2,5,8,11 --train_mask_mode full_mask \
  --pca_ckpt "/cpfs/pengyu/DINO-Foresight/dinov2_pca_224_l[2,_5,_8,_11]_1152.pth" \
  --max_epochs 10 --save_every_n_steps 10000 --num_sanity_val_steps 5 \
  --limit_val_batches 1.0 --val_check_interval 1.0 \
  --dst_path "${DST_PATH}" \
  --wandb_project dino-foresight --wandb_name opendv_mini_b16_6gpu_lang \
  --wandb_mode online
