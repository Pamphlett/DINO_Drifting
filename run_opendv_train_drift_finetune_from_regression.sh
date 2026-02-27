#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1,2,3,4
NGPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

OPENDV_ROOT="/pfs/pengyu/OpenDV-YouTube"
OPENDV_LANG_ROOT="/pfs/pengyu/OpenDV-YouTube-Language"
LANG_CACHE_TRAIN="${OPENDV_LANG_ROOT}/mini_train_cache.json"
LANG_CACHE_VAL="${OPENDV_LANG_ROOT}/mini_val_cache.json"

MASTER_ADDR="127.0.0.1"
MASTER_PORT="29502"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
DST_ROOT="/cpfs/pengyu/DINO-Foresight/logs/dino_foresight_lowres_opendv_drift_finetune"
DST_PATH="${DST_ROOT}/${RUN_TS}"
LOG_PATH="${DST_PATH}/train.log"

: "${NCCL_ASYNC_ERROR_HANDLING:=1}"
: "${TORCH_NCCL_BLOCKING_WAIT:=1}"
: "${DINO_REPO:=/cpfs/pengyu/.cache/torch/hub/facebookresearch_dinov2_main}"
: "${HF_HOME:=/cpfs/pengyu/hfcaches}"
: "${CLIP_CACHE_DIR:=${HF_HOME}}"
: "${CLIP_LOCAL_ONLY:=1}"
export NCCL_ASYNC_ERROR_HANDLING TORCH_NCCL_BLOCKING_WAIT DINO_REPO HF_HOME CLIP_CACHE_DIR CLIP_LOCAL_ONLY

mkdir -p "${DST_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

CKPT="/cpfs/pengyu/DINO-Foresight/logs/dino_foresight_lowres_opendv_drift_noallgather/20260215_011651/checkpoints/epoch=9-step=535547.ckpt"

echo "Starting drifting finetune from regression checkpoint..."
echo "Checkpoint: ${CKPT}"
echo "Output: ${DST_PATH}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" \
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
  /cpfs/pengyu/DINO-Foresight/train.py \
  --dataset opendv \
  --opendv_root "${OPENDV_ROOT}" \
  --opendv_lang_root "${OPENDV_LANG_ROOT}" \
  --opendv_use_lang_annos \
  --opendv_lang_cache_train "${LANG_CACHE_TRAIN}" \
  --opendv_lang_cache_val "${LANG_CACHE_VAL}" \
  --ckpt "${CKPT}" \
  --ckpt_weights_only \
  --num_gpus "${NGPU}" --strategy ddp --ddp_timeout 120 --find_unused_parameters \
  --precision 16-mixed --num_workers 4 --num_workers_val 4 --dataloader_timeout 120 \
  --batch_size 4 --accum_iter 1 \
  --sequence_length 5 --img_size 224,448 \
  --hidden_dim 768 --heads 8 --layers 8 --dropout 0.1 \
  --single_step_sample_train \
  --use_drifting_loss \
  --drift_temperatures 0.02,0.05,0.2 \
  --drift_v_clip 20 \
  --drift_train_token_cap 0 \
  --drift_diversity_weight 0.1 \
  --drift_log_interval 20 --drift_antisymmetry_interval 400 \
  --drift_metric_token_cap 256 --drift_diversity_k 2 \
  --drift_noise_dim 256 \
  --lr_base 5e-5 --warmup_p 0.1 --gclip 1.0 \
  --masking simple_replace --random_horizontal_flip --random_crop --use_fc_bias \
  --dinov2_variant vitb14_reg --d_layers 2,5,8,11 --train_mask_mode full_mask \
  --pca_ckpt "/cpfs/pengyu/DINO-Foresight/dinov2_pca_224_l[2,_5,_8,_11]_1152.pth" \
  --max_epochs 10 --num_sanity_val_steps 0 --limit_val_batches 0.2 --val_check_interval 0.25 \
  --dst_path "${DST_PATH}" \
  --wandb_project dino-foresight --wandb_name drift_finetune_from_regression \
  --wandb_mode online \
  --use_tensorboard --tensorboard_name tensorboard \
  --use_csv_logger --csv_logger_name csv
