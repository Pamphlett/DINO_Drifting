  #!/usr/bin/env bash
  set -euo pipefail

  # ---- GPU visible devices ----
  export CUDA_VISIBLE_DEVICES=1,2,3,4
  NGPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
  # 4090-friendly defaults (override by env when needed).
  BATCH_SIZE="${BATCH_SIZE:-2}"
  ACCUM_ITER="${ACCUM_ITER:-1}"
  DRIFT_TEMP_TOKEN_CAP="${DRIFT_TEMP_TOKEN_CAP:-256}"
  DRIFT_METRIC_TOKEN_CAP="${DRIFT_METRIC_TOKEN_CAP:-128}"
  DRIFT_TRAIN_TOKEN_CAP="${DRIFT_TRAIN_TOKEN_CAP:-512}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  NUM_WORKERS_VAL="${NUM_WORKERS_VAL:-0}"
  DATALOADER_TIMEOUT="${DATALOADER_TIMEOUT:-0}"
  LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0}"
  VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1.0}"
  NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
  DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
  DRIFT_TEMPERATURES="${DRIFT_TEMPERATURES:-0.05,0.1,0.2}"
  DRIFT_ADAPTIVE_TEMP="${DRIFT_ADAPTIVE_TEMP:-0}"
  DRIFT_V_CLIP="${DRIFT_V_CLIP:-10}"

  OPENDV_ROOT="/pfs/pengyu/OpenDV-YouTube"
  OPENDV_LANG_ROOT="/pfs/pengyu/OpenDV-YouTube-Language"
  LANG_CACHE_TRAIN="${OPENDV_LANG_ROOT}/mini_train_cache.json"
  LANG_CACHE_VAL="${OPENDV_LANG_ROOT}/mini_val_cache.json"

  MASTER_ADDR="127.0.0.1"
  MASTER_PORT="29503"
  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  DST_ROOT="/cpfs/pengyu/DINO-Foresight/logs/dino_foresight_lowres_opendv_drift_noallgather_optimized"
  DST_PATH="${DST_ROOT}/${RUN_TS}"
  DATALOADER_LOG_PATH="${DST_PATH}/dataloader_debug.log"
  DDP_STAGE_LOG_PATH="${DST_PATH}/ddp_stage.log"
  LOG_PATH="${DST_PATH}/train.log"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TRAIN_PY="${SCRIPT_DIR}/train.py"

  : "${NCCL_ASYNC_ERROR_HANDLING:=1}"
  : "${TORCH_NCCL_BLOCKING_WAIT:=1}"
  : "${DINO_REPO:=/cpfs/pengyu/.cache/torch/hub/facebookresearch_dinov2_main}"
  : "${HF_HOME:=/cpfs/pengyu/hfcaches}"
  : "${CLIP_CACHE_DIR:=${HF_HOME}}"
  : "${CLIP_LOCAL_ONLY:=1}"
  : "${PYTORCH_CUDA_ALLOC_CONF:=max_split_size_mb:128,garbage_collection_threshold:0.8}"
  export NCCL_ASYNC_ERROR_HANDLING
  export TORCH_NCCL_BLOCKING_WAIT
  export DINO_REPO
  export HF_HOME
  export CLIP_CACHE_DIR
  export CLIP_LOCAL_ONLY
  export PYTORCH_CUDA_ALLOC_CONF

  mkdir -p "${DST_PATH}"

  # Write all stdout/stderr to text log while keeping terminal output.
  exec > >(tee -a "${LOG_PATH}") 2>&1

  echo "Starting ${NGPU}-GPU optimized drifting training on OpenDV..."
  echo "Logs/checkpoints will be saved to: ${DST_PATH}"
  echo "Train log: ${LOG_PATH}"
  echo "CSV metrics dir: ${DST_PATH}/csv"
  echo "DINO_REPO=${DINO_REPO}"
  echo "HF_HOME=${HF_HOME}"
  echo "TRAIN_PY=${TRAIN_PY}"
  echo "DDP_TIMEOUT=${DDP_TIMEOUT}"
  echo "LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES} VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL}"
  echo "DRIFT: temps=${DRIFT_TEMPERATURES} adaptive=${DRIFT_ADAPTIVE_TEMP} vclip=${DRIFT_V_CLIP}"

  DRIFT_ADAPTIVE_FLAG=()
  if [ "${DRIFT_ADAPTIVE_TEMP}" = "1" ]; then
    DRIFT_ADAPTIVE_FLAG+=(--drift_adaptive_temp)
  fi

  torchrun --standalone --nnodes=1 --nproc_per_node="${NGPU}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${TRAIN_PY}" \
    --dataset opendv \
    --opendv_root "${OPENDV_ROOT}" \
    --opendv_lang_root "${OPENDV_LANG_ROOT}" \
    --opendv_use_lang_annos \
    --opendv_lang_cache_train "${LANG_CACHE_TRAIN}" \
    --opendv_lang_cache_val "${LANG_CACHE_VAL}" \
    --num_gpus "${NGPU}" --strategy ddp --ddp_timeout "${DDP_TIMEOUT}" --find_unused_parameters \
    --precision 16-mixed --num_workers "${NUM_WORKERS}" --num_workers_val "${NUM_WORKERS_VAL}" --dataloader_timeout "${DATALOADER_TIMEOUT}" \
    --dataloader_log_path "${DATALOADER_LOG_PATH}" --dataloader_log_every 0 \
    --ddp_stage_log_path "${DDP_STAGE_LOG_PATH}" \
    --batch_size "${BATCH_SIZE}" --accum_iter "${ACCUM_ITER}" \
    --sequence_length 5 --img_size 224,448 \
    --hidden_dim 768 --heads 8 --layers 8 --dropout 0.1 \
    --single_step_sample_train \
    --use_drifting_loss \
    --drift_temperatures "${DRIFT_TEMPERATURES}" \
    "${DRIFT_ADAPTIVE_FLAG[@]}" \
    --drift_temp_ema_decay 0.95 \
    --drift_temp_update_interval 20 \
    --drift_temp_token_cap "${DRIFT_TEMP_TOKEN_CAP}" \
    --drift_temp_min_scale 0.5 \
    --drift_temp_max_scale 2.0 \
    --drift_temp_ref_dist 0.0 \
    --drift_v_clip "${DRIFT_V_CLIP}" \
    --drift_log_interval 20 --drift_antisymmetry_interval 400 \
    --drift_metric_token_cap "${DRIFT_METRIC_TOKEN_CAP}" \
    --drift_train_token_cap "${DRIFT_TRAIN_TOKEN_CAP}" \
    --drift_diversity_k 2 \
    --lr_base 1e-5 --warmup_p 0.05 --gclip 1.0 \
    --masking simple_replace --random_horizontal_flip --random_crop --use_fc_bias \
    --dinov2_variant vitb14_reg --d_layers 2,5,8,11 --train_mask_mode full_mask \
    --pca_ckpt "/cpfs/pengyu/DINO-Foresight/dinov2_pca_224_l[2,_5,_8,_11]_1152.pth" \
    --pca_whiten_alpha 0.5 --pca_whiten_eps 1e-8 \
    --max_epochs 10 --num_sanity_val_steps "${NUM_SANITY_VAL_STEPS}" --limit_val_batches "${LIMIT_VAL_BATCHES}" --val_check_interval "${VAL_CHECK_INTERVAL}" \
    --dst_path "${DST_PATH}" \
    --wandb_project dino-foresight --wandb_name drift_opt_fp16_ddp_noallgather_e10 \
    --wandb_mode online \
    --use_tensorboard --tensorboard_name tensorboard \
    --use_csv_logger --csv_logger_name csv
