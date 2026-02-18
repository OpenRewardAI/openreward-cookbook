#!/bin/bash
set -euo pipefail

# ============================================================================
# OpenReward + Miles training launch script
#
# Run from the miles repo root:
#   cd /path/to/miles
#   bash /path/to/this/run.sh [OPTIONS]
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Model & checkpoint:
  --model NAME              HF checkpoint name (default: Qwen/Qwen3-30B-A3B)
  --save DIR                Checkpoint save directory
  --load DIR                Checkpoint load directory
  --save-interval N         Save every N steps (default: 10)

Optimizer:
  --lr FLOAT                Learning rate (default: 1e-5)
  --weight-decay FLOAT      Weight decay (default: 0.0)
  --adam-beta1 FLOAT        Adam beta1 (default: 0.9)
  --adam-beta2 FLOAT        Adam beta2 (default: 0.999)
  --adam-eps FLOAT           Adam epsilon (default: 1e-8)
  --lr-decay-style STR      LR schedule: constant, cosine, linear (default: constant)

Cluster:
  --actor-gpus N            GPUs for training (default: 4)
  --rollout-gpus N          GPUs for rollout (default: 4)
  --tp N                    GPUs per rollout engine / tensor parallel (default: 4)
  --actor-nodes N           Number of training nodes (default: 1)
  --train-backend STR       Training backend: fsdp or megatron (default: fsdp)

Rollout:
  --rollout-batch-size N    Prompts per rollout batch (default: 32)
  --n-samples N             Rollouts per prompt (default: 16)
  --max-response-len N      Max response tokens per turn (default: 4096)
  --max-total-len N         Max total tokens per rollout sequence (default: 8192)
  --max-tokens-per-gpu N    Max tokens per GPU in training (OOM prevention, default: 8192)
  --temperature FLOAT       Sampling temperature (default: 1.0)
  --num-rollout N           Total training rounds (default: 3000)

Data:
  --task-data PATH          Path to tasks.jsonl
  --config PATH             Path to train_config.yaml

Extra:
  --wandb-project STR       Wandb project name
  --wandb-run STR           Wandb run name
  -- EXTRA_ARGS...          Pass remaining args directly to train_async.py
EOF
    exit 0
}

# ============================================================================
# Defaults
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="Qwen/Qwen3-30B-A3B"
SAVE_DIR="${SCRIPT_DIR}/checkpoints/"
LOAD_DIR="${SCRIPT_DIR}/checkpoints/"
SAVE_INTERVAL=10

LR=1e-5
WEIGHT_DECAY=0.0
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPS=1e-8
LR_DECAY_STYLE=constant

ACTOR_GPUS=4
ROLLOUT_GPUS=4
TP=4
ACTOR_NODES=1
TRAIN_BACKEND=fsdp

ROLLOUT_BATCH_SIZE=32
N_SAMPLES=16
MAX_RESPONSE_LEN=4096
MAX_TOTAL_LEN=8192
MAX_TOKENS_PER_GPU=8192
TEMPERATURE=1.0
NUM_ROLLOUT=3000

TASK_DATA="${SCRIPT_DIR}/tasks.jsonl"
TRAIN_CONFIG="${SCRIPT_DIR}/train_config.yaml"

WANDB_PROJECT="miles-openreward"
WANDB_RUN="miles-rl-openreward-$(date +%Y%m%d-%H%M%S)"
EXTRA_ARGS=()

# ============================================================================
# Parse CLI
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)           usage ;;
        --model)             MODEL="$2"; shift 2 ;;
        --save)              SAVE_DIR="$2"; shift 2 ;;
        --load)              LOAD_DIR="$2"; shift 2 ;;
        --save-interval)     SAVE_INTERVAL="$2"; shift 2 ;;
        --lr)                LR="$2"; shift 2 ;;
        --weight-decay)      WEIGHT_DECAY="$2"; shift 2 ;;
        --adam-beta1)        ADAM_BETA1="$2"; shift 2 ;;
        --adam-beta2)        ADAM_BETA2="$2"; shift 2 ;;
        --adam-eps)          ADAM_EPS="$2"; shift 2 ;;
        --lr-decay-style)    LR_DECAY_STYLE="$2"; shift 2 ;;
        --actor-gpus)        ACTOR_GPUS="$2"; shift 2 ;;
        --rollout-gpus)      ROLLOUT_GPUS="$2"; shift 2 ;;
        --tp)                TP="$2"; shift 2 ;;
        --actor-nodes)       ACTOR_NODES="$2"; shift 2 ;;
        --train-backend)     TRAIN_BACKEND="$2"; shift 2 ;;
        --rollout-batch-size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
        --n-samples)         N_SAMPLES="$2"; shift 2 ;;
        --max-response-len)  MAX_RESPONSE_LEN="$2"; shift 2 ;;
        --max-total-len)     MAX_TOTAL_LEN="$2"; shift 2 ;;
        --max-tokens-per-gpu) MAX_TOKENS_PER_GPU="$2"; shift 2 ;;
        --temperature)       TEMPERATURE="$2"; shift 2 ;;
        --num-rollout)       NUM_ROLLOUT="$2"; shift 2 ;;
        --task-data)         TASK_DATA="$2"; shift 2 ;;
        --config)            TRAIN_CONFIG="$2"; shift 2 ;;
        --wandb-project)     WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-run)         WANDB_RUN="$2"; shift 2 ;;
        --)                  shift; EXTRA_ARGS+=("$@"); break ;;
        *)                   echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Derived values
# ============================================================================
export OPENREWARD_API_KEY="${OPENREWARD_API_KEY:?Set OPENREWARD_API_KEY}"
export OPENREWARD_RUN_NAME="${WANDB_RUN}"
export TRAIN_CONFIG
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# global_batch_size = rollout_batch_size × n_samples ÷ num_steps_per_rollout
GLOBAL_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES ))

# ============================================================================
# Launch
# ============================================================================
echo "Starting OpenReward + Miles training"
echo "  Model:      ${MODEL}"
echo "  LR:         ${LR}"
echo "  Tasks:      ${TASK_DATA}"
echo "  Config:     ${TRAIN_CONFIG}"
echo "  Wandb:      ${WANDB_PROJECT} / ${WANDB_RUN}"
echo "  Batch:      ${ROLLOUT_BATCH_SIZE} prompts × ${N_SAMPLES} samples = ${GLOBAL_BATCH_SIZE}"
echo "  GPUs:       ${ACTOR_GPUS} train, ${ROLLOUT_GPUS} rollout (tp=${TP})"
echo "  MaxTok/GPU: ${MAX_TOKENS_PER_GPU}"
echo "  Backend:    ${TRAIN_BACKEND}"

python train_async.py \
    --actor-num-nodes "${ACTOR_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --rollout-num-gpus-per-engine "${TP}" \
    \
    --train-backend "${TRAIN_BACKEND}" \
    \
    --hf-checkpoint "${MODEL}" \
    --save "${SAVE_DIR}" \
    --load "${LOAD_DIR}" \
    --save-interval "${SAVE_INTERVAL}" \
    \
    --optimizer adam \
    --lr "${LR}" \
    --lr-decay-style "${LR_DECAY_STYLE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --adam-beta1 "${ADAM_BETA1}" \
    --adam-beta2 "${ADAM_BETA2}" \
    --adam-eps "${ADAM_EPS}" \
    \
    --advantage-estimator grpo \
    \
    --prompt-data "${TASK_DATA}" \
    --input-key prompt \
    --label-key label \
    --rollout-shuffle \
    --custom-generate-function-path generate.generate \
    --n-samples-per-prompt "${N_SAMPLES}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --num-steps-per-rollout 1 \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --rollout-max-response-len "${MAX_RESPONSE_LEN}" \
    --rollout-temperature "${TEMPERATURE}" \
    --num-rollout "${NUM_ROLLOUT}" \
    \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --gradient-checkpointing \
    \
    --sglang-mem-fraction-static 0.8 \
    \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN}" \
    --wandb-key "${WANDB_API_KEY}" \
    --wandb-group "${WANDB_PROJECT}" \
    \
    "${EXTRA_ARGS[@]}"