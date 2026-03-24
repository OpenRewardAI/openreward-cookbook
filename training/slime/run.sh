#!/bin/bash
set -euo pipefail

# ============================================================================
# OpenReward + SLIME training launch script
#
# Prerequisites:
#   1. SLIME installed with all dependencies
#   2. HF checkpoint converted to Megatron format (see README.md)
#   3. Tasks prepared with prepare_tasks.py
#   4. Environment variables set (OPENREWARD_API_KEY, WANDB_API_KEY)
#
# Usage:
#   bash run.sh [OPTIONS]
# ============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Model & checkpoint:
  --model NAME              HF model name or path (default: Qwen/Qwen3-30B-A3B)
  --megatron-ckpt PATH      Megatron checkpoint path (for training, from convert step)
  --model-args PATH         Path to SLIME model args script (from slime/scripts/models/)
  --save DIR                Checkpoint save directory
  --save-interval N         Save every N steps (default: 10)

Optimizer:
  --lr FLOAT                Learning rate (default: 1e-5)
  --weight-decay FLOAT      Weight decay (default: 0.0)
  --adam-beta1 FLOAT        Adam beta1 (default: 0.9)
  --adam-beta2 FLOAT        Adam beta2 (default: 0.999)
  --adam-eps FLOAT           Adam epsilon (default: 1e-8)
  --lr-decay-style STR      LR schedule: constant, cosine, linear (default: constant)

Cluster:
  --num-gpus N              Total GPUs (default: 4, uses --colocate mode)
  --tp N                    Tensor parallel size per rollout engine (default: 4)

Rollout:
  --rollout-batch-size N    Prompts per rollout batch (default: 32)
  --n-samples N             Rollouts per prompt (default: 16)
  --max-response-len N      Max response tokens per turn (default: 4096)
  --max-tokens-per-gpu N    Max tokens per GPU in training (OOM prevention, default: 8192)
  --temperature FLOAT       Sampling temperature (default: 1.0)
  --num-rollout N           Total training rounds (default: 3000)

Data:
  --task-data PATH          Path to tasks.jsonl
  --config PATH             Path to train_config.yaml

Extra:
  --wandb-project STR       Wandb project name
  --wandb-run STR           Wandb run name
  --megatron-path PATH      Path to Megatron-LM repo (default: auto-detect)
  --slime-path PATH         Path to SLIME repo (default: auto-detect)
  -- EXTRA_ARGS...          Pass remaining args directly to train.py
EOF
    exit 0
}

# ============================================================================
# Defaults
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="Qwen/Qwen3-30B-A3B"
MEGATRON_CKPT=""
MODEL_ARGS_SCRIPT=""
SAVE_DIR="${SCRIPT_DIR}/checkpoints/"
SAVE_INTERVAL=10

LR=1e-5
WEIGHT_DECAY=0.0
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPS=1e-8
LR_DECAY_STYLE=constant

NUM_GPUS=4
TP=4

ROLLOUT_BATCH_SIZE=32
N_SAMPLES=16
MAX_RESPONSE_LEN=4096
MAX_TOKENS_PER_GPU=8192
TEMPERATURE=1.0
NUM_ROLLOUT=3000

TASK_DATA="${SCRIPT_DIR}/tasks.jsonl"
TRAIN_CONFIG="${SCRIPT_DIR}/train_config.yaml"

WANDB_PROJECT="slime-openreward"
WANDB_RUN="slime-openreward-$(date +%Y%m%d-%H%M%S)"

MEGATRON_PATH=""
SLIME_PATH=""
EXTRA_ARGS=()

# ============================================================================
# Parse CLI
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)           usage ;;
        --model)             MODEL="$2"; shift 2 ;;
        --megatron-ckpt)     MEGATRON_CKPT="$2"; shift 2 ;;
        --model-args)        MODEL_ARGS_SCRIPT="$2"; shift 2 ;;
        --save)              SAVE_DIR="$2"; shift 2 ;;
        --save-interval)     SAVE_INTERVAL="$2"; shift 2 ;;
        --lr)                LR="$2"; shift 2 ;;
        --weight-decay)      WEIGHT_DECAY="$2"; shift 2 ;;
        --adam-beta1)        ADAM_BETA1="$2"; shift 2 ;;
        --adam-beta2)        ADAM_BETA2="$2"; shift 2 ;;
        --adam-eps)          ADAM_EPS="$2"; shift 2 ;;
        --lr-decay-style)    LR_DECAY_STYLE="$2"; shift 2 ;;
        --num-gpus)          NUM_GPUS="$2"; shift 2 ;;
        --tp)                TP="$2"; shift 2 ;;
        --rollout-batch-size) ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
        --n-samples)         N_SAMPLES="$2"; shift 2 ;;
        --max-response-len)  MAX_RESPONSE_LEN="$2"; shift 2 ;;
        --max-tokens-per-gpu) MAX_TOKENS_PER_GPU="$2"; shift 2 ;;
        --temperature)       TEMPERATURE="$2"; shift 2 ;;
        --num-rollout)       NUM_ROLLOUT="$2"; shift 2 ;;
        --task-data)         TASK_DATA="$2"; shift 2 ;;
        --config)            TRAIN_CONFIG="$2"; shift 2 ;;
        --wandb-project)     WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-run)         WANDB_RUN="$2"; shift 2 ;;
        --megatron-path)     MEGATRON_PATH="$2"; shift 2 ;;
        --slime-path)        SLIME_PATH="$2"; shift 2 ;;
        --)                  shift; EXTRA_ARGS+=("$@"); break ;;
        *)                   echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Validate required args
# ============================================================================
if [[ -z "${MODEL}" ]]; then
    echo "ERROR: --model is required"; exit 1
fi
if [[ -z "${MEGATRON_CKPT}" ]]; then
    echo "ERROR: --megatron-ckpt is required (run the checkpoint conversion step first, see README.md)"; exit 1
fi
if [[ -z "${MODEL_ARGS_SCRIPT}" ]]; then
    echo "ERROR: --model-args is required (see slime/scripts/models/ for available model configs)"; exit 1
fi

export OPENREWARD_API_KEY="${OPENREWARD_API_KEY:?Set OPENREWARD_API_KEY}"
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

# ============================================================================
# Auto-detect paths
# ============================================================================
if [[ -z "${SLIME_PATH}" ]]; then
    SLIME_PATH="$(python3 -c 'import slime; import os; print(os.path.dirname(os.path.dirname(slime.__file__)))' 2>/dev/null || true)"
    if [[ -z "${SLIME_PATH}" ]] || [[ ! -f "${SLIME_PATH}/train.py" ]]; then
        echo "ERROR: Cannot find SLIME repo. Set --slime-path or install slime with: pip install -e /path/to/slime"; exit 1
    fi
fi

if [[ -z "${MEGATRON_PATH}" ]]; then
    MEGATRON_PATH="$(python3 -c 'import megatron.core, os; print(os.path.dirname(os.path.dirname(os.path.dirname(megatron.core.__file__))))' 2>/dev/null || true)"
    if [[ -z "${MEGATRON_PATH}" ]]; then
        echo "ERROR: Cannot find Megatron-LM. Set --megatron-path or install megatron-core"; exit 1
    fi
fi

# Source model architecture args
source "${MODEL_ARGS_SCRIPT}"

# ============================================================================
# Derived values
# ============================================================================
export OPENREWARD_RUN_NAME="${WANDB_RUN}"
export TRAIN_CONFIG
GLOBAL_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES ))

# ============================================================================
# Ensure Ray is running
# ============================================================================
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

if ! ray status 2>/dev/null | grep -q "Active:"; then
    echo "No Ray cluster found, starting a local one..."
    ray start --head \
        --node-ip-address 127.0.0.1 \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --temp-dir /tmp/ray_slime
    sleep 3
fi

# ============================================================================
# Launch
# ============================================================================
echo "Starting OpenReward + SLIME training"
echo "  Model:         ${MODEL}"
echo "  Megatron ckpt: ${MEGATRON_CKPT}"
echo "  LR:            ${LR}"
echo "  Tasks:         ${TASK_DATA}"
echo "  Config:        ${TRAIN_CONFIG}"
echo "  Wandb:         ${WANDB_PROJECT} / ${WANDB_RUN}"
echo "  Batch:         ${ROLLOUT_BATCH_SIZE} prompts × ${N_SAMPLES} samples = ${GLOBAL_BATCH_SIZE}"
echo "  GPUs:          ${NUM_GPUS} (colocate mode, tp=${TP})"
echo "  MaxTok/GPU:    ${MAX_TOKENS_PER_GPU}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENREWARD_API_KEY\": \"${OPENREWARD_API_KEY}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY:-}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"TRAIN_CONFIG\": \"${TRAIN_CONFIG}\"
  }
}"

ray job submit --address="${RAY_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_PATH}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --rollout-num-gpus "${NUM_GPUS}" \
    --colocate \
    --rollout-num-gpus-per-engine "${TP}" \
    \
    "${MODEL_ARGS[@]}" \
    \
    --hf-checkpoint "${MODEL}" \
    --load "${MEGATRON_CKPT}" \
    --save "${SAVE_DIR}" \
    --save-interval "${SAVE_INTERVAL}" \
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
    --advantage-estimator grpo \
    --eps-clip 0.2 \
    \
    --optimizer adam \
    --lr "${LR}" \
    --lr-decay-style "${LR_DECAY_STYLE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --adam-beta1 "${ADAM_BETA1}" \
    --adam-beta2 "${ADAM_BETA2}" \
    --adam-eps "${ADAM_EPS}" \
    \
    --tensor-model-parallel-size "${TP}" \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    \
    --sglang-mem-fraction-static 0.7 \
    \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN}" \
    --wandb-key "${WANDB_API_KEY}" \
    --wandb-group "${WANDB_PROJECT}" \
    \
    "${EXTRA_ARGS[@]}"