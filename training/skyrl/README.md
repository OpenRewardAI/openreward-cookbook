# SkyRL Training with OpenReward

Multi-environment reinforcement learning training using [SkyRL](https://github.com/NovaSky-AI/SkyRL) (RL post-training framework) and [OpenReward](https://openreward.ai) environments.

## Overview

SkyRL is an RL post-training framework that supports multi-turn agent interactions with tool use. The OpenReward integration ([PR #1458](https://github.com/NovaSky-AI/SkyRL/pull/1458)) has been merged into SkyRL's main branch. To keep a single source of truth, please use the integration directly from the SkyRL repository rather than copying code here.

Key features of the integration:

- Runs multi-turn agent interactions with tool use against OpenReward environments
- Uses vLLM for inference and SkyRL's FSDP2 backend for training
- Implements GRPO advantage estimation with per-token loss masking
- Uploads trajectories to OpenReward for visualization
- Supports training on multiple environments simultaneously
- Supports Modal for cloud GPU training

## Quick Start

The full code and documentation live in the SkyRL repository:

👉 **[NovaSky-AI/SkyRL → examples/train_integrations/openreward](https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train_integrations/openreward)**

### 1. Install SkyRL

```bash
pip install -e /path/to/SkyRL
```

See the [SkyRL installation guide](https://docs.skyrl.ai/docs/getting-started/installation) for details.

### 2. Set environment variables

```bash
export OPENREWARD_API_KEY=your_openreward_key_here
export WANDB_API_KEY=your_wandb_key_here       # optional
export OPENAI_API_KEY=your_openai_key_here      # if environments use LLM-based graders
```

### 3. Prepare tasks

Fetch tasks from OpenReward and write a SkyRL-compatible Parquet dataset:

```bash
python examples/train_integrations/openreward/prepare_tasks.py \
    --env "GeneralReasoning/WhoDunit" \
    --split train \
    --max-tasks 50 \
    --output /root/data/openreward/train.parquet
```

### 4. Run training

**Local GPU:**

```bash
bash examples/train_integrations/openreward/run_openreward.sh
```

**Modal (cloud GPU):**

```bash
MODAL_GPU=A100:4 modal run examples/train_integrations/modal/main.py \
    --command "OPENREWARD_API_KEY=<your-key> WANDB_API_KEY=<your-key> \
        OPENREWARD_UPLOAD_ROLLOUT=true \
        bash examples/train_integrations/openreward/run_openreward.sh"
```

### Override config

```bash
bash examples/train_integrations/openreward/run_openreward.sh \
    trainer.epochs=2 generator.max_turns=8
```

## Training Config Summary

| Parameter              | Default                  | Description                                          |
| ---------------------- | ------------------------ | ---------------------------------------------------- |
| `NUM_GPUS`             | 4                        | Number of GPUs (colocated: policy + ref + inference) |
| `MODEL`                | Qwen/Qwen2.5-3B-Instruct | Base model                                           |
| `train_batch_size`     | 32                       | Unique prompts per training step                     |
| `n_samples_per_prompt` | 4                        | Rollouts per prompt (GRPO group size)                |
| `max_turns`            | 10                       | Max agent-environment interaction turns              |

## Environment Variables

| Variable                    | Default            | Description                                                   |
| --------------------------- | ------------------ | ------------------------------------------------------------- |
| `OPENREWARD_API_KEY`        | (required)         | API key from [openreward.ai/keys](https://openreward.ai/keys) |
| `OPENREWARD_UPLOAD_ROLLOUT` | `true`             | Whether to upload rollouts to OpenReward for visualization    |
| `OPENREWARD_RUN_NAME`       | `skyrl-openreward` | Run name used for rollout uploads                             |

## Adding More Environments

Browse available environments at [openreward.ai/environments](https://openreward.ai/environments). The `env_class` is always `"openreward"` — the specific environment is determined by `env_name` in the dataset. To train on multiple environments:

```bash
python examples/train_integrations/openreward/prepare_tasks.py \
    --env "GeneralReasoning/WhoDunit" \
    --env "GeneralReasoning/CTF" \
    --split train --max-tasks 50 \
    --output /root/data/openreward/train.parquet
```

## How It Works

1. **`prepare_tasks.py`** queries OpenReward for tasks, creates a temporary session per task to fetch the initial prompt and available tools, then writes a Parquet dataset.
2. **`OpenRewardEnv`** implements `BaseTextEnv`. On `init()`, it opens an OpenReward session. On each `step()`, it parses tool calls from the model output, calls `session.call_tool()`, and returns the result.
3. SkyRL's `agent_loop` handles the multi-turn generation loop.
