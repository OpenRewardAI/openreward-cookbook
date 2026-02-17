# SLIME RL Training with OpenReward

Multi-environment reinforcement learning training using [SLIME](https://github.com/THUDM/slime) (RL post-training framework), [OpenReward](https://openreward.ai) environments, and Weights & Biases tracking.

## Overview

This project implements a custom SLIME rollout integration that:
- Runs multi-turn agent interactions with tool use against OpenReward environments
- Uses SGLang for fast inference and SLIME's FSDP or Megatron backend for training
- Implements GRPO advantage estimation with per-token loss masking
- Tracks per-token log probabilities from rollout for importance sampling
- Uploads trajectories to OpenReward for visualization
- Supports training on multiple environments simultaneously

## Installation

### Requirements
- Python 3.11+
- SLIME installed (`pip install -e /path/to/slime`)
- NVIDIA GPUs (tested on H100/H200)

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `openreward` — Environment management and task definitions
- `transformers` — Tokenizer support
- `aiohttp` — Async HTTP client for SGLang router
- `pydantic` — Configuration validation
- `numpy` — Token array construction
- `pyyaml` — Config file parsing
- `wandb` — Experiment tracking (used by SLIME)

PyTorch and SLIME are expected to already be installed in your environment.

## Environment Variables

```bash
export OPENREWARD_API_KEY=your_openreward_key_here
export WANDB_API_KEY=your_wandb_key_here
export OPENAI_API_KEY=your_openai_key_here  # If environments use LLM-based graders
```

## Configuration

Training is configured via two files:

### `train_config.yaml` — Environment & agent settings

```yaml
environments:
  GeneralReasoning/WhoDunit:  # Browse at https://openreward.ai/environments
    splits: [train]
    nonterminal_reward: 0.0
    reward_reduction: sum
    max_turns: 20

secrets:
  openai_api_key: null  # null = read from OPENAI_API_KEY env var

system_prompt_template: |
  You are an agent that takes actions in a stateful environment to achieve a goal.

  # Tools
  ...
  {tools}
  ...

thinking: false
openreward_run_name: slime-rl-test-openreward
```

### `run.sh` — Training hyperparameters

All training, optimizer, cluster, and rollout settings are passed via `run.sh` CLI flags. Run `bash run.sh --help` for the full list.

### Environment Configuration

**Browse available environments at: https://openreward.ai/environments**

Environment names follow the format: `Organization/EnvironmentName`

Per-environment options:

| Field | Default | Description |
|---|---|---|
| `splits` | `[train]` | Which task splits to pull from |
| `num_samples` | `null` | Cap on tasks per split (`null` = all) |
| `nonterminal_reward` | `null` | Penalty if agent doesn't reach terminal state |
| `reward_reduction` | `sum` | How to reduce step rewards: `sum`, `mean`, `max`, `min` |
| `min_reward` / `max_reward` | `null` | Clamp final reward |
| `max_turns` | `20` | Maximum agent turns per rollout |
| `secrets` | `{}` | Environment-specific secrets |

Train on multiple environments by adding entries:

```yaml
environments:
  GeneralReasoning/WhoDunit:
    splits: [train]
    reward_reduction: sum
    max_turns: 20

  MATH/GSM8K:
    splits: [train]
    reward_reduction: mean
    max_turns: 10
```

## Usage

### 1. Prepare tasks

Fetch tasks from OpenReward and write a SLIME-compatible JSONL dataset:

```bash
python prepare_tasks.py --config train_config.yaml --output tasks.jsonl
```

### 2. Run training

From the SLIME repo root:

```bash
cd /path/to/slime
bash /path/to/this/run.sh
```

Common overrides:

```bash
# Different model
bash run.sh --model Qwen/Qwen3-4B

# Adjust GPU allocation
bash run.sh --actor-gpus 4 --rollout-gpus 4 --tp 4

# Tune training
bash run.sh --lr 5e-6 --n-samples 8 --rollout-batch-size 16

# Enable gradient checkpointing (reduces memory, ~10% slower)
bash run.sh -- --gradient-checkpointing

# Pass arbitrary SLIME args after --
bash run.sh -- --context-parallel-size 2 --use-kl-loss --kl-loss-coef 0.01
```

### Key training flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-30B-A3B` | HuggingFace checkpoint |
| `--lr` | `1e-5` | Learning rate |
| `--n-samples` | `16` | Rollouts per prompt (for GRPO) |
| `--rollout-batch-size` | `32` | Prompts per rollout batch |
| `--max-response-len` | `4096` | Max response tokens per generation call |
| `--max-tokens-per-gpu` | `8192` | Token cap per GPU in training (OOM prevention) |
| `--temperature` | `1.0` | Sampling temperature |
| `--train-backend` | `fsdp` | `fsdp` or `megatron` |

### Resuming from checkpoint

```bash
bash run.sh --load /path/to/checkpoints/
```

SLIME auto-resumes from the latest checkpoint in `--load` if one exists.

## Project Structure

```
openreward_slime/
├── generate.py          # Custom SLIME generate function (multi-turn agent loop)
├── reward.py            # Custom SLIME reward function (reduces per-step rewards)
├── config.py            # Shared config, utilities, tool formatting, parsing
├── prepare_tasks.py     # Fetches tasks from OpenReward → writes tasks.jsonl
├── train_config.yaml    # Environment & agent configuration
├── run.sh               # Training launch script with CLI interface
├── requirements.txt     # Python dependencies
└── README.md
```

### How it works

1. **`prepare_tasks.py`** connects to OpenReward, lists tasks for each configured environment, and writes a JSONL file where each line has `prompt`, `label`, and `metadata` (containing env name, task spec, deployment info).

2. **`generate.py`** is called by SLIME once per sample. It:
   - Opens an OpenReward session for the task
   - Gets tools and prompt from the environment
   - Runs a multi-turn loop: generate via SGLang → parse tool call → execute via OpenReward → append result
   - Tracks per-token log probabilities from SGLang for importance sampling
   - Builds the loss mask (1 for model-generated tokens, 0 for environment/system tokens)
   - Fire-and-forgets a trajectory upload to OpenReward

3. **`reward.py`** reads the per-step rewards stored in `sample.metadata` by `generate.py`, applies nonterminal penalties, reduces them (sum/mean/max/min), and clamps.

4. **SLIME** handles everything else: batching, GRPO advantage estimation, training, weight sync to SGLang, checkpointing.

## Memory Considerations

Multi-turn agent rollouts produce long sequences (system prompt + tools + N turns of generation + tool responses). This can cause OOM during training.

Key levers:

- **`--max-tokens-per-gpu N`** + **`--use-dynamic-batch-size`**: Caps tokens packed per GPU per training step. Set in `run.sh` by default. Start at `max_response_len` and increase for throughput.
- **`--gradient-checkpointing`**: Trades ~10% speed for significantly less activation memory. Recommended for models with large vocabularies (e.g. Qwen3's 152k vocab).
- **`--context-parallel-size N`**: Splits long sequences across N GPUs (requires N actor GPUs).
- **`max_turns` in `train_config.yaml`**: Fewer turns = shorter sequences.

## Additional Resources

- **OpenReward Environments**: https://openreward.ai/environments
- **OpenReward Documentation**: https://docs.openreward.ai/
- **SLIME Documentation**: https://github.com/THUDM/slime
- **WandB Documentation**: https://docs.wandb.ai/
