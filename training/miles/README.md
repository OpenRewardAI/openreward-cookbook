# Miles RL Training with OpenReward

Multi-environment reinforcement learning training using [Miles](https://github.com/radixark/miles) (RL post-training framework), [OpenReward](https://openreward.ai) environments, and Weights & Biases tracking.

## Overview

This project implements a custom Miles rollout integration that:
- Runs multi-turn agent interactions with tool use against OpenReward environments
- Uses SGLang for fast inference and Miles' FSDP or Megatron backend for training
- Implements GRPO advantage estimation with per-token loss masking
- Tracks per-token log probabilities from rollout for importance sampling
- Uploads trajectories to OpenReward for visualization
- Supports training on multiple environments simultaneously

Miles is a fork of [slime](https://github.com/THUDM/slime) that adds production-grade stability features. This integration uses the same plugin interface (custom generate/reward functions) and benefits from Miles-specific improvements automatically:

- **Graceful OOM recovery** — benign OOMs from variable-length multi-turn rollouts are caught and propagated instead of crashing the job
- **True on-policy with FSDP** — zero train-inference mismatch via aligned numerics (FlashAttention-3, DeepGEMM, batch-invariant kernels)
- **FSDP memory fixes** — reduced excessive memory usage, move-based offloading, host peak memory savings
- **Partial rollout & over-sampling** — handles the long-tail effect in multi-turn RL by over-sampling and recycling half-finished trajectories

## Installation

### Requirements
- Python 3.11+
- Miles installed (`pip install -e /path/to/miles`)
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
- `wandb` — Experiment tracking (used by Miles)

PyTorch and Miles are expected to already be installed in your environment.

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
openreward_run_name: miles-rl-openreward
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

Fetch tasks from OpenReward and write a Miles-compatible JSONL dataset:

```bash
python prepare_tasks.py --config train_config.yaml --output tasks.jsonl
```

### 2. Run training

From the Miles repo root:

```bash
cd /path/to/miles
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

# Pass arbitrary Miles args after --
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

Miles auto-resumes from the latest checkpoint in `--load` if one exists.

## Project Structure

```
openreward_miles/
├── generate.py          # Custom Miles generate function (multi-turn agent loop)
├── reward.py            # Custom Miles reward function (reduces per-step rewards)
├── config.py            # Shared config, utilities, tool formatting, parsing
├── prepare_tasks.py     # Fetches tasks from OpenReward → writes tasks.jsonl
├── train_config.yaml    # Environment & agent configuration
├── run.sh               # Training launch script with CLI interface
├── requirements.txt     # Python dependencies
└── README.md
```

### How it works

1. **`prepare_tasks.py`** connects to OpenReward, lists tasks for each configured environment, and writes a JSONL file where each line has `prompt`, `label`, and `metadata` (containing env name, task spec, deployment info).

2. **`generate.py`** is called by Miles once per sample. It:
   - Opens an OpenReward session for the task
   - Gets tools and prompt from the environment
   - Runs a multi-turn loop: generate via SGLang → parse tool call → execute via OpenReward → append result
   - Tracks per-token log probabilities from SGLang for importance sampling
   - Builds the loss mask (1 for model-generated tokens, 0 for environment/system tokens)
   - Fire-and-forgets a trajectory upload to OpenReward

3. **`reward.py`** reads the per-step rewards stored in `sample.metadata` by `generate.py`, applies nonterminal penalties, reduces them (sum/mean/max/min), and clamps.

4. **Miles** handles everything else: batching, GRPO advantage estimation, training, weight sync to SGLang, checkpointing. Miles' graceful OOM handling and memory margins make the variable-length sequences from multi-turn rollouts less likely to crash training.

## Known Issues

### FSDP logging crash with custom generate (`Attribute tokens is not found in packed batch`)

**Workaround:** wrap the logging call in a try/except in `miles/backends/fsdp_utils/actor.py`:
```python
# around line 560, change:
self._log_rollout_data(rollout_id, rollout_data, packed_batches)

# to:
try:
    self._log_rollout_data(rollout_id, rollout_data, packed_batches)
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Failed to log rollout data: {e}")
```

This preserves all training behavior and reward logging — you only lose some per-step
`rollout/log_probs`, `rollout/advantages` etc. metrics in wandb for affected batches.

## Memory Considerations

Multi-turn agent rollouts produce long sequences (system prompt + tools + N turns of generation + tool responses). This can cause OOM during training.

Key levers:

- **`--max-tokens-per-gpu N`** + **`--use-dynamic-batch-size`**: Caps tokens packed per GPU per training step. Set in `run.sh` by default. Start at `max_response_len` and increase for throughput.
- **`--gradient-checkpointing`**: Trades ~10% speed for significantly less activation memory. Enabled by default in `run.sh`. Recommended for models with large vocabularies (e.g. Qwen3's 152k vocab).
- **`--context-parallel-size N`**: Splits long sequences across N GPUs (requires N actor GPUs).
- **`max_turns` in `train_config.yaml`**: Fewer turns = shorter sequences.

Miles' graceful OOM recovery means that if a rare batch does exceed memory, the job won't crash — the error is propagated and training continues. This is particularly valuable for multi-turn rollouts where sequence length variance is high.

## Additional Resources

- **OpenReward Environments**: https://openreward.ai/environments
- **OpenReward Documentation**: https://docs.openreward.ai/
- **Miles GitHub**: https://github.com/radixark/miles
- **slime GitHub**: https://github.com/THUDM/slime (upstream framework)
- **WandB Documentation**: https://docs.wandb.ai/