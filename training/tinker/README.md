# Tinker RL Training with OpenReward

Multi-environment reinforcement learning training system using Tinker's hosted infrastructure, OpenReward environments, and Weights & Biases tracking.

## Overview

This project implements a distributed RL training loop that:
- Trains large language models using LoRA fine-tuning via Tinker's hosted infrastructure
- Trains uses an environment from OpenReward
- Uses multi-turn agent interactions with tool use
- Implements GRPO-style training with advantage normalization
- Tracks experiments in WandB and uploads trajectories to OpenReward
- Supports checkpoint resumption

## Installation

### Requirements
- Python 3.11 or higher
- API keys for Tinker, OpenReward, WandB, and OpenAI (if using environments that require it)

### Dependencies

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Or using uv (faster):
```bash
uv pip install -r requirements.txt
```

Core dependencies:
- `openreward` - Environment management and task definitions
- `tinker` - Distributed training and sampling infrastructure
- `wandb` - Experiment tracking and metrics logging
- `transformers` - Tokenizer support
- `torch` - PyTorch for tensor operations
- `pydantic` - Configuration validation

## Environment Variables

Create a `.env` file in the project root with the following API keys:

```bash
TINKER_API_KEY=your_tinker_key_here
OPENREWARD_API_KEY=your_openreward_key_here
WANDB_API_KEY=your_wandb_key_here
OPENAI_API_KEY=your_openai_key_here  # You might need additional keys, e.g. for environments that have LLM based graders
```

## Configuration

The training is configured via `tinker-config.yaml`. Here's an overview of the key settings:

### WandB Settings
```yaml
wandb_project_name: "tinker-test"
wandb_run_name: "tinker-rl-test-openreward"
openreward_run_name: "tinker-rl-test-openreward"
log_path: "/tmp/tinker-rl"
```

### Model Configuration
```yaml
model_name: "Qwen/Qwen3-30B-A3B"
lora_rank: 8
# resume_from: "tinker://run-id/weights/checkpoint_000050"  # Optional
```

### Training Parameters
```yaml
loss_fn: "ppo"  # Options: importance_sampling, ppo, cispo, dro
batch_size: 32
learning_rate: 1e-5
save_every: 10  # Save checkpoint every N steps
```

### Environment Configuration

**Browse available environments at: https://openreward.ai/environments**

Environment names follow the format: `Organization/EnvironmentName`

Example configuration:
```yaml
environments:
  GeneralReasoning/WhoDunIt:  # Browse at https://openreward.ai/environments
    splits:
      train:
        shuffle: true
        num_samples: null  # null = use all tasks
    num_rollouts: 16
    max_failing_rollouts: 2
    temperature: 1.0
    reward_reduction: "sum"
    nonterminal_reward: 0.0
```

### Finding and Configuring Environments

1. Visit https://openreward.ai/environments to browse available environments
2. Find an environment you want to train on (e.g., `GeneralReasoning/WhoDunIt`, `MATH/GSM8K`, etc.)
3. Add it to the `environments` section in `tinker-config.yaml`
4. Configure the parameters as needed

You can train on multiple environments simultaneously by adding multiple entries:

```yaml
environments:
  GeneralReasoning/WhoDunIt:
    splits:
      train:
        shuffle: true
    num_rollouts: 16
    temperature: 1.0
    reward_reduction: "sum"

  MATH/GSM8K:
    splits:
      train:
        shuffle: true
    num_rollouts: 8
    temperature: 0.7
    reward_reduction: "mean"
```

## Usage

### Running Training

Run the training loop with the default configuration:
```bash
python main.py --config_path tinker-config.yaml
```

### Configuration Overrides

You can override configuration values from the command line using dotted notation:

```bash
# Override batch size and learning rate
python main.py --config_path tinker-config.yaml --overrides batch_size=64 learning_rate=2e-5

# Override multiple settings
python main.py --config_path tinker-config.yaml --overrides \
    batch_size=64 \
    learning_rate=2e-5 \
    loss_fn=importance_sampling
```

### Resuming from Checkpoint

To resume training from a previous checkpoint, uncomment and set the `resume_from` parameter in your config:

```yaml
resume_from: "tinker://run-id/weights/checkpoint_000050"
```

## Project Structure

```
tinker/
├── main.py                  # Main training loop with Tinker integration
├── tinker-config.yaml       # Configuration file for training runs
├── requirements.txt         # Python dependencies
├── .env                     # API keys and secrets (not committed to git)
└── README.md               # This file
```

### OpenReward Trajectories
View detailed rollout trajectories at your OpenReward runs page.

### Local Logs
Logs are written to the path specified in `log_path` (default: `/tmp/tinker-rl`).

## Additional Resources

- **OpenReward Environments**: https://openreward.ai/environments
- **OpenReward Documentation**: https://docs.openreward.ai/
- **Tinker Documentation**: https://tinker-docs.thinkingmachines.ai/
- **WandB Documentation**: https://docs.wandb.ai/
