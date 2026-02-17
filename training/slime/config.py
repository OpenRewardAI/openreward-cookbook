"""Shared configuration and utilities for OpenReward + SLIME integration."""
import json
import os
from typing import Any

from pydantic import BaseModel, Field


class EnvConfig(BaseModel, extra="forbid"):
    """Per-environment configuration."""

    splits: list[str] = ["train"]
    """Which splits to pull tasks from."""

    num_samples: int | None = None
    """Cap on tasks per split, or all if None."""

    nonterminal_reward: float | None = None
    """Reward when agent doesn't reach terminal state."""

    reward_reduction: str = "sum"
    """How to reduce multiple rewards: sum, mean, max, min."""

    min_reward: float | None = None
    max_reward: float | None = None

    secrets: dict[str, str | None] = Field(default_factory=dict)
    """Environment-specific secrets (key → value, or key → null to read from env)."""

    max_turns: int = 20
    """Maximum agent turns per rollout."""


class IntegrationConfig(BaseModel, extra="forbid"):
    """Top-level config for the OpenReward + SLIME integration."""

    environments: dict[str, EnvConfig]
    """Map of environment name → config."""

    secrets: dict[str, str | None] = Field(default_factory=dict)
    """Global secrets (overridden by per-env secrets)."""

    system_prompt_template: str = (
        "You are an agent that takes actions in a stateful environment to achieve a goal.\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n{tools}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        'within <tool_call></tool_call> XML tags:\n<tool_call>\n'
        '{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>'
    )

    thinking: bool = False
    """Whether the model uses <think> blocks."""

    openreward_run_name: str = "slime-training"
    """Run name for OpenReward rollout uploads."""


def load_integration_config(path: str | None = None) -> IntegrationConfig:
    """Load the integration config from a YAML file."""
    import yaml
    path = path or os.environ.get("TRAIN_CONFIG", "train_config.yaml")
    with open(path) as f:
        return IntegrationConfig(**yaml.safe_load(f))


def resolve_secrets(
    env_config: EnvConfig,
    global_secrets: dict[str, str | None],
) -> dict[str, str]:
    """Resolve secrets: env-specific > global > environment variables."""
    result: dict[str, str] = {}

    for key, value in global_secrets.items():
        resolved = value if value else os.environ.get(key, "")
        if resolved:
            result[key] = resolved

    for key, value in env_config.secrets.items():
        if value is None:
            env_val = os.environ.get(key, "")
            if env_val:
                result[key] = env_val
            else:
                result.pop(key, None)
        elif value:
            result[key] = value
        else:
            result.pop(key, None)

    return result


def reduce_rewards(rewards: list[float], method: str) -> float:
    """Reduce a list of rewards to a scalar."""
    if not rewards:
        return 0.0
    if method == "sum":
        return sum(rewards)
    elif method == "mean":
        return sum(rewards) / len(rewards)
    elif method == "max":
        return max(rewards)
    elif method == "min":
        return min(rewards)
    raise ValueError(f"Unknown reduction: {method}")


def format_tool_spec(spec: Any) -> str:
    """Format a tool spec as JSON for the system prompt."""
    return json.dumps({
        "name": spec.name,
        "arguments_schema": spec.input_schema,
        "description": spec.description,
    })


def parse_tool_call(text: str) -> dict | None:
    """Parse a <tool_call>...</tool_call> block from generated text.

    Returns None if no tool call found, or a dict with type='success'/'error'.
    """
    start_tag, end_tag = "<tool_call>", "</tool_call>"
    si = text.find(start_tag)
    if si == -1:
        return None

    # If stopped at </tool_call>, it may be missing from text
    ei = text.find(end_tag, si)
    json_str = text[si + len(start_tag):ei].strip() if ei != -1 else text[si + len(start_tag):].strip()

    try:
        data = json.loads(json_str)
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            return {"type": "error", "error": f"arguments not a dict: {type(args)}"}
        return {"type": "success", "name": data["name"], "arguments": args}
    except (json.JSONDecodeError, KeyError) as e:
        return {"type": "error", "error": str(e)}


def apply_chat_template(messages: list[dict[str, str]]) -> str:
    """Build Qwen-style chat formatted text from messages."""
    return "\n".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
        for m in messages
    )