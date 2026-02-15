import asyncio
import hashlib
import itertools
import json
import logging
import os
import random
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
from transformers import AutoTokenizer

import tinker
import tinker.types as tinker_types
from openreward import AsyncOpenReward
from openreward.client import DEFAULT_BASE_URL as OPENREWARD_BASE_URL
from openreward.api.environments.client import AsyncEnvironment, Task
from openreward.api.environments.types import TextBlock, ToolCallError, ToolOutput
from openreward.api.rollouts.serializers.base import (
    AssistantMessage,
    ReasoningItem,
    SystemMessage,
    ToolCall,
    ToolResult,
    UploadType,
    UserMessage,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Type aliases
# ============================================================================

RewardReduction: TypeAlias = Literal["sum", "mean", "max", "min"]
AdvantageCalculation: TypeAlias = Literal["direct", "centered", "centered_normalized"]
StopReason: TypeAlias = Literal["max_tokens", "reached_terminal_state", "max_retries", "error"]
TinkerLossFn: TypeAlias = Literal["importance_sampling", "ppo", "cispo", "dro"]

# ============================================================================
# Configuration
# ============================================================================


class SplitConfig(BaseModel, extra="forbid"):
    shuffle: bool = True
    """Shuffle received tasks."""

    num_samples: int | None = None
    """Number of samples to draw from this split, or all if None."""


class EnvironmentConfig(BaseModel, extra="forbid"):
    splits: dict[str, SplitConfig]
    """Splits to use during sampling."""

    nonterminal_reward: float | None = None
    """Reward to assign when the agent does not reach a terminal state."""

    num_rollouts: int = Field(ge=2)
    """Number of rollouts per task."""

    max_failing_rollouts: int = Field(ge=0, default=0)
    """Maximum number of rollouts allowed to fail before the group is excluded."""

    temperature: float = Field(ge=0.0, le=2.0, default=1.0)
    """Temperature for sampling."""

    top_p: float | None = Field(ge=0.0, le=1.0, default=None)
    """Top-p (nucleus) sampling threshold."""

    top_k: int | None = Field(ge=0, default=None)
    """Top-k sampling."""

    reward_reduction: RewardReduction = "sum"
    """How to reduce multiple rewards within a rollout."""

    min_reward: float | None = None
    """Optional minimum reward clamp."""

    max_reward: float | None = None
    """Optional maximum reward clamp."""

    secrets: dict[str, str | None] | None = None
    """Optional environment-specific secrets (overrides top-level secrets)."""

    @model_validator(mode="after")
    def check_rollouts(self):
        if self.num_rollouts - self.max_failing_rollouts < 2:
            raise ValueError("num_rollouts - max_failing_rollouts must be >= 2")
        return self


class Config(BaseModel, extra="forbid"):
    # -- Tracking --
    wandb_project_name: str
    wandb_run_name: str
    openreward_run_name: str
    log_path: str = "/tmp/tinker-rl"

    # -- Environments --
    environments: dict[str, EnvironmentConfig]

    secrets: dict[str, str | None] | None = None
    """Global secrets for all environments."""

    # -- Model / Tinker --
    model_name: str
    """Base model on Tinker (e.g. 'Qwen/Qwen3-8B')."""

    lora_rank: int = 32
    """LoRA rank for fine-tuning."""

    tokenizer_name: str | None = None
    """Tokenizer name; defaults to model_name."""

    resume_from: str | None = None
    """Tinker checkpoint path to resume from (tinker://...)."""

    # -- Generation --
    max_response_tokens: int = 4096
    """Maximum tokens per completion."""

    max_total_tokens: int = 32768
    """Maximum total tokens per rollout."""

    max_rollout_retries: int = 3
    """Maximum retries per rollout."""

    # -- Training --
    loss_fn: TinkerLossFn = "ppo"
    """Loss function: importance_sampling, ppo, cispo, or dro."""

    loss_fn_config: dict[str, float] = Field(default_factory=dict)
    """Optional loss function parameters (e.g. clip thresholds)."""

    advantage_calculation: AdvantageCalculation = "centered_normalized"
    """How to compute advantage from rewards."""

    batch_size: int
    """Number of task groups per training step."""

    learning_rate: float = 4e-5
    """Learning rate for Adam."""

    weight_decay: float = 0.0
    """Weight decay for Adam."""

    num_epochs: int | None = None
    """Number of epochs; None for infinite."""

    save_every: int = 10
    """Save checkpoint every N steps."""

    max_active_tasks: int = 32
    """Maximum concurrent rollout tasks."""

    max_rollout_concurrency: int | None = None
    """Optional maximum concurrency for rollouts."""

    rollout_timeout: float = 300.0
    """Timeout in seconds for a single rollout (default 5 minutes)."""

    # -- Agent --
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
    """Whether to use thinking mode."""

    seed: int = 42


class Settings(BaseSettings):
    openreward_api_key: str
    openreward_base_url: str = OPENREWARD_BASE_URL
    openreward_environments_url: str | None = None
    wandb_api_key: str
    tinker_api_key: str = ""  # Can also be set via TINKER_API_KEY env var

    # Optional secrets for environment sessions
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class AgentBlock:
    """A single block in an agent rollout (message + optional reward)."""
    message: UploadType
    reward: float | None = None
    is_finished: bool = False
    metadata: dict | None = None


@dataclass
class RolloutResult:
    """Result of a single rollout."""
    task: Task
    env_name: str
    split: str
    blocks: list[AgentBlock]
    prompt_tokens: list[int]      # all tokens fed to model (prompt + prior turns)
    response_tokens: list[int]    # only the generated response tokens across all turns
    response_logprobs: list[float]  # logprobs for response tokens
    stop: StopReason
    reward: float | None
    policy_step: int
    errors: list[str]
    extra: dict[str, Any]


@dataclass
class RolloutGroup:
    """A group of rollouts for the same task."""
    rollouts: list[RolloutResult]


# ============================================================================
# Reward / advantage utilities
# ============================================================================


def reduce_rewards(rewards: list[float], reduction: RewardReduction) -> float:
    """Reduce a list of rewards to a single value."""
    if reduction == "sum":
        return sum(rewards)
    elif reduction == "mean":
        return mean(rewards)
    elif reduction == "max":
        return max(rewards)
    elif reduction == "min":
        return min(rewards)
    raise ValueError(f"Unknown reduction: {reduction}")


def compute_advantages(
    rewards: list[float],
    method: AdvantageCalculation,
) -> list[float]:
    """Compute per-rollout advantages from rewards within a group."""
    mean_reward = mean(rewards)
    std_reward = stdev(rewards) if len(rewards) > 1 else 0.0

    if method == "direct":
        return rewards
    elif method == "centered":
        return [r - mean_reward for r in rewards]
    elif method == "centered_normalized":
        if std_reward < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean_reward) / std_reward for r in rewards]
    raise ValueError(f"Unknown method: {method}")


def resolve_secrets(
    env_config: EnvironmentConfig,
    global_secrets: dict[str, str | None] | None,
    settings: Settings,
) -> dict[str, str]:
    """Resolve secrets with proper precedence: env-specific > global > env vars.

    Args:
        env_config: The environment configuration
        global_secrets: Global secrets from top-level config
        settings: Settings object with environment variables

    Returns:
        Dictionary of resolved secret key-value pairs (only non-null values)
    """
    result: dict[str, str] = {}

    # Start with global secrets from config
    if global_secrets:
        for key, value in global_secrets.items():
            if value is None:
                # null means use environment variable
                env_value = getattr(settings, key, None)
                if env_value:
                    result[key] = env_value
            elif value:  # Non-empty string
                result[key] = value

    # Apply environment-specific overrides
    if env_config.secrets:
        for key, value in env_config.secrets.items():
            if value is None:
                # null means use environment variable
                env_value = getattr(settings, key, None)
                if env_value:
                    result[key] = env_value
                else:
                    # Remove the key if explicitly set null and no env var
                    result.pop(key, None)
            elif value:  # Non-empty string
                result[key] = value
            else:  # Empty string - remove
                result.pop(key, None)

    return result


# ============================================================================
# Stats utilities
# ============================================================================


def get_stats(values: list[int] | list[float]) -> dict[str, float]:
    """Compute basic statistics for a list of values."""
    if not values:
        return {}
    fvalues = [float(v) for v in values]
    return {
        "mean": mean(fvalues),
        "min": min(fvalues),
        "max": max(fvalues),
        "std": stdev(fvalues) if len(fvalues) > 1 else 0.0,
        "count": len(fvalues),
    }


def get_rollout_stats(groups: list[RolloutGroup]) -> dict[str, Any]:
    """Compute statistics across all rollouts."""
    flat = [r for g in groups for r in g.rollouts]
    if not flat:
        return {}

    out: dict[str, Any] = {}
    rewards = [r.reward for r in flat if r.reward is not None]
    if rewards:
        out["reward"] = get_stats(rewards)

    token_counts = [len(r.response_tokens) for r in flat if r.response_tokens]
    if token_counts:
        out["response_tokens"] = get_stats(token_counts)

    # Stop reason breakdown
    stop_reasons = set(r.stop for r in flat)
    out["stop_reason"] = {
        k: {"count": sum(1 for r in flat if r.stop == k), "frac": sum(1 for r in flat if r.stop == k) / len(flat)}
        for k in stop_reasons
    }

    # Per-environment
    env_rollouts: dict[str, list[RolloutResult]] = defaultdict(list)
    for r in flat:
        env_rollouts[r.env_name].append(r)
    for env_name, rollouts in env_rollouts.items():
        env_rewards = [r.reward for r in rollouts if r.reward is not None]
        if env_rewards:
            out[f"env/{env_name}/reward"] = get_stats(env_rewards)

    return out


# ============================================================================
# Agent: drives multi-turn tool-use rollouts via Tinker SamplingClient
# ============================================================================


def format_tool_spec(spec: Any) -> str:
    """Format a tool spec as a JSON string for the system prompt."""
    return json.dumps({
        "name": spec.name,
        "arguments_schema": spec.input_schema,
        "description": spec.description,
    })


def parse_tool_call(text: str) -> dict | None:
    """Parse a <tool_call>...</tool_call> block from text."""
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None
    raw = text[start_idx:end_idx + len(end_tag)]
    json_str = text[start_idx + len(start_tag):end_idx].strip()
    try:
        data = json.loads(json_str)
        args = data["arguments"]
        if not isinstance(args, dict):
            return {"type": "error", "raw": raw, "error": f"Expected dict, got {type(args)}"}
        return {"type": "success", "name": data["name"], "arguments": args, "raw": raw}
    except (json.JSONDecodeError, KeyError) as e:
        return {"type": "error", "raw": raw, "error": str(e)}


def parse_response(text: str, enable_thinking: bool) -> dict:
    """Parse thinking + content + tool_call from a response."""
    thinking = None
    content = text
    if enable_thinking:
        ts, te = "<think>", "</think>"
        si = text.find(ts)
        if si != -1:
            ei = text.find(te, si)
            if ei != -1:
                thinking = text[si + len(ts):ei].strip()
                content = text[ei + len(te):].strip()

    tool_call = parse_tool_call(content)
    if tool_call is not None:
        ti = content.find("<tool_call>")
        if ti != -1:
            content = content[:ti].strip()

    return {"thinking": thinking, "content": content, "tool_call": tool_call}


async def run_agent_rollout(
    sampling_client: Any,
    tokenizer: Any,
    session: Any,
    config: Config,
    env_config: EnvironmentConfig,
) -> tuple[list[AgentBlock], list[int], list[int], list[float], StopReason, dict[str, Any]]:
    """Run a single multi-turn agent rollout using Tinker sampling.

    Returns (blocks, all_tokens, response_tokens, response_logprobs, stop_reason, extra).
    """
    tools = await session.list_tools()
    tools_str = "\n".join(format_tool_spec(t) for t in tools)
    system_message = config.system_prompt_template.format(tools=tools_str)

    prompt = await session.get_prompt()
    user_text = "".join(b.text for b in prompt if b.type == "text")

    blocks: list[AgentBlock] = [
        AgentBlock(message=SystemMessage(content=system_message)),
        AgentBlock(message=UserMessage(content=user_text)),
    ]

    # Build the initial token sequence (system + user turns)
    dialog_text = _apply_chat_format(
        [{"role": "system", "content": system_message}, {"role": "user", "content": user_text}],
        tokenizer,
    )
    all_tokens: list[int] = tokenizer.encode(dialog_text, add_special_tokens=False)
    response_tokens: list[int] = []
    response_logprobs: list[float] = []

    stop: StopReason = "max_tokens"
    num_tool_calls = 0
    num_failed_tool_calls = 0
    extra: dict[str, Any] = {}

    while True:
        # Add assistant turn prefix
        assistant_prefix = _apply_chat_format_assistant_prefix(config.thinking, tokenizer)
        prefix_tokens = tokenizer.encode(assistant_prefix, add_special_tokens=False)
        all_tokens.extend(prefix_tokens)

        # Sample a completion
        model_input = tinker_types.ModelInput.from_ints(all_tokens)
        remaining = config.max_total_tokens - len(all_tokens)
        if remaining <= 0:
            stop = "max_tokens"
            break
        effective_max_tokens = min(config.max_response_tokens, remaining)
        params = tinker_types.SamplingParams(
            max_tokens=effective_max_tokens,
            temperature=env_config.temperature,
            top_p=env_config.top_p if env_config.top_p is not None else 1.0,
            stop=["<|im_end|>", "</tool_call>"],
        )

        try:
            response = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=params,
            )
            # sample_async returns APIFuture; resolve it
            sample_result: Any
            if hasattr(response, "result") and callable(response.result):
                sample_result = response.result()
            else:
                sample_result = response
        except Exception as e:
            logger.warning(f"Sampling failed: {e}")
            raise

        # Resolve the actual sample from the response
        if hasattr(sample_result, "sequences"):
            sample = sample_result.sequences[0]
            output_tokens = list(sample.tokens)
            output_logprobs = list(sample.logprobs) if sample.logprobs else [0.0] * len(output_tokens)
        elif hasattr(sample_result, "samples"):
            sample = sample_result.samples[0]
            output_tokens = list(sample.tokens)
            output_logprobs = list(sample.logprobs) if sample.logprobs else [0.0] * len(output_tokens)
        else:
            public_attrs = [a for a in dir(sample_result) if not a.startswith("_")]
            raise AttributeError(
                f"Unexpected SampleResponse structure: type={type(sample_result).__name__}, "
                f"attrs={public_attrs}"
            )
        all_tokens.extend(output_tokens)
        response_tokens.extend(output_tokens)
        response_logprobs.extend(output_logprobs)

        # Decode and parse
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)
        clean_text = output_text.replace("<|im_end|>", "")
        parsed = parse_response(clean_text, config.thinking)

        if parsed["thinking"] is not None:
            blocks.append(AgentBlock(message=ReasoningItem(content=parsed["thinking"])))
        if (c := parsed["content"].strip()):
            blocks.append(AgentBlock(message=AssistantMessage(content=c)))

        tool_call_id = uuid.uuid4().hex
        if parsed["tool_call"] is not None:
            num_tool_calls += 1
            tc = parsed["tool_call"]

            if tc["type"] == "success":
                blocks.append(AgentBlock(
                    message=ToolCall(name=tc["name"], content=json.dumps(tc["arguments"]), call_id=tool_call_id)
                ))
                try:
                    tool_out = await session.call_tool(tool_name=tc["name"], input=tc["arguments"])
                except ToolCallError as exc:
                    tool_out = ToolOutput(
                        blocks=[TextBlock(text=f"Error: {exc}")],
                        metadata={"error": str(exc)},
                        finished=False,
                    )
                    num_failed_tool_calls += 1

                tool_out_str = "".join(b.text for b in tool_out.blocks if b.type == "text")
                if tool_out.finished:
                    stop = "reached_terminal_state"

                user_msg = f"<tool_response>\n{tool_out_str}\n</tool_response>"
                blocks.append(AgentBlock(
                    message=ToolResult(content=tool_out_str, call_id=tool_call_id),
                    reward=tool_out.reward,
                    is_finished=tool_out.finished,
                    metadata=cast(Any, tool_out.metadata),
                ))
            else:
                num_failed_tool_calls += 1
                blocks.append(AgentBlock(message=AssistantMessage(content=tc["raw"])))
                user_msg = f"Tool call parse error: {tc['error']}. Please ensure tool arguments are valid JSON."
                blocks.append(AgentBlock(message=UserMessage(content=user_msg)))

            # Encode the user reply for next turn
            user_turn_text = _apply_chat_format_user_turn(user_msg, tokenizer)
            user_tokens = tokenizer.encode(user_turn_text, add_special_tokens=False)
            all_tokens.extend(user_tokens)

            if stop == "reached_terminal_state":
                break
        else:
            user_msg = "No tool call detected. Please use the provided tools to complete the task."
            blocks.append(AgentBlock(message=UserMessage(content=user_msg)))
            user_turn_text = _apply_chat_format_user_turn(user_msg, tokenizer)
            user_tokens = tokenizer.encode(user_turn_text, add_special_tokens=False)
            all_tokens.extend(user_tokens)

            if stop == "reached_terminal_state":
                break

    extra["num_tool_calls"] = num_tool_calls
    extra["num_failed_tool_calls"] = num_failed_tool_calls
    extra["called_answer"] = float(any(
        isinstance(b.message, ToolCall) and b.message.name == "answer" for b in blocks
    ))

    return blocks, all_tokens, response_tokens, response_logprobs, stop, extra


# ============================================================================
# Chat formatting helpers (Qwen-style, override for other models)
# ============================================================================


def _apply_chat_format(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Apply Qwen-style chat template to messages."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts)


def _apply_chat_format_assistant_prefix(thinking: bool, tokenizer: Any) -> str:
    """Get the assistant turn prefix for generation."""
    if thinking:
        return "\n<|im_start|>assistant\n"
    return "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _apply_chat_format_user_turn(content: str, tokenizer: Any) -> str:
    """Format a user turn for appending."""
    return f"\n<|im_start|>user\n{content}<|im_end|>"


# ============================================================================
# Build Tinker Datum from rollout
# ============================================================================


def build_datum(
    all_tokens: list[int],
    response_tokens: list[int],
    response_logprobs: list[float],
    advantage: float,
    max_length: int,
) -> tinker.Datum | None:
    """Build a tinker.Datum from rollout data for RL training.

    Per Tinker's loss function docs, loss_fn_inputs needs:
    - target_tokens: int tensor of token IDs (shape [N])
    - logprobs: float tensor of sampling logprobs (shape [N])
    - advantages: float tensor of per-token advantages (shape [N])
    """
    n = len(all_tokens)
    if n == 0 or not response_tokens:
        return None

    if n > max_length:
        all_tokens = all_tokens[:max_length]
        n = len(all_tokens)

    # Build logprobs array: 0.0 for prompt tokens, actual logprobs for response tokens
    prompt_len = n - len(response_tokens)
    if prompt_len < 0:
        response_tokens = response_tokens[:n]
        response_logprobs = response_logprobs[:n]
        prompt_len = 0

    full_logprobs = torch.zeros(n, dtype=torch.float32)
    resp_end = min(prompt_len + len(response_logprobs), n)
    resp_lp_len = resp_end - prompt_len
    full_logprobs[prompt_len:resp_end] = torch.tensor(response_logprobs[:resp_lp_len], dtype=torch.float32)

    # Build advantages array: 0.0 for prompt, advantage for response tokens
    full_advantages = torch.zeros(n, dtype=torch.float32)
    full_advantages[prompt_len:resp_end] = advantage

    target_tokens = torch.tensor(all_tokens, dtype=torch.int32)
    model_input = tinker_types.ModelInput.from_ints(all_tokens)

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(target_tokens),
            "logprobs": tinker.TensorData.from_torch(full_logprobs),
            "advantages": tinker.TensorData.from_torch(full_advantages),
        },
    )


# ============================================================================
# OpenReward upload (fire-and-forget)
# ============================================================================


async def upload_rollout(
    or_client: AsyncOpenReward,
    run_name: str,
    rollout_name: str,
    deployment_name: str,
    split: str,
    blocks: list[AgentBlock],
    task_spec: Any,
    metadata: dict[str, Any],
) -> None:
    """Upload rollout to OpenReward."""
    try:
        def _sync():
            rollout = or_client.rollout.create(
                run_name=run_name,
                rollout_name=rollout_name,
                environment=deployment_name,
                split=split,
                task_spec=task_spec,
                metadata=metadata,
            )
            for block in blocks:
                rollout.log(
                    message=block.message,
                    reward=block.reward,
                    is_finished=block.is_finished,
                    metadata=block.metadata,
                )
        await asyncio.to_thread(_sync)
    except Exception:
        logger.warning(f"Failed to upload rollout {rollout_name}: {traceback.format_exc()}")


# ============================================================================
# Main training loop
# ============================================================================


@dataclass
class TaskItem:
    """A single rollout task in the dataset."""
    task: Task
    env_name: str
    deployment_name: str
    split: str
    rollout_idx: int


async def run(config: Config, settings: Settings) -> None:
    """Main async training loop."""
    log = logging.getLogger("run")
    os.makedirs(config.log_path, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # -- Initialize Tinker --
    service_client = tinker.ServiceClient()

    if config.resume_from:
        log.info(f"Resuming from checkpoint: {config.resume_from}")
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            config.resume_from
        )
    else:
        log.info(f"Creating new LoRA training client for {config.model_name}, rank={config.lora_rank}")
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )

    # Save initial weights for sampler
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="step_000000")

    tokenizer_name = config.tokenizer_name or config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # -- Initialize OpenReward --
    or_client = AsyncOpenReward(
        api_key=settings.openreward_api_key,
        base_url=settings.openreward_base_url,
    )
    if settings.openreward_environments_url:
        or_client.environments_base_url = settings.openreward_environments_url

    # -- Fetch tasks from all environments --
    env_configs: dict[str, EnvironmentConfig] = {}
    env_deployments: dict[str, str] = {}
    tasks: dict[str, dict[str, list[Task]]] = defaultdict(dict)

    async def get_tasks(env: AsyncEnvironment, split: str) -> None:
        env_tasks = await env.list_tasks(split)
        split_config = env_configs[env.name].splits[split]
        if split_config.shuffle:
            random.shuffle(env_tasks)
        if split_config.num_samples is not None:
            env_tasks = env_tasks[:split_config.num_samples]
        tasks[env.name][split] = env_tasks
        env_deployments[env.name] = env.deployment_name

    fetch_coros = []
    for name, env_config in config.environments.items():
        env = or_client.environments.get(name)
        env_configs[env.name] = env_config
        for split_name in env_config.splits:
            fetch_coros.append(get_tasks(env, split_name))

    log.info("Fetching tasks from environments...")
    await asyncio.gather(*fetch_coros)

    # -- Build flat dataset --
    dataset: dict[str, list[TaskItem]] = defaultdict(list)
    for env_name, splits in tasks.items():
        env = or_client.environments.get(env_deployments[env_name])
        ec = env_configs[env_name]
        for split_name, split_tasks in splits.items():
            for task in split_tasks:
                task_id = f"{env_name}:{split_name}:{hashlib.sha256(json.dumps(task.task_spec).encode()).hexdigest()[:16]}"
                for i in range(ec.num_rollouts):
                    dataset[task_id].append(TaskItem(
                        task=task,
                        env_name=env_name,
                        deployment_name=env.deployment_name,
                        split=split_name,
                        rollout_idx=i,
                    ))

    task_ids = list(dataset.keys())
    num_tasks = len(task_ids)
    log.info(f"Found {num_tasks} tasks, {sum(len(v) for v in dataset.values())} total rollouts")

    num_steps = None
    if config.num_epochs is not None:
        num_steps = config.num_epochs * (num_tasks // config.batch_size)
        log.info(f"Training for {num_steps} steps ({config.num_epochs} epochs)")

    # -- Wandb --
    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_name,
        config=config.model_dump(mode="json"),
    )

    # -- Concurrency control --
    sem = asyncio.Semaphore(config.max_rollout_concurrency or config.max_active_tasks)

    # -- Rollout helper --
    rollout_progress: dict[int, int] = {}  # step -> completed count
    rollout_totals: dict[int, int] = {}    # step -> total count
    pending_rollouts: dict[int, set[str]] = {}  # step -> set of "env:rN" labels still running

    async def complete_rollout(task_id: str, item: TaskItem, step: int) -> RolloutResult:
        """Run a single rollout for a task item."""
        ec = env_configs[item.env_name]
        env = or_client.environments.get(item.deployment_name)

        # Resolve secrets for this environment
        secrets = resolve_secrets(ec, config.secrets, settings)

        result = RolloutResult(
            task=item.task,
            env_name=item.env_name,
            split=item.split,
            blocks=[],
            prompt_tokens=[],
            response_tokens=[],
            response_logprobs=[],
            stop="error",
            reward=None,
            policy_step=step,
            errors=[],
            extra={},
        )

        t0 = time.perf_counter()
        for attempt in range(config.max_rollout_retries + 1):
            try:
                async with sem:
                    # Pass secrets to session
                    session = env.session(item.task, secrets=secrets if secrets else None)
                    async with session as active_session:
                        blocks, all_tok, resp_tok, resp_lp, stop_reason, extra = await run_agent_rollout(
                            sampling_client=sampling_client,
                            tokenizer=tokenizer,
                            session=active_session,
                            config=config,
                            env_config=ec,
                        )
                        result.blocks = blocks
                        result.prompt_tokens = all_tok
                        result.response_tokens = resp_tok
                        result.response_logprobs = resp_lp
                        result.stop = stop_reason
                        result.extra = extra
                        break
            except Exception:
                err = traceback.format_exc()
                result.errors.append(err)
                log.warning(f"Rollout attempt {attempt + 1} failed: {err}")
                if attempt >= config.max_rollout_retries:
                    result.stop = "max_retries"

        # Compute reward
        rewards = [b.reward for b in result.blocks if b.reward is not None]
        if ec.nonterminal_reward is not None and result.stop != "reached_terminal_state":
            rewards.append(ec.nonterminal_reward)
        if rewards:
            reward = reduce_rewards(rewards, ec.reward_reduction)
            if ec.min_reward is not None:
                reward = max(ec.min_reward, reward)
            if ec.max_reward is not None:
                reward = min(ec.max_reward, reward)
            result.reward = reward

        # Upload to OpenReward
        if result.blocks:
            task_hash = hashlib.sha256(json.dumps(item.task.task_spec).encode()).hexdigest()[:16]
            rollout_name = f"{item.env_name}-{task_hash}-step{step}-r{item.rollout_idx}"
            asyncio.create_task(upload_rollout(
                or_client=or_client,
                run_name=config.openreward_run_name,
                rollout_name=rollout_name,
                deployment_name=item.deployment_name,
                split=item.split,
                blocks=result.blocks,
                task_spec=item.task.task_spec,
                metadata={"step": step, "rollout_idx": item.rollout_idx, "stop": result.stop, **result.extra},
            ))

        # Track progress
        elapsed = time.perf_counter() - t0
        rollout_progress[step] = rollout_progress.get(step, 0) + 1
        label = f"{item.env_name}:r{item.rollout_idx}"
        if step in pending_rollouts:
            pending_rollouts[step].discard(label)
        total = rollout_totals.get(step, 0)
        log.info(
            f"Step {step}: rollout {rollout_progress[step]}/{total} done "
            f"({label}) "
            f"reward={result.reward} stop={result.stop} "
            f"tokens={len(result.response_tokens)} elapsed={elapsed:.1f}s"
        )

        return result

    # -- Main loop --
    step = 0
    metrics_file = open(os.path.join(config.log_path, "metrics.jsonl"), "a")

    while True:
        if num_steps is not None and step >= num_steps:
            break

        t_step_start = time.perf_counter()

        # Select batch of task groups
        random.shuffle(task_ids)
        batch_task_ids = task_ids[:config.batch_size]

        # Run all rollouts for this batch concurrently
        rollout_coros: list[tuple[str, Any]] = []
        for tid in batch_task_ids:
            for item in dataset[tid]:
                coro = complete_rollout(tid, item, step)
                rollout_coros.append((tid, coro))

        total_rollouts = len(rollout_coros)
        rollout_progress[step] = 0
        rollout_totals[step] = total_rollouts
        log.info(f"Step {step}: launching {total_rollouts} rollouts for {len(batch_task_ids)} task groups")
        t_rollout_start = time.perf_counter()

        # Wrap each rollout with a timeout
        async def timed_rollout(coro: Any) -> RolloutResult:
            return await asyncio.wait_for(coro, timeout=config.rollout_timeout)

        # Periodic progress reporter
        progress_done = asyncio.Event()

        async def progress_reporter() -> None:
            while not progress_done.is_set():
                await asyncio.sleep(15.0)
                if not progress_done.is_set():
                    done = rollout_progress.get(step, 0)
                    elapsed = time.perf_counter() - t_rollout_start
                    log.info(
                        f"Step {step}: rollout progress {done}/{total_rollouts} "
                        f"({elapsed:.0f}s elapsed)"
                    )

        reporter_task = asyncio.create_task(progress_reporter())

        # Gather all, allowing individual failures (including timeouts)
        raw_results = await asyncio.gather(
            *(timed_rollout(coro) for _, coro in rollout_coros),
            return_exceptions=True,
        )

        progress_done.set()
        reporter_task.cancel()

        # Group rollouts by task_id
        rollout_map: dict[str, list[RolloutResult]] = defaultdict(list)
        n_timeout = 0
        n_error = 0
        for (tid, _), result in zip(rollout_coros, raw_results):
            if isinstance(result, asyncio.TimeoutError):
                n_timeout += 1
                continue
            if isinstance(result, BaseException):
                n_error += 1
                log.warning(f"Rollout raised exception: {result}")
                continue
            rollout_map[tid].append(result)

        if n_timeout > 0 or n_error > 0:
            log.warning(
                f"Step {step}: {n_timeout} rollouts timed out, {n_error} raised errors "
                f"({total_rollouts - n_timeout - n_error}/{total_rollouts} succeeded)"
            )

        t_rollout_end = time.perf_counter()

        # Build groups and compute advantages
        included_groups: list[RolloutGroup] = []
        excluded_groups: list[RolloutGroup] = []

        for tid in batch_task_ids:
            rollouts = rollout_map.get(tid, [])
            if not rollouts:
                continue

            successful = [r for r in rollouts if r.response_tokens and r.reward is not None]
            n_failed = len(rollouts) - len(successful)
            ec = env_configs[rollouts[0].env_name]

            if not successful or n_failed > ec.max_failing_rollouts:
                excluded_groups.append(RolloutGroup(rollouts=rollouts))
                continue

            rewards = [cast(float, r.reward) for r in successful]
            if len(rewards) > 1 and stdev(rewards) < 1e-8:
                excluded_groups.append(RolloutGroup(rollouts=successful))
                continue

            advantages = compute_advantages(rewards, config.advantage_calculation)
            for adv, rollout in zip(advantages, successful):
                rollout.extra["advantage"] = adv

            included_groups.append(RolloutGroup(rollouts=successful))

        if not included_groups:
            log.warning(f"Step {step}: no valid groups, skipping")
            step += 1
            continue

        # Build Datum list for training
        data: list[tinker.Datum] = []
        for group in included_groups:
            for rollout in group.rollouts:
                adv = rollout.extra.get("advantage", 0.0)
                datum = build_datum(
                    all_tokens=rollout.prompt_tokens,
                    response_tokens=rollout.response_tokens,
                    response_logprobs=rollout.response_logprobs,
                    advantage=adv,
                    max_length=config.max_total_tokens,
                )
                if datum is not None:
                    data.append(datum)

        if not data:
            log.warning(f"Step {step}: no valid data, skipping")
            step += 1
            continue

        # -- Training step --
        log.info(f"Step {step}: training on {len(data)} rollouts from {len(included_groups)} groups")
        t_train_start = time.perf_counter()

        fwdbwd_future = await training_client.forward_backward_async(
            data,
            loss_fn=config.loss_fn,
            loss_fn_config=config.loss_fn_config if config.loss_fn_config else None,
        )
        optim_future = await training_client.optim_step_async(
            tinker_types.AdamParams(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        )

        fwdbwd_result = await fwdbwd_future
        optim_result = await optim_future

        # Loss is in fwdbwd_result.metrics["loss:sum"] per Tinker docs
        train_loss: float | None = None
        if hasattr(fwdbwd_result, "metrics") and isinstance(fwdbwd_result.metrics, dict):
            train_loss = fwdbwd_result.metrics.get("loss:sum")
        if train_loss is None:
            fwdbwd_fields = {k: type(v).__name__ for k, v in fwdbwd_result.__dict__.items()} if hasattr(fwdbwd_result, "__dict__") else {}
            log.warning(f"Could not find loss:sum in metrics. Fields: {fwdbwd_fields}, metrics: {getattr(fwdbwd_result, 'metrics', 'N/A')}")

        t_train_end = time.perf_counter()
        log.info(f"Step {step}: training done in {t_train_end - t_train_start:.1f}s, loss={train_loss}")

        # -- Update sampler --
        t_sampler_start = time.perf_counter()
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"step_{step + 1:06d}"
        )
        t_sampler_end = time.perf_counter()
        log.info(f"Step {step}: sampler updated in {t_sampler_end - t_sampler_start:.1f}s")

        # -- Checkpointing --
        if config.save_every > 0 and (step + 1) % config.save_every == 0:
            save_name = f"checkpoint_{step + 1:06d}"
            save_future = await training_client.save_state_async(save_name)
            await save_future
            log.info(f"Saved checkpoint: {save_name}")

        # -- Metrics --
        rollout_stats = get_rollout_stats(included_groups)
        all_advantages = [r.extra.get("advantage", 0.0) for g in included_groups for r in g.rollouts]

        # Forward all tinker-reported metrics (loss:sum, kl, clip frac, etc.)
        tinker_metrics: dict[str, Any] = {}
        if hasattr(fwdbwd_result, "metrics") and isinstance(fwdbwd_result.metrics, dict):
            for k, v in fwdbwd_result.metrics.items():
                safe_key = k.replace(":", "/")
                tinker_metrics[f"tinker/{safe_key}"] = v

        metrics = {
            "step": step,
            "lr": config.learning_rate,
            "loss": train_loss,
            **tinker_metrics,
            "num_groups": len(included_groups),
            "num_excluded_groups": len(excluded_groups),
            "num_rollouts": len(data),
            "num_timed_out": n_timeout,
            "num_errors": n_error,
            "rollout": rollout_stats,
            "advantage": get_stats(all_advantages) if all_advantages else {},
            "time": {
                "rollout_s": t_rollout_end - t_rollout_start,
                "train_s": t_train_end - t_train_start,
                "sampler_s": t_sampler_end - t_sampler_start,
                "total_s": time.perf_counter() - t_step_start,
            },
        }

        # Flatten for wandb
        flat_metrics = _flatten_dict(metrics)
        wandb.log(flat_metrics, step=step)

        metrics_file.write(json.dumps(metrics, default=str) + "\n")
        metrics_file.flush()

        log.info(
            f"Step {step} complete: "
            f"loss={train_loss}, "
            f"groups={len(included_groups)}, "
            f"reward_mean={rollout_stats.get('reward', {}).get('mean', '?')}, "
            f"time={time.perf_counter() - t_step_start:.1f}s "
            f"(rollout={t_rollout_end - t_rollout_start:.1f}s "
            f"train={t_train_end - t_train_start:.1f}s "
            f"sampler={t_sampler_end - t_sampler_start:.1f}s)"
        )

        # Clean up progress tracking
        rollout_progress.pop(step, None)
        rollout_totals.pop(step, None)

        step += 1

    metrics_file.close()
    log.info("Training complete")
    wandb.finish()


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict for wandb logging."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


# ============================================================================
# Entry point
# ============================================================================


def main(config_path: str, **overrides: Any) -> None:
    """Load config from YAML/JSON and run training."""
    with open(config_path) as f:
        if config_path.endswith(".json"):
            raw = json.load(f)
        else:
            import yaml
            raw = yaml.safe_load(f)

    # Apply overrides
    for k, v in overrides.items():
        parts = k.split(".")
        obj = raw
        for part in parts[:-1]:
            obj = obj.setdefault(part, {})
        obj[parts[-1]] = v

    config = Config(**raw)
    settings = Settings()  # type: ignore

    asyncio.run(run(config, settings))


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    import fire
    fire.Fire(main)
