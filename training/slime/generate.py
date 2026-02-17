"""Custom SLIME generate function backed by OpenReward environments.

Used with: --custom-generate-function-path generate.py

Runs a multi-turn agent rollout per sample:
  1. Opens an OpenReward session for the task
  2. Gets tools + prompt from the environment
  3. Loops: generate (SGLang) → parse tool call → execute (OpenReward) → append result
  4. Returns a Sample with tokens, response_length, loss_mask populated
"""
import asyncio
import hashlib
import json
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np
from transformers import AutoTokenizer

from openreward import AsyncOpenReward
from openreward.api.environments.client import Task
from openreward.api.environments.types import TextBlock, ToolCallError, ToolOutput
from openreward.api.rollouts.serializers.base import (
    AssistantMessage,
    ReasoningItem,
    SystemMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)

from config import (
    IntegrationConfig,
    apply_chat_template,
    format_tool_spec,
    load_integration_config,
    parse_tool_call,
    reduce_rewards,
    resolve_secrets,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Module-level singleton — initialized once on first generate() call
# ============================================================================

class _State:
    or_client: AsyncOpenReward | None = None
    http_session: aiohttp.ClientSession | None = None
    tokenizer: Any | None = None
    config: IntegrationConfig | None = None
    router_url: str | None = None


_state = _State()


async def _ensure_state(args: Any) -> _State:
    """Lazily initialize shared state on first call."""
    if _state.or_client is None:
        _state.or_client = AsyncOpenReward(api_key=os.environ["OPENREWARD_API_KEY"])

    if _state.tokenizer is None:
        model_name = getattr(args, "tokenizer_name_or_path", None) or getattr(args, "hf_checkpoint", None)
        if model_name is None:
            raise RuntimeError("Cannot determine tokenizer: set --hf-checkpoint or tokenizer_name_or_path")
        _state.tokenizer = AutoTokenizer.from_pretrained(model_name)

    if _state.http_session is None:
        _state.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
        )

    if _state.config is None:
        _state.config = load_integration_config()

    if _state.router_url is None:
        # SLIME exposes the router address in args
        ip = getattr(args, "sglang_router_ip", None) or "localhost"
        port = getattr(args, "sglang_router_port", None) or 30000
        _state.router_url = f"http://{ip}:{port}"

    return _state


# ============================================================================
# SGLang HTTP generation
# ============================================================================

@dataclass
class GenResult:
    """Result from a single SGLang generation call."""
    text: str
    output_ids: list[int]
    logprobs: list[float]


async def sglang_generate(
    session: aiohttp.ClientSession,
    router_url: str,
    text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
) -> GenResult:
    """Call SGLang's /generate endpoint with logprob return."""
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "include_stop_str_in_output": True,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with session.post(f"{router_url}/generate", json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()

    output_ids = data.get("output_ids", [])
    raw_logprobs = data.get("meta_info", {}).get("output_token_logprobs", [])
    # Each entry is [logprob, token_id, decoded_text_or_null]
    logprobs = [entry[0] for entry in raw_logprobs] if raw_logprobs else []

    # Ensure alignment — pad/truncate if lengths mismatch
    if len(logprobs) < len(output_ids):
        logprobs.extend([0.0] * (len(output_ids) - len(logprobs)))
    elif len(logprobs) > len(output_ids):
        logprobs = logprobs[:len(output_ids)]

    return GenResult(text=data["text"], output_ids=output_ids, logprobs=logprobs)


def _append_text_tokens(
    text: str,
    tokenizer: Any,
    ids: list[int],
    lps: list[float],
    mask: list[int],
    mask_val: int,
) -> None:
    """Tokenize text and append to token-level tracking lists with logprob=0."""
    toks = tokenizer.encode(text, add_special_tokens=False)
    ids.extend(toks)
    lps.extend([0.0] * len(toks))
    mask.extend([mask_val] * len(toks))


def _find_think_boundary(text: str, output_ids: list[int], tokenizer: Any) -> int:
    """Find token index where content starts (after </think> and trailing whitespace).

    Returns index into output_ids. Everything before is thinking (mask=0),
    everything from this index onward is content (mask=1).
    """
    idx = text.find("</think>")
    if idx < 0:
        return 0
    end = idx + len("</think>")
    # Skip trailing whitespace between </think> and content
    while end < len(text) and text[end] in "\n\r ":
        end += 1
    # Approximate token boundary by encoding the thinking prefix
    n_think_tokens = len(tokenizer.encode(text[:end], add_special_tokens=False))
    return min(n_think_tokens, len(output_ids))


# ============================================================================
# OpenReward rollout upload (fire-and-forget, mirrors the Tinker version)
# ============================================================================

async def _upload_rollout(
    or_client: AsyncOpenReward,
    run_name: str,
    rollout_name: str,
    deployment_name: str,
    split: str,
    blocks: list[dict],
    task_spec: Any,
    metadata: dict,
) -> None:
    """Upload rollout to OpenReward for logging/visualization."""
    try:
        def _sync() -> None:
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
                    message=block["message"],
                    reward=block.get("reward"),
                    is_finished=block.get("is_finished", False),
                    metadata=block.get("metadata"),
                )
        await asyncio.to_thread(_sync)
    except Exception:
        logger.warning(f"Failed to upload rollout {rollout_name}: {traceback.format_exc()}")


# ============================================================================
# Core generate function — SLIME custom generate interface
# ============================================================================

async def generate(args: Any, sample: Any, sampling_params: Any) -> Any:
    """Run a multi-turn agent rollout through an OpenReward environment.

    This is the entry point called by SLIME via --custom-generate-function-path.
    It receives a Sample with metadata containing the task/env info, runs the
    agent loop using SGLang for generation and OpenReward for tool execution,
    and returns the Sample with tokens, response_length, and loss_mask set.
    """
    state = await _ensure_state(args)
    assert state.config is not None
    assert state.or_client is not None
    assert state.http_session is not None
    assert state.tokenizer is not None
    assert state.router_url is not None

    meta = json.loads(sample.metadata)
    env_name: str = meta["env_name"]
    env_cfg = state.config.environments[env_name]
    task = Task(
        server_name=meta["server_name"],
        environment_name=meta["environment_name"],
        task_spec=meta["task_spec"],
        namespace=meta.get("namespace"),
    )

    secrets = resolve_secrets(env_cfg, state.config.secrets)
    env = state.or_client.environments.get(meta["deployment_name"])
    tokenizer = state.tokenizer

    max_response_tokens = getattr(args, "rollout_max_response_len", 4096)
    temperature = getattr(args, "rollout_temperature", 1.0)
    top_p = getattr(args, "rollout_top_p", 1.0)
    stop_strings = ["<|im_end|>", "</tool_call>"]

    # Track text segments for conversation building and OpenReward upload
    segments: list[tuple[str, bool]] = []
    rewards: list[float] = []
    or_blocks: list[dict] = []  # for OpenReward upload
    stop_reason = "max_turns"
    num_tool_calls = 0
    num_failed_tool_calls = 0

    # Token-level tracking (parallel to segments, but exact token alignment)
    response_ids: list[int] = []
    response_lps: list[float] = []
    response_mask: list[int] = []

    try:
        session = env.session(task, secrets=secrets if secrets else None)
        async with session as active:
            # --- Build initial prompt from environment ---
            tools = await active.list_tools()
            tools_str = "\n".join(format_tool_spec(t) for t in tools)
            system_msg = state.config.system_prompt_template.format(tools=tools_str)

            prompt_blocks = await active.get_prompt()
            user_text = "".join(b.text for b in prompt_blocks if b.type == "text")

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_text},
            ]
            prompt_text = apply_chat_template(messages)
            segments.append((prompt_text, False))

            or_blocks.append({"message": SystemMessage(content=system_msg)})
            or_blocks.append({"message": UserMessage(content=user_text)})

            # --- Multi-turn agent loop ---
            for _turn in range(env_cfg.max_turns):
                # Assistant turn prefix
                if state.config.thinking:
                    prefix = "\n<|im_start|>assistant\n"
                else:
                    prefix = "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                segments.append((prefix, False))
                _append_text_tokens(prefix, tokenizer, response_ids, response_lps, response_mask, 0)

                # Build full text and generate
                full_text = "".join(seg[0] for seg in segments)

                # Rough token budget check
                prompt_tok_count = len(tokenizer.encode(full_text, add_special_tokens=False))
                effective_max = min(
                    max_response_tokens,
                    getattr(args, "rollout_max_total_len", 32768) - prompt_tok_count,
                )
                if effective_max <= 0:
                    stop_reason = "max_tokens"
                    break

                gen = await sglang_generate(
                    state.http_session,
                    state.router_url,
                    full_text,
                    max_new_tokens=effective_max,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_strings,
                )

                if not gen.text.strip():
                    nudge = "No tool call detected. Please use the provided tools to complete the task."
                    segments.append(("<|im_end|>", False))
                    segments.append((f"\n<|im_start|>user\n{nudge}<|im_end|>", False))
                    _append_text_tokens("<|im_end|>", tokenizer, response_ids, response_lps, response_mask, 0)
                    _append_text_tokens(f"\n<|im_start|>user\n{nudge}<|im_end|>", tokenizer, response_ids, response_lps, response_mask, 0)
                    or_blocks.append({"message": UserMessage(content=nudge)})
                    continue

                # --- Append SGLang tokens with thinking/content mask split ---
                thinking_text, content_text = _split_thinking(gen.text, state.config.thinking)
                if thinking_text is not None:
                    # Text segments for conversation building (reformatted)
                    segments.append((f"<think>\n{thinking_text}\n</think>\n\n", False))
                    or_blocks.append({"message": ReasoningItem(content=thinking_text)})
                    generated_for_parse = content_text

                    # Token-level: split SGLang output_ids at thinking/content boundary
                    boundary = _find_think_boundary(gen.text, gen.output_ids, tokenizer)
                    response_ids.extend(gen.output_ids[:boundary])
                    response_lps.extend(gen.logprobs[:boundary])
                    response_mask.extend([0] * boundary)
                    # Content portion gets mask=1 — will be added with the text segment below
                    content_ids = gen.output_ids[boundary:]
                    content_lps = gen.logprobs[boundary:]
                else:
                    generated_for_parse = gen.text
                    content_ids = gen.output_ids
                    content_lps = gen.logprobs

                tc = parse_tool_call(generated_for_parse)

                if tc is None:
                    # No tool call — content is model-generated (mask=1), then nudge
                    segments.append((generated_for_parse, True))
                    response_ids.extend(content_ids)
                    response_lps.extend(content_lps)
                    response_mask.extend([1] * len(content_ids))

                    segments.append(("<|im_end|>", False))
                    _append_text_tokens("<|im_end|>", tokenizer, response_ids, response_lps, response_mask, 0)

                    nudge = "No tool call detected. Please use the provided tools to complete the task."
                    segments.append((f"\n<|im_start|>user\n{nudge}<|im_end|>", False))
                    _append_text_tokens(f"\n<|im_start|>user\n{nudge}<|im_end|>", tokenizer, response_ids, response_lps, response_mask, 0)

                    if generated_for_parse.strip():
                        or_blocks.append({"message": AssistantMessage(content=generated_for_parse.strip())})
                    or_blocks.append({"message": UserMessage(content=nudge)})
                    continue

                if tc["type"] == "error":
                    num_failed_tool_calls += 1
                    segments.append((generated_for_parse, True))
                    response_ids.extend(content_ids)
                    response_lps.extend(content_lps)
                    response_mask.extend([1] * len(content_ids))

                    segments.append(("<|im_end|>", False))
                    _append_text_tokens("<|im_end|>", tokenizer, response_ids, response_lps, response_mask, 0)

                    error_msg = f"Tool call parse error: {tc['error']}. Please ensure arguments are valid JSON."
                    segments.append((f"\n<|im_start|>user\n{error_msg}<|im_end|>", False))
                    _append_text_tokens(f"\n<|im_start|>user\n{error_msg}<|im_end|>", tokenizer, response_ids, response_lps, response_mask, 0)

                    or_blocks.append({"message": AssistantMessage(content=generated_for_parse.strip())})
                    or_blocks.append({"message": UserMessage(content=error_msg)})
                    continue

                # --- Successful tool call ---
                num_tool_calls += 1

                # Model-generated content including tool call (mask=1)
                segments.append((generated_for_parse, True))
                response_ids.extend(content_ids)
                response_lps.extend(content_lps)
                response_mask.extend([1] * len(content_ids))

                if "</tool_call>" not in generated_for_parse:
                    segments.append(("</tool_call>", True))
                    _append_text_tokens("</tool_call>", tokenizer, response_ids, response_lps, response_mask, 1)

                segments.append(("<|im_end|>", False))
                _append_text_tokens("<|im_end|>", tokenizer, response_ids, response_lps, response_mask, 0)

                call_id = hashlib.md5(f"{_turn}:{tc['name']}".encode()).hexdigest()[:12]
                or_blocks.append({"message": ToolCall(
                    name=tc["name"], content=json.dumps(tc["arguments"]), call_id=call_id,
                )})

                # Execute tool via OpenReward
                try:
                    tool_out = await active.call_tool(tc["name"], tc["arguments"])
                except ToolCallError as exc:
                    tool_out = ToolOutput(
                        blocks=[TextBlock(text=f"Error: {exc}")],
                        metadata={"error": str(exc)},
                        finished=False,
                    )
                    num_failed_tool_calls += 1

                tool_text = "".join(b.text for b in tool_out.blocks if b.type == "text")
                if tool_out.reward is not None:
                    rewards.append(tool_out.reward)

                or_blocks.append({
                    "message": ToolResult(content=tool_text, call_id=call_id),
                    "reward": tool_out.reward,
                    "is_finished": tool_out.finished,
                    "metadata": getattr(tool_out, "metadata", None),
                })

                # Tool response as user turn — loss_mask=0 (environment-generated)
                tool_turn = f"\n<|im_start|>user\n<tool_response>\n{tool_text}\n</tool_response><|im_end|>"
                segments.append((tool_turn, False))
                _append_text_tokens(tool_turn, tokenizer, response_ids, response_lps, response_mask, 0)

                if tool_out.finished:
                    stop_reason = "reached_terminal_state"
                    break

    except Exception as e:
        logger.warning(f"Rollout failed for {env_name}: {e}\n{traceback.format_exc()}")
        stop_reason = "error"

    # --- Build final sample from token-level tracking ---
    prompt_tokens = tokenizer.encode(segments[0][0], add_special_tokens=False) if segments else []
    all_tokens = prompt_tokens + response_ids

    sample.tokens = np.array(all_tokens, dtype=np.int64)
    sample.response_length = len(response_ids)
    sample.loss_mask = np.array(response_mask, dtype=np.int32)
    sample.rollout_log_probs = response_lps
    sample.response = "".join(seg[0] for seg in segments[1:])

    # Compute reward directly on the sample
    final_rewards = list(rewards)
    if stop_reason != "reached_terminal_state" and env_cfg.nonterminal_reward is not None:
        final_rewards.append(env_cfg.nonterminal_reward)
    reward = reduce_rewards(final_rewards, env_cfg.reward_reduction)
    if env_cfg.min_reward is not None:
        reward = max(env_cfg.min_reward, reward)
    if env_cfg.max_reward is not None:
        reward = min(env_cfg.max_reward, reward)
    sample.reward = reward

    # Store extra info in metadata
    meta["rewards"] = rewards
    meta["stop_reason"] = stop_reason
    meta["num_tool_calls"] = num_tool_calls
    meta["num_failed_tool_calls"] = num_failed_tool_calls
    sample.metadata = json.dumps(meta)

    # Fire-and-forget upload to OpenReward
    if or_blocks and state.or_client is not None:
        task_hash = meta.get("task_hash", "unknown")
        rollout_name = f"{env_name}-{task_hash}-{os.urandom(4).hex()}"
        asyncio.create_task(_upload_rollout(
            or_client=state.or_client,
            run_name=os.environ.get("OPENREWARD_RUN_NAME", state.config.openreward_run_name),
            rollout_name=rollout_name,
            deployment_name=meta["deployment_name"],
            split=meta.get("split", "train"),
            blocks=or_blocks,
            task_spec=meta["task_spec"],
            metadata={
                "stop_reason": stop_reason,
                "num_tool_calls": num_tool_calls,
                "reward_sum": sum(rewards) if rewards else None,
            },
        ))

    return sample


def _split_thinking(text: str, enable_thinking: bool) -> tuple[str | None, str]:
    """Split <think>...</think> block from generated text."""
    if not enable_thinking:
        return None, text
    ts, te = "<think>", "</think>"
    si = text.find(ts)
    if si == -1:
        return None, text
    ei = text.find(te, si)
    if ei == -1:
        return None, text
    thinking = text[si + len(ts):ei].strip()
    content = text[ei + len(te):].strip()
    return thinking, content