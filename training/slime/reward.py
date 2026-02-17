"""Custom SLIME reward function for OpenReward environments.

Used with: --custom-rm-path reward.py

Reads intermediate rewards stored in sample.metadata by generate.py
and reduces them to a single scalar reward per rollout.
"""
import json
import os
from typing import Any

from config import load_integration_config, reduce_rewards, IntegrationConfig

_config: IntegrationConfig | None = None


def _get_config() -> IntegrationConfig:
    """Lazily load the integration config."""
    global _config
    if _config is None:
        _config = load_integration_config()
    return _config


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> float:
    """Compute the final scalar reward for a rollout.

    Reads the per-step rewards and stop_reason from sample.metadata,
    applies nonterminal penalty, reduction, and clamping per env config.
    """
    config = _get_config()
    meta = json.loads(sample.metadata)

    env_name: str = meta["env_name"]
    env_cfg = config.environments[env_name]

    rewards: list[float] = meta.get("rewards", [])
    stop_reason: str = meta.get("stop_reason", "max_turns")

    # Append nonterminal penalty if agent didn't finish
    if stop_reason != "reached_terminal_state" and env_cfg.nonterminal_reward is not None:
        rewards.append(env_cfg.nonterminal_reward)

    if not rewards:
        return 0.0

    reward = reduce_rewards(rewards, env_cfg.reward_reduction)

    if env_cfg.min_reward is not None:
        reward = max(env_cfg.min_reward, reward)
    if env_cfg.max_reward is not None:
        reward = min(env_cfg.max_reward, reward)

    return reward