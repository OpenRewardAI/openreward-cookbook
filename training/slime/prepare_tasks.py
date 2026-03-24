"""Fetch tasks from OpenReward environments and write SLIME-compatible JSONL.

Handles both list_tasks and index-based (num_tasks + get_task) APIs.

Usage:
    python prepare_tasks.py --config train_config.yaml --output tasks.jsonl [--max-tasks 50]
"""
import argparse
import asyncio
import hashlib
import json
import logging
import random

from openreward import AsyncOpenReward

from config import load_integration_config

logger = logging.getLogger(__name__)


async def _fetch_tasks_for_split(env, env_name, split, cap):
    """Fetch tasks using list_tasks or falling back to index-based API."""
    try:
        tasks = await env.list_tasks(split)
        random.shuffle(tasks)
        if cap is not None:
            tasks = tasks[:cap]
        logger.info(f"  {split}: {len(tasks)} tasks (list_tasks)")
        return tasks
    except Exception as e:
        if "not supported" not in str(e) and "index" not in str(e).lower():
            raise
        logger.info(f"  list_tasks not supported for {env_name}, using index-based API")
        num = await env.num_tasks(split)
        effective_cap = min(cap, num) if cap is not None else num
        indices = list(range(num))
        random.shuffle(indices)
        indices = indices[:effective_cap]
        logger.info(f"  {split}: fetching {effective_cap}/{num} tasks")
        tasks = []
        for idx in indices:
            tasks.append(await env.get_task(split, idx))
        return tasks


async def fetch_and_write(config_path: str, output_path: str, max_tasks: int | None, seed: int = 42) -> None:
    """Fetch all tasks from configured environments and write to JSONL."""
    config = load_integration_config(config_path)
    client = AsyncOpenReward()

    random.seed(seed)
    rows: list[dict] = []

    for env_name, env_config in config.environments.items():
        env = client.environments.get(env_name)
        logger.info(f"Fetching tasks from {env_name} (deployment: {env.deployment_name})")

        for split in env_config.splits:
            cap = env_config.num_samples or max_tasks
            tasks = await _fetch_tasks_for_split(env, env_name, split, cap)

            for task in tasks:
                spec_hash = hashlib.sha256(
                    json.dumps(task.task_spec, sort_keys=True).encode()
                ).hexdigest()[:16]

                rows.append({
                    "prompt": f"[{env_name}:{split}:{spec_hash}]",
                    "label": "",
                    "metadata": json.dumps({
                        "env_name": env_name,
                        "deployment_name": env.deployment_name,
                        "server_name": task.server_name,
                        "environment_name": task.environment_name,
                        "namespace": getattr(task, "namespace", None),
                        "split": split,
                        "task_spec": task.task_spec,
                        "task_hash": spec_hash,
                    }),
                })

    random.shuffle(rows)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    logger.info(f"Wrote {len(rows)} tasks to {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prepare OpenReward tasks for SLIME training")
    parser.add_argument("--config", default="train_config.yaml", help="Integration config path")
    parser.add_argument("--output", default="tasks.jsonl", help="Output JSONL path")
    parser.add_argument("--max-tasks", type=int, default=None, help="Cap total tasks per environment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(fetch_and_write(args.config, args.output, args.max_tasks, args.seed))


if __name__ == "__main__":
    main()
