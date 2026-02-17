"""Fetch tasks from OpenReward environments and write SLIME-compatible JSONL.

Usage:
    python prepare_tasks.py --config train_config.yaml --output tasks.jsonl
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


async def fetch_and_write(config_path: str, output_path: str, seed: int = 42) -> None:
    """Fetch all tasks from configured environments and write to JSONL."""
    config = load_integration_config(config_path)
    client = AsyncOpenReward(api_key=__import__("os").environ["OPENREWARD_API_KEY"])

    random.seed(seed)
    rows: list[dict] = []

    for env_name, env_config in config.environments.items():
        env = client.environments.get(env_name)
        logger.info(f"Fetching tasks from {env_name} (deployment: {env.deployment_name})")

        for split in env_config.splits:
            tasks = await env.list_tasks(split)
            random.shuffle(tasks)
            if env_config.num_samples is not None:
                tasks = tasks[:env_config.num_samples]

            logger.info(f"  {split}: {len(tasks)} tasks")

            for task in tasks:
                spec_hash = hashlib.sha256(
                    json.dumps(task.task_spec, sort_keys=True).encode()
                ).hexdigest()[:16]

                rows.append({
                    "prompt": f"[{env_name}:{split}:{spec_hash}]",
                    "label": "",
                    "metadata": json.dumps({
                        "env_name": env_name,
                        "deployment_name": task.deployment_name,
                        "server_name": task.server_name,
                        "environment_name": task.environment_name,
                        "namespace": task.namespace,
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(fetch_and_write(args.config, args.output, args.seed))


if __name__ == "__main__":
    main()