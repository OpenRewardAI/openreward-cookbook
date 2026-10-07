"""An OpenReward environment whose tasks run in smol machines.

The OpenReward sandbox template with one change: the sandbox is a
``SmolSandbox`` instead of ``AsyncOpenReward.sandbox(...)``. The task asks the
agent to fix a bug so a test passes; ``submit`` runs the test and returns the
reward.
"""
from typing import List

from openreward import SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from pydantic import BaseModel

from smol_sandbox import SmolSandbox

TASKS = {
    "train": [
        {
            "task_id": "add",
            "source": "def add(a, b):\n    return a - b\n",
            "test": "from solution import add\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
        },
    ],
    "test": [
        {
            "task_id": "mean",
            "source": "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n",
            "test": "from solution import mean\nassert mean([1, 2, 3]) == 2\nassert mean([5]) == 5\n",
        },
    ],
}


class BashParams(BaseModel, extra="forbid"):
    command: str


class SubmitParams(BaseModel, extra="forbid"):
    pass


class TaskSpec(BaseModel):
    task_id: str
    source: str
    test: str


class SmolEnv(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.task = TaskSpec.model_validate(task_spec)
        self.sandbox = SmolSandbox(
            SandboxSettings(
                environment="smol/SmolEnv",
                image="python:3.12-slim",
                machine_size="1:2",
                block_network=True,
            ),
            # Runs locally by default; a smol cloud key sends it to smol cloud.
            api_key=secrets.get("smol_api_key"),
        )

    async def setup(self) -> None:
        await self.sandbox.start()
        await self.sandbox.check_run(
            "mkdir -p /task && cd /task"
            f" && cat > solution.py <<'EOF'\n{self.task.source}EOF\n"
            f"cat > /task/test_solution.py <<'EOF'\n{self.task.test}EOF"
        )

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @tool(concurrent=True)
    async def bash(self, params: BashParams) -> ToolOutput:
        """Executes a bash command in the environment."""
        result = await self.sandbox.run(params.command.strip())
        output, code = result
        if result.truncated:
            output = f"...(truncated, output exceeded limit)\n{output}"
        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code, "truncated": result.truncated},
            reward=0.0,
            finished=False,
        )

    @tool(exclusive=True)
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Runs the test. Reward is 1 if it passes, else 0. Ends the episode."""
        output, code = await self.sandbox.run("cd /task && python3 test_solution.py", timeout=60)
        passed = code == 0
        return ToolOutput(
            blocks=[TextBlock(text="Test passed." if passed else f"Test failed.\n{output}")],
            metadata={"passed": passed},
            reward=1.0 if passed else 0.0,
            finished=True,
        )

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=(
            "/task/solution.py has a bug, so /task/test_solution.py fails. "
            "Fix solution.py without changing the test, then call submit."
        ))]

    @classmethod
    def list_tasks(cls, split: str) -> List[JSONObject]:
        if split not in TASKS:
            raise ValueError(f"Unknown split: {split}")
        return TASKS[split]

    @classmethod
    def list_splits(cls) -> list[str]:
        return list(TASKS)
