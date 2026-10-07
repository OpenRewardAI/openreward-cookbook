"""End-to-end checks for SmolSandbox on real smol machines.

Needs smolvm installed (local) or SMOL_CLOUD_TOKEN set (cloud):
    pytest test_smol_sandbox.py
"""
import asyncio

import pytest
from openreward import SandboxSettings
from openreward.api.sandboxes.types import RunResult

from smol_env import BashParams, SmolEnv, SubmitParams
from smol_sandbox import SmolSandbox

SETTINGS = SandboxSettings(environment="smol/test", image="python:3.12-slim", machine_size="1:2")


def run(coro):
    return asyncio.run(coro)


def test_run_matches_the_openreward_sandbox(tmp_path):
    async def go():
        async with SmolSandbox(SETTINGS) as sb:
            r = await sb.run("echo out; echo err >&2; exit 3")
            assert isinstance(r, RunResult)
            output, code = r
            assert (output, code) == ("out\nerr\n", 3)
            assert not r.truncated and not r.timed_out

            r = await sb.run("head -c 1000 /dev/zero | tr '\\0' x", max_bytes=10)
            assert r.truncated and r.output == "x" * 10

            r = await sb.run("sleep 30", timeout=2)
            assert r.timed_out and r.return_code == 124

            assert await sb.check_run("python3 -c 'print(6 * 7)'") == "42\n"
            with pytest.raises(RuntimeError, match="Command failed"):
                await sb.check_run("false")

            local = tmp_path / "data.bin"
            local.write_bytes(bytes(range(256)) * 100)
            await sb.upload(local, "/tmp/data.bin")
            assert await sb.download("/tmp/data.bin") == local.read_bytes()
            assert (await sb.check_run("wc -c < /tmp/data.bin")).strip() == "25600"

    run(go())


def test_branch_copies_a_running_machine():
    async def go():
        async with SmolSandbox(SETTINGS, branchable=True) as sb:
            await sb.check_run("echo prepared > /root/state")
            children = [await sb.branch() for _ in range(3)]
            try:
                for i, child in enumerate(children):
                    await child.check_run(f"echo {i} >> /root/state")
                for i, child in enumerate(children):
                    assert await child.check_run("cat /root/state") == f"prepared\n{i}\n"
                assert await sb.check_run("cat /root/state") == "prepared\n"
            finally:
                for child in children:
                    await child.stop()

    run(go())


def test_example_environment_rewards_a_fix():
    async def go():
        task = SmolEnv.list_tasks("train")[0]
        env = SmolEnv(task)
        await env.setup()
        try:
            out = await env.submit(SubmitParams())
            assert out.reward == 0.0 and out.finished
            await env.bash(BashParams(command="sed -i 's/a - b/a + b/' /task/solution.py"))
            out = await env.submit(SubmitParams())
            assert out.reward == 1.0, out.blocks[0].text
        finally:
            await env.teardown()

    run(go())
