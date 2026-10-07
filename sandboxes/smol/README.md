# OpenReward environments on smol machines

[smol machines](https://smolmachines.com) are microVMs that boot in about 200 ms, run on a laptop (macOS, Linux) or on smol cloud, and can be branched while running.

`SmolSandbox` is a drop-in for the sandbox `AsyncOpenReward.sandbox(settings)` returns. It has the same `start`, `stop`, `run`, `check_run`, `upload` and `download` methods and returns the same `RunResult`. Environments and toolsets need no other change.

## Why

- **Develop and run locally.** Environments boot on your own machine, with no account and no network. The same code runs on smol cloud by setting one key.
- **Branch a prepared state.** Run the slow setup once (clone a repo, install dependencies, start services). Then `branch()` it: each rollout gets its own copy of the running machine, memory and disk included, in well under a second.
- **VM isolation.** Every rollout gets its own kernel, so agent code can't reach the host or other rollouts.

## Install

```bash
curl -sSL https://smolmachines.com/install.sh | bash   # local engine (skip for cloud only)
pip install openreward smolmachines
```

To use smol cloud, set `SMOL_CLOUD_TOKEN`, or pass `target="cloud"` and `api_key=...`.

## Use

Swap the sandbox in your environment:

```python
from openreward import SandboxSettings
from smol_sandbox import SmolSandbox

settings = SandboxSettings(
    environment="you/YourEnv",
    image="python:3.12-slim",
    machine_size="1:2",
    block_network=True,
)

# before: self.sandbox = AsyncOpenReward(api_key=...).sandbox(settings)
self.sandbox = SmolSandbox(settings)
```

Everything else stays the same:

```python
await self.sandbox.start()
output, code = await self.sandbox.run("pytest -q")
```

Branch a prepared machine for parallel rollouts:

```python
base = SmolSandbox(settings, branchable=True)
await base.start()
await base.check_run("git clone https://github.com/org/repo /repo && pip install -e /repo")

rollouts = [await base.branch() for _ in range(16)]   # each starts with /repo installed
```

`smol_env.py` is a complete example. It is the `orwd init` sandbox template with the sandbox swapped, plus a task that rewards fixing a failing test. `server.py` serves it.

```bash
python server.py
```

## Settings mapping

| `SandboxSettings` | smol machine |
|---|---|
| `image` | image (any OCI image) |
| `machine_size="C:G"` | `C` vCPUs (rounded up) and `G` GB of memory |
| `env` | environment variables |
| `block_network=True` | networking off |
| `host_aliases` | written to `/etc/hosts` at start |
| `bucket_config`, `sidecars` | not supported: raises `NotImplementedError` |

`run` keeps OpenReward's behavior: bash, stdout and stderr in one output, truncation at `max_bytes`, and exit 124 with `timed_out=True` at `timeout`. `upload` has no 10 MB limit.

## Test

```bash
pytest test_smol_sandbox.py
```
