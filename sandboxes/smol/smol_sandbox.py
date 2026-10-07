"""Run OpenReward environments in smol machines.

``SmolSandbox`` is a drop-in replacement for the sandbox that
``AsyncOpenReward.sandbox(settings)`` returns: same ``start`` / ``stop`` /
``run`` / ``check_run`` / ``upload`` / ``download`` methods and the same
``RunResult``, so an environment and every OpenReward toolset (CLI, Claude Code,
Codex, ...) work unchanged on a smol machine.

It adds what smol machines are for: ``branch()`` copies the running machine,
memory and disk included, so many rollouts start from one prepared state
instead of rebuilding it, and ``checkpoint()`` saves that state to a file.

Machines run locally (the default, on macOS or Linux with KVM) or on smol cloud
when ``SMOL_CLOUD_TOKEN`` is set or ``target="cloud"`` is passed.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Optional, Union

from openreward import SandboxSettings
from openreward.api.sandboxes.types import RunResult
from smol import AsyncMachine, ConnectOptions, ExecOptions, MachineConfig, ResourceSpec

try:  # Match OpenReward's output cleaning when it is available.
    from openreward.api.sandboxes.client import _sanitise_content
except ImportError:  # pragma: no cover - older or newer SDKs
    def _sanitise_content(text: str) -> str:
        return text


def _resources(machine_size: str, block_network: bool) -> ResourceSpec:
    """``"0.5:1"`` (CPUs : GB of memory) as smol resources. Fractions of a CPU
    round up to a whole vCPU."""
    cpus, _, memory_gb = str(machine_size).partition(":")
    return ResourceSpec(
        cpus=max(1, math.ceil(float(cpus))),
        memory_mb=max(512, int(float(memory_gb or 1) * 1024)),
        network=not block_network,
    )


class SmolSandbox:
    """An OpenReward-compatible sandbox backed by a smol machine.

    Args:
        settings: the same ``SandboxSettings`` an OpenReward sandbox takes.
            ``environment`` is not needed and ignored. ``bucket_config`` and
            ``sidecars`` have no smol equivalent and are refused rather than
            silently dropped.
        target: ``"local"``, ``"cloud"`` or ``None`` (cloud when
            ``SMOL_CLOUD_TOKEN`` is set, else local).
        api_key: smol cloud API key; defaults to ``SMOL_CLOUD_TOKEN``.
        branchable: start the machine as a branch source, so ``branch()``
            can copy it while it runs.
        name: machine name; generated when omitted.
    """

    def __init__(
        self,
        settings: SandboxSettings,
        *,
        target: Optional[str] = None,
        api_key: Optional[str] = None,
        branchable: bool = False,
        name: Optional[str] = None,
        _machine: Optional[AsyncMachine] = None,
    ) -> None:
        if settings.bucket_config is not None:
            raise NotImplementedError("bucket_config has no smol equivalent; upload files instead")
        if settings.sidecars:
            raise NotImplementedError("sidecars are not supported; run the service inside the machine")
        self.settings = settings
        self._conn = ConnectOptions(target=target, api_key=api_key)
        self._branchable = branchable
        self._name = name or f"or-{uuid.uuid4().hex[:12]}"
        self._machine = _machine

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def sid(self) -> Optional[str]:
        return self._machine.id if self._machine else None

    async def start(self) -> None:
        if self._machine is not None:
            return
        self._machine = await AsyncMachine.create(
            MachineConfig(
                name=self._name,
                image=self.settings.image,
                env=self.settings.env,
                resources=_resources(self.settings.machine_size, self.settings.block_network),
                branchable=self._branchable or None,
            ),
            self._conn,
        )
        for alias in self.settings.host_aliases or []:
            line = f"{alias.ip} {' '.join(alias.hostnames)}"
            await self.check_run(f"echo '{line}' >> /etc/hosts")

    async def stop(self) -> None:
        machine, self._machine = self._machine, None
        if machine is not None:
            await machine.delete()

    async def __aenter__(self) -> "SmolSandbox":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    def _started(self) -> AsyncMachine:
        if self._machine is None:
            raise RuntimeError("Sandbox not started. Call start() or use as context manager.")
        return self._machine

    # ── the OpenReward sandbox API ────────────────────────────────────────

    async def run(
        self,
        cmd: str,
        timeout: Optional[float] = 300,
        max_bytes: Optional[int] = 50_000,
        sanitise: bool = True,
    ) -> RunResult:
        """Run ``cmd`` with bash, stdout and stderr interleaved as one output,
        like an OpenReward sandbox. A command still running at ``timeout``
        is killed and exits 124."""
        result = await self._started().exec(
            ["/bin/bash", "-c", "exec 2>&1\n" + cmd],
            ExecOptions(timeout=timeout),
        )
        output = result.stdout + result.stderr
        if sanitise:
            output = _sanitise_content(output)
        truncated = max_bytes is not None and len(output) > max_bytes
        if truncated:
            output = output[:max_bytes]
        return RunResult(
            output=output,
            return_code=result.exit_code,
            truncated=truncated,
            timed_out=timeout is not None and result.exit_code == 124,
            sanitised=sanitise,
        )

    async def check_run(
        self,
        cmd: str,
        timeout: Optional[float] = 300,
        max_bytes: Optional[int] = 50_000,
        sanitise: bool = True,
    ) -> str:
        """Run ``cmd`` and raise if it exits non-zero."""
        result = await self.run(cmd, timeout=timeout, max_bytes=max_bytes, sanitise=sanitise)
        if result.return_code != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.output}")
        return result.output

    async def upload(self, local_path: Union[str, Path], container_path: str) -> None:
        """Copy a local file into the machine. Unlike the hosted sandbox,
        there is no 10 MB limit."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        await self._started().write_file(container_path, local_path.read_bytes())

    async def download(self, container_path: str) -> bytes:
        """Read a file from the machine."""
        return await self._started().read_file(container_path)

    # ── smol extras ───────────────────────────────────────────────────────

    async def branch(self, name: Optional[str] = None) -> "SmolSandbox":
        """Copy this running sandbox, memory and disk included, into a new
        independent one. This sandbox keeps running. Requires
        ``branchable=True``. Stop the returned sandbox when done."""
        child_name = name or f"{self._name}-{uuid.uuid4().hex[:6]}"
        child = await self._started().branch(child_name)
        return SmolSandbox(self.settings, target=self._conn.target, api_key=self._conn.api_key,
                           name=child_name, _machine=child)

    async def checkpoint(self, output: Optional[str] = None):
        """Save this running sandbox's full state. Locally ``output`` is the
        ``.smolcheckpoint`` path; on smol cloud it is stored for you."""
        return await self._started().checkpoint(output)
