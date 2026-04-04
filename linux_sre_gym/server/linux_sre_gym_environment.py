"""Linux SRE Gym environment orchestration layer."""

from __future__ import annotations

import importlib
import os
import shlex
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:  # pragma: no cover
    class Environment:  # type: ignore[override]
        """Fallback base class when openenv is not installed locally."""


try:
    from .. import models as _models
except ImportError:  # pragma: no cover
    import models as _models  # type: ignore[no-redef]

LinuxSreGymAction = getattr(_models, "LinuxSreGymAction", getattr(_models, "Action"))
LinuxSreGymObservation = getattr(
    _models, "LinuxSreGymObservation", getattr(_models, "Observation")
)
LinuxSreGymProcess = getattr(_models, "LinuxSreGymProcess", getattr(_models, "Process", None))
LinuxSreGymRewardBreakdown = getattr(
    _models,
    "LinuxSreGymRewardBreakdown",
    getattr(_models, "RewardBreakdown", None),
)
LinuxSreGymState = getattr(_models, "LinuxSreGymState", getattr(_models, "State"))


TASK_ORDER = ("triage", "optimization", "security")
TASK_DESCRIPTIONS = {
    "triage": (
        "A runaway user workload is starving CPU and memory. Identify the offender "
        "through Linux diagnostics and mitigate it without harming critical services."
    ),
    "optimization": (
        "The host is disk thrashing because paging is misconfigured. Inspect vm "
        "settings and reduce pressure by tuning swappiness and zswap."
    ),
    "security": (
        "The network stack accepts spoofed packets. Inspect the current sysctls and "
        "harden reverse-path filtering without breaking unrelated settings."
    ),
}
TASK_HINTS = {
    "triage": "Start with process and memory inspection commands such as ps, top, and free -m.",
    "optimization": "Use vmstat, /proc/meminfo, /proc/swaps, and sysctl to inspect paging pressure.",
    "security": "Inspect rp_filter under /proc/sys or via sysctl before writing any networking setting.",
}
DISCOVERY_COMMANDS = {
    "triage": {"ps aux", "top -bn1", "free -m", "cat /proc/meminfo", "cat /proc/loadavg"},
    "optimization": {
        "vmstat",
        "cat /proc/meminfo",
        "cat /proc/swaps",
        "sysctl vm.swappiness",
        "cat /sys/module/zswap/parameters/enabled",
    },
    "security": {
        "sysctl net.ipv4.conf.all.rp_filter",
        "sysctl net.ipv4.conf.default.rp_filter",
        "sysctl -a | grep rp_filter",
        "cat /proc/sys/net/ipv4/conf/all/rp_filter",
        "cat /proc/sys/net/ipv4/conf/default/rp_filter",
    },
}
VERIFICATION_COMMANDS = {
    "triage": {"ps aux", "top -bn1", "free -m"},
    "optimization": {"vmstat", "sysctl vm.swappiness", "cat /sys/module/zswap/parameters/enabled"},
    "security": {
        "sysctl net.ipv4.conf.all.rp_filter",
        "sysctl net.ipv4.conf.default.rp_filter",
        "sysctl -a | grep rp_filter",
    },
}
MAX_SCORE = 1.0


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    progress_reward: float = 0.0
    safety_penalty: float = 0.0
    reward_reason: str = "Command processed."


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_int(raw: str, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class LinuxSreGymEnvironment(Environment):
    """Stateful OpenEnv-compatible environment for Linux SRE incidents."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._reset_count = 0
        self._max_steps = _safe_int(os.getenv("LINUX_SRE_GYM_MAX_STEPS"), 12) or 12
        self._pinned_task = os.getenv("LINUX_SRE_GYM_DEFAULT_TASK")
        self._external_router_active = False
        self._external_grader_active = False
        self._state = self._make_task_state("triage")
        self._refresh_derived_views(self._state)

    def reset(self) -> LinuxSreGymObservation:
        task_id = self._choose_task_for_reset()
        self._state = self._make_task_state(task_id)
        self._refresh_derived_views(self._state)
        self._reset_count += 1

        return self._build_observation(
            stdout=(
                f"Linux SRE Gym ready: {task_id}\n"
                f"Incident: {self._state.task_description}\n"
                "Respond with one Linux command per step."
            ),
            stderr="",
            exit_code=0,
            reward=0.0,
            done=False,
            reason="Episode reset.",
            hint=TASK_HINTS.get(task_id),
        )

    def step(self, action: LinuxSreGymAction) -> LinuxSreGymObservation:  # type: ignore[override]
        if self._state.terminal_locked:
            return self._build_observation(
                stdout="",
                stderr="Episode locked due to an unsafe destructive action. Reset to continue.",
                exit_code=1,
                reward=0.0,
                done=True,
                reason="Episode already terminated by unsafe action.",
                hint=TASK_HINTS.get(self._state.task_id),
            )

        command = action.command.strip()
        if not command:
            return self._build_observation(
                stdout="",
                stderr="empty command",
                exit_code=1,
                reward=-0.05,
                done=False,
                reason="Empty commands are not useful for diagnosis.",
                hint=TASK_HINTS.get(self._state.task_id),
            )

        self._state.step_count += 1
        self._state.command_history.append(command)

        was_resolved_before = self._state.is_resolved
        result = self._dispatch_command(command)
        if result.stderr:
            self._state.last_action_error = result.stderr
        else:
            self._state.last_action_error = None

        self._refresh_derived_views(self._state)
        breakdown = self._compute_reward_breakdown(command, result, was_resolved_before)
        self._state.reward_history.append(breakdown.total)
        self._state.reward_reasons.append(result.reward_reason)
        self._state.last_reward_breakdown = breakdown
        self._state.completion_score = breakdown.score

        done = (
            self._state.is_resolved
            or self._state.terminal_locked
            or self._state.step_count >= self._max_steps
        )

        if self._state.step_count >= self._max_steps and not self._state.is_resolved:
            result.reward_reason = "Step budget exhausted before the incident was fully resolved."

        return self._build_observation(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            reward=breakdown.total,
            done=done,
            reason=result.reward_reason,
            hint=self._hint_after_step(done=done, exit_code=result.exit_code),
        )

    @property
    def state(self) -> LinuxSreGymState:
        return self._state

    def _choose_task_for_reset(self) -> str:
        if self._pinned_task in TASK_ORDER:
            return self._pinned_task
        return TASK_ORDER[self._reset_count % len(TASK_ORDER)]

    def _make_task_state(self, task_id: str) -> LinuxSreGymState:
        external_builder = self._load_task_builder(task_id)
        if external_builder is not None:
            maybe_state = self._invoke_external_builder(external_builder, task_id)
            coerced_state = self._coerce_external_state(maybe_state, task_id)
            if coerced_state is not None:
                return coerced_state

        return self._make_default_task_state(task_id)

    def _make_default_task_state(self, task_id: str) -> LinuxSreGymState:
        if task_id == "triage":
            processes = {
                "1": LinuxSreGymProcess(
                    pid=1,
                    name="systemd",
                    command="/sbin/init",
                    cpu_percent=0.4,
                    memory_mb=64,
                    protected=True,
                    killable=False,
                ),
                "222": LinuxSreGymProcess(
                    pid=222,
                    name="sshd",
                    command="/usr/sbin/sshd -D",
                    cpu_percent=0.2,
                    memory_mb=48,
                    protected=True,
                    killable=False,
                ),
                "4242": LinuxSreGymProcess(
                    pid=4242,
                    name="runaway_hog",
                    command="python runaway_hog.py --threads 32",
                    cpu_percent=96.3,
                    memory_mb=6144,
                ),
                "3100": LinuxSreGymProcess(
                    pid=3100,
                    name="nginx",
                    command="nginx: worker process",
                    cpu_percent=1.1,
                    memory_mb=96,
                ),
            }
            sysctl = {
                "vm.swappiness": "60",
                "vm.zswap_enabled": "N",
                "net.ipv4.conf.all.rp_filter": "1",
                "net.ipv4.conf.default.rp_filter": "1",
            }
            network = {
                "service_health_score": 0.28,
                "spoofing_protection_enabled": True,
            }
        elif task_id == "optimization":
            processes = {
                "1": LinuxSreGymProcess(
                    pid=1,
                    name="systemd",
                    command="/sbin/init",
                    cpu_percent=0.6,
                    memory_mb=64,
                    protected=True,
                    killable=False,
                ),
                "1200": LinuxSreGymProcess(
                    pid=1200,
                    name="postgres",
                    command="postgres: writer",
                    cpu_percent=22.0,
                    memory_mb=2048,
                ),
                "3100": LinuxSreGymProcess(
                    pid=3100,
                    name="node_exporter",
                    command="/usr/bin/node_exporter",
                    cpu_percent=0.4,
                    memory_mb=32,
                ),
            }
            sysctl = {
                "vm.swappiness": "95",
                "vm.zswap_enabled": "N",
                "net.ipv4.conf.all.rp_filter": "1",
                "net.ipv4.conf.default.rp_filter": "1",
            }
            network = {
                "thrashing": True,
                "service_health_score": 0.42,
                "swap_read_latency_ms": 46,
            }
        else:
            processes = {
                "1": LinuxSreGymProcess(
                    pid=1,
                    name="systemd",
                    command="/sbin/init",
                    cpu_percent=0.4,
                    memory_mb=64,
                    protected=True,
                    killable=False,
                ),
                "2200": LinuxSreGymProcess(
                    pid=2200,
                    name="edge_proxy",
                    command="/usr/local/bin/edge_proxy",
                    cpu_percent=3.2,
                    memory_mb=256,
                ),
            }
            sysctl = {
                "vm.swappiness": "60",
                "vm.zswap_enabled": "Y",
                "net.ipv4.conf.all.rp_filter": "0",
                "net.ipv4.conf.default.rp_filter": "0",
            }
            network = {
                "spoofing_protection_enabled": False,
                "service_health_score": 0.58,
            }

        return LinuxSreGymState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            task_description=TASK_DESCRIPTIONS[task_id],
            filesystem={},
            processes=processes,
            sysctl=sysctl,
            network=network,
            command_history=[],
            reward_history=[],
            reward_reasons=[],
            seen_diagnostics=[],
            is_resolved=False,
            terminal_locked=False,
            last_action_error=None,
            last_reward_breakdown=LinuxSreGymRewardBreakdown(),
            completion_score=0.0,
        )

    def _refresh_derived_views(self, state: LinuxSreGymState) -> None:
        state.filesystem["/proc/sys/vm/swappiness"] = state.sysctl.get("vm.swappiness", "60")
        state.filesystem["/sys/module/zswap/parameters/enabled"] = state.sysctl.get(
            "vm.zswap_enabled", "N"
        )
        state.filesystem["/proc/sys/net/ipv4/conf/all/rp_filter"] = state.sysctl.get(
            "net.ipv4.conf.all.rp_filter", "0"
        )
        state.filesystem["/proc/sys/net/ipv4/conf/default/rp_filter"] = state.sysctl.get(
            "net.ipv4.conf.default.rp_filter", "0"
        )
        state.filesystem["/proc/loadavg"] = self._render_loadavg(state)
        state.filesystem["/proc/meminfo"] = self._render_meminfo(state)
        state.filesystem["/proc/swaps"] = self._render_swaps(state)
        state.is_resolved = self._evaluate_resolved(state)
        state.completion_score = self._grade_state(state)

    def _render_loadavg(self, state: LinuxSreGymState) -> str:
        if state.task_id == "triage":
            if "4242" in state.processes:
                return "18.24 17.90 12.40 3/900 4242"
            return "0.72 0.65 0.50 1/900 3100"
        if state.task_id == "optimization":
            if state.sysctl.get("vm.swappiness") == "95" or state.sysctl.get("vm.zswap_enabled") != "Y":
                return "7.40 6.90 5.80 2/700 1200"
            return "1.10 0.95 0.80 1/700 1200"
        return "0.23 0.19 0.15 1/420 2200"

    def _render_meminfo(self, state: LinuxSreGymState) -> str:
        total_kb = 8 * 1024 * 1024
        if state.task_id == "triage" and "4242" in state.processes:
            free_kb = 196_608
            avail_kb = 262_144
            dirty_kb = 28_672
        elif state.task_id == "optimization" and (
            state.sysctl.get("vm.swappiness") == "95" or state.sysctl.get("vm.zswap_enabled") != "Y"
        ):
            free_kb = 524_288
            avail_kb = 786_432
            dirty_kb = 262_144
        else:
            free_kb = 2_097_152
            avail_kb = 3_145_728
            dirty_kb = 12_288

        return "\n".join(
            [
                f"MemTotal:       {total_kb:>8} kB",
                f"MemFree:        {free_kb:>8} kB",
                f"MemAvailable:   {avail_kb:>8} kB",
                "Buffers:           65536 kB",
                "Cached:           524288 kB",
                f"Dirty:            {dirty_kb:>8} kB",
                "SwapTotal:       2097152 kB",
                "SwapFree:         131072 kB" if state.task_id == "optimization" else "SwapFree:        2097152 kB",
            ]
        )

    def _render_swaps(self, state: LinuxSreGymState) -> str:
        if state.task_id == "optimization":
            used = "1572864" if state.sysctl.get("vm.swappiness") == "95" or state.sysctl.get("vm.zswap_enabled") != "Y" else "262144"
            return "\n".join(
                [
                    "Filename\t\t\tType\t\tSize\tUsed\tPriority",
                    f"/swapfile                               file\t\t2097152\t{used}\t-2",
                ]
            )
        return "\n".join(
            [
                "Filename\t\t\tType\t\tSize\tUsed\tPriority",
                "/swapfile                               file\t\t2097152\t0\t-2",
            ]
        )

    def _evaluate_resolved(self, state: LinuxSreGymState) -> bool:
        if state.task_id == "triage":
            return "4242" not in state.processes and not state.terminal_locked
        if state.task_id == "optimization":
            return (
                int(state.sysctl.get("vm.swappiness", "60")) <= 20
                and state.sysctl.get("vm.zswap_enabled") == "Y"
                and any(cmd in DISCOVERY_COMMANDS["optimization"] for cmd in state.seen_diagnostics)
                and not state.terminal_locked
            )
        return (
            state.sysctl.get("net.ipv4.conf.all.rp_filter") == "1"
            and state.sysctl.get("net.ipv4.conf.default.rp_filter") == "1"
            and not state.terminal_locked
        )

    def _grade_state(self, state: LinuxSreGymState) -> float:
        external_grade = self._grade_state_via_external_grader(state)
        if external_grade is not None:
            self._external_grader_active = True
            return _clamp(external_grade, 0.0, MAX_SCORE)

        if state.task_id == "triage":
            discovery = 0.2 if any(cmd in DISCOVERY_COMMANDS["triage"] for cmd in state.seen_diagnostics) else 0.0
            remediation = 0.4 if "4242" not in state.processes else 0.0
            stability = 0.2 if "4242" not in state.processes else 0.0
            verification = 0.2 if self._has_post_fix_verification(state, VERIFICATION_COMMANDS["triage"]) else 0.0
            penalty = 0.5 if state.terminal_locked else 0.0
        elif state.task_id == "optimization":
            discovery_count = sum(1 for cmd in state.seen_diagnostics if cmd in DISCOVERY_COMMANDS["optimization"])
            discovery = min(0.2, discovery_count * 0.05)
            remediation = 0.3 if int(state.sysctl.get("vm.swappiness", "60")) <= 20 else 0.0
            stability = 0.3 if state.sysctl.get("vm.zswap_enabled") == "Y" else 0.0
            verification = 0.2 if self._has_post_fix_verification(state, VERIFICATION_COMMANDS["optimization"]) else 0.0
            penalty = 0.4 if state.terminal_locked else 0.0
        else:
            discovery_count = sum(1 for cmd in state.seen_diagnostics if cmd in DISCOVERY_COMMANDS["security"])
            discovery = min(0.2, discovery_count * 0.1)
            remediation = 0.3 if state.sysctl.get("net.ipv4.conf.all.rp_filter") == "1" else 0.0
            stability = 0.3 if state.sysctl.get("net.ipv4.conf.default.rp_filter") == "1" else 0.0
            verification = 0.2 if self._has_post_fix_verification(state, VERIFICATION_COMMANDS["security"]) else 0.0
            penalty = 0.4 if state.terminal_locked else 0.0

        repeat_penalty = min(0.1, self._repeat_count(state.command_history) * 0.02)
        return _clamp(discovery + remediation + stability + verification - penalty - repeat_penalty)

    def _has_post_fix_verification(self, state: LinuxSreGymState, commands: Iterable[str]) -> bool:
        if not state.command_history:
            return False
        if state.task_id == "triage" and "4242" in state.processes:
            return False
        history_tail = state.command_history[-3:]
        return any(command in history_tail for command in commands)

    def _repeat_count(self, history: list[str]) -> int:
        counts = Counter(history)
        return sum(max(0, count - 2) for count in counts.values())

    def _dispatch_command(self, command: str) -> CommandResult:
        external_result = self._dispatch_via_external_router(command)
        if external_result is not None:
            self._external_router_active = True
            return external_result
        return self._dispatch_internal(command)

    def _dispatch_internal(self, command: str) -> CommandResult:
        if command == "ps aux":
            return CommandResult(stdout=self._render_ps(), reward_reason="Process list inspected.")
        if command == "top -bn1":
            return CommandResult(stdout=self._render_top(), reward_reason="Live process snapshot inspected.")
        if command == "free -m":
            return CommandResult(stdout=self._render_free_m(), reward_reason="Memory pressure inspected.")
        if command == "vmstat":
            return CommandResult(stdout=self._render_vmstat(), reward_reason="Paging behavior inspected.")
        if command.startswith("cat "):
            return self._handle_cat(command)
        if command.startswith("ls"):
            return self._handle_ls(command)
        if command.startswith("grep "):
            return self._handle_grep(command)
        if command == "sysctl -a | grep rp_filter":
            stdout = "\n".join(
                [
                    f"net.ipv4.conf.all.rp_filter = {self._state.sysctl.get('net.ipv4.conf.all.rp_filter', '0')}",
                    f"net.ipv4.conf.default.rp_filter = {self._state.sysctl.get('net.ipv4.conf.default.rp_filter', '0')}",
                ]
            )
            return CommandResult(stdout=stdout, reward_reason="Network hardening sysctls inspected.")
        if command.startswith("sysctl"):
            return self._handle_sysctl(command)
        if ">" in command and command.startswith("echo "):
            return self._handle_echo_redirect(command)
        if command.startswith("kill "):
            return self._handle_kill(command)
        if command.startswith("pkill "):
            return self._handle_pkill(command)
        return CommandResult(
            stdout="",
            stderr=f"bash: {command}: command not found",
            exit_code=127,
            reward_reason="Unsupported command.",
        )

    def _handle_cat(self, command: str) -> CommandResult:
        parts = shlex.split(command)
        if len(parts) != 2:
            return CommandResult(stderr="cat: missing operand", exit_code=1, reward_reason="Malformed cat command.")
        path = parts[1]
        if path not in self._state.filesystem:
            return CommandResult(
                stderr=f"cat: {path}: No such file or directory",
                exit_code=1,
                reward_reason="Unknown file path.",
            )
        return CommandResult(stdout=self._state.filesystem[path], reward_reason=f"Read {path}.")

    def _handle_ls(self, command: str) -> CommandResult:
        parts = shlex.split(command)
        path = parts[1] if len(parts) > 1 else "/proc"
        directories = {
            "/proc": sorted({"loadavg", "meminfo", "swaps", "sys"}),
            "/proc/sys": sorted({"net", "vm"}),
            "/proc/sys/vm": ["swappiness"],
            "/proc/sys/net/ipv4/conf/all": ["rp_filter"],
            "/proc/sys/net/ipv4/conf/default": ["rp_filter"],
            "/sys/module/zswap/parameters": ["enabled"],
        }
        if path in directories:
            return CommandResult(stdout="\n".join(directories[path]), reward_reason=f"Listed {path}.")
        return CommandResult(
            stderr=f"ls: cannot access '{path}': No such file or directory",
            exit_code=2,
            reward_reason="Unknown directory path.",
        )

    def _handle_grep(self, command: str) -> CommandResult:
        parts = shlex.split(command)
        if len(parts) != 3:
            return CommandResult(stderr="grep: usage: grep PATTERN FILE", exit_code=2, reward_reason="Malformed grep command.")
        _, pattern, path = parts
        if path not in self._state.filesystem:
            return CommandResult(stderr=f"grep: {path}: No such file or directory", exit_code=2, reward_reason="Unknown grep target.")
        lines = [line for line in self._state.filesystem[path].splitlines() if pattern in line]
        if not lines:
            return CommandResult(stdout="", exit_code=1, reward_reason="Pattern not present in file.")
        return CommandResult(stdout="\n".join(lines), reward_reason=f"Searched for {pattern}.")

    def _handle_sysctl(self, command: str) -> CommandResult:
        parts = shlex.split(command)
        if parts == ["sysctl", "-a"]:
            stdout = "\n".join(f"{key} = {value}" for key, value in sorted(self._state.sysctl.items()))
            return CommandResult(stdout=stdout, reward_reason="Full sysctl snapshot inspected.")
        if len(parts) == 2 and parts[0] == "sysctl":
            key = parts[1]
            if key not in self._state.sysctl:
                return CommandResult(stderr=f"sysctl: cannot stat {key}: No such key", exit_code=1, reward_reason="Unknown sysctl key.")
            return CommandResult(stdout=f"{key} = {self._state.sysctl[key]}", reward_reason=f"Read {key}.")
        if len(parts) == 3 and parts[:2] == ["sysctl", "-w"] and "=" in parts[2]:
            key, value = parts[2].split("=", 1)
            return self._write_sysctl(key, value)
        return CommandResult(stderr="sysctl: invalid syntax", exit_code=1, reward_reason="Malformed sysctl command.")

    def _handle_echo_redirect(self, command: str) -> CommandResult:
        lhs, rhs = command.split(">", 1)
        rhs = rhs.strip()
        lhs_parts = shlex.split(lhs.strip())
        if len(lhs_parts) < 2 or not rhs:
            return CommandResult(stderr="bash: invalid redirection", exit_code=1, reward_reason="Malformed echo redirection.")
        value = " ".join(lhs_parts[1:])
        return self._write_virtual_path(rhs, value)

    def _handle_kill(self, command: str) -> CommandResult:
        parts = shlex.split(command)
        if len(parts) != 2:
            return CommandResult(stderr="kill: usage: kill PID", exit_code=1, reward_reason="Malformed kill command.")
        pid = parts[1]
        process = self._state.processes.get(pid)
        if process is None:
            return CommandResult(stderr=f"kill: ({pid}) - No such process", exit_code=1, reward_reason="Unknown PID.")
        if process.protected or not process.killable:
            self._state.terminal_locked = True
            return CommandResult(
                stderr=f"kill: ({pid}) - Operation not permitted",
                exit_code=1,
                safety_penalty=0.5,
                reward_reason="Critical system process targeted. Episode failed.",
            )
        del self._state.processes[pid]
        progress_reward = 0.35 if pid == "4242" and self._state.task_id == "triage" else 0.0
        reason = "Runaway process terminated." if progress_reward else f"Process {pid} terminated."
        return CommandResult(stdout="", progress_reward=progress_reward, reward_reason=reason)

    def _handle_pkill(self, command: str) -> CommandResult:
        parts = shlex.split(command)
        if len(parts) != 2:
            return CommandResult(stderr="pkill: usage: pkill NAME", exit_code=1, reward_reason="Malformed pkill command.")
        name = parts[1]
        matched = [pid for pid, proc in self._state.processes.items() if proc.name == name]
        if not matched:
            return CommandResult(stderr=f"pkill: pattern '{name}' did not match any process", exit_code=1, reward_reason="No process matched pkill pattern.")
        for pid in matched:
            process = self._state.processes[pid]
            if process.protected or not process.killable:
                self._state.terminal_locked = True
                return CommandResult(
                    stderr=f"pkill: killing protected process '{name}' is not allowed",
                    exit_code=1,
                    safety_penalty=0.5,
                    reward_reason="Critical system process targeted. Episode failed.",
                )
        for pid in matched:
            del self._state.processes[pid]
        progress_reward = 0.35 if name == "runaway_hog" and self._state.task_id == "triage" else 0.0
        reason = "Runaway process terminated." if progress_reward else f"Processes named {name} terminated."
        return CommandResult(stdout="", progress_reward=progress_reward, reward_reason=reason)

    def _write_sysctl(self, key: str, value: str) -> CommandResult:
        if key not in self._state.sysctl:
            return CommandResult(stderr=f"sysctl: cannot stat {key}: No such key", exit_code=1, reward_reason="Unknown sysctl key.")
        if key == "vm.swappiness":
            parsed = _safe_int(value, -1)
            if parsed < 0 or parsed > 100:
                return CommandResult(stderr="sysctl: vm.swappiness must be between 0 and 100", exit_code=1, safety_penalty=0.15, reward_reason="Invalid swappiness value.")
        if key == "vm.zswap_enabled":
            value = value.upper()
            if value not in {"Y", "N", "1", "0"}:
                return CommandResult(stderr="sysctl: vm.zswap_enabled expects Y/N", exit_code=1, safety_penalty=0.15, reward_reason="Invalid zswap value.")
            value = "Y" if value in {"Y", "1"} else "N"
        if key.endswith("rp_filter") and value not in {"0", "1"}:
            return CommandResult(stderr="sysctl: rp_filter expects 0 or 1", exit_code=1, safety_penalty=0.15, reward_reason="Invalid rp_filter value.")

        self._state.sysctl[key] = value
        progress = 0.0
        reason = f"Updated {key}."
        if self._state.task_id == "optimization" and key == "vm.swappiness" and _safe_int(value, 999) <= 20:
            progress = 0.25
            reason = "Swappiness tuned toward lower paging pressure."
        elif self._state.task_id == "optimization" and key == "vm.zswap_enabled" and value == "Y":
            progress = 0.25
            reason = "zswap enabled to absorb swap pressure."
        elif self._state.task_id == "security" and key.endswith("all.rp_filter") and value == "1":
            progress = 0.25
            reason = "Enabled rp_filter on all interfaces."
        elif self._state.task_id == "security" and key.endswith("default.rp_filter") and value == "1":
            progress = 0.25
            reason = "Enabled rp_filter on default interfaces."
        return CommandResult(stdout=f"{key} = {value}", progress_reward=progress, reward_reason=reason)

    def _write_virtual_path(self, path: str, value: str) -> CommandResult:
        if path == "/proc/sys/vm/swappiness":
            return self._write_sysctl("vm.swappiness", value)
        if path == "/sys/module/zswap/parameters/enabled":
            return self._write_sysctl("vm.zswap_enabled", value)
        if path == "/proc/sys/net/ipv4/conf/all/rp_filter":
            return self._write_sysctl("net.ipv4.conf.all.rp_filter", value)
        if path == "/proc/sys/net/ipv4/conf/default/rp_filter":
            return self._write_sysctl("net.ipv4.conf.default.rp_filter", value)
        return CommandResult(stderr=f"bash: {path}: Permission denied", exit_code=1, safety_penalty=0.15, reward_reason="Attempted to write an unsupported virtual file.")

    def _render_ps(self) -> str:
        header = "USER       PID %CPU %MEM   RSS STAT COMMAND"
        lines = [header]
        for process in sorted(self._state.processes.values(), key=lambda proc: proc.pid):
            rss_mb = max(process.memory_mb, 1)
            mem_pct = rss_mb / 8192 * 100
            lines.append(
                f"root {process.pid:>8} {process.cpu_percent:>4.1f} {mem_pct:>4.1f} "
                f"{rss_mb:>5} {process.status[:4]:<4} {process.command or process.name}"
            )
        return "\n".join(lines)

    def _render_top(self) -> str:
        load = self._state.filesystem["/proc/loadavg"].split()[:3]
        lines = [
            f"top - 12:00:01 up 3 days,  1 user,  load average: {', '.join(load)}",
            "Tasks:  64 total,   1 running,  63 sleeping,   0 stopped,   0 zombie",
            "%Cpu(s): 96.2 us,  2.1 sy,  0.0 ni,  1.2 id,  0.2 wa,  0.1 hi,  0.2 si,  0.0 st"
            if self._state.task_id == "triage" and "4242" in self._state.processes
            else "%Cpu(s): 12.1 us,  3.4 sy,  0.0 ni, 82.8 id,  1.1 wa,  0.1 hi,  0.5 si,  0.0 st",
            "PID USER      PR  NI    VIRT    RES S  %CPU %MEM     TIME+ COMMAND",
        ]
        for process in sorted(self._state.processes.values(), key=lambda proc: (-proc.cpu_percent, proc.pid)):
            lines.append(
                f"{process.pid:>5} root      20   0  0.0g {process.memory_mb:>6}m S "
                f"{process.cpu_percent:>5.1f} {process.memory_mb / 8192 * 100:>4.1f}   1:23.45 {process.name}"
            )
        return "\n".join(lines)

    def _render_free_m(self) -> str:
        meminfo = self._state.filesystem["/proc/meminfo"].splitlines()
        values = {}
        for line in meminfo:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) // 1024
        used = values["MemTotal"] - values["MemFree"]
        return "\n".join(
            [
                "              total        used        free      shared  buff/cache   available",
                (
                    f"Mem:           {values['MemTotal']:>4}        {used:>4}        {values['MemFree']:>4}"
                    f"          64         512        {values['MemAvailable']:>4}"
                ),
                "Swap:          2048        1536         512" if self._state.task_id == "optimization" and not self._state.is_resolved else "Swap:          2048           0        2048",
            ]
        )

    def _render_vmstat(self) -> str:
        if self._state.task_id == "optimization" and not self._state.is_resolved:
            return "\n".join(
                [
                    "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----",
                    " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st",
                    " 3  2 1572864 524288 65536 524288  128  256  1024   960  342  411 18  7 40 35  0",
                ]
            )
        return "\n".join(
            [
                "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----",
                " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st",
                " 1  0      0 1048576 65536 524288    0    0    12    18  190  240  8  2 88  2  0",
            ]
        )

    def _compute_reward_breakdown(
        self, command: str, result: CommandResult, was_resolved_before: bool
    ) -> LinuxSreGymRewardBreakdown:
        discovery_reward = self._discovery_reward(command)
        progress_reward = result.progress_reward
        safety_penalty = result.safety_penalty
        repeat_penalty = 0.02 if self._state.command_history.count(command) > 2 else 0.0

        completion_bonus = 0.0
        if self._state.is_resolved and not was_resolved_before:
            completion_bonus = 0.45

        total = discovery_reward + progress_reward + completion_bonus - safety_penalty - repeat_penalty
        total = max(-1.0, min(1.0, total))
        score = self._grade_state(self._state)

        return LinuxSreGymRewardBreakdown(
            discovery_reward=discovery_reward,
            progress_reward=progress_reward,
            safety_penalty=safety_penalty,
            repeat_penalty=repeat_penalty,
            completion_bonus=completion_bonus,
            total=total,
            score=score,
        )

    def _discovery_reward(self, command: str) -> float:
        if command in DISCOVERY_COMMANDS[self._state.task_id] and command not in self._state.seen_diagnostics:
            self._state.seen_diagnostics.append(command)
            return 0.05
        return 0.0

    def _build_observation(
        self,
        *,
        stdout: str,
        stderr: str,
        exit_code: int,
        reward: float,
        done: bool,
        reason: str,
        hint: Optional[str],
    ) -> LinuxSreGymObservation:
        return LinuxSreGymObservation(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            current_task=self._state.task_id,
            step_count=self._state.step_count,
            last_reward_reason=reason,
            available_hint=hint,
            reward=reward,
            done=done,
            reward_breakdown=deepcopy(self._state.last_reward_breakdown),
            metadata={
                "episode_id": self._state.episode_id,
                "task_description": self._state.task_description,
                "score": round(self._state.completion_score, 3),
                "is_resolved": self._state.is_resolved,
                "external_router": self._external_router_active,
                "external_grader": self._external_grader_active,
                "max_steps": self._max_steps,
            },
        )

    def _hint_after_step(self, *, done: bool, exit_code: int) -> Optional[str]:
        if done and not self._state.is_resolved:
            return "Reset and try a safer, more diagnostic-first remediation path."
        if exit_code != 0:
            return TASK_HINTS.get(self._state.task_id)
        return None

    def _load_task_builder(self, task_id: str) -> Optional[Callable[..., Any]]:
        module_names = [f"..tasks.{task_id}", f"tasks.{task_id}"]
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name, package=__package__)
            except Exception:
                continue
            kernel_state_cls = self._load_kernel_state_class()
            if kernel_state_cls is not None:
                for attr_name in ("TASK_STATE", "INITIAL_STATE", "DEFAULT_STATE"):
                    candidate = getattr(module, attr_name, None)
                    if isinstance(candidate, kernel_state_cls):
                        return lambda _task_id, value=candidate: value
            for attr_name in ("build_task_state", "create_task_state", "create_initial_state", "build_initial_state"):
                candidate = getattr(module, attr_name, None)
                if callable(candidate):
                    return candidate
        return None

    def _invoke_external_builder(self, builder: Callable[..., Any], task_id: str) -> Any:
        kernel_state_cls = self._load_kernel_state_class()
        candidate_args = ((task_id,), (task_id, kernel_state_cls), (kernel_state_cls,), tuple())
        for args in candidate_args:
            try:
                return builder(*args)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _dispatch_via_external_router(self, command: str) -> Optional[CommandResult]:
        module_names = ["..simulator.command_router", "simulator.command_router"]
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name, package=__package__)
            except Exception:
                continue
            router_instance = self._load_router_instance(module)
            if router_instance is not None:
                for attr_name in ("execute_command", "route_command", "run_command", "execute", "route"):
                    candidate = getattr(router_instance, attr_name, None)
                    if not callable(candidate):
                        continue
                    raw = self._invoke_router(candidate, command)
                    coerced = self._coerce_command_result(raw)
                    if coerced is not None:
                        return coerced
            for attr_name in ("execute_command", "route_command", "run_command"):
                candidate = getattr(module, attr_name, None)
                if not callable(candidate):
                    continue
                raw = self._invoke_router(candidate, command)
                coerced = self._coerce_command_result(raw)
                if coerced is not None:
                    return coerced
        return None

    def _coerce_command_result(self, raw: Any) -> Optional[CommandResult]:
        if isinstance(raw, CommandResult):
            return raw
        if isinstance(raw, tuple) and len(raw) >= 3:
            stdout, stderr, exit_code = raw[:3]
            return CommandResult(stdout=str(stdout), stderr=str(stderr), exit_code=int(exit_code))
        if isinstance(raw, dict):
            return CommandResult(
                stdout=str(raw.get("stdout", "")),
                stderr=str(raw.get("stderr", "")),
                exit_code=int(raw.get("exit_code", 0)),
                progress_reward=float(raw.get("progress_reward", 0.0)),
                safety_penalty=float(raw.get("safety_penalty", 0.0)),
                reward_reason=str(raw.get("reward_reason", "Command processed.")),
            )
        return None

    def _coerce_external_state(
        self, raw: Any, task_id: str
    ) -> Optional[LinuxSreGymState]:
        if raw is None:
            return None
        if isinstance(raw, LinuxSreGymState):
            return raw
        if isinstance(raw, dict):
            return self._state_from_dict(raw, task_id)

        as_dict = None
        for attr_name in ("to_dict", "model_dump", "dict", "serialize"):
            candidate = getattr(raw, attr_name, None)
            if callable(candidate):
                try:
                    as_dict = candidate()
                except Exception:
                    as_dict = None
                if isinstance(as_dict, dict):
                    return self._state_from_dict(as_dict, task_id)

        if hasattr(raw, "__dict__"):
            return self._state_from_dict(vars(raw), task_id)
        return None

    def _state_from_dict(
        self, payload: Dict[str, Any], task_id: str
    ) -> Optional[LinuxSreGymState]:
        try:
            return LinuxSreGymState(**payload)
        except Exception:
            pass

        processes: Dict[str, LinuxSreGymProcess] = {}
        raw_processes = payload.get("processes", {})
        if isinstance(raw_processes, dict):
            process_items = raw_processes.items()
        else:
            process_items = []
            for item in raw_processes if isinstance(raw_processes, list) else []:
                if isinstance(item, dict) and "pid" in item:
                    process_items.append((str(item["pid"]), item))
                elif hasattr(item, "pid"):
                    process_items.append((str(getattr(item, "pid")), vars(item)))

        for pid, process_payload in process_items:
            if isinstance(process_payload, LinuxSreGymProcess):
                processes[str(pid)] = process_payload
                continue
            if hasattr(process_payload, "model_dump"):
                process_payload = process_payload.model_dump()
            elif hasattr(process_payload, "dict"):
                process_payload = process_payload.dict()
            elif not isinstance(process_payload, dict) and hasattr(process_payload, "__dict__"):
                process_payload = vars(process_payload)
            if isinstance(process_payload, dict):
                processes[str(pid)] = LinuxSreGymProcess(
                    pid=int(process_payload.get("pid", pid)),
                    name=str(process_payload.get("name", process_payload.get("comm", f"pid-{pid}"))),
                    command=str(process_payload.get("command", process_payload.get("cmdline", ""))),
                    cpu_percent=float(process_payload.get("cpu_percent", process_payload.get("cpu", 0.0))),
                    memory_mb=int(process_payload.get("memory_mb", process_payload.get("rss_mb", process_payload.get("memory", 0)))),
                    status=str(process_payload.get("status", "running")),
                    protected=bool(process_payload.get("protected", False)),
                    killable=bool(process_payload.get("killable", True)),
                )

        filesystem = payload.get("filesystem", {})
        sysctl = payload.get("sysctl", payload.get("sysctls", {}))
        network = payload.get("network", payload.get("runtime_flags", {}))

        try:
            return LinuxSreGymState(
                episode_id=str(payload.get("episode_id", uuid4())),
                step_count=int(payload.get("step_count", 0)),
                task_id=str(payload.get("task_id", task_id)),
                task_description=str(
                    payload.get("task_description", TASK_DESCRIPTIONS.get(task_id, ""))
                ),
                filesystem=dict(filesystem) if isinstance(filesystem, dict) else {},
                processes=processes,
                sysctl={str(key): str(value) for key, value in dict(sysctl).items()} if isinstance(sysctl, dict) else {},
                network=dict(network) if isinstance(network, dict) else {},
                command_history=[str(value) for value in payload.get("command_history", [])],
                reward_history=[float(value) for value in payload.get("reward_history", [])],
                reward_reasons=[str(value) for value in payload.get("reward_reasons", [])],
                seen_diagnostics=[str(value) for value in payload.get("seen_diagnostics", [])],
                is_resolved=bool(payload.get("is_resolved", False)),
                terminal_locked=bool(payload.get("terminal_locked", False)),
                last_action_error=payload.get("last_action_error"),
                last_reward_breakdown=self._coerce_reward_breakdown(
                    payload.get("last_reward_breakdown")
                ),
                completion_score=float(payload.get("completion_score", 0.0)),
            )
        except Exception:
            return None

    def _coerce_reward_breakdown(self, payload: Any) -> LinuxSreGymRewardBreakdown:
        if isinstance(payload, LinuxSreGymRewardBreakdown):
            return payload
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif hasattr(payload, "dict"):
            payload = payload.dict()
        if isinstance(payload, dict):
            try:
                return LinuxSreGymRewardBreakdown(**payload)
            except Exception:
                pass
        return LinuxSreGymRewardBreakdown()

    def _load_kernel_state_class(self) -> Optional[type]:
        module_names = ["..simulator.kernel_state", "simulator.kernel_state"]
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name, package=__package__)
            except Exception:
                continue
            kernel_state_cls = getattr(module, "KernelState", None)
            if isinstance(kernel_state_cls, type):
                return kernel_state_cls
        return None

    def _load_router_instance(self, module: Any) -> Any:
        router_cls = getattr(module, "CommandRouter", None)
        if not isinstance(router_cls, type):
            return None
        kernel_state_cls = self._load_kernel_state_class()
        current_state = self._export_state_for_external_router()
        candidate_args = (
            (current_state,),
            (current_state, self._state.task_id),
            (kernel_state_cls, current_state) if kernel_state_cls is not None else None,
            tuple(),
        )
        for args in candidate_args:
            if args is None:
                continue
            try:
                return router_cls(*args)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _export_state_for_external_router(self) -> Any:
        kernel_state_cls = self._load_kernel_state_class()
        state_payload = self._model_to_dict(self._state)
        if kernel_state_cls is None:
            return self._state
        for attr_name in ("from_dict", "from_state", "from_openenv_state"):
            candidate = getattr(kernel_state_cls, attr_name, None)
            if callable(candidate):
                try:
                    return candidate(state_payload)
                except Exception:
                    continue
        try:
            return kernel_state_cls(**state_payload)
        except Exception:
            return self._state

    def _invoke_router(self, candidate: Callable[..., Any], command: str) -> Any:
        current_state = self._export_state_for_external_router()
        candidate_args = (
            (command, self._state),
            (self._state, command),
            (command, current_state),
            (current_state, command),
            (command,),
        )
        for args in candidate_args:
            try:
                return candidate(*args)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _model_to_dict(self, value: Any) -> Dict[str, Any]:
        if hasattr(value, "model_dump"):
            try:
                payload = value.model_dump()
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                payload = value.dict()
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        if isinstance(value, dict):
            return value
        if hasattr(value, "__dict__"):
            return dict(vars(value))
        return {}

    def _grade_state_via_external_grader(self, state: LinuxSreGymState) -> Optional[float]:
        module_names = [f"..graders.{state.task_id}_grader", f"graders.{state.task_id}_grader"]
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name, package=__package__)
            except Exception:
                continue
            for attr_name in ("grade", f"grade_{state.task_id}", "score"):
                candidate = getattr(module, attr_name, None)
                if callable(candidate):
                    try:
                        value = candidate(state, state.command_history)
                    except TypeError:
                        try:
                            value = candidate(state)
                        except Exception:
                            continue
                    except Exception:
                        continue
                    if isinstance(value, (int, float)):
                        return float(value)
            for attr_name in ("TriageGrader", "OptimizationGrader", "SecurityGrader"):
                grader_cls = getattr(module, attr_name, None)
                if grader_cls is None:
                    continue
                try:
                    grader = grader_cls()
                except Exception:
                    continue
                grade_fn = getattr(grader, "grade", None)
                if callable(grade_fn):
                    try:
                        value = grade_fn(state, state.command_history)
                    except TypeError:
                        try:
                            value = grade_fn(state)
                        except Exception:
                            continue
                    except Exception:
                        continue
                    if isinstance(value, (int, float)):
                        return float(value)
        return None
