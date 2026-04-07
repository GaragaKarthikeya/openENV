"""Shared helpers for deterministic task scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DANGEROUS_COMMAND_MARKERS = (
    "rm -rf /",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    ":(){:|:&};:",
    "iptables -F",
    "nft flush ruleset",
)
OPEN_SCORE_EPSILON = 0.001


def clamp_score(score: float) -> float:
    bounded = max(OPEN_SCORE_EPSILON, min(1.0 - OPEN_SCORE_EPSILON, float(score)))
    return round(bounded, 4)


def read_state_value(state: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(field_name, default)
    return getattr(state, field_name, default)


def read_mapping(state: Any, field_name: str) -> dict[Any, Any]:
    value = read_state_value(state, field_name, {})
    if not value and field_name == "runtime_flags":
        value = read_state_value(state, "network", {})
    return dict(value) if isinstance(value, Mapping) else {}


def read_commands(state: Any, action_history: Sequence[str] | None = None) -> list[str]:
    if action_history is not None:
        return [str(command).strip() for command in action_history if str(command).strip()]

    history = read_state_value(state, "command_history", [])
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        return [str(command).strip() for command in history if str(command).strip()]
    return []


def has_command(commands: Sequence[str], markers: Sequence[str]) -> bool:
    return any(
        any(marker in command.lower() for marker in markers)
        for command in commands
    )


def first_index(commands: Sequence[str], markers: Sequence[str]) -> int | None:
    for index, command in enumerate(commands):
        lowered = command.lower()
        if any(marker in lowered for marker in markers):
            return index
    return None


def count_repeated_commands(commands: Sequence[str]) -> int:
    seen = set()
    repeated = 0
    for command in commands:
        normalized = " ".join(command.lower().split())
        if normalized in seen:
            repeated += 1
        else:
            seen.add(normalized)
    return repeated


def dangerous_command_penalty(commands: Sequence[str]) -> float:
    return 0.3 if has_command(commands, DANGEROUS_COMMAND_MARKERS) else 0.0


def repetition_penalty(commands: Sequence[str]) -> float:
    return 0.05 * count_repeated_commands(commands)


def sysctl_value(state: Any, key: str, default: Any = None) -> Any:
    sysctl = read_mapping(state, "sysctl")
    if key in sysctl:
        return sysctl[key]

    filesystem = read_mapping(state, "filesystem")
    path = f"/proc/sys/{key.replace('.', '/')}"
    if path in filesystem:
        return str(filesystem[path]).strip()
    return default


def truthy_kernel_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "y", "yes", "true", "on"}


def process_table(state: Any) -> dict[int, dict[str, Any]]:
    processes = read_state_value(state, "processes", {})
    normalized: dict[int, dict[str, Any]] = {}
    
    if isinstance(processes, Mapping):
        for pid, details in processes.items():
            try:
                normalized[int(pid)] = dict(details)
            except (TypeError, ValueError):
                continue
    elif isinstance(processes, Sequence):
        for p in processes:
            try:
                if isinstance(p, Mapping):
                    pid = p.get("pid")
                    details = dict(p)
                else:
                    pid = getattr(p, "pid", None)
                    details = {
                        "pid": pid,
                        "name": getattr(p, "name", ""),
                        "user": getattr(p, "user", ""),
                        "cpu_percent": getattr(p, "cpu_percent", 0.0),
                        "memory_mb": getattr(p, "memory_mb", 0),
                        "status": getattr(p, "status", ""),
                        "killable": getattr(p, "killable", True),
                        "command": getattr(p, "command", ""),
                        "nice": getattr(p, "nice", 0),
                    }
                if pid is not None:
                    normalized[int(pid)] = details
            except (TypeError, ValueError):
                continue
    return normalized
