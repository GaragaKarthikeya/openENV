"""Helpers for constructing task-specific kernel state objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, MutableMapping


@dataclass
class ScenarioKernelState:
    """Fallback state object used until the simulator's KernelState lands."""

    task_id: str
    current_task: str
    filesystem: dict[str, str] = field(default_factory=dict)
    processes: dict[int, dict[str, Any]] = field(default_factory=dict)
    sysctl: dict[str, Any] = field(default_factory=dict)
    network: dict[str, Any] = field(default_factory=dict)
    runtime_flags: dict[str, Any] = field(default_factory=dict)
    command_history: list[str] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    is_resolved: bool = False
    terminal_locked: bool = False


def create_kernel_state(task_id: str) -> Any:
    """Create a simulator state object and populate the common attributes."""

    state = _instantiate_simulator_kernel_state(task_id)
    _ensure_attr(state, "task_id", task_id)
    _ensure_attr(state, "current_task", task_id)
    _ensure_attr(state, "filesystem", {})
    _ensure_attr(state, "processes", {})
    _ensure_attr(state, "sysctl", {})
    _ensure_attr(state, "network", {})
    _ensure_attr(state, "runtime_flags", {})
    _ensure_attr(state, "command_history", [])
    _ensure_attr(state, "reward_history", [])
    _ensure_attr(state, "is_resolved", False)
    _ensure_attr(state, "terminal_locked", False)
    return state


def put_process(
    state: Any,
    *,
    pid: int,
    name: str,
    cpu_percent: float,
    memory_mb: int,
    status: str = "S",
    killable: bool = True,
) -> None:
    processes = _as_mapping_attr(state, "processes")
    process_dict = {
        "pid": pid,
        "name": name,
        "cpu_percent": cpu_percent,
        "memory_mb": memory_mb,
        "status": status,
        "killable": killable,
    }
    if isinstance(processes, list):
        try:
            from linux_sre_gym.simulator.kernel_state import ProcessEntry
            processes.append(ProcessEntry(**process_dict))
        except ImportError:
            processes.append(process_dict)
    else:
        processes[pid] = process_dict


def put_file(state: Any, path: str, content: str) -> None:
    _as_mapping_attr(state, "filesystem")[path] = content


def put_sysctl(state: Any, key: str, value: Any) -> None:
    _as_mapping_attr(state, "sysctl")[key] = value


def put_network_flag(state: Any, key: str, value: Any) -> None:
    _as_mapping_attr(state, "runtime_flags")[key] = value


def put_runtime_flag(state: Any, key: str, value: Any) -> None:
    _as_mapping_attr(state, "runtime_flags")[key] = value


def _instantiate_simulator_kernel_state(task_id: str) -> Any:
    try:
        from linux_sre_gym.simulator.kernel_state import KernelState
    except Exception:
        KernelState = None

    if callable(KernelState):
        for args, kwargs in (
            ((), {"task_id": task_id}),
            ((task_id,), {}),
            ((), {}),
        ):
            try:
                return KernelState(*args, **kwargs)
            except TypeError:
                continue

    return ScenarioKernelState(task_id=task_id, current_task=task_id)


def _ensure_attr(state: Any, attr_name: str, default_value: Any) -> None:
    if not hasattr(state, attr_name):
        setattr(state, attr_name, default_value)


def _as_mapping_attr(state: Any, attr_name: str) -> MutableMapping[str, Any]:
    mapping = getattr(state, attr_name, None)
    if mapping is None:
        mapping = {}
        setattr(state, attr_name, mapping)
    return mapping
