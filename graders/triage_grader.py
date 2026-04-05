"""Deterministic grader for the triage scenario."""

from __future__ import annotations

from typing import Any, Sequence

from linux_sre_gym.graders.common import (
    clamp_score,
    dangerous_command_penalty,
    has_command,
    process_table,
    read_commands,
    read_mapping,
    repetition_penalty,
)


DIAGNOSTIC_MARKERS = (
    "ps",
    "top",
    "free -m",
    "cat /proc/meminfo",
    "cat /proc/loadavg",
)
PROTECTED_PROCESS_NAMES = {"systemd", "sshd", "journald", "postgres", "nginx"}


class TriageGrader:
    """Score triage runs from final state plus action history."""

    task_id = "triage"

    def grade(self, state: Any, action_history: Sequence[str] | None = None) -> float:
        commands = read_commands(state, action_history)
        processes = process_table(state)
        runtime_flags = read_mapping(state, "runtime_flags")

        score = 0.0
        if has_command(commands, DIAGNOSTIC_MARKERS):
            score += 0.2
        if not self._runaway_process_active(processes, runtime_flags):
            score += 0.35
        if self._memory_pressure_recovered(processes, runtime_flags):
            score += 0.2
        if self._protected_processes_alive(processes):
            score += 0.15
        if bool(getattr(state, "is_resolved", False)) or not self._runaway_process_active(processes, runtime_flags):
            score += 0.1

        score -= dangerous_command_penalty(commands)
        score -= repetition_penalty(commands)
        if self._protected_process_targeted(commands, processes):
            score -= 0.25
        return clamp_score(score)

    def score(self, state: Any, action_history: Sequence[str] | None = None) -> float:
        return self.grade(state, action_history)

    @staticmethod
    def _runaway_process_active(
        processes: dict[int, dict[str, Any]],
        runtime_flags: dict[Any, Any],
    ) -> bool:
        target_pid = runtime_flags.get("triage_target_pid")
        if target_pid is not None:
            details = processes.get(int(target_pid))
            if details is not None:
                return str(details.get("status", "")).upper() != "Z"

        for details in processes.values():
            if (
                bool(details.get("killable", True))
                and float(details.get("cpu_percent", 0.0)) >= 85.0
                and int(details.get("memory_mb", 0)) >= 2048
                and str(details.get("status", "")).upper() != "Z"
            ):
                return True
        return bool(runtime_flags.get("runaway_process_present", False))

    @staticmethod
    def _memory_pressure_recovered(
        processes: dict[int, dict[str, Any]],
        runtime_flags: dict[Any, Any],
    ) -> bool:
        if bool(runtime_flags.get("thrashing", False)):
            return False
        heaviest_process_mb = max(
            (int(details.get("memory_mb", 0)) for details in processes.values()),
            default=0,
        )
        return heaviest_process_mb < 2048 and float(runtime_flags.get("load_average_1m", 0.0)) < 2.5

    @staticmethod
    def _protected_processes_alive(processes: dict[int, dict[str, Any]]) -> bool:
        names = {str(details.get("name", "")) for details in processes.values()}
        return {"systemd", "sshd"}.issubset(names)

    @staticmethod
    def _protected_process_targeted(
        commands: Sequence[str],
        processes: dict[int, dict[str, Any]],
    ) -> bool:
        protected_pids = {
            pid
            for pid, details in processes.items()
            if not bool(details.get("killable", True))
            or str(details.get("name", "")) in PROTECTED_PROCESS_NAMES
        }
        for command in commands:
            lowered = command.lower().strip()
            if lowered.startswith("pkill ") and any(name in lowered for name in PROTECTED_PROCESS_NAMES):
                return True
            if lowered.startswith("kill "):
                for pid in protected_pids:
                    if f" {pid}" in f" {lowered}":
                        return True
        return False
