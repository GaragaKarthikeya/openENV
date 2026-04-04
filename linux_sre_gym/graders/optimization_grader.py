"""Deterministic grader for the optimization scenario."""

from __future__ import annotations

from typing import Any, Sequence

from linux_sre_gym.graders.common import (
    clamp_score,
    dangerous_command_penalty,
    first_index,
    has_command,
    read_commands,
    read_mapping,
    repetition_penalty,
    sysctl_value,
    truthy_kernel_value,
)


DIAGNOSTIC_MARKERS = (
    "vmstat",
    "cat /proc/swaps",
    "cat /proc/meminfo",
    "sysctl vm.swappiness",
)
MUTATION_MARKERS = (
    "sysctl -w",
    "echo",
)
VERIFY_MARKERS = (
    "sysctl vm.swappiness",
    "cat /proc/sys/vm/swappiness",
    "cat /sys/module/zswap/parameters/enabled",
    "vmstat",
)


class OptimizationGrader:
    """Score memory optimization runs from final state plus action history."""

    task_id = "optimization"

    def grade(self, state: Any, action_history: Sequence[str] | None = None) -> float:
        commands = read_commands(state, action_history)
        runtime_flags = read_mapping(state, "runtime_flags")

        score = 0.0
        if self._diagnosed_before_mutation(commands):
            score += 0.2
        if self._swappiness_is_fixed(state, runtime_flags):
            score += 0.3
        if self._zswap_enabled(state, runtime_flags):
            score += 0.25
        if not bool(runtime_flags.get("thrashing", True)):
            score += 0.15
        if self._verified_after_mutation(commands):
            score += 0.1

        score -= dangerous_command_penalty(commands)
        score -= repetition_penalty(commands)
        if self._set_invalid_swappiness(commands):
            score -= 0.2
        return clamp_score(score)

    def score(self, state: Any, action_history: Sequence[str] | None = None) -> float:
        return self.grade(state, action_history)

    @staticmethod
    def _diagnosed_before_mutation(commands: Sequence[str]) -> bool:
        diagnostic_index = first_index(commands, DIAGNOSTIC_MARKERS)
        mutation_index = first_index(commands, MUTATION_MARKERS)
        return diagnostic_index is not None and (
            mutation_index is None or diagnostic_index < mutation_index
        )

    @staticmethod
    def _swappiness_is_fixed(state: Any, runtime_flags: dict[Any, Any]) -> bool:
        target = int(runtime_flags.get("target_swappiness", 10))
        current = sysctl_value(state, "vm.swappiness", default=999)
        try:
            return int(str(current).strip()) == target
        except ValueError:
            return False

    @staticmethod
    def _zswap_enabled(state: Any, runtime_flags: dict[Any, Any]) -> bool:
        if bool(runtime_flags.get("zswap_enabled", False)):
            return True
        filesystem = read_mapping(state, "filesystem")
        value = filesystem.get("/sys/module/zswap/parameters/enabled", "N")
        return truthy_kernel_value(value)

    @staticmethod
    def _verified_after_mutation(commands: Sequence[str]) -> bool:
        mutation_index = first_index(commands, MUTATION_MARKERS)
        if mutation_index is None:
            return False
        return has_command(commands[mutation_index + 1 :], VERIFY_MARKERS)

    @staticmethod
    def _set_invalid_swappiness(commands: Sequence[str]) -> bool:
        for command in commands:
            lowered = command.lower().replace(" ", "")
            if "vm.swappiness=" not in lowered:
                continue
            try:
                value = int(lowered.split("vm.swappiness=", 1)[1].split(">", 1)[0])
            except ValueError:
                return True
            if value < 0 or value > 100:
                return True
        return False
