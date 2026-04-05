"""Deterministic grader for the security hardening scenario."""

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
)


DIAGNOSTIC_MARKERS = (
    "rp_filter",
    "sysctl -a",
    "sysctl net.ipv4.conf",
    "cat /proc/sys/net/ipv4/conf",
)
MUTATION_MARKERS = (
    "sysctl -w",
    "echo",
)
VERIFY_MARKERS = (
    "sysctl -a",
    "sysctl net.ipv4.conf",
    "grep rp_filter",
    "cat /proc/sys/net/ipv4/conf/all/rp_filter",
    "cat /proc/sys/net/ipv4/conf/default/rp_filter",
)
UNSAFE_NETWORK_MARKERS = (
    "accept_redirects=1",
    "send_redirects=1",
    "rp_filter=0",
    "net.ipv4.ip_forward=1",
)


class SecurityGrader:
    """Score security hardening runs from final state plus action history."""

    task_id = "security"

    def grade(self, state: Any, action_history: Sequence[str] | None = None) -> float:
        commands = read_commands(state, action_history)
        runtime_flags = read_mapping(state, "runtime_flags")

        score = 0.0
        if self._diagnosed_before_mutation(commands):
            score += 0.15
        if self._rp_filter_enabled(state, "net.ipv4.conf.all.rp_filter", runtime_flags.get("rp_filter_all")):
            score += 0.25
        if self._rp_filter_enabled(
            state,
            "net.ipv4.conf.default.rp_filter",
            runtime_flags.get("rp_filter_default"),
        ):
            score += 0.25
        if bool(runtime_flags.get("spoofing_protection_enabled", False)) or self._all_filters_enabled(state):
            score += 0.25
        if self._verified_after_mutation(commands):
            score += 0.1

        score -= dangerous_command_penalty(commands)
        score -= repetition_penalty(commands)
        if has_command(commands, UNSAFE_NETWORK_MARKERS):
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
    def _rp_filter_enabled(state: Any, key: str, fallback_value: Any) -> bool:
        value = sysctl_value(state, key, default=fallback_value)
        try:
            return int(str(value).strip()) == 1
        except (TypeError, ValueError):
            return False

    def _all_filters_enabled(self, state: Any) -> bool:
        return (
            self._rp_filter_enabled(state, "net.ipv4.conf.all.rp_filter", None)
            and self._rp_filter_enabled(state, "net.ipv4.conf.default.rp_filter", None)
        )

    @staticmethod
    def _verified_after_mutation(commands: Sequence[str]) -> bool:
        mutation_index = first_index(commands, MUTATION_MARKERS)
        if mutation_index is None:
            return False
        return has_command(commands[mutation_index + 1 :], VERIFY_MARKERS)
