from linux_sre_gym.graders import OptimizationGrader, SecurityGrader, TriageGrader
from linux_sre_gym.tasks import (
    RUNAWAY_PID,
    TARGET_SWAPPINESS,
    build_optimization_state,
    build_security_state,
    build_triage_state,
)


def test_triage_state_contains_runaway_process() -> None:
    state = build_triage_state()

    assert state.current_task == "triage"
    runaway_proc = next(p for p in state.processes if (p.get("pid") if isinstance(p, dict) else getattr(p, "pid", None)) == RUNAWAY_PID)
    assert (runaway_proc.get("cpu_percent", 0) if isinstance(runaway_proc, dict) else getattr(runaway_proc, "cpu_percent", 0)) >= 300
    assert (runaway_proc.get("memory_mb", 0) if isinstance(runaway_proc, dict) else getattr(runaway_proc, "memory_mb", 0)) >= 4096
    assert state.runtime_flags["runaway_process_present"] is True
    assert "/proc/meminfo" in state.filesystem


def test_triage_grader_rewards_resolution_and_penalizes_repetition() -> None:
    state = build_triage_state()
    state.command_history = ["ps aux", "cat /proc/meminfo", f"kill {RUNAWAY_PID}", "ps aux"]
    state.processes = [p for p in state.processes if (p.get("pid") if isinstance(p, dict) else getattr(p, "pid", None)) != RUNAWAY_PID]
    state.runtime_flags["runaway_process_present"] = False
    state.runtime_flags["thrashing"] = False
    state.runtime_flags["load_average_1m"] = 0.42
    state.is_resolved = True

    grader = TriageGrader()
    score = grader.grade(state)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score >= 0.85
    assert grader.grade(state, action_history=["ps aux"] * 5) < score


def test_optimization_state_starts_with_bad_paging_settings() -> None:
    state = build_optimization_state()

    assert state.current_task == "optimization"
    assert state.sysctl["vm.swappiness"] == 95
    assert state.filesystem["/sys/module/zswap/parameters/enabled"].strip() == "N"
    assert state.runtime_flags["thrashing"] is True
    assert state.runtime_flags["target_swappiness"] == TARGET_SWAPPINESS


def test_optimization_grader_scores_completed_fix() -> None:
    state = build_optimization_state()
    state.command_history = [
        "vmstat",
        "sysctl vm.swappiness",
        "sysctl -w vm.swappiness=10",
        "echo 1 > /sys/module/zswap/parameters/enabled",
        "cat /sys/module/zswap/parameters/enabled",
    ]
    state.sysctl["vm.swappiness"] = TARGET_SWAPPINESS
    state.filesystem["/proc/sys/vm/swappiness"] = "10\n"
    state.filesystem["/sys/module/zswap/parameters/enabled"] = "Y\n"
    state.runtime_flags["zswap_enabled"] = True
    state.runtime_flags["thrashing"] = False
    state.is_resolved = True

    score = OptimizationGrader().grade(state)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score >= 0.95


def test_security_state_starts_with_rp_filter_disabled() -> None:
    state = build_security_state()

    assert state.current_task == "security"
    assert state.sysctl["net.ipv4.conf.all.rp_filter"] == 0
    assert state.sysctl["net.ipv4.conf.default.rp_filter"] == 0
    assert state.runtime_flags["spoofing_protection_enabled"] is False


def test_security_grader_rewards_hardening_and_penalizes_unsafe_network_changes() -> None:
    state = build_security_state()
    state.command_history = [
        "sysctl net.ipv4.conf.all.rp_filter",
        "sysctl -w net.ipv4.conf.all.rp_filter=1",
        "sysctl -w net.ipv4.conf.default.rp_filter=1",
        "sysctl -a | grep rp_filter",
    ]
    state.sysctl["net.ipv4.conf.all.rp_filter"] = 1
    state.sysctl["net.ipv4.conf.default.rp_filter"] = 1
    state.runtime_flags["rp_filter_all"] = 1
    state.runtime_flags["rp_filter_default"] = 1
    state.runtime_flags["spoofing_protection_enabled"] = True
    state.is_resolved = True

    grader = SecurityGrader()
    score = grader.grade(state)
    unsafe_score = grader.grade(
        state,
        action_history=["sysctl -w net.ipv4.conf.all.accept_redirects=1"],
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score >= 0.95
    assert unsafe_score < score
