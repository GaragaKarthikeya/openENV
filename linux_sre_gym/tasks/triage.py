"""Task 1: incident triage with a runaway CPU and memory hog."""

from __future__ import annotations

from typing import Any

from linux_sre_gym.tasks._kernel_state import (
    create_kernel_state,
    put_file,
    put_network_flag,
    put_process,
    put_runtime_flag,
    put_sysctl,
)


TRIAGE_TASK_ID = "triage"
RUNAWAY_PID = 4242
RUNAWAY_PROCESS_NAME = "image-indexer"


def build_triage_state() -> Any:
    """Return the initial kernel state for the triage scenario."""

    state = create_kernel_state(TRIAGE_TASK_ID)

    put_process(
        state,
        pid=1,
        name="systemd",
        cpu_percent=0.5,
        memory_mb=180,
        status="S",
        killable=False,
    )
    put_process(
        state,
        pid=221,
        name="sshd",
        cpu_percent=0.2,
        memory_mb=42,
        status="S",
        killable=False,
    )
    put_process(
        state,
        pid=677,
        name="journald",
        cpu_percent=0.8,
        memory_mb=96,
        status="S",
        killable=False,
    )
    put_process(
        state,
        pid=RUNAWAY_PID,
        name=RUNAWAY_PROCESS_NAME,
        cpu_percent=384.0,
        memory_mb=6144,
        status="R",
        killable=True,
    )

    put_file(
        state,
        "/proc/meminfo",
        "\n".join(
            [
                "MemTotal:       16384000 kB",
                "MemFree:          196608 kB",
                "MemAvailable:     524288 kB",
                "SwapTotal:       2097152 kB",
                "SwapFree:         262144 kB",
            ]
        )
        + "\n",
    )
    put_file(state, "/proc/loadavg", "8.92 7.88 6.45 7/904 4242\n")
    put_file(state, "/proc/swaps", "Filename\tType\tSize\tUsed\tPriority\n/swapfile\tfile\t2097148\t1835008\t-2\n")
    put_file(state, "/proc/sys/vm/swappiness", "60\n")
    put_file(state, "/sys/module/zswap/parameters/enabled", "Y\n")
    put_file(state, "/proc/sys/net/ipv4/conf/all/rp_filter", "1\n")
    put_file(state, "/proc/sys/net/ipv4/conf/default/rp_filter", "1\n")

    put_sysctl(state, "vm.swappiness", 60)
    put_sysctl(state, "net.ipv4.conf.all.rp_filter", 1)
    put_sysctl(state, "net.ipv4.conf.default.rp_filter", 1)

    put_network_flag(state, "spoofing_protection_enabled", True)
    put_runtime_flag(state, "runaway_process_present", True)
    put_runtime_flag(state, "thrashing", True)
    put_runtime_flag(state, "service_health_score", 0.18)
    put_runtime_flag(state, "load_average_1m", 8.92)
    put_runtime_flag(state, "triage_target_pid", RUNAWAY_PID)
    put_runtime_flag(state, "triage_target_process", RUNAWAY_PROCESS_NAME)

    state.command_history = []
    state.reward_history = []
    state.is_resolved = False
    state.terminal_locked = False
    return state


def create_task() -> Any:
    """Compatibility wrapper for environment code that expects create_task()."""

    return build_triage_state()
