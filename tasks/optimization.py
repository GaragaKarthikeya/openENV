"""Task 2: memory optimization by fixing paging configuration."""

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


OPTIMIZATION_TASK_ID = "optimization"
TARGET_SWAPPINESS = 10


def build_optimization_state() -> Any:
    """Return the initial kernel state for the optimization scenario."""

    state = create_kernel_state(OPTIMIZATION_TASK_ID)

    put_process(
        state,
        pid=1,
        name="systemd",
        cpu_percent=0.4,
        memory_mb=176,
        status="S",
        killable=False,
    )
    put_process(
        state,
        pid=811,
        name="postgres",
        cpu_percent=18.5,
        memory_mb=1536,
        status="S",
        killable=False,
    )
    put_process(
        state,
        pid=902,
        name="api-worker",
        cpu_percent=24.0,
        memory_mb=768,
        status="S",
        killable=True,
    )

    put_file(
        state,
        "/proc/meminfo",
        "\n".join(
            [
                "MemTotal:       16384000 kB",
                "MemFree:          917504 kB",
                "MemAvailable:    1572864 kB",
                "Buffers:          131072 kB",
                "Cached:           262144 kB",
                "SwapTotal:       4194304 kB",
                "SwapFree:         524288 kB",
            ]
        )
        + "\n",
    )
    put_file(
        state,
        "/proc/swaps",
        "Filename\tType\tSize\tUsed\tPriority\n/swapfile\tfile\t4194300\t3670016\t-2\n",
    )
    put_file(state, "/proc/loadavg", "2.40 2.12 1.88 3/640 902\n")
    put_file(state, "/proc/sys/vm/swappiness", "95\n")
    put_file(state, "/sys/module/zswap/parameters/enabled", "N\n")
    put_file(state, "/proc/sys/net/ipv4/conf/all/rp_filter", "1\n")
    put_file(state, "/proc/sys/net/ipv4/conf/default/rp_filter", "1\n")

    put_sysctl(state, "vm.swappiness", 95)
    put_sysctl(state, "net.ipv4.conf.all.rp_filter", 1)
    put_sysctl(state, "net.ipv4.conf.default.rp_filter", 1)

    put_network_flag(state, "spoofing_protection_enabled", True)
    put_runtime_flag(state, "runaway_process_present", False)
    put_runtime_flag(state, "thrashing", True)
    put_runtime_flag(state, "service_health_score", 0.52)
    put_runtime_flag(state, "target_swappiness", TARGET_SWAPPINESS)
    put_runtime_flag(state, "zswap_enabled", False)

    state.command_history = []
    state.reward_history = []
    state.is_resolved = False
    state.terminal_locked = False
    return state


def create_task() -> Any:
    """Compatibility wrapper for environment code that expects create_task()."""

    return build_optimization_state()
