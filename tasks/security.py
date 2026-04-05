"""Task 3: network hardening by enabling reverse path filtering."""

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


SECURITY_TASK_ID = "security"
REQUIRED_RP_FILTER_VALUE = 1


def build_security_state() -> Any:
    """Return the initial kernel state for the security hardening scenario."""

    state = create_kernel_state(SECURITY_TASK_ID)

    put_process(
        state,
        pid=1,
        name="systemd",
        cpu_percent=0.4,
        memory_mb=180,
        status="S",
        killable=False,
    )
    put_process(
        state,
        pid=343,
        name="nginx",
        cpu_percent=4.8,
        memory_mb=320,
        status="S",
        killable=False,
    )
    put_process(
        state,
        pid=777,
        name="wireguard-agent",
        cpu_percent=1.1,
        memory_mb=96,
        status="S",
        killable=False,
    )

    put_file(
        state,
        "/proc/meminfo",
        "\n".join(
            [
                "MemTotal:       16384000 kB",
                "MemFree:        10485760 kB",
                "MemAvailable:   12288000 kB",
                "SwapTotal:       2097152 kB",
                "SwapFree:        2097152 kB",
            ]
        )
        + "\n",
    )
    put_file(state, "/proc/loadavg", "0.36 0.29 0.22 1/316 777\n")
    put_file(state, "/proc/swaps", "Filename\tType\tSize\tUsed\tPriority\n/swapfile\tfile\t2097148\t0\t-2\n")
    put_file(state, "/proc/sys/vm/swappiness", "60\n")
    put_file(state, "/sys/module/zswap/parameters/enabled", "Y\n")
    put_file(state, "/proc/sys/net/ipv4/conf/all/rp_filter", "0\n")
    put_file(state, "/proc/sys/net/ipv4/conf/default/rp_filter", "0\n")

    put_sysctl(state, "vm.swappiness", 60)
    put_sysctl(state, "net.ipv4.conf.all.rp_filter", 0)
    put_sysctl(state, "net.ipv4.conf.default.rp_filter", 0)
    put_sysctl(state, "net.ipv4.conf.all.accept_redirects", 1)
    put_sysctl(state, "net.ipv4.conf.default.accept_redirects", 1)

    put_network_flag(state, "spoofing_protection_enabled", False)
    put_network_flag(state, "rp_filter_all", 0)
    put_network_flag(state, "rp_filter_default", 0)
    put_runtime_flag(state, "runaway_process_present", False)
    put_runtime_flag(state, "thrashing", False)
    put_runtime_flag(state, "service_health_score", 0.78)

    state.command_history = []
    state.reward_history = []
    state.is_resolved = False
    state.terminal_locked = False
    return state


def create_task() -> Any:
    """Compatibility wrapper for environment code that expects create_task()."""

    return build_security_state()
