"""
Smoke test: verify KernelState + CommandRouter work end-to-end for all 3 task scenarios.

Uses direct file imports to avoid the broken linux_sre_gym/__init__.py
(Person 1 needs to fix the LinuxSreGymAction import alias in models.py).
"""
import sys
import os
import types
import importlib.util

# ---------------------------------------------------------------------------
# Direct file-based imports (bypass linux_sre_gym/__init__.py)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
_sim_dir = os.path.join(_project_root, "linux_sre_gym", "simulator")


def _import_from_file(fqn: str, file_path: str):
    """Import a module by file path, registering it under *fqn* in sys.modules."""
    spec = importlib.util.spec_from_file_location(
        fqn, file_path,
        submodule_search_locations=[]  # treat as package-aware
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


# 1. Create stub packages so relative imports resolve
_pkg = types.ModuleType("linux_sre_gym")
_pkg.__path__ = [os.path.join(_project_root, "linux_sre_gym")]
_pkg.__package__ = "linux_sre_gym"
sys.modules["linux_sre_gym"] = _pkg

_sim_pkg = types.ModuleType("linux_sre_gym.simulator")
_sim_pkg.__path__ = [_sim_dir]
_sim_pkg.__package__ = "linux_sre_gym.simulator"
sys.modules["linux_sre_gym.simulator"] = _sim_pkg

# 2. Import kernel_state first (command_router depends on it)
_ks_mod = _import_from_file(
    "linux_sre_gym.simulator.kernel_state",
    os.path.join(_sim_dir, "kernel_state.py"),
)

# 3. Import command_router (its `from .kernel_state import ...` now resolves)
_cr_mod = _import_from_file(
    "linux_sre_gym.simulator.command_router",
    os.path.join(_sim_dir, "command_router.py"),
)

KernelState = _ks_mod.KernelState
ProcessEntry = _ks_mod.ProcessEntry
CommandRouter = _cr_mod.CommandRouter
CommandResult = _cr_mod.CommandResult


def sep(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run(router: CommandRouter, state: KernelState, cmd: str) -> CommandResult:
    print(f"\n$ {cmd}")
    result = router.execute(cmd, state)
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"  {line}")
    if result.stderr:
        print(f"  [STDERR] {result.stderr}")
    if result.exit_code != 0:
        print(f"  [EXIT CODE] {result.exit_code}")
    return result


def test_basic_commands():
    """Test every supported command against a default KernelState."""
    sep("BASIC COMMAND TESTS")

    state = KernelState()
    router = CommandRouter()

    # Read-only commands
    run(router, state, "hostname")
    run(router, state, "whoami")
    run(router, state, "uname -a")
    run(router, state, "uptime")
    run(router, state, "cat /proc/meminfo")
    run(router, state, "cat /proc/loadavg")
    run(router, state, "cat /proc/swaps")
    run(router, state, "ps aux")
    run(router, state, "top -bn1")
    run(router, state, "free -m")
    run(router, state, "vmstat")
    run(router, state, "dmesg")
    run(router, state, "ls /proc")
    run(router, state, "ls /sys")
    run(router, state, "sysctl vm.swappiness")
    run(router, state, "sysctl -a")
    run(router, state, "sysctl -a | grep rp_filter")
    run(router, state, "grep Mem /proc/meminfo")

    # Unsupported command
    r = run(router, state, "apt-get install htop")
    assert r.exit_code == 127, f"Expected 127, got {r.exit_code}"

    # Bad file
    r = run(router, state, "cat /etc/nonexistent")
    assert r.exit_code == 1

    print("\n[PASS] All basic command tests passed!")


def test_task1_triage():
    """Simulate Task 1: Triage — runaway process eating CPU/RAM."""
    sep("TASK 1: TRIAGE")

    state = KernelState(
        processes=None,
        total_memory_mb=8192,
        free_memory_mb=512,
        swap_used_mb=1800,
        load_avg=(12.5, 11.2, 8.4),
        sysctl_overrides=None,
        runtime_flags={
            "runaway_process_present": True,
            "service_health_score": 0.2,
        },
    )
    state.processes.append(ProcessEntry(
        pid=9999, name="crypto_miner", user="www-data",
        cpu_percent=95.0, memory_mb=4096.0, status="running",
        killable=True, command="/tmp/crypto_miner --threads=4",
    ))
    state.regenerate_dynamic_files()

    router = CommandRouter()

    run(router, state, "ps aux")
    run(router, state, "top -bn1")
    run(router, state, "free -m")
    run(router, state, "cat /proc/meminfo")
    run(router, state, "cat /proc/loadavg")

    r = run(router, state, "kill -9 9999")
    assert r.exit_code == 0, f"Kill failed: {r.stderr}"
    assert state.runtime_flags["runaway_process_present"] is False
    assert state.get_process_by_pid(9999) is None

    run(router, state, "ps aux")
    run(router, state, "free -m")
    run(router, state, "cat /proc/loadavg")
    run(router, state, "dmesg")

    print(f"\n  Runtime flags: {state.runtime_flags}")
    print(f"  Free memory: {state.free_memory_mb} MB (was 512 MB)")
    assert state.free_memory_mb > 512
    print("\n[PASS] Task 1 Triage scenario passed!")


def test_task2_optimization():
    """Simulate Task 2: Optimization — bad swappiness + zswap disabled."""
    sep("TASK 2: OPTIMIZATION")

    state = KernelState(
        total_memory_mb=8192,
        swap_used_mb=1400,
        sysctl_overrides={"vm.swappiness": "100"},
        runtime_flags={
            "thrashing": True,
        },
    )
    state._zswap_enabled = "N"
    state.regenerate_dynamic_files()

    router = CommandRouter()

    run(router, state, "sysctl vm.swappiness")
    run(router, state, "cat /sys/module/zswap/parameters/enabled")
    run(router, state, "vmstat")
    run(router, state, "free -m")
    run(router, state, "cat /proc/swaps")

    r = run(router, state, "sysctl -w vm.swappiness=10")
    assert r.exit_code == 0
    assert state.sysctl["vm.swappiness"] == "10"

    r = run(router, state, "echo Y > /sys/module/zswap/parameters/enabled")
    assert r.exit_code == 0
    assert state.zswap_enabled is True

    run(router, state, "sysctl vm.swappiness")
    run(router, state, "cat /sys/module/zswap/parameters/enabled")
    run(router, state, "vmstat")

    assert state.runtime_flags["thrashing"] is False
    print(f"\n  Runtime flags: {state.runtime_flags}")
    print("\n[PASS] Task 2 Optimization scenario passed!")


def test_task3_security():
    """Simulate Task 3: Security — rp_filter disabled."""
    sep("TASK 3: SECURITY HARDENING")

    state = KernelState(
        sysctl_overrides={
            "net.ipv4.conf.all.rp_filter": "0",
            "net.ipv4.conf.default.rp_filter": "0",
        },
        runtime_flags={
            "spoofing_protection_enabled": False,
        },
    )

    router = CommandRouter()

    run(router, state, "sysctl -a | grep rp_filter")
    run(router, state, "cat /proc/sys/net/ipv4/conf/all/rp_filter")
    run(router, state, "cat /proc/sys/net/ipv4/conf/default/rp_filter")

    r = run(router, state, "sysctl -w net.ipv4.conf.all.rp_filter=1")
    assert r.exit_code == 0

    r = run(router, state, "sysctl -w net.ipv4.conf.default.rp_filter=1")
    assert r.exit_code == 0

    run(router, state, "sysctl -a | grep rp_filter")
    assert state.sysctl["net.ipv4.conf.all.rp_filter"] == "1"
    assert state.sysctl["net.ipv4.conf.default.rp_filter"] == "1"
    assert state.runtime_flags["spoofing_protection_enabled"] is True

    print(f"\n  Runtime flags: {state.runtime_flags}")
    print("\n[PASS] Task 3 Security scenario passed!")


def test_serialization():
    """Test to_dict / from_dict / clone roundtrips."""
    sep("SERIALIZATION TESTS")

    state = KernelState(
        sysctl_overrides={"vm.swappiness": "80"},
        runtime_flags={"thrashing": True},
    )

    d = state.to_dict()
    restored = KernelState.from_dict(d)
    assert restored.sysctl["vm.swappiness"] == "80"
    assert restored.runtime_flags["thrashing"] is True
    assert len(restored.processes) == len(state.processes)

    cloned = state.clone()
    cloned.sysctl["vm.swappiness"] = "10"
    assert state.sysctl["vm.swappiness"] == "80", "Clone should not affect original"

    print("\n[PASS] Serialization tests passed!")


def test_protected_process():
    """Ensure system-critical processes cannot be killed."""
    sep("PROTECTED PROCESS TEST")

    state = KernelState()
    router = CommandRouter()

    r = run(router, state, "kill 1")
    assert r.exit_code == 1
    assert state.get_process_by_pid(1) is not None, "systemd should NOT be killed"

    print("\n[PASS] Protected process test passed!")


if __name__ == "__main__":
    test_basic_commands()
    test_task1_triage()
    test_task2_optimization()
    test_task3_security()
    test_serialization()
    test_protected_process()

    print(f"\n{'='*70}")
    print("  ALL TESTS PASSED")
    print(f"{'='*70}")
