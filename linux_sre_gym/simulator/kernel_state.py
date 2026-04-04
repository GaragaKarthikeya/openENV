"""
KernelState — the simulated Linux kernel state.

Holds a fake process table, virtual filesystem (/proc, /sys), sysctl
registry, runtime flags, and system metrics.  Designed so that:

* Person 3 can inject a task-specific state via constructor kwargs.
* CommandRouter can read/mutate state in-place during an episode.
* The environment can snapshot/restore state for deterministic replay.
"""

from __future__ import annotations

import copy
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Process entry helper
# ---------------------------------------------------------------------------

@dataclass
class ProcessEntry:
    """A single row in the simulated process table."""

    pid: int
    name: str
    user: str = "root"
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = "sleeping"       # running | sleeping | zombie
    killable: bool = True
    command: str = ""              # full cmdline shown in ps
    nice: int = 0

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "name": self.name,
            "user": self.user,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "status": self.status,
            "killable": self.killable,
            "command": self.command or self.name,
            "nice": self.nice,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessEntry":
        return cls(**d)


# ---------------------------------------------------------------------------
# Mapping between sysctl dotted-key ↔ filesystem path
# ---------------------------------------------------------------------------

_SYSCTL_PATH_MAP: Dict[str, str] = {
    "vm.swappiness": "/proc/sys/vm/swappiness",
    "net.ipv4.conf.all.rp_filter": "/proc/sys/net/ipv4/conf/all/rp_filter",
    "net.ipv4.conf.default.rp_filter": "/proc/sys/net/ipv4/conf/default/rp_filter",
}

_PATH_SYSCTL_MAP: Dict[str, str] = {v: k for k, v in _SYSCTL_PATH_MAP.items()}


# ---------------------------------------------------------------------------
# Default background processes (realistic filler for ps/top output)
# ---------------------------------------------------------------------------

def _default_background_processes() -> List[ProcessEntry]:
    """Return ~10 realistic background processes."""
    return [
        ProcessEntry(pid=1,    name="systemd",        user="root",   cpu_percent=0.1,  memory_mb=12.4,  status="sleeping", killable=False, command="/sbin/init"),
        ProcessEntry(pid=2,    name="kthreadd",       user="root",   cpu_percent=0.0,  memory_mb=0.0,   status="sleeping", killable=False, command="[kthreadd]"),
        ProcessEntry(pid=45,   name="ksoftirqd/0",    user="root",   cpu_percent=0.0,  memory_mb=0.0,   status="sleeping", killable=False, command="[ksoftirqd/0]"),
        ProcessEntry(pid=112,  name="kworker/0:1H",   user="root",   cpu_percent=0.0,  memory_mb=0.0,   status="sleeping", killable=False, command="[kworker/0:1H]"),
        ProcessEntry(pid=256,  name="systemd-journal", user="root",  cpu_percent=0.2,  memory_mb=42.0,  status="sleeping", killable=False, command="/lib/systemd/systemd-journald"),
        ProcessEntry(pid=312,  name="sshd",           user="root",   cpu_percent=0.0,  memory_mb=8.2,   status="sleeping", killable=False, command="/usr/sbin/sshd -D"),
        ProcessEntry(pid=340,  name="cron",           user="root",   cpu_percent=0.0,  memory_mb=3.8,   status="sleeping", killable=False, command="/usr/sbin/cron -f"),
        ProcessEntry(pid=355,  name="rsyslogd",       user="syslog", cpu_percent=0.0,  memory_mb=5.6,   status="sleeping", killable=False, command="/usr/sbin/rsyslogd -n"),
        ProcessEntry(pid=410,  name="dbus-daemon",    user="messagebus", cpu_percent=0.0, memory_mb=4.2, status="sleeping", killable=False, command="/usr/bin/dbus-daemon --system"),
        ProcessEntry(pid=450,  name="networkd",       user="systemd-network", cpu_percent=0.1, memory_mb=6.0, status="sleeping", killable=False, command="/lib/systemd/systemd-networkd"),
    ]


# ---------------------------------------------------------------------------
# KernelState
# ---------------------------------------------------------------------------

class KernelState:
    """
    The complete simulated Linux machine state.

    Parameters
    ----------
    processes : list[ProcessEntry | dict], optional
        Initial process table.  If dicts are supplied they are converted to
        ``ProcessEntry`` instances.  If *None*, a set of realistic background
        processes is used.
    total_memory_mb : int
        Total physical RAM in MiB (default 8192 = 8 GiB).
    free_memory_mb : int | None
        Free RAM.  Computed from processes if *None*.
    cached_mb : int
        Page cache size.
    buffers_mb : int
        Buffer cache size.
    swap_total_mb : int
        Total swap.
    swap_used_mb : int
        Used swap.
    load_avg : tuple[float, float, float]
        1 / 5 / 15-minute load averages.
    cpu_count : int
        Number of CPUs.
    sysctl_overrides : dict[str, str] | None
        Extra sysctl values merged *on top* of defaults.
    runtime_flags : dict[str, Any] | None
        Boolean / numeric runtime flags (thrashing, spoofing_protection_enabled, …).
    extra_files : dict[str, str] | None
        Additional virtual filesystem entries.
    dmesg_lines : list[str] | None
        Initial kernel ring-buffer entries.
    """

    # Reasonable sysctl defaults for a "healthy" system
    _DEFAULT_SYSCTL: Dict[str, str] = {
        "vm.swappiness": "30",
        "vm.dirty_ratio": "20",
        "vm.dirty_background_ratio": "10",
        "vm.overcommit_memory": "0",
        "vm.min_free_kbytes": "67584",
        "net.ipv4.conf.all.rp_filter": "1",
        "net.ipv4.conf.default.rp_filter": "1",
        "net.ipv4.ip_forward": "1",
        "net.ipv4.tcp_syncookies": "1",
        "kernel.hostname": "sre-lab",
        "kernel.osrelease": "5.15.0-94-generic",
    }

    def __init__(
        self,
        *,
        processes: Optional[List] = None,
        total_memory_mb: int = 8192,
        free_memory_mb: Optional[int] = None,
        cached_mb: int = 1024,
        buffers_mb: int = 256,
        swap_total_mb: int = 2048,
        swap_used_mb: int = 128,
        load_avg: Tuple[float, float, float] = (0.45, 0.38, 0.32),
        cpu_count: int = 4,
        sysctl_overrides: Optional[Dict[str, str]] = None,
        runtime_flags: Optional[Dict[str, Any]] = None,
        extra_files: Optional[Dict[str, str]] = None,
        dmesg_lines: Optional[List[str]] = None,
    ):
        # -- Process table --------------------------------------------------
        if processes is None:
            self.processes: List[ProcessEntry] = _default_background_processes()
        else:
            self.processes = [
                p if isinstance(p, ProcessEntry) else ProcessEntry.from_dict(p)
                for p in processes
            ]

        # -- System metrics -------------------------------------------------
        self.total_memory_mb = total_memory_mb
        self.cached_mb = cached_mb
        self.buffers_mb = buffers_mb
        self.swap_total_mb = swap_total_mb
        self.swap_used_mb = swap_used_mb
        self.load_avg = load_avg
        self.cpu_count = cpu_count

        # Free memory: if not given, derive from total minus process usage
        if free_memory_mb is not None:
            self.free_memory_mb = free_memory_mb
        else:
            used = sum(p.memory_mb for p in self.processes)
            self.free_memory_mb = max(64, int(self.total_memory_mb - used - self.cached_mb - self.buffers_mb))

        # -- Sysctl registry (dotted keys → string values) ------------------
        self.sysctl: Dict[str, str] = dict(self._DEFAULT_SYSCTL)
        if sysctl_overrides:
            self.sysctl.update(sysctl_overrides)

        # -- Runtime flags --------------------------------------------------
        self.runtime_flags: Dict[str, Any] = {
            "thrashing": False,
            "spoofing_protection_enabled": True,
            "runaway_process_present": False,
            "service_health_score": 1.0,
        }
        if runtime_flags:
            self.runtime_flags.update(runtime_flags)

        # -- Virtual filesystem ---------------------------------------------
        self.filesystem: Dict[str, str] = {}
        if extra_files:
            self.filesystem.update(extra_files)

        # -- dmesg ring buffer ----------------------------------------------
        self.dmesg: List[str] = dmesg_lines if dmesg_lines is not None else [
            "[    0.000000] Linux version 5.15.0-94-generic (buildd@lcy02-amd64-044)",
            "[    0.000000] Command line: BOOT_IMAGE=/vmlinuz-5.15.0-94-generic root=/dev/sda1",
            "[    0.523412] ACPI: Core revision 20210730",
            "[    1.234567] systemd[1]: Detected architecture x86-64.",
            "[    1.234890] systemd[1]: Set hostname to <sre-lab>.",
            "[    2.345000] EXT4-fs (sda1): mounted filesystem with ordered data mode.",
            "[    3.124000] systemd[1]: Started Journal Service.",
        ]

        # -- Zswap state (separate from sysctl for filesystem sync) ---------
        self._zswap_enabled: str = "N"  # "Y" or "N"

        # Build all dynamic virtual files from current metrics
        self.regenerate_dynamic_files()

    # =====================================================================
    # Dynamic file generation
    # =====================================================================

    def regenerate_dynamic_files(self) -> None:
        """
        Rebuild every virtual file whose content derives from current metrics,
        process table, or sysctl values.  Call this after any state mutation.
        """
        self._gen_proc_meminfo()
        self._gen_proc_loadavg()
        self._gen_proc_swaps()
        self._gen_proc_stat()
        self._sync_sysctl_to_filesystem()
        self._gen_zswap_enabled()

    # -- /proc/meminfo ------------------------------------------------------

    def _gen_proc_meminfo(self) -> None:
        total_kb = self.total_memory_mb * 1024
        free_kb = self.free_memory_mb * 1024
        cached_kb = self.cached_mb * 1024
        buffers_kb = self.buffers_mb * 1024
        available_kb = free_kb + cached_kb + buffers_kb
        swap_total_kb = self.swap_total_mb * 1024
        swap_free_kb = (self.swap_total_mb - self.swap_used_mb) * 1024
        used_kb = total_kb - free_kb - cached_kb - buffers_kb

        # Slab / SReclaimable are small fixed values for realism
        slab_kb = 98304
        sreclaimable_kb = 65536

        self.filesystem["/proc/meminfo"] = textwrap.dedent(f"""\
            MemTotal:       {total_kb:>8} kB
            MemFree:        {free_kb:>8} kB
            MemAvailable:   {available_kb:>8} kB
            Buffers:        {buffers_kb:>8} kB
            Cached:         {cached_kb:>8} kB
            SwapCached:            0 kB
            Active:         {used_kb // 2:>8} kB
            Inactive:       {used_kb // 2:>8} kB
            SwapTotal:      {swap_total_kb:>8} kB
            SwapFree:       {swap_free_kb:>8} kB
            Dirty:               128 kB
            Writeback:             0 kB
            AnonPages:      {used_kb // 3:>8} kB
            Mapped:         {used_kb // 6:>8} kB
            Shmem:             65536 kB
            Slab:           {slab_kb:>8} kB
            SReclaimable:   {sreclaimable_kb:>8} kB
            SUnreclaim:     {slab_kb - sreclaimable_kb:>8} kB
            KernelStack:       16384 kB
            PageTables:        12288 kB
            CommitLimit:    {swap_total_kb + total_kb // 2:>8} kB
            Committed_AS:   {used_kb:>8} kB
            VmallocTotal:   34359738367 kB
            VmallocUsed:       45056 kB
            VmallocChunk:          0 kB""")

    # -- /proc/loadavg ------------------------------------------------------

    def _gen_proc_loadavg(self) -> None:
        running = sum(1 for p in self.processes if p.status == "running")
        total = len(self.processes)
        last_pid = max((p.pid for p in self.processes), default=1)
        self.filesystem["/proc/loadavg"] = (
            f"{self.load_avg[0]:.2f} {self.load_avg[1]:.2f} {self.load_avg[2]:.2f} "
            f"{running}/{total} {last_pid}"
        )

    # -- /proc/swaps --------------------------------------------------------

    def _gen_proc_swaps(self) -> None:
        swap_size_kb = self.swap_total_mb * 1024
        swap_used_kb = self.swap_used_mb * 1024
        self.filesystem["/proc/swaps"] = (
            "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n"
            f"/dev/sda2                               partition\t{swap_size_kb}\t\t{swap_used_kb}\t\t-2"
        )

    # -- /proc/stat (simplified) --------------------------------------------

    def _gen_proc_stat(self) -> None:
        lines = [
            "cpu  12345 678 9012 345678 1234 0 567 0 0 0",
        ]
        for i in range(self.cpu_count):
            lines.append(f"cpu{i} 3086 169 2253 86419 308 0 141 0 0 0")
        lines.append(f"procs_running {sum(1 for p in self.processes if p.status == 'running')}")
        lines.append(f"procs_blocked 0")
        self.filesystem["/proc/stat"] = "\n".join(lines)

    # -- sysctl → filesystem sync ------------------------------------------

    def _sync_sysctl_to_filesystem(self) -> None:
        """Write every sysctl key that has a known filesystem path."""
        for key, path in _SYSCTL_PATH_MAP.items():
            if key in self.sysctl:
                self.filesystem[path] = self.sysctl[key] + "\n"

    # -- /sys/module/zswap/parameters/enabled -------------------------------

    def _gen_zswap_enabled(self) -> None:
        self.filesystem["/sys/module/zswap/parameters/enabled"] = self._zswap_enabled + "\n"

    # -- directory listing helpers ------------------------------------------

    def list_directory(self, path: str) -> Optional[List[str]]:
        """
        Return child names for a virtual directory, or None if not a directory.

        A 'directory' is any prefix that at least one filesystem key starts with.
        """
        # Normalise: ensure trailing /
        if not path.endswith("/"):
            path += "/"

        children: set[str] = set()
        for fpath in self.filesystem:
            if fpath.startswith(path) and fpath != path:
                remainder = fpath[len(path):]
                child = remainder.split("/")[0]
                children.add(child)

        if not children:
            return None
        return sorted(children)

    # =====================================================================
    # State-mutating operations
    # =====================================================================

    def kill_process(self, pid: int) -> Tuple[bool, str]:
        """
        Kill a process by PID.

        Returns (success: bool, message: str).
        If the process is not killable (system-critical), returns failure.
        """
        target = None
        for p in self.processes:
            if p.pid == pid:
                target = p
                break

        if target is None:
            return False, f"kill: ({pid}) - No such process"

        if not target.killable:
            return False, f"kill: ({pid}) - Operation not permitted (system-critical process: {target.name})"

        # Remove from table
        self.processes.remove(target)

        # Recalculate metrics
        self._recalculate_metrics_after_kill(target)

        # Append dmesg entry
        self.dmesg.append(
            f"[{self._next_dmesg_ts()}] Killed process {pid} ({target.name}) "
            f"total-vm:{int(target.memory_mb * 1024)}kB, freed {int(target.memory_mb * 1024)}kB"
        )

        # Update runtime flags
        if target.cpu_percent > 50.0 or target.memory_mb > 2000:
            self.runtime_flags["runaway_process_present"] = False
            # Reduce load
            self.load_avg = (
                max(0.1, self.load_avg[0] - target.cpu_percent / 100 * self.cpu_count),
                max(0.1, self.load_avg[1] - target.cpu_percent / 120 * self.cpu_count),
                self.load_avg[2],  # 15-min avg hasn't caught up yet
            )
            self.runtime_flags["service_health_score"] = min(
                1.0, self.runtime_flags["service_health_score"] + 0.4
            )

        self.regenerate_dynamic_files()
        return True, ""

    def kill_process_by_name(self, name: str) -> Tuple[int, str]:
        """
        Kill all processes matching *name* (like pkill).

        Returns (killed_count, message).
        """
        targets = [p for p in self.processes if p.name == name]
        if not targets:
            return 0, f"pkill: no process found with name '{name}'"

        killed = 0
        errors: list[str] = []
        for t in targets:
            ok, msg = self.kill_process(t.pid)
            if ok:
                killed += 1
            else:
                errors.append(msg)

        if errors:
            return killed, "\n".join(errors)
        return killed, ""

    def write_file(self, path: str, value: str) -> Tuple[bool, str]:
        """
        Write *value* to a virtual filesystem path.

        If the path corresponds to a sysctl, the sysctl registry is updated
        too.  Returns (success, error_message).
        """
        value = value.strip()

        # Validate known paths
        if path == "/sys/module/zswap/parameters/enabled":
            if value.upper() in ("Y", "N"):
                self._zswap_enabled = value.upper()
                self.filesystem[path] = self._zswap_enabled + "\n"
                if self._zswap_enabled == "Y":
                    self.runtime_flags["thrashing"] = False
                    self.dmesg.append(f"[{self._next_dmesg_ts()}] zswap: enabled")
                else:
                    self.dmesg.append(f"[{self._next_dmesg_ts()}] zswap: disabled")
                return True, ""
            else:
                return False, f"bash: echo: invalid value '{value}' for zswap (expected Y or N)"

        # Sysctl-backed paths
        if path in _PATH_SYSCTL_MAP:
            sysctl_key = _PATH_SYSCTL_MAP[path]
            return self.write_sysctl(sysctl_key, value)

        # Generic writable path (only under /proc/sys or /sys)
        if path.startswith(("/proc/sys/", "/sys/")):
            self.filesystem[path] = value + "\n"
            return True, ""

        return False, f"bash: {path}: Read-only file system"

    def read_sysctl(self, key: str) -> Optional[str]:
        """Return the value for a dotted sysctl key, or None."""
        return self.sysctl.get(key)

    def write_sysctl(self, key: str, value: str) -> Tuple[bool, str]:
        """
        Write a sysctl value.  Syncs to filesystem and updates runtime flags.
        """
        value = value.strip()

        # Validate numeric sysctl values where we know the range
        if key == "vm.swappiness":
            try:
                v = int(value)
                if not (0 <= v <= 200):
                    return False, f"sysctl: setting key \"{key}\": Invalid argument"
            except ValueError:
                return False, f"sysctl: setting key \"{key}\": Invalid argument"
            self.sysctl[key] = value
            # Side-effects
            if v <= 30:
                self.runtime_flags["thrashing"] = False
                self.swap_used_mb = max(0, self.swap_used_mb - self.swap_used_mb // 2)
            self.dmesg.append(f"[{self._next_dmesg_ts()}] sysctl: vm.swappiness set to {value}")

        elif key in ("net.ipv4.conf.all.rp_filter", "net.ipv4.conf.default.rp_filter"):
            if value not in ("0", "1", "2"):
                return False, f"sysctl: setting key \"{key}\": Invalid argument"
            self.sysctl[key] = value
            # Check if *both* are now enabled
            all_rp = self.sysctl.get("net.ipv4.conf.all.rp_filter", "0")
            def_rp = self.sysctl.get("net.ipv4.conf.default.rp_filter", "0")
            if all_rp in ("1", "2") and def_rp in ("1", "2"):
                self.runtime_flags["spoofing_protection_enabled"] = True
            self.dmesg.append(f"[{self._next_dmesg_ts()}] sysctl: {key} set to {value}")

        else:
            # Generic: just store it
            self.sysctl[key] = value

        # Sync to filesystem
        if key in _SYSCTL_PATH_MAP:
            self.filesystem[_SYSCTL_PATH_MAP[key]] = value + "\n"
        self.regenerate_dynamic_files()
        return True, ""

    # =====================================================================
    # Helpers
    # =====================================================================

    def _recalculate_metrics_after_kill(self, killed: ProcessEntry) -> None:
        """Reclaim memory from a killed process."""
        self.free_memory_mb = min(
            self.total_memory_mb,
            self.free_memory_mb + int(killed.memory_mb),
        )

    def _next_dmesg_ts(self) -> str:
        """Generate a synthetic kernel timestamp for dmesg."""
        if self.dmesg:
            # Parse last timestamp and increment
            last = self.dmesg[-1]
            try:
                ts_str = last.split("]")[0].strip("[").strip()
                ts = float(ts_str) + 0.001
            except (ValueError, IndexError):
                ts = 9999.0
        else:
            ts = 0.0
        return f"{ts:>12.6f}"

    # =====================================================================
    # Serialization
    # =====================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full state to a plain dict (JSON-safe)."""
        return {
            "processes": [p.to_dict() for p in self.processes],
            "total_memory_mb": self.total_memory_mb,
            "free_memory_mb": self.free_memory_mb,
            "cached_mb": self.cached_mb,
            "buffers_mb": self.buffers_mb,
            "swap_total_mb": self.swap_total_mb,
            "swap_used_mb": self.swap_used_mb,
            "load_avg": list(self.load_avg),
            "cpu_count": self.cpu_count,
            "sysctl": dict(self.sysctl),
            "runtime_flags": dict(self.runtime_flags),
            "filesystem": dict(self.filesystem),
            "dmesg": list(self.dmesg),
            "zswap_enabled": self._zswap_enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelState":
        """Restore a KernelState from a dict produced by ``to_dict``."""
        ks = cls(
            processes=[ProcessEntry.from_dict(p) for p in d["processes"]],
            total_memory_mb=d["total_memory_mb"],
            free_memory_mb=d["free_memory_mb"],
            cached_mb=d["cached_mb"],
            buffers_mb=d["buffers_mb"],
            swap_total_mb=d["swap_total_mb"],
            swap_used_mb=d["swap_used_mb"],
            load_avg=tuple(d["load_avg"]),
            cpu_count=d["cpu_count"],
            runtime_flags=d.get("runtime_flags"),
            dmesg_lines=d.get("dmesg"),
        )
        ks.sysctl = dict(d.get("sysctl", {}))
        ks._zswap_enabled = d.get("zswap_enabled", "N")
        if "filesystem" in d:
            ks.filesystem.update(d["filesystem"])
        ks.regenerate_dynamic_files()
        return ks

    def clone(self) -> "KernelState":
        """Return a deep copy of this state."""
        return copy.deepcopy(self)

    # =====================================================================
    # Convenience — quick inspection
    # =====================================================================

    def get_process_by_pid(self, pid: int) -> Optional[ProcessEntry]:
        for p in self.processes:
            if p.pid == pid:
                return p
        return None

    def get_processes_by_name(self, name: str) -> List[ProcessEntry]:
        return [p for p in self.processes if p.name == name]

    @property
    def zswap_enabled(self) -> bool:
        return self._zswap_enabled == "Y"

    def __repr__(self) -> str:
        return (
            f"<KernelState procs={len(self.processes)} "
            f"mem={self.free_memory_mb}/{self.total_memory_mb}MB "
            f"swap={self.swap_used_mb}/{self.swap_total_mb}MB "
            f"load={self.load_avg} "
            f"flags={self.runtime_flags}>"
        )
