"""
CommandRouter — parse and dispatch bash-like commands against a KernelState.

All parsing is done in pure Python (no real shell execution).  Each handler
reads or mutates the ``KernelState`` and returns a ``CommandResult``.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .kernel_state import KernelState


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    """Value returned by every command handler."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


# ---------------------------------------------------------------------------
# CommandRouter
# ---------------------------------------------------------------------------

class CommandRouter:
    """
    Tokenise a command string, match it to a handler, and execute against
    a ``KernelState``.

    Usage::

        router = CommandRouter()
        result = router.execute("cat /proc/meminfo", kernel_state)
        print(result.stdout)
    """

    def __init__(self) -> None:
        # Map first token → handler method
        self._dispatch: Dict[str, Callable] = {
            "ls": self._handle_ls,
            "cat": self._handle_cat,
            "ps": self._handle_ps,
            "top": self._handle_top,
            "grep": self._handle_grep,
            "sysctl": self._handle_sysctl,
            "echo": self._handle_echo,
            "kill": self._handle_kill,
            "pkill": self._handle_pkill,
            "free": self._handle_free,
            "vmstat": self._handle_vmstat,
            "dmesg": self._handle_dmesg,
            "hostname": self._handle_hostname,
            "uname": self._handle_uname,
            "whoami": self._handle_whoami,
            "uptime": self._handle_uptime,
        }

    # =================================================================
    # Public API
    # =================================================================

    def execute(self, raw_command: str, state: KernelState) -> CommandResult:
        """
        Parse and run *raw_command* against *state*.

        Supports a limited pipe: ``cmd1 | grep pattern``.
        Everything else with ``|``, ``;``, ``&&`` is unsupported.
        """
        raw_command = raw_command.strip()

        if not raw_command:
            return CommandResult(stdout="", stderr="", exit_code=0)

        # ---- Handle simple pipe: <anything> | grep <pattern> -----------
        if "|" in raw_command:
            return self._handle_pipe(raw_command, state)

        # ---- Handle echo … > path  (redirect) -------------------------
        if ">" in raw_command and not raw_command.strip().startswith(">"):
            return self._handle_redirect(raw_command, state)

        # ---- Normal command dispatch -----------------------------------
        try:
            tokens = shlex.split(raw_command)
        except ValueError as exc:
            return CommandResult(stderr=f"bash: syntax error: {exc}", exit_code=2)

        if not tokens:
            return CommandResult()

        cmd = tokens[0]
        handler = self._dispatch.get(cmd)
        if handler is None:
            return CommandResult(
                stderr=f"bash: {cmd}: command not found",
                exit_code=127,
            )

        try:
            return handler(tokens, state)
        except Exception as exc:  # pragma: no cover — safety net
            return CommandResult(stderr=f"{cmd}: error: {exc}", exit_code=1)

    # =================================================================
    # Pipe & redirect helpers
    # =================================================================

    def _handle_pipe(self, raw: str, state: KernelState) -> CommandResult:
        """
        Support ``<cmd> | grep <pattern>`` (and ``| grep -i``).
        Other pipes are rejected.
        """
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) != 2:
            return CommandResult(
                stderr="bash: only single-pipe commands are supported",
                exit_code=1,
            )

        left_raw, right_raw = parts
        try:
            right_tokens = shlex.split(right_raw)
        except ValueError as exc:
            return CommandResult(stderr=f"bash: syntax error: {exc}", exit_code=2)

        if not right_tokens or right_tokens[0] != "grep":
            return CommandResult(
                stderr="bash: only 'grep' is supported on the right side of a pipe",
                exit_code=1,
            )

        # Execute the left side
        left_result = self.execute(left_raw, state)
        if left_result.exit_code != 0:
            return left_result

        # Apply grep to the left output
        return self._grep_text(left_result.stdout, right_tokens[1:])

    def _handle_redirect(self, raw: str, state: KernelState) -> CommandResult:
        """Handle ``echo <value> > <path>``."""
        # Split on >
        # Support both `echo val > path` and `echo val >path`
        match = re.match(r'^echo\s+(.*?)\s*>\s*(.+)$', raw)
        if not match:
            return CommandResult(
                stderr="bash: syntax error near unexpected token '>'",
                exit_code=2,
            )

        value = match.group(1).strip().strip('"').strip("'")
        path = match.group(2).strip().strip('"').strip("'")

        ok, err = state.write_file(path, value)
        if ok:
            return CommandResult()
        else:
            return CommandResult(stderr=err, exit_code=1)

    # =================================================================
    # Individual command handlers
    # =================================================================

    # -- ls -------------------------------------------------------------

    def _handle_ls(self, tokens: List[str], state: KernelState) -> CommandResult:
        path = tokens[1] if len(tokens) > 1 else "/"
        # Remove trailing flags like -la (we just show names)
        if path.startswith("-"):
            path = tokens[2] if len(tokens) > 2 else "/"

        children = state.list_directory(path)
        if children is None:
            # Maybe it's a file
            if path in state.filesystem:
                return CommandResult(stdout=path)
            return CommandResult(
                stderr=f"ls: cannot access '{path}': No such file or directory",
                exit_code=2,
            )
        return CommandResult(stdout="\n".join(children))

    # -- cat ------------------------------------------------------------

    def _handle_cat(self, tokens: List[str], state: KernelState) -> CommandResult:
        if len(tokens) < 2:
            return CommandResult(stderr="cat: missing operand", exit_code=1)

        path = tokens[1]
        content = state.filesystem.get(path)
        if content is not None:
            return CommandResult(stdout=content)

        # Maybe it's a directory
        if state.list_directory(path) is not None:
            return CommandResult(
                stderr=f"cat: {path}: Is a directory",
                exit_code=1,
            )

        return CommandResult(
            stderr=f"cat: {path}: No such file or directory",
            exit_code=1,
        )

    # -- ps -------------------------------------------------------------

    def _handle_ps(self, tokens: List[str], state: KernelState) -> CommandResult:
        # ps aux  — full process listing
        lines: list[str] = []
        lines.append(f"{'USER':<12} {'PID':>6} {'%CPU':>5} {'%MEM':>5} "
                      f"{'VSZ':>8} {'RSS':>8} {'TTY':<6} {'STAT':<5} "
                      f"{'START':<6} {'TIME':<6} COMMAND")
        for p in sorted(state.processes, key=lambda x: x.pid):
            stat_char = {"running": "R", "sleeping": "S", "zombie": "Z"}.get(p.status, "S")
            mem_pct = (p.memory_mb / state.total_memory_mb * 100) if state.total_memory_mb else 0.0
            vsz = int(p.memory_mb * 1024)
            rss = int(p.memory_mb * 900)  # rough approximation
            lines.append(
                f"{p.user:<12} {p.pid:>6} {p.cpu_percent:>5.1f} {mem_pct:>5.1f} "
                f"{vsz:>8} {rss:>8} {'?':<6} {stat_char + ('N' if p.nice > 0 else ''):<5} "
                f"{'00:00':<6} {'0:00':<6} {p.command or p.name}"
            )
        return CommandResult(stdout="\n".join(lines))

    # -- top -bn1 -------------------------------------------------------

    def _handle_top(self, tokens: List[str], state: KernelState) -> CommandResult:
        total = len(state.processes)
        running = sum(1 for p in state.processes if p.status == "running")
        sleeping = sum(1 for p in state.processes if p.status == "sleeping")
        zombie = sum(1 for p in state.processes if p.status == "zombie")

        total_cpu = sum(p.cpu_percent for p in state.processes)
        idle = max(0.0, 100.0 - total_cpu / state.cpu_count)

        mem_used = state.total_memory_mb - state.free_memory_mb
        swap_free = state.swap_total_mb - state.swap_used_mb

        header = (
            f"top - 12:00:00 up 14 days,  3:22,  1 user,  "
            f"load average: {state.load_avg[0]:.2f}, {state.load_avg[1]:.2f}, {state.load_avg[2]:.2f}\n"
            f"Tasks: {total:>4} total, {running:>4} running, {sleeping:>4} sleeping, "
            f"   0 stopped, {zombie:>4} zombie\n"
            f"%Cpu(s): {total_cpu / state.cpu_count:>5.1f} us,  1.2 sy,  0.0 ni, "
            f"{idle:>5.1f} id,  0.3 wa,  0.0 hi,  0.1 si,  0.0 st\n"
            f"MiB Mem : {state.total_memory_mb:>8.1f} total, {state.free_memory_mb:>8.1f} free, "
            f"{mem_used:>8.1f} used, {state.cached_mb + state.buffers_mb:>8.1f} buff/cache\n"
            f"MiB Swap: {state.swap_total_mb:>8.1f} total, {swap_free:>8.1f} free, "
            f"{state.swap_used_mb:>8.1f} used. {state.free_memory_mb + state.cached_mb:>8.1f} avail Mem\n"
            f"\n"
            f"{'PID':>7} {'USER':<10} {'PR':>3} {'NI':>3} {'VIRT':>8} {'RES':>8} "
            f"{'SHR':>8} {'S':<2} {'%CPU':>5} {'%MEM':>5}  {'TIME+':>9} COMMAND"
        )

        proc_lines: list[str] = []
        for p in sorted(state.processes, key=lambda x: x.cpu_percent, reverse=True):
            s_char = {"running": "R", "sleeping": "S", "zombie": "Z"}.get(p.status, "S")
            mem_pct = (p.memory_mb / state.total_memory_mb * 100) if state.total_memory_mb else 0.0
            virt = int(p.memory_mb * 1024)
            res = int(p.memory_mb * 900)
            shr = int(p.memory_mb * 100)
            proc_lines.append(
                f"{p.pid:>7} {p.user:<10} {20:>3} {p.nice:>3} {virt:>8} {res:>8} "
                f"{shr:>8} {s_char:<2} {p.cpu_percent:>5.1f} {mem_pct:>5.1f}  {'0:00.00':>9} {p.name}"
            )

        return CommandResult(stdout=header + "\n" + "\n".join(proc_lines))

    # -- grep -----------------------------------------------------------

    def _handle_grep(self, tokens: List[str], state: KernelState) -> CommandResult:
        """grep <pattern> <path>  OR  grep -i <pattern> <path>"""
        args = tokens[1:]
        case_insensitive = False
        while args and args[0].startswith("-"):
            flag = args.pop(0)
            if "i" in flag:
                case_insensitive = True

        if len(args) < 2:
            return CommandResult(
                stderr="grep: missing operand",
                exit_code=2,
            )

        pattern = args[0]
        path = args[1]

        content = state.filesystem.get(path)
        if content is None:
            return CommandResult(
                stderr=f"grep: {path}: No such file or directory",
                exit_code=2,
            )

        return self._grep_text(content, ["-i", pattern] if case_insensitive else [pattern])

    def _grep_text(self, text: str, grep_args: List[str]) -> CommandResult:
        """Filter *text* line-by-line with a grep-like pattern."""
        case_insensitive = False
        args = list(grep_args)
        while args and args[0].startswith("-"):
            flag = args.pop(0)
            if "i" in flag:
                case_insensitive = True

        if not args:
            return CommandResult(stderr="grep: missing pattern", exit_code=2)

        pattern = args[0]
        flags = re.IGNORECASE if case_insensitive else 0

        try:
            compiled = re.compile(pattern, flags)
        except re.error:
            # Fall back to literal matching
            compiled = re.compile(re.escape(pattern), flags)

        matches = [line for line in text.splitlines() if compiled.search(line)]

        if matches:
            return CommandResult(stdout="\n".join(matches))
        return CommandResult(stdout="", exit_code=1)  # grep returns 1 on no match

    # -- sysctl ---------------------------------------------------------

    def _handle_sysctl(self, tokens: List[str], state: KernelState) -> CommandResult:
        args = tokens[1:]

        if not args:
            return CommandResult(stderr="sysctl: no variables specified", exit_code=1)

        # sysctl -a  — list all
        if args[0] == "-a":
            lines = [f"{k} = {v}" for k, v in sorted(state.sysctl.items())]
            return CommandResult(stdout="\n".join(lines))

        # sysctl -w key=value  — write
        if args[0] == "-w":
            if len(args) < 2:
                return CommandResult(stderr="sysctl: no variable specified", exit_code=1)
            kv = args[1]
            if "=" not in kv:
                return CommandResult(
                    stderr=f"sysctl: \"{kv}\" must be of the form name=value",
                    exit_code=1,
                )
            key, _, value = kv.partition("=")
            ok, err = state.write_sysctl(key.strip(), value.strip())
            if ok:
                return CommandResult(stdout=f"{key.strip()} = {value.strip()}")
            return CommandResult(stderr=err, exit_code=1)

        # sysctl key  — read single
        key = args[0]
        val = state.read_sysctl(key)
        if val is not None:
            return CommandResult(stdout=f"{key} = {val}")

        return CommandResult(
            stderr=f"sysctl: cannot stat /proc/sys/{key.replace('.', '/')}: "
                   f"No such file or directory",
            exit_code=1,
        )

    # -- echo (no redirect — just print) --------------------------------

    def _handle_echo(self, tokens: List[str], state: KernelState) -> CommandResult:
        # Standalone echo (redirect is handled separately before dispatch)
        text = " ".join(tokens[1:])
        # Strip quotes
        text = text.strip('"').strip("'")
        return CommandResult(stdout=text)

    # -- kill -----------------------------------------------------------

    def _handle_kill(self, tokens: List[str], state: KernelState) -> CommandResult:
        args = tokens[1:]

        # Remove signal flags like -9, -SIGTERM etc.
        signal = "SIGTERM"
        while args and args[0].startswith("-"):
            sig_arg = args.pop(0)
            signal = sig_arg.lstrip("-").upper()
            if signal.isdigit():
                signal = {
                    "9": "SIGKILL", "15": "SIGTERM", "2": "SIGINT"
                }.get(signal, f"SIG{signal}")

        if not args:
            return CommandResult(stderr="kill: usage: kill [-s sigspec | -n signum | -sigspec] pid", exit_code=2)

        errors: list[str] = []
        for arg in args:
            try:
                pid = int(arg)
            except ValueError:
                errors.append(f"kill: ({arg}): arguments must be process or job IDs")
                continue

            ok, msg = state.kill_process(pid)
            if not ok:
                errors.append(msg)

        if errors:
            return CommandResult(stderr="\n".join(errors), exit_code=1)
        return CommandResult()

    # -- pkill ----------------------------------------------------------

    def _handle_pkill(self, tokens: List[str], state: KernelState) -> CommandResult:
        if len(tokens) < 2:
            return CommandResult(stderr="pkill: no process name specified", exit_code=2)

        name = tokens[1]
        killed, msg = state.kill_process_by_name(name)

        if killed == 0:
            return CommandResult(stderr=msg, exit_code=1)

        if msg:  # partial errors
            return CommandResult(stderr=msg, exit_code=1)

        return CommandResult()

    # -- free -----------------------------------------------------------

    def _handle_free(self, tokens: List[str], state: KernelState) -> CommandResult:
        # Always output in MiB (matches free -m)
        mem_used = state.total_memory_mb - state.free_memory_mb - state.cached_mb - state.buffers_mb
        mem_shared = 64  # fixed placeholder
        mem_available = state.free_memory_mb + state.cached_mb + state.buffers_mb
        swap_free = state.swap_total_mb - state.swap_used_mb

        lines = [
            f"{'':>14} {'total':>10} {'used':>10} {'free':>10} "
            f"{'shared':>10} {'buff/cache':>12} {'available':>10}",
            f"{'Mem:':>14} {state.total_memory_mb:>10} {mem_used:>10} {state.free_memory_mb:>10} "
            f"{mem_shared:>10} {state.cached_mb + state.buffers_mb:>12} {mem_available:>10}",
            f"{'Swap:':>14} {state.swap_total_mb:>10} {state.swap_used_mb:>10} {swap_free:>10}",
        ]
        return CommandResult(stdout="\n".join(lines))

    # -- vmstat ---------------------------------------------------------

    def _handle_vmstat(self, tokens: List[str], state: KernelState) -> CommandResult:
        total_cpu = sum(p.cpu_percent for p in state.processes)
        us = int(total_cpu / state.cpu_count)
        sy = 2
        idle = max(0, 100 - us - sy)
        running = sum(1 for p in state.processes if p.status == "running")
        blocked = 0

        # si/so reflect swapping activity
        si = 0
        so = 0
        if state.runtime_flags.get("thrashing"):
            si = 12800
            so = 8400

        free_kb = state.free_memory_mb * 1024
        buff_kb = state.buffers_mb * 1024
        cache_kb = state.cached_mb * 1024
        swap_in = si
        swap_out = so

        lines = [
            "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----",
            " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st",
            f" {running:>1}  {blocked:>1} {state.swap_used_mb * 1024:>6} {free_kb:>6} {buff_kb:>6} "
            f"{cache_kb:>6} {swap_in:>4} {swap_out:>4}   128    64  256  512 {us:>2} {sy:>2} {idle:>2}  0  0",
        ]
        return CommandResult(stdout="\n".join(lines))

    # -- dmesg ----------------------------------------------------------

    def _handle_dmesg(self, tokens: List[str], state: KernelState) -> CommandResult:
        if not state.dmesg:
            return CommandResult(stdout="")

        # Support `dmesg | tail` style by just returning all
        # (tail is handled at pipe level if ever needed)
        return CommandResult(stdout="\n".join(state.dmesg))

    # -- hostname -------------------------------------------------------

    def _handle_hostname(self, tokens: List[str], state: KernelState) -> CommandResult:
        return CommandResult(stdout=state.sysctl.get("kernel.hostname", "sre-lab"))

    # -- uname ----------------------------------------------------------

    def _handle_uname(self, tokens: List[str], state: KernelState) -> CommandResult:
        release = state.sysctl.get("kernel.osrelease", "5.15.0-94-generic")
        if "-a" in tokens:
            return CommandResult(
                stdout=f"Linux sre-lab {release} #1 SMP x86_64 GNU/Linux"
            )
        return CommandResult(stdout="Linux")

    # -- whoami ---------------------------------------------------------

    def _handle_whoami(self, tokens: List[str], state: KernelState) -> CommandResult:
        return CommandResult(stdout="root")

    # -- uptime ---------------------------------------------------------

    def _handle_uptime(self, tokens: List[str], state: KernelState) -> CommandResult:
        return CommandResult(
            stdout=(
                f" 12:00:00 up 14 days,  3:22,  1 user,  "
                f"load average: {state.load_avg[0]:.2f}, "
                f"{state.load_avg[1]:.2f}, {state.load_avg[2]:.2f}"
            )
        )
