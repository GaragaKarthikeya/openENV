"""Typed models for the Linux SRE Gym environment."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LinuxSreGymAction(BaseModel):
    """One terminal command issued by the agent."""

    command: str = Field(
        ...,
        min_length=1,
        description=(
            "A bash-like command to run against the simulated Linux host, "
            "for example 'cat /proc/meminfo' or 'sysctl -w vm.swappiness=10'."
        ),
    )


class LinuxSreGymRewardBreakdown(BaseModel):
    """Reward components for the latest step."""

    discovery_reward: float = Field(default=0.0, description="Reward for useful state discovery.")
    progress_reward: float = Field(default=0.0, description="Reward for making relevant remediation progress.")
    safety_penalty: float = Field(default=0.0, description="Penalty for destructive or unsafe actions.")
    repeat_penalty: float = Field(default=0.0, description="Penalty for repeated low-value actions.")
    completion_bonus: float = Field(default=0.0, description="Bonus for fully resolving the task.")
    total: float = Field(default=0.0, description="Net reward for the step after penalties and bonuses.")
    score: float = Field(default=0.0, description="Current normalized task score in the range [0.0, 1.0].")


class LinuxSreGymProcess(BaseModel):
    """A simulated process visible to the agent."""

    pid: int = Field(..., description="Process identifier.")
    name: str = Field(..., description="Short process name.")
    command: str = Field(default="", description="Full command line shown in ps/top output.")
    cpu_percent: float = Field(default=0.0, description="Current CPU utilization percentage.")
    memory_mb: int = Field(default=0, description="Current RSS memory usage in megabytes.")
    status: str = Field(default="running", description="Process status.")
    protected: bool = Field(default=False, description="Whether killing this process is considered destructive.")
    killable: bool = Field(default=True, description="Whether the process can be terminated by the agent.")


class LinuxSreGymObservation(BaseModel):
    """Terminal output returned after a command."""

    stdout: str = Field(default="", description="Standard output produced by the command.")
    stderr: str = Field(default="", description="Standard error produced by the command.")
    exit_code: int = Field(default=0, description="Zero for success, non-zero for failure.")
    current_task: str = Field(..., description="The active benchmark task id.")
    step_count: int = Field(default=0, description="Current episode step number after applying the action.")
    last_reward_reason: str = Field(
        default="",
        description="Human-readable explanation for the latest reward update.",
    )
    available_hint: Optional[str] = Field(
        default=None,
        description="Optional hint shown to the agent, usually on reset or after repeated mistakes.",
    )
    reward: float = Field(default=0.0, description="Net reward for the latest step.")
    done: bool = Field(default=False, description="Whether the current episode has ended.")
    reward_breakdown: LinuxSreGymRewardBreakdown = Field(
        default_factory=LinuxSreGymRewardBreakdown,
        description="Detailed reward components for the latest step.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata such as task description, score, and remediation status.",
    )


class LinuxSreGymState(BaseModel):
    """The hidden simulator state exposed for debugging and evaluation."""

    episode_id: str = Field(..., description="Unique id for the current episode.")
    step_count: int = Field(default=0, description="Number of actions taken in the episode.")
    task_id: str = Field(..., description="Current task id.")
    task_description: str = Field(default="", description="Natural-language summary of the incident.")
    filesystem: Dict[str, str] = Field(
        default_factory=dict,
        description="Simulated file contents for procfs/sysfs-style paths.",
    )
    processes: Dict[str, LinuxSreGymProcess] = Field(
        default_factory=dict,
        description="Current simulated process table keyed by PID string.",
    )
    sysctl: Dict[str, str] = Field(
        default_factory=dict,
        description="Kernel tunables tracked by the simulator.",
    )
    network: Dict[str, Any] = Field(
        default_factory=dict,
        description="Networking-related state such as spoofing protection flags.",
    )
    command_history: List[str] = Field(
        default_factory=list,
        description="All commands issued in the current episode.",
    )
    reward_history: List[float] = Field(
        default_factory=list,
        description="Per-step reward history.",
    )
    reward_reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable reasons paired with reward history.",
    )
    seen_diagnostics: List[str] = Field(
        default_factory=list,
        description="Useful diagnostic commands already observed by the grader.",
    )
    is_resolved: bool = Field(default=False, description="Whether the incident objective is resolved.")
    terminal_locked: bool = Field(
        default=False,
        description="Whether the episode is irrecoverably failed due to destructive behavior.",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Latest command error, if any.",
    )
    last_reward_breakdown: LinuxSreGymRewardBreakdown = Field(
        default_factory=LinuxSreGymRewardBreakdown,
        description="Most recent reward computation.",
    )
    completion_score: float = Field(
        default=0.0,
        description="Current normalized task score in the range [0.0, 1.0].",
    )


Action = LinuxSreGymAction
Observation = LinuxSreGymObservation
State = LinuxSreGymState
RewardBreakdown = LinuxSreGymRewardBreakdown
Process = LinuxSreGymProcess
