from pydantic import BaseModel, Field
from typing import Dict, Any, List

class Action(BaseModel):
    """The command the agent wants to run on the server."""
    command: str = Field(..., description="A bash command to execute (e.g., 'cat /proc/meminfo', 'sysctl -w vm.swappiness=10')")

class Observation(BaseModel):
    """What the terminal outputs back to the agent."""
    stdout: str = Field(..., description="Standard output from the command")
    stderr: str = Field(..., description="Standard error, if the command failed")
    exit_code: int = Field(..., description="0 for success, non-zero for failure")

class State(BaseModel):
    """The hidden internal state of our simulated Linux kernel."""
    current_task: str = Field(..., description="The ID of the active task (easy, medium, hard)")
    # We use a dictionary to mock the sysfs/procfs files and active parameters
    filesystem: Dict[str, Any] = Field(default_factory=dict, description="Simulated file contents and system parameters")
    processes: Dict[int, Dict[str, Any]] = Field(default_factory=dict, description="Simulated running processes")
    command_history: List[str] = Field(default_factory=list, description="Track what the agent has done")
    is_resolved: bool = Field(default=False, description="Whether the current task's objective is met")