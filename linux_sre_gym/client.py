"""Client for interacting with the Linux SRE Gym server."""

from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
except ImportError:  # pragma: no cover
    A = TypeVar("A")
    O = TypeVar("O")
    S = TypeVar("S")

    class EnvClient(Generic[A, O, S]):  # type: ignore[misc]
        """Fallback stub used when openenv is not installed locally."""

    @dataclass
    class StepResult(Generic[O]):
        observation: O
        reward: float | None = None
        done: bool = False

from . import models as _models

LinuxSreGymAction = getattr(_models, "LinuxSreGymAction", getattr(_models, "Action"))
LinuxSreGymObservation = getattr(
    _models, "LinuxSreGymObservation", getattr(_models, "Observation")
)
LinuxSreGymState = getattr(_models, "LinuxSreGymState", getattr(_models, "State"))
LinuxSreGymRewardBreakdown = getattr(
    _models,
    "LinuxSreGymRewardBreakdown",
    getattr(_models, "RewardBreakdown", None),
)


class LinuxSreGymEnv(
    EnvClient[LinuxSreGymAction, LinuxSreGymObservation, LinuxSreGymState]
):
    """Typed OpenEnv client for Linux SRE Gym."""

    def _step_payload(self, action: LinuxSreGymAction) -> Dict[str, Any]:
        return {"command": action.command}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[LinuxSreGymObservation]:
        obs_data = payload.get("observation", {})
        observation = LinuxSreGymObservation(
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", payload.get("exit_code", 0)),
            current_task=obs_data.get("current_task", payload.get("task_id", "triage")),
            step_count=obs_data.get("step_count", payload.get("step_count", 0)),
            last_reward_reason=obs_data.get("last_reward_reason", ""),
            available_hint=obs_data.get("available_hint"),
            reward=payload.get("reward", obs_data.get("reward", 0.0)) or 0.0,
            done=payload.get("done", obs_data.get("done", False)),
            reward_breakdown=LinuxSreGymRewardBreakdown(
                **obs_data.get("reward_breakdown", {})
            ),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> LinuxSreGymState:
        return LinuxSreGymState(**payload)
