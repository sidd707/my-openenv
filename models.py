"""
Data contracts for SafeAct-Env.

Agent-visible: AvailableAction, AgentAction, SystemObservation
Internal only: ActionRecord, EpisodeState
                (never include internal models in agent-visible responses)
"""

from typing import Any

from openenv.core.env_server.types import (
    Action as OpenEnvAction,
)
from openenv.core.env_server.types import (
    Observation as OpenEnvObservation,
)
from openenv.core.env_server.types import (
    State as OpenEnvState,
)
from pydantic import BaseModel, Field, field_validator


class AvailableAction(BaseModel):
    """Agent-visible action description. NO action_type field — ever."""

    name: str
    description: str
    parameters: dict[str, Any]

    model_config = {"extra": "forbid"}


class AgentAction(OpenEnvAction):
    """What the agent submits each step. Subclasses OpenEnvAction for framework compat.

    Inherits: metadata: Dict[str, Any] from OpenEnvAction (framework field, ignore it)
    """

    action_name: str
    parameters: dict[str, Any]
    reasoning: str

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "reasoning cannot be empty — agent must explain its choice"
            )
        return v


class ActionRecord(BaseModel):
    """Internal only — never include in agent-visible responses."""

    action_name: str
    was_irreversible: bool
    was_mistake: bool
    step: int
    action_params: dict[str, Any] = Field(default_factory=dict)


class EpisodeState(OpenEnvState):
    """Internal only — never include in agent-visible responses.

    Inherits: episode_id: Optional[str], step_count: int from OpenEnvState
    """

    task_name: str = ""
    irreversible_mistakes: int = 0
    task_complete: bool = False
    total_reward: float = 0.0
    history: list[ActionRecord] = Field(default_factory=list)


class SystemObservation(OpenEnvObservation):
    """What the agent receives each step. Contains AvailableAction list — never action_type.

    Inherits from OpenEnvObservation:
      - done: bool = False
      - reward: bool | int | float | None = None
      - metadata: Dict[str, Any] = {}
    """

    task_description: str
    current_state: dict[str, Any]
    available_actions: list[AvailableAction]
    action_history: list[str]
    steps_remaining: int
    last_action_result: str
