"""
Pydantic models for SafeAct-Env.
Defines the typed contracts for the OpenEnv step/reset/state interface.
Inherits from openenv.core.env_server: Action, Observation, State.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Internal enums (never sent to the agent)
# ---------------------------------------------------------------------------

class ActionRisk(str, Enum):
    """Risk classification for actions. HIDDEN from the agent."""
    SAFE = "safe"                # read, query, preview — no side effects
    RISKY = "risky"              # modify, rename, archive — reversible side effects
    IRREVERSIBLE = "irreversible"  # delete, drop, send — cannot be undone
    ESCALATE = "escalate"        # ask human for approval


# ---------------------------------------------------------------------------
# Action definitions (what the agent sees vs what the environment knows)
# ---------------------------------------------------------------------------

class ActionDefinition(BaseModel):
    """An available action as presented to the agent.
    Contains ONLY the name, description, and required parameters.
    The agent must infer risk from context — no risk labels here.
    """
    name: str = Field(description="Unique action identifier")
    description: str = Field(description="Human-readable description of what this action does")
    required_params: Dict[str, str] = Field(
        default_factory=dict,
        description="Parameter name → type string, e.g. {'path': 'string', 'table': 'string'}",
    )


class InternalActionDefinition(ActionDefinition):
    """Extended action definition with hidden risk metadata.
    Used internally by the environment — NEVER serialized to the agent.
    """
    risk_level: ActionRisk = Field(description="True risk classification")
    has_safer_alternative: bool = Field(
        default=False,
        description="Whether a safer action exists that achieves a similar outcome",
    )
    escalation_appropriate: bool = Field(
        default=False,
        description="Whether escalating before this action is the correct choice",
    )


# ---------------------------------------------------------------------------
# Agent ↔ Environment contract
# ---------------------------------------------------------------------------

class AgentAction(Action):
    """What the agent submits in step().
    Inherits `metadata: Dict[str, Any]` from openenv Action.
    """
    action_name: str = Field(description="Name of the action to execute (must match an available action)")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters, e.g. {'path': '/temp/cache.tmp'}",
    )
    reasoning: str = Field(
        default="",
        description="Agent's explanation for why it chose this action",
    )


class SystemObservation(Observation):
    """What the agent sees after each step.
    Inherits from openenv Observation: done (bool), reward (float|None), metadata (dict).
    The metadata dict is used to pass post-action info (e.g. revealed risk level).
    """
    task_description: str = Field(default="", description="What the agent needs to accomplish")
    task_id: str = Field(default="", description="Task identifier (easy, medium, hard)")
    current_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current state of the simulated system (files, tables, services)",
    )
    available_actions: List[ActionDefinition] = Field(
        default_factory=list,
        description="Actions the agent can take right now (no risk labels)",
    )
    action_history: List[str] = Field(
        default_factory=list,
        description="Summary of actions taken so far in this episode",
    )
    steps_remaining: int = Field(default=0, description="Steps left before episode timeout")
    last_action_result: str = Field(
        default="Episode started.",
        description="Feedback from the last action executed",
    )


class EpisodeState(State):
    """Episode metadata — returned by state().
    Inherits from openenv State: episode_id (str|None), step_count (int).
    """
    task_id: str = Field(default="", description="Which task is being played")
    max_steps: int = Field(default=20, description="Maximum steps for this task")
    irreversible_mistakes: int = Field(default=0, description="Count of irreversible errors made")
    correct_escalations: int = Field(default=0, description="Count of appropriate escalations")
    unnecessary_escalations: int = Field(default=0, description="Count of pointless escalations")
    task_progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Task completion 0-1")
    task_complete: bool = Field(default=False, description="Whether the task objective is achieved")
    catastrophic_failure: bool = Field(default=False, description="Whether a catastrophic mistake ended the episode")
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode")


# ---------------------------------------------------------------------------
# Endpoint response models
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    """Response model for GET /tasks — describes one task."""
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    action_schema: List[ActionDefinition] = Field(
        description="All possible actions for this task with parameter schemas",
    )


class GraderResult(BaseModel):
    """Response model for POST /grader — episode grading result."""
    task_id: str
    score: float = Field(ge=0.0, le=1.0, description="Final score 0.0 to 1.0")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Score breakdown by component, e.g. {'disk_freed': 0.35, 'no_critical_lost': 0.3}",
    )
    episode_id: Optional[str] = None
    steps_taken: int = 0


class BaselineResult(BaseModel):
    """Response model for POST /baseline — baseline inference results."""
    results: List[GraderResult] = Field(
        default_factory=list,
        description="Grading results for each task",
    )
    model_used: str = Field(default="", description="LLM model used for inference")


# ---------------------------------------------------------------------------
# Action history record (used internally for grading)
# ---------------------------------------------------------------------------

class ActionRecord(BaseModel):
    """Record of a single action taken during an episode.
    Used by graders to evaluate agent behavior.
    """
    step: int
    action_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    risk_level: ActionRisk = ActionRisk.SAFE
    result: str = ""
    reward: float = 0.0
    had_safer_alternative: bool = False
    escalation_was_appropriate: bool = False
