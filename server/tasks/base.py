"""
Abstract base class for all SafeAct-Env tasks.
All task classes must subclass BaseTask and implement all abstract methods.
grade() must be deterministic pure Python — no LLM calls ever.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from models import ActionRecord, AvailableAction


class BaseTask(ABC):

    @abstractmethod
    def get_initial_state(self) -> Dict[str, Any]:
        """Return the starting system state. Must be a fresh copy each call."""
        ...

    @abstractmethod
    def get_available_actions(self, state: Dict[str, Any]) -> List[AvailableAction]:
        """Return actions the agent can see. Never include risk labels."""
        ...

    @abstractmethod
    def get_action_risk(self, action_name: str) -> str:
        """Return internal risk level. NEVER include in agent-visible models.
        Returns one of: 'safe', 'risky', 'irreversible', 'escalate', 'unknown'
        """
        ...

    @abstractmethod
    def execute_action(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute action, mutate and return updated state.
        Must not raise — return state unchanged for unknown actions.
        """
        ...

    def can_proceed_with_irreversible(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        state: Dict[str, Any],
    ) -> bool:
        """Return True if prerequisites are met to downgrade an irreversible action.
        Default: never downgrade. Override in tasks with safe-after-checks patterns.
        """
        return False

    @abstractmethod
    def grade(
        self,
        history: List[ActionRecord],
        final_state: Dict[str, Any],
    ) -> float:
        """Score the episode. Must return float in [0.0, 1.0].
        Must be deterministic — same inputs always produce same output.
        No LLM calls. Pure Python math only.
        """
        ...
