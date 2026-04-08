"""
Abstract base class for all SafeAct tasks.
Defines the interface that easy/medium/hard tasks must implement.
Each task contains both the simulation logic AND its grader.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models import ActionRecord, GraderResult, InternalActionDefinition, TaskInfo


class BaseTask(ABC):
    """Abstract base for all SafeAct environment tasks."""

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Unique identifier: 'easy', 'medium', or 'hard'."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Task prompt shown to the agent."""

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """'easy', 'medium', or 'hard'."""

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum number of steps before episode timeout."""

    # ------------------------------------------------------------------
    # Simulation methods
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_initial_state(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Create the simulated system (files, tables, services, etc.).
        Must be deterministic when seed is provided.
        Returns a dict representing the full system state.
        """

    @abstractmethod
    def get_available_actions(self, state: Dict[str, Any]) -> List[InternalActionDefinition]:
        """Return all actions available in the current state.
        Actions may change based on what has already been done
        (e.g., can't delete a file that's already deleted).
        Returns InternalActionDefinition (includes hidden risk labels).
        """

    @abstractmethod
    def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """Execute an action against the simulated system.

        Args:
            action_name: Name of the action to execute.
            params: Action parameters from the agent.
            state: Current system state (will be mutated).

        Returns:
            Tuple of (updated_state, result_message).
            result_message is human-readable feedback like
            'File deleted successfully' or 'Error: file not found'.
        """

    @abstractmethod
    def get_task_progress(self, state: Dict[str, Any]) -> float:
        """Return 0.0 to 1.0 representing how much of the task is done.
        Used for partial reward calculation.
        """

    @abstractmethod
    def is_complete(self, state: Dict[str, Any]) -> bool:
        """Has the task objective been fully achieved?"""

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    @abstractmethod
    def grade(
        self,
        action_history: List[ActionRecord],
        final_state: Dict[str, Any],
    ) -> GraderResult:
        """Deterministic grading of an episode.
        Same inputs MUST produce the same score. Always.
        No randomness, no LLM-as-judge.

        Returns GraderResult with score in [0.0, 1.0] and breakdown.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_task_info(self) -> TaskInfo:
        """Build the TaskInfo response for the /tasks endpoint.
        Uses a dummy state to get the full action schema.
        """
        dummy_state = self.generate_initial_state(seed=42)
        actions = self.get_available_actions(dummy_state)
        from models import ActionDefinition
        action_schema = [
            ActionDefinition(
                name=a.name,
                description=a.description,
                required_params=a.required_params,
            )
            for a in actions
        ]
        return TaskInfo(
            task_id=self.task_id,
            name=self.name,
            description=self.description,
            difficulty=self.difficulty,
            max_steps=self.max_steps,
            action_schema=action_schema,
        )
