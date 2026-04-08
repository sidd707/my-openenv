"""
WebSocket client for SafeAct-Env.
Subclasses EnvClient[AgentAction, SystemObservation, EpisodeState] from openenv.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import AgentAction, EpisodeState, SystemObservation


class SafeActClient(EnvClient[AgentAction, SystemObservation, EpisodeState]):
    """Client for interacting with a SafeAct-Env server over WebSocket."""

    def _step_payload(self, action: AgentAction) -> Dict[str, Any]:
        """Convert an AgentAction to the JSON payload expected by the server."""
        return {"action": action.model_dump()}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SystemObservation]:
        """Parse the server's step response into a StepResult."""
        obs = SystemObservation(**payload)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EpisodeState:
        """Parse the server's state response into an EpisodeState."""
        return EpisodeState(**payload)
