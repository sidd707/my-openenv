"""
WebSocket client for SafeAct-Env.
Subclasses EnvClient for typed interactions with the environment server.
"""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import AgentAction, EpisodeState, SystemObservation


class SafeActClient(EnvClient[AgentAction, SystemObservation, EpisodeState]):
    """
    Typed client for SafeAct-Env.
    Connects via WebSocket to the environment server.

    Example (async):
        async with SafeActClient(base_url="http://localhost:8000") as env:
            result = await env.reset(task_name="easy")
            result = await env.step(AgentAction(
                action_name="read_file_metadata",
                parameters={"path": "temp_cache_1.tmp"},
                reasoning="Reading metadata before acting",
            ))

    Example (sync):
        with SafeActClient(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_name="easy")
    """

    def _step_payload(self, action: AgentAction) -> dict:
        return action.model_dump()

    def _parse_result(
        self, payload: dict
    ) -> StepResult[SystemObservation]:
        obs = SystemObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> EpisodeState:
        return EpisodeState(**payload)
