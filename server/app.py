"""
FastAPI application entry point for SafeAct-Env.
Uses openenv-core's create_app for standard endpoints + static demo UI.
Adds hackathon-required endpoints: /tasks, /grader, /baseline.
"""

import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import Body, HTTPException
from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app
from pydantic import BaseModel, ValidationError

from models import ActionRecord, AgentAction, SystemObservation
from server.environment import _TASK_CONFIG, IrreversibleActionEnv
from server.tasks.cloud_infra import CloudInfraTask
from server.tasks.easy import FileCleanupTask
from server.tasks.hard import ServerMigrationTask
from server.tasks.medical import MedicalTriageTask
from server.tasks.medium import DatabaseMaintenanceTask

# ── Task registry for grader endpoint ────────────────────────
_GRADER_REGISTRY = {
    "easy": FileCleanupTask,
    "medium": DatabaseMaintenanceTask,
    "hard": ServerMigrationTask,
    "medical": MedicalTriageTask,
    "cloud_infra": CloudInfraTask,
}

_ACTION_SCHEMA = {
    "required": ["action_name", "parameters", "reasoning"],
    "properties": {
        "action_name": {
            "type": "string",
            "description": "Name of the action to execute",
        },
        "parameters": {"type": "object", "description": "Action-specific parameters"},
        "reasoning": {
            "type": "string",
            "minLength": 1,
            "description": "Agent must explain its choice",
        },
    },
}

# ── Demo tab for Gradio TabbedInterface ────────────────────────


def build_demo_tab(
    web_manager, action_fields, metadata, is_chat_env, title, quick_start_md
):
    with gr.Blocks() as demo:
        gr.HTML("""
            <div style="text-align:center; padding: 10px 0 4px 0;">
                <b>🎮 Interactive Demo — auto-play mode included</b><br>
                <small>Use the ⚙️ API Playground tab for manual step/reset control</small>
            </div>
            <iframe
                src="/demo"
                width="100%"
                height="850px"
                style="border:none; border-radius:8px;">
            </iframe>
        """)
    return demo


# ── Monkey-patch TabbedInterface to customize tab layout ─────
_OrigTabbedInterface = gr.TabbedInterface

_HEADER_MD = """\
# 🛡️ SafeAct-Env

**Train AI agents to handle irreversible actions safely.**

Actions have hidden risk levels. Some are traps with plausible \
names. One wrong move ends the episode with −1.0 reward.

→ Open the **Interactive Demo** tab to see an agent navigate \
a real-world task in real time."""

_API_WARNING_MD = (
    "> ⚠️ This tab is for programmatic API testing.  \n"
    "> For the interactive demo, use the **🛡️ Interactive Demo** tab."
)


def _patched_tabbed_interface(interface_list, tab_names=None, **kwargs):
    if tab_names is not None and len(tab_names) >= 1 and tab_names[0] == "Playground":
        playground_blocks = interface_list[0]
        demo_blocks = interface_list[1]
        try:
            with gr.Blocks(title=kwargs.get("title", "")) as wrapper:
                gr.Markdown(_HEADER_MD)
                with gr.Tabs():
                    with gr.Tab("🛡️ Interactive Demo"):
                        demo_blocks.render()
                    with gr.Tab("⚙️ API Playground"):
                        gr.Markdown(_API_WARNING_MD)
                        playground_blocks.render()
            return wrapper
        except Exception:
            # Fallback: use original TabbedInterface (loses per-tab warning)
            with gr.Blocks(title=kwargs.get("title", "")) as wrapper:
                gr.Markdown(_HEADER_MD)
                _OrigTabbedInterface(
                    [demo_blocks, playground_blocks],
                    tab_names=["🛡️ Interactive Demo", "⚙️ API Playground"],
                )
            return wrapper
    return _OrigTabbedInterface(interface_list, tab_names=tab_names, **kwargs)


gr.TabbedInterface = _patched_tabbed_interface

# ── Create base app from openenv-core ────────────────────────
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
app = create_app(
    env=IrreversibleActionEnv,
    action_cls=AgentAction,
    observation_cls=SystemObservation,
    max_concurrent_envs=4,
    gradio_builder=build_demo_tab,
)

# ── Replace stateless /reset and /step with session-aware versions ──
# The openenv-core creates a fresh env per request which breaks multi-step
# episodes over HTTP. We store envs keyed by episode_id.
_ENV_SESSIONS: dict[str, tuple[IrreversibleActionEnv, float]] = {}
_SESSIONS_LOCK = threading.Lock()

_SESSION_TTL = 300  # 5 minutes


def _cleanup_stale_sessions() -> None:
    now = time.time()
    stale = [k for k, (_, ts) in _ENV_SESSIONS.items() if now - ts > _SESSION_TTL]
    for k in stale:
        del _ENV_SESSIONS[k]


# Remove the default /reset and /step routes so ours take precedence
app.router.routes = [
    r
    for r in app.router.routes
    if getattr(r, "path", None) not in ("/reset", "/step", "/state")
]


_VERSION = importlib.metadata.version("my-openenv")

logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return {"status": "ok", "environment": "safeact-env", "version": _VERSION}


class ResetRequest(BaseModel):
    task_name: str = "easy"
    episode_id: str | None = None
    seed: int | None = None


class StepRequest(BaseModel):
    action: dict[str, Any]
    episode_id: str | None = None


def _serialize_observation(obs: SystemObservation) -> dict:
    data = obs.model_dump()
    return {
        "observation": data,
        "reward": data.get("reward", 0.0),
        "done": data.get("done", False),
    }


@app.post("/reset")
def reset_episode(request: ResetRequest | None = Body(default=None)):  # noqa: B008
    if request is None:
        request = ResetRequest()
    _cleanup_stale_sessions()
    episode_id = request.episode_id or str(uuid.uuid4())
    env = IrreversibleActionEnv()
    obs = env.reset(
        seed=request.seed, episode_id=episode_id, task_name=request.task_name
    )
    with _SESSIONS_LOCK:
        _ENV_SESSIONS[episode_id] = (env, time.time())
    response = _serialize_observation(obs)
    response["episode_id"] = episode_id
    return response


@app.post("/step")
def step_episode(request: StepRequest):
    # Strict session lookup — no silent fallback
    if not request.episode_id or request.episode_id not in _ENV_SESSIONS:
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing episode_id. Call /reset first to start an episode.",
        )
    env, _ = _ENV_SESSIONS[request.episode_id]

    try:
        action = AgentAction(**request.action)
    except (ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid action payload: {exc}"
        ) from exc
    obs = env.step(action)

    # Clean up completed episodes
    with _SESSIONS_LOCK:
        if getattr(obs, "done", False):
            _ENV_SESSIONS.pop(request.episode_id, None)
        elif request.episode_id:  # only write back if we have a valid key
            _ENV_SESSIONS[request.episode_id] = (env, time.time())

    return _serialize_observation(obs)


@app.get("/state")
def get_state(episode_id: str | None = None):
    if episode_id and episode_id in _ENV_SESSIONS:
        env, _ = _ENV_SESSIONS[episode_id]
        return env.state
    if not episode_id:
        raise HTTPException(
            status_code=400,
            detail="episode_id query parameter is required.",
        )
    raise HTTPException(
        status_code=404,
        detail=f"Episode '{episode_id}' not found. It may have expired.",
    )


@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    return HTMLResponse(
        content=(Path(__file__).parent.parent / "static" / "demo.html").read_text()
    )


# ── Request/Response models ───────────────────────────────────


class GraderRequest(BaseModel):
    task_name: str
    history: list[dict[str, Any]]
    final_state: dict[str, Any]


class GraderResponse(BaseModel):
    score: float


class BaselineRequest(BaseModel):
    task_name: str | None = None


class BaselineResponse(BaseModel):
    scores: dict[str, float | None]


# ── Hackathon-required endpoints ──────────────────────────────


@app.get("/tasks")
def get_tasks():
    """List all tasks with their action schemas."""
    tasks = []
    for name, config in _TASK_CONFIG.items():
        tasks.append(
            {
                "name": name,
                "description": config["description"],
                "max_steps": config["max_steps"],
                "difficulty": config["difficulty"],
                "action_schema": _ACTION_SCHEMA,
            }
        )
    return {"tasks": tasks}


@app.post("/grader", response_model=GraderResponse)
def run_grader(request: GraderRequest):
    """Score a completed episode deterministically."""
    if request.task_name not in _GRADER_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {request.task_name!r}. "
            f"Valid: {list(_GRADER_REGISTRY)}",
        )
    task = _GRADER_REGISTRY[request.task_name]()
    history = [ActionRecord(**r) for r in request.history]
    score = task.grade(history, request.final_state)
    return GraderResponse(score=score)


@app.post("/baseline", response_model=BaselineResponse)
def run_baseline(request: BaselineRequest):
    """Trigger baseline agent run via subprocess."""
    tasks = [request.task_name] if request.task_name else list(_GRADER_REGISTRY.keys())

    # Early exit if no LLM credentials are available
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get(
        "AZURE_OPENAI_API_KEY"
    ):
        return BaselineResponse(scores={t: None for t in tasks})

    scores: dict[str, float | None] = {}

    for task in tasks:
        try:
            result = subprocess.run(
                [sys.executable, "scripts/baseline.py", "--task", task, "--json"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            data = json.loads(result.stdout.strip())
            scores[task] = data["score"]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Baseline run failed for task %s: %s", task, exc)
            scores[task] = None

    return BaselineResponse(scores=scores)


def main():
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
