"""
Inference script for SafeAct-Env (HuggingFace Space evaluation).
Runs one episode per task using the OpenAI-compatible API provided by the Space.

Environment variables:
    API_BASE_URL  — base URL for the OpenAI-compatible endpoint
    MODEL_NAME    — model name to use (default: gpt-4o)
    HF_TOKEN      — HuggingFace token used as api_key

Backward compat: if AZURE_OPENAI_API_KEY is set, uses Azure backend instead.

Usage:
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o HF_TOKEN=sk-... \
        uv run python inference.py
    # or single task:
    uv run python inference.py --task easy --json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import time

START_TIME: float = 0.0
MAX_RUNTIME_SECONDS = 18 * 60  # 18 minutes safety buffer
SUCCESS_SCORE_THRESHOLD = 0.1

_step_rewards: list[float] = []


def log_start(task: str, env: str, model: str) -> None:
    global START_TIME
    START_TIME = time.time()
    _step_rewards.clear()
    print(f"[START] task={task} env=safeact-env model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    _step_rewards.append(reward)
    done_str = "true" if done else "false"
    error_str = "null" if error is None else str(error)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


from openai import AzureOpenAI, OpenAI

from safeact_env.runner import run_episode

# ── LLM client ────────────────────────────────────────────────


def _make_client():
    # Primary path (HF Space): API_BASE_URL is set
    if os.getenv("API_BASE_URL"):
        return OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.environ["HF_TOKEN"],
        )

    # Backward compat: Azure backend
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

    raise OSError(
        "Set API_BASE_URL (+ HF_TOKEN) or AZURE_OPENAI_API_KEY (+ AZURE_OPENAI_ENDPOINT)."
    )


def _get_model() -> str:
    if os.getenv("API_BASE_URL"):
        return os.environ.get("MODEL_NAME", "gpt-4o")
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeAct-Env inference runner")
    parser.add_argument(
        "--task", type=str, default=None, help="Run only this task (default: all)"
    )
    parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        help='Print only {"score": float} to stdout',
    )
    args = parser.parse_args()

    client = _make_client()
    model = _get_model()

    from server.environment import IrreversibleActionEnv

    task_names = (
        [args.task]
        if args.task
        else ["easy", "medium", "hard", "medical", "cloud_infra"]
    )

    if args.task:
        env = IrreversibleActionEnv()
        results = {}
        log_start(task=args.task, env="SafeAct-Env", model=model)
        result = {"score": 0.0, "steps": 0, "error": None}
        try:
            result = run_episode(
                env,
                args.task,
                client,
                model,
                log_step_fn=log_step,
                start_time=START_TIME,
                max_runtime=MAX_RUNTIME_SECONDS,
            )
            results[args.task] = result
        except Exception as e:
            logger.error("[%s] Episode failed: %s: %s", args.task, type(e).__name__, e)
            results[args.task] = {"score": 0.0, "steps": 0, "error": str(e)}
            result = results[args.task]
        log_end(
            success=result["score"] >= SUCCESS_SCORE_THRESHOLD,
            steps=result["steps"],
            score=result["score"],
            rewards=list(_step_rewards),
        )
    else:
        results = {}
        for task_id in task_names:
            log_start(task=task_id, env="SafeAct-Env", model=model)
            result = {"score": 0.0, "steps": 0, "error": None}
            try:
                env = IrreversibleActionEnv()
                result = run_episode(
                    env,
                    task_id,
                    client,
                    model,
                    log_step_fn=log_step,
                    start_time=START_TIME,
                    max_runtime=MAX_RUNTIME_SECONDS,
                )
                results[task_id] = result
            except Exception as e:
                logger.error(
                    "[%s] Episode failed: %s: %s", task_id, type(e).__name__, e
                )
                results[task_id] = {"score": 0.0, "steps": 0, "error": str(e)}
                result = results[task_id]
            log_end(
                success=result["score"] >= SUCCESS_SCORE_THRESHOLD,
                steps=result["steps"],
                score=result["score"],
                rewards=list(_step_rewards),
            )

    if args.json_mode:
        if args.task:
            score = results[args.task]["score"]
        else:
            scores = [r["score"] for r in results.values()]
            score = round(sum(scores) / len(scores), 4) if scores else 0.0
        print(json.dumps({"score": score}))
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
