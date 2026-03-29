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

from openai import AzureOpenAI, OpenAI

from safeact_env.runner import run_all_tasks, run_episode

# ── LLM client ────────────────────────────────────────────────


def _make_client():
    # Primary path (HF Space): API_BASE_URL is set
    if os.getenv("API_BASE_URL"):
        return OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ.get("HF_TOKEN", "hf-token"),
        )

    # Backward compat: Azure backend
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )

    raise EnvironmentError(
        "Set API_BASE_URL (+ HF_TOKEN) or AZURE_OPENAI_API_KEY (+ AZURE_OPENAI_ENDPOINT)."
    )


def _get_model() -> str:
    if os.getenv("API_BASE_URL"):
        return os.environ.get("MODEL_NAME", "gpt-4o")
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeAct-Env inference runner")
    parser.add_argument("--task", type=str, default=None,
                        help="Run only this task (default: all)")
    parser.add_argument("--json", dest="json_mode", action="store_true",
                        help="Print only {\"score\": float} to stdout")
    args = parser.parse_args()

    client = _make_client()
    model = _get_model()

    from server.environment import IrreversibleActionEnv

    task_names = (
        [args.task] if args.task
        else ["easy", "medium", "hard", "medical", "cloud_infra"]
    )

    if args.task:
        env = IrreversibleActionEnv()
        results = {}
        try:
            results[args.task] = run_episode(env, args.task, client, model)
        except Exception as e:
            logger.error("[%s] Episode failed: %s: %s", args.task, type(e).__name__, e)
            results[args.task] = {"score": 0.0, "steps": 0, "error": str(e)}
    else:
        results = run_all_tasks(
            IrreversibleActionEnv, client, model, task_names=task_names
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
