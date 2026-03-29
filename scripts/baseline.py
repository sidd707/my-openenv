"""
Baseline inference script for SafeAct-Env.
Runs one episode per task using OpenAI (or Azure OpenAI) as the agent.

    Set OPENAI_BACKEND=openai (default) for standard OpenAI API.
    Set OPENAI_BACKEND=azure for Azure OpenAI.

Usage:
    # Copy .env.example to .env and fill in your credentials
    cp .env.example .env
    # Then run:
    uv run python scripts/baseline.py

Output: JSON to stdout with scores for all 5 tasks.
Progress: printed to stderr.
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

load_dotenv(Path(__file__).parent.parent / ".env")

from openai import AzureOpenAI

from safeact_env.runner import run_all_tasks, run_episode
from server.environment import IrreversibleActionEnv

# ── LLM client ────────────────────────────────────────────────


def _make_client():
    backend = os.getenv("OPENAI_BACKEND", "openai").lower()
    if backend == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        ):
            raise OSError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for azure backend."
            )
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise OSError(
                "OPENAI_API_KEY must be set. Copy .env.example to .env and fill in credentials."
            )
        from openai import OpenAI

        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _get_model() -> str:
    backend = os.getenv("OPENAI_BACKEND", "openai").lower()
    if backend == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    return os.getenv("OPENAI_MODEL", "gpt-4.1")


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeAct-Env baseline runner")
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

    task_names = (
        [args.task]
        if args.task
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
