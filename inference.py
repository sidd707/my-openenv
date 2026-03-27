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
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import AzureOpenAI, OpenAI

from server.environment import IrreversibleActionEnv
from shared.llm_utils import (
    MAX_STEPS_PER_TASK,
    SYSTEM_PROMPT,
    TASK_REGISTRY,
    build_user_prompt,
    parse_action,
)

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


# ── Episode runner ────────────────────────────────────────────


def run_episode(task_name: str, client, model: str) -> dict:
    """Run one episode for a task. Returns score, steps, error."""
    print(f"\n[{task_name}] Starting episode...", file=sys.stderr)

    env = IrreversibleActionEnv()
    obs = env.reset(
        task_name=task_name,
        episode_id=f"inference-{task_name}",
    )

    steps = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not obs.done and steps < MAX_STEPS_PER_TASK:
        user_prompt = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        action = parse_action(content)
        print(
            f"[{task_name}] step {steps + 1}: {action.action_name}",
            file=sys.stderr,
        )

        obs = env.step(action)
        steps += 1

    # Grade the episode
    task_obj = TASK_REGISTRY[task_name]()
    score = task_obj.grade(
        history=env.state.history,
        final_state=env._current_state,
    )

    print(
        f"[{task_name}] Done. steps={steps} score={score:.3f}",
        file=sys.stderr,
    )
    return {"score": round(score, 4), "steps": steps, "error": None}


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

    task_names = (
        [args.task] if args.task
        else ["easy", "medium", "hard", "medical", "cloud_infra"]
    )
    results = {}

    for task_name in task_names:
        try:
            results[task_name] = run_episode(task_name, client, model)
        except Exception as e:
            print(f"[{task_name}] ERROR: {e}", file=sys.stderr)
            results[task_name] = {
                "score": 0.0,
                "steps": 0,
                "error": str(e),
            }

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
