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
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from openai import AzureOpenAI

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
    backend = os.getenv("OPENAI_BACKEND", "openai").lower()
    if backend == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for azure backend."
            )
        return AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY must be set. Copy .env.example to .env and fill in credentials."
            )
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Episode runner ────────────────────────────────────────────


def run_episode(
    task_name: str,
    client,
) -> Dict[str, Any]:
    """Run one episode for a task. Returns score, steps, error."""
    print(f"\n[{task_name}] Starting episode...", file=sys.stderr)

    env = IrreversibleActionEnv()
    obs = env.reset(
        task_name=task_name,
        episode_id=f"baseline-{task_name}",
    )

    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1") if os.getenv("OPENAI_BACKEND", "openai").lower() == "azure" else os.getenv("OPENAI_MODEL", "gpt-4.1")

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

    # Grade the episode using the task grader directly
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
    parser = argparse.ArgumentParser(description="SafeAct-Env baseline runner")
    parser.add_argument("--task", type=str, default=None,
                        help="Run only this task (default: all)")
    parser.add_argument("--json", dest="json_mode", action="store_true",
                        help="Print only {\"score\": float} to stdout")
    args = parser.parse_args()

    client = _make_client()

    task_names = [args.task] if args.task else ["easy", "medium", "hard", "medical", "cloud_infra"]
    results = {}

    for task_name in task_names:
        try:
            results[task_name] = run_episode(task_name, client)
        except Exception as e:
            print(f"[{task_name}] ERROR: {e}", file=sys.stderr)
            results[task_name] = {
                "score": 0.0,
                "steps": 0,
                "error": str(e),
            }

    if args.json_mode:
        # Always output {"score": float} — used by /baseline subprocess calls.
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
