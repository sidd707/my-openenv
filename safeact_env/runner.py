"""
Shared episode runner for SafeAct-Env.
Used by both inference.py and scripts/baseline.py to avoid code duplication.
"""

import logging

from shared.llm_utils import (
    MAX_STEPS_PER_TASK,
    SYSTEM_PROMPT,
    TASK_REGISTRY,
    build_user_prompt,
    parse_action,
)

logger = logging.getLogger(__name__)


def run_episode(
    env,
    task_id: str,
    client,
    model: str,
    max_steps: int = MAX_STEPS_PER_TASK,
) -> dict:
    """Run one episode for a task. Returns {score, steps, error}."""
    logger.info("[%s] Starting episode...", task_id)

    obs = env.reset(
        task_name=task_id,
        episode_id=f"run-{task_id}",
    )

    steps = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not obs.done and steps < max_steps:
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
        logger.info("[%s] step %d: %s", task_id, steps + 1, action.action_name)

        obs = env.step(action)
        steps += 1

    # Grade the episode
    task_obj = TASK_REGISTRY[task_id]()
    score = task_obj.grade(
        history=env.state.history,
        final_state=env._current_state,
    )

    logger.info("[%s] Done. steps=%d score=%.3f", task_id, steps, score)
    return {"score": round(score, 4), "steps": steps, "error": None}


def run_all_tasks(
    env_cls,
    client,
    model: str,
    task_names: list[str] | None = None,
) -> dict[str, dict]:
    """Run all tasks, returning {task_id: {score, steps, error}}."""
    if task_names is None:
        task_names = ["easy", "medium", "hard", "medical", "cloud_infra"]

    results = {}
    for task_id in task_names:
        env = env_cls()
        try:
            results[task_id] = run_episode(env, task_id, client, model)
        except Exception as e:
            logger.error("[%s] Episode failed: %s: %s", task_id, type(e).__name__, e)
            results[task_id] = {
                "score": 0.0,
                "steps": 0,
                "error": str(e),
            }
    return results
