"""
Inference Script — Email Triage OpenEnv
===================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier
    HF_TOKEN       Your Hugging Face / API key

Usage:
    python inference.py [--task easy|medium|hard|all] [--episodes N]

Produces reproducible baseline scores across all 3 tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Local imports — works when run from repo root
sys.path.insert(0, os.path.dirname(__file__))
from env import EmailTriageEnv
from models import Action, ActionType, Category, Priority

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

TEMPERATURE: float = 0.1
MAX_TOKENS: int = 512
FALLBACK_ACTION = Action(action_type=ActionType.NOOP)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert email triage agent. You manage a corporate inbox.

    At each step you receive a JSON observation containing:
    - `inbox`: list of email states (each with id, subject, sender, body, current assignments)
    - `goal`: what you must accomplish
    - `step_count` / `max_steps`: budget remaining
    - `stats`: summary counts

    You must respond with a SINGLE JSON action object. The action schema is:

    {
      "action_type": <one of: assign_priority, assign_category, draft_reply,
                               mark_resolved, escalate, delete, noop>,
      "email_id": "<id of the email to act on>",
      "priority": "<urgent|high|normal|low|spam>",          // only for assign_priority
      "category": "<customer_support|billing|bug_report|feature_request|internal|sales|spam|other>",  // only for assign_category
      "reply_text": "<reply body>"                          // only for draft_reply
    }

    Strategy:
    1. Start by assigning priorities (most important first: urgent > high > normal > low > spam).
    2. Assign categories to each email.
    3. Draft replies for emails that clearly need a response.
    4. Escalate emails tagged urgent that involve security, legal, or outages.
    5. Delete spam emails.
    6. Mark everything else resolved.

    Respond ONLY with the JSON action object. No explanation. No markdown.
    """
).strip()


def _obs_to_prompt(obs_dict: Dict[str, Any]) -> str:
    """Convert observation dict to a concise text prompt."""
    inbox_summary = []
    for i, es in enumerate(obs_dict.get("inbox", [])):
        email = es.get("email", {})
        inbox_summary.append(
            f"[{i}] id={email.get('id')} | "
            f"subject={email.get('subject', '')[:60]} | "
            f"from={email.get('sender', '')} | "
            f"priority={es.get('assigned_priority') or 'unset'} | "
            f"category={es.get('assigned_category') or 'unset'} | "
            f"reply={'yes' if es.get('draft_reply') else 'no'} | "
            f"escalated={es.get('is_escalated')} | "
            f"resolved={es.get('is_resolved')} | "
            f"deleted={es.get('is_deleted')}"
        )

    stats = obs_dict.get("stats", {})
    lines = [
        f"Goal: {obs_dict.get('goal', '')[:200]}",
        f"Step: {obs_dict.get('step_count')}/{obs_dict.get('max_steps')}",
        f"Stats: {json.dumps(stats)}",
        "Inbox:",
        *inbox_summary,
        "",
        "Pick the MOST IMPACTFUL next action and return it as JSON.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Action:
    """Parse model JSON response into an Action object."""
    if not response_text:
        return FALLBACK_ACTION
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()
    try:
        data = json.loads(text)
        at = data.get("action_type", "noop")
        kwargs: Dict[str, Any] = {"action_type": ActionType(at)}
        if "email_id" in data:
            kwargs["email_id"] = str(data["email_id"])
        if "priority" in data and data["priority"]:
            kwargs["priority"] = Priority(data["priority"])
        if "category" in data and data["category"]:
            kwargs["category"] = Category(data["category"])
        if "reply_text" in data and data["reply_text"]:
            kwargs["reply_text"] = str(data["reply_text"])
        return Action(**kwargs)
    except Exception as exc:
        print(f"  [parse error] {exc} | raw: {response_text[:120]}")
        return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_id: str, verbose: bool = True) -> Dict[str, Any]:
    """Run one full episode and return result stats."""
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()

    cumulative_reward = 0.0
    steps_taken = 0
    done = False
    history: List[str] = []

    while not done:
        obs_dict = obs.model_dump()
        user_prompt = _obs_to_prompt(obs_dict)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[{"role": m["role"], "content": m["content"]} for m in history[-6:]],
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [API error] {exc}")
            response_text = json.dumps({"action_type": "noop"})

        action = parse_action(response_text)

        if verbose:
            print(f"  Step {obs.step_count + 1}: {action.action_type.value}"
                  f"  email={action.email_id}  "
                  f"priority={action.priority}  category={action.category}")

        result = env.step(action)
        obs = result.observation
        cumulative_reward += result.reward
        steps_taken += 1
        done = result.done

        # Append assistant action to history for context
        history.append({"role": "assistant", "content": response_text})

        if result.info.get("error"):
            history.append({"role": "user", "content": f"[Error]: {result.info['error']}"})

    final_score = result.info.get("score", 0.0)
    return {
        "task_id": task_id,
        "steps_taken": steps_taken,
        "final_score": final_score,
        "cumulative_reward": round(cumulative_reward, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv baseline inference")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks_to_run = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    all_results: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  Email Triage OpenEnv — Baseline Inference")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Tasks : {tasks_to_run}")
    print(f"{'='*60}\n")

    for task_id in tasks_to_run:
        print(f"\n--- Task: {task_id.upper()} ---")
        episode_results = []
        for ep in range(args.episodes):
            print(f"  Episode {ep + 1}/{args.episodes}")
            res = run_episode(client, task_id, verbose=args.verbose)
            episode_results.append(res)
            print(f"  → Score: {res['final_score']:.4f}  "
                  f"Steps: {res['steps_taken']}  "
                  f"CumReward: {res['cumulative_reward']:.4f}")

        avg_score = sum(r["final_score"] for r in episode_results) / len(episode_results)
        print(f"\n  Average score [{task_id}]: {avg_score:.4f}")
        all_results.extend(episode_results)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for task_id in tasks_to_run:
        task_results = [r for r in all_results if r["task_id"] == task_id]
        avg = sum(r["final_score"] for r in task_results) / len(task_results)
        print(f"  {task_id:8s}: avg_score={avg:.4f}  (n={len(task_results)})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
