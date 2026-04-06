"""
Inference Script — Email Triage OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env import EmailTriageEnv
from models import Action, ActionType, Category, Priority

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "email-triage-v1"
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.1
MAX_TOKENS = 512
FALLBACK_ACTION = Action(action_type=ActionType.NOOP)

# ---------------------------------------------------------------------------
# Structured stdout logging — EXACT format required by judges
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
      "priority": "<urgent|high|normal|low|spam>",
      "category": "<customer_support|billing|bug_report|feature_request|internal|sales|spam|other>",
      "reply_text": "<reply body>"
    }

    Strategy:
    1. Go through EACH email one by one — do not repeat actions on the same email.
    2. For each email assign priority first, then category, then reply/escalate/delete if needed.
    3. Once an email has priority AND category assigned, move to the NEXT email immediately.
    4. Delete spam emails (priority=spam). Escalate urgent emails involving outages/security/legal.
    5. After all emails are handled, mark remaining ones as resolved.

    IMPORTANT: Check the inbox state carefully. If an email already has a priority assigned,
    do NOT assign priority again — move on to assigning its category or the next email.
    Never repeat the same action on the same email twice.

    Respond ONLY with the JSON action object. No explanation. No markdown fences.
    """
).strip()


def _obs_to_prompt(obs_dict: Dict[str, Any]) -> str:
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
    if not response_text:
        return FALLBACK_ACTION
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()
    try:
        data = json.loads(text)
        at = data.get("action_type", "noop")
        kwargs: Dict[str, Any] = {"action_type": ActionType(at)}
        if "email_id" in data:
            kwargs["email_id"] = str(data["email_id"])
        if data.get("priority"):
            kwargs["priority"] = Priority(data["priority"])
        if data.get("category"):
            kwargs["category"] = Category(data["category"])
        if data.get("reply_text"):
            kwargs["reply_text"] = str(data["reply_text"])
        return Action(**kwargs)
    except Exception as exc:
        print(f"[DEBUG] parse error: {exc} | raw: {response_text[:120]}", flush=True)
        return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_id: str) -> None:
    """Run one full episode, emitting [START], [STEP]..., [END] to stdout."""

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[Dict] = []

    try:
        done = False
        while not done:
            obs_dict = obs.model_dump()
            user_prompt = _obs_to_prompt(obs_dict)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history[-6:],
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
                print(f"[DEBUG] API error: {exc}", flush=True)
                response_text = '{"action_type": "noop"}'

            action = parse_action(response_text)

            # Compact action string for logging
            action_str = action.action_type.value
            if action.email_id:
                action_str += f"(email={action.email_id}"
                if action.priority:
                    action_str += f",priority={action.priority.value}"
                if action.category:
                    action_str += f",category={action.category.value}"
                action_str += ")"

            result = env.step(action)
            obs = result.observation
            reward = round(result.reward, 2)
            done = result.done
            error = result.observation.last_action_error or None

            rewards.append(reward)
            steps_taken += 1
            score = result.info.get("score", 0.0)

            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            history.append({"role": "assistant", "content": response_text})
            if error:
                history.append({"role": "user", "content": f"[Error]: {error}"})

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASKS:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
