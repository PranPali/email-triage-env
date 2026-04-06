"""
EmailTriageEnv — OpenEnv-compliant environment for email triage tasks.

Implements:
    reset()  → Observation
    step(action) → StepResult(observation, reward, done, info)
    state()  → EnvironmentState
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from models import (
    Action,
    ActionType,
    Category,
    Email,
    EmailState,
    EnvironmentState,
    Observation,
    Priority,
    StepResult,
)
from data.emails import TASK_INBOXES
from tasks.graders import TASKS


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.

    Parameters
    ----------
    task_id : str
        One of "easy", "medium", "hard".
    """

    VERSION = "1.0.0"
    ENV_ID = "email-triage-v1"

    def __init__(self, task_id: str = "easy") -> None:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASKS)}")
        self.task_id = task_id
        self._task_cfg = TASKS[task_id]
        self._state: Optional[EnvironmentState] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        raw_emails = TASK_INBOXES[self.task_id]
        inbox = [
            EmailState(email=Email(**e_dict))
            for e_dict in raw_emails
        ]
        self._state = EnvironmentState(
            inbox=inbox,
            step_count=0,
            cumulative_reward=0.0,
            done=False,
            task_id=self.task_id,
            goal=self._task_cfg["description"],
            max_steps=self._task_cfg["max_steps"],
            last_action_error="",
            grader_data={},
        )
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and return (observation, reward, done, info).

        Reward is the *delta* in grader score from before to after the action.
        Done is True when max_steps reached or all emails are resolved/deleted.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.done:
            obs = self._build_observation()
            return StepResult(observation=obs, reward=0.0, done=True, info={"msg": "already done"})

        # Validate
        error_msg = action.validate_action()
        self._state.last_action_error = error_msg or ""

        # Score before action
        score_before = self._grade()

        if not error_msg:
            self._apply_action(action)

        self._state.step_count += 1

        # Score after action
        score_after = self._grade()
        delta_reward = round(score_after - score_before, 4)

        # Penalise invalid actions
        if error_msg:
            delta_reward = -0.01

        # Small step penalty to encourage efficiency
        delta_reward -= 0.001

        self._state.cumulative_reward = round(
            self._state.cumulative_reward + delta_reward, 4
        )

        # Check done conditions
        done = self._check_done()
        self._state.done = done

        obs = self._build_observation()
        info: Dict[str, Any] = {
            "score": score_after,
            "delta_reward": delta_reward,
            "step": self._state.step_count,
            "error": error_msg or "",
        }
        return StepResult(
            observation=obs,
            reward=delta_reward,
            done=done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """Return the full internal environment state."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._state)

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> None:
        """Mutate internal state based on validated action."""
        s = self._state

        # Find target email state
        email_state = self._find_email(action.email_id)
        if email_state is None and action.action_type not in (ActionType.NOOP,):
            s.last_action_error = f"Email id '{action.email_id}' not found."
            return

        atype = action.action_type

        if atype == ActionType.NOOP:
            pass  # deliberate no-op

        elif atype == ActionType.ASSIGN_PRIORITY:
            email_state.assigned_priority = action.priority
            email_state.actions_taken.append(f"priority={action.priority.value}")

        elif atype == ActionType.ASSIGN_CATEGORY:
            email_state.assigned_category = action.category
            email_state.actions_taken.append(f"category={action.category.value}")

        elif atype == ActionType.DRAFT_REPLY:
            email_state.draft_reply = action.reply_text
            email_state.actions_taken.append("draft_reply")

        elif atype == ActionType.MARK_RESOLVED:
            email_state.is_resolved = True
            email_state.actions_taken.append("resolved")

        elif atype == ActionType.ESCALATE:
            email_state.is_escalated = True
            email_state.actions_taken.append("escalated")

        elif atype == ActionType.DELETE:
            email_state.is_deleted = True
            email_state.actions_taken.append("deleted")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_email(self, email_id: Optional[str]) -> Optional[EmailState]:
        if email_id is None:
            return None
        for es in self._state.inbox:
            if es.email.id == email_id:
                return es
        return None

    def _grade(self) -> float:
        return self._task_cfg["grader"](self._state)

    def _check_done(self) -> bool:
        s = self._state
        if s.step_count >= s.max_steps:
            return True
        # All emails resolved or deleted
        all_handled = all(
            es.is_resolved or es.is_deleted
            for es in s.inbox
        )
        return all_handled

    def _build_observation(self) -> Observation:
        s = self._state
        stats = {
            "total": len(s.inbox),
            "resolved": sum(1 for e in s.inbox if e.is_resolved),
            "deleted": sum(1 for e in s.inbox if e.is_deleted),
            "escalated": sum(1 for e in s.inbox if e.is_escalated),
            "with_reply": sum(1 for e in s.inbox if e.draft_reply),
            "unhandled": sum(
                1 for e in s.inbox if not e.is_resolved and not e.is_deleted
            ),
        }
        return Observation(
            inbox=list(s.inbox),
            current_email_idx=0,
            goal=s.goal,
            step_count=s.step_count,
            max_steps=s.max_steps,
            cumulative_reward=s.cumulative_reward,
            last_action_error=s.last_action_error,
            stats=stats,
        )
