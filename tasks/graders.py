"""
Task definitions and graders for the Email Triage OpenEnv.

Each task exposes:
    description  — natural-language goal given to the agent
    grader(state) → float in [0.0, 1.0]
"""

from __future__ import annotations

from typing import Dict, Any
from models import EnvironmentState, Priority, Category, ActionType


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_PRIORITY_ORDER = {
    Priority.URGENT: 4,
    Priority.HIGH: 3,
    Priority.NORMAL: 2,
    Priority.LOW: 1,
    Priority.SPAM: 0,
}


def _priority_score(assigned: Priority | None, expected: str) -> float:
    """Score 1.0 for exact match, 0.5 for one level off, 0.0 for two+ off or None."""
    if assigned is None:
        return 0.0
    try:
        exp = Priority(expected)
    except ValueError:
        return 0.0
    if assigned == exp:
        return 1.0
    diff = abs(_PRIORITY_ORDER[assigned] - _PRIORITY_ORDER[exp])
    return max(0.0, 1.0 - diff * 0.5)


def _category_score(assigned: Category | None, expected: str) -> float:
    """1.0 for exact match, 0.0 otherwise."""
    if assigned is None:
        return 0.0
    try:
        exp = Category(expected)
    except ValueError:
        return 0.0
    return 1.0 if assigned == exp else 0.0


def _reply_quality_score(reply: str | None, requires_reply: bool) -> float:
    """
    Heuristic reply quality:
        - If reply not required, no credit but no penalty.
        - If required, score based on length/completeness proxy.
          < 5 words  → 0.2 (token reply)
          < 15 words → 0.6 (brief but present)
          15+ words  → 1.0 (substantive reply)
    """
    if not requires_reply:
        return 0.0
    if not reply:
        return 0.0
    words = len(reply.split())
    if words < 5:
        return 0.2
    if words < 15:
        return 0.6
    return 1.0


# ---------------------------------------------------------------------------
# Task 1 — EASY
# ---------------------------------------------------------------------------

EASY_TASK = {
    "id": "easy",
    "description": (
        "You are a customer-support agent. "
        "For EACH email in the inbox:\n"
        "1. Assign the correct priority (urgent / high / normal / low / spam).\n"
        "2. Assign the correct category.\n"
        "3. Mark spam emails as deleted.\n"
        "4. Escalate any urgent emails.\n\n"
        "The inbox contains 5 clearly labelled emails. "
        "Score 1.0 means all emails are correctly triaged."
    ),
    "max_steps": 20,
}


def grade_easy(state: EnvironmentState) -> float:
    """
    Grade: average of priority + category accuracy across all 5 emails.
    Bonus +0.2 for each correctly deleted spam (capped at 1.0).
    Bonus +0.2 for each correctly escalated urgent (capped at 1.0).
    """
    if not state.inbox:
        return 0.0

    total = 0.0
    for es in state.inbox:
        meta = es.email.metadata
        p_score = _priority_score(es.assigned_priority, meta.get("expected_priority", ""))
        c_score = _category_score(es.assigned_category, meta.get("expected_category", ""))

        email_score = (p_score * 0.5) + (c_score * 0.5)

        # Bonus for spam deletion
        if meta.get("expected_priority") == "spam" and es.is_deleted:
            email_score = min(1.0, email_score + 0.2)

        # Bonus for urgent escalation
        if meta.get("expected_priority") == "urgent" and es.is_escalated:
            email_score = min(1.0, email_score + 0.2)

        total += email_score

    return round(total / len(state.inbox), 4)


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM
# ---------------------------------------------------------------------------

MEDIUM_TASK = {
    "id": "medium",
    "description": (
        "You are a customer-support triage agent managing a mixed inbox.\n"
        "For EACH email:\n"
        "1. Assign priority and category.\n"
        "2. Draft a reply for any email that requires one (requires_reply=True in context).\n"
        "3. Escalate emails that need engineering or management attention.\n"
        "4. Delete confirmed spam.\n\n"
        "The inbox has 8 emails with varying urgency. "
        "Replies must be professional and address the sender's concern."
    ),
    "max_steps": 35,
}


def grade_medium(state: EnvironmentState) -> float:
    """
    Grade: weighted average across all 8 emails.
    Priority: 30%, Category: 30%, Reply quality: 25%, Escalation: 15%.
    """
    if not state.inbox:
        return 0.0

    total = 0.0
    for es in state.inbox:
        meta = es.email.metadata

        p_score = _priority_score(es.assigned_priority, meta.get("expected_priority", ""))
        c_score = _category_score(es.assigned_category, meta.get("expected_category", ""))
        r_score = _reply_quality_score(es.draft_reply, meta.get("requires_reply", False))

        # Escalation score
        should_escalate = meta.get("requires_escalation", False)
        e_score = 1.0 if (should_escalate == es.is_escalated) else 0.0

        # Spam deletion bonus
        spam_bonus = 0.0
        if meta.get("expected_priority") == "spam" and es.is_deleted:
            spam_bonus = 0.1

        email_score = (
            p_score * 0.30
            + c_score * 0.30
            + r_score * 0.25
            + e_score * 0.15
            + spam_bonus
        )
        total += min(1.0, email_score)

    return round(total / len(state.inbox), 4)


# ---------------------------------------------------------------------------
# Task 3 — HARD
# ---------------------------------------------------------------------------

HARD_TASK = {
    "id": "hard",
    "description": (
        "You are the head of customer operations. "
        "Manage a high-pressure inbox of 12 emails that includes:\n"
        "- Critical security vulnerabilities\n"
        "- Legal compliance demands\n"
        "- Major enterprise sales opportunities\n"
        "- Internal incidents requiring coordination\n"
        "- Routine requests and spam\n\n"
        "For EACH email you must:\n"
        "1. Assign the correct priority (subtle signals required).\n"
        "2. Assign the correct category.\n"
        "3. Draft actionable replies where needed.\n"
        "4. Escalate critical issues.\n"
        "5. Correctly resolve or delete items needing no action.\n\n"
        "Precision matters: marking a GDPR issue as 'normal' or missing a "
        "security escalation will heavily penalise your score."
    ),
    "max_steps": 55,
}


def grade_hard(state: EnvironmentState) -> float:
    """
    Hard grader applies stricter scoring:
    - No partial credit for priority on critical emails (urgent/high must be exact).
    - Reply quality is graded more strictly (minimum 30 words for full score).
    - Missing a required escalation is a -0.2 penalty on that email's score.
    """
    if not state.inbox:
        return 0.0

    total = 0.0
    for es in state.inbox:
        meta = es.email.metadata
        expected_p = meta.get("expected_priority", "normal")

        # Priority: strict for urgent/high
        if expected_p in ("urgent", "high"):
            p_score = 1.0 if (es.assigned_priority and es.assigned_priority.value == expected_p) else 0.0
        else:
            p_score = _priority_score(es.assigned_priority, expected_p)

        c_score = _category_score(es.assigned_category, meta.get("expected_category", ""))

        # Reply: stricter length requirement
        requires_reply = meta.get("requires_reply", False)
        if requires_reply:
            if not es.draft_reply:
                r_score = 0.0
            else:
                words = len(es.draft_reply.split())
                if words < 5:
                    r_score = 0.1
                elif words < 20:
                    r_score = 0.5
                else:
                    r_score = 1.0
        else:
            r_score = 1.0  # No reply needed → full score for leaving blank

        # Escalation: penalty for missing
        should_escalate = meta.get("requires_escalation", False)
        if should_escalate and not es.is_escalated:
            e_penalty = -0.2
        elif not should_escalate and es.is_escalated:
            e_penalty = -0.1   # unnecessary escalation minor penalty
        else:
            e_penalty = 0.0

        email_score = (
            p_score * 0.35
            + c_score * 0.25
            + r_score * 0.25
            + 0.15             # base escalation credit
            + e_penalty
        )
        total += max(0.0, min(1.0, email_score))

    return round(total / len(state.inbox), 4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {
    "easy":   {**EASY_TASK,   "grader": grade_easy},
    "medium": {**MEDIUM_TASK, "grader": grade_medium},
    "hard":   {**HARD_TASK,   "grader": grade_hard},
}
