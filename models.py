"""
Typed Pydantic models for the Email Triage OpenEnv environment.
Observation, Action, and Reward follow the OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SPAM = "spam"


class Category(str, Enum):
    CUSTOMER_SUPPORT = "customer_support"
    BILLING = "billing"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    INTERNAL = "internal"
    SALES = "sales"
    SPAM = "spam"
    OTHER = "other"


class ActionType(str, Enum):
    ASSIGN_PRIORITY = "assign_priority"
    ASSIGN_CATEGORY = "assign_category"
    DRAFT_REPLY = "draft_reply"
    MARK_RESOLVED = "mark_resolved"
    ESCALATE = "escalate"
    DELETE = "delete"
    NOOP = "noop"


# ---------------------------------------------------------------------------
# Email data model
# ---------------------------------------------------------------------------

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    sender_domain: str
    body: str
    received_at: str                        # ISO-8601 string
    thread_length: int = 1
    has_attachment: bool = False
    cc: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailState(BaseModel):
    email: Email
    assigned_priority: Optional[Priority] = None
    assigned_category: Optional[Category] = None
    draft_reply: Optional[str] = None
    is_resolved: bool = False
    is_escalated: bool = False
    is_deleted: bool = False
    actions_taken: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Full observation returned by step() and reset()."""

    # Current inbox snapshot
    inbox: List[EmailState]

    # Index of the email the agent is currently focusing on (-1 = choose any)
    current_email_idx: int = 0

    # Task goal description
    goal: str

    # How many steps have been taken this episode
    step_count: int = 0

    # Maximum steps allowed
    max_steps: int = 20

    # Cumulative reward so far
    cumulative_reward: float = 0.0

    # Last action error message (empty string = no error)
    last_action_error: str = ""

    # Rich context: unread count, urgent count, etc.
    stats: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Action the agent sends to step()."""

    action_type: ActionType

    # Which email to act on (by email id)
    email_id: Optional[str] = None

    # Payload depending on action_type
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    reply_text: Optional[str] = None

    # Move focus to a different email index
    focus_index: Optional[int] = None

    def validate_action(self) -> Optional[str]:
        """Return an error string if the action is malformed, else None."""
        if self.action_type == ActionType.ASSIGN_PRIORITY and self.priority is None:
            return "assign_priority requires 'priority' field"
        if self.action_type == ActionType.ASSIGN_CATEGORY and self.category is None:
            return "assign_category requires 'category' field"
        if self.action_type == ActionType.DRAFT_REPLY and not self.reply_text:
            return "draft_reply requires 'reply_text' field"
        return None


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State (full internal state, superset of observation)
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    inbox: List[EmailState]
    step_count: int = 0
    cumulative_reward: float = 0.0
    done: bool = False
    task_id: str = ""
    goal: str = ""
    max_steps: int = 20
    last_action_error: str = ""
    grader_data: Dict[str, Any] = Field(default_factory=dict)
