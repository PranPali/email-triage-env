"""
Test suite for Email Triage OpenEnv.
Validates OpenEnv spec compliance: reset(), step(), state(), reward, done conditions.

Run:
    python -m pytest tests/test_env.py -v
"""

from __future__ import annotations

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env import EmailTriageEnv
from models import (
    Action, ActionType, Category, EnvironmentState,
    Observation, Priority, StepResult
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["easy", "medium", "hard"])
def env(request):
    e = EmailTriageEnv(task_id=request.param)
    e.reset()
    return e


@pytest.fixture
def easy_env():
    e = EmailTriageEnv(task_id="easy")
    e.reset()
    return e


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, Observation)

    def test_observation_has_inbox(self, env):
        obs = env.reset()
        assert len(obs.inbox) > 0

    def test_step_count_zero(self, env):
        obs = env.reset()
        assert obs.step_count == 0

    def test_cumulative_reward_zero(self, env):
        obs = env.reset()
        assert obs.cumulative_reward == 0.0

    def test_goal_non_empty(self, env):
        obs = env.reset()
        assert len(obs.goal) > 10

    def test_easy_has_5_emails(self):
        env = EmailTriageEnv(task_id="easy")
        obs = env.reset()
        assert len(obs.inbox) == 5

    def test_medium_has_8_emails(self):
        env = EmailTriageEnv(task_id="medium")
        obs = env.reset()
        assert len(obs.inbox) == 8

    def test_hard_has_12_emails(self):
        env = EmailTriageEnv(task_id="hard")
        obs = env.reset()
        assert len(obs.inbox) == 12

    def test_reset_clears_previous_state(self, easy_env):
        # Take an action, then reset
        first_email_id = easy_env.state().inbox[0].email.id
        easy_env.step(Action(
            action_type=ActionType.ASSIGN_PRIORITY,
            email_id=first_email_id,
            priority=Priority.URGENT,
        ))
        obs = easy_env.reset()
        # After reset priority should be unset
        assert obs.inbox[0].assigned_priority is None


# ---------------------------------------------------------------------------
# step() tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_step_result(self, easy_env):
        action = Action(action_type=ActionType.NOOP)
        result = easy_env.step(action)
        assert isinstance(result, StepResult)

    def test_step_result_fields(self, easy_env):
        action = Action(action_type=ActionType.NOOP)
        result = easy_env.step(action)
        assert isinstance(result.observation, Observation)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_increments_count(self, easy_env):
        obs_before = easy_env.reset()
        email_id = obs_before.inbox[0].email.id
        result = easy_env.step(Action(
            action_type=ActionType.ASSIGN_PRIORITY,
            email_id=email_id,
            priority=Priority.NORMAL,
        ))
        assert result.observation.step_count == 1

    def test_assign_priority_updates_state(self, easy_env):
        obs = easy_env.reset()
        email_id = obs.inbox[0].email.id
        easy_env.step(Action(
            action_type=ActionType.ASSIGN_PRIORITY,
            email_id=email_id,
            priority=Priority.URGENT,
        ))
        state = easy_env.state()
        assert state.inbox[0].assigned_priority == Priority.URGENT

    def test_assign_category_updates_state(self, easy_env):
        obs = easy_env.reset()
        email_id = obs.inbox[0].email.id
        easy_env.step(Action(
            action_type=ActionType.ASSIGN_CATEGORY,
            email_id=email_id,
            category=Category.BUG_REPORT,
        ))
        state = easy_env.state()
        assert state.inbox[0].assigned_category == Category.BUG_REPORT

    def test_draft_reply_updates_state(self, easy_env):
        obs = easy_env.reset()
        email_id = obs.inbox[0].email.id
        reply_text = "Thank you for reaching out. We are investigating the issue."
        easy_env.step(Action(
            action_type=ActionType.DRAFT_REPLY,
            email_id=email_id,
            reply_text=reply_text,
        ))
        state = easy_env.state()
        assert state.inbox[0].draft_reply == reply_text

    def test_escalate_updates_state(self, easy_env):
        obs = easy_env.reset()
        email_id = obs.inbox[0].email.id
        easy_env.step(Action(action_type=ActionType.ESCALATE, email_id=email_id))
        state = easy_env.state()
        assert state.inbox[0].is_escalated is True

    def test_delete_updates_state(self, easy_env):
        obs = easy_env.reset()
        email_id = obs.inbox[2].email.id  # spam email
        easy_env.step(Action(action_type=ActionType.DELETE, email_id=email_id))
        state = easy_env.state()
        assert state.inbox[2].is_deleted is True

    def test_mark_resolved_updates_state(self, easy_env):
        obs = easy_env.reset()
        email_id = obs.inbox[0].email.id
        easy_env.step(Action(action_type=ActionType.MARK_RESOLVED, email_id=email_id))
        state = easy_env.state()
        assert state.inbox[0].is_resolved is True

    def test_invalid_action_returns_negative_reward(self, easy_env):
        # Missing required 'priority' field
        action = Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e001")
        result = easy_env.step(action)
        assert result.reward < 0

    def test_invalid_action_sets_error(self, easy_env):
        action = Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e001")
        result = easy_env.step(action)
        assert result.observation.last_action_error != ""

    def test_max_steps_terminates_episode(self):
        env = EmailTriageEnv(task_id="easy")
        env.reset()
        done = False
        for _ in range(25):  # max_steps = 20
            result = env.step(Action(action_type=ActionType.NOOP))
            done = result.done
        assert done is True

    def test_all_resolved_terminates_episode(self, easy_env):
        obs = easy_env.reset()
        done = False
        for es in obs.inbox:
            result = easy_env.step(Action(
                action_type=ActionType.MARK_RESOLVED,
                email_id=es.email.id,
            ))
            done = result.done
        assert done is True


# ---------------------------------------------------------------------------
# state() tests
# ---------------------------------------------------------------------------

class TestState:
    def test_returns_environment_state(self, env):
        result = env.state()
        assert isinstance(result, EnvironmentState)

    def test_state_before_reset_raises(self):
        env = EmailTriageEnv(task_id="easy")
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_reflects_actions(self, easy_env):
        obs = easy_env.reset()
        email_id = obs.inbox[0].email.id
        easy_env.step(Action(
            action_type=ActionType.ASSIGN_PRIORITY,
            email_id=email_id,
            priority=Priority.HIGH,
        ))
        state = easy_env.state()
        assert state.step_count == 1
        assert state.inbox[0].assigned_priority == Priority.HIGH


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------

class TestReward:
    def test_correct_priority_gives_positive_reward(self, easy_env):
        """e001 is urgent — assigning urgent should give positive reward."""
        easy_env.reset()
        result = easy_env.step(Action(
            action_type=ActionType.ASSIGN_PRIORITY,
            email_id="e001",
            priority=Priority.URGENT,
        ))
        assert result.reward > 0

    def test_wrong_priority_gives_less_reward_than_correct(self, easy_env):
        """Marking an urgent email as low should score less than marking it urgent."""
        easy_env.reset()
        result_wrong = easy_env.step(Action(
            action_type=ActionType.ASSIGN_PRIORITY,
            email_id="e001",
            priority=Priority.LOW,
        ))
        easy_env.reset()
        result_correct = easy_env.step(Action(
            action_type=ActionType.ASSIGN_PRIORITY,
            email_id="e001",
            priority=Priority.URGENT,
        ))
        assert result_correct.reward > result_wrong.reward

    def test_reward_is_dense_not_sparse(self, easy_env):
        """Agent should get reward on intermediate steps, not just at episode end."""
        easy_env.reset()
        rewards = []
        for email_id in ["e001", "e002", "e003"]:
            r = easy_env.step(Action(
                action_type=ActionType.ASSIGN_PRIORITY,
                email_id=email_id,
                priority=Priority.NORMAL,
            ))
            rewards.append(r.reward)
        # At least some intermediate rewards should be non-trivially non-zero
        nonzero = [r for r in rewards if abs(r) > 0.001]
        assert len(nonzero) > 0

    def test_final_score_in_range(self, easy_env):
        """End-of-episode score must be in [0.0, 1.0]."""
        easy_env.reset()
        result = None
        # Perfect triage of easy inbox
        actions_easy = [
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e001", priority=Priority.URGENT),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e001", category=Category.BUG_REPORT),
            Action(action_type=ActionType.ESCALATE, email_id="e001"),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e002", priority=Priority.NORMAL),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e002", category=Category.BILLING),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e003", priority=Priority.SPAM),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e003", category=Category.SPAM),
            Action(action_type=ActionType.DELETE, email_id="e003"),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e004", priority=Priority.LOW),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e004", category=Category.FEATURE_REQUEST),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e005", priority=Priority.LOW),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e005", category=Category.INTERNAL),
        ]
        for a in actions_easy:
            result = easy_env.step(a)
        score = result.info.get("score", 0.0)
        assert 0.0 <= score <= 1.0

    def test_perfect_easy_score_above_baseline(self, easy_env):
        """An oracle agent should significantly outperform random."""
        easy_env.reset()
        actions_easy = [
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e001", priority=Priority.URGENT),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e001", category=Category.BUG_REPORT),
            Action(action_type=ActionType.ESCALATE, email_id="e001"),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e002", priority=Priority.NORMAL),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e002", category=Category.BILLING),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e003", priority=Priority.SPAM),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e003", category=Category.SPAM),
            Action(action_type=ActionType.DELETE, email_id="e003"),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e004", priority=Priority.LOW),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e004", category=Category.FEATURE_REQUEST),
            Action(action_type=ActionType.ASSIGN_PRIORITY, email_id="e005", priority=Priority.LOW),
            Action(action_type=ActionType.ASSIGN_CATEGORY, email_id="e005", category=Category.INTERNAL),
        ]
        for a in actions_easy:
            result = easy_env.step(a)
        score = result.info.get("score", 0.0)
        assert score >= 0.8, f"Oracle easy score should be ≥ 0.8, got {score}"


# ---------------------------------------------------------------------------
# Invalid task_id test
# ---------------------------------------------------------------------------

def test_invalid_task_id_raises():
    with pytest.raises(ValueError):
        EmailTriageEnv(task_id="impossible")
