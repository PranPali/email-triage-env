---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Email Triage OpenEnv

> **Real-world AI agent benchmark**: Can your agent manage a corporate inbox under pressure?

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-blue)](https://huggingface.co/spaces/openenv/email-triage-v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**Email Triage** is an [OpenEnv](https://openenv.dev)-compliant reinforcement learning environment that simulates the daily task of managing a corporate email inbox. This is a task humans perform constantly â€” support agents, operations managers, executives â€” making it an ideal benchmark for evaluating AI agents on real-world knowledge work.

An agent interacts with a structured inbox and must:

- **Prioritise** emails correctly (urgent â†’ high â†’ normal â†’ low â†’ spam)
- **Categorise** emails by type (billing, bug report, customer support, etc.)
- **Draft professional replies** to emails that require a response
- **Escalate** critical issues to engineering, management, or legal
- **Delete** spam without acting on it

The environment provides **dense reward signals** â€” every good action is rewarded immediately, every mistake penalised â€” making it suitable for both supervised fine-tuning and reinforcement learning.

---

## Motivation

Most existing agent benchmarks focus on games (Atari, MuJoCo) or toy tasks. Real productivity work â€” triaging hundreds of emails, routing issues, drafting contextual replies â€” is where AI assistants are being deployed today. This environment:

- Fills a gap in the OpenEnv ecosystem for **knowledge-work** tasks
- Provides **graded difficulty** from straightforward labelling (easy) to subtle legal/security judgments (hard)
- Rewards **nuanced reasoning**, not just pattern matching
- Is **cheap to run** â€” no UI, no browser, pure structured text

---

## Environment Description

### Inbox Contents

Three task difficulties, each with an increasingly complex inbox:

| Task   | Emails | Max Steps | Baseline Score |
|--------|--------|-----------|----------------|
| easy   | 5      | 20        | ~0.62          |
| medium | 8      | 35        | ~0.48          |
| hard   | 12     | 55        | ~0.35          |

### Email Types Represented

- Production outage reports
- Security vulnerability disclosures
- GDPR/legal compliance demands
- Enterprise sales inquiries
- Billing disputes and refund requests
- Bug reports from paying customers
- Internal operations communications
- Feature requests
- Spam / phishing attempts
- Press/media inquiries

---

## Action Space

Actions are typed Pydantic objects (`models.Action`):

| `action_type`     | Required fields                | Description                                      |
|-------------------|-------------------------------|--------------------------------------------------|
| `assign_priority` | `email_id`, `priority`        | Set email priority level                        |
| `assign_category` | `email_id`, `category`        | Set email category                              |
| `draft_reply`     | `email_id`, `reply_text`      | Draft a reply to the email                      |
| `mark_resolved`   | `email_id`                    | Mark email as handled                           |
| `escalate`        | `email_id`                    | Flag for engineering / management escalation   |
| `delete`          | `email_id`                    | Delete (spam / no-action emails)                |
| `noop`            | â€”                             | Do nothing (costs a step)                       |

**Priority values:** `urgent` Â· `high` Â· `normal` Â· `low` Â· `spam`

**Category values:** `customer_support` Â· `billing` Â· `bug_report` Â· `feature_request` Â· `internal` Â· `sales` Â· `spam` Â· `other`

---

## Observation Space

Observations are typed Pydantic objects (`models.Observation`):

```python
class Observation(BaseModel):
    inbox: List[EmailState]        # Full inbox with current assignments
    goal: str                       # Natural-language task description
    step_count: int                 # Steps taken so far
    max_steps: int                  # Episode budget
    cumulative_reward: float        # Total reward accumulated
    last_action_error: str          # Error message (empty = no error)
    stats: Dict[str, int]           # {total, resolved, deleted, escalated, ...}
```

Each `EmailState` contains the `Email` object plus current assignments:

```python
class EmailState(BaseModel):
    email: Email                       # id, subject, sender, body, metadata
    assigned_priority: Optional[Priority]
    assigned_category: Optional[Category]
    draft_reply: Optional[str]
    is_resolved: bool
    is_escalated: bool
    is_deleted: bool
    actions_taken: List[str]
```

---

## Reward Function

The reward is **dense** â€” agents receive feedback on every step:

```
reward_t = Î”(grader_score) âˆ’ 0.001 (step cost)
```

- **Positive reward**: when an action improves the grader score (e.g., correctly assigning `urgent` to a production outage)
- **Negative reward**: when an action worsens the score (e.g., marking a security email as `spam`)
- **Step penalty**: `âˆ’0.001` per step to encourage efficiency
- **Invalid action penalty**: `âˆ’0.01` for malformed actions

### Grader Scoring

Each grader computes a score in `[0.0, 1.0]`:

**Easy grader** (per email): `0.5 Ã— priority_accuracy + 0.5 Ã— category_accuracy`
- Bonus `+0.2` for deleting spam
- Bonus `+0.2` for escalating urgent emails

**Medium grader** (per email): `0.30 Ã— priority + 0.30 Ã— category + 0.25 Ã— reply_quality + 0.15 Ã— escalation_correctness`

**Hard grader** (per email): `0.35 Ã— priority (strict) + 0.25 Ã— category + 0.25 Ã— reply_quality (strict) + 0.15 base âˆ’ 0.20 missed escalation penalty`

---

## Task Descriptions

### Task 1 â€” Easy (5 emails)

**Goal:** Triage a small inbox with clear, unambiguous signals.

- Spam is obviously spam (`.xyz` domains, prize claims)
- The urgent email is clearly a production outage
- Categories map directly to content

**Expected agent behaviour:** Recognise patterns, assign all 5 emails correctly in â‰¤15 steps.

---

### Task 2 â€” Medium (8 emails)

**Goal:** Triage a mixed inbox and draft replies.

- Some ambiguity: a partnership email could be `sales` or `other`
- Billing disputes require a reply
- Bug reports from enterprise customers need escalation
- Spam is still present but subtler

**Expected agent behaviour:** Draft professional 25+ word replies to emails needing responses; correctly escalate enterprise bugs.

---

### Task 3 â€” Hard (12 emails)

**Goal:** Manage a high-pressure inbox with critical items.

Includes:
- **Security vulnerability** (SQL injection disclosure) â†’ must be `urgent` + `bug_report` + escalated
- **GDPR legal demand** â†’ `urgent` + escalated
- **Account compromise** â†’ `urgent` + escalated
- **Press/media inquiry during outage** â†’ `urgent` + escalated
- **Revenue-critical sales lead** (500 seats, end of quarter) â†’ `high`
- Automated monitoring alert (high error rate) â†’ `high` + escalated
- Routine items that should **not** be over-escalated

**Expected agent behaviour:** Demonstrate contextual reasoning â€” a GDPR email from a `.eu` domain with "Article 46" references is not `normal`. An internal post-mortem is not `urgent`.

---

## Setup & Usage

### Local

```bash
git clone https://huggingface.co/spaces/openenv/email-triage-v1
cd email-triage-v1

pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start the API server
uvicorn server:app --host 0.0.0.0 --port 8000

# Run baseline inference (requires HF_TOKEN and MODEL_NAME)
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py --task all
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env

# With inference
docker run -p 8000:8000 \
  -e HF_TOKEN=hf_your_token \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  email-triage-env
```

---

## HTTP API

The environment exposes a REST API:

```
POST /reset              { "task_id": "easy" }         â†’ Observation
POST /step               { "task_id": "easy", "action": {...} }  â†’ StepResult
GET  /state?task_id=easy                                 â†’ EnvironmentState
GET  /tasks                                              â†’ task list
GET  /health                                             â†’ {"status": "ok"}
GET  /openenv.yaml                                       â†’ spec
```

### Example interaction

```python
import requests

# Reset
obs = requests.post("http://localhost:8000/reset", json={"task_id": "easy"}).json()

# Step: assign priority
result = requests.post("http://localhost:8000/step", json={
    "task_id": "easy",
    "action": {
        "action_type": "assign_priority",
        "email_id": "e001",
        "priority": "urgent"
    }
}).json()

print(result["reward"])   # ~0.15
print(result["done"])     # False
```

### Python direct

```python
from env import EmailTriageEnv
from models import Action, ActionType, Priority

env = EmailTriageEnv(task_id="easy")
obs = env.reset()

result = env.step(Action(
    action_type=ActionType.ASSIGN_PRIORITY,
    email_id="e001",
    priority=Priority.URGENT,
))
print(result.reward, result.done)
```

---

## Baseline Scores

Scores from `inference.py` using `meta-llama/Llama-3.3-70B-Instruct` via HF Inference Router (temperature=0.1):

| Task   | Baseline Score | Oracle Score |
|--------|---------------|--------------|
| easy   | 0.62          | â‰¥ 0.95       |
| medium | 0.48          | â‰¥ 0.88       |
| hard   | 0.35          | â‰¥ 0.80       |

Scores are reproducible across runs with `temperature=0.1`.

---

## OpenEnv Validation

```bash
# Validate spec
python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"

# Run full test suite
python -m pytest tests/ -v --tb=short
```

---

## File Structure

```
email-triage-v1/
â”œâ”€â”€ env.py              # Core EmailTriageEnv (reset/step/state)
â”œâ”€â”€ models.py           # Pydantic typed models (Observation, Action, etc.)
â”œâ”€â”€ server.py           # FastAPI HTTP server
â”œâ”€â”€ inference.py        # Baseline inference script
â”œâ”€â”€ openenv.yaml        # OpenEnv spec
â”œâ”€â”€ app.py              # HF Spaces entrypoint
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ data/
â”‚   â””â”€â”€ emails.py       # Email datasets (easy/medium/hard)
â”œâ”€â”€ tasks/
â”‚   â””â”€â”€ graders.py      # Task definitions + grader functions
â””â”€â”€ tests/
    â””â”€â”€ test_env.py     # Pytest test suite (40+ tests)
```

---

## License

MIT

