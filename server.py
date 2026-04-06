"""
FastAPI HTTP server for the Email Triage OpenEnv.

Endpoints:
    POST /reset          → Observation
    POST /step           → StepResult
    GET  /state          → EnvironmentState
    GET  /health         → {"status": "ok"}
    GET  /openenv.yaml   → spec file

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from env import EmailTriageEnv
from models import Action, ActionType, Category, Priority, Observation, StepResult, EnvironmentState


app = FastAPI(
    title="Email Triage OpenEnv",
    version="1.0.0",
    description="Real-world email triage environment following the OpenEnv spec.",
)

# One environment instance per session (stateful; for multi-agent use,
# clients should manage separate instances via task_id query param).
_envs: Dict[str, EmailTriageEnv] = {}


def _get_env(task_id: str = "easy") -> EmailTriageEnv:
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id)
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Action


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "email-triage-v1"}


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def get_spec() -> str:
    spec_path = Path(__file__).parent / "openenv.yaml"
    return spec_path.read_text()


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    """Reset the environment for the given task and return the initial observation."""
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'. Choose: easy, medium, hard")
    env = _get_env(req.task_id)
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    """Execute one action and return (observation, reward, done, info)."""
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"Unknown task_id '{req.task_id}'")
    env = _get_env(req.task_id)
    result = env.step(req.action)
    return result


@app.get("/state", response_model=EnvironmentState)
def state(task_id: str = "easy") -> EnvironmentState:
    """Return the full internal environment state (for debugging/evaluation)."""
    if task_id not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"Unknown task_id '{task_id}'")
    env = _get_env(task_id)
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": "easy",
                "difficulty": "easy",
                "email_count": 5,
                "max_steps": 20,
                "description": "Triage 5 clearly-labelled emails.",
            },
            {
                "id": "medium",
                "difficulty": "medium",
                "email_count": 8,
                "max_steps": 35,
                "description": "Triage 8 emails — draft replies, escalate issues.",
            },
            {
                "id": "hard",
                "difficulty": "hard",
                "email_count": 12,
                "max_steps": 55,
                "description": "Triage 12 high-pressure emails. Precision required.",
            },
        ]
    }
