"""
server/app.py — Entry point for multi-mode deployment.
Imports and re-exports the FastAPI app and main() from the root server.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import app, main  # noqa: F401

__all__ = ["app", "main"]
