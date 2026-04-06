"""
Hugging Face Spaces entrypoint.
Launches the FastAPI server when deployed as a HF Space.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "server:app",
            "--host", "0.0.0.0",
            "--port", "7860",   # HF Spaces uses port 7860
            "--workers", "1",
        ],
        check=True,
    )
