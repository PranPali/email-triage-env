import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app  # noqa: F401


def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
