#!/usr/bin/env python3
"""
Run the Step 4 FastAPI model API locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import uvicorn

    host = os.getenv("MODEL_API_HOST", "0.0.0.0")
    port = int(os.getenv("MODEL_API_PORT", "8000"))
    uvicorn.run("src.inference.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

