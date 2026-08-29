#!/usr/bin/env python3
"""
Run the Step 5 A/B router locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import uvicorn

    host = os.getenv("AB_ROUTER_HOST", "0.0.0.0")
    port = int(os.getenv("AB_ROUTER_PORT", "8080"))
    uvicorn.run("src.router.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

