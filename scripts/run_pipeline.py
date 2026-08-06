"""
Run Pipeline — AdaptiveRAG
===========================
Convenience script that imports and executes the project entry point.

Usage::

    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.main import main

if __name__ == "__main__":
    main()
