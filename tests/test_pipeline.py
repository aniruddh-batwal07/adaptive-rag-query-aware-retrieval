"""
Smoke Tests — AdaptiveRAG
==========================
Minimal pytest test suite to verify the project skeleton is intact.
"""

import sys
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def test_project_skeleton_loads() -> None:
    """Verify that the project skeleton is importable and the config loads."""
    from src.utils.config import load_config

    config = load_config()

    assert config is not None, "Configuration should not be None."
    assert config["project"]["name"] == "AdaptiveRAG"
