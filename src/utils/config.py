"""
Configuration Loader for AdaptiveRAG
=====================================
Helper for loading and accessing ``configs/config.yaml``.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


# Default path to the project configuration file
_DEFAULT_CONFIG_PATH: str = str(
    Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
)


def load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Absolute or relative path to the YAML file.
            Defaults to ``configs/config.yaml`` at the project root.

    Returns:
        A dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)

    return config
