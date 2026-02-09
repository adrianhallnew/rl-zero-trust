"""YAML configuration parser for the RL-driven adaptive security system.

Loads and validates configuration from YAML files in the config/ directory.
All hyperparameters and settings are externalized to config files.
"""

import os
from typing import Any, Dict

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_name: Name of the config file (with or without .yaml extension).
            Examples: ``"dqn_config"``, ``"network_config.yaml"``

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"

    config_path = os.path.join(CONFIG_DIR, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config: %s", config_name)
    return config


def get_reward_config(algorithm: str = "dqn") -> Dict[str, Any]:
    """Load reward function configuration for a specific algorithm.

    Args:
        algorithm: Either ``"dqn"`` or ``"ppo"``.

    Returns:
        Dictionary containing reward weights and thresholds.
    """
    config = load_config(f"{algorithm}_config")
    return config.get("reward", {})


def get_hyperparameters(algorithm: str = "dqn") -> Dict[str, Any]:
    """Load hyperparameters for a specific algorithm.

    Args:
        algorithm: Either ``"dqn"`` or ``"ppo"``.

    Returns:
        Dictionary containing algorithm hyperparameters.
    """
    config = load_config(f"{algorithm}_config")
    return config.get(algorithm, {})


def get_network_config() -> Dict[str, Any]:
    """Load network topology and traffic configuration.

    Returns:
        Dictionary containing topology, link, and traffic parameters.
    """
    return load_config("network_config")
