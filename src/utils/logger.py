"""Centralized logging configuration for the RL-driven adaptive security system.

Provides a consistent logging setup across all modules with configurable
log levels and formatting. Uses Python's built-in logging module.
"""

import logging
import os
import sys
from typing import Optional


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path to write logs to.
    """
    global _initialized
    if _initialized:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured logger instance.
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
