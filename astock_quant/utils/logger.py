"""Logging helper used by CLI scripts and library modules."""

from __future__ import annotations

import logging


def get_logger(name: str = "astock_quant") -> logging.Logger:
    """Return a configured logger without adding duplicate handlers."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
