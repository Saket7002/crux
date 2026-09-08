"""
Logger construction, in one place.

Import as:

import crux.infra.logging as ilog
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    :param name: Usually ``__name__``.
    :return: The module's logger.
    """
    return logging.getLogger(name)


def configure(*, verbose: bool = False) -> None:
    """
    Set up logging for a CLI run.

    A library never configures logging for its host; this is called only from
    :mod:`crux.cli.main`.

    :param verbose: Whether to show crux's own info-level lines.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # litellm is extremely chatty at info level and none of it is ours.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
