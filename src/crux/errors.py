"""
The exception hierarchy.

Every exception crux raises descends from :class:`CruxError`, so a host can
catch the whole library with one clause.

Import as:

import crux.errors as cerrors
"""

from __future__ import annotations


class CruxError(Exception):
    """
    Base for everything crux raises.
    """


class ConfigurationError(CruxError):
    """
    crux was wired up in a way that cannot work.

    Host-facing, so the message says what to do about it.
    """


class ReasonerError(CruxError):
    """
    The model failed to produce a usable answer.
    """


class ReasonerParseError(ReasonerError):
    """
    The model answered, but the answer did not fit the expected shape.

    Kept distinct from :class:`ReasonerError` because it is the one worth
    retrying: a malformed tool call is often transient, an authentication
    failure never is.
    """


class RetrievalError(CruxError):
    """
    A retriever failed outright.

    A retriever that finds nothing has not failed; it returns an empty
    ``BriefResult`` and the router treats zero candidates as a legal outcome.
    """


class SessionError(CruxError):
    """
    A session was used in a way its state does not allow.
    """


class UnknownQuestionError(SessionError):
    """
    A reply named a question the session never asked.
    """


class UnknownDecisionError(CruxError):
    """
    A decision id was referenced that is not in the graph.
    """
