"""
The session-persistence seam.

Optional. Most hosts already have somewhere to put JSON and will never implement
this; it exists so the CLI and the blocking wrapper have somewhere to keep a
session between turns.

Import as:

import crux.ports.store as pstore
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import crux.domain.session as csessn


@runtime_checkable
class SessionStore(Protocol):
    """
    Keeps sessions between a yield and a resume.
    """

    def save(self, session: csessn.Session) -> None:
        """
        Write a session, replacing any earlier version of it.

        :param session: The session to keep.
        """
        ...

    def load(self, session_id: str) -> csessn.Session | None:
        """
        Read a session back.

        :param session_id: The session to find.
        :return: The session, or ``None`` when it was never saved.
        """
        ...
