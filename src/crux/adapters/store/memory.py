"""
An in-process session store.

Import as:

import crux.adapters.store.memory as xmemsto
"""

from __future__ import annotations

import crux.domain.session as csessn


class MemorySessionStore:
    """
    Keeps sessions in a dict for the life of the process.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, csessn.Session] = {}

    def save(self, session: csessn.Session) -> None:
        """
        :param session: The session to keep.
        """
        self._sessions[session.id] = session

    def load(self, session_id: str) -> csessn.Session | None:
        """
        :param session_id: The session to find.
        :return: The session, or ``None``.
        """
        return self._sessions.get(session_id)
