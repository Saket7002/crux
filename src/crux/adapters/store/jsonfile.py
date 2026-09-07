"""
A session store backed by one JSON file per session.

Import as:

import crux.adapters.store.jsonfile as xjsonst
"""

from __future__ import annotations

import pathlib

import crux.domain.session as csessn


class JsonFileSessionStore:
    """
    Writes each session to ``<root>/<id>.json``.
    """

    def __init__(self, root: pathlib.Path) -> None:
        """
        :param root: Directory to write into. Created if it does not exist.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, session: csessn.Session) -> None:
        """
        :param session: The session to keep.
        """
        self._path(session.id).write_text(session.model_dump_json(indent=2), encoding="utf-8")

    def load(self, session_id: str) -> csessn.Session | None:
        """
        :param session_id: The session to find.
        :return: The session, or ``None`` when the file is absent.
        """
        path = self._path(session_id)
        if not path.is_file():
            return None
        return csessn.Session.model_validate_json(path.read_text(encoding="utf-8"))

    def _path(self, session_id: str) -> pathlib.Path:
        """
        :param session_id: The session to locate.
        :return: Where it lives on disk.
        """
        return self._root / f"{session_id}.json"
