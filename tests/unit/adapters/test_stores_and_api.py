"""
Session persistence, and the guard on the synchronous entry point.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

import crux.adapters.store.jsonfile as xjsonst
import crux.adapters.store.memory as xmemsto
import crux.api as capi
import crux.domain.decisions as cdecis
import crux.domain.session as csessn
import crux.errors as cerrors
import tests.support.reasoner as fakes


def _session(session_id: str = "s1") -> csessn.Session:
    """
    Build a session with something in its graph.

    :param session_id: Id to use.
    :return: The session.
    """
    graph = csessn.Session(id=session_id, prompt="p").graph.add(
        cdecis.OpenDecision(
            id="software.scope.build",
            undecided="build scope",
            type="underspecification",
            cost_if_wrong="high",
            reversibility="hard",
            origin=cdecis.Origin(seeded_by="pack"),
        ).resolve("mvp", source="respondent")
    )
    return csessn.Session(id=session_id, prompt="add rate limiting", graph=graph)


class TestJsonFileStore:
    """
    The store a host uses to resume across a process boundary.
    """

    def test_a_session_survives_a_write_and_a_read(self, tmp_path: pathlib.Path) -> None:
        """
        Test the claim yield/resume rests on: the session that comes back off
        disk is the one that went on, graph and all.
        """
        store = xjsonst.JsonFileSessionStore(tmp_path / "sessions")
        original = _session()

        store.save(original)

        assert store.load("s1") == original

    def test_an_unknown_session_is_none_rather_than_an_error(self, tmp_path: pathlib.Path) -> None:
        """
        Test that a host asking for a session it never saved gets nothing back,
        not an exception it has to catch on a normal path.
        """
        assert xjsonst.JsonFileSessionStore(tmp_path).load("never-saved") is None

    def test_saving_twice_keeps_the_later_version(self, tmp_path: pathlib.Path) -> None:
        """
        Test that a resumed round overwrites rather than accumulating. Each round
        saves, and a store that kept the first would resume from stale state.
        """
        store = xjsonst.JsonFileSessionStore(tmp_path)
        store.save(_session())
        later = _session().touched(phase="done")

        store.save(later)

        found = store.load("s1")
        assert found is not None
        assert found.phase == "done"

    def test_the_directory_is_created_on_demand(self, tmp_path: pathlib.Path) -> None:
        """
        Test that pointing at a directory that does not exist yet works, since
        the CLI takes the path from a flag.
        """
        store = xjsonst.JsonFileSessionStore(tmp_path / "a" / "b" / "c")
        store.save(_session())

        assert store.load("s1") is not None


class TestMemoryStore:
    """
    The in-process store.
    """

    def test_it_round_trips_and_misses_cleanly(self) -> None:
        """
        Test both halves at once; there is not much to it, and that is the point.
        """
        store = xmemsto.MemorySessionStore()
        store.save(_session())

        assert store.load("s1") is not None
        assert store.load("other") is None


class TestSynchronousEntryPoint:
    """
    The blocking wrapper, and where it refuses to run.
    """

    def test_it_works_from_ordinary_synchronous_code(self) -> None:
        """
        Test the four-line README case: a script with a human at a terminal
        should not have to know what an event loop is.
        """
        compiled = capi.clarify(
            "add rate limiting to the API",
            reasoner=fakes.FakeReasoner(),
            budget=csessn.Budget(max_questions_total=0),
        )

        assert compiled.task

    async def test_calling_it_inside_a_loop_says_what_to_do_instead(self) -> None:
        """
        Test that a host already inside asyncio gets told to await the async
        version, rather than the bare "asyncio.run() cannot be called from a
        running event loop" that names neither crux nor the fix.
        """
        with pytest.raises(cerrors.ConfigurationError, match="clarify_async"):
            capi.clarify(
                "add rate limiting",
                reasoner=fakes.FakeReasoner(),
                budget=csessn.Budget(max_questions_total=0),
            )

    def test_no_clarifier_means_headless_rather_than_an_error(self) -> None:
        """
        Test the invariant from CLAUDE.md: a host with nobody to ask still gets a
        compiled prompt. Raising here would make crux unusable in exactly the
        setting a library is most often embedded in.
        """
        compiled = capi.clarify(
            "add rate limiting to the API",
            reasoner=fakes.FakeReasoner(),
            clarifier=None,
        )

        assert compiled.task
        assert asyncio.get_event_loop_policy() is not None
