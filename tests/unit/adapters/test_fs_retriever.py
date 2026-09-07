"""
Tier-4 tests: the shipped retriever, against a real tree.

A real directory rather than a stub, because the thing under test is precisely
the file walking, the skipping and the scoring — a mock of a filesystem would
test nothing.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import crux.adapters.retrieval.fs as xfsretr
import crux.domain.evidence as cevid


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Build a small project tree.

    :param tmp_path: pytest's per-test directory.
    :return: The project root.
    """
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "cache.py").write_text(
        "import redis\n\nclient = redis.Redis()\n", encoding="utf-8"
    )
    (tmp_path / "app" / "routes.py").write_text("router = APIRouter()\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = ["redis>=5.0"]\n', encoding="utf-8"
    )
    (tmp_path / "secrets.txt").write_text("redis password hunter2\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("secrets.txt\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("redis redis redis\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n")
    return tmp_path


def _brief(query: str, *, max_items: int = 10) -> cevid.Brief:
    """
    Build a brief.

    :param query: What to look for.
    :param max_items: Cap on results.
    :return: The brief.
    """
    return cevid.Brief(
        id="brief:test",
        decision_id="test",
        query=query,
        accept_if="a store is named",
        max_items=max_items,
    )


class TestFilesystemRetriever:
    """
    What it finds, and what it refuses to look at.
    """

    async def test_it_finds_the_file_that_mentions_the_term(self, tree: pathlib.Path) -> None:
        """
        Test the basic claim: a brief about caching surfaces the caching module,
        with a locator a reader can open and an excerpt they can check.
        """
        result = (await xfsretr.FilesystemRetriever(tree).retrieve([_brief("redis cache")]))[0]

        locators = [item.locator for item in result.items]
        assert "app/cache.py" in locators
        assert any(item.excerpt for item in result.items)

    async def test_it_skips_vendored_directories(self, tree: pathlib.Path) -> None:
        """
        Test that node_modules is never read. It would otherwise dominate every
        result set on term frequency alone, and none of it is the user's code.
        """
        result = (await xfsretr.FilesystemRetriever(tree).retrieve([_brief("redis")]))[0]

        assert all("node_modules" not in item.locator for item in result.items)

    async def test_it_honours_gitignore(self, tree: pathlib.Path) -> None:
        """
        Test that an ignored file stays unread. Ignored files are where secrets
        live, and an excerpt of one would reach the model and the compiled
        prompt.
        """
        result = (await xfsretr.FilesystemRetriever(tree).retrieve([_brief("redis password")]))[0]

        assert all("secrets.txt" not in item.locator for item in result.items)

    async def test_it_ignores_binaries(self, tree: pathlib.Path) -> None:
        """
        Test that non-text files are not offered as evidence.
        """
        result = (await xfsretr.FilesystemRetriever(tree).retrieve([_brief("logo")]))[0]

        assert all(not item.locator.endswith(".png") for item in result.items)

    async def test_it_surfaces_the_manifest_for_a_dependency_question(
        self, tree: pathlib.Path
    ) -> None:
        """
        Test that "what does this project already depend on" reaches
        pyproject.toml. A manifest answers that question whether or not it
        happens to contain the query's words, which is why landmarks score
        without a term hit.
        """
        result = (
            await xfsretr.FilesystemRetriever(tree).retrieve(
                [_brief("which third party dependencies exist")]
            )
        )[0]

        assert "pyproject.toml" in [item.locator for item in result.items]

    async def test_it_honours_the_item_cap_and_reports_exhaustion(self, tree: pathlib.Path) -> None:
        """
        Test that a brief's depth budget is respected. Depth is how crux keeps a
        cheap decision from paying for a deep crawl.
        """
        result = (await xfsretr.FilesystemRetriever(tree).retrieve([_brief("redis", max_items=1)]))[
            0
        ]

        assert len(result.items) == 1
        assert result.exhausted is False

    async def test_the_whole_batch_is_answered(self, tree: pathlib.Path) -> None:
        """
        Test that every brief comes back with a result, so the caller need not
        special-case a missing one.
        """
        briefs = [
            _brief("redis"),
            cevid.Brief(
                id="brief:other", decision_id="other", query="authentication", accept_if="x"
            ),
        ]

        results = await xfsretr.FilesystemRetriever(tree).retrieve(briefs)

        assert {r.brief_id for r in results} == {"brief:test", "brief:other"}

    async def test_an_empty_result_is_not_an_error(self, tree: pathlib.Path) -> None:
        """
        Test that finding nothing returns an empty result rather than raising.
        Zero candidates is a legal routing outcome, not a failure.
        """
        result = (await xfsretr.FilesystemRetriever(tree).retrieve([_brief("kubernetes")]))[0]

        assert result.found_nothing
        assert result.error == ""

    async def test_prose_does_not_outrank_the_module_that_implements_it(
        self, tree: pathlib.Path
    ) -> None:
        """
        Test that a README mentioning a thing ranks below the module using it.

        This test exists because it once went the other way: README.md was in the
        landmark list with a bonus of 2.0, so a one-line mention of Redis beat
        app/cache.py, and a compiled prompt cited the README as the source of a
        decision about the cache module. A citation that points at prose instead
        of code is worse than none, because the reader trusts it.
        """
        (tree / "README.md").write_text("This project uses redis for counters.\n", encoding="utf-8")

        result = (await xfsretr.FilesystemRetriever(tree).retrieve([_brief("redis cache")]))[0]

        locators = [item.locator for item in result.items]
        assert locators.index("app/cache.py") < locators.index("README.md")


class TestDotenvLoading:
    """
    Where credentials come from, and the trap this avoids.
    """

    def test_a_provider_key_in_dotenv_reaches_the_environment(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that ANTHROPIC_API_KEY in a .env file actually reaches litellm.

        This test exists because it once did not. pydantic-settings reads .env
        but filters by the CRUX_ prefix, and litellm reads the real environment,
        so an unprefixed provider key was silently ignored — while the CLI
        reported an authentication failure that pointed at the environment the
        user had, as far as they could tell, just configured.
        """
        import crux.infra.settings as isettn

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text('ANTHROPIC_API_KEY="sk-ant-test"\nCRUX_MODEL=claude-opus-5\n')

        loaded = isettn.load_dotenv(env)

        assert "ANTHROPIC_API_KEY" in loaded
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"

    def test_a_real_export_beats_the_file(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that a shell export is treated as a deliberate override. Otherwise a
        stale .env would quietly win over the key someone just exported to test
        something, which is maddening to debug.
        """
        import crux.infra.settings as isettn

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=sk-from-file\n")

        isettn.load_dotenv(env)

        assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-shell"

    def test_a_missing_dotenv_is_not_an_error(self, tmp_path: pathlib.Path) -> None:
        """
        Test that running without a .env file works, since exporting variables is
        the more common setup.
        """
        import crux.infra.settings as isettn

        assert isettn.load_dotenv(tmp_path / "nope.env") == ()


class TestRunTrace:
    """
    What a run leaves behind for someone asking "why did it do that?".
    """

    def test_a_trace_records_the_run_a_person_can_read(self, tmp_path: pathlib.Path) -> None:
        """
        Test that the narrative carries the numbers you would otherwise have to
        deserialise a session to find: the expansion curve, edge counts, and what
        closed each decision.
        """
        import crux.domain.decisions as cdecis
        import crux.domain.session as csessn
        import crux.infra.evidence as ievid

        graph = csessn.Session(id="s", prompt="p").graph.add(
            cdecis.OpenDecision(
                id="software.scope.build",
                undecided="build scope",
                type="underspecification",
                cost_if_wrong="high",
                reversibility="hard",
                origin=cdecis.Origin(seeded_by="pack"),
            ).resolve("mvp", source="respondent")
        )
        session = csessn.Session(
            id="abc12345",
            prompt="add rate limiting",
            graph=graph,
            passes=csessn.PassLog(
                records=(csessn.PassRecord(index=1, lens="blind", proposed=7, new_after_dedup=7),)
            ),
        )
        recorder = ievid.RunRecorder(tmp_path, session_id=session.id)
        recorder.record_call(
            operation="propose_decisions",
            model="test/model",
            messages=[{"role": "user", "content": "what is undecided?"}],
            forced_tool="propose_decisions",
            response={"tool_calls": [{"name": "propose_decisions", "arguments": {}}]},
            elapsed_seconds=1.5,
        )

        where = recorder.write(session)

        narrative = (where / "run.md").read_text()
        assert "add rate limiting" in narrative
        assert "| 1 | blind | 7 | 7 | 1.00 |" in narrative
        assert "software.scope.build" in narrative
        assert "respondent" in narrative
        assert (where / "session.json").is_file()
        assert (where / "calls" / "01-propose_decisions.json").is_file()

    def test_a_failed_run_still_leaves_its_calls_behind(self, tmp_path: pathlib.Path) -> None:
        """
        Test that a crash is traced too. A failed run is the one you most want a
        record of, and the calls leading up to it are the evidence.
        """
        import crux.domain.session as csessn
        import crux.infra.evidence as ievid

        session = csessn.Session(id="dead", prompt="p")
        recorder = ievid.RunRecorder(tmp_path, session_id=session.id)
        recorder.record_call(
            operation="propose_decisions",
            model="test/model",
            messages=[{"role": "user", "content": "x"}],
            forced_tool="propose_decisions",
            error="rate limited",
        )

        where = recorder.write(session, None, error="model stayed rate limited")

        assert "## Failed" in (where / "run.md").read_text()
        assert (where / "calls" / "01-propose_decisions-failed.json").is_file()
        assert not (where / "compiled.md").exists()
