"""
Which model serves which operation, and where that answer comes from.

Two subjects that are easy to get subtly wrong and hard to notice: the tier
table, and a precedence order that deliberately inverts the usual convention.
"""

from __future__ import annotations

import pathlib

import pytest

import crux.adapters.llm.routing as xroute
import crux.errors as cerrors
import crux.infra.settings as isettn
import crux.ports.reasoner as preason

ALL_OPERATIONS: tuple[preason.Operation, ...] = (
    "expand",
    "adjudicate",
    "phrase",
    "classify",
    "answer_counter",
    "draft",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Remove crux variables so a developer's own shell cannot change the result.

    :param monkeypatch: pytest's patcher.
    """
    for name in list(dict(__import__("os").environ)):
        if name.startswith("CRUX_"):
            monkeypatch.delenv(name, raising=False)


def _toml(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    """
    Write a crux.toml.

    :param tmp_path: pytest's per-test directory.
    :param body: The file contents.
    :return: The path.
    """
    path = tmp_path / "crux.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestTiers:
    """
    The grouping, and what happens when nobody configures one.
    """

    def test_one_model_still_serves_everything(self) -> None:
        """
        Test that the default is exactly today's behaviour. Anyone who never
        learns the tiers must be unaffected by their existence.
        """
        routing = xroute.ModelRouting.of(isettn.CruxSettings(model="one/model"))

        assert {routing.model_for(op) for op in ALL_OPERATIONS} == {"one/model"}
        assert routing.is_uniform

    def test_the_weak_tier_takes_the_mechanical_work(self) -> None:
        """
        Test the split that is the point of the feature: open-ended judgement on
        the strong model, bounded transformation on the cheap one.
        """
        routing = xroute.ModelRouting.of(
            isettn.CruxSettings(model="big/one", weak_model="small/one")
        )

        assert routing.model_for("expand") == "big/one"
        assert routing.model_for("adjudicate") == "big/one"
        for operation in ("phrase", "classify", "answer_counter", "draft"):
            assert routing.model_for(operation) == "small/one"

    def test_draft_is_mechanical(self) -> None:
        """
        Test the placement worth defending. The compiled prompt's substance comes
        from the decision records and its assumptions block is derived by code;
        the model writes only the task sentence, which a cheap model does fine.
        """
        assert "draft" in xroute.MECHANICAL
        assert "draft" not in xroute.REASONING

    def test_every_operation_resolves(self) -> None:
        """
        Test that the table is total. An operation with no model would fail at
        call time, in one of six places, on somebody else's machine.
        """
        routing = xroute.ModelRouting.of(isettn.CruxSettings(model="m"))
        assert all(routing.model_for(op) for op in ALL_OPERATIONS)

    def test_a_per_operation_override_beats_its_tier(self) -> None:
        """
        Test the escape hatch, and that it moves only the operation named.
        """
        routing = xroute.ModelRouting.of(
            isettn.CruxSettings(model="big/one", weak_model="small/one", model_classify="tiny/one")
        )

        assert routing.model_for("classify") == "tiny/one"
        assert routing.model_for("phrase") == "small/one"
        assert routing.model_for("expand") == "big/one"


class TestFallbackChain:
    """
    The ordered list of models to try.
    """

    def test_a_comma_separated_list_parses(self) -> None:
        """
        Test the shape anyone would actually type.

        pydantic-settings JSON-decodes complex fields inside the source, before
        any validator runs, so without NoDecode this raises a SettingsError
        rather than parsing. It is the first thing that would break on day one.
        """
        settings = isettn.CruxSettings(model="a", fallback_models="b, c ,d")

        assert settings.fallback_models == ("b", "c", "d")

    def test_the_chain_starts_with_the_primary_and_keeps_order(self) -> None:
        """
        Test that fallbacks are tried after the chosen model, not instead of it.
        """
        settings = isettn.CruxSettings(model="primary", fallback_models=("b", "c"))

        assert settings.chain_for("primary") == ("primary", "b", "c")

    def test_a_repeated_model_is_not_tried_twice(self) -> None:
        """
        Test that a chain listing the primary again does not waste an attempt and
        a retry budget re-running something that just failed.
        """
        settings = isettn.CruxSettings(model="a", fallback_models=("b", "a", "c"))

        assert settings.chain_for("a") == ("a", "b", "c")

    def test_a_routed_model_leads_its_own_chain(self) -> None:
        """
        Test that the fallbacks apply to whichever model routing picked, not only
        to the configured default.
        """
        settings = isettn.CruxSettings(model="default", fallback_models=("b",))

        assert settings.chain_for("weak/one") == ("weak/one", "b")


class TestPerModelOptions:
    """
    A free tier and a paid one want different numbers.
    """

    def test_per_model_options_override_the_globals(self, tmp_path: pathlib.Path) -> None:
        """
        Test that one model can carry its own ceiling. Groq's free tier counts
        reserved tokens, so it needs a small one; the same value would truncate a
        frontier model mid-answer.
        """
        path = _toml(
            tmp_path,
            '[tiers]\nreasoning="big/one"\n'
            "[limits]\nmax_tokens=8192\n"
            '[models."small/one"]\nmax_tokens=3500\nreasoning_effort="low"\n',
        )
        settings = isettn.CruxSettings(config_file=path)

        assert settings.spec_for("big/one").max_tokens == 8192
        assert settings.spec_for("small/one").max_tokens == 3500
        assert settings.spec_for("small/one").reasoning_effort == "low"

    def test_a_global_key_is_withheld_from_a_chain(self) -> None:
        """
        Test the credential trap. A global api_key belongs to one provider, so
        applying it to a fallback on another produces an authentication failure
        that looks like a broken key rather than a misconfiguration.
        """
        settings = isettn.CruxSettings(model="a", api_key="sk-for-provider-a")

        assert settings.spec_for("a").api_key is not None
        assert settings.spec_for("b", drop_global_key=True).api_key is None


class TestPrecedence:
    """
    Flag beats file beats environment beats default.
    """

    def test_the_file_beats_the_environment(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test the inversion of the usual convention, which is deliberate: a repo
        that checks a file in wants every contributor on the same models
        regardless of their shell.
        """
        monkeypatch.setenv("CRUX_MODEL", "env/model")
        path = _toml(tmp_path, '[tiers]\nreasoning="file/model"\n')

        assert isettn.CruxSettings(config_file=path).model == "file/model"

    def test_a_flag_beats_the_file(self, tmp_path: pathlib.Path) -> None:
        """
        Test that an explicit argument still wins, so a one-off run does not
        require editing a committed file.
        """
        path = _toml(tmp_path, '[tiers]\nreasoning="file/model"\n')

        assert isettn.CruxSettings(config_file=path, model="flag/model").model == "flag/model"

    def test_the_environment_wins_when_there_is_no_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that nothing changes for anyone who never writes a crux.toml.
        """
        monkeypatch.setenv("CRUX_MODEL", "env/model")

        assert isettn.CruxSettings().model == "env/model"

    def test_a_missing_file_is_not_an_error(self, tmp_path: pathlib.Path) -> None:
        """
        Test that pointing at a file that is not there falls back to the
        environment rather than raising, since the CLI passes a path it found by
        searching and the search can come up empty.
        """
        assert isettn.CruxSettings(config_file=tmp_path / "nope.toml").model

    def test_sections_and_operations_flatten(self, tmp_path: pathlib.Path) -> None:
        """
        Test that the human-facing nesting reaches the flat fields. Making the
        settings class itself nested would have renamed every environment
        variable to CRUX_TIERS__REASONING and broken every existing .env.
        """
        path = _toml(
            tmp_path,
            '[tiers]\nreasoning="r"\nmechanical="m"\n'
            '[operations]\nexpand="e"\nclassify="c"\n'
            '[fallback]\nmodels=["f1","f2"]\nretries=2\n'
            "[limits]\nmax_passes=7\nmax_tokens=1234\n",
        )
        settings = isettn.CruxSettings(config_file=path)

        assert (settings.model, settings.weak_model) == ("r", "m")
        assert (settings.model_expand, settings.model_classify) == ("e", "c")
        assert settings.fallback_models == ("f1", "f2")
        assert (settings.fallback_retries, settings.max_passes) == (2, 7)
        assert settings.max_tokens == 1234


class TestSecretsStayOutOfTheFile:
    """
    The one thing that must not follow the file-beats-environment rule.
    """

    def test_a_key_in_the_file_is_refused(self, tmp_path: pathlib.Path) -> None:
        """
        Test that a committed secret cannot beat an operator's exported one.

        This is what makes the inverted precedence safe to ship. Without it,
        checking a crux.toml into a repo would silently override the key every
        contributor set in their own shell.
        """
        path = _toml(tmp_path, 'api_key="leaked-into-source-control"\n')

        with pytest.raises(cerrors.ConfigurationError, match="must not set"):
            isettn.CruxSettings(config_file=path)


class TestDiscovery:
    """
    Finding a crux.toml, and refusing to do it on a library's behalf.
    """

    def test_it_walks_up_from_a_directory(self, tmp_path: pathlib.Path) -> None:
        """
        Test that running from a subdirectory still finds the repo's file.
        """
        (tmp_path / "crux.toml").write_text('[tiers]\nreasoning="found"\n')
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)

        assert isettn.find_config_file(nested) == tmp_path / "crux.toml"

    def test_no_file_above_is_none_not_an_error(self, tmp_path: pathlib.Path) -> None:
        """
        Test the ordinary case of a project that has no crux.toml.
        """
        assert isettn.find_config_file(tmp_path) is None

    def test_settings_never_search_on_their_own(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that a library embedding crux does not pick up whatever repo it
        happens to be running inside.

        Same category as not mutating a host's environment: a host that wants
        file configuration passes the path, and crux never goes looking on its
        behalf.
        """
        (tmp_path / "crux.toml").write_text('[tiers]\nreasoning="ambient/model"\n')
        monkeypatch.chdir(tmp_path)

        assert isettn.CruxSettings().model != "ambient/model"
