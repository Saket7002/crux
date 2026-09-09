"""
Tests for the platform adapter, the scorers and the judge.

Not marked ``live``: none of this touches a model or a network. A backend that
mislabels a score, or a judge that reads a prose reply as a pass, would put a
confident wrong number in a dashboard, which is worse than no dashboard.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

import pytest

import crux.adapters.llm.prompts as xprompt
import crux.domain.output as coutput
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.ports.llm as pllm
import tests.evals.compare as compare
import tests.evals.harness as harness
import tests.evals.judge as judge
import tests.evals.models as models
import tests.evals.optimise as optimise
import tests.evals.platform.adapter as adapter
import tests.evals.runner as runner
import tests.evals.scorers as scorers
import tests.evals.systems as systems
import tests.support.llm as support_llm


def _score(**overrides: object) -> harness.CaseScore:
    defaults: dict[str, object] = {
        "case_id": "c",
        "stratum": "feature",
        "recall": 1.0,
        "surfaced": (),
        "missed": (),
        "should_have_been_quiet": (),
        "retrieval_expected_but_not": (),
        "questions_asked": 1,
        "over_question_budget": False,
        "assumptions": 2,
        "decisions_total": 5,
        "llm_calls": 4,
    }
    return harness.CaseScore.model_validate(defaults | overrides)


def _rubric() -> harness.Rubric:
    return harness.Rubric(
        task_should=("add rate limiting",),
        assumptions_should_cite=("how callers are identified",),
        must_not_claim=("that a middleware already exists",),
    )


class TestScorers:
    """
    Turning one case score into named 0..1 scores.
    """

    def test_every_named_score_is_present_and_in_range(self) -> None:
        """
        Test that a platform always sees the same five names, so an experiment
        diff compares like with like even when a case expects nothing.
        """
        case = harness.EvalCase(author="test", id="c", prompt="p")
        scores = scorers.decompose(_score(), case)
        assert set(scores) == {"recall", "quiet", "retrieval", "within_budget", "clean"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_an_empty_expectation_set_scores_perfect_not_undefined(self) -> None:
        """
        Test that a case with no must_not_surface cannot have been noisy. A NaN
        here would poison the experiment mean.
        """
        case = harness.EvalCase(author="test", id="c", prompt="p")
        assert scorers.decompose(_score(), case)["quiet"] == 1.0

    def test_quiet_is_the_fraction_of_expectations_kept(self) -> None:
        """
        Test that one noisy decision out of two expected-quiet ones scores 0.5,
        which is what the ratchet's precision-miss count means per case.
        """
        case = harness.EvalCase(
            author="test",
            id="c",
            prompt="p",
            must_not_surface=(harness.Expectation(id="a"), harness.Expectation(id="b")),
        )
        scores = scorers.decompose(_score(should_have_been_quiet=("a",)), case)
        assert scores["quiet"] == 0.5
        assert scores["clean"] == 0.0

    def test_metrics_are_counts_not_scores(self) -> None:
        """
        Test that the integer metrics stay separate from the 0..1 scores, so a
        platform never averages a question count with a recall.
        """
        assert scorers.metrics(_score()) == {
            "questions_asked": 1,
            "assumptions": 2,
            "decisions_total": 5,
            "llm_calls": 4,
        }


class TestFeedback:
    """
    Saying in words why a case scored what it did.
    """

    def test_one_line_per_finding_in_the_corpus_vocabulary(self) -> None:
        """
        Test that a missed decision, a noisy one, a retrieval miss, a budget
        breach and a failed rubric bullet each become exactly one line a
        reflection model can act on, and nothing else is said.
        """
        case = harness.EvalCase(author="test", id="c", prompt="p", max_questions=2)
        score = _score(
            missed=("~what gets cached",),
            should_have_been_quiet=("software.deps.policy",),
            retrieval_expected_but_not=("software.target.files",),
            questions_asked=3,
            over_question_budget=True,
        )
        judgement = judge.Judgement(
            verdicts=(
                judge.Verdict(section="task_should", bullet="a", met=True),
                judge.Verdict(section="must_not_claim", bullet="b", met=False, evidence="line 4"),
            )
        )
        text = scorers.feedback(score, case, judgement)
        assert text.splitlines() == [
            "missed: ~what gets cached",
            "asked, though the repo answers it: software.deps.policy",
            "expected from retrieval, but not resolved that way: software.target.files",
            "over question budget: asked 3, allowed 2",
            "rubric not met [must_not_claim]: b (line 4)",
        ]

    def test_a_clean_case_says_so_rather_than_nothing(self) -> None:
        """
        Test that silence is never the feedback: an empty string would read as
        a missing field on a dashboard.
        """
        assert scorers.feedback(_score(), harness.EvalCase(author="test", id="c", prompt="p")) == (
            "every expectation met"
        )


class TestJudge:
    """
    The rubric judge, through a scripted model.
    """

    async def test_the_score_is_the_fraction_of_bullets_met(self) -> None:
        """
        Test the arithmetic through the real parser: two of three met is 0.67.
        """
        client = support_llm.ScriptedLlm(
            [
                {
                    "verdicts": [
                        {"section": "task_should", "bullet": "a", "met": True},
                        {"section": "assumptions_should_cite", "bullet": "b", "met": False},
                        {"section": "must_not_claim", "bullet": "c", "met": True},
                    ],
                    "rationale": "two of three",
                }
            ]
        )
        result = await judge.judge(coutput.CompiledPrompt(task="t"), _rubric(), client)
        assert result.score == pytest.approx(2 / 3)
        assert result.rationale == "two of three"
        assert client.forced == [judge.JUDGE_TOOL]

    async def test_a_prose_reply_is_a_parse_error_not_a_pass(self) -> None:
        """
        Test the failure the judge exists to make visible: a model that ignores
        the forced tool must raise, never score 1.0 on an empty verdict list.
        """
        client = support_llm.ScriptedLlm(["Looks fine to me."])
        with pytest.raises(cerrors.ReasonerParseError):
            await judge.judge(coutput.CompiledPrompt(task="t"), _rubric(), client)

    async def test_the_named_model_reaches_the_client(self) -> None:
        """
        Test that --judge-model is honoured, because it is the one mitigation
        for a model grading its own work.
        """
        client = support_llm.ScriptedLlm([{"verdicts": [], "rationale": ""}])
        await judge.judge(coutput.CompiledPrompt(task="t"), _rubric(), client, model="other")
        assert client.models == ["other"]

    def test_every_bullet_and_the_rendered_prompt_reach_the_model(self) -> None:
        """
        Test the prompt carries what the judge needs, asserting on bullet
        presence rather than on the surrounding prose.
        """
        rendered = coutput.CompiledPrompt(task="Add rate limiting to the API").render()
        text = "\n".join(m.content for m in judge.build_messages(rendered, _rubric()))
        for _, bullet in _rubric().bullets():
            assert bullet in text
        assert "Add rate limiting to the API" in text


class TestTracing:
    """
    The client wrapper that turns completions into call records.
    """

    async def test_each_completion_becomes_one_record(self) -> None:
        """
        Test that a replayed call is traced like a live one, keyed by the tool
        that was forced, so a platform span tree matches the evidence trace.
        """
        log = adapter.CallLog()
        traced = adapter.TracingLlm(
            support_llm.ScriptedLlm([{"x": 1}]), log, default_model="m-default"
        )
        await traced.complete((pllm.Message(role="user", content="hi"),), force_tool="expand")
        (record,) = log.drain()
        assert record.operation == "expand"
        assert record.model == "m-default"
        assert record.response["tool_calls"][0]["arguments"] == {"x": 1}
        assert log.drain() == ()

    async def test_a_failure_is_recorded_then_re_raised(self) -> None:
        """
        Test that an error still leaves a record, because the failed call is
        the one worth clicking into.
        """
        log = adapter.CallLog()
        traced = adapter.TracingLlm(support_llm.ScriptedLlm([]), log, default_model="m")
        with pytest.raises(cerrors.ReasonerError):
            await traced.complete((pllm.Message(role="user", content="hi"),), force_tool="x")
        (record,) = log.drain()
        assert record.error
        assert record.operation == "x"


class FakeBackend:
    """
    Records the order it was called in and what it was handed.
    """

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.results: list[adapter.CaseResult] = []
        self.meta: adapter.RunMetadata | None = None
        self.item_ids: dict[str, str] = {}

    def upsert_dataset(self, cases: Sequence[harness.EvalCase]) -> dict[str, str]:
        self.calls.append("upsert_dataset")
        return {case.id: f"item-{case.id}" for case in cases}

    def start_experiment(self, meta: adapter.RunMetadata, item_ids: dict[str, str]) -> None:
        self.calls.append("start_experiment")
        self.meta = meta
        self.item_ids = item_ids

    def log_result(self, result: adapter.CaseResult) -> None:
        self.calls.append("log_result")
        self.results.append(result)

    def finish(self) -> str:
        self.calls.append("finish")
        return "done"


def _meta(**overrides: object) -> adapter.RunMetadata:
    import datetime as dt

    defaults: dict[str, object] = {
        "git_sha": "abc",
        "crux_version": "0.1.0",
        "model": "m",
        "judge_model": None,
        "cassette_mode": "replay",
        "corpus_size": 1,
        "started_at": dt.datetime.now(tz=dt.UTC),
    }
    return adapter.RunMetadata.model_validate(defaults | overrides)


class TestRunExperiment:
    """
    The shared loop every backend is driven by.
    """

    async def _run(
        self, cases: Sequence[harness.EvalCase], judge_client: pllm.LlmClient | None
    ) -> FakeBackend:
        backend = FakeBackend()
        compiled = coutput.CompiledPrompt(task="done")

        async def run_case(case: harness.EvalCase, client: pllm.LlmClient) -> csessn.Session:
            await client.complete(
                (pllm.Message(role="user", content=case.prompt),), force_tool="expand"
            )
            return csessn.Session(id=case.id, prompt=case.prompt, outcome=compiled)

        async def finish_case(
            case: harness.EvalCase, client: pllm.LlmClient, session: csessn.Session
        ) -> coutput.CompiledPrompt | None:
            return session.outcome

        await adapter.run_experiment(
            cases,
            backend,
            support_llm.ScriptedLlm(by_tool={"expand": [{"decisions": []}]}),
            judge_client,
            _meta(corpus_size=len(cases)),
            run_case=run_case,
            finish_case=finish_case,
        )
        return backend

    async def test_backend_calls_happen_in_order_with_ids_flowing_through(self) -> None:
        """
        Test the contract a backend can rely on: dataset before experiment,
        every result before finish, and the item ids it returned handed back.
        """
        cases = (
            harness.EvalCase(author="test", id="a", prompt="p"),
            harness.EvalCase(author="test", id="b", prompt="q"),
        )
        backend = await self._run(cases, judge_client=None)
        assert backend.calls == [
            "upsert_dataset",
            "start_experiment",
            "log_result",
            "log_result",
            "finish",
        ]
        assert backend.item_ids == {"a": "item-a", "b": "item-b"}
        assert backend.meta is not None and backend.meta.cassette_mode == "replay"

    async def test_the_judge_is_skipped_without_a_rubric_or_a_judge_client(self) -> None:
        """
        Test that absence is absence: no rubric means no judge score, rather
        than a silent 1.0 that reads as a pass on a dashboard.
        """
        plain = harness.EvalCase(author="test", id="a", prompt="p")
        backend = await self._run((plain,), judge_client=support_llm.ScriptedLlm([]))
        assert "judge" not in backend.results[0].scores

        with_rubric = harness.EvalCase(author="test", id="b", prompt="p", expected=_rubric())
        backend = await self._run((with_rubric,), judge_client=None)
        assert "judge" not in backend.results[0].scores

    async def test_the_judge_score_and_calls_land_on_the_result(self) -> None:
        """
        Test that a judged case carries its score, its rationale, and the
        traced calls crux made, which is everything a backend logs.
        """
        case = harness.EvalCase(author="test", id="a", prompt="p", expected=_rubric())
        judge_client = support_llm.ScriptedLlm(
            [
                {
                    "verdicts": [{"section": "task_should", "bullet": "x", "met": True}],
                    "rationale": "met",
                }
            ]
        )
        backend = await self._run((case,), judge_client=judge_client)
        result = backend.results[0]
        assert result.scores["judge"] == 1.0
        assert result.judge_rationale == "met"
        assert result.feedback == "every expectation met"
        assert [c.operation for c in result.calls] == ["expand"]
        assert result.rendered.startswith("# Task")


class TestOneFailureDoesNotCostTheRun:
    """
    A provider declining one case leaves the other cases' numbers intact.
    """

    async def test_a_failed_case_is_logged_with_its_error_and_the_loop_continues(self) -> None:
        """
        Test the lesson the saturation experiment paid for: a single
        ReasonerError must become a logged failure, not a lost run.
        """
        backend = FakeBackend()

        async def run_case(case: harness.EvalCase, client: pllm.LlmClient) -> csessn.Session:
            if case.id == "bad":
                raise cerrors.ReasonerError("provider declined")
            return csessn.Session(id=case.id, prompt=case.prompt)

        async def finish_case(
            case: harness.EvalCase, client: pllm.LlmClient, session: csessn.Session
        ) -> coutput.CompiledPrompt | None:
            return None

        cases = (
            harness.EvalCase(author="test", id="bad", prompt="p"),
            harness.EvalCase(author="test", id="good", prompt="p"),
        )
        await adapter.run_experiment(
            cases,
            backend,
            support_llm.ScriptedLlm([]),
            None,
            _meta(corpus_size=2),
            run_case=run_case,
            finish_case=finish_case,
        )
        bad, good = backend.results
        assert bad.error == "provider declined" and bad.score is None
        assert bad.feedback == "failed: provider declined"
        assert good.error == "" and good.score is not None
        assert backend.calls[-1] == "finish"


class TestBackendsLoadLazily:
    """
    Importing a backend module must not import its SDK.
    """

    @pytest.mark.parametrize(
        ("module", "sdk"),
        [
            ("tests.evals.platform.braintrust_backend", "braintrust"),
            ("tests.evals.platform.opik_backend", "opik"),
        ],
    )
    def test_the_sdk_is_imported_only_in_the_constructor(self, module: str, sdk: str) -> None:
        """
        Test that the CI gate, which has no SDK installed, can still import and
        type-check the backend, and that a missing SDK fails with the install
        instruction rather than an ImportError from deep inside.
        """
        import importlib

        importlib.import_module(module)
        if sdk in sys.modules:
            pytest.skip(f"{sdk} is installed in this environment")
        backend_class = next(
            v for k, v in vars(sys.modules[module]).items() if k.endswith("Backend")
        )
        with pytest.raises(cerrors.ConfigurationError, match="extra evals"):
            backend_class(project="crux")


class TestPrintBackend:
    """
    The built-in backend that needs no SDK.
    """

    def test_it_renders_the_report_and_every_named_score(self) -> None:
        """
        Test that a replay run can be eyeballed before anything is uploaded.
        """
        backend = adapter.PrintBackend()
        case = harness.EvalCase(author="test", id="a", prompt="p")
        backend.upsert_dataset((case,))
        backend.start_experiment(_meta(), {"a": "a"})
        backend.log_result(
            adapter.CaseResult(
                case=case,
                score=_score(case_id="a"),
                scores={"recall": 1.0, "judge": 0.5},
                metrics={},
                rendered="",
            )
        )
        summary = backend.finish()
        assert "judge" in summary and "recall" in summary
        assert "cassette replay" in summary


def _run(name: str, rows: dict[str, dict[str, float]]) -> compare.RunResults:
    import datetime as dt

    return compare.RunResults(
        experiment=name,
        model="m",
        git_sha="abc",
        cassette_mode="replay",
        recorded_at=dt.datetime.now(tz=dt.UTC),
        rows=tuple(
            compare.CaseRow(case_id=case_id, stratum="feature", scores=scores, metrics={})
            for case_id, scores in rows.items()
        ),
    )


class TestCompare:
    """
    Two runs compared case by case.
    """

    def test_leads_count_cases_not_points(self) -> None:
        """
        Test the thing the mean hides: two small wins and one large loss is
        flat on the mean but reads as leading two cases and trailing one.
        """
        current = _run("new", {"a": {"recall": 0.6}, "b": {"recall": 0.6}, "c": {"recall": 0.0}})
        baseline = _run("old", {"a": {"recall": 0.5}, "b": {"recall": 0.5}, "c": {"recall": 0.2}})
        assert current.mean("recall") == pytest.approx(baseline.mean("recall"))
        assert compare.leads(current, baseline)["recall"] == compare.Leads(wins=2, losses=1, ties=0)

    def test_regressions_name_the_case_worst_first(self) -> None:
        """
        Test that a regression is reported by case and score with both values,
        so the reader can go straight to the corpus file.
        """
        current = _run("new", {"a": {"recall": 0.0, "quiet": 1.0}, "b": {"recall": 0.4}})
        baseline = _run("old", {"a": {"recall": 0.5, "quiet": 1.0}, "b": {"recall": 0.5}})
        worse = compare.regressions(current, baseline)
        assert [(r.case_id, r.score) for r in worse] == [("a", "recall"), ("b", "recall")]

    def test_a_case_missing_from_one_run_neither_wins_nor_loses(self) -> None:
        """
        Test that a case that failed in one run, or a score only one run has,
        stays out of the count rather than counting as a loss.
        """
        current = _run("new", {"a": {"recall": 1.0}, "b": {"recall": 1.0, "judge": 0.5}})
        baseline = _run("old", {"a": {"recall": 1.0}})
        result = compare.leads(current, baseline)
        assert result["recall"] == compare.Leads(wins=0, losses=0, ties=1)
        assert result["judge"] == compare.Leads(wins=0, losses=0, ties=0)

    def test_results_round_trip_through_disk(self, tmp_path: pathlib.Path) -> None:
        """
        Test that what the command writes is what the comparison reads.
        """
        run = _run("new", {"a": {"recall": 0.5}})
        compare.save(run, tmp_path / "r" / "new.json")
        assert compare.load(tmp_path / "r" / "new.json") == run

    def test_render_shows_leads_and_regressions(self) -> None:
        """
        Test the table names the regression rather than only counting it.
        """
        current = _run("new", {"a": {"recall": 0.0}})
        baseline = _run("old", {"a": {"recall": 0.5}})
        text = compare.render(current, baseline)
        assert "1 regression(s):" in text
        assert "0.50 -> 0.00" in text


class TestOptimise:
    """
    The gepa adapter, without gepa or a model.
    """

    def test_the_candidate_instruction_reaches_the_expansion_prompt(self) -> None:
        """
        Test the wiring the whole experiment rests on: the text gepa proposes
        is what the model is shown, in place of the built-in instruction and
        with the role and no-invention lines intact.
        """
        messages = xprompt.expand_messages(
            prompt="p", lens="l", existing="", evidence="", host_notes="", instruction="CANDIDATE"
        )
        system = messages[0].content
        assert "CANDIDATE" in system
        assert xprompt.EXPAND_INSTRUCTION not in system
        assert "Never invent a value" in system

    def test_the_seed_candidate_is_the_shipped_instruction(self) -> None:
        """
        Test that optimisation starts from the truth, so the first score is
        the current prompt's score and the file's before/after means something.
        """
        default = xprompt.expand_messages(
            prompt="p", lens="l", existing="", evidence="", host_notes=""
        )
        assert xprompt.EXPAND_INSTRUCTION in default[0].content

    def test_a_split_is_stratified_and_repeatable(self) -> None:
        """
        Test that the held-out third has every prompt size in it and the same
        seed gives the same split, so two runs are comparable.
        """
        cases = harness.load_corpus()
        train, held = optimise.split(cases, seed=1)
        assert len(train) + len(held) == len(cases)
        assert {c.stratum for c in held} == {"one_liner", "feature", "project"}
        assert optimise.split(cases, seed=1) == (train, held)
        assert not set(c.id for c in train) & set(c.id for c in held)

    async def test_a_rollout_scores_the_objective_and_explains_it(self) -> None:
        """
        Test that one rollout yields the recall-and-quiet objective plus the
        feedback text, and a failing case scores zero with its error rather
        than raising, which is gepa's contract.
        """

        async def run_case(
            case: harness.EvalCase, client: pllm.LlmClient, instruction: str
        ) -> csessn.Session:
            if case.id == "bad":
                raise cerrors.ReasonerError("declined")
            return csessn.Session(id=case.id, prompt=case.prompt)

        adapter_ = optimise.ExpandAdapter(support_llm.ScriptedLlm([]), run_case=run_case)
        good = harness.EvalCase(
            author="test", id="good", prompt="p", must_surface=(harness.Expectation(id="nowhere"),)
        )
        bad = harness.EvalCase(author="test", id="bad", prompt="p")
        good_t = await adapter_._rollout(good, "x")
        bad_t = await adapter_._rollout(bad, "x")
        assert good_t["score"] == 0.5
        assert good_t["feedback"] == "missed: nowhere"
        assert bad_t["score"] == 0.0 and bad_t["feedback"] == "failed: declined"

    def test_the_reflective_dataset_carries_prompt_output_and_feedback(self) -> None:
        """
        Test the three fields gepa's proposer reads are filled from the
        trajectory, keyed by the one component under optimisation.
        """

        class Batch:
            trajectories = [
                optimise.Trajectory(
                    case_id="c",
                    prompt="add caching",
                    surfaced=["what gets cached"],
                    asked=["software.scope.build"],
                    feedback="missed: ~how long entries live",
                    score=0.5,
                )
            ]

        adapter_ = optimise.ExpandAdapter(support_llm.ScriptedLlm([]))
        dataset = adapter_.make_reflective_dataset({"expand": "x"}, Batch(), ["expand"])
        (record,) = dataset["expand"]
        assert record["Inputs"] == "add caching"
        assert "what gets cached" in record["Generated Outputs"]
        assert record["Feedback"] == "missed: ~how long entries live"

    def test_the_report_says_it_was_not_applied(self) -> None:
        """
        Test that the written file tells the reader the winner is a candidate
        to review, with before and after, never something already in effect.
        """
        text = optimise.render("do better", 0.3, 0.5, 120)
        assert "0.30 -> 0.50" in text and "120 rollouts" in text
        assert "Not applied" in text and "do better" in text


class TestSystems:
    """
    The baselines, scored by the same harness as crux.
    """

    def test_none_misses_everything_and_asks_nothing(self) -> None:
        """
        Test the honest floor: with the raw prompt as the output, every
        expected decision is missed and nothing the repo answers was asked.
        """
        case = harness.EvalCase(
            author="test",
            id="c",
            prompt="add caching",
            must_surface=(harness.Expectation(id="software.scope.build"),),
            must_not_surface=(harness.Expectation(id="software.deps.policy"),),
        )
        score = harness.score_case(case, systems.run_none(case))
        assert score.recall == 0.0 and score.questions_asked == 0
        assert scorers.decompose(score, case)["quiet"] == 1.0

    async def test_ask3_questions_are_matched_like_decisions(self) -> None:
        """
        Test that a baseline question worded like an expected decision counts
        as surfaced, and that the three questions count against the budget.
        """
        client = support_llm.ScriptedLlm(
            [
                {
                    "questions": [
                        {"text": "which endpoints the limit applies to"},
                        {"text": "how callers are identified for counting"},
                        {"text": "what colour the dashboard should be"},
                    ]
                }
            ]
        )
        case = harness.EvalCase(
            author="test",
            id="c",
            prompt="add rate limiting",
            max_questions=3,
            must_surface=(
                harness.Expectation(match="which endpoints the limit applies to"),
                harness.Expectation(match="how long entries live before expiring"),
            ),
        )
        session = await systems.run_ask3(case, client)
        score = harness.score_case(case, session)
        assert score.recall == 0.5
        assert score.questions_asked == 3 and not score.over_question_budget
        assert score.llm_calls == 1
        assert client.forced == [systems.ASK_TOOL.name]

    async def test_ask3_prose_is_a_parse_error(self) -> None:
        """
        Test that a baseline that ignores its tool fails the case rather than
        scoring as if it asked nothing.
        """
        client = support_llm.ScriptedLlm(["Sure, here are some questions."])
        with pytest.raises(cerrors.ReasonerParseError):
            await systems.run_ask3(harness.EvalCase(author="test", id="c", prompt="p"), client)

    async def test_flat_strips_every_edge_but_keeps_the_proposals(self) -> None:
        """
        Test the B arm: the same proposals reach the graph, none of their
        edges do, and every other operation passes straight through.
        """
        import crux.ports.reasoner as preason
        import tests.support.reasoner as fakes

        inner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal("which delivery channel", ref="channel"),
                        fakes.proposal(
                            "digest cadence",
                            edges=(
                                preason.ProposedEdge(
                                    kind="requires", source_id="channel", when_value="email"
                                ),
                            ),
                        ),
                    )
                )
            ]
        )
        stripped = systems.EdgeStripper(inner)
        result = await stripped.expand(preason.ExpansionRequest(prompt="p", lens="l"))
        assert [p.undecided for p in result.proposed] == [
            "which delivery channel",
            "digest cadence",
        ]
        assert all(p.edges == () for p in result.proposed)
        assert stripped.adjudicate.__func__ is inner.adjudicate.__func__  # type: ignore[attr-defined]

    async def test_run_system_rejects_an_unknown_name(self) -> None:
        """
        Test that a typo in --system fails loudly rather than silently scoring crux.
        """
        with pytest.raises(cerrors.ConfigurationError):
            await systems.run_system(
                "typo",  # type: ignore[arg-type]
                harness.EvalCase(author="test", id="c", prompt="p"),
                support_llm.ScriptedLlm([]),
            )


class TestPerModel:
    """
    Recordings and results that never collide across models.
    """

    def test_cassette_name_carries_the_model(self) -> None:
        """
        Test that two models get two cassettes, and the name is safe on disk.
        """
        name = runner.cassette_name("recall", "cohere_chat/command-a-03-2025")
        assert name == "recall-cohere-chat-command-a-03-2025"
        assert "/" not in name
        assert runner.cassette_name("recall", "claude-sonnet-5") != name

    def test_the_models_table_ranks_and_counts_leads(self) -> None:
        """
        Test that the summary orders models by recall, shows ask rate per
        stratum, and counts case-level leads over the weakest model rather
        than only reporting means.
        """
        strong = _run("s", {"a": {"recall": 1.0}, "b": {"recall": 0.5}})
        weak = _run("w", {"a": {"recall": 0.0}, "b": {"recall": 0.5}})
        strong = strong.model_copy(update={"model": "strong"})
        weak = weak.model_copy(update={"model": "weak"})
        text = models.render([weak, strong])
        lines = text.splitlines()
        assert lines[2].startswith("strong") and lines[3].startswith("weak")
        assert "leads over weak" in text
        assert "recall 1/0/1" in text
