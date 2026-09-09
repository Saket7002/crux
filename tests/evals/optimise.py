"""
Reflective optimisation of the expansion instruction against the corpus.

    uv run python -m tests.evals.optimise [--budget N] [--reflection-model M]

Follows GEPA (arXiv 2507.19457): run a candidate instruction on a minibatch of
cases, hand a reflection model the prompt, what surfaced, and the scorer's
feedback text, let it propose a new instruction, keep it if the minibatch
improved, and pick the next candidate from the Pareto front over cases rather
than from the single best mean.

What is optimised is exactly ``xprompt.EXPAND_INSTRUCTION``. The role line and
the no-invention rule stay fixed, the other five operations stay fixed, and the
winner is written to a file for a person to read and apply by hand. Nothing
here edits a prompt in the source tree.

Every rollout is a real model call by definition: a candidate instruction is a
request the cassette has never seen. So this refuses a fallback chain like
every other measurement, and it is never run in CI.

Import as:

import tests.evals.optimise as optimise
"""

from __future__ import annotations

import argparse
import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypedDict

import crux.adapters.llm.litellm as xlitell
import crux.adapters.llm.prompts as xprompt
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.infra.settings as isettn
import crux.ports.llm as pllm
import tests.evals.compare as compare
import tests.evals.harness as harness
import tests.evals.runner as runner
import tests.evals.scorers as scorers

COMPONENT = "expand"
"""The one component under optimisation, by name."""

OUTPUT = compare.RESULTS / "expand_instruction.md"

RunCase = Callable[[harness.EvalCase, pllm.LlmClient, str], Awaitable[csessn.Session]]


class Trajectory(TypedDict):
    """
    What one rollout leaves behind for the reflection model.
    """

    case_id: str
    prompt: str
    surfaced: list[str]
    asked: list[str]
    feedback: str
    score: float


class Record(TypedDict):
    """
    One row of the reflective dataset, in the keys gepa's proposer expects.
    """

    Inputs: str
    Generated_Outputs: str
    Feedback: str


def objective(scores: Mapping[str, float]) -> float:
    """
    The number being maximised.

    Recall and quiet, equally weighted: surface what matters, and do not ask
    what the repo already answers. Budget and retrieval are left out because
    the expansion instruction does not decide them.

    :param scores: The decomposed scores for one case.
    :return: A value in 0..1.
    """
    return (scores["recall"] + scores["quiet"]) / 2


class ExpandAdapter:
    """
    gepa's view of crux: run a candidate, score it, explain the score.

    Implements ``gepa.core.adapter.GEPAAdapter`` structurally, so the module
    imports without the SDK.
    """

    def __init__(self, client: pllm.LlmClient, *, run_case: RunCase | None = None) -> None:
        """
        :param client: Where crux's completions come from. A real one.
        :param run_case: How to run one case with one instruction. Injected by
            the tests; defaults to the shared runner.
        """
        self._client = client
        self._run_case = run_case or _default_run_case

    def evaluate(
        self,
        batch: list[harness.EvalCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        """
        Run every case in the batch under the candidate instruction.

        A case that fails scores 0.0 with its error as feedback, never raises:
        gepa's contract, and also what keeps one provider hiccup from ending a
        run that cost real money.

        :param batch: Cases to run.
        :param candidate: Component name to text; only ``expand`` is read.
        :param capture_traces: Whether to keep trajectories for reflection.
        :return: A ``gepa.EvaluationBatch``.
        """
        import gepa.core.adapter as gadapt

        instruction = candidate[COMPONENT]
        outputs: list[dict[str, float]] = []
        scores: list[float] = []
        trajectories: list[Trajectory] = []
        for case in batch:
            trajectory = asyncio.run(self._rollout(case, instruction))
            trajectories.append(trajectory)
            scores.append(trajectory["score"])
            outputs.append({"score": trajectory["score"]})
        return gadapt.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Turn trajectories into what the reflection model reads.

        :param candidate: The candidate that was evaluated.
        :param eval_batch: Its evaluation, with trajectories.
        :param components_to_update: Always ``["expand"]`` here.
        :return: Records keyed by component.
        """
        trajectories: list[Trajectory] = eval_batch.trajectories or []
        records = [
            {
                "Inputs": trajectory["prompt"],
                "Generated Outputs": "Decisions surfaced:\n"
                + "\n".join(f"- {d}" for d in trajectory["surfaced"])
                + "\nAsked:\n"
                + "\n".join(f"- {q}" for q in trajectory["asked"]),
                "Feedback": trajectory["feedback"],
            }
            for trajectory in trajectories
        ]
        return {component: records for component in components_to_update}

    async def _rollout(self, case: harness.EvalCase, instruction: str) -> Trajectory:
        """
        :param case: The case to run.
        :param instruction: The candidate expansion instruction.
        :return: What happened, scored and explained.
        """
        try:
            session = await self._run_case(case, self._client, instruction)
        except cerrors.CruxError as exc:
            return Trajectory(
                case_id=case.id,
                prompt=case.prompt,
                surfaced=[],
                asked=[],
                feedback=f"failed: {exc}",
                score=0.0,
            )
        score = harness.score_case(case, session)
        decomposed = scorers.decompose(score, case)
        return Trajectory(
            case_id=case.id,
            prompt=case.prompt,
            surfaced=[n.undecided for n in session.graph.nodes.values()],
            asked=[q.decision_id for q in session.pending]
            + [e.question.decision_id for e in session.transcript],
            feedback=scorers.feedback(score, case),
            score=objective(decomposed),
        )


def split(
    cases: Sequence[harness.EvalCase], *, seed: int = 0
) -> tuple[tuple[harness.EvalCase, ...], tuple[harness.EvalCase, ...]]:
    """
    Two thirds for training minibatches, one third held for Pareto selection.

    Stratified by prompt size, so the held-out third cannot end up all
    one-liners and reward an instruction that only works on those.

    :param cases: The corpus.
    :param seed: For a repeatable split.
    :return: Training cases, then held-out cases.
    """
    rng = random.Random(seed)
    train: list[harness.EvalCase] = []
    held: list[harness.EvalCase] = []
    for stratum in ("one_liner", "feature", "project"):
        members = [c for c in cases if c.stratum == stratum]
        rng.shuffle(members)
        cut = len(members) - len(members) // 3
        train.extend(members[:cut])
        held.extend(members[cut:])
    return tuple(sorted(train, key=lambda c: c.id)), tuple(sorted(held, key=lambda c: c.id))


def render(instruction: str, seed_score: float, best_score: float, budget: int) -> str:
    """
    :param instruction: The winning instruction.
    :param seed_score: The current instruction's held-out mean.
    :param best_score: The winner's held-out mean.
    :param budget: Rollouts spent.
    :return: A file a person reads before deciding whether to apply it.
    """
    return (
        "# Candidate expansion instruction\n\n"
        f"Held-out objective (mean of recall and quiet): {seed_score:.2f} -> {best_score:.2f} "
        f"after {budget} rollouts.\n\n"
        "Not applied. Review it, then replace `EXPAND_INSTRUCTION` in "
        "`src/crux/adapters/llm/prompts.py` and re-record the cassettes.\n\n"
        "```\n" + instruction.strip() + "\n```\n"
    )


async def _default_run_case(
    case: harness.EvalCase, client: pllm.LlmClient, instruction: str
) -> csessn.Session:
    return await runner.run_case(case, client, expand_instruction=instruction)


def _main(args: argparse.Namespace) -> int:
    """
    :param args: Parsed command line.
    :return: Process exit code.
    """
    try:
        import gepa.api as gapi
    except ImportError as exc:
        raise cerrors.ConfigurationError(
            "The gepa package is not installed. Run `uv sync --extra evals`."
        ) from exc

    isettn.load_dotenv()
    settings = runner.require_single_model(isettn.CruxSettings())
    cases = [c for c in harness.load_corpus() if runner.fixture_exists(c)]
    train, held = split(cases, seed=args.seed)
    # Held as Any at the SDK boundary: gepa's adapter Protocol is generic over
    # trajectory and output types it never inspects, and its result type moves
    # between releases.
    adapter: Any = ExpandAdapter(xlitell.LiteLlmClient(settings))
    seed_candidate = {COMPONENT: xprompt.EXPAND_INSTRUCTION}

    # The seed is scored on every held-out case before any reflection, and a
    # candidate that improves its minibatch is scored on all of them again. A
    # budget below that never proposes anything and reports the seed as best.
    floor = len(held) + args.minibatch + len(held)
    if args.budget < floor:
        print(
            f"budget {args.budget} is below the {floor} rollouts one reflection step needs "
            f"({len(held)} held-out, {args.minibatch} minibatch, {len(held)} re-score); "
            "the seed will be reported unchanged"
        )
    print(f"train {len(train)} · held-out {len(held)} · budget {args.budget} rollouts")
    result: Any = gapi.optimize(
        seed_candidate=seed_candidate,
        trainset=list(train),
        valset=list(held),
        adapter=adapter,
        reflection_lm=args.reflection_model or settings.model,
        candidate_selection_strategy="pareto",
        use_merge=False,
        max_metric_calls=args.budget,
        reflection_minibatch_size=args.minibatch,
        run_dir=str(compare.RESULTS / "gepa"),
        seed=args.seed,
        display_progress_bar=True,
    )
    winner = result.best_candidate
    best = winner[COMPONENT] if isinstance(winner, dict) else str(winner)
    scores = result.val_aggregate_scores
    seed_score = scores[0] if scores else 0.0
    best_score = scores[result.best_idx] if scores else 0.0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(best, seed_score, best_score, result.total_metric_calls or 0))
    print(f"\nheld-out {seed_score:.2f} -> {best_score:.2f}; wrote {OUTPUT}")
    return 0


def main() -> int:
    """
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(prog="tests.evals.optimise", description=__doc__)
    parser.add_argument("--budget", type=int, default=300, help="total rollouts allowed")
    parser.add_argument("--minibatch", type=int, default=4, help="cases per reflection step")
    parser.add_argument("--reflection-model", default=None, help="model that rewrites prompts")
    parser.add_argument("--seed", type=int, default=0)
    return _main(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
