"""
The saturation experiment.

This is the measurement the graph's existence turns on, and the falsification
criteria are written down here *before* it runs so the result cannot be read
generously after the fact.

    uv run python -m tests.evals.saturation --record        # real model
    uv run python -m tests.evals.saturation                 # replay a cassette

Six metrics, and the fifth is the one most likely to be skipped and most likely
to change the conclusion:

1. marginal yield per pass, per stratum
2. k* — the smallest pass index whose yield falls below 10%
3. edge density — how often the expander attaches a conditional edge
4. edge *firing* density — how often one of those ever does anything
5. noise floor — pass one, sampled three times, compared to itself
6. questions and assumptions, with the graph and with it flattened

Without (5), k* is uninterpretable: a "new" decision on pass two may simply be
the model sampling differently, not finding anything deeper.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import pathlib

import crux.application.engine as aengine
import crux.application.expansion as aexpand
import crux.domain.graph as cgraph
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.packs.base as kspec
import crux.packs.software  # noqa: F401 - registers the shipped pack
import crux.ports.llm as pllm
import crux.ports.reasoner as preason
import tests.evals.harness as harness
import tests.evals.runner as runner
import tests.evals.systems as systems

# A pass yielding less than this is not finding anything new worth paying for.
K_STAR_RATIO = 0.1

# How many times pass one is resampled to establish the noise floor.
NOISE_SAMPLES = 3

# --- The decision rule, committed in advance -------------------------------
CEREMONY_K_STAR = 1.0
CEREMONY_FIRING = 0.05
EARNS_K_STAR = 2.0
EARNS_FIRING = 0.20

# Below this, pass one disagrees with itself so much that "new decisions on pass
# two" is mostly resampling rather than depth, and k* means nothing. Measured at
# 0.26 on the first real run, which is why this guard exists at all.
K_STAR_TRUSTWORTHY_NOISE = 0.5


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """
    Overlap between two sets of decision ids.

    :param left: One sample.
    :param right: The other.
    :return: Overlap between 0.0 and 1.0; two empty sets count as identical.
    """
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def k_star(ratios: list[float]) -> int:
    """
    Find the first pass whose yield fell below the threshold.

    :param ratios: New-after-dedup over proposed, one per pass, in order.
    :return: The one-based pass index, or the pass count when it never drops.
    """
    for index, ratio in enumerate(ratios, start=1):
        if ratio < K_STAR_RATIO:
            return index
    return len(ratios)


def firing_density(stats: dict[str, tuple[int, int]]) -> float:
    """
    Fraction of conditional edges that ever fired.

    ``informs`` is excluded: it shares evidence rather than gating or deleting
    anything, so counting it would flatter the number the decision rule reads.

    :param stats: Edge kind mapped to (attached, fired).
    :return: Fired over attached, or 0.0 when nothing was attached.
    """
    attached = sum(a for kind, (a, _) in stats.items() if kind != "informs")
    fired = sum(f for kind, (_, f) in stats.items() if kind != "informs")
    return fired / attached if attached else 0.0


def edge_coherence(graph: cgraph.DecisionGraph) -> tuple[int, int]:
    """
    Count conditional edges that *could* ever fire.

    Firing density depends on what the respondent happened to answer, so it is a
    lower bound rather than a property of the graph. This is the deterministic
    companion: an edge whose ``when_value`` is not among its source's options can
    never fire under any answer, and is therefore dead the moment it is written.

    A model that emits edges nobody can trigger is producing decoration, and that
    shows up here regardless of how the questions were answered.

    :param graph: The graph to inspect.
    :return: (conditional edges, of those, ones whose trigger is reachable).
    """
    total = 0
    reachable = 0
    for node in graph.nodes.values():
        for edge in node.depends_on:
            if edge.kind == "informs" or edge.when_value is None:
                continue
            total += 1
            source = graph.nodes.get(edge.source_id)
            if source is None:
                continue
            values = {o.value for o in (source.options or ())}
            if edge.when_value in values:
                reachable += 1
    return total, reachable


@dataclasses.dataclass
class CaseMeasurement:
    """
    Everything measured for one prompt.
    """

    case_id: str
    stratum: str
    ratios: list[float]
    new_per_pass: list[int]
    noise_floor: float | None
    edge_stats: dict[str, tuple[int, int]]
    conditional_edges: int
    reachable_edges: int
    questions_with_graph: int
    questions_flat: int
    assumptions_with_graph: int
    assumptions_flat: int
    decisions_total: int

    @property
    def k_star(self) -> int:
        """
        :return: This case's saturation point.
        """
        return k_star(self.ratios)

    @property
    def coherence(self) -> float:
        """
        :return: Fraction of conditional edges that could fire under some answer.
        """
        if not self.conditional_edges:
            return 0.0
        return self.reachable_edges / self.conditional_edges

    @property
    def firing(self) -> float:
        """
        :return: This case's edge-firing density.
        """
        return firing_density(self.edge_stats)


async def _expansion_curve(
    case: harness.EvalCase,
    reasoner: preason.Reasoner,
    passes: int,
) -> tuple[list[float], list[int], cgraph.DecisionGraph]:
    """
    Run N expansion passes, ignoring the saturation stop.

    Driven directly rather than through the engine, because the engine stops at
    saturation and the whole point here is to see what the passes *after* it
    would have produced.

    :param case: The prompt to expand.
    :param reasoner: The model boundary.
    :param passes: How many passes to run.
    :return: Yield ratios, new-decision counts, and the resulting graph.
    """
    graph = cgraph.DecisionGraph()
    for decision in kspec.seed(case.prompt, ("software",)):
        graph = graph.add(decision)

    ratios: list[float] = []
    new_per_pass: list[int] = []
    for index in range(1, passes + 1):
        result = await reasoner.expand(
            preason.ExpansionRequest(
                prompt=case.prompt,
                lens=aexpand.lens_for(index),
                existing=aexpand.sketches(graph),
            )
        )
        graph, record = aexpand.absorb(graph, result, pass_index=index, pack_ids=("software",))
        ratios.append(record.ratio)
        new_per_pass.append(record.new_after_dedup)
    return ratios, new_per_pass, graph


async def _noise_floor(
    case: harness.EvalCase,
    client: pllm.LlmClient,
    record: bool,
) -> float | None:
    """
    Sample pass one repeatedly and see how much it agrees with itself.

    Only meaningful against a live model: a cassette keyed on request content
    returns the same answer every time, which would report a noise floor of 1.0
    and make every later pass look like genuine depth.

    :param case: The prompt to sample.
    :param client: Where completions come from.
    :param record: Whether real calls are being made.
    :return: Mean pairwise overlap, or ``None`` when replaying.
    """
    if not record:
        return None
    import crux.adapters.llm.reasoner as xreason

    samples: list[frozenset[str]] = []
    for sample in range(NOISE_SAMPLES):
        # Each sample must reach the model, not the cassette entry the last one
        # wrote. Identical requests are the whole point here.
        if isinstance(client, harness.CassetteLlm):
            client.salt = f"noise-{sample}"
        _, _, graph = await _expansion_curve(case, xreason.LlmReasoner(client), 1)
        samples.append(frozenset(graph.nodes))
    if isinstance(client, harness.CassetteLlm):
        client.salt = ""
    pairs = [
        jaccard(samples[i], samples[j])
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    ]
    return sum(pairs) / len(pairs) if pairs else None


async def measure_case(
    case: harness.EvalCase,
    client: pllm.LlmClient,
    *,
    passes: int,
    record: bool,
) -> CaseMeasurement:
    """
    Measure one case on all six metrics.

    :param case: The case to measure.
    :param client: Where completions come from.
    :param passes: How many expansion passes to force.
    :param record: Whether real calls are allowed.
    :return: The measurement.
    """
    import crux.adapters.llm.reasoner as xreason

    reasoner = xreason.LlmReasoner(client)
    ratios, new_per_pass, _ = await _expansion_curve(case, reasoner, passes)
    noise = await _noise_floor(case, client, record)

    graph_session = await runner.run_case(case, client, answer=True)
    flat_session = await _run_flat(case, client)
    conditional, reachable = edge_coherence(graph_session.graph)

    return CaseMeasurement(
        case_id=case.id,
        stratum=case.stratum,
        ratios=ratios,
        new_per_pass=new_per_pass,
        noise_floor=noise,
        edge_stats=graph_session.graph.edge_stats(),
        conditional_edges=conditional,
        reachable_edges=reachable,
        questions_with_graph=graph_session.budget.spent_questions,
        questions_flat=flat_session.budget.spent_questions,
        assumptions_with_graph=(
            len(graph_session.outcome.assumptions) if graph_session.outcome else 0
        ),
        assumptions_flat=(len(flat_session.outcome.assumptions) if flat_session.outcome else 0),
        decisions_total=len(graph_session.graph.nodes),
    )


async def _run_flat(case: harness.EvalCase, client: pllm.LlmClient) -> csessn.Session:
    """
    Run the same case with every conditional edge stripped.

    :param case: The case to run.
    :param client: Where completions come from.
    :return: The finished session.
    """
    import crux.adapters.llm.reasoner as xreason
    import crux.adapters.retrieval.fs as xfsretr

    crux = aengine.Crux(
        reasoner=systems.EdgeStripper(xreason.LlmReasoner(client)),
        retriever=xfsretr.FilesystemRetriever(case.root) if case.root else None,
        budget=csessn.Budget(max_questions_total=case.max_questions),
    )
    step = await crux.start(
        case.prompt,
        context=csessn.SessionContext(root=case.root) if case.root else None,
        session_id=f"{case.id}-flat",
    )
    respondent = runner.FirstOptionRespondent()
    while isinstance(step, csessn.NeedsInput):
        step = await crux.resume(step.session, respondent.answer(step.questions))
    return step.session


def verdict(measurements: list[CaseMeasurement]) -> str:
    """
    Apply the decision rule committed to before the run.

    :param measurements: Every case's measurement.
    :return: What to do about the graph machinery.
    """
    if not measurements:
        return "no data"
    mean_k = sum(m.k_star for m in measurements) / len(measurements)
    mean_firing = sum(m.firing for m in measurements) / len(measurements)
    saved = sum(m.questions_flat - m.questions_with_graph for m in measurements)
    conditional = sum(m.conditional_edges for m in measurements)
    reachable = sum(m.reachable_edges for m in measurements)
    coherence = reachable / conditional if conditional else 0.0

    noises = [m.noise_floor for m in measurements if m.noise_floor is not None]
    noise = sum(noises) / len(noises) if noises else None
    if noise is not None and noise < K_STAR_TRUSTWORTHY_NOISE:
        return (
            f"mean k* = {mean_k:.2f} · noise floor = {noise:.2f} · "
            f"edge coherence = {coherence:.0%} ({reachable}/{conditional}) · "
            f"firing = {mean_firing:.0%} · questions saved = {saved}\n"
            f"NO VERDICT — pass one agrees with itself only {noise:.0%} of the "
            f"time, so k* is measuring resampling rather than depth. The edge "
            f"numbers below stand on their own; the saturation half does not."
        )

    if mean_k <= CEREMONY_K_STAR and mean_firing < CEREMONY_FIRING:
        call = "CEREMONY — delete requires/prunes evaluation, keep a flat ranked list."
    elif mean_k >= EARNS_K_STAR or mean_firing > EARNS_FIRING:
        call = "EARNS ITS KEEP — the graph stays, and prunes is what earns it."
    else:
        call = "PARTIAL — keep prunes only; drop requires, constrains, informs."

    return (
        f"mean k* = {mean_k:.2f} · noise floor = "
        f"{f'{noise:.2f}' if noise is not None else 'not measured'} · "
        f"edge coherence = {coherence:.0%} ({reachable}/{conditional}) · "
        f"firing = {mean_firing:.0%} · questions saved = {saved}\n{call}"
    )


def render(measurements: list[CaseMeasurement]) -> str:
    """
    :param measurements: Every case's measurement.
    :return: A table a person can read.
    """
    lines = [
        f"{'case':<24} {'stratum':<10} {'k*':>3} {'yield per pass':<20} "
        f"{'cond':>5} {'reach':>6} {'fired':>6} {'q(g/f)':>8} {'noise':>6}",
        "-" * 100,
    ]
    for m in measurements:
        curve = ",".join(str(n) for n in m.new_per_pass)
        noise = f"{m.noise_floor:.2f}" if m.noise_floor is not None else "  -"
        lines.append(
            f"{m.case_id:<24} {m.stratum:<10} {m.k_star:>3} {curve:<20} "
            f"{m.conditional_edges:>5} {m.coherence:>5.0%} {m.firing:>5.0%} "
            f"{m.questions_with_graph}/{m.questions_flat:<6} {noise:>6}"
        )
    lines.append("-" * 100)
    lines.append(verdict(measurements))
    if all(m.noise_floor is None for m in measurements):
        lines.append(
            "\nNOISE FLOOR NOT MEASURED (replay mode). k* is uninterpretable "
            "without it — rerun with --record before acting on the verdict."
        )
    return "\n".join(lines)


async def _main(args: argparse.Namespace) -> int:
    """
    Run the experiment.

    :param args: Parsed command line.
    :return: Process exit code.
    """
    cases = [c for c in harness.load_corpus() if runner.fixture_exists(c)]
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    if not cases:
        print("no cases to run")
        return 1

    client = runner.build_client("saturation", record=args.record)
    measurements: list[CaseMeasurement] = []
    failed: list[tuple[str, str]] = []
    for case in cases:
        print(f"measuring {case.id} ...", flush=True)
        try:
            measurements.append(
                await measure_case(case, client, passes=args.passes, record=args.record)
            )
        except KeyError as exc:
            print(f"  cassette miss ({exc}); rerun with --record")
            return 1
        except cerrors.CruxError as exc:
            # One case must not cost the run. A provider that declines a single
            # request took out eight completed cases the first time this ran, and
            # the surviving measurements were more valuable than the tidy exit.
            print(f"  FAILED: {str(exc)[:120]}")
            failed.append((case.id, str(exc)[:200]))

    print()
    print(render(measurements))
    if failed:
        print(f"\n{len(failed)} case(s) failed and are excluded from the numbers above:")
        for case_id, why in failed:
            print(f"  {case_id}: {why}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([dataclasses.asdict(m) for m in measurements], indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    """
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="call the real model")
    parser.add_argument("--passes", type=int, default=5, help="expansion passes to force")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument(
        "--out",
        default="tests/evals/results/saturation.json",
        help="where to write the raw measurements",
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
