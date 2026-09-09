"""
Upload one eval run to a platform, or print it.

    uv run python -m tests.evals.platform --backend stdout
    uv run python -m tests.evals.platform --backend braintrust --project crux
    uv run python -m tests.evals.platform --backend opik --record

``--system`` scores a baseline instead of crux: ``none`` (raw prompt), ``ask3``
(one call, three questions, nothing else) or ``flat`` (crux with every edge
stripped). Baselines write their own results file, named after the system, so
``--baseline`` compares them to crux case by case.

``--misses`` prints every missed expectation beside the surfaced decisions
nearest to it and writes a template; a person marks which were really covered
and ``--rescored`` reports recall corrected for matcher error.

Every run also writes its per-case scores to ``results/<experiment>.json``, and
``--baseline`` compares against an earlier file case by case: which run leads
on how many cases, and which cases regressed. The mean hides both.

``--model`` picks the model. Cassettes and results files are named per model
(``recall-<model>.json``, ``results/<system>-<model>.json``), so recordings
never collide and ``python -m tests.evals.models`` can compare them.

Replays from the model's recall and judge cassettes unless
``--record`` is given. The run shares the recall cassette on purpose: the
scores uploaded are then byte-for-byte the ones the ratchet asserts on, and one
recording serves both. A separate command rather than a pytest option so the
ratchet stays a pure test and never needs platform credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib

import crux.domain.session as csessn
import crux.errors as cerrors
import crux.infra.settings as isettn
import crux.ports.llm as pllm
import tests.evals.compare as compare
import tests.evals.harness as harness
import tests.evals.platform.adapter as adapter
import tests.evals.rescoring as rescoring
import tests.evals.runner as runner
import tests.evals.systems as systems

CASSETTE = "recall"
JUDGE_CASSETTE = "judge"


def build_backend(name: str, *, project: str, experiment: str | None) -> adapter.EvalBackend:
    """
    :param name: Which backend.
    :param project: The platform project.
    :param experiment: The experiment name, when the caller chose one.
    :return: The backend.
    :raises ConfigurationError: When the name is unknown or the SDK is missing.
    """
    if name == "stdout":
        return adapter.PrintBackend()
    if name == "braintrust":
        import tests.evals.platform.braintrust_backend as btback

        return btback.BraintrustBackend(project=project, experiment=experiment)
    if name == "opik":
        import tests.evals.platform.opik_backend as opback

        return opback.OpikBackend(project=project, experiment=experiment)
    raise cerrors.ConfigurationError(f"Unknown backend {name!r}.")


async def _main(args: argparse.Namespace) -> int:
    """
    :param args: Parsed command line.
    :return: Process exit code.
    """
    cases = [c for c in harness.load_corpus() if runner.fixture_exists(c)]
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    if not cases:
        print("no cases to run")
        return 1

    # Provider keys live in .env under the provider's own name, which litellm
    # reads from the process environment, not from settings.
    isettn.load_dotenv()
    settings = isettn.CruxSettings(model=args.model) if args.model else isettn.CruxSettings()
    client = runner.build_client(CASSETTE, record=args.record, settings=settings)
    judge_client = (
        None
        if args.no_judge
        else runner.build_client(JUDGE_CASSETTE, record=args.record, settings=settings)
    )
    meta = adapter.collect_metadata(
        settings,
        record=args.record,
        judge_model=None if args.no_judge else args.judge_model,
        corpus_size=len(cases),
    )
    name = args.backend if args.system == "crux" else args.system
    experiment = args.experiment or f"{name}-{runner.model_slug(settings.model)}"
    backend = build_backend(args.backend, project=args.project, experiment=experiment)

    async def run_case(case: harness.EvalCase, llm: pllm.LlmClient) -> csessn.Session:
        return await systems.run_system(args.system, case, llm)

    results = await adapter.run_experiment(
        cases,
        backend,
        client,
        judge_client,
        meta,
        run_case=run_case,
        judge_model=args.judge_model,
    )
    print(f"\ncassette hits {client.hits}, misses {client.misses}")
    if any("Cassette miss" in r.error for r in results):
        print("some cases hit a cassette miss; rerun with --record to capture them")

    run = adapter.to_results(results, meta, experiment=experiment)
    out = pathlib.Path(args.out) if args.out else compare.RESULTS / f"{experiment}.json"
    compare.save(run, out)
    print(f"wrote {out}")
    if args.baseline:
        print()
        print(compare.render(run, compare.load(pathlib.Path(args.baseline))))
    if args.misses:
        found = rescoring.misses(run)
        template = out.with_name(f"{out.stem}-misses.yaml")
        template.write_text(rescoring.template(found), encoding="utf-8")
        print()
        print(rescoring.render(found))
        print(f"fill in covered_by in {template}, then rerun with --rescored {template}")
    if args.rescored:
        print()
        print(rescoring.render_corrected(run, rescoring.load_rescores(pathlib.Path(args.rescored))))
    return 0


def main() -> int:
    """
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(prog="tests.evals.platform", description=__doc__)
    parser.add_argument("--backend", choices=("stdout", "braintrust", "opik"), default="stdout")
    parser.add_argument(
        "--system",
        choices=systems.SYSTEMS,
        default="crux",
        help="what to score: crux, or a baseline (flat, ask3, none)",
    )
    parser.add_argument("--model", default=None, help="model to run, overriding CRUX_MODEL")
    parser.add_argument("--record", action="store_true", help="call the real model")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--project", default="crux", help="platform project name")
    parser.add_argument("--experiment", default=None, help="experiment name")
    parser.add_argument("--no-judge", action="store_true", help="skip the rubric judge")
    parser.add_argument("--judge-model", default=None, help="model that judges rubrics")
    parser.add_argument("--out", default=None, help="where to write this run's results")
    parser.add_argument("--baseline", default=None, help="a results file to compare against")
    parser.add_argument(
        "--misses", action="store_true", help="print every miss with its nearest surfaced decisions"
    )
    parser.add_argument(
        "--rescored", default=None, help="a filled-in misses file; prints corrected recall"
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
