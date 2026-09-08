"""
Upload one eval run to a platform, or print it.

    uv run python -m tests.evals.platform --backend stdout
    uv run python -m tests.evals.platform --backend braintrust --project crux
    uv run python -m tests.evals.platform --backend opik --record

Replays from ``cassettes/recall.json`` and ``cassettes/judge.json`` unless
``--record`` is given. The run shares the recall cassette on purpose: the
scores uploaded are then byte-for-byte the ones the ratchet asserts on, and one
recording serves both. A separate command rather than a pytest option so the
ratchet stays a pure test and never needs platform credentials.
"""

from __future__ import annotations

import argparse
import asyncio

import crux.errors as cerrors
import crux.infra.settings as isettn
import tests.evals.harness as harness
import tests.evals.platform.adapter as adapter
import tests.evals.runner as runner

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
    settings = isettn.CruxSettings()
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
    backend = build_backend(args.backend, project=args.project, experiment=args.experiment)
    try:
        await adapter.run_experiment(
            cases, backend, client, judge_client, meta, judge_model=args.judge_model
        )
    except KeyError as exc:
        print(f"cassette miss ({exc}); rerun with --record")
        return 1
    print(f"\ncassette hits {client.hits}, misses {client.misses}")
    return 0


def main() -> int:
    """
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(prog="tests.evals.platform", description=__doc__)
    parser.add_argument("--backend", choices=("stdout", "braintrust", "opik"), default="stdout")
    parser.add_argument("--record", action="store_true", help="call the real model")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--project", default="crux", help="platform project name")
    parser.add_argument("--experiment", default=None, help="experiment name")
    parser.add_argument("--no-judge", action="store_true", help="skip the rubric judge")
    parser.add_argument("--judge-model", default=None, help="model that judges rubrics")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
