"""
The demo driver: one prompt in, one compiled prompt out.

Import as:

import crux.cli.main as ccli
"""

from __future__ import annotations

import pathlib
import uuid
from typing import Annotated

import rich.console
import rich.syntax
import typer

import crux.adapters.clarifier.tty as xtty
import crux.adapters.llm.litellm as xlitell
import crux.adapters.llm.reasoner as xreason
import crux.adapters.llm.routing as xroute
import crux.adapters.retrieval.fs as xfsretr
import crux.adapters.store.jsonfile as xjsonst
import crux.adapters.store.memory as xmemsto
import crux.api as capi
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.infra.evidence as ievid
import crux.infra.logging as ilog
import crux.infra.settings as isettn
import crux.packs.base as kspec
import crux.packs.software  # noqa: F401 - registers the shipped pack
import crux.ports.store as pstore

app = typer.Typer(
    add_completion=False,
    help="Work out what a prompt leaves undecided, then compile it.",
)


@app.command(name="clarify", help="Work out what a prompt leaves undecided, then compile it.")
def clarify(
    prompt: Annotated[str, typer.Argument(help="What you want done.")],
    root: Annotated[
        pathlib.Path,
        typer.Option("--root", "-r", help="Directory the work concerns."),
    ] = pathlib.Path(),
    headless: Annotated[
        bool,
        typer.Option("--headless", help="Ask nothing; state every guess as an assumption."),
    ] = False,
    max_questions: Annotated[
        int, typer.Option("--max-questions", "-q", help="Cap on questions asked.")
    ] = 6,
    no_retrieval: Annotated[
        bool, typer.Option("--no-retrieval", help="Do not read the directory at all.")
    ] = False,
    save_to: Annotated[
        pathlib.Path | None,
        typer.Option("--save-to", help="Directory to write the session JSON into."),
    ] = None,
    evidence: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--evidence",
            help="Directory to write a full run trace into: what was asked, what "
            "was found, every model exchange, and why each decision closed.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="The reasoning model: expand and adjudicate."),
    ] = None,
    weak_model: Annotated[
        str | None,
        typer.Option(
            "--weak-model",
            help="The mechanical model: phrasing, classifying, counter-answers, drafting.",
        ),
    ] = None,
    config: Annotated[
        pathlib.Path | None,
        typer.Option("--config", help="A crux.toml to use instead of searching for one."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show what crux is doing.")
    ] = False,
) -> None:
    """
    Clarify a prompt and print the compiled result.

    :param prompt: What you want done.
    :param root: Directory the work concerns.
    :param headless: Whether to skip asking entirely.
    :param max_questions: Cap on questions asked.
    :param no_retrieval: Whether to skip reading the directory.
    :param model: The reasoning model.
    :param weak_model: The mechanical model.
    :param config: An explicit crux.toml.
    :param save_to: Where to write the session JSON.
    :param evidence: Where to write a full run trace.
    :param verbose: Whether to show crux's own logging.
    """
    ilog.configure(verbose=verbose)
    console = rich.console.Console()
    session_id = str(uuid.uuid4())
    # Before settings are read, so a provider key in .env reaches litellm.
    isettn.load_dotenv()
    config_file = config or isettn.find_config_file()
    overrides: dict[str, object] = {"config_file": config_file}
    if model:
        overrides["model"] = model
    if weak_model:
        overrides["weak_model"] = weak_model
    settings = isettn.CruxSettings(**overrides)  # type: ignore[arg-type]
    routing = xroute.ModelRouting.of(settings)

    recorder = ievid.RunRecorder(evidence, session_id=session_id) if evidence else None
    reasoner = xreason.LlmReasoner(xlitell.LiteLlmClient(settings, recorder), routing)
    retriever = None if no_retrieval else xfsretr.FilesystemRetriever(root)
    clarifier = None if headless else xtty.TtyClarifier(console)
    # One store serves both jobs: it keeps the session between rounds, and it is
    # where the trace reads the finished session back from.
    store: pstore.SessionStore | None = None
    if save_to is not None:
        store = xjsonst.JsonFileSessionStore(save_to)
    elif recorder is not None:
        store = xmemsto.MemorySessionStore()

    budget = csessn.Budget(
        max_passes=settings.max_passes,
        max_rounds=settings.max_rounds,
        max_questions_total=0 if headless else max_questions,
        max_questions_per_round=settings.max_questions_per_round,
    )

    console.print(_banner(settings, routing, config_file, root))
    try:
        compiled = capi.clarify(
            prompt,
            reasoner=reasoner,
            retriever=retriever,
            clarifier=clarifier,
            root=root,
            budget=budget,
            store=store,
            session_id=session_id,
        )
    except cerrors.CruxError as exc:
        console.print(f"[red]{exc}[/red]")
        if recorder is not None:
            last = store.load(session_id) if store is not None else None
            where = recorder.write(last, None, error=str(exc))
            console.print(f"[dim]trace written to {where}[/dim]")
        raise typer.Exit(code=1) from exc

    if recorder is not None and store is not None:
        last = store.load(session_id)
        if last is not None:
            where = recorder.write(last, compiled)
            console.print(f"[dim]trace written to {where}[/dim]")

    console.print()
    console.print(rich.syntax.Syntax(compiled.render(), "markdown", word_wrap=True))
    console.print(
        f"\n[dim]{len(compiled.decided)} decided · "
        f"{len(compiled.assumptions)} assumed · "
        f"{len(compiled.delegations)} left to the agent · "
        f"{len(compiled.out_of_scope)} pruned[/dim]"
    )


def _banner(
    settings: isettn.CruxSettings,
    routing: xroute.ModelRouting,
    config_file: pathlib.Path | None,
    root: pathlib.Path,
) -> str:
    """
    Say which models are in play and where the configuration came from.

    Worth the line because crux.toml takes priority over the environment, which
    is the reverse of the usual convention: someone whose exported CRUX_MODEL is
    being ignored should be able to see why without reading the source.

    :param settings: The resolved settings.
    :param routing: The resolved routing.
    :param config_file: The file that was read, if any.
    :param root: The directory being clarified.
    :return: A dim one-or-two-line banner.
    """
    where = f" · config {config_file}" if config_file else ""
    if routing.is_uniform:
        models = settings.model
    else:
        models = f"{routing.reasoning} + {routing.mechanical} (weak)"
    chain = (
        f" · fallback {' -> '.join(settings.fallback_models)}" if settings.fallback_models else ""
    )
    return f"[dim]crux · {models}{chain} · {root}{where}[/dim]\n"


@app.command(name="config", help="Show the resolved configuration and where it came from.")
def show_config(
    config: Annotated[
        pathlib.Path | None,
        typer.Option("--config", help="A crux.toml to use instead of searching for one."),
    ] = None,
) -> None:
    """
    Print every setting that matters, with its source.

    :param config: An explicit crux.toml.
    """
    console = rich.console.Console()
    isettn.load_dotenv()
    config_file = config or isettn.find_config_file()
    settings = isettn.CruxSettings(config_file=config_file)
    routing = xroute.ModelRouting.of(settings)

    console.print(f"[bold]config file[/bold]  {config_file or '(none found)'}")
    console.print(
        "[dim]precedence: command-line flag, then crux.toml, then the "
        "environment, then defaults[/dim]\n"
    )
    console.print("[bold]models by operation[/bold]")
    for operation in (*xroute.REASONING, *xroute.MECHANICAL):
        tier = "reasoning" if operation in xroute.REASONING else "mechanical"
        override = " [yellow](override)[/yellow]" if operation in routing.overrides else ""
        console.print(
            f"  {operation:<16} [cyan]{routing.model_for(operation)}[/cyan]  "
            f"[dim]{tier}[/dim]{override}"
        )
    if settings.fallback_models:
        console.print("\n[bold]fallback chain[/bold]")
        for position, name in enumerate(settings.fallback_models, start=1):
            console.print(f"  {position}. {name}")
        console.print(
            f"  [dim]{settings.fallback_retries} retries each, "
            f"{settings.fallback_deadline:.0f}s deadline for the whole chain[/dim]"
        )
        if settings.api_key is not None or settings.api_base is not None:
            console.print(
                "\n[yellow]CRUX_API_KEY/CRUX_API_BASE are set and will be ignored: "
                "a global key belongs to one provider. Use each provider's own "
                "variable.[/yellow]"
            )
    console.print("\n[bold]limits[/bold]")
    for field in (
        "max_tokens",
        "max_retries",
        "max_passes",
        "max_rounds",
        "max_questions_per_round",
        "reasoning_effort",
    ):
        console.print(f"  {field:<24} {getattr(settings, field)}")


@app.command(name="packs", help="Show the decision packs a prompt would seed from.")
def packs(
    prompt: Annotated[
        str,
        typer.Argument(help="Optional prompt, to see which specs it triggers."),
    ] = "",
) -> None:
    """
    List registered decision packs and what they would seed.

    Useful when an eval case surfaces a decision you did not expect: the pack
    specs carry authored cost and reversibility, and those are what the ask
    threshold reads.

    :param prompt: A prompt to test the triggers against.
    """
    console = rich.console.Console()
    for pack_id in kspec.registered():
        pack = kspec.get(pack_id)
        if pack is None:
            continue
        console.print(f"[bold]{pack.id}[/bold] — {pack.description}")
        seeded = {d.id for d in pack.seed(prompt)} if prompt else set()
        for spec in pack.specs:
            mark = "[green]seeds[/green]" if spec.canonical_id in seeded else "     "
            when = "always" if spec.always else "on trigger"
            console.print(
                f"  {mark} [cyan]{spec.canonical_id}[/cyan]  "
                f"{spec.cost_if_wrong}/{spec.reversibility}, {when}"
            )
            console.print(f"         {spec.undecided}")
            if spec.default:
                console.print(f"         [dim]default: {spec.default}[/dim]")
        console.print()


if __name__ == "__main__":
    app()
