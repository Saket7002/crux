"""
Run traceability: what crux did, what it was told, and what it produced.

A clarification is a chain of judgements, and "why did it ask me that?" is the
question people actually have. The session answers it in principle — it holds the
graph, the transcript and the pass log — but only if someone deserialises it.
This writes the same information out in a form a person can read, alongside the
raw model exchanges that produced it.

Recording is off unless a run asks for it. It writes prompts and retrieved
excerpts to disk, which is exactly what you want when debugging and exactly what
you do not want happening silently.

Import as:

import crux.infra.evidence as ievid
"""

from __future__ import annotations

import datetime as dt
import pathlib
from typing import Any

import pydantic

import crux.domain.output as coutput
import crux.domain.session as csessn


class CallRecord(pydantic.BaseModel):
    """
    One exchange with the model.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    index: int
    operation: str
    model: str
    messages: tuple[dict[str, Any], ...] = ()
    forced_tool: str | None = None
    response: dict[str, Any] = pydantic.Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str = ""
    attempt: int = 0
    """Position in the fallback chain. Non-zero means an earlier model failed."""


class RunRecorder:
    """
    Collects everything one run did, then writes it out as a directory.
    """

    def __init__(self, root: pathlib.Path, *, session_id: str) -> None:
        """
        :param root: Directory to write runs under.
        :param session_id: The session being recorded.
        """
        stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d-%H%M%S")
        self.directory = root / f"{stamp}-{session_id[:8]}"
        self._calls: list[CallRecord] = []

    def record_call(
        self,
        *,
        operation: str,
        model: str,
        messages: list[dict[str, Any]],
        forced_tool: str | None,
        response: dict[str, Any] | None = None,
        elapsed_seconds: float = 0.0,
        error: str = "",
        attempt: int = 0,
    ) -> None:
        """
        Keep one model exchange.

        :param operation: Which reasoner operation this served.
        :param model: The model string.
        :param messages: What was sent.
        :param forced_tool: The tool the model was made to call.
        :param response: What came back.
        :param elapsed_seconds: How long it took.
        :param error: What went wrong, when something did.
        :param attempt: Position in the fallback chain this attempt sat at.
        """
        self._calls.append(
            CallRecord(
                index=len(self._calls) + 1,
                operation=operation,
                model=model,
                messages=tuple(messages),
                forced_tool=forced_tool,
                response=response or {},
                elapsed_seconds=elapsed_seconds,
                error=error,
                attempt=attempt,
            )
        )

    @property
    def call_count(self) -> int:
        """
        :return: How many model exchanges were recorded.
        """
        return len(self._calls)

    def write(
        self,
        session: csessn.Session | None = None,
        compiled: coutput.CompiledPrompt | None = None,
        *,
        error: str = "",
    ) -> pathlib.Path:
        """
        Write the whole run to disk.

        Called even when a run failed — a failed run is the one you most want a
        trace of, and the calls leading up to it are the evidence.

        :param session: The session as it ended, when there is one. A run that
            died before completing a round never got saved, and the recorded
            calls are then the only evidence there is — so this is optional.
        :param compiled: The output, when the run got that far.
        :param error: What killed the run, when something did.
        :return: The directory written.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        if session is not None:
            (self.directory / "session.json").write_text(
                session.model_dump_json(indent=2), encoding="utf-8"
            )
        if compiled is not None:
            (self.directory / "compiled.md").write_text(compiled.render(), encoding="utf-8")
        calls = self.directory / "calls"
        calls.mkdir(exist_ok=True)
        for call in self._calls:
            name = f"{call.index:02d}-{call.operation}{'-failed' if call.error else ''}.json"
            (calls / name).write_text(call.model_dump_json(indent=2), encoding="utf-8")
        (self.directory / "run.md").write_text(
            self._narrate(session, compiled, error), encoding="utf-8"
        )
        return self.directory

    def _narrate(
        self,
        session: csessn.Session | None,
        compiled: coutput.CompiledPrompt | None,
        error: str,
    ) -> str:
        """
        Write the run as prose a person can follow.

        :param session: The session as it ended.
        :param compiled: The output, when there was one.
        :param error: What killed the run, when something did.
        :return: Markdown.
        """
        if session is None:
            return self._narrate_callsonly(error)
        lines = [
            f"# Run {session.id}",
            "",
            f"- **Prompt**: {session.prompt}",
            f"- **Root**: {session.context.root or '(none)'}",
            f"- **Packs**: {', '.join(session.context.pack_ids)}",
            f"- **Ended in phase**: `{session.phase}`",
            f"- **Model calls**: {self.call_count}",
            f"- **Questions asked**: {session.budget.spent_questions}",
            f"- **Rounds**: {session.budget.spent_rounds}",
        ]
        if error:
            lines += ["", "## Failed", "", f"```\n{error}\n```"]

        lines += ["", "## Expansion", ""]
        if session.passes.records:
            lines.append("| pass | lens | proposed | new | ratio | saw evidence |")
            lines.append("|---|---|---|---|---|---|")
            for record in session.passes.records:
                lens = "blind" if record.index == 1 else "consequences"
                lines.append(
                    f"| {record.index} | {lens} | {record.proposed} | "
                    f"{record.new_after_dedup} | {record.ratio:.2f} | "
                    f"{'yes' if record.saw_evidence else 'no'} |"
                )
            lines.append("")
            lines.append(
                f"Saturated: **{session.passes.saturated}** "
                f"(a pass yielding under {csessn.SATURATION_RATIO:.0%} new is spent)."
            )
        else:
            lines.append("No expansion pass ran.")

        lines += ["", "## Retrieval", ""]
        if session.evidence.briefs:
            lines.append("| brief | depth | items found |")
            lines.append("|---|---|---|")
            for brief in session.evidence.briefs.values():
                found = len(session.evidence.by_decision.get(brief.decision_id, ()))
                lines.append(f"| `{brief.decision_id}` | {brief.depth} | {found} |")
        else:
            lines.append("No retrieval ran.")

        lines += ["", "## Edges", ""]
        stats = session.graph.edge_stats()
        if stats:
            lines.append("| kind | attached | fired |")
            lines.append("|---|---|---|")
            for kind, (attached, fired) in sorted(stats.items()):
                lines.append(f"| {kind} | {attached} | {fired} |")
            lines.append("")
            lines.append(
                "An edge only fires when its source resolves to a *particular "
                "value*, so a run where nothing was answered fires none."
            )
        else:
            lines.append("No edges were attached.")

        lines += ["", "## Decisions", "", f"{len(session.graph.nodes)} in total.", ""]
        lines.append("| decision | type | cost/rev | closed by | value |")
        lines.append("|---|---|---|---|---|")
        for node in session.graph.nodes.values():
            closed = node.resolution.source if node.resolution else node.status
            value = node.resolution.value if node.resolution else "—"
            lines.append(
                f"| `{node.id}` | {node.type} | {node.cost_if_wrong}/{node.reversibility} "
                f"| {closed} | {value[:60]} |"
            )

        if session.transcript:
            lines += ["", "## What was asked", ""]
            for exchange in session.transcript:
                lines.append(f"**Q:** {exchange.question.text}")
                if exchange.question.options:
                    lines.append(
                        "  - offered: " + ", ".join(o.value for o in exchange.question.options)
                    )
                typed = exchange.reply.text or ", ".join(exchange.reply.selected)
                lines.append(f"  - **they said:** {typed}")
                lines.append(f"  - read as: `{exchange.classified.kind}`")
                lines.append("")

        lines += ["", "## Model calls", ""]
        lines.append("| # | operation | model | elapsed | outcome |")
        lines.append("|---|---|---|---|---|")
        for call in self._calls:
            outcome = f"error: {call.error[:50]}" if call.error else "ok"
            if call.attempt:
                outcome = f"fallback #{call.attempt}, {outcome}"
            lines.append(
                f"| {call.index} | {call.operation} | `{call.model}` | "
                f"{call.elapsed_seconds:.1f}s | {outcome} |"
            )
        lines.append("")
        lines.append("Full prompts and responses are in `calls/`.")

        if compiled is not None:
            lines += [
                "",
                "## Output",
                "",
                f"- decided: {len(compiled.decided)}",
                f"- assumed: {len(compiled.assumptions)}",
                f"- delegated: {len(compiled.delegations)}",
                f"- pruned: {len(compiled.out_of_scope)}",
                "",
                "See `compiled.md`.",
            ]
        return "\n".join(lines) + "\n"

    def _narrate_callsonly(self, error: str) -> str:
        """
        Narrate a run that died before a session was ever saved.

        :param error: What killed it.
        :return: Markdown.
        """
        lines = [
            "# Run failed before any round completed",
            "",
            "No session was saved, so the model exchanges below are the whole "
            "record. They are in `calls/`.",
            "",
            f"```\n{error}\n```",
            "",
            "| # | operation | elapsed | outcome |",
            "|---|---|---|---|",
        ]
        for call in self._calls:
            outcome = f"error: {call.error[:60]}" if call.error else "ok"
            lines.append(
                f"| {call.index} | {call.operation} | `{call.model}` | "
                f"{call.elapsed_seconds:.1f}s | {outcome} |"
            )
        return "\n".join(lines) + "\n"


def summarise_turn(turn: Any) -> dict[str, Any]:
    """
    Reduce a model turn to something worth writing down.

    :param turn: The turn as crux models it.
    :return: A JSON-safe summary including the parsed tool arguments, which are
        the part anyone debugging actually wants.
    """
    return {
        "text": getattr(turn, "text", ""),
        "finish_reason": getattr(turn, "finish_reason", ""),
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments}
            for call in getattr(turn, "tool_calls", ())
        ],
    }
