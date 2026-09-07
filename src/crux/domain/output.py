"""
The compiled prompt: what the downstream agent actually receives.

The honesty of this output is a property of the code, not of a model. A model
writes the task prose and the constraints and nothing else; the assumptions, the
delegations and the citations are *derived* from the graph by
:meth:`CompiledPrompt.assemble`. A model asked to list its own assumptions will
list the flattering ones.

Import as:

import crux.domain.output as coutput
"""

from __future__ import annotations

import datetime as dt

import pydantic

import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import crux.domain.graph as cgraph

# How many open judgements the rendered prompt lists before summarising the rest.
#
# Measured rather than assumed: the first live run against a real model produced
# twenty-two, which is not a prompt, it is a decision dump. Every one of them is
# still in `delegations` and `all_decisions` for a caller that wants them — this
# caps what a *reader* is asked to hold, not what crux recorded.
MAX_RENDERED_DELEGATIONS = 8

_COST_RANK = {"high": 0, "medium": 1, "low": 2}


class DecisionRecord(pydantic.BaseModel):
    """
    One resolved decision, flattened for the output and the audit trail.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    undecided: str
    type: cdecis.DecisionType
    value: str
    source: cdecis.ResolutionSource
    grade: cdecis.Grade
    cost_if_wrong: cdecis.Cost = "medium"
    reversibility: cdecis.Reversibility = "easy"
    rationale: str = ""
    locators: tuple[str, ...] = ()
    options_considered: tuple[str, ...] = ()

    def line(self) -> str:
        """
        Render one bullet, with its provenance and any citation.

        :return: The rendered line.
        """
        tag = f"[{self.source}]" if self.grade != "inferred" else "[retrieved]"
        cited = f" ({', '.join(self.locators)})" if self.locators else ""
        return f"- {self.undecided} → {self.value}{cited}   {tag}"


class Citation(pydantic.BaseModel):
    """
    A place in the world that a resolution leaned on.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    locator: str
    excerpt: str = ""
    decision_ids: tuple[str, ...] = ()


class CompiledPrompt(pydantic.BaseModel):
    """
    The output. Every line carries where it came from.
    """

    task: str
    constraints: tuple[str, ...] = ()
    decided: tuple[DecisionRecord, ...] = ()
    assumptions: tuple[DecisionRecord, ...] = ()
    delegations: tuple[DecisionRecord, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    context: tuple[Citation, ...] = ()
    all_decisions: tuple[DecisionRecord, ...] = ()

    session_id: str = ""
    compiled_at: dt.datetime = pydantic.Field(
        default_factory=lambda: dt.datetime.now(tz=dt.UTC),
    )

    @staticmethod
    def assemble(
        *,
        task: str,
        constraints: tuple[str, ...],
        graph: cgraph.DecisionGraph,
        evidence: cevid.EvidenceGraph,
        session_id: str = "",
    ) -> CompiledPrompt:
        """
        Build the output from the graph, taking only prose from the model.

        The split is the point: ``task`` and ``constraints`` are written, and
        every other section is computed. A decision cannot be resolved without
        appearing here, and it cannot appear under a stronger heading than its
        grade allows.

        :param task: The task prose the model drafted.
        :param constraints: The constraints the model drafted.
        :param graph: The resolved decision graph.
        :param evidence: The session's evidence, for citations.
        :param session_id: The session this came from.
        :return: The compiled prompt.
        """
        decided: list[DecisionRecord] = []
        assumptions: list[DecisionRecord] = []
        delegations: list[DecisionRecord] = []
        every: list[DecisionRecord] = []
        cited: dict[str, list[str]] = {}

        for node in graph.nodes.values():
            if not node.is_resolved or node.resolution is None:
                continue
            resolution = node.resolution
            locators = evidence.locators(resolution.evidence)
            for locator in locators:
                cited.setdefault(locator, []).append(node.id)
            record = DecisionRecord(
                id=node.id,
                undecided=node.undecided,
                type=node.type,
                value=resolution.value,
                source=resolution.source,
                grade=resolution.grade,
                cost_if_wrong=node.cost_if_wrong,
                reversibility=node.reversibility,
                rationale=resolution.rationale,
                locators=locators,
                options_considered=tuple(o.value for o in (node.options or ())),
            )
            every.append(record)
            if resolution.source == "delegated":
                delegations.append(record)
            elif resolution.grade in ("confirmed", "inferred"):
                decided.append(record)
            else:
                assumptions.append(record)

        # Pruned decisions, plus ones that were never born because a `requires`
        # edge did not fire. Both are genuinely out of scope, and both have to be
        # SAID.
        #
        # Leaving the unborn ones out was silent data loss: a live run attached
        # 121 edges waiting on values their source could never take, every one of
        # them suppressed a decision, and none of those decisions appeared
        # anywhere in the output. It looked like the graph was saving questions.
        # It was deleting them.
        out_of_scope: list[str] = []
        for node in graph.nodes.values():
            if node.status == "pruned":
                out_of_scope.append(node.undecided)
            elif node.status == "open" and not graph.is_live(node):
                out_of_scope.append(f"{node.undecided} ({_unborn_reason(node, graph)})")
        context = tuple(
            Citation(
                locator=locator,
                excerpt=_excerpt_for(evidence, locator),
                decision_ids=tuple(ids),
            )
            for locator, ids in sorted(cited.items())
        )
        return CompiledPrompt(
            task=task,
            constraints=constraints,
            decided=tuple(decided),
            assumptions=tuple(assumptions),
            delegations=tuple(delegations),
            out_of_scope=tuple(out_of_scope),
            context=context,
            all_decisions=tuple(every),
            session_id=session_id,
        )

    def render(self) -> str:
        """
        Render the prompt a downstream agent reads.

        :return: Markdown.
        """
        parts = [f"# Task\n{self.task.strip()}"]
        if self.constraints:
            parts.append("## Constraints\n" + "\n".join(f"- {c}" for c in self.constraints))
        if self.decided:
            parts.append("## Decided\n" + "\n".join(r.line() for r in self.decided))
        if self.assumptions:
            parts.append(
                "## Assumptions — I guessed these. Correct me if wrong.\n"
                + "\n".join(r.line() for r in self.assumptions)
            )
        if self.delegations:
            parts.append(self._render_delegations())
        if self.out_of_scope:
            parts.append("## Out of scope\n" + "\n".join(f"- {s}" for s in self.out_of_scope))
        if self.context:
            parts.append("## Read first\n" + "\n".join(f"- {c.locator}" for c in self.context))
        return "\n\n".join(parts) + "\n"

    def _render_delegations(self) -> str:
        """
        Render the open judgements, worst first, and stop before the reader does.

        :return: The rendered section.
        """
        ranked = sorted(
            self.delegations,
            key=lambda r: (
                _COST_RANK[r.cost_if_wrong],
                0 if r.reversibility == "hard" else 1,
                r.undecided,
            ),
        )
        shown = ranked[:MAX_RENDERED_DELEGATIONS]
        lines = [_delegation_line(record) for record in shown]
        remaining = len(ranked) - len(shown)
        if remaining:
            lines.append(
                f"- ...and {remaining} lower-stakes judgements, listed in full in "
                f"the session's decision record."
            )
        return "## Left to you\n" + "\n".join(lines)


def _unborn_reason(node: cdecis.OpenDecision, graph: cgraph.DecisionGraph) -> str:
    """
    Say why a decision never came into existence.

    Naming the unmet condition is what makes a nonsense edge visible. An edge
    waiting on a value its source cannot take reads here as "only applies if X
    is <something the reader knows X can never be>", which is a bug report.

    :param node: The decision that was never born.
    :param graph: The graph, for reading what its sources resolved to.
    :return: A short clause naming the condition that did not hold.
    """
    unmet = [
        edge for edge in node.edges("requires") if not edge.fires(graph.value_of(edge.source_id))
    ]
    if not unmet:
        return "prerequisite not met"
    edge = unmet[0]
    source = graph.nodes.get(edge.source_id)
    subject = source.undecided if source is not None else edge.source_id
    actual = graph.value_of(edge.source_id)
    if actual is None:
        return f"only applies once {subject} is settled"
    return f"only applies if {subject} is {edge.when_value!r}, and it is {actual!r}"


def _delegation_line(record: DecisionRecord) -> str:
    """
    Render a delegation, naming what was considered so the agent need not
    rediscover the option space.

    :param record: The delegated decision.
    :return: The rendered line.
    """
    considered = (
        f" — considered: {', '.join(record.options_considered)}"
        if record.options_considered
        else ""
    )
    return f"- {record.undecided}{considered}"


def _excerpt_for(evidence: cevid.EvidenceGraph, locator: str) -> str:
    """
    Find the first excerpt recorded at a locator.

    :param evidence: The session's evidence.
    :param locator: The place to look up.
    :return: The excerpt, or an empty string when there is none.
    """
    for item in evidence.items.values():
        if item.locator == locator and item.excerpt:
            return item.excerpt
    return ""
