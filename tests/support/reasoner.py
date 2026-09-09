"""
Fakes for the two seams the engine talks to.

A fake records what it was asked, because half the value of these tests is
asserting that the engine **batched** — one adjudication call for six decisions,
not six calls. A design whose whole economic argument rests on batching needs a
test that would notice if it stopped.

No global state: build one in the test and pass it in.
"""

from __future__ import annotations

from collections.abc import Sequence

import crux.domain.evidence as cevid
import crux.ports.reasoner as preason


class FakeReasoner:
    """
    Returns queued, already-typed results and remembers every request.
    """

    def __init__(
        self,
        *,
        expansions: Sequence[preason.ExpansionResult] = (),
        adjudications: Sequence[preason.AdjudicationResult] = (),
        classifications: Sequence[preason.ClassifyResult] = (),
        counters: Sequence[preason.CounterResult] = (),
        task: str = "",
        constraints: Sequence[str] = (),
        pack_choice: str | None = None,
    ) -> None:
        """
        :param expansions: One result per expansion pass; exhausting the queue
            yields empty results, which is what makes a session saturate.
        :param adjudications: One per retrieval round.
        :param classifications: One per resume that had free text.
        :param counters: One per resume that had a counter-question.
        :param task: Task prose to draft.
        :param constraints: Constraints to draft.
        :param pack_choice: What ``select_pack`` answers. ``None`` answers with
            the first pack offered.
        """
        self._expansions = list(expansions)
        self._adjudications = list(adjudications)
        self._classifications = list(classifications)
        self._counters = list(counters)
        self._task = task
        self._constraints = tuple(constraints)
        self._pack_choice = pack_choice
        self.calls: list[tuple[str, object]] = []

    def count(self, method: str) -> int:
        """
        :param method: The reasoner method to count.
        :return: How many times it was called.
        """
        return sum(1 for name, _ in self.calls if name == method)

    def last(self, method: str) -> object:
        """
        :param method: The reasoner method to look up.
        :return: The most recent request it received.
        """
        return [request for name, request in self.calls if name == method][-1]

    async def expand(self, request: preason.ExpansionRequest) -> preason.ExpansionResult:
        self.calls.append(("expand", request))
        if not self._expansions:
            return preason.ExpansionResult()
        return self._expansions.pop(0)

    async def adjudicate(self, request: preason.AdjudicationRequest) -> preason.AdjudicationResult:
        self.calls.append(("adjudicate", request))
        if not self._adjudications:
            return preason.AdjudicationResult()
        return self._adjudications.pop(0)

    async def phrase(self, request: preason.PhraseRequest) -> preason.PhraseResult:
        self.calls.append(("phrase", request))
        return preason.PhraseResult(
            questions=tuple(
                preason.PhrasedQuestion(
                    decision_id=d.id,
                    text=d.undecided,
                    why=f"cost={d.cost_if_wrong}",
                )
                for d in request.decisions
            )
        )

    async def classify(self, request: preason.ClassifyRequest) -> preason.ClassifyResult:
        self.calls.append(("classify", request))
        if not self._classifications:
            return preason.ClassifyResult()
        return self._classifications.pop(0)

    async def answer_counter(self, request: preason.CounterRequest) -> preason.CounterResult:
        self.calls.append(("answer_counter", request))
        if not self._counters:
            return preason.CounterResult(
                answers=tuple(
                    preason.CounterAnswer(
                        question_id=item.question_id,
                        answer="I do not know.",
                        answered=False,
                    )
                    for item in request.items
                )
            )
        return self._counters.pop(0)

    async def select_pack(self, request: preason.PackSelectRequest) -> preason.PackSelectResult:
        self.calls.append(("select_pack", request))
        chosen = self._pack_choice or (request.packs[0].id if request.packs else "")
        return preason.PackSelectResult(pack_id=chosen, rationale="scripted")

    async def draft(self, request: preason.DraftRequest) -> preason.DraftResult:
        self.calls.append(("draft", request))
        return preason.DraftResult(task=self._task or request.prompt, constraints=self._constraints)


class FakeRetriever:
    """
    Answers briefs from a canned map, and records how many trips it took.
    """

    def __init__(self, items: dict[str, tuple[cevid.EvidenceItem, ...]] | None = None) -> None:
        """
        :param items: Decision id mapped to the items retrieval should find.
        """
        self._items = items or {}
        self.batches: list[tuple[str, ...]] = []

    @property
    def trips(self) -> int:
        """
        :return: How many times the retriever was called at all.

        The number the batching argument lives or dies on: four decisions must
        cost one trip, not four.
        """
        return len(self.batches)

    async def retrieve(self, briefs: Sequence[cevid.Brief]) -> Sequence[cevid.BriefResult]:
        self.batches.append(tuple(b.id for b in briefs))
        return [
            cevid.BriefResult(
                brief_id=brief.id,
                items=self._items.get(brief.decision_id, ()),
            )
            for brief in briefs
        ]


def item(decision_id: str, index: int, locator: str, excerpt: str = "") -> cevid.EvidenceItem:
    """
    Build one evidence item.

    :param decision_id: The decision it answers.
    :param index: Position within that decision's results.
    :param locator: Where it was found.
    :param excerpt: What it says.
    :return: The item.
    """
    return cevid.EvidenceItem(
        id=f"ev:{decision_id}:{index}",
        brief_id=f"brief:{decision_id}",
        locator=locator,
        kind="file",
        excerpt=excerpt,
    )


def proposal(
    undecided: str,
    *,
    ref: str = "",
    decision_type: str = "underspecification",
    reversibility: str = "hard",
    options: Sequence[str] = (),
    edges: Sequence[preason.ProposedEdge] = (),
    retrieval_hint: str = "",
) -> preason.ProposedDecision:
    """
    Build one expansion proposal.

    :param undecided: What it says is undecided.
    :param ref: Short label a sibling can depend on.
    :param decision_type: Its type.
    :param reversibility: How hard it is to change later.
    :param options: Candidate answers.
    :param edges: Dependencies it claims.
    :param retrieval_hint: Where to look, if anywhere.
    :return: The proposal.
    """
    return preason.ProposedDecision.model_validate(
        {
            "ref": ref,
            "undecided": undecided,
            "type": decision_type,
            "reversibility": reversibility,
            "options": tuple(options),
            "edges": tuple(edges),
            "retrieval_hint": retrieval_hint,
        }
    )
