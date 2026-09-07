"""
Minting ids, and deciding when two proposed decisions are the same decision.

Node identity matters more here than it would in a tree. A duplicate decision
does not merely ask the same question twice — it splits the edges that should
have pointed at one node, so a prune fires against one copy and misses the other.

Identity is exact where it can be and fuzzy only in the tail: a decision seeded
from a pack carries that pack's canonical id, and only freeform decisions the
expander invented need matching by text.

Import as:

import crux.domain.ids as cids
"""

from __future__ import annotations

import re

# Two freeform decisions are the same decision above this overlap. Set high
# because a false merge silently loses a question, which is worse than asking a
# near-duplicate: the merged-away decision never reaches the output at all.
DUPLICATE_THRESHOLD = 0.8

# A freeform decision inherits cost and reversibility from a pack sibling above
# this, much lower bar. Getting this wrong costs a mis-graded decision, not a
# lost one, and an authored grade beats a model-invented one even when the match
# is loose.
SIBLING_THRESHOLD = 0.5

_NON_WORD = re.compile(r"[^a-z0-9]+")

# Words that carry no signal about *what* is undecided. Kept deliberately short:
# an aggressive stoplist makes unrelated decisions look alike, and a false merge
# is the expensive direction.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "by",
        "do",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "should",
        "the",
        "this",
        "to",
        "we",
        "what",
        "which",
        "with",
    }
)


def slug(text: str, *, max_length: int = 48) -> str:
    """
    Reduce free text to something usable as an id fragment.

    :param text: The text to reduce.
    :param max_length: Longest slug to produce.
    :return: A lowercase, hyphen-separated fragment.
    """
    cleaned = _NON_WORD.sub("-", text.strip().lower()).strip("-")
    return cleaned[:max_length].rstrip("-")


def freeform_id(undecided: str, *, pass_index: int) -> str:
    """
    Mint an id for a decision the expander invented.

    Namespaced under ``open.`` so a freeform id can never collide with a pack's
    canonical id, and so the compiled prompt's audit trail shows at a glance
    which decisions were authored and which were generated.

    :param undecided: What the decision says is undecided.
    :param pass_index: Which expansion pass produced it.
    :return: The id.
    """
    return f"open.p{pass_index}.{slug(undecided)}"


def brief_id(decision_id: str) -> str:
    """
    :param decision_id: The decision the brief serves.
    :return: The brief's id.
    """
    return f"brief:{decision_id}"


def item_id(brief: str, index: int) -> str:
    """
    :param brief: The brief the item answers.
    :param index: Position within that brief's results.
    :return: The evidence item's id.
    """
    return f"ev:{brief}:{index}"


def question_id(decision_id: str, *, exchange: int) -> str:
    """
    :param decision_id: The decision being asked about.
    :param exchange: Which round-trip this is, counting from zero.
    :return: The question's id.
    """
    return f"q:{decision_id}:{exchange}"


def tokens(text: str) -> frozenset[str]:
    """
    Reduce text to the words that say what it is about.

    :param text: The text to reduce.
    :return: Its significant tokens, lowercased.
    """
    words = _NON_WORD.sub(" ", text.lower()).split()
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def similarity(left: str, right: str) -> float:
    """
    Score how alike two pieces of text are, by token overlap.

    Jaccard rather than embeddings: at the thirty-node scale a session actually
    reaches, token overlap is enough, and an embedding model would add a network
    dependency and a cache to the domain layer for unproven gain. Revisit when a
    measurement says it is wrong, not before.

    :param left: One text.
    :param right: The other.
    :return: Overlap between 0.0 and 1.0.
    """
    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_duplicate(left: str, right: str, *, threshold: float = DUPLICATE_THRESHOLD) -> bool:
    """
    Say whether two decisions are the same decision worded differently.

    :param left: One decision's ``undecided`` text.
    :param right: The other's.
    :param threshold: Overlap above which they are considered the same.
    :return: Whether to merge them.
    """
    return similarity(left, right) >= threshold


def closest(
    needle: str,
    haystack: dict[str, str],
    *,
    threshold: float = SIBLING_THRESHOLD,
) -> str | None:
    """
    Find the best match for a decision among known ones.

    Used to give a freeform decision the authored cost and reversibility of its
    nearest pack sibling, which is the only calibration anchor crux has.

    :param needle: The text to match.
    :param haystack: Candidate ids mapped to their ``undecided`` text.
    :param threshold: Minimum overlap to count as a match.
    :return: The best-matching id, or ``None`` when nothing clears the bar.
    """
    best_id: str | None = None
    best_score = threshold
    for candidate_id, text in haystack.items():
        score = similarity(needle, text)
        if score >= best_score:
            best_id, best_score = candidate_id, score
    return best_id
