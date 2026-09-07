"""
The shipped retriever: glob, grep and bounded reads over a directory.

Deliberately not an agent. It exists so the README example runs and so crux has
something to test against — not to compete with the explore agent a host already
has. A host that plugs their own retriever in gets everything this does and more,
and that is the intended path.

Import as:

import crux.adapters.retrieval.fs as xfsretr
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import re
from collections.abc import Sequence

import crux.domain.evidence as cevid
import crux.domain.ids as cids

_LOG = logging.getLogger(__name__)

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".next",
        "target",
    }
)

_TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".php",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".cfg",
        ".ini",
        ".env",
        ".sql",
        ".md",
        ".txt",
        ".sh",
        ".dockerfile",
        ".tf",
    }
)

# Manifests answer "what does this project already depend on" more reliably than
# any other file, so a matching one is worth a nudge up the ranking.
#
# README.md was in this list and had to come out: prose that *mentions* Redis
# outranked the module that *uses* it, which is exactly backwards. It still gets
# found on its own merits as an ordinary text file.
_LANDMARKS = (
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "requirements.txt",
    "Gemfile",
    "pom.xml",
)

# Small on purpose. At 2.0 a manifest with one incidental term hit beat an
# implementation file with several, which is how README.md came to be cited as
# the source of a decision about a cache module.
_LANDMARK_BONUS = 1.0

_MAX_FILE_BYTES = 200_000
_EXCERPT_CHARS = 180


class FilesystemRetriever:
    """
    Finds evidence by reading the tree under one root.
    """

    def __init__(self, root: pathlib.Path, *, respect_gitignore: bool = True) -> None:
        """
        :param root: The directory to search.
        :param respect_gitignore: Whether to skip paths named in a root
            ``.gitignore``. Coarse: only literal and single-star patterns.
        """
        self._root = root
        self._ignored = _read_gitignore(root) if respect_gitignore else frozenset()

    async def retrieve(self, briefs: Sequence[cevid.Brief]) -> Sequence[cevid.BriefResult]:
        """
        Answer every brief in the batch concurrently.

        :param briefs: What to find out.
        :return: One result per brief.
        """
        return list(
            await asyncio.gather(*(asyncio.to_thread(self._one, brief) for brief in briefs))
        )

    def _one(self, brief: cevid.Brief) -> cevid.BriefResult:
        """
        Answer one brief.

        :param brief: What to find out.
        :return: What was found, capped at the brief's depth.
        """
        terms = _terms(brief)
        if not terms:
            return cevid.BriefResult(brief_id=brief.id)
        hits: list[tuple[float, str, str]] = []
        for path in self._walk():
            score, excerpt = _score(path, terms)
            if score > 0:
                hits.append((score, str(path.relative_to(self._root)), excerpt))
        hits.sort(key=lambda h: (-h[0], h[1]))
        capped = hits[: brief.max_items]
        items = tuple(
            cevid.EvidenceItem(
                id=cids.item_id(brief.id, index),
                brief_id=brief.id,
                locator=locator,
                kind="file",
                excerpt=excerpt,
                score=score,
                retriever="filesystem",
            )
            for index, (score, locator, excerpt) in enumerate(capped)
        )
        _LOG.info("Brief %s matched %d files, kept %d", brief.id, len(hits), len(items))
        return cevid.BriefResult(
            brief_id=brief.id, items=items, exhausted=len(hits) <= brief.max_items
        )

    def _walk(self) -> list[pathlib.Path]:
        """
        List every readable text file under the root.

        :return: The paths.
        """
        found: list[pathlib.Path] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.name in _LANDMARKS:
                found.append(path)
                continue
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if self._is_ignored(path):
                continue
            found.append(path)
        return found

    def _is_ignored(self, path: pathlib.Path) -> bool:
        """
        :param path: The path to test.
        :return: Whether a gitignore pattern covers it.
        """
        relative = str(path.relative_to(self._root))
        return any(
            relative == pattern or relative.startswith(f"{pattern}/") for pattern in self._ignored
        )


def _terms(brief: cevid.Brief) -> tuple[str, ...]:
    """
    Work out what to grep for.

    :param brief: The brief.
    :return: Significant terms from its query and hints.
    """
    words = cids.tokens(brief.query) | cids.tokens(" ".join(brief.hints))
    return tuple(sorted(w for w in words if len(w) > 2))


def _score(path: pathlib.Path, terms: tuple[str, ...]) -> tuple[float, str]:
    """
    Score one file against the brief's terms.

    A landmark file scores even without a term hit, because "what does this
    project already depend on" is answered by a manifest whether or not the
    manifest happens to contain the query's words.

    :param path: The file to read.
    :param terms: What to look for.
    :return: The score, and the first matching line as an excerpt.
    """
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return 0.0, ""
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0.0, ""

    lowered = text.lower()
    score = 0.0
    excerpt = ""
    for term in terms:
        count = lowered.count(term)
        if count == 0:
            continue
        score += 1.0 + min(count, 5) * 0.2
        if not excerpt:
            excerpt = _first_line(text, term)
    if score == 0.0:
        # A landmark that matched nothing is not evidence. Scoring it anyway put
        # pyproject.toml at the top of every brief, including ones about
        # authentication.
        return 0.0, ""
    if path.name in _LANDMARKS:
        score += _LANDMARK_BONUS
    if terms and terms[0] in path.name.lower():
        score += 0.5
    return score, excerpt


def _first_line(text: str, term: str) -> str:
    """
    :param text: The file's contents.
    :param term: The term that matched.
    :return: The first line containing it, trimmed.
    """
    for line in text.splitlines():
        if term in line.lower():
            return line.strip()[:_EXCERPT_CHARS]
    return ""


def _read_gitignore(root: pathlib.Path) -> frozenset[str]:
    """
    Read the root gitignore, coarsely.

    :param root: The directory to look in.
    :return: Literal path prefixes to skip.
    """
    path = root / ".gitignore"
    if not path.is_file():
        return frozenset()
    patterns: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or re.search(r"[*?\[]", stripped):
            continue
        patterns.add(stripped.strip("/"))
    return frozenset(patterns)
