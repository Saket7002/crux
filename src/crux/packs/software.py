"""
The software-tasks pack: the five decisions that recur in almost every code ask.

Every cost and reversibility here is **authored**, deliberately. These five are
the calibration anchor the ask threshold rests on, and a freeform decision the
expander invents inherits its grading from whichever of these it most resembles.
Change a value here and you change how often crux interrupts people.

Import as:

import crux.packs.software as ksoft
"""

from __future__ import annotations

import crux.packs.base as kspec

BUILD_SCOPE = kspec.DecisionSpec(
    canonical_id="software.scope.build",
    undecided="Build scope: a minimal working version, or production-grade with "
    "retries, dead-lettering and rate limiting",
    type="underspecification",
    # Highest-leverage question in the pack. It is a threefold difference in
    # effort, nobody can infer it from a repo, and answering it early prunes
    # whole branches before they are ever expanded.
    cost_if_wrong="high",
    reversibility="hard",
    options=("mvp", "production"),
    always=True,
)

TARGET_FILES = kspec.DecisionSpec(
    canonical_id="software.target.files",
    undecided="Which part of the codebase this change belongs in",
    type="missing_context",
    # Almost always answerable from the repo, and cheap to correct in review when
    # retrieval gets it wrong.
    cost_if_wrong="medium",
    reversibility="easy",
    retrieval_hint="entry points, route definitions, module layout",
    accept_if="one directory or module clearly owns this concern",
    always=True,
)

BREAKING_CHANGES = kspec.DecisionSpec(
    canonical_id="software.compat.breaking",
    undecided="Whether breaking existing callers is acceptable",
    type="underspecification",
    # The definition of expensive and irreversible: once a consumer breaks, the
    # cost lands on somebody who never saw this prompt.
    cost_if_wrong="high",
    reversibility="hard",
    options=("additive only", "breaking changes acceptable"),
    triggers=(
        r"\b(api|endpoint|schema|interface|migrat|refactor|rename|signature|"
        r"contract|version)\b",
    ),
)

TEST_EXPECTATIONS = kspec.DecisionSpec(
    canonical_id="software.tests.expectations",
    undecided="What test coverage this change should ship with",
    type="underspecification",
    cost_if_wrong="low",
    reversibility="easy",
    default="follow the conventions already in the repo's test suite",
    retrieval_hint="test directory, test framework, existing test conventions",
    accept_if="the repo's test framework and layout are identifiable",
    always=True,
)

DEPENDENCY_POLICY = kspec.DecisionSpec(
    canonical_id="software.deps.policy",
    undecided="Whether adding a new third-party dependency is acceptable",
    type="underspecification",
    # Adding one is easy; removing one after it has spread is not. Medium cost
    # rather than high because the blast radius is contained to this change.
    cost_if_wrong="medium",
    reversibility="hard",
    options=("prefer the standard library", "a well-maintained dependency is fine"),
    default="prefer what the project already depends on",
    retrieval_hint="dependency manifest, lockfile, existing libraries in use",
    accept_if="the project's existing dependencies are identifiable",
    always=True,
)

PACK = kspec.DecisionPack(
    id="software",
    description="Decisions that recur in almost every software change",
    specs=(
        BUILD_SCOPE,
        TARGET_FILES,
        BREAKING_CHANGES,
        TEST_EXPECTATIONS,
        DEPENDENCY_POLICY,
    ),
)

kspec.register(PACK)
