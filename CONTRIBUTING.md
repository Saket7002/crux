# Contributing

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Thanks for looking. crux is early, so the most useful contributions right now are
the ones that tell us something we do not know: a measurement, a provider that
behaves differently, a case where the ask threshold gets it wrong.

## The gate

Four commands. Run them before you finish, and run them **before reading a file
back**, because `ruff format` rewrites what you just wrote.

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check --fix .
uv run mypy
uv run pytest -m "not live"
```

The gate must report nothing and change nothing on a second run. CI checks that
separately, because a formatter that keeps finding new work turns every review
into an argument about whitespace.

## House rules that are not obvious

Read [`docs/code_guidelines/`](docs/code_guidelines/) before writing code. Three
things surprise people:

**Import modules, never names.** `import crux.domain.decisions as cdecis`, not
`from crux.domain.decisions import OpenDecision`. Every module has one canonical
alias, at most eight characters, and it is listed in
[`CLAUDE.md`](CLAUDE.md#import-aliases). Adding a module means adding a row. The
point is that `cdecis.OpenDecision` says where the type came from at the place it
is used, in a repo where three layers have their own vocabulary for the same
nouns.

**Modern typing.** `X | None`, `list[X]`, `tuple[X, ...]`. Every closed set is a
`Literal`, never an `Enum`, because a `Session` has to round-trip through a
host's JSON storage with no crux code on the other side.

**Docstrings say why, comments say why.** A comment that restates the code is
noise somebody has to maintain. A comment that records a decision is the only
place that decision survives. Where a rule was measured rather than assumed, say
so: several comments in this repo name the number that produced them.

## Tests

Three tiers, and which one a test belongs in is decided by what it fakes.

| Tier | Location | Fakes | Catches |
|---|---|---|---|
| 1 | `tests/unit/domain/` | nothing | routing, graph operations, dedup, output assembly |
| 2 | `tests/unit/application/` | `FakeReasoner` | phase transitions, loop guards, budgets |
| 3 | `tests/unit/adapters/` | `ScriptedLlm` | prompt construction, response parsing, schema drift |

Never write a tier-2 test that hand-writes tool-call JSON. That is what tier 3 is
for, and mixing them means a prompt edit breaks two hundred logic tests.

**Name tests after the claim**, not the function:
`test_counter_question_terminates_after_two_exchanges`. These names are the
specification.

**Every test has a docstring saying what is tested and why it matters.** Where a
test exists because something once went wrong, say so. Several tests in this repo
name the live run that produced them, and that context is the most valuable thing
in the file.

**Prove a test can fail.** Remove the fix, watch it fail, put it back. A test that
has never been seen to fail is a guess.

## Evals are not tests

`tests/evals/` hits a real model, is marked `live`, and is excluded from CI. It
measures prompt quality and decision recall, which move for reasons that are not
regressions. Assertions there are on the aggregate, never per case.

If you change a prompt, say what happened to the ask rate. That number is the
product.

## What is most wanted

- **Another provider.** crux is meant to run on whatever model a host already
  has. Every provider we have tried behaved differently in a way worth writing
  down: one validates tool schemas server-side and rejects a whole batch over one
  nested field, one counts reserved rather than used tokens, one answers in prose
  where a schema was demanded. If yours does something new, that is a good issue.
- **A decision pack.** `src/crux/packs/software.py` is the only one. Packs are
  declarative data, and their `cost_if_wrong` and `reversibility` are **authored
  by a person**, which is the only calibration the ask threshold has.
- **Evidence that the graph is ceremony.** The conditional-edge machinery is on
  probation, and `tests/evals/saturation.py` carries the falsification criteria.
  A run showing edges never fire is a genuinely useful contribution, and deleting
  code is a fine outcome.

## Pull requests

Small and single-purpose. Say what you measured, not only what you changed.

If you found something by running crux for real, put the number in the PR. Most
of the sharper bugs in this repo were invisible to mocked tests and only showed up
against a live provider.
