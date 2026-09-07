# Coding style

<!-- toc -->

- [The gate](#the-gate)
- [Imports](#imports)
- [Typing](#typing)
- [Paths](#paths)
- [Docstrings](#docstrings)
- [Comments](#comments)
- [Logging](#logging)
- [Errors](#errors)
- [Suppressions](#suppressions)
- [Secrets](#secrets)
- [Line length](#line-length)

<!-- tocstop -->

The rules the linter cannot check for you. Everything it _can_ check —
formatting, import order, line length — `ruff` applies, and the right response to
a disagreement with it is to change the code.

These guidelines started from the convalesce engine's, which in turn started from
the causify toolchain. They are rewritten here from what this repo actually
enforces rather than copied, so a rule that stops being true stops being written
down. The helpers-library rules (`hdbg.dassert`, `cunitest.TestCase`, the
`make lint` gate) are gone because we vendor no helpers.

## The gate

```bash
ruff format . && ruff check --fix . && mypy
```

Must report nothing and change nothing on a second run. Run it on every file you
touch before you finish, and run it _before_ reading a file back: `ruff format`
rewrites formatting and import order, so what you last wrote is not what is on
disk.

## Imports

- Import modules, never names: `import crux.domain.decisions as cdecis`, not
  `from crux.domain.decisions import OpenDecision`. The exceptions are `typing`,
  `dataclasses`, `collections.abc`, and `crux/__init__.py`.
- Every module has one canonical alias, at most eight characters, listed in
  `CLAUDE.md`. The budget caps the package tree at about four components.
- No relative imports. `ruff` bans them.

Reading `cdecis.OpenDecision` tells you where `OpenDecision` came from at the
point of use. Reading `OpenDecision` does not, and in a repo with a `domain`, an
`application` and an `adapters` layer that each have their own vocabulary for the
same nouns, that matters.

## Typing

- Modern spelling: `X | None`, `list[X]`, `dict[K, V]`, `tuple[X, ...]`.
- `Literal[...]` for closed sets. There are a lot of them here — decision types,
  resolution sources, grades, phases — and every one is a `Literal`, never a
  bare `str` and never an `Enum`. A `Literal` round-trips through JSON without a
  custom encoder, which the serialisation invariant needs.
- `Any` only where a value crosses a boundary we do not own — a litellm response,
  a host's retriever return — and narrow it immediately.

The convalesce guide requires `typing.List` and `Optional[X]` because the causify
parsers predate PEP 604 and 585. We have no such parsers, run Python 3.12, and
select ruff's `UP` rules, so the newer spelling is the one that passes the gate.

## Paths

`pathlib.Path`, not `os.path`.

## Docstrings

reST, triple quotes on their own lines, `:param x:` / `:return:` / `:raises E:`.
No types in `:param:` — the annotation already says it, and a type written twice
is a type that will disagree with itself.

Every public function, method, class and module has one. A module docstring ends
with its import line:

```python
"""
The open-decision primitive and everything that closes one.

Import as:

import crux.domain.decisions as cdecis
"""
```

A test method's docstring says what it tests, not what it does.

## Comments

Above the line, capitalised, ending in a period, and explaining _why_. A comment
that restates the code is noise that has to be maintained; a comment that records
a decision is the only place that decision survives.

Where a rule was measured rather than assumed, say so. "A grep for a common term
returns ~20 files, so counting raw hits escalated every decision; counting
adjudicated candidates does not" is worth more than any amount of description.

## Logging

- `_LOG = logging.getLogger(__name__)` in every module that logs.
- Lazy `%` formatting: `_LOG.info("Expanded %d decisions", count)`. Never an
  f-string in a log call — it is formatted even when the level is off, and it
  defeats structured handlers.
- Never log a credential, a respondent's free-text answer, or a retrieved
  excerpt. The first is obvious; the second and third are user content that a
  host may be obliged not to persist.

## Errors

- Custom exceptions, always, all descending from `cerrors.CruxError`. Never a
  bare `assert` for a runtime condition — asserts vanish under `-O`.
- Catch specific exceptions. Never a bare `except:`, never a bare
  `except Exception:` outside an adapter boundary.
- An exception takes a formatted string naming what was wrong and, where the
  reader can act, what to do about it.
- A host-facing error says what to do. An internal one says what happened.

## Suppressions

Symbolic, never a bare code, and always with a comment saying why:

```python
# litellm's stub types the tool-call argument as str, but the runtime hands
# back a parsed dict when the provider supports native tool use. Measured
# against Anthropic and OpenAI; both return dict.
tool_args = cast(dict[str, Any], call.function.arguments)  # type: ignore[arg-type]
```

A suppression without a reason is a bug someone will restore.

## Secrets

- Every secret field is a `pydantic.SecretStr`.
- Only `crux.infra.settings` and `crux.adapters.llm.litellm` may call
  `get_secret_value()`.
- A `Session` never holds one. It is serialised to a host's storage by design,
  and redaction covers logs and serialisation, not memory.

## Line length

100 columns, enforced by `ruff format` and `ruff check`.
