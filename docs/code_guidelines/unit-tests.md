# Unit tests

<!-- toc -->

- [Where and what they are called](#where-and-what-they-are-called)
- [The three tiers](#the-three-tiers)
- [Assertions](#assertions)
- [Docstrings](#docstrings)
- [What to test](#what-to-test)
- [Fakes](#fakes)
- [Timeouts](#timeouts)

<!-- tocstop -->

## Where and what they are called

- Under `tests/`, mirroring the package: `tests/unit/domain/test_graph.py`.
- pytest, with `asyncio_mode = "auto"`. No `unittest.TestCase`, no
  `@pytest.mark.asyncio`.
- Test names are **behavioural**, describing the claim:
  `test_counter_question_terminates_after_two_exchanges`, never `test1`. The
  convalesce guide numbers them so a second case needs no rename; we take the
  readable name instead, because these names are the specification.
- Group with a plain class when a file has several clusters:
  `class TestFrontier:`. No base class.

## The three tiers

Which tier a test belongs in is decided by what it fakes, and each tier catches a
different class of bug.

| Tier | Location | Fakes | Catches |
|---|---|---|---|
| 1 | `tests/unit/domain/` | nothing | frontier, prune, dedup, routing, grade derivation |
| 2 | `tests/unit/application/` | `FakeReasoner`, `FakeRetriever` | phase transitions, loop guards, budget, cascade |
| 3 | `tests/unit/adapters/` | `ScriptedLlm` | prompt construction, response parsing, schema drift |

Tier 1 is most of the correctness and costs nothing to run — the routing table,
the graph operations and the compiler's fact half are pure functions over
pydantic models. If you find yourself wanting to mock something in tier 1, logic
has leaked out of the domain and into an adapter.

Never write a tier-2 test that hand-writes tool-call JSON. That is what tier 3
is for, and mixing them means a prompt edit breaks two hundred logic tests.

Evals live in `tests/evals/`, marked `live`, and are excluded from CI. They hit a
real model and they measure prompt quality, not correctness.

## Assertions

Plain `assert`. Compare the thing itself, not its `repr`:

```python
assert graph.frontier() == ("scope", "channel")
```

For the rendered compiled prompt, assert on the **decision-id set and the
assumption set**, never on the prose. The prose is model output and will drift;
the sets are the contract.

## Docstrings

Every test has one, and it says what is being tested and why that matters. "Test
that a decision with no evidence and no pack default is retyped rather than
dropped" is a rule someone can check the code against. "Test routing" is not.

Where a test exists because something once went wrong, say so. The docstring is
the only place that survives.

## What to test

Test the claim, not the implementation. The useful tests here are the ones that
would fail if a real guarantee broke:

- that every unasked decision reaches the output as a named assumption
- that a `Session` round-trips through JSON at every phase
- that a `Session` holds no port references
- that a grep returning forty hits but two candidate answers escalates once
- that budget exhaustion resolves every remaining decision rather than hanging
- that the counter-question loop terminates in at most two extra exchanges

**Prove a test can fail.** Where a test exists to catch a specific bug, remove
the fix, watch it fail, put the fix back. A test that has never been seen to fail
is a guess.

The routing table gets a `@pytest.mark.parametrize` case per row, plus the
boundaries: dominance ratio exactly 2.0, zero candidates, budget spent mid-table.

## Fakes

No global mutable state. Build the fake in the test and pass it in; a fake that
remembers across tests will pass in isolation and fail in a suite, or worse, the
other way round.

A fake records what it was asked. Half the value of `FakeReasoner` is asserting
that the engine batched — that it made *one* adjudication call for six decisions,
not six calls.

Prefer a real thing where one is reachable. `FilesystemRetriever` against a
`tmp_path` tree is closer to the truth than a stub and costs the same to write.

## Timeouts

The suite runs with `--timeout=60`. A retriever or an LLM adapter that hangs will
otherwise stall a run for ever rather than failing it.
