# Type hints

<!-- toc -->

- [The spelling](#the-spelling)
- [Annotate everything](#annotate-everything)
- [`Literal` for closed sets](#literal-for-closed-sets)
- [Frozen models and tuples](#frozen-models-and-tuples)
- [`Any` is a boundary marker](#any-is-a-boundary-marker)
- [Protocols over base classes](#protocols-over-base-classes)
- [Discriminated unions](#discriminated-unions)
- [`cast` sparingly, and say why](#cast-sparingly-and-say-why)

<!-- tocstop -->

mypy runs in strict mode over `src` and `tests`, with the pydantic plugin and
`warn_unreachable`. That decides most of what follows.

## The spelling

Modern generics, always:

```python
def frontier(self, *, limit: int | None = None) -> tuple[str, ...]: ...
```

Not `List[str]`, not `Optional[X]`, not `Dict[K, V]`. ruff's `UP` rules rewrite
them, so the older spelling fails the gate.

## Annotate everything

Every parameter and every return, including `-> None`. Strict mode requires it,
and an unannotated function is invisible to the checker rather than merely
undocumented.

## `Literal` for closed sets

Decision types, resolution sources, grades, phases, edge kinds, statuses — every
closed set is a `Literal`, never a bare `str` and never an `Enum`:

```python
DecisionType = Literal["ambiguity", "underspecification", "vagueness", "missing_context"]
```

An `Enum` needs a custom encoder to survive `model_dump_json()`. A `Literal` is
already a string on the wire, which is what the serialisation invariant needs —
a `Session` must round-trip through a host's JSON storage with no crux code on
the other side.

## Frozen models and tuples

Leaf value objects are `frozen=True` and hold `tuple[X, ...]`, not `list[X]`.
Two reasons, both load-bearing: a frozen model is hashable, so decisions can go
in sets during dedup; and `list` is invariant, so a function taking
`list[Option]` will not accept a `list[PackOption]`. Take `Sequence` in a
parameter position unless you intend to mutate.

Aggregates that genuinely change — `DecisionGraph`, `Session` — are mutable by
copy: every operation returns a new one.

## `Any` is a boundary marker

Use it where a value crosses into code we do not own — litellm's response, a
host's `Retriever` return, a YAML eval case — and narrow it as soon as it is
inside:

```python
def to_items(raw: Sequence[Mapping[str, Any]]) -> tuple[cevid.EvidenceItem, ...]:
```

What comes in is `Any`. What comes out is not.

## Protocols over base classes

Where a thing is defined by what it can do rather than what it is — a retriever,
a reasoner, a session store — a `Protocol` says so without forcing an import
relationship that would run the wrong way. All five ports are
`@runtime_checkable` Protocols, and the application layer never imports an
adapter.

`DecisionPack` is the deliberate exception: it is declarative data, not a
Protocol, because a pack with callables in it could not be serialised and would
let a host smuggle I/O into the domain.

## Discriminated unions

Where a value is one of several shapes, use a `Literal` tag plus
`Field(discriminator=...)`:

```python
ClassifiedReply = Annotated[
    ChoiceReply | ValueReply | CounterQuestionReply | ProposalReply | DeferReply | RejectReply,
    Field(discriminator="kind"),
]
```

pydantic then validates the right arm without a try-each-in-turn pass, mypy
narrows on `match reply.kind:`, and a missing arm is a type error rather than a
runtime surprise.

## `cast` sparingly, and say why

A `cast` is an assertion the checker cannot verify. When one is needed, the
comment says what makes it true:

```python
# The router only ever calls this on a decision it has already checked is
# resolved, so the Optional is not one here.
resolution = cast(cdecis.Resolution, decision.resolution)
```
