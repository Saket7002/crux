# crux — rules the linter cannot check

`crux` turns an underspecified prompt into a compiled prompt whose every guess is
visible. It is a library embedded in someone else's agent system, never an app.

Read `docs/code_guidelines/` before writing code. What follows is the alias table
and the handful of invariants that have no test-shaped form.

## Import aliases

Import modules, never names. Every module has exactly one canonical alias, at
most eight characters, and it is listed here. `crux/__init__.py` is the one
sanctioned place for `from X import Y`, because it is the public surface.

| Module | Alias |
|---|---|
| `crux.api` | `capi` |
| `crux.errors` | `cerrors` |
| `crux.domain.decisions` | `cdecis` |
| `crux.domain.graph` | `cgraph` |
| `crux.domain.evidence` | `cevid` |
| `crux.domain.replies` | `creply` |
| `crux.domain.session` | `csessn` |
| `crux.domain.output` | `coutput` |
| `crux.domain.ids` | `cids` |
| `crux.ports.reasoner` | `preason` |
| `crux.ports.llm` | `pllm` |
| `crux.ports.retrieval` | `pretr` |
| `crux.ports.clarifier` | `pclarif` |
| `crux.ports.store` | `pstore` |
| `crux.application.engine` | `aengine` |
| `crux.application.expansion` | `aexpand` |
| `crux.application.retrieval` | `aretr` |
| `crux.application.routing` | `aroute` |
| `crux.application.questions` | `aquest` |
| `crux.application.replies` | `arepl` |
| `crux.application.stopping` | `astop` |
| `crux.application.compile` | `acomp` |
| `crux.packs.base` | `kspec` |
| `crux.packs.software` | `ksoft` |
| `crux.adapters.llm.reasoner` | `xreason` |
| `crux.adapters.llm.litellm` | `xlitell` |
| `crux.adapters.llm.prompts` | `xprompt` |
| `crux.adapters.llm.routing` | `xroute` |
| `crux.adapters.retrieval.fs` | `xfsretr` |
| `crux.adapters.clarifier.tty` | `xtty` |
| `crux.adapters.store.memory` | `xmemsto` |
| `crux.adapters.store.jsonfile` | `xjsonst` |
| `crux.infra.evidence` | `ievid` |
| `crux.infra.settings` | `isettn` |
| `crux.infra.logging` | `ilog` |
| `crux.cli.main` | `ccli` |

The eight-character budget caps the package tree at about four components. That
is a feature: `crux.adapters.llm.litellm` is as deep as this repo goes.

## Invariants

These are load-bearing. Each has a test; the test is the enforcement, this is the
explanation.

- **A `Session` holds zero port references.** The engine holds the ports; the
  session holds only data. This is what lets a host `start()` on one worker,
  persist the session, and `resume()` on another.
- **`CompiledPrompt.assumptions` is derived, never generated.** Every resolution
  whose source is not `respondent` becomes an assumption. The model writes the
  task prose and the constraints; it never writes the honesty section.
- **A `default` resolution cites a named pack default.** A model-invented value
  is a bug, not a default. There is no code path from the expander to a value.
- **Every question carries a free-text field**, always, alongside any options. A
  respondent must be able to answer with a question, a counter-proposal, or a
  refusal, not just a click.
- **A credential-shaped decision never travels the `Clarifier`.** Clarifier
  answers are echoed back into model context by design; credentials must not be.
- **No `Clarifier` configured is not an error.** crux runs headless in plenty of
  hosts. It means resolve everything by default and delegation, and say so.
- **A fallback is never tried for our own malformed request.** It would fail
  identically on every provider, so the chain burns and the bug hides behind
  three timeouts. Provider-side failures fall back; schema failures stop.
- **A turn that ignores a forced tool is a failed attempt, not a success.**
  Otherwise the commonest weak-model failure is invisible to the fallback chain.
  `draft` is the one exception, because prose is what it asked for.
- **Measurement runs refuse a fallback chain.** A substitution mid-run turns the
  noise floor into a model difference and records the wrong answer under the
  primary's cassette key. Both corruptions are silent, so it raises.
- **A secret is never settable from `crux.toml`.** The file beats the
  environment, so without that rule a committed key would override an operator's.
- **Over-expansion costs tokens, never questions.** Question count is bounded by
  ask-worthiness and the budget, not by how many decisions the expander produced.
  Keep it that way; it is what makes an over-eager expander survivable.

## Vocabulary

Use these words and no synonyms. Terminology drift here shows up as duplicate
concepts in the code within a week.

- **Open decision** — something undecided about the intent. The node type. Not a
  "question": asking is one of four ways to close one.
- **Resolution** — how an open decision closed, and by what: `respondent`,
  `retrieval`, `default`, `delegated`.
- **Respondent** — whoever answers. Often a human, sometimes another agent.
- **Frontier** — the open decisions whose prerequisites are already settled, so
  they can be asked now.
- **Evidence graph** — entities surfaced by retrieval for this session. A
  citation structure, never an index of the repo.
- **Candidate** — a distinct possible answer adjudicated out of evidence items.
  Escalation counts candidates, never raw retrieval hits.
- **Compiled prompt** — the output. Not a "refined prompt", not a "final prompt".
- **Downstream agent** — whoever consumes the compiled prompt.
- **Host** — the system embedding crux.
