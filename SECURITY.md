# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:
[**Report a vulnerability**](https://github.com/Vedanshu7/crux/security/advisories/new).
Please do not open a public issue for a security problem.

Expect a first response within a week. crux is maintained by one person, so it
may take longer to fix than to acknowledge.

## What is in scope

crux handles two things worth being careful with, and both have rules the code
enforces.

**Credentials.** Every secret field is a `pydantic.SecretStr`. A `Session` never
holds one, because sessions are serialised to a host's storage by design. Only
`crux.infra.settings` and `crux.adapters.llm.litellm` may call
`get_secret_value()`. A `crux.toml` cannot set a secret at all: the file takes
priority over the environment, so without that rule a committed file would
silently override an operator's key.

A finding that a credential reaches a log, a `Session`, a compiled prompt, or the
`--evidence` trace is in scope and worth reporting.

**Content sent to a model.** Retrieved file excerpts and a respondent's free-text
answers reach the model by design. `--evidence` writes both to disk, which is why
it is off unless asked for. A path that sends more than the briefs asked for, or
that writes a trace nobody requested, is in scope.

## What is not

- A model producing a wrong or poor-quality decision. That is a measurement
  problem, and `tests/evals/` is where it belongs.
- Cost from an unbounded run. Every loop is bounded by `Budget`; if you have
  found one that is not, that is a bug and an ordinary issue is fine.
- Anything requiring a `Retriever` or `Clarifier` you wrote yourself to be
  hostile. Those are your code running in your process.
