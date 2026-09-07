# Markdown

<!-- toc -->

- [Structure](#structure)
- [Writing](#writing)
- [Line length](#line-length)
- [Code blocks](#code-blocks)
- [What belongs where](#what-belongs-where)

<!-- tocstop -->

## Structure

- One `# Title` per file, then `##` sections.
- A table of contents between `<!-- toc -->` and `<!-- tocstop -->` where a file
  has enough sections to need one.
- Every markdown file is referenced from `README.md` or from a file that is. An
  orphan is a file nobody will find and nobody will update.

## Writing

- One idea per bullet. No filler, no "in order to", no restating the heading.
- Say what is true now, not what is planned, unless the file's job is to plan.
- Where something is not done, say so plainly and say why. A caveat buried in a
  paragraph is a caveat nobody reads.
- Prefer the concrete: name the field, the flag, the threshold. "Escalates when
  two candidates survive adjudication and neither scores twice the other" beats
  "escalates when the evidence is ambiguous".

## Line length

80 columns.

## Code blocks

Fenced, with a language. A shell block shows the command a reader would actually
type.

## What belongs where

- `README.md`: what crux is, how to run it, where everything lives.
- `CLAUDE.md`: the alias table, the invariants, the vocabulary.
- `docs/code_guidelines/`: these guides, internal.
- `docs/adr/`: one file per decision that is hard to reverse, surprising without
  context, and the result of a real trade-off. If any of the three is missing,
  it is not an ADR.
- `plans/`: one file per iteration, written as the iteration happens, recording
  decisions and their reasons including the ones that turned out wrong.
