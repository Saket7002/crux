## What this changes

<!-- One or two sentences. The diff shows the rest. -->

## What you measured

<!--
If this touches a prompt, the routing table, or the ask threshold, say what
happened to the numbers. Ask rate is the product metric: a change that quietly
moves it from two questions to six is a regression even if every test passes.

If you found this by running crux for real, put the number here. Most of the
sharper bugs in this repo were invisible to mocked tests.
-->

## Checklist

- [ ] `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -m "not live"` passes
- [ ] The gate changes nothing on a second run
- [ ] New tests are named after the claim, and say why the claim matters
- [ ] If a test exists because something went wrong, the docstring says what
- [ ] New module added? Its alias is in the `CLAUDE.md` table
