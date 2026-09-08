# Eval fixtures

Synthetic projects, not real ones. They exist so crux's filesystem retriever has
something to read when the eval corpus asks a question like "what does this
project already depend on".

Nothing here is built, installed, or executed. Dependency versions are
deliberately not real releases: with real numbers, GitHub's dependency graph
raised three dozen security alerts against projects that do not exist, which is
the fastest way to teach a maintainer to ignore Dependabot entirely.

Package *names* are realistic on purpose, because the adjudicator has to
recognise them as a stack.
