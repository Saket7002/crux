# Eval fixtures

Synthetic projects, not real ones. They exist so crux's filesystem retriever has
something to read when the eval corpus asks a question like "what does this
project already depend on".

Nothing here is built, installed, or executed. Dependency versions are pinned
above every advisory range on purpose. Real numbers raised three dozen security
alerts against projects that do not exist, which is the fastest way to teach a
maintainer to ignore Dependabot entirely. The obvious fix of 0.0.0 is worse:
advisory ranges are open at the bottom, so `<= 0.0.125` matches it.

Package *names* are realistic on purpose, because the adjudicator has to
recognise them as a stack.
