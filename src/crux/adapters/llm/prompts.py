"""
Every prompt template and every response schema, in one file.

Kept together on purpose: a schema and the prompt that has to satisfy it drift
apart the moment they live in different modules, and the tier-3 adapter tests
exist precisely to catch that drift.

Structured output goes through a **forced tool call** rather than a response
format, because tool calling is the more portable of the two across the providers
litellm fronts.

Import as:

import crux.adapters.llm.prompts as xprompt
"""

from __future__ import annotations

from typing import Any, Final

import crux.ports.llm as pllm

# The one rule every prompt here repeats, lifted from operant's discovery loop
# where it was learned the hard way: a model asked what it does not know will
# invent something plausible unless told not to.
_NO_INVENTION: Final = (
    "Never invent a value. If you do not know something, say so or leave it out. "
    "Never write a placeholder like example.com, TODO, or a made-up name."
)

# An expansion pass is capped rather than open-ended. Three reasons, and the
# first was measured: an uncapped pass generated a tool call large enough to
# truncate against a free tier's per-minute token budget, which fails the call
# outright. The second is that the ask threshold only ever surfaces the top few,
# so proposing thirty is work nobody reads. The third is that the cap makes cost
# ranking meaningful -- ranking eight things is a judgement, ranking thirty is a
# list.
MAX_DECISIONS_PER_PASS: Final = 8

# Allowed values live in the descriptions rather than in `enum` constraints, and
# `when_value` accepts null. This is deliberate and was measured: several
# providers validate tool arguments server-side, so a single bad nested field —
# one model wrote `"reversibility": "moderate"`, another wrote
# `"when_value": null` — rejects the WHOLE call. Client-side the schema is still
# strict: pydantic drops the offending entry and the rest of the batch survives.
# A dropped decision is recoverable; a failed session is not.
_ROLE: Final = (
    "You work out what a software request leaves undecided, so an agent does not "
    "have to guess. You are not writing code and you are not answering the "
    "request."
)


# #############################################################################
# Expansion
# #############################################################################

EXPAND_TOOL: Final = pllm.ToolSchema(
    name="propose_decisions",
    description=(
        f"Report at most {MAX_DECISIONS_PER_PASS} open decisions — the ones that "
        "matter most, not everything conceivable. "
        "Order them MOST COSTLY "
        "FIRST: the decision whose wrong answer would waste the most work goes "
        "at index 0. The ordering is used directly, so it matters more than any "
        "individual entry."
    ),
    parameters={
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "A short label for this decision, like "
                            "'channel' or 'scope'. Other decisions IN THIS SAME "
                            "LIST depend on it by this label.",
                        },
                        "undecided": {
                            "type": "string",
                            "description": "What is undecided, as a noun phrase. Not a question.",
                        },
                        "type": {
                            "type": "string",
                            "description": (
                                "One of exactly: ambiguity, underspecification, "
                                "vagueness, missing_context.\n"
                                "ambiguity: two or more incompatible readings. "
                                "underspecification: one reading, open choices. "
                                "vagueness: a criterion with no threshold. "
                                "missing_context: a fact about the codebase that "
                                "could be looked up rather than asked."
                            ),
                        },
                        "reversibility": {
                            "type": "string",
                            "description": "Exactly 'easy' or 'hard' — no other "
                            "value is accepted. How costly it is to change this "
                            "decision later, once code exists.",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Candidate answers, where the set is "
                            "genuinely closed. Omit otherwise.",
                        },
                        "retrieval_hint": {
                            "type": "string",
                            "description": "Comma-separated places worth looking, if "
                            "this could be answered from the codebase.",
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "description": "One of exactly: requires, "
                                        "prunes, constrains, informs.",
                                    },
                                    "source_id": {"type": "string"},
                                    "when_value": {
                                        "type": ["string", "null"],
                                        "description": "The source value that "
                                        "triggers this edge. Required for "
                                        "requires, prunes and constrains; null "
                                        "for informs.",
                                    },
                                },
                                "required": ["kind", "source_id"],
                            },
                            "description": (
                                "What this decision depends on. source_id is either "
                                "the ref of another decision in THIS list, or the id "
                                "of one from the already-found list.\n"
                                "requires: this decision does not exist at all "
                                "until the source resolves to when_value — use it "
                                "so nobody is asked a question their earlier answer "
                                "made irrelevant.\n"
                                "prunes: the source resolving to when_value deletes "
                                "this decision outright.\n"
                                "constrains: the source resolving to when_value "
                                "makes this decision mandatory.\n"
                                "informs: the source's evidence bears on this one."
                            ),
                        },
                    },
                    "required": ["ref", "undecided", "type", "reversibility"],
                },
            }
        },
        "required": ["decisions"],
    },
)


# The part of the expansion system prompt that is *instruction* rather than
# role or safety. Named so the eval optimiser can propose replacements for it
# without touching the role line or the no-invention rule, which are not up
# for negotiation.
EXPAND_INSTRUCTION: Final = (
    "A decision is something a competent engineer would have to settle before "
    "writing code, and that this request does not settle.\n\n"
    "Do not propose a decision that duplicates one already listed. Do not "
    "propose style preferences, and do not propose anything the request "
    "already answers.\n\n"
    f"Keep it short. At most {MAX_DECISIONS_PER_PASS} decisions, and give options only "
    "where the set is genuinely closed and short. A long list of marginal "
    "decisions is worse than a short list of real ones.\n\n"
    "Use edges. Most decisions only matter once an earlier one goes a "
    "particular way: a question about email digests is meaningless if the "
    "answer was in-app only, and a question about retry policy is "
    "meaningless if the scope is a minimal version. Say so with a requires "
    "or prunes edge, referring to the other decision by its ref. An edge "
    "you leave out is a question somebody gets asked for no reason."
)


def expand_messages(
    *,
    prompt: str,
    lens: str,
    existing: str,
    evidence: str,
    host_notes: str,
    instruction: str | None = None,
) -> tuple[pllm.Message, ...]:
    """
    Build the expansion conversation.

    :param prompt: What the user asked for.
    :param lens: The question this pass is asking.
    :param existing: Decisions already found, rendered.
    :param evidence: What retrieval has turned up, rendered.
    :param host_notes: Anything the host wanted to add.
    :param instruction: Replaces :data:`EXPAND_INSTRUCTION`. Only the eval
        optimiser passes this.
    :return: The messages.
    """
    system = f"{_ROLE}\n\n{instruction or EXPAND_INSTRUCTION}\n\n{_NO_INVENTION}"
    parts = [f"Request:\n{prompt}", f"\nThis pass asks: {lens}"]
    if existing:
        parts.append(f"\nDecisions already found (do not repeat these):\n{existing}")
    if evidence:
        parts.append(f"\nWhat has been found in the codebase:\n{evidence}")
    if host_notes:
        parts.append(f"\nNotes from the host system:\n{host_notes}")
    return (
        pllm.Message(role="system", content=system),
        pllm.Message(role="user", content="\n".join(parts)),
    )


# #############################################################################
# Adjudication
# #############################################################################

ADJUDICATE_TOOL: Final = pllm.ToolSchema(
    name="adjudicate",
    description=(
        "For each decision, collapse its raw search hits into DISTINCT possible "
        "answers. Twenty files matching a search is not twenty answers. Score "
        "each candidate by how strongly the evidence supports it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "decision_id": {"type": "string"},
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "value": {
                                        "type": "string",
                                        "description": "One possible answer, stated "
                                        "as the answer itself.",
                                    },
                                    "support": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Ids of the evidence items "
                                        "backing this candidate.",
                                    },
                                    "score": {
                                        "type": "number",
                                        "description": "Strength of support, 0-10.",
                                    },
                                },
                                "required": ["value", "score"],
                            },
                        },
                    },
                    "required": ["decision_id", "candidates"],
                },
            }
        },
        "required": ["decisions"],
    },
)


def adjudicate_messages(*, prompt: str, rendered: str) -> tuple[pllm.Message, ...]:
    """
    Build the adjudication conversation.

    :param prompt: What the user asked for.
    :param rendered: Each decision with its evidence, rendered.
    :return: The messages.
    """
    system = (
        f"{_ROLE}\n\n"
        "Turn search results into distinct candidate answers. If the evidence "
        "clearly points one way, return ONE candidate. Return several only when "
        "the evidence genuinely supports several different answers. Return none "
        "when it supports nothing.\n\n"
        "Returning spurious extra candidates causes a human to be interrupted "
        "unnecessarily, so do not pad the list.\n\n"
        f"{_NO_INVENTION}"
    )
    return (
        pllm.Message(role="system", content=system),
        pllm.Message(role="user", content=f"Request:\n{prompt}\n\n{rendered}"),
    )


# #############################################################################
# Phrasing
# #############################################################################

PHRASE_TOOL: Final = pllm.ToolSchema(
    name="phrase_questions",
    description="Word each decision as one short question for a busy engineer.",
    parameters={
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "decision_id": {"type": "string"},
                        "text": {
                            "type": "string",
                            "description": "The question. One sentence, no preamble, no apology.",
                        },
                        "why": {
                            "type": "string",
                            "description": "One short clause on what turns on the "
                            "answer. Shown under the question.",
                        },
                        "recommended": {
                            "type": "string",
                            "description": "Only if the evidence already points "
                            "somewhere. Omit otherwise.",
                        },
                    },
                    "required": ["decision_id", "text"],
                },
            }
        },
        "required": ["questions"],
    },
)


def phrase_messages(*, prompt: str, rendered: str) -> tuple[pllm.Message, ...]:
    """
    Build the phrasing conversation.

    :param prompt: What the user asked for.
    :param rendered: The frontier decisions, rendered.
    :return: The messages.
    """
    system = (
        f"{_ROLE}\n\n"
        "You are writing questions someone will answer in a few seconds between "
        "other work. Be direct. Do not thank them, do not apologise for asking, "
        "and do not explain what crux is.\n\n"
        f"{_NO_INVENTION}"
    )
    return (
        pllm.Message(role="system", content=system),
        pllm.Message(role="user", content=f"Request:\n{prompt}\n\n{rendered}"),
    )


# #############################################################################
# Classification
# #############################################################################

CLASSIFY_TOOL: Final = pllm.ToolSchema(
    name="classify_replies",
    description="Work out what each reply meant.",
    parameters={
        "type": "object",
        "properties": {
            "replies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "choice",
                                "value",
                                "question",
                                "proposal",
                                "defer",
                                "reject",
                            ],
                            "description": (
                                "choice: they picked one of the options offered. "
                                "value: they answered in their own words. "
                                "question: they asked something back. "
                                "proposal: they suggested an answer that was not "
                                "offered. defer: they said you decide. "
                                "reject: they said the question itself is wrong."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": "For choice, value or proposal: the answer.",
                        },
                        "asks": {
                            "type": "string",
                            "description": "For question: what they want to know.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "For proposal: why they suggested it.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "For reject: what is wrong with the question.",
                        },
                    },
                    "required": ["question_id", "kind"],
                },
            }
        },
        "required": ["replies"],
    },
)


def classify_messages(*, rendered: str) -> tuple[pllm.Message, ...]:
    """
    Build the classification conversation.

    :param rendered: Question and reply pairs, rendered.
    :return: The messages.
    """
    system = (
        f"{_ROLE}\n\n"
        "Classify each reply. Prefer the literal reading: a reply that answers "
        "the question is an answer, not a proposal. Only use 'reject' when they "
        "say the question itself is wrong, not merely that they dislike the "
        "options.\n\n"
        f"{_NO_INVENTION}"
    )
    return (
        pllm.Message(role="system", content=system),
        pllm.Message(role="user", content=rendered),
    )


# #############################################################################
# Counter-questions
# #############################################################################

COUNTER_TOOL: Final = pllm.ToolSchema(
    name="answer_back",
    description="Answer what the respondent asked, or say plainly that you cannot.",
    parameters={
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "string"},
                        "answer": {"type": "string"},
                        "answered": {
                            "type": "boolean",
                            "description": (
                                "True ONLY if the evidence provided actually "
                                "answers them. False if you are guessing. Setting "
                                "this to true when you are guessing sends them a "
                                "confident wrong answer."
                            ),
                        },
                        "recommended": {
                            "type": "string",
                            "description": "A suggested answer to the original "
                            "question, if the evidence supports one.",
                        },
                    },
                    "required": ["question_id", "answer", "answered"],
                },
            }
        },
        "required": ["answers"],
    },
)


def counter_messages(*, prompt: str, rendered: str) -> tuple[pllm.Message, ...]:
    """
    Build the counter-question conversation.

    :param prompt: What the user asked for.
    :param rendered: The questions asked back, with their evidence.
    :return: The messages.
    """
    system = (
        f"{_ROLE}\n\n"
        "Answer only from the evidence given. If it does not answer them, set "
        "answered to false and say what you would need to know. An invented "
        "answer here is worse than no answer, because they will act on it.\n\n"
        f"{_NO_INVENTION}"
    )
    return (
        pllm.Message(role="system", content=system),
        pllm.Message(role="user", content=f"Original request:\n{prompt}\n\n{rendered}"),
    )


# #############################################################################
# Drafting
# #############################################################################

DRAFT_TOOL: Final = pllm.ToolSchema(
    name="draft_task",
    description=(
        "Write the task statement and its constraints. Write NOTHING about "
        "assumptions, guesses, confidence or provenance — those are added from "
        "the decision record and are not yours to write."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What to build, in two or three sentences, "
                "incorporating the settled decisions.",
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hard requirements the implementation must meet, one per entry.",
            },
        },
        "required": ["task"],
    },
)


def draft_messages(*, prompt: str, rendered: str) -> tuple[pllm.Message, ...]:
    """
    Build the drafting conversation.

    :param prompt: What the user asked for.
    :param rendered: The settled decisions, rendered.
    :return: The messages.
    """
    system = (
        f"{_ROLE}\n\n"
        "Write a task statement an implementing agent can act on. Fold the "
        "settled decisions into it rather than listing them; they appear "
        "separately with their provenance.\n\n"
        "Do not add an assumptions section. Do not hedge. Do not mention that "
        "questions were asked.\n\n"
        f"{_NO_INVENTION}"
    )
    return (
        pllm.Message(role="system", content=system),
        pllm.Message(role="user", content=f"Original request:\n{prompt}\n\n{rendered}"),
    )


def tool_of(schema: pllm.ToolSchema) -> dict[str, Any]:
    """
    Render a tool schema the way litellm wants it.

    :param schema: The tool.
    :return: Its OpenAI-shaped dict.
    """
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }
