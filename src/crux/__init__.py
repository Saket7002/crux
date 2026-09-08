"""
crux: an embeddable clarification agent.

Turns an underspecified prompt into a compiled prompt whose every guess is
visible. This module is the public surface and the one sanctioned place in the
package for ``from X import Y``.
"""

from crux.adapters.clarifier.tty import TtyClarifier
from crux.adapters.llm.litellm import LiteLlmClient
from crux.adapters.llm.reasoner import LlmReasoner
from crux.adapters.retrieval.fs import FilesystemRetriever
from crux.adapters.store.jsonfile import JsonFileSessionStore
from crux.adapters.store.memory import MemorySessionStore
from crux.api import Crux, clarify, clarify_async
from crux.domain.decisions import OpenDecision, Option, Resolution
from crux.domain.output import CompiledPrompt
from crux.domain.replies import Question, RawReply
from crux.domain.session import Budget, Done, NeedsInput, Session, SessionContext, Step
from crux.errors import (
    ConfigurationError,
    CruxError,
    ReasonerError,
    ReasonerParseError,
    RetrievalError,
    SessionError,
)
from crux.ports.clarifier import Clarifier
from crux.ports.reasoner import Reasoner
from crux.ports.retrieval import Retriever
from crux.ports.store import SessionStore

__all__ = [
    "Budget",
    "Clarifier",
    "CompiledPrompt",
    "ConfigurationError",
    "Crux",
    "CruxError",
    "Done",
    "FilesystemRetriever",
    "JsonFileSessionStore",
    "LiteLlmClient",
    "LlmReasoner",
    "MemorySessionStore",
    "NeedsInput",
    "OpenDecision",
    "Option",
    "Question",
    "RawReply",
    "Reasoner",
    "ReasonerError",
    "ReasonerParseError",
    "Resolution",
    "RetrievalError",
    "Retriever",
    "Session",
    "SessionContext",
    "SessionError",
    "SessionStore",
    "Step",
    "TtyClarifier",
    "clarify",
    "clarify_async",
]
