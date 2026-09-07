"""
A console clarifier, for the CLI demo.

Import as:

import crux.adapters.clarifier.tty as xtty
"""

from __future__ import annotations

from collections.abc import Sequence

import rich.console
import rich.panel

import crux.domain.replies as creply


class TtyClarifier:
    """
    Puts a round of questions to whoever is at the terminal.
    """

    def __init__(self, console: rich.console.Console | None = None) -> None:
        """
        :param console: Where to write. A fresh one is made when not supplied.
        """
        self._console = console or rich.console.Console()

    def ask(self, questions: Sequence[creply.Question]) -> Sequence[creply.RawReply]:
        """
        Show each question and read one line back.

        The free-text prompt is always offered, never only the options: a
        respondent has to be able to ask back, propose something unlisted, or say
        the question is wrong.

        :param questions: The round to put.
        :return: What was typed, one reply per question.
        """
        replies: list[creply.RawReply] = []
        for index, question in enumerate(questions, start=1):
            body = [f"[bold]{question.text}[/bold]"]
            if question.why:
                body.append(f"[dim]{question.why}[/dim]")
            if question.options:
                body.append("")
                for number, option in enumerate(question.options, start=1):
                    marker = (
                        " [green](suggested)[/green]"
                        if (question.recommended == option.value)
                        else ""
                    )
                    body.append(f"  [cyan]{number}[/cyan]. {option.display()}{marker}")
            if question.citations:
                body.append(f"\n[dim]from: {', '.join(question.citations[:3])}[/dim]")
            body.append(
                "\n[dim]Type a number, or your own answer. You can also ask me a "
                "question, suggest something else, say 'you decide', or say the "
                "question is wrong.[/dim]"
            )
            self._console.print(
                rich.panel.Panel(
                    "\n".join(body),
                    title=f"[{index}/{len(questions)}]",
                    border_style="yellow",
                )
            )
            answer = self._console.input("[bold yellow]> [/bold yellow]").strip()
            replies.append(_to_reply(question, answer))
        return replies


def _to_reply(question: creply.Question, answer: str) -> creply.RawReply:
    """
    Turn a typed line into a raw reply.

    A bare option number becomes a selection, which the engine then classifies
    without a model call. Everything else goes through as free text.

    :param question: What was asked.
    :param answer: What they typed.
    :return: The reply.
    """
    if answer.isdigit() and question.options:
        index = int(answer) - 1
        if 0 <= index < len(question.options):
            return creply.RawReply(
                question_id=question.id, selected=(question.options[index].value,)
            )
    if not answer and question.recommended:
        return creply.RawReply(question_id=question.id, selected=(question.recommended,))
    return creply.RawReply(question_id=question.id, text=answer)
