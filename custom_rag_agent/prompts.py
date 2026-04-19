"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Prompt templates used by the custom RAG agent.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

QUERY_REWRITE_PROMPT_TEMPLATE = """You are a search query rewriter.
Given the conversation history and the latest user request, rewrite the latest
request as a standalone search query that is explicit and context-complete.
Return only the standalone search query text.
Do not add explanations, comments, labels, markdown, tags, XML, or any extra text.
Do not output chain-of-thought or reasoning markers such as <think>.
Output exactly one line containing only the final search query.

Conversation history:
{history}

Latest user request:
{user_input}

Standalone search query:
"""

ANSWER_PROMPT_TEMPLATE = """You are a helpful assistant.
Use only the provided context to answer the user request.
If the context is not enough, say that the information is not available.

Context:
{context}

User request:
{user_input}

Answer:
"""


def build_answer_prompt(user_input: str, context: str) -> str:
    """Build the answer-generation prompt from input and context.

    Args:
        user_input: Original user question.
        context: Retrieved text context provided to the model.

    Returns:
        str: Prompt text ready to send to the LLM.
    """
    return ANSWER_PROMPT_TEMPLATE.format(user_input=user_input, context=context)


def _format_history_for_prompt(history: Sequence[Dict[str, Any]]) -> str:
    """Convert history messages into readable lines for prompts.

    Args:
        history: Conversation history containing ``role`` and ``content`` fields.

    Returns:
        str: Multi-line string suitable for prompt injection.
    """
    if not history:
        return "(empty)"

    lines: list[str] = []
    for message in history:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "(empty)"


def build_query_rewrite_prompt(
    user_input: str,
    history: Sequence[Dict[str, Any]],
) -> str:
    """Build query-rewrite prompt using user input and prior history.

    Args:
        user_input: Latest user request.
        history: Previous messages used to preserve conversational context.

    Returns:
        str: Prompt text for the query rewriting LLM step.
    """
    history_text = _format_history_for_prompt(history)
    return QUERY_REWRITE_PROMPT_TEMPLATE.format(
        user_input=user_input,
        history=history_text,
    )
