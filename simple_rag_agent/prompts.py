"""
Author: L. Saetta
Date last modified: 2026-04-18
License: MIT
Description: Prompt templates used by the simple RAG agent.
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
