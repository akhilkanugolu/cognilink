"""CogniLink LLM invocation — uses any LangChain BaseChatModel.

The user passes their own configured LangChain model (ChatOpenAI, ChatAnthropic,
ChatBedrock, ChatOllama, etc.) and CogniLink simply calls model.invoke() with
the appropriate messages.

No custom LLM abstraction. No LiteLLM. Just LangChain's standard interface.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


def invoke_llm(model: Any, system_prompt: str, user_prompt: str) -> str:
    """Invoke a LangChain chat model and return the response text.

    Args:
        model: Any LangChain BaseChatModel instance (ChatOpenAI, ChatBedrock,
               ChatOllama, ChatAnthropic, etc.)
        system_prompt: The system-level instruction for the model.
        user_prompt: The user-level input/content to process.

    Returns:
        The generated text content from the model's response.

    Raises:
        Exception: Propagates any LangChain exceptions (timeouts, auth errors,
            rate limits, etc.) for the caller to handle.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = model.invoke(messages)
    # response is an AIMessage — extract the text content
    content = response.content
    if isinstance(content, list):
        # Some models return list of content blocks
        return "".join(str(block) for block in content)
    return str(content)
