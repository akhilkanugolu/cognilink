"""CogniLink inference — thin wrapper around any LangChain BaseChatModel."""

from cognilink.inference.provider import invoke_llm

__all__ = ["invoke_llm"]
