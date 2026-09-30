from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class ChatProvider(ABC):
    """abstract interface for all chat‐completion backends."""
    @abstractmethod
    def create_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Any = None
    ) -> Any:
        """
        Sends the chat request. Return the raw provider response.
        """
        pass

    @abstractmethod
    def parse_response(
        self,
        response: Any,
        include_tokens: bool
    ) -> Tuple[str, int, int]:
        """
        Extracts (message, prompt_tokens, completion_tokens) from the raw response.
        If include_tokens is False, you may return (message, None, None).
        """
        pass


def parse_chat_response(response: Any, include_tokens: bool) -> Tuple[str, int, int]:
    """Shared adapter for OpenAI-compatible chat responses."""
    choice = response.choices[0].message
    message = getattr(choice, "parsed", choice.content)
    if not include_tokens:
        return message, None, None
    prompt_tokens = getattr(response.usage, "prompt_tokens", None)
    completion_tokens = getattr(response.usage, "completion_tokens", None)
    return message, prompt_tokens, completion_tokens


class EmbeddingProvider(ABC):
    """abstract interface for all embedding backends."""
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of strings into a list of float vectors.
        """
        pass
