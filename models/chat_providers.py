from typing import Any, Dict, List, Tuple
from .providers import ChatProvider

class OpenAIProvider(ChatProvider):
    """
    ChatProvider implementation for OpenAI's API via the openai-python client.
    """
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def create_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Any = None,
    ) -> Any:
        """
        Send a chat completion request to OpenAI.
        """
        if response_format:
            return self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def parse_response(
        self,
        response: Any,
        include_tokens: bool
    ) -> Tuple[str, int, int]:
        """
        Extract the assistant message and token usage from the response.
        Returns (message, prompt_tokens, completion_tokens).
        If include_tokens=False, returns (message, None, None).
        """
        choice = response.choices[0].message
        message = getattr(choice, "parsed", choice.content)
        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
        completion_tokens = getattr(response.usage, "completion_tokens", None)
        if include_tokens:
            return message, prompt_tokens, completion_tokens
        return message, None, None


class TogetherProvider(ChatProvider):
    """
    ChatProvider implementation for TogetherAI.
    """
    def __init__(self, api_key: str):
        from together import Together
        self.client = Together(api_key=api_key)

    def create_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Any = None,
    ) -> Any:
        """
        Send a chat completion request to Together.
        """
        if response_format:
            return self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def parse_response(
        self,
        response: Any,
        include_tokens: bool
    ) -> Tuple[str, int, int]:
        """
        Extract the assistant message and token usage from the response.
        Returns (message, prompt_tokens, completion_tokens).
        If include_tokens=False, returns (message, None, None).
        """
        choice = response.choices[0].message
        message = getattr(choice, "parsed", choice.content)
        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
        completion_tokens = getattr(response.usage, "completion_tokens", None)
        if include_tokens:
            return message, prompt_tokens, completion_tokens
        return message, None, None
