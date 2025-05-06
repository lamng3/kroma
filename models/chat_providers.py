from typing import Any, Dict, Tuple

class ChatProvider:
    """abstract interface for chat backends."""
    def create_completion(
        self,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
        response_format: Any = None
    ) -> Any:
        raise NotImplementedError

    def parse_response(
        self,
        response: Any,
        include_tokens: bool
    ) -> Tuple[str, int, int]:
        raise NotImplementedError


class OpenAIProvider(ChatProvider):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        # assume beta vs stable can be configured here

    def create_completion(self, model, messages, temperature, max_tokens, response_format=None):
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
            response_format=response_format,
        )

    def parse_response(self, response, include_tokens):
        # grab both parsed and raw paths
        msg = (response.choices[0].message.parsed
               if hasattr(response.choices[0].message, 'parsed')
               else response.choices[0].message.content)
        inp = response.usage.prompt_tokens
        out = response.usage.completion_tokens
        return (msg, inp, out) if include_tokens else (msg, None, None)


class TogetherProvider(ChatProvider):
    def __init__(self, api_key: str):
        from together import Together
        self.client = Together(api_key=api_key)

    def create_completion(self, model, messages, temperature, max_tokens, response_format=None):
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
            response_format=response_format,
        )

    def parse_response(self, response, include_tokens):
        msg = (response.choices[0].message.parsed
               if hasattr(response.choices[0].message, 'parsed')
               else response.choices[0].message.content)
        inp = response.usage.prompt_tokens
        out = response.usage.completion_tokens
        return (msg, inp, out) if include_tokens else (msg, None, None)