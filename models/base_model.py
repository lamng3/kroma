from typing import Any, Dict, List, Optional, Tuple
from config.arguments import (
    ModelArguments, 
    ModelMetadata, 
    EmbeddingArguments,
    APIStats,
)
from models.providers import (
    ChatProvider, 
    EmbeddingProvider,
)
from config.constants import TOTAL_TOKENS_LIMIT

class BaseModel:
    """
    Wraps a ChatProvider (OpenAI, Together, etc.) to provide:
      - generate(...)
      - built‑in token accounting and (optional) truncation
    """

    def __init__(
        self,
        provider: ChatProvider,
        args: ModelArguments,
        metadata: ModelMetadata,
    ):
        self.provider = provider
        self.args = args
        self.metadata = metadata
        self.stats = APIStats()  # your existing stats class

    def generate(
        self,
        system_prompt: str,
        prompt: str,
        include_tokens: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Any = None,
        truncate: bool = True,
    ) -> Any:
        """
        Build messages, optionally truncate, call the provider,
        update costs, and return either (msg, in_t, out_t) or msg only.
        """
        # 1. construct message list
        messages: List[Dict[str,str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]

        # 2. truncate if needed
        if truncate:
            messages = self._truncate_conversation(messages)

        # 3. pick config
        temp = temperature if temperature is not None else self.args.temperature
        mtok = max_tokens if max_tokens is not None else self.args.max_tokens

        # 4. send to provider
        raw = self.provider.create_completion(
            model=self.args.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=mtok,
            response_format=response_format,
        )

        # 5. parse response
        msg, in_tok, out_tok = self.provider.parse_response(raw, include_tokens)

        # 6. update stats (if we know tokens)
        if include_tokens and in_tok is not None and out_tok is not None:
            cost = (
                in_tok  * self.metadata.cost_per_input_token +
                out_tok * self.metadata.cost_per_output_token
            )
            self.stats.total_cost      += cost
            self.stats.instance_cost   = cost
            self.stats.tokens_sent    += in_tok
            self.stats.tokens_received+= out_tok
            self.stats.api_calls      += 1

        return (msg, in_tok, out_tok) if include_tokens else msg

    def _truncate_conversation(self, messages: List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Keep the system prompt plus as many most recent messages
        as fit under TOTAL_TOKENS_LIMIT - max_tokens.
        """
        # quick token estimate (1 token ~4 chars)
        def est_tokens(text: str) -> int:
            return len(text) // 4

        keep = [messages[0]]
        used = est_tokens(messages[0]["content"])
        allowed = TOTAL_TOKENS_LIMIT - self.args.max_tokens

        # walk from the end backwards
        for msg in reversed(messages[1:]):
            tok = est_tokens(msg["content"])
            if used + tok <= allowed:
                keep.append(msg)
                used += tok
        keep.reverse()
        if len(keep) < len(messages):
            # insert a notice
            keep.insert(1, {
                "role": "system",
                "content": "⚠️ previous messages truncated due to length."
            })
        return keep

class BaseEmbeddingModel:
    """
    Wraps any EmbeddingProvider (HF, Ollama, etc.) to expose a
    simple .encode(...) method.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        args: EmbeddingArguments,
    ):
        self.provider = provider
        self.args = args

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Returns a list of embedding vectors, one per input string.
        """
        return self.provider.embed(texts)
