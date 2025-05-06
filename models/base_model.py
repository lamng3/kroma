from typing import List, Dict, Any

from config.arguments import ModelArguments, ModelMetadata
from config.constants import TOTAL_TOKENS_LIMIT

from models.providers import EmbeddingProvider
from models.stats import APIStats

class BaseModel:
    def __init__(self, args: ModelArguments, metadata: ModelMetadata):
        # initialise with provided arguments and metadata
        self.args = args
        self.metadata = metadata
        self.stats = APIStats()

    def generate(self, system_prompt: str, prompt: str,
                 include_tokens: bool = False, truncate: bool = True) -> str:
        # to be implemented in subclasses
        raise NotImplementedError("use a subclass of BaseModel")

    def desc(self) -> str:
        # to be implemented in subclasses
        raise NotImplementedError("use a subclass of BaseModel")

    def estimate_tokens(self, text: str) -> int:
        # rough estimate: one token per four characters
        return len(text) // 4

    def total_estimated_tokens(self, messages: List[Dict[str, Any]]) -> int:
        # sum token estimates for each message content
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def update_stats(self, input_tokens: int, output_tokens: int) -> float:
        # compute cost and update stats counters
        cost = (
            self.metadata.cost_per_input_token * input_tokens
            + self.metadata.cost_per_output_token * output_tokens
        )
        self.stats.total_cost += cost
        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1
        return cost

    def truncate_conversation(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int
    ) -> List[Dict[str, str]]:
        # always preserve the system prompt at index 0
        system = messages[0]
        tokens_used = self.estimate_tokens(system["content"])
        kept = []

        # include most recent messages until max_tokens is reached
        for msg in reversed(messages[1:]):
            tok = self.estimate_tokens(msg["content"])
            if tokens_used + tok <= max_tokens:
                kept.append(msg)
                tokens_used += tok

        kept.reverse()
        # if truncated, insert a notice before user messages
        if len(kept) < len(messages) - 1:
            notice = {
                "role": "system",
                "content": "note: previous conversation messages were truncated due to token limits."
            }
            kept.insert(0, notice)

        return [system] + kept

    def adjust_conversation(
        self,
        conversation: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        # ensure input + desired output fit within the total token limit
        desired_output = self.args.max_tokens
        allowed_input = TOTAL_TOKENS_LIMIT - desired_output

        if self.total_estimated_tokens(conversation) > allowed_input:
            conversation = self.truncate_conversation(conversation, allowed_input)
        return conversation
    

class EmbeddingModel:
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def encode(self, texts: List[str]) -> List[List[float]]:
        # you could add batching or caching here
        return self.provider.embed(texts)