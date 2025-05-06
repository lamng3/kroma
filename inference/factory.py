from typing import Type
from utils.env import get_env
from config.constants import METADATA, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL

from models.base_model import BaseModel, EmbeddingModel
from config.arguments import ModelArguments, ModelMetadata, EmbeddingArguments

from models.chat_providers import (
    OpenAIProvider,
    TogetherProvider,
)
from models.embedding_providers import (
    HFEmbeddingProvider,
)

# registry of available chat-completion backends
CHAT_BACKENDS: dict[str, Type] = {
    "openai": OpenAIProvider,
    "togetherai": TogetherProvider,
}

# registry of available embedding backends
EMBED_BACKENDS: dict[str, Type] = {
    "huggingface": HFEmbeddingProvider,
}


def create_inference_model(
    backend: str,
    model_name: str = DEFAULT_CHAT_MODEL,
) -> BaseModel:
    """
    instantiate a BaseModel given a backend key and optional model name.
    looks up cost metadata and pulls API keys from environment.
    """
    ProviderCls = CHAT_BACKENDS.get(backend)
    if not ProviderCls:
        raise ValueError(f"chat backend '{backend}' not supported")

    api_key = get_env(f"{backend.upper()}_API_KEY")
    args = ModelArguments(model_name=model_name, api_key=api_key)
    meta_block = METADATA.get(backend, {})
    model_meta = meta_block.get(model_name, {})
    metadata = ModelMetadata(
        cost_per_input_token=model_meta.get("cost_per_input_token", 0.0),
        cost_per_output_token=model_meta.get("cost_per_output_token", 0.0),
    )

    provider = ProviderCls(api_key=api_key, model_name=model_name)
    return BaseModel(provider, args, metadata)


def create_embedding_model(
    backend: str,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> EmbeddingModel:
    """
    instantiate an EmbeddingModel given a backend key and optional model name.
    """
    ProviderCls = EMBED_BACKENDS.get(backend)
    if not ProviderCls:
        raise ValueError(f"embedding backend '{backend}' not supported")

    args = EmbeddingArguments(model_name=model_name)
    provider = ProviderCls(model_name=model_name)
    return EmbeddingModel(provider, args)


def generate_response(
    chat_backend: str,
    system_prompt: str,
    prompt: str,
    model_name: str = DEFAULT_CHAT_MODEL,
    include_tokens: bool = False,
    **kwargs,
):
    """
    convenience wrapper: spins up a chat model and returns its generated reply.
    """
    model = create_inference_model(chat_backend, model_name)
    return model.generate(
        system_prompt=system_prompt,
        prompt=prompt,
        include_tokens=include_tokens,
        **kwargs,
    )
