from typing import Type, Dict
from utils.env import get_env
from config.constants import METADATA, DEFAULT_CHAT_BACKEND, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from config.arguments import ModelArguments, ModelMetadata, EmbeddingArguments

from models.base_model import BaseModel, BaseEmbeddingModel
from models.chat_providers import OpenAIProvider, TogetherProvider
from models.embedding_providers import HFEmbeddingProvider

# registry of available chat‐completion backends
CHAT_BACKENDS: Dict[str, Type] = {
    "openai": OpenAIProvider,
    "togetherai": TogetherProvider,
}

# registry of available embedding backends
EMBED_BACKENDS: Dict[str, Type] = {
    "huggingface": HFEmbeddingProvider,
}


def create_inference_model(
    backend: str,
    model_name: str = DEFAULT_CHAT_MODEL,
) -> BaseModel:
    """
    instantiate a ChatProvider and wrap it in BaseModel
    """
    ProviderCls = CHAT_BACKENDS.get(backend)
    if ProviderCls is None:
        raise ValueError(f"unsupported chat backend: {backend}")

    # pull API key & build args/metadata
    api_key = get_env(f"{backend.upper()}_API_KEY")
    args = ModelArguments(model_name=model_name, api_key=api_key)

    meta_block = METADATA.get(backend, {})
    model_meta = meta_block.get(model_name, {})
    metadata = ModelMetadata(
        cost_per_input_token=model_meta.get("cost_per_input_token", 0.0),
        cost_per_output_token=model_meta.get("cost_per_output_token", 0.0),
    )

    # instantiate provider and wrap
    provider = ProviderCls(api_key=api_key)
    return BaseModel(provider, args, metadata)


def create_embedding_model(
    backend: str,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> BaseEmbeddingModel:
    """
    instantiate an EmbeddingProvider and wrap it in BaseEmbeddingModel
    """
    ProviderCls = EMBED_BACKENDS.get(backend)
    if ProviderCls is None:
        raise ValueError(f"unsupported embedding backend: {backend}")

    args = EmbeddingArguments(model_name=model_name)
    provider = ProviderCls(model_name=model_name)
    return BaseEmbeddingModel(provider, args)


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
