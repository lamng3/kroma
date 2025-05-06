from dataclasses import dataclass, fields
from utils.env import get_env

@dataclass
class APIStats():
    total_cost: float = 0
    instance_cost: float = 0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

@dataclass
class ModelArguments:
    model_name: str
    api_key: str = None
    temperature: float = 0.7
    max_tokens: int = 4000
    host_url: str = get_env('OLLAMA_HOST')  # if using an ollama server

@dataclass
class ModelMetadata:
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0

@dataclass
class EmbeddingArguments():
    model_name: str
    host_url: str = get_env('OLLAMA_HOST')