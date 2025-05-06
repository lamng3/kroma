from typing import Dict, Any, List
from inference.factory import create_inference_model

def create_agents_from_config(
    configs: List[Dict[str, str]]
) -> Dict[int, Any]:
    """
    Given a list of agent configs:
      [{ 'agent_type':'openai', 'agent_name':'gpt-4o-mini' }, ...]
    instantiate and return a mapping idx -> BaseModel.
    """
    agents: Dict[int, Any] = {}
    for idx, cfg in enumerate(configs):
        agents[idx] = create_inference_model(cfg['agent_type'], cfg['agent_name'])
    return agents


def create_agents(
    n_agents: int,
    backend: str,
    model_name: str
) -> Dict[int, Any]:
    """
    Create n_agents identical agents using the same backend+model.
    """
    return {
        i: create_inference_model(backend, model_name)
        for i in range(n_agents)
    }
