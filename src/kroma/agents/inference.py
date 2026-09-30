"""Single-model and debate inference strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from kroma.agents.factory import create_agents, create_agents_from_config
from kroma.agents.utils import (
    active_learning_score,
    extract_confidence,
    extract_yes_no,
    majority_vote,
)
from kroma.tasks.prompts.builders import (
    build_last_round_prompt,
    build_round_prompt,
    build_task_prompt,
)
from kroma.tasks.query_engine import query


def expand_term(term, pcmap, dictionary, options):
    parents, children, synonyms, labels = query(term[0], pcmap, dictionary, options)
    return term[0], parents, children, synonyms, labels


def _run_rounds(agents, n_rounds, src, tgt, context_block, reasoning):
    history: List[Tuple[int, int]] = []
    metrics = {"input_token": 0, "output_token": 0, "api_calls": 0}
    for round_idx in range(n_rounds):
        if round_idx == 0:
            system_prompt, base_prompt = build_task_prompt(src, tgt, reasoning=reasoning)
        elif round_idx == n_rounds - 1:
            system_prompt, base_prompt = build_last_round_prompt(src, tgt, history, perturb=False)
        else:
            system_prompt, base_prompt = build_round_prompt(src, tgt, history, perturb=False)
        user_prompt = context_block + base_prompt
        for agent in agents.values():
            response, input_tokens, output_tokens = agent.generate(
                system_prompt,
                user_prompt,
                include_tokens=True,
                truncate=False,
            )
            history.append((extract_yes_no(response), extract_confidence(response)))
            metrics["input_token"] += input_tokens
            metrics["output_token"] += output_tokens
            metrics["api_calls"] += 1
    prediction, confidence = majority_vote(history)
    return prediction, metrics, confidence


class InferenceStrategy(ABC):
    """Chooses agents and how many debate rounds to run."""

    @abstractmethod
    def agents(self, backend, model_name, agent_configs):
        """Return the agent mapping and the number of rounds."""

    def run(
        self,
        backend,
        model_name,
        source_term,
        target_term,
        pcmaps,
        dictionary,
        options,
        agent_configs,
        n_rounds=3,
        reasoning=False,
        active_learning=False,
        f1_score=0.0,
        context=None,
    ):
        agents, rounds = self.agents(backend, model_name, agent_configs)
        if rounds is None:
            rounds = n_rounds
        source = expand_term(source_term, pcmaps["source"], dictionary, options)
        target = expand_term(target_term, pcmaps["target"], dictionary, options)
        prediction, metrics, confidence = _run_rounds(
            agents,
            rounds,
            source,
            target,
            context or "",
            reasoning,
        )
        accept = True
        if active_learning:
            accept = active_learning_score(confidence, f1_score) > 0
        return prediction, metrics, confidence, accept


class SingleModelInference(InferenceStrategy):
    """One model, one round."""

    def agents(self, backend, model_name, agent_configs):
        return create_agents(1, backend, model_name), 1


class DebateInference(InferenceStrategy):
    """One agent per config, across the configured number of rounds."""

    def __init__(self, n_rounds: int):
        self.n_rounds = n_rounds

    def agents(self, backend, model_name, agent_configs):
        return create_agents_from_config(agent_configs), self.n_rounds


def infer(
    debate: bool,
    backend: str,
    model_name: str,
    source_term: Tuple[str, str],
    target_term: Tuple[str, str],
    pcmaps: Dict[str, Dict[str, set]],
    dictionary: Dict[str, Any],
    options: List[str],
    embed_model=None,
    store=None,
    agent_configs: List[Dict[str, str]] = None,
    n_rounds: int = 3,
    dropout: float = 0.5,
    reasoning: bool = False,
    active_learning: bool = False,
    f1_score: float = 0.0,
    bisim: bool = False,
    context: str = None,
):
    """Run single-model or debate inference for one alignment pair.

    embed_model, store, dropout, and bisim are accepted so existing callers
    keep working. Context is built by the matching run and passed in.
    """
    strategy = DebateInference(n_rounds) if debate else SingleModelInference()
    return strategy.run(
        backend=backend,
        model_name=model_name,
        source_term=source_term,
        target_term=target_term,
        pcmaps=pcmaps,
        dictionary=dictionary,
        options=options,
        agent_configs=agent_configs or [],
        n_rounds=n_rounds,
        reasoning=reasoning,
        active_learning=active_learning,
        f1_score=f1_score,
        context=context,
    )
