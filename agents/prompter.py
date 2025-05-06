import random
from typing import Any, Dict, List, Tuple

from agents.utils import (
    extract_yes_no,
    extract_confidence,
    majority_vote,
    active_learning_score,
    simple_match,
)
from agents.factory import create_agents
from tasks.query_engine import query
from tasks.prompts.builders import (
    build_task_prompt,
    build_round_prompt,
    build_last_round_prompt,
)
from retrieval.vector_store import VectorStore
from inference.factory import create_embedding_model

random.seed(2025)


def inference(
    backend: str,
    model_name: str,
    source_term: str,
    target_term: str,
    pcmaps: Dict[str, Dict[str, set]],
    dictionary: Dict[str, Any],
    options: List[str],
    embed_model,               # EmbeddingModel from factory
    store: VectorStore,        # prebuilt vector store
    n_rounds: int = 3,
    dropout: float = 0.5,
    reasoning: bool = False,
    active_learning: bool = False,
    f1_score: float = 0.0,
    bisim: bool = False,
) -> Tuple[int, Dict[str,int], int, bool]:
    """
    RAG-enabled multi-round inference:
      - expand terms
      - retrieve contexts via store
      - build prompts with contexts + demos
      - run agents, vote, optionally AL
    """
    # 1) instantiate agent ensemble
    agents = create_agents(1 if bisim else len(pcmaps), backend, model_name)
    metrics = {'input_token':0,'output_token':0,'api_calls':0}

    # 2) expand term structure
    src = expand_term(source_term, pcmaps['source'], dictionary, options)
    tgt = expand_term(target_term, pcmaps['target'], dictionary, options)

    # 3) bisimulation shortcut
    if bisim and simple_match(src, tgt):
        return 1, metrics, 10, True

    # 4) retrieve RAG contexts
    src_vec = embed_model.encode([src[0]])[0]
    tgt_vec = embed_model.encode([tgt[0]])[0]
    src_hits = store.query(src_vec, top_k=3)
    tgt_hits = store.query(tgt_vec, top_k=3)
    src_ctx = [store.get_text(_id) for _id,_ in src_hits]
    tgt_ctx = [store.get_text(_id) for _id,_ in tgt_hits]

    context_block = (
        "Source context:\n- " + "\n- ".join(src_ctx) +
        "\n\nTarget context:\n- " + "\n- ".join(tgt_ctx) + "\n\n"
    )

    history: List[Tuple[int,int]] = []

    # 5) multi-round debate
    for round_idx in range(n_rounds):
        if round_idx == 0:
            sys_p, base_p = build_task_prompt(src, tgt)
        elif round_idx == n_rounds - 1:
            sys_p, base_p = build_last_round_prompt(history)
        else:
            sys_p, base_p = build_round_prompt(history)

        user_p = context_block + base_p

        # broadcast to agents
        round_answers: List[Tuple[int,int]] = []
        for agent in agents.values():
            resp, inp, out = agent.generate(sys_p, user_p, include_tokens=True, truncate=False)
            ans  = extract_yes_no(resp)
            conf = extract_confidence(resp)
            round_answers.append((ans,conf))

            metrics['input_token']  += inp
            metrics['output_token'] += out
            metrics['api_calls']     += 1

        history.extend(round_answers)

    # 6) majority vote
    pred, conf = majority_vote(history)

    # 7) active learning decision
    accept = True
    if active_learning:
        score = active_learning_score(conf, f1_score)
        accept = score > 0

    return pred, metrics, conf, accept


def expand_term(
    term: str,
    pcmap: Dict[str, set],
    dictionary: Dict[str, Any],
    options: List[str],
) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    """Fetch structural and synonym contexts via query engine."""
    p, c, s, l = query(term, pcmap, dictionary, options)
    return term, p, c, s, l
