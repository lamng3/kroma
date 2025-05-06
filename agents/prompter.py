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

# bring in bisimulation and ranking
from algorithms.bisimulation import bisimulation
from algorithms.bisimulation import compute_node_ranks
from algorithms.incremental_refinement import incremental_refinement

random.seed(2025)

# we’ll keep a module‑level compressed graph and rank_attr
compressed_graph: Dict[Any, List[Any]] = {}
rank_attr: Dict[Any, int] = {}

def initialize_graph(OS: Dict, OT: Dict):
    """Call once at startup with your raw ontologies"""
    global compressed_graph, rank_attr
    compressed_graph = {**OS, **OT}
    rank_attr = compute_node_ranks(compressed_graph)

def inference(
    backend: str,
    model_name: str,
    source_term: Tuple[str,str],
    target_term: Tuple[str,str],
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
    RAG‐enabled multi‐round inference, with a bisimulation pre‐check
    and optional incremental refinement.
    """
    global compressed_graph, rank_attr

    # 1) try bisimulation structural match
    if bisim:
        aligned = bisimulation(compressed_graph)
        if (source_term, target_term) in aligned:
            return 1, {'input_token':0,'output_token':0,'api_call_cnt':0}, 10, True

    # 2) normal multi‐round debate
    agents = create_agents(1 if bisim else len(pcmaps), backend, model_name)
    metrics = {'input_token':0,'output_token':0,'api_calls':0}

    src = expand_term(source_term, pcmaps['source'], dictionary, options)
    tgt = expand_term(target_term, pcmaps['target'], dictionary, options)

    # fetch RAG contexts
    src_ctx = [store.get_text(i) for i,_ in store.query(embed_model.encode([src[0]])[0], 3)]
    tgt_ctx = [store.get_text(i) for i,_ in store.query(embed_model.encode([tgt[0]])[0], 3)]
    context_block = (
        "Source context:\n- " + "\n- ".join(src_ctx) +
        "\n\nTarget context:\n- " + "\n- ".join(tgt_ctx) + "\n\n"
    )

    history: List[Tuple[int,int]] = []
    for round_idx in range(n_rounds):
        if round_idx == 0:
            sys_p, base_p = build_task_prompt(src, tgt, reasoning=reasoning)
        elif round_idx == n_rounds - 1:
            sys_p, base_p = build_last_round_prompt(src, tgt, history, perturb=False)
        else:
            sys_p, base_p = build_round_prompt(src, tgt, history, perturb=False)

        user_p = context_block + base_p

        for agent in agents.values():
            resp, inp, out = agent.generate(sys_p, user_p, include_tokens=True, truncate=False)
            ans  = extract_yes_no(resp)
            conf = extract_confidence(resp)
            history.append((ans,conf))

            metrics['input_token']  += inp
            metrics['output_token'] += out
            metrics['api_calls']     += 1

    pred, conf = majority_vote(history)

    accept = True
    if active_learning:
        accept = active_learning_score(conf, f1_score) > 0

    # 3) if accepted, refine compressed_graph
    if accept:
        compressed_graph, newQ = incremental_refinement(
            O_S={}, O_T={},  # pass actual OS/OT if you track them here
            G_r=compressed_graph,
            delta_G=[(source_term, target_term)],
            rank_attr=rank_attr
        )
        if newQ:
            # you could log or expose these
            print("Expert queries needed:", newQ)

    return pred, metrics, conf, accept


def expand_term(
    term: Tuple[str,str],
    pcmap: Dict[str, set],
    dictionary: Dict[str, Any],
    options: List[str],
) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    p, c, s, l = query(term[0], pcmap, dictionary, options)
    return term[0], p, c, s, l
