import networkx as nx
from node2vec import Node2Vec
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_graph(adj_dict):
    G = nx.DiGraph()
    for u, children in adj_dict.items():
        for v in children:
            G.add_edge(u, v)
    return G


def train_node2vec(
    graph,
    dimensions=128,
    walk_length=30,
    num_walks=200,
    p=1,
    q=1,
    workers=4,
    seed=42,
    window=10,
    min_count=1
):
    logger.info("Training node2vec model with %d dimensions", dimensions)
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed
    )
    model = node2vec.fit(window=window, min_count=min_count)
    return model


def get_structural_embeddings(model, node_keys):
    vecs = []
    for k in node_keys:
        try:
            vecs.append(model.wv[str(k)])
        except KeyError:
            logger.warning("Node %s not found in node2vec vocabulary, using zeros", k)
            vecs.append(np.zeros(model.vector_size))
    return np.vstack(vecs)


def combine_embeddings(text_embs, struct_embs, method="concat", alpha=0.5):
    if method == "concat":
        return np.hstack([text_embs, struct_embs])
    elif method == "weighted_sum":
        return alpha * text_embs + (1 - alpha) * struct_embs
    else:
        raise ValueError(f"Unknown combine method: {method}")


def compute_combined_embeddings(
    adj_dict,
    node_keys,
    text_embs,
    n2v_kwargs=None,
    combine_method="concat",
    alpha=0.5
):
    if n2v_kwargs is None:
        n2v_kwargs = {}

    graph = build_graph(adj_dict)
    model = train_node2vec(graph, **n2v_kwargs)
    struct_embs = get_structural_embeddings(model, node_keys)
    combined = combine_embeddings(text_embs, struct_embs, method=combine_method, alpha=alpha)
    return combined
