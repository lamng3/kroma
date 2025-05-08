from typing import Dict, Tuple, Set, List, Any
from algorithms.bisimulation import compute_node_ranks

# type aliases for ontology nodes (code, uri) pairs and graph
Node = Tuple[str, str]
Graph = Dict[Node, Set[Node]]
Edge = Tuple[Node, Node]

def merge_graphs(*graphs: Dict[Any, Set[Any]]) -> Dict[Any, Set[Any]]:
    merged: Dict[Any, Set[Any]] = {}
    for G in graphs:
        for u, vs in G.items():
            merged.setdefault(u, set()).update(vs)
    return merged

def update_ranks_for_delta(
    G_r: Graph,
    delta_edges: List[Edge]
) -> Dict[Node, int]:
    """recompute ranks for nodes touched by new edges"""
    # identify affected nodes
    affected = {u for u, v in delta_edges} | {v for u, v in delta_edges}
    # compute full ranks on G_r
    full_ranks = compute_node_ranks(G_r)
    # return only affected updates
    return {n: full_ranks[n] for n in affected if n in full_ranks}


def apply_rank_updates(
    rank_attr: Dict[Node, int],
    updates: Dict[Node, int]
) -> None:
    """apply computed rank updates to the existing rank attribute map"""
    for node, r in updates.items():
        rank_attr[node] = r


def inc_rcm_plus(
    e: Edge,
    block_u: Set[Node],
    block_u_prime: Set[Node],
    G_r: Graph,
    rank_attr: Dict[Node, int]
) -> Tuple[Graph, List[Edge]]:
    """
    Implements incRCM^+:
      - splits super-node blocks
      - merges based on rank comparisons
      - flags expert queries when ranks inconsistent
    """
    from algorithms.bisimulation import collapse

    new_queries: List[Edge] = []
    u, u_prime = e

    # Split super-node blocks if they contain more than one node
    for blk, center in [(block_u, u), (block_u_prime, u_prime)]:
        if len(blk) > 1:
            others = blk - {center}
            for v in others:
                if center in G_r and v in G_r[center]:
                    G_r[center].remove(v)
                if v in G_r and center in G_r[v]:
                    G_r[v].remove(center)
            # record split operation
            new_queries.append((center, 'split_block'))

    # bucket nodes by rank
    buckets: Dict[int, List[Node]] = {}
    for node, r in rank_attr.items():
        buckets.setdefault(r, []).append(node)

    rank_u = rank_attr.get(u, 0)
    rank_up = rank_attr.get(u_prime, 0)

    if rank_u > rank_up:
        # merge u with its own rank peers
        for v in buckets.get(rank_u, []):
            if v != u:
                collapse(G_r, {u, v})
        # merge u' with its rank peers
        for v in buckets.get(rank_up, []):
            if v != u_prime:
                collapse(G_r, {u_prime, v})
    elif rank_u == rank_up:
        # merge u with parents of u'
        parents_up = [p for p, children in G_r.items() if u_prime in children]
        for v in parents_up:
            collapse(G_r, {u, v})
        # merge u' with children of u
        for v in G_r.get(u, set()):
            collapse(G_r, {u_prime, v})
    else:
        # inconsistent rank direction → expert review needed
        new_queries.append((u, u_prime))

    return G_r, new_queries


def incremental_refinement(
    G_r: Graph,
    delta_G: List[Edge],
    rank_attr: Dict[Node, int]
) -> Tuple[Graph, List[Edge]]:
    """
    Incrementally refine the compressed graph G_r with insertions delta_G.
    returns updated graph and expert queries for manual review
    """
    expert_queries: List[Edge] = []
        # apply new edges to the graph
    for u, v in delta_G:
        G_r.setdefault(u, set()).add(v)
    # recompute and apply rank updates for affected nodes
    updates = update_ranks_for_delta(G_r, delta_G)
    # recompute and apply rank updates for affected nodes
    updates = update_ranks_for_delta(G_r, delta_G)
    apply_rank_updates(rank_attr, updates)
    # process each insertion in order of ascending rank of source
    sorted_edges = sorted(delta_G, key=lambda edge: rank_attr.get(edge[0], 0))
    for e in sorted_edges:
        u, v = e
        # treat each node as its own super-node block (compressed graph entries are singletons)
        block_u = {u}
        block_up = {v}
        G_r, newQ = inc_rcm_plus(e, block_u, block_up, G_r, rank_attr)
        expert_queries.extend(newQ)

    return G_r, expert_queries
