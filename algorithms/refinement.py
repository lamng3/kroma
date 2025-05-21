from typing import Any, Dict, Set, List, Tuple
from collections import defaultdict

def offline_refine(adj: Dict[Any, List[Any]], rank_attr: Dict[Any, int]) -> Dict[Any, List[Any]]:
    V = set(adj)
    rho = max(rank_attr.values(), default=0)

    # build B_i and initial partition P = {B_0, …, B_rho}
    B: Dict[int, Set[Any]] = {i: {c for c, r in rank_attr.items() if r == i} for i in range(rho + 1)}
    P: List[Set[Any]] = [set(B[i]) for i in range(rho + 1)]

    # build incoming edges map
    incoming: Dict[Any, Set[Any]] = defaultdict(set)
    for u, children in adj.items():
        for v in children:
            incoming[v].add(u)

    # refinement passes
    for i in range(rho + 1):
        Bi = B[i]
        downstream = set().union(*(B[j] for j in range(i + 1, rho + 1)))
        new_P: List[Set[Any]] = []

        for C in P:
            if C and C.issubset(downstream):
                split_done = False
                for c in Bi:
                    C1 = {n for n in C if (n in adj.get(c, [])) or (c in incoming.get(n, []))}
                    C2 = C - C1
                    if C1 and C2:
                        new_P.extend([C1, C2])
                        split_done = True
                        break
                if not split_done:
                    new_P.append(C)
            else:
                new_P.append(C)
        P = new_P

    # collapse each block
    rep_map: Dict[Any, Any] = {}
    for block in P:
        rep = min(block, key=lambda n: str(n))
        for n in block:
            rep_map[n] = rep
    
    for u in adj:
        rep_map.setdefault(u, u)
    for vs in adj.values():
        for v in vs:
            rep_map.setdefault(v, v)

    G_o: Dict[Any, Set[Any]] = defaultdict(set)
    for u, children in adj.items():
        u_rep = rep_map[u]
        for v in children:
            v_rep = rep_map[v]
            if u_rep != v_rep:
                G_o[u_rep].add(v_rep)

    return {u: list(vs) for u, vs in G_o.items()}


def online_refine(
    full_graph: Dict[Any, Set[Any]],
    rank_attr: Dict[Any, int],
    equiv_classes: Dict[Any, Any],
    delta_edges: List[Tuple[Any, Any]],
    pred: int
) -> Tuple[Dict[Any, Any], List[Tuple[Any, Any]]]:
    # ensure all nodes are present
    for u, v in delta_edges:
        full_graph.setdefault(u, set())
        full_graph.setdefault(v, set())
        rank_attr.setdefault(u, 0)
        rank_attr.setdefault(v, 0)
        equiv_classes.setdefault(u, u)
        equiv_classes.setdefault(v, v)
        # also guard block IDs
        bu = equiv_classes[u]
        bv = equiv_classes[v]
        rank_attr.setdefault(bu, 0)
        rank_attr.setdefault(bv, 0)

    # incorporate new edges
    for u, v in delta_edges:
        full_graph[u].add(v)

    # sort edges by block rank
    block_rank = lambda n: rank_attr[equiv_classes[n]]
    delta_edges.sort(key=lambda e: block_rank(e[0]))

    # build block -> members map
    blocks: Dict[Any, Set[Any]] = defaultdict(set)
    for node, bid in equiv_classes.items():
        blocks[bid].add(node)

    expert_q: List[Tuple[Any, Any]] = []
    for u, v in delta_edges:
        bu, bv = equiv_classes[u], equiv_classes[v]
        ru, rv = rank_attr[bu], rank_attr[bv]

        if pred == 1 and bu != bv:
            if ru > rv:
                # merge block bv into bu
                for n in blocks[bv]:
                    equiv_classes[n] = bu
                blocks[bu].update(blocks[bv])
                del blocks[bv]

            elif ru == rv:
                # equal ranks: partial merges
                parents = {w for w, cs in full_graph.items() if cs & blocks[bv]}
                children = {ch for m in blocks[bu] for ch in full_graph[m]}
                for w in parents:
                    equiv_classes[w] = bu
                    blocks[bu].add(w)
                for ch in children:
                    equiv_classes[ch] = bv
                    blocks[bv].add(ch)

            else:
                # symmetric case: merge bu into bv
                for n in blocks[bu]:
                    equiv_classes[n] = bv
                blocks[bv].update(blocks[bu])
                del blocks[bu]

        elif pred == 0 and bu == bv:
            # split decision: carve out v's bisimulation frontier
            new_block = {
                n for n in blocks[bu]
                if not (
                    full_graph[n] == full_graph[v] and
                    {p for p in full_graph if n in full_graph[p]} ==
                    {p for p in full_graph if v in full_graph[p]}
                )
            }
            # move those into their own blocks
            for n in new_block:
                equiv_classes[n] = n
            blocks[bu] -= new_block
            for n in new_block:
                blocks[n] = {n}
            expert_q.append((u, v))

    return equiv_classes, expert_q