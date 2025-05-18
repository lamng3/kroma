from collections import defaultdict, deque

def offline_refine(adj, rank_attr):
    V = set(adj)
    
    rho = max(rank_attr.values())
    
    # build B_i and initial partition P = {B₀, …, B_ρ}
    B = {i: {c for c, r in rank_attr.items() if r == i} for i in range(rho+1)}
    P = [set(B[i]) for i in range(rho+1)]
    
    # build incoming edges map
    incoming = defaultdict(set)
    for u, children in adj.items():
        for v in children:
            incoming[v].add(u)
    
    # refinement passes
    for i in range(rho+1):
        Bi = B[i]
        Di = [X for X in P if X and X.issubset(Bi)]
        # collapse(P, X) — we defer actual graph collapse until the end,
        # but we leave this here so future splits use the updated P
        # (no-op since we only use P for splitting)
        
        downstream = set().union(*(B[j] for j in range(i+1, rho+1)))
        new_P = []
        for C in P:
            # If C lives strictly in downstream levels, consider splitting
            if C and C.issubset(downstream):
                # for each pivot c in B_i, attempt to split C by adjacency
                split_occurred = False
                for c in Bi:
                    # C1 = members of C that have an edge to or from c
                    C1 = {n for n in C if (n in adj.get(c, [])) or (c in incoming.get(n, []))}
                    C2 = C - C1
                    if C1 and C2:
                        # split
                        new_P.append(C1)
                        new_P.append(C2)
                        split_occurred = True
                        break
                if not split_occurred:
                    new_P.append(C)
            else:
                # leave blocks that are not downstream untouched
                new_P.append(C)
        P = new_P
    
    # construct G_o by collapsing each block into its representative
    # choose the minimal node id as rep for determinism
    rep_map = {}
    for block in P:
        rep = min(block)
        for n in block:
            rep_map[n] = rep
    
    G_o = defaultdict(set)
    for u, children in adj.items():
        u_rep = rep_map[u]
        for v in children:
            v_rep = rep_map[v]
            if u_rep != v_rep:
                G_o[u_rep].add(v_rep)
    
    # to keep the same type, convert sets back to lists
    return {u: list(vs) for u, vs in G_o.items()}


def online_refine(equiv_classes, src, tgt, pred):
    expert_q = []
    src_cid = equiv_classes.get(src, src)
    tgt_cid = equiv_classes.get(tgt, tgt)

    if pred == 1:
        # merge classes
        new_cid = min(src_cid, tgt_cid)
        for n, cid in list(equiv_classes.items()):
            if cid == src_cid or cid == tgt_cid:
                equiv_classes[n] = new_cid

    else:
        # split if they were in the same class: flag for expert review
        if src_cid == tgt_cid:
            equiv_classes[tgt] = tgt
            expert_q.append((src, tgt))

    return equiv_classes, expert_q
