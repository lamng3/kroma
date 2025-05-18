from collections import defaultdict, deque

def offline_refine(equiv_classes, rank_attr):
    # Build reverse mapping: class_id -> set(nodes)
    classes = defaultdict(set)
    for node, cid in equiv_classes.items():
        classes[cid].add(node)

    # Iteratively refine until stable
    changed = True
    while changed:
        changed = False
        new_classes = {}
        # For each class, split by sorted multiset of neighbor class-ids
        for cid, members in classes.items():
            signature_map = defaultdict(set)
            for u in members:
                # signature: (rank, sorted(incoming_class_ids), sorted(outgoing_class_ids))
                incoming = sorted({equiv_classes[p] for p in u.in_edges})
                outgoing = sorted({equiv_classes[c] for c in u.out_edges})
                sig = (rank_attr[u], tuple(incoming), tuple(outgoing))
                signature_map[sig].add(u)
            # If you split into >1 group:
            if len(signature_map) > 1:
                changed = True
            # assign new class ids
            for group in signature_map.values():
                new_cid = min(hash(n) for n in group)  # or any deterministic ID
                for n in group:
                    new_classes[n] = new_cid
        # re-index classes for next pass
        classes = defaultdict(set)
        for n, cid in new_classes.items():
            classes[cid].add(n)
        equiv_classes = new_classes

    return equiv_classes


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
