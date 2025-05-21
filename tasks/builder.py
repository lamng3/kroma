import random
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from tasks.query_engine import query

random.seed(2025)


def build_ontology_expansion_task(
    G: Dict[Any, List[Any]],
    keep_ratio: float = 0.5
) -> Tuple[Dict[Any, set], Dict[Any, set], Dict[Any, List[Any]]]:
    """
        G_keep: a subset to retain in the prompt
        G_expand: the remainder to use for expansion context
    """
    G_keep, G_expand = defaultdict(set), defaultdict(set)
    for node, neighbors in G.items():
        n = len(neighbors)
        k = max(1, int(n * keep_ratio)) if n else 0
        kept = set(random.sample(neighbors, k)) if k else set()
        G_keep[node] = kept
        G_expand[node] = set(neighbors) - kept
    return G_keep, G_expand, G


def build_parent_child_map(
    G: Dict[Tuple[str, str], Any]
) -> Tuple[Dict[str, set], Dict[str, set]]:
    parent_map, child_map = defaultdict(set), defaultdict(set)
    for (label, _), children in G.items():
        child_map[label] = {c[0] for c in children}
        for c in children:
            parent_map[c[0]].add(label)
    return parent_map, child_map


def build_ontology_matching_task(
    G1: Dict[Tuple[str, str], Any],
    G2: Dict[Tuple[str, str], Any],
    alignments: List[Tuple[Any, Any, str]],
    sample_sz: int = 50,
    dictionary: Dict[str, Any] = None,
    query_opts: List[str] = None
) -> Tuple[
    Dict[Tuple[str, str], Any],
    Dict[Tuple[str, str], Any],
    List[Tuple[Any, Any, str]],
    Dict[Tuple[str, str], Dict[str, List[str]]],
    Dict[Tuple[str, str], Dict[str, List[str]]],
    Dict[str, set],
    Dict[str, set]
]:
    # sample positives/negatives
    positives = [m for m in alignments if m[2] == "1"]
    negatives = [m for m in alignments if m[2] != "1"]

    if not negatives:
        seen = {(s[0], t[0]) for s, t, _ in alignments}
        negatives = [
            (s, t, "0")
            for s, t in itertools.product(G1, G2)
            if (s[0], t[0]) not in seen
        ]

    pos_sel = random.sample(positives, min(len(positives), sample_sz))
    neg_sel = random.sample(negatives, min(len(negatives), sample_sz))
    sampled = pos_sel + neg_sel
    random.shuffle(sampled)

    # build parent/child maps
    pm1, cm1 = build_parent_child_map(G1)
    pm2, cm2 = build_parent_child_map(G2)

    # enrich metadata using URI lookup
    OS_meta: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    OT_meta: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    if dictionary and query_opts:
        for key in G1:
            lookup_key = key[0]  # use source URI as the dictionary lookup
            p, c, s, l = query(
                lookup_key,
                {'parent': pm1, 'child': pm1},
                dictionary,
                query_opts
            )
            OS_meta[lookup_key] = {'parents': p, 'children': c, 'synonyms': s, 'labels': l}

        for key in G2:
            lookup_key = key[0]
            p, c, s, l = query(
                lookup_key,
                {'parent': pm2, 'child': pm2},
                dictionary,
                query_opts
            )
            OT_meta[lookup_key] = {'parents': p, 'children': c, 'synonyms': s, 'labels': l}

    return (
        G1,
        G2,
        sampled,
        OS_meta,
        OT_meta,
        {'parent': pm1, 'child': cm1},
        {'parent': pm2, 'child': cm2}
    )
