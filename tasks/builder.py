import random
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Any

random.seed(2025)


def build_ontology_expansion_task(
    G: Dict[Any, List[Any]],
    keep_ratio: float = 0.5
) -> Tuple[Dict[Any, set], Dict[Any, set], Dict[Any, List[Any]]]:
    """
    Splits each node's neighbor set into:
      - G_keep: a subset to retain in the prompt
      - G_expand: the remainder to use for expansion context
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
    """
    Constructs:
      - parent_map[label] = set(child_labels)
      - child_map[label]  = set(parent_labels)
    """
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
    sample_sz: int = 50
) -> Tuple[
    Dict[Tuple[str, str], Any],
    Dict[Tuple[str, str], Any],
    List[Tuple[Any, Any, str]],
    Dict[str, set],
    Dict[str, set]
]:
    """
    Samples up to sample_sz positives and negatives, then returns:
      G1, G2, sampled_alignments, 
      {'parent':parent_map1,'child':child_map1},
      {'parent':parent_map2,'child':child_map2}
    """
    positives = [m for m in alignments if m[2] == "1"]
    negatives = [m for m in alignments if m[2] != "1"]

    # generate negatives if none exist
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

    pm1, cm1 = build_parent_child_map(G1)
    pm2, cm2 = build_parent_child_map(G2)

    return G1, G2, sampled, {"parent": pm1, "child": cm1}, {"parent": pm2, "child": cm2}
