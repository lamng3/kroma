from collections import defaultdict, deque
from typing import Dict, Iterable, Any, Tuple, List, Set

# -----------------------------------------------------------------------------
# Step 1: Compute node ranks in a DAG
# -----------------------------------------------------------------------------

def compute_node_ranks(
    graph: Dict[Any, Set[Any]]
) -> Dict[Any, int]:
    """
    Given a DAG (parent -> children), compute for each node its 'rank'
    (longest distance from any root). Roots (no incoming edges) have rank 0.
    """
    indegree = defaultdict(int)
    for parent, children in graph.items():
        indegree[parent]  # ensure key exists
        for c in children:
            indegree[c] += 1

    queue = deque([n for n, deg in indegree.items() if deg == 0])
    ranks = {n: 0 for n in queue}

    while queue:
        node = queue.popleft()
        for child in graph.get(node, set()):
            ranks[child] = max(ranks.get(child, 0), ranks[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return ranks

# -----------------------------------------------------------------------------
# Step 2: Bucket nodes by their rank
# -----------------------------------------------------------------------------

def bucket_by_rank(
    ranks: Dict[Any, int]
) -> Tuple[Dict[int, Set[Any]], int]:
    """
    Group nodes into buckets by rank value.
    Returns a dict rank->set(nodes) and the maximum rank.
    """
    buckets: Dict[int, Set[Any]] = defaultdict(set)
    max_rank = 0
    for node, r in ranks.items():
        buckets[r].add(node)
        if r > max_rank:
            max_rank = r
    return buckets, max_rank

# -----------------------------------------------------------------------------
# Step 3: Initial partition (all nodes in one block)
# -----------------------------------------------------------------------------

def initialize_partition(
    nodes: Iterable[Any]
) -> List[Set[Any]]:
    """
    The initial partition P is a list containing one block of all nodes.
    """
    return [set(nodes)]

# -----------------------------------------------------------------------------
# Step 4: Collapse a block of nodes into a single representative
# -----------------------------------------------------------------------------

def collapse(
    graph: Dict[Any, Set[Any]],
    block: Set[Any]
) -> None:
    """
    Collapse all nodes in `block` into a single representative node.
    Picks the lexicographically smallest element as rep, reassigns edges, and removes others.
    """
    if not block:
        return
    rep = sorted(block)[0]
    to_merge = block - {rep}

    # ensure rep has an entry
    graph.setdefault(rep, set())

    # redirect children from merged nodes to rep
    for node in to_merge:
        for child in list(graph.get(node, set())):
            if child not in block:
                graph[rep].add(child)
        graph.pop(node, None)

    # remove any edges from rep to merged nodes
    graph[rep] -= to_merge

    # update parent pointers for all remaining nodes
    for parent, children in list(graph.items()):
        if parent not in block:
            new_children = set()
            for c in children:
                if c in to_merge:
                    new_children.add(rep)
                else:
                    new_children.add(c)
            graph[parent] = new_children

# -----------------------------------------------------------------------------
# Step 5: Refine partition at a given rank
# -----------------------------------------------------------------------------

def refine_partition(
    graph: Dict[Any, Set[Any]],
    P: List[Set[Any]],
    buckets: Dict[int, Set[Any]],
    current_rank: int,
    max_rank: int
) -> List[Set[Any]]:
    """
    Refine partition P by splitting blocks based on connectivity to nodes of rank=current_rank,
    and collapsing blocks entirely at that rank.
    """
    new_P: List[Set[Any]] = []
    # nodes with rank higher than current_rank
    higher_nodes = set().union(*(buckets[r] for r in range(current_rank+1, max_rank+1)))

    for block in P:
        # collapse entire block if it's exactly the current bucket
        if block.issubset(buckets[current_rank]):
            collapse(graph, block)
            new_P.append(block)
            continue

        # otherwise, split block based on connections to each node e in current bucket
        subsets = [block]
        for e in buckets[current_rank]:
            updated: List[Set[Any]] = []
            for subset in subsets:
                C1 = {m for m in subset if m in graph.get(e, set())}
                C2 = subset - C1
                if C1 and C2:
                    updated.extend([C1, C2])
                else:
                    updated.append(subset)
            subsets = updated
        new_P.extend(subsets)

    return new_P

# -----------------------------------------------------------------------------
# Step 6: Top-level bisimulation
# -----------------------------------------------------------------------------

def bisimulation(
    graph: Dict[Any, Set[Any]]
) -> List[Tuple[Any, Any]]:
    """
    Run bisimulation and extract pairs in same final blocks.
    """
    ranks = compute_node_ranks(graph)
    buckets, rho = bucket_by_rank(ranks)
    P = initialize_partition(graph.keys())

    for i in range(rho + 1):
        P = refine_partition(graph, P, buckets, i, rho)

    A_prime: List[Tuple[Any, Any]] = []
    for block in P:
        if len(block) > 1:
            for a in block:
                for b in block:
                    if a != b:
                        A_prime.append((a, b))
    return A_prime
