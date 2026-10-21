"""KROMA as a matching agent on a MemCommit blackboard, refining equivalence classes with KROMA's own algorithm.

`KromaMatcher` consumes `match_request` memories (a source concept and a
target concept). For each it asks a judge for KROMA's three outputs
(prediction, confidence, accept), writes a `match` memory, and for an
accepted pair runs `kroma.algorithms.refinement.online_refine`, the online
refinement step of the ISWC 2025 paper, on the merged concept graph. Pairs
that refinement cannot settle become `review` memories for an expert.

An agent that looks up a concept's matches derives its result from
`alignment_read(concept)` as well as from the matches it found, so a match
that arrives later, including the first one, sends that work back to be redone.

The judge is where KROMA's language-model debate plugs in
(`kroma.agents.prompter.inference`); any callable with the same outputs works,
which is how the tests run without a model.
MemCommit is optional for KROMA: install it, or clone it beside this repository.

The matcher's in-process state (the graph, ranks, equivalence classes) is a
cache of the store: on its first input it replays every live accepted match,
so a restarted matcher refines from where the memory says it was.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from memcommit import AgentSession, Memory, Transaction, query_ref

MATCH_REQUEST, MATCH, REVIEW = "match_request", "match", "review"

# judge(source, target) -> (prediction 0 or 1, confidence, accept)
Judge = Callable[[str, str], tuple[int, float, bool]]


def _kroma() -> tuple[Any, Any]:
    from kroma.algorithms.refinement import online_refine
    from kroma.algorithms.utils import compute_node_ranks

    return online_refine, compute_node_ranks


def match_key(source: str, target: str) -> str:
    return f"{source}|{target}"


class KromaMatcher:
    consumes = frozenset({MATCH_REQUEST})

    def __init__(
        self,
        graph: dict[str, Iterable[str]],
        judge: Judge,
        name: str = "kroma.matcher",
        derive_from: Callable[[], Iterable[str]] = tuple,
    ) -> None:
        """`graph` maps each concept of both ontologies to its children; `derive_from` adds provenance refs."""
        self.online_refine, compute_node_ranks = _kroma()
        self.name = name
        self.judge = judge
        self.derive_from = derive_from
        self.graph = {node: set(children) for node, children in graph.items()}
        for children in list(self.graph.values()):
            for child in children:
                self.graph.setdefault(child, set())
        self.ranks = compute_node_ranks(self.graph)
        self.equiv = {node: node for node in self.graph}
        self._replayed = False

    def block(self, concept: str) -> str:
        return self.equiv.get(concept, concept)

    def _refine(self, source: str, target: str, prediction: int) -> list[tuple[str, str]]:
        self.equiv, expert = self.online_refine(self.graph, self.ranks, self.equiv, [(source, target)], prediction)
        return list(expert or [])

    def _replay(self, session: AgentSession) -> None:
        for m in session.find(kind=MATCH, source=self.name):
            if m.body["accepted"]:
                self._refine(m.body["source"], m.body["target"], m.body["prediction"])
        self._replayed = True

    def handle(self, memory: Memory, tx: Transaction, session: AgentSession) -> None:
        if not self._replayed:
            self._replay(session)
        source, target = memory.body["source"], memory.body["target"]
        prediction, confidence, accept = self.judge(source, target)
        refs = [memory.id, *self.derive_from()]
        body = {
            "source": source,
            "target": target,
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4),
            "accepted": bool(accept),
        }
        if accept:
            for u, v in self._refine(source, target, int(prediction)):
                tx.put(REVIEW, match_key(u, v), {"source": u, "target": v}, derived_from=refs)
            body["same_class"] = self.block(source) == self.block(target)
        tx.put(MATCH, match_key(source, target), body, derived_from=refs)


def alignment_read(source: str) -> str:
    """What to derive from after looking up `source`'s matches: any later match for it makes the result stale."""
    return query_ref(MATCH, source=source)


def accepted_matches(session: AgentSession, source: str | None = None) -> list[Memory]:
    """Live matches the matcher accepted as the same concept, optionally for one source concept."""
    return [
        m
        for m in session.find(kind=MATCH)
        if m.body["accepted"] and m.body["prediction"] == 1 and (source is None or m.body["source"] == source)
    ]
