"""KROMA's matcher on a MemCommit blackboard. Skips unless memcommit is installed or cloned beside kroma."""

import sys
from pathlib import Path

import pytest

MEMCOMMIT = Path(__file__).resolve().parents[2] / "memcommit" / "src"
if MEMCOMMIT.is_dir() and str(MEMCOMMIT) not in sys.path:
    sys.path.append(str(MEMCOMMIT))
memcommit = pytest.importorskip("memcommit")

from kroma.integrations.memcommit import KromaMatcher, accepted_matches, alignment_read  # noqa: E402

GRAPH = {"Material": {"Metal", "Polymer"}, "Substance": {"MetallicMaterial", "Plastic"}}
VERDICTS = {("Metal", "MetallicMaterial"): (1, 0.93, True), ("Metal", "Plastic"): (0, 0.88, True)}


def judge(s, t):
    return VERDICTS.get((s, t), (0, 0.5, False))


def test_matcher_refines_and_resumes_from_memory():
    store = memcommit.MemoryStore()
    matcher = KromaMatcher(GRAPH, judge)
    board = memcommit.Blackboard(store, [matcher])
    for s, t in [("Metal", "MetallicMaterial"), ("Metal", "Plastic"), ("Polymer", "Plastic")]:
        store.put("user", "match_request", f"{s}|{t}", {"source": s, "target": t})
    board.run()
    session = board.sessions["kroma.matcher"]
    assert [m.body["target"] for m in accepted_matches(session)] == ["MetallicMaterial"]
    assert store.latest("match", "Polymer|Plastic").body["accepted"] is False
    # Whatever KROMA's refinement decided is recorded with the match...
    same = matcher.block("Metal") == matcher.block("MetallicMaterial")
    assert store.latest("match", "Metal|MetallicMaterial").body["same_class"] is same

    # ...and a restarted matcher rebuilds exactly that state from memory alone.
    restarted = KromaMatcher(GRAPH, judge)
    restarted._replay(session)
    assert restarted.equiv == matcher.equiv


def test_a_first_match_redoes_work_that_read_no_match():
    store = memcommit.MemoryStore()
    board = memcommit.Blackboard(store, [KromaMatcher(GRAPH, judge)])
    label = store.put("agent", "label", "Metal", "unaligned", derived_from=[alignment_read("Metal")])
    store.put("user", "match_request", "Metal|MetallicMaterial", {"source": "Metal", "target": "MetallicMaterial"})
    board.run()
    assert store.status(label.id) == memcommit.STALE
