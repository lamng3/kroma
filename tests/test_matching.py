from kroma.agents.utils import extract_confidence, extract_yes_no, majority_vote
from kroma.cli import build_run_parser
from kroma.metrics.scoring import calculate_metrics
from kroma.tasks.query_engine import query


def test_extract_yes_no():
    assert extract_yes_no("<answer>yes</answer>") == 1
    assert extract_yes_no("<answer>No</answer>") == 0
    assert extract_yes_no("maybe") == 0


def test_extract_confidence():
    assert extract_confidence("<confidence>7</confidence>") == 7
    assert extract_confidence("<confidence>99</confidence>") == 0
    assert extract_confidence("none") == 0


def test_majority_vote():
    assert majority_vote([(1, 8), (1, 6), (0, 9)]) == (1, 7)


def test_metrics():
    precision, recall, f1 = calculate_metrics([1, 0, 1], [1, 1, 0])
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5
    assert calculate_metrics([], []) == (0.0, 0.0, 0.0)


def test_debate_flag_defaults_to_false():
    args = build_run_parser().parse_args(["--method_config", "cfg", "--llm", "model"])
    assert args.debate is False


def test_debate_flag_is_set():
    args = build_run_parser().parse_args(
        ["--method_config", "cfg", "--llm", "model", "--debate"]
    )
    assert args.debate is True


def test_dictionary_query_strategy():
    dictionary = {
        "envo": {
            "water": {
                "parents": ["http://example.org/liquid"],
                "children": [],
                "synonyms": ["H2O"],
                "labels": ["water"],
            }
        }
    }
    parents, children, synonyms, labels = query(
        "water",
        {"parent": {}, "child": {}},
        dictionary,
        ["envo"],
    )
    assert parents == ["liquid"]
    assert children == []
    assert synonyms == ["H2O"]
    assert labels == ["water"]
