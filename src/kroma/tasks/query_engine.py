"""Query strategies for dictionary, SPARQL, ontology, and ConceptNet context."""

import requests
from typing import List, Tuple, Dict, Any
from urllib.parse import urlparse

from kroma.config.constants import (
    CONCEPTNET_TOP_K,
    DEFAULT_EMBED_MODEL,
    DICT_QUERY_TOP_K,
    SPARQLEndpoints,
)
from kroma.tasks.query_templates import (
    ncit_query_templates,
    doid_query_templates,
    dbpedia_query_templates,
)

DICT_OPTS = {
    "omim", "ordo", "matonto", "mi2", "envo", "sweet", "mi", "emmo",
    "yago", "wikidata", "mouse", "human",
}
SPARQL_OPTS = {
    "ncit": ncit_query_templates,
    "doid": doid_query_templates,
    "dbpedia": dbpedia_query_templates,
}


def _clean_uri_list(uri_list: List[str]) -> List[str]:
    """Extract the final path or fragment segment from each URI."""
    cleaned = []
    for uri in uri_list:
        for part in uri.split("|"):
            parsed = urlparse(part)
            segment = parsed.path.rstrip("/").split("/")[-1] if parsed.path else part
            if "#" in segment:
                segment = segment.split("#")[-1]
            cleaned.append(segment)
    return cleaned


def _truncate(parents, children, synonyms, labels, topk):
    return (
        list(dict.fromkeys(parents))[:topk],
        list(dict.fromkeys(children))[:topk],
        list(dict.fromkeys(synonyms))[:topk],
        list(dict.fromkeys(labels))[:topk],
    )


class DictionaryStrategy:
    """Read parents, children, synonyms, and labels from a cached dictionary."""

    def fetch(self, code, pcmap, dictionary, topk=DICT_QUERY_TOP_K):
        record = dictionary.get(code, {})
        parents = [urlparse(uri).path.lstrip("/") for uri in _clean_uri_list(record.get("parents", []))]
        children = [urlparse(uri).path.lstrip("/") for uri in _clean_uri_list(record.get("children", []))]
        return _truncate(parents, children, record.get("synonyms", []), record.get("labels", []), topk)


class SparqlStrategy:
    """Read context from a SPARQL endpoint template."""

    def __init__(self, endpoint: str, templates):
        self.endpoint = endpoint
        self.templates = templates

    def fetch(self, code, pcmap, dictionary, topk=DICT_QUERY_TOP_K):
        from SPARQLWrapper import SPARQLWrapper2

        sparql = SPARQLWrapper2(SPARQLEndpoints[self.endpoint])

        def run(template: str) -> List[str]:
            sparql.setQuery(template.replace("<code>", code))
            return [binding.value.lower() for binding in sparql.query().bindings]

        return _truncate(
            run(self.templates.query_parents_template),
            run(self.templates.query_children_template),
            run(self.templates.query_synonyms_template),
            run(self.templates.query_label_template),
            topk,
        )


class OntologyStrategy:
    """Read parents and children from the in-memory ontology maps."""

    def fetch(self, code, pcmap, dictionary, topk=DICT_QUERY_TOP_K):
        parents = list(pcmap["parent"].get(code, []))
        children = list(pcmap["child"].get(code, []))
        return parents, children, [], []


class ConceptNetStrategy:
    """Score ConceptNet neighbors with the run's text embedder.

    NLTK and the embedding model load on the first ConceptNet lookup.
    """

    def __init__(self):
        self._embedder = None
        self._words = None

    def bind(self, embedder) -> None:
        self._embedder = embedder

    def _english_words(self):
        if self._words is None:
            import nltk
            from nltk.corpus import words
            nltk.download("words", quiet=True)
            self._words = set(words.words())
        return self._words

    def _model(self):
        if self._embedder is None:
            from kroma.inference.factory import create_embedding_model
            self._embedder = create_embedding_model("huggingface", DEFAULT_EMBED_MODEL)
        return self._embedder

    def fetch(self, concept, pcmap, dictionary, topk=CONCEPTNET_TOP_K):
        if concept not in self._english_words():
            return [], [], [], []

        from sklearn.metrics.pairwise import cosine_similarity

        data = requests.get(f"http://api.conceptnet.io/c/en/{concept}").json()
        buckets = {"parent": [], "child": [], "synonym": []}
        embedder = self._model()
        embedding = embedder.encode([concept])[0]

        for edge in data.get("edges", []):
            start = edge["start"]["label"].lower()
            end = edge["end"]["label"].lower()
            relation = edge["rel"]["label"].lower()
            if "syn" in relation:
                key = "synonym"
            elif "ClassOf" in relation or "subClassOf" in relation:
                key = "parent"
            else:
                key = "child"
            start_score = cosine_similarity([embedder.encode([start])[0]], [embedding])[0][0]
            end_score = cosine_similarity([embedder.encode([end])[0]], [embedding])[0][0]
            term = start if start_score > end_score else end
            buckets[key].append((term, edge.get("weight", 1.0)))

        def top_terms(items):
            scores = {}
            for term, weight in items:
                scores[term] = max(scores.get(term, 0.0), weight)
            ordered = sorted(scores.items(), key=lambda item: -item[1])
            return [term for term, _ in ordered[:topk]]

        return (
            top_terms(buckets["parent"]),
            top_terms(buckets["child"]),
            top_terms(buckets["synonym"]),
            [concept],
        )


QUERY_STRATEGIES: Dict[str, Any] = {}
_DICTIONARY = DictionaryStrategy()
_ONTOLOGY = OntologyStrategy()
_CONCEPTNET = ConceptNetStrategy()

for _name in DICT_OPTS:
    QUERY_STRATEGIES[_name] = _DICTIONARY
for _name, _templates in SPARQL_OPTS.items():
    QUERY_STRATEGIES[_name] = SparqlStrategy(_name, _templates)
QUERY_STRATEGIES["ontology"] = _ONTOLOGY
QUERY_STRATEGIES["conceptnet"] = _CONCEPTNET


def register_query_strategy(name: str, strategy) -> None:
    """Register a named context source used by query()."""
    QUERY_STRATEGIES[name] = strategy


def query(
    concept_code: str,
    pcmap: Dict[str, set],
    dictionary: Dict[str, Any],
    options: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Merge parents, children, synonyms, and labels from each requested source."""
    parents, children, synonyms, labels = [], [], [], []
    for option in options:
        strategy = QUERY_STRATEGIES.get(option)
        if strategy is None:
            continue
        try:
            found_parents, found_children, found_synonyms, found_labels = strategy.fetch(
                concept_code,
                pcmap,
                dictionary.get(option, {}) if option in DICT_OPTS else dictionary,
            )
        except Exception:
            continue
        parents.extend(found_parents)
        children.extend(found_children)
        synonyms.extend(found_synonyms)
        labels.extend(found_labels)
    return (
        [*dict.fromkeys(parents)],
        [*dict.fromkeys(children)],
        [*dict.fromkeys(synonyms)],
        [*dict.fromkeys(labels)],
    )
