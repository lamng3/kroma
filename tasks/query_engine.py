import requests
from collections import defaultdict
from typing import List, Tuple, Dict, Any

from SPARQLWrapper import SPARQLWrapper2  
from config.constants import SPARQLEndpoints
from tasks.query_templates import (
    ncit_query_templates,
    doid_query_templates,
    dbpedia_query_templates,
)

from inference.factory import create_embedding_model
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import words, wordnet

# ensure necessary NLTK data is available
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)

ENGLISH_WORDS = set(words.words())

# initialize SciBERT embedder via our factory
scibert = create_embedding_model("huggingface", "allenai/scibert_scivocab_uncased")


def _dict_query(
    code: str,
    dict_src: Dict[str, Any],
    topk: int = 3
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Generic retrieval from a pre‑loaded dictionary of ontology records.
    Returns (parents, children, synonyms, labels).
    """
    rec = dict_src.get(code, {})
    parents   = rec.get('parents', [])
    children  = rec.get('children', [])
    synonyms  = rec.get('synonyms', [])
    labels    = rec.get('labels', [])
    # dedupe & truncate
    return (
        list(dict.fromkeys(parents))[:topk],
        list(dict.fromkeys(children))[:topk],
        list(dict.fromkeys(synonyms))[:topk],
        list(dict.fromkeys(labels))[:topk],
    )


def _sparql_query(
    endpoint: str,
    tmpl,
    code: str,
    topk: int = 3
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    SPARQL-based retrieval using provided query template class.
    """
    sparql = SPARQLWrapper2(SPARQLEndpoints[endpoint])

    def run(template: str) -> List[str]:
        sparql.setQuery(template.replace('<code>', code))
        return [b.value.lower() for b in sparql.query().bindings]

    parents, children = run(tmpl.query_parents_template), run(tmpl.query_children_template)
    synonyms, labels = run(tmpl.query_synonyms_template), run(tmpl.query_label_template)
    return (
        list(dict.fromkeys(parents))[:topk],
        list(dict.fromkeys(children))[:topk],
        list(dict.fromkeys(synonyms))[:topk],
        list(dict.fromkeys(labels))[:topk],
    )


def _conceptnet_query(
    concept: str,
    topk: int = 5
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    ConceptNet lookup with embedding‑based scoring to pick top related terms.
    """
    if concept not in ENGLISH_WORDS:
        return [], [], [], []

    data = requests.get(f'http://api.conceptnet.io/c/en/{concept}').json()
    buckets: Dict[str, List[Tuple[str, float]]] = {'parent': [], 'child': [], 'synonym': []}
    emb = scibert.encode([concept])[0]

    for edge in data.get('edges', []):
        s_lbl = edge['start']['label'].lower()
        e_lbl = edge['end']['label'].lower()
        rel   = edge['rel']['label'].lower()

        # determine relation type
        if 'syn' in rel:
            key = 'synonym'
        elif 'ClassOf' in rel or 'subClassOf' in rel:
            key = 'parent'
        else:
            key = 'child'

        # compute similarity to choose direction
        sim_s = cosine_similarity([scibert.encode([s_lbl])[0]], [emb])[0][0]
        sim_e = cosine_similarity([scibert.encode([e_lbl])[0]], [emb])[0][0]
        term  = s_lbl if sim_s > sim_e else e_lbl
        buckets[key].append((term, edge.get('weight', 1.0)))

    def top_terms(items: List[Tuple[str, float]]) -> List[str]:
        agg: Dict[str, float] = {}
        for term, wt in items:
            agg[term] = max(agg.get(term, 0.0), wt)
        # sort by weight descending
        sorted_terms = sorted(agg.items(), key=lambda x: -x[1])
        return [t for t, _ in sorted_terms[:topk]]

    return (
        top_terms(buckets['parent']),
        top_terms(buckets['child']),
        top_terms(buckets['synonym']),
        [concept],
    )


# option categories
DICT_OPTS   = {'omim','ordo','matonto','mi2','envo','sweet','mi','emmo','yago','wikidata','mouse','human'}
SPARQL_OPTS = {
    'ncit': ncit_query_templates,
    'doid': doid_query_templates,
    'dbpedia': dbpedia_query_templates,
}


def query(
    concept_code: str,
    pcmap: Dict[str, set],
    dictionary: Dict[str, Any],
    options: List[str]
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Main dispatcher: for each option, gather parents, children, synonyms, labels
    from the appropriate source and merge results.
    """
    parents, children, synonyms, labels = [], [], [], []

    for opt in options:
        try:
            if opt in DICT_OPTS:
                p, c, s, l = _dict_query(concept_code, dictionary.get(opt, {}))
            elif opt in SPARQL_OPTS:
                p, c, s, l = _sparql_query(opt, SPARQL_OPTS[opt], concept_code)
            elif opt == 'ontology':
                p = list(pcmap['parent'].get(concept_code, []))
                c = list(pcmap['child'].get(concept_code, []))
                s, l = [], []
            elif opt == 'conceptnet':
                p, c, s, l = _conceptnet_query(concept_code)
            else:
                continue

            parents.extend(p)
            children.extend(c)
            synonyms.extend(s)
            labels.extend(l)

        except Exception:
            # if any source fails, skip it
            continue

    # final de-duplication while preserving order
    return (
        [*dict.fromkeys(parents)],
        [*dict.fromkeys(children)],
        [*dict.fromkeys(synonyms)],
        [*dict.fromkeys(labels)],
    )
