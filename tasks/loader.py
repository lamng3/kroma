import csv
import random
import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
from rdflib import Graph, RDF, OWL, RDFS

random.seed(2025)


class OntologyLoader(ABC):
    """Abstract base for ontology data loaders."""
    @abstractmethod
    def load(self) -> Tuple[
        Dict[Tuple[str, str], Any],  # source graph
        Dict[Tuple[str, str], Any],  # target graph
        List[Tuple[Tuple[str, str], Tuple[str, str], str]]  # alignments
    ]:
        pass


class CsvAlignmentLoader(OntologyLoader):
    """Loads pairwise alignments from CSV benchmark files."""
    def __init__(self, csv_paths: List[str], dataset: str = "ncit-doid"):
        self.csv_paths = csv_paths
        self.dataset = dataset

    def load(self) -> Tuple[
        Dict[Tuple[str, str], Any],
        Dict[Tuple[str, str], Any],
        List[Tuple[Tuple[str, str], Tuple[str, str], str]]
    ]:
        G1 = defaultdict(list)
        G2 = defaultdict(list)
        alignments = []

        for path in self.csv_paths:
            reader = csv.DictReader(Path(path).open(encoding="utf-8"))
            for row in reader:
                if self.dataset == 'ncit-doid':
                    s_lbl, s_uri = row["src_code"].strip(), row["src_ety"].strip()
                    t_lbl, t_uri = row["tgt_code"].strip(), row["tgt_ety"].strip()
                    candidates = ast.literal_eval(row.get("tgt_cands", "[]"))
                    truth = row.get("score", "0").strip()
                    for cand in candidates or [t_uri]:
                        lbl = '1' if (cand == t_uri and truth == '1') else '0'
                        cand_lbl = cand.split('#')[-1] if '#' in cand else cand
                        alignments.append(((s_lbl, s_uri), (cand_lbl, cand), lbl))
                else:
                    s_lbl, s_uri = row.get('source',''), row.get('source_uri','')
                    t_lbl, t_uri = row.get('target',''), row.get('target_uri','')
                    lbl = row.get('Relation', row.get('relation','1')).strip()
                    alignments.append(((s_lbl, s_uri), (t_lbl, t_uri), lbl))

                # ensure nodes appear
                G1.setdefault((s_lbl, s_uri), [])
                G2.setdefault((t_lbl, t_uri), [])

        return G1, G2, alignments


class OwlOntologyLoader(OntologyLoader):
    """Loads class hierarchy and synonyms from an OWL/RDF file."""
    def __init__(self, owl_path: str):
        self.owl_path = owl_path

    def load(self) -> Tuple[
        Dict[Tuple[str, str], Any],
        Dict[Tuple[str, str], Any],
        List[Tuple[Tuple[str, str], Tuple[str, str], str]]
    ]:
        g = Graph()
        g.parse(self.owl_path, format='xml')

        classes = {}
        for s, _, o in g.triples((None, RDF.type, None)):
            if o in (OWL.Class, RDFS.Class):
                classes[s] = _get_uri_name(str(s))

        G1 = defaultdict(set)
        for s, _, o in g.triples((None, RDFS.subClassOf, None)):
            if s in classes and o in classes:
                G1[(classes[o], str(o))].add((classes[s], str(s)))

        synonyms = defaultdict(set)
        for s, _, o in g.triples((None, OWL.equivalentClass, None)):
            if s in classes and o in classes:
                synonyms[classes[s]].add(classes[o])
                synonyms[classes[o]].add(classes[s])

        return G1, synonyms, []


def _get_uri_name(uri: str) -> str:
    if '#' in uri:
        return uri.split('#')[-1]
    return uri.rstrip('/').split('/')[-1].lower()
