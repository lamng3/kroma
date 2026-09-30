import json
import os
from pathlib import Path
from typing import Tuple, Set, TextIO, Dict, Any, List

def load_config(filepath: str) -> Any:
    """
    Load configuration from a JSON or JSONL file. Returns:
      - dict for .json
      - dict for single-line .jsonl
      - list of dicts for multi-line .jsonl
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_cache(filepath: str, key_field: str = 'concept_code') -> Dict[str, Dict[str, List[Any]]]:
    """
    Load a JSONL cache file, merging entries by the given key_field.
    Each entry should have lists under 'parents', 'children', 'synonyms', 'labels'.
    Returns a dict mapping key -> merged record with lists.
    """
    cache: Dict[str, Dict[str, Set[Any]]] = {}
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {filepath}")

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get(key_field)
            if key is None:
                continue
            if key not in cache:
                cache[key] = {field: set() for field in ['parents','children','synonyms','labels']}
            for field in ['parents','children','synonyms','labels']:
                values = record.get(field)
                if isinstance(values, list):
                    cache[key][field].update(values)
    # convert sets back to lists
    return {k: {fld: list(vset) for fld, vset in fields.items()} for k, fields in cache.items()}


def get_predicted_pairs_and_file(
    task_name: str,
    method_name: str,
    agent_name: str = None,
    debate: bool = False,
    size: str = None,
    reasoning: bool = False,
    baseline: bool = False,
    active_learning: bool = False,
    compare_models: bool = False,
    bisim: bool = True
) -> Tuple[Set[Tuple[str,str]], TextIO]:
    # decide mode folder
    mode = "baseline" if baseline else "results"

    # sanitize agent_name to a filesystem‐safe string
    safe_agent = agent_name.replace("/", "-")

    # build directory: results/<mode>/<task_name>/
    results_dir = Path("results") / mode / task_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # build filename and full path
    filename = f"{safe_agent}_{task_name}.jsonl"
    filepath = results_dir / filename

    # ensure the file exists
    filepath.touch(exist_ok=True)

    # load existing predictions
    keys: Set[Tuple[str,str]] = set()
    predicted: List[Dict] = []
    for line in filepath.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            key = (rec['source'], rec['target'])
            if key not in keys:
                keys.add(key)
                predicted.append(rec)
        except json.JSONDecodeError:
            continue

    # open for append+read and seek to end
    handle = filepath.open("a+", encoding="utf-8")
    handle.seek(0, os.SEEK_END)

    print(f"Using prediction file: {filepath}")
    print(keys)
    print(predicted)
    return keys, predicted, handle