import json
from pathlib import Path
from typing import Tuple, Set, TextIO, Dict, Any, List


def load_config(filepath: str) -> Any:
    """
    Load configuration from a JSON or JSONL file. Returns:
      - dict for .json
      - list of dicts for .jsonl
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    if path.suffix == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    elif path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    else:
        raise ValueError(f"Unsupported config extension: {path.suffix}")


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
    """
    Determine predictions JSONL path, load existing pairs, and open file handle.
    Returns (set of (source, target), file handle in append+ mode).
    """
    # decide subfolder and filename
    suffix = f"{agent_name or method_name}_predictions.jsonl"
    if debate:
        subdir = f"results/debate/{task_name}"
    elif active_learning:
        subdir = f"results/active/{task_name}"
    elif compare_models:
        subdir = f"results/compare/{task_name}"
    elif baseline:
        subdir = f"results/baseline/{task_name}"
    elif reasoning:
        subdir = f"results/reasoning/{task_name}"
    elif size:
        subdir = f"results/scale/{task_name}"
    elif not bisim:
        subdir = f"results/nobisim/{task_name}"
    else:
        subdir = f"results/{task_name}"

    # override filename for scale/reasoning if needed
    if size and not reasoning:
        suffix = f"{size}_predictions.jsonl"
    if reasoning:
        suffix = f"{size or 'all'}_predictions.jsonl"
    if active_learning or compare_models or baseline or not bisim:
        suffix = f"{agent_name}_predictions.jsonl"

    base_dir = Path("experiments/kroma-eval") / subdir
    base_dir.mkdir(parents=True, exist_ok=True)
    filepath = base_dir / suffix

    predicted: Set[Tuple[str,str]] = set()
    if filepath.exists():
        for line in filepath.read_text(encoding='utf-8').splitlines():
            if not line:
                continue
            try:
                rec = json.loads(line)
                src = rec.get('source')
                tgt = rec.get('target')
                if src and tgt:
                    predicted.add((src, tgt))
            except json.JSONDecodeError:
                continue

    # open file for append+read
    file_handle = filepath.open('a+', encoding='utf-8')
    file_handle.seek(0)
    print(f"Using prediction file: {filepath}")
    return predicted, file_handle
