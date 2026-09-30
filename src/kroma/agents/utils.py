import re
import random
from collections import defaultdict
from typing import List, Tuple, Dict

def extract_yes_no(response: str) -> int:
    """Parse <answer>yes/no</answer> tags into 1/0."""
    m = re.search(r'<answer>\s*(yes|no)\s*</answer>', response, re.IGNORECASE)
    return 1 if m and m.group(1).lower() == 'yes' else 0

def extract_confidence(response: str) -> int:
    """Parse <confidence>0-10</confidence> tags into integer."""
    m = re.search(r'<confidence>\s*(\d{1,2})\s*</confidence>', response)
    val = int(m.group(1)) if m else 0
    return val if 0 <= val <= 10 else 0

def majority_vote(answers: List[Tuple[int,int]]) -> Tuple[int,int]:
    """
    Compute majority answer and average confidence.
    answers: list of (answer, confidence) tuples.
    """
    counts: Dict[int, List[int]] = defaultdict(list)
    for ans, conf in answers:
        counts[ans].append(conf)
    # pick ans with most votes
    maj = max(counts, key=lambda k: len(counts[k]))
    avg_conf = int(sum(counts[maj]) / len(counts[maj]))
    return maj, avg_conf

def active_learning_score(
    confidence: int,
    f1_score: float,
    gamma: float = 1.0,
    delta: float = 1.0,
    epsilon: float = 0.1
) -> float:
    """Compute noisy AL acceptance score."""
    noise = random.gauss(0,1) * epsilon * f1_score
    return (gamma + delta) * confidence - 11 * delta + noise

def simple_match(src: Tuple, tgt: Tuple) -> bool:
    """Basic bisimulation: same term string."""
    return src[0] == tgt[0]
