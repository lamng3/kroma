import nltk
from typing import List, Tuple, Dict, Any, Optional

from .templates import SYSTEM_PROMPT, TASK_PROMPT, ROUND_PROMPT, LAST_ROUND_PROMPT

# ensure wordlist for cleaning
nltk.download('words', quiet=True)
from nltk.corpus import words
ENGLISH_WORDS = set(words.words())

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def clean_terms(terms: List[str]) -> List[str]:
    """Filter out short, numeric or underscore-containing terms."""
    return [t for t in terms if len(t) >= 2 and '_' not in t and not any(c.isdigit() for c in t)]


def list_to_str(items: List[Any]) -> str:
    """Convert list of arbitrary items into a comma-separated string (or 'none')."""
    return ', '.join(map(str, items)) if items else 'none'


# ----------------------------------------------------------------------------
# Few‑shot formatting
# ----------------------------------------------------------------------------

def format_demonstrations(
    examples: List[Tuple[
        Tuple[str, List[str], List[str], List[str], List[str]],
        Tuple[str, List[str], List[str], List[str], List[str]],
        int,  # 1=Yes, 0=No
        int
    ]]
) -> str:
    """Turn few‑shot examples into a prompt block."""
    lines = ["Examples:"]
    for src, tgt, lbl, conf in examples:
        ans = 'Yes' if lbl == 1 else 'No'
        lines.append(f"Source: {src[0]}; Target: {tgt[0]} -> {ans} (confidence {conf})")
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------------

def build_task_prompt(
    source: Tuple[str, List[str], List[str], List[str], List[str]],
    target: Tuple[str, List[str], List[str], List[str], List[str]],
    examples: Optional[List[Tuple[Any, Any, int, int]]] = None,
    reasoning: bool = False,
) -> Tuple[str, str]:
    """
    Build (system, user) prompts for the main task.
    """
    _, s_par, s_chi, s_syn, s_lab = source
    _, t_par, t_chi, t_syn, t_lab = target

    # prepare and clean
    s_lab = clean_terms(list(set(s_lab + [source[0]])))
    t_lab = clean_terms(list(set(t_lab + [target[0]])))
    s_par, s_chi, s_syn = map(clean_terms, (s_par, s_chi, s_syn))
    t_par, t_chi, t_syn = map(clean_terms, (t_par, t_chi, t_syn))

    demos = format_demonstrations(examples) if examples else ''
    user = TASK_PROMPT.format(
        demonstrations=demos,
        src_names=list_to_str(s_lab),
        src_parents=list_to_str(s_par),
        src_children=list_to_str(s_chi),
        src_synonyms=list_to_str(s_syn),
        tgt_names=list_to_str(t_lab),
        tgt_parents=list_to_str(t_par),
        tgt_children=list_to_str(t_chi),
        tgt_synonyms=list_to_str(t_syn),
    )
    return SYSTEM_PROMPT, user


def build_round_prompt(
    history: Dict[int, List[int]]
) -> Tuple[str, str]:
    """
    Build (system, user) prompt for intermediate rounds.
    """
    agent_str = '\n'.join(f"Agent {aid+1}: {' '.join(map(str, ans))}"
                          for aid, ans in history.items())
    return SYSTEM_PROMPT, ROUND_PROMPT.format(agent_answers=agent_str)


def build_last_round_prompt(
    history: Dict[int, List[int]]
) -> Tuple[str, str]:
    """
    Build (system, user) prompt for the final round.
    """
    agent_str = '\n'.join(f"Agent {aid+1}: {' '.join(map(str, ans))}"
                          for aid, ans in history.items())
    return SYSTEM_PROMPT, LAST_ROUND_PROMPT.format(agent_answers=agent_str)
