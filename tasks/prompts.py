import re
from typing import List, Tuple, Dict, Any
import nltk
from nltk.corpus import words

# ensure wordlist
nltk.download('words', quiet=True)
ENGLISH_WORDS = set(words.words())

# ----------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are OntoMapper, an agent that determines whether two ontology concepts are closely related based on their names and hierarchical relationships. Given the following inputs for each concept: names, parent concepts, child concepts, and synonyms, output Yes or No enclosed in <answer> tags, followed by a confidence 0-10 in <confidence> tags.
"""

TASK_PROMPT = """
Task: Determine if two ontology concepts refer to the same real-world entity.

Source Names: {src_names}
Source Parents: {src_parents}
Source Children: {src_children}
Source Synonyms: {src_synonyms}

Target Names: {tgt_names}
Target Parents: {tgt_parents}
Target Children: {tgt_children}
Target Synonyms: {tgt_synonyms}

Only output:<answer>Yes</answer> or <answer>No</answer>
and <confidence>0-10</confidence>.
"""

ROUND_PROMPT = """
Previous agent answers:
{agent_answers}

Based on these, give an updated <answer>Yes</answer> or <answer>No</answer> and <confidence>0-10</confidence>.
"""

LAST_ROUND_PROMPT = """
Final round. Previous answers:
{agent_answers}

Provide a conclusive <answer>Yes</answer> or <answer>No</answer> and <confidence>0-10</confidence>.
"""

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def clean_terms(terms: List[str]) -> List[str]:
    """Filter out short, numeric, underscore-containing, or empty terms."""
    cleaned = []
    for t in terms:
        if len(t) < 2 or '_' in t or any(c.isdigit() for c in t):
            continue
        cleaned.append(t)
    return cleaned


def list_to_str(items: List[str]) -> str:
    return ', '.join(items) if items else 'none'

# ----------------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------------

def build_task_prompt(
    source: Tuple[str, List[str], List[str], List[str], List[str]],
    target: Tuple[str, List[str], List[str], List[str], List[str]]
) -> Tuple[str, str]:
    """
    Build the system and task-level user prompt.
    """
    _, s_par, s_chi, s_syn, s_lab = source
    _, t_par, t_chi, t_syn, t_lab = target

    # include term itself in labels
    s_lab = clean_terms(list(set(s_lab + [source[0]])))
    t_lab = clean_terms(list(set(t_lab + [target[0]])))
    s_par, s_chi, s_syn = map(clean_terms, (s_par, s_chi, s_syn))
    t_par, t_chi, t_syn = map(clean_terms, (t_par, t_chi, t_syn))

    user_prompt = TASK_PROMPT.format(
        src_names=list_to_str(s_lab),
        src_parents=list_to_str(s_par),
        src_children=list_to_str(s_chi),
        src_synonyms=list_to_str(s_syn),
        tgt_names=list_to_str(t_lab),
        tgt_parents=list_to_str(t_par),
        tgt_children=list_to_str(t_chi),
        tgt_synonyms=list_to_str(t_syn)
    )
    return SYSTEM_PROMPT, user_prompt


def build_round_prompt(
    history: Dict[int, List[int]]
) -> Tuple[str, str]:
    """
    Build system and user prompt for intermediate rounds given agent answer history.
    history maps agent_id to list of 0/1 answers.
    """
    lines = []
    for aid, answers in history.items():
        lines.append(f"Agent {aid+1}: {' '.join(str(a) for a in answers)}")
    agent_str = '\n'.join(lines)
    user_prompt = ROUND_PROMPT.format(agent_answers=agent_str)
    return SYSTEM_PROMPT, user_prompt


def build_last_round_prompt(
    history: Dict[int, List[int]]
) -> Tuple[str, str]:
    """
    Build final round prompt.
    """
    lines = []
    for aid, answers in history.items():
        lines.append(f"Agent {aid+1}: {' '.join(str(a) for a in answers)}")
    agent_str = '\n'.join(lines)
    user_prompt = LAST_ROUND_PROMPT.format(agent_answers=agent_str)
    return SYSTEM_PROMPT, user_prompt
