SYSTEM_PROMPT = """
You are OntoMapper, an agent that determines whether two ontology concepts are closely related based on their names and hierarchical relationships.
"""

TASK_PROMPT = """
{demonstrations}
Task: Determine if two ontology concepts refer to the same real-world entity.

Source Names: {src_names}
Source Parents: {src_parents}
Source Children: {src_children}
Source Synonyms: {src_synonyms}

Target Names: {tgt_names}
Target Parents: {tgt_parents}
Target Children: {tgt_children}
Target Synonyms: {tgt_synonyms}

Only output:<answer>Yes</answer> or <answer>No</answer> and <confidence>0-10</confidence>.
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
