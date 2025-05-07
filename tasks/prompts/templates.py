SYSTEM_PROMPT = """
You are OntoMapper, an agent that determines whether two ontology concepts are closedly related based on their names and hierarchical relationships. Given the following inputs for each concept: names, parent concepts, and child concepts, analyze the provided information and output "Yes" if the concepts are closedly related, or "No" if they are different. Your decision should be based solely on the comparison of the concept names (which may include multiple labels) and their hierarchical context.
"""

TASK_PROMPT = """
{demonstrations}
Task definition: The task of Ontology Mapping can be defined as follows. Given the source and target ontologies, the objective is to generate a set of mappings between the concepts of source ontologies and target ontologies. This is essentially a binary classification task that determines if two concepts, given their names (multiple labels per concept possible) and/or additional structural contexts, refer to the same real-world entity.
Given the lists of names and hierarchical relationships associated with two concepts, your task is to determine whether these concepts refer to the same real-world entity or not. Consider the following:

Source Concept Names: {src_names}
Parent Concepts of the Source Concept: {src_parents}
Child Concepts of the Source Concept: {src_children}
Synonym Concepts of the Source Concept: {src_synonyms}

Target Concept Names: {tgt_names}
Parent Concepts of the Target Concept: {tgt_parents}
Child Concepts of the Target Concept: {tgt_children}
Synonym Concepts of the Target Concept: {tgt_synonyms}

Analyze the names and the hierarchical information provided for each concept and provide a conclusion on whether these two concepts refer to the same real-world entity or not (“Yes” or “No”) based on their associated names and hierarchical relationships. 
Only output <answer>Yes</answer> or <answer>No</answer>.

As you completed the above task, give a score of how confident you are with your answer? Only a single number within the range of 0 to 10 (i.e. <confidence>0-10</confidence>).
"""

TASK_REASONING_PROMPT = task_prompt_reasoning_template = """
<think>\n
Task definition: The task of Ontology Mapping can be defined as follows. Given the source and target ontologies, the objective is to generate a set of mappings between the concepts of source ontologies and target ontologies. This is essentially a binary classification task that determines if two concepts, given their names (multiple labels per concept possible) and/or additional structural contexts, refer to the same real-world entity.
Given the lists of names and hierarchical relationships associated with two concepts, your task is to determine whether these concepts refer to the same real-world entity or not. Consider the following:

Source Concept Names: {src_names}
Parent Concepts of the Source Concept: {src_parents}
Child Concepts of the Source Concept: {src_children}
Synonym Concepts of the Source Concept: {src_synonyms}

Target Concept Names: {tgt_names}
Parent Concepts of the Target Concept: {tgt_parents}
Child Concepts of the Target Concept: {tgt_children}
Synonym Concepts of the Target Concept: {tgt_synonyms}

Analyze the names and the hierarchical information provided for each concept and provide a conclusion on whether these two concepts refer to the same real-world entity or not (“Yes” or “No”) based on their associated names and hierarchical relationships. 
Only output <answer>Yes</answer> or <answer>No</answer>.

As you completed the above task, give a score of how confident you are with your answer? Only a single number within the range of 0 to 10 (i.e. <confidence>0-10</confidence>).
\n</think>
Rethink of your solution.
"""

ROUND_PROMPT = """
These are the solutions to the problem from other agents:
{agent_answers}
Using the solutions from other agents as additional information, can you give an updated response by analyzing the names and the hierarchical information provided for each concept and provide a conclusion on whether these two concepts are closedly related or different (“Yes” or “No”) based on their associated names and hierarchical relationships. Only output "Yes" or "No" enclosed within the tags <output_tag>.
"""

LAST_ROUND_PROMPT = """
This is the last round of debate. Come up with a conclusive answer. These are the solutions to the problem from other agents:
{agent_answers}
Using the solutions from other agents as additional information, can you give an updated response by analyzing the names and the hierarchical information provided for each concept and provide a conclusion on whether these two concepts are closedly related or different (“Yes” or “No”) based on their associated names and hierarchical relationships. Only output "Yes" or "No" enclosed within the tags <output_tag>.
"""
