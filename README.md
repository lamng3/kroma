# KROMA
Knowledge Retrieval Ontology Matching using Large Language Model

## Environment Setup
We provide the minimum environment requirements to support the running of our project. This means there can be a slight difference depending on the actual automatic dependency-solving result of different systems.

Should one be interested in reproducing a certain method, please look up the corresponding requirement file and install listed packages accordingly.
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset and Access Preparation
Currently, our method features 6 datasets (Mouse-Human, NCIT-DOID, Nell-DBpedia, YAGO-Wikidata, ENVO-SWEET, MI-MatOnto). We provide 

Our paper features models provided by `TogetherAI`'s API. So please supply your TogetherAI access token in the `.env` file. 

## Experiment Reproduction
We supply a sample script to run an experiment on `ENVO-SWEET` track with `Llama-3.3-70B` in the `scripts/run_envo_sweet.sh` .

