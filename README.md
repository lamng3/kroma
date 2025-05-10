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
Currently, our method features 6 datasets (Mouse-Human, NCIT-DOID, Nell-DBpedia, YAGO-Wikidata, ENVO-SWEET, MI-MatOnto). Please provide the downloaded datasets in `experiments/dataset/`.

Our paper features models provided by `TogetherAI`'s API. So please supply your TogetherAI access token in the `.env` file. 

## Experiment Reproduction
We supply a sample script to run an experiment on `ENVO-SWEET` track with `Llama-3.3-70B` in the `scripts/run_envo_sweet.sh`. We provided a dynamic results saving to the script so that the results files will be automatically updated with the newest predicted pairs.

## Result Digestion
The final results can be find under 2 folders. For example, after running the `ENVO-SWEET` experiment, you can find the results under `results/baseline/envo_sweet` for accepted alignments and `reviews/baseline/envo_sweet` for reviews needed by experts

## Codebase Design and Contribution
Should you want to add a new evaluation, you may consider adding an `experiments/configs/method/<llm>` folder for the LLMs you want to evaluated on, and supply corresponding `.jsonl` for the dataset and the configurations you want to evaluate on. For a new dataset, add a folder `experiments/configs/dataset/<dataset>.json` to point towards the dataset you wish to add.

The OAEI datasets will be stored at `experiments/datasets/OAEI/<track>/<dataset>`. If you want to test multiple times, we advised to cache the query results and provide a path to it at `experiments/configs/dictionary.json`.

---

Should you need to refer to this work or find our codebase useful, please consider citing our work as:
```
@inproceedings{2025_knowledge_retrieval_ontology_matching_llm,
    title={KROMA: Knowledge Retrieval Ontology Matching using Large Language Model},
    author={Anonymous},
    year={2025},
}
```