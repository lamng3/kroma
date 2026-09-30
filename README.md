# KROMA

KROMA matches concepts across ontologies by retrieving local context and asking a large language model whether two concepts refer to the same thing. It evaluates six OAEI-style tracks: Mouse-Human, NCIT-DOID, Nell-DBpedia, YAGO-Wikidata, ENVO-SWEET, and MI-MatOnto.

[Paper](https://arxiv.org/abs/2507.14032) ·
[Full PDF](full.pdf) ·
[Documentation](docs/index.html)

The full paper PDF is in this repository at [`full.pdf`](full.pdf). The arXiv version is [https://arxiv.org/abs/2507.14032](https://arxiv.org/abs/2507.14032).

## Quick start

KROMA requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lamng3/kroma.git
cd kroma
./setup.sh
source .venv/bin/activate
kroma --help
```

Copy `.env.example` to `.env` and set `TOGETHERAI_API_KEY` or `OPENAI_API_KEY`. Place the downloaded alignment CSVs and dictionary caches under `experiments/dataset/`, using the paths in `experiments/configs/`.

Install the model backends before a real run:

```bash
uv pip install ".[models]"
```

Reproduce the ENVO-SWEET baseline with Llama 3.3 70B:

```bash
kroma run \
  --method_config kroma_scibert_envo_sweet \
  --llm meta-llama/Llama-3.3-70B-Instruct-Turbo-Free \
  --baseline
```

Accepted alignments are written under `results/baseline/envo_sweet`. Pairs that need an expert are written under `reviews/baseline/envo_sweet`.

Open the docs from the repository:

```bash
python -m http.server -d docs 8000
```

Then visit `http://127.0.0.1:8000`.

## Citation

```
@inproceedings{nguyen_2025_kroma,
    title={KROMA: Knowledge Retrieval Ontology Matching using Large Language Models},
    author={Lam Nguyen and Erika Barcelos and Roger French and Yinghui Wu},
    journal={Proceedings of the 24th International Semantic Web Conference Research Track},
    year={2025},
}
```
