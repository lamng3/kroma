# KROMA

KROMA matches concepts across two ontologies. For each candidate pair it
retrieves the concepts' local context, asks a large language model whether the
two refer to the same thing, and refines the alignment as answers arrive. It was
published at ISWC 2025 and is evaluated on six OAEI-style tracks: Mouse-Human,
NCIT-DOID, Nell-DBpedia, YAGO-Wikidata, ENVO-SWEET, and MI-MatOnto.

[Paper](https://doi.org/10.1007/978-3-032-09527-5_34) ·
[arXiv](https://arxiv.org/abs/2507.14032) ·
[Full PDF](full.pdf) ·
[Documentation](https://lamng3.github.io/kroma-docs/)

## How it works

One run scores every alignment pair of a track in four steps.

1. **Load.** The track's alignment CSVs become a source graph, a target graph,
   and a list of labeled source-target pairs.
2. **Retrieve.** Each concept is embedded with SciBERT text vectors fused with
   Node2Vec graph vectors. Its context is gathered from the configured sources:
   the ontologies' dictionary caches, the ontology graph itself, ConceptNet,
   and SPARQL endpoints for NCIT, DOID, and DBpedia.
3. **Decide.** A language model answers yes or no with a confidence, either
   alone or as a multi-agent debate.
4. **Refine.** Accepted matches update the equivalence classes as they arrive
   (online refinement), and the whole graph is re-collapsed every 5 pairs
   (offline refinement). Pairs the refinement cannot settle are written out for
   an expert to review.

## Quick start

KROMA requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lamng3/kroma.git
cd kroma
./setup.sh
source .venv/bin/activate
kroma --help
```

`kroma --help` needs no model. A matching run needs the model backends:

```bash
uv pip install ".[models]"
```

## Configure

**API keys.** Copy `.env.example` to `.env`. Together models read
`TOGETHERAI_API_KEY`, and OpenAI models read `OPENAI_API_KEY`.

**Datasets.** The benchmarks are downloaded separately. Place the alignment
CSVs and dictionary caches under `experiments/dataset/`. Each track's
dataset config names its CSV folder, and `experiments/configs/dictionary.json`
names each dictionary cache. These paths are gitignored.

| Track | Config | CSV folder |
| --- | --- | --- |
| Mouse-Human | `mouse_human` | `experiments/dataset/OAEI/Anatomy` |
| NCIT-DOID | `ncit_doid` | `experiments/dataset/OAEI/BioLLM` |
| Nell-DBpedia | `nell_dbpedia` | `experiments/dataset/OAEI/CommonKG` |
| YAGO-Wikidata | `yago_wikidata` | `experiments/dataset/OAEI/CommonKG` |
| ENVO-SWEET | `envo_sweet` | `experiments/dataset/OAEI/Biodiv` |
| MI-MatOnto | `mi_matonto` | `experiments/dataset/OAEI/MSE` |

**Methods.** A method config is a JSONL file at
`experiments/configs/method/<llm>/<method_config>.jsonl`. It names the model
backend (`openai` or `togetherai`), the embedding model, the track, and the
context sources to query. Configs are included for Llama 3.3 70B, Llama 3.1 70B,
Llama 3.2 3B, Llama 2 70B, DeepSeek-R1-Distill (Llama 70B and Qwen 1.5B),
GPT-4o-mini, Gemma 2B, and Mistral 7B, plus debate configs such as
`kroma_scibert_mouse_human_debate_2A_3R` (2 agents, 3 rounds).

## Run an experiment

Reproduce the ENVO-SWEET baseline with Llama 3.3 70B
(`scripts/run_envo_sweet.sh` runs the same command):

```bash
kroma run \
  --method_config kroma_scibert_envo_sweet \
  --llm meta-llama/Llama-3.3-70B-Instruct-Turbo-Free \
  --baseline
```

| Flag | Effect |
| --- | --- |
| `--method_config` | Method config name, without `.jsonl` (required) |
| `--llm` | Model folder under `experiments/configs/method/` (required) |
| `--size` | Fraction of pairs to score: `xsmall` (20%), `small`, `medium`, `large`, or `full` (default) |
| `--reasoning` | Ask the model to reason before it answers |
| `--debate` | Decide each pair by a multi-agent debate |
| `--active_learning` | Accept an answer only when a confidence-based acceptance score is positive |
| `--baseline` | Write results under `results/baseline/` instead of `results/results/` |

The same flags work with `python -m main`. A run skips pairs it has already
scored, so an interrupted run resumes where it stopped.

## Outputs

| Path | Contents |
| --- | --- |
| `results/baseline/<track>/<model>_<track>.jsonl` | One accepted prediction per line |
| `results/baseline/<track>/<model>_<track>.metrics.json` | Precision, recall, F1, token counts and API calls, and refinement timings |
| `reviews/baseline/<track>/expert_queries.csv` | Pairs sent for expert review |
| `logs/kroma.log` | Run log |

## Extend KROMA

| To add | Use |
| --- | --- |
| A chat or embedding model provider | `register_chat_backend`, `register_embedding_backend` |
| A context source, such as a dictionary or SPARQL endpoint | `register_query_strategy` |
| An alignment file format | subclass `OntologyLoader` |
| A judging procedure | subclass `InferenceStrategy` |

To run from Python, `kroma.pipeline.MatchingRun(args).execute()` takes the
same arguments as the CLI and returns the metrics dictionary.

## Repository layout

```text
src/kroma/         Installable package
  cli.py           kroma run
  pipeline.py      MatchingRun: prepare, score pairs, write metrics
  tasks/           Loaders, prompt builders, context sources
  agents/          Single-model and debate inference
  inference/       Chat and embedding backend factory
  algorithms/      Node2Vec fusion, online and offline refinement
  repository.py    Prediction JSONL and expert-review CSV
experiments/       Dataset and method configs (data is downloaded separately)
scripts/           Reproduction scripts
docs/              Documentation pages
full.pdf           Full version of the paper
```

## Development

```bash
uv pip install ".[dev]"
pytest -q
```

## Citation

If you use KROMA, please cite:

```bibtex
@inproceedings{nguyen_2025_kroma,
    title={KROMA: Ontology Matching with Knowledge Retrieval and Large Language Models},
    author={Lam Nguyen and Erika Barcelos and Roger French and Yinghui Wu},
    booktitle={The Semantic Web -- ISWC 2025: 24th International Semantic Web Conference, Proceedings, Part I},
    series={Lecture Notes in Computer Science},
    publisher={Springer},
    pages={629--649},
    year={2025},
    doi={10.1007/978-3-032-09527-5_34},
}
```
