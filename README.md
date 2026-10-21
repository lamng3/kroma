<div align="center">

# 🧭 KROMA

**Ontology Matching with Knowledge Retrieval and Large Language Models**

[![ISWC 2025](https://img.shields.io/badge/ISWC-2025-b31b1b)](https://doi.org/10.1007/978-3-032-09527-5_34)
[![arXiv](https://img.shields.io/badge/arXiv-2507.14032-b31b1b)](https://arxiv.org/abs/2507.14032)
[![Docs](https://img.shields.io/badge/docs-kroma--docs-8b5cf6)](https://lamng3.github.io/kroma-docs/)
[![Python](https://img.shields.io/badge/python-3.12-3b82f6)](pyproject.toml)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

[⚡ Quick start](#-quick-start) · [✨ What it is](#-what-it-is) · [📊 Results](#-results) · [🔧 Usage](#-usage) · [📝 Citation](#-citation)

</div>

<p align="center">
  <b>Lam Nguyen</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Erika Barcelos</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Roger French</b><sup>1</sup> &nbsp;&nbsp;&nbsp; <b>Yinghui Wu</b><sup>1</sup>
  <br>
  <sub><sup>1</sup>&nbsp;Case Western Reserve University</sub>
</p>

> [!NOTE]
> **Published at ISWC 2025** (Research Track, LNCS). The [full version](full.pdf),
> with every table, is in this repository.

## ⚡ Quick start

```bash
git clone https://github.com/lamng3/kroma.git
cd kroma
./setup.sh
source .venv/bin/activate
kroma --help
```

`kroma --help` needs no model. A matching run needs the model backends and an
API key:

```bash
uv pip install ".[models]"
cp .env.example .env          # TOGETHERAI_API_KEY or OPENAI_API_KEY

kroma run \
  --method_config kroma_scibert_envo_sweet \
  --llm meta-llama/Llama-3.3-70B-Instruct-Turbo-Free \
  --baseline
```

KROMA requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The benchmark
data is downloaded separately; see Datasets under [Usage](#-usage).

## ✨ What it is

KROMA matches concepts across two ontologies. For each candidate pair it
retrieves the concepts' local context, asks a large language model whether the
two refer to the same thing, and refines the alignment as answers arrive.

1. **Load.** A track's alignment CSVs become a source graph, a target graph, and labeled source-target pairs.
2. **Retrieve.** Each concept is embedded with SciBERT text vectors fused with Node2Vec graph vectors. Its context comes from the ontologies' dictionaries, the ontology graph, ConceptNet, and SPARQL endpoints for NCIT, DOID, and DBpedia.
3. **Decide.** A language model answers yes or no with a confidence, alone or as a multi-agent debate.
4. **Refine.** Accepted matches update the equivalence classes as they arrive (online refinement), and the whole graph is re-collapsed every 5 pairs (offline refinement). Pairs refinement cannot settle go to an expert for review.

## 📊 Results

<div align="center">

| Track | KROMA | Best baseline |
|:--|:--:|:--:|
| Mouse–Human | **94.94** | 92.20 <sub>MILA</sub> |
| NCIT–DOID | **98.63** | 94.80 <sub>MILA</sub> |
| Nell–DBpedia | **97.08** | 96.00 <sub>OLaLa</sub> |
| YAGO–Wikidata | **95.54** | 92.19 <sub>LLM4OM</sub> |
| ENVO–SWEET | **93.18** <sub>GPT-4o-mini</sub> | 83.70 <sub>MILA</sub> |
| MI–MatOnto | **61.25** | 32.97 <sub>MILA</sub> |

<sub>F1 (%). KROMA with Llama-3.3-70B unless noted. Baselines are MILA, OLaLa, and LLM4OM with their reported models; each row shows the best of them · Table 3 of [full.pdf](full.pdf)</sub>

</div>

KROMA has the highest F1 on all six tracks. Removing knowledge retrieval or
ontology refinement lowers F1 on every track reported, and with the small
Qwen2.5-1.5B it still beats the best baseline on NCIT–DOID, YAGO–Wikidata, and
MI–MatOnto.

<details>
<summary><b>Ablations and other models</b></summary>

<div align="center">

| Track | KROMA | without knowledge retrieval | without refinement |
|:--|:--:|:--:|:--:|
| Mouse–Human | **94.94** | 91.25 | 86.59 |
| NCIT–DOID | **98.63** | 95.01 | 90.91 |
| Nell–DBpedia | **97.08** | 94.26 | 93.58 |
| YAGO–Wikidata | **95.54** | 91.98 | 89.74 |
| MI–MatOnto | **61.25** | 59.95 | 55.00 |

<sub>F1 (%), Llama-3.3-70B · Table 3 of [full.pdf](full.pdf)</sub>

| Track | Qwen2.5-1.5B | GPT-4o-mini | Llama-3.3-70B |
|:--|:--:|:--:|:--:|
| Mouse–Human | 83.6 | 92.0 | **94.9** |
| NCIT–DOID | 97.4 | 97.6 | **98.6** |
| Nell–DBpedia | 95.0 | 95.7 | **97.1** |
| YAGO–Wikidata | **95.5** | 95.4 | **95.5** |
| ENVO–SWEET | 79.8 | **93.2** | 92.0 |
| MI–MatOnto | 61.3 | 60.9 | **62.2** |

<sub>F1 (%) · Table 4 of [full.pdf](full.pdf)</sub>

</div>

</details>

<details>
<summary><b>Cost: tokens, calls, and time</b></summary>

<div align="center">

| Track | Tokens sent vs full pairwise: Qwen2.5 | GPT-4o-mini | Llama-3.3 | Speedup, offline vs online refinement |
|:--|:--:|:--:|:--:|:--:|
| Mouse–Human | 3.06% | 2.88% | 6.76% | 7.17× |
| NCIT–DOID | 54.48% | 54.63% | 54.48% | 2.33× |
| Nell–DBpedia | 26.09% | 25.76% | 26.09% | 5.63× |
| YAGO–Wikidata | 21.86% | 21.93% | 22.01% | 18.76× |
| ENVO–SWEET | 6.99% | 6.82% | 6.99% | 7.00× |
| MI–MatOnto | 43.04% | 41.66% | 41.81% | 13.31× |
| **Overall** | **25.92%** | **25.61%** | **26.36%** | **9.03×** |

<sub>Table 9 of [full.pdf](full.pdf)</sub>

| Model | Total tokens | Output tokens | Tokens per call |
|:--|--:|--:|--:|
| Llama-3.3-70B | 480K | 15K | 470 |
| DR1D-Llama-3-70B (deep reasoning) | 620K | 130K | 605 |

<sub>All calls of one run · Table 10 of [full.pdf](full.pdf)</sub>

| ENVO–SWEET, 1,000 candidate pairs | Seconds |
|:--|--:|
| LLM inference, single agent | 10,916 |
| Offline refinement | 68 |
| Online refinement | 14 |

<sub>§ KROMA Runtime of [full.pdf](full.pdf)</sub>

</div>

Retrieval sends the model about a quarter of what full pairwise matching would,
and under 7% on the smaller ontologies; inference, not refinement, is where the
time goes. New runs write a `cost` block next to their metrics (`kroma.costs`:
API calls, tokens per call, and dollars at list price for hosted models).

</details>

## 🔧 Usage

<details>
<summary><b>Datasets</b></summary>

Place the alignment CSVs and dictionary caches under `experiments/dataset/`.
Each track's dataset config names its CSV folder, and
`experiments/configs/dictionary.json` names each dictionary cache. These paths
are gitignored.

| Track | Config | CSV folder |
| --- | --- | --- |
| Mouse-Human | `mouse_human` | `experiments/dataset/OAEI/Anatomy` |
| NCIT-DOID | `ncit_doid` | `experiments/dataset/OAEI/BioLLM` |
| Nell-DBpedia | `nell_dbpedia` | `experiments/dataset/OAEI/CommonKG` |
| YAGO-Wikidata | `yago_wikidata` | `experiments/dataset/OAEI/CommonKG` |
| ENVO-SWEET | `envo_sweet` | `experiments/dataset/OAEI/Biodiv` |
| MI-MatOnto | `mi_matonto` | `experiments/dataset/OAEI/MSE` |

</details>

<details>
<summary><b>Methods and models</b></summary>

A method config is a JSONL file at
`experiments/configs/method/<llm>/<method_config>.jsonl`. It names the model
backend (`openai` or `togetherai`), the embedding model, the track, and the
context sources to query. Configs are included for Llama 3.3 70B, Llama 3.1 70B,
Llama 3.2 3B, Llama 2 70B, DeepSeek-R1-Distill (Llama 70B and Qwen 1.5B),
GPT-4o-mini, Gemma 2B, and Mistral 7B, plus debate configs such as
`kroma_scibert_mouse_human_debate_2A_3R` (2 agents, 3 rounds).

Together models read `TOGETHERAI_API_KEY`, and OpenAI models read
`OPENAI_API_KEY`, from `.env`.

</details>

<details>
<summary><b>Running an experiment</b></summary>

`scripts/run_envo_sweet.sh` runs the quick-start command.

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
scored, so an interrupted run resumes where it stopped. From Python,
`kroma.pipeline.MatchingRun(args).execute()` takes the same arguments and
returns the metrics dictionary.

| Output | Contents |
| --- | --- |
| `results/baseline/<track>/<model>_<track>.jsonl` | One accepted prediction per line |
| `results/baseline/<track>/<model>_<track>.metrics.json` | Precision, recall, F1, token counts and API calls, and refinement timings |
| `reviews/baseline/<track>/expert_queries.csv` | Pairs sent for expert review |
| `logs/kroma.log` | Run log |

</details>

<details>
<summary><b>Extending KROMA</b></summary>

| To add | Use |
| --- | --- |
| A chat or embedding model provider | `register_chat_backend`, `register_embedding_backend` |
| A context source, such as a dictionary or SPARQL endpoint | `register_query_strategy` |
| An alignment file format | subclass `OntologyLoader` |
| A judging procedure | subclass `InferenceStrategy` |

</details>

<details>
<summary><b>KROMA as an agent on a shared memory</b></summary>

`kroma.integrations.memcommit.KromaMatcher` runs KROMA's online refinement as an
agent on a [MemCommit](https://github.com/lamng3/memcommit) blackboard, so it can share
one memory with other agents. It consumes
`match_request` memories and writes `match` and `review` memories with their
provenance; a restarted matcher rebuilds its equivalence classes from memory.
The judge is any function returning KROMA's (prediction, confidence, accept),
such as the language-model debate. An agent that looks up a concept's matches
derives from `alignment_read(concept)`, so a later match makes its result stale.

</details>

<details>
<summary><b>Repository layout and development</b></summary>

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

```bash
uv pip install ".[dev]"
pytest -q
```

</details>

## 📝 Citation

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

## License

[CC BY 4.0](LICENSE). Copyright (c) 2025 Lam Nguyen, Erika Barcelos, Roger French, and Yinghui Wu.
