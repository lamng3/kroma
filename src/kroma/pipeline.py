"""Ontology matching run.

MatchingRun is the template: prepare the task, score each pair, then write
metrics. The retrieval and refinement steps stay the same as the original script.
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

from kroma.agents.prompter import inference as agent_inference
from kroma.algorithms.node2vec import compute_combined_embeddings
from kroma.algorithms.refinement import offline_refine, online_refine
from kroma.algorithms.utils import compute_node_ranks, merge_graphs
from kroma.config.constants import (
    EVAL_SIZE_FRACTIONS,
    N_SHOT_DEMO,
    NODE2VEC_DEFAULTS,
    OFFLINE_REFINE_EVERY,
    RANDOM_SEED,
    RETRIEVAL_TOP_K,
)
from kroma.decorators import timed
from kroma.inference.factory import create_embedding_model
from kroma.metrics.scoring import calculate_metrics
from kroma.observers import MetricsListener
from kroma.repository import PredictionRepository, ReviewRepository
from kroma.retrieval.vector_store import VectorStore
from kroma.tasks.builder import build_ontology_matching_task
from kroma.tasks.loader import CsvAlignmentLoader
from kroma.tasks.prompts.builders import list_to_str
from kroma.utils.file import load_cache, load_config


class MatchingRun:
    """One KROMA evaluation over a configured dataset and model."""

    def __init__(self, args, listener=None):
        self.args = args
        self.listener = listener or MetricsListener()

    def execute(self) -> dict:
        self.prepare()
        self.score_pairs()
        return self.finish()

    def prepare(self) -> None:
        random.seed(RANDOM_SEED)
        self._configure_logging()
        self._load_configs()
        self._open_outputs()
        self._build_task()
        self._prepare_graph()
        self._build_stores()
        self.y_true = []
        self.y_pred = []
        self.output = {
            "method": self.method_name,
            "datasets": [str(path) for path in self.csv_paths],
            "agent_model": self.agent_name,
            "task": self.dataset_cfg["task_name"],
            "query_options": self.query_opts,
            "api_metrics": dict(self.listener.api),
            "metrics": dict(precision=-1, recall=-1, f1=-1),
        }
        self.started = time.time()

    def score_pairs(self) -> None:
        limit = int(len(self.task_aligns) * EVAL_SIZE_FRACTIONS[self.args.size])
        if limit < len(self.task_aligns):
            self.task_aligns = random.sample(self.task_aligns, limit)

        for index, (src_key, tgt_key, label) in enumerate(self.task_aligns, 1):
            precision, recall, f1 = calculate_metrics(self.y_true, self.y_pred)
            self.output["metrics"] = dict(precision=precision, recall=recall, f1=f1)
            print(f"\nObservation {index}/{len(self.task_aligns)} — F1 so far: {f1:.3f}")
            if index % OFFLINE_REFINE_EVERY == 0:
                self._snapshot_refine(index)

            src_code, tgt_code = src_key[0], tgt_key[0]
            if self.predictions.seen((src_code, tgt_code)):
                print("  • already done")
                continue

            self._score_pair(src_key, tgt_key, label, f1)

    def finish(self) -> dict:
        timed(self.listener, "offline_refinement")(offline_refine)(
            adj=self.compressed_graph,
            rank_attr=self.rank_attr_map,
        )
        total = time.time() - self.started
        precision, recall, f1 = calculate_metrics(self.y_true, self.y_pred)
        self.output.update({
            "running_time": total,
            "api_metrics": dict(self.listener.api),
            "metrics": {"precision": precision, "recall": recall, "f1": f1},
            "timings": {
                "offline_refinement": self.listener.timings["offline_refinement"],
                "online_refinement": self.listener.timings["online_refinement"],
                "total": round(total, 3),
            },
        })
        print("\nFinal metrics:", self.output)
        metrics_path = Path(self.predictions.name).with_suffix(".metrics.json")
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(self.output, handle, indent=2)
        print(f"Saved metrics to {metrics_path}")
        return self.output

    def _configure_logging(self) -> None:
        print("Starting KROMA evaluation...")
        log_path = Path("logs/kroma.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_path),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(console)
        self.logger = logging.getLogger("kroma")
        self.logger.info(f"Logging initialized — writing to {log_path}")

    def _load_configs(self) -> None:
        method_path = Path("experiments/configs/method") / self.args.llm / f"{self.args.method_config}.jsonl"
        self.logger.info(f"Method config path: {method_path}")
        method_cfg = load_config(method_path)
        self.agent_type = method_cfg["agent_type"]
        self.agent_name = method_cfg["agent_name"]
        self.method_name = method_cfg["method_name"]
        self.task_key = method_cfg["task"]
        self.query_opts = method_cfg["query_options"]
        self.agent_configs = method_cfg.get("agent_configs", [])
        self.n_rounds = method_cfg.get("n_rounds", 1)
        self.dropout = method_cfg.get("dropout", 0.0)

        dataset_path = Path("experiments/configs/datasets") / f"{self.task_key}.json"
        self.logger.info(f"Dataset config path: {dataset_path}")
        self.dataset_cfg = load_config(dataset_path)
        self.csv_paths = [
            Path(self.dataset_cfg["csv_folder"]) / f"{name}.csv"
            for name in self.dataset_cfg["datasets"]
        ]
        dictionary_paths = load_config("experiments/configs/dictionary.json")
        self.dictionary = {key: load_cache(path) for key, path in dictionary_paths.items()}

    def _open_outputs(self) -> None:
        self.predictions = PredictionRepository(
            self.task_key,
            self.method_name,
            self.agent_name,
            debate=False,
            size=self.args.size,
            reasoning=self.args.reasoning,
            baseline=self.args.baseline,
            active_learning=self.args.active_learning,
            compare_models=self.args.compare_models,
            bisim=self.args.bisim,
        )
        self.reviews = ReviewRepository(self.dataset_cfg["subtask"])

    def _build_task(self) -> None:
        loader = CsvAlignmentLoader(self.csv_paths, dataset=self.dataset_cfg["dataset_type"])
        source, target, raw_alignments = loader.load()
        (
            self.source,
            self.target,
            self.task_aligns,
            self.source_meta,
            self.target_meta,
            self.source_map,
            self.target_map,
        ) = build_ontology_matching_task(
            source,
            target,
            raw_alignments,
            sample_sz=self.dataset_cfg["sample_sz"],
            dictionary=self.dictionary,
            query_opts=self.query_opts,
        )

    def _prepare_graph(self) -> None:
        self.compressed_graph = merge_graphs(self.source, self.target)
        for src, tgt, _ in self.task_aligns:
            self.compressed_graph.setdefault(src, set())
            self.compressed_graph.setdefault(tgt, set())
        nodes = set(self.compressed_graph) | {
            child for children in self.compressed_graph.values() for child in children
        }
        self.rank_attr_map = compute_node_ranks(self.compressed_graph)
        self.equiv_classes = {node: node for node in nodes}
        offline_refine(adj=self.compressed_graph, rank_attr=self.rank_attr_map)

    def _build_stores(self) -> None:
        self.embedder = create_embedding_model("huggingface", "allenai/scibert_scivocab_uncased")
        eval_src = {src[0] for src, _, _ in self.task_aligns}
        eval_tgt = {tgt[0] for _, tgt, _ in self.task_aligns}
        source_ids = list(self.source.keys())
        target_ids = list(self.target.keys())
        source_vectors = self._combined(source_ids, self.source_meta)
        target_vectors = self._combined(target_ids, self.target_meta)

        pool_src = [key for key in source_ids if key not in eval_src]
        pool_tgt = [key for key in target_ids if key not in eval_tgt]
        self.src_store = VectorStore()
        self.tgt_store = VectorStore()
        self.src_store.add(
            ext_ids=pool_src,
            embeddings=np.vstack([source_vectors[source_ids.index(key)] for key in pool_src]),
        )
        self.tgt_store.add(
            ext_ids=pool_tgt,
            embeddings=np.vstack([target_vectors[target_ids.index(key)] for key in pool_tgt]),
        )

    def _combined(self, ids, meta):
        labels = [f"{meta[key]['labels'][0]}" for key in ids]
        text = self.embedder.encode(labels)
        return compute_combined_embeddings(
            adj_dict=self.compressed_graph,
            node_keys=ids,
            text_embs=text,
            n2v_kwargs=dict(NODE2VEC_DEFAULTS),
            combine_method="concat",
            alpha=0.5,
        )

    def _snapshot_refine(self, index: int) -> None:
        snapshot = {node: set(children) for node, children in self.compressed_graph.items()}
        snapshot_ranks = compute_node_ranks(snapshot)
        started = time.time()
        refined = timed(self.listener, "offline_refinement")(offline_refine)(snapshot, snapshot_ranks)
        self.rank_attr_map = compute_node_ranks(refined)
        print(f"[Benchmark] offline_refine at idx={index} took {time.time() - started:.3f}s")

    def _score_pair(self, src_key, tgt_key, label, f1) -> None:
        src_code, tgt_code = src_key[0], tgt_key[0]
        src_meta = self.source_meta[src_code]
        tgt_meta = self.target_meta[tgt_code]
        src_label = list_to_str(src_meta["labels"])
        tgt_label = list_to_str(tgt_meta["labels"])
        src_emb = self.embedder.encode([src_label])[0]
        tgt_emb = self.embedder.encode([tgt_label])[0]
        similarity = float(
            np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb))
        )
        self.logger.info(f"Pair cosine(sim)={similarity:.3f} for {src_label} ↔ {tgt_label}")

        demonstrations = random.sample(self.predictions.rows, min(N_SHOT_DEMO, len(self.predictions.rows)))
        demo_block = "Here are some examples of source↔target → relation:\n"
        for record in demonstrations:
            if record["pred_relation"] == record["true_relation"]:
                related = "are related." if record["true_relation"] == 0 else "do not appear to be related."
                demo_block += (
                    f"The concept “{record['source_label']}” and “{record['target_label']}” "
                    f"{related}\n\n"
                )
        meta_block = (
            f"The source concept “{src_label}” has parents {', '.join(src_meta['parents'])}, "
            f"children {', '.join(src_meta['children'])}, synonyms {', '.join(src_meta['synonyms'])}, "
            f"and labels {', '.join(src_meta['labels'])}.\n"
            f"The target concept “{tgt_label}” has parents {', '.join(tgt_meta['parents'])}, "
            f"children {', '.join(tgt_meta['children'])}, synonyms {', '.join(tgt_meta['synonyms'])}, "
            f"and labels {', '.join(tgt_meta['labels'])}.\n\n"
        )
        src_hits = self.src_store.query(src_emb, top_k=RETRIEVAL_TOP_K)
        tgt_hits = self.tgt_store.query(tgt_emb, top_k=RETRIEVAL_TOP_K)
        src_ctx = [f"{list_to_str(self.source_meta[name]['labels'])}" for (name, _), _ in src_hits if name in self.source_meta]
        tgt_ctx = [f"{list_to_str(self.target_meta[name]['labels'])}" for (name, _), _ in tgt_hits if name in self.target_meta]
        rag_block = (
            f"For the source concept “{src_label}”, the most similar context labels are: {', '.join(src_ctx)}. "
            f"For the target concept “{tgt_label}”, the most similar context labels are: {', '.join(tgt_ctx)}.\n\n"
        )

        prediction, llm_metrics, confidence, accept = agent_inference(
            debate=self.args.debate,
            agent_configs=self.agent_configs,
            backend=self.agent_type,
            model_name=self.agent_name,
            source_term=(src_code, self.source[src_code]),
            target_term=(tgt_code, self.target[tgt_code]),
            pcmaps={"source": self.source_map, "target": self.target_map},
            dictionary=self.dictionary,
            options=self.query_opts,
            embed_model=self.embedder,
            store=(self.src_store, self.tgt_store),
            n_rounds=self.n_rounds,
            dropout=self.dropout,
            reasoning=self.args.reasoning,
            active_learning=self.args.active_learning,
            f1_score=f1,
            bisim=self.args.bisim,
            context=demo_block + meta_block + rag_block,
        )
        self.listener.add_api(llm_metrics)
        row = dict(
            source_label=src_label,
            target_label=tgt_label,
            source=src_code,
            source_uri=src_key[1],
            target=tgt_code,
            target_uri=tgt_key[1],
            similarity=similarity,
            confidence=confidence,
            true_relation=int(label),
            pred_relation=int(prediction),
            api_metrics=llm_metrics,
            accepted=accept,
        )
        self.predictions.append(row)
        self.y_true.append(int(label))
        self.y_pred.append(int(prediction))
        if not accept:
            return

        self.equiv_classes, expert_questions = timed(self.listener, "online_refinement")(online_refine)(
            self.compressed_graph,
            self.rank_attr_map,
            self.equiv_classes,
            [(src_key, tgt_key)],
            prediction,
        )
        for question in expert_questions or []:
            print("  → expert review:", question)
        self.reviews.append(src_key, tgt_key, expert_questions)
