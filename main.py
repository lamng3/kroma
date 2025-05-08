#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import logging
import numpy as np

from config.constants import METADATA
from utils.env import get_env
from utils.file import load_config, load_cache, get_predicted_pairs_and_file

from agents.factory import create_agents
from agents.prompter import inference as agent_inference, initialize_graph

from inference.factory import create_embedding_model

from tasks.loader import CsvAlignmentLoader
from tasks.builder import build_ontology_matching_task
from tasks.prompts.builders import list_to_str

from retrieval.vector_store import VectorStore

from algorithms.bisimulation import bisimulation, compute_node_ranks
from algorithms.incremental_refinement import incremental_refinement, merge_graphs

from metrics.scoring import calculate_metrics

# ----------------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------------
random.seed(42)
print("Starting KROMA evaluation...")

# logging: file + console
log_path = Path("logs/kroma.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(log_path), filemode="w",
    level=logging.INFO, format="[%(levelname)s] %(message)s",
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console)
logger = logging.getLogger()
logger.info(f"Logging initialized — writing to {log_path}")

# ----------------------------------------------------------------------------
# ARGUMENTS
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate KROMA ontology matching")
parser.add_argument("--method_config", required=True)
parser.add_argument("--llm", required=True)
parser.add_argument("--size", choices=['xsmall','small','medium','large','full'], default="full")
parser.add_argument("--reasoning", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--active_learning", action="store_true")
parser.add_argument("--compare_models", action="store_true")
parser.add_argument("--bisim", action="store_true")
parser.add_argument("--debate", action="store_true", default="false")
args = parser.parse_args()

# ----------------------------------------------------------------------------
# LOAD CONFIGS
# ----------------------------------------------------------------------------
method_cfg_path = Path("experiments/configs/method") / args.llm / f"{args.method_config}.jsonl"
logger.info(f"Method config path: {method_cfg_path}")
method_cfg = load_config(method_cfg_path)
agent_type, agent_name = method_cfg['agent_type'], method_cfg['agent_name']
method_name = method_cfg['method_name']
task_key = method_cfg['task']
query_opts = method_cfg['query_options']

agent_configs = method_cfg['agent_configs'] if 'agent_configs' in method_cfg else []
n_agents = method_cfg['n_agents'] if 'n_agents' in method_cfg else 1
n_rounds = method_cfg['n_rounds'] if 'n_rounds' in method_cfg else 1
dropout = method_cfg['dropout'] if 'dropout' in method_cfg else 0.0

ds_cfg_path = Path("experiments/configs/datasets") / f"{task_key}.json"
logger.info(f"Dataset config path: {ds_cfg_path}")
ds_cfg = load_config(ds_cfg_path)
csv_paths = [Path(ds_cfg['csv_folder']) / f"{n}.csv" for n in ds_cfg['datasets']]

dict_paths = load_config("experiments/configs/dictionary.json")
dictionary = {k: load_cache(dict_paths[k]) for k in dict_paths}

n_shot_demo = 5

# ----------------------------------------------------------------------------
# PREDICTION FILES
# ----------------------------------------------------------------------------
predicted_pairs, prediction_file = get_predicted_pairs_and_file(
    task_key, method_name, agent_name,
    debate=False, size=args.size,
    reasoning=args.reasoning, baseline=args.baseline,
    active_learning=args.active_learning, compare_models=args.compare_models,
    bisim=args.bisim,
)

# expert review files
review_dir = Path("reviews") / "baseline" / ds_cfg['subtask']
review_dir.mkdir(parents=True, exist_ok=True)
review_file = review_dir / "expert_queries.csv"

# write a header if the file is new
if not review_file.exists():
    review_file.write_text("source,target,expert_edge\n")

# ----------------------------------------------------------------------------
# BUILD ONTOLOGY TASK + EARLY METADATA
# ----------------------------------------------------------------------------
loader = CsvAlignmentLoader(csv_paths, dataset=ds_cfg['dataset_type'])
OS, OT, raw_aligns = loader.load()
OS, OT, task_aligns, OS_meta, OT_meta, OS_map, OT_map = build_ontology_matching_task(
    OS, OT, raw_aligns,
    sample_sz=ds_cfg['sample_sz'],
    dictionary=dictionary,
    query_opts=query_opts,
)
initialize_graph(OS, OT)

# ----------------------------------------------------------------------------
# COMPUTE NODE RANKS
# ----------------------------------------------------------------------------
compressed_graph = merge_graphs(OS, OT)
rank_attr = compute_node_ranks(compressed_graph)

# ----------------------------------------------------------------------------
# SETUP EMBEDDER + VECTOR STORES
# ----------------------------------------------------------------------------
embedder = create_embedding_model("huggingface", "allenai/scibert_scivocab_uncased")
src_store = VectorStore()
tgt_store = VectorStore()

src_ids = list(OS.keys())
src_texts = [f"{lbl}" for lbl, uri in src_ids]
src_embs = embedder.encode(src_texts)
src_store.add(ext_ids=src_ids, embeddings=src_embs)

tgt_ids = list(OT.keys())
tgt_texts = [f"{lbl}" for lbl, uri in tgt_ids]
tgt_embs = embedder.encode(tgt_texts)
tgt_store.add(ext_ids=tgt_ids, embeddings=tgt_embs)

# ----------------------------------------------------------------------------
# EVALUATION LOOP
# ----------------------------------------------------------------------------
y_true, y_pred = [], []
output_metrics = {
    'method': method_name, 'datasets': [str(p) for p in csv_paths],
    'agent_model': agent_name, 'task': ds_cfg['task_name'],
    'query_options': query_opts,
    'api_metrics': dict(input_token=0, output_token=0, token_count=0, api_call_cnt=0),
    'metrics': dict(precision=-1, recall=-1, f1=-1)
}
start_time = time.time()

size_map = {'xsmall':0.2, 'small':0.4, 'medium':0.6, 'large':0.8, 'full':1.0}
limit = int(len(task_aligns) * size_map[args.size])
if limit < len(task_aligns): task_aligns = random.sample(task_aligns, limit)

for idx, (src_key, tgt_key, label) in enumerate(task_aligns, 1):
    prec, rec, f1 = calculate_metrics(y_true, y_pred)
    output_metrics['metrics'] = dict(precision=prec, recall=rec, f1=f1)
    print(f"\nObservation {idx}/{len(task_aligns)} — F1 so far: {f1:.3f}")

    src_keycode = src_key[0]
    tgt_keycode = tgt_key[0]
    src_node = OS[src_keycode]
    tgt_node = OT[tgt_keycode]
    src_meta = OS_meta[src_keycode]
    tgt_meta = OT_meta[tgt_keycode]
    src_label, tgt_label = list_to_str(src_meta['labels']), list_to_str(tgt_meta['labels'])

    if (src_keycode, tgt_keycode) in predicted_pairs:
        print("  • already done")
        continue

    src_emb = embedder.encode([src_label])[0]
    tgt_emb = embedder.encode([tgt_label])[0]
    sim_score = float(np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb)))
    logger.info(f"Pair cosine(sim)={sim_score:.3f} for {src_label} ↔ {tgt_label}")

    demonstrations = [r for r in predicted_pairs]
    demonstrations = random.sample(demonstrations, min(n_shot_demo, len(demonstrations)))
    demo_block = "Here are some examples of source↔target → relation:\n"
    for rec in demonstrations:
        if rec['pred_relation'] == rec['true_relation']:
            demo_block += (
                f"- Source: {rec['source_label']}  \n"
                f"  Target: {rec['target_label']}  \n"
                f"  Relation: {"Related" if rec['true_relation'] == 0 else "Not related"}\n\n"
            )

    if args.bisim and (src_keycode, tgt_keycode) in bisimulation(compressed_graph):
        pred, conf, accept = 1, 10, True
        llm_metrics = dict(input_token=0, output_token=0, api_call_cnt=0)
    else:
        meta_block = (
            "Source meta:\n"
            f"- parents: {', '.join(src_meta['parents'])}\n"
            f"- children: {', '.join(src_meta['children'])}\n"
            f"- synonyms: {', '.join(src_meta['synonyms'])}\n"
            f"- labels: {', '.join(src_meta['labels'])}\n\n"
            "Target meta:\n"
            f"- parents: {', '.join(tgt_meta['parents'])}\n"
            f"- children: {', '.join(tgt_meta['children'])}\n"
            f"- synonyms: {', '.join(tgt_meta['synonyms'])}\n"
            f"- labels: {', '.join(tgt_meta['labels'])}\n\n"
        )
        src_hits = src_store.query(src_emb, top_k=3)
        tgt_hits = tgt_store.query(tgt_emb, top_k=3)
        src_ctx = [f"{list_to_str(OS_meta[lbl]['labels'])}" for (lbl, uri), _ in src_hits if lbl in OS_meta]
        tgt_ctx = [f"{list_to_str(OT_meta[lbl]['labels'])}" for (lbl, uri), _ in tgt_hits if lbl in OT_meta]
        rag_block = (
            "Source context:\n- " + "\n- ".join(src_ctx) +
            "\n\nTarget context:\n- " + "\n- ".join(tgt_ctx) + "\n\n"
        )
        context_block = demo_block + meta_block + rag_block
        agents = create_agents(1, agent_type, agent_name)
        pred, llm_metrics, conf, accept = agent_inference(
            debate=False if args.debate == "false" else True,
            agent_configs=agent_configs,
            backend=agent_type, model_name=agent_name,
            source_term=(src_keycode, src_node), target_term=(tgt_keycode, tgt_node),
            pcmaps={'source': OS_map, 'target': OT_map}, dictionary=dictionary,
            options=query_opts, embed_model=embedder,
            store=(src_store, tgt_store), n_rounds=n_rounds, dropout=dropout,
            reasoning=args.reasoning, active_learning=args.active_learning,
            f1_score=f1, bisim=args.bisim, context=context_block
        )
        for k in output_metrics['api_metrics']:
            output_metrics['api_metrics'][k] += llm_metrics.get(k, 0)

    predicted_pairs.add((src_keycode, tgt_keycode))
    row = dict(
        source_label=src_label, target_label=tgt_label,
        source=src_keycode, source_uri=src_key[1],
        target=tgt_keycode, target_uri=tgt_key[1],
        similarity=sim_score, confidence=conf,
        true_relation=int(label), pred_relation=int(pred),
        api_metrics=llm_metrics, accepted=accept
    )
    prediction_file.write(json.dumps(row) + "\n")
    prediction_file.flush(); os.fsync(prediction_file.fileno())

    y_true.append(int(label)); y_pred.append(int(pred))

    if accept:
        compressed_graph, expert_q = incremental_refinement(
            G_r=compressed_graph,
            delta_G=[(src_keycode, tgt_keycode)], 
            rank_attr=rank_attr
        )
        # print to console
        for eq in (expert_q or []): print("  → expert review:", eq)
        # **append to CSV file**
        with review_file.open("a+") as rf:
            for (u, v) in expert_q or []:
                rf.write(f"{src_key},{tgt_key},{u}->{v}\n")

running_time = time.time() - start_time
prec, rec, f1 = calculate_metrics(y_true, y_pred)
output_metrics.update(running_time=running_time, metrics=dict(precision=prec, recall=rec, f1=f1))
print("\nFinal metrics:\n", json.dumps(output_metrics, indent=2))
