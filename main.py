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

from algorithms.utils import compute_node_ranks, merge_graphs
from algorithms.node2vec import compute_combined_embeddings
from algorithms.refinement import online_refine, offline_refine

from retrieval.vector_store import VectorStore

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
# PROFILING SETUP
# ----------------------------------------------------------------------------
online_call_count = 0
online_total_time = 0.0
offline_call_count = 0
offline_total_time = 0.0

start_all = time.time()

# ----------------------------------------------------------------------------
# PREDICTION FILES
# ----------------------------------------------------------------------------
predicted_keys, predicted_pairs, prediction_file = get_predicted_pairs_and_file(
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
# ensure every aligned src/tgt is in the graph
for src, tgt, _ in task_aligns:
    compressed_graph.setdefault(src, set())
    compressed_graph.setdefault(tgt, set())

# now recompute your nodes, ranks, and equivalence‐classes:
all_nodes     = set(compressed_graph) | {c for vs in compressed_graph.values() for c in vs}
rank_attr_map = compute_node_ranks(compressed_graph)
equiv_classes = {n: n for n in all_nodes}

# warm up offline
_ = offline_refine(adj=compressed_graph, rank_attr=rank_attr_map)

# ----------------------------------------------------------------------------
# SETUP EMBEDDER + VECTOR STORES
# ----------------------------------------------------------------------------
# Node2Vec parameters
n2v_kwargs = {
    'dimensions': 128,
    'walk_length': 30,
    'num_walks': 200,
    'p': 1,
    'q': 1,
    'workers': 4,
    'seed': 42
}

# embedding generation build with text
embedder = create_embedding_model("huggingface", "allenai/scibert_scivocab_uncased")

# determine which source/target keys are in the eval set
eval_src = {src_key[0] for src_key, _, _ in task_aligns}
eval_tgt = {tgt_key[0] for _, tgt_key, _ in task_aligns}

# prepare IDs and text embeddings
all_src_ids      = list(OS.keys())
all_src_labels   = [f"{OS_meta[k]['labels'][0]}" for k in all_src_ids]
all_src_text_emb = embedder.encode(all_src_labels)

all_tgt_ids      = list(OT.keys())
all_tgt_labels   = [f"{OT_meta[k]['labels'][0]}" for k in all_tgt_ids]
all_tgt_text_emb = embedder.encode(all_tgt_labels)

# compute combined (text + node2vec) embeddings
all_src_combined = compute_combined_embeddings(
    adj_dict       = compressed_graph,
    node_keys      = all_src_ids,
    text_embs      = all_src_text_emb,
    n2v_kwargs     = n2v_kwargs,
    combine_method = 'concat',
    alpha          = 0.5
)
all_tgt_combined = compute_combined_embeddings(
    adj_dict       = compressed_graph,
    node_keys      = all_tgt_ids,
    text_embs      = all_tgt_text_emb,
    n2v_kwargs     = n2v_kwargs,
    combine_method = 'concat',
    alpha          = 0.5
)

# split sampled pool
pool_src_ids    = [k for k in all_src_ids if k not in eval_src]
pool_src_embs   = np.vstack([
    all_src_combined[all_src_ids.index(k)] for k in pool_src_ids
])

pool_tgt_ids    = [k for k in all_tgt_ids if k not in eval_tgt]
pool_tgt_embs   = np.vstack([
    all_tgt_combined[all_tgt_ids.index(k)] for k in pool_tgt_ids
])

# populate vector stores with the combined embeddings
src_store = VectorStore()
src_store.add(ext_ids=pool_src_ids, embeddings=pool_src_embs)

tgt_store = VectorStore()
tgt_store.add(ext_ids=pool_tgt_ids, embeddings=pool_tgt_embs)

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

start = time.time()
online_time = 0.0
offline_time = 0.0

for idx, (src_key, tgt_key, label) in enumerate(task_aligns, 1):
    prec, rec, f1 = calculate_metrics(y_true, y_pred)
    output_metrics['metrics'] = dict(precision=prec, recall=rec, f1=f1)
    print(f"\nObservation {idx}/{len(task_aligns)} — F1 so far: {f1:.3f}")

    if idx % 5 == 0:
        # take a snapshot of the current graph
        snapshot = {u: set(vs) for u, vs in compressed_graph.items()}
        # recompute ranks for that snapshot
        snap_ranks = compute_node_ranks(snapshot)

        # time the full offline pass
        t0_snap = time.time()
        _ = offline_refine(snapshot, snap_ranks)
        rank_attr_map = compute_node_ranks(_)
        t1_snap = time.time()
        offline_time += t1_snap - t0_snap
        print(f"[Benchmark] offline_refine at idx={idx} took {t1_snap - t0_snap:.3f}s")

    src_keycode = src_key[0]
    tgt_keycode = tgt_key[0]
    src_node = OS[src_keycode]
    tgt_node = OT[tgt_keycode]
    src_meta = OS_meta[src_keycode]
    tgt_meta = OT_meta[tgt_keycode]
    src_label, tgt_label = list_to_str(src_meta['labels']), list_to_str(tgt_meta['labels'])

    if (src_keycode, tgt_keycode) in predicted_keys:
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
                f"The concept “{rec['source_label']}” and “{rec['target_label']}” "
                f"{'are related.' if rec['true_relation'] == 0 else 'do not appear to be related.'}\n\n"
            )

    meta_block = (
        f"The source concept “{src_label}” has parents {', '.join(src_meta['parents'])}, children {', '.join(src_meta['children'])}, "
        f"synonyms {', '.join(src_meta['synonyms'])}, and labels {', '.join(src_meta['labels'])}.\n"
        f"The target concept “{tgt_label}” has parents {', '.join(tgt_meta['parents'])}, children {', '.join(tgt_meta['children'])}, "
        f"synonyms {', '.join(tgt_meta['synonyms'])}, and labels {', '.join(tgt_meta['labels'])}.\n\n"
    )

    # neighbor sampling
    src_hits = src_store.query(src_emb, top_k=3)
    tgt_hits = tgt_store.query(tgt_emb, top_k=3)

    src_ctx = [f"{list_to_str(OS_meta[lbl]['labels'])}" for (lbl, uri), _ in src_hits if lbl in OS_meta]
    tgt_ctx = [f"{list_to_str(OT_meta[lbl]['labels'])}" for (lbl, uri), _ in tgt_hits if lbl in OT_meta]
    rag_block = (
        f"For the source concept “{src_label}”, the most similar context labels are: {', '.join(src_ctx)}. "
        f"For the target concept “{tgt_label}”, the most similar context labels are: {', '.join(tgt_ctx)}.\n\n"
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

    predicted_keys.add((src_keycode, tgt_keycode))
    row = dict(
        source_label=src_label, target_label=tgt_label,
        source=src_keycode, source_uri=src_key[1],
        target=tgt_keycode, target_uri=tgt_key[1],
        similarity=sim_score, confidence=conf,
        true_relation=int(label), pred_relation=int(pred),
        api_metrics=llm_metrics, accepted=accept
    )
    prediction_file.write(json.dumps(row) + "\n")
    prediction_file.flush()
    os.fsync(prediction_file.fileno())

    y_true.append(int(label)); y_pred.append(int(pred))

    if accept:
        # --- PROFILE online_refine ---
        t0 = time.time()
        delta_edges = [(src, tgt)]
        equiv_classes, expert_qs = online_refine(
            compressed_graph,
            rank_attr_map,
            equiv_classes,
            delta_edges,
            pred
        )
        t1 = time.time()
        online_time += (t1 - t0)

        # print to console
        for eq in (expert_qs or []): print("  → expert review:", eq)
        # append to CSV file
        with review_file.open("a+") as rf:
            for (u, v) in expert_qs or []:
                rf.write(f"{src_key},{tgt_key},{u}->{v}\n")

# ----------------------------------------------------------------------------
# PROFILE offline_refine
# ----------------------------------------------------------------------------
t0_off = time.time()
refined_graph = offline_refine(adj=compressed_graph, rank_attr=rank_attr_map)
t1_off = time.time()
offline_time += t1_off - t0_off

# ----------------------------------------------------------------------------
# FINAL METRICS & PROFILING REPORT
# ----------------------------------------------------------------------------
end = time.time()
total_time = end - start
prec, rec, f1 = calculate_metrics(y_true=y_true, y_pred=y_pred)
output_metrics.update({
    'running_time': total_time,  
    'metrics': {'precision': prec, 'recall': rec, 'f1': f1},
    'timings': { 'offline_refinement': offline_time, 'online_refinement': online_time, 'total': round(total_time, 3) }
})
print("\nFinal metrics:", output_metrics)

metrics_path = Path(prediction_file.name).with_suffix('.metrics.json')
with metrics_path.open('w', encoding='utf-8') as mf:
    json.dump(output_metrics, mf, indent=2)
print(f"Saved metrics to {metrics_path}")
