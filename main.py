import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import logging

from config.constants import METADATA
from utils.env import get_env
from utils.file import load_config, load_cache, get_predicted_pairs_and_file

from agents.factory import create_agents
from agents.prompter import inference as agent_inference
from agents.utils import simple_match

from inference.factory import create_embedding_model

from tasks.loader import CsvAlignmentLoader
from tasks.builder import build_ontology_matching_task

from retrieval.vector_store import VectorStore

from algorithms.bisimulation import bisimulation
from algorithms.incremental_refinement import incremental_refinement

from metrics.scoring import calculate_metrics

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------
random.seed(42)
print("Starting KROMA evaluation...")

# logging: write to file and also print INFO to console if desired
log_path = Path("logs/kroma.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

# configure file handler
logging.basicConfig(
    filename=str(log_path),
    filemode="w",
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

# also add a console handler so logger.info() appears on stdout
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console)

logger = logging.getLogger()  
logger.info(f"Logging initialized — writing to {log_path}")

# ------------------------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate KROMA ontology matching")
parser.add_argument("--method_config", required=True)
parser.add_argument("--llm", required=True)
parser.add_argument("--size", choices=['xsmall','small','medium','large','full'], default="full")
parser.add_argument("--reasoning", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--active_learning", action="store_true")
parser.add_argument("--compare_models", action="store_true")
parser.add_argument("--bisim", action="store_true")
args = parser.parse_args()

# ------------------------------------------------------------------------------
# LOAD CONFIGS
# ------------------------------------------------------------------------------
method_cfg_path = Path("experiments/configs/method")/args.llm/f"{args.method_config}.jsonl"
logger.info(f"Method config path: {method_cfg_path}")
method_cfg = load_config(method_cfg_path)
agent_type, agent_name = method_cfg['agent_type'], method_cfg['agent_name']
method_name   = method_cfg['method_name']
task_key      = method_cfg['task']
query_opts    = method_cfg['query_options']

ds_cfg_path = Path("experiments/configs/datasets")/f"{task_key}.json"
logger.info(f"Dataset config path: {ds_cfg_path}")
ds_cfg = load_config(ds_cfg_path)
csv_paths = [ Path(ds_cfg['csv_folder'])/f"{n}.csv" for n in ds_cfg['datasets'] ]

dict_paths = load_config("experiments/configs/dictionary.json")
dictionary = {k: load_cache(dict_paths[k]) for k in dict_paths}

# ------------------------------------------------------------------------------
# PREDICTION FILES
# ------------------------------------------------------------------------------
predicted_pairs, prediction_file = get_predicted_pairs_and_file(
    task_key, method_name, agent_name,
    debate=False,
    size=args.size,
    reasoning=args.reasoning,
    baseline=args.baseline,
    active_learning=args.active_learning,
    compare_models=args.compare_models,
    bisim=args.bisim,
)

# ------------------------------------------------------------------------------
# BUILD ONTOLOGY TASK
# ------------------------------------------------------------------------------
# load raw graphs + alignments
loader = CsvAlignmentLoader(csv_paths, dataset=ds_cfg['dataset_type'])
OS, OT, raw_aligns = loader.load()

# sample & build parent/child maps
OS, OT, task_aligns, OS_map, OT_map = build_ontology_matching_task(
    OS, OT, raw_aligns, sample_sz=ds_cfg['sample_sz']
)

# initialize the shared compressed_graph and rank_attr for bisimulation & refinement
from agents.prompter import initialize_graph
initialize_graph(OS, OT)

# ------------------------------------------------------------------------------
# PREPARE COMPRESSED GRAPH + RANKS
# ------------------------------------------------------------------------------
compressed_graph = {**OS, **OT}
from algorithms.bisimulation import compute_node_ranks
rank_attr = compute_node_ranks(compressed_graph)

# ------------------------------------------------------------------------------
# SETUP RAG VECTOR STORE (in‐memory cosine similarity)
# ------------------------------------------------------------------------------
embedder = create_embedding_model("huggingface", "allenai/scibert_scivocab_uncased")
store = VectorStore()

# index all concept identifiers (use node tuple as ext_id)
concept_ids = list(compressed_graph.keys())
concept_texts = [f"{lbl} {uri}" for lbl, uri in concept_ids]
concept_embs = embedder.encode(concept_texts)

# now `add(ext_ids, embeddings)`
store.add(ext_ids=concept_ids, embeddings=concept_embs)

# ------------------------------------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------------------------------------
y_true, y_pred = [], []
output_metrics = {
    'method': method_name,
    'datasets': [str(p) for p in csv_paths],
    'agent_model': agent_name,
    'task': ds_cfg['task_name'],
    'query_options': query_opts,
    'api_metrics': {'input_token':0,'output_token':0,'token_count':0,'api_call_cnt':0},
    'metrics': {'precision':-1,'recall':-1,'f1':-1}
}
start_time = time.time()

size_map = {'xsmall':0.2,'small':0.4,'medium':0.6,'large':0.8,'full':1.0}
limit = int(len(task_aligns)*size_map[args.size])
if limit < len(task_aligns):
    task_aligns = random.sample(task_aligns, limit)

for idx, (p, c, label) in enumerate(task_aligns, 1):
    prec, rec, f1 = calculate_metrics(y_true, y_pred)
    output_metrics['metrics'] = {'precision':prec,'recall':rec,'f1':f1}
    print(f"\nObservation {idx}/{len(task_aligns)} — F1 so far: {f1:.3f}")

    src_node, tgt_node = p[0], c[0]
    if (src_node, tgt_node) in predicted_pairs:
        print("  • already done")
        continue

    # bisimulation quick accept
    if args.bisim and (src_node, tgt_node) in bisimulation(compressed_graph):
        pred, conf, accept = 1, 10, True
        llm_metrics = {'input_token':0,'output_token':0,'api_call_cnt':0}
    else:
        # RAG contexts via in‐memory store
        src_emb = embedder.encode([src_node[0]])[0]
        tgt_emb = embedder.encode([tgt_node[0]])[0]
        src_hits = store.query(src_emb, top_k=3)     # List[(ext_id, score)]
        tgt_hits = store.query(tgt_emb, top_k=3)

        # retrieve texts for context
        src_ctx = [f"{lbl} {uri}" for (lbl,uri) in [hid for hid,_ in src_hits]]
        tgt_ctx = [f"{lbl} {uri}" for (lbl,uri) in [hid for hid,_ in tgt_hits]]
        context_block = (
            "Source context:\n- " + "\n- ".join(src_ctx) +
            "\n\nTarget context:\n- " + "\n- ".join(tgt_ctx) + "\n\n"
        )

        # LLM inference
        agents = create_agents(1, agent_type, agent_name)
        pred, llm_metrics, conf, accept = agent_inference(
            backend=agent_type,
            model_name=agent_name,
            source_term=src_node,
            target_term=tgt_node,
            pcmaps={'source':OS_map,'target':OT_map},
            dictionary=dictionary,
            options=query_opts,
            embed_model=embedder,
            store=store,
            n_rounds=1,
            dropout=0.0,
            reasoning=args.reasoning,
            active_learning=args.active_learning,
            f1_score=f1,
            bisim=args.bisim,
        )
        for k in output_metrics['api_metrics']:
            output_metrics['api_metrics'][k] += llm_metrics.get(k,0)

    # record prediction
    predicted_pairs.add((src_node, tgt_node))
    row = {
        "source": src_node[0],
        "source_uri": src_node[1],
        "target": tgt_node[0],
        "target_uri": tgt_node[1],
        "confidence": conf,
        "true_relation": int(label),
        "pred_relation": int(pred),
        "api_metrics": llm_metrics,
        "accepted": accept
    }
    prediction_file.write(json.dumps(row) + "\n")
    prediction_file.flush()
    os.fsync(prediction_file.fileno())

    y_true.append(int(label))
    y_pred.append(int(pred))

    # incremental refinement on accept
    if accept:
        compressed_graph, expert_q = incremental_refinement(
            O_S=OS,
            O_T=OT,
            G_r=compressed_graph,
            delta_G=[(src_node, tgt_node)],
            rank_attr=rank_attr
        )
        if expert_q:
            for eq in expert_q:
                print("  → expert review:", eq)

# finalize
running_time = time.time() - start_time
prec, rec, f1 = calculate_metrics(y_true, y_pred)
output_metrics.update({'running_time':running_time,'metrics':{'precision':prec,'recall':rec,'f1':f1}})
print("\nFinal metrics:\n", json.dumps(output_metrics, indent=2))
