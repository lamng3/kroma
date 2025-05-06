import io
import os
import json
import sys
import tempfile
from pathlib import Path

import pytest

# we import your main as a module, not as __main__
import importlib

@pytest.fixture(autouse=True)
def dummy_environment(monkeypatch, tmp_path, caplog):
    """
    Stub out all external dependencies:
    - load_config, load_cache
    - load_ontologies_from_csv, build_matching_task
    - agent inference, bisimulation, incremental_refinement
    - vector_store
    And create minimal dummy files for configs & CSVs.
    """
    # create dummy method_config JSONL
    mc_dir = tmp_path / "experiments/configs/method/dummyllm"
    mc_dir.mkdir(parents=True)
    mc_file = mc_dir / "dummy.jsonl"
    mc_file.write_text(json.dumps({
        "agent_type": "openai",
        "agent_name": "dummy-model",
        "method_name": "dummy",
        "task": "dummytask",
        "query_options": ["ncit"]
    }) + "\n")

    # dummy dataset config
    ds_dir = tmp_path / "experiments/configs/datasets"
    ds_dir.mkdir(parents=True)
    ds_file = ds_dir / "dummytask.json"
    ds_file.write_text(json.dumps({
        "task_name": "Dummy Task",
        "sample_sz": 1,
        "dataset_type": "ncit-doid",
        "csv_folder": str(tmp_path / "data"),
        "datasets": ["set1"]
    }))

    # dummy CSV folder and file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "set1.csv"
    csv_file.write_text("src_code,src_ety,tgt_code,tgt_ety,score,tgt_cands\n"
                        "A,http://a,B,http://b,1,['http://b']\n")

    # monkey‐patch load_config and load_cache to use tmp_path base
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    # override paths inside main
    from utils.file import load_config, load_cache
    monkeypatch.setattr("utils.file.load_config", lambda p: load_config(str(tmp_path / Path(p).relative_to(Path().cwd()))))
    monkeypatch.setattr("utils.file.load_cache", lambda p: {})

    # stub agents.factory.create_agents
    import agents.factory as af
    monkeypatch.setattr(af, "create_agents", lambda n,t,m: {0: None})

    # stub agent_inference to always return no, metrics zeros, conf 0, accepted False
    import agents.prompter as ap
    monkeypatch.setattr(ap, "inference", lambda *args, **kwargs: (0, {'input_token':0,'output_token':0,'api_call_cnt':0}, 0, False))

    # stub bisimulation to empty
    import algorithms.bisimulation as ab
    monkeypatch.setattr(ab, "bisimulation", lambda g: set())

    # stub incremental_refinement to pass through
    import algorithms.incremental_refinement as ir
    monkeypatch.setattr(ir, "incremental_refinement", lambda O_S,O_T,G,Δ,rank_attr: (G, []))

    # stub RAG store query
    import retrieval.vector_store as vs
    class DummyStore:
        def __init__(self,*a,**k): pass
        def add(self,*a,**k): pass
        def query(self,*a,**k): return []
        def get_text(self,*a,**k): return ""
    monkeypatch.setattr(vs, "VectorStore", DummyStore)
    # stub embedder
    import inference.factory as inf
    monkeypatch.setattr(inf, "create_embedding_model", lambda *a,**k: type("E",[object],{"encode":lambda self,x: [[0]*384 for _ in x]})())

    yield

def test_main_runs(tmp_path, capsys):
    # move cwd to tmp_path so all relative paths resolve there
    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        # import and run main
        import main
        sys.argv = ["main.py", "--method_config", "dummy", "--llm", "dummyllm"]
        main.main()
    finally:
        os.chdir(old_cwd)

    # capture stdout for the “Starting KROMA evaluation...” message
    captured = capsys.readouterr()
    assert "Starting KROMA evaluation" in captured.out
