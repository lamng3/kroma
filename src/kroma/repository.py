"""Writers for prediction JSONL files and expert-review CSV files."""

import json
import os
from pathlib import Path

from kroma.utils.file import get_predicted_pairs_and_file


class PredictionRepository:
    """Append-only store of predicted alignment rows."""

    def __init__(self, task_key, method_name, agent_name, **flags):
        self.keys, self.rows, self._handle = get_predicted_pairs_and_file(
            task_key,
            method_name,
            agent_name,
            **flags,
        )

    def seen(self, key) -> bool:
        return key in self.keys

    def append(self, row: dict) -> None:
        self._handle.write(json.dumps(row) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.keys.add((row["source"], row["target"]))
        self.rows.append(row)

    @property
    def name(self) -> str:
        return self._handle.name


class ReviewRepository:
    """CSV of pairs sent to an expert after online refinement."""

    def __init__(self, subtask: str):
        review_dir = Path("reviews") / "baseline" / subtask
        review_dir.mkdir(parents=True, exist_ok=True)
        self.path = review_dir / "expert_queries.csv"
        if not self.path.exists():
            self.path.write_text("source,target,expert_edge\n")

    def append(self, src_key, tgt_key, expert_qs) -> None:
        with self.path.open("a+") as handle:
            for src, tgt in expert_qs or []:
                handle.write(f"{src_key},{tgt_key},{src}->{tgt}\n")
