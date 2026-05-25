import json
from pathlib import Path

import pytest

from swecare_split import split_train_eval, load_split


class TestSplit:
    def test_split_is_seeded_reproducibly(self):
        rows = [{"instance_id": f"id_{i}"} for i in range(100)]
        train1, eval1 = split_train_eval(rows, seed=42, eval_fraction=0.2)
        train2, eval2 = split_train_eval(rows, seed=42, eval_fraction=0.2)
        assert [r["instance_id"] for r in train1] == [r["instance_id"] for r in train2]
        assert [r["instance_id"] for r in eval1] == [r["instance_id"] for r in eval2]

    def test_split_no_overlap(self):
        rows = [{"instance_id": f"id_{i}"} for i in range(100)]
        train, eval_ = split_train_eval(rows, seed=42, eval_fraction=0.2)
        train_ids = {r["instance_id"] for r in train}
        eval_ids = {r["instance_id"] for r in eval_}
        assert train_ids.isdisjoint(eval_ids)
        assert len(train_ids) + len(eval_ids) == 100

    def test_split_eval_fraction_approx(self):
        rows = [{"instance_id": f"id_{i}"} for i in range(100)]
        train, eval_ = split_train_eval(rows, seed=42, eval_fraction=0.2)
        assert len(eval_) == 20
        assert len(train) == 80

    def test_different_seeds_give_different_splits(self):
        rows = [{"instance_id": f"id_{i}"} for i in range(100)]
        _, eval1 = split_train_eval(rows, seed=42, eval_fraction=0.2)
        _, eval2 = split_train_eval(rows, seed=43, eval_fraction=0.2)
        assert {r["instance_id"] for r in eval1} != {r["instance_id"] for r in eval2}


class TestLoadSplit:
    def test_load_split_from_jsonl(self, tmp_path):
        rows = [{"instance_id": f"id_{i}", "diff": f"diff{i}"} for i in range(50)]
        input_path = tmp_path / "ood_input.jsonl"
        with input_path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        train, eval_ = load_split(input_path, seed=42, eval_fraction=0.2)
        assert len(eval_) == 10
        assert len(train) == 40
