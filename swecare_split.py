"""Seeded train/eval split helper.

Generic 80/20 (or other ratio) random splitter for any list-of-dicts dataset.
NOTE: the CoRPO notebook trains on the SWE-CARE dev split and evals on test;
this splitter is retained for future tasks that need to subdivide one corpus.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def split_train_eval(
    rows: list[dict],
    seed: int = 42,
    eval_fraction: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Deterministically partition rows into (train, eval) lists.

    Shuffles by seeded RNG; eval takes the first eval_fraction of the shuffle.
    """
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_fraction)
    return shuffled[n_eval:], shuffled[:n_eval]


def load_split(
    jsonl_path: Path | str,
    seed: int = 42,
    eval_fraction: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Load JSONL and return seeded (train, eval) split."""
    rows: list[dict] = []
    with Path(jsonl_path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return split_train_eval(rows, seed=seed, eval_fraction=eval_fraction)
