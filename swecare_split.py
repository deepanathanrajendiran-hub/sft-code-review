"""Seeded train/eval split helper.

Generic random splitter (default 80/20) for any list-of-dicts dataset. Not
wired into the CoRPO training path, which already ships its own dev/test split
this is here for tasks that need to carve up a single corpus.
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
    """Deterministically partition rows into (train, eval).

    Seeded shuffle, then eval takes the leading eval_fraction.
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
    """Load a JSONL file and return the seeded (train, eval) split."""
    rows: list[dict] = []
    with Path(jsonl_path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return split_train_eval(rows, seed=seed, eval_fraction=eval_fraction)
