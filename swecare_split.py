"""Seeded 80/20 train/eval split of SWE-CARE OOD prompts.

The 632-row OOD eval set lives at ood_input.jsonl. CoRPO trains on the 80%
train slice; the 20% eval slice is used for variance-gate sanity but NOT for
the final OOD pairwise eval (which uses the full 632 rows).
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
