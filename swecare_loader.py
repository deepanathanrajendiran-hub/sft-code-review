"""Download a SWE-CARE split and emit an OOD eval / training input file.

Supports two splits:
- ``test`` (671 raw rows) -- the held-out OOD eval set; default for backward compat.
- ``dev`` (7086 raw rows) -- the larger pool used for CoRPO training prompts.

Both splits go through the same repo-overlap filter: rows whose repository name
matches one of our 4 training repos (transformers, sklearn, pydantic, fastapi)
are dropped. Match is by lowercased name suffix; owner is ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TRAIN_REPOS = {"transformers", "sklearn", "scikit-learn", "pydantic", "fastapi"}


def get_train_repo_names(train_jsonl: Path | str) -> set[str]:
    """Return the set of lowercased repo names found in a training JSONL file.

    Falls back to DEFAULT_TRAIN_REPOS if the file is missing or empty.
    """
    p = Path(train_jsonl)
    if not p.exists():
        return set(DEFAULT_TRAIN_REPOS)
    names: set[str] = set()
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            repo = row.get("repo", "")
            if "/" in repo:
                names.add(repo.split("/", 1)[1].lower())
            elif repo:
                names.add(repo.lower())
    return names or set(DEFAULT_TRAIN_REPOS)


def matches_train_repo(swecare_repo: str, train_names: set[str]) -> bool:
    """True iff the repo name (case-insensitive on swecare_repo) matches any
    train repo name exactly.

    Caller MUST pass train_names already lowercased (get_train_repo_names
    and DEFAULT_TRAIN_REPOS both satisfy this).
    """
    name = swecare_repo.split("/", 1)[-1].lower()
    return name in train_names


def map_swecare_row(row: dict) -> dict:
    """Convert a SWE-CARE dataset row to our common eval schema."""
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "diff": row["commit_to_review"]["patch_to_review"],
        "reference_comments": [
            {
                "path": c.get("path"),
                "line": c.get("line") if c.get("line") is not None else c.get("original_line"),
                "text": c.get("text", ""),
            }
            for c in row.get("reference_review_comments", [])
        ],
        "reference_text": "\n\n".join(
            c.get("text", "")
            for c in row.get("reference_review_comments", [])
            if c.get("text")
        ),
        "difficulty": row.get("metadata", {}).get("difficulty", "unknown"),
        "problem_domain": row.get("metadata", {}).get("problem_domain", "unknown"),
    }


def load_and_filter(
    train_jsonl: Path | str,
    dry_run: int | None = None,
    split: str = "test",
) -> list[dict]:
    """Load a SWE-CARE split, drop overlap, return list of preprocessed rows.

    Args:
        train_jsonl: Path to training JSONL file used to derive the exclusion repo set.
        dry_run: If set, only process the first N rows of the split.
        split: SWE-CARE split name to load. "test" (671 raw rows) is the historical
            default and is reserved for OOD eval. "dev" (7086 raw rows) is the larger
            split used for CoRPO training prompts. Both splits go through the same
            repo-overlap filter.

    If filtered set is < 300 rows, this function logs a warning to stderr.
    """
    from datasets import load_dataset

    train_names = get_train_repo_names(train_jsonl)
    print(f"[swecare_loader] excluding train repos: {sorted(train_names)}", file=sys.stderr)

    ds = load_dataset("inclusionAI/SWE-CARE", split=split)
    if dry_run is not None:
        ds = ds.select(range(min(dry_run, len(ds))))

    out: list[dict] = []
    excluded = 0
    for row in ds:
        if matches_train_repo(row["repo"], train_names):
            excluded += 1
            continue
        out.append(map_swecare_row(row))

    print(
        f"[swecare_loader] split={split} kept {len(out)} / {len(out) + excluded} rows "
        f"({excluded} excluded by repo-overlap)",
        file=sys.stderr,
    )
    if len(out) < 300 and dry_run is None:
        print(
            "[swecare_loader] WARNING: filtered set < 300 rows; consider relaxing "
            "the exclusion criterion",
            file=sys.stderr,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="ood_input.jsonl")
    ap.add_argument("--train-jsonl", default="train_dataset_clean.jsonl")
    ap.add_argument("--dry-run", type=int, default=None, help="Only process N rows")
    ap.add_argument(
        "--split",
        choices=["test", "dev"],
        default="test",
        help="SWE-CARE split to load. 'test' (default, 671 raw rows) reserved for "
             "OOD eval; 'dev' (7086 raw rows) used for CoRPO training prompts.",
    )
    args = ap.parse_args()

    rows = load_and_filter(args.train_jsonl, args.dry_run, split=args.split)
    out_path = Path(args.output)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"[swecare_loader] wrote {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
