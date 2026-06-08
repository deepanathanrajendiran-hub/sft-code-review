"""Download a SWE-CARE split and emit an OOD eval / training input file.

Two splits are available: ``test`` is the held-out OOD eval set (the default),
``dev`` is the larger pool used for CoRPO training prompts. Both go through the
same filter that drops rows whose repo matches one of our 4 training repos
(transformers, sklearn, pydantic, fastapi), matched by lowercased name suffix
with owner ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TRAIN_REPOS = {"transformers", "sklearn", "scikit-learn", "pydantic", "fastapi"}


def get_train_repo_names(train_jsonl: Path | str) -> set[str]:
    """Lowercased repo names from a training JSONL, or DEFAULT_TRAIN_REPOS if missing/empty."""
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
    """Whether swecare_repo's name matches any train repo.

    train_names must already be lowercased (both get_train_repo_names and
    DEFAULT_TRAIN_REPOS satisfy this).
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
    """Load a SWE-CARE split, drop repo-overlap rows, return preprocessed rows.

    train_jsonl seeds the exclusion repo set. dry_run, if set, limits to the
    first N rows. split is "test" (OOD eval) or "dev" (CoRPO training prompts);
    both get the same overlap filter. Warns on stderr if fewer than 300 rows survive.
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
