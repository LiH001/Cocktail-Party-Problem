"""Split a JSONL manifest into train and validation manifests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split manifest into train/validation files")
    parser.add_argument("--manifest", default="data/librispeech_easy_generated/manifest.jsonl")
    parser.add_argument("--train-out", default="data/librispeech_easy_generated/train.jsonl")
    parser.add_argument("--valid-out", default="data/librispeech_easy_generated/valid.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError("Need at least two rows to split a manifest")

    random.Random(args.seed).shuffle(rows)
    valid_count = max(1, int(round(len(rows) * args.valid_ratio)))
    valid_rows = rows[:valid_count]
    train_rows = rows[valid_count:]

    train_out = Path(args.train_out)
    valid_out = Path(args.valid_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    valid_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    valid_out.write_text("\n".join(valid_rows) + "\n", encoding="utf-8")

    print(f"Train rows: {len(train_rows)} -> {train_out.resolve()}")
    print(f"Valid rows: {len(valid_rows)} -> {valid_out.resolve()}")


if __name__ == "__main__":
    main()
