#!/usr/bin/env python3
"""Prepare fixed-size round subsets for iterative DPO training."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List

from dpo_utils import iter_jsonl, str2bool


LOGGER = logging.getLogger("prepare_iterative_dpo_subsets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a DPO pair jsonl into round-wise small subsets."
    )
    parser.add_argument("--pair_jsonl", type=str, required=True, help="Input DPO pair jsonl.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--pairs_per_round",
        type=int,
        required=True,
        help="How many pairs each round subset contains.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=0,
        help="Maximum rounds to generate. 0 means auto (all possible rounds).",
    )
    parser.add_argument(
        "--shuffle",
        type=str,
        default="true",
        help="Whether to shuffle before chunking.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle.")
    parser.add_argument(
        "--drop_last",
        type=str,
        default="false",
        help="If true, discard the final partial round.",
    )
    parser.add_argument(
        "--round_prefix",
        type=str,
        default="round_",
        help="Subset filename prefix.",
    )
    parser.add_argument(
        "--start_round_idx",
        type=int,
        default=0,
        help="Round index offset for naming (default: 0).",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="iterative_dpo_rounds_manifest.json",
        help="Output manifest filename under output_dir.",
    )
    parser.add_argument(
        "--train_val_ratio",
        type=str,
        default="10:1",
        help="Train:val split ratio applied before round chunking, e.g. 10:1.",
    )
    parser.add_argument(
        "--val_file_name",
        type=str,
        default="val_pairs.jsonl",
        help="Validation pair jsonl filename under output_dir.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="true",
        help="If true and subset file exists, keep it instead of rewriting.",
    )
    parser.add_argument("--dry_run", type=str, default="false")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_ratio(ratio: str) -> tuple[int, int]:
    text = str(ratio).strip()
    for sep in (":", "/"):
        if sep in text:
            lhs, rhs = text.split(sep, 1)
            break
    else:
        raise ValueError(f"Invalid ratio '{ratio}', expected format like 10:1")

    try:
        train_part = int(lhs)
        val_part = int(rhs)
    except Exception as e:
        raise ValueError(f"Invalid ratio '{ratio}', expected integers like 10:1") from e

    if train_part <= 0 or val_part <= 0:
        raise ValueError(f"Invalid ratio '{ratio}', both parts must be > 0")
    return train_part, val_part


def main() -> None:
    args = parse_args()
    setup_logging()

    if args.pairs_per_round <= 0:
        raise ValueError("--pairs_per_round must be > 0")
    if args.num_rounds < 0:
        raise ValueError("--num_rounds must be >= 0")
    if args.start_round_idx < 0:
        raise ValueError("--start_round_idx must be >= 0")

    shuffle = str2bool(args.shuffle)
    drop_last = str2bool(args.drop_last)
    resume = str2bool(args.resume)
    dry_run = str2bool(args.dry_run)

    pair_jsonl = Path(args.pair_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    subset_dir = output_dir / "subsets"
    manifest_path = output_dir / args.manifest_name
    val_path = output_dir / args.val_file_name
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(pair_jsonl))
    total_input = len(rows)
    if total_input == 0:
        raise ValueError(f"Input pair jsonl is empty: {pair_jsonl}")
    if total_input < 2:
        raise ValueError(
            f"Input pair jsonl has only {total_input} row, cannot split train/val."
        )

    if shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    train_part, val_part = _parse_ratio(args.train_val_ratio)
    ratio_total = train_part + val_part
    val_count_target = int(round(total_input * (val_part / ratio_total)))
    val_count_target = max(1, min(total_input - 1, val_count_target))
    train_rows = rows[val_count_target:]
    val_rows = rows[:val_count_target]
    train_total = len(train_rows)

    val_reused_existing = False
    val_pairs_written = 0
    if resume and val_path.is_file():
        val_reused_existing = True
        val_pairs_written = _count_jsonl(val_path)
    else:
        if not dry_run:
            _write_jsonl(val_path, val_rows)
        val_pairs_written = len(val_rows)

    if drop_last:
        max_rounds_auto = train_total // args.pairs_per_round
    else:
        max_rounds_auto = math.ceil(train_total / args.pairs_per_round)
    num_rounds = max_rounds_auto if args.num_rounds == 0 else min(args.num_rounds, max_rounds_auto)

    round_entries: List[Dict] = []
    total_pairs_written = 0
    total_pairs_skipped_existing = 0

    for local_round_idx in range(num_rounds):
        start = local_round_idx * args.pairs_per_round
        end = min((local_round_idx + 1) * args.pairs_per_round, train_total)
        chunk = train_rows[start:end]
        if len(chunk) == 0:
            continue
        if drop_last and len(chunk) < args.pairs_per_round:
            continue

        round_idx = args.start_round_idx + local_round_idx
        round_name = f"{args.round_prefix}{round_idx:03d}"
        subset_path = subset_dir / f"{round_name}.jsonl"

        reused_existing = False
        existing_count = 0
        if resume and subset_path.is_file():
            reused_existing = True
            existing_count = _count_jsonl(subset_path)
            total_pairs_skipped_existing += existing_count
        else:
            if not dry_run:
                _write_jsonl(subset_path, chunk)
            total_pairs_written += len(chunk)
            existing_count = len(chunk)

        entry = {
            "round_id": round_idx,
            "local_round_id": local_round_idx,
            "round_name": round_name,
            "subset_jsonl": str(subset_path),
            "num_pairs": existing_count,
            "start_offset": start,
            "end_offset_exclusive": end,
            "reused_existing_file": reused_existing,
            "val_pair_jsonl": str(val_path.resolve()),
        }
        round_entries.append(entry)

    manifest = {
        "source_pair_jsonl": str(pair_jsonl),
        "output_dir": str(output_dir),
        "subset_dir": str(subset_dir),
        "val_pair_jsonl": str(val_path.resolve()),
        "total_input_pairs": total_input,
        "train_pool_pairs": train_total,
        "val_pairs_target": len(val_rows),
        "val_pairs_written_or_reused": val_pairs_written,
        "val_file_reused_existing": val_reused_existing,
        "train_val_ratio": args.train_val_ratio,
        "train_ratio_part": train_part,
        "val_ratio_part": val_part,
        "pairs_per_round": args.pairs_per_round,
        "shuffle": shuffle,
        "seed": args.seed,
        "drop_last": drop_last,
        "start_round_idx": args.start_round_idx,
        "num_rounds_requested": args.num_rounds,
        "num_rounds_generated": len(round_entries),
        "total_pairs_written_this_run": total_pairs_written,
        "total_pairs_reused_existing": total_pairs_skipped_existing,
        "rounds": round_entries,
    }

    if not dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    LOGGER.info(
        "[prepare_iterative_dpo_subsets] input=%d train=%d val=%d ratio=%s rounds=%d pairs_per_round=%d written=%d reused=%d val_file=%s manifest=%s",
        total_input,
        train_total,
        len(val_rows),
        args.train_val_ratio,
        len(round_entries),
        args.pairs_per_round,
        total_pairs_written,
        total_pairs_skipped_existing,
        val_path,
        manifest_path,
    )


if __name__ == "__main__":
    main()
