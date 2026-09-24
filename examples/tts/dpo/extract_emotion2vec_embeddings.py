#!/usr/bin/env python3
"""Step 1: extract emotion2vec embeddings from ground-truth training audio."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from funasr import AutoModel

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from dpo_utils import (
    EMOTION_LABELS,
    canonical_emotion,
    iter_jsonl,
    resolve_existing_path,
    str2bool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract emotion2vec embeddings for DPO.")
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Input dataset jsonl.")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output metadata jsonl path.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory to save .npy embedding files.",
    )
    parser.add_argument(
        "--emotion_model",
        type=str,
        default="iic/emotion2vec_plus_large",
        help="funasr model name/path.",
    )
    parser.add_argument("--audio_field", type=str, default="target_wav")
    parser.add_argument("--key_field", type=str, default="key")
    parser.add_argument("--emotion_field", type=str, default="emotion")
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="If true, append to output_jsonl and skip keys already written.",
    )
    parser.add_argument(
        "--unique_key_only",
        type=str,
        default="true",
        help="If true, only keep first occurrence for each key in dataset.",
    )
    parser.add_argument(
        "--path_prefix",
        action="append",
        default=[],
        help="Extra path prefix for resolving relative audio paths. Repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume = str2bool(args.resume)
    unique_key_only = str2bool(args.unique_key_only)
    dataset_path = Path(args.dataset_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    embedding_dir = Path(args.embedding_dir).resolve()
    embedding_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = CURRENT_DIR.parents[3]
    default_prefixes = [dataset_path.parent, repo_root, Path.cwd()]
    extra_prefixes = [Path(p).resolve() for p in args.path_prefix]
    prefixes = [*extra_prefixes, *default_prefixes]

    existing_keys = set()
    if resume and output_path.is_file():
        for rec in iter_jsonl(output_path):
            k = rec.get("key")
            if k not in (None, "", "None"):
                existing_keys.add(str(k))

    model = AutoModel(model=args.emotion_model)

    total = 0
    ok = 0
    failed = 0
    skipped_existing = 0
    skipped_duplicate_keys = 0
    seen_keys = set()
    wrote = 0
    open_mode = "a" if resume and output_path.is_file() else "w"
    with open(output_path, open_mode, encoding="utf-8") as wf:
        for item in iter_jsonl(dataset_path):
            total += 1
            key = str(item.get(args.key_field, f"sample_{total:08d}"))

            if unique_key_only and key in seen_keys:
                skipped_duplicate_keys += 1
                continue
            seen_keys.add(key)

            if key in existing_keys:
                skipped_existing += 1
                continue

            raw_audio = item.get(args.audio_field)
            if raw_audio is None:
                failed += 1
                continue

            audio_path = resolve_existing_path(str(raw_audio), prefixes)
            if not os.path.isfile(audio_path):
                failed += 1
                continue

            try:
                out = model.generate(audio_path, granularity="utterance", extract_embedding=True)
                emb = np.asarray(out[0]["feats"], dtype=np.float32)
                scores = out[0]["scores"]
                probs = {lab: float(scores[i]) for i, lab in enumerate(EMOTION_LABELS)}
                pred_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
                pred_emotion = EMOTION_LABELS[pred_idx]

                emb_path = embedding_dir / f"{key}.npy"
                np.save(emb_path, emb)

                rec = {
                    "key": key,
                    "audio_path": audio_path,
                    "emotion": canonical_emotion(item.get(args.emotion_field)),
                    "raw_emotion": item.get(args.emotion_field),
                    "predicted_emotion": pred_emotion,
                    "classifier_probs": probs,
                    "embedding_path": str(emb_path),
                }
                ok += 1
            except Exception as exc:
                failed += 1
                rec = {
                    "key": key,
                    "audio_path": audio_path,
                    "emotion": canonical_emotion(item.get(args.emotion_field)),
                    "raw_emotion": item.get(args.emotion_field),
                    "error": str(exc),
                }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1
            existing_keys.add(key)

    print(
        "[extract_emotion2vec_embeddings] "
        f"total={total} ok={ok} failed={failed} wrote={wrote} "
        f"skipped_existing={skipped_existing} skipped_duplicate_keys={skipped_duplicate_keys}"
    )
    print(f"[extract_emotion2vec_embeddings] output_jsonl={output_path}")
    print(f"[extract_emotion2vec_embeddings] embedding_dir={embedding_dir}")


if __name__ == "__main__":
    main()
