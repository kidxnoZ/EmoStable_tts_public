#!/usr/bin/env python3
"""Step 2: compute per-emotion centroids from extracted emotion2vec embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from dpo_utils import EMOTION_LABELS, canonical_emotion, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-emotion centroids for DPO scoring.")
    parser.add_argument(
        "--embedding_meta_jsonl",
        type=str,
        required=True,
        help="Metadata jsonl from extract_emotion2vec_embeddings.py",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output centroid json file path.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="If true and output_json exists, skip recomputation.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum samples to keep an emotion centroid.",
    )
    parser.add_argument(
        "--reject_predicted_emotions",
        type=str,
        default="",
        help="Comma-separated predicted_emotion labels to exclude (e.g., 'unk,other').",
    )
    parser.add_argument(
        "--min_target_prob",
        type=float,
        default=0.0,
        help="Minimum P(target_emotion|GT_audio) required to keep a GT sample.",
    )
    parser.add_argument(
        "--require_pred_match",
        type=str,
        default="false",
        help="If true, require predicted_emotion == emotion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json).resolve()
    resume = str(args.resume).strip().lower() in {"1", "true", "t", "yes", "y"}
    if resume and output_path.is_file():
        print(f"[compute_emotion_centroids] resume hit, existing output found: {output_path}")
        return

    buckets: Dict[str, List[np.ndarray]] = {k: [] for k in EMOTION_LABELS}
    total = 0
    used = 0
    missing = 0
    filtered_conf = 0
    require_pred_match = str(args.require_pred_match).strip().lower() in {"1", "true", "t", "yes", "y"}
    reject_labels = {
        canonical_emotion(x)
        for x in str(args.reject_predicted_emotions).split(",")
        if str(x).strip()
    }

    for rec in iter_jsonl(args.embedding_meta_jsonl):
        total += 1
        emb_path = rec.get("embedding_path")
        if not emb_path:
            continue
        p = Path(emb_path)
        if not p.is_file():
            missing += 1
            continue
        emo = canonical_emotion(rec.get("emotion"))
        pred_emo = canonical_emotion(rec.get("predicted_emotion"))
        probs = rec.get("classifier_probs", {})
        try:
            target_prob = float(probs.get(emo, 0.0))
        except Exception:
            target_prob = 0.0
        conf_ok = (
            pred_emo not in reject_labels
            and target_prob >= args.min_target_prob
            and ((not require_pred_match) or (pred_emo == emo))
        )
        if not conf_ok:
            filtered_conf += 1
            continue
        emb = np.load(p)
        buckets.setdefault(emo, []).append(emb.astype(np.float32))
        used += 1

    centroids = {}
    counts = {}
    for emo, vecs in buckets.items():
        if len(vecs) < args.min_count:
            continue
        mat = np.stack(vecs, axis=0)
        centroid = mat.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[emo] = centroid.astype(np.float32).tolist()
        counts[emo] = int(len(vecs))

    out = {
        "source_meta": str(Path(args.embedding_meta_jsonl).resolve()),
        "total_meta_rows": total,
        "used_rows": used,
        "missing_embedding_files": missing,
        "filtered_by_confidence": filtered_conf,
        "min_count": args.min_count,
        "reject_predicted_emotions": sorted(list(reject_labels)),
        "min_target_prob": args.min_target_prob,
        "require_pred_match": require_pred_match,
        "counts": counts,
        "centroids": centroids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(
        "[compute_emotion_centroids] "
        f"total={total} used={used} missing={missing} filtered_conf={filtered_conf} "
        f"centroids={len(centroids)} output={output_path}"
    )
    print(f"[compute_emotion_centroids] counts={counts}")


if __name__ == "__main__":
    main()
