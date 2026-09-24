#!/usr/bin/env python3
"""Step 5: build chosen/rejected DPO pairs from scored candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from dpo_utils import canonical_emotion, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs.")
    parser.add_argument("--scored_jsonl", type=str, required=True, help="Output of score_dpo_candidates.py")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Pair jsonl output path")
    parser.add_argument("--summary_json", type=str, default="", help="Optional summary json path")
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="If true, append missing keys to output_jsonl and skip existing keys.",
    )
    return parser.parse_args()


def _candidate_key(rec: Dict) -> Tuple[str, int]:
    return str(rec.get("key", "")), int(rec.get("candidate_id", -1))


def _pack_candidate(rec: Dict) -> Dict:
    score_block = rec.get("scoring", {})
    return {
        "candidate_id": rec.get("candidate_id"),
        "candidate_path": rec.get("candidate_path"),
        "generated_text": rec.get("generated_text", ""),
        "generated_text_token_ids": rec.get("generated_text_token_ids"),
        "generated_audio_token_ids": rec.get("generated_audio_token_ids"),
        "score": float(score_block.get("score", -1e9)),
        "predicted_emotion": score_block.get("predicted_emotion"),
        "classifier_probs": score_block.get("classifier_probs", {}),
        "emo2vec_centroid_similarity": score_block.get("emo2vec_centroid_similarity"),
        "wer": score_block.get("wer"),
        "speaker_similarity": score_block.get("speaker_similarity"),
    }


def select_pair(candidates: List[Dict], target_emotion: str) -> Optional[Tuple[Dict, Dict, str]]:
    if len(candidates) < 2:
        return None
    sorted_by_score = sorted(candidates, key=lambda x: x["scoring"]["score"], reverse=True)
    chosen = sorted_by_score[0]
    others = sorted_by_score[1:]

    reason = "lowest_score"
    rejected = None
    if target_emotion != "neutral":
        neutral_mis = [c for c in others if c["scoring"].get("predicted_emotion") == "neutral"]
        if neutral_mis:
            # Hard negative: among neutral-misclassified ones, use the highest-score one.
            rejected = sorted(neutral_mis, key=lambda x: x["scoring"]["score"], reverse=True)[0]
            reason = "hard_negative_neutral_misclassified"
    if rejected is None:
        rejected = sorted(others, key=lambda x: x["scoring"]["score"])[0]
    return chosen, rejected, reason


def main() -> None:
    args = parse_args()
    rows = [r for r in iter_jsonl(args.scored_jsonl) if r.get("status") == "ok" and "scoring" in r]
    resume = str(args.resume).strip().lower() in {"1", "true", "t", "yes", "y"}

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("key", ""))].append(r)

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and output_path.is_file() else "w"
    existing_keys = set()
    if resume and output_path.is_file():
        for rec in iter_jsonl(output_path):
            k = rec.get("key")
            if k not in (None, "", "None"):
                existing_keys.add(str(k))

    new_pairs = []
    skipped = 0
    skipped_existing = 0

    for key, cands in grouped.items():
        if not cands:
            continue
        if key in existing_keys:
            skipped_existing += 1
            continue
        target_emotion = canonical_emotion(cands[0].get("emotion"))
        pair = select_pair(cands, target_emotion)
        if pair is None:
            skipped += 1
            continue
        chosen, rejected, reason = pair

        chosen_score = float(chosen["scoring"]["score"])
        rejected_score = float(rejected["scoring"]["score"])
        margin = chosen_score - rejected_score

        pair_rec = {
            "key": key,
            "condition_hash": chosen.get("condition_hash"),
            "source_text": chosen.get("source_text"),
            "target_text": chosen.get("target_text"),
            "emotion": target_emotion,
            "raw_emotion": chosen.get("raw_emotion"),
            "emotion_text_prompt": chosen.get("emotion_text_prompt"),
            "neutral_speaker_wav": chosen.get("neutral_speaker_wav"),
            "target_wav": chosen.get("target_wav"),
            "chosen": _pack_candidate(chosen),
            "rejected": _pack_candidate(rejected),
            "score_margin": margin,
            "reject_reason": reason,
            "num_candidates_for_key": len(cands),
            "all_candidate_ids": [int(x.get("candidate_id", -1)) for x in sorted(cands, key=lambda t: t.get("candidate_id", -1))],
        }
        new_pairs.append(pair_rec)

    with open(output_path, open_mode, encoding="utf-8") as f:
        for p in new_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    all_pairs = list(iter_jsonl(output_path))
    per_emo = Counter()
    reject_reason_counter = Counter()
    margins = []
    chosen_scores = []
    rejected_scores = []
    for pair in all_pairs:
        emo = canonical_emotion(pair.get("emotion"))
        per_emo[emo] += 1
        reason = pair.get("reject_reason", "unknown")
        reject_reason_counter[reason] += 1
        try:
            margins.append(float(pair.get("score_margin", 0.0)))
        except Exception:
            margins.append(0.0)
        chosen = pair.get("chosen", {})
        rejected = pair.get("rejected", {})
        try:
            chosen_scores.append(float(chosen.get("score", 0.0)))
        except Exception:
            chosen_scores.append(0.0)
        try:
            rejected_scores.append(float(rejected.get("score", 0.0)))
        except Exception:
            rejected_scores.append(0.0)

    summary = {
        "input_rows": len(rows),
        "num_keys": len(grouped),
        "num_pairs": len(all_pairs),
        "new_pairs_this_run": len(new_pairs),
        "skipped_existing_keys": skipped_existing,
        "skipped_due_to_insufficient_candidates": skipped,
        "per_emotion_pair_count": dict(per_emo),
        "reject_reason_count": dict(reject_reason_counter),
        "avg_margin": (sum(margins) / len(margins)) if margins else 0.0,
        "avg_chosen_score": (sum(chosen_scores) / len(chosen_scores)) if chosen_scores else 0.0,
        "avg_rejected_score": (sum(rejected_scores) / len(rejected_scores)) if rejected_scores else 0.0,
    }

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    else:
        summary_path = None

    print(
        "[build_dpo_pairs] "
        f"num_keys={summary['num_keys']} num_pairs={summary['num_pairs']} "
        f"new_pairs={summary['new_pairs_this_run']} skipped_existing_keys={skipped_existing} skipped={skipped}"
    )
    print(f"[build_dpo_pairs] avg_margin={summary['avg_margin']:.4f}")
    print(f"[build_dpo_pairs] reject_reason_count={summary['reject_reason_count']}")
    print(f"[build_dpo_pairs] per_emotion_pair_count={summary['per_emotion_pair_count']}")
    print(f"[build_dpo_pairs] output_jsonl={output_path}")
    if summary_path is not None:
        print(f"[build_dpo_pairs] summary_json={summary_path}")


if __name__ == "__main__":
    main()
