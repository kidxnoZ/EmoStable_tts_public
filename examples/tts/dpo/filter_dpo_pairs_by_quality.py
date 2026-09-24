#!/usr/bin/env python3
"""Step 6: compute WER + UTMOS for DPO pairs and optionally filter by thresholds."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import whisper
from whisper_normalizer.english import EnglishTextNormalizer

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from dpo_utils import iter_jsonl, str2bool


LOGGER = logging.getLogger("filter_dpo_pairs_by_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute WER/UTMOS for DPO pairs and optionally filter them.")
    parser.add_argument("--pair_jsonl", type=str, required=True, help="Input dpo_pairs.jsonl")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output jsonl with computed metrics")
    parser.add_argument(
        "--filtered_output_jsonl",
        type=str,
        default="",
        help="Optional filtered output jsonl (only keep=true rows).",
    )
    parser.add_argument("--summary_json", type=str, default="", help="Optional summary json path")
    parser.add_argument("--resume", type=str, default="true")
    parser.add_argument("--max_pairs", type=int, default=0, help="For debugging only; 0 means all.")

    parser.add_argument("--compute_wer", type=str, default="true")
    parser.add_argument("--whisper_model", type=str, default="large-v3")
    parser.add_argument("--compute_utmos", type=str, default="true")
    parser.add_argument("--utmos_min_samples", type=int, default=640)

    parser.add_argument(
        "--max_chosen_wer",
        type=str,
        default="",
        help="Optional threshold. Keep row only if chosen.wer <= this value.",
    )
    parser.add_argument(
        "--max_rejected_wer",
        type=str,
        default="",
        help="Optional threshold. Keep row only if rejected.wer <= this value.",
    )
    parser.add_argument(
        "--min_chosen_utmos",
        type=str,
        default="",
        help="Optional threshold. Keep row only if chosen.utmos >= this value.",
    )
    parser.add_argument(
        "--min_rejected_utmos",
        type=str,
        default="",
        help="Optional threshold. Keep row only if rejected.utmos >= this value.",
    )
    parser.add_argument(
        "--drop_if_metric_missing",
        type=str,
        default="false",
        help="If true, drop rows when a thresholded metric is missing/failed.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_optional_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan"}:
        return None
    return float(text)


def _clean_error(exc: Exception) -> str:
    return str(exc).splitlines()[0][:500]


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


class FallbackEnglishNormalizer:
    def __call__(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def build_text_normalizer() -> Callable[[str], str]:
    try:
        return EnglishTextNormalizer()
    except Exception as exc:
        LOGGER.warning(
            "Failed to initialize EnglishTextNormalizer (%s); fallback to basic normalizer.",
            _clean_error(exc),
        )
        return FallbackEnglishNormalizer()


def compute_wer(ref_text: str, hyp_text: str, normalizer: Callable[[str], str]) -> float:
    ref = normalizer(ref_text or "").split()
    hyp = normalizer(hyp_text or "").split()
    if len(ref) == 0:
        return float(len(hyp) > 0)
    return float(levenshtein_distance(ref, hyp) / max(len(ref), 1))


def _extract_audio_metrics_cache(rec: Dict, audio_cache: Dict[str, Dict]) -> None:
    q = rec.get("quality_metrics", {})
    for side in ("chosen", "rejected"):
        sub = q.get(side, {})
        path = str(sub.get("candidate_path", "")).strip()
        if path and path not in audio_cache:
            audio_cache[path] = dict(sub)


def _resolve_candidate_path(pair: Dict, side: str) -> str:
    node = pair.get(side, {})
    path = str(node.get("candidate_path", "")).strip()
    if path:
        return path
    return ""


def compute_audio_quality(
    wav_path: str,
    target_text: str,
    whisper_model: Optional[whisper.Whisper],
    text_normalizer: Callable[[str], str],
    utmos_predictor: Optional[torch.nn.Module],
    device: str,
    compute_wer_flag: bool,
    compute_utmos_flag: bool,
    utmos_min_samples: int,
) -> Dict:
    out = {
        "candidate_path": wav_path,
        "wer": None,
        "asr_text": None,
        "utmos": None,
        "num_samples": None,
        "sr": None,
        "status": "ok",
        "errors": [],
    }

    if not wav_path or not os.path.isfile(wav_path):
        out["status"] = "failed"
        out["errors"].append(f"missing_wav:{wav_path}")
        return out

    if compute_wer_flag and whisper_model is not None:
        try:
            asr_text = whisper_model.transcribe(wav_path, language="en")["text"].strip()
            wer = compute_wer(target_text, asr_text, text_normalizer)
            out["asr_text"] = asr_text
            out["wer"] = float(wer)
        except Exception as exc:
            out["status"] = "partial_failed"
            out["errors"].append(f"wer_failed:{_clean_error(exc)}")

    if compute_utmos_flag and utmos_predictor is not None:
        try:
            wav, sr = librosa.load(wav_path, sr=None, mono=True)
            num_samples = int(wav.shape[0])
            out["sr"] = int(sr)
            out["num_samples"] = num_samples
            if num_samples < utmos_min_samples:
                out["status"] = "partial_failed"
                out["errors"].append(
                    f"utmos_too_short:num_samples={num_samples}<min_samples={utmos_min_samples}"
                )
            else:
                wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
                with torch.no_grad():
                    score = utmos_predictor(wav_tensor, sr)
                out["utmos"] = float(score.item())
        except Exception as exc:
            out["status"] = "partial_failed"
            out["errors"].append(f"utmos_failed:{_clean_error(exc)}")

    if out["status"] == "ok" and len(out["errors"]) > 0:
        out["status"] = "partial_failed"
    return out


def apply_thresholds(
    chosen: Dict,
    rejected: Dict,
    max_chosen_wer: Optional[float],
    max_rejected_wer: Optional[float],
    min_chosen_utmos: Optional[float],
    min_rejected_utmos: Optional[float],
    drop_if_metric_missing: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    def check_limit(
        metric_name: str,
        value: Optional[float],
        limit: Optional[float],
        relation: str,
    ) -> None:
        if limit is None:
            return
        if value is None:
            if drop_if_metric_missing:
                reasons.append(f"{metric_name}_missing")
            return
        if relation == "<=" and float(value) > float(limit):
            reasons.append(f"{metric_name}_gt_{limit}")
        if relation == ">=" and float(value) < float(limit):
            reasons.append(f"{metric_name}_lt_{limit}")

    check_limit("chosen_wer", chosen.get("wer"), max_chosen_wer, "<=")
    check_limit("rejected_wer", rejected.get("wer"), max_rejected_wer, "<=")
    check_limit("chosen_utmos", chosen.get("utmos"), min_chosen_utmos, ">=")
    check_limit("rejected_utmos", rejected.get("utmos"), min_rejected_utmos, ">=")
    return len(reasons) == 0, reasons


def _safe_mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if len(values) > 0 else None


def _percentile(values: List[float], q: float) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    setup_logging()

    compute_wer_flag = str2bool(args.compute_wer)
    compute_utmos_flag = str2bool(args.compute_utmos)
    resume = str2bool(args.resume)
    drop_if_metric_missing = str2bool(args.drop_if_metric_missing)

    max_chosen_wer = parse_optional_float(args.max_chosen_wer)
    max_rejected_wer = parse_optional_float(args.max_rejected_wer)
    min_chosen_utmos = parse_optional_float(args.min_chosen_utmos)
    min_rejected_utmos = parse_optional_float(args.min_rejected_utmos)

    thresholds = {
        "max_chosen_wer": max_chosen_wer,
        "max_rejected_wer": max_rejected_wer,
        "min_chosen_utmos": min_chosen_utmos,
        "min_rejected_utmos": min_rejected_utmos,
        "drop_if_metric_missing": drop_if_metric_missing,
    }
    threshold_enabled = any(v is not None for k, v in thresholds.items() if k != "drop_if_metric_missing")

    input_rows = list(iter_jsonl(args.pair_jsonl))
    if args.max_pairs > 0:
        input_rows = input_rows[: args.max_pairs]

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = set()
    audio_cache: Dict[str, Dict] = {}
    if resume and output_path.is_file():
        for rec in iter_jsonl(output_path):
            key = str(rec.get("key", "")).strip()
            if key:
                existing_keys.add(key)
            _extract_audio_metrics_cache(rec, audio_cache)

    open_mode = "a" if resume and output_path.is_file() else "w"

    device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    whisper_model = None
    normalizer = None
    if compute_wer_flag:
        normalizer = build_text_normalizer()
        whisper_model = whisper.load_model(args.whisper_model, device=("cuda" if device == "cuda" else "cpu"))
        LOGGER.info("Loaded Whisper model: %s", args.whisper_model)

    utmos_predictor = None
    if compute_utmos_flag:
        utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
            skip_validation=True,
        )
        utmos_predictor = utmos_predictor.to(device)
        utmos_predictor.eval()
        LOGGER.info("Loaded UTMOS predictor on device=%s", device)

    total = 0
    written = 0
    skipped_existing = 0
    kept_count = 0
    dropped_count = 0
    reason_counter = Counter()

    with open(output_path, open_mode, encoding="utf-8") as wf:
        for row in input_rows:
            total += 1
            key = str(row.get("key", "")).strip()
            if resume and key and key in existing_keys:
                skipped_existing += 1
                continue

            rec = dict(row)
            target_text = str(rec.get("target_text", ""))
            chosen_path = _resolve_candidate_path(rec, "chosen")
            rejected_path = _resolve_candidate_path(rec, "rejected")

            if chosen_path in audio_cache:
                chosen_metrics = dict(audio_cache[chosen_path])
            else:
                chosen_metrics = compute_audio_quality(
                    wav_path=chosen_path,
                    target_text=target_text,
                    whisper_model=whisper_model,
                    text_normalizer=normalizer if normalizer is not None else FallbackEnglishNormalizer(),
                    utmos_predictor=utmos_predictor,
                    device=device,
                    compute_wer_flag=compute_wer_flag,
                    compute_utmos_flag=compute_utmos_flag,
                    utmos_min_samples=args.utmos_min_samples,
                )
                audio_cache[chosen_path] = dict(chosen_metrics)

            if rejected_path in audio_cache:
                rejected_metrics = dict(audio_cache[rejected_path])
            else:
                rejected_metrics = compute_audio_quality(
                    wav_path=rejected_path,
                    target_text=target_text,
                    whisper_model=whisper_model,
                    text_normalizer=normalizer if normalizer is not None else FallbackEnglishNormalizer(),
                    utmos_predictor=utmos_predictor,
                    device=device,
                    compute_wer_flag=compute_wer_flag,
                    compute_utmos_flag=compute_utmos_flag,
                    utmos_min_samples=args.utmos_min_samples,
                )
                audio_cache[rejected_path] = dict(rejected_metrics)

            keep, reasons = apply_thresholds(
                chosen=chosen_metrics,
                rejected=rejected_metrics,
                max_chosen_wer=max_chosen_wer,
                max_rejected_wer=max_rejected_wer,
                min_chosen_utmos=min_chosen_utmos,
                min_rejected_utmos=min_rejected_utmos,
                drop_if_metric_missing=drop_if_metric_missing,
            )
            if not threshold_enabled:
                keep = True
                reasons = []

            if keep:
                kept_count += 1
            else:
                dropped_count += 1
                for r in reasons:
                    reason_counter[r] += 1

            rec["quality_metrics"] = {
                "chosen": chosen_metrics,
                "rejected": rejected_metrics,
            }
            rec["quality_filter"] = {
                "keep": keep,
                "reasons": reasons,
                "thresholds": thresholds,
                "threshold_enabled": threshold_enabled,
            }

            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if key:
                existing_keys.add(key)

    all_rows = list(iter_jsonl(output_path))
    chosen_wer = []
    rejected_wer = []
    chosen_utmos = []
    rejected_utmos = []
    keep_rows = []
    for rec in all_rows:
        q = rec.get("quality_metrics", {})
        ch = q.get("chosen", {})
        rj = q.get("rejected", {})
        if ch.get("wer") is not None:
            chosen_wer.append(float(ch["wer"]))
        if rj.get("wer") is not None:
            rejected_wer.append(float(rj["wer"]))
        if ch.get("utmos") is not None:
            chosen_utmos.append(float(ch["utmos"]))
        if rj.get("utmos") is not None:
            rejected_utmos.append(float(rj["utmos"]))
        keep = bool(rec.get("quality_filter", {}).get("keep", True))
        if keep:
            keep_rows.append(rec)

    filtered_output_path = None
    if args.filtered_output_jsonl:
        filtered_output_path = Path(args.filtered_output_jsonl).resolve()
        filtered_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(filtered_output_path, "w", encoding="utf-8") as f:
            for rec in keep_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "input_rows_this_run": total,
        "written_rows_this_run": written,
        "skipped_existing_rows": skipped_existing,
        "total_rows_in_output": len(all_rows),
        "kept_rows": len(keep_rows),
        "dropped_rows": len(all_rows) - len(keep_rows),
        "drop_reason_count": dict(reason_counter),
        "thresholds": thresholds,
        "threshold_enabled": threshold_enabled,
        "chosen_wer": {
            "count": len(chosen_wer),
            "mean": _safe_mean(chosen_wer),
            "p50": _percentile(chosen_wer, 50),
            "p90": _percentile(chosen_wer, 90),
        },
        "rejected_wer": {
            "count": len(rejected_wer),
            "mean": _safe_mean(rejected_wer),
            "p50": _percentile(rejected_wer, 50),
            "p90": _percentile(rejected_wer, 90),
        },
        "chosen_utmos": {
            "count": len(chosen_utmos),
            "mean": _safe_mean(chosen_utmos),
            "p10": _percentile(chosen_utmos, 10),
            "p50": _percentile(chosen_utmos, 50),
        },
        "rejected_utmos": {
            "count": len(rejected_utmos),
            "mean": _safe_mean(rejected_utmos),
            "p10": _percentile(rejected_utmos, 10),
            "p50": _percentile(rejected_utmos, 50),
        },
        "output_jsonl": str(output_path),
        "filtered_output_jsonl": str(filtered_output_path) if filtered_output_path else "",
    }

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    else:
        summary_path = None

    print(
        "[filter_dpo_pairs_by_quality] "
        f"input_this_run={total} written={written} skipped_existing={skipped_existing} "
        f"total_output={len(all_rows)} kept={len(keep_rows)} dropped={len(all_rows)-len(keep_rows)}"
    )
    print(
        "[filter_dpo_pairs_by_quality] "
        f"chosen_wer_mean={summary['chosen_wer']['mean']} chosen_utmos_mean={summary['chosen_utmos']['mean']} "
        f"rejected_wer_mean={summary['rejected_wer']['mean']} rejected_utmos_mean={summary['rejected_utmos']['mean']}"
    )
    if threshold_enabled:
        print(f"[filter_dpo_pairs_by_quality] drop_reason_count={dict(reason_counter)}")
    print(f"[filter_dpo_pairs_by_quality] output_jsonl={output_path}")
    if filtered_output_path is not None:
        print(f"[filter_dpo_pairs_by_quality] filtered_output_jsonl={filtered_output_path}")
    if summary_path is not None:
        print(f"[filter_dpo_pairs_by_quality] summary_json={summary_path}")


if __name__ == "__main__":
    main()
