#!/usr/bin/env python3
"""Compute GT WER + UTMOS for dataset jsonl (e.g., test.jsonl)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import librosa
import numpy as np
import torch
import whisper
from whisper_normalizer.english import EnglishTextNormalizer

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
for p in [CURRENT_DIR, REPO_ROOT]:
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

from dpo_utils import iter_jsonl, resolve_existing_path, str2bool


LOGGER = logging.getLogger("compute_gt_quality")


class FallbackEnglishNormalizer:
    def __call__(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9' ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute GT WER + UTMOS and store into one jsonl.")
    parser.add_argument("--dataset_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--summary_json", type=str, default="")
    parser.add_argument("--resume", type=str, default="true")
    parser.add_argument("--max_rows", type=int, default=0)

    parser.add_argument("--audio_field", type=str, default="target_wav")
    parser.add_argument("--text_field", type=str, default="target_text")
    parser.add_argument(
        "--path_prefix",
        action="append",
        default=[],
        help="Extra path prefix for resolving relative wav paths. Repeatable.",
    )

    parser.add_argument("--compute_wer", type=str, default="true")
    parser.add_argument("--whisper_model", type=str, default="large-v3")
    parser.add_argument("--compute_utmos", type=str, default="true")
    parser.add_argument("--utmos_min_samples", type=int, default=640)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _clean_error(exc: Exception) -> str:
    return str(exc).splitlines()[0][:500]


def build_normalizer() -> Callable[[str], str]:
    try:
        return EnglishTextNormalizer()
    except Exception as exc:
        LOGGER.warning(
            "Failed to initialize EnglishTextNormalizer (%s), fallback to basic normalizer.",
            _clean_error(exc),
        )
        return FallbackEnglishNormalizer()


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


def compute_wer(ref_text: str, hyp_text: str, normalizer: Callable[[str], str]) -> float:
    ref = normalizer(ref_text or "").split()
    hyp = normalizer(hyp_text or "").split()
    if len(ref) == 0:
        return float(len(hyp) > 0)
    return float(levenshtein_distance(ref, hyp) / max(len(ref), 1))


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

    rows = list(iter_jsonl(args.dataset_jsonl))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path_prefixes = [Path(p).resolve() for p in args.path_prefix]
    path_prefixes.extend(
        [
            Path(args.dataset_jsonl).resolve().parent,
            REPO_ROOT,
            Path.cwd(),
        ]
    )

    existing_keys = set()
    if resume and output_path.is_file():
        for rec in iter_jsonl(output_path):
            key = str(rec.get("key", "")).strip()
            if key:
                existing_keys.add(key)

    open_mode = "a" if resume and output_path.is_file() else "w"

    device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    normalizer = build_normalizer() if compute_wer_flag else FallbackEnglishNormalizer()

    whisper_model = None
    if compute_wer_flag:
        whisper_device = "cuda" if device == "cuda" else "cpu"
        whisper_model = whisper.load_model(args.whisper_model, device=whisper_device)
        LOGGER.info("Loaded Whisper model=%s on device=%s", args.whisper_model, whisper_device)

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
    failed = 0

    wer_values = []
    utmos_values = []

    with open(output_path, open_mode, encoding="utf-8") as wf:
        for item in rows:
            total += 1
            rec = dict(item)
            key = str(rec.get("key", "")).strip()
            if resume and key and key in existing_keys:
                skipped_existing += 1
                continue

            raw_wav = str(rec.get(args.audio_field, "")).strip()
            wav_path = resolve_existing_path(raw_wav, path_prefixes)
            target_text = str(rec.get(args.text_field, ""))

            out = {
                "key": key,
                "audio_field": args.audio_field,
                "audio_path": wav_path,
                "text_field": args.text_field,
                "target_text": target_text,
                "wer": None,
                "asr_text": None,
                "utmos": None,
                "status": "ok",
                "errors": [],
            }

            if not os.path.isfile(wav_path):
                out["status"] = "failed"
                out["errors"].append(f"missing_wav:{wav_path}")
                failed += 1
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                if key:
                    existing_keys.add(key)
                written += 1
                continue

            if compute_wer_flag and whisper_model is not None:
                try:
                    asr_text = whisper_model.transcribe(wav_path, language="en")["text"].strip()
                    out["asr_text"] = asr_text
                    out["wer"] = compute_wer(target_text, asr_text, normalizer)
                    wer_values.append(float(out["wer"]))
                except Exception as exc:
                    out["status"] = "partial_failed"
                    out["errors"].append(f"wer_failed:{_clean_error(exc)}")

            if compute_utmos_flag and utmos_predictor is not None:
                try:
                    wav, sr = librosa.load(wav_path, sr=None, mono=True)
                    if int(wav.shape[0]) < args.utmos_min_samples:
                        out["status"] = "partial_failed"
                        out["errors"].append(
                            f"utmos_too_short:num_samples={int(wav.shape[0])}<min_samples={args.utmos_min_samples}"
                        )
                    else:
                        wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
                        with torch.no_grad():
                            score = utmos_predictor(wav_tensor, sr)
                        out["utmos"] = float(score.item())
                        utmos_values.append(float(out["utmos"]))
                except Exception as exc:
                    out["status"] = "partial_failed"
                    out["errors"].append(f"utmos_failed:{_clean_error(exc)}")

            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            if key:
                existing_keys.add(key)
            written += 1

    all_rows = list(iter_jsonl(output_path))
    summary = {
        "input_rows_this_run": total,
        "written_rows_this_run": written,
        "skipped_existing_rows": skipped_existing,
        "failed_rows_this_run": failed,
        "total_rows_in_output": len(all_rows),
        "wer": {
            "count": len(wer_values),
            "mean": _safe_mean(wer_values),
            "p50": _percentile(wer_values, 50),
            "p90": _percentile(wer_values, 90),
        },
        "utmos": {
            "count": len(utmos_values),
            "mean": _safe_mean(utmos_values),
            "p10": _percentile(utmos_values, 10),
            "p50": _percentile(utmos_values, 50),
        },
        "output_jsonl": str(output_path),
    }

    summary_path = None
    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        "[compute_gt_quality] "
        f"input_this_run={total} written={written} skipped_existing={skipped_existing} "
        f"failed_this_run={failed} total_output={len(all_rows)}"
    )
    print(
        "[compute_gt_quality] "
        f"wer_mean={summary['wer']['mean']} utmos_mean={summary['utmos']['mean']}"
    )
    print(f"[compute_gt_quality] output_jsonl={output_path}")
    if summary_path is not None:
        print(f"[compute_gt_quality] summary_json={summary_path}")


if __name__ == "__main__":
    main()
