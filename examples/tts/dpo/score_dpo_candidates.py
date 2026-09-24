#!/usr/bin/env python3
"""Step 4: score each candidate with emotion + centroid + WER + speaker similarity."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import whisper
from funasr import AutoModel
from whisper_normalizer.english import EnglishTextNormalizer

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
EXAMPLE_TTS_DIR = REPO_ROOT / "examples" / "tts"
for p in [CURRENT_DIR, EXAMPLE_TTS_DIR]:
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

from dpo_utils import (
    EMOTION_LABELS,
    canonical_emotion,
    cosine_similarity,
    iter_jsonl,
    normalize_probs,
    resolve_existing_path,
    safe_log_prob,
    str2bool,
)


LOGGER = logging.getLogger("score_dpo_candidates")


class CampplusSpeakerEmbedder:
    def __init__(self, model_dir: str):
        model_dir = str(Path(model_dir).resolve())
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"codec model dir not found: {model_dir}")

        if str(EXAMPLE_TTS_DIR / "utils") not in sys.path:
            sys.path.append(str(EXAMPLE_TTS_DIR / "utils"))
        if str(EXAMPLE_TTS_DIR / "utils" / "third_party" / "Matcha-TTS") not in sys.path:
            sys.path.append(str(EXAMPLE_TTS_DIR / "utils" / "third_party" / "Matcha-TTS"))

        from hyperpyyaml import load_hyperpyyaml
        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        from cosyvoice.utils.file_utils import load_wav

        with open(os.path.join(model_dir, "cosyvoice.yaml"), "r", encoding="utf-8") as f:
            configs = load_hyperpyyaml(f)
        speech_tokenizer = (
            os.path.join(model_dir, "speech_tokenizer_v2.onnx")
            if os.path.isfile(os.path.join(model_dir, "speech_tokenizer_v2.onnx"))
            else os.path.join(model_dir, "speech_tokenizer_v1.onnx")
        )
        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            os.path.join(model_dir, "campplus.onnx"),
            speech_tokenizer,
            os.path.join(model_dir, "spk2info.pt"),
            configs["allowed_special"],
        )
        self.load_wav = load_wav

    def extract(self, wav_path: str) -> np.ndarray:
        speech = self.load_wav(wav_path, 16000)
        emb = self.frontend._extract_spk_embedding(speech).squeeze(0).detach().cpu().numpy()
        return emb.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score DPO candidates.")
    parser.add_argument("--candidate_meta_jsonl", type=str, required=True)
    parser.add_argument("--centroids_json", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="If true, append to output_jsonl and skip existing (key, candidate_id).",
    )
    parser.add_argument(
        "--gt_embedding_meta_jsonl",
        type=str,
        default="",
        help="Optional GT embedding metadata jsonl for high-confidence key filtering.",
    )
    parser.add_argument(
        "--filter_by_gt_confidence",
        type=str,
        default="false",
        help="If true, only score keys that pass GT confidence filters.",
    )
    parser.add_argument(
        "--gt_reject_predicted_emotions",
        type=str,
        default="unk,other",
        help="Comma-separated predicted_emotion labels to reject in GT confidence filtering.",
    )
    parser.add_argument(
        "--gt_min_target_prob",
        type=float,
        default=0.0,
        help="Minimum P(target_emotion|GT_audio) required to keep a key.",
    )
    parser.add_argument(
        "--gt_require_pred_match",
        type=str,
        default="false",
        help="If true, require GT predicted_emotion == GT emotion.",
    )

    parser.add_argument("--emotion_model", type=str, default="iic/emotion2vec_plus_large")
    parser.add_argument("--speaker_backend", type=str, default="auto", choices=["auto", "campplus", "emotion2vec", "none"])
    parser.add_argument("--codec_decoder_path", type=str, default="")
    parser.add_argument("--speaker_ref_field", type=str, default="neutral_speaker_wav")

    parser.add_argument("--compute_wer", type=str, default="true")
    parser.add_argument("--whisper_model", type=str, default="large-v3")
    parser.add_argument(
        "--path_prefix",
        action="append",
        default=[],
        help="Extra path prefix for resolving relative wav paths. Repeatable.",
    )

    parser.add_argument("--w_target", type=float, default=1.0)
    parser.add_argument("--w_centroid", type=float, default=0.5)
    parser.add_argument("--w_neutral", type=float, default=0.8)
    parser.add_argument("--w_wer", type=float, default=0.3)
    parser.add_argument("--w_speaker", type=float, default=0.2)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_centroids(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    centroids = {}
    for k, v in data.get("centroids", {}).items():
        centroids[str(k)] = np.asarray(v, dtype=np.float32)
    return centroids


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


def compute_wer(ref_text: str, hyp_text: str, normalizer: EnglishTextNormalizer) -> float:
    ref = normalizer(ref_text or "").split()
    hyp = normalizer(hyp_text or "").split()
    if len(ref) == 0:
        return float(len(hyp) > 0)
    return float(levenshtein_distance(ref, hyp) / max(len(ref), 1))


def load_high_confidence_gt_keys(
    gt_meta_jsonl: str,
    reject_pred_labels: set[str],
    min_target_prob: float,
    require_pred_match: bool,
) -> set[str]:
    key_pass = {}
    for rec in iter_jsonl(gt_meta_jsonl):
        key = str(rec.get("key", ""))
        if not key:
            continue

        gt_emo = canonical_emotion(rec.get("emotion"))
        pred_emo = canonical_emotion(rec.get("predicted_emotion"))
        probs = rec.get("classifier_probs", {})
        try:
            target_prob = float(probs.get(gt_emo, 0.0))
        except Exception:
            target_prob = 0.0

        passed = (
            pred_emo not in reject_pred_labels
            and target_prob >= min_target_prob
            and ((not require_pred_match) or (pred_emo == gt_emo))
        )
        if key not in key_pass:
            key_pass[key] = passed
        else:
            # Keep key if any duplicate row passes.
            key_pass[key] = key_pass[key] or passed

    return {k for k, v in key_pass.items() if v}


def main() -> None:
    args = parse_args()
    setup_logging()
    compute_wer_flag = str2bool(args.compute_wer)
    filter_by_gt_conf = str2bool(args.filter_by_gt_confidence)
    gt_require_pred_match = str2bool(args.gt_require_pred_match)
    resume = str2bool(args.resume)

    candidate_rows = list(iter_jsonl(args.candidate_meta_jsonl))
    centroids = load_centroids(args.centroids_json)
    path_prefixes = [Path(p).resolve() for p in args.path_prefix]
    path_prefixes.extend([Path(args.candidate_meta_jsonl).resolve().parent, REPO_ROOT, Path.cwd()])
    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and output_path.is_file() else "w"

    existing_scored_pairs = set()
    if resume and output_path.is_file():
        for rec in iter_jsonl(output_path):
            key = rec.get("key")
            cand_id = rec.get("candidate_id")
            if key in (None, "", "None") or cand_id in (None, "", "None"):
                continue
            try:
                existing_scored_pairs.add((str(key), int(cand_id)))
            except Exception:
                continue

    allowed_keys = None
    if filter_by_gt_conf:
        if not args.gt_embedding_meta_jsonl:
            raise ValueError("--gt_embedding_meta_jsonl is required when --filter_by_gt_confidence true")
        reject_labels = {
            canonical_emotion(x)
            for x in str(args.gt_reject_predicted_emotions).split(",")
            if str(x).strip()
        }
        allowed_keys = load_high_confidence_gt_keys(
            args.gt_embedding_meta_jsonl,
            reject_labels,
            args.gt_min_target_prob,
            gt_require_pred_match,
        )
        LOGGER.info(
            "GT confidence filtering enabled: allowed_keys=%d reject_labels=%s min_target_prob=%.4f require_pred_match=%s",
            len(allowed_keys),
            sorted(list(reject_labels)),
            args.gt_min_target_prob,
            gt_require_pred_match,
        )

    emotion_model = AutoModel(model=args.emotion_model)
    whisper_model = None
    text_normalizer = EnglishTextNormalizer()
    if compute_wer_flag:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(args.whisper_model, device=device)

    speaker_backend = args.speaker_backend
    campplus = None
    if speaker_backend in {"auto", "campplus"}:
        if args.codec_decoder_path:
            try:
                campplus = CampplusSpeakerEmbedder(args.codec_decoder_path)
                speaker_backend = "campplus"
                LOGGER.info("Speaker backend: campplus")
            except Exception as exc:
                if args.speaker_backend == "campplus":
                    raise
                LOGGER.warning("Failed to init campplus speaker embedder, fallback to emotion2vec: %s", exc)
                speaker_backend = "emotion2vec"
        elif speaker_backend == "auto":
            speaker_backend = "emotion2vec"

    if speaker_backend == "none":
        LOGGER.info("Speaker backend disabled")
    elif speaker_backend == "emotion2vec":
        LOGGER.info("Speaker backend: emotion2vec")

    emo_cache: Dict[str, Tuple[np.ndarray, Dict[str, float], str]] = {}
    spk_cache: Dict[str, np.ndarray] = {}
    scored_rows = []

    total = 0
    ok = 0
    failed = 0
    filtered = 0
    skipped_existing = 0
    pred_counter = Counter()
    sum_score = 0.0
    sum_wer = 0.0
    sum_emo_sim = 0.0
    sum_spk = 0.0

    def get_emotion_info(wav_path: str) -> Tuple[np.ndarray, Dict[str, float], str]:
        if wav_path in emo_cache:
            return emo_cache[wav_path]
        out = emotion_model.generate(wav_path, granularity="utterance", extract_embedding=True)
        emb = np.asarray(out[0]["feats"], dtype=np.float32)
        probs = normalize_probs(list(out[0]["scores"]), EMOTION_LABELS)
        pred = max(probs, key=probs.get)
        emo_cache[wav_path] = (emb, probs, pred)
        return emb, probs, pred

    def get_speaker_emb(wav_path: str, emo_emb: Optional[np.ndarray] = None) -> np.ndarray:
        if wav_path in spk_cache:
            return spk_cache[wav_path]
        if speaker_backend == "campplus" and campplus is not None:
            emb = campplus.extract(wav_path)
        elif speaker_backend == "emotion2vec":
            if emo_emb is None:
                emo_emb, _, _ = get_emotion_info(wav_path)
            emb = emo_emb
        else:
            emb = np.zeros((1,), dtype=np.float32)
        spk_cache[wav_path] = emb
        return emb

    with open(output_path, open_mode, encoding="utf-8") as wf:
        for row in candidate_rows:
            total += 1
            rec = dict(row)
            pair_key_name = str(row.get("key", ""))
            try:
                pair_candidate_id = int(row.get("candidate_id", -1))
            except Exception:
                pair_candidate_id = -1
            pair_key = (pair_key_name, pair_candidate_id)
            try:
                if pair_key in existing_scored_pairs:
                    skipped_existing += 1
                    continue

                if allowed_keys is not None and str(row.get("key", "")) not in allowed_keys:
                    rec["status"] = "filtered_gt_confidence"
                    rec["filter_reason"] = "key_not_in_high_confidence_gt_set"
                    filtered += 1
                    scored_rows.append(rec)
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    existing_scored_pairs.add(pair_key)
                    continue

                if row.get("status") == "failed":
                    raise RuntimeError(f"candidate generation failed: {row.get('error', '')}")

                cand_path = resolve_existing_path(str(row.get("candidate_path", "")), path_prefixes)
                if not os.path.isfile(cand_path):
                    raise FileNotFoundError(f"candidate wav not found: {cand_path}")
                rec["candidate_path"] = cand_path

                target_emotion = canonical_emotion(row.get("emotion"))
                target_text = str(row.get("target_text", ""))
                ref_path = resolve_existing_path(str(row.get(args.speaker_ref_field, "")), path_prefixes)

                cand_emb, cand_probs, pred_emo = get_emotion_info(cand_path)
                pred_counter[pred_emo] += 1

                target_prob = float(cand_probs.get(target_emotion, 0.0))
                neutral_prob = float(cand_probs.get("neutral", 0.0))
                target_logp = safe_log_prob(target_prob)
                neutral_logp = safe_log_prob(neutral_prob)

                centroid = centroids.get(target_emotion)
                emo_sim = cosine_similarity(cand_emb, centroid) if centroid is not None else 0.0

                if compute_wer_flag and whisper_model is not None:
                    asr_text = whisper_model.transcribe(cand_path, language="en")["text"].strip()
                    wer = compute_wer(target_text, asr_text, text_normalizer)
                else:
                    asr_text = ""
                    wer = 0.0

                if speaker_backend == "none" or not ref_path or not os.path.isfile(ref_path):
                    spk_sim = 0.0
                else:
                    ref_emo_emb = None
                    if speaker_backend == "emotion2vec":
                        ref_emo_emb, _, _ = get_emotion_info(ref_path)
                    ref_spk_emb = get_speaker_emb(ref_path, ref_emo_emb)
                    cand_spk_emb = get_speaker_emb(cand_path, cand_emb if speaker_backend == "emotion2vec" else None)
                    spk_sim = cosine_similarity(cand_spk_emb, ref_spk_emb)

                neutral_term = 0.0
                if target_emotion != "neutral":
                    neutral_term = -args.w_neutral * neutral_logp

                score = (
                    args.w_target * target_logp
                    + args.w_centroid * emo_sim
                    + neutral_term
                    - args.w_wer * wer
                    + args.w_speaker * spk_sim
                )

                rec.update(
                    {
                        "status": "ok",
                        "scoring": {
                            "target_emotion": target_emotion,
                            "predicted_emotion": pred_emo,
                            "classifier_probs": cand_probs,
                            "target_log_prob": target_logp,
                            "neutral_log_prob": neutral_logp,
                            "emo2vec_centroid_similarity": emo_sim,
                            "wer": wer,
                            "speaker_similarity": spk_sim,
                            "transcribed_text": asr_text,
                            "score": score,
                            "weights": {
                                "target": args.w_target,
                                "centroid": args.w_centroid,
                                "neutral": args.w_neutral,
                                "wer": args.w_wer,
                                "speaker": args.w_speaker,
                            },
                        },
                    }
                )
                ok += 1
                sum_score += score
                sum_wer += wer
                sum_emo_sim += emo_sim
                sum_spk += spk_sim
            except Exception as exc:
                rec["status"] = "failed"
                rec["score_error"] = str(exc)
                failed += 1

            scored_rows.append(rec)
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_scored_pairs.add(pair_key)

    if ok > 0:
        avg_score = sum_score / ok
        avg_wer = sum_wer / ok
        avg_emo_sim = sum_emo_sim / ok
        avg_spk = sum_spk / ok
    else:
        avg_score = avg_wer = avg_emo_sim = avg_spk = 0.0

    print(
        f"[score_dpo_candidates] total={total} ok={ok} filtered={filtered} "
        f"skipped_existing={skipped_existing} failed={failed}"
    )
    print(
        f"[score_dpo_candidates] avg_score={avg_score:.4f} "
        f"avg_wer={avg_wer:.4f} avg_emo_sim={avg_emo_sim:.4f} avg_spk_sim={avg_spk:.4f}"
    )
    print(f"[score_dpo_candidates] predicted_emotion_counts={dict(pred_counter)}")
    print(f"[score_dpo_candidates] output_jsonl={output_path}")


if __name__ == "__main__":
    main()
