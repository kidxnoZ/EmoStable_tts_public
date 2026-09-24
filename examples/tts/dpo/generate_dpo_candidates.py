#!/usr/bin/env python3
"""Step 3: generate K candidate audios for each sample under fixed conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
try:
    import soundfile as sf
except Exception:
    sf = None

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
EXAMPLE_TTS_DIR = REPO_ROOT / "examples" / "tts"
SRC_DIR = REPO_ROOT / "src"
for p in [CURRENT_DIR, EXAMPLE_TTS_DIR, SRC_DIR]:
    p_str = str(p)
    if p_str in sys.path:
        sys.path.remove(p_str)
    sys.path.insert(0, p_str)

from dpo_utils import (
    canonical_emotion,
    iter_jsonl,
    load_dataset_by_key,
    resolve_existing_path,
    str2bool,
)
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from slam_llm.utils.model_utils import get_custom_model_factory
from tts_config import DataConfig, DecodeConfig, ModelConfig, TrainConfig
from utils.codec_utils import audio_decode_cosyvoice


LOGGER = logging.getLogger("generate_dpo_candidates")


def _attach_get_method(obj) -> None:
    """Make dataclass-style config objects compatible with code expecting .get()."""
    cls = obj.__class__
    if not hasattr(cls, "get"):
        def _get(self, key, default=None):
            return getattr(self, key, default)
        setattr(cls, "get", _get)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DPO candidate audios.")
    parser.add_argument("--dataset_jsonl", type=str, required=True, help="Input train/dev jsonl.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--metadata_jsonl", type=str, default="", help="Candidate metadata jsonl path.")
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="If true, append to metadata_jsonl and skip existing (key, candidate_id).",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of candidates per sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--unique_key_only",
        type=str,
        default="true",
        help="If true, only keep the first occurrence for each key and skip later duplicates.",
    )
    parser.add_argument(
        "--fallback_ref_to_target_wav",
        type=str,
        default="true",
        help="If true, fallback to target_wav when neutral_speaker_wav is missing.",
    )
    parser.add_argument(
        "--default_ref_wav",
        type=str,
        default="",
        help="Global reference wav path used when sample-level reference is missing.",
    )
    parser.add_argument(
        "--speaker_ref_jsonl",
        type=str,
        default="",
        help=(
            "Jsonl that contains neutral_speaker_wav and can be used as speaker->neutral reference map. "
            "If empty, auto-try <dataset_jsonl_dir>/test.jsonl."
        ),
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
        help="If true, only generate keys that pass GT confidence filters.",
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

    parser.add_argument("--llm_name", type=str, default="qwen2.5-0.5b")
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--llm_dim", type=int, default=896)
    parser.add_argument("--phn_tokenizer", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--peft_ckpt", type=str, default="")

    parser.add_argument("--codec_decode", type=str, default="true")
    parser.add_argument("--codec_decoder_path", type=str, required=True)
    parser.add_argument("--codec_decoder_type", type=str, default="CosyVoice")
    parser.add_argument("--cosyvoice_version", type=int, default=1)

    parser.add_argument("--group_decode", type=str, default="true")
    parser.add_argument("--group_decode_adapter_type", type=str, default="linear")
    parser.add_argument("--use_text_stream", type=str, default="false")
    parser.add_argument("--modeling_paradigm", type=str, default="parallel")

    parser.add_argument("--code_layer", type=int, default=3)
    parser.add_argument("--text_vocabsize", type=int, default=151936)
    parser.add_argument("--text_specialtokens", type=int, default=64)
    parser.add_argument("--audio_vocabsize", type=int, default=4096)
    parser.add_argument("--audio_specialtokens", type=int, default=64)
    parser.add_argument("--total_audio_vocabsize", type=int, default=4160)
    parser.add_argument("--total_vocabsize", type=int, default=156160)

    parser.add_argument("--num_latency_tokens", type=int, default=0)
    parser.add_argument("--do_layershift", type=str, default="false")
    parser.add_argument("--use_emo", type=str, default="true")

    parser.add_argument("--do_sample", type=str, default="true")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--text_repetition_penalty", type=float, default=1.2)
    parser.add_argument("--audio_repetition_penalty", type=float, default=1.2)
    parser.add_argument("--decode_text_only", type=str, default="false")
    parser.add_argument("--speech_sample_rate", type=int, default=22050)

    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def override_vocab(model_config: ModelConfig, dataset_config: DataConfig, args: argparse.Namespace) -> None:
    for vocab in [model_config.vocab_config, dataset_config.vocab_config]:
        vocab.text_vocabsize = args.text_vocabsize
        vocab.text_specialtokens = args.text_specialtokens
        vocab.audio_vocabsize = args.audio_vocabsize
        vocab.audio_specialtokens = args.audio_specialtokens
        vocab.code_layer = args.code_layer
        vocab.__post_init__()
        vocab.total_audio_vocabsize = args.total_audio_vocabsize
        vocab.total_vocabsize = args.total_vocabsize


def build_configs(args: argparse.Namespace) -> Tuple[TrainConfig, ModelConfig, DataConfig, DecodeConfig]:
    train_config = TrainConfig()
    train_config.model_name = "tts"
    train_config.seed = args.seed
    train_config.num_workers_dataloader = args.num_workers
    train_config.val_batch_size = 1
    train_config.freeze_encoder = True
    train_config.freeze_llm = True
    train_config.freeze_group_decode_adapter = True
    train_config.batching_strategy = "custom"
    train_config.modeling_paradigm = args.modeling_paradigm
    train_config.use_text_stream = str2bool(args.use_text_stream)

    model_config = ModelConfig()
    model_config.file = f"{(REPO_ROOT / 'examples' / 'tts' / 'model' / 'slam_model_tts.py').as_posix()}:model_factory"
    model_config.llm_name = args.llm_name
    model_config.llm_path = args.llm_path
    model_config.llm_dim = args.llm_dim
    model_config.phn_tokenizer = args.phn_tokenizer or None
    model_config.codec_decode = str2bool(args.codec_decode)
    model_config.codec_decoder_type = args.codec_decoder_type
    model_config.codec_decoder_path = args.codec_decoder_path
    model_config.cosyvoice_version = args.cosyvoice_version
    model_config.group_decode = str2bool(args.group_decode)
    model_config.group_decode_adapter_type = args.group_decode_adapter_type
    model_config.use_text_stream = str2bool(args.use_text_stream)
    model_config.modeling_paradigm = args.modeling_paradigm

    dataset_config = DataConfig()
    dataset_config.file = f"{(REPO_ROOT / 'examples' / 'tts' / 'speech_dataset_tts.py').as_posix()}:get_speech_dataset"
    dataset_config.train_data_path = args.dataset_jsonl
    dataset_config.val_data_path = args.dataset_jsonl
    dataset_config.inference_mode = True
    dataset_config.use_emo = str2bool(args.use_emo)
    dataset_config.num_latency_tokens = args.num_latency_tokens
    dataset_config.do_layershift = str2bool(args.do_layershift)
    dataset_config.modeling_paradigm = args.modeling_paradigm
    dataset_config.use_text_stream = str2bool(args.use_text_stream)
    dataset_config.seed = args.seed

    decode_config = DecodeConfig()
    decode_config.do_sample = str2bool(args.do_sample)
    decode_config.top_p = args.top_p
    decode_config.top_k = args.top_k
    decode_config.temperature = args.temperature
    decode_config.max_new_tokens = args.max_new_tokens
    decode_config.text_repetition_penalty = args.text_repetition_penalty
    decode_config.audio_repetition_penalty = args.audio_repetition_penalty
    decode_config.decode_text_only = str2bool(args.decode_text_only)
    decode_config.num_latency_tokens = args.num_latency_tokens
    decode_config.do_layershift = str2bool(args.do_layershift)
    decode_config.task_type = "TTS"

    override_vocab(model_config, dataset_config, args)
    for cfg in [train_config, model_config, dataset_config, decode_config, model_config.vocab_config, dataset_config.vocab_config]:
        _attach_get_method(cfg)
    return train_config, model_config, dataset_config, decode_config


def hash_condition(rec: Dict) -> str:
    payload = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def infer_speaker_id(record: Dict, key: str = "", target_wav: str = "") -> str:
    for field in ["speaker", "spk", "speaker_id", "spk_id", "voice"]:
        value = record.get(field)
        if value not in (None, "", "None"):
            return str(value)
    key = str(key or record.get("key", ""))
    if key and "_" in key:
        return key.rsplit("_", 1)[-1]
    target_wav = str(target_wav or record.get("target_wav", ""))
    if target_wav:
        stem = Path(target_wav).stem
        if "_" in stem:
            return stem.rsplit("_", 1)[-1]
    return ""


def load_speaker_ref_map(
    speaker_ref_jsonl: Path | None,
    base_prefixes: List[Path],
) -> Tuple[Dict[str, str], str]:
    if speaker_ref_jsonl is None:
        return {}, ""

    if not speaker_ref_jsonl.is_file():
        raise FileNotFoundError(f"speaker_ref_jsonl does not exist: {speaker_ref_jsonl}")

    map_prefixes = [speaker_ref_jsonl.parent, *base_prefixes]
    speaker_to_ref: Dict[str, str] = {}
    collisions = 0
    used = 0

    for rec in iter_jsonl(speaker_ref_jsonl):
        neutral_raw = rec.get("neutral_speaker_wav")
        if neutral_raw in (None, "", "None"):
            continue
        speaker_id = infer_speaker_id(rec)
        if not speaker_id:
            continue
        neutral_ref = resolve_existing_path(str(neutral_raw), map_prefixes)
        prev = speaker_to_ref.get(speaker_id)
        if prev is None:
            speaker_to_ref[speaker_id] = neutral_ref
            used += 1
        elif prev != neutral_ref:
            collisions += 1

    LOGGER.info(
        "Loaded speaker neutral reference map from %s: speakers=%d used=%d collisions=%d",
        speaker_ref_jsonl,
        len(speaker_to_ref),
        used,
        collisions,
    )
    return speaker_to_ref, str(speaker_ref_jsonl.resolve())


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
            key_pass[key] = key_pass[key] or passed
    return {k for k, v in key_pass.items() if v}


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    resume = str2bool(args.resume)
    filter_by_gt_conf = str2bool(args.filter_by_gt_confidence)
    gt_require_pred_match = str2bool(args.gt_require_pred_match)

    output_dir = Path(args.output_dir).resolve()
    metadata_path = (
        Path(args.metadata_jsonl).resolve()
        if args.metadata_jsonl
        else output_dir / "candidate_metadata.jsonl"
    )
    candidates_root = output_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and metadata_path.is_file() else "w"

    existing_candidate_pairs = set()
    if resume and metadata_path.is_file():
        for rec in iter_jsonl(metadata_path):
            key = rec.get("key")
            cand_id = rec.get("candidate_id")
            if key in (None, "", "None") or cand_id in (None, "", "None"):
                continue
            try:
                existing_candidate_pairs.add((str(key), int(cand_id)))
            except Exception:
                continue

    dataset_jsonl_path = Path(args.dataset_jsonl).resolve()
    dataset_index = load_dataset_by_key(dataset_jsonl_path)
    path_prefixes = [dataset_jsonl_path.parent, REPO_ROOT, Path.cwd()]
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

    speaker_ref_jsonl: Path | None
    if args.speaker_ref_jsonl:
        speaker_ref_jsonl = Path(args.speaker_ref_jsonl).expanduser().resolve()
    else:
        auto_speaker_ref = dataset_jsonl_path.parent / "test.jsonl"
        speaker_ref_jsonl = auto_speaker_ref if auto_speaker_ref.is_file() else None
    speaker_ref_map, speaker_ref_source = load_speaker_ref_map(speaker_ref_jsonl, path_prefixes)

    train_cfg, model_cfg, dataset_cfg, decode_cfg = build_configs(args)
    model_factory = get_custom_model_factory(model_cfg, LOGGER)
    model, tokenizer = model_factory(
        train_cfg,
        model_cfg,
        ckpt_path=args.ckpt_path,
        peft_ckpt=(args.peft_ckpt or None),
        metric="acc",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    codec_decoder = model.codec_decoder

    dataset = get_preprocessed_dataset(tokenizer, dataset_cfg, split="test")
    filtered_low_conf_samples = 0
    if allowed_keys is not None and hasattr(dataset, "data_list"):
        before_len = len(dataset.data_list)
        dataset.data_list = [x for x in dataset.data_list if str(x.get("key", "")) in allowed_keys]
        filtered_low_conf_samples = before_len - len(dataset.data_list)
        LOGGER.info(
            "Applied GT confidence key filter on generation dataset: before=%d after=%d filtered=%d",
            before_len,
            len(dataset.data_list),
            filtered_low_conf_samples,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collator,
    )

    generated = 0
    failed = 0
    recovered_existing_files = 0
    skipped_existing_candidates = 0
    seen_samples = 0
    seen_raw_records = 0
    skipped_duplicate_keys = 0
    skipped_low_confidence_keys = 0
    used_keys = set()
    with open(metadata_path, open_mode, encoding="utf-8") as wf:
        for sample_idx, batch in enumerate(dataloader):
            if sample_idx < args.start_index:
                continue
            if args.max_samples > 0 and seen_samples >= args.max_samples:
                break

            seen_raw_records += 1
            key = str(batch["keys"][0])
            if allowed_keys is not None and key not in allowed_keys:
                skipped_low_confidence_keys += 1
                continue
            if str2bool(args.unique_key_only):
                if key in used_keys:
                    skipped_duplicate_keys += 1
                    continue
                used_keys.add(key)
            seen_samples += 1

            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            src_record = dataset_index.get(key, {})
            source_text = src_record.get("source_text", batch["source_texts"][0])
            target_text = src_record.get("target_text", batch["target_texts"][0])
            raw_emotion = src_record.get("emotion")
            emotion = canonical_emotion(raw_emotion)
            emotion_text_prompt = src_record.get("emotion_text_prompt")
            target_wav_raw = src_record.get("target_wav", "")
            speaker_id = infer_speaker_id(src_record, key=key, target_wav=target_wav_raw)

            batch_neutral_ref = (
                batch["neutral_speaker_wav"][0]
                if "neutral_speaker_wav" in batch and len(batch["neutral_speaker_wav"]) > 0
                else None
            )
            neutral_ref_raw = src_record.get("neutral_speaker_wav", batch_neutral_ref)
            neutral_ref_source = "sample.neutral_speaker_wav"

            if neutral_ref_raw in (None, "", "None"):
                if speaker_id and speaker_id in speaker_ref_map:
                    neutral_ref_raw = speaker_ref_map[speaker_id]
                    neutral_ref_source = "speaker_ref_jsonl"
                elif args.default_ref_wav:
                    neutral_ref_raw = args.default_ref_wav
                    neutral_ref_source = "default_ref_wav"
                elif str2bool(args.fallback_ref_to_target_wav):
                    neutral_ref_raw = target_wav_raw
                    neutral_ref_source = "target_wav"
                else:
                    neutral_ref_source = "missing"

            neutral_ref = resolve_existing_path(str(neutral_ref_raw), path_prefixes)
            target_wav = resolve_existing_path(str(target_wav_raw), path_prefixes) if target_wav_raw else ""

            input_condition = {
                "source_text": source_text,
                "target_text": target_text,
                "emotion": emotion,
                "raw_emotion": raw_emotion,
                "emotion_text_prompt": emotion_text_prompt,
                "neutral_speaker_wav": neutral_ref,
                "speaker_id": speaker_id,
            }
            condition_hash = hash_condition(input_condition)
            sample_out_dir = candidates_root / key
            sample_out_dir.mkdir(parents=True, exist_ok=True)

            for cand_idx in range(args.k):
                candidate_key = (key, cand_idx)
                if candidate_key in existing_candidate_pairs:
                    skipped_existing_candidates += 1
                    continue

                run_seed = args.seed + sample_idx * max(args.k, 1) + cand_idx
                set_seed(run_seed)
                candidate_path = sample_out_dir / f"cand_{cand_idx:03d}.wav"
                meta = {
                    "key": key,
                    "candidate_id": cand_idx,
                    "seed": run_seed,
                    "condition_hash": condition_hash,
                    "source_text": source_text,
                    "target_text": target_text,
                    "emotion": emotion,
                    "raw_emotion": raw_emotion,
                    "emotion_text_prompt": emotion_text_prompt,
                    "speaker_id": speaker_id,
                    "neutral_speaker_wav": neutral_ref,
                    "neutral_ref_source": neutral_ref_source,
                    "speaker_ref_jsonl": speaker_ref_source,
                    "target_wav": target_wav,
                    "decode_config": {
                        "do_sample": decode_cfg.do_sample,
                        "top_p": decode_cfg.top_p,
                        "top_k": decode_cfg.top_k,
                        "temperature": decode_cfg.temperature,
                        "max_new_tokens": decode_cfg.max_new_tokens,
                        "text_repetition_penalty": decode_cfg.text_repetition_penalty,
                        "audio_repetition_penalty": decode_cfg.audio_repetition_penalty,
                        "decode_text_only": decode_cfg.decode_text_only,
                    },
                    "status": "ok",
                }

                if resume and candidate_path.is_file():
                    meta["candidate_path"] = str(candidate_path.resolve())
                    meta["status"] = "ok"
                    meta["recovered_from_existing_file"] = True
                    wf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    existing_candidate_pairs.add(candidate_key)
                    recovered_existing_files += 1
                    continue

                try:
                    with torch.no_grad():
                        if dataset_cfg.modeling_paradigm == "serial":
                            outputs = model.serial_generate(**batch, **decode_cfg.__dict__)
                        else:
                            outputs = model.generate(**batch, **decode_cfg.__dict__)

                    if dataset_cfg.modeling_paradigm in {"parallel", "serial"}:
                        text_outputs = outputs[model_cfg.vocab_config.code_layer]
                        audio_outputs = outputs[: model_cfg.vocab_config.code_layer]
                    elif dataset_cfg.modeling_paradigm == "interleaved":
                        text_outputs = outputs["text"]
                        audio_outputs = outputs["audio"]
                    else:
                        raise NotImplementedError(f"Unsupported paradigm: {dataset_cfg.modeling_paradigm}")

                    generated_text = model.tokenizer.decode(
                        text_outputs, add_special_tokens=False, skip_special_tokens=True
                    ).replace("\n", " ")
                    meta["generated_text"] = generated_text
                    meta["generated_text_token_ids"] = [int(x) for x in text_outputs.detach().cpu().tolist()]

                    if decode_cfg.decode_text_only:
                        meta["candidate_path"] = ""
                    else:
                        if dataset_cfg.modeling_paradigm != "serial":
                            if audio_outputs[0].shape[0] >= decode_cfg.max_new_tokens:
                                raise RuntimeError("audio token too long and likely degenerate")
                        else:
                            if isinstance(audio_outputs[0], list) and len(audio_outputs[0]) == 0:
                                raise RuntimeError("serial generation did not stop text stream")

                        audio_tokens = (
                            [audio_outputs[layer] for layer in range(model_cfg.vocab_config.code_layer)]
                            if model_cfg.vocab_config.code_layer > 0
                            else audio_outputs
                        )
                        if model_cfg.vocab_config.code_layer > 1:
                            audio_token_matrix = torch.stack(
                                [audio_tokens[layer].detach().cpu() for layer in range(model_cfg.vocab_config.code_layer)],
                                dim=0,
                            )
                            flat_audio_tokens = audio_token_matrix.permute(1, 0).reshape(-1).tolist()
                        elif model_cfg.vocab_config.code_layer == 1:
                            flat_audio_tokens = audio_tokens[0].detach().cpu().tolist()
                        else:
                            flat_audio_tokens = []
                        meta["generated_audio_token_ids"] = [int(x) for x in flat_audio_tokens]
                        if neutral_ref in (None, "", "None"):
                            raise RuntimeError(
                                f"missing reference wav for key={key}; set --speaker_ref_jsonl or --default_ref_wav"
                            )
                        audio_hat = audio_decode_cosyvoice(
                            audio_tokens,
                            model_cfg,
                            codec_decoder,
                            neutral_ref,
                            code_layer=model_cfg.vocab_config.code_layer,
                            num_latency_tokens=dataset_cfg.num_latency_tokens,
                            speed=1.0,
                        )
                        if audio_hat is None:
                            raise RuntimeError("audio decode failed (empty or no EOA)")

                        wav_np = audio_hat.squeeze().detach().cpu().numpy()
                        if sf is not None:
                            sf.write(str(candidate_path), wav_np, args.speech_sample_rate)
                        else:
                            wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
                            torchaudio.save(str(candidate_path), wav_t, sample_rate=args.speech_sample_rate)
                        meta["candidate_path"] = str(candidate_path.resolve())
                    generated += 1
                except Exception as exc:
                    failed += 1
                    meta["status"] = "failed"
                    meta["error"] = str(exc)
                    meta["candidate_path"] = ""
                wf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                existing_candidate_pairs.add(candidate_key)

    print(
        "[generate_dpo_candidates] "
        f"samples={seen_samples} raw_records={seen_raw_records} skipped_duplicate_keys={skipped_duplicate_keys} "
        f"skipped_low_confidence_keys={skipped_low_confidence_keys} filtered_low_conf_samples={filtered_low_conf_samples} "
        f"candidates={seen_samples * args.k} generated={generated} recovered_existing_files={recovered_existing_files} "
        f"skipped_existing_candidates={skipped_existing_candidates} failed={failed}"
    )
    print(f"[generate_dpo_candidates] metadata_jsonl={metadata_path}")
    print(f"[generate_dpo_candidates] candidate_root={candidates_root}")


if __name__ == "__main__":
    main()
