import json
import logging
import os
import random
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from slam_llm.utils.model_utils import get_custom_model_factory
from tts_config import DataConfig, FSDPConfig, LogConfig, ModelConfig, TrainConfig


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    debug: bool = False
    metric: str = "acc"
    ckpt_path: Optional[str] = None
    peft_ckpt: Optional[str] = None
    probe_log: str = "output/eoa_probe"
    manifest_path: Optional[str] = None
    probe_topk: int = 10


def _configure_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    if logger.handlers:
        logger.handlers[0].setLevel(logging.INFO)
        logger.handlers[0].setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger


def _init_bucket_metrics(num_streams, stream_names=None):
    if stream_names is None:
        stream_names = [f"layer_{i}" for i in range(num_streams)]
    return {
        "overall": {"count": 0, "top1": 0, "top5": 0, "top10": 0, "p_eoa": [], "rank": []},
        "per_stream": [
            {
                "name": stream_names[i],
                "count": 0,
                "top1": 0,
                "top5": 0,
                "top10": 0,
                "p_eoa": [],
                "rank": [],
            }
            for i in range(num_streams)
        ],
    }


def _update_metric(metric_store, layer_idx, p_eoa, rank, topk_hit_5, topk_hit_10, top1_hit):
    metric_store["overall"]["count"] += 1
    metric_store["overall"]["top1"] += int(top1_hit)
    metric_store["overall"]["top5"] += int(topk_hit_5)
    metric_store["overall"]["top10"] += int(topk_hit_10)
    metric_store["overall"]["p_eoa"].append(float(p_eoa))
    metric_store["overall"]["rank"].append(int(rank))

    layer_store = metric_store["per_stream"][layer_idx]
    layer_store["count"] += 1
    layer_store["top1"] += int(top1_hit)
    layer_store["top5"] += int(topk_hit_5)
    layer_store["top10"] += int(topk_hit_10)
    layer_store["p_eoa"].append(float(p_eoa))
    layer_store["rank"].append(int(rank))


def _finalize_metric(metric_store):
    def finalize_leaf(leaf):
        count = leaf["count"]
        if count <= 0:
            return {
                "count": 0,
                "eoa_acc": None,
                "eoa_top5": None,
                "eoa_top10": None,
                "mean_p_eoa": None,
                "median_p_eoa": None,
                "mean_rank": None,
                "median_rank": None,
            }
        return {
            "count": count,
            "eoa_acc": leaf["top1"] / count,
            "eoa_top5": leaf["top5"] / count,
            "eoa_top10": leaf["top10"] / count,
            "mean_p_eoa": mean(leaf["p_eoa"]),
            "median_p_eoa": median(leaf["p_eoa"]),
            "mean_rank": mean(leaf["rank"]),
            "median_rank": median(leaf["rank"]),
        }

    return {
        "overall": finalize_leaf(metric_store["overall"]),
        "per_stream": [
            {
                "name": layer_store["name"],
                **finalize_leaf(layer_store),
            }
            for layer_store in metric_store["per_stream"]
        ],
    }


def _load_manifest_buckets(manifest_path):
    if manifest_path in (None, "", "None"):
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["key"]: item.get("bucket", "unknown") for item in data.get("items", [])}


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    main(cfg)


def main(kwargs: DictConfig):
    train_config = kwargs.train_config
    fsdp_config = kwargs.fsdp_config
    model_config = kwargs.model_config
    log_config = kwargs.log_config
    dataset_config = kwargs.dataset_config
    factory_kwargs = OmegaConf.create(OmegaConf.to_container(kwargs, resolve=True))
    for reserved_key in ["train_config", "fsdp_config", "model_config", "log_config", "dataset_config"]:
        if reserved_key in factory_kwargs:
            del factory_kwargs[reserved_key]

    probe_dir = kwargs.get("probe_log")
    os.makedirs(probe_dir, exist_ok=True)
    log_file = os.path.join(probe_dir, "probe.log")
    logger = _configure_logger(log_file)

    logger.info("train_config: %s", train_config)
    logger.info("fsdp_config: %s", fsdp_config)
    logger.info("model_config: %s", model_config)
    logger.info("dataset_config(before override): %s", dataset_config)
    logger.info("ckpt_path: %s", kwargs.get("ckpt_path"))

    dataset_config.inference_mode = False
    logger.info("dataset_config(inference_mode forced false): %s", dataset_config)

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **factory_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset_test = get_preprocessed_dataset(tokenizer, dataset_config, split="test")
    logger.info("teacher-forced probe dataset size: %d", len(dataset_test))

    manifest_buckets = _load_manifest_buckets(kwargs.get("manifest_path"))
    topk = max(int(kwargs.get("probe_topk", 10)), 1)
    eoa_id = model_config.vocab_config.eoa
    code_layer = model_config.vocab_config.code_layer
    use_future_mtp = bool(getattr(model, "use_future_mtp", False))
    if use_future_mtp:
        num_probe_streams = int(getattr(model, "future_mtp_num", 0)) + 1
        stream_names = ["main"] + [f"future{i}" for i in range(1, num_probe_streams)]
    else:
        num_probe_streams = code_layer
        stream_names = [f"layer_{i}" for i in range(code_layer)]

    aggregate = {
        "all": _init_bucket_metrics(num_probe_streams, stream_names),
        "warning": _init_bucket_metrics(num_probe_streams, stream_names),
        "normal": _init_bucket_metrics(num_probe_streams, stream_names),
        "unknown": _init_bucket_metrics(num_probe_streams, stream_names),
    }
    per_sample = []

    with torch.no_grad():
        for idx in range(len(dataset_test)):
            sample = dataset_test[idx]
            key = sample["key"]
            bucket = manifest_buckets.get(key, "unknown")
            batch = dataset_test.collator([sample])
            for batch_key, batch_value in list(batch.items()):
                if isinstance(batch_value, torch.Tensor):
                    batch[batch_key] = batch_value.to(device)

            model_outputs, text_acc, audio_acc, _ = model(**batch)
            _, xa = model._build_parallel_logits(model_outputs, batch["attention_mask"], batch["labels"])
            audio_labels = batch["labels"][:, :code_layer]
            audio_stream = audio_labels[:, 0, :] if audio_labels.dim() == 3 else audio_labels

            sample_result = {
                "key": key,
                "bucket": bucket,
                "target_text": sample["target_text"],
                "teacher_forced_text_acc": None if text_acc == -1 else float(text_acc),
                "teacher_forced_audio_acc": [float(v) for v in audio_acc] if isinstance(audio_acc, list) else audio_acc,
                "streams": [],
            }

            logger.info("[EOA_PROBE] key=%s bucket=%s", key, bucket)

            if use_future_mtp:
                stream_specs = []
                seq_len = audio_stream.shape[1]
                for stream_idx, stream_logits in enumerate(xa):
                    target_shift = stream_idx + 1
                    if seq_len <= target_shift:
                        shift_logits = stream_logits[:, :0, :]
                        shift_labels = audio_stream[:, :0]
                    else:
                        shift_logits = stream_logits[:, :-target_shift, :]
                        shift_labels = audio_stream[:, target_shift:]
                    stream_specs.append((stream_idx, stream_names[stream_idx], target_shift, shift_logits, shift_labels))
            else:
                stream_specs = []
                for layer_idx in range(code_layer):
                    shift_logits = xa[layer_idx][:, :-1, :]
                    shift_labels = audio_labels[:, layer_idx, 1:]
                    stream_specs.append((layer_idx, stream_names[layer_idx], 1, shift_logits, shift_labels))

            for layer_idx, stream_name, target_shift, shift_logits, shift_labels in stream_specs:
                eoa_positions = torch.nonzero(shift_labels.eq(eoa_id), as_tuple=False)
                layer_entries = []

                for pos in eoa_positions:
                    batch_index = int(pos[0].item())
                    time_index = int(pos[1].item())
                    logits_vec = shift_logits[batch_index, time_index]
                    probs_vec = torch.softmax(logits_vec, dim=-1)
                    pred_id = int(torch.argmax(logits_vec).item())
                    rank = int((logits_vec > logits_vec[eoa_id]).sum().item()) + 1
                    topk_ids = torch.topk(logits_vec, k=min(topk, logits_vec.shape[-1])).indices.tolist()
                    top5_ids = torch.topk(logits_vec, k=min(5, logits_vec.shape[-1])).indices.tolist()
                    top10_ids = torch.topk(logits_vec, k=min(10, logits_vec.shape[-1])).indices.tolist()
                    p_eoa = float(probs_vec[eoa_id].item())
                    top1_hit = pred_id == eoa_id
                    top5_hit = eoa_id in top5_ids
                    top10_hit = eoa_id in top10_ids

                    entry = {
                        "position": time_index,
                        "p_eoa": p_eoa,
                        "rank": rank,
                        "pred_id": pred_id,
                        "topk_ids": [int(v) for v in topk_ids],
                        "top1_hit": top1_hit,
                        "top5_hit": top5_hit,
                        "top10_hit": top10_hit,
                    }
                    layer_entries.append(entry)

                    _update_metric(aggregate["all"], layer_idx, p_eoa, rank, top5_hit, top10_hit, top1_hit)
                    _update_metric(aggregate[bucket], layer_idx, p_eoa, rank, top5_hit, top10_hit, top1_hit)

                    logger.info(
                        "[EOA_PROBE] key=%s stream=%s shift=%d pos=%d p_eoa=%.4f rank=%d pred=%d top1=%s top5=%s top10=%s topk=%s",
                        key,
                        stream_name,
                        target_shift,
                        time_index,
                        p_eoa,
                        rank,
                        pred_id,
                        top1_hit,
                        top5_hit,
                        top10_hit,
                        topk_ids,
                    )

                sample_result["streams"].append(
                    {
                        "stream_idx": layer_idx,
                        "name": stream_name,
                        "target_shift": target_shift,
                        "gold_eoa_count": len(layer_entries),
                        "entries": layer_entries,
                    }
                )
            per_sample.append(sample_result)

    summary = {
        "ckpt_path": kwargs.get("ckpt_path"),
        "dataset_path": dataset_config.val_data_path,
        "manifest_path": kwargs.get("manifest_path"),
        "topk": topk,
        "aggregate": {
            bucket_name: _finalize_metric(metric_store)
            for bucket_name, metric_store in aggregate.items()
        },
        "per_sample": per_sample,
    }

    summary_path = os.path.join(probe_dir, "teacher_forced_eoa_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for bucket_name, bucket_summary in summary["aggregate"].items():
        logger.info(
            "[EOA_PROBE][SUMMARY] bucket=%s overall=%s per_stream=%s",
            bucket_name,
            bucket_summary["overall"],
            bucket_summary["per_stream"],
        )

    logger.info("[EOA_PROBE] summary_path=%s", summary_path)


if __name__ == "__main__":
    main_hydra()
