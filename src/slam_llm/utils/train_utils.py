# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
import json
import math
import shutil
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer

from slam_llm.utils.checkpoint_handler import (
    save_model_checkpoint, 
    save_model_and_optimizer_sharded, 
    save_optimizer_checkpoint, 
    save_model_checkpoint_peft,
    save_model_checkpoint_peft_full_shard
)
from slam_llm.policies import fpSixteen,bfSixteen_mixed, get_llama_wrapper
from slam_llm.utils.memory_utils import MemoryTrace

import wandb
import logging
logger = logging.getLogger(__name__)

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _move_batch_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            for k2 in batch[key].keys():
                if isinstance(batch[key][k2], torch.Tensor):
                    batch[key][k2] = batch[key][k2].to(device)
    return batch


def _extract_prefixed_batch(batch, prefix):
    sub = {}
    prefix_key = f"{prefix}_"
    for key, value in batch.items():
        if key.startswith(prefix_key):
            sub[key[len(prefix_key):]] = value
    return sub


def _clone_tensor_batch(batch):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        elif isinstance(value, dict):
            out[key] = {}
            for k2, v2 in value.items():
                out[key][k2] = v2.clone() if isinstance(v2, torch.Tensor) else v2
        else:
            out[key] = value
    return out


def _is_dpo_batch(batch):
    return "chosen_input_ids" in batch and "rejected_input_ids" in batch


def _compute_dpo_step(model, reference_model, batch, train_config):
    if reference_model is None:
        raise ValueError("reference_model is required when use_dpo=true")

    chosen_batch = _extract_prefixed_batch(batch, "chosen")
    rejected_batch = _extract_prefixed_batch(batch, "rejected")
    if "labels" not in chosen_batch or "labels" not in rejected_batch:
        raise ValueError("DPO batch must contain chosen_labels and rejected_labels")

    chosen_outputs, *chosen_rest = model(**chosen_batch)
    rejected_outputs, *_ = model(**rejected_batch)

    model_core = _unwrap_model(model)
    if not hasattr(model_core, "compute_sequence_logp"):
        raise AttributeError("Model does not implement compute_sequence_logp required for DPO")
    chosen_logp = model_core.compute_sequence_logp(chosen_outputs.logits, chosen_batch["labels"])
    rejected_logp = model_core.compute_sequence_logp(rejected_outputs.logits, rejected_batch["labels"])

    with torch.no_grad():
        # Use cloned tensors for ref forward to isolate any in-place ops from policy graph.
        ref_chosen_batch = _clone_tensor_batch(chosen_batch)
        ref_rejected_batch = _clone_tensor_batch(rejected_batch)
        ref_chosen_outputs, *_ = reference_model(**ref_chosen_batch)
        ref_rejected_outputs, *_ = reference_model(**ref_rejected_batch)
        reference_core = _unwrap_model(reference_model)
        if not hasattr(reference_core, "compute_sequence_logp"):
            raise AttributeError("Reference model does not implement compute_sequence_logp required for DPO")
        ref_chosen_logp = reference_core.compute_sequence_logp(ref_chosen_outputs.logits, ref_chosen_batch["labels"])
        ref_rejected_logp = reference_core.compute_sequence_logp(ref_rejected_outputs.logits, ref_rejected_batch["labels"])

    beta = float(train_config.get("dpo_beta", 0.1))
    alpha = float(train_config.get("dpo_alpha", 0.0))
    policy_margin = chosen_logp - rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    dpo_margin = policy_margin - ref_margin
    dpo_loss = (-F.logsigmoid(beta * dpo_margin)).mean()
    sft_loss = chosen_outputs.loss
    total_loss = dpo_loss + alpha * sft_loss
    if _debug_train_sanity_enabled(train_config):
        if not torch.isfinite(dpo_loss).all() or not torch.isfinite(sft_loss).all() or not torch.isfinite(total_loss).all():
            logger.warning(
                "[DPO] [DIAGNOSIS] non-finite detected in _compute_dpo_step: dpo_loss=%s sft_loss=%s total_loss=%s",
                dpo_loss,
                sft_loss,
                total_loss,
            )

    pair_margin = batch.get("pair_score_margin", None)
    if isinstance(pair_margin, torch.Tensor):
        pair_margin_stat = pair_margin.float().mean()
        pair_margin_pos_ratio = pair_margin.float().gt(0).float().mean()
    else:
        pair_margin_stat = None
        pair_margin_pos_ratio = None

    dpo_stats = {
        "dpo_loss": dpo_loss.detach(),
        "sft_loss": sft_loss.detach(),
        "chosen_logp": chosen_logp.detach().mean(),
        "rejected_logp": rejected_logp.detach().mean(),
        "ref_chosen_logp": ref_chosen_logp.detach().mean(),
        "ref_rejected_logp": ref_rejected_logp.detach().mean(),
        "policy_margin": policy_margin.detach().mean(),
        "ref_margin": ref_margin.detach().mean(),
        "dpo_margin": dpo_margin.detach().mean(),
        "policy_pref_rate": policy_margin.detach().gt(0).float().mean(),
        "dpo_pref_rate": dpo_margin.detach().gt(0).float().mean(),
        "pair_margin": pair_margin_stat.detach() if isinstance(pair_margin_stat, torch.Tensor) else None,
        "pair_margin_pos_rate": pair_margin_pos_ratio.detach() if isinstance(pair_margin_pos_ratio, torch.Tensor) else None,
    }
    return total_loss, chosen_outputs, chosen_rest, dpo_stats


def _to_python_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_optional_aux_stats(outputs, batch):
    stats = {}
    total_loss = _to_python_float(getattr(outputs, "loss", None))
    if total_loss is not None:
        stats["total_loss"] = total_loss

    for key in ("base_loss",):
        value = _to_python_float(getattr(outputs, key, None))
        if value is not None:
            stats[key] = value

    stop_metrics = getattr(outputs, "stop_metrics", None)
    if isinstance(stop_metrics, dict):
        for key in (
            "stop_loss",
            "stop_pos_rate",
            "stop_pred_mean",
            "stop_pred_pos_mean",
            "stop_pred_neg_mean",
        ):
            value = _to_python_float(stop_metrics.get(key, None))
            if value is not None:
                stats[key] = value

    return stats


def _format_aux_wandb_logs(aux_stats, prefix="train_inner"):
    logs = {}
    for key in (
        "base_loss",
        "total_loss",
        "stop_loss",
        "stop_pos_rate",
        "stop_pred_mean",
        "stop_pred_pos_mean",
        "stop_pred_neg_mean",
    ):
        if key in aux_stats:
            logs[f"{prefix}/{key}"] = aux_stats[key]
    return logs


def _compute_binary_auc_metrics(probs, targets):
    if probs is None or targets is None:
        return {"pr_auc": None, "roc_auc": None}
    probs = probs.reshape(-1).float()
    targets = targets.reshape(-1).float()
    if probs.numel() == 0 or targets.numel() == 0:
        return {"pr_auc": None, "roc_auc": None}
    pos_count = float(targets.sum().item())
    neg_count = float(targets.numel() - int(round(pos_count)))
    if pos_count <= 0 or neg_count <= 0:
        return {"pr_auc": None, "roc_auc": None}

    order = torch.argsort(probs, descending=True)
    sorted_targets = targets[order]
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)
    tpr = tp / pos_count
    fpr = fp / neg_count
    zero = torch.zeros(1, dtype=probs.dtype, device=probs.device)
    one = torch.ones(1, dtype=probs.dtype, device=probs.device)
    roc_auc = float(torch.trapz(torch.cat([zero, tpr, one]), torch.cat([zero, fpr, one])).item())

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tpr
    recall_prev = torch.cat([zero, recall[:-1]], dim=0)
    pr_auc = float(torch.sum((recall - recall_prev) * precision).item())
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def _compute_binary_threshold_metrics(probs, targets, threshold=0.5):
    if probs is None or targets is None:
        return {"precision": None, "recall": None, "f1": None}
    probs = probs.reshape(-1).float()
    targets = targets.reshape(-1).float()
    if probs.numel() == 0 or targets.numel() == 0:
        return {"precision": None, "recall": None, "f1": None}

    pred_pos = probs >= float(threshold)
    target_pos = targets > 0.5
    tp = int((pred_pos & target_pos).sum().item())
    fp = int((pred_pos & (~target_pos)).sum().item())
    fn = int(((~pred_pos) & target_pos).sum().item())

    precision = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def _str2bool(value):
    value = str(value).strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on"}


def _debug_train_sanity_enabled(train_config):
    return bool(train_config.get("debug_train_sanity", False)) or _str2bool(
        os.environ.get("DEBUG_TRAIN_SANITY", "0")
    )


def _tensor_meta(x):
    if not isinstance(x, torch.Tensor):
        return {"type": str(type(x))}
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    }


def _labels_valid_stats(labels):
    if not isinstance(labels, torch.Tensor):
        return {"exists": False}
    total = int(labels.numel())
    valid = int(labels.ne(-100).sum().item())
    stats = {
        "exists": True,
        "shape": list(labels.shape),
        "dim": int(labels.dim()),
        "valid_tokens": valid,
        "total_tokens": total,
        "valid_ratio": (float(valid) / float(total)) if total > 0 else 0.0,
    }
    if labels.dim() == 3:
        stream_num = labels.shape[1]
        if stream_num > 0:
            text_idx = stream_num - 1
            text_valid = int(labels[:, text_idx, :].ne(-100).sum().item())
            text_total = int(labels[:, text_idx, :].numel())
            stats["text_valid_tokens"] = text_valid
            stats["text_total_tokens"] = text_total
            stats["text_valid_ratio"] = (float(text_valid) / float(text_total)) if text_total > 0 else 0.0
            audio_stats = []
            for i in range(max(0, stream_num - 1)):
                audio_valid = int(labels[:, i, :].ne(-100).sum().item())
                audio_total = int(labels[:, i, :].numel())
                audio_stats.append(
                    {
                        "layer": i,
                        "valid_tokens": audio_valid,
                        "total_tokens": audio_total,
                        "valid_ratio": (float(audio_valid) / float(audio_total)) if audio_total > 0 else 0.0,
                    }
                )
            stats["audio_layers"] = audio_stats
    return stats


def _log_batch_sanity(batch, step, use_dpo, max_pairs=3):
    keys = list(batch.keys())
    logger.info(f"[BATCH] [SANITY] step={step} keys={keys}")
    for key in keys[:40]:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            m = _tensor_meta(value)
            logger.info(
                "[BATCH] [SANITY] step=%d key=%s shape=%s dtype=%s device=%s",
                step,
                key,
                m["shape"],
                m["dtype"],
                m["device"],
            )
        elif isinstance(value, list):
            logger.info("[BATCH] [SANITY] step=%d key=%s list_len=%d", step, key, len(value))
        elif value is None:
            logger.info("[BATCH] [SANITY] step=%d key=%s is None", step, key)

    if use_dpo:
        chosen_stats = _labels_valid_stats(batch.get("chosen_labels", None))
        rejected_stats = _labels_valid_stats(batch.get("rejected_labels", None))
        logger.info(f"[BATCH] [SANITY] step={step} chosen_label_stats={json.dumps(chosen_stats, ensure_ascii=False)}")
        logger.info(f"[BATCH] [SANITY] step={step} rejected_label_stats={json.dumps(rejected_stats, ensure_ascii=False)}")
        pair_margin = batch.get("pair_score_margin", None)
        pair_keys = batch.get("pair_keys", None)
        if isinstance(pair_margin, torch.Tensor):
            pos = int(pair_margin.gt(0).sum().item())
            non_pos = int(pair_margin.le(0).sum().item())
            logger.info(
                "[DPO] [BATCH] step=%d pair_margin mean=%.6f min=%.6f max=%.6f pos=%d non_pos=%d",
                step,
                float(pair_margin.float().mean().item()),
                float(pair_margin.float().min().item()),
                float(pair_margin.float().max().item()),
                pos,
                non_pos,
            )
            if non_pos > 0:
                logger.warning("[DPO] [BATCH] step=%d detected non-positive pair_score_margin entries.", step)
        if isinstance(pair_keys, list) and len(pair_keys) > 0:
            chosen_labels = batch.get("chosen_labels", None)
            rejected_labels = batch.get("rejected_labels", None)
            sample_n = min(max_pairs, len(pair_keys))
            for i in range(sample_n):
                ch_valid = int(chosen_labels[i].ne(-100).sum().item()) if isinstance(chosen_labels, torch.Tensor) else -1
                rej_valid = int(rejected_labels[i].ne(-100).sum().item()) if isinstance(rejected_labels, torch.Tensor) else -1
                margin_i = None
                if isinstance(pair_margin, torch.Tensor):
                    margin_i = float(pair_margin[i].item())
                logger.info(
                    "[DPO] [BATCH] step=%d sample=%d key=%s pair_margin=%s chosen_valid=%d rejected_valid=%d",
                    step,
                    i,
                    pair_keys[i],
                    margin_i,
                    ch_valid,
                    rej_valid,
                )
    else:
        label_stats = _labels_valid_stats(batch.get("labels", None))
        logger.info(f"[BATCH] [SANITY] step={step} label_stats={json.dumps(label_stats, ensure_ascii=False)}")


def _gather_param_diagnostics(model, optimizer):
    model_core = _unwrap_model(model)
    all_named_params = list(model_core.named_parameters())
    total_params = sum(p.numel() for _, p in all_named_params)
    trainable_named_params = [(n, p) for n, p in all_named_params if p.requires_grad]
    trainable_params = sum(p.numel() for _, p in trainable_named_params)

    module_trainable = {}
    for name, p in trainable_named_params:
        module_name = name.rsplit(".", 1)[0] if "." in name else name
        module_trainable[module_name] = module_trainable.get(module_name, 0) + int(p.numel())
    module_trainable_sorted = sorted(module_trainable.items(), key=lambda x: x[1], reverse=True)

    optim_named = []
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            if p is None:
                continue
            optimizer_param_ids.add(id(p))
            optim_named.append(p)
    optimizer_params = sum(p.numel() for p in optim_named)
    trainable_param_ids = {id(p) for _, p in trainable_named_params}
    trainable_not_in_optimizer = len(trainable_param_ids - optimizer_param_ids)
    optimizer_not_trainable = len(optimizer_param_ids - trainable_param_ids)

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_fraction": (float(trainable_params) / float(total_params)) if total_params > 0 else 0.0,
        "trainable_named_params": trainable_named_params,
        "module_trainable_sorted": module_trainable_sorted,
        "optimizer_params": int(optimizer_params),
        "trainable_not_in_optimizer": int(trainable_not_in_optimizer),
        "optimizer_not_trainable": int(optimizer_not_trainable),
    }


def _select_tracked_params(trainable_named_params, track_n):
    if track_n <= 0 or len(trainable_named_params) == 0:
        return []
    priority_keywords = [
        "llm.lm_head",
        "group_decode_adapter",
        "embed_tokens",
        "layers.0",
    ]
    selected = []
    used_names = set()
    for kw in priority_keywords:
        for name, p in trainable_named_params:
            if kw in name and name not in used_names:
                selected.append((name, p))
                used_names.add(name)
                break
        if len(selected) >= track_n:
            return selected
    for name, p in trainable_named_params:
        if name in used_names:
            continue
        selected.append((name, p))
        used_names.add(name)
        if len(selected) >= track_n:
            break
    return selected


def _compute_grad_norm(optimizer):
    grad_sq_sum = 0.0
    grad_param_count = 0
    nonzero_grad_count = 0
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            if p is None or p.grad is None:
                continue
            g = p.grad.detach().float()
            norm_val = float(torch.norm(g).item())
            grad_sq_sum += norm_val * norm_val
            grad_param_count += 1
            if norm_val > 0:
                nonzero_grad_count += 1
    total_grad_norm = grad_sq_sum ** 0.5
    return total_grad_norm, grad_param_count, nonzero_grad_count


def _estimate_checkpoint_path(train_config, fsdp_config, checkpoint_name, epoch):
    try:
        if train_config.enable_fsdp:
            if train_config.use_peft or train_config.freeze_llm:
                shard_strategy = getattr(fsdp_config, "sharding_strategy", None) if fsdp_config is not None else None
                if str(shard_strategy) == str(ShardingStrategy.FULL_SHARD):
                    return os.path.join(train_config.output_dir, train_config.model_name, str(epoch + 1), "model.pt")
                return os.path.join(train_config.output_dir, checkpoint_name, "model.pt")
            checkpoint_type = getattr(fsdp_config, "checkpoint_type", None) if fsdp_config is not None else None
            if checkpoint_type == "FULL_STATE_DICT":
                folder_name = (
                    train_config.dist_checkpoint_root_folder
                    + "/"
                    + train_config.dist_checkpoint_folder
                    + "-"
                    + train_config.model_name
                )
                return str(Path.cwd() / folder_name / f"{train_config.model_name}-{epoch}.pt")
            folder_name = (
                train_config.dist_checkpoint_root_folder
                + "/"
                + train_config.dist_checkpoint_folder
                + "-"
                + train_config.model_name
            )
            return str(Path.cwd() / folder_name)
        return os.path.join(train_config.output_dir, checkpoint_name, "model.pt")
    except Exception:
        return None


def _normalize_checkpoint_monitor_mode(mode):
    mode = str(mode if mode is not None else "min").strip().lower()
    if mode in {"min", "lower", "smaller"}:
        return "min"
    if mode in {"max", "higher", "larger"}:
        return "max"
    logger.warning(f"[CKPT] unknown checkpoint_monitor_mode={mode}, fallback to 'min'.")
    return "min"


def _coerce_metric_value(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = float(value.detach().float().mean().item())
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(value):
        return None
    return value


def _get_checkpoint_monitor_value(monitor_name, eval_epoch_loss, eval_epoch_acc, eval_epoch_audio_acc, eval_extra):
    normalized_name = str(monitor_name if monitor_name is not None else "val_loss").strip().lower()
    alias_map = {
        "loss": "val_loss",
        "acc": "val_acc",
        "accuracy": "val_acc",
        "audio_acc": "val_audio_acc",
        "audio_accuracy": "val_audio_acc",
    }
    normalized_name = alias_map.get(normalized_name, normalized_name)
    base_metrics = {
        "val_loss": eval_epoch_loss,
        "val_acc": eval_epoch_acc,
        "val_audio_acc": eval_epoch_audio_acc,
    }
    if normalized_name in base_metrics:
        return _coerce_metric_value(base_metrics[normalized_name])
    if normalized_name.startswith("val_"):
        eval_key = f"eval_{normalized_name[4:]}"
        if isinstance(eval_extra, dict) and eval_key in eval_extra:
            return _coerce_metric_value(eval_extra.get(eval_key))
    if isinstance(eval_extra, dict) and normalized_name in eval_extra:
        return _coerce_metric_value(eval_extra.get(normalized_name))
    return None


def _is_better_checkpoint_metric(candidate_value, reference_value, mode):
    candidate_value = _coerce_metric_value(candidate_value)
    reference_value = _coerce_metric_value(reference_value)
    if candidate_value is None:
        return False
    if reference_value is None:
        return True
    if mode == "max":
        return candidate_value > reference_value
    return candidate_value < reference_value


def _sort_checkpoint_records(records, mode):
    reverse = (mode == "max")
    return sorted(
        records,
        key=lambda item: item.get("monitor_value", float("-inf") if reverse else float("inf")),
        reverse=reverse,
    )


def _checkpoint_cleanup_target(ckpt_path):
    if ckpt_path in (None, "", "None"):
        return None
    path = Path(str(ckpt_path))
    if path.name == "model.pt":
        return path.parent
    return path


def _delete_checkpoint_artifact(ckpt_path):
    target = _checkpoint_cleanup_target(ckpt_path)
    if target is None or not target.exists():
        return False
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return True


def _write_checkpoint_topk_summary(train_config, checkpoint_records, monitor_name, monitor_mode, top_k):
    output_dir = train_config.get("output_dir", None)
    if output_dir in (None, "", "None"):
        return None
    try:
        out_path = Path(str(output_dir)) / "checkpoint_topk.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "monitor": monitor_name,
            "mode": monitor_mode,
            "top_k": int(top_k),
            "count": len(checkpoint_records),
            "checkpoints": checkpoint_records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(out_path.resolve())
    except Exception as exc:
        logger.warning(f"[CKPT] failed to write checkpoint_topk summary: {exc}")
        return None


def _build_diagnosis_summary(diag_state):
    suspects = []
    severity_rank = {"confirmed": 0, "highly suspicious": 1, "possible but unconfirmed": 2}

    def add_suspect(level, key, reason, evidence):
        suspects.append(
            {
                "level": level,
                "key": key,
                "reason": reason,
                "evidence": evidence,
            }
        )

    if diag_state.get("trainable_params", 0) == 0:
        add_suspect(
            "confirmed",
            "no_trainable_params",
            "No trainable parameters detected.",
            {"trainable_params": diag_state.get("trainable_params", 0)},
        )
    elif diag_state.get("trainable_fraction", 0.0) < 1e-4:
        add_suspect(
            "highly suspicious",
            "too_few_trainable_params",
            "Trainable parameter fraction is extremely small.",
            {
                "trainable_fraction": diag_state.get("trainable_fraction", 0.0),
                "trainable_params": diag_state.get("trainable_params", 0),
            },
        )

    if diag_state.get("optimizer_params", 0) == 0:
        add_suspect(
            "confirmed",
            "optimizer_empty",
            "Optimizer has zero parameters.",
            {"optimizer_params": diag_state.get("optimizer_params", 0)},
        )

    if diag_state.get("trainable_not_in_optimizer", 0) > 0:
        add_suspect(
            "highly suspicious",
            "trainable_missing_in_optimizer",
            "Some trainable parameters are missing from optimizer param groups.",
            {"count": diag_state.get("trainable_not_in_optimizer", 0)},
        )

    if diag_state.get("nonfinite_loss_steps", 0) > 0:
        add_suspect(
            "confirmed",
            "nonfinite_loss",
            "NaN/Inf detected in losses or margins.",
            {"nonfinite_steps": diag_state.get("nonfinite_loss_steps", 0)},
        )

    if diag_state.get("low_valid_label_steps", 0) > 0:
        add_suspect(
            "highly suspicious",
            "invalid_or_sparse_labels",
            "Low valid-label ratio detected in one or more steps.",
            {"low_valid_label_steps": diag_state.get("low_valid_label_steps", 0)},
        )

    if diag_state.get("grad_observed_steps", 0) > 0:
        zero_grad_ratio = float(diag_state.get("zero_grad_steps", 0)) / float(diag_state["grad_observed_steps"])
        if zero_grad_ratio > 0.7:
            add_suspect(
                "highly suspicious",
                "gradient_vanishing_or_disconnected",
                "Most observed steps have near-zero gradient norm.",
                {"zero_grad_ratio": zero_grad_ratio},
            )

    if diag_state.get("param_update_observed_steps", 0) > 0:
        tiny_update_ratio = float(diag_state.get("tiny_param_update_steps", 0)) / float(diag_state["param_update_observed_steps"])
        if tiny_update_ratio > 0.7:
            add_suspect(
                "highly suspicious",
                "parameters_not_updating",
                "Tracked trainable parameters barely changed in most observed optimizer steps.",
                {"tiny_update_ratio": tiny_update_ratio},
            )

    if diag_state.get("use_dpo", False):
        if diag_state.get("dpo_observed_steps", 0) > 0:
            policy_non_pos_ratio = float(diag_state.get("policy_margin_nonpos_steps", 0)) / float(diag_state["dpo_observed_steps"])
            dpo_non_pos_ratio = float(diag_state.get("dpo_margin_nonpos_steps", 0)) / float(diag_state["dpo_observed_steps"])
            if policy_non_pos_ratio > 0.7:
                add_suspect(
                    "highly suspicious",
                    "policy_not_preferring_chosen",
                    "Policy margin (chosen - rejected) is non-positive in most DPO steps.",
                    {"policy_margin_nonpositive_ratio": policy_non_pos_ratio},
                )
            if dpo_non_pos_ratio > 0.7:
                add_suspect(
                    "highly suspicious",
                    "dpo_signal_not_beating_reference",
                    "DPO margin is non-positive in most DPO steps.",
                    {"dpo_margin_nonpositive_ratio": dpo_non_pos_ratio},
                )

        if diag_state.get("pair_margin_observed_steps", 0) > 0:
            pair_non_pos_ratio = float(diag_state.get("pair_margin_nonpos_steps", 0)) / float(diag_state["pair_margin_observed_steps"])
            if pair_non_pos_ratio > 0.2:
                add_suspect(
                    "possible but unconfirmed",
                    "pair_direction_issue",
                    "A notable fraction of pair_score_margin is non-positive.",
                    {"pair_margin_nonpositive_ratio": pair_non_pos_ratio},
                )
        add_suspect(
            "possible but unconfirmed",
            "acc_target_mismatch",
            "acc comes from text/audio token accuracy, while optimization target is DPO(+SFT) loss.",
            {"note": "acc may not reflect the main optimization target"},
        )

    if diag_state.get("inference_ckpt_mismatch", False):
        add_suspect(
            "confirmed",
            "best_ckpt_not_used_for_inference",
            "Loaded inference checkpoint path differs from recorded best checkpoint path.",
            {
                "best_ckpt_path": diag_state.get("best_ckpt_path"),
                "inference_ckpt_path": diag_state.get("inference_ckpt_path"),
            },
        )

    suspects = sorted(suspects, key=lambda x: severity_rank.get(x["level"], 99))
    return {
        "suspects": suspects,
        "has_confirmed_issue": any(x["level"] == "confirmed" for x in suspects),
        "has_highly_suspicious_issue": any(x["level"] == "highly suspicious" for x in suspects),
    }


def _write_diagnosis_summary(train_config, payload):
    output_dir = train_config.get("output_dir", None)
    if output_dir in (None, "", "None"):
        return None
    try:
        out_path = Path(str(output_dir)) / "sanity_diagnosis_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(out_path.resolve())
    except Exception as exc:
        logger.warning(f"[DIAGNOSIS] failed to write sanity summary file: {exc}")
        return None

def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, log_config, fsdp_config=None, local_rank=None, rank=None, reference_model=None, run_context=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        log_config: The logging configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    if train_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        if train_config.enable_fsdp:
            scaler = ShardedGradScaler()
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    use_dpo = bool(train_config.get("use_dpo", False))
    if use_dpo and reference_model is None:
        raise ValueError("use_dpo=true requires a loaded reference_model")
    run_context = run_context or {}
    debug_train_sanity = _debug_train_sanity_enabled(train_config)
    debug_interval = max(1, int(train_config.get("debug_sanity_interval", 100)))
    debug_first_steps = max(0, int(train_config.get("debug_sanity_first_steps", 5)))
    debug_track_params = max(0, int(train_config.get("debug_sanity_track_params", 5)))
    tiny_overfit_mode = bool(train_config.get("tiny_overfit_mode", False))
    tiny_overfit_max_steps = max(0, int(train_config.get("tiny_overfit_max_steps", 0)))
    if debug_train_sanity:
        logger.warning(
            "[SANITY] enabled (interval=%d first_steps=%d track_params=%d tiny_overfit_mode=%s tiny_overfit_max_steps=%d)",
            debug_interval,
            debug_first_steps,
            debug_track_params,
            tiny_overfit_mode,
            tiny_overfit_max_steps,
        )
        if use_dpo:
            logger.warning(
                "[DIAGNOSIS] acc may not reflect the main optimization target under DPO; optimize target is total_loss=dpo_loss + alpha*sft_loss."
            )
            logger.warning("[DIAGNOSIS] acc source: rest[0]=text_acc, rest[1]=audio_acc from policy forward.")

    param_diag = _gather_param_diagnostics(model, optimizer)
    if debug_train_sanity:
        logger.info(
            "[GRAD] [SANITY] total_params=%d trainable_params=%d trainable_fraction=%.6f optimizer_params=%d",
            param_diag["total_params"],
            param_diag["trainable_params"],
            param_diag["trainable_fraction"],
            param_diag["optimizer_params"],
        )
        logger.info(
            "[GRAD] [SANITY] trainable_not_in_optimizer=%d optimizer_not_trainable=%d",
            param_diag["trainable_not_in_optimizer"],
            param_diag["optimizer_not_trainable"],
        )
        for module_name, module_params in param_diag["module_trainable_sorted"][:40]:
            logger.info(
                "[GRAD] [SANITY] trainable_module=%s params=%d",
                module_name,
                module_params,
            )
        if param_diag["trainable_params"] <= 0:
            logger.warning("[GRAD] [DIAGNOSIS] trainable parameters are zero.")
        if param_diag["optimizer_params"] <= 0:
            logger.warning("[GRAD] [DIAGNOSIS] optimizer parameter count is zero.")
        if param_diag["trainable_not_in_optimizer"] > 0:
            logger.warning("[GRAD] [DIAGNOSIS] some trainable params are missing from optimizer.")

    tracked_params = _select_tracked_params(param_diag["trainable_named_params"], debug_track_params)
    tracked_prev_values = {
        name: p.detach().float().clone()
        for name, p in tracked_params
    }

    diag_state = {
        "use_dpo": use_dpo,
        "trainable_params": param_diag["trainable_params"],
        "trainable_fraction": param_diag["trainable_fraction"],
        "optimizer_params": param_diag["optimizer_params"],
        "trainable_not_in_optimizer": param_diag["trainable_not_in_optimizer"],
        "nonfinite_loss_steps": 0,
        "low_valid_label_steps": 0,
        "grad_observed_steps": 0,
        "zero_grad_steps": 0,
        "param_update_observed_steps": 0,
        "tiny_param_update_steps": 0,
        "dpo_observed_steps": 0,
        "policy_margin_nonpos_steps": 0,
        "dpo_margin_nonpos_steps": 0,
        "pair_margin_observed_steps": 0,
        "pair_margin_nonpos_steps": 0,
        "best_ckpt_path": None,
        "inference_ckpt_path": run_context.get("inference_ckpt_path", None),
        "inference_ckpt_mismatch": False,
    }
    train_prep = []
    train_loss = []
    train_acc = []
    train_audio_acc = []
    train_dpo_loss = []
    train_sft_loss = []
    train_chosen_logp = []
    train_rejected_logp = []
    train_ref_chosen_logp = []
    train_ref_rejected_logp = []
    train_policy_margin = []
    train_ref_margin = []
    train_dpo_margin = []
    train_policy_pref_rate = []
    train_dpo_pref_rate = []
    train_pair_margin = []
    train_pair_margin_pos_rate = []
    train_base_loss = []
    train_total_loss = []
    train_stop_loss = []
    train_stop_pos_rate = []
    train_stop_pred_mean = []
    train_stop_pred_pos_mean = []
    train_stop_pred_neg_mean = []
    val_prep = []
    val_loss =[]
    val_acc = []
    val_audio_acc = []
    val_dpo_loss = []
    val_sft_loss = []
    val_chosen_logp = []
    val_rejected_logp = []
    val_ref_chosen_logp = []
    val_ref_rejected_logp = []
    val_policy_margin = []
    val_ref_margin = []
    val_dpo_margin = []
    val_policy_pref_rate = []
    val_dpo_pref_rate = []
    val_pair_margin = []
    val_pair_margin_pos_rate = []
    val_stop_loss = []
    val_stop_pr_auc = []
    val_stop_roc_auc = []
    val_stop_precision_at_0_5 = []
    val_stop_recall_at_0_5 = []
    val_stop_f1_at_0_5 = []
    val_stop_pred_pos_mean = []
    val_stop_pred_neg_mean = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_audio_acc =0.0
    checkpoint_top_k = int(train_config.get("checkpoint_top_k", -1))
    checkpoint_monitor = str(train_config.get("checkpoint_monitor", "val_loss"))
    checkpoint_monitor_mode = _normalize_checkpoint_monitor_mode(train_config.get("checkpoint_monitor_mode", "min"))
    retained_checkpoint_records = []
    best_ckpt_path = None
    best_ckpt_epoch = None
    best_ckpt_step = None
    global_step = 0
    stop_after_epoch = False
    if train_config.save_model:
        if checkpoint_top_k > 0:
            logger.info(
                "[CKPT] top-k checkpointing enabled: top_k=%d monitor=%s mode=%s",
                checkpoint_top_k,
                checkpoint_monitor,
                checkpoint_monitor_mode,
            )
        else:
            logger.info("[CKPT] legacy checkpointing enabled: save every validation interval.")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            if use_dpo and reference_model is not None:
                reference_model.eval()
            total_loss = 0.0
            total_acc = 0.0
            total_audio_acc = 0.0
            total_dpo_loss = 0.0
            total_sft_loss = 0.0
            total_chosen_logp = 0.0
            total_rejected_logp = 0.0
            total_ref_chosen_logp = 0.0
            total_ref_rejected_logp = 0.0
            total_policy_margin = 0.0
            total_ref_margin = 0.0
            total_dpo_margin = 0.0
            total_policy_pref_rate = 0.0
            total_dpo_pref_rate = 0.0
            total_pair_margin = 0.0
            total_pair_margin_pos_rate = 0.0
            total_pair_margin_count = 0
            total_base_loss = 0.0
            total_total_loss = 0.0
            total_base_loss_count = 0
            total_total_loss_count = 0
            total_stop_loss = 0.0
            total_stop_loss_count = 0
            total_stop_pos_rate = 0.0
            total_stop_pos_rate_count = 0
            total_stop_pred_mean = 0.0
            total_stop_pred_mean_count = 0
            total_stop_pred_pos_mean = 0.0
            total_stop_pred_pos_mean_count = 0
            total_stop_pred_neg_mean = 0.0
            total_stop_pred_neg_mean_count = 0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                global_step += 1
                step_device = local_rank if (train_config.enable_fsdp or train_config.enable_ddp) else 'cuda:0'
                batch = _move_batch_to_device(batch, step_device)
                should_log_sanity_step = debug_train_sanity and (step < debug_first_steps or (step % debug_interval == 0))
                if should_log_sanity_step:
                    _log_batch_sanity(batch, step, use_dpo)

                if use_dpo:
                    chosen_label_stats = _labels_valid_stats(batch.get("chosen_labels", None))
                    rejected_label_stats = _labels_valid_stats(batch.get("rejected_labels", None))
                    min_valid_ratio = min(
                        chosen_label_stats.get("valid_ratio", 0.0),
                        rejected_label_stats.get("valid_ratio", 0.0),
                    )
                    if min_valid_ratio < 1e-2:
                        diag_state["low_valid_label_steps"] += 1
                        if should_log_sanity_step:
                            logger.warning(
                                "[BATCH] [DIAGNOSIS] step=%d extremely low valid label ratio in DPO batch (chosen=%.6f rejected=%.6f).",
                                step,
                                chosen_label_stats.get("valid_ratio", 0.0),
                                rejected_label_stats.get("valid_ratio", 0.0),
                            )
                else:
                    label_stats = _labels_valid_stats(batch.get("labels", None))
                    if label_stats.get("valid_ratio", 0.0) < 1e-2:
                        diag_state["low_valid_label_steps"] += 1
                        if should_log_sanity_step:
                            logger.warning(
                                "[BATCH] [DIAGNOSIS] step=%d extremely low valid label ratio (%.6f).",
                                step,
                                label_stats.get("valid_ratio", 0.0),
                            )

                dpo_stats = None
                with autocast():
                    if use_dpo:
                        if not _is_dpo_batch(batch):
                            raise ValueError("use_dpo=true expects DPO batch fields (chosen_*/rejected_*)")
                        loss, outputs, rest, dpo_stats = _compute_dpo_step(
                            model=model,
                            reference_model=reference_model,
                            batch=batch,
                            train_config=train_config,
                        )
                    else:
                        outputs, *rest = model(**batch)
                        loss = outputs.loss
                if isinstance(loss, torch.Tensor) and (not torch.isfinite(loss).all()):
                    diag_state["nonfinite_loss_steps"] += 1
                    logger.warning(f"[DPO] [DIAGNOSIS] step={step} non-finite loss detected: {loss}")
                if dpo_stats is not None:
                    diag_state["dpo_observed_steps"] += 1
                    try:
                        policy_margin_val = float(dpo_stats["policy_margin"].item())
                        ref_margin_val = float(dpo_stats["ref_margin"].item())
                        dpo_margin_val = float(dpo_stats["dpo_margin"].item())
                    except Exception:
                        policy_margin_val = 0.0
                        ref_margin_val = 0.0
                        dpo_margin_val = 0.0
                    if policy_margin_val <= 0:
                        diag_state["policy_margin_nonpos_steps"] += 1
                    if dpo_margin_val <= 0:
                        diag_state["dpo_margin_nonpos_steps"] += 1
                    if dpo_stats.get("pair_margin", None) is not None:
                        diag_state["pair_margin_observed_steps"] += 1
                        pair_margin_val = float(dpo_stats["pair_margin"].item())
                        if pair_margin_val <= 0:
                            diag_state["pair_margin_nonpos_steps"] += 1
                    if should_log_sanity_step:
                        logger.info(
                            "[DPO] step=%d total_loss=%.6f dpo_loss=%.6f sft_loss=%.6f chosen_logp=%.6f rejected_logp=%.6f ref_chosen_logp=%.6f ref_rejected_logp=%.6f policy_margin=%.6f ref_margin=%.6f dpo_margin=%.6f acc=%.6f",
                            step,
                            float(loss.detach().float().item()),
                            float(dpo_stats["dpo_loss"].detach().float().item()),
                            float(dpo_stats["sft_loss"].detach().float().item()),
                            float(dpo_stats["chosen_logp"].detach().float().item()),
                            float(dpo_stats["rejected_logp"].detach().float().item()),
                            float(dpo_stats["ref_chosen_logp"].detach().float().item()),
                            float(dpo_stats["ref_rejected_logp"].detach().float().item()),
                            policy_margin_val,
                            ref_margin_val,
                            dpo_margin_val,
                            float(rest[0].detach().float().item()) if isinstance(rest[0], torch.Tensor) else float(rest[0]) if rest else -1.0,
                        )
                        if policy_margin_val <= 0:
                            logger.warning("[DPO] [DIAGNOSIS] step=%d policy_margin<=0; policy is not preferring chosen.", step)
                        if dpo_margin_val <= 0:
                            logger.warning("[DPO] [DIAGNOSIS] step=%d dpo_margin<=0; policy not outperforming reference on preference gap.", step)
                    if any([
                        not torch.isfinite(dpo_stats["dpo_loss"]).all(),
                        not torch.isfinite(dpo_stats["sft_loss"]).all(),
                        not torch.isfinite(dpo_stats["chosen_logp"]).all(),
                        not torch.isfinite(dpo_stats["rejected_logp"]).all(),
                        not torch.isfinite(dpo_stats["ref_chosen_logp"]).all(),
                        not torch.isfinite(dpo_stats["ref_rejected_logp"]).all(),
                    ]):
                        diag_state["nonfinite_loss_steps"] += 1
                        logger.warning(f"[DPO] [DIAGNOSIS] step={step} non-finite values detected in dpo_stats.")

                aux_train_stats = _extract_optional_aux_stats(outputs, batch)
                # original logic retained: main optimizer backward still uses `loss` below.
                aux_train_stats["total_loss"] = float(loss.detach().float().item())

                acc = rest[0] if rest else -1 # text_acc
                audio_acc = rest[1] if rest else -1   # audio acc
                if train_config.modeling_paradigm == "parallel" or train_config.modeling_paradigm == "serial":
                    layer_loss = rest[2] if rest else -1
                else:
                    layer_loss = [0]

                loss = loss / gradient_accumulation_steps
                layer_loss = [l / gradient_accumulation_steps for l in layer_loss]
                acc = acc / gradient_accumulation_steps
                audio_acc = [acc / gradient_accumulation_steps for acc in audio_acc]

                if log_config.use_wandb and step % log_config.log_interval == 0:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank==0:
                            wandb.log({"train_inner/train_inner_loss":loss, "train_inner/train_inner_text_accuracy":acc}, step=(epoch * total_length + step))
                            for layer, acc in enumerate(audio_acc):
                                wandb.log({f"train_inner/train_inner_audio_accuracy_layer{layer}":acc}, step=(epoch * total_length + step))
                            for layer, l in enumerate(layer_loss[:-1]):
                                wandb.log({f"train_inner/train_inner_audio_loss_layer{layer}":l}, step=(epoch * total_length + step))
                            wandb.log({f"train_inner/train_inner_text_loss":layer_loss[-1]}, step=(epoch * total_length + step))
                            if dpo_stats is not None:
                                dpo_log = {
                                    "train_inner/dpo_loss": dpo_stats["dpo_loss"],
                                    "train_inner/sft_loss": dpo_stats["sft_loss"],
                                    "train_inner/chosen_logp": dpo_stats["chosen_logp"],
                                    "train_inner/rejected_logp": dpo_stats["rejected_logp"],
                                    "train_inner/ref_chosen_logp": dpo_stats["ref_chosen_logp"],
                                    "train_inner/ref_rejected_logp": dpo_stats["ref_rejected_logp"],
                                    "train_inner/policy_margin": dpo_stats["policy_margin"],
                                    "train_inner/ref_margin": dpo_stats["ref_margin"],
                                    "train_inner/dpo_margin": dpo_stats["dpo_margin"],
                                    "train_inner/policy_pref_rate": dpo_stats["policy_pref_rate"],
                                    "train_inner/dpo_pref_rate": dpo_stats["dpo_pref_rate"],
                                }
                                if dpo_stats["pair_margin"] is not None:
                                    dpo_log["train_inner/pair_margin"] = dpo_stats["pair_margin"]
                                if dpo_stats.get("pair_margin_pos_rate", None) is not None:
                                    dpo_log["train_inner/pair_margin_pos_rate"] = dpo_stats["pair_margin_pos_rate"]
                                wandb.log(dpo_log, step=(epoch * total_length + step))
                    else:
                        wandb.log({"train_inner/train_inner_loss":loss, "train_inner/train_inner_text_accuracy":acc}, step=(epoch * total_length + step))
                        for layer, acc in enumerate(audio_acc):
                            wandb.log({f"train_inner/train_inner_audio_accuracy_layer{layer}":acc}, step=(epoch * total_length + step))
                        for layer, l in enumerate(layer_loss[:-1]):
                            wandb.log({f"train_inner/train_inner_audio_loss_layer{layer}":l}, step=(epoch * total_length + step))
                        wandb.log({f"train_inner/train_inner_text_loss":layer_loss[-1]}, step=(epoch * total_length + step))
                        if dpo_stats is not None:
                            dpo_log = {
                                "train_inner/dpo_loss": dpo_stats["dpo_loss"],
                                "train_inner/sft_loss": dpo_stats["sft_loss"],
                                "train_inner/chosen_logp": dpo_stats["chosen_logp"],
                                "train_inner/rejected_logp": dpo_stats["rejected_logp"],
                                "train_inner/ref_chosen_logp": dpo_stats["ref_chosen_logp"],
                                "train_inner/ref_rejected_logp": dpo_stats["ref_rejected_logp"],
                                "train_inner/policy_margin": dpo_stats["policy_margin"],
                                "train_inner/ref_margin": dpo_stats["ref_margin"],
                                "train_inner/dpo_margin": dpo_stats["dpo_margin"],
                                "train_inner/policy_pref_rate": dpo_stats["policy_pref_rate"],
                                "train_inner/dpo_pref_rate": dpo_stats["dpo_pref_rate"],
                            }
                            if dpo_stats["pair_margin"] is not None:
                                dpo_log["train_inner/pair_margin"] = dpo_stats["pair_margin"]
                            if dpo_stats.get("pair_margin_pos_rate", None) is not None:
                                dpo_log["train_inner/pair_margin_pos_rate"] = dpo_stats["pair_margin_pos_rate"]
                            wandb.log(dpo_log, step=(epoch * total_length + step))
                should_log_step = (step % log_config.log_interval == 0)
                is_main_process = (not (train_config.enable_fsdp or train_config.enable_ddp)) or (rank == 0)
                if should_log_step and is_main_process:
                    aux_wandb_log = _format_aux_wandb_logs(aux_train_stats, prefix="train_inner")
                    if log_config.use_wandb and len(aux_wandb_log) > 0:
                        wandb.log(aux_wandb_log, step=(epoch * total_length + step))
                    if len(aux_wandb_log) > 0:
                        aux_log_str = ", ".join(
                            [f"{k.split('/')[-1]}={v:.6f}" for k, v in aux_wandb_log.items()]
                        )
                        logger.info(f"[train_aux] epoch={epoch+1} step={step} {aux_log_str}")
                    stop_log_parts = []
                    for stop_key in (
                        "stop_loss",
                        "stop_pos_rate",
                        "stop_pred_mean",
                        "stop_pred_pos_mean",
                        "stop_pred_neg_mean",
                    ):
                        if stop_key in aux_train_stats:
                            stop_log_parts.append(f"{stop_key}={aux_train_stats[stop_key]:.6f}")
                    if len(stop_log_parts) > 0:
                        logger.info(f"[STOP][TRAIN] epoch={epoch+1} step={step} " + ", ".join(stop_log_parts))
                    if dpo_stats is not None:
                        dpo_log_str = (
                            f"total_loss={float(loss.detach().float().item()):.6f}, "
                            f"dpo_loss={float(dpo_stats['dpo_loss'].detach().float().item()):.6f}, "
                            f"sft_loss={float(dpo_stats['sft_loss'].detach().float().item()):.6f}, "
                            f"chosen_logp={float(dpo_stats['chosen_logp'].detach().float().item()):.6f}, "
                            f"rejected_logp={float(dpo_stats['rejected_logp'].detach().float().item()):.6f}, "
                            f"ref_chosen_logp={float(dpo_stats['ref_chosen_logp'].detach().float().item()):.6f}, "
                            f"ref_rejected_logp={float(dpo_stats['ref_rejected_logp'].detach().float().item()):.6f}, "
                            f"policy_margin={float(dpo_stats['policy_margin'].detach().float().item()):.6f}, "
                            f"ref_margin={float(dpo_stats['ref_margin'].detach().float().item()):.6f}, "
                            f"dpo_margin={float(dpo_stats['dpo_margin'].detach().float().item()):.6f}, "
                            f"policy_pref_rate={float(dpo_stats['policy_pref_rate'].detach().float().item()):.6f}, "
                            f"dpo_pref_rate={float(dpo_stats['dpo_pref_rate'].detach().float().item()):.6f}"
                        )
                        if dpo_stats["pair_margin"] is not None:
                            dpo_log_str += f", pair_margin={float(dpo_stats['pair_margin'].detach().float().item()):.6f}"
                        if dpo_stats.get("pair_margin_pos_rate", None) is not None:
                            dpo_log_str += f", pair_margin_pos_rate={float(dpo_stats['pair_margin_pos_rate'].detach().float().item()):.6f}"
                        logger.info(f"[DPO][TRAIN] epoch={epoch+1} step={step} {dpo_log_str}")

                if "base_loss" in aux_train_stats:
                    total_base_loss += aux_train_stats["base_loss"]
                    total_base_loss_count += 1
                if "total_loss" in aux_train_stats:
                    total_total_loss += aux_train_stats["total_loss"]
                    total_total_loss_count += 1
                if "stop_loss" in aux_train_stats:
                    total_stop_loss += aux_train_stats["stop_loss"]
                    total_stop_loss_count += 1
                if "stop_pos_rate" in aux_train_stats:
                    total_stop_pos_rate += aux_train_stats["stop_pos_rate"]
                    total_stop_pos_rate_count += 1
                if "stop_pred_mean" in aux_train_stats:
                    total_stop_pred_mean += aux_train_stats["stop_pred_mean"]
                    total_stop_pred_mean_count += 1
                if "stop_pred_pos_mean" in aux_train_stats:
                    total_stop_pred_pos_mean += aux_train_stats["stop_pred_pos_mean"]
                    total_stop_pred_pos_mean_count += 1
                if "stop_pred_neg_mean" in aux_train_stats:
                    total_stop_pred_neg_mean += aux_train_stats["stop_pred_neg_mean"]
                    total_stop_pred_neg_mean_count += 1

                total_loss += loss.detach().float()
                total_acc += acc # text_acc
                total_audio_acc += audio_acc[0]
                if dpo_stats is not None:
                    total_dpo_loss += (dpo_stats["dpo_loss"] / gradient_accumulation_steps)
                    total_sft_loss += (dpo_stats["sft_loss"] / gradient_accumulation_steps)
                    total_chosen_logp += (dpo_stats["chosen_logp"] / gradient_accumulation_steps)
                    total_rejected_logp += (dpo_stats["rejected_logp"] / gradient_accumulation_steps)
                    total_ref_chosen_logp += (dpo_stats["ref_chosen_logp"] / gradient_accumulation_steps)
                    total_ref_rejected_logp += (dpo_stats["ref_rejected_logp"] / gradient_accumulation_steps)
                    total_policy_margin += (dpo_stats["policy_margin"] / gradient_accumulation_steps)
                    total_ref_margin += (dpo_stats["ref_margin"] / gradient_accumulation_steps)
                    total_dpo_margin += (dpo_stats["dpo_margin"] / gradient_accumulation_steps)
                    total_policy_pref_rate += (dpo_stats["policy_pref_rate"] / gradient_accumulation_steps)
                    total_dpo_pref_rate += (dpo_stats["dpo_pref_rate"] / gradient_accumulation_steps)
                    if dpo_stats["pair_margin"] is not None:
                        total_pair_margin += float(dpo_stats["pair_margin"])
                        total_pair_margin_count += 1
                    if dpo_stats.get("pair_margin_pos_rate", None) is not None:
                        total_pair_margin_pos_rate += float(dpo_stats["pair_margin_pos_rate"])
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if debug_train_sanity:
                            scaler.unscale_(optimizer)
                            grad_norm, grad_param_count, nonzero_grad_count = _compute_grad_norm(optimizer)
                            diag_state["grad_observed_steps"] += 1
                            if grad_norm <= 1e-12 or nonzero_grad_count == 0:
                                diag_state["zero_grad_steps"] += 1
                            if should_log_sanity_step:
                                logger.info(
                                    "[GRAD] step=%d grad_norm=%.8f grad_param_count=%d nonzero_grad_count=%d",
                                    step,
                                    grad_norm,
                                    grad_param_count,
                                    nonzero_grad_count,
                                )
                        scaler.step(optimizer)
                        scaler.update()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step))
                            else:
                                wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step))
                        if debug_train_sanity and len(tracked_params) > 0:
                            diag_state["param_update_observed_steps"] += 1
                            tiny_updates = 0
                            update_logs = []
                            for name, p in tracked_params:
                                cur = p.detach().float()
                                prev = tracked_prev_values[name]
                                delta = cur - prev
                                l2_diff = float(torch.norm(delta).item())
                                max_abs_diff = float(delta.abs().max().item())
                                update_logs.append((name, l2_diff, max_abs_diff))
                                tracked_prev_values[name] = cur.clone()
                                if l2_diff <= 1e-12 and max_abs_diff <= 1e-12:
                                    tiny_updates += 1
                            if tiny_updates == len(tracked_params):
                                diag_state["tiny_param_update_steps"] += 1
                            if should_log_sanity_step:
                                for name, l2_diff, max_abs_diff in update_logs:
                                    logger.info(
                                        "[GRAD] [SANITY] step=%d param_update name=%s l2_diff=%.8e max_abs_diff=%.8e",
                                        step,
                                        name,
                                        l2_diff,
                                        max_abs_diff,
                                    )
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if debug_train_sanity:
                            grad_norm, grad_param_count, nonzero_grad_count = _compute_grad_norm(optimizer)
                            diag_state["grad_observed_steps"] += 1
                            if grad_norm <= 1e-12 or nonzero_grad_count == 0:
                                diag_state["zero_grad_steps"] += 1
                            if should_log_sanity_step:
                                logger.info(
                                    "[GRAD] step=%d grad_norm=%.8f grad_param_count=%d nonzero_grad_count=%d",
                                    step,
                                    grad_norm,
                                    grad_param_count,
                                    nonzero_grad_count,
                                )
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step))
                            else:
                                wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step))
                        if debug_train_sanity and len(tracked_params) > 0:
                            diag_state["param_update_observed_steps"] += 1
                            tiny_updates = 0
                            update_logs = []
                            for name, p in tracked_params:
                                cur = p.detach().float()
                                prev = tracked_prev_values[name]
                                delta = cur - prev
                                l2_diff = float(torch.norm(delta).item())
                                max_abs_diff = float(delta.abs().max().item())
                                update_logs.append((name, l2_diff, max_abs_diff))
                                tracked_prev_values[name] = cur.clone()
                                if l2_diff <= 1e-12 and max_abs_diff <= 1e-12:
                                    tiny_updates += 1
                            if tiny_updates == len(tracked_params):
                                diag_state["tiny_param_update_steps"] += 1
                            if should_log_sanity_step:
                                for name, l2_diff, max_abs_diff in update_logs:
                                    logger.info(
                                        "[GRAD] [SANITY] step=%d param_update name=%s l2_diff=%.8e max_abs_diff=%.8e",
                                        step,
                                        name,
                                        l2_diff,
                                        max_abs_diff,
                                    )
                        optimizer.zero_grad()
                        pbar.update(1)

                if dpo_stats is not None:
                    pbar.set_description(
                        f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} "
                        f"(loss: {loss.detach().float():.4f}, dpo: {dpo_stats['dpo_loss'].item():.4f}, sft: {dpo_stats['sft_loss'].item():.4f}, "
                        f"audio_acc: {audio_acc[0]:.4f}, text_acc: {acc:.4f})"
                    )
                else:
                    pbar.set_description(
                        f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} "
                        f"completed (loss: {loss.detach().float():.4f}, audio_acc: {audio_acc[0]:.4f}, text_acc: {acc:.4f})"
                    )
                
                if (epoch * total_length + step + 1) % train_config.validation_interval == 0 and train_config.run_validation:
                    eval_ppl, eval_epoch_loss, eval_epoch_acc, eval_epoch_audio_acc, eval_extra = evaluation(
                        model,
                        train_config,
                        eval_dataloader,
                        local_rank,
                        tokenizer,
                        reference_model=reference_model if use_dpo else None,
                    )
                    checkpoint_start_time = time.perf_counter()
                    checkpoint_name = f"{train_config.model_name}_epoch_{str(epoch+1)}_step_{step+1}"
                    checkpoint_metric_value = _get_checkpoint_monitor_value(
                        checkpoint_monitor,
                        eval_epoch_loss,
                        eval_epoch_acc,
                        eval_epoch_audio_acc,
                        eval_extra,
                    )
                    saved_ckpt_path = None
                    should_save_checkpoint = bool(train_config.save_model)
                    if should_save_checkpoint and checkpoint_top_k > 0:
                        if checkpoint_metric_value is None:
                            should_save_checkpoint = False
                            if is_main_process:
                                logger.warning(
                                    "[CKPT] skip saving checkpoint because monitor=%s is unavailable for epoch=%d step=%d.",
                                    checkpoint_monitor,
                                    epoch + 1,
                                    step + 1,
                                )
                        elif len(retained_checkpoint_records) >= checkpoint_top_k:
                            worst_record = _sort_checkpoint_records(retained_checkpoint_records, checkpoint_monitor_mode)[-1]
                            should_save_checkpoint = _is_better_checkpoint_metric(
                                checkpoint_metric_value,
                                worst_record.get("monitor_value"),
                                checkpoint_monitor_mode,
                            )
                            if (not should_save_checkpoint) and is_main_process:
                                logger.info(
                                    "[CKPT] skip checkpoint %s: monitor=%s value=%.6f does not enter top-%d (worst kept=%.6f at %s).",
                                    checkpoint_name,
                                    checkpoint_monitor,
                                    checkpoint_metric_value,
                                    checkpoint_top_k,
                                    float(worst_record.get("monitor_value")),
                                    worst_record.get("checkpoint_name"),
                                )
                    if should_save_checkpoint:
                        saved_ckpt_path = _estimate_checkpoint_path(train_config, fsdp_config, checkpoint_name, epoch)
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                        if train_config.use_peft:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    logger.info(f"we are about to save the PEFT modules")
                            else:
                                logger.info(f"we are about to save the PEFT modules")
                            if train_config.enable_fsdp:
                                if fsdp_config.sharding_strategy == ShardingStrategy.FULL_SHARD:
                                    save_model_checkpoint_peft_full_shard(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                elif fsdp_config.sharding_strategy == ShardingStrategy.NO_SHARD:
                                    if rank==0:
                                        save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                    dist.barrier()
                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                            else:
                                logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                        
                        elif not train_config.use_peft and train_config.freeze_llm:
                            logger.info(f"llm is frozen, we are about to save other parts.")
                            if train_config.enable_fsdp:
                                if fsdp_config.sharding_strategy == ShardingStrategy.FULL_SHARD:
                                    save_model_checkpoint_peft_full_shard(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                elif fsdp_config.sharding_strategy == ShardingStrategy.NO_SHARD:
                                    if rank==0:
                                        save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                    dist.barrier()
                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )

                        else: #
                            if train_config.enable_fsdp:
                                if getattr(StateDictType, fsdp_config.checkpoint_type) == StateDictType.FULL_STATE_DICT:
                                    save_model_checkpoint(
                                        model, optimizer, rank, train_config, epoch=epoch
                                    )
                                elif getattr(StateDictType, fsdp_config.checkpoint_type) == StateDictType.SHARDED_STATE_DICT:
                                    logger.info(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                    logger.info("=====================================================")

                                    save_model_and_optimizer_sharded(model, rank, train_config)
                                    if train_config.save_optimizer:
                                        save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                        logger.info(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                        logger.info("=====================================================")

                                if train_config.save_optimizer:
                                    save_optimizer_checkpoint(
                                        model, optimizer, rank, train_config, epoch=epoch
                                    )
                                    logger.info(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                    logger.info("=====================================================")

                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                                    
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )
                                
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                        if debug_train_sanity:
                            logger.info(f"[CKPT] saved checkpoint candidate path={saved_ckpt_path}")
                        if checkpoint_top_k > 0 and saved_ckpt_path not in (None, "", "None"):
                            retained_checkpoint_records.append(
                                {
                                    "checkpoint_name": checkpoint_name,
                                    "path": saved_ckpt_path,
                                    "epoch": epoch + 1,
                                    "step": step + 1,
                                    "global_step": epoch * total_length + step + 1,
                                    "monitor": checkpoint_monitor,
                                    "monitor_value": checkpoint_metric_value,
                                    "val_loss": _coerce_metric_value(eval_epoch_loss),
                                    "val_acc": _coerce_metric_value(eval_epoch_acc),
                                    "val_audio_acc": _coerce_metric_value(eval_epoch_audio_acc),
                                }
                            )
                            retained_checkpoint_records = _sort_checkpoint_records(
                                retained_checkpoint_records,
                                checkpoint_monitor_mode,
                            )
                            removed_records = []
                            while len(retained_checkpoint_records) > checkpoint_top_k:
                                removed_records.append(retained_checkpoint_records.pop())
                            if is_main_process:
                                for removed_record in removed_records:
                                    removed_path = removed_record.get("path")
                                    if removed_path == saved_ckpt_path:
                                        continue
                                    removed_ok = _delete_checkpoint_artifact(removed_path)
                                    if removed_ok:
                                        logger.info(
                                            "[CKPT] pruned checkpoint %s (monitor=%s value=%.6f).",
                                            removed_record.get("checkpoint_name"),
                                            checkpoint_monitor,
                                            float(removed_record.get("monitor_value")),
                                        )
                                summary_path = _write_checkpoint_topk_summary(
                                    train_config,
                                    retained_checkpoint_records,
                                    checkpoint_monitor,
                                    checkpoint_monitor_mode,
                                    checkpoint_top_k,
                                )
                                if summary_path is not None:
                                    logger.info(f"[CKPT] top-k summary updated at {summary_path}")
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                dist.barrier()
                            best_record = retained_checkpoint_records[0] if len(retained_checkpoint_records) > 0 else None
                            prev_best_ckpt_path = best_ckpt_path
                            if best_record is not None:
                                best_ckpt_path = best_record.get("path")
                                best_ckpt_epoch = best_record.get("epoch")
                                best_ckpt_step = best_record.get("step")
                                if is_main_process and best_ckpt_path != prev_best_ckpt_path:
                                    logger.info(
                                        "[CKPT] new best checkpoint by %s=%0.6f at epoch=%s step=%s path=%s",
                                        checkpoint_monitor,
                                        float(best_record.get("monitor_value")),
                                        best_ckpt_epoch,
                                        best_ckpt_step,
                                        best_ckpt_path,
                                    )
                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_epoch_loss < best_val_loss:
                        best_val_loss = eval_epoch_loss
                        if checkpoint_top_k <= 0 and train_config.save_model:
                            best_ckpt_path = saved_ckpt_path if saved_ckpt_path is not None else best_ckpt_path
                            best_ckpt_epoch = epoch + 1
                            best_ckpt_step = step + 1
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank==0:
                                logger.info(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                        else:
                            logger.info(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                        if debug_train_sanity:
                            logger.info(
                                f"[CKPT] [SANITY] new best by val_loss={float(best_val_loss):.6f} epoch={best_ckpt_epoch} step={best_ckpt_step} path={best_ckpt_path}"
                            )
                    val_loss.append(eval_epoch_loss)
                    val_prep.append(eval_ppl)
                    if eval_epoch_acc > best_val_acc:
                        best_val_acc = eval_epoch_acc
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank==0:
                                logger.info(f"best eval acc on epoch {epoch+1} is {best_val_acc}")
                        else:
                            logger.info(f"best eval acc on epoch {epoch+1} is {best_val_acc}")
                    val_acc.append(eval_epoch_acc)

                    if eval_epoch_audio_acc > best_val_audio_acc:
                        best_val_audio_acc = eval_epoch_audio_acc
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank==0:
                                logger.info(f"best eval audio acc on epoch {epoch+1} is {best_val_audio_acc}")
                        else:
                            logger.info(f"best eval audio acc on epoch {epoch+1} is {best_val_audio_acc}")
                    val_audio_acc.append(eval_epoch_audio_acc)

                    has_val_stop = bool(eval_extra.get("has_stop_metrics", False))
                    if has_val_stop:
                        val_stop_loss.append(eval_extra["eval_stop_loss"])
                        if eval_extra.get("eval_stop_pr_auc", None) is not None:
                            val_stop_pr_auc.append(eval_extra["eval_stop_pr_auc"])
                        if eval_extra.get("eval_stop_roc_auc", None) is not None:
                            val_stop_roc_auc.append(eval_extra["eval_stop_roc_auc"])
                        if eval_extra.get("eval_stop_precision@0.5", None) is not None:
                            val_stop_precision_at_0_5.append(eval_extra["eval_stop_precision@0.5"])
                        if eval_extra.get("eval_stop_recall@0.5", None) is not None:
                            val_stop_recall_at_0_5.append(eval_extra["eval_stop_recall@0.5"])
                        if eval_extra.get("eval_stop_f1@0.5", None) is not None:
                            val_stop_f1_at_0_5.append(eval_extra["eval_stop_f1@0.5"])
                        if eval_extra.get("eval_stop_pred_pos_mean", None) is not None:
                            val_stop_pred_pos_mean.append(eval_extra["eval_stop_pred_pos_mean"])
                        if eval_extra.get("eval_stop_pred_neg_mean", None) is not None:
                            val_stop_pred_neg_mean.append(eval_extra["eval_stop_pred_neg_mean"])
                        if (not (train_config.enable_fsdp or train_config.enable_ddp)) or rank == 0:
                            logger.info(
                                "[STOP][VAL] epoch=%d step=%d val_stop_loss=%.6f val_stop_pr_auc=%s val_stop_roc_auc=%s val_stop_precision@0.5=%.6f val_stop_recall@0.5=%.6f val_stop_f1@0.5=%.6f val_stop_pred_pos_mean=%s val_stop_pred_neg_mean=%s",
                                epoch + 1,
                                step + 1,
                                float(eval_extra["eval_stop_loss"]),
                                f"{float(eval_extra['eval_stop_pr_auc']):.6f}" if eval_extra.get("eval_stop_pr_auc", None) is not None else "None",
                                f"{float(eval_extra['eval_stop_roc_auc']):.6f}" if eval_extra.get("eval_stop_roc_auc", None) is not None else "None",
                                float(eval_extra["eval_stop_precision@0.5"]),
                                float(eval_extra["eval_stop_recall@0.5"]),
                                float(eval_extra["eval_stop_f1@0.5"]),
                                f"{float(eval_extra['eval_stop_pred_pos_mean']):.6f}" if eval_extra.get("eval_stop_pred_pos_mean", None) is not None else "None",
                                f"{float(eval_extra['eval_stop_pred_neg_mean']):.6f}" if eval_extra.get("eval_stop_pred_neg_mean", None) is not None else "None",
                            )

                    has_val_dpo = bool(eval_extra.get("has_dpo_metrics", False))
                    if has_val_dpo:
                        val_dpo_loss.append(eval_extra["eval_dpo_loss"])
                        val_sft_loss.append(eval_extra["eval_sft_loss"])
                        val_chosen_logp.append(eval_extra["eval_chosen_logp"])
                        val_rejected_logp.append(eval_extra["eval_rejected_logp"])
                        val_ref_chosen_logp.append(eval_extra["eval_ref_chosen_logp"])
                        val_ref_rejected_logp.append(eval_extra["eval_ref_rejected_logp"])
                        val_policy_margin.append(eval_extra["eval_policy_margin"])
                        val_ref_margin.append(eval_extra["eval_ref_margin"])
                        val_dpo_margin.append(eval_extra["eval_dpo_margin"])
                        val_policy_pref_rate.append(eval_extra["eval_policy_pref_rate"])
                        val_dpo_pref_rate.append(eval_extra["eval_dpo_pref_rate"])
                        if eval_extra.get("eval_pair_margin", None) is not None:
                            val_pair_margin.append(eval_extra["eval_pair_margin"])
                        if eval_extra.get("eval_pair_margin_pos_rate", None) is not None:
                            val_pair_margin_pos_rate.append(eval_extra["eval_pair_margin_pos_rate"])
                        if (not (train_config.enable_fsdp or train_config.enable_ddp)) or rank == 0:
                            logger.info(
                                "[DPO][VAL] epoch=%d step=%d dpo_loss=%.6f sft_loss=%.6f chosen_logp=%.6f rejected_logp=%.6f ref_chosen_logp=%.6f ref_rejected_logp=%.6f policy_margin=%.6f ref_margin=%.6f dpo_margin=%.6f policy_pref_rate=%.6f dpo_pref_rate=%.6f pair_margin=%s pair_margin_pos_rate=%s",
                                epoch + 1,
                                step + 1,
                                float(eval_extra["eval_dpo_loss"]),
                                float(eval_extra["eval_sft_loss"]),
                                float(eval_extra["eval_chosen_logp"]),
                                float(eval_extra["eval_rejected_logp"]),
                                float(eval_extra["eval_ref_chosen_logp"]),
                                float(eval_extra["eval_ref_rejected_logp"]),
                                float(eval_extra["eval_policy_margin"]),
                                float(eval_extra["eval_ref_margin"]),
                                float(eval_extra["eval_dpo_margin"]),
                                float(eval_extra["eval_policy_pref_rate"]),
                                float(eval_extra["eval_dpo_pref_rate"]),
                                f"{float(eval_extra['eval_pair_margin']):.6f}" if eval_extra.get("eval_pair_margin", None) is not None else "None",
                                f"{float(eval_extra['eval_pair_margin_pos_rate']):.6f}" if eval_extra.get("eval_pair_margin_pos_rate", None) is not None else "None",
                            )
                    
                    if log_config.use_wandb:
                        valid_log = {
                            "valid/val_epoch_loss": eval_epoch_loss,
                            "valid/val_perplexity": eval_ppl,
                            "valid/best_val_loss": best_val_loss,
                            "valid/val_accuracy": val_acc[-1],
                            "valid/val_audio_accuracy": val_audio_acc[-1],
                            "valid/val_best_audio_accuracy": best_val_audio_acc,
                            "valid/val_best_accuracy": best_val_acc,
                        }
                        if has_val_stop:
                            valid_log.update(
                                {
                                    "valid/val_stop_loss": eval_extra["eval_stop_loss"],
                                    "valid/val_stop_pr_auc": eval_extra.get("eval_stop_pr_auc", None),
                                    "valid/val_stop_roc_auc": eval_extra.get("eval_stop_roc_auc", None),
                                    "valid/val_stop_precision@0.5": eval_extra["eval_stop_precision@0.5"],
                                    "valid/val_stop_recall@0.5": eval_extra["eval_stop_recall@0.5"],
                                    "valid/val_stop_f1@0.5": eval_extra["eval_stop_f1@0.5"],
                                    "valid/val_stop_pred_pos_mean": eval_extra.get("eval_stop_pred_pos_mean", None),
                                    "valid/val_stop_pred_neg_mean": eval_extra.get("eval_stop_pred_neg_mean", None),
                                }
                            )
                        if has_val_dpo:
                            valid_log.update(
                                {
                                    "valid/val_dpo_loss": eval_extra["eval_dpo_loss"],
                                    "valid/val_sft_loss": eval_extra["eval_sft_loss"],
                                    "valid/val_chosen_logp": eval_extra["eval_chosen_logp"],
                                    "valid/val_rejected_logp": eval_extra["eval_rejected_logp"],
                                    "valid/val_ref_chosen_logp": eval_extra["eval_ref_chosen_logp"],
                                    "valid/val_ref_rejected_logp": eval_extra["eval_ref_rejected_logp"],
                                    "valid/val_policy_margin": eval_extra["eval_policy_margin"],
                                    "valid/val_ref_margin": eval_extra["eval_ref_margin"],
                                    "valid/val_dpo_margin": eval_extra["eval_dpo_margin"],
                                    "valid/val_policy_pref_rate": eval_extra["eval_policy_pref_rate"],
                                    "valid/val_dpo_pref_rate": eval_extra["eval_dpo_pref_rate"],
                                }
                            )
                            if eval_extra.get("eval_pair_margin", None) is not None:
                                valid_log["valid/val_pair_margin"] = eval_extra["eval_pair_margin"]
                            if eval_extra.get("eval_pair_margin_pos_rate", None) is not None:
                                valid_log["valid/val_pair_margin_pos_rate"] = eval_extra["eval_pair_margin_pos_rate"]
                        valid_log = {k: v for k, v in valid_log.items() if v is not None}
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank==0:
                                wandb.log(valid_log)
                        else:
                            wandb.log(valid_log)

                if train_config.run_test_during_validation:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank==0:
                            logger.info("=====================================")
                            logger.info(f"Test the file {train_config.run_test_during_validation_file} during validation:")
                            with autocast():
                                logger.info(model.inference(train_config.run_test_during_validation_file, train_config.run_test_during_validation_prompt))
                            logger.info("=====================================")
                        dist.barrier()
                    else:
                        logger.info("=====================================")
                        logger.info(f"Test the file {train_config.run_test_during_validation_file} during validation:")
                        with autocast():
                            logger.info(model.inference(train_config.run_test_during_validation_file, train_config.run_test_during_validation_prompt))
                        logger.info("=====================================")
            pbar.close()
            if tiny_overfit_mode and tiny_overfit_max_steps > 0 and global_step >= tiny_overfit_max_steps:
                logger.warning(
                    "[SANITY] tiny_overfit_max_steps reached (global_step=%d). Stopping training loop early for sanity run.",
                    global_step,
                )
                # Avoid breaking here: we still need epoch-level aggregation/logging
                # to keep result lists non-empty and diagnosis summary valid.
                stop_after_epoch = True

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_audio_acc, op=dist.ReduceOp.SUM)
            if use_dpo:
                dist.all_reduce(total_dpo_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_sft_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_chosen_logp, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_rejected_logp, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_ref_chosen_logp, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_ref_rejected_logp, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_policy_margin, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_ref_margin, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_dpo_margin, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_policy_pref_rate, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_dpo_pref_rate, op=dist.ReduceOp.SUM)
                pair_reduce = torch.tensor(
                    [
                        float(total_pair_margin),
                        float(total_pair_margin_pos_rate),
                        float(total_pair_margin_count),
                    ],
                    device=total_loss.device if isinstance(total_loss, torch.Tensor) else torch.device(f"cuda:{local_rank}"),
                )
                dist.all_reduce(pair_reduce, op=dist.ReduceOp.SUM)
                total_pair_margin = float(pair_reduce[0].item())
                total_pair_margin_pos_rate = float(pair_reduce[1].item())
                total_pair_margin_count = int(round(float(pair_reduce[2].item())))
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_acc = total_acc / len(train_dataloader)
        train_epoch_audio_acc = total_audio_acc / len(train_dataloader)
        if use_dpo:
            train_epoch_dpo_loss = total_dpo_loss / len(train_dataloader)
            train_epoch_sft_loss = total_sft_loss / len(train_dataloader)
            train_epoch_chosen_logp = total_chosen_logp / len(train_dataloader)
            train_epoch_rejected_logp = total_rejected_logp / len(train_dataloader)
            train_epoch_ref_chosen_logp = total_ref_chosen_logp / len(train_dataloader)
            train_epoch_ref_rejected_logp = total_ref_rejected_logp / len(train_dataloader)
            train_epoch_policy_margin = total_policy_margin / len(train_dataloader)
            train_epoch_ref_margin = total_ref_margin / len(train_dataloader)
            train_epoch_dpo_margin = total_dpo_margin / len(train_dataloader)
            train_epoch_policy_pref_rate = total_policy_pref_rate / len(train_dataloader)
            train_epoch_dpo_pref_rate = total_dpo_pref_rate / len(train_dataloader)
            train_epoch_pair_margin = (
                total_pair_margin / total_pair_margin_count if total_pair_margin_count > 0 else 0.0
            )
            train_epoch_pair_margin_pos_rate = (
                total_pair_margin_pos_rate / total_pair_margin_count if total_pair_margin_count > 0 else None
            )
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss/world_size
            train_epoch_acc = train_epoch_acc/world_size
            train_epoch_audio_acc = train_epoch_audio_acc/world_size
            if use_dpo:
                train_epoch_dpo_loss = train_epoch_dpo_loss/world_size
                train_epoch_sft_loss = train_epoch_sft_loss/world_size
                train_epoch_chosen_logp = train_epoch_chosen_logp/world_size
                train_epoch_rejected_logp = train_epoch_rejected_logp/world_size
                train_epoch_ref_chosen_logp = train_epoch_ref_chosen_logp/world_size
                train_epoch_ref_rejected_logp = train_epoch_ref_rejected_logp/world_size
                train_epoch_policy_margin = train_epoch_policy_margin/world_size
                train_epoch_ref_margin = train_epoch_ref_margin/world_size
                train_epoch_dpo_margin = train_epoch_dpo_margin/world_size
                train_epoch_policy_pref_rate = train_epoch_policy_pref_rate/world_size
                train_epoch_dpo_pref_rate = train_epoch_dpo_pref_rate/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        train_epoch_base_loss = total_base_loss / total_base_loss_count if total_base_loss_count > 0 else None
        train_epoch_total_loss = total_total_loss / total_total_loss_count if total_total_loss_count > 0 else None
        train_epoch_stop_loss = total_stop_loss / total_stop_loss_count if total_stop_loss_count > 0 else None
        train_epoch_stop_pos_rate = total_stop_pos_rate / total_stop_pos_rate_count if total_stop_pos_rate_count > 0 else None
        train_epoch_stop_pred_mean = total_stop_pred_mean / total_stop_pred_mean_count if total_stop_pred_mean_count > 0 else None
        train_epoch_stop_pred_pos_mean = (
            total_stop_pred_pos_mean / total_stop_pred_pos_mean_count if total_stop_pred_pos_mean_count > 0 else None
        )
        train_epoch_stop_pred_neg_mean = (
            total_stop_pred_neg_mean / total_stop_pred_neg_mean_count if total_stop_pred_neg_mean_count > 0 else None
        )

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        train_audio_acc.append(train_epoch_audio_acc)
        if train_epoch_base_loss is not None:
            train_base_loss.append(train_epoch_base_loss)
        if train_epoch_total_loss is not None:
            train_total_loss.append(train_epoch_total_loss)
        if train_epoch_stop_loss is not None:
            train_stop_loss.append(train_epoch_stop_loss)
        if train_epoch_stop_pos_rate is not None:
            train_stop_pos_rate.append(train_epoch_stop_pos_rate)
        if train_epoch_stop_pred_mean is not None:
            train_stop_pred_mean.append(train_epoch_stop_pred_mean)
        if train_epoch_stop_pred_pos_mean is not None:
            train_stop_pred_pos_mean.append(train_epoch_stop_pred_pos_mean)
        if train_epoch_stop_pred_neg_mean is not None:
            train_stop_pred_neg_mean.append(train_epoch_stop_pred_neg_mean)
        if use_dpo:
            train_dpo_loss.append(train_epoch_dpo_loss)
            train_sft_loss.append(train_epoch_sft_loss)
            train_chosen_logp.append(train_epoch_chosen_logp)
            train_rejected_logp.append(train_epoch_rejected_logp)
            train_ref_chosen_logp.append(train_epoch_ref_chosen_logp)
            train_ref_rejected_logp.append(train_epoch_ref_rejected_logp)
            train_policy_margin.append(train_epoch_policy_margin)
            train_ref_margin.append(train_epoch_ref_margin)
            train_dpo_margin.append(train_epoch_dpo_margin)
            train_policy_pref_rate.append(train_epoch_policy_pref_rate)
            train_dpo_pref_rate.append(train_epoch_dpo_pref_rate)
            train_pair_margin.append(train_epoch_pair_margin)
            if train_epoch_pair_margin_pos_rate is not None:
                train_pair_margin_pos_rate.append(train_epoch_pair_margin_pos_rate)

        if log_config.use_wandb:
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank==0:
                    wandb.log({"train/train_perplexity":train_perplexity, "train/train_epoch_loss":train_epoch_loss, "train/train_epoch_acc":train_epoch_acc, "train/train_epoch_audio_acc":train_epoch_audio_acc})
                    epoch_aux_log = {}
                    if train_epoch_base_loss is not None:
                        epoch_aux_log["train/train_epoch_base_loss"] = train_epoch_base_loss
                    if train_epoch_total_loss is not None:
                        epoch_aux_log["train/train_epoch_total_loss"] = train_epoch_total_loss
                    if train_epoch_stop_loss is not None:
                        epoch_aux_log["train/train_epoch_stop_loss"] = train_epoch_stop_loss
                    if train_epoch_stop_pos_rate is not None:
                        epoch_aux_log["train/train_epoch_stop_pos_rate"] = train_epoch_stop_pos_rate
                    if train_epoch_stop_pred_mean is not None:
                        epoch_aux_log["train/train_epoch_stop_pred_mean"] = train_epoch_stop_pred_mean
                    if train_epoch_stop_pred_pos_mean is not None:
                        epoch_aux_log["train/train_epoch_stop_pred_pos_mean"] = train_epoch_stop_pred_pos_mean
                    if train_epoch_stop_pred_neg_mean is not None:
                        epoch_aux_log["train/train_epoch_stop_pred_neg_mean"] = train_epoch_stop_pred_neg_mean
                    if len(epoch_aux_log) > 0:
                        wandb.log(epoch_aux_log)
                    if use_dpo:
                        wandb.log(
                            {
                                "train/train_epoch_dpo_loss": train_epoch_dpo_loss,
                                "train/train_epoch_sft_loss": train_epoch_sft_loss,
                                "train/train_epoch_chosen_logp": train_epoch_chosen_logp,
                                "train/train_epoch_rejected_logp": train_epoch_rejected_logp,
                                "train/train_epoch_ref_chosen_logp": train_epoch_ref_chosen_logp,
                                "train/train_epoch_ref_rejected_logp": train_epoch_ref_rejected_logp,
                                "train/train_epoch_policy_margin": train_epoch_policy_margin,
                                "train/train_epoch_ref_margin": train_epoch_ref_margin,
                                "train/train_epoch_dpo_margin": train_epoch_dpo_margin,
                                "train/train_epoch_policy_pref_rate": train_epoch_policy_pref_rate,
                                "train/train_epoch_dpo_pref_rate": train_epoch_dpo_pref_rate,
                                "train/train_epoch_pair_margin": train_epoch_pair_margin,
                            }
                        )
                        if train_epoch_pair_margin_pos_rate is not None:
                            wandb.log({"train/train_epoch_pair_margin_pos_rate": train_epoch_pair_margin_pos_rate})
            else:
                wandb.log({"train/train_perplexity":train_perplexity, "train/train_epoch_loss":train_epoch_loss, "train/train_epoch_acc":train_epoch_acc, "train/train_epoch_audio_acc":train_epoch_audio_acc})
                epoch_aux_log = {}
                if train_epoch_base_loss is not None:
                    epoch_aux_log["train/train_epoch_base_loss"] = train_epoch_base_loss
                if train_epoch_total_loss is not None:
                    epoch_aux_log["train/train_epoch_total_loss"] = train_epoch_total_loss
                if train_epoch_stop_loss is not None:
                    epoch_aux_log["train/train_epoch_stop_loss"] = train_epoch_stop_loss
                if train_epoch_stop_pos_rate is not None:
                    epoch_aux_log["train/train_epoch_stop_pos_rate"] = train_epoch_stop_pos_rate
                if train_epoch_stop_pred_mean is not None:
                    epoch_aux_log["train/train_epoch_stop_pred_mean"] = train_epoch_stop_pred_mean
                if train_epoch_stop_pred_pos_mean is not None:
                    epoch_aux_log["train/train_epoch_stop_pred_pos_mean"] = train_epoch_stop_pred_pos_mean
                if train_epoch_stop_pred_neg_mean is not None:
                    epoch_aux_log["train/train_epoch_stop_pred_neg_mean"] = train_epoch_stop_pred_neg_mean
                if len(epoch_aux_log) > 0:
                    wandb.log(epoch_aux_log)
                if use_dpo:
                    wandb.log(
                        {
                            "train/train_epoch_dpo_loss": train_epoch_dpo_loss,
                            "train/train_epoch_sft_loss": train_epoch_sft_loss,
                            "train/train_epoch_chosen_logp": train_epoch_chosen_logp,
                            "train/train_epoch_rejected_logp": train_epoch_rejected_logp,
                            "train/train_epoch_ref_chosen_logp": train_epoch_ref_chosen_logp,
                            "train/train_epoch_ref_rejected_logp": train_epoch_ref_rejected_logp,
                            "train/train_epoch_policy_margin": train_epoch_policy_margin,
                            "train/train_epoch_ref_margin": train_epoch_ref_margin,
                            "train/train_epoch_dpo_margin": train_epoch_dpo_margin,
                            "train/train_epoch_policy_pref_rate": train_epoch_policy_pref_rate,
                            "train/train_epoch_dpo_pref_rate": train_epoch_dpo_pref_rate,
                            "train/train_epoch_pair_margin": train_epoch_pair_margin,
                        }
                    )
                    if train_epoch_pair_margin_pos_rate is not None:
                        wandb.log({"train/train_epoch_pair_margin_pos_rate": train_epoch_pair_margin_pos_rate})

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                logger.info(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
                if use_dpo:
                    logger.info(
                        f"[DPO][EPOCH][TRAIN] dpo_loss={train_epoch_dpo_loss:.6f}, sft_loss={train_epoch_sft_loss:.6f}, "
                        f"chosen_logp={train_epoch_chosen_logp:.6f}, rejected_logp={train_epoch_rejected_logp:.6f}, "
                        f"ref_chosen_logp={train_epoch_ref_chosen_logp:.6f}, ref_rejected_logp={train_epoch_ref_rejected_logp:.6f}, "
                        f"policy_margin={train_epoch_policy_margin:.6f}, ref_margin={train_epoch_ref_margin:.6f}, dpo_margin={train_epoch_dpo_margin:.6f}, "
                        f"policy_pref_rate={train_epoch_policy_pref_rate:.6f}, dpo_pref_rate={train_epoch_dpo_pref_rate:.6f}, "
                        f"pair_margin={train_epoch_pair_margin:.6f}, pair_margin_pos_rate={train_epoch_pair_margin_pos_rate if train_epoch_pair_margin_pos_rate is not None else 'None'}"
                    )
        else:
            logger.info(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            if use_dpo:
                logger.info(
                    f"[DPO][EPOCH][TRAIN] dpo_loss={train_epoch_dpo_loss:.6f}, sft_loss={train_epoch_sft_loss:.6f}, "
                    f"chosen_logp={train_epoch_chosen_logp:.6f}, rejected_logp={train_epoch_rejected_logp:.6f}, "
                    f"ref_chosen_logp={train_epoch_ref_chosen_logp:.6f}, ref_rejected_logp={train_epoch_ref_rejected_logp:.6f}, "
                    f"policy_margin={train_epoch_policy_margin:.6f}, ref_margin={train_epoch_ref_margin:.6f}, dpo_margin={train_epoch_dpo_margin:.6f}, "
                    f"policy_pref_rate={train_epoch_policy_pref_rate:.6f}, dpo_pref_rate={train_epoch_dpo_pref_rate:.6f}, "
                    f"pair_margin={train_epoch_pair_margin:.6f}, pair_margin_pos_rate={train_epoch_pair_margin_pos_rate if train_epoch_pair_margin_pos_rate is not None else 'None'}"
                )
        epoch_aux_parts = []
        if train_epoch_base_loss is not None:
            epoch_aux_parts.append(f"base_loss={train_epoch_base_loss:.6f}")
        if train_epoch_total_loss is not None:
            epoch_aux_parts.append(f"total_loss={train_epoch_total_loss:.6f}")
        if train_epoch_stop_loss is not None:
            epoch_aux_parts.append(f"stop_loss={train_epoch_stop_loss:.6f}")
        if train_epoch_stop_pos_rate is not None:
            epoch_aux_parts.append(f"stop_pos_rate={train_epoch_stop_pos_rate:.6f}")
        if train_epoch_stop_pred_mean is not None:
            epoch_aux_parts.append(f"stop_pred_mean={train_epoch_stop_pred_mean:.6f}")
        if train_epoch_stop_pred_pos_mean is not None:
            epoch_aux_parts.append(f"stop_pred_pos_mean={train_epoch_stop_pred_pos_mean:.6f}")
        if train_epoch_stop_pred_neg_mean is not None:
            epoch_aux_parts.append(f"stop_pred_neg_mean={train_epoch_stop_pred_neg_mean:.6f}")
        if len(epoch_aux_parts) > 0:
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank == 0:
                    logger.info(f"Epoch {epoch+1} aux metrics: " + ", ".join(epoch_aux_parts))
            else:
                logger.info(f"Epoch {epoch+1} aux metrics: " + ", ".join(epoch_aux_parts))

        if train_config.enable_fsdp:
            if rank==0:
                logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
                logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                logger.info(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
            logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            logger.info(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        if stop_after_epoch:
            break

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    avg_train_acc = sum(train_acc)/len(train_acc)
    avg_train_base_loss = sum(train_base_loss)/len(train_base_loss) if len(train_base_loss) > 0 else None
    avg_train_total_loss = sum(train_total_loss)/len(train_total_loss) if len(train_total_loss) > 0 else None
    avg_train_stop_loss = sum(train_stop_loss)/len(train_stop_loss) if len(train_stop_loss) > 0 else None
    avg_train_stop_pos_rate = sum(train_stop_pos_rate)/len(train_stop_pos_rate) if len(train_stop_pos_rate) > 0 else None
    avg_train_stop_pred_mean = sum(train_stop_pred_mean)/len(train_stop_pred_mean) if len(train_stop_pred_mean) > 0 else None
    avg_train_stop_pred_pos_mean = (
        sum(train_stop_pred_pos_mean)/len(train_stop_pred_pos_mean) if len(train_stop_pred_pos_mean) > 0 else None
    )
    avg_train_stop_pred_neg_mean = (
        sum(train_stop_pred_neg_mean)/len(train_stop_pred_neg_mean) if len(train_stop_pred_neg_mean) > 0 else None
    )
    if use_dpo and len(train_dpo_loss) > 0:
        avg_train_dpo_loss = sum(train_dpo_loss)/len(train_dpo_loss)
        avg_train_sft_loss = sum(train_sft_loss)/len(train_sft_loss)
        avg_train_chosen_logp = sum(train_chosen_logp)/len(train_chosen_logp)
        avg_train_rejected_logp = sum(train_rejected_logp)/len(train_rejected_logp)
        avg_train_ref_chosen_logp = sum(train_ref_chosen_logp)/len(train_ref_chosen_logp)
        avg_train_ref_rejected_logp = sum(train_ref_rejected_logp)/len(train_ref_rejected_logp)
        avg_train_policy_margin = sum(train_policy_margin)/len(train_policy_margin)
        avg_train_ref_margin = sum(train_ref_margin)/len(train_ref_margin)
        avg_train_dpo_margin = sum(train_dpo_margin)/len(train_dpo_margin)
        avg_train_policy_pref_rate = sum(train_policy_pref_rate)/len(train_policy_pref_rate)
        avg_train_dpo_pref_rate = sum(train_dpo_pref_rate)/len(train_dpo_pref_rate)
        avg_train_pair_margin = sum(train_pair_margin)/len(train_pair_margin) if len(train_pair_margin) > 0 else 0.0
        avg_train_pair_margin_pos_rate = (
            sum(train_pair_margin_pos_rate)/len(train_pair_margin_pos_rate) if len(train_pair_margin_pos_rate) > 0 else None
        )
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)
        avg_eval_acc = sum(val_acc)/len(val_acc)
        avg_eval_audio_acc = sum(val_audio_acc)/len(val_audio_acc)
        avg_eval_stop_loss = sum(val_stop_loss)/len(val_stop_loss) if len(val_stop_loss) > 0 else None
        avg_eval_stop_pr_auc = sum(val_stop_pr_auc)/len(val_stop_pr_auc) if len(val_stop_pr_auc) > 0 else None
        avg_eval_stop_roc_auc = sum(val_stop_roc_auc)/len(val_stop_roc_auc) if len(val_stop_roc_auc) > 0 else None
        avg_eval_stop_precision_at_0_5 = (
            sum(val_stop_precision_at_0_5)/len(val_stop_precision_at_0_5) if len(val_stop_precision_at_0_5) > 0 else None
        )
        avg_eval_stop_recall_at_0_5 = (
            sum(val_stop_recall_at_0_5)/len(val_stop_recall_at_0_5) if len(val_stop_recall_at_0_5) > 0 else None
        )
        avg_eval_stop_f1_at_0_5 = sum(val_stop_f1_at_0_5)/len(val_stop_f1_at_0_5) if len(val_stop_f1_at_0_5) > 0 else None
        avg_eval_stop_pred_pos_mean = (
            sum(val_stop_pred_pos_mean)/len(val_stop_pred_pos_mean) if len(val_stop_pred_pos_mean) > 0 else None
        )
        avg_eval_stop_pred_neg_mean = (
            sum(val_stop_pred_neg_mean)/len(val_stop_pred_neg_mean) if len(val_stop_pred_neg_mean) > 0 else None
        )
        if len(val_dpo_loss) > 0:
            avg_eval_dpo_loss = sum(val_dpo_loss)/len(val_dpo_loss)
            avg_eval_sft_loss = sum(val_sft_loss)/len(val_sft_loss)
            avg_eval_chosen_logp = sum(val_chosen_logp)/len(val_chosen_logp)
            avg_eval_rejected_logp = sum(val_rejected_logp)/len(val_rejected_logp)
            avg_eval_ref_chosen_logp = sum(val_ref_chosen_logp)/len(val_ref_chosen_logp)
            avg_eval_ref_rejected_logp = sum(val_ref_rejected_logp)/len(val_ref_rejected_logp)
            avg_eval_policy_margin = sum(val_policy_margin)/len(val_policy_margin)
            avg_eval_ref_margin = sum(val_ref_margin)/len(val_ref_margin)
            avg_eval_dpo_margin = sum(val_dpo_margin)/len(val_dpo_margin)
            avg_eval_policy_pref_rate = sum(val_policy_pref_rate)/len(val_policy_pref_rate)
            avg_eval_dpo_pref_rate = sum(val_dpo_pref_rate)/len(val_dpo_pref_rate)
            avg_eval_pair_margin = sum(val_pair_margin)/len(val_pair_margin) if len(val_pair_margin) > 0 else None
            avg_eval_pair_margin_pos_rate = (
                sum(val_pair_margin_pos_rate)/len(val_pair_margin_pos_rate) if len(val_pair_margin_pos_rate) > 0 else None
            )

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results['avg_train_acc'] = avg_train_acc
    if avg_train_base_loss is not None:
        results['avg_train_base_loss'] = avg_train_base_loss
    if avg_train_total_loss is not None:
        results['avg_train_total_loss'] = avg_train_total_loss
    if avg_train_stop_loss is not None:
        results['avg_train_stop_loss'] = avg_train_stop_loss
    if avg_train_stop_pos_rate is not None:
        results['avg_train_stop_pos_rate'] = avg_train_stop_pos_rate
    if avg_train_stop_pred_mean is not None:
        results['avg_train_stop_pred_mean'] = avg_train_stop_pred_mean
    if avg_train_stop_pred_pos_mean is not None:
        results['avg_train_stop_pred_pos_mean'] = avg_train_stop_pred_pos_mean
    if avg_train_stop_pred_neg_mean is not None:
        results['avg_train_stop_pred_neg_mean'] = avg_train_stop_pred_neg_mean
    if use_dpo and len(train_dpo_loss) > 0:
        results['avg_train_dpo_loss'] = avg_train_dpo_loss
        results['avg_train_sft_loss'] = avg_train_sft_loss
        results['avg_train_chosen_logp'] = avg_train_chosen_logp
        results['avg_train_rejected_logp'] = avg_train_rejected_logp
        results['avg_train_ref_chosen_logp'] = avg_train_ref_chosen_logp
        results['avg_train_ref_rejected_logp'] = avg_train_ref_rejected_logp
        results['avg_train_policy_margin'] = avg_train_policy_margin
        results['avg_train_ref_margin'] = avg_train_ref_margin
        results['avg_train_dpo_margin'] = avg_train_dpo_margin
        results['avg_train_policy_pref_rate'] = avg_train_policy_pref_rate
        results['avg_train_dpo_pref_rate'] = avg_train_dpo_pref_rate
        results['avg_train_pair_margin'] = avg_train_pair_margin
        if avg_train_pair_margin_pos_rate is not None:
            results['avg_train_pair_margin_pos_rate'] = avg_train_pair_margin_pos_rate
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
        results['avg_eval_acc'] = avg_eval_acc
        results['avg_eval_audio_acc'] = avg_eval_audio_acc
        if avg_eval_stop_loss is not None:
            results['avg_eval_stop_loss'] = avg_eval_stop_loss
        if avg_eval_stop_pr_auc is not None:
            results['avg_eval_stop_pr_auc'] = avg_eval_stop_pr_auc
        if avg_eval_stop_roc_auc is not None:
            results['avg_eval_stop_roc_auc'] = avg_eval_stop_roc_auc
        if avg_eval_stop_precision_at_0_5 is not None:
            results['avg_eval_stop_precision@0.5'] = avg_eval_stop_precision_at_0_5
        if avg_eval_stop_recall_at_0_5 is not None:
            results['avg_eval_stop_recall@0.5'] = avg_eval_stop_recall_at_0_5
        if avg_eval_stop_f1_at_0_5 is not None:
            results['avg_eval_stop_f1@0.5'] = avg_eval_stop_f1_at_0_5
        if avg_eval_stop_pred_pos_mean is not None:
            results['avg_eval_stop_pred_pos_mean'] = avg_eval_stop_pred_pos_mean
        if avg_eval_stop_pred_neg_mean is not None:
            results['avg_eval_stop_pred_neg_mean'] = avg_eval_stop_pred_neg_mean
        if len(val_dpo_loss) > 0:
            results['avg_eval_dpo_loss'] = avg_eval_dpo_loss
            results['avg_eval_sft_loss'] = avg_eval_sft_loss
            results['avg_eval_chosen_logp'] = avg_eval_chosen_logp
            results['avg_eval_rejected_logp'] = avg_eval_rejected_logp
            results['avg_eval_ref_chosen_logp'] = avg_eval_ref_chosen_logp
            results['avg_eval_ref_rejected_logp'] = avg_eval_ref_rejected_logp
            results['avg_eval_policy_margin'] = avg_eval_policy_margin
            results['avg_eval_ref_margin'] = avg_eval_ref_margin
            results['avg_eval_dpo_margin'] = avg_eval_dpo_margin
            results['avg_eval_policy_pref_rate'] = avg_eval_policy_pref_rate
            results['avg_eval_dpo_pref_rate'] = avg_eval_dpo_pref_rate
            if avg_eval_pair_margin is not None:
                results['avg_eval_pair_margin'] = avg_eval_pair_margin
            if avg_eval_pair_margin_pos_rate is not None:
                results['avg_eval_pair_margin_pos_rate'] = avg_eval_pair_margin_pos_rate
    results["best_ckpt_path"] = best_ckpt_path
    results["best_ckpt_epoch"] = best_ckpt_epoch
    results["best_ckpt_step"] = best_ckpt_step
    results["checkpoint_monitor"] = checkpoint_monitor
    results["checkpoint_monitor_mode"] = checkpoint_monitor_mode
    if checkpoint_top_k > 0:
        results["checkpoint_top_k"] = checkpoint_top_k
        results["retained_checkpoints"] = retained_checkpoint_records
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    diag_state["best_ckpt_path"] = best_ckpt_path
    infer_ckpt_path = run_context.get("inference_ckpt_path", None)
    if infer_ckpt_path in (None, "", "None"):
        infer_ckpt_path = train_config.get("debug_inference_ckpt_path", None)
    if infer_ckpt_path in (None, "", "None"):
        infer_ckpt_path = os.environ.get("SANITY_INFER_CKPT_PATH", None)
    diag_state["inference_ckpt_path"] = infer_ckpt_path
    if infer_ckpt_path not in (None, "", "None") and best_ckpt_path not in (None, "", "None"):
        try:
            if str(Path(infer_ckpt_path).resolve()) != str(Path(best_ckpt_path).resolve()):
                diag_state["inference_ckpt_mismatch"] = True
        except Exception:
            diag_state["inference_ckpt_mismatch"] = str(infer_ckpt_path) != str(best_ckpt_path)

    diagnosis_payload = {
        "summary": _build_diagnosis_summary(diag_state),
        "raw_state": {
            k: v for k, v in diag_state.items()
            if k not in {"trainable_named_params"}
        },
        "run_context": run_context,
    }
    results["diagnosis_summary"] = diagnosis_payload["summary"]
    if debug_train_sanity:
        logger.info(f"[CKPT] best_ckpt_path={best_ckpt_path} best_epoch={best_ckpt_epoch} best_step={best_ckpt_step}")
        logger.info(f"[CKPT] inference_ckpt_path={infer_ckpt_path}")
        if diag_state["inference_ckpt_mismatch"]:
            logger.warning("[CKPT] [DIAGNOSIS] loaded inference ckpt path mismatches recorded best ckpt path.")
        logger.info(f"[DIAGNOSIS] structured_summary={json.dumps(diagnosis_payload, ensure_ascii=False)}")
        summary_path = _write_diagnosis_summary(train_config, diagnosis_payload)
        if summary_path is not None:
            logger.info(f"[DIAGNOSIS] summary_file={summary_path}")

    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, reference_model=None):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss, eval_epoch_acc, eval_epoch_audio_acc, eval_extra_metrics
    """
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    use_dpo_eval = bool(train_config.get("use_dpo", False)) and (reference_model is not None)
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_acc = 0.0
    eval_audio_acc = 0.0
    eval_dpo_loss = 0.0
    eval_sft_loss = 0.0
    eval_chosen_logp = 0.0
    eval_rejected_logp = 0.0
    eval_ref_chosen_logp = 0.0
    eval_ref_rejected_logp = 0.0
    eval_policy_margin = 0.0
    eval_ref_margin = 0.0
    eval_dpo_margin = 0.0
    eval_policy_pref_rate = 0.0
    eval_dpo_pref_rate = 0.0
    eval_pair_margin = 0.0
    eval_pair_margin_pos_rate = 0.0
    eval_pair_margin_count = 0
    eval_dpo_count = 0
    eval_stop_loss = 0.0
    eval_stop_loss_count = 0
    eval_stop_prob_sum = 0.0
    eval_stop_prob_count = 0
    eval_stop_pos_prob_sum = 0.0
    eval_stop_pos_count = 0
    eval_stop_neg_prob_sum = 0.0
    eval_stop_neg_count = 0
    eval_stop_tp = 0.0
    eval_stop_fp = 0.0
    eval_stop_fn = 0.0
    eval_stop_tn = 0.0
    eval_stop_prob_chunks = []
    eval_stop_target_chunks = []
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    with MemoryTrace() as memtrace:
        total_length = len(eval_dataloader)
        pbar = tqdm(colour="green", desc=f"Evaluating Epoch", total=total_length, dynamic_ncols=True)
        for step, batch in enumerate(eval_dataloader):
            step_device = local_rank if (train_config.enable_fsdp or train_config.enable_ddp) else "cuda:0"
            batch = _move_batch_to_device(batch, step_device)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                with autocast():
                    dpo_stats = None
                    if use_dpo_eval and _is_dpo_batch(batch):
                        loss, outputs, rest, dpo_stats = _compute_dpo_step(
                            model=model,
                            reference_model=reference_model,
                            batch=batch,
                            train_config=train_config,
                        )
                    else:
                        outputs, *rest = model(**batch)
                        loss = outputs.loss
                acc = rest[0] if rest else -1 #text_acc
                audio_acc = rest[1] if rest else -1   # seven layers of audio acc
                if not isinstance(acc, torch.Tensor):
                    acc = torch.tensor(float(acc), device=loss.device if isinstance(loss, torch.Tensor) else step_device)
                if isinstance(audio_acc, (list, tuple)) and len(audio_acc) > 0:
                    audio_acc = audio_acc[0]
                if not isinstance(audio_acc, torch.Tensor):
                    audio_acc = torch.tensor(float(audio_acc), device=loss.device if isinstance(loss, torch.Tensor) else step_device)

                aux_eval_stats = _extract_optional_aux_stats(outputs, batch)
                stop_logits_detached = getattr(outputs, "stop_logits_detached", None)
                stop_target_detached = getattr(outputs, "stop_target_detached", None)
                stop_valid_detached = getattr(outputs, "stop_valid_detached", None)
                if (
                    isinstance(stop_logits_detached, torch.Tensor)
                    and isinstance(stop_target_detached, torch.Tensor)
                    and isinstance(stop_valid_detached, torch.Tensor)
                ):
                    stop_valid_mask = stop_valid_detached.bool()
                    if stop_valid_mask.any():
                        stop_probs = torch.sigmoid(stop_logits_detached.detach())[stop_valid_mask].float()
                        stop_targets = stop_target_detached.detach()[stop_valid_mask].float()
                        if "stop_loss" in aux_eval_stats:
                            eval_stop_loss += float(aux_eval_stats["stop_loss"])
                            eval_stop_loss_count += 1

                        eval_stop_prob_sum += float(stop_probs.sum().item())
                        eval_stop_prob_count += int(stop_probs.numel())
                        stop_pos_mask = stop_targets > 0.5
                        stop_neg_mask = ~stop_pos_mask
                        if stop_pos_mask.any():
                            eval_stop_pos_prob_sum += float(stop_probs[stop_pos_mask].sum().item())
                            eval_stop_pos_count += int(stop_pos_mask.sum().item())
                        if stop_neg_mask.any():
                            eval_stop_neg_prob_sum += float(stop_probs[stop_neg_mask].sum().item())
                            eval_stop_neg_count += int(stop_neg_mask.sum().item())

                        stop_pred_pos = stop_probs >= 0.5
                        eval_stop_tp += float((stop_pred_pos & stop_pos_mask).sum().item())
                        eval_stop_fp += float((stop_pred_pos & stop_neg_mask).sum().item())
                        eval_stop_fn += float(((~stop_pred_pos) & stop_pos_mask).sum().item())
                        eval_stop_tn += float(((~stop_pred_pos) & stop_neg_mask).sum().item())

                        if not (train_config.enable_fsdp or train_config.enable_ddp):
                            eval_stop_prob_chunks.append(stop_probs.cpu())
                            eval_stop_target_chunks.append(stop_targets.cpu())

                eval_loss += loss.detach().float()
                eval_acc += acc
                eval_audio_acc += audio_acc
                if dpo_stats is not None:
                    eval_dpo_count += 1
                    eval_dpo_loss += float(dpo_stats["dpo_loss"].detach().float().item())
                    eval_sft_loss += float(dpo_stats["sft_loss"].detach().float().item())
                    eval_chosen_logp += float(dpo_stats["chosen_logp"].detach().float().item())
                    eval_rejected_logp += float(dpo_stats["rejected_logp"].detach().float().item())
                    eval_ref_chosen_logp += float(dpo_stats["ref_chosen_logp"].detach().float().item())
                    eval_ref_rejected_logp += float(dpo_stats["ref_rejected_logp"].detach().float().item())
                    eval_policy_margin += float(dpo_stats["policy_margin"].detach().float().item())
                    eval_ref_margin += float(dpo_stats["ref_margin"].detach().float().item())
                    eval_dpo_margin += float(dpo_stats["dpo_margin"].detach().float().item())
                    eval_policy_pref_rate += float(dpo_stats["policy_pref_rate"].detach().float().item())
                    eval_dpo_pref_rate += float(dpo_stats["dpo_pref_rate"].detach().float().item())
                    if dpo_stats.get("pair_margin", None) is not None:
                        eval_pair_margin += float(dpo_stats["pair_margin"].detach().float().item())
                        eval_pair_margin_count += 1
                    if dpo_stats.get("pair_margin_pos_rate", None) is not None:
                        eval_pair_margin_pos_rate += float(dpo_stats["pair_margin_pos_rate"].detach().float().item())
            # Decode predictions and add to evaluation predictions list
            try:
                preds = torch.argmax(outputs.logits, -1)
                eval_preds.extend(
                    tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
                )
            except Exception:
                pass  # vallex does not need to show it's result (we can't view any thing from abstract acoustic token)
            pbar.update(1)
            if eval_dpo_count > 0:
                pbar.set_description(
                    f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, eval_dpo: {eval_dpo_loss/max(eval_dpo_count,1):.4f}, "
                    f"eval_audio_acc: {eval_audio_acc/(step+1):.4f}, eval_acc: {eval_acc/(step+1):.4f}"
                )
            else:
                pbar.set_description(
                    f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, eval_audio_acc: {eval_audio_acc/(step+1):.4f}, eval_acc: {eval_acc/(step+1):.4f}"
                )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_audio_acc, op=dist.ReduceOp.SUM)
        if use_dpo_eval:
            reduce_device = eval_loss.device if isinstance(eval_loss, torch.Tensor) else (
                torch.device(f"cuda:{local_rank}") if (train_config.enable_fsdp or train_config.enable_ddp) else torch.device("cuda:0")
            )
            dpo_reduce = {
                "eval_dpo_loss": eval_dpo_loss,
                "eval_sft_loss": eval_sft_loss,
                "eval_chosen_logp": eval_chosen_logp,
                "eval_rejected_logp": eval_rejected_logp,
                "eval_ref_chosen_logp": eval_ref_chosen_logp,
                "eval_ref_rejected_logp": eval_ref_rejected_logp,
                "eval_policy_margin": eval_policy_margin,
                "eval_ref_margin": eval_ref_margin,
                "eval_dpo_margin": eval_dpo_margin,
                "eval_policy_pref_rate": eval_policy_pref_rate,
                "eval_dpo_pref_rate": eval_dpo_pref_rate,
                "eval_pair_margin": eval_pair_margin,
                "eval_pair_margin_pos_rate": eval_pair_margin_pos_rate,
                "eval_dpo_count": float(eval_dpo_count),
                "eval_pair_margin_count": float(eval_pair_margin_count),
            }
            for name, value in dpo_reduce.items():
                tensor_val = torch.tensor(float(value), device=reduce_device)
                dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
                dpo_reduce[name] = float(tensor_val.item())
            eval_dpo_loss = dpo_reduce["eval_dpo_loss"]
            eval_sft_loss = dpo_reduce["eval_sft_loss"]
            eval_chosen_logp = dpo_reduce["eval_chosen_logp"]
            eval_rejected_logp = dpo_reduce["eval_rejected_logp"]
            eval_ref_chosen_logp = dpo_reduce["eval_ref_chosen_logp"]
            eval_ref_rejected_logp = dpo_reduce["eval_ref_rejected_logp"]
            eval_policy_margin = dpo_reduce["eval_policy_margin"]
            eval_ref_margin = dpo_reduce["eval_ref_margin"]
            eval_dpo_margin = dpo_reduce["eval_dpo_margin"]
            eval_policy_pref_rate = dpo_reduce["eval_policy_pref_rate"]
            eval_dpo_pref_rate = dpo_reduce["eval_dpo_pref_rate"]
            eval_pair_margin = dpo_reduce["eval_pair_margin"]
            eval_pair_margin_pos_rate = dpo_reduce["eval_pair_margin_pos_rate"]
            eval_dpo_count = int(round(dpo_reduce["eval_dpo_count"]))
            eval_pair_margin_count = int(round(dpo_reduce["eval_pair_margin_count"]))
        stop_reduce = {
            "eval_stop_loss": eval_stop_loss,
            "eval_stop_loss_count": float(eval_stop_loss_count),

            "eval_stop_prob_sum": eval_stop_prob_sum,
            "eval_stop_prob_count": float(eval_stop_prob_count),
            "eval_stop_pos_prob_sum": eval_stop_pos_prob_sum,
            "eval_stop_pos_count": float(eval_stop_pos_count),
            "eval_stop_neg_prob_sum": eval_stop_neg_prob_sum,
            "eval_stop_neg_count": float(eval_stop_neg_count),
            "eval_stop_tp": eval_stop_tp,
            "eval_stop_fp": eval_stop_fp,
            "eval_stop_fn": eval_stop_fn,
            "eval_stop_tn": eval_stop_tn,
        }
        reduce_device = eval_loss.device if isinstance(eval_loss, torch.Tensor) else (
            torch.device(f"cuda:{local_rank}") if (train_config.enable_fsdp or train_config.enable_ddp) else torch.device("cuda:0")
        )
        for name, value in stop_reduce.items():
            tensor_val = torch.tensor(float(value), device=reduce_device)
            dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
            stop_reduce[name] = float(tensor_val.item())
        eval_stop_loss = stop_reduce["eval_stop_loss"]
        eval_stop_loss_count = int(round(stop_reduce["eval_stop_loss_count"]))
        eval_stop_prob_sum = stop_reduce["eval_stop_prob_sum"]
        eval_stop_prob_count = int(round(stop_reduce["eval_stop_prob_count"]))
        eval_stop_pos_prob_sum = stop_reduce["eval_stop_pos_prob_sum"]
        eval_stop_pos_count = int(round(stop_reduce["eval_stop_pos_count"]))
        eval_stop_neg_prob_sum = stop_reduce["eval_stop_neg_prob_sum"]
        eval_stop_neg_count = int(round(stop_reduce["eval_stop_neg_count"]))
        eval_stop_tp = stop_reduce["eval_stop_tp"]
        eval_stop_fp = stop_reduce["eval_stop_fp"]
        eval_stop_fn = stop_reduce["eval_stop_fn"]
        eval_stop_tn = stop_reduce["eval_stop_tn"]

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_acc = eval_acc / len(eval_dataloader)
    eval_epoch_audio_acc = eval_audio_acc / len(eval_dataloader)
    if train_config.enable_fsdp or train_config.enable_ddp:
        eval_epoch_loss = eval_epoch_loss/world_size
        eval_epoch_acc = eval_epoch_acc/world_size
        eval_epoch_audio_acc = eval_epoch_audio_acc/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    eval_extra = {"has_dpo_metrics": False, "has_stop_metrics": False}
    if eval_dpo_count > 0:
        eval_extra = {
            "has_dpo_metrics": True,
            "has_stop_metrics": False,
            "eval_dpo_loss": eval_dpo_loss / eval_dpo_count,
            "eval_sft_loss": eval_sft_loss / eval_dpo_count,
            "eval_chosen_logp": eval_chosen_logp / eval_dpo_count,
            "eval_rejected_logp": eval_rejected_logp / eval_dpo_count,
            "eval_ref_chosen_logp": eval_ref_chosen_logp / eval_dpo_count,
            "eval_ref_rejected_logp": eval_ref_rejected_logp / eval_dpo_count,
            "eval_policy_margin": eval_policy_margin / eval_dpo_count,
            "eval_ref_margin": eval_ref_margin / eval_dpo_count,
            "eval_dpo_margin": eval_dpo_margin / eval_dpo_count,
            "eval_policy_pref_rate": eval_policy_pref_rate / eval_dpo_count,
            "eval_dpo_pref_rate": eval_dpo_pref_rate / eval_dpo_count,
            "eval_pair_margin": (eval_pair_margin / eval_pair_margin_count) if eval_pair_margin_count > 0 else None,
            "eval_pair_margin_pos_rate": (eval_pair_margin_pos_rate / eval_pair_margin_count) if eval_pair_margin_count > 0 else None,
        }
    if eval_stop_prob_count > 0:
        stop_precision = (float(eval_stop_tp) / float(eval_stop_tp + eval_stop_fp)) if (eval_stop_tp + eval_stop_fp) > 0 else 0.0
        stop_recall = (float(eval_stop_tp) / float(eval_stop_tp + eval_stop_fn)) if (eval_stop_tp + eval_stop_fn) > 0 else 0.0
        stop_f1 = (
            2.0 * stop_precision * stop_recall / (stop_precision + stop_recall)
        ) if (stop_precision + stop_recall) > 0 else 0.0
        stop_pr_auc = None
        stop_roc_auc = None
        if not (train_config.enable_fsdp or train_config.enable_ddp) and len(eval_stop_prob_chunks) > 0:
            all_stop_probs = torch.cat(eval_stop_prob_chunks, dim=0)
            all_stop_targets = torch.cat(eval_stop_target_chunks, dim=0)
            stop_auc_metrics = _compute_binary_auc_metrics(all_stop_probs, all_stop_targets)
            stop_pr_auc = stop_auc_metrics["pr_auc"]
            stop_roc_auc = stop_auc_metrics["roc_auc"]
        eval_extra.update(
            {
                "has_stop_metrics": True,
                "eval_stop_loss": (eval_stop_loss / eval_stop_loss_count) if eval_stop_loss_count > 0 else None,
                "eval_stop_pr_auc": stop_pr_auc,
                "eval_stop_roc_auc": stop_roc_auc,
                "eval_stop_precision@0.5": stop_precision,
                "eval_stop_recall@0.5": stop_recall,
                "eval_stop_f1@0.5": stop_f1,
                "eval_stop_pred_pos_mean": (eval_stop_pos_prob_sum / eval_stop_pos_count) if eval_stop_pos_count > 0 else None,
                "eval_stop_pred_neg_mean": (eval_stop_neg_prob_sum / eval_stop_neg_count) if eval_stop_neg_count > 0 else None,
            }
        )

    if train_config.enable_fsdp or train_config.enable_ddp:
        if local_rank==0:
            logger.info(f" {eval_ppl=} {eval_epoch_loss=} {eval_epoch_acc=} {eval_epoch_audio_acc=}")
            if eval_extra["has_dpo_metrics"]:
                logger.info(f"[DPO][VAL] metrics={json.dumps(eval_extra, ensure_ascii=False)}")
            if eval_extra["has_stop_metrics"]:
                logger.info(f"[STOP][VAL] metrics={json.dumps(eval_extra, ensure_ascii=False)}")
    else:
        logger.info(f" {eval_ppl=} {eval_epoch_loss=} {eval_epoch_acc=} {eval_epoch_audio_acc=}")
        if eval_extra["has_dpo_metrics"]:
            logger.info(f"[DPO][VAL] metrics={json.dumps(eval_extra, ensure_ascii=False)}")
        if eval_extra["has_stop_metrics"]:
            logger.info(f"[STOP][VAL] metrics={json.dumps(eval_extra, ensure_ascii=False)}")

    return eval_ppl, eval_epoch_loss, eval_epoch_acc, eval_epoch_audio_acc, eval_extra

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                logger.info(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        logger.info(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    log model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"--> {config.model_name} has {total_params / 1e6} Million params\n")

def print_module_size(module, module_name, rank: int = 0) -> None:
    """
    Print module name, the number of trainable parameters and initialization time.

    Args:
        module: The PyTorch module.
        module_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Module {module_name}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"--> {module_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )
    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                logger.info(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                logger.info(f"FP16 enabled")
        else:
            logger.info(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        logger.info(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            logger.info(f"training params are saved in {file_name}")
