#!/usr/bin/env python3
"""Shared utilities for EmoVoice DPO data pipeline scripts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np


EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unk",
]

EMOTION_ALIAS = {
    "disgust": "disgusted",
    "fear": "fearful",
    "surprise": "surprised",
    "calm": "other",
    "cry": "other",
    "excited": "other",
}


def str2bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: str | Path) -> Iterator[Dict]:
    bad_prefix = "\ufeff\u00a0 \t\r\n"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.lstrip(bad_prefix).strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str | Path) -> List[Dict]:
    return list(iter_jsonl(path))


def dump_jsonl(path: str | Path, records: Iterable[Dict]) -> int:
    ensure_parent_dir(path)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def canonical_emotion(label: Optional[str]) -> str:
    if label is None:
        return "other"
    key = str(label).strip().lower()
    key = EMOTION_ALIAS.get(key, key)
    return key if key in EMOTION_LABELS else "other"


def normalize_probs(scores: List[float], labels: List[str]) -> Dict[str, float]:
    if len(scores) != len(labels):
        raise ValueError(f"scores length ({len(scores)}) != labels length ({len(labels)})")
    probs = np.asarray(scores, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total <= 0:
        probs = np.ones_like(probs) / max(len(probs), 1)
    else:
        probs = probs / total
    return {lab: float(p) for lab, p in zip(labels, probs.tolist())}


def safe_log_prob(prob: float, eps: float = 1e-12) -> float:
    return float(math.log(max(float(prob), eps)))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(vec_a, dtype=np.float64).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def resolve_existing_path(
    raw_path: str,
    prefixes: List[str | Path],
) -> str:
    raw_path = str(raw_path)
    p = Path(raw_path)
    if p.is_file():
        return str(p.resolve())
    for prefix in prefixes:
        candidate = Path(prefix) / raw_path
        if candidate.is_file():
            return str(candidate.resolve())
    return raw_path


def load_dataset_by_key(dataset_jsonl: str | Path) -> Dict[str, Dict]:
    data = {}
    for item in iter_jsonl(dataset_jsonl):
        key = item.get("key")
        if key is not None:
            data[str(key)] = item
    return data
