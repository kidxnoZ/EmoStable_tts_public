import torch
import sys
import logging
from collections import Counter
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from speech_dataset_tts import SpeechDatasetJsonl

logger = logging.getLogger(__name__)


class SpeechDpoPairDatasetJsonl(SpeechDatasetJsonl):
    """DPO pair dataset that reuses SpeechDatasetJsonl sequence construction."""

    _AUDIO_TOKEN_KEYS = (
        "generated_audio_token_ids",
        "generated_audio_tokens",
        "answer_cosyvoice_speech_token",
        "audio_tokens",
    )

    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__(dataset_config=dataset_config, tokenizer=tokenizer, split=split)
        self._apply_emotion_filter()

    @staticmethod
    def _normalize_emotion(emotion):
        if emotion is None:
            return None
        emotion_norm = str(emotion).strip().lower()
        return emotion_norm if emotion_norm else None

    def _extract_pair_emotion(self, pair):
        if not isinstance(pair, dict):
            return None

        for key in ("emotion", "raw_emotion"):
            emo = self._normalize_emotion(pair.get(key))
            if emo is not None:
                return emo

        for key in ("chosen", "rejected"):
            candidate = pair.get(key, None)
            if not isinstance(candidate, dict):
                continue
            for emotion_key in ("emotion", "raw_emotion"):
                emo = self._normalize_emotion(candidate.get(emotion_key))
                if emo is not None:
                    return emo
        return None

    def _apply_emotion_filter(self):
        excluded_raw = self.dataset_config.get("dpo_excluded_emotions", [])
        excluded = set()
        for emotion in excluded_raw:
            emotion_norm = self._normalize_emotion(emotion)
            if emotion_norm is not None:
                excluded.add(emotion_norm)

        if len(excluded) == 0:
            logger.info("[DPO-DATA] no emotion filter applied (dpo_excluded_emotions is empty).")
            return

        before = len(self.data_list)
        filtered = []
        removed_counter = Counter()
        unknown_kept = 0

        for pair in self.data_list:
            emo = self._extract_pair_emotion(pair)
            if emo is None:
                filtered.append(pair)
                unknown_kept += 1
                continue
            if emo in excluded:
                removed_counter[emo] += 1
                continue
            filtered.append(pair)

        self.data_list = filtered
        removed_total = before - len(self.data_list)
        logger.info(
            "[DPO-DATA] emotion filter applied: before=%d after=%d removed=%d excluded=%s removed_breakdown=%s unknown_kept=%d",
            before,
            len(self.data_list),
            removed_total,
            sorted(list(excluded)),
            dict(sorted(removed_counter.items())),
            unknown_kept,
        )

    def _extract_audio_tokens(self, candidate):
        if not isinstance(candidate, dict):
            return None
        for key in self._AUDIO_TOKEN_KEYS:
            tokens = candidate.get(key)
            if isinstance(tokens, list) and len(tokens) > 0:
                try:
                    return [int(x) for x in tokens]
                except Exception:
                    continue
        return None

    def _build_single_sample(self, record):
        # Reuse base __getitem__ logic without modifying baseline dataset implementation.
        backup = self.data_list
        self.data_list = [record]
        try:
            sample = super().__getitem__(0)
        finally:
            self.data_list = backup
        return sample

    def __getitem__(self, index):
        pair = self.data_list[index]
        chosen = pair.get("chosen", {})
        rejected = pair.get("rejected", {})

        chosen_audio_tokens = self._extract_audio_tokens(chosen)
        rejected_audio_tokens = self._extract_audio_tokens(rejected)
        if chosen_audio_tokens is None or rejected_audio_tokens is None:
            key = pair.get("key", "")
            raise ValueError(
                f"DPO pair missing audio token sequence for key={key}. "
                "Expected chosen/rejected to contain one of "
                f"{self._AUDIO_TOKEN_KEYS}."
            )

        source_text = pair.get("source_text") or pair.get("target_text") or ""
        target_text = pair.get("target_text") or source_text
        emotion_text_prompt = pair.get("emotion_text_prompt")
        key = pair.get("key")

        chosen_record = {
            "key": key,
            "source_text": source_text,
            "target_text": target_text,
            "emotion_text_prompt": emotion_text_prompt,
            "answer_cosyvoice_speech_token": chosen_audio_tokens,
            "neutral_speaker_wav": pair.get("neutral_speaker_wav"),
        }
        rejected_record = {
            "key": key,
            "source_text": source_text,
            "target_text": target_text,
            "emotion_text_prompt": emotion_text_prompt,
            "answer_cosyvoice_speech_token": rejected_audio_tokens,
            "neutral_speaker_wav": pair.get("neutral_speaker_wav"),
        }

        chosen_sample = self._build_single_sample(chosen_record)
        rejected_sample = self._build_single_sample(rejected_record)

        score_margin = pair.get("score_margin", None)
        try:
            score_margin = float(score_margin) if score_margin is not None else None
        except Exception:
            score_margin = None

        return {
            "key": key,
            "pair_score_margin": score_margin,
            "chosen": chosen_sample,
            "rejected": rejected_sample,
        }

    def collator(self, samples):
        chosen_samples = [s["chosen"] for s in samples]
        rejected_samples = [s["rejected"] for s in samples]

        chosen_batch = super().collator(chosen_samples)
        rejected_batch = super().collator(rejected_samples)

        result = {}
        for k, v in chosen_batch.items():
            result[f"chosen_{k}"] = v
        for k, v in rejected_batch.items():
            result[f"rejected_{k}"] = v

        keys = [s.get("key") for s in samples]
        result["pair_keys"] = keys

        margins = [s.get("pair_score_margin", None) for s in samples]
        margin_values = [m for m in margins if m is not None]
        if margin_values:
            result["pair_score_margin"] = torch.tensor(
                [0.0 if m is None else float(m) for m in margins],
                dtype=torch.float32,
            )
        else:
            result["pair_score_margin"] = None

        return result


def get_speech_dataset_dpo(dataset_config, tokenizer, split):
    dataset = SpeechDpoPairDatasetJsonl(dataset_config, tokenizer, split)
    return dataset
