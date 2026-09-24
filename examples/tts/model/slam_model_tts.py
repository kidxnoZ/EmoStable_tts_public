import torch
import os
import json
import logging
import torch.nn.functional as F
from collections import Counter
from model.mtp import Qwen2MTPAudioDecoder
from model.future_mtp import Qwen2FutureTokenDecoder
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from typing import List, Optional
from slam_llm.utils.metric import compute_accuracy
from tqdm import tqdm
from utils.codec_utils import setup_codec
from utils.codec_utils import audio_decode_cosyvoice
from utils.codec_utils import layershift as layer_shift, simple_shift
from utils.projector_utils import setup_group_decode_adapter
from slam_llm.utils.config_utils import generate_peft_config
from peft import get_peft_model

logger = logging.getLogger(__name__)

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    
    encoder = None
    encoder_projector = None

    llm = setup_llm(train_config, model_config, **kwargs)

    codec_decoder = None
    if model_config.codec_decode:
        codec_decoder = setup_codec(train_config, model_config, **kwargs)

    group_decode_adapter = None
    use_mtp = bool(model_config.get("use_mtp", False))
    use_future_mtp = bool(model_config.get("use_future_mtp", False))
    if model_config.group_decode and not use_mtp and not use_future_mtp:
        group_decode_adapter = setup_group_decode_adapter(model_config, train_config, **kwargs)
        if train_config.freeze_group_decode_adapter:
            for name, param in group_decode_adapter.named_parameters():
                param.requires_grad = False
            group_decode_adapter.eval()

    model = slam_model_tts(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        codec_decoder,
        group_decode_adapter,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}\n".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    if train_config.use_peft:
        logger.info("setup peft for llm")
        peft_config = generate_peft_config(train_config)
        model.llm = get_peft_model(model.llm, peft_config)
        if int(os.environ.get("RANK", "0")) == 0:
            model.llm.print_trainable_parameters()

    if kwargs.get("peft_ckpt", None):
        logger.info("loading peft-stage ckpt from: {}\n".format(kwargs.get("peft_ckpt")))
        ckpt_dict = torch.load(kwargs.get("peft_ckpt"), map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    model._configure_stop_head_trainability()
        
    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class slam_model_tts(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        codec_decoder,
        group_decode_adapter,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )

        # resize llm embedding layer
        self.original_vocabsize = self.llm.lm_head.weight.size(0)
        if self.model_config.vocab_config.total_vocabsize != self.original_vocabsize:
            self.llm.resize_token_embeddings(self.model_config.vocab_config.total_vocabsize)
            if int(os.environ.get("RANK", "0")) == 0:
                logger.info("Resize llm embedding layer's vocab size to {}\n".format(self.model_config.vocab_config.total_vocabsize))

        self.codec_decoder = codec_decoder
        self.code_layer = self.model_config.vocab_config.code_layer
        self.group_decode_adapter = group_decode_adapter
        self.use_mtp = bool(self.model_config.get("use_mtp", False))
        self.use_future_mtp = bool(self.model_config.get("use_future_mtp", False))
        self.use_stop_head = bool(self.model_config.get("use_stop_head", False))
        self.mtp_num = self.model_config.get("mtp_num", None)
        self.mtp_loss_decay = float(self.model_config.get("mtp_loss_decay", 1.0))
        self.stop_pos_weight = float(self.model_config.get("stop_pos_weight", 96.0))
        self.stop_loss_weight = float(self.model_config.get("stop_loss_weight", 1.0))
        self.stop_min_step = int(self.model_config.get("stop_min_step", 48))
        self.stop_bias_lambda = float(self.model_config.get("stop_bias_lambda", 4.0))
        self.use_sampled_audio_prefix_training = bool(self.model_config.get("use_sampled_audio_prefix_training", False))
        self.sampled_audio_prefix_prob_start = float(self.model_config.get("sampled_audio_prefix_prob_start", 0.0))
        self.sampled_audio_prefix_prob_end = float(self.model_config.get("sampled_audio_prefix_prob_end", 0.2))
        self.sampled_audio_prefix_warmup_steps = int(self.model_config.get("sampled_audio_prefix_warmup_steps", 5000))
        self.sampled_audio_prefix_chunk_len = int(self.model_config.get("sampled_audio_prefix_chunk_len", 16))
        self.sampled_audio_prefix_sampling = str(self.model_config.get("sampled_audio_prefix_sampling", "greedy")).lower()
        self.freeze_stop_head_during_sampled_prefix = bool(self.model_config.get("freeze_stop_head_during_sampled_prefix", True))
        self.disable_stop_loss_during_sampled_prefix = bool(self.model_config.get("disable_stop_loss_during_sampled_prefix", True))
        self.mtp_audio_decoder = None
        self.future_mtp_num = self.model_config.get("future_mtp_num", None)
        self.future_mtp_loss_decay = float(self.model_config.get("future_mtp_loss_decay", 1.0))
        self.future_mtp_decoder = None
        self.stop_hidden_norm = None
        self.stop_raw_proj = None
        self.stop_mlp = None
        self.debug_train_sanity = bool(getattr(self.train_config, "debug_train_sanity", False)) or (
            str(os.environ.get("DEBUG_TRAIN_SANITY", "0")).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        )
        self.debug_sanity_model_max_calls = int(getattr(self.train_config, "debug_sanity_model_max_calls", 8))
        self._debug_seq_logp_calls = 0
        self._sampled_audio_prefix_train_calls = 0

        if self.use_sampled_audio_prefix_training:
            if self.sampled_audio_prefix_sampling != "greedy":
                raise NotImplementedError(
                    "sampled_audio_prefix_training currently only supports sampled_audio_prefix_sampling='greedy'."
                )
            if self.train_config.modeling_paradigm != "parallel":
                raise NotImplementedError(
                    "sampled_audio_prefix_training currently only supports modeling_paradigm='parallel'."
                )

        if self.use_mtp and self.use_future_mtp:
            raise ValueError("use_mtp and use_future_mtp are mutually exclusive.")

        if self.use_mtp:
            if self.train_config.modeling_paradigm != "parallel":
                raise NotImplementedError("use_mtp=True currently only supports modeling_paradigm='parallel'.")
            if self.code_layer <= 1:
                raise ValueError("use_mtp=True requires code_layer > 1.")
            if self.mtp_num is None:
                self.mtp_num = self.code_layer - 1
            if self.mtp_num != self.code_layer - 1:
                raise NotImplementedError(
                    f"use_mtp=True currently requires mtp_num == code_layer - 1, got mtp_num={self.mtp_num}, code_layer={self.code_layer}."
                )
            text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
            audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
            expected_total_vocabsize = text_vocab_size + audio_vocab_size
            if self.model_config.vocab_config.total_vocabsize != expected_total_vocabsize:
                raise ValueError(
                    "use_mtp=True expects total_vocabsize == padded_text_vocabsize + padded_audio_vocabsize "
                    f"(got total_vocabsize={self.model_config.vocab_config.total_vocabsize}, expected={expected_total_vocabsize})."
                )
            self.mtp_audio_decoder = Qwen2MTPAudioDecoder(
                qwen_config=self.llm.config,
                hidden_size=self.llm.config.hidden_size,
                audio_vocab_size=audio_vocab_size,
                mtp_num=self.mtp_num,
                # audio_pad_token=self.model_config.vocab_config.pad_a,
            )
            # self._initialize_mtp_audio_embedding(text_vocab_size, audio_vocab_size)
            logger.info(
                "Initialized MTP audio decoder: mtp_num=%d hidden_size=%d audio_vocab_size=%d mtp_loss_decay=%.4f",
                self.mtp_num,
                self.llm.config.hidden_size,
                audio_vocab_size,
                self.mtp_loss_decay,
            )

        if self.use_future_mtp:
            if self.train_config.modeling_paradigm != "parallel":
                raise NotImplementedError("use_future_mtp=True currently only supports modeling_paradigm='parallel'.")
            if self.code_layer != 1:
                raise ValueError(f"use_future_mtp=True requires code_layer == 1, got {self.code_layer}.")
            text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
            audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
            expected_total_vocabsize = text_vocab_size + audio_vocab_size
            if self.model_config.vocab_config.total_vocabsize != expected_total_vocabsize:
                raise ValueError(
                    "use_future_mtp=True expects total_vocabsize == padded_text_vocabsize + padded_audio_vocabsize "
                    f"(got total_vocabsize={self.model_config.vocab_config.total_vocabsize}, expected={expected_total_vocabsize})."
                )
            if self.future_mtp_num is None:
                self.future_mtp_num = 2
            self.future_mtp_decoder = Qwen2FutureTokenDecoder(
                qwen_config=self.llm.config,
                hidden_size=self.llm.config.hidden_size,
                audio_vocab_size=audio_vocab_size,
                mtp_num=self.future_mtp_num,
                audio_pad_token=self.model_config.vocab_config.pad_a,
            )
            self._initialize_future_mtp_audio_embedding(text_vocab_size, audio_vocab_size)
            logger.info(
                "Initialized future-token MTP decoder: future_mtp_num=%d hidden_size=%d audio_vocab_size=%d future_mtp_loss_decay=%.4f",
                self.future_mtp_num,
                self.llm.config.hidden_size,
                audio_vocab_size,
                self.future_mtp_loss_decay,
            )

        if self.use_stop_head:
            audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
            hidden_size = self.llm.config.hidden_size
            self.stop_hidden_norm = torch.nn.LayerNorm(hidden_size)
            self.stop_raw_proj = torch.nn.Linear(audio_vocab_size, 128)
            self.stop_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size + 128, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 1),
            )
            logger.info(
                "Initialized stop head: hidden_size=%d raw_proj_dim=%d stop_loss_weight=%.4f stop_pos_weight=%.4f stop_min_step=%d stop_bias_lambda=%.4f",
                hidden_size,
                128,
                self.stop_loss_weight,
                self.stop_pos_weight,
                self.stop_min_step,
                self.stop_bias_lambda,
            )

        self.stop_head_only_training = bool(
            self.use_stop_head
            and getattr(self.train_config, "freeze_llm", False)
            and not self.use_sampled_audio_prefix_training
        )
        if self.use_sampled_audio_prefix_training and getattr(self.train_config, "freeze_llm", False):
            logger.info(
                "Sampled-audio-prefix training is enabled with freeze_llm=True: stop-head-only mode is disabled, "
                "but the backbone LLM remains frozen by train_config.freeze_llm."
            )

    def _configure_stop_head_trainability(self):
        stop_prefixes = ("stop_hidden_norm.", "stop_raw_proj.", "stop_mlp.")
        if self.use_stop_head and self.stop_head_only_training:
            for name, param in self.named_parameters():
                param.requires_grad = name.startswith(stop_prefixes)
            logger.info("Stop-head-only training enabled: all parameters frozen except stop head modules.")
            return
        if self.use_stop_head and self.use_sampled_audio_prefix_training and self.freeze_stop_head_during_sampled_prefix:
            for name, param in self.named_parameters():
                if name.startswith(stop_prefixes):
                    param.requires_grad = False
            logger.info("Sampled-audio-prefix training enabled: stop head parameters frozen.")

    def _set_frozen_backbone_eval_mode(self):
        if not self.stop_head_only_training:
            return
        self.llm.eval()
        if self.encoder is not None:
            self.encoder.eval()
        if self.encoder_projector is not None:
            self.encoder_projector.eval()
        if self.mtp_audio_decoder is not None:
            self.mtp_audio_decoder.eval()
        if self.future_mtp_decoder is not None:
            self.future_mtp_decoder.eval()
        if self.group_decode_adapter is not None:
            self.group_decode_adapter.eval()

    def _build_stop_head_features(self, hidden_states, raw_audio_logits):
        if not self.use_stop_head:
            raise ValueError("Stop head is disabled.")
        hidden_states = self.stop_hidden_norm(hidden_states)
        raw_audio_logits = self.stop_raw_proj(raw_audio_logits)
        return torch.cat([hidden_states, raw_audio_logits], dim=-1)

    def _compute_stop_logits(self, hidden_states, raw_audio_logits):
        stop_features = self._build_stop_head_features(hidden_states, raw_audio_logits)
        return self.stop_mlp(stop_features).squeeze(-1)

    def _compute_stop_targets(self, audio_labels, eoa_id):
        if audio_labels is None:
            return None, None
        stop_target = audio_labels.eq(eoa_id).any(dim=1)[:, 1:]
        stop_valid = audio_labels.ne(-100).any(dim=1)[:, 1:]
        return stop_target.float(), stop_valid

   
    def _compute_stop_loss(self, stop_logits, audio_labels):
        eoa_id = self.model_config.vocab_config.eoa
        stop_target, stop_valid = self._compute_stop_targets(audio_labels, eoa_id)
    # def _compute_stop_loss(self, stop_logits, stop_target, stop_valid):
        if stop_target is None or stop_valid is None:
            return stop_logits.new_zeros(())
        if stop_logits.shape != stop_target.shape:
            raise ValueError(
                f"stop_logits shape {tuple(stop_logits.shape)} must match stop_target shape {tuple(stop_target.shape)}."
            )
        pos_weight = torch.tensor(self.stop_pos_weight, dtype=stop_logits.dtype, device=stop_logits.device)
        stop_loss = F.binary_cross_entropy_with_logits(
            stop_logits,
            stop_target,
            pos_weight=pos_weight,
            reduction="none",
        )
        stop_valid = stop_valid.to(dtype=stop_loss.dtype)
        denom = stop_valid.sum()
        if denom.item() <= 0:
            return stop_logits.new_zeros(())
        return (stop_loss * stop_valid).sum() / denom

    def _summarize_stop_metrics(self, stop_logits, stop_target, stop_valid, stop_loss=None):
        metrics = {}
        if stop_loss is not None:
            metrics["stop_loss"] = stop_loss.detach()
        
        if stop_target is None or stop_valid is None:
            return metrics
        valid_mask = stop_valid.bool()
        if valid_mask.sum().item() <= 0:
            return metrics
        stop_probs = torch.sigmoid(stop_logits.detach())
        valid_probs = stop_probs[valid_mask]
        valid_targets = stop_target.detach()[valid_mask].float()
        metrics["stop_pos_rate"] = valid_targets.mean()
        metrics["stop_pred_mean"] = valid_probs.mean()
        pos_mask = valid_targets > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any():
            metrics["stop_pred_pos_mean"] = valid_probs[pos_mask].mean()
        if neg_mask.any():
            metrics["stop_pred_neg_mean"] = valid_probs[neg_mask].mean()
        return metrics

    def _get_llm_embed_tokens_module(self):
        if hasattr(self.llm.model, "embed_tokens"):
            return self.llm.model.embed_tokens
        if hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
            return self.llm.model.model.embed_tokens
        if hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "model") and hasattr(self.llm.model.model.model, "embed_tokens"):
            return self.llm.model.model.model.embed_tokens
        raise AttributeError("Unable to locate the backbone LLM input embedding module.")

    def _embed_input_ids(self, input_ids):
        embed_tokens = self._get_llm_embed_tokens_module()
        return embed_tokens(input_ids)

    def _compute_sampled_audio_prefix_prob(self):
        current_step = self._sampled_audio_prefix_train_calls
        self._sampled_audio_prefix_train_calls += 1
        if self.sampled_audio_prefix_warmup_steps <= 0:
            return self.sampled_audio_prefix_prob_end
        progress = min(max(float(current_step) / float(self.sampled_audio_prefix_warmup_steps), 0.0), 1.0)
        return self.sampled_audio_prefix_prob_start + (
            self.sampled_audio_prefix_prob_end - self.sampled_audio_prefix_prob_start
        ) * progress

    def _sample_audio_token_from_logits(self, logits):
        if self.sampled_audio_prefix_sampling != "greedy":
            raise NotImplementedError(
                "sampled_audio_prefix_training currently only supports sampled_audio_prefix_sampling='greedy'."
            )
        return torch.argmax(logits, dim=-1)

    def _build_sampled_audio_prefix_input_ids(self, input_ids, attention_mask, labels):
        if (
            not self.training
            or not self.use_sampled_audio_prefix_training
            or input_ids is None
            or labels is None
            or self.train_config.modeling_paradigm != "parallel"
        ):
            return input_ids, None

        sampled_prob = self._compute_sampled_audio_prefix_prob()
        stats = {
            "sampled_audio_prefix_prob": float(sampled_prob),
            "sampled_audio_prefix_replaced": 0,
            "sampled_audio_prefix_candidates": 0,
            "sampled_audio_prefix_samples": 0,
        }
        if sampled_prob <= 0.0:
            return input_ids, stats

        audio_labels = labels[:, : self.code_layer]
        valid_audio_steps = audio_labels.ne(-100).any(dim=1)
        if not valid_audio_steps.any():
            return input_ids, stats

        working_input_ids = input_ids.clone()
        reference_input_ids = input_ids.clone()
        model_attention_mask = attention_mask
        if model_attention_mask is None:
            model_attention_mask = torch.ones(
                input_ids.shape[0],
                input_ids.shape[-1],
                dtype=torch.long,
                device=input_ids.device,
            )

        llm_was_training = self.llm.training
        mtp_was_training = self.mtp_audio_decoder.training if self.mtp_audio_decoder is not None else None
        try:
            self.llm.eval()
            if self.mtp_audio_decoder is not None:
                self.mtp_audio_decoder.eval()

            with torch.no_grad():
                for batch_idx in range(input_ids.shape[0]):
                    valid_positions = torch.nonzero(valid_audio_steps[batch_idx], as_tuple=False).flatten()
                    if valid_positions.numel() <= 0:
                        continue
                    suffix_positions = valid_positions[-self.sampled_audio_prefix_chunk_len :].tolist()
                    for pos in suffix_positions:
                        if pos <= 0:
                            continue
                        stats["sampled_audio_prefix_candidates"] += self.code_layer
                        if torch.rand((), device=input_ids.device).item() >= sampled_prob:
                            continue

                        sample_embeds = self._embed_input_ids(working_input_ids[batch_idx : batch_idx + 1])
                        sample_embeds = torch.mean(sample_embeds, dim=1)
                        sample_outputs = self.llm(
                            inputs_embeds=sample_embeds,
                            attention_mask=model_attention_mask[batch_idx : batch_idx + 1],
                            output_hidden_states=self.use_mtp or self.use_future_mtp or self.use_stop_head,
                            return_dict=True,
                        )
                        _, sample_xa = self._build_parallel_logits(
                            sample_outputs,
                            model_attention_mask[batch_idx : batch_idx + 1],
                            labels[batch_idx : batch_idx + 1],
                        )
                        prev_pos = pos - 1
                        for layer_idx in range(self.code_layer):
                            label_token = audio_labels[batch_idx, layer_idx, pos]
                            if int(label_token.item()) < 0:
                                continue
                            sampled_token = self._sample_audio_token_from_logits(sample_xa[layer_idx][0, prev_pos, :])
                            sampled_token_value = int(sampled_token.item())
                            reference_shifted_token = int(reference_input_ids[batch_idx, layer_idx, pos].item())
                            reference_label_token = int(label_token.item())
                            shift_offset = reference_shifted_token - reference_label_token
                            working_input_ids[batch_idx, layer_idx, pos] = sampled_token_value + shift_offset
                            stats["sampled_audio_prefix_samples"] += 1
                            if sampled_token_value != reference_label_token:
                                stats["sampled_audio_prefix_replaced"] += 1
        finally:
            if llm_was_training:
                self.llm.train()
            if self.mtp_audio_decoder is not None and mtp_was_training is not None:
                if mtp_was_training:
                    self.mtp_audio_decoder.train()
                else:
                    self.mtp_audio_decoder.eval()

        return working_input_ids, stats

    # def _get_llm_input_embedding_weight(self):
    #     if hasattr(self.llm.model, "embed_tokens"):
    #         return self.llm.model.embed_tokens.weight
    #     if hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
    #         return self.llm.model.model.embed_tokens.weight
    #     if hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "model") and hasattr(self.llm.model.model.model, "embed_tokens"):
    #         return self.llm.model.model.model.embed_tokens.weight
    #     raise AttributeError("Unable to locate the backbone LLM input embedding table.")

    def _initialize_audio_embedding_from_llm(self, decoder_module, text_vocab_size, audio_vocab_size):
        if decoder_module is None or not hasattr(decoder_module, "audio_embedding"):
            return
        with torch.no_grad():
            llm_input_weight = self._get_llm_input_embedding_weight()
            audio_embedding_slice = llm_input_weight[text_vocab_size : text_vocab_size + audio_vocab_size]
            decoder_module.audio_embedding.weight.copy_(audio_embedding_slice)
            pad_a = self.model_config.vocab_config.pad_a
            if 0 <= pad_a < audio_vocab_size:
                decoder_module.audio_embedding.weight[pad_a].zero_()

    # def _initialize_mtp_audio_embedding(self, text_vocab_size, audio_vocab_size):
    #     if not self.use_mtp or self.mtp_audio_decoder is None:
    #         return
    #     self._initialize_audio_embedding_from_llm(self.mtp_audio_decoder, text_vocab_size, audio_vocab_size)

    def _initialize_future_mtp_audio_embedding(self, text_vocab_size, audio_vocab_size):
        if not self.use_future_mtp or self.future_mtp_decoder is None:
            return
        self._initialize_audio_embedding_from_llm(self.future_mtp_decoder, text_vocab_size, audio_vocab_size)


    # def _ensure_mtp_audio_histories(self, mtp_hidden_history, mtp_audio_histories, pad_a):
    #     batch_size, history_len = mtp_hidden_history.shape[:2]
    #     device = mtp_hidden_history.device
    #     if mtp_audio_histories is None:
    #         return [
    #             torch.full((batch_size, history_len), pad_a, dtype=torch.long, device=device)
    #             for _ in range(self.mtp_num)
    #         ]

    #     if len(mtp_audio_histories) != self.mtp_num:
    #         raise ValueError(
    #             f"Expected {self.mtp_num} MTP audio history tensors, got {len(mtp_audio_histories)}."
    #         )

    #     updated_histories = []
    #     for layer_idx, audio_history in enumerate(mtp_audio_histories):
    #         if audio_history.shape[0] != batch_size:
    #             raise ValueError(
    #                 f"MTP audio history batch size mismatch for layer {layer_idx}: "
    #                 f"{audio_history.shape[0]} vs {batch_size}."
    #             )
    #         if audio_history.shape[1] > history_len:
    #             audio_history = audio_history[:, -history_len:]
    #         elif audio_history.shape[1] < history_len:
    #             pad_columns = torch.full(
    #                 (batch_size, history_len - audio_history.shape[1]),
    #                 pad_a,
    #                 dtype=audio_history.dtype,
    #                 device=audio_history.device,
    #             )
    #             audio_history = torch.cat([audio_history, pad_columns], dim=1)
    #         updated_histories.append(audio_history.to(device=device, dtype=torch.long))
    #     return updated_histories

    # def _build_mtp_layer_generate_logits(self, mtp_hidden_history, mtp_audio_histories, target_audio_layer):
    #     if not self.use_mtp or self.mtp_audio_decoder is None:
    #         raise RuntimeError("_build_mtp_layer_generate_logits should only be used when use_mtp=True.")
    #     if mtp_hidden_history is None:
    #         raise ValueError("mtp_hidden_history must not be None when building MTP generate logits.")
    #     if target_audio_layer <= 0 or target_audio_layer >= self.code_layer:
    #         raise ValueError(
    #             f"target_audio_layer must be in [1, {self.code_layer - 1}], got {target_audio_layer}."
    #         )
    #     conditioning_histories = [mtp_audio_histories[i] for i in range(target_audio_layer)]
    #     return self.mtp_audio_decoder.infer(mtp_hidden_history, conditioning_histories)[-1][0]

    def _build_parallel_logits(self, model_outputs, attention_mask, labels):
        x_ori = model_outputs.logits
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        xt = x_ori[..., :text_vocab_size]

        if self.use_future_mtp:
            if model_outputs.hidden_states is None:
                raise ValueError("use_future_mtp=True requires output_hidden_states=True from the backbone LLM.")
            if labels is None:
                raise ValueError("use_future_mtp=True expects labels during forward so future-audio supervision can be computed.")
            audio_labels = labels[:, 0, :]
            xa = [x_ori[..., text_vocab_size:]]
            xa.extend(self.future_mtp_decoder(model_outputs.hidden_states[-1], attention_mask, audio_labels))
            return xt, xa

        if self.use_mtp:
            if model_outputs.hidden_states is None:
                raise ValueError("use_mtp=True requires output_hidden_states=True from the backbone LLM.")
            if labels is None:
                raise ValueError("use_mtp=True currently expects labels during forward so audio supervision can be computed.")
            audio_labels = labels[:, :self.code_layer]
            xa = [x_ori[..., text_vocab_size:]]
            xa.extend(self.mtp_audio_decoder(model_outputs.hidden_states[-1], attention_mask, audio_labels))
            return xt, xa

        xa = []
        if self.group_decode_adapter is not None:
            x_audio_ori = x_ori[..., text_vocab_size:]
            x_audio = self.group_decode_adapter(x_audio_ori)
            for i in range(self.code_layer):
                xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])
        else:
            for i in range(self.code_layer):
                xa.append(x_ori[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)])
        return xt, xa

    def _build_generate_step_logits(self, logits, mtp_hidden_history=None):
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        xt_logits = logits[..., :text_vocab_size]

        if self.use_future_mtp:
            return xt_logits, [logits[..., text_vocab_size:]]

        if self.use_mtp:
            if mtp_hidden_history is None:
                raise ValueError("use_mtp=True requires mtp_hidden_history during generation.")
            xa_logits = [logits[..., text_vocab_size:]]
            xa_logits.extend(layer_logits[0] for layer_logits in self.mtp_audio_decoder.infer(mtp_hidden_history))
            # if mtp_conditioning_audio_histories is not None:
            #     if mtp_hidden_history is None:
            #         raise ValueError("use_mtp=True requires mtp_hidden_history during generation.")
            #     xa_logits.extend(
            #         layer_logits[0]
            #         for layer_logits in self.mtp_audio_decoder.infer(
            #             mtp_hidden_history,
            #             mtp_conditioning_audio_histories,
            #         )
            #     )
            return xt_logits, xa_logits

        if self.group_decode_adapter is not None:
            x_audio = self.group_decode_adapter(logits[..., text_vocab_size:])
            xa_logits = [x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
        else:
            xa_logits = [
                logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)]
                for i in range(self.code_layer)
            ]
        return xt_logits, xa_logits

    def _format_debug_audio_summary(self, prefix, logits, selected_token, topk, eoa, pad_a):
        audio_probs, audio_topk_entries = self._debug_topk_entries(
            logits[-1, :],
            topk,
            kind="audio",
            eoa=eoa,
            pad_a=pad_a,
        )
        selected_token = int(selected_token)
        return (
            f"{prefix}={self._debug_audio_token_repr(selected_token, eoa, pad_a)} "
            f"sel_p={float(audio_probs[selected_token].item()):.4f} "
            f"p_eoa={float(audio_probs[eoa].item()):.4f} "
            f"topk=[{', '.join(audio_topk_entries)}]"
        )

    def _debug_text_token_repr(self, token_id, eot, pad_t):
        token_id = int(token_id)
        if token_id == eot:
            return "<EOT>"
        if token_id == pad_t:
            return "<PAD_T>"
        try:
            token_text = self.tokenizer.decode([token_id], add_special_tokens=False, skip_special_tokens=False)
        except Exception:
            token_text = ""
        token_text = token_text.replace("\n", "\\n")
        if token_text == "":
            token_text = "<EMPTY>"
        return f"{token_id}:{token_text}"

    @staticmethod
    def _debug_audio_token_repr(token_id, eoa, pad_a):
        token_id = int(token_id)
        if token_id == eoa:
            return "<EOA>"
        if token_id == pad_a:
            return "<PAD_A>"
        return str(token_id)

    @staticmethod
    def _debug_cache_length(cache_obj):
        if cache_obj is None:
            return 0
        if hasattr(cache_obj, "get_seq_length"):
            return int(cache_obj.get_seq_length())
        if isinstance(cache_obj, tuple) and len(cache_obj) > 0 and len(cache_obj[0]) > 0:
            return int(cache_obj[0][0].shape[2])
        return 0

    def _debug_topk_entries(self, logits, topk, kind, eot=None, pad_t=None, eoa=None, pad_a=None):
        probs = F.softmax(logits.float(), dim=-1)
        topk = min(max(int(topk), 1), probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs, k=topk)
        entries = []
        for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            if kind == "text":
                token_repr = self._debug_text_token_repr(idx, eot, pad_t)
            else:
                token_repr = self._debug_audio_token_repr(idx, eoa, pad_a)
            entries.append(f"{token_repr}:{prob:.4f}")
        return probs, entries

    def _debug_audio_distribution_stats(self, logits, topk, eoa, pad_a):
        probs, topk_entries = self._debug_topk_entries(
            logits,
            topk,
            kind="audio",
            eoa=eoa,
            pad_a=pad_a,
        )
        top1_id = int(torch.argmax(probs).item())
        top1_prob = float(probs[top1_id].item())
        entropy = float((-(probs * torch.log(torch.clamp(probs, min=1e-12))).sum()).item())
        p_eoa = float(probs[eoa].item())
        return {
            "top1_id": top1_id,
            "top1_prob": top1_prob,
            "entropy": entropy,
            "p_eoa": p_eoa,
            "topk_entries": topk_entries,
        }

    @staticmethod
    def _config_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        value = str(value).strip().lower()
        if value in ("1", "true", "yes", "y", "on"):
            return True
        if value in ("0", "false", "no", "n", "off"):
            return False
        return bool(default)

    @staticmethod
    def _collapse_tail_silence_run(flag_history):
        run = 0
        for item in reversed(flag_history):
            if item.get("flag", False):
                run += 1
            else:
                break
        return run

    @staticmethod
    def _looks_like_hard_case_path(path_value):
        if path_value in (None, "", "None"):
            return False
        path_value = str(path_value).lower()
        return "hard_case" in path_value or "hardcase" in path_value

    @staticmethod
    def _detect_audio_collapse(
        top1_histories,
        entropy_histories,
        current_audio_stats,
        tail_silence_run,
        window_size,
        small_set_size,
        entropy_threshold,
        tail_silence_patience,
        attractor_min_step,
        step,
    ):
        state = {
            "collapse_active": False,
            "triggered_layers": [],
            "tail_silent": bool(tail_silence_run >= tail_silence_patience),
            "tail_silence_run": int(tail_silence_run),
            "attractors_by_layer": {},
            "layers": {},
        }
        if step < attractor_min_step or window_size <= 0:
            return state

        dominance_threshold = max(2, (3 * window_size + 3) // 4)
        known_attractor_threshold = max(2, window_size // 2)
        attractor_count_threshold = max(2, window_size // 4)
        known_attractor_tokens = {66, 40, 113, 685, 1089}

        for layer_idx, layer_stats in enumerate(current_audio_stats):
            top1_window_source = list(top1_histories[layer_idx]) + [int(layer_stats["top1_id"])]
            entropy_window_source = list(entropy_histories[layer_idx]) + [float(layer_stats["entropy"])]
            layer_key = f"a{layer_idx}"
            if len(top1_window_source) < window_size or len(entropy_window_source) < window_size:
                state["layers"][layer_key] = {
                    "triggered": False,
                    "reasons": [],
                    "window_ready": False,
                    "unique_top1": None,
                    "dominant_token": None,
                    "dominant_count": None,
                    "attractors": [],
                }
                continue

            top1_window = top1_window_source[-window_size:]
            entropy_window = entropy_window_source[-window_size:]
            counts = Counter(top1_window)
            dominant_token, dominant_count = counts.most_common(1)[0]
            current_top1 = int(layer_stats["top1_id"])
            current_top1_prob = float(layer_stats["top1_prob"])
            current_entropy = float(layer_stats["entropy"])
            unique_top1 = len(counts)
            small_set = unique_top1 <= small_set_size
            low_entropy = all(value <= entropy_threshold for value in entropy_window)
            dominant = dominant_count >= dominance_threshold
            known_dominant = any(
                counts.get(token_id, 0) >= known_attractor_threshold
                for token_id in known_attractor_tokens
            )
            current_known_attractor = (
                current_top1 in known_attractor_tokens
                and current_entropy <= entropy_threshold
            )
            current_peaked_attractor = (
                current_top1_prob >= 0.75
                and current_entropy <= entropy_threshold
            )

            reasons = []
            if small_set:
                reasons.append("small_top1_set")
            if low_entropy:
                reasons.append("low_entropy")
            if dominant:
                reasons.append("dominant_token")
            if known_dominant:
                reasons.append("known_attractor")
            if current_known_attractor:
                reasons.append("current_known_attractor_low_entropy")
            if current_peaked_attractor:
                reasons.append("current_peaked_low_entropy")
            if state["tail_silent"]:
                reasons.append("tail_silent")

            triggered = (
                (small_set and (low_entropy or dominant or state["tail_silent"]))
                or (low_entropy and (dominant or known_dominant))
                or (state["tail_silent"] and (small_set or dominant or known_dominant))
                or current_known_attractor
                or current_peaked_attractor
            )
            attractors = sorted(
                token_id
                for token_id, count in counts.items()
                if count >= attractor_count_threshold
            )
            if triggered and (current_known_attractor or current_peaked_attractor):
                attractors = sorted(set(attractors + [current_top1]))
            if triggered and not attractors:
                attractors = [int(dominant_token)]

            state["layers"][layer_key] = {
                "triggered": bool(triggered),
                "reasons": reasons if triggered else [],
                "window_ready": True,
                "window_top1": [int(token_id) for token_id in top1_window],
                "current_top1": int(current_top1),
                "current_top1_prob": float(current_top1_prob),
                "unique_top1": int(unique_top1),
                "dominant_token": int(dominant_token),
                "dominant_count": int(dominant_count),
                "entropy_max": float(max(entropy_window)),
                "entropy_min": float(min(entropy_window)),
                "attractors": [int(token_id) for token_id in attractors],
            }
            if triggered:
                state["triggered_layers"].append(layer_key)
                state["attractors_by_layer"][layer_key] = [int(token_id) for token_id in attractors]

        state["collapse_active"] = len(state["triggered_layers"]) > 0
        return state

    @staticmethod
    def _apply_attractor_penalty(xa_logits, collapse_state, penalty, eoa, pad_a):
        applied = {}
        if penalty <= 0.0 or not collapse_state.get("collapse_active", False):
            return applied

        for layer_key, tokens in collapse_state.get("attractors_by_layer", {}).items():
            try:
                layer_idx = int(layer_key[1:])
            except (TypeError, ValueError):
                continue
            if layer_idx < 0 or layer_idx >= len(xa_logits):
                continue
            valid_tokens = []
            vocab_size = xa_logits[layer_idx].shape[-1]
            for token_id in tokens:
                token_id = int(token_id)
                if token_id in (int(eoa), int(pad_a)):
                    continue
                if 0 <= token_id < vocab_size:
                    valid_tokens.append(token_id)
            if not valid_tokens:
                continue
            token_tensor = torch.tensor(valid_tokens, dtype=torch.long, device=xa_logits[layer_idx].device)
            xa_logits[layer_idx][-1, token_tensor] = xa_logits[layer_idx][-1, token_tensor] - float(penalty)
            applied[layer_key] = [
                {"token": int(token_id), "penalty": float(penalty)}
                for token_id in valid_tokens
            ]
        return applied

    @staticmethod
    def _debug_first_small_set_step(token_history, window_size, small_set_size):
        if window_size <= 0 or len(token_history) < window_size:
            return None
        for start in range(len(token_history) - window_size + 1):
            if len(set(token_history[start:start + window_size])) <= small_set_size:
                return start
        return None

    @staticmethod
    def _debug_first_low_entropy_step(entropy_history, window_size, threshold):
        if window_size <= 0 or len(entropy_history) < window_size:
            return None
        for start in range(len(entropy_history) - window_size + 1):
            window = entropy_history[start:start + window_size]
            if all(value <= threshold for value in window):
                return start
        return None

    @staticmethod
    def _debug_first_true_window(flag_history, window_size):
        if window_size <= 0 or len(flag_history) < window_size:
            return None
        for start in range(len(flag_history) - window_size + 1):
            window = flag_history[start:start + window_size]
            if all(item["flag"] for item in window):
                return int(window[0]["step"])
        return None

    @staticmethod
    def _debug_pick_first_path(path_value):
        if isinstance(path_value, (list, tuple)):
            for item in path_value:
                if item not in (None, "", "None"):
                    path_value = item
                    break
            else:
                return None
        if path_value in (None, "", "None"):
            return None
        path_value = os.path.expanduser(str(path_value))
        if os.path.isabs(path_value):
            return path_value if os.path.exists(path_value) else None
        abs_path = os.path.abspath(path_value)
        return abs_path if os.path.exists(abs_path) else None

    def _debug_decode_prefix_audio(
        self,
        token_histories,
        audio_prompt_path,
        speech_sample_rate,
        tail_window_ms,
        tail_silence_threshold,
        num_latency_tokens,
    ):
        if self.codec_decoder is None or audio_prompt_path is None:
            return None
        if not token_histories or not any(len(history) > 0 for history in token_histories):
            return None
        audio_tokens = [
            torch.tensor(history, dtype=torch.long)
            for history in token_histories
        ]
        audio_hat = audio_decode_cosyvoice(
            audio_tokens,
            self.model_config,
            self.codec_decoder,
            audio_prompt_path=audio_prompt_path,
            code_layer=self.code_layer,
            num_latency_tokens=num_latency_tokens,
            allow_missing_eoa=True,
        )
        if audio_hat is None or not isinstance(audio_hat, torch.Tensor) or audio_hat.numel() <= 0:
            return None
        waveform = audio_hat.detach().float().reshape(-1)
        full_rms = float(torch.sqrt(torch.clamp(waveform.pow(2).mean(), min=1e-12)).item())
        tail_window_samples = max(int(float(speech_sample_rate) * float(tail_window_ms) / 1000.0), 1)
        tail_waveform = waveform[-tail_window_samples:] if waveform.numel() > tail_window_samples else waveform
        tail_rms = float(torch.sqrt(torch.clamp(tail_waveform.pow(2).mean(), min=1e-12)).item())
        return {
            "num_samples": int(waveform.numel()),
            "full_rms": full_rms,
            "tail_rms": tail_rms,
            "tail_silent": bool(tail_rms <= float(tail_silence_threshold)),
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        modality_mask = kwargs.get("modality_mask", None)
        encoder_outs = None
        sampled_audio_prefix_stats = None
        if self.stop_head_only_training and self.training:
            self._set_frozen_backbone_eval_mode()

        if input_ids is not None:
            # Avoid in-place mutation on input_ids to prevent autograd version mismatch
            # when the same batch tensor is reused (e.g., policy/ref forward in DPO).
            input_ids = input_ids.masked_fill(input_ids == -1, 0)  # [btz, code_layer + 1, seq_length]
            input_ids, sampled_audio_prefix_stats = self._build_sampled_audio_prefix_input_ids(
                input_ids,
                attention_mask,
                labels,
            )
            inputs_embeds = self._embed_input_ids(input_ids)

        if modality_mask is not None and encoder_outs is not None:
            if self.train_config.modeling_paradigm == "parallel":
                modality_mask = modality_mask.unsqueeze(1).repeat(1, self.code_layer, 1)  # [btz, code_layer, seq_length]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=2)
                modality_lengths = torch.clamp(modality_mask.sum(dim=2), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    for j in range(self.code_layer):
                        start_idx = modality_mask_start_indices[i, j].item()
                        length = modality_lengths[i][j]
                        encoder_outs_pad[i, j, start_idx:start_idx+length] = encoder_outs[i, :length]
                
                inputs_embeds[:, :self.code_layer, :, :] = encoder_outs_pad[:, :self.code_layer, :, :] + inputs_embeds[:, :self.code_layer, :, :] * (~modality_mask[:, :, :, None])
        
                inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers

            elif self.train_config.modeling_paradigm == "interleaved":
                inputs_embeds = inputs_embeds.squeeze(1)  # [btz, seq_length, emb_dim]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
                modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    encoder_outs_pad[
                        i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                    ] = encoder_outs[i][:modality_lengths[i]]
                
                inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
            
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        if self.train_config.modeling_paradigm == "serial":
            temp_labels = labels[:,self.code_layer - 1] if labels is not None else None
            # need_hidden_states = self.use_mtp or kwargs.get("output_hidden_states", False)
            need_hidden_states = self.use_mtp or self.use_future_mtp or self.use_stop_head or kwargs.get("output_hidden_states", False)
            llm_forward_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "labels": temp_labels,
                "output_hidden_states": need_hidden_states,
                "return_dict": True,
            }
            # original logic retained: serial path still uses temp_labels as LM supervision label.
            model_outputs = self.llm(**llm_forward_kwargs)

            eot = self.model_config.vocab_config.eot
            text_pad_token = self.model_config.vocab_config.pad_t
            audio_pad_token = self.model_config.vocab_config.pad_a
            text_labels = torch.full_like(labels, -100)
            audio_labels = torch.full_like(labels, -100)
            batch_size, seq_size, length = labels.shape
            for i in range(batch_size):
                eot_position = (labels[i, 0] == eot).nonzero(as_tuple=True)[0]
                eot_pos = eot_position.item()  
                text_labels[i, :, :eot_pos+1] = labels[i, :, :eot_pos+1]
                text_labels[i, :, eot_pos+1:] = text_pad_token

                audio_labels[i, :, eot_pos+1:] = labels[i, :, eot_pos+1:]
                ignore_pos = torch.where(labels[i, 0]!= -100)[0][0].item()
                audio_labels[i, :, ignore_pos:eot_pos+1] = audio_pad_token
            text_labels = text_labels[:, 0, :]
        else:
            text_labels = labels[:,self.code_layer] if labels is not None else None
            audio_labels = labels[:, :self.code_layer] if labels is not None else None
            # need_hidden_states = self.use_mtp or kwargs.get("output_hidden_states", False)
            need_hidden_states = self.use_mtp or self.use_future_mtp or self.use_stop_head or kwargs.get("output_hidden_states", False)
            llm_forward_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "labels": text_labels,
                "output_hidden_states": need_hidden_states,
                "return_dict": True,
            }
            # original logic retained: the text token layer is still used as the LM supervision label.
            model_outputs = self.llm(**llm_forward_kwargs)

        if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
            xt, xa = self._build_parallel_logits(model_outputs, attention_mask, labels)
            loss_recorder = []
            if self.use_future_mtp:
                base_loss, loss_recorder = self.compute_future_mtp_loss(xt, text_labels, xa, audio_labels)
            else:
                base_loss, loss_recorder = self.compute_parallel_loss(xt, text_labels, xa, audio_labels)
            stop_loss_disabled = (
                self.training
                and self.use_sampled_audio_prefix_training
                and self.disable_stop_loss_during_sampled_prefix
            )
            if self.use_stop_head and audio_labels is not None and not stop_loss_disabled:
                text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
                eoa_id = self.model_config.vocab_config.eoa
                stop_hidden = model_outputs.hidden_states[-1][:, :-1, :]
                stop_raw = model_outputs.logits[..., text_vocab_size:][:, :-1, :]
                stop_logits = self._compute_stop_logits(stop_hidden, stop_raw)
                stop_target, stop_valid = self._compute_stop_targets(audio_labels, eoa_id)
                stop_loss = self._compute_stop_loss(stop_logits, audio_labels)
                model_outputs.stop_loss = stop_loss
                model_outputs.stop_metrics = self._summarize_stop_metrics(
                    stop_logits,
                    stop_target,
                    stop_valid,
                    stop_loss=stop_loss,
                )
                model_outputs.stop_logits_detached = stop_logits.detach()
                model_outputs.stop_target_detached = stop_target.detach() if stop_target is not None else None
                model_outputs.stop_valid_detached = stop_valid.detach() if stop_valid is not None else None
                base_loss = base_loss + self.stop_loss_weight * stop_loss
            elif self.use_stop_head and audio_labels is not None and stop_loss_disabled:
                zero_stop_loss = model_outputs.logits.new_zeros(())
                model_outputs.stop_loss = zero_stop_loss
                model_outputs.stop_metrics = {"stop_loss": zero_stop_loss.detach()}
                model_outputs.stop_logits_detached = None
                model_outputs.stop_target_detached = None
                model_outputs.stop_valid_detached = None
            if sampled_audio_prefix_stats is not None:
                model_outputs.sampled_audio_prefix_stats = sampled_audio_prefix_stats
            model_outputs.loss = base_loss
        elif self.train_config.modeling_paradigm == "interleaved":
            x_ori = model_outputs.logits
        else:
            raise NotImplementedError

        text_acc = -1
        audio_acc = [-1 for _ in range(self.code_layer)] if self.code_layer > 0 else -1
        if self.metric:
            with torch.no_grad():
                if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
                    preds = torch.argmax(xt, -1)
                    text_acc = compute_accuracy(preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                    preds_audio = [torch.argmax(xa[i], -1) for i in range(self.code_layer)]
                    audio_acc = [compute_accuracy(preds_audio[i].detach()[:, :-1], audio_labels[:, i, 1:], ignore_label=-100) for i in range(self.code_layer)]
                elif self.train_config.modeling_paradigm == "interleaved":
                    preds_start_idx = (text_labels != -100).float().argmax(dim=1)
                    preds = torch.argmax(x_ori, -1)
                    text_preds, text_labels, audio_preds, audio_labels = self.extract_interleaved_tokens(preds, text_labels, preds_start_idx) 

                    text_pad_token = self.model_config.vocab_config.pad_t
                    text_labels[text_labels == text_pad_token] = -100
                    
                    text_acc = compute_accuracy(text_preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                    audio_pad_token = self.model_config.vocab_config.pad_a
                    audio_labels[audio_labels == audio_pad_token] = -100
                    
                    audio_acc = [ compute_accuracy(audio_preds.detach()[:, :-1], audio_labels.detach()[:, 1:], ignore_label=-100)]

                    loss_recorder = None
                else:
                    raise NotImplementedError

        return model_outputs, text_acc, audio_acc, loss_recorder

    def compute_parallel_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute the parallel loss for text and audio layers.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        layer_loss = [0 for _ in range(self.code_layer+1) ] 
        
        text_weight = 0.0
        if text_labels is not None:
            text_loss = F.cross_entropy(xt[:, :-1, :].reshape(-1, text_vocab_size), text_labels[:, 1:].reshape(-1), ignore_index=-100)
            layer_loss[self.code_layer] = text_loss
            text_weight = 1.0
        else:
            text_loss = 0

        total_audio_loss = 0
        single_audio_loss = 0
        total_audio_weight = 0.0
        for i in range(self.code_layer):
            if audio_labels[:,i] is not None:
                single_audio_loss = F.cross_entropy(xa[i][:, :-1, :].reshape(-1, audio_vocab_size), audio_labels[:, i, 1:].reshape(-1), ignore_index=-100)
                layer_loss[i] = single_audio_loss
                audio_weight = self.mtp_loss_decay ** max(i, 0) if self.use_mtp else 1.0
                total_audio_loss += audio_weight * single_audio_loss
                total_audio_weight += audio_weight

        total_weight = text_weight + total_audio_weight
        total_loss = (text_loss + total_audio_loss) / max(total_weight, 1.0)
        return total_loss, layer_loss

    def compute_future_mtp_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute loss for the single-stream future-token MTP path.

        xa[0] predicts the next audio token.
        xa[1] predicts the token after that, xa[2] predicts one step further, etc.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        audio_stream = audio_labels[:, 0, :] if audio_labels.dim() == 3 else audio_labels

        loss_recorder = [torch.tensor(0.0, device=xt.device) for _ in range(len(xa) + 1)]

        text_weight = 0.0
        if text_labels is not None:
            text_loss = F.cross_entropy(
                xt[:, :-1, :].reshape(-1, text_vocab_size),
                text_labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss_recorder[-1] = text_loss
            text_weight = 1.0
        else:
            text_loss = torch.tensor(0.0, device=xt.device)

        total_audio_loss = torch.tensor(0.0, device=xt.device)
        total_audio_weight = 0.0
        seq_len = audio_stream.shape[1]
        for future_idx, audio_logits in enumerate(xa):
            target_shift = future_idx + 1
            if seq_len <= target_shift:
                continue
            shifted_logits = audio_logits[:, :-target_shift, :]
            shifted_labels = audio_stream[:, target_shift:]
            single_audio_loss = F.cross_entropy(
                shifted_logits.reshape(-1, audio_vocab_size),
                shifted_labels.reshape(-1),
                ignore_index=-100,
            )
            loss_recorder[future_idx] = single_audio_loss
            audio_weight = self.future_mtp_loss_decay ** future_idx
            total_audio_loss = total_audio_loss + audio_weight * single_audio_loss
            total_audio_weight += audio_weight

        total_weight = text_weight + total_audio_weight
        total_loss = (text_loss + total_audio_loss) / max(total_weight, 1.0)
        return total_loss, loss_recorder

    def _compute_future_mtp_lookahead_rebuild(self, hidden_history, main_token, generated_audio, audio_repetition_penalty, **kwargs):
        """
        Rebuild future caches from full history every step.

        This is kept as a validation reference to catch cache alignment bugs in the
        persistent rollout path.
        """
        if self.future_mtp_decoder is None or self.future_mtp_num <= 0:
            return [], []

        batch_size = hidden_history.shape[0]
        history_len = hidden_history.shape[1]
        if history_len == 0:
            return [], []

        future_caches = self.future_mtp_decoder.init_generation(hidden_history[:, :-1, :])
        future_hidden = hidden_history[:, -1:, :]
        conditioning_token = main_token.view(batch_size, 1)
        future_logits = []
        future_tokens = []
        generated_prefix = generated_audio.clone()

        for layer_idx in range(self.future_mtp_num):
            future_hidden, layer_logits = self.future_mtp_decoder.infer_one_layer_step(
                future_hidden,
                conditioning_token,
                layer_idx,
                future_caches[layer_idx],
            )
            layer_logits = layer_logits[0]

            seen_tokens = generated_prefix
            if seen_tokens.numel() > 0:
                layer_logits = self.repetition_penalty(layer_logits, seen_tokens, audio_repetition_penalty)

            next_future_token = self.sample_next_token(layer_logits[-1, :], **kwargs)
            future_logits.append(layer_logits)
            future_tokens.append(next_future_token)

            generated_prefix = torch.cat([generated_prefix, next_future_token.view(1)], dim=0)
            conditioning_token = next_future_token.view(batch_size, 1)

        return future_logits, future_tokens

    def _compute_future_mtp_lookahead(self, current_anchor_hidden, main_token, future_caches, generated_audio, audio_repetition_penalty, **kwargs):
        """
        Warmup rollout used before the chunk window is filled.
        """
        if self.future_mtp_decoder is None or self.future_mtp_num <= 0:
            return [], [], [], []
        if current_anchor_hidden is None or current_anchor_hidden.dim() != 3 or current_anchor_hidden.shape[1] != 1:
            raise ValueError(
                "current_anchor_hidden must have shape [B, 1, H] for future-token rollout, "
                f"got {tuple(current_anchor_hidden.shape) if current_anchor_hidden is not None else None}."
            )
        if future_caches is None or len(future_caches) < self.future_mtp_num:
            raise ValueError("future_caches must contain one cache per future-token branch.")

        batch_size = current_anchor_hidden.shape[0]
        future_hidden = current_anchor_hidden
        conditioning_token = main_token.view(batch_size, 1)
        future_logits = []
        future_tokens = []
        generated_prefix = generated_audio.clone()
        cache_lens_before = [cache.get_seq_length() for cache in future_caches[: self.future_mtp_num]]

        for layer_idx in range(self.future_mtp_num):
            future_hidden, layer_logits = self.future_mtp_decoder.infer_one_layer_step(
                future_hidden,
                conditioning_token,
                layer_idx,
                future_caches[layer_idx],
            )
            layer_logits = layer_logits[0]

            if generated_prefix.numel() > 0:
                layer_logits = self.repetition_penalty(layer_logits, generated_prefix, audio_repetition_penalty)

            next_future_token = self.sample_next_token(layer_logits[-1, :], **kwargs)
            future_logits.append(layer_logits)
            future_tokens.append(next_future_token)

            generated_prefix = torch.cat([generated_prefix, next_future_token.view(1)], dim=0)
            conditioning_token = next_future_token.view(batch_size, 1)

        cache_lens_after = [cache.get_seq_length() for cache in future_caches[: self.future_mtp_num]]
        return future_logits, future_tokens, cache_lens_before, cache_lens_after

    def _compute_future_mtp_chunk_rollout(self, chunk_hidden_states, chunk_audio_tokens, future_caches, generated_audio, audio_repetition_penalty, **kwargs):
        """
        Ref-like steady-state rollout where every future branch consumes the same chunk
        window and updates its own persistent cache with the full chunk length.
        """
        if self.future_mtp_decoder is None or self.future_mtp_num <= 0:
            return [], [], [], []
        if chunk_hidden_states is None or chunk_hidden_states.dim() != 3:
            raise ValueError(
                "chunk_hidden_states must have shape [B, T, H] for future-token chunk rollout, "
                f"got {tuple(chunk_hidden_states.shape) if chunk_hidden_states is not None else None}."
            )
        if chunk_audio_tokens is None or chunk_audio_tokens.dim() != 2:
            raise ValueError(
                "chunk_audio_tokens must have shape [B, T] for future-token chunk rollout, "
                f"got {tuple(chunk_audio_tokens.shape) if chunk_audio_tokens is not None else None}."
            )
        if chunk_hidden_states.shape[:2] != chunk_audio_tokens.shape:
            raise ValueError(
                "chunk_hidden_states and chunk_audio_tokens must share the same [B, T] shape, "
                f"got hidden={tuple(chunk_hidden_states.shape[:2])} vs tokens={tuple(chunk_audio_tokens.shape)}."
            )
        if future_caches is None or len(future_caches) < self.future_mtp_num:
            raise ValueError("future_caches must contain one cache per future-token branch.")

        future_hidden = chunk_hidden_states
        future_logits = []
        future_tokens = []
        generated_prefix = generated_audio.clone()
        cache_lens_before = [cache.get_seq_length() for cache in future_caches[: self.future_mtp_num]]

        for layer_idx in range(self.future_mtp_num):
            future_hidden, layer_logits = self.future_mtp_decoder.infer_one_layer_chunk(
                future_hidden,
                chunk_audio_tokens,
                layer_idx,
                future_caches[layer_idx],
            )
            layer_logits = layer_logits[0]

            if generated_prefix.numel() > 0:
                layer_logits = self.repetition_penalty(layer_logits, generated_prefix, audio_repetition_penalty)

            next_future_token = self.sample_next_token(layer_logits[-1, :], **kwargs)
            future_logits.append(layer_logits)
            future_tokens.append(next_future_token)
            generated_prefix = torch.cat([generated_prefix, next_future_token.view(1)], dim=0)

        cache_lens_after = [cache.get_seq_length() for cache in future_caches[: self.future_mtp_num]]
        return future_logits, future_tokens, cache_lens_before, cache_lens_after

    def _rebuild_future_chunk_caches(self, prefix_hidden_history, warmup_chunk_tokens, replay_hidden_chunks, replay_chunk_tokens):
        if self.future_mtp_decoder is None or self.future_mtp_num <= 0:
            return []
        if prefix_hidden_history.dim() != 3 or prefix_hidden_history.shape[1] == 0:
            return self.future_mtp_decoder.init_generation(prefix_hidden_history)

        future_caches = self.future_mtp_decoder.init_generation(prefix_hidden_history[:, :-1, :])

        if warmup_chunk_tokens is not None and warmup_chunk_tokens.numel() > 0:
            future_hidden = prefix_hidden_history[:, -1:, :]
            for layer_idx in range(min(self.future_mtp_num, warmup_chunk_tokens.shape[1] - 1)):
                conditioning_token = warmup_chunk_tokens[:, layer_idx : layer_idx + 1]
                future_hidden, _ = self.future_mtp_decoder.infer_one_layer_step(
                    future_hidden,
                    conditioning_token,
                    layer_idx,
                    future_caches[layer_idx],
                )

        for hidden_chunk, chunk_tokens in zip(replay_hidden_chunks, replay_chunk_tokens):
            future_hidden = hidden_chunk
            for layer_idx in range(self.future_mtp_num):
                future_hidden, _ = self.future_mtp_decoder.infer_one_layer_chunk(
                    future_hidden,
                    chunk_tokens,
                    layer_idx,
                    future_caches[layer_idx],
                )

        return future_caches

    def _compute_future_mtp_chunk_rollout_rebuild(
        self,
        prefix_hidden_history,
        warmup_chunk_tokens,
        replay_hidden_chunks,
        replay_chunk_tokens,
        current_chunk_hidden,
        current_chunk_tokens,
        generated_audio,
        audio_repetition_penalty,
        **kwargs,
    ):
        future_caches = self._rebuild_future_chunk_caches(
            prefix_hidden_history,
            warmup_chunk_tokens,
            replay_hidden_chunks,
            replay_chunk_tokens,
        )
        future_logits, future_tokens, _, _ = self._compute_future_mtp_chunk_rollout(
            current_chunk_hidden,
            current_chunk_tokens,
            future_caches,
            generated_audio,
            audio_repetition_penalty,
            **kwargs,
        )
        return future_logits, future_tokens

    def _run_future_backbone_chunk(self, audio_chunk_tokens, current_text_token, attention_mask, backbone_past_key_values):
        if audio_chunk_tokens is None or audio_chunk_tokens.dim() != 2:
            raise ValueError(
                "audio_chunk_tokens must have shape [B, T] when advancing the future-MTP backbone cache, "
                f"got {tuple(audio_chunk_tokens.shape) if audio_chunk_tokens is not None else None}."
            )

        text_chunk_tokens = current_text_token.view(1, 1).expand(audio_chunk_tokens.shape[0], audio_chunk_tokens.shape[1])
        shifted_audio_tokens = simple_shift(audio_chunk_tokens, 0)
        if self.train_config.use_peft:
            embed_tokens = self.llm.model.model.embed_tokens
        else:
            embed_tokens = self.llm.model.embed_tokens

        audio_embeds = embed_tokens(shifted_audio_tokens)
        text_embeds = embed_tokens(text_chunk_tokens)
        chunk_inputs_embeds = (audio_embeds + text_embeds) / 2
        attention_mask = torch.cat(
            [attention_mask, torch.ones((audio_chunk_tokens.size(0), audio_chunk_tokens.size(1)), device=audio_chunk_tokens.device)],
            dim=1,
        )

        outputs = self.llm(
            inputs_embeds=chunk_inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=backbone_past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs, attention_mask

    @torch.no_grad()
    def _generate_future_mtp(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if self.model_config.use_text_stream:
            raise NotImplementedError("use_future_mtp=True currently only supports use_text_stream=false during generation.")

        kwargs["inference_mode"] = True
        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        max_new_tokens = kwargs.get("max_new_tokens", 360)
        debug_generation = bool(kwargs.get("debug_generation", False))
        debug_generation_topk = max(int(kwargs.get("debug_generation_topk", 5)), 1)
        debug_generation_max_steps = int(kwargs.get("debug_generation_max_steps", 0))
        debug_generation_log_interval = max(int(kwargs.get("debug_generation_log_interval", 1)), 1)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eoa = self.model_config.vocab_config.eoa

        sample_keys = kwargs.get("keys", None)
        sample_key = sample_keys[0] if isinstance(sample_keys, (list, tuple)) and len(sample_keys) > 0 else sample_keys

        if debug_generation:
            logger.info(
                "[GEN_DEBUG] key=%s start max_new_tokens=%d topk=%d log_interval=%d max_steps=%d use_future_mtp=%s decode_text_only=%s",
                sample_key,
                max_new_tokens,
                debug_generation_topk,
                debug_generation_log_interval,
                debug_generation_max_steps,
                self.use_future_mtp,
                decode_text_only,
            )

        generated_audio = torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device)
        generated_audio_len = 0
        generated_text = torch.empty((0,), dtype=torch.long, device=input_ids.device)
        current_text_token = torch.tensor([pad_t], device=input_ids.device)
        audio_end = False

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        consumed_audio_history = torch.empty((0,), dtype=torch.long, device=input_ids.device)
        lookahead_sample_kwargs = dict(kwargs)
        lookahead_sample_kwargs.pop("audio_repetition_penalty", None)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        backbone_past_key_values = outputs.past_key_values
        prefix_hidden_history = outputs.hidden_states[-1]
        future_prefill_hidden = prefix_hidden_history[:, :-1, :]
        future_caches = self.future_mtp_decoder.init_generation(future_prefill_hidden)
        future_cache_prefill_lens = [cache.get_seq_length() for cache in future_caches[: self.future_mtp_num]]
        expected_prefill_len = future_prefill_hidden.shape[1]
        prefill_ok = all(cache_len == expected_prefill_len for cache_len in future_cache_prefill_lens)
        if debug_generation:
            logger.info(
                "[GEN_DEBUG] key=%s future_cache_prefill expected=%d actual=%s prefill_ok=%s",
                sample_key,
                expected_prefill_len,
                future_cache_prefill_lens,
                prefill_ok,
            )

        current_cycle_outputs = outputs
        current_chunk_hidden = None
        current_chunk_tokens = None
        warmup_chunk_tokens = None
        replay_hidden_chunks = []
        replay_chunk_tokens = []

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            logits = current_cycle_outputs.logits[0]
            xt_logits = logits[..., :text_vocab_size]
            xa_main = logits[..., text_vocab_size:]
            if consumed_audio_history.numel() > 0:
                xa_main = self.repetition_penalty(xa_main, consumed_audio_history, audio_repetition_penalty)

            main_token = self.sample_next_token(xa_main[-1, :], **kwargs)
            steady_mode = current_chunk_tokens is not None
            if steady_mode:
                future_logits, future_tokens, future_cache_lens_before, future_cache_lens_after = self._compute_future_mtp_chunk_rollout(
                    current_chunk_hidden,
                    current_chunk_tokens,
                    future_caches,
                    consumed_audio_history,
                    audio_repetition_penalty,
                    **lookahead_sample_kwargs,
                )
            else:
                current_anchor_hidden = prefix_hidden_history[:, -1:, :]
                future_logits, future_tokens, future_cache_lens_before, future_cache_lens_after = self._compute_future_mtp_lookahead(
                    current_anchor_hidden,
                    main_token,
                    future_caches,
                    consumed_audio_history,
                    audio_repetition_penalty,
                    **lookahead_sample_kwargs,
                )

            rebuild_validation = None
            rebuild_future_tokens = None
            if debug_generation and (
                debug_generation_max_steps <= 0 or step < debug_generation_max_steps
            ):
                if steady_mode:
                    _, rebuild_future_tokens = self._compute_future_mtp_chunk_rollout_rebuild(
                        prefix_hidden_history,
                        warmup_chunk_tokens,
                        replay_hidden_chunks,
                        replay_chunk_tokens,
                        current_chunk_hidden,
                        current_chunk_tokens,
                        consumed_audio_history,
                        audio_repetition_penalty,
                        **lookahead_sample_kwargs,
                    )
                else:
                    _, rebuild_future_tokens = self._compute_future_mtp_lookahead_rebuild(
                        prefix_hidden_history,
                        main_token,
                        consumed_audio_history,
                        audio_repetition_penalty,
                        **lookahead_sample_kwargs,
                    )
                rebuild_validation = [int(tok.item()) for tok in future_tokens] == [int(tok.item()) for tok in rebuild_future_tokens]

            should_log_debug_step = False
            if debug_generation:
                should_log_debug_step = (
                    debug_generation_max_steps <= 0 or step < debug_generation_max_steps
                ) and (step % debug_generation_log_interval == 0)
                if int(main_token.item()) == eoa or any(int(tok.item()) == eoa for tok in future_tokens):
                    should_log_debug_step = True
                if step == max_new_tokens - 1:
                    should_log_debug_step = True
            chunk_tokens = [int(main_token.item())]
            for future_token in future_tokens:
                chunk_tokens.append(int(future_token.item()))
                if int(future_token.item()) == eoa:
                    break
            first_eoa_idx = next((idx for idx, tok in enumerate(chunk_tokens) if tok == eoa), None)
            if first_eoa_idx is not None:
                chunk_tokens = chunk_tokens[: first_eoa_idx + 1]

            if should_log_debug_step:
                text_summary = f"text={self._debug_text_token_repr(int(current_text_token.item()), -1, pad_t)} sel_p=1.0000 p_eot=0.0000 topk=[<PAD_T>:1.0000]"
                audio_summaries = [
                    self._format_debug_audio_summary(
                        "main",
                        xa_main,
                        int(main_token.item()),
                        debug_generation_topk,
                        eoa,
                        pad_a,
                    )
                ]
                for future_idx, (future_logit, future_token) in enumerate(zip(future_logits, future_tokens), start=1):
                    audio_summaries.append(
                        self._format_debug_audio_summary(
                            f"future{future_idx}",
                            future_logit,
                            int(future_token.item()),
                            debug_generation_topk,
                            eoa,
                            pad_a,
                        )
                    )
                emitted_summary = ", ".join(self._debug_audio_token_repr(tok, eoa, pad_a) for tok in chunk_tokens)
                backbone_cache_len = self._debug_cache_length(backbone_past_key_values)
                if steady_mode:
                    chunk_input_summary = ", ".join(
                        self._debug_audio_token_repr(int(tok.item()), eoa, pad_a) for tok in current_chunk_tokens[0]
                    )
                else:
                    chunk_input_summary = "<warmup>"
                cache_summary = (
                    f"cache(mode={'steady' if steady_mode else 'warmup'}, chunk_in=[{chunk_input_summary}], backbone={backbone_cache_len}, future_before={future_cache_lens_before}, "
                    f"future_after={future_cache_lens_after}, prefill_ok={prefill_ok})"
                )
                if rebuild_validation is None:
                    validation_summary = "rebuild_match=NA"
                else:
                    validation_summary = (
                        f"rebuild_match={rebuild_validation} "
                        f"persistent={[int(tok.item()) for tok in future_tokens]} "
                        f"rebuild={[int(tok.item()) for tok in rebuild_future_tokens]}"
                    )
                logger.info(
                    "[GEN_DEBUG] key=%s step=%d audio_end=%s emit=[%s] %s %s %s %s",
                    sample_key,
                    step,
                    audio_end,
                    emitted_summary,
                    cache_summary,
                    validation_summary,
                    text_summary,
                    " ".join(audio_summaries),
                )

            if not decode_text_only:
                remaining = max_new_tokens - generated_audio_len
                committed_chunk = chunk_tokens[:remaining]
                for token_id in committed_chunk:
                    generated_audio[generated_audio_len] = token_id
                    generated_audio_len += 1
                if committed_chunk:
                    committed_chunk_tensor = torch.tensor(committed_chunk, dtype=torch.long, device=input_ids.device)
                    consumed_audio_history = torch.cat([consumed_audio_history, committed_chunk_tensor], dim=0)
                audio_end = first_eoa_idx is not None and first_eoa_idx < len(committed_chunk)
            else:
                committed_chunk = chunk_tokens
                audio_end = first_eoa_idx is not None

            if audio_end or generated_audio_len >= max_new_tokens:
                break

            if steady_mode:
                replay_hidden_chunks.append(current_chunk_hidden.detach().clone())
                replay_chunk_tokens.append(current_chunk_tokens.detach().clone())

            committed_chunk_tensor = torch.tensor(committed_chunk, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            current_cycle_outputs, attention_mask = self._run_future_backbone_chunk(
                committed_chunk_tensor,
                current_text_token,
                attention_mask,
                backbone_past_key_values,
            )
            backbone_past_key_values = current_cycle_outputs.past_key_values
            current_chunk_hidden = current_cycle_outputs.hidden_states[-1]
            current_chunk_tokens = committed_chunk_tensor
            if warmup_chunk_tokens is None:
                warmup_chunk_tokens = committed_chunk_tensor.detach().clone()

            current_text_token = torch.tensor([pad_t], device=input_ids.device)

        if debug_generation:
            logger.info(
                "[GEN_DEBUG] key=%s finish text_end=%s audio_end=%s generated_audio_steps=%d",
                sample_key,
                False,
                audio_end,
                int(generated_audio_len),
            )

        return [generated_audio[:generated_audio_len], generated_text]

    @staticmethod
    def _token_logp_sum(logits, labels, ignore_index=-100):
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(ignore_index)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
        token_logp = F.log_softmax(shift_logits, dim=-1).gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        token_logp = token_logp * valid_mask.to(token_logp.dtype)
        seq_logp = token_logp.sum(dim=-1)
        token_count = valid_mask.sum(dim=-1)
        return seq_logp, token_count

    def _prepare_text_audio_labels(self, labels):
        if self.train_config.modeling_paradigm == "serial":
            eot = self.model_config.vocab_config.eot
            text_pad_token = self.model_config.vocab_config.pad_t
            audio_pad_token = self.model_config.vocab_config.pad_a
            text_labels = torch.full_like(labels, -100)
            audio_labels = torch.full_like(labels, -100)
            batch_size = labels.shape[0]
            for i in range(batch_size):
                eot_position = (labels[i, 0] == eot).nonzero(as_tuple=True)[0]
                eot_pos = eot_position[0].item() if len(eot_position) > 0 else labels.shape[-1] - 1
                text_labels[i, :, :eot_pos + 1] = labels[i, :, :eot_pos + 1]
                text_labels[i, :, eot_pos + 1:] = text_pad_token

                audio_labels[i, :, eot_pos + 1:] = labels[i, :, eot_pos + 1:]
                ignore_pos = torch.where(labels[i, 0] != -100)[0]
                ignore_pos = ignore_pos[0].item() if len(ignore_pos) > 0 else 0
                audio_labels[i, :, ignore_pos:eot_pos + 1] = audio_pad_token
            text_labels = text_labels[:, 0, :]
            return text_labels, audio_labels

        text_labels = labels[:, self.code_layer] if labels is not None else None
        audio_labels = labels[:, :self.code_layer] if labels is not None else None
        return text_labels, audio_labels

    def compute_sequence_logp(self, logits, labels):
        """
        Compute per-sample sequence log-probability under current modeling paradigm.
        """
        if labels is None:
            return None
        if self.use_mtp:
            raise NotImplementedError("compute_sequence_logp is not yet adapted for use_mtp=True.")
        if self.use_future_mtp:
            raise NotImplementedError("compute_sequence_logp is not yet adapted for use_future_mtp=True.")
        self._debug_seq_logp_calls += 1
        debug_this_call = self.debug_train_sanity and (self._debug_seq_logp_calls <= self.debug_sanity_model_max_calls)
        if debug_this_call:
            logger.info(
                "[SANITY] [DPO] compute_sequence_logp call=%d paradigm=%s logits_shape=%s labels_shape=%s labels_dim=%s code_layer=%s",
                self._debug_seq_logp_calls,
                self.train_config.modeling_paradigm,
                tuple(logits.shape) if isinstance(logits, torch.Tensor) else None,
                tuple(labels.shape) if isinstance(labels, torch.Tensor) else None,
                labels.dim() if isinstance(labels, torch.Tensor) else None,
                self.code_layer,
            )
        if self.train_config.modeling_paradigm == "interleaved":
            if labels.dim() == 3:
                stream_labels = labels[:, 0, :]
            else:
                stream_labels = labels
            if debug_this_call:
                valid_stream = int(stream_labels.ne(-100).sum().item())
                total_stream = int(stream_labels.numel())
                logger.info(
                    "[SANITY] [DPO] interleaved label split: stream_labels_shape=%s valid_tokens=%d/%d",
                    tuple(stream_labels.shape),
                    valid_stream,
                    total_stream,
                )
                if valid_stream <= 0:
                    logger.warning("[SANITY] [DPO] interleaved stream has no valid supervision tokens.")
            seq_logp, _ = self._token_logp_sum(logits, stream_labels, ignore_index=-100)
            return seq_logp


        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        if debug_this_call:
            logger.info(
                "[SANITY] [DPO] vocab split: text_vocab_size=%d audio_vocab_size=%d group_decode_adapter=%s",
                text_vocab_size,
                audio_vocab_size,
                self.group_decode_adapter is not None,
            )
        xt = logits[..., :text_vocab_size]
        xa = []

        if self.group_decode_adapter is not None:
            x_audio_ori = logits[..., text_vocab_size:]
            x_audio = self.group_decode_adapter(x_audio_ori)
            for i in range(self.code_layer):
                xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])
        else:
            for i in range(self.code_layer):
                xa.append(logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)])

        text_labels, audio_labels = self._prepare_text_audio_labels(labels)
        if debug_this_call:
            text_valid = int(text_labels.ne(-100).sum().item()) if isinstance(text_labels, torch.Tensor) else 0
            text_total = int(text_labels.numel()) if isinstance(text_labels, torch.Tensor) else 0
            logger.info(
                "[SANITY] [DPO] prepared labels: text_labels_shape=%s audio_labels_shape=%s text_valid=%d/%d",
                tuple(text_labels.shape) if isinstance(text_labels, torch.Tensor) else None,
                tuple(audio_labels.shape) if isinstance(audio_labels, torch.Tensor) else None,
                text_valid,
                text_total,
            )
            if text_valid <= 0:
                logger.warning("[SANITY] [DPO] text_labels has no valid tokens after split.")
            if isinstance(audio_labels, torch.Tensor) and audio_labels.dim() >= 3:
                for i in range(min(audio_labels.shape[1], self.code_layer)):
                    audio_valid = int(audio_labels[:, i, :].ne(-100).sum().item())
                    audio_total = int(audio_labels[:, i, :].numel())
                    logger.info(
                        "[SANITY] [DPO] audio_layer=%d valid_tokens=%d/%d",
                        i,
                        audio_valid,
                        audio_total,
                    )
                    if audio_valid <= 0:
                        logger.warning(f"[SANITY] [DPO] audio layer {i} has near-zero valid tokens.")
        text_seq_logp, _ = self._token_logp_sum(xt, text_labels, ignore_index=-100)

        audio_seq_logp_total = torch.zeros_like(text_seq_logp)
        for i in range(self.code_layer):
            single_audio_seq_logp, _ = self._token_logp_sum(xa[i], audio_labels[:, i, :], ignore_index=-100)
            audio_seq_logp_total = audio_seq_logp_total + single_audio_seq_logp

        return text_seq_logp + audio_seq_logp_total



    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        if self.use_future_mtp:
            return self._generate_future_mtp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None
        mtp_hidden_history = None
        # mtp_audio_histories = None
        # mtp_caches = None
        oracle_audio_targets = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        # audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        num_latency_tokens = kwargs.get("num_latency_tokens", 0)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        debug_generation = bool(kwargs.get("debug_generation", False))
        debug_generation_topk = max(int(kwargs.get("debug_generation_topk", 5)), 1)
        debug_generation_max_steps = int(kwargs.get("debug_generation_max_steps", 0))
        debug_generation_log_interval = max(int(kwargs.get("debug_generation_log_interval", 1)), 1)
        oracle_mtp_conditioning = bool(kwargs.get("oracle_mtp_conditioning", False))
        teacher_forced_audio_prefix = bool(kwargs.get("teacher_forced_audio_prefix", False))
        debug_generation_decode_prefix = bool(kwargs.get("debug_generation_decode_prefix", False))
        debug_generation_decode_prefix_every = max(int(kwargs.get("debug_generation_decode_prefix_every", 1)), 1)
        debug_generation_decode_prefix_min_step = max(int(kwargs.get("debug_generation_decode_prefix_min_step", 0)), 0)
        debug_collapse_window = max(int(kwargs.get("debug_collapse_window", 8)), 1)
        debug_collapse_small_set_size = max(int(kwargs.get("debug_collapse_small_set_size", 2)), 1)
        debug_entropy_threshold = float(kwargs.get("debug_entropy_threshold", 2.0))
        debug_prefix_tail_window_ms = int(kwargs.get("debug_prefix_tail_window_ms", 250))
        debug_prefix_tail_silence_threshold = float(kwargs.get("debug_prefix_tail_silence_threshold", 0.003))
        debug_prefix_silence_patience = max(int(kwargs.get("debug_prefix_silence_patience", 3)), 1)
        use_collapse_aware_decoding = self._config_bool(kwargs.get("use_collapse_aware_decoding", False), False)
        collapse_window = max(int(kwargs.get("collapse_window", debug_collapse_window)), 1)
        collapse_small_set_size = max(int(kwargs.get("collapse_small_set_size", debug_collapse_small_set_size)), 1)
        collapse_entropy_threshold = float(kwargs.get("collapse_entropy_threshold", debug_entropy_threshold))
        collapse_tail_silence_patience = max(
            int(kwargs.get("collapse_tail_silence_patience", debug_prefix_silence_patience)),
            1,
        )
        attractor_penalty = float(kwargs.get("attractor_penalty", 1.5))
        attractor_min_step = max(int(kwargs.get("attractor_min_step", 4)), 0)
        collapse_hardcase_only = self._config_bool(kwargs.get("collapse_hardcase_only", True), True)
        collapse_debug_dataset_path = kwargs.get("collapse_debug_dataset_path", None)
        collapse_hardcase_match = self._looks_like_hard_case_path(collapse_debug_dataset_path)
        collapse_runtime_enabled = (
            use_collapse_aware_decoding
            and not decode_text_only
            and (not collapse_hardcase_only or collapse_hardcase_match)
        )
        speech_sample_rate = int(kwargs.get("speech_sample_rate", 22050))
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False
        audio_end = False
        sample_keys = kwargs.get("keys", None)
        sample_key = sample_keys[0] if isinstance(sample_keys, (list, tuple)) and len(sample_keys) > 0 else sample_keys
        trace_mode = "teacher_forced_prefix" if teacher_forced_audio_prefix else "free_run"
        debug_audio_top1_histories = [[] for _ in range(self.code_layer)]
        debug_audio_entropy_histories = [[] for _ in range(self.code_layer)]
        debug_prefix_tail_flags = []
        debug_prefix_audio_prompt_path = self._debug_pick_first_path(kwargs.get("audio_prompt_path"))
        if debug_prefix_audio_prompt_path is None:
            debug_prefix_audio_prompt_path = self._debug_pick_first_path(kwargs.get("neutral_speaker_wav"))
        if debug_prefix_audio_prompt_path is None:
            debug_prefix_audio_prompt_path = self._debug_pick_first_path(kwargs.get("target_wav"))

        if debug_generation:
            logger.info(
                "[GEN_DEBUG] key=%s start trace_mode=%s max_new_tokens=%d topk=%d log_interval=%d max_steps=%d use_mtp=%s decode_text_only=%s oracle_mtp_conditioning=%s teacher_forced_audio_prefix=%s decode_prefix=%s",
                sample_key,
                trace_mode,
                max_new_tokens,
                debug_generation_topk,
                debug_generation_log_interval,
                debug_generation_max_steps,
                self.use_mtp,
                decode_text_only,
                oracle_mtp_conditioning,
                teacher_forced_audio_prefix,
                debug_generation_decode_prefix,
            )
            logger.info(
                "[GEN_DEBUG][COLLAPSE_CFG] key=%s use_collapse_aware_decoding=%s runtime_enabled=%s hardcase_only=%s hardcase_match=%s dataset_path=%s window=%d small_set_size=%d entropy_threshold=%.4f tail_silence_patience=%d attractor_penalty=%.4f attractor_min_step=%d",
                sample_key,
                use_collapse_aware_decoding,
                collapse_runtime_enabled,
                collapse_hardcase_only,
                collapse_hardcase_match,
                collapse_debug_dataset_path,
                collapse_window,
                collapse_small_set_size,
                collapse_entropy_threshold,
                collapse_tail_silence_patience,
                attractor_penalty,
                attractor_min_step,
            )

        teacher_forced_audio_targets = None
        if oracle_mtp_conditioning or teacher_forced_audio_prefix:
            if oracle_mtp_conditioning and not self.use_mtp:
                raise ValueError("oracle_mtp_conditioning requires use_mtp=True.")
            raw_target_audio = kwargs.get("target_audio", None)
            if raw_target_audio is None:
                raise ValueError(
                    "teacher_forced_audio_prefix / oracle_mtp_conditioning requires target_audio in the generation batch."
                )
            if isinstance(raw_target_audio, list):
                if len(raw_target_audio) != input_ids.size(0):
                    raise ValueError(
                        "teacher_forced_audio_prefix expects one target_audio tensor per batch item, "
                        f"got {len(raw_target_audio)} entries for batch size {input_ids.size(0)}."
                    )
                teacher_forced_audio_targets = torch.stack(
                    [item.to(input_ids.device) if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=input_ids.device) for item in raw_target_audio],
                    dim=0,
                )
            elif isinstance(raw_target_audio, torch.Tensor):
                teacher_forced_audio_targets = raw_target_audio.to(input_ids.device)
                if teacher_forced_audio_targets.dim() == 2:
                    teacher_forced_audio_targets = teacher_forced_audio_targets.unsqueeze(0)
            else:
                raise TypeError(
                    "teacher_forced_audio_prefix expects target_audio to be a tensor or list of tensors, "
                    f"got {type(raw_target_audio).__name__}."
                )
            if teacher_forced_audio_targets.dim() != 3 or teacher_forced_audio_targets.shape[1] < self.code_layer:
                raise ValueError(
                    "teacher_forced_audio_prefix expects target_audio with shape [B, code_layer, T], "
                    f"got {tuple(teacher_forced_audio_targets.shape)}."
                )
            if oracle_mtp_conditioning:
                oracle_audio_targets = teacher_forced_audio_targets

        if self.train_config.modeling_paradigm == "interleaved":
            model_outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                num_beams=kwargs.get("num_beams", 4),
                do_sample=kwargs.get("do_sample", False),
                min_length=kwargs.get("min_length", 1),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=text_repetition_penalty,
                length_penalty=kwargs.get("length_penalty", 1.0),
                temperature=kwargs.get("temperature", 1.0),
                attention_mask=attention_mask,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=layershift(eoa, 0),
                pad_token_id=self.tokenizer.pad_token_id
            )
            model_outputs = self.process_interleaved_output(model_outputs)
            return model_outputs

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if current_input_text is not None:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens, current_input_text.unsqueeze(1)], dim=1)
                if self.train_config.use_peft:
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=self.use_mtp or self.use_stop_head,
                return_dict=True,
            )
            
            logits = outputs.logits[0]                      # batch size is 1
            past_key_values = outputs.past_key_values       # Update past_key_values for the next step
            if self.use_mtp:
                if mtp_hidden_history is None:
                    mtp_hidden_history = outputs.hidden_states[-1]
                else:
                    current_anchor_hidden = outputs.hidden_states[-1][:, -1:, :]
                    mtp_hidden_history = torch.cat([mtp_hidden_history, current_anchor_hidden], dim=1)
                # mtp_audio_histories = self._ensure_mtp_audio_histories(
                #     mtp_hidden_history,
                #     mtp_audio_histories,
                #     pad_a,
                # )
                # prefix_hidden_states = outputs.hidden_states[-1]
                # if mtp_caches is None:
                #     mtp_caches = self.mtp_audio_decoder.init_generation(outputs.hidden_states[-1])
                # current_anchor_hidden = outputs.hidden_states[-1][:, -1:, :]
                #     # Prefill branch caches with the strict prefix only.
                #     # The last prefix position is the first online decoding step
                #     # and must not be consumed twice by the cache path.
                #     mtp_caches = self.mtp_audio_decoder.init_generation(prefix_hidden_states[:, :-1, :])
                # current_anchor_hidden = prefix_hidden_states[:, -1:, :]
            xt_logits, xa_logits = self._build_generate_step_logits(logits, mtp_hidden_history=mtp_hidden_history)
            # if self.use_mtp:
            #     xt_logits, xa_logits = self._build_generate_step_logits(logits)
            # else:
            #     xt_logits, xa_logits = self._build_generate_step_logits(logits, mtp_hidden_history=mtp_hidden_history)
            # # Split logits into text and audio layers based on vocab size
            # xt_logits, xa_logits = self._build_generate_step_logits(logits)

            # Apply repetition penalty to the logits
            xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
            for i in range(self.code_layer):
                xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)
            # xa_logits[0] = self.repetition_penalty(xa_logits[0], generated_ids[0][:step], audio_repetition_penalty)
            # # if self.use_mtp:
            #     xa_logits[0] = self.repetition_penalty(xa_logits[0], generated_ids[0][:step], audio_repetition_penalty)
            # else:
            #     for i in range(self.code_layer):
            #         xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)

            stop_logit_value = None
            stop_prob_value = None
            stop_bias_value = 0.0
            effective_min_step = max(self.stop_min_step, num_latency_tokens) if self.use_stop_head else None
            stop_ready = False
            if self.use_stop_head:
                text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
                eoa_id = self.model_config.vocab_config.eoa
                current_hidden = outputs.hidden_states[-1][:, -1, :]
                current_raw = logits[-1, text_vocab_size:].unsqueeze(0)
                stop_logit = self._compute_stop_logits(current_hidden, current_raw)
                stop_prob = torch.sigmoid(stop_logit).reshape(-1)[0]
                stop_logit_value = float(stop_logit.reshape(-1)[0].detach().float().item())
                stop_prob_value = float(stop_prob.detach().float().item())
                stop_ready = step >= effective_min_step
                if stop_ready:
                    stop_bias = self.stop_bias_lambda * stop_prob.to(dtype=xa_logits[0].dtype)
                    stop_bias_value = float(stop_bias.detach().float().item())
                    for i in range(self.code_layer):
                        xa_logits[i][-1, eoa_id] = xa_logits[i][-1, eoa_id] + stop_bias

            pre_penalty_audio_step_stats = []
            if debug_generation or collapse_runtime_enabled:
                for i in range(self.code_layer):
                    layer_stats = self._debug_audio_distribution_stats(
                        xa_logits[i][-1, :],
                        debug_generation_topk,
                        eoa,
                        pad_a,
                    )
                    pre_penalty_audio_step_stats.append(layer_stats)

            tail_silence_run = self._collapse_tail_silence_run(debug_prefix_tail_flags)
            collapse_state = self._detect_audio_collapse(
                debug_audio_top1_histories,
                debug_audio_entropy_histories,
                pre_penalty_audio_step_stats,
                tail_silence_run,
                collapse_window,
                collapse_small_set_size,
                collapse_entropy_threshold,
                collapse_tail_silence_patience,
                attractor_min_step,
                step,
            ) if pre_penalty_audio_step_stats else {
                "collapse_active": False,
                "triggered_layers": [],
                "tail_silent": bool(tail_silence_run >= collapse_tail_silence_patience),
                "tail_silence_run": int(tail_silence_run),
                "attractors_by_layer": {},
                "layers": {},
            }

            collapse_penalties = {}
            if (
                collapse_runtime_enabled
                and collapse_state.get("collapse_active", False)
                and not audio_end
                and not decode_text_only
                and num_latency_tokens <= step
            ):
                collapse_penalties = self._apply_attractor_penalty(
                    xa_logits,
                    collapse_state,
                    attractor_penalty,
                    eoa,
                    pad_a,
                )

            audio_step_stats = []
            if debug_generation or collapse_runtime_enabled:
                for i in range(self.code_layer):
                    audio_step_stats.append(
                        self._debug_audio_distribution_stats(
                            xa_logits[i][-1, :],
                            debug_generation_topk,
                            eoa,
                            pad_a,
                        )
                    )

            if debug_generation or collapse_runtime_enabled:
                for i, layer_stats in enumerate(pre_penalty_audio_step_stats):
                    debug_audio_top1_histories[i].append(layer_stats["top1_id"])
                    debug_audio_entropy_histories[i].append(layer_stats["entropy"])

            should_log_collapse_step = debug_generation and (
                debug_generation_max_steps <= 0 or step < debug_generation_max_steps
            )
            if should_log_collapse_step:
                collapse_log = {
                    "enabled": bool(collapse_runtime_enabled),
                    "collapse_active": bool(collapse_state.get("collapse_active", False)),
                    "triggered_layers": collapse_state.get("triggered_layers", []),
                    "tail_silent": bool(collapse_state.get("tail_silent", False)),
                    "tail_silence_run": int(collapse_state.get("tail_silence_run", 0)),
                    "attractors_by_layer": collapse_state.get("attractors_by_layer", {}),
                    "penalties": collapse_penalties,
                    "pre_topk": {
                        f"a{i}": layer_stats.get("topk_entries", [])
                        for i, layer_stats in enumerate(pre_penalty_audio_step_stats)
                    },
                    "post_topk": {
                        f"a{i}": layer_stats.get("topk_entries", [])
                        for i, layer_stats in enumerate(audio_step_stats)
                    },
                    "layers": collapse_state.get("layers", {}),
                }
                logger.info(
                    "[GEN_DEBUG][COLLAPSE_STEP] key=%s step=%d %s",
                    sample_key,
                    step,
                    json.dumps(collapse_log, ensure_ascii=False, sort_keys=True),
                )

            raw_a0_stats = pre_penalty_audio_step_stats[0] if pre_penalty_audio_step_stats else None

            if not text_end:
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
            else:
                next_token_text = torch.tensor([pad_t], device=input_ids.device)
            
            next_tokens_audio = []
            for i in range(self.code_layer):
                if not audio_end and not decode_text_only and num_latency_tokens <= step:
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                else:
                    next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                next_tokens_audio.append(next_token_audio)
            # if not audio_end and not decode_text_only and num_latency_tokens <= step:
            #     next_token_audio = self.sample_next_token(xa_logits[0][-1, :], **kwargs)
            # else:
            #     next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
            # next_tokens_audio.append(next_token_audio)

            # if self.use_mtp:
            #     mtp_audio_histories[0][:, -1] = next_token_audio.view(input_ids.size(0))
            #     # mtp_hidden_state = current_anchor_hidden
            #     for mtp_layer_idx in range(self.mtp_num):
            #         audio_layer_idx = mtp_layer_idx + 1
            #         # if oracle_audio_targets is not None:
            #         #     if step < oracle_audio_targets.shape[-1]:
            #         #         conditioning_audio_token = oracle_audio_targets[:, mtp_layer_idx, step].view(input_ids.size(0), 1)
            #         #     else:
            #         #         conditioning_audio_token = torch.full((input_ids.size(0), 1), pad_a, device=input_ids.device, dtype=torch.long)
            #         # else:
            #         #     conditioning_audio_token = next_tokens_audio[mtp_layer_idx].view(input_ids.size(0), 1)
            #         # mtp_hidden_state, mtp_layer_logits = self.mtp_audio_decoder.infer_one_layer_step(
            #         #     mtp_hidden_state,
            #         #     conditioning_audio_token,
            #         #     mtp_layer_idx,
            #         #     mtp_caches[mtp_layer_idx],
            #         # )
            #         # mtp_layer_logits = mtp_layer_logits[0]
            #         if not audio_end and not decode_text_only and num_latency_tokens <= step:
            #             mtp_layer_logits = self._build_mtp_layer_generate_logits(
            #                 mtp_hidden_history,
            #                 mtp_audio_histories,
            #                 target_audio_layer=audio_layer_idx,
            #             )
            #             mtp_layer_logits = self.repetition_penalty(
            #                 mtp_layer_logits,
            #                 generated_ids[audio_layer_idx][:step],
            #                 audio_repetition_penalty,
            #             )
            #             next_token_audio = self.sample_next_token(mtp_layer_logits[-1, :], **kwargs)
            #         else:
            #             mtp_layer_logits = torch.full(
            #                 (mtp_hidden_history.shape[1], audio_vocab_size),
            #                 fill_value=-1e9,
            #                 dtype=logits.dtype,
            #                 device=logits.device,
            #             )
            #             mtp_layer_logits[:, pad_a] = 0
            #             next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
            #         xa_logits.append(mtp_layer_logits)
            #         next_tokens_audio.append(next_token_audio)
            #         if audio_layer_idx < self.mtp_num:
            #             mtp_audio_histories[audio_layer_idx][:, -1] = next_token_audio.view(input_ids.size(0))
            # else:
            #     for i in range(1, self.code_layer):
            #         if not audio_end and not decode_text_only and num_latency_tokens <= step:
            #             next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
            #         else:
            #             next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
            #         next_tokens_audio.append(next_token_audio)

            selected_text_token = int(next_token_text.item())
            selected_audio_tokens = [int(token.item()) for token in next_tokens_audio]
            teacher_forced_condition_tokens = None
            if teacher_forced_audio_prefix and teacher_forced_audio_targets is not None:
                teacher_forced_condition_tokens = []
                for i in range(self.code_layer):
                    if step < teacher_forced_audio_targets.shape[-1]:
                        teacher_token = teacher_forced_audio_targets[:, i, step].to(device=input_ids.device, dtype=torch.long)
                    else:
                        teacher_token = torch.full((input_ids.size(0),), pad_a, device=input_ids.device, dtype=torch.long)
                    teacher_forced_condition_tokens.append(teacher_token.view(input_ids.size(0)))
            if eoa in next_tokens_audio or decode_text_only:
                audio_end = True
            if next_token_text == eot:
                text_end = True

            should_log_debug_step = False
            if debug_generation:
                should_log_debug_step = (
                    debug_generation_max_steps <= 0 or step < debug_generation_max_steps
                ) and (step % debug_generation_log_interval == 0)
                if selected_text_token == eot or any(token == eoa for token in selected_audio_tokens):
                    should_log_debug_step = True
                if step == max_new_tokens - 1:
                    should_log_debug_step = True
            if should_log_debug_step:
                if self.use_stop_head:
                    logger.info(
                        "[GEN_DEBUG][STOP] key=%s step=%d stop_logit=%.6f stop_prob=%.6f stop_ready=%s effective_min_step=%d stop_bias=%.6f",
                        sample_key,
                        step,
                        stop_logit_value if stop_logit_value is not None else float("nan"),
                        stop_prob_value if stop_prob_value is not None else float("nan"),
                        stop_ready,
                        effective_min_step if effective_min_step is not None else -1,
                        stop_bias_value,
                    )
                if raw_a0_stats is not None:
                    logger.info(
                        "[GEN_DEBUG][RAW_A0] key=%s step=%d top1_id=%d top1=%s top1_prob=%.6f entropy=%.6f p_eoa=%.6f topk=[%s]",
                        sample_key,
                        step,
                        raw_a0_stats["top1_id"],
                        self._debug_audio_token_repr(raw_a0_stats["top1_id"], eoa, pad_a),
                        raw_a0_stats["top1_prob"],
                        raw_a0_stats["entropy"],
                        raw_a0_stats["p_eoa"],
                        ", ".join(raw_a0_stats["topk_entries"]),
                    )
                text_probs, text_topk_entries = self._debug_topk_entries(
                    xt_logits[-1, :],
                    debug_generation_topk,
                    kind="text",
                    eot=eot,
                    pad_t=pad_t,
                )
                text_selected_prob = float(text_probs[selected_text_token].item())
                text_eot_prob = float(text_probs[eot].item())
                text_summary = (
                    f"text={self._debug_text_token_repr(selected_text_token, eot, pad_t)} "
                    f"sel_p={text_selected_prob:.4f} p_eot={text_eot_prob:.4f} "
                    f"topk=[{', '.join(text_topk_entries)}]"
                )

                audio_summaries = []
                for i in range(self.code_layer):
                    layer_stats = audio_step_stats[i]
                    selected_audio_token = selected_audio_tokens[i]
                    audio_summaries.append(
                        f"a{i}=top1_id:{layer_stats['top1_id']} "
                        f"top1:{self._debug_audio_token_repr(layer_stats['top1_id'], eoa, pad_a)} "
                        f"top1_p={layer_stats['top1_prob']:.4f} "
                        f"entropy={layer_stats['entropy']:.4f} "
                        f"p_eoa={layer_stats['p_eoa']:.4f} "
                        f"sampled={self._debug_audio_token_repr(selected_audio_token, eoa, pad_a)} "
                        f"topk=[{', '.join(layer_stats['topk_entries'])}]"
                    )
                if teacher_forced_condition_tokens is not None:
                    teacher_condition_summary = ", ".join(
                        f"a{i}={self._debug_audio_token_repr(int(token.item()), eoa, pad_a)}"
                        for i, token in enumerate(teacher_forced_condition_tokens)
                    )
                    logger.info(
                        "[GEN_DEBUG][TF] key=%s step=%d cond_prefix=[%s]",
                        sample_key,
                        step,
                        teacher_condition_summary,
                    )
                if (
                    debug_generation_decode_prefix
                    and step >= debug_generation_decode_prefix_min_step
                    and step % debug_generation_decode_prefix_every == 0
                ):
                    prefix_audio_stats = self._debug_decode_prefix_audio(
                        debug_audio_top1_histories,
                        debug_prefix_audio_prompt_path,
                        speech_sample_rate,
                        debug_prefix_tail_window_ms,
                        debug_prefix_tail_silence_threshold,
                        num_latency_tokens,
                    )
                    if prefix_audio_stats is not None:
                        debug_prefix_tail_flags.append({"step": step, "flag": prefix_audio_stats["tail_silent"]})
                        logger.info(
                            "[GEN_DEBUG][PREFIX_AUDIO] key=%s step=%d num_samples=%d full_rms=%.6f tail_rms=%.6f tail_silent=%s",
                            sample_key,
                            step,
                            prefix_audio_stats["num_samples"],
                            prefix_audio_stats["full_rms"],
                            prefix_audio_stats["tail_rms"],
                            prefix_audio_stats["tail_silent"],
                        )
                logger.info(
                    "[GEN_DEBUG] key=%s step=%d trace_mode=%s text_end=%s audio_end=%s %s %s",
                    sample_key,
                    step,
                    trace_mode,
                    text_end,
                    audio_end,
                    text_summary,
                    " ".join(audio_summaries),
                )
            
            # Update input_ids for the next step
            current_input_text = next_token_text
            for i in range(self.code_layer):
                if teacher_forced_condition_tokens is not None:
                    current_audio_tokens[i] = teacher_forced_condition_tokens[i]
                else:
                    current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            # Append generated tokens to the tensor
            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]  # Audio layers
            generated_ids[self.code_layer][step] = next_token_text  # Text layer

            if self.model_config.use_text_stream:
                if audio_end and text_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break       
            else:
                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break     

        if debug_generation:
            collapse_summary = {
                "trace_mode": trace_mode,
                "collapse_runtime_enabled": collapse_runtime_enabled,
                "collapse_window": collapse_window,
                "small_set_size": collapse_small_set_size,
                "entropy_threshold": collapse_entropy_threshold,
                "tail_silence_patience": collapse_tail_silence_patience,
                "attractor_penalty": attractor_penalty,
                "attractor_min_step": attractor_min_step,
                "layers": {},
                "first_prefix_tail_silence_step": self._debug_first_true_window(
                    debug_prefix_tail_flags,
                    collapse_tail_silence_patience,
                ),
            }
            for i in range(self.code_layer):
                collapse_summary["layers"][f"a{i}"] = {
                    "first_small_set_step": self._debug_first_small_set_step(
                        debug_audio_top1_histories[i],
                        collapse_window,
                        collapse_small_set_size,
                    ),
                    "first_low_entropy_step": self._debug_first_low_entropy_step(
                        debug_audio_entropy_histories[i],
                        collapse_window,
                        collapse_entropy_threshold,
                    ),
                }
            logger.info(
                "[GEN_DEBUG] key=%s finish trace_mode=%s text_end=%s audio_end=%s generated_audio_steps=%d",
                sample_key,
                trace_mode,
                text_end,
                audio_end,
                int(generated_ids[0].shape[0]) if self.code_layer > 0 else 0,
            )
            logger.info(
                "[GEN_DEBUG][COLLAPSE] key=%s summary=%s",
                sample_key,
                json.dumps(collapse_summary, ensure_ascii=False, sort_keys=True),
            )

        # Concatenate the generated tokens to form the complete sequence
        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][0]] if eot in text_tokens else text_tokens

        if eoa in generated_ids[self.code_layer - 1] and do_layershift:
            end_ids = (generated_ids[self.code_layer - 1] == eoa).nonzero(as_tuple=True)[0][0]
            for i in range(self.code_layer):
                audio_tokens = generated_ids[i]
                generated_ids[i] = audio_tokens[:end_ids]

        if upsampling_factor > 1:
            generated_ids[self.code_layer] = generated_ids[self.code_layer][::upsampling_factor]
            
        return generated_ids


    @torch.no_grad()
    def serial_generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        if self.use_mtp:
            raise NotImplementedError("serial_generate is not yet adapted for use_mtp=True.")
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer )]
        current_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False     # Track whether text generation has ended
        audio_end = False    # Track whether audio generation has ended

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if None not in current_tokens:
                if text_end:
                    if current_tokens[0].item() == eot:
                        combined_input_ids = torch.cat([ current_tokens[i].unsqueeze(1) for i in range(self.code_layer)], dim=1)
                        text_step = step
                    else:
                        combined_input_ids = torch.cat([layershift(current_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                else:
                    combined_input_ids = torch.cat([ current_tokens[i].unsqueeze(1) for i in range(self.code_layer)], dim=1)
                if self.train_config.use_peft:
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]                      # batch size is 1
            past_key_values = outputs.past_key_values       # Update past_key_values for the next step

            if not text_end:
                xt_logits = logits[..., :text_vocab_size]
                xt_logits = self.repetition_penalty(xt_logits, generated_ids[0][:step], text_repetition_penalty)
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)

                if next_token_text == eot:
                    text_end = True
                
                for i in range(self.code_layer):
                    current_tokens[i] = next_token_text

                attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

                for i in range(self.code_layer):
                    generated_ids[i][step] = next_token_text
                
            else:
                if self.group_decode_adapter is not None:
                    xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                    xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
                else:
                    xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

                for i in range(self.code_layer):
                    xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][text_step:step], audio_repetition_penalty)

                next_tokens_audio = []
                for i in range(self.code_layer):
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                    next_tokens_audio.append(next_token_audio)
                
                if eoa in next_tokens_audio or decode_text_only:
                    audio_end = True

                for i in range(self.code_layer):
                    current_tokens[i] = next_tokens_audio[i]

                attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)
               
                for i in range(self.code_layer):
                    generated_ids[i][step] = next_tokens_audio[i]  # Audio layers

                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break            

        text_tokens = generated_ids[0]
        if eot in text_tokens:
            end_text_id = (text_tokens == eot).nonzero(as_tuple=True)[0][0]
            generated_ids.append(text_tokens[: end_text_id ] if eot in text_tokens else text_tokens)
            for i in range(self.code_layer):
                generated_ids[i] = generated_ids[i][ end_text_id+1: ]
        else:
            generated_ids.append(text_tokens)
            for i in range(self.code_layer):
                generated_ids[i] = []     

        return generated_ids
    
    @torch.no_grad()
    def sample_next_token(self, logits, **kwargs):
        """
        Generate the next token based on the model output logits.
        Supports both greedy decoding, top-k sampling, and top-p (nucleus) sampling.
        """
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 0)
        top_p = kwargs.get("top_p", 1.0)
        num_samples = kwargs.get("num_samples", 1)

        # Adjust logits with temperature
        logits = logits.squeeze(0)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Make sure top_k is within the vocab size
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[..., [-1]]] = -float('Inf')  # Filter tokens not in top_k

        # Top-p filtering (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            # Perform sampling
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples)
        else:
            # Greedy decoding (argmax)
            return torch.argmax(logits, dim=-1, keepdim=True)

    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits
        
    def extract_interleaved_tokens(self, preds, text_labels, preds_start_idx):
        """
        Extract predictions and labels in interleaved mode.
        """
        interleaved_text_num = self.train_config.interleaved_text_token_num
        interleaved_audio_num = self.train_config.interleaved_audio_token_num
        
        text_preds_batch = []
        text_labels_batch = []
        audio_preds_batch = []
        audio_labels_batch = []

        for i in range(preds.size(0)):
            text_preds = []
            text_labels_list = []
            audio_preds = []
            audio_labels = []

            start_idx = preds_start_idx[i].item()
            total_length = preds.size(1)
            idx = start_idx
            
            while idx < total_length:
                if idx + interleaved_text_num <= total_length:
                    text_preds.append(preds[i, idx:idx + interleaved_text_num].unsqueeze(0))
                    text_labels_list.append(text_labels[i, idx:idx + interleaved_text_num].unsqueeze(0))
                idx += interleaved_text_num
                
                if idx + interleaved_audio_num <= total_length:
                    audio_preds.append(preds[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                    audio_labels.append(text_labels[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                idx += interleaved_audio_num
            
            text_preds_batch.append(torch.cat(text_preds, dim=1))
            text_labels_batch.append(torch.cat(text_labels_list, dim=1))
            audio_preds_batch.append(torch.cat(audio_preds, dim=1))
            audio_labels_batch.append(torch.cat(audio_labels, dim=1))

        text_preds = torch.cat(text_preds_batch, dim=0) if text_preds_batch else torch.empty(0, interleaved_text_num)
        text_labels = torch.cat(text_labels_batch, dim=0) if text_labels_batch else torch.empty(0, interleaved_text_num)
        audio_preds = torch.cat(audio_preds_batch, dim=0) if audio_preds_batch else torch.empty(0, interleaved_audio_num)
        audio_labels = torch.cat(audio_labels_batch, dim=0) if audio_labels_batch else torch.empty(0, interleaved_audio_num)
        
        return text_preds, text_labels, audio_preds, audio_labels
        
    def process_interleaved_output(self, model_outputs):
        """
        Parse the interleaved generation results and separate tokens into audio and text.
        """
        batch_size, seq_len = model_outputs.shape
        interleaved_audio_token_num = self.train_config.interleaved_audio_token_num
        interleaved_text_token_num = self.train_config.interleaved_text_token_num
        audio_shift = self.model_config.vocab_config.padded_text_vocabsize

        audio_tokens, text_tokens = [], []

        for i in range(batch_size):
            current_audio, current_text = [], []
            sequence = model_outputs[i]

            idx = 0
            while idx < seq_len:
                text_chunk = sequence[idx: idx + interleaved_text_token_num]
                current_text.append(text_chunk)
                idx += interleaved_text_token_num

                if idx < seq_len:
                    audio_chunk = sequence[idx: idx + interleaved_audio_token_num] - audio_shift
                    current_audio.append(audio_chunk)
                    idx += interleaved_audio_token_num

            audio_tokens.append(torch.cat(current_audio) if current_audio else torch.tensor([], device=model_outputs.device))
            text_tokens.append(torch.cat(current_text) if current_text else torch.tensor([], device=model_outputs.device))

        audio_tokens = torch.stack(audio_tokens) if audio_tokens else torch.empty((batch_size, 0), device=model_outputs.device)
        text_tokens = torch.stack(text_tokens) if text_tokens else torch.empty((batch_size, 0), device=model_outputs.device)

        return {
            "audio": audio_tokens,
            "text": text_tokens.squeeze(0),
        }
