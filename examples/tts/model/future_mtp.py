import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm


class Qwen2FutureTokenDecoder(nn.Module):
    def __init__(self, qwen_config, hidden_size, audio_vocab_size, mtp_num, audio_pad_token):
        super().__init__()
        self.hidden_size = hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.mtp_num = int(mtp_num)
        self.audio_pad_token = int(audio_pad_token)
        self.audio_embedding = nn.Embedding(
            audio_vocab_size,
            hidden_size,
            padding_idx=self.audio_pad_token,
        )

        self.mtp_layers = nn.ModuleList()
        for _ in range(self.mtp_num):
            self.mtp_layers.append(
                nn.ModuleDict(
                    {
                        "decoder_layer": Qwen2DecoderLayer(qwen_config, layer_idx=0),
                        "norm_final": Qwen2RMSNorm(qwen_config.hidden_size, eps=qwen_config.rms_norm_eps),
                        "output_proj": nn.Linear(hidden_size, audio_vocab_size),
                    }
                )
            )

    @staticmethod
    def _build_position_ids(attention_mask):
        if attention_mask is None:
            return None
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return position_ids

    def _sanitize_audio_tokens(self, audio_tokens):
        if audio_tokens is None:
            raise ValueError("audio_tokens must not be None.")
        if audio_tokens.dim() != 2:
            raise ValueError(f"audio_tokens must have shape [B, T], got {tuple(audio_tokens.shape)}")

        safe_tokens = audio_tokens.long().clone()
        safe_tokens.masked_fill_(safe_tokens.eq(-100), self.audio_pad_token)
        safe_tokens.masked_fill_(safe_tokens.lt(0), self.audio_pad_token)
        if safe_tokens.numel() > 0:
            max_token = int(safe_tokens.max().item())
            min_token = int(safe_tokens.min().item())
            if min_token < 0 or max_token >= self.audio_vocab_size:
                raise ValueError(
                    "audio token ids must stay within the audio vocabulary range after sanitization "
                    f"(got min={min_token}, max={max_token}, vocab={self.audio_vocab_size})."
                )
        return safe_tokens

    def _build_teacher_conditioning_tokens(self, audio_labels):
        if audio_labels is None:
            return None
        if audio_labels.dim() == 3:
            if audio_labels.shape[1] != 1:
                raise ValueError(
                    "future MTP expects a single audio stream during training, "
                    f"got shape {tuple(audio_labels.shape)}."
                )
            audio_labels = audio_labels[:, 0, :]
        elif audio_labels.dim() != 2:
            raise ValueError(f"audio_labels must have shape [B, T] or [B, 1, T], got {tuple(audio_labels.shape)}")

        safe_tokens = self._sanitize_audio_tokens(audio_labels)
        batch_size, seq_len = safe_tokens.shape
        conditioning_tokens = []
        for layer_idx in range(self.mtp_num):
            shift = layer_idx + 1
            if shift >= seq_len:
                conditioning_tokens.append(
                    torch.full(
                        (batch_size, seq_len),
                        self.audio_pad_token,
                        dtype=safe_tokens.dtype,
                        device=safe_tokens.device,
                    )
                )
                continue
            pad_column = torch.full(
                (batch_size, shift),
                self.audio_pad_token,
                dtype=safe_tokens.dtype,
                device=safe_tokens.device,
            )
            conditioning_tokens.append(torch.cat([safe_tokens[:, shift:], pad_column], dim=1))
        return conditioning_tokens

    def _validate_conditioning_tokens(self, hidden_states, conditioning_audio_tokens):
        if conditioning_audio_tokens is None:
            raise ValueError("conditioning_audio_tokens must not be None.")
        if not isinstance(conditioning_audio_tokens, (list, tuple)):
            raise TypeError(
                "conditioning_audio_tokens must be a list or tuple of [B, T] audio token tensors, "
                f"got {type(conditioning_audio_tokens).__name__}."
            )
        if len(conditioning_audio_tokens) <= 0:
            raise ValueError("conditioning_audio_tokens must contain at least one future conditioning stream.")
        if len(conditioning_audio_tokens) > self.mtp_num:
            raise ValueError(
                f"conditioning_audio_tokens length must be <= mtp_num ({self.mtp_num}), got {len(conditioning_audio_tokens)}."
            )

        batch_size, seq_len = hidden_states.shape[:2]
        safe_conditioning_tokens = []
        for layer_idx, audio_tokens in enumerate(conditioning_audio_tokens):
            safe_tokens = self._sanitize_audio_tokens(audio_tokens)
            if safe_tokens.shape != (batch_size, seq_len):
                raise ValueError(
                    "conditioning audio token shape must match hidden state sequence shape "
                    f"(layer={layer_idx}, got {tuple(safe_tokens.shape)} vs {(batch_size, seq_len)})."
                )
            safe_conditioning_tokens.append(safe_tokens)
        return safe_conditioning_tokens

    @staticmethod
    def _build_full_causal_mask(hidden_states, attention_mask=None):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        dtype = hidden_states.dtype
        min_dtype = torch.finfo(dtype).min

        if attention_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        else:
            valid_mask = attention_mask.bool()
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        allowed = causal.unsqueeze(0) & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)

        future_attention_mask = torch.full(
            (batch_size, 1, seq_len, seq_len),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        future_attention_mask = future_attention_mask.masked_fill(allowed.unsqueeze(1), 0)
        return future_attention_mask

    def _decode(self, hidden_states, mtp_attention_mask, position_ids, conditioning_audio_tokens):
        mtp_logits = []
        current_hidden_states = hidden_states
        for layer_idx, mtp_layer in enumerate(self.mtp_layers[: len(conditioning_audio_tokens)]):
            audio_token_embeds = self.audio_embedding(conditioning_audio_tokens[layer_idx])
            conditioned_hidden_states = current_hidden_states + audio_token_embeds
            current_hidden_states = mtp_layer["decoder_layer"](
                conditioned_hidden_states,
                attention_mask=mtp_attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                use_cache=False,
            )[0]
            current_output = mtp_layer["norm_final"](current_hidden_states)
            mtp_logits.append(mtp_layer["output_proj"](current_output))
        return mtp_logits

    @staticmethod
    def _build_incremental_causal_mask(hidden_states, past_length):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        dtype = hidden_states.dtype
        min_dtype = torch.finfo(dtype).min

        total_len = seq_len + int(past_length)
        causal = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
        causal = causal[-seq_len:, :].unsqueeze(0).unsqueeze(1)
        attention_mask = torch.full(
            (batch_size, 1, seq_len, total_len),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        attention_mask = attention_mask.masked_fill(causal, 0)
        return attention_mask

    def init_generation(self, prefix_hidden_states):
        if prefix_hidden_states.dim() != 3:
            raise ValueError(
                f"prefix_hidden_states must have shape [B, T, H], got {tuple(prefix_hidden_states.shape)}"
            )

        batch_size, seq_len = prefix_hidden_states.shape[:2]
        future_caches = []
        if seq_len == 0:
            for _ in self.mtp_layers:
                future_caches.append(DynamicCache.from_legacy_cache(None))
            return future_caches

        device = prefix_hidden_states.device
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        future_attention_mask = self._build_full_causal_mask(prefix_hidden_states, attention_mask)
        position_ids = self._build_position_ids(attention_mask)
        pad_tokens = torch.full(
            (batch_size, seq_len),
            self.audio_pad_token,
            dtype=torch.long,
            device=device,
        )

        current_hidden_states = prefix_hidden_states
        for mtp_layer in self.mtp_layers:
            future_cache = DynamicCache.from_legacy_cache(None)
            conditioned_hidden_states = current_hidden_states + self.audio_embedding(pad_tokens)
            current_hidden_states = mtp_layer["decoder_layer"](
                conditioned_hidden_states,
                attention_mask=future_attention_mask,
                position_ids=position_ids,
                past_key_value=future_cache,
                output_attentions=False,
                use_cache=True,
            )[0]
            future_caches.append(future_cache)
        return future_caches

    def infer_one_layer_chunk(self, hidden_states, conditioning_audio_tokens, layer_idx, future_cache):
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape [B, T, H] for incremental future-MTP inference, "
                f"got {tuple(hidden_states.shape)}."
            )
        if layer_idx < 0 or layer_idx >= self.mtp_num:
            raise ValueError(f"layer_idx must be in [0, {self.mtp_num - 1}], got {layer_idx}.")
        if future_cache is None:
            raise ValueError("future_cache must not be None for incremental future-MTP inference.")

        safe_conditioning_tokens = self._sanitize_audio_tokens(conditioning_audio_tokens)
        if safe_conditioning_tokens.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "conditioning_audio_tokens must match hidden_states shape [B, T] during incremental future-MTP inference, "
                f"got conditioning={tuple(safe_conditioning_tokens.shape)} vs hidden={tuple(hidden_states.shape[:2])}."
            )

        conditioned_hidden_states = hidden_states + self.audio_embedding(safe_conditioning_tokens)
        past_length = future_cache.get_seq_length()
        attention_mask = self._build_incremental_causal_mask(conditioned_hidden_states, past_length)
        position_ids = torch.arange(
            past_length,
            past_length + conditioned_hidden_states.shape[1],
            device=conditioned_hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0)

        mtp_layer = self.mtp_layers[layer_idx]
        next_hidden_states = mtp_layer["decoder_layer"](
            conditioned_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=future_cache,
            output_attentions=False,
            use_cache=True,
        )[0]
        logits = mtp_layer["output_proj"](mtp_layer["norm_final"](next_hidden_states))
        return next_hidden_states, logits

    def infer_one_layer_step(self, hidden_states, conditioning_audio_token, layer_idx, future_cache):
        if hidden_states.dim() != 3 or hidden_states.shape[1] != 1:
            raise ValueError(
                "hidden_states must have shape [B, 1, H] for single-step future-MTP inference, "
                f"got {tuple(hidden_states.shape)}."
            )
        if conditioning_audio_token.dim() != 2 or conditioning_audio_token.shape[1] != 1:
            raise ValueError(
                "conditioning_audio_token must have shape [B, 1] for single-step future-MTP inference, "
                f"got {tuple(conditioning_audio_token.shape)}."
            )
        return self.infer_one_layer_chunk(hidden_states, conditioning_audio_token, layer_idx, future_cache)

    def forward(self, hidden_states, attention_mask, audio_labels):
        conditioning_audio_tokens = self._build_teacher_conditioning_tokens(audio_labels)
        conditioning_audio_tokens = self._validate_conditioning_tokens(hidden_states, conditioning_audio_tokens)
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
        future_attention_mask = self._build_full_causal_mask(hidden_states, attention_mask)
        position_ids = self._build_position_ids(attention_mask)
        return self._decode(hidden_states, future_attention_mask, position_ids, conditioning_audio_tokens)
