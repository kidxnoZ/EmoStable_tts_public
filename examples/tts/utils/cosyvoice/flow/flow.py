# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000},
                 enable_emotion_consistency_loss: bool = False,
                 enable_f0_loss: bool = False,
                 enable_energy_loss: bool = False,
                 enable_duration_loss: bool = False,
                 num_emotions: int = 7,
                 lambda_ecl: float = 0.0,
                 lambda_f0: float = 0.0,
                 lambda_energy: float = 0.0,
                 lambda_dur: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss
        self.enable_emotion_consistency_loss = enable_emotion_consistency_loss
        self.enable_f0_loss = enable_f0_loss
        self.enable_energy_loss = enable_energy_loss
        self.enable_duration_loss = enable_duration_loss
        self.num_emotions = num_emotions
        self.lambda_ecl = lambda_ecl
        self.lambda_f0 = lambda_f0
        self.lambda_energy = lambda_energy
        self.lambda_dur = lambda_dur

        hidden_size = max(output_size // 2, 16)
        self.emotion_classifier = None
        if self.enable_emotion_consistency_loss:
            self.emotion_classifier = nn.Sequential(
                nn.Linear(output_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_emotions),
            )

        self.f0_predictor = None
        if self.enable_f0_loss:
            self.f0_predictor = nn.Linear(output_size, 1)

        self.energy_predictor = None
        if self.enable_energy_loss:
            self.energy_predictor = nn.Linear(output_size, 1)

        self.duration_predictor = None
        if self.enable_duration_loss:
            self.duration_predictor = nn.Sequential(
                nn.Linear(output_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

    @staticmethod
    def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(hidden_states.dtype).unsqueeze(-1)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weight).sum(dim=1) / denom

    @staticmethod
    def _masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(pred.dtype)
        denom = weight.sum().clamp_min(1.0)
        return (torch.abs(pred - target) * weight).sum() / denom

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_mask = (~make_pad_mask(token_len)).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * token_mask.float().unsqueeze(-1)

        # text encode
        h_token, h_lengths = self.encoder(token, token_len)
        h_token = self.encoder_proj(h_token)
        h, h_lengths = self.length_regulator(h_token, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        feat_mask = (~make_pad_mask(feat_len)).to(h)
        feat_mask_bool = feat_mask.bool()
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)
        base_loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            feat_mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )

        loss_ecl = base_loss.new_zeros(())
        loss_f0 = base_loss.new_zeros(())
        loss_energy = base_loss.new_zeros(())
        loss_dur = base_loss.new_zeros(())

        if self.enable_emotion_consistency_loss and self.emotion_classifier is not None and 'emotion_id' in batch:
            emotion_target = batch['emotion_id'].to(device).long()
            valid_emotion = (emotion_target >= 0) & (emotion_target < self.num_emotions)
            if valid_emotion.any():
                prosody_summary = self._masked_mean(h, feat_mask_bool)
                emotion_logits = self.emotion_classifier(prosody_summary)
                loss_ecl = F.cross_entropy(emotion_logits[valid_emotion], emotion_target[valid_emotion])

        if self.enable_f0_loss and self.f0_predictor is not None and 'pitch_feat' in batch:
            pitch_target = batch['pitch_feat'].to(device).to(h.dtype)
            if pitch_target.dim() == 3:
                pitch_target = pitch_target.squeeze(-1)
            if pitch_target.shape[1] != h.shape[1]:
                pitch_target = F.interpolate(
                    pitch_target.unsqueeze(1),
                    size=h.shape[1],
                    mode='linear',
                    align_corners=False,
                ).squeeze(1)
            pitch_pred = self.f0_predictor(h).squeeze(-1)
            pitch_mask = feat_mask_bool
            if 'pitch_feat_valid' in batch:
                pitch_valid = batch['pitch_feat_valid'].to(device).bool()
                pitch_mask = pitch_mask & pitch_valid.unsqueeze(1)
            if pitch_mask.any():
                loss_f0 = self._masked_l1_loss(pitch_pred, pitch_target, pitch_mask)

        if self.enable_energy_loss and self.energy_predictor is not None:
            energy_target = feat.abs().mean(dim=-1)
            energy_pred = self.energy_predictor(h).squeeze(-1)
            loss_energy = self._masked_l1_loss(energy_pred, energy_target, feat_mask_bool)

        if self.enable_duration_loss and self.duration_predictor is not None:
            duration_hidden = self._masked_mean(h_token, token_mask)
            duration_pred = F.softplus(self.duration_predictor(duration_hidden).squeeze(-1))
            duration_target = feat_len.to(duration_pred.dtype)
            loss_dur = F.l1_loss(torch.log1p(duration_pred), torch.log1p(duration_target))

        total_loss = base_loss + self.lambda_ecl * loss_ecl + self.lambda_f0 * loss_f0 + \
            self.lambda_energy * loss_energy + self.lambda_dur * loss_dur
        return {
            'loss': total_loss,
            'base_loss': base_loss,
            'emotion_cls_loss': loss_ecl,
            'f0_loss': loss_f0,
            'energy_loss': loss_energy,
            'dur_loss': loss_dur,
        }

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None
