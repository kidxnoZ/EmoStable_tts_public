from slam_llm.utils.train_utils import print_module_size
import torch
import torchaudio
import os
import torch.nn as nn
import uuid
import logging
logger = logging.getLogger(__name__)

def setup_codec(train_config, model_config, **kwargs):
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
    if model_config.cosyvoice_version==1:
        codec_decoder = CosyVoice(model_config.codec_decoder_path, load_jit=False, load_trt=False, fp16=False)
    elif model_config.cosyvoice_version==2:
        codec_decoder = CosyVoice2(model_config.codec_decoder_path, load_jit=False, load_trt=False, fp16=False)
    else:
        raise NotImplementedError
    codec_decoder_module = nn.ModuleList((codec_decoder.model.flow,codec_decoder.model.hift))

    print_module_size(codec_decoder_module, model_config.codec_decoder_type + " Codec", int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return codec_decoder

def get_single_layer_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio):
    audio_length = len(audio_tokens) + num_latency_tokens + 1  # 1 is due to end of audio token
    result = [padding_token] * num_latency_tokens + list(audio_tokens) + [end_of_audio]
    result_tensor = torch.tensor(result).unsqueeze(0)
    return result_tensor, audio_length

def get_group_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio, num_layers):
    padded_audio_tokens = audio_tokens + [end_of_audio]
    padding_needed = (num_layers - len(padded_audio_tokens) % num_layers ) % num_layers
    
    # Add padding to ensure even distribution across layers
    padded_audio_tokens = padded_audio_tokens + [padding_token] * padding_needed
    total_length = len(padded_audio_tokens)
    audio_length = total_length // num_layers + num_latency_tokens

    # Create the result for each layer
    result = []
    for layer in range(num_layers):
        layer_tokens = [padding_token] * num_latency_tokens
        layer_tokens.extend(padded_audio_tokens[layer::num_layers])
        result.append(torch.tensor(layer_tokens))
    
    result_tensor = torch.stack(result)
    return result_tensor, audio_length

def audio_decode_cosyvoice(
    audio_tokens,
    model_config,
    codec_decoder,
    audio_prompt_path=None,
    code_layer=1,
    num_latency_tokens=1,
    speed=1.0,
    replace_token=4095,
    allow_missing_eoa=False,
):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Generated audio waveform.
    """
    # Reshape audio tokens based on code_layer
    if code_layer > 1:
        audio_tokens_tensor = torch.stack(audio_tokens, dim=0)
        audio_tokens_permuted = audio_tokens_tensor.permute(1, 0)
        audio_tokens = audio_tokens_permuted.reshape(-1).unsqueeze(0)
        audio_tokens = audio_tokens[..., num_latency_tokens * code_layer:]
    elif code_layer == 1:
        audio_tokens = torch.cat(audio_tokens, dim=-1).unsqueeze(0)
        audio_tokens = audio_tokens[..., num_latency_tokens:]
    else:
        audio_tokens = audio_tokens[..., num_latency_tokens:]

    # Get vocabulary configuration for end of audio (EOA) and padding token
    eoa = model_config.vocab_config.eoa
    pad_a = model_config.vocab_config.pad_a

    # Truncate audio tokens at the EOA token when present.
    # In debug mode we may explicitly allow decoding the full generated
    # sequence even if EOA never appears, to inspect overlong failures.
    if eoa in audio_tokens[0]:
        end_index = torch.nonzero(audio_tokens[0] == eoa)[0]
        audio_tokens = audio_tokens[..., :end_index]
    elif not allow_missing_eoa:
        return None
    else:
        logger.warning("EOA not found in generated audio tokens; decoding full generated length because allow_missing_eoa=True.")

    # Handle padding tokens if present, # FIXME: this is a temporary fix for the padding issue, where the padding token may be included in the audio tokens
    if pad_a in audio_tokens:
        audio_tokens = audio_tokens.masked_fill(audio_tokens == pad_a, replace_token)
    if model_config.save_audio_token:
        return audio_tokens
    if audio_tokens.numel()==0: 
        return None

    this_uuid = str(uuid.uuid1())  # Generate a unique ID for this audio generation

    from utils.cosyvoice.utils.file_utils import load_wav
    prompt_speech_16k = load_wav(audio_prompt_path, 16000)
    flow_prompt_speech_token, flow_prompt_speech_token_len = codec_decoder.frontend._extract_speech_token(prompt_speech_16k)
    if model_config.cosyvoice_version==1:
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_22050)
    elif model_config.cosyvoice_version==2:
        prompt_speech_24000 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_24000)
    flow_embedding = codec_decoder.frontend._extract_spk_embedding(prompt_speech_16k)

    # Convert tokens to audio waveform
    if model_config.cosyvoice_version==1:
        audio_hat = codec_decoder.model.token2wav(
            token=audio_tokens,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            finalize=True,
            speed=speed
        )
    elif model_config.cosyvoice_version==2:
        audio_hat = codec_decoder.model.token2wav(
            token=audio_tokens,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            token_offset=0,
            finalize=True,
            speed=speed
        )
    else:
        raise NotImplementedError
    return audio_hat

def layershift(input_id, layer, stride=4160, shift=152000):
    return input_id + shift + layer * stride

def simple_shift(input_id, layer, stride=4160, shift=152000):
    return input_id + shift
