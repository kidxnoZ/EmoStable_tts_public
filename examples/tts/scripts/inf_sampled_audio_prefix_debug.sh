#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
env_file="${ENV_FILE:-$repo_root/.env}"

if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
fi

require_path() {
    local name="$1"
    local value="${!name:-}"
    if [[ -z "$value" || "$value" == /path/to/* ]]; then
        echo "ERROR: set $name in $env_file or the environment" >&2
        exit 2
    fi
    if [[ ! -e "$value" ]]; then
        echo "ERROR: $name does not exist: $value" >&2
        exit 2
    fi
}

for name in LLM_PATH CODEC_DECODER_PATH INFERENCE_CKPT VAL_DATA_PATH; do
    require_path "$name"
done

if [[ -z "${DECODE_LOG:-}" || "$DECODE_LOG" == /path/to/* ]]; then
    echo "ERROR: set DECODE_LOG in $env_file or the environment" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

code_dir="$repo_root/examples/tts"
mkdir -p "$DECODE_LOG"

hydra_args=(
    "hydra.run.dir=$DECODE_LOG"
    "++ckpt_path=$INFERENCE_CKPT"
    "++model_config.llm_name=${LLM_NAME:-Qwen2.5-1.5b}"
    "++model_config.llm_path=$LLM_PATH"
    "++model_config.llm_dim=${LLM_DIM:-1536}"
    "++model_config.codec_decoder_path=$CODEC_DECODER_PATH"
    "++model_config.codec_decode=true"
    "++model_config.vocab_config.code_layer=${CODE_LAYER:-3}"
    "++model_config.vocab_config.total_audio_vocabsize=${TOTAL_AUDIO_VOCABSIZE:-4160}"
    "++model_config.vocab_config.total_vocabsize=${TOTAL_VOCABSIZE:-156160}"
    "++model_config.codec_decoder_type=CosyVoice"
    "++model_config.group_decode=false"
    "++model_config.group_decode_adapter_type=linear"
    "++model_config.use_mtp=true"
    "++model_config.mtp_num=2"
    "++model_config.use_text_stream=false"
    "++model_config.use_stop_head=${USE_STOP_HEAD:-true}"
    "++model_config.stop_pos_weight=${STOP_POS_WEIGHT:-96}"
    "++model_config.stop_loss_weight=${STOP_LOSS_WEIGHT:-0.0}"
    "++model_config.stop_min_step=${STOP_MIN_STEP:-48}"
    "++model_config.stop_bias_lambda=${STOP_BIAS_LAMBDA:-4.0}"
    "++dataset_config.dataset=speech_dataset_tts"
    "++dataset_config.val_data_path=$VAL_DATA_PATH"
    "++dataset_config.train_data_path=$VAL_DATA_PATH"
    "++dataset_config.inference_mode=true"
    "++dataset_config.vocab_config.code_layer=${CODE_LAYER:-3}"
    "++dataset_config.vocab_config.total_audio_vocabsize=${TOTAL_AUDIO_VOCABSIZE:-4160}"
    "++dataset_config.vocab_config.total_vocabsize=${TOTAL_VOCABSIZE:-156160}"
    "++dataset_config.num_latency_tokens=${NUM_LATENCY_TOKENS:-0}"
    "++dataset_config.do_layershift=${DO_LAYERSHIFT:-false}"
    "++dataset_config.use_emo=true"
    "++dataset_config.use_text_stream=false"
    "++train_config.model_name=tts"
    "++train_config.freeze_encoder=true"
    "++train_config.freeze_llm=true"
    "++train_config.freeze_group_decode_adapter=true"
    "++train_config.batching_strategy=custom"
    "++train_config.num_epochs=1"
    "++train_config.val_batch_size=1"
    "++train_config.num_workers_dataloader=${NUM_WORKERS_DATALOADER:-2}"
    "++decode_config.text_repetition_penalty=${TEXT_REPETITION_PENALTY:-1.2}"
    "++decode_config.audio_repetition_penalty=${AUDIO_REPETITION_PENALTY:-1.2}"
    "++decode_config.max_new_tokens=${MAX_NEW_TOKENS:-256}"
    "++decode_config.do_sample=${DO_SAMPLE:-false}"
    "++decode_config.top_p=${TOP_P:-1.0}"
    "++decode_config.top_k=${TOP_K:-0}"
    "++decode_config.temperature=${TEMPERATURE:-1.0}"
    "++decode_config.decode_text_only=${DECODE_TEXT_ONLY:-false}"
    "++decode_config.num_latency_tokens=${NUM_LATENCY_TOKENS:-0}"
    "++decode_config.do_layershift=${DO_LAYERSHIFT:-false}"
    "++decode_config.debug_generation=${DEBUG_GENERATION:-false}"
    "++decode_config.debug_generation_topk=${DEBUG_GENERATION_TOPK:-8}"
    "++decode_config.debug_generation_max_steps=${DEBUG_GENERATION_MAX_STEPS:-256}"
    "++decode_config.debug_generation_log_interval=${DEBUG_GENERATION_LOG_INTERVAL:-1}"
    "++decode_config.debug_max_samples=${DEBUG_MAX_SAMPLES:-0}"
    "++decode_config.oracle_mtp_conditioning=${ORACLE_MTP_CONDITIONING:-false}"
    "++decode_config.debug_decode_overlong_audio=${DEBUG_DECODE_OVERLONG_AUDIO:-true}"
    "++decode_config.teacher_forced_audio_prefix=${TEACHER_FORCED_AUDIO_PREFIX:-false}"
    "++decode_config.debug_generation_decode_prefix=${DEBUG_GENERATION_DECODE_PREFIX:-false}"
    "++decode_config.debug_generation_decode_prefix_every=${DEBUG_GENERATION_DECODE_PREFIX_EVERY:-1}"
    "++decode_config.debug_generation_decode_prefix_min_step=${DEBUG_GENERATION_DECODE_PREFIX_MIN_STEP:-0}"
    "++decode_config.debug_collapse_window=${DEBUG_COLLAPSE_WINDOW:-8}"
    "++decode_config.debug_collapse_small_set_size=${DEBUG_COLLAPSE_SMALL_SET_SIZE:-2}"
    "++decode_config.debug_entropy_threshold=${DEBUG_ENTROPY_THRESHOLD:-2.0}"
    "++decode_config.debug_prefix_tail_window_ms=${DEBUG_PREFIX_TAIL_WINDOW_MS:-250}"
    "++decode_config.debug_prefix_tail_silence_threshold=${DEBUG_PREFIX_TAIL_SILENCE_THRESHOLD:-0.003}"
    "++decode_config.debug_prefix_silence_patience=${DEBUG_PREFIX_SILENCE_PATIENCE:-3}"
    "++decode_config.use_collapse_aware_decoding=${USE_COLLAPSE_AWARE_DECODING:-true}"
    "++decode_config.collapse_window=${COLLAPSE_WINDOW:-8}"
    "++decode_config.collapse_small_set_size=${COLLAPSE_SMALL_SET_SIZE:-2}"
    "++decode_config.collapse_entropy_threshold=${COLLAPSE_ENTROPY_THRESHOLD:-2.0}"
    "++decode_config.collapse_tail_silence_patience=${COLLAPSE_TAIL_SILENCE_PATIENCE:-3}"
    "++decode_config.attractor_penalty=${ATTRACTOR_PENALTY:-1.5}"
    "++decode_config.attractor_min_step=${ATTRACTOR_MIN_STEP:-4}"
    "++decode_config.collapse_hardcase_only=${COLLAPSE_HARDCASE_ONLY:-false}"
    "++decode_config.collapse_debug_dataset_path=$VAL_DATA_PATH"
    "++decode_log=$DECODE_LOG"
    "++output_text_only=${OUTPUT_TEXT_ONLY:-false}"
    "++speech_sample_rate=${SPEECH_SAMPLE_RATE:-22050}"
    "++log_config.log_file=$DECODE_LOG/infer.log"
)

exec "$python_bin" "$code_dir/inference_tts.py" "${hydra_args[@]}" "$@"

