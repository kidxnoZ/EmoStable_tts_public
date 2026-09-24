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

for name in LLM_PATH BASE_CKPT TRAIN_DATA_PATH VAL_DATA_PATH; do
    require_path "$name"
done

if [[ -z "${OUTPUT_DIR:-}" || "$OUTPUT_DIR" == /path/to/* ]]; then
    echo "ERROR: set OUTPUT_DIR in $env_file or the environment" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

code_dir="$repo_root/examples/tts"
num_gpus_per_node=$(( $(tr -cd ',' <<<"$CUDA_VISIBLE_DEVICES" | wc -c) + 1 ))
num_nodes="${NUM_NODES:-1}"

llm_name="${LLM_NAME:-Qwen2.5-1.5b}"
llm_dim="${LLM_DIM:-1536}"
code_layer="${CODE_LAYER:-3}"
total_audio_vocabsize="${TOTAL_AUDIO_VOCABSIZE:-4160}"
llm_vocabsize="${LLM_VOCABSIZE:-152000}"
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

batch_size_training="${BATCH_SIZE_TRAINING:-1}"
num_epochs="${NUM_EPOCHS:-5}"
learning_rate="${LEARNING_RATE:-5e-6}"
warmup_steps="${WARMUP_STEPS:-2000}"
total_steps="${TOTAL_STEPS:-200000}"
validation_interval="${VALIDATION_INTERVAL:-2500}"
split_size="${SPLIT_SIZE:-0.01}"

stop_pos_weight="${STOP_POS_WEIGHT:-96}"
stop_loss_weight="${STOP_LOSS_WEIGHT:-0.5}"
stop_min_step="${STOP_MIN_STEP:-48}"
stop_bias_lambda="${STOP_BIAS_LAMBDA:-4.0}"

exp_name="${EXP_NAME:-emovoice_sampled_prefix_stop_joint}"
use_wandb="${USE_WANDB:-false}"
wandb_entity="${WANDB_ENTITY:-}"
wandb_project="${WANDB_PROJECT:-emovoice-stability}"

mkdir -p "$OUTPUT_DIR"

hydra_args=(
    "hydra.run.dir=$OUTPUT_DIR"
    "++ckpt_path=$BASE_CKPT"
    "++model_config.llm_name=$llm_name"
    "++model_config.llm_path=$LLM_PATH"
    "++model_config.llm_dim=$llm_dim"
    "++model_config.vocab_config.code_layer=$code_layer"
    "++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize"
    "++model_config.vocab_config.total_vocabsize=$total_vocabsize"
    "++model_config.group_decode=false"
    "++model_config.group_decode_adapter_type=linear"
    "++model_config.use_mtp=true"
    "++model_config.mtp_num=2"
    "++model_config.use_text_stream=false"
    "++model_config.use_stop_head=true"
    "++model_config.stop_pos_weight=$stop_pos_weight"
    "++model_config.stop_loss_weight=$stop_loss_weight"
    "++model_config.stop_min_step=$stop_min_step"
    "++model_config.stop_bias_lambda=$stop_bias_lambda"
    "++model_config.use_sampled_audio_prefix_training=true"
    "++model_config.sampled_audio_prefix_prob_start=${SAMPLED_PREFIX_PROB_START:-0.0}"
    "++model_config.sampled_audio_prefix_prob_end=${SAMPLED_PREFIX_PROB_END:-0.2}"
    "++model_config.sampled_audio_prefix_warmup_steps=${SAMPLED_PREFIX_WARMUP_STEPS:-5000}"
    "++model_config.sampled_audio_prefix_chunk_len=${SAMPLED_PREFIX_CHUNK_LEN:-16}"
    "++model_config.sampled_audio_prefix_sampling=${SAMPLED_PREFIX_SAMPLING:-greedy}"
    "++model_config.freeze_stop_head_during_sampled_prefix=${FREEZE_STOP_HEAD_DURING_SAMPLED_PREFIX:-false}"
    "++model_config.disable_stop_loss_during_sampled_prefix=${DISABLE_STOP_LOSS_DURING_SAMPLED_PREFIX:-false}"
    "++dataset_config.dataset=speech_dataset_tts"
    "++dataset_config.train_data_path=$TRAIN_DATA_PATH"
    "++dataset_config.val_data_path=$VAL_DATA_PATH"
    "++dataset_config.seed=${SEED:-42}"
    "++dataset_config.split_size=$split_size"
    "++dataset_config.vocab_config.code_layer=$code_layer"
    "++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize"
    "++dataset_config.vocab_config.total_vocabsize=$total_vocabsize"
    "++dataset_config.num_latency_tokens=${NUM_LATENCY_TOKENS:-0}"
    "++dataset_config.do_layershift=${DO_LAYERSHIFT:-false}"
    "++dataset_config.use_emo=true"
    "++dataset_config.use_text_stream=false"
    "++train_config.model_name=tts"
    "++train_config.num_epochs=$num_epochs"
    "++train_config.freeze_encoder=true"
    "++train_config.freeze_llm=false"
    "++train_config.freeze_group_decode_adapter=true"
    "++train_config.batching_strategy=custom"
    "++train_config.warmup_steps=$warmup_steps"
    "++train_config.total_steps=$total_steps"
    "++train_config.lr=$learning_rate"
    "++train_config.validation_interval=$validation_interval"
    "++train_config.batch_size_training=$batch_size_training"
    "++train_config.val_batch_size=$batch_size_training"
    "++train_config.checkpoint_top_k=${CHECKPOINT_TOP_K:-6}"
    "++train_config.checkpoint_monitor=${CHECKPOINT_MONITOR:-val_loss}"
    "++train_config.checkpoint_monitor_mode=${CHECKPOINT_MONITOR_MODE:-min}"
    "++train_config.num_workers_dataloader=${NUM_WORKERS_DATALOADER:-0}"
    "++train_config.output_dir=$OUTPUT_DIR"
    "++train_config.use_fp16=${USE_FP16:-true}"
    "++train_config.use_peft=${USE_PEFT:-false}"
    "++metric=acc"
    "++log_config.use_wandb=$use_wandb"
    "++log_config.wandb_entity_name=$wandb_entity"
    "++log_config.wandb_project_name=$wandb_project"
    "++log_config.wandb_exp_name=$exp_name"
    "++log_config.wandb_dir=$OUTPUT_DIR"
    "++log_config.log_file=$OUTPUT_DIR/train.log"
    "++log_config.log_interval=${LOG_INTERVAL:-100}"
)

if [[ "$num_gpus_per_node" -eq 1 ]]; then
    exec "$python_bin" "$code_dir/finetune_tts.py" "${hydra_args[@]}" "$@"
fi

exec torchrun \
    --nnodes "$num_nodes" \
    --nproc_per_node "$num_gpus_per_node" \
    --master_port "${MASTER_PORT:-29503}" \
    "$code_dir/finetune_tts.py" \
    "++train_config.enable_ddp=true" \
    "++train_config.enable_fsdp=false" \
    "${hydra_args[@]}" \
    "$@"

