#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 DECODE_LOG_DIR" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
parent_dir=$1
gt_text="$parent_dir/gt_text"
pred_text="$parent_dir/pred_text"
pred_whisper="$parent_dir/pred_whisper_text"

python "$repo_root/src/slam_llm/utils/whisper_tn.py" "$gt_text" "${gt_text}.proc"
python "$repo_root/src/slam_llm/utils/whisper_tn.py" "$pred_text" "${pred_text}.proc"
python "$repo_root/src/slam_llm/utils/compute_wer.py" "${gt_text}.proc" "${pred_text}.proc" "${pred_text}.wer"
tail -3 "${pred_text}.wer"

python "$repo_root/src/slam_llm/utils/whisper_tn.py" "$pred_whisper" "${pred_whisper}.proc"
python "$repo_root/src/slam_llm/utils/compute_wer.py" "${gt_text}.proc" "${pred_whisper}.proc" "${pred_whisper}.wer"
tail -3 "${pred_whisper}.wer"
