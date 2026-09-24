# EmoVoice Emotional DPO Data Pipeline

This folder adds a minimal-invasive, script-based pipeline to construct DPO pairs for emotional TTS.

## Scripts

1. `extract_emotion2vec_embeddings.py`  
Extract emotion2vec embeddings + classifier probabilities from ground-truth audio.

2. `compute_emotion_centroids.py`  
Compute per-emotion centroid vectors from extracted embeddings.

3. `generate_dpo_candidates.py`  
Generate `K` candidate audios per sample with fixed input conditions.

4. `score_dpo_candidates.py`  
Score each candidate with:

```
score =
  1.0 * log P(target_emotion | audio)
+ 0.5 * emo2vec cosine similarity to target centroid
- 0.8 * log P(neutral | audio)   # only for non-neutral targets
- 0.3 * WER(audio, text)
+ 0.2 * speaker similarity(audio, ref)
```

5. `build_dpo_pairs.py`  
Build `(chosen, rejected)` pairs with hard negative preference.

6. `filter_dpo_pairs_by_quality.py`  
Compute pair-level `WER + UTMOS`, save enriched metadata, and optionally drop pairs below quality thresholds.

## Required Inputs

- Dataset jsonl with at least:
  - `key`
  - `source_text`
  - `target_text`
  - `emotion` (recommended)
  - `emotion_text_prompt` (recommended)
  - `target_wav` (for step 1)
  - `neutral_speaker_wav` (recommended for speaker similarity and candidate generation)
- A trained EmoVoice checkpoint (`model.pt`) and related model paths for generation.

## Dependencies

The scripts reuse dependencies already used in this repo:

- `funasr` (`iic/emotion2vec_plus_large`)
- `whisper`
- `whisper_normalizer`
- `torch`, `torchaudio`, `numpy`, `soundfile`
- `cosyvoice` frontend stack for campplus speaker embedding (optional but recommended)

If one of these is missing in your environment, install it before running.

## End-to-End Example

Run from the repository root.

### 1) Extract emotion2vec embeddings from GT audio

```bash
python examples/tts/dpo/extract_emotion2vec_embeddings.py \
  --dataset_jsonl train.jsonl \
  --embedding_dir dpo_work/gt_embeddings \
  --output_jsonl dpo_work/gt_embedding_meta.jsonl \
  --resume true
```

### 2) Compute per-emotion centroids

```bash
python examples/tts/dpo/compute_emotion_centroids.py \
  --embedding_meta_jsonl dpo_work/gt_embedding_meta.jsonl \
  --reject_predicted_emotions unk,other \
  --min_target_prob 0.05 \
  --output_json dpo_work/emotion_centroids.json \
  --resume true
```

### 3) Generate K candidates (fixed condition, different sampling seeds)

```bash
python examples/tts/dpo/generate_dpo_candidates.py \
  --dataset_jsonl train.jsonl \
  --output_dir /path/to/dpo_work/candidates_run1 \
  --k 4 \
  --seed 42 \
  --speaker_ref_jsonl test.jsonl \
  --gt_embedding_meta_jsonl dpo_work/gt_embedding_meta.jsonl \
  --filter_by_gt_confidence true \
  --gt_reject_predicted_emotions unk,other \
  --gt_min_target_prob 0.05 \
  --llm_path /path/to/Qwen2.5-0.5B \
  --ckpt_path /path/to/EmoVoice-PP.pt \
  --codec_decoder_path /path/to/CosyVoice-300M-SFT \
  --do_sample true \
  --top_p 1.0 \
  --temperature 1.0 \
  --max_new_tokens 3000 \
  --fallback_ref_to_target_wav true \
  --resume true
```

Output:

- candidate wavs under `.../candidates/<key>/cand_XXX.wav`
- candidate metadata jsonl: `.../candidate_metadata.jsonl`
- metadata also stores `generated_text_token_ids` and `generated_audio_token_ids` for DPO training log-prob computation

Note:

- Recommended: if `train.jsonl` has no `neutral_speaker_wav`, provide `--speaker_ref_jsonl test.jsonl`
  (script auto-maps by speaker and uses `neutral_speaker_wav` from that file).
- If `--speaker_ref_jsonl` is not set, script will auto-try `<dataset_jsonl_dir>/test.jsonl`.
- For faster runs, you can pre-filter low-confidence GT keys during generation:
  `--filter_by_gt_confidence true --gt_embedding_meta_jsonl ... --gt_reject_predicted_emotions ... --gt_min_target_prob ...`
- `--unique_key_only true` (default) keeps only the first sample for each `key` to avoid duplicated candidates.
- `--fallback_ref_to_target_wav true` is kept only as last-resort fallback.
- You can also set a fixed reference prompt with `--default_ref_wav /abs/path/ref.wav`.

### 4) Score candidates

```bash
python examples/tts/dpo/score_dpo_candidates.py \
  --candidate_meta_jsonl dpo_work/candidates_run1/candidate_metadata.jsonl \
  --centroids_json dpo_work/emotion_centroids.json \
  --gt_embedding_meta_jsonl dpo_work/gt_embedding_meta.jsonl \
  --filter_by_gt_confidence true \
  --gt_reject_predicted_emotions unk,other \
  --gt_min_target_prob 0.05 \
  --output_jsonl dpo_work/scored_candidates.jsonl \
  --speaker_backend auto \
  --codec_decoder_path /path/to/CosyVoice-300M-SFT \
  --compute_wer true \
  --resume true
```

### 5) Build chosen/rejected pairs

```bash
python examples/tts/dpo/build_dpo_pairs.py \
  --scored_jsonl dpo_work/scored_candidates.jsonl \
  --output_jsonl dpo_work/dpo_pairs.jsonl \
  --summary_json dpo_work/dpo_pair_summary.json \
  --resume true
```

### 6) Compute/store pair quality metrics (WER + UTMOS), then optional filtering

First pass (no threshold, only compute and store):

```bash
python examples/tts/dpo/filter_dpo_pairs_by_quality.py \
  --pair_jsonl dpo_work/dpo_pairs.jsonl \
  --output_jsonl dpo_work/dpo_pairs_with_quality.jsonl \
  --summary_json dpo_work/dpo_pairs_quality_summary.json \
  --compute_wer true \
  --compute_utmos true \
  --whisper_model large-v3 \
  --resume true
```

Then, once thresholds are decided, filter directly:

```bash
python examples/tts/dpo/filter_dpo_pairs_by_quality.py \
  --pair_jsonl dpo_work/dpo_pairs.jsonl \
  --output_jsonl dpo_work/dpo_pairs_with_quality.jsonl \
  --filtered_output_jsonl dpo_work/dpo_pairs_filtered.jsonl \
  --summary_json dpo_work/dpo_pairs_quality_summary.json \
  --compute_wer true \
  --compute_utmos true \
  --max_chosen_wer 0.25 \
  --min_chosen_utmos 3.7 \
  --drop_if_metric_missing true \
  --resume true
```

Notes:

- `--output_jsonl` stores all rows plus:
  - `quality_metrics.chosen/rejected.{wer, asr_text, utmos, status, errors}`
  - `quality_filter.{keep, reasons, thresholds}`
- If `--filtered_output_jsonl` is set, only `keep=true` rows are exported there.
- Threshold args are optional. If not set, the script computes and stores metrics without dropping rows.

## Hard Negative Rule

`build_dpo_pairs.py` applies:

- For non-neutral targets, prefer rejected candidates with predicted emotion == `neutral`.
- If none, pick the lowest-score candidate.

Chosen candidate is always the highest-score candidate for that key.

## Intermediate Metadata

The pipeline keeps jsonl metadata at every stage, including:

- candidate path
- candidate token sequences (`generated_text_token_ids`, `generated_audio_token_ids`)
- total score and all score terms
- classifier probabilities
- emo2vec centroid similarity
- WER
- speaker similarity
- chosen/rejected pair mapping and score margin

## High-Confidence GT Filtering

To run high-confidence DPO only on GT samples that emotion2vec also recognizes:

- Exclude GT with `predicted_emotion in {unk, other}`.
- Exclude GT with very low `P(target_emotion | GT audio)` (for example `< 0.05`).

Recommended flags:

- Step 2 (`compute_emotion_centroids.py`):
  - `--reject_predicted_emotions unk,other`
  - `--min_target_prob 0.05`
- Step 3 (`generate_dpo_candidates.py`, optional but recommended for speed):
  - `--filter_by_gt_confidence true`
  - `--gt_embedding_meta_jsonl dpo_work/gt_embedding_meta.jsonl`
  - `--gt_reject_predicted_emotions unk,other`
  - `--gt_min_target_prob 0.05`
- Step 4 (`score_dpo_candidates.py`):
  - `--filter_by_gt_confidence true`
  - `--gt_embedding_meta_jsonl dpo_work/gt_embedding_meta.jsonl`
  - `--gt_reject_predicted_emotions unk,other`
  - `--gt_min_target_prob 0.05`

## Resume / Continue

All scripts support `--resume true` for same-folder continuation:

- Step 1 (`extract_emotion2vec_embeddings.py`):
  - Append mode; skip keys already written in output jsonl.
- Step 2 (`compute_emotion_centroids.py`):
  - If output json already exists, skip recomputation.
- Step 3 (`generate_dpo_candidates.py`):
  - Append mode; skip existing `(key, candidate_id)` in metadata.
  - If candidate wav exists but metadata missing, recover it without regeneration.
- Step 4 (`score_dpo_candidates.py`):
  - Append mode; skip existing `(key, candidate_id)` in scored output.
- Step 5 (`build_dpo_pairs.py`):
  - Append mode; skip keys already present in pair output.
- Step 6 (`filter_dpo_pairs_by_quality.py`):
  - Append mode; skip keys already present in `--output_jsonl`.
  - Supports "compute first, filter later" workflow using the same output file.

## Iterative DPO (Round-Based Short Training)

This repo now supports a first-version "multi-round short DPO" workflow with fixed reference per round:

- `round_0`: `policy_init = base_ckpt`, `reference = base_ckpt` (frozen)
- `round_k`: `policy_init = best_ckpt(round_{k-1})`, `reference = best_ckpt(round_{k-1})` (frozen)
- reference is updated only between rounds, never inside a round.

### A) Prepare round subsets (shuffle + chunk)

```bash
python examples/tts/dpo/prepare_iterative_dpo_subsets.py \
  --pair_jsonl dpo_work/dpo_pairs_wer_lt_0.1.jsonl \
  --output_dir dpo_work/iterative_rounds_v1 \
  --pairs_per_round 800 \
  --seed 42 \
  --shuffle true \
  --resume true
```

Output:

- subset files: `.../iterative_rounds_v1/subsets/round_000.jsonl`, `round_001.jsonl`, ...
- manifest: `.../iterative_rounds_v1/iterative_dpo_rounds_manifest.json`

### B) Train round subsets

The historical snapshot referenced a local round orchestrator and fine-tuning
wrapper that are not present in this cleaned tree. They are intentionally not
documented as runnable. Use the generated subset manifest with the standard
training entry point, and explicitly set the policy/reference checkpoint for
each round. A public orchestrator should be added only after it has its own
smoke test and contains no machine-specific paths.
