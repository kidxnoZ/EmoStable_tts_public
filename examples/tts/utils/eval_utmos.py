import argparse
import json
import os
from pathlib import Path

import librosa
import torch
from tqdm import tqdm




def main():
    parser = argparse.ArgumentParser(description="UTMOS Evaluation")
    parser.add_argument("--audio_dir", type=str, required=True, help="Audio file path.")
    parser.add_argument("--ext", type=str, default="wav", help="Audio extension.")
    parser.add_argument(
        "--min_samples",
        type=int,
        default=640,
        help="Skip audio shorter than this many samples before predictor call.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"


    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True,
        skip_validation=True,
    )
    predictor = predictor.to(device)
    predictor.eval()

    audio_paths = sorted(Path(args.audio_dir).rglob(f"*.{args.ext}"))
    utmos_score = 0.0
    scored_count = 0
    skipped_count = 0

    utmos_result_path = Path(args.audio_dir) / "_utmos_results.jsonl"
    with open(utmos_result_path, "w", encoding="utf-8") as f:
        for audio_path in tqdm(audio_paths, desc="Processing"):
            line = {"wav": str(audio_path.stem), "path": str(audio_path)}
            try:
                wav, sr = librosa.load(audio_path, sr=None, mono=True)
            except Exception as e:
                skipped_count += 1
                line["status"] = "skipped"
                line["reason"] = "load_failed"
                line["error"] = str(e).splitlines()[0]
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                continue

            num_samples = int(wav.shape[0])
            line["sr"] = int(sr)
            line["num_samples"] = num_samples
            if num_samples < args.min_samples:
                skipped_count += 1
                line["status"] = "skipped"
                line["reason"] = "too_short"
                line["error"] = f"num_samples={num_samples} < min_samples={args.min_samples}"
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                continue

            try:
                wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
                with torch.no_grad():
                    score = predictor(wav_tensor, sr)
                score_item = float(score.item())
            except Exception as e:
                skipped_count += 1
                line["status"] = "skipped"
                line["reason"] = "predict_failed"
                line["error"] = str(e).splitlines()[0]
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                continue

            line["status"] = "ok"
            line["utmos"] = score_item
            utmos_score += score_item
            scored_count += 1
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        avg_score = utmos_score / scored_count if scored_count > 0 else 0.0
        summary = {
            "type": "summary",
            "total": len(audio_paths),
            "scored": scored_count,
            "skipped": skipped_count,
            "avg_utmos": round(avg_score, 6),
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"UTMOS: {avg_score:.4f}")
    print(f"Total: {len(audio_paths)}, Scored: {scored_count}, Skipped: {skipped_count}")
    print(f"UTMOS results saved to {utmos_result_path}")


if __name__ == "__main__":
    main()
