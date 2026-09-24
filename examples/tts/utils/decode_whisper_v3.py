import argparse
import os

import torch
import whisper
from tqdm import tqdm


def main(parent_dir, audio_subdir):
    gt_text_path = os.path.join(parent_dir, "gt_text")
    audio_dir = os.path.join(parent_dir, audio_subdir)
    output_dir = os.path.join(parent_dir, "pred_whisper_text")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = whisper.load_model("~/.cache/whisper/large-v3.pt", device=device)
    model = whisper.load_model("large-v3", device=device)
    
    with open(gt_text_path, 'r') as file, open(output_dir, 'w') as f:
        for line in tqdm(file):
            id = line.split('\t')[0]
            audio_filename = id + '.wav'
            audio_filepath = os.path.join(audio_dir, audio_filename)
            try:
                result = model.transcribe(audio_filepath, language='en')
                transcription = result['text'].strip()
                f.write(f"{id}\t{transcription}\n")
            except Exception as e:
                print(f"Error processing {audio_filepath}: {e}")
                f.write(f"{id}\t\n")

    print("Transcription completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio files.')
    parser.add_argument('--parent_dir', type=str, required=True, help='Path to the parent directory.')
    parser.add_argument('--audio_subdir', type=str, default='pred_audio/default_tone', help='Subdirectory for audio files relative to the parent directory.')


    args = parser.parse_args()
    main(args.parent_dir, args.audio_subdir)
