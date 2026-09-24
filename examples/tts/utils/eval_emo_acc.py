import torch
import argparse
import os
import torch.nn.functional as F
import numpy as np
import json
from funasr import AutoModel



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--gt', type=str, default="../test.jsonl")
    parser.add_argument('--pred', type=str)
    parser.add_argument('--audio_subdir', type=str, default='pred_audio/default_tone', help='Subdirectory for audio files relative to the parent directory.')
    
    args = parser.parse_args()

    pred_dir = os.path.join(args.pred, args.audio_subdir)
    output_path = os.path.join(args.pred, "emo1.log")
    model = AutoModel(model="iic/emotion2vec_plus_large")
    simis = []

    correct_predictions = 0
    total_predictions = 0

    # Define all labels and the selected subset for processing
    all_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unk']
    selected_labels = ['angry', 'happy', 'neutral', 'other', 'sad']
    selected_indices = [all_labels.index(label) for label in selected_labels] #[0, 3, 4, 5, 6]
    # selected_labels = ['angry', 'happy', 'neutral', 'sad']
    # selected_indices = [all_labels.index(label) for label in selected_labels] #[0, 3, 4, 6]

    # Initialize dictionaries to track recall information
    recall_stats = {label: {'correct': 0, 'total': 0} for label in selected_labels}

    with torch.no_grad():
        with open(args.gt, "r") as rf, open(output_path, "w") as f:
            for line in rf:
                data = json.loads(line.strip())
                id =data["key"]
                gt_path=data["target_wav"]
                pred_path=pred_dir+'/'+id+'.wav'
                tgt_emo=data["emotion"]
                if tgt_emo not in selected_labels:
                    tgt_emo = "other"
                    # continue

                if not os.path.exists(pred_path):
                    print(pred_path)
                    continue

                try:
                    pred_result = model.generate(pred_path, granularity="utterance", extract_embedding=True)
                    pred_emb = pred_result[0]["feats"]

                    # Filter scores and labels for selected labels only
                    pred_scores = pred_result[0]['scores']
                    pred_scores_filtered = [pred_scores[i] for i in selected_indices]
                    pred_emo = selected_labels[pred_scores_filtered.index(max(pred_scores_filtered))]
                except Exception as e:
                    print(f"Error processing {pred_path}: {e}")
                    continue

                try:
                    tgt_result = model.generate(gt_path, granularity="utterance", extract_embedding=True)
                    tgt_emb = tgt_result[0]["feats"]
                except Exception as e:
                    print(f"Error processing {gt_path}: {e}")
                    continue

                # Update total and correct predictions
                total_predictions += 1
                if pred_emo == tgt_emo:
                    correct_predictions += 1
                    recall_stats[tgt_emo]['correct'] += 1
                recall_stats[tgt_emo]['total'] += 1

                simi = float(F.cosine_similarity(torch.FloatTensor([pred_emb]), torch.FloatTensor([tgt_emb])).item())
                simis.append(simi)
                
                print("%s %s %f"%(pred_path, gt_path, simi), file=f)
            print("------------------------------------------", file=f)
            print("len:", len(simis),file=f)
            print("emo2vec large:", np.mean(simis), file=f)

            overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print("------------------------------------------", file=f)
            print(f"Total predictions: {total_predictions}", file=f)
            print(f"Correct predictions: {correct_predictions}", file=f)
            print(f"Overall Accuracy: {overall_accuracy:.3f}", file=f)

            # Calculate recall for each emotion
            recalls = []
            for label in selected_labels:
                recall = (recall_stats[label]['correct'] / recall_stats[label]['total']) if recall_stats[label]['total'] > 0 else 0
                recalls.append(recall)
                print(f"Recall for {label}: {recall:.3f}", file=f)

            # Calculate and print the average recall
            average_recall = np.mean(recalls)
            print(f"Average Recall: {average_recall:.3f}", file=f)
