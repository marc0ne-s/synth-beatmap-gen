import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from model import FeasibilityScorer
import os

class MapDataset(Dataset):
    def __init__(self, samples, max_seq=512):
        self.samples = samples
        self.max_seq = max_seq
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = np.array(s["seq"])
        if len(seq) > self.max_seq: seq = seq[:self.max_seq]
        else:
            if len(seq) == 0: seq = np.zeros((self.max_seq, 7))
            else:
                pad = np.zeros((self.max_seq - len(seq), seq.shape[1]))
                seq = np.vstack([seq, pad])
        return (torch.tensor(seq, dtype=torch.float32), 
                torch.tensor(s["global"], dtype=torch.float32), 
                torch.tensor([s["label"]], dtype=torch.float32),
                idx)

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    with open("/Users/marcus/.gemini/antigravity/scratch/dataset_v0.pkl", 'rb') as f:
        data = pickle.load(f)
    
    ds = MapDataset(data)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    
    model = FeasibilityScorer().to(device)
    model.load_state_dict(torch.load("/Users/marcus/.gemini/antigravity/scratch/scorer_v0.pt"))
    model.eval()
    
    all_scores = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for seq, glob, label, idx in loader:
            seq, glob, label = seq.to(device), glob.to(device), label.to(device)
            out = model(seq, glob)
            all_scores.extend(out.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())
            
    df = pd.DataFrame({
        "uuid": [s["uuid"] for s in data],
        "difficulty": [s["difficulty"] for s in data],
        "score": all_scores,
        "label": all_labels,
        "imbalance": [s["global"][6] for s in data], # imbalance score was 7th feature in global
        "nps": [s["global"][3] * 20.0 for s in data] # denormalize nps
    })
    
    # 1. Overall Metrics
    print("\n=== Overall Metrics ===")
    auc = roc_auc_score(df["label"], df["score"])
    print(f"Total AUROC: {auc:.4f}")
    
    # 2. Per-Difficulty Audit
    print("\n=== Per-Difficulty Audit ===")
    diffs = ["Easy", "Normal", "Hard", "Expert", "Master"]
    for d in diffs:
        sub = df[df["difficulty"] == d]
        if sub.empty: continue
        d_auc = roc_auc_score(sub["label"], sub["score"])
        avg_score = sub["score"].mean()
        pass_rate = sub["label"].mean()
        print(f"{d:<10} | AUROC: {d_auc:.4f} | Pass Rate: {pass_rate:.1%} | Avg Score: {avg_score:.3f}")
        
    # 3. Easy Difficulty Focus: Left-Hand Bias
    print("\n=== Easy Difficulty: Hand Bias Analysis ===")
    easy = df[df["difficulty"] == "Easy"]
    # Check correlation between imbalance and score
    corr = easy["imbalance"].corr(easy["score"])
    print(f"Correlation between Imbalance and Playability Score: {corr:.4f}")
    
    # Bucket by imbalance
    easy["imb_bucket"] = pd.cut(easy["imbalance"], bins=[0, 0.1, 0.2, 0.35, 0.5, 1.0])
    imb_stats = easy.groupby("imb_bucket")["score"].mean()
    print("\nAvg Score by Imbalance Bucket (Easy):")
    print(imb_stats)
    
    # 4. Disagreements (Rules vs Model)
    # False Positives: Rules say Fail (0), Model says Playable (Score > 0.8)
    # False Negatives: Rules say Pass (1), Model says Unplayable (Score < 0.2)
    
    fps = df[(df["label"] == 0) & (df["score"] > 0.8)].sort_values("score", ascending=False)
    fns = df[(df["label"] == 1) & (df["score"] < 0.2)].sort_values("score", ascending=True)
    
    print(f"\nDisagreements: Found {len(fps)} False Positives and {len(fns)} False Negatives.")
    
    if not fps.empty:
        print("\nTop False Positives (Rules: Fail, Model: High Score):")
        print(fps[["uuid", "difficulty", "score", "imbalance"]].head(10))
        
    if not fns.empty:
        print("\nTop False Negatives (Rules: Pass, Model: Low Score):")
        print(fns[["uuid", "difficulty", "score", "imbalance"]].head(10))
        
    # Save results for report
    df.to_csv("/Users/marcus/.gemini/antigravity/scratch/evaluation_results.csv", index=False)
    
if __name__ == "__main__":
    evaluate()
