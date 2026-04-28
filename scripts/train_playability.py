import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import time
from model import FeasibilityScorer

class MapDataset(Dataset):
    def __init__(self, samples, max_seq=512):
        self.samples = samples
        self.max_seq = max_seq
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = np.array(s["seq"])
        
        # Pad or truncate sequence
        if len(seq) > self.max_seq:
            seq = seq[:self.max_seq]
        else:
            if len(seq) == 0:
                seq = np.zeros((self.max_seq, 7))
            else:
                pad = np.zeros((self.max_seq - len(seq), seq.shape[1]))
                seq = np.vstack([seq, pad])
            
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(s["global"], dtype=torch.float32),
            torch.tensor([s["label"]], dtype=torch.float32)
        )

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")
    
    with open("/Users/marcus/.gemini/antigravity/scratch/dataset_v0.pkl", 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples.")
    
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    
    train_ds = MapDataset(train_data)
    val_ds = MapDataset(val_data)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True) # Increased batch size
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = FeasibilityScorer().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_auc = 0
    num_epochs = 15 # More epochs
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start = time.time()
        
        for seq, glob, label in train_loader:
            seq, glob, label = seq.to(device), glob.to(device), label.to(device)
            
            optimizer.zero_grad()
            out = model(seq, glob)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Eval
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for seq, glob, label in val_loader:
                seq, glob, label = seq.to(device), glob.to(device), label.to(device)
                out = model(seq, glob)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(out.cpu().numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | AUC: {auc:.4f} | Time: {time.time()-start:.2f}s")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "/Users/marcus/.gemini/antigravity/scratch/scorer_v0.pt")
            
    # Final metrics
    print("\nTraining Complete. Best AUC:", best_auc)
    # Load best
    model.load_state_dict(torch.load("/Users/marcus/.gemini/antigravity/scratch/scorer_v0.pt"))
    model.eval()
    all_labels = []
    all_preds_prob = []
    with torch.no_grad():
        for seq, glob, label in val_loader:
            seq, glob, label = seq.to(device), glob.to(device), label.to(device)
            out = model(seq, glob)
            all_labels.extend(label.cpu().numpy())
            all_preds_prob.extend(out.cpu().numpy())
            
    preds_binary = (np.array(all_preds_prob) > 0.5).astype(int)
    print(classification_report(all_labels, preds_binary, digits=4))

if __name__ == "__main__":
    train()


if __name__ == "__main__":
    train()
