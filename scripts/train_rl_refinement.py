import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import time
import math
from pathlib import Path
from tqdm import tqdm
import random

# Absolute paths
BASE_PATH = Path("/Volumes/Second-Brain-1/AI/Synth")
sys.path.insert(0, str(BASE_PATH))

from src.models.transformer import TransformerCausalDecoder
from scripts.playability_model import FeasibilityScorer
from scripts.feasibility_checker import FeasibilityChecker, DIFFICULTY_PARAMS # For cross-checking

# --- Hyperparameters ---
LEARNING_RATE = 2e-4
BETA_START = 0.5
BETA_END = 0.1
EPOCHS = 30
BATCH_SIZE = 1
MAX_SEQ_LEN = 2000
NMS_WINDOW_MS = 100.0

# Target NPS for density bonus (from corpus averages)
DENSITY_TARGETS = {
    0: 2.1,   # Easy
    1: 4.5,   # Normal
    2: 7.5,   # Hard
    3: 11.0,  # Expert
    4: 15.0   # Master
}

# --- Audio Features (Simplified for training) ---
def compute_augmented_features(mel: torch.Tensor) -> torch.Tensor:
    T, num_freq = mel.shape[1], mel.shape[2]
    mel_sq = mel[0]
    shifted_mel = torch.cat([torch.zeros(1, num_freq, device=mel.device), mel_sq[:-1, :]], dim=0)
    delta = mel_sq - shifted_mel
    delta_cpu = delta.cpu().unsqueeze(0)
    delta_44 = F.adaptive_avg_pool1d(delta_cpu, 44).squeeze(0).to(mel.device)
    flux = F.relu(delta).sum(dim=1, keepdim=True) / 10.0
    f_idx = torch.arange(num_freq, dtype=torch.float32, device=mel.device) / (num_freq - 1)
    mel_sum = mel_sq.sum(dim=1, keepdim=True) + 1e-8
    centroid = (mel_sq * f_idx.unsqueeze(0)).sum(dim=1, keepdim=True) / mel_sum
    f_diff_sq = (f_idx.unsqueeze(0) - centroid)**2
    bandwidth = torch.sqrt((mel_sq * f_diff_sq).sum(dim=1, keepdim=True) / mel_sum)
    rms = torch.sqrt((mel_sq**2).mean(dim=1, keepdim=True))
    augmented = torch.cat([mel_sq, delta_44, flux, centroid, bandwidth, rms], dim=1)
    return augmented.unsqueeze(0)

# --- Reward Engine ---
class RLRewardEngine:
    def __init__(self, scorer, device):
        self.scorer = scorer
        self.device = device
        self.rules = FeasibilityChecker(DIFFICULTY_PARAMS["Easy"])

    def temporal_nms_lite(self, notes, window_ms=100.0):
        if not notes: return []
        sorted_notes = sorted(notes, key=lambda x: x["prob"], reverse=True)
        keep = []
        while sorted_notes:
            best = sorted_notes.pop(0)
            keep.append(best)
            sorted_notes = [n for n in sorted_notes if not (n["type"] == best["type"] and abs(n["time"] - best["time"]) < window_ms)]
        return sorted(keep, key=lambda x: x["time"])

    def compute_features(self, notes, diff_idx, duration_s):
        """Standardize note list into Scorer feature vector."""
        if not notes:
            return None, 0.0
            
        hands = {0: [], 1: []}
        for n in notes: hands[n["type"]].append(n)
        
        feat_list = []
        for h_type, h_notes in hands.items():
            for i in range(len(h_notes)):
                n = h_notes[i]
                dt, dist, vel, ang = 0, 0, 0, 0
                if i > 0:
                    p = h_notes[i-1]
                    dt = (n["time"] - p["time"]) / 1000.0
                    dist = math.sqrt((n["x"]/4.0 - p["x"]/4.0)**2 + (n["y"]/3.0 - p["y"]/3.0)**2)
                    if dt > 0: vel = dist / dt
                feat_list.append([h_type, n["x"]/4.0, n["y"]/3.0, min(dt, 5.0), dist, min(vel, 20.0), 0.0])
        
        feat_list.sort(key=lambda x: x[3]) # Sort by relative dt? No, actually extractor sorted by absolute time.
        # Fix: need absolute time for features.
        
        # NPS Calculation
        nps = len(notes) / duration_s if duration_s > 0 else 0
        
        # Balance
        l_pct = sum(1 for n in notes if n["x"] < 0) / len(notes)
        r_pct = 1.0 - l_pct
        imb = abs(l_pct - r_pct)
        
        # Build tensor
        # Scorer seq size: (1, 512, 7)
        seq_tensor = torch.zeros(1, 512, 7).to(self.device)
        # Sort notes by absolute time for meaningful dt/vel
        notes_sorted = sorted(notes, key=lambda x: x["time"])
        
        # Track previous per hand for dt/dist
        prev_hand = {0: None, 1: None}
        
        for i, n in enumerate(notes_sorted[:512]):
            h = n["type"]
            dt, dist, vel = 0, 0, 0
            
            if prev_hand[h] is not None:
                p = prev_hand[h]
                dt = (n["time"] - p["time"]) / 1000.0
                dist = math.sqrt((n["x"]/4.0 - p["x"]/4.0)**2 + (n["y"]/3.0 - p["y"]/3.0)**2)
                if dt > 1e-4:
                    vel = dist / dt
            
            seq_tensor[0, i, 0] = h
            seq_tensor[0, i, 1] = n["x"] / 4.0
            seq_tensor[0, i, 2] = n["y"] / 3.0
            seq_tensor[0, i, 3] = min(dt, 5.0)
            seq_tensor[0, i, 4] = dist
            seq_tensor[0, i, 5] = min(vel, 20.0)
            seq_tensor[0, i, 6] = 0.0 # Angle placeholder
            
            prev_hand[h] = n
            
        map_feats = torch.tensor([[diff_idx/4.0, len(notes)/2000.0, duration_s/300.0, nps/20.0, l_pct, r_pct, imb, 1.0/5.0]], dtype=torch.float32).to(self.device)
        
        return seq_tensor, map_feats, nps

    def get_reward(self, probs, pos_pred, diff_idx, duration_s, ref_logits, current_epoch, is_sampled=True):
        T_len = probs.shape[1]
        
        # 1. Extract notes
        notes = []
        for t in range(T_len):
            for h in range(2):
                p = probs[0, t, h].item()
                if is_sampled:
                    if random.random() < p:
                        notes.append({
                            "time": t*10.0, 
                            "type": h, 
                            "prob": p, 
                            "x": pos_pred[0, t, h*2].detach().item()*4.0, 
                            "y": pos_pred[0, t, h*2+1].detach().item()*3.0
                        })
                else:
                    if p > 0.5:
                        notes.append({
                            "time": t*10.0, 
                            "type": h, 
                            "prob": p, 
                            "x": pos_pred[0, t, h*2].detach().item()*4.0, 
                            "y": pos_pred[0, t, h*2+1].detach().item()*3.0
                        })
        
        # 2. NMS
        notes = self.temporal_nms_lite(notes)
        
        if not notes: 
            print("DEBUG: No notes generated.")
            return 0.0
        
        print(f"DEBUG: {len(notes)} notes generated.")
        
        # 3. Score
        seq_t, map_t, nps = self.compute_features(notes, diff_idx, duration_s)
        with torch.no_grad():
            playability_score = self.scorer(seq_t, map_t).item()
            
        # 4. Density Bonus (Exponential for long-tail gradient to reach sparse targets)
        target_nps = DENSITY_TARGETS[diff_idx]
        density_bonus = math.exp(-abs(nps - target_nps) / (target_nps + 1e-6))
        
        # 5. Hard Floor Guardrail (Total zero below 1.0 NPS)
        if nps < 1.0:
            density_bonus = 0.0
            
        # 6. Temporal Uniformity (Prevent clumping)
        mid_time = duration_s * 500.0 # ms
        early_notes = sum(1 for n in notes if n["time"] < mid_time)
        late_notes = len(notes) - early_notes
        ratio = (early_notes + 1.0) / (late_notes + 1.0)
        uniformity_bonus = math.exp(-abs(1.0 - ratio) / 2.0)
            
        # 7. Stratified Alignment (Fixed-K Oracle)
        # Force the mask to a fixed count based on the difficulty target.
        # This makes alignment_score = Precision, rewarding sparsity and beat-lock.
        with torch.no_grad():
            ref_probs = torch.sigmoid(ref_logits).max(dim=2)[0] # Max salience across hands (B, T)
            seg_len = 800
            T_total = ref_probs.shape[1]
            num_segs = max(1, T_total // seg_len)
            
            # Use global target count to lock K
            target_count = int(DENSITY_TARGETS[diff_idx] * duration_s)
            
            # Use strict ceil to ensure every segment gets at least one anchor
            k_per_seg = max(1, target_count // num_segs)
            
            # Squeeze batch dimension for mask calculation (B=1)
            rp = ref_probs.squeeze() 
            T = rp.shape[0]
            mask = torch.zeros_like(rp)
            
            for i in range(num_segs):
                s, e = i*seg_len, min((i+1)*seg_len, T)
                if e - s < 2: continue
                seg = rp[s:e]
                k = min(k_per_seg, e - s)
                _, topk = torch.topk(seg, k, dim=0)
                mask[s:e][topk] = 1.0
            
            # Precision Alignment: notes on mask / total notes
            mask_2d = mask.unsqueeze(1).expand(-1, 2)
            alignment_score = (probs[0] * mask_2d).sum() / (probs.sum() + 1e-6)
        
        # 8. Exploration Bias (User-advised triage)
        biased_score = playability_score + 0.1
        
        # 9. Multiplier: 10x Alignment pull for Easy mode search (Decays over epochs)
        # We use a curriculum: 10.0 (Search) -> 1.0 (Integration)
        # We'll pass the current epoch to get_reward for this.
        align_mult = 1.0
        if diff_idx == 0:
            # Decay from 20.0 to 1.0 over epochs 2-5
            if current_epoch < 2: align_mult = 20.0
            elif current_epoch < 3: align_mult = 10.0
            elif current_epoch < 4: align_mult = 5.0
            else: align_mult = 1.0
            
        print(f"DEBUG: Score={playability_score:.6f}, Bonus={density_bonus:.4f}, NPS={nps:.2f}, Align={alignment_score:.4f}")
        total_reward = biased_score * density_bonus * uniformity_bonus * (1.0 + align_mult * alignment_score)
        
        return {
            "reward": total_reward,
            "mask": mask_2d,
            "nps": nps
        }

# --- Difficulty-Specific Hyperparams ---
BETA_MAP = {
    0: 0.3, # Easy: More freedom to change
    1: 0.4,
    2: 0.5,
    3: 0.6,
    4: 0.7  # Master: More anchor to original dense style
}

BIAS_MAP = {
    0: 5.0, # Easy bias: Extreme hammer to force density landing
    1: 2.0, # Normal bias
    2: 1.0, 
    3: 0.0,
    4: 0.0
}

# --- Training Loop ---
def refine():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Models
    gen = TransformerCausalDecoder(d_model=256).to(device)
    gen_ckpt = BASE_PATH / "models/checkpoints/transformer_phase12b_rl_multi_ep1.pt"
    gen.load_state_dict(torch.load(gen_ckpt, map_location=device))
    
    ref = TransformerCausalDecoder(d_model=256).to(device)
    ref.load_state_dict(torch.load(gen_ckpt, map_location=device))
    ref.eval()
    
    scorer = FeasibilityScorer().to(device)
    scorer.load_state_dict(torch.load(BASE_PATH / "models/checkpoints/scorer_v0.pt", map_location=device))
    scorer.eval()
    
    reward_engine = RLRewardEngine(scorer, device)
    optimizer = optim.Adam(gen.parameters(), lr=LEARNING_RATE)
    
    # Tracks
    track_dir = BASE_PATH / "dataset/audio_features"
    tracks = sorted([str(p) for p in track_dir.glob("*.npz")])[:500]
    
    print(f"[+] Starting Multi-Difficulty Contrastive RL Refinement")
    
    for epoch in range(20): # More epochs since they are shorter
        random.shuffle(tracks)
        for i, audio_path in enumerate(tracks[:50]):
            # Load audio (80-dim mel)
            data = np.load(audio_path)
            audio = torch.from_numpy(data["audio_mel"]).float().unsqueeze(0).to(device)
            # Pad 80 -> 128 for model compatibility (D_audio=128)
            # Use small noise instead of zeros to prevent NaNs in attention and ensure device pinning
            pad = torch.randn((1, audio.shape[1], 48), device=device, dtype=audio.dtype) * 0.01
            audio = torch.cat([audio, pad], dim=-1)
            
            # Crop to 20s Hurricane Window (Accelerate RL descent)
            max_f = 2000
            if audio.shape[1] > max_f:
                audio = audio[:, :max_f, :]
            
            T_len = audio.shape[1]
            dur_s = T_len * 0.01 # 10ms hop
            
            # --- Multi-Difficulty Sampling ---
            diff_idx = random.choice([0, 1, 2, 3, 4])
            # Non-uniform sampling to protect high-density poles (User-suggested 40/40 mix)
            if random.random() < 0.4:
                diff_idx = 4 # Force Master to prevent shared-layer collapse
            elif random.random() < 0.2:
                diff_idx = random.choice([0, 1]) # Focus on sparse search
            
            diff_t = torch.tensor([diff_idx]).to(device)
            targets = torch.zeros((1, T_len, 8)).to(device)
            
            # Forward
            out = gen(audio, targets, diff_t)
            with torch.no_grad():
                ref_out = ref(audio, targets, diff_t)
                ref_probs = torch.sigmoid(ref_out["presence_logits"])
            
            logits = out["presence_logits"]
            probs = torch.sigmoid(logits)
            
            # --- SUPERVISED PHASES (TURBO BYPASS) ---
            nps_est = probs.sum().item() / (dur_s + 1e-6)
            
            if nps_est > DENSITY_TARGETS[diff_idx] * 1.2:
                # Skip Scorer and RL for high-density maps to reach sparse valley 100x faster
                with torch.no_grad():
                    # Quick Oracle Mask generation (No Scorer)
                    ref_p = torch.sigmoid(ref_out["presence_logits"][:, :T_len, :]).max(dim=2)[0].squeeze()
                    mask = torch.zeros_like(ref_p)
                    sj_len = 800
                    target_count = int(DENSITY_TARGETS[diff_idx] * dur_s)
                    num_sj = max(1, T_len // sj_len)
                    k_per_sj = max(1, target_count // num_sj)
                    for sj in range(num_sj):
                        s, e = sj*sj_len, min((sj+1)*sj_len, T_len)
                        if e - s < 2: continue
                        _, topk = torch.topk(ref_p[s:e], min(k_per_sj, e-s), dim=0)
                        mask[s:e][topk] = 1.0
                    mask_2d = mask.unsqueeze(1).expand(-1, 2).unsqueeze(0)
                
                clamped_logits = torch.clamp(logits, max=5.0)
                loss = 20.0 * F.binary_cross_entropy_with_logits(clamped_logits, mask_2d, reduction='mean')
                print(f"DEBUG [TURBO]: NPS={nps_est:.2f}, Target={DENSITY_TARGETS[diff_idx]}")
                
            else:
                # RL MODE: Density is in-range, fine-tune for playability/rhythm
                res_s = reward_engine.get_reward(probs, out["position_pred"], diff_idx, dur_s, ref_out["presence_logits"][:, :T_len, :], epoch, is_sampled=True)
                res_g = reward_engine.get_reward(probs, out["position_pred"], diff_idx, dur_s, ref_out["presence_logits"][:, :T_len, :], epoch, is_sampled=False)
                
                R_sample = res_s["reward"]
                R_greedy = res_g["reward"]
                mask_2d = res_s["mask"]
                nps = res_s["nps"]
                
                # Apply Gated Exploration Bias locally
                bias_val = BIAS_MAP.get(diff_idx, 0.1)
                R_sample += bias_val
                
                advantage = R_sample - R_greedy
                sample_mask = (torch.rand_like(probs) < probs).float()
                pg_loss = - (torch.log(probs + 1e-8) * sample_mask + torch.log(1 - probs + 1e-8) * (1 - sample_mask)) * advantage
                pg_loss = pg_loss.mean()
                
                # Difficulty-specific Gated KL
                beta_val = BETA_MAP.get(diff_idx, 0.05)
                # Gated KL: Disable for search modes (Easy/Normal) to prevent dense bias
                if diff_idx < 2:
                    kl_loss = torch.tensor(0.0).to(device)
                else:
                    kl_loss = F.binary_cross_entropy_with_logits(logits, ref_probs[:, :T_len, :], reduction='mean')
                
                loss = pg_loss + beta_val * kl_loss
                
                # Final rhythmic refinement
                sup_weight = 2.0 * min(1.0, (nps / DENSITY_TARGETS[diff_idx] - 0.8))
                clamped_logits = torch.clamp(logits, max=5.0)
                sup_loss = F.binary_cross_entropy_with_logits(clamped_logits, mask_2d.unsqueeze(0), reduction='mean')
                loss = loss + sup_weight * sup_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Ep {epoch} | D{diff_idx} | Adv: {advantage:.4f} | R_samp: {R_sample:.4f}")

        # Periodic check with Rules
        # Save Every Epoch for fast auditing
        if (epoch + 1) % 1 == 0:
            torch.save(gen.state_dict(), BASE_PATH / f"models/checkpoints/transformer_phase12b_rl_multi_ep{epoch+1}.pt")
            print(f"\n[Checkpoint] Epoch {epoch+1} Saved.")
    
    torch.save(gen.state_dict(), BASE_PATH / "models/checkpoints/transformer_phase12b_rl_multi_final.pt")

if __name__ == "__main__":
    refine()
