import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import TransformerCausalDecoder
from scripts.audit_zero_shot import get_disjoint_audit_loader

def run_latent_extraction():
    print("=====================================================")
    print("  PHASE 8: THE LATENT SPACE PROJECTION (PCA)")
    print("=====================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[X] Environment: {device}")
    
    # 1. Boot up the Intelligence Sub-routines
    loader = get_disjoint_audit_loader(
        features_dir="/Volumes/Second-Brain-1/AI/Synth/dataset/features",
        audio_dir="/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features",
        num_maps=20 # Scale down to 20 maps so we easily grab 10k points fast
    )
    
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    model.load_state_dict(torch.load("models/checkpoints/transformer_pilot_ep5.pt", weights_only=True))
    model.to(device)
    model.eval()
    
    latent_vectors = []
    kinetic_magnitudes = []
    
    MAX_POINTS = 10000
    points_gathered = 0
    
    print("[X] Ripping Latent State from Active Tensors...")
    context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
    
    with torch.no_grad():
        for batch_idx, (audio, targets, diff, lengths) in enumerate(loader):
            if points_gathered >= MAX_POINTS:
                break
                
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            with context:
                preds = model(audio, targets, diff)
                
            # Extract Pure Intermediary Array
            latent = preds["latent_state"] # (B, T, 256)
            
            # Identify Genuine Geometry (Ignore Silence padded margins)
            target_pres = targets[..., 0:2]
            target_vel = targets[..., 6:8]
            
            B, T, _ = target_pres.shape
            mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
            active_target_binary = (target_pres > 0.5) & mask.unsqueeze(-1)
            
            for b in range(B):
                for t in range(T):
                    # For every valid note (e.g., Right Hand [0] or Left Hand [1])
                    for h in range(2):
                        if active_target_binary[b, t, h] and points_gathered < MAX_POINTS:
                            v_raw = latent[b, t, :].cpu().float().numpy()
                            # Physical Kinetic Velocity scale
                            vel = target_vel[b, t, h].cpu().float().item() 
                            
                            latent_vectors.append(v_raw)
                            kinetic_magnitudes.append(abs(vel))
                            points_gathered += 1

    
    print(f"[X] Extraction complete. Feeding {points_gathered} parameters into PCA...")
    X = np.array(latent_vectors)
    C = np.array(kinetic_magnitudes)
    
    # 2. PCA Projection (256D -> 3D)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    variance = sum(pca.explained_variance_ratio_)
    print(f"[X] PCA Engine Completed | Top 3 Vectors encompass {variance * 100:.2f}% of Matrix variance.")
    
    # 3. Artifact Render Synthesis
    print("[X] Exporting 3D Density Matrix into Artifacts...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    sc = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
        c=C, cmap='magma', s=3, alpha=0.6, linewidths=0
    )
    
    cbar = plt.colorbar(sc, shrink=0.5, aspect=10)
    cbar.set_label('Kinetic Velocity Magnitude ($\Delta$)')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title("128-D Latent Space Projection (PCA) - $\gamma=2.0$ Precision Convergence", color='white')
    ax.set_xlabel('Principal Axis 1')
    ax.set_ylabel('Principal Axis 2')
    ax.set_zlabel('Principal Axis 3')
    
    # Grid cleanup
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='#30363d', linestyle='-', linewidth=0.5)

    artifact_dir = Path("/Volumes/Second-Brain-1/AI/Synth/evaluation/phase10")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "latent_final_pca.png"
    
    plt.savefig(out_path, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"[X] Render Successfully Bound to: {out_path}")
    print("=====================================================")

if __name__ == "__main__":
    run_latent_extraction()
