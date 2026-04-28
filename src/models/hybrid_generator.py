import torch
import torch.nn as nn
from typing import Optional

from src.models.baseline import BaselineBeatmapModel
from src.models.transformer import TransformerCausalDecoder

class HybridCascadeModel(nn.Module):
    """
    Phase 11: Orchestrates the Convolutional LSTM scaffold to guide 
    the active Residual Transformer Refiner.
    """
    def __init__(self, baseline_ckpt: Optional[str] = None, transformer_ckpt: Optional[str] = None):
        super().__init__()
        
        self.baseline = BaselineBeatmapModel(audio_features=80)
        
        self.transformer = TransformerCausalDecoder(
            d_model=256, 
            num_layers=4, 
            d_audio=128, 
            d_target=8
        )
        
    def load_states(self, baseline_ckpt: str, transformer_ckpt: Optional[str] = None, device: torch.device = None):
        if device is None:
            device = torch.device("cpu")
            
        print(f"[Hybrid] Loading Baseline LSTM Scaffold from {baseline_ckpt}")
        base_state = torch.load(baseline_ckpt, map_location=device, weights_only=True)
        if "model_state_dict" in base_state:
            self.baseline.load_state_dict(base_state["model_state_dict"])
        else:
            self.baseline.load_state_dict(base_state)
            
        # Hard Freeze Baseline
        self.baseline.eval()
        for param in self.baseline.parameters():
            param.requires_grad = False
            
        if transformer_ckpt:
            print(f"[Hybrid] Loading Transformer Memory from {transformer_ckpt}")
            trans_state = torch.load(transformer_ckpt, map_location=device, weights_only=True)
            self.transformer.load_state_dict(trans_state, strict=False)
            
    def forward(self, audio_features_128d: torch.Tensor, target_features: torch.Tensor, difficulty_idx: torch.Tensor):
        # 1. We must slice the 128D audio context array back into its raw 80D Mel for the LSTM scaffold.
        # The first 80 channels are the exact mel bins.
        audio_80d = audio_features_128d[..., :80]
        
        # 2. Extract Coarse scaffold output (No-Grad locked context execution)
        with torch.no_grad():
            self.baseline.eval()
            base_out = self.baseline(audio_80d)
            coarse_memory = base_out["lstm_out"] # (B, T, 256)
            
        # 3. Direct Refinement Pass
        refined_preds = self.transformer(
            audio_features_128d, 
            target_features, 
            difficulty_idx, 
            coarse_memory=coarse_memory
        )
        
        return refined_preds
