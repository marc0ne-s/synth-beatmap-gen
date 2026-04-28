"""
Transformer Architecture Scaffold.
Implements the Hybrid Projection (Audio + Note embeddings) combined with 
Causal Scaled-Dot-Product Cross-Attention natively optimized for MPS.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        # Using exact precision for safe initialization
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is expected to have shape (Batch, SeqLen, Dims)"""
        return x + self.pe[:, :x.size(1), :]

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to have shape (Batch, Channels, SeqLen)
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x

class NativeCausalBlock(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, kernel_size: int = 7):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.norm_conv = nn.LayerNorm(d_model)
        self.causal_conv = nn.Sequential(
            CausalConv1d(d_model, d_model, kernel_size),
            nn.SiLU()
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x, memory_audio):
        # 1. Causal Self-Attention (Notes tracking Notes)
        nx = self.norm1(x)
        # F.scaled_dot_product_attention is lightning fast on MPS
        sa_out = F.scaled_dot_product_attention(nx, nx, nx, is_causal=True)
        x = x + sa_out
        
        # 2. Causal Cross-Attention (Notes querying Audio)
        # Because Audio and Notes both have length T, is_causal=True perfectly
        # masks out future audio frames, preserving strict causality.
        nx2 = self.norm2(x)
        ca_out = F.scaled_dot_product_attention(nx2, memory_audio, memory_audio, is_causal=True)
        x = x + ca_out
        
        # 3. Local Temporal Bridge (1D Causal Convolution)
        nx_conv = self.norm_conv(x)
        nx_conv = nx_conv.transpose(1, 2)
        conv_out = self.causal_conv(nx_conv)
        conv_out = conv_out.transpose(1, 2)
        x = x + conv_out
        
        # 4. Feed Forward
        x = x + self.ffn(self.norm3(x))
        return x

class TransformerCausalDecoder(nn.Module):
    def __init__(
        self, 
        d_model: int = 256, 
        num_layers: int = 4, 
        d_audio: int = 128, 
        d_target: int = 8, 
        num_diff: int = 6
    ):
        super().__init__()
        self.d_model = d_model
        
        # Core Projections
        self.audio_proj = nn.Linear(d_audio, d_model)
        self.target_proj = nn.Linear(d_target, d_model)
        self.diff_emb = nn.Embedding(num_diff, d_model)
        
        self.pos_enc = PositionalEncoding(d_model)
        
        # Decoder Pipeline
        self.layers = nn.ModuleList([
            NativeCausalBlock(d_model, d_model * 4) for _ in range(num_layers)
        ])
        
        # Output Heads (Segregated for Modular Stability)
        self.presence_out = nn.Linear(d_model, 2)
        self.position_out = nn.Linear(d_model, 4)
        self.velocity_out = nn.Linear(d_model, 2)

    def forward(self, audio_features, target_features, difficulty_idx):
        """
        Calculates causal predictions.
        audio_features: (B, T, 128)
        target_features: (B, T, 8)
        difficulty_idx: (B)
        """
        B, T, _ = audio_features.shape
        
        # 1. Shift targets right computationally to emulate true autoregressive generation
        shifted_targets = torch.zeros_like(target_features)
        shifted_targets[:, 1:, :] = target_features[:, :-1, :]
        
        # 2. Project
        v_audio = self.pos_enc(self.audio_proj(audio_features))
        v_tgt = self.pos_enc(self.target_proj(shifted_targets))
        v_diff = self.diff_emb(difficulty_idx).unsqueeze(1).expand(-1, T, -1)
        
        # 3. Primary Hybrid Base Sequence
        x = v_tgt + v_audio + v_diff
        
        # 4. Deep Decoder processing
        for layer in self.layers:
            x = layer(x, memory_audio=v_audio)
            
        # 5. Extract multi-task logits
        presence_logits = self.presence_out(x)
        position_pred = torch.tanh(self.position_out(x))
        velocity_pred = torch.tanh(self.velocity_out(x))
        
        return {
            "presence_logits": presence_logits,
            "position_pred": position_pred,
            "velocity_pred": velocity_pred,
            "latent_state": x
        }
