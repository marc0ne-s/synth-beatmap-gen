import torch
import torch.nn as nn

class FeasibilityScorer(nn.Module):
    def __init__(self, seq_in=7, glob_in=8, hidden=128):
        super().__init__()
        # LSTM for sequence features
        self.lstm = nn.LSTM(input_size=seq_in, hidden_size=hidden, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        
        # Projection for global features
        self.glob_proj = nn.Sequential(
            nn.Linear(glob_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2 + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_data, glob_data):
        # seq_data: (Batch, SeqLen, Features)
        # glob_data: (Batch, Features)
        
        # LSTM output
        lstm_out, _ = self.lstm(seq_data)
        
        # Take the last relevant hidden state or max pool
        # Using max pool across time to capture "peaks" of difficulty/bad-flow
        seq_repr, _ = torch.max(lstm_out, dim=1) # (Batch, hidden * 2)
        
        # Global repr
        glob_repr = self.glob_proj(glob_data) # (Batch, hidden)
        
        # Combine
        combined = torch.cat([seq_repr, glob_repr], dim=1)
        
        return self.fc(combined)
