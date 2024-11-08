import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180

class BaseNetwork(nn.Module):
    """Base network with shared components for Actor and Critic"""
    def __init__(self, lab_size: int, demo_size: int, di_size: int, time_stamp: int):
        super().__init__()
        self.time_stamp = time_stamp
        
        # Lab test processing
        self.lab_dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            input_size=lab_size,
            hidden_size=HIDDEN2_UNITS,
            batch_first=True
        )
        
        # Demographics processing
        self.demo_fc = nn.Linear(demo_size, HIDDEN1_UNITS)
        self.demo_prelu = nn.PReLU()
        
        # Disease processing
        self.disease_dropout = nn.Dropout(0.2)
        self.disease_embedding = nn.Embedding(
            num_embeddings=2001,  # Max disease codes + 1 for padding
            embedding_dim=HIDDEN1_UNITS,
            padding_idx=0
        )
        
    def masked_mean(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute masked mean."""
        if mask is None:
            return torch.mean(x, dim=-2)
        mask = mask.float().unsqueeze(-1)
        x = x * mask
        return x.sum(-2) / mask.sum(-2).clamp(min=1e-10)

    def process_labs(self, lab_tests: torch.Tensor) -> torch.Tensor:
        """Process lab test values through LSTM."""
        x = self.lab_dropout(lab_tests)
        x, _ = self.lstm(x)
        return x

    def process_demographics(self, demographics: torch.Tensor) -> torch.Tensor:
        """Process demographic features."""
        x = self.demo_fc(demographics)
        x = self.demo_prelu(x)
        x = x.unsqueeze(1).repeat(1, self.time_stamp, 1)
        return x

    def process_diseases(self, disease: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Process disease codes."""
        x = self.disease_dropout(disease)
        x = self.disease_embedding(x)
        x = self.masked_mean(x, mask)
        x = x.unsqueeze(1).repeat(1, self.time_stamp, 1)
        return x

    def forward(self, lab_tests: torch.Tensor, disease: torch.Tensor, 
                demographics: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through base network.
        
        Args:
            lab_tests: Lab test values [batch, time, lab_size]
            disease: Disease codes [batch, di_size]
            demographics: Demographic features [batch, demo_size]
            mask: Optional mask for disease codes [batch, di_size]
            
        Returns:
            Combined features [batch, time, hidden_size]
        """
        lab_features = self.process_labs(lab_tests)
        demo_features = self.process_demographics(demographics)
        disease_features = self.process_diseases(disease, mask)
        
        return torch.cat([lab_features, disease_features, demo_features], dim=-1)
