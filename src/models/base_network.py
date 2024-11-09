# base_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants used across networks
HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180

class BaseNetwork(nn.Module):
    """Base network with shared components for Actor and Critic"""
    def __init__(self):
        """Initialize base network"""
        super().__init__()

        # Shared attention layer that will be used by both Actor and Critic
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN2_UNITS * 2, HIDDEN2_UNITS),
            nn.Tanh(),
            nn.Linear(HIDDEN2_UNITS, 1)
        )
    
    def apply_attention(self, sequence, lengths, attention_layer):
        """
        Apply attention mechanism to sequence
        
        Args:
            sequence: Tensor of shape [batch_size, seq_len, hidden_size]
            lengths: Tensor of shape [batch_size] containing actual sequence lengths
            attention_layer: The attention layer to use
        """
        batch_size, seq_len, hidden_size = sequence.shape
        
        # Squeeze lengths if needed
        lengths = lengths.squeeze(-1)  # [batch_size]
        
        # Create attention mask
        device = sequence.device
        mask = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(-1)

        # Calculate attention scores
        attention_scores = attention_layer(sequence)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention to sequence
        context = (attention_weights * sequence).sum(dim=1)
        return context
    
    def target_train(self):
        """
        Update target network using soft update
        
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        if self.target_network is None:
            return
            
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
            
    def process_sequences(self, sequences, lengths):
        """Process variable length sequences with masking"""
        if lengths is not None:
            mask = torch.arange(sequences.size(1), device=self.device)[None, :] < lengths[:, None]
            mask = mask.float().unsqueeze(-1)
            masked_seq = sequences * mask
            return masked_seq.sum(dim=1) / lengths.float().unsqueeze(-1).clamp(min=1)
        else:
            return sequences.mean(dim=1)