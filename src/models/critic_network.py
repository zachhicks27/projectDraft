import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .base_network import BaseNetwork

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180

class CriticNetwork(BaseNetwork):
    def __init__(self, state_size, action_size, batch_size, tau, learning_rate, 
                 epsilon, time_stamp, med_size, lab_size, demo_size, di_size, 
                 action_dim, device, vocab_size=507,  # One more than unique codes for padding
                 create_target=True):
        super().__init__()
        
        # Save initialization parameters (same as actor)
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.time_stamp = time_stamp
        self.med_size = med_size
        self.lab_size = lab_size
        self.demo_size = demo_size
        self.di_size = di_size
        self.action_dim = action_dim
        self.device = device

        # Lab test processing (same as actor)
        self.lab_dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            input_size=lab_size,
            hidden_size=HIDDEN2_UNITS,
            batch_first=True,
            bidirectional=True
        )

        # Demographics processing (same as actor)
        self.demo_fc = nn.Linear(demo_size, HIDDEN1_UNITS)
        self.demo_prelu = nn.PReLU()

        # Disease processing (same as actor)
        self.disease_dropout = nn.Dropout(0.2)
        self.disease_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=HIDDEN1_UNITS,
            padding_idx=0
        )
        self.disease_fc = nn.Linear(HIDDEN1_UNITS, HIDDEN1_UNITS)
        self.disease_prelu = nn.PReLU()

        # Separate attention mechanisms for labs and diseases
        self.lab_attention = nn.Sequential(
            nn.Linear(HIDDEN2_UNITS * 2, HIDDEN2_UNITS),
            nn.Tanh(),
            nn.Linear(HIDDEN2_UNITS, 1)
        )

        self.disease_attention = nn.Sequential(
            nn.Linear(HIDDEN1_UNITS, HIDDEN1_UNITS),
            nn.Tanh(),
            nn.Linear(HIDDEN1_UNITS, 1)
        )

        self.attention = nn.Sequential(
            nn.Linear(HIDDEN2_UNITS * 2, HIDDEN2_UNITS),  # *2 because bidirectional LSTM
            nn.Tanh(),
            nn.Linear(HIDDEN2_UNITS, 1)
        )

        # Action processing
        self.action_fc = nn.Linear(action_dim, HIDDEN2_UNITS)

        # Q-value layers
        merged_size = HIDDEN2_UNITS * 2 + HIDDEN1_UNITS * 2 + HIDDEN2_UNITS
        self.q_layers = nn.Sequential(
            nn.Linear(merged_size, HIDDEN2_UNITS),
            nn.PReLU(),
            nn.Linear(HIDDEN2_UNITS, 1)
        )

        # Move to device
        self.to(device)
        
        # Initialize target network
        self.target_network = None
        if create_target:
            self.create_target_network()

    def forward(self, lab_tests, actions, disease, demographics, lengths=None):
        # The actions input should match the expected shape before processing
        # Add a check or reshape if needed
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected actions with shape [..., {self.action_dim}], got {actions.shape}")
        
        batch_size = lab_tests.size(0)
        
        # Process lab tests [batch_size, seq_len, lab_size]
        x_lab = self.lab_dropout(lab_tests)
        x_lab, _ = self.lstm(x_lab)  # [batch_size, seq_len, hidden_size*2]
        
        # Apply attention to labs
        if lengths is not None:
            x_lab = self.apply_attention(x_lab, lengths, self.lab_attention)
        else:
            x_lab = x_lab.mean(dim=1)
            
        # Process demographics [batch_size, demo_size]
        x_demo = self.demo_fc(demographics)
        x_demo = self.demo_prelu(x_demo)  # [batch_size, hidden1]
        
        # Process diseases [batch_size, seq_len]
        x_disease = disease.long()  # Convert to long first
        x_disease = torch.clamp(x_disease, min=0, max=self.disease_embedding.num_embeddings - 1)
        x_disease = self.disease_dropout(x_disease.float())  # Apply dropout to float version
        x_disease = x_disease.long()  # Convert back to long for embedding
        x_disease = torch.clamp(x_disease, min=0, max=self.disease_embedding.num_embeddings - 1)
        
        x_disease = self.disease_embedding(x_disease)  # [batch_size, seq_len, hidden1]
        
        # Apply attention to diseases
        if lengths is not None:
            x_disease = self.apply_attention(x_disease, lengths, self.disease_attention)
        else:
            x_disease = x_disease.mean(dim=1)
        
        x_disease = self.disease_fc(x_disease)
        x_disease = self.disease_prelu(x_disease)  # [batch_size, hidden1]
        
        # Process actions [batch_size, action_dim]
        x_action = self.action_fc(actions)  # [batch_size, hidden2]
        
        # Combine features
        combined = torch.cat([x_lab, x_disease, x_demo, x_action], dim=1)
        
        # Output Q-values
        return self.q_layers(combined)
    
    def create_target_network(self):
        """Create target network for critic"""
        self.target_network = CriticNetwork(
            self.state_size,
            self.action_size,
            self.batch_size,
            self.tau,
            self.learning_rate,
            self.epsilon,
            self.time_stamp,
            self.med_size,
            self.lab_size,
            self.demo_size,
            self.di_size,
            self.action_dim,
            self.device,
            create_target=False
        )
        
        # Copy weights manually without target network parameters
        state_dict = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                            if not k.startswith('target_network.')}
        self.target_network.load_state_dict(filtered_state_dict)

def apply_attention(self, sequence, lengths, attention_layer):
        """
        Apply attention mechanism to sequence
        
        Args:
            sequence: [batch_size, seq_len, hidden_size]
            lengths: [batch_size, 1]
            attention_layer: attention mechanism to use
        Returns:
            [batch_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = sequence.shape
        
        # Squeeze lengths if needed
        lengths = lengths.squeeze(-1)  # [batch_size]
        
        # Create attention mask
        device = sequence.device
        mask = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]  # [batch_size, seq_len]
        mask = mask.float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Calculate attention scores
        attention_scores = attention_layer(sequence)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.masked_fill(~mask.bool(), float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention (broadcasting will handle the dimensions)
        context = (attention_weights * sequence).sum(dim=1)  # [batch_size, hidden_size]
        
        return context