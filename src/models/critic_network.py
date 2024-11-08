
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, batch_size, tau, learning_rate, 
                 epsilon, time_stamp, med_size, lab_size, demo_size, di_size, 
                 action_dim, device):
        super().__init__()
        
        # Save initialization parameters
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
            num_embeddings=2001,
            embedding_dim=HIDDEN1_UNITS,
            padding_idx=0
        )
        self.disease_fc = nn.Linear(HIDDEN1_UNITS, HIDDEN1_UNITS)
        self.disease_prelu = nn.PReLU()

        # Action processing
        self.action_fc = nn.Linear(action_dim, HIDDEN2_UNITS)

        # Q-value layers
        merged_size = HIDDEN2_UNITS + HIDDEN1_UNITS * 2  # LSTM + disease + demo
        self.q_layers = nn.Sequential(
            nn.Linear(merged_size + HIDDEN2_UNITS, HIDDEN2_UNITS),
            nn.PReLU(),
            nn.Linear(HIDDEN2_UNITS, 1)
        )

        # Move to device
        self.to(device)
        
        # Initialize target network
        self.target_network = None
        self.create_target_network()

    def masked_mean(self, x, mask=None):
        """Compute masked mean."""
        if mask is None:
            return torch.mean(x, dim=-2)
        mask = mask.float().unsqueeze(-1)
        x = x * mask
        return x.sum(-2) / mask.sum(-2).clamp(min=1e-10)

    def process_labs(self, lab_tests):
        """Process lab test values through LSTM."""
        x = self.lab_dropout(lab_tests)
        x, _ = self.lstm(x)
        return x

    def process_demographics(self, demographics):
        """Process demographic features."""
        x = self.demo_fc(demographics)
        x = self.demo_prelu(x)
        x = x.unsqueeze(1).repeat(1, self.time_stamp, 1)
        return x

    def process_diseases(self, disease, mask=None):
        """Process disease codes."""
        x = self.disease_dropout(disease)
        x = self.disease_embedding(x)
        x = self.masked_mean(x, mask)
        x = self.disease_fc(x)
        x = self.disease_prelu(x)
        x = x.unsqueeze(1).repeat(1, self.time_stamp, 1)
        return x

    def process_actions(self, actions):
        """Process actions."""
        return self.action_fc(actions)

    def forward(self, lab_tests, actions, disease, demographics, mask=None):
        """
        Forward pass of critic network.
        
        Args:
            lab_tests: Lab test values [batch, time, lab_size]
            actions: Actions taken [batch, time, action_dim]
            disease: Disease codes [batch, di_size]
            demographics: Demographic features [batch, demo_size]
            mask: Optional mask for disease codes [batch, di_size]
        """
        # Process each input type
        lab_features = self.process_labs(lab_tests)
        demo_features = self.process_demographics(demographics)
        disease_features = self.process_diseases(disease, mask)
        action_features = self.process_actions(actions)
        
        # Combine all features
        combined = torch.cat([
            lab_features,
            disease_features,
            demo_features,
            action_features
        ], dim=-1)
        
        # Output Q-values
        q_values = self.q_layers(combined)
        return q_values

    def create_target_network(self):
        """Create a target network as a copy of the current network."""
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
            self.device
        )
        self.target_network.load_state_dict(self.state_dict())
        self.target_network.eval()

    def target_train(self):
        """Update target network using soft update."""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
