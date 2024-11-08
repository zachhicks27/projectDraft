import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, batch_size, tau, learning_rate, 
                 epsilon, time_stamp, med_size, lab_size, demo_size, di_size, 
                 action_dim, device, create_target=True):  # Added flag to prevent recursion
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
        merged_size = HIDDEN2_UNITS + HIDDEN1_UNITS * 2 + HIDDEN2_UNITS
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

    def forward(self, lab_tests, actions, disease, demographics, mask=None):
        # Process lab tests
        x_lab = self.lab_dropout(lab_tests)
        x_lab, _ = self.lstm(x_lab)
        
        # Process demographics
        x_demo = self.demo_fc(demographics)
        x_demo = self.demo_prelu(x_demo)
        x_demo = x_demo.unsqueeze(1).repeat(1, self.time_stamp, 1)
        
        # Process diseases
        x_disease = self.disease_dropout(disease)
        x_disease = self.disease_embedding(x_disease)
        if mask is not None:
            mask = mask.float().unsqueeze(-1)
            x_disease = x_disease * mask
            x_disease = x_disease.sum(-2) / mask.sum(-2).clamp(min=1e-10)
        else:
            x_disease = torch.mean(x_disease, dim=-2)
        x_disease = self.disease_fc(x_disease)
        x_disease = self.disease_prelu(x_disease)
        x_disease = x_disease.unsqueeze(1).repeat(1, self.time_stamp, 1)
        
        # Process actions
        x_action = self.action_fc(actions)
        
        # Combine all features
        combined = torch.cat([x_lab, x_disease, x_demo, x_action], dim=-1)
        
        # Output Q-values
        return self.q_layers(combined)

    def create_target_network(self):
        """Create target network without recursion"""
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
            create_target=False  # Prevent recursion
        )
        # Copy weights manually without target network parameters
        state_dict = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('target_network.')}
        self.target_network.load_state_dict(filtered_state_dict)

    def target_train(self):
        """Update target network using soft update"""
        if self.target_network is None:
            return
            
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )