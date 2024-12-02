import torch
import torch.nn as nn
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork

class SRLModel(nn.Module):
    """Combined SRL model with Actor and Critic networks"""
    def __init__(self, lab_size, demo_size, di_size, med_size, time_stamp, config, device):
        super(SRLModel, self).__init__()
        
        self.lab_size = lab_size
        self.demo_size = demo_size
        self.di_size = di_size
        self.med_size = med_size
        self.time_stamp = time_stamp
        self.device = device
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(
            state_size=config.state_dim,
            action_size=med_size,
            batch_size=config.batch_size,
            tau=config.tau,
            learning_rate=config.lra,
            epsilon=config.epsilon,
            time_stamp=time_stamp,
            med_size=med_size,
            lab_size=lab_size,
            demo_size=demo_size,
            di_size=di_size,
            device=device
        )
        
        self.critic = CriticNetwork(
            state_size=config.state_dim,
            action_size=med_size,
            batch_size=config.batch_size,
            tau=config.tau,
            learning_rate=config.lrc,
            epsilon=config.epsilon,
            time_stamp=time_stamp,
            med_size=med_size,
            lab_size=lab_size,
            demo_size=demo_size,
            di_size=di_size,
            action_dim=med_size,
            device=device
        )

    @classmethod
    def from_config(cls, config, device):
        """Create model from config"""
        return cls(
            lab_size=config.lab_size,
            demo_size=config.demo_size,
            di_size=config.di_size,
            med_size=config.med_size,
            time_stamp=config.time_stamp,
            config=config,
            device=device
        )

    def forward(self, states, disease, demos):
        """Forward pass through both networks"""
        # Get actions from actor
        actions = self.actor(states, disease, demos)
        
        # Get Q-values from critic
        q_values = self.critic(states, actions, disease, demos)
        
        return actions, q_values

    def update_targets(self, tau):
        """Update target networks"""
        self.actor.target_train()
        self.critic.target_train()
