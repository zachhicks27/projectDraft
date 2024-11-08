import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.model import SRLModel
import pandas as pd
from pathlib import Path

class MIMICDataset(Dataset):
    """Dataset class for MIMIC data"""
    def __init__(self, data_path, split='train'):
        self.data_path = Path(data_path)
        self.split = split
        self.load_data()
        
    def load_data(self):
        # Load split-specific data
        prefix = self.split
        self.demographics = pd.read_csv(self.data_path / f'{prefix}_demographics.csv')
        self.timeseries = pd.read_csv(self.data_path / f'{prefix}_timeseries.csv')
        self.diagnoses = pd.read_csv(self.data_path / f'{prefix}_diagnoses.csv')
        self.prescriptions = pd.read_csv(self.data_path / f'{prefix}_prescriptions.csv')
        
        # Get unique admission IDs
        self.hadm_ids = self.demographics['hadm_id'].unique()
        
    def __len__(self):
        return len(self.hadm_ids)
        
    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        return self._get_admission_data(hadm_id)
    
    def _get_admission_data(self, hadm_id):
        # Get data for specific admission
        demo = self.demographics[self.demographics['hadm_id'] == hadm_id]
        labs = self.timeseries[self.timeseries['hadm_id'] == hadm_id]
        diag = self.diagnoses[self.diagnoses['hadm_id'] == hadm_id]
        meds = self.prescriptions[self.prescriptions['hadm_id'] == hadm_id]
        
        return {
            'demographics': torch.FloatTensor(demo.values),
            'labs': torch.FloatTensor(labs.values),
            'diagnoses': torch.LongTensor(diag.values),
            'medications': torch.FloatTensor(meds.values),
            'hadm_id': hadm_id
        }

class SRL_RNN:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize model using the from_config class method
        self.model = SRLModel.from_config(config, self.device).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), 
            lr=config.lra
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), 
            lr=config.lrc
        )
        
        # Setup data
        self.setup_data()

    
    def setup_data(self):
        """Setup data loaders"""
        data_path = Path('data/processed')
        
        # Create datasets
        self.train_dataset = MIMICDataset(data_path, split='train')
        self.val_dataset = MIMICDataset(data_path, split='val')
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def train_step(self, batch):
        """Single training step"""
        # Move data to device
        demographics = batch['demographics'].to(self.device)
        labs = batch['labs'].to(self.device)
        diagnoses = batch['diagnoses'].to(self.device)
        medications = batch['medications'].to(self.device)
        
        # Forward pass
        actions, q_values = self.model(labs, diagnoses, demographics)
        
        # Calculate losses
        critic_loss = nn.MSELoss()(q_values, medications)
        actor_loss = -q_values.mean()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.model.update_targets(self.config.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def validate(self):
        """Validation step"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                demographics = batch['demographics'].to(self.device)
                labs = batch['labs'].to(self.device)
                diagnoses = batch['diagnoses'].to(self.device)
                medications = batch['medications'].to(self.device)
                
                actions, q_values = self.model(labs, diagnoses, demographics)
                loss = nn.MSELoss()(q_values, medications)
                val_losses.append(loss.item())
        
        self.model.train()
        return np.mean(val_losses)
    
    def DTR(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.config.episode_count):
            # Training
            epoch_losses = []
            for batch in self.train_loader:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
            
            # Validation
            if epoch % 10 == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}")
                print(f"Training Losses: {np.mean(epoch_losses, axis=0)}")
                print(f"Validation Loss: {val_loss}")
                
            # Save checkpoint
            if epoch % 100 == 0:
                checkpoint_path = Path(f'checkpoints/model_epoch_{epoch}.pt')
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(checkpoint_path)
