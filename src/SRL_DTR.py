import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.models import ActorNetwork, CriticNetwork

class MIMICDataset(Dataset):
    # Class variables to store global mappings
    icd9_to_idx = None
    atc_to_idx = None
    vocab_size = None
    med_vocab_size = None
    
    def __init__(self, data_path, split='train'):
        self.data_path = Path(data_path)
        self.split = split
        
        # Load split-specific data
        print(f"\nLoading {split} data...")
        self.demographics = pd.read_csv(self.data_path / f'{split}_demographics.csv')
        self.timeseries = pd.read_csv(self.data_path / f'{split}_timeseries.csv')
        self.diagnoses = pd.read_csv(self.data_path / f'{split}_diagnoses.csv')
        self.prescriptions = pd.read_csv(self.data_path / f'{split}_prescriptions.csv')
        
        # Create mappings for first time with training data
        if split == 'train':
            if MIMICDataset.icd9_to_idx is None:
                print("\nCreating global ICD9 code mapping...")
                all_codes = set(self.diagnoses['icd9_code'].unique())
                MIMICDataset.icd9_to_idx = {'PAD': 0}
                for idx, code in enumerate(sorted(all_codes), 1):
                    MIMICDataset.icd9_to_idx[str(code)] = idx
                MIMICDataset.vocab_size = len(MIMICDataset.icd9_to_idx)
                print(f"Total unique ICD9 codes (including padding): {MIMICDataset.vocab_size}")
            
            if MIMICDataset.atc_to_idx is None:
                print("\nCreating global ATC code mapping...")
                all_atc = set(self.prescriptions['atc_code'].dropna().unique())
                MIMICDataset.atc_to_idx = {'PAD': 0}
                for idx, code in enumerate(sorted(all_atc), 1):
                    MIMICDataset.atc_to_idx[str(code)] = idx
                MIMICDataset.med_vocab_size = len(MIMICDataset.atc_to_idx)
                print(f"Total unique ATC codes (including padding): {MIMICDataset.med_vocab_size}")
        
        # Print info
        print("\nAvailable columns:")
        print("Timeseries columns:", self.timeseries.columns.tolist())
        print("Demographics columns:", self.demographics.columns.tolist())
        print("Diagnoses columns:", self.diagnoses.columns.tolist())
        print("Prescriptions columns:", self.prescriptions.columns.tolist())
        
        # Define feature columns
        self.lab_cols = ['dbp', 'fio2', 'gcs', 'sbp', 'hr', 'rr', 'spo2', 'temp']
        self.demo_cols = ['age', 'gender', 'weight', 'height']
        self.max_seq_length = 50
        
        # Get unique admission IDs
        self.hadm_ids = self.demographics['hadm_id'].unique()
        print(f"Loaded {len(self.hadm_ids)} admissions for {split}")
        
        # Define feature columns
        self.lab_cols = ['dbp', 'fio2', 'gcs', 'sbp', 'hr', 'rr', 'spo2', 'temp']
        self.demo_cols = ['age', 'gender', 'weight', 'height']
        self.max_seq_length = 50

        print("\nColumn data types:")
        print("\nDemographics types:")
        print(self.demographics[self.demo_cols].dtypes)
        print("\nLab tests types:")
        print(self.timeseries[self.lab_cols].dtypes)
        print("\nDiagnoses types:")
        print(self.diagnoses['icd9_code'].dtype)
        print("\nPrescriptions types:")
        print(self.prescriptions['atc_code'].dtype)

    def pad_sequence(self, sequence, max_length, pad_value=0):
        """Pad sequence to max_length"""
        sequence = np.array(sequence)
        
        if len(sequence) == 0:
            if len(sequence.shape) > 1:
                return np.zeros((max_length, sequence.shape[1]))
            return np.zeros((max_length,))
        
        if len(sequence) > max_length:
            return sequence[:max_length]
        
        pad_width = [(0, max_length - len(sequence))]
        if len(sequence.shape) > 1:
            pad_width.append((0, 0))
            
        return np.pad(sequence, pad_width, 'constant', constant_values=pad_value)

    def convert_icd9_to_idx(self, code):
        """Safely convert ICD9 code to index"""
        if MIMICDataset.icd9_to_idx is None:
            raise ValueError("ICD9 mapping not initialized!")
        return MIMICDataset.icd9_to_idx.get(str(code), 0)

    def convert_atc_to_idx(self, code):
        """Safely convert ATC code to index"""
        if MIMICDataset.atc_to_idx is None:
            raise ValueError("ATC mapping not initialized!")
        return MIMICDataset.atc_to_idx.get(str(code), 0)

    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        
        # Get demographics data
        demo_df = self.demographics[self.demographics['hadm_id'] == hadm_id][self.demo_cols]
        if 'gender' in demo_df.columns:
            demo_df['gender'] = (demo_df['gender'] == 'M').astype(float)
        demo = demo_df.astype(float).fillna(0).values[0]
        
        # Get lab data
        labs = self.timeseries[self.timeseries['hadm_id'] == hadm_id][self.lab_cols].fillna(0).values
        
        # Convert ICD9 codes to indices
        diag_codes = self.diagnoses[self.diagnoses['hadm_id'] == hadm_id]['icd9_code'].values
        diag = np.array([self.convert_icd9_to_idx(code) for code in diag_codes], dtype=np.int64)
        
        # Convert ATC codes to one-hot encoding
        med_codes = self.prescriptions[self.prescriptions['hadm_id'] == hadm_id]['atc_code'].values
        meds = np.zeros(self.med_vocab_size)  # Initialize zero vector of vocabulary size
        for code in med_codes:
            if pd.notna(code):  # Check if code is not NaN
                idx = self.convert_atc_to_idx(str(code))
                if idx > 0:  # Ignore padding index 0
                    meds[idx] = 1
        
        # Ensure arrays are 2D
        if len(meds.shape) == 1:
            meds = meds.reshape(-1, 1)
        if len(diag.shape) == 1:
            diag = diag.reshape(-1, 1)
        
        # Pad sequences
        labs = self.pad_sequence(labs, self.max_seq_length)
        diag = self.pad_sequence(diag, self.max_seq_length)
        meds = self.pad_sequence(meds, self.max_seq_length)
        
        original_length = len(labs) if len(labs.shape) > 1 else 1
        
        # Print shapes for debugging
        print(f"\nDataset item shapes:")
        print(f"demographics: {demo.shape}")
        print(f"labs: {labs.shape}")
        print(f"diagnoses: {diag.shape}")
        print(f"medications: {meds.shape}")
        print(f"seq_length: {[original_length]}")
        
        return {
            'demographics': torch.FloatTensor(demo),
            'labs': torch.FloatTensor(labs),
            'diagnoses': torch.LongTensor(diag).squeeze(-1),
            'medications': torch.FloatTensor(meds),  # Now a one-hot encoded vector
            'hadm_id': hadm_id,
            'seq_length': torch.LongTensor([original_length])
        }

    def __len__(self):
        return len(self.hadm_ids)

class ExperienceBuffer:
    """Experience replay buffer"""
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (
            {k: torch.stack([s[k] for s in states]) for k in states[0].keys()},
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32),
            {k: torch.stack([s[k] for s in next_states]) for k in next_states[0].keys()},
            torch.tensor(dones, dtype=torch.float32)
        )

class SRL_DTR:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Setup data first
        self.setup_data()
        
        # Initialize networks with correct vocab sizes
        vocab_size = MIMICDataset.vocab_size
        med_vocab_size = MIMICDataset.med_vocab_size
        
        if vocab_size is None or med_vocab_size is None:
            raise ValueError("Vocabulary sizes not initialized! Dataset setup failed.")
        
        print(f"Initializing networks with diagnosis vocabulary size: {vocab_size}")
        print(f"Medication vocabulary size: {med_vocab_size}")
        
        # Update config with correct medication size
        config.med_size = med_vocab_size
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(config=self.config,
                                  state_size=config.state_dim,
                                  action_size=config.med_size,
                                  batch_size=config.batch_size,
                                  tau=config.tau,
                                  learning_rate=config.lra,
                                  epsilon=config.epsilon,
                                  time_stamp=config.time_stamp,
                                  med_size=config.med_size,
                                  lab_size=config.lab_size,
                                  demo_size=config.demo_size,
                                  di_size=config.di_size,
                                  device=self.device,
                                  vocab_size=vocab_size
                                 ).to(self.device)
        
        self.critic = CriticNetwork(
            state_size=config.state_dim,
            action_size=config.med_size,
            batch_size=config.batch_size,
            tau=config.tau,
            learning_rate=config.lrc,
            epsilon=config.epsilon,
            time_stamp=config.time_stamp,
            med_size=config.med_size,
            lab_size=config.lab_size,
            demo_size=config.demo_size,
            di_size=config.di_size,
            action_dim=config.med_size,
            device=self.device,
            vocab_size=vocab_size
        ).to(self.device)
        
        # Initialize experience buffer
        self.buffer = ExperienceBuffer(max_size=10000)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=config.lra
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=config.lrc
        )
        
        self.max_grad_norm = 1.0
    
    def setup_data(self):
        """Initialize datasets and data loaders"""
        self.train_dataset = MIMICDataset(self.config.processed_path, split='train')
        self.val_dataset = MIMICDataset(self.config.processed_path, split='val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for easier debugging
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def calculate_reward(self, mortality):
        """Calculate reward based on mortality flag"""
        return torch.where(mortality == 1, 
                         torch.tensor(-15.0), 
                         torch.where(mortality == 0,
                                   torch.tensor(15.0),
                                   torch.tensor(0.0)))
    
    def train_step(self, batch):
        """Single training step"""
        torch.autograd.set_detect_anomaly(True)
        
        # Move batch to device and ensure we're working with fresh tensors
        states = {k: v.clone().detach().to(self.device) for k, v in batch.items() 
                if k not in ['hadm_id', 'medications', 'seq_length']}
        doctor_actions = batch['medications'].clone().detach().to(self.device)
        seq_lengths = batch['seq_length'].clone().detach().to(self.device)
        
        # Forward pass through actor
        actions = self.actor(
            states['labs'],
            states['diagnoses'],
            states['demographics'],
            seq_lengths
        ).detach()  # Detach actions for critic update
        
        # Forward pass through critic
        q_values = self.critic(
            states['labs'].clone(),
            actions.clone(),
            states['diagnoses'].clone(),
            states['demographics'].clone(),
            seq_lengths.clone()
        )
        
        # Create new tensors for expanded q_values
        q_values_expanded = q_values.unsqueeze(-1).clone()
        q_values_expanded = q_values_expanded.expand(-1, doctor_actions.size(1), -1).clone()
        
        # Calculate critic loss
        critic_loss = F.mse_loss(q_values_expanded.clone(), doctor_actions)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Get fresh actions for actor update
        actions = self.actor(
            states['labs'],
            states['diagnoses'],
            states['demographics'],
            seq_lengths
        )
        
        # Get fresh Q-values
        q_values = self.critic(
            states['labs'],
            actions,
            states['diagnoses'],
            states['demographics'],
            seq_lengths
        )
        
        # Create one-hot encoded tensor
        batch_size = doctor_actions.size(0)
        doctor_actions_squeezed = doctor_actions.squeeze(-1)
        flattened_actions = torch.zeros(batch_size, self.config.med_size, device=self.device)
        
        for b in range(batch_size):
            for t in range(doctor_actions_squeezed.size(1)):
                idx = doctor_actions_squeezed[b, t].long()
                if idx < self.config.med_size:
                    flattened_actions[b, idx] = 1.0
        
        # Calculate actor losses with fresh tensors
        rl_loss = -q_values.mean()
        sl_loss = F.binary_cross_entropy(actions.clone(), flattened_actions)
        actor_loss = (1 - self.config.epsilon) * rl_loss + self.config.epsilon * sl_loss
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update target networks
        self.actor.target_train()
        self.critic.target_train()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'rl_loss': rl_loss.item(),
            'sl_loss': sl_loss.item()
        }
    
    def validate(self):
        """Validation step"""
        self.actor.eval()
        self.critic.eval()
        val_metrics = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                states = {k: v.to(self.device) for k, v in batch.items() 
                         if k not in ['hadm_id', 'medications']}
                doctor_actions = batch['medications'].to(self.device)
                
                # Forward pass through actor with sequence lengths
                actions = self.actor(
                    states['labs'],
                    states['diagnoses'],
                    states['demographics'],
                    states['seq_length']  # Add sequence length parameter
                )
                
                # Convert doctor_actions to same format as predictions
                batch_size = doctor_actions.size(0)
                doctor_actions_one_hot = torch.zeros(batch_size, self.config.med_size, 
                                                   device=self.device)
                
                # Convert sequential actions to one-hot encoded format
                for b in range(batch_size):
                    for t in range(doctor_actions.size(1)):
                        idx = doctor_actions[b, t].long()
                        if idx < self.config.med_size:
                            doctor_actions_one_hot[b, idx] = 1.0
                
                # Calculate Jaccard similarity with matched dimensions
                pred_binary = (actions > 0.5).float()
                intersection = (pred_binary * doctor_actions_one_hot).sum(1)
                union = (pred_binary + doctor_actions_one_hot).clamp(0, 1).sum(1)
                jaccard = (intersection / union.clamp(min=1e-8)).mean()
                
                val_metrics.append({
                    'jaccard': jaccard.item()
                })
        
        self.actor.train()
        self.critic.train()
        return {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
    
    def save_checkpoint(self, filename, epoch):
        """Save model checkpoint"""
        checkpoint_path = self.config.processed_path / 'checkpoints' / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, checkpoint_path)
    
    def train(self):
        """Main training loop"""
        print("Starting training on device:", self.device)
        best_jaccard = 0
        
        for epoch in range(self.config.episode_count):
            # Training
            self.actor.train()
            self.critic.train()
            epoch_losses = []
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                losses = self.train_step(batch)
                epoch_losses.append(losses)
        
            # Calculate average losses
            avg_losses = {k: np.mean([loss[k] for loss in epoch_losses]) 
                         for k in epoch_losses[0].keys()}
        
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            for k, v in avg_losses.items():
                print(f"  {k}: {v:.4f}")
        
            # Validation
            val_metrics = self.validate()
            print(f"Validation Jaccard: {val_metrics['jaccard']:.4f}")

    def DTR(self):
        """Main entry point for training"""
        print("Starting training...")
        self.train()