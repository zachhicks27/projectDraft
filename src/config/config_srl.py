from pathlib import Path

class Config:
    def __init__(self):
        # Set up paths relative to project root
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / 'data'
        self.processed_path = self.data_path / 'processed'

        # Model hyperparameters adjusted for your data
        self.batch_size = 30
        self.gamma = 0.99
        self.tau = 0.001
        self.lra = 0.001  # Learning rate for actor
        self.lrc = 0.005  # Learning rate for critic
        self.epsilon = 0.5  # Balance between SL and RL
        self.time_stamp = 50  # Based on your sequence length
        self.lab_size = 8   # From your timeseries shape
        self.demo_size = 4  # [age, gender, weight, height]
        self.med_size = 118  # From your ATC codes + 1 for padding
        self.di_size = 507   # From your ICD9 codes + 1 for padding
        self.debug_embeddings = True  # Add this line
        self.state_dim = 12  # State dimension
        self.seed = 1337
        self.episode_count = 100  # Changed from 100000 to 100

        # Create directories if they don't exist
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def check_files(self) -> bool:
        """Check if all required files exist"""
        required_files = [
            self.processed_path / 'train_demographics.csv',
            self.processed_path / 'val_demographics.csv',
            self.processed_path / 'train_diagnoses.csv',
            self.processed_path / 'val_diagnoses.csv',
            self.processed_path / 'train_timeseries.csv',
            self.processed_path / 'val_timeseries.csv',
            self.processed_path / 'train_prescriptions.csv',
            self.processed_path / 'val_prescriptions.csv'
        ]
        
        missing_files = [str(f) for f in required_files if not f.exists()]
        if missing_files:
            print("\nMissing required data files:")
            for f in missing_files:
                print(f"  - {f}")
            print("\nPlease run preprocessing first.")
            return False
        return True

    def get_data_paths(self):
        """Get dictionary of data file paths"""
        return {
            'train_demographics': self.processed_path / 'train_demographics.csv',
            'val_demographics': self.processed_path / 'val_demographics.csv',
            'train_diagnoses': self.processed_path / 'train_diagnoses.csv',
            'val_diagnoses': self.processed_path / 'val_diagnoses.csv',
            'train_timeseries': self.processed_path / 'train_timeseries.csv',
            'val_timeseries': self.processed_path / 'val_timeseries.csv',
            'train_prescriptions': self.processed_path / 'train_prescriptions.csv',
            'val_prescriptions': self.processed_path / 'val_prescriptions.csv'
        }

def get_config():
    """Get default configuration"""
    return Config()

# Create a single instance
config = get_config()
