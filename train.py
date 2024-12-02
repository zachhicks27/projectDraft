import os
from pathlib import Path
import torch
from src.config.config_srl import get_config
from src.SRL_DTR import SRL_DTR  # Changed from SRL_RNN to SRL_DTR

def check_data_ready(config):
    """Check if data is ready for training"""
    if not config.check_files():
        raise FileNotFoundError(
            "\nProcessed data files are missing. Please run preprocessing first:\n"
            "python -m src.preprocessing.preprocessing"
        )

def main():
    # Get configuration
    config = get_config()
    print("Configuration loaded.")
    
    # Check for processed data
    check_data_ready(config)
    print("Data files verified.")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Initializing model...")
    model = SRL_DTR(config)  # Changed from SRL_RNN to SRL_DTR
    
    # Train model
    print("Starting training...")
    model.DTR()
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = model.test_and_evaluate()
    
    # Save final metrics
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    import json
    with open(results_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

if __name__ == "__main__":
    main()