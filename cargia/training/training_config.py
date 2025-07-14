"""
Configuration settings for model training.
"""
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    verbose: bool = True # when true, will print out the training progress
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data paths
    data_dir: str = "data/solves_and_thoughts" # where the data is stored
    source_folder: str = "data/arc_agi_2_reformatted" # where the source ARC AGI 2 data is stored

    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    
    # Model configuration
    start_checkpoint_path: str = "C:\\Users\\thomas\\proj\\arcagi\\local_data\\model_weights\\gemma3-4b-it-ORIGINAL" # if None, will start from scratch
    

    # Loss weights
    grid_loss_weight: float = 1.0  # λ in the loss equation
    intermediate_weight: float = 0.0  # α for intermediate supervision
    
    # Data augmentation flags
    use_any_augmentation: bool = True
    use_color_invariance: bool = True
    use_char_invariance: bool = True
    use_spatial_aug: bool = False
    
    # Validation settings
    validation_interval: int = 100  # Steps between validation
    snapshot_interval: int = 500  # Steps between saving examples
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Optional settings
    seed: Optional[int] = None
    num_workers: int = 4
    mixed_precision: bool = True 

TRAINING_CONFIG = TrainingConfig()