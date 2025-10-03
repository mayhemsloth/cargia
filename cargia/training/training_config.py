"""
Configuration settings for model training.
"""
from dataclasses import dataclass
from typing import Optional
import os
import torch

@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    verbose: bool = True # when true, will print out the training progress
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data paths
    data_dir: str = "data/solves_and_thoughts" # where the data is stored
    source_folder: str = "data/arc_agi_2_reformatted" # where the source ARC AGI 2 data is stored
    
    # Data sampling control
    data_sample_maximum_limit: Optional[int] = None  # None = use full dataset, int = limit total samples
    
    # Model configuration
    start_checkpoint_path: str = os.getenv("GEMMA3_MODEL_PATH", "gemma3-4b-it-ORIGINAL") # if None, will start from scratch
    
    # Training control
    max_steps: int = 1000  # Maximum training steps (useful for overfitting tests)
    num_train_epochs: int = 1  # Number of training epochs
    
    # Loss weights
    assistant_only_loss: bool = True
    grid_loss_weight: float = 10.0  # Weight multiplier for grid output tokens (higher = more emphasis on grid accuracy)
    binary_grid_loss_weight: float = 50.0  # Weight multiplier for binary grid accuracy (all-or-nothing penalty)
    binary_loss_type: str = "exponential"  # Type of binary loss: "exponential", "sigmoid", "threshold"
    binary_loss_sensitivity: float = 10.0  # Sensitivity factor for binary loss (higher = more spiky)
    
    # Data augmentation flags
    use_any_augmentation: bool = False
    use_color_invariance: bool = True
    use_char_invariance: bool = True
    use_spatial_aug: bool = False
    use_grid_loss: bool = True

    # Quantization options
    use_bitsnbytes: bool = False
    

TRAINING_CONFIG = TrainingConfig()