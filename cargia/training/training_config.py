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
    
    # Model configuration
    start_checkpoint_path: str = "C:\\Users\\thomas\\proj\\arcagi\\local_data\\model_weights\\gemma3-4b-it-ORIGINAL" # if None, will start from scratch
    
    # Loss weights
    grid_loss_weight: float = 1.0  # λ in the loss equation
    intermediate_weight: float = 0.0  # α for intermediate supervision
    
    # Data augmentation flags
    use_any_augmentation: bool = False
    use_color_invariance: bool = True
    use_char_invariance: bool = True
    use_spatial_aug: bool = False

    use_bitsnbytes: bool = False
    

TRAINING_CONFIG = TrainingConfig()