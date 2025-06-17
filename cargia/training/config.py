"""
Configuration settings for model training.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    weight_decay: float = 0.01
    
    # Model configuration
    model_name: str = "google/gemma-2b"  # Will be updated to Gemma3 when available
    vision_encoder_name: str = "google/vit-base-patch16-224"  # Example vision encoder
    
    # Loss weights
    grid_loss_weight: float = 1.0  # λ in the loss equation
    intermediate_weight: float = 0.0  # α for intermediate supervision
    
    # Data augmentation flags
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