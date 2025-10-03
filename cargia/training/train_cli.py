#!/usr/bin/env python3
"""
Command-line interface for Cargia training.

This script provides a CLI interface for running training with different configurations,
supporting both local and cloud deployment scenarios.

Usage:
    python train_cli.py --config configs/step_1_overfit_single.yaml --local
    python train_cli.py --config configs/step_1_overfit_single.yaml --cloud
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Import from installed package
from cargia.training.trainer import CargiaGoogleGemma3Trainer
from cargia.training.training_config import TrainingConfig


class TrainingCLI:
    """Command-line interface for Cargia training."""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('training.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded YAML configuration from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in configuration file: {e}")
            sys.exit(1)
            
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and required fields."""
        required_sections = ['training', 'augmentation', 'training_args', 'paths']
        
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Missing required section: {section}")
                return False
                
        # Validate training section
        training = config['training']
        if 'start_checkpoint_path' not in training:
            self.logger.error("Missing required field: training.start_checkpoint_path")
            return False
            
        # Validate paths section
        paths = config['paths']
        if 'local' not in paths or 'cloud' not in paths:
            self.logger.error("Missing required path configurations: local and cloud")
            return False
            
        self.logger.info("Configuration validation passed")
        return True
        
    def resolve_paths(self, config: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Resolve paths based on environment (local vs cloud)."""
        if environment not in ['local', 'cloud']:
            raise ValueError(f"Invalid environment: {environment}")
            
        # Get environment-specific paths
        env_paths = config['paths'][environment]
        
        # Create a copy of the training config with resolved paths
        training_config = config['training'].copy()
        training_config.update(env_paths)
        
        self.logger.info(f"Using {environment} environment paths")
        self.logger.info(f"Data directory: {training_config.get('data_dir', 'Not specified')}")
        self.logger.info(f"Model path: {training_config.get('start_checkpoint_path', 'Not specified')}")
        
        return training_config
        
    def create_training_config(self, config: Dict[str, Any], environment: str) -> TrainingConfig:
        """Create TrainingConfig instance from configuration."""
        # Resolve paths for the environment
        training_config_dict = self.resolve_paths(config, environment)
        
        # Create TrainingConfig with all the settings
        training_config = TrainingConfig()
        
        # Update with configuration values
        for key, value in training_config_dict.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
                self.logger.debug(f"Set {key} = {value}")
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
                
        # Handle augmentation settings
        augmentation = config['augmentation']
        for key, value in augmentation.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
                self.logger.debug(f"Set augmentation {key} = {value}")
                
        # Handle training arguments (these will be passed to the trainer)
        training_config.training_args = config['training_args']
        
        return training_config
        
    def run_training(self, config: Dict[str, Any], environment: str):
        """Execute the training run."""
        try:
            # Create training configuration
            training_config = self.create_training_config(config, environment)
            
            self.logger.info("Starting training with configuration:")
            self.logger.info(f"  Model: {training_config.start_checkpoint_path}")
            self.logger.info(f"  Data directory: {training_config.data_dir}")
            self.logger.info(f"  Augmentations: {training_config.use_any_augmentation}")
            self.logger.info(f"  Device: {training_config.device}")
            
            # Initialize trainer
            self.logger.info("Initializing trainer...")
            trainer = CargiaGoogleGemma3Trainer(training_config)
            self.logger.info("Trainer initialized successfully")
            
            # Start training
            self.logger.info("Starting training...")
            trainer.sft_trainer.train()
            self.logger.info("Training completed successfully")

            # Save model
            self.logger.info(f"Saving model to {training_config.output_dir}")
            trainer.sft_trainer.save_model(training_config.output_dir)
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.exception("Full traceback:")
            sys.exit(1)
            
    def run(self, args):
        """Main execution method."""
        self.logger.info("=== Cargia Training CLI Started ===")
        
        # Load configuration
        config = self.load_config(args.config)
        
        # Validate configuration
        if not self.validate_config(config):
            self.logger.error("Configuration validation failed")
            sys.exit(1)
            
        # Determine environment
        environment = 'cloud' if args.cloud else 'local'
        self.logger.info(f"Running in {environment} environment")
        
        # Execute training
        self.run_training(config, environment)
        
        self.logger.info("=== Cargia Training CLI Completed ===")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cargia Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local testing
  python train_cli.py --config configs/step_1_overfit_single.yaml --local
  
  # Cloud deployment
  python train_cli.py --config configs/step_1_overfit_single.yaml --cloud
  
  # With custom model path
  python train_cli.py --config configs/step_1_overfit_single.yaml --cloud --model-path /custom/path
        """
    )
    
    parser.add_argument(
        '--config', 
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local environment paths'
    )
    
    parser.add_argument(
        '--cloud',
        action='store_true',
        help='Use cloud environment paths'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running training'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.local and args.cloud:
        print("Error: Cannot specify both --local and --cloud")
        sys.exit(1)
        
    if not args.local and not args.cloud:
        print("Error: Must specify either --local or --cloud")
        sys.exit(1)
        
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create and run CLI
    cli = TrainingCLI()
    
    if args.validate_only:
        # Only validate configuration
        config = cli.load_config(args.config)
        if cli.validate_config(config):
            print("Configuration validation passed!")
            sys.exit(0)
        else:
            print("Configuration validation failed!")
            sys.exit(1)
    else:
        # Run training
        cli.run(args)


if __name__ == "__main__":
    main() 