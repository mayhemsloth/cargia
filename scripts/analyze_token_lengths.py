#!/usr/bin/env python3
"""
Script to analyze token lengths in the Cargia dataset.
Run this from the project root directory.
"""

import sys
import os

from cargia.training.trainer import CargiaGoogleGemma3Trainer, TRAINING_CONFIG

def main():
    """
    Main function to run the token length analysis.
    """
    print("===== Token Length Analysis Started =====")
    
    # Initialize trainer (this will load the model and datasets)
    print("Initializing trainer...")
    trainer = CargiaGoogleGemma3Trainer(TRAINING_CONFIG)
    
    # Analyze with a sample first (for quick testing)
    print("\n=== Quick Sample Analysis ===")
    sample_results = trainer.analyze_dataset(max_samples=10)
    
    # Ask user if they want to analyze the full dataset
    response = input("\nDo you want to analyze the full dataset? This may take a while. (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        print("\n=== Full Dataset Analysis ===")
        full_results = trainer.analyze_dataset()
        
        # Estimate memory requirements
        trainer.estimate_memory_requirements(full_results)
    else:
        print("Full dataset analysis skipped.")
        trainer.estimate_memory_requirements(sample_results)
    
    print("===== Token Length Analysis Completed =====")

if __name__ == "__main__":
    main() 