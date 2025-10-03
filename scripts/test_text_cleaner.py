#!/usr/bin/env python3
"""
Test script for the text cleaner functionality.
"""
import os
import sys
import json
from cargia.text_cleaner import TextCleaner
from cargia.data_manager import get_repo_root

def main():
    """Main function to test text cleaning."""
    try:
        # Get cargia directory
        cargia_dir = get_repo_root()
        
        # Load settings to get data directory
        settings_path = os.path.join(cargia_dir, "settings.json")
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            data_dir = os.path.abspath(settings.get('data_dir', os.path.join(cargia_dir, "data")))
        else:
            data_dir = os.path.join(cargia_dir, "data")
        
        print(f"Using data directory: {data_dir}")
        
        # Initialize text cleaner
        cleaner = TextCleaner(data_dir)
        
        # Get cleaning statistics
        stats = cleaner.get_cleaning_stats()
        print(f"\nCleaning Statistics:")
        print(f"  Total thoughts with text: {stats['total_thoughts']}")
        print(f"  Already cleaned: {stats['cleaned_thoughts']}")
        print(f"  Need cleaning: {stats['uncleaned_thoughts']}")
        
        if stats['uncleaned_thoughts'] == 0:
            print("\nNo thoughts need cleaning!")
            return
        
        # Get thoughts that need cleaning
        thoughts_to_clean = cleaner.get_thoughts_needing_cleaning()
        print(f"\nFound {len(thoughts_to_clean)} thoughts that need cleaning.")
        
        # Show a sample of thoughts that need cleaning
        if thoughts_to_clean:
            print("\nSample thoughts that need cleaning:")
            for i, thought in enumerate(thoughts_to_clean[:3]):  # Show first 3
                print(f"\nThought {i+1} (ID: {thought['id']}):")
                print(f"  Solve ID: {thought['solve_id']}")
                print(f"  Pair Label: {thought['pair_label']}")
                print(f"  Pair Type: {thought['pair_type']}")
                print(f"  Sequence Index: {thought['sequence_index']}")
                print(f"  Raw Text: {thought['thought_text'][:200]}...")
        
        # Ask user if they want to proceed with cleaning
        response = input(f"\nDo you want to clean {len(thoughts_to_clean)} thoughts? (y/n): ").lower().strip()
        
        if response != 'y':
            print("Cleaning cancelled.")
            return
        
        # Initialize Gemma3
        print("\nInitializing Gemma3...")
        cleaner.initialize_gemma()
        print("Gemma3 initialized successfully!")
        
        # Clean all thoughts
        print("\nStarting text cleaning...")
        results = cleaner.clean_all_thoughts()
        
        print(f"\nCleaning completed!")
        print(f"  Total thoughts processed: {results['total_thoughts']}")
        print(f"  Successfully cleaned: {results['cleaned_thoughts']}")
        print(f"  Failed to clean: {results['failed_thoughts']}")
        
        # Show updated statistics
        updated_stats = cleaner.get_cleaning_stats()
        print(f"\nUpdated Cleaning Statistics:")
        print(f"  Total thoughts with text: {updated_stats['total_thoughts']}")
        print(f"  Already cleaned: {updated_stats['cleaned_thoughts']}")
        print(f"  Need cleaning: {updated_stats['uncleaned_thoughts']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 