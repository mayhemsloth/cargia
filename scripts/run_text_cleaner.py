#!/usr/bin/env python3
"""
CLI script to run the text cleaner functionality.
"""
import os
import sys
import json
import argparse
from cargia.data_manager import get_repo_root, DataManager

def main():
    """Main function for the text cleaner CLI."""
    parser = argparse.ArgumentParser(description="Run text cleaner on thoughts database")
    parser.add_argument("--data-dir", help="Data directory path (default: from settings)")
    parser.add_argument("--source-folder", help="Source folder path (default: from settings)")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics, don't clean")
    parser.add_argument("--force", action="store_true", help="Force cleaning without confirmation")
    
    args = parser.parse_args()
    
    try:
        # Get cargia directory
        cargia_dir = get_repo_root()
        
        # Load settings
        settings_path = os.path.join(cargia_dir, "settings.json")
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
        
        # Use provided data_dir or default from settings
        data_dir = args.data_dir or settings.get('data_dir', os.path.join(cargia_dir, "data"))
        source_folder = args.source_folder or settings.get('source_folder', os.path.join(cargia_dir, "data"))
        
        print(f"Using data directory: {data_dir}")
        print(f"Using source folder: {source_folder}")
        
        # Initialize data manager
        data_manager = DataManager(data_dir, source_folder)
        
        # Get cleaning statistics
        stats = data_manager.get_text_cleaning_stats()
        print(f"\nCleaning Statistics:")
        print(f"  Total thoughts with text: {stats['total_thoughts']}")
        print(f"  Already cleaned: {stats['cleaned_thoughts']}")
        print(f"  Need cleaning: {stats['uncleaned_thoughts']}")
        
        if args.stats_only:
            return
        
        if stats['uncleaned_thoughts'] == 0:
            print("\nNo thoughts need cleaning!")
            return
        
        # Ask for confirmation unless --force is used
        if not args.force:
            response = input(f"\nDo you want to clean {stats['uncleaned_thoughts']} thoughts? (y/n): ").lower().strip()
            if response != 'y':
                print("Cleaning cancelled.")
                return
        
        # Run text cleaner
        print("\nStarting text cleaning...")
        results = data_manager.run_text_cleaner()
        
        print(f"\nCleaning completed!")
        print(f"  Total thoughts processed: {results['total_thoughts']}")
        print(f"  Successfully cleaned: {results['cleaned_thoughts']}")
        print(f"  Failed to clean: {results['failed_thoughts']}")
        
        # Show updated statistics
        updated_stats = data_manager.get_text_cleaning_stats()
        print(f"\nUpdated Cleaning Statistics:")
        print(f"  Total thoughts with text: {updated_stats['total_thoughts']}")
        print(f"  Already cleaned: {updated_stats['cleaned_thoughts']}")
        print(f"  Need cleaning: {updated_stats['uncleaned_thoughts']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 