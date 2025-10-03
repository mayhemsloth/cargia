#!/usr/bin/env python3
"""
Test script to verify the text comparison functionality.
"""
import sys
from cargia.data_manager import get_repo_root, DataManager

def test_text_comparison():
    """Test the text comparison functionality."""
    try:
        print("Testing text comparison functionality...")
        
        # Get cargia directory
        cargia_dir = get_repo_root()
        
        # Initialize data manager
        data_manager = DataManager("data", "data")
        
        # Test getting thoughts for comparison
        thoughts = data_manager.get_all_thoughts_for_comparison()
        print(f"Found {len(thoughts)} thoughts for comparison")
        
        if thoughts:
            # Test getting a specific thought
            first_thought = data_manager.get_thought_by_id(thoughts[0]['id'])
            if first_thought:
                print(f"Successfully retrieved thought ID {first_thought['id']}")
                print(f"  Original text length: {len(first_thought['thought_text'] or '')}")
                print(f"  Cleaned text length: {len(first_thought['cleaned_thought_text'] or '')}")
                print(f"  Has cleaned text: {bool(first_thought['cleaned_thought_text'])}")
            else:
                print("Failed to retrieve thought by ID")
        
        # Test cleaning stats
        stats = data_manager.get_text_cleaning_stats()
        print(f"\nCleaning statistics:")
        print(f"  Total thoughts with text: {stats['total_thoughts']}")
        print(f"  Already cleaned: {stats['cleaned_thoughts']}")
        print(f"  Need cleaning: {stats['uncleaned_thoughts']}")
        
        print("\n✅ Text comparison functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_text_comparison()
    sys.exit(0 if success else 1) 