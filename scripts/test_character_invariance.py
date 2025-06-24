#!/usr/bin/env python3
"""
Test script for character invariance functionality.
"""
import sys
import json
import os
from cargia.training.augment import CharacterInvariance

def test_character_invariance():
    """Test the character invariance functionality."""
    try:
        print("Testing Character Invariance with sample task...")
        
        # Load sample task
        sample_task_path = "sample_task.json"
        if not os.path.exists(sample_task_path):
            print(f"❌ Sample task file not found: {sample_task_path}")
            return False
        
        with open(sample_task_path, 'r') as f:
            sample_task = json.load(f)
        
        print(f"Loaded sample task with {len(sample_task['train'])} training pairs and {len(sample_task['test'])} test pairs")
        
        # Create augmenter
        char_augmenter = CharacterInvariance()
        
        # Generate character map
        char_map = char_augmenter.generate_character_map()
        print(f"\nGenerated character map: {char_map}")
        
        # Test with first training pair
        first_train_pair = sample_task['train'][0]
        print(f"\nOriginal first training pair:")
        print(f"  Input grid (first 3 rows):")
        for i, row in enumerate(first_train_pair['input'][:3]):
            print(f"    {row}")
        print(f"  Output grid (first 3 rows):")
        for i, row in enumerate(first_train_pair['output'][:3]):
            print(f"    {row}")
        
        # Apply character map to first training pair
        transformed_input = char_augmenter.apply_character_map_to_digit_grid(first_train_pair['input'], char_map)
        transformed_output = char_augmenter.apply_character_map_to_digit_grid(first_train_pair['output'], char_map)
        
        print(f"\nTransformed first training pair:")
        print(f"  Input grid (first 3 rows):")
        for i, row in enumerate(transformed_input[:3]):
            print(f"    {row}")
        print(f"  Output grid (first 3 rows):")
        for i, row in enumerate(transformed_output[:3]):
            print(f"    {row}")
        
        # Test inverse transformation
        inverse_input = char_augmenter.apply_inverse_map(transformed_input, char_map)
        inverse_output = char_augmenter.apply_inverse_map(transformed_output, char_map)
        
        # Verify round-trip transformation
        if (first_train_pair['input'] == inverse_input and 
            first_train_pair['output'] == inverse_output):
            print("\n✅ Round-trip transformation successful!")
        else:
            print("\n❌ Round-trip transformation failed!")
            return False
        
        # Test with entire task
        print(f"\nTesting with entire task...")
        transformed_task = char_augmenter.apply_character_map_to_task(sample_task, char_map)
        
        print(f"Transformed task has {len(transformed_task['train'])} training pairs and {len(transformed_task['test'])} test pairs")
        
        # Test inverse transformation on entire task
        inverse_task = {
            'train': [],
            'test': []
        }
        
        for pair in transformed_task['train']:
            inverse_pair = {
                'input': char_augmenter.apply_inverse_map(pair['input'], char_map),
                'output': char_augmenter.apply_inverse_map(pair['output'], char_map)
            }
            inverse_task['train'].append(inverse_pair)
        
        for pair in transformed_task['test']:
            inverse_pair = {
                'input': char_augmenter.apply_inverse_map(pair['input'], char_map),
                'output': char_augmenter.apply_inverse_map(pair['output'], char_map)
            }
            inverse_task['test'].append(inverse_pair)
        
        # Verify entire task round-trip
        if (sample_task['train'] == inverse_task['train'] and 
            sample_task['test'] == inverse_task['test']):
            print("✅ Entire task round-trip transformation successful!")
        else:
            print("❌ Entire task round-trip transformation failed!")
            return False
        
        # Test multiple character maps
        print(f"\nTesting multiple character maps...")
        char_maps = char_augmenter.generate_multiple_maps(3)
        print(f"Generated {len(char_maps)} character maps")
        
        for i, char_map in enumerate(char_maps):
            print(f"  Map {i+1}: {char_map}")
        
        # Test that maps are different
        map_strings = [str(sorted(char_map.items())) for char_map in char_maps]
        if len(set(map_strings)) == len(char_maps):
            print("✅ All character maps are unique!")
        else:
            print("❌ Some character maps are identical!")
            return False
        
        print("\n✅ All character invariance tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_character_invariance()
    sys.exit(0 if success else 1) 