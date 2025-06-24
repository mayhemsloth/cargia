#!/usr/bin/env python3
"""
Test script for color invariance functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cargia.training.augment import ColorInvariance
import json


def test_color_invariance_basic():
    """Test basic color invariance functionality."""
    print("=== Testing Basic Color Invariance ===")
    
    # Create augmenter
    color_augmenter = ColorInvariance()
    
    # Test color map generation
    print("\n1. Testing color map generation...")
    color_map = color_augmenter.generate_color_map()
    print(f"Generated color map: {json.dumps(color_map, indent=2)}")
    
    # Verify all digits 0-9 are mapped
    expected_digits = set(str(i) for i in range(10))
    actual_digits = set(color_map.keys())
    assert expected_digits == actual_digits, f"Missing digits: {expected_digits - actual_digits}"
    print("✅ All digits 0-9 are mapped")
    
    # Verify all colors are from valid set
    valid_color_names = set(color_augmenter.valid_colors.keys())
    mapped_color_names = set(color_map[digit]['name'] for digit in color_map)
    assert mapped_color_names.issubset(valid_color_names), f"Invalid colors: {mapped_color_names - valid_color_names}"
    print("✅ All mapped colors are valid")


def test_text_transformation():
    """Test text transformation with color maps."""
    print("\n=== Testing Text Transformation ===")
    
    color_augmenter = ColorInvariance()
    
    # Test cases
    test_cases = [
        {
            "text": "The red object is next to the blue tile.",
            "original_map": {
                "1": {"name": "red", "color": [255, 0, 0]},
                "2": {"name": "blue", "color": [0, 0, 255]}
            },
            "target_map": {
                "1": {"name": "green", "color": [0, 255, 0]},
                "2": {"name": "purple", "color": [128, 0, 128]}
            },
            "expected": "The green object is next to the purple tile."
        },
        {
            "text": "There are gray squares and grey tiles.",
            "original_map": {
                "3": {"name": "gray", "color": [128, 128, 128]}
            },
            "target_map": {
                "3": {"name": "orange", "color": [255, 165, 0]}
            },
            "expected": "There are orange squares and orange tiles."
        },
        {
            "text": "The RED and Blue objects are visible.",
            "original_map": {
                "1": {"name": "red", "color": [255, 0, 0]},
                "2": {"name": "blue", "color": [0, 0, 255]}
            },
            "target_map": {
                "1": {"name": "yellow", "color": [255, 255, 0]},
                "2": {"name": "pink", "color": [255, 192, 203]}
            },
            "expected": "The yellow and pink objects are visible."
        },
        {
            "text": "Multiple reds and blues in the pattern.",
            "original_map": {
                "1": {"name": "red", "color": [255, 0, 0]},
                "2": {"name": "blue", "color": [0, 0, 255]}
            },
            "target_map": {
                "1": {"name": "brown", "color": [165, 42, 42]},
                "2": {"name": "aqua", "color": [0, 255, 255]}
            },
            "expected": "Multiple browns and aquas in the pattern."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"  Original: '{test_case['text']}'")
        print(f"  Original map: {test_case['original_map']}")
        print(f"  Target map: {test_case['target_map']}")
        
        result = color_augmenter.apply_color_map_to_text(
            test_case['text'],
            test_case['original_map'],
            test_case['target_map']
        )
        
        print(f"  Result: '{result}'")
        print(f"  Expected: '{test_case['expected']}'")
        
        if result == test_case['expected']:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
            return False
    
    return True


def test_edge_cases():
    """Test edge cases and robustness."""
    print("\n=== Testing Edge Cases ===")
    
    color_augmenter = ColorInvariance()
    
    # Test empty text
    result = color_augmenter.apply_color_map_to_text("", {}, {})
    assert result == "", f"Empty text should return empty string, got: '{result}'"
    print("✅ Empty text handled correctly")
    
    # Test text with no color references
    text = "This text has no color references."
    result = color_augmenter.apply_color_map_to_text(text, {}, {})
    assert result == text, f"Text without colors should be unchanged, got: '{result}'"
    print("✅ Text without colors unchanged")
    
    # Test text with unknown colors (should be left unchanged)
    text = "The reddish object and blueish tile."
    original_map = {"1": {"name": "red", "color": [255, 0, 0]}}
    target_map = {"1": {"name": "green", "color": [0, 255, 0]}}
    result = color_augmenter.apply_color_map_to_text(text, original_map, target_map)
    # Should only change "red" to "green", leave "reddish" and "blueish" unchanged
    expected = "The reddish object and blueish tile."
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✅ Unknown colors left unchanged")
    
    # Test cascading replacement prevention
    text = "The red object becomes green."
    original_map = {
        "1": {"name": "red", "color": [255, 0, 0]},
        "2": {"name": "green", "color": [0, 255, 0]}
    }
    target_map = {
        "1": {"name": "green", "color": [0, 255, 0]},
        "2": {"name": "blue", "color": [0, 0, 255]}
    }
    result = color_augmenter.apply_color_map_to_text(text, original_map, target_map)
    # Should be "The green object becomes blue" (not "The green object becomes green")
    expected = "The green object becomes blue."
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✅ Cascading replacement prevented")


def test_multiple_color_maps():
    """Test generating multiple color maps."""
    print("\n=== Testing Multiple Color Maps ===")
    
    color_augmenter = ColorInvariance()
    
    # Generate multiple maps
    count = 5
    color_maps = color_augmenter.generate_multiple_color_maps(count)
    
    print(f"Generated {len(color_maps)} color maps:")
    for i, color_map in enumerate(color_maps, 1):
        print(f"  Map {i}: {color_map}")
    
    # Verify all maps are different
    map_strings = [json.dumps(color_map, sort_keys=True) for color_map in color_maps]
    unique_maps = set(map_strings)
    assert len(unique_maps) == count, f"Expected {count} unique maps, got {len(unique_maps)}"
    print("✅ All generated maps are unique")


def test_color_name_extraction():
    """Test color name extraction from maps."""
    print("\n=== Testing Color Name Extraction ===")
    
    color_augmenter = ColorInvariance()
    
    color_map = {
        "1": {"name": "red", "color": [255, 0, 0]},
        "2": {"name": "blue", "color": [0, 0, 255]},
        "3": {"name": "green", "color": [0, 255, 0]}
    }
    
    # Test valid digits
    assert color_augmenter.get_color_name_from_map("1", color_map) == "red"
    assert color_augmenter.get_color_name_from_map("2", color_map) == "blue"
    assert color_augmenter.get_color_name_from_map("3", color_map) == "green"
    
    # Test invalid digit
    assert color_augmenter.get_color_name_from_map("9", color_map) == ""
    
    print("✅ Color name extraction works correctly")


def main():
    """Run all tests."""
    print("🧪 Testing Color Invariance Functionality")
    print("=" * 50)
    
    try:
        test_color_invariance_basic()
        if not test_text_transformation():
            print("\n❌ Text transformation tests failed")
            return False
        test_edge_cases()
        test_multiple_color_maps()
        test_color_name_extraction()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 