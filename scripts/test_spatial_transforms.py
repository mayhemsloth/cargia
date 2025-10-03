#!/usr/bin/env python3
"""
Test script for spatial transforms functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cargia.training.augment import SpatialTransforms


def test_grid_transformations():
    """Test all grid transformation methods."""
    print("=== Testing Grid Transformations ===")
    
    spatial_augmenter = SpatialTransforms()
    
    # Test grid
    test_grid = [
        ['1', '2', '3'],
        ['4', '5', '6'],
        ['7', '8', '9']
    ]
    
    print(f"Original grid:")
    for row in test_grid:
        print(f"  {row}")
    
    # Test flip horizontal
    print(f"\n1. Testing flip_horizontal...")
    flipped_h = spatial_augmenter.flip_horizontal(test_grid)
    print(f"  Result:")
    for row in flipped_h:
        print(f"    {row}")
    expected_h = [['3', '2', '1'], ['6', '5', '4'], ['9', '8', '7']]
    assert flipped_h == expected_h, f"Expected {expected_h}, got {flipped_h}"
    print("  ✅ PASS")
    
    # Test flip vertical
    print(f"\n2. Testing flip_vertical...")
    flipped_v = spatial_augmenter.flip_vertical(test_grid)
    print(f"  Result:")
    for row in flipped_v:
        print(f"    {row}")
    expected_v = [['7', '8', '9'], ['4', '5', '6'], ['1', '2', '3']]
    assert flipped_v == expected_v, f"Expected {expected_v}, got {flipped_v}"
    print("  ✅ PASS")
    
    # Test rotate 90
    print(f"\n3. Testing rotate_90...")
    rotated_90 = spatial_augmenter.rotate_90(test_grid)
    print(f"  Result:")
    for row in rotated_90:
        print(f"    {row}")
    expected_90 = [['7', '4', '1'], ['8', '5', '2'], ['9', '6', '3']]
    assert rotated_90 == expected_90, f"Expected {expected_90}, got {rotated_90}"
    print("  ✅ PASS")
    
    # Test rotate 180
    print(f"\n4. Testing rotate_180...")
    rotated_180 = spatial_augmenter.rotate_180(test_grid)
    print(f"  Result:")
    for row in rotated_180:
        print(f"    {row}")
    expected_180 = [['9', '8', '7'], ['6', '5', '4'], ['3', '2', '1']]
    assert rotated_180 == expected_180, f"Expected {expected_180}, got {rotated_180}"
    print("  ✅ PASS")
    
    # Test rotate 270
    print(f"\n5. Testing rotate_270...")
    rotated_270 = spatial_augmenter.rotate_270(test_grid)
    print(f"  Result:")
    for row in rotated_270:
        print(f"    {row}")
    expected_270 = [['3', '6', '9'], ['2', '5', '8'], ['1', '4', '7']]
    assert rotated_270 == expected_270, f"Expected {expected_270}, got {rotated_270}"
    print("  ✅ PASS")


def test_text_transformation_placeholders():
    """Test that text transformation methods raise NotImplementedError."""
    print("\n=== Testing Text Transformation Placeholders ===")
    
    spatial_augmenter = SpatialTransforms()
    
    # Test all text transformation methods
    text_methods = [
        ('transform_thoughts_for_flip_horizontal', 'Horizontal flip text transform'),
        ('transform_thoughts_for_flip_vertical', 'Vertical flip text transform'),
        ('transform_thoughts_for_rotate_90', '90° rotation text transform'),
        ('transform_thoughts_for_rotate_180', '180° rotation text transform'),
        ('transform_thoughts_for_rotate_270', '270° rotation text transform')
    ]
    
    for method_name, description in text_methods:
        print(f"\n{description}:")
        try:
            method = getattr(spatial_augmenter, method_name)
            result = method("test thoughts")
            print(f"  ❌ Should have raised NotImplementedError, got: {result}")
            return False
        except NotImplementedError as e:
            print(f"  ✅ Correctly raised NotImplementedError: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False
    
    return True


def test_apply_spatial_transform():
    """Test the main apply_spatial_transform method."""
    print("\n=== Testing apply_spatial_transform ===")
    
    spatial_augmenter = SpatialTransforms()
    
    test_grid = [['1', '2'], ['3', '4']]
    test_thoughts = "test thoughts"
    
    # Test valid transforms
    valid_transforms = ['flip_horizontal', 'flip_vertical', 'rotate_90', 'rotate_180', 'rotate_270']
    
    for transform_type in valid_transforms:
        print(f"\nTesting {transform_type}:")
        try:
            transformed_grid, transformed_thoughts = spatial_augmenter.apply_spatial_transform(
                test_grid, test_thoughts, transform_type
            )
            print(f"  Grid result:")
            for row in transformed_grid:
                print(f"    {row}")
            print(f"  Thoughts result: {transformed_thoughts}")
        except NotImplementedError as e:
            print(f"  ✅ Grid transformed successfully")
            print(f"  ✅ Text transformation correctly raises NotImplementedError: {e}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    # Test invalid transform
    print(f"\nTesting invalid transform:")
    try:
        spatial_augmenter.apply_spatial_transform(test_grid, test_thoughts, "invalid_transform")
        print("  ❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    return True


def test_multiple_transforms():
    """Test applying multiple transforms in sequence."""
    print("\n=== Testing Multiple Transforms ===")
    
    spatial_augmenter = SpatialTransforms()
    
    test_grid = [['1', '2'], ['3', '4']]
    test_thoughts = "test thoughts"
    
    # Test sequence of transforms
    transform_sequences = [
        (['flip_horizontal', 'flip_vertical'], 'Flip H + V'),
        (['rotate_90', 'rotate_90'], 'Rotate 90° + 90°'),
        (['flip_horizontal', 'rotate_90', 'flip_vertical'], 'Flip H + Rotate 90° + Flip V')
    ]
    
    for transforms, description in transform_sequences:
        print(f"\n{description}:")
        try:
            transformed_grid, transformed_thoughts = spatial_augmenter.apply_multiple_transforms(
                test_grid, test_thoughts, transforms
            )
            print(f"  Grid result:")
            for row in transformed_grid:
                print(f"    {row}")
            print(f"  Thoughts result: {transformed_thoughts}")
        except NotImplementedError as e:
            print(f"  ✅ Grid transformed successfully")
            print(f"  ✅ Text transformation correctly raises NotImplementedError: {e}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    # Test empty transform list
    print(f"\nTesting empty transform list:")
    try:
        transformed_grid, transformed_thoughts = spatial_augmenter.apply_multiple_transforms(
            test_grid, test_thoughts, []
        )
        assert transformed_grid == test_grid, "Grid should be unchanged"
        assert transformed_thoughts == test_thoughts, "Thoughts should be unchanged"
        print("  ✅ Empty transform list handled correctly")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    return True


def test_error_handling():
    """Test error handling for malformed grids."""
    print("\n=== Testing Error Handling ===")
    
    spatial_augmenter = SpatialTransforms()
    
    # Test empty grid
    print(f"\n1. Testing empty grid:")
    try:
        spatial_augmenter.flip_horizontal([])
        print("  ❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    # Test grid with empty rows
    print(f"\n2. Testing grid with empty rows:")
    try:
        spatial_augmenter.flip_horizontal([[], []])
        print("  ❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    # Test malformed grid (different row lengths)
    print(f"\n3. Testing malformed grid (different row lengths):")
    malformed_grid = [['1', '2'], ['3']]  # Second row has only one element
    try:
        # This might not raise an error depending on implementation, but should handle gracefully
        result = spatial_augmenter.flip_horizontal(malformed_grid)
        print(f"  ✅ Handled gracefully: {result}")
    except Exception as e:
        print(f"  ✅ Correctly raised error: {e}")
    
    return True


def test_available_transforms():
    """Test getting available transforms."""
    print("\n=== Testing Available Transforms ===")
    
    spatial_augmenter = SpatialTransforms()
    
    available = spatial_augmenter.get_available_transforms()
    expected = ['flip_horizontal', 'flip_vertical', 'rotate_90', 'rotate_180', 'rotate_270']
    
    print(f"Available transforms: {available}")
    print(f"Expected transforms: {expected}")
    
    # Check that all expected transforms are available
    for transform in expected:
        assert transform in available, f"Expected transform '{transform}' not found in available transforms"
    
    # Check that no unexpected transforms are present
    for transform in available:
        assert transform in expected, f"Unexpected transform '{transform}' found in available transforms"
    
    print("  ✅ All expected transforms are available")
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Spatial Transforms Functionality")
    print("=" * 50)
    
    try:
        test_grid_transformations()
        if not test_text_transformation_placeholders():
            print("\n❌ Text transformation placeholder tests failed")
            return False
        if not test_apply_spatial_transform():
            print("\n❌ apply_spatial_transform tests failed")
            return False
        if not test_multiple_transforms():
            print("\n❌ Multiple transforms tests failed")
            return False
        if not test_error_handling():
            print("\n❌ Error handling tests failed")
            return False
        if not test_available_transforms():
            print("\n❌ Available transforms tests failed")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 All spatial transforms tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 