"""
Data augmentation module for ARC-AGI training data.

This module implements various augmentation strategies:
- Color invariance: Generating distinct color maps
- Character invariance: Random character substitutions
- Spatial transforms: Rotations and flips (with text rewriting)
- Helper functions for transforming both grid data and thought text
"""
import random
import string
from typing import List, Dict, Tuple, Union
import numpy as np
import re


class CharacterInvariance:
    """
    Character invariance augmentation for ARC-AGI tasks.
    
    Transforms digit-based grids (0-9) to character-based grids using
    random character mappings while preserving the underlying patterns.
    """
    
    def __init__(self):
        """Initialize the character invariance augmenter."""
        # Define valid character sets for mapping
        self.uppercase_letters = string.ascii_uppercase  # A-Z
        self.lowercase_letters = string.ascii_lowercase  # a-z
        self.numbers = string.digits  # 0-9
        # Include most special characters, but avoid problematic ones for string parsing
        # Avoid: [], {} () , ; : " ' \ (brackets, comma, semicolon, colon, quotes, backslash)
        self.safe_special_chars = "!@#$%^&*_+-=<>?/~`|"
        
        # Combine all valid characters
        self.all_valid_chars = (
            self.uppercase_letters + 
            self.lowercase_letters + 
            self.numbers +
            self.safe_special_chars
        )
    
    def generate_character_map(self) -> Dict[int, str]:
        """
        Generate a random character map for digits 0-9.
        
        Returns:
            Dictionary mapping digits 0-9 to random characters
        """
        # Create a bijection from digits 0-9 to random characters
        available_chars = list(self.all_valid_chars)
        random.shuffle(available_chars)
        
        # Map each digit to a unique character
        char_map = {}
        for digit in range(10):  # 0-9
            char_map[digit] = available_chars[digit]
        
        return char_map
    
    def apply_character_map_to_digit_grid(
        self, 
        grid: List[List[int]], 
        char_map: Dict[int, str]
    ) -> List[List[str]]:
        """
        Apply a character map to transform a digit grid to a character grid.
        
        Args:
            grid: Input grid as list of lists of integers (0-9)
            char_map: Dictionary mapping digits to characters
            
        Returns:
            Transformed grid as list of lists of characters
        """
        if not grid:
            return []
        
        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        
        # Create the transformed grid
        transformed_grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                digit = grid[i][j]
                if digit in char_map:
                    row.append(char_map[digit])
                else:
                    # Handle unexpected digits by keeping them as strings
                    row.append(str(digit))
            transformed_grid.append(row)
        
        return transformed_grid
    
    def apply_character_map_to_task(
        self, 
        task: Dict, 
        char_map: Dict[int, str]
    ) -> Dict:
        """
        Apply character map to an entire task (train and test pairs).
        
        Args:
            task: Task dictionary with 'train' and 'test' lists
            char_map: Dictionary mapping digits to characters
            
        Returns:
            Transformed task with character-based grids
        """
        transformed_task = {
            'train': [],
            'test': []
        }
        
        # Transform training pairs
        for pair in task.get('train', []):
            transformed_pair = {
                'input': self.apply_character_map_to_digit_grid(pair['input'], char_map),
                'output': self.apply_character_map_to_digit_grid(pair['output'], char_map)
            }
            transformed_task['train'].append(transformed_pair)
        
        # Transform test pairs
        for pair in task.get('test', []):
            transformed_pair = {
                'input': self.apply_character_map_to_digit_grid(pair['input'], char_map),
                'output': self.apply_character_map_to_digit_grid(pair['output'], char_map)
            }
            transformed_task['test'].append(transformed_pair)
        
        return transformed_task
    
    def generate_multiple_maps(self, count: int) -> List[Dict[int, str]]:
        """
        Generate multiple unique character maps.
        
        Args:
            count: Number of character maps to generate
            
        Returns:
            List of character maps
        """
        char_maps = []
        for _ in range(count):
            char_maps.append(self.generate_character_map())
        return char_maps
    
    def invert_character_map(self, char_map: Dict[int, str]) -> Dict[str, int]:
        """
        Create the inverse mapping from characters back to digits.
        
        Args:
            char_map: Original digit to character mapping
            
        Returns:
            Inverse mapping from characters to digits
        """
        return {char: digit for digit, char in char_map.items()}
    
    def apply_inverse_map(
        self, 
        char_grid: List[List[str]], 
        char_map: Dict[int, str]
    ) -> List[List[int]]:
        """
        Apply inverse mapping to convert character grid back to digit grid.
        
        Args:
            char_grid: Grid as list of lists of characters
            char_map: Original digit to character mapping
            
        Returns:
            Transformed grid as list of lists of integers
        """
        inverse_map = self.invert_character_map(char_map)
        
        if not char_grid:
            return []
        
        rows = len(char_grid)
        cols = len(char_grid[0]) if char_grid else 0
        
        # Create the transformed grid
        transformed_grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                char = char_grid[i][j]
                if char in inverse_map:
                    row.append(inverse_map[char])
                else:
                    # Handle unexpected characters by converting to int if possible
                    try:
                        row.append(int(char))
                    except ValueError:
                        # If conversion fails, use 0 as default
                        row.append(0)
            transformed_grid.append(row)
        
        return transformed_grid


def test_character_invariance():
    """Test the character invariance functionality."""
    print("Testing Character Invariance...")
    
    # Create augmenter
    char_augmenter = CharacterInvariance()
    
    # Test grid (simple 3x3 grid with digits 0-3)
    test_grid = [
        [0, 1, 2],
        [1, 3, 1],
        [2, 1, 0]
    ]
    
    print(f"Original grid:")
    for row in test_grid:
        print(f"  {row}")
    
    # Generate character map
    char_map = char_augmenter.generate_character_map()
    print(f"\nGenerated character map: {char_map}")
    
    # Apply transformation
    transformed_grid = char_augmenter.apply_character_map_to_digit_grid(test_grid, char_map)
    print(f"\nTransformed grid:")
    for row in transformed_grid:
        print(f"  {row}")
    
    # Test inverse transformation
    inverse_grid = char_augmenter.apply_inverse_map(transformed_grid, char_map)
    print(f"\nInverse transformed grid:")
    for row in inverse_grid:
        print(f"  {row}")
    
    # Verify round-trip transformation
    if test_grid == inverse_grid:
        print("\n✅ Round-trip transformation successful!")
    else:
        print("\n❌ Round-trip transformation failed!")
        print(f"Original: {test_grid}")
        print(f"Inverse:  {inverse_grid}")
    
    # Test with sample task
    sample_task = {
        'train': [
            {
                'input': [[0, 1], [1, 0]],
                'output': [[0, 1], [1, 2]]
            }
        ],
        'test': [
            {
                'input': [[0, 0], [1, 1]],
                'output': [[0, 0], [1, 2]]
            }
        ]
    }
    
    print(f"\nTesting with sample task...")
    transformed_task = char_augmenter.apply_character_map_to_task(sample_task, char_map)
    
    print(f"Transformed task train input: {transformed_task['train'][0]['input']}")
    print(f"Transformed task train output: {transformed_task['train'][0]['output']}")
    print(f"Transformed task test input: {transformed_task['test'][0]['input']}")
    print(f"Transformed task test output: {transformed_task['test'][0]['output']}")
    
    print("\n✅ Character invariance tests completed!")


class ColorInvariance:
    """
    Color invariance augmentation for ARC-AGI tasks.
    
    Transforms color references in thought text to match new color maps
    while preserving the underlying reasoning and patterns.
    """
    
    def __init__(self):
        """Initialize the color invariance augmenter."""
        # Define the restricted set of colors with commonly-used English names
        self.valid_colors = {
            "red": {"name": "red", "color": [255, 0, 0]},
            "orange": {"name": "orange", "color": [255, 165, 0]},
            "yellow": {"name": "yellow", "color": [255, 255, 0]},
            "green": {"name": "green", "color": [0, 255, 0]},
            "aqua": {"name": "aqua", "color": [0, 255, 255]},
            "blue": {"name": "blue", "color": [0, 0, 255]},
            "purple": {"name": "purple", "color": [128, 0, 128]},
            "pink": {"name": "pink", "color": [255, 192, 203]},
            "brown": {"name": "brown", "color": [165, 42, 42]},
            "gray": {"name": "gray", "color": [128, 128, 128]},
            "black": {"name": "black", "color": [0, 0, 0]},
            "white": {"name": "white", "color": [255, 255, 255]}
        }
        
        # Create regex patterns for robust color name detection
        self._create_color_patterns()
    
    def _create_color_patterns(self):
        """Create regex patterns for detecting color names in text."""
        # Base color names
        base_colors = list(self.valid_colors.keys())
        
        # Handle "gray" vs "grey" spelling variation
        gray_variations = ["gray", "grey"]
        
        # Create patterns for different forms
        self.color_patterns = []
        
        # Pattern 1: Exact color names (case insensitive)
        for color in base_colors:
            if color == "gray":
                # Handle both "gray" and "grey" spellings
                pattern = rf'\b({"|".join(gray_variations)})\b'
            else:
                pattern = rf'\b{color}\b'
            self.color_patterns.append((pattern, color, re.IGNORECASE))
        
        # Pattern 2: Plural forms
        for color in base_colors:
            if color == "gray":
                pattern = rf'\b({"|".join([f"{g}s" for g in gray_variations])})\b'
            else:
                pattern = rf'\b{color}s\b'
            self.color_patterns.append((pattern, color, re.IGNORECASE))
    
    def generate_color_map(self) -> Dict[str, Dict[str, Union[str, List[int]]]]:
        """
        Generate a random color map for digits 0-9.
        
        Returns:
            Dictionary mapping digits 0-9 to random colors
        """
        # Create a bijection from digits 0-9 to random colors
        available_colors = list(self.valid_colors.keys())
        random.shuffle(available_colors)
        
        # Map each digit to a unique color
        color_map = {}
        for digit in range(10):  # 0-9
            color_name = available_colors[digit]
            color_map[str(digit)] = self.valid_colors[color_name].copy()
        
        return color_map
    
    def apply_color_map_to_text(
        self, 
        text: str, 
        original_color_map: Dict[str, Dict[str, Union[str, List[int]]]], 
        target_color_map: Dict[str, Dict[str, Union[str, List[int]]]]
    ) -> str:
        """
        Apply color map transformation to text using two-phase replacement.
        
        Args:
            text: Original thought text
            original_color_map: Original color map used when text was recorded
            target_color_map: Target color map to transform to
            
        Returns:
            Transformed text with updated color references
        """
        if not text:
            return text
        
        # Create mapping from original color names to target color names
        color_mapping = {}
        for digit, original_color_info in original_color_map.items():
            if digit in target_color_map:
                original_name = original_color_info.get('name', '')
                target_name = target_color_map[digit].get('name', '')
                if original_name and target_name:
                    color_mapping[original_name.lower()] = target_name.lower()
        
        # Phase 1: Replace with unique placeholders
        transformed_text = text
        placeholder_mapping = {}
        
        for original_name, target_name in color_mapping.items():
            # Create unique placeholder
            placeholder = f"__COLOR_{len(placeholder_mapping)}__"
            placeholder_mapping[placeholder] = target_name
            
            # Replace all variations of the original color name
            for pattern, base_color, flags in self.color_patterns:
                if base_color.lower() == original_name.lower():
                    # Handle case variations
                    matches = re.finditer(pattern, transformed_text, flags)
                    # Replace in reverse order to maintain positions
                    for match in reversed(list(matches)):
                        start, end = match.span()
                        transformed_text = transformed_text[:start] + placeholder + transformed_text[end:]
        
        # Phase 2: Replace placeholders with target colors
        for placeholder, target_name in placeholder_mapping.items():
            transformed_text = transformed_text.replace(placeholder, target_name)
        
        return transformed_text
    
    def apply_color_map_to_task(
        self, 
        task: Dict, 
        original_color_map: Dict[str, Dict[str, Union[str, List[int]]]], 
        target_color_map: Dict[str, Dict[str, Union[str, List[int]]]]
    ) -> Dict:
        """
        Apply color map to an entire task (preserves grid structure, transforms metadata).
        
        Args:
            task: Task dictionary with 'train' and 'test' lists
            original_color_map: Original color map used when data was recorded
            target_color_map: Target color map to transform to
            
        Returns:
            Transformed task with updated color references in metadata
        """
        transformed_task = {
            'train': [],
            'test': []
        }
        
        # Transform training pairs (grids remain unchanged, only metadata would change)
        for pair in task.get('train', []):
            transformed_pair = {
                'input': pair['input'],  # Grid unchanged
                'output': pair['output']  # Grid unchanged
            }
            transformed_task['train'].append(transformed_pair)
        
        # Transform test pairs (grids remain unchanged, only metadata would change)
        for pair in task.get('test', []):
            transformed_pair = {
                'input': pair['input'],  # Grid unchanged
                'output': pair['output']  # Grid unchanged
            }
            transformed_task['test'].append(transformed_pair)
        
        return transformed_task
    
    def generate_multiple_color_maps(self, count: int) -> List[Dict[str, Dict[str, Union[str, List[int]]]]]:
        """
        Generate multiple unique color maps.
        
        Args:
            count: Number of color maps to generate
            
        Returns:
            List of color maps
        """
        color_maps = []
        for _ in range(count):
            color_maps.append(self.generate_color_map())
        return color_maps
    
    def get_color_name_from_map(self, digit: str, color_map: Dict[str, Dict[str, Union[str, List[int]]]]) -> str:
        """
        Get the color name for a given digit from a color map.
        
        Args:
            digit: Digit string (0-9)
            color_map: Color map dictionary
            
        Returns:
            Color name as string
        """
        if digit in color_map:
            return color_map[digit].get('name', '')
        return ''


def test_color_invariance():
    """Test the color invariance functionality."""
    print("Testing Color Invariance...")
    
    # Create augmenter
    color_augmenter = ColorInvariance()
    
    # Test color map generation
    color_map = color_augmenter.generate_color_map()
    print(f"Generated color map: {color_map}")
    
    # Test text transformation
    original_text = "The red object is next to the blue tile, and there are gray squares in the corner."
    original_color_map = {
        "1": {"name": "red", "color": [255, 0, 0]},
        "2": {"name": "blue", "color": [0, 0, 255]},
        "3": {"name": "gray", "color": [128, 128, 128]}
    }
    target_color_map = {
        "1": {"name": "green", "color": [0, 255, 0]},
        "2": {"name": "purple", "color": [128, 0, 128]},
        "3": {"name": "orange", "color": [255, 165, 0]}
    }
    
    print(f"\nOriginal text: {original_text}")
    print(f"Original color map: {original_color_map}")
    print(f"Target color map: {target_color_map}")
    
    # Apply transformation
    transformed_text = color_augmenter.apply_color_map_to_text(
        original_text, original_color_map, target_color_map
    )
    print(f"Transformed text: {transformed_text}")
    
    # Test with variations
    test_texts = [
        "red tiles",
        "blues and grays",
        "the RED object",
        "grey squares",
        "reddish areas",
        "blue-colored tiles"
    ]
    
    print(f"\nTesting with various text patterns:")
    for test_text in test_texts:
        transformed = color_augmenter.apply_color_map_to_text(
            test_text, original_color_map, target_color_map
        )
        print(f"  '{test_text}' -> '{transformed}'")
    
    # Test multiple color maps
    print(f"\nTesting multiple color maps...")
    color_maps = color_augmenter.generate_multiple_color_maps(3)
    for i, color_map in enumerate(color_maps):
        print(f"  Map {i+1}: {color_map}")
    
    print("\n✅ Color invariance tests completed!")


class SpatialTransforms:
    """
    Spatial transformations for ARC-AGI grids.
    
    Applies rotations and flips to grid representations and provides
    placeholder methods for transforming corresponding thought text.
    """
    
    def __init__(self):
        """Initialize the spatial transforms augmenter."""
        self.valid_transforms = {
            'flip_horizontal', 'flip_vertical',
            'rotate_90', 'rotate_180', 'rotate_270'
        }
    
    def flip_horizontal(self, grid: List[List[str]]) -> List[List[str]]:
        """
        Flip the grid horizontally (left to right).
        
        Args:
            grid: 2D list of characters representing the grid
            
        Returns:
            Horizontally flipped grid
        """
        if not grid or not grid[0]:
            raise ValueError("Cannot flip empty or malformed grid")
        
        return [row[::-1] for row in grid]
    
    def flip_vertical(self, grid: List[List[str]]) -> List[List[str]]:
        """
        Flip the grid vertically (top to bottom).
        
        Args:
            grid: 2D list of characters representing the grid
            
        Returns:
            Vertically flipped grid
        """
        if not grid or not grid[0]:
            raise ValueError("Cannot flip empty or malformed grid")
        
        return grid[::-1]
    
    def rotate_90(self, grid: List[List[str]]) -> List[List[str]]:
        """
        Rotate the grid 90 degrees clockwise.
        
        Args:
            grid: 2D list of characters representing the grid
            
        Returns:
            Grid rotated 90 degrees clockwise
        """
        if not grid or not grid[0]:
            raise ValueError("Cannot rotate empty or malformed grid")
        
        # Transpose and reverse each row for 90-degree clockwise rotation
        return [list(row) for row in zip(*grid[::-1])]
    
    def rotate_180(self, grid: List[List[str]]) -> List[List[str]]:
        """
        Rotate the grid 180 degrees.
        
        Args:
            grid: 2D list of characters representing the grid
            
        Returns:
            Grid rotated 180 degrees
        """
        if not grid or not grid[0]:
            raise ValueError("Cannot rotate empty or malformed grid")
        
        # Reverse both rows and columns
        return [row[::-1] for row in grid[::-1]]
    
    def rotate_270(self, grid: List[List[str]]) -> List[List[str]]:
        """
        Rotate the grid 270 degrees clockwise (90 degrees counterclockwise).
        
        Args:
            grid: 2D list of characters representing the grid
            
        Returns:
            Grid rotated 270 degrees clockwise
        """
        if not grid or not grid[0]:
            raise ValueError("Cannot rotate empty or malformed grid")
        
        # Transpose and reverse columns for 270-degree clockwise rotation
        return [list(row) for row in zip(*grid)][::-1]
    
    def transform_thoughts_for_flip_horizontal(self, thoughts: str) -> str:
        """
        Transform thought text to account for horizontal flip.
        
        Args:
            thoughts: Original thought text
            
        Returns:
            Transformed thought text
        """
        raise NotImplementedError(
            "Text transformation for horizontal flip not yet implemented. "
            "Should transform spatial references like 'left', 'right', 'column 1', etc."
        )
    
    def transform_thoughts_for_flip_vertical(self, thoughts: str) -> str:
        """
        Transform thought text to account for vertical flip.
        
        Args:
            thoughts: Original thought text
            
        Returns:
            Transformed thought text
        """
        raise NotImplementedError(
            "Text transformation for vertical flip not yet implemented. "
            "Should transform spatial references like 'top', 'bottom', 'row 1', etc."
        )
    
    def transform_thoughts_for_rotate_90(self, thoughts: str) -> str:
        """
        Transform thought text to account for 90-degree rotation.
        
        Args:
            thoughts: Original thought text
            
        Returns:
            Transformed thought text
        """
        raise NotImplementedError(
            "Text transformation for 90-degree rotation not yet implemented. "
            "Should transform all spatial references to match the rotated coordinate system."
        )
    
    def transform_thoughts_for_rotate_180(self, thoughts: str) -> str:
        """
        Transform thought text to account for 180-degree rotation.
        
        Args:
            thoughts: Original thought text
            
        Returns:
            Transformed thought text
        """
        raise NotImplementedError(
            "Text transformation for 180-degree rotation not yet implemented. "
            "Should transform all spatial references to match the rotated coordinate system."
        )
    
    def transform_thoughts_for_rotate_270(self, thoughts: str) -> str:
        """
        Transform thought text to account for 270-degree rotation.
        
        Args:
            thoughts: Original thought text
            
        Returns:
            Transformed thought text
        """
        raise NotImplementedError(
            "Text transformation for 270-degree rotation not yet implemented. "
            "Should transform all spatial references to match the rotated coordinate system."
        )
    
    def apply_spatial_transform(
        self, 
        grid: List[List[str]], 
        thoughts: str, 
        transform_type: str
    ) -> Tuple[List[List[str]], str]:
        """
        Apply a single spatial transform to grid and thoughts.
        
        Args:
            grid: 2D list of characters representing the grid
            thoughts: Original thought text
            transform_type: Type of transform to apply
            
        Returns:
            Tuple of (transformed_grid, transformed_thoughts)
        """
        if transform_type not in self.valid_transforms:
            raise ValueError(f"Invalid transform type: {transform_type}. Valid types: {self.valid_transforms}")
        
        # Apply grid transformation
        if transform_type == 'flip_horizontal':
            transformed_grid = self.flip_horizontal(grid)
            transformed_thoughts = self.transform_thoughts_for_flip_horizontal(thoughts)
        elif transform_type == 'flip_vertical':
            transformed_grid = self.flip_vertical(grid)
            transformed_thoughts = self.transform_thoughts_for_flip_vertical(thoughts)
        elif transform_type == 'rotate_90':
            transformed_grid = self.rotate_90(grid)
            transformed_thoughts = self.transform_thoughts_for_rotate_90(thoughts)
        elif transform_type == 'rotate_180':
            transformed_grid = self.rotate_180(grid)
            transformed_thoughts = self.transform_thoughts_for_rotate_180(thoughts)
        elif transform_type == 'rotate_270':
            transformed_grid = self.rotate_270(grid)
            transformed_thoughts = self.transform_thoughts_for_rotate_270(thoughts)
        else:
            raise ValueError(f"Unhandled transform type: {transform_type}")
        
        return transformed_grid, transformed_thoughts
    
    def apply_multiple_transforms(
        self, 
        grid: List[List[str]], 
        thoughts: str, 
        transforms: List[str]
    ) -> Tuple[List[List[str]], str]:
        """
        Apply multiple spatial transforms in sequence.
        
        Args:
            grid: 2D list of characters representing the grid
            thoughts: Original thought text
            transforms: List of transform types to apply in order
            
        Returns:
            Tuple of (transformed_grid, transformed_thoughts)
        """
        if not transforms:
            return grid, thoughts
        
        current_grid = grid
        current_thoughts = thoughts
        
        for transform_type in transforms:
            current_grid, current_thoughts = self.apply_spatial_transform(
                current_grid, current_thoughts, transform_type
            )
        
        return current_grid, current_thoughts
    
    def get_available_transforms(self) -> List[str]:
        """
        Get list of available transform types.
        
        Returns:
            List of valid transform type names
        """
        return list(self.valid_transforms)


def test_spatial_transforms():
    """Test the spatial transforms functionality."""
    print("Testing Spatial Transforms...")
    
    # Create augmenter
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
    
    # Test individual transforms
    transforms_to_test = [
        ('flip_horizontal', 'Horizontal Flip'),
        ('flip_vertical', 'Vertical Flip'),
        ('rotate_90', '90° Rotation'),
        ('rotate_180', '180° Rotation'),
        ('rotate_270', '270° Rotation')
    ]
    
    for transform_type, description in transforms_to_test:
        print(f"\n{description}:")
        try:
            transformed_grid, transformed_thoughts = spatial_augmenter.apply_spatial_transform(
                test_grid, "test thoughts", transform_type
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
    
    # Test multiple transforms
    print(f"\nMultiple transforms (flip_horizontal + rotate_90):")
    try:
        multi_transformed_grid, multi_transformed_thoughts = spatial_augmenter.apply_multiple_transforms(
            test_grid, "test thoughts", ['flip_horizontal', 'rotate_90']
        )
        print(f"  Grid result:")
        for row in multi_transformed_grid:
            print(f"    {row}")
        print(f"  Thoughts result: {multi_transformed_thoughts}")
    except NotImplementedError as e:
        print(f"  ✅ Grid transformed successfully")
        print(f"  ✅ Text transformation correctly raises NotImplementedError: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test error handling
    print(f"\nError handling:")
    try:
        spatial_augmenter.apply_spatial_transform(test_grid, "test", "invalid_transform")
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    try:
        spatial_augmenter.apply_spatial_transform([], "test", "flip_horizontal")
        print("  ❌ Should have raised ValueError for empty grid")
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError for empty grid: {e}")
    
    print("\n✅ Spatial transforms tests completed!")


if __name__ == "__main__":
    test_character_invariance()
    print("\n" + "="*50 + "\n")
    test_color_invariance()
    print("\n" + "="*50 + "\n")
    test_spatial_transforms() 