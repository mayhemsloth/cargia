"""
Data harness for preprocessing ARC-AGI training data.

This module handles the complete preprocessing pipeline:
- Loading raw task data from databases
- Applying task-level augmentations (color, character, spatial)
- Converting grids to images using GridImageBuilder
- Formatting conversations for Gemma3 multi-turn training
- Handling training vs validation sets differently
"""
from typing import Dict, List, Optional, Tuple
import json
from PIL import Image

from cargia.training.training_config import TrainingConfig
from cargia.training.augment import CharacterInvariance, ColorInvariance, SpatialTransforms
from cargia.common.grid_image import GridImageBuilder

""" USAGE EXAMPLE (used in a different Python file)
from cargia.training.data_harness import TaskDataset
from cargia.training.config import TrainingConfig

# Load tasks from database
tasks = load_tasks_from_database() # STILL NEED TO IMPLEMENT THIS

# Create datasets
config = TrainingConfig()
train_dataset = TaskDataset(tasks[:100], config, is_training=True)
val_dataset = TaskDataset(tasks[100:], config, is_training=False)

# Get a conversation sequence
conversation = train_dataset[0]  # Returns List[Dict] for Gemma3

"""

class DataHarness:
    """
    Main orchestrator for data preprocessing pipeline.
    
    Handles the complete flow from raw task data to model-ready conversation
    sequences, including task-level augmentations and proper message formatting
    for Gemma3 multi-turn training.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the data harness.
        
        Args:
            config: Training configuration including augmentation settings
        """
        self.config = config
        self.augmenter = AugmentationPipeline(config)
        self.image_builder = GridImageBuilder()
        
    def create_training_conversation(self, task: Dict, is_training: bool = True) -> List[Dict]:
        """
        Create a complete conversation sequence for one task.
        
        Args:
            task (Dict): Raw task data with 'train' and 'test', each with a list of pairs of "input" and "output" grids and "thoughts" strings
            task = {
                "train": [
                    {
                        "input": List[List[int]],  # 2D grid of integers (0-9 typically)
                        "output": List[List[int]], # 2D grid of integers (0-9 typically)
                        "thoughts": str            # Ground truth reasoning text
                    },
                    # ... more training pairs
                ],
                "test": [
                    {
                        "input": List[List[int]],  # 2D grid of integers (0-9 typically)
                        "output": List[List[int]], # 2D grid of integers (0-9 typically)
                        "thoughts": str            # Ground truth reasoning text
                    },
                    # ... more test pairs
                ]
            }
            is_training: Whether this is for training (augmentations applied) or validation
            
        Returns:
            List of message dictionaries formatted for Gemma3
        
        Note: If you have SolveData objects from SolveLoader, you can convert them to this format
        using the to_data_harness_format() method:
        
        solve_data = SolveLoader(...).load_all_solves()[0]
        task = solve_data.to_data_harness_format(use_cleaned_thoughts=True)
        conversation = data_harness.create_training_conversation(task)
        """
        # Apply augmentations to entire task if training
        if is_training:
            task = self.augmenter.apply_to_task(task)
            
        # Build conversation sequence
        messages = []
        
        # Add system prompt
        messages.append(self._format_system_message())
        
        # Add training pairs
        for i, pair in enumerate(task['train']):
            messages.append(self._format_training_pair_message(pair, i))
            messages.append(self._format_assistant_response(pair['thoughts']))
            
        # Add test pairs (no output grid shown)
        for i, pair in enumerate(task['test']):
            messages.append(self._format_test_pair_message(pair, i))
            messages.append(self._format_assistant_response(pair['thoughts']))
            
        return messages
    
    def _format_system_message(self) -> Dict:
        """Format the system prompt message."""
        system_prompt = (
            "You are an AI assistant solving ARC-AGI puzzles. "
            "You will be shown training pairs with input and output grids, "
            "followed by test pairs with only input grids. "
            "Analyze the patterns in the training pairs to understand the transformation rules, "
            "then apply those rules to predict the output for test pairs. "
            "Express your reasoning clearly and provide your predictions in JSON format."
        )
        
        return {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }
    
    def _format_training_pair_message(self, pair: Dict, pair_idx: int) -> Dict:
        """
        Format a training pair as a user message with input and output grids.
        
        Args:
            pair: Training pair with 'input', 'output', and 'thoughts'
            pair_idx: Index of the training pair (0-based)
            
        Returns:
            Formatted user message for Gemma3
        """
        content = [
            {"type": "text", "text": f"This is training pair {pair_idx + 1}:"},
            {"type": "text", "text": f"Input grid: {json.dumps(pair['input'])}"},
            {"type": "image", "image": self.image_builder.build(pair['input'])},
            {"type": "text", "text": f"Output grid: {json.dumps(pair['output'])}"},
            {"type": "image", "image": self.image_builder.build(pair['output'])}
        ]
        return {"role": "user", "content": content}
    
    def _format_test_pair_message(self, pair: Dict, pair_idx: int) -> Dict:
        """
        Format a test pair as a user message with input grid only.
        
        Args:
            pair: Test pair with 'input', 'output', and 'thoughts'
            pair_idx: Index of the test pair (0-based)
            
        Returns:
            Formatted user message for Gemma3
        """
        content = [
            {"type": "text", "text": f"This is test pair {pair_idx + 1}, so you'll only get the input grid image and text. "
                                   f"Respond with text description of your thoughts and then I'll ask for one more message "
                                   f"for the JSON message response which should be the output of the test grid."},
            {"type": "text", "text": f"Input grid: {json.dumps(pair['input'])}"},
            {"type": "image", "image": self.image_builder.build(pair['input'])}
        ]
        return {"role": "user", "content": content}
    
    def _format_assistant_response(self, thoughts: str) -> Dict:
        """
        Format ground truth thoughts as assistant response.
        
        Args:
            thoughts: The ground truth thoughts text
            
        Returns:
            Formatted assistant message for Gemma3
        """
        return {
            "role": "assistant", 
            "content": [{"type": "text", "text": thoughts}]
        }


class AugmentationPipeline:
    """
    Pipeline for applying augmentations to entire tasks.
    
    Applies the same augmentation parameters to all pairs (train and test)
    within a task to maintain consistency.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the augmentation pipeline.
        
        Args:
            config: Training configuration with augmentation settings
        """
        self.config = config
        self.char_invariance = CharacterInvariance() if config.use_char_invariance else None
        self.color_invariance = ColorInvariance() if config.use_color_invariance else None
        self.spatial_transforms = SpatialTransforms() if config.use_spatial_aug else None
        
    def apply_to_task(self, task: Dict) -> Dict:
        """
        Apply augmentations to an entire task.
        
        Args:
            task: Raw task data with 'train' and 'test' pairs
            
        Returns:
            Augmented task with same transformations applied to all pairs
        """
        # Generate augmentation parameters once for the entire task
        char_map = None
        color_map = None
        spatial_transform = None
        
        if self.char_invariance:
            char_map = self.char_invariance.generate_character_map()
            
        if self.color_invariance:
            color_map = self.color_invariance.generate_color_map()
            
        if self.spatial_transforms:
            # For now, randomly select one spatial transform
            # This could be enhanced to apply multiple transforms
            available_transforms = self.spatial_transforms.get_available_transforms()
            if available_transforms:
                import random
                spatial_transform = random.choice(available_transforms)
        
        # Apply augmentations to all pairs consistently
        augmented_task = {
            'train': [self._augment_pair(pair, char_map, color_map, spatial_transform) 
                     for pair in task['train']],
            'test': [self._augment_pair(pair, char_map, color_map, spatial_transform) 
                    for pair in task['test']]
        }
        
        return augmented_task
    
    def _augment_pair(self, pair: Dict, char_map: Optional[Dict], 
                     color_map: Optional[Dict], spatial_transform: Optional[str]) -> Dict:
        """
        Apply augmentations to a single pair.
        
        Args:
            pair: Input/output pair with grids and thoughts
            char_map: Character mapping for invariance
            color_map: Color mapping for invariance  
            spatial_transform: Spatial transformation to apply
            
        Returns:
            Augmented pair
        """
        augmented_pair = pair.copy()
        
        # Apply character invariance
        if char_map:
            augmented_pair['input'] = self.char_invariance.apply_character_map_to_digit_grid(
                pair['input'], char_map)
            augmented_pair['output'] = self.char_invariance.apply_character_map_to_digit_grid(
                pair['output'], char_map)
        
        # Apply spatial transforms
        if spatial_transform:
            # Convert back to string format for spatial transforms
            input_grid = [[str(cell) for cell in row] for row in augmented_pair['input']]
            output_grid = [[str(cell) for cell in row] for row in augmented_pair['output']]
            
            # Apply spatial transform to both grids
            input_grid, input_thoughts = self.spatial_transforms.apply_spatial_transform(
                input_grid, pair['thoughts'], spatial_transform)
            output_grid, output_thoughts = self.spatial_transforms.apply_spatial_transform(
                output_grid, "", spatial_transform)  # No thoughts for output grid
            
            # Convert back to integer format
            augmented_pair['input'] = [[int(cell) if cell.isdigit() else 0 for cell in row] 
                                     for row in input_grid]
            augmented_pair['output'] = [[int(cell) if cell.isdigit() else 0 for cell in row] 
                                      for row in output_grid]
            augmented_pair['thoughts'] = input_thoughts
        
        # Note: Color invariance is handled at image generation time
        # by passing the color_map to GridImageBuilder
        
        return augmented_pair