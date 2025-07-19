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
import random
from PIL import Image

from cargia.training.training_config import TrainingConfig
from cargia.training.augment import CharacterInvariance, ColorInvariance, SpatialTransforms
from cargia.common.grid_image import GridImageBuilder
from cargia.training.solve_loader import SolveData

""" USAGE EXAMPLE (used in a different Python file)
from cargia.training.data_harness import DataHarness
from cargia.training.config import TrainingConfig
from cargia.training.solve_loader import SolveLoader

# Load SolveData objects from database
solve_loader = SolveLoader(data_dir, source_folder)
solves = solve_loader.load_all_solves()

# Create datasets
config = TrainingConfig()
data_harness = DataHarness(config)

# Create conversation for a SolveData object
conversation = data_harness.create_training_conversation(solves[0], is_training=True)

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
        
    def create_training_conversation(self, solve_data: SolveData, is_training: bool = True) -> List[Dict]:
        """
        Create a complete conversation sequence for one task.
        
        Args:
            solve_data: A SolveData object containing all task information
            is_training: Whether this is for training (augmentations applied) or validation
            
        Returns:
            List of message dictionaries formatted for Gemma3
        """
        # Apply augmentations to entire task if training
        if is_training:
            solve_data = self.augmenter.apply_to_solve_data(solve_data)
            
        # Build conversation sequence
        messages = []
        
        # Add system prompt
        messages.append(self._format_system_message(solve_data.solve_metadata))
        
        # Add training pairs
        for i, pair in enumerate(solve_data.train_pairs):
            messages.append(self._format_training_pair_message(pair, i))
            messages.append(self._format_assistant_response(pair.get('cleaned_thought_text', '') or pair.get('thought_text', '')))
            
        # Add test pairs (no output grid shown, seperate message for the request for the output grid, and another message for the output grid text)
        for i, pair in enumerate(solve_data.test_pairs):
            messages.append(self._format_test_pair_input_grid_message(pair, i))
            messages.append(self._format_assistant_response(pair.get('cleaned_thought_text', '') or pair.get('thought_text', '')))
            messages.append(self._format_test_pair_request_output_grid_message(pair, i))  # separate message for the request for the output grid
            messages.append(self._format_assistant_response_test_pair_output_grid_text_message(pair, i))  # separate message for the output grid text
            
        return messages
    
    def _format_system_message(self, solve_metadata: Dict) -> Dict:
        """Format the system prompt message."""
        system_prompt = (
            "You are an AI assistant solving ARC-AGI puzzles. "
            "You will be shown training pairs with input and output grids, "
            "followed by test pairs with only input grids. "
            "Analyze the patterns in the training pairs to understand the transformation rules, "
            "then apply those rules to predict the output for test pairs. "
            "Express your reasoning clearly and provide your predictions in JSON format. "
        )

        def _format_color_map(color_map: Dict) -> str:
            color_map_str = ""
            for key, value in color_map.items():
                color_map_str += f"Character {key}: {value['name']}\n"
            return color_map_str

        color_map_message = ("The color map for all pairs in this task connecting the characters seen "
                             "in the text grid representation and the colors of tiles seen in the images is: \n"
                             f"{_format_color_map(solve_metadata['color_map'])}")

        system_prompt += color_map_message

        return {    
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        }
    
    def _format_training_pair_message(self, pair: Dict, pair_idx: int) -> Dict:
        """
        Format a training pair as a user message with input and output grids.
        
        Args:
            pair: Training pair with 'input', 'output', and thought fields
            pair_idx: Index of the training pair (0-based)
            
        Returns:
            Formatted user message for Gemma3
        """
        content = [
            {"type": "text", "text": f"This is training pair {pair_idx + 1}: "},
            {"type": "text", "text": f"Input grid: {json.dumps(pair['input'])}"},
            {"type": "image", "image": self.image_builder.build(pair['input'])},
            {"type": "text", "text": f"Output grid: {json.dumps(pair['output'])}"},
            {"type": "image", "image": self.image_builder.build(pair['output'])}
        ]
        return {"role": "user", "content": content}
    
    def _format_test_pair_input_grid_message(self, pair: Dict, pair_idx: int) -> Dict:
        """
        Format a test pair as a user message with input grid only.
        
        Args:
            pair: Test pair with 'input', 'output', and thought fields
            pair_idx: Index of the test pair (0-based)
            
        Returns:
            Formatted user message for Gemma3
        """
        content = [
            {"type": "text", "text": f"This is test pair {pair_idx + 1}, so you'll only get the input grid image and text. "
                                   f"Respond with a text description of your thoughts and then I'll follow-up with another message"
                                   f"for your JSON response which should be the output of this test grid."},
            {"type": "text", "text": f"Input grid: {json.dumps(pair['input'])}"},
            {"type": "image", "image": self.image_builder.build(pair['input'])}
        ]
        return {"role": "user", "content": content}
    
    def _format_test_pair_request_output_grid_message(self, pair: Dict, pair_idx: int) -> Dict:
        """
        Format the request for the output grid as a user message with some request text.
        
        Args:
            pair: Test pair with 'input', 'output', and thought fields
            pair_idx: Index of the test pair (0-based)
            
        Returns:
            Formatted user message for Gemma3
        """
        content = [
            {"type": "text", "text": f"Referencing specifically your comments on test pair {pair_idx + 1} and all your other thoughts leading up to it, "
                                     f"respond now with ONLY the correct output grid for test pair {pair_idx + 1} in JSON format."}
        ]
        return {"role": "user", "content": content}
    
    def _format_assistant_response_test_pair_output_grid_text_message(self, pair: Dict, pair_idx: int) -> Dict:
        """
        Format the output grid text as a user message with some request text.
        
        Args:
            pair: Test pair with 'input', 'output', and thought fields
            pair_idx: Index of the test pair (0-based)
            
        Returns:
            Formatted user message for Gemma3
        """
        content = [
            {"type": "text", "text": f"{json.dumps(pair['output'])}"}
        ]
        return {"role": "assistant", "content": content}
    
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
        self.use_any_augmentation = config.use_any_augmentation
        self.char_invariance = CharacterInvariance() if config.use_char_invariance and self.use_any_augmentation else None
        self.color_invariance = ColorInvariance() if config.use_color_invariance and self.use_any_augmentation else None
        self.spatial_transforms = SpatialTransforms() if config.use_spatial_aug and self.use_any_augmentation else None
        
    def apply_to_solve_data(self, solve_data: SolveData) -> SolveData:
        """
        Apply augmentations to an entire SolveData object.
        
        Args:
            solve_data: SolveData object containing train and test pairs
            
        Returns:
            Augmented SolveData with same transformations applied to all pairs
        """

        original_color_map = solve_data.get_color_map()
        
        # Generate augmentation parameters once for the entire task
        target_char_map = None
        target_color_map = None
        spatial_transform = None
        
        if self.char_invariance:
            target_char_map = self.char_invariance.generate_character_map()
            
        if self.color_invariance:
            target_color_map = self.color_invariance.generate_color_map()
            
        if self.spatial_transforms:
            # For now, randomly select one spatial transform
            # This could be enhanced to apply multiple transforms
            available_transforms = self.spatial_transforms.get_available_transforms()
            if available_transforms:
                spatial_transform = random.choice(available_transforms)
        
        # Apply augmentations to all pairs consistently
        augmented_train_pairs = [
            self._augment_pair(pair, target_char_map, original_color_map, target_color_map, spatial_transform) 
            for pair in solve_data.train_pairs
        ]
        
        augmented_test_pairs = [
            self._augment_pair(pair, target_char_map, original_color_map, target_color_map, spatial_transform) 
            for pair in solve_data.test_pairs
        ]
        
        # Note that in the SolveData new object eblow, I need to add the target_color_map to the solve_metadata, overwriting the original color map
        if target_color_map:
            solve_data.solve_metadata['color_map'] = target_color_map
            
        # If both character and color invariance are applied, update the color map to reflect character transformations
        if target_char_map and target_color_map:
            # Create a new color map that maps the transformed characters to the target colors
            updated_color_map = {}
            for original_char, target_char in target_char_map.items():
                updated_color_map[target_char] = target_color_map[original_char]
            solve_data.solve_metadata['color_map'] = updated_color_map

        # Create a new SolveData object with augmented pairs
        augmented_solve_data = SolveData(
            task_id=solve_data.task_id,
            raw_task=solve_data.raw_task,
            solve_metadata=solve_data.solve_metadata,
            train_pairs=augmented_train_pairs,
            test_pairs=augmented_test_pairs
        )
        
        return augmented_solve_data
    
       
    def _augment_pair(self, pair: Dict, char_map: Optional[Dict], 
                     original_color_map: Optional[Dict], target_color_map: Optional[Dict],
                     spatial_transform: Optional[str]) -> Dict: 
        """
        Apply augmentations to a single pair.
        
        Args:
            pair: Input/output pair with grids and thoughts
            char_map: Character mapping for invariance
            original_color_map: Original color mapping for invariance  
            target_color_map: Target color mapping for invariance
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
        
        # apply color invariance
        if original_color_map and target_color_map:
            # Update the thought text with the transformed version
            if 'cleaned_thought_text' in augmented_pair:
                augmented_pair['cleaned_thought_text'] = self.color_invariance.apply_color_map_to_text(
                    augmented_pair['cleaned_thought_text'], original_color_map, target_color_map)
            else:
                augmented_pair['thought_text'] = self.color_invariance.apply_color_map_to_text(
                    augmented_pair['thought_text'], original_color_map, target_color_map)
        
        # Apply spatial transforms
        if spatial_transform:
            # Convert back to string format for spatial transforms
            input_grid = [[str(cell) for cell in row] for row in augmented_pair['input']]
            output_grid = [[str(cell) for cell in row] for row in augmented_pair['output']]
            
            # Get the thought text to transform
            thought_text = pair.get('cleaned_thought_text', '') or pair.get('thought_text', '')
            
            # Apply spatial transform to both grids
            input_grid, input_thoughts = self.spatial_transforms.apply_spatial_transform(
                input_grid, thought_text, spatial_transform)
            output_grid, output_thoughts = self.spatial_transforms.apply_spatial_transform(
                output_grid, "", spatial_transform)  # No thoughts for output grid
            
            # Convert back to integer format
            augmented_pair['input'] = [[int(cell) if cell.isdigit() else 0 for cell in row] 
                                     for row in input_grid]
            augmented_pair['output'] = [[int(cell) if cell.isdigit() else 0 for cell in row] 
                                      for row in output_grid]
            
            # Update the thought text with the transformed version
            if 'cleaned_thought_text' in augmented_pair:
                augmented_pair['cleaned_thought_text'] = input_thoughts
            else:
                augmented_pair['thought_text'] = input_thoughts
        
        return augmented_pair