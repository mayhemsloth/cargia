"""
Solve data loading and unification for ARC-AGI training.

This module provides classes for loading and unifying ARC-AGI solve data from:
- Raw task JSON files from filesystem
- Solve metadata from solves database
- Thought text from thoughts database

The result is a unified SolveData representation that contains all information
needed for training and difficulty analysis.
"""
import os
import json
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime

from cargia.data_manager import log_error

class AugmentAndTokenise:
    """ Class version to simplify the creation of a transformation method in conjunction with the HuggingFace Dataset's .with_transform method
    Note that the calling of the class IS THE FUNCTION to be passed into .with_transform method.
    """
    def __init__(self, harness, processor, is_training: bool = True):
        self.harness   = harness
        self.processor = processor
        self.is_training = is_training

    def __call__(self, example):

        if isinstance(example["task_raw"], dict):
            task = SolveData(**example.pop("task_raw")) # task variable is a SolveData object
        else:
            task = example.pop("task_raw") # task variable is a SolveData object

        # ➊ random augmentation on **raw text**
        conv  = self.harness.create_training_conversation(task, is_training=self.is_training)

        # ➋ chat-template command that mimics the Google tutorial
        text  = self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False).strip()

        # ➌ extracting images in order from the conversation, mimicking the Google tutorial
        imgs  = [part["image"]
                 for msg in conv
                 for part in msg.get("content", [])
                 if isinstance(part, dict) and part.get("type") == "image"]

        # ➍ processor → tensors, mimicking the Google tutorial
        enc   = self.processor(text=text,
                               images=imgs,
                               padding="max_length",
                               truncation=True,
                               max_length=8192,
                               return_tensors="pt")

        # ➎ label mask (same as Google tutorial)
        labels          = enc["input_ids"].clone()
        pad_id          = self.processor.tokenizer.pad_token_id
        boi_id          = self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.special_tokens_map["boi_token"])
        labels[(labels == pad_id) | (labels == boi_id) | (labels == 262144)] = -100

        # # --- NEW: mask user-role tokens ----------------------------
        # user_role_ids = self.processor.tokenizer.convert_tokens_to_ids("<|user|>")  # adjust to your template
        # mask_until_assistant = False
        # for i, tok in enumerate(enc["input_ids"]):
        #     if tok == user_role_ids:
        #         mask_until_assistant = True
        #     elif tok == self.processor.tokenizer.convert_tokens_to_ids("<|assistant|>"):
        #         mask_until_assistant = False
        #     if mask_until_assistant:
        #         labels[i] = -100
        # # -----------------------------------------------------------

        enc["labels"] = labels
        return {k: v.squeeze(0) for k, v in enc.items()}
    
class SolveData:
    """
    Complete representation of a solved ARC-AGI task with all associated metadata.
    
    This class unifies the raw task data, solve metadata, and collected thoughts
    into a single, comprehensive data structure for training and analysis.
    """
    
    def __init__(self, task_id: str, raw_task: Dict, solve_metadata: Dict, 
                 train_pairs: List[Dict], test_pairs: List[Dict], **kwargs):
        """
        Initialize SolveData with all components.
        
        Args:
            task_id: Unique identifier for the task
            raw_task: Original JSON task structure from filesystem
            solve_metadata: Database metadata (start_time, end_time, user_id, etc.)
            train_pairs: Training pairs with thoughts and metadata
            test_pairs: Test pairs with thoughts and metadata
        """
        self.task_id = task_id
        self.raw_task = raw_task
        self.solve_metadata = solve_metadata
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.difficulty_score = None  # Will be calculated later
        self._proxy_difficulty_scores = None  # Cache for individual scores
        self.kwargs = kwargs
        
        # Calculate proxy difficulty score automatically during initialization
        self.calculate_proxy_difficulty_score()
    
    def get_all_pairs(self) -> List[Dict]:
        """Get all pairs (train + test) in sequence."""
        return self.train_pairs + self.test_pairs
    
    def get_pair_by_label(self, pair_label: str) -> Optional[Dict]:
        """Get a specific pair by its label (a, b, c, etc.)."""
        # Search in both train and test pairs
        for pair in self.get_all_pairs():
            if pair.get('pair_label') == pair_label:
                return pair
        return None
    
    def get_thoughts_by_pair_type(self, pair_type: str) -> List[str]:
        """Get all thoughts for a specific pair type (train/test)."""
        pairs = self.train_pairs if pair_type == 'train' else self.test_pairs
        return [pair.get('thought_text', '') for pair in pairs]
    
    def get_solve_duration_seconds(self) -> Optional[int]:
        """Calculate solve duration in seconds if both start and end times exist."""
        start_time = self.solve_metadata.get('start_time')
        end_time = self.solve_metadata.get('end_time')
        
        if not start_time or not end_time:
            return None
            
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            return int((end - start).total_seconds())
        except (ValueError, TypeError):
            return None
        
    def get_color_map(self) -> Dict:
        """Get the color map for the task."""
        return self.solve_metadata.get('color_map', {})
    
    def get_order_map(self) -> Dict:
        """Get the order map for the task."""
        return self.solve_metadata.get('order_map', {})
    
    def to_dict(self) -> Dict:
        """
        Convert the SolveData object to a Python dictionary.
        
        This method dumps all class attributes into a dictionary, preserving
        the structure of nested dictionaries and lists. This is useful for
        serialization, debugging, or when you need to work with the data
        in a dictionary format.
        
        Returns:
            Dictionary containing all SolveData attributes with their current values
        """
        return {
            'task_id': self.task_id,
            'raw_task': self.raw_task,
            'solve_metadata': self.solve_metadata,
            'train_pairs': self.train_pairs,
            'test_pairs': self.test_pairs,
            'difficulty_score': self.difficulty_score,
            '_proxy_difficulty_scores': self._proxy_difficulty_scores
        }
    
    
    def __repr__(self) -> str:
        return f"SolveData(task_id='{self.task_id}', user='{self.solve_metadata.get('user_id', 'unknown')}', train_pairs={len(self.train_pairs)}, test_pairs={len(self.test_pairs)})"
    
    def pretty_print(self, include_grids: bool = True, include_thoughts: bool = True, max_grid_size: int = 10) -> str:
        """
        Create a detailed, formatted string representation of all SolveData information.
        
        Args:
            include_grids: Whether to include the actual grid data (can be large)
            include_thoughts: Whether to include the thought text (can be long)
            max_grid_size: Maximum grid size to display (truncate larger grids)
            
        Returns:
            Formatted string with all SolveData information
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"SOLVE DATA: {self.task_id}")
        lines.append("=" * 80)
        
        # Basic information
        lines.append("\n📋 BASIC INFORMATION")
        lines.append("-" * 40)
        lines.append(f"Task ID: {self.task_id}")
        lines.append(f"User ID: {self.solve_metadata.get('user_id', 'unknown')}")
        lines.append(f"Order Map Type: {self.solve_metadata.get('order_map_type', 'unknown')}")
        lines.append(f"Train Pairs: {len(self.train_pairs)}")
        lines.append(f"Test Pairs: {len(self.test_pairs)}")
        
        # Timing information
        duration = self.get_solve_duration_seconds()
        if duration:
            lines.append(f"Solve Duration: {duration} seconds ({duration//60}m {duration%60}s)")
        else:
            lines.append("Solve Duration: Unknown (missing start/end time)")
        
        lines.append(f"Start Time: {self.solve_metadata.get('start_time', 'unknown')}")
        lines.append(f"End Time: {self.solve_metadata.get('end_time', 'unknown')}")
        
        # Order map
        order_map = self.solve_metadata.get('order_map', {})
        lines.append(f"Order Map: {order_map}")
        
        # Color map
        color_map = self.solve_metadata.get('color_map', {})
        if color_map:
            lines.append(f"Color Map: {color_map}")
        else:
            lines.append("Color Map: None")
        
        # Metadata labels
        metadata_labels = self.solve_metadata.get('metadata_labels', {})
        if metadata_labels:
            lines.append(f"Metadata Labels: {metadata_labels}")
        else:
            lines.append("Metadata Labels: None")
        
        # Raw task information
        lines.append("\n📄 RAW TASK INFORMATION")
        lines.append("-" * 40)
        if 'default_splits' in self.raw_task:
            lines.append(f"Default Splits: {self.raw_task['default_splits']}")
        if 'pairs' in self.raw_task:
            lines.append(f"Total Pairs in Raw Task: {len(self.raw_task['pairs'])}")
            lines.append("Available Pair Labels: " + ", ".join(sorted(self.raw_task['pairs'].keys())))
        
        # Training pairs
        if self.train_pairs:
            lines.append("\n🎯 TRAINING PAIRS")
            lines.append("-" * 40)
            for i, pair in enumerate(self.train_pairs):
                lines.append(f"\nTraining Pair {i+1} (Label: {pair['pair_label']}):")
                lines.append(f"  Sequence Index: {pair['sequence_index']}")
                
                if include_grids:
                    input_grid = pair['input']
                    output_grid = pair['output']
                    
                    # Truncate large grids
                    if len(input_grid) > max_grid_size or (input_grid and len(input_grid[0]) > max_grid_size):
                        lines.append(f"  Input Grid: {len(input_grid)}x{len(input_grid[0]) if input_grid else 0} (truncated)")
                    else:
                        lines.append("  Input Grid:")
                        for row in input_grid:
                            lines.append(f"    {row}")
                    
                    if len(output_grid) > max_grid_size or (output_grid and len(output_grid[0]) > max_grid_size):
                        lines.append(f"  Output Grid: {len(output_grid)}x{len(output_grid[0]) if output_grid else 0} (truncated)")
                    else:
                        lines.append("  Output Grid:")
                        for row in output_grid:
                            lines.append(f"    {row}")
                else:
                    lines.append(f"  Input Grid: {len(pair['input'])}x{len(pair['input'][0]) if pair['input'] else 0}")
                    lines.append(f"  Output Grid: {len(pair['output'])}x{len(pair['output'][0]) if pair['output'] else 0}")
                
                if include_thoughts:
                    thought_text = pair.get('thought_text', '')
                    cleaned_thought = pair.get('cleaned_thought_text', '')
                    
                    if thought_text:
                        lines.append(f"  Thought Text: {repr(thought_text)}")
                    else:
                        lines.append("  Thought Text: (empty)")
                    
                    if cleaned_thought and cleaned_thought != thought_text:
                        lines.append(f"  Cleaned Thought: {repr(cleaned_thought)}")
                else:
                    thought_length = len(pair.get('thought_text', ''))
                    cleaned_length = len(pair.get('cleaned_thought_text', ''))
                    lines.append(f"  Thought Text Length: {thought_length} chars")
                    lines.append(f"  Cleaned Thought Length: {cleaned_length} chars")
        
        # Test pairs
        if self.test_pairs:
            lines.append("\n🧪 TEST PAIRS")
            lines.append("-" * 40)
            for i, pair in enumerate(self.test_pairs):
                lines.append(f"\nTest Pair {i+1} (Label: {pair['pair_label']}):")
                lines.append(f"  Sequence Index: {pair['sequence_index']}")
                
                if include_grids:
                    input_grid = pair['input']
                    output_grid = pair['output']
                    
                    # Truncate large grids
                    if len(input_grid) > max_grid_size or (input_grid and len(input_grid[0]) > max_grid_size):
                        lines.append(f"  Input Grid: {len(input_grid)}x{len(input_grid[0]) if input_grid else 0} (truncated)")
                    else:
                        lines.append("  Input Grid:")
                        for row in input_grid:
                            lines.append(f"    {row}")
                    
                    if len(output_grid) > max_grid_size or (output_grid and len(output_grid[0]) > max_grid_size):
                        lines.append(f"  Output Grid: {len(output_grid)}x{len(output_grid[0]) if output_grid else 0} (truncated)")
                    else:
                        lines.append("  Output Grid:")
                        for row in output_grid:
                            lines.append(f"    {row}")
                else:
                    lines.append(f"  Input Grid: {len(pair['input'])}x{len(pair['input'][0]) if pair['input'] else 0}")
                    lines.append(f"  Output Grid: {len(pair['output'])}x{len(pair['output'][0]) if pair['output'] else 0}")
                
                if include_thoughts:
                    thought_text = pair.get('thought_text', '')
                    cleaned_thought = pair.get('cleaned_thought_text', '')
                    
                    if thought_text:
                        lines.append(f"  Thought Text: {repr(thought_text)}")
                    else:
                        lines.append("  Thought Text: (empty)")
                    
                    if cleaned_thought and cleaned_thought != thought_text:
                        lines.append(f"  Cleaned Thought: {repr(cleaned_thought)}")
                else:
                    thought_length = len(pair.get('thought_text', ''))
                    cleaned_length = len(pair.get('cleaned_thought_text', ''))
                    lines.append(f"  Thought Text Length: {thought_length} chars")
                    lines.append(f"  Cleaned Thought Length: {cleaned_length} chars")
        
        # Difficulty score (if available)
        if self.difficulty_score is not None:
            lines.append("\n📊 PROXY DIFFICULTY ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Final Proxy Difficulty Score: {self.difficulty_score:.3f}")
            
            if self._proxy_difficulty_scores:
                lines.append("\nIndividual Attribute Scores:")
                lines.append(f"  Training Pairs Count: {self._proxy_difficulty_scores['training_pairs_count']}")
                lines.append(f"  Average Grid Size: {self._proxy_difficulty_scores['average_grid_size']:.1f}")
                lines.append(f"  Average Unique Colors: {self._proxy_difficulty_scores['average_unique_colors']:.1f}")
                lines.append(f"  Grid Variability: {self._proxy_difficulty_scores['grid_variability']:.3f} (placeholder)")
                lines.append(f"  Average Thought Length: {self._proxy_difficulty_scores['average_thought_length']:.1f}")
                lines.append(f"  Size Consistency: {self._proxy_difficulty_scores['size_consistency']:.3f}")
        else:
            lines.append("\n📊 PROXY DIFFICULTY ANALYSIS")
            lines.append("-" * 40)
            lines.append("Proxy difficulty score not calculated yet.")
            lines.append("Call calculate_proxy_difficulty_score() to compute.")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    
    def _calculate_training_pairs_count(self) -> int:
        """Calculate attribute 1: Total number of training pairs."""
        return len(self.train_pairs)
    
    def _calculate_average_grid_size(self) -> float:
        """Calculate attribute 2: Average tile size (rows × columns) of training grids."""
        if not self.train_pairs:
            return 0.0
        
        total_size = 0
        total_grids = 0
        
        for pair in self.train_pairs:
            # Input grid size
            input_grid = pair['input']
            if input_grid:
                total_size += len(input_grid) * len(input_grid[0])
                total_grids += 1
            
            # Output grid size
            output_grid = pair['output']
            if output_grid:
                total_size += len(output_grid) * len(output_grid[0])
                total_grids += 1
        
        return total_size / total_grids if total_grids > 0 else 0.0
    
    def _calculate_average_unique_colors(self) -> float:
        """Calculate attribute 3: Average number of unique colors/characters in training grids."""
        if not self.train_pairs:
            return 0.0
        
        total_unique_colors = 0
        total_grids = 0
        
        for pair in self.train_pairs:
            # Input grid unique colors
            input_grid = pair['input']
            if input_grid:
                unique_colors = set()
                for row in input_grid:
                    for cell in row:
                        unique_colors.add(cell)
                total_unique_colors += len(unique_colors)
                total_grids += 1
            
            # Output grid unique colors
            output_grid = pair['output']
            if output_grid:
                unique_colors = set()
                for row in output_grid:
                    for cell in row:
                        unique_colors.add(cell)
                total_unique_colors += len(unique_colors)
                total_grids += 1
        
        return total_unique_colors / total_grids if total_grids > 0 else 0.0
    
    def _calculate_grid_variability(self) -> float:
        """Calculate attribute 4: Measure of grid 'smoothness' vs 'chaoticness'."""
        # TODO: Implement grid variability calculation
        # For now, return a placeholder value
        return 0.0
    
    def _calculate_average_thought_length(self) -> float:
        """Calculate attribute 5: Average character length of training thoughts."""
        if not self.train_pairs:
            return 0.0
        
        total_length = 0
        total_thoughts = 0
        
        for pair in self.train_pairs:
            thought_text = pair.get('thought_text', '')
            if thought_text:
                total_length += len(thought_text)
                total_thoughts += 1
        
        return total_length / total_thoughts if total_thoughts > 0 else 0.0
    
    def _calculate_size_consistency(self) -> float:
        """Calculate attribute 6: Proportion of pairs with same input/output grid sizes."""
        if not self.train_pairs:
            return 0.0
        
        consistent_pairs = 0
        total_pairs = 0
        
        for pair in self.train_pairs:
            input_grid = pair['input']
            output_grid = pair['output']
            
            if input_grid and output_grid:
                input_rows, input_cols = len(input_grid), len(input_grid[0])
                output_rows, output_cols = len(output_grid), len(output_grid[0])
                
                # Check if input and output grids have the same dimensions
                if input_rows == output_rows and input_cols == output_cols:
                    consistent_pairs += 1
                total_pairs += 1
        
        return consistent_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def calculate_proxy_difficulty_score(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate a proxy difficulty score based on various task characteristics.
        
        This method analyzes the task data to estimate difficulty based on:
        1. Number of training pairs (more pairs = easier to understand pattern)
        2. Average grid size (larger grids = more complex)
        3. Average number of unique colors (more colors = more complex)
        4. Grid size variability (inconsistent sizes = more complex)
        5. Average thought length (longer thoughts = more complex reasoning)
        6. Size consistency between input/output (inconsistent = more complex)
        
        Args:
            weights: Optional dictionary to weight different factors differently
            
        Returns:
            Dictionary with individual scores and weighted total
        """
        # Default weights (all factors equally important)
        default_weights = {
            'training_pairs_count': 1.0,
            'average_grid_size': 1.0,
            'average_unique_colors': 1.0,
            'grid_variability': 1.0,
            'average_thought_length': 1.0,
            'size_consistency': 1.0
        }
        
        if weights:
            default_weights.update(weights)
        
        # Calculate individual scores
        scores = {
            'training_pairs_count': self._calculate_training_pairs_count(),
            'average_grid_size': self._calculate_average_grid_size(),
            'average_unique_colors': self._calculate_average_unique_colors(),
            'grid_variability': self._calculate_grid_variability(),
            'average_thought_length': self._calculate_average_thought_length(),
            'size_consistency': self._calculate_size_consistency()
        }
        
        # Calculate weighted total
        weighted_total = sum(scores[key] * default_weights[key] for key in scores)
        
        # Store results
        self._proxy_difficulty_scores = {
            'individual_scores': scores,
            'weights': default_weights,
            'weighted_total': weighted_total
        }
        
        return self._proxy_difficulty_scores
    
    def to_data_harness_format(self, use_cleaned_thoughts: bool = True) -> Dict:
        """
        Convert SolveData to the format expected by DataHarness.create_training_conversation().
        
        Args:
            use_cleaned_thoughts: Whether to use cleaned_thought_text (True) or thought_text (False)
            
        Returns:
            Dictionary in the format expected by DataHarness:
            {
                "train": [
                    {
                        "input": List[List[int]],
                        "output": List[List[int]], 
                        "thoughts": str
                    },
                    ...
                ],
                "test": [
                    {
                        "input": List[List[int]],
                        "output": List[List[int]],
                        "thoughts": str
                    },
                    ...
                ]
            }
            
        Note: Pairs are sorted by sequence_index to preserve the exact order in which they
        were presented during data collection. The sequence_index does not reset between
        train and test pairs - it starts at 0 and increments across all pairs.
        """
        # Choose which thought field to use
        thought_field = 'cleaned_thought_text' if use_cleaned_thoughts else 'thought_text'
        
        # Combine all pairs and sort by sequence_index to preserve original order
        all_pairs = []
        
        # Add train pairs with their sequence_index
        for pair in self.train_pairs:
            all_pairs.append({
                'input': pair['input'],
                'output': pair['output'],
                'thoughts': pair.get(thought_field, '') or pair.get('thought_text', ''),
                'sequence_index': pair['sequence_index'],
                'pair_type': 'train'
            })
        
        # Add test pairs with their sequence_index
        for pair in self.test_pairs:
            all_pairs.append({
                'input': pair['input'],
                'output': pair['output'],
                'thoughts': pair.get(thought_field, '') or pair.get('thought_text', ''),
                'sequence_index': pair['sequence_index'],
                'pair_type': 'test'
            })
        
        # Sort by sequence_index to preserve original presentation order
        all_pairs.sort(key=lambda x: x['sequence_index'])
        
        # Separate back into train and test based on pair_type
        train_pairs = []
        test_pairs = []
        
        for pair in all_pairs:
            # Remove the temporary fields we added for sorting
            formatted_pair = {
                'input': pair['input'],
                'output': pair['output'],
                'thoughts': pair['thoughts']
            }
            
            if pair['pair_type'] == 'train':
                train_pairs.append(formatted_pair)
            else:  # pair_type == 'test'
                test_pairs.append(formatted_pair)
        
        return {
            'train': train_pairs,
            'test': test_pairs
        }


class SolveLoader:
    """
    Loads and unifies ARC-AGI solve data from multiple sources.
    
    This class handles:
    - Loading raw task JSON files from filesystem
    - Querying solves and thoughts databases
    - Joining data to create complete SolveData objects
    - Handling missing data gracefully
    """
    
    def __init__(self, data_dir: str, source_folder: str):
        """
        Initialize the solve loader.
        
        Args:
            data_dir: Path to directory containing solves.db and thoughts.db
            source_folder: Path to directory containing task JSON files
        """
        self.data_dir = data_dir
        self.source_folder = source_folder
        self.solves_db_path = os.path.join(data_dir, "solves.db")
        self.thoughts_db_path = os.path.join(data_dir, "thoughts.db")
        self.training_tasks_path = os.path.join(source_folder, "training.txt")
        
        # Validate paths exist
        self._validate_paths()
    
    def load_all_solves(self, user_filter: Optional[str] = None) -> List[SolveData]:
        """
        Load all completed solves with their associated thoughts.
        
        Args:
            user_filter: Optional user ID to filter solves by specific user
            
        Returns:
            List of SolveData objects representing all completed solves
        """
        solve_records = self._load_solve_records(user_filter)
        solve_data_list = []
        
        for solve_record in solve_records:
            try:
                # Load task JSON
                task_data = self._load_task_json(solve_record['task_id'])
                if not task_data:
                    continue
                
                # Load thoughts for this solve
                thoughts = self._load_thoughts_for_solve(solve_record['solve_id'])
                
                # Join data and create SolveData
                solve_data = self._join_task_with_solve_data(task_data, solve_record, thoughts)
                if solve_data:
                    solve_data_list.append(solve_data)
                    
            except Exception as e:
                log_error(f"Failed to load solve {solve_record['solve_id']} for task {solve_record['task_id']}", e)
                continue
        
        return solve_data_list
    
    def load_solves_by_task_id(self, task_ids: List[str]) -> List[SolveData]:
        """Load solves for specific task IDs."""
        solve_records = self._load_solve_records_by_task_ids(task_ids)
        solve_data_list = []
        
        for solve_record in solve_records:
            try:
                task_data = self._load_task_json(solve_record['task_id'])
                if not task_data:
                    continue
                
                thoughts = self._load_thoughts_for_solve(solve_record['solve_id'])
                solve_data = self._join_task_with_solve_data(task_data, solve_record, thoughts)
                if solve_data:
                    solve_data_list.append(solve_data)
                    
            except Exception as e:
                log_error(f"Failed to load solve {solve_record['solve_id']} for task {solve_record['task_id']}", e)
                continue
        
        return solve_data_list
    
    def load_solves_by_user(self, user_id: str) -> List[SolveData]:
        """Load all solves by a specific user."""
        return self.load_all_solves(user_filter=user_id)
    
    def _load_task_json(self, task_id: str) -> Optional[Dict]:
        """Load raw task JSON from filesystem."""
        task_path = os.path.join(self.source_folder, "training", f"{task_id}.json")
        
        if not os.path.exists(task_path):
            log_error(f"Task JSON file not found: {task_path}")
            return None
        
        try:
            with open(task_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            log_error(f"Failed to load task JSON from {task_path}", e)
            return None
    
    def _load_solve_records(self, user_filter: Optional[str] = None) -> List[Dict]:
        """Load solve records from database."""
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            if user_filter:
                cursor.execute("""
                    SELECT id, task_id, user_id, order_map_type, order_map, 
                           color_map, metadata_labels, start_time, end_time
                    FROM solves
                    WHERE user_id = ? AND end_time IS NOT NULL
                    ORDER BY id
                """, (user_filter,))
            else:
                cursor.execute("""
                    SELECT id, task_id, user_id, order_map_type, order_map, 
                           color_map, metadata_labels, start_time, end_time
                    FROM solves
                    WHERE end_time IS NOT NULL
                    ORDER BY id
                """)
            
            solve_records = []
            for row in cursor.fetchall():
                solve_records.append({
                    'solve_id': row[0],
                    'task_id': row[1],
                    'user_id': row[2],
                    'order_map_type': row[3],
                    'order_map': json.loads(row[4]) if row[4] else {},
                    'color_map': json.loads(row[5]) if row[5] else {},
                    'metadata_labels': json.loads(row[6]) if row[6] else {},
                    'start_time': row[7],
                    'end_time': row[8]
                })
            
            return solve_records
            
        finally:
            conn.close()
    
    def _load_solve_records_by_task_ids(self, task_ids: List[str]) -> List[Dict]:
        """Load solve records for specific task IDs."""
        if not task_ids:
            return []
            
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            placeholders = ','.join('?' * len(task_ids))
            cursor.execute(f"""
                SELECT id, task_id, user_id, order_map_type, order_map, 
                       color_map, metadata_labels, start_time, end_time
                FROM solves
                WHERE task_id IN ({placeholders}) AND end_time IS NOT NULL
                ORDER BY id
            """, task_ids)
            
            solve_records = []
            for row in cursor.fetchall():
                solve_records.append({
                    'solve_id': row[0],
                    'task_id': row[1],
                    'user_id': row[2],
                    'order_map_type': row[3],
                    'order_map': json.loads(row[4]) if row[4] else {},
                    'color_map': json.loads(row[5]) if row[5] else {},
                    'metadata_labels': json.loads(row[6]) if row[6] else {},
                    'start_time': row[7],
                    'end_time': row[8]
                })
            
            return solve_records
            
        finally:
            conn.close()
    
    def _load_thoughts_for_solve(self, solve_id: int) -> List[Dict]:
        """Load all thoughts for a specific solve."""
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, solve_id, pair_label, pair_type, sequence_index, 
                       thought_text, cleaned_thought_text
                FROM thoughts
                WHERE solve_id = ?
                ORDER BY sequence_index
            """, (solve_id,))
            
            thoughts = []
            for row in cursor.fetchall():
                thoughts.append({
                    'id': row[0],
                    'solve_id': row[1],
                    'pair_label': row[2],
                    'pair_type': row[3],
                    'sequence_index': row[4],
                    'thought_text': row[5] or '',
                    'cleaned_thought_text': row[6] or ''
                })
            
            return thoughts
            
        finally:
            conn.close()
    
    def _join_task_with_solve_data(self, task_data: Dict, solve_record: Dict, 
                                  thoughts: List[Dict]) -> Optional[SolveData]:
        """Join task data with solve metadata and thoughts to create SolveData."""
        try:
            # Get the order map to determine which pairs are train vs test
            order_map = solve_record.get('order_map', {})
            train_labels = order_map.get('train', [])
            test_labels = order_map.get('test', [])
            
            # Create thought lookup by pair label
            thought_lookup = {thought['pair_label']: thought for thought in thoughts}
            
            # Build train pairs
            train_pairs = []
            for label in train_labels:
                if label in task_data.get('pairs', {}):
                    pair_data = task_data['pairs'][label]
                    thought_data = thought_lookup.get(label, {})
                    
                    train_pairs.append({
                        'pair_label': label,
                        'pair_type': 'train',
                        'sequence_index': len(train_pairs),
                        'input': pair_data.get('input', []),
                        'output': pair_data.get('output', []),
                        'thought_text': thought_data.get('thought_text', ''),
                        'cleaned_thought_text': thought_data.get('cleaned_thought_text', '')
                    })
            
            # Build test pairs
            test_pairs = []
            for label in test_labels:
                if label in task_data.get('pairs', {}):
                    pair_data = task_data['pairs'][label]
                    thought_data = thought_lookup.get(label, {})
                    
                    test_pairs.append({
                        'pair_label': label,
                        'pair_type': 'test',
                        'sequence_index': len(test_pairs),
                        'input': pair_data.get('input', []),
                        'output': pair_data.get('output', []),
                        'thought_text': thought_data.get('thought_text', ''),
                        'cleaned_thought_text': thought_data.get('cleaned_thought_text', '')
                    })
            
            return SolveData(
                task_id=solve_record['task_id'],
                raw_task=task_data,
                solve_metadata=solve_record,
                train_pairs=train_pairs,
                test_pairs=test_pairs
            )
            
        except Exception as e:
            log_error(f"Failed to join task data for solve {solve_record['solve_id']}", e)
            return None
    
    def _validate_paths(self):
        """Validate that all required paths exist."""
        if not os.path.exists(self.solves_db_path):
            raise FileNotFoundError(f"Solves database not found: {self.solves_db_path}")
        
        if not os.path.exists(self.thoughts_db_path):
            raise FileNotFoundError(f"Thoughts database not found: {self.thoughts_db_path}")
        
        if not os.path.exists(self.source_folder):
            raise FileNotFoundError(f"Source folder not found: {self.source_folder}")
        
        training_folder = os.path.join(self.source_folder, "training")
        if not os.path.exists(training_folder):
            raise FileNotFoundError(f"Training folder not found: {training_folder}")
        
        if not os.path.exists(self.training_tasks_path):
            raise FileNotFoundError(f"Training tasks file not found: {self.training_tasks_path}")
    
    def _handle_missing_data(self, task_id: str, solve_record: Dict, 
                           thoughts: List[Dict]) -> Optional[SolveData]:
        """Handle cases where task JSON or thoughts are missing."""
        # This method can be expanded later to handle various missing data scenarios
        # For now, we'll just log warnings and return None
        log_error(f"Missing data for task {task_id}, solve {solve_record['solve_id']}")
        return None 