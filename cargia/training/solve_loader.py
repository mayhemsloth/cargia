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


class SolveData:
    """
    Complete representation of a solved ARC-AGI task with all associated metadata.
    
    This class unifies the raw task data, solve metadata, and collected thoughts
    into a single, comprehensive data structure for training and analysis.
    """
    
    def __init__(self, task_id: str, raw_task: Dict, solve_metadata: Dict, 
                 train_pairs: List[Dict], test_pairs: List[Dict]):
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
    
    def __repr__(self) -> str:
        return f"SolveData(task_id='{self.task_id}', user='{self.solve_metadata.get('user_id', 'unknown')}', train_pairs={len(self.train_pairs)}, test_pairs={len(self.test_pairs)})"


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