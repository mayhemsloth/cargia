import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import uuid

class DataManager:
    def __init__(self, data_dir: str = "data"):
        """Initialize the data manager with the specified data directory."""
        self.data_dir = data_dir
        self.solves_df = None
        self.thoughts_df = None
        self.current_user = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize or load existing databases
        self._initialize_databases()
    
    def _initialize_databases(self):
        """Initialize or load the solves and thoughts databases."""
        solves_path = os.path.join(self.data_dir, "solves.csv")
        thoughts_path = os.path.join(self.data_dir, "thoughts.csv")
        
        # Initialize solves DataFrame if it doesn't exist
        if os.path.exists(solves_path):
            self.solves_df = pd.read_csv(solves_path)
        else:
            self.solves_df = pd.DataFrame(columns=[
                'data_entry_id',
                'task_id',
                'order_map',
                'user',
                'color_map',
                'metadata_labels',
                'start_time',
                'end_time'
            ])
        
        # Initialize thoughts DataFrame if it doesn't exist
        if os.path.exists(thoughts_path):
            self.thoughts_df = pd.read_csv(thoughts_path)
        else:
            self.thoughts_df = pd.DataFrame(columns=[
                'data_entry_id',
                'pair_label',
                'pair_type',
                'sequence_index',
                'thought_text'
            ])
    
    def save_databases(self):
        """Save both databases to CSV files."""
        self.solves_df.to_csv(os.path.join(self.data_dir, "solves.csv"), index=False)
        self.thoughts_df.to_csv(os.path.join(self.data_dir, "thoughts.csv"), index=False)
    
    def set_current_user(self, username: str):
        """Set the current user for the session."""
        self.current_user = username
    
    def create_solve(
        self,
        task_id: str,
        order_map: Dict[str, List[str]],
        color_map: Dict[int, str],
        metadata_labels: Dict[str, bool]
    ) -> int:
        """Create a new solve entry and return its data_entry_id."""
        if not self.current_user:
            raise ValueError("No current user set. Call set_current_user first.")
        
        # Generate a new data_entry_id
        data_entry_id = len(self.solves_df) + 1
        
        # Create new solve entry
        new_solve = {
            'data_entry_id': data_entry_id,
            'task_id': task_id,
            'order_map': json.dumps(order_map),
            'user': self.current_user,
            'color_map': json.dumps(color_map),
            'metadata_labels': json.dumps(metadata_labels),
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        # Add to solves DataFrame
        self.solves_df = pd.concat([
            self.solves_df,
            pd.DataFrame([new_solve])
        ], ignore_index=True)
        
        return data_entry_id
    
    def add_thought(
        self,
        data_entry_id: int,
        pair_label: str,
        pair_type: str,
        sequence_index: int,
        thought_text: str
    ):
        """Add a thought entry to the thoughts database."""
        # Validate data_entry_id exists
        if data_entry_id not in self.solves_df['data_entry_id'].values:
            raise ValueError(f"Invalid data_entry_id: {data_entry_id}")
        
        # Create new thought entry
        new_thought = {
            'data_entry_id': data_entry_id,
            'pair_label': pair_label,
            'pair_type': pair_type,
            'sequence_index': sequence_index,
            'thought_text': thought_text
        }
        
        # Add to thoughts DataFrame
        self.thoughts_df = pd.concat([
            self.thoughts_df,
            pd.DataFrame([new_thought])
        ], ignore_index=True)
    
    def complete_solve(self, data_entry_id: int):
        """Mark a solve as complete by setting its end_time."""
        if data_entry_id not in self.solves_df['data_entry_id'].values:
            raise ValueError(f"Invalid data_entry_id: {data_entry_id}")
        
        # Update end_time
        self.solves_df.loc[
            self.solves_df['data_entry_id'] == data_entry_id,
            'end_time'
        ] = datetime.now().isoformat()
        
        # Save databases
        self.save_databases()
    
    def get_solve(self, data_entry_id: int) -> Dict:
        """Get a solve entry by its data_entry_id."""
        solve = self.solves_df[self.solves_df['data_entry_id'] == data_entry_id].iloc[0]
        return {
            'data_entry_id': solve['data_entry_id'],
            'task_id': solve['task_id'],
            'order_map': json.loads(solve['order_map']),
            'user': solve['user'],
            'color_map': json.loads(solve['color_map']),
            'metadata_labels': json.loads(solve['metadata_labels']),
            'start_time': solve['start_time'],
            'end_time': solve['end_time']
        }
    
    def get_thoughts(self, data_entry_id: int) -> List[Dict]:
        """Get all thoughts for a specific solve."""
        thoughts = self.thoughts_df[self.thoughts_df['data_entry_id'] == data_entry_id]
        return thoughts.to_dict('records')
    
    def get_solves_by_task(self, task_id: str) -> List[Dict]:
        """Get all solves for a specific task."""
        solves = self.solves_df[self.solves_df['task_id'] == task_id]
        return solves.to_dict('records')
    
    def get_solves_by_user(self, username: str) -> List[Dict]:
        """Get all solves by a specific user."""
        solves = self.solves_df[self.solves_df['user'] == username]
        return solves.to_dict('records') 