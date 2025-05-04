import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import uuid
import sqlite3
import traceback
import random

def get_repo_root():
    """Get the absolute path to the cargia directory."""
    # Get the directory containing this file
    return os.path.dirname(os.path.abspath(__file__))

def log_error(message, error=None):
    """Helper function to log errors with traceback"""
    print(f"\nERROR: {message}")
    if error:
        print(f"Exception: {str(error)}")
        print("Traceback:")
        traceback.print_exc()
    print("-" * 80)

class DataManager:
    def __init__(self, data_dir: str, source_folder: str):
        """Initialize the data manager with the specified data directory and source folder."""
        # Get cargia directory
        cargia_dir = get_repo_root()
        
        # Convert paths to absolute paths
        self.data_dir = os.path.abspath(data_dir)
        self.source_folder = os.path.abspath(source_folder)
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.source_folder, exist_ok=True)
        
        # Set database paths
        self.solves_db_path = os.path.join(self.data_dir, "solves.db")
        self.thoughts_db_path = os.path.join(self.data_dir, "thoughts.db")
        self.training_tasks_path = os.path.join(self.source_folder, "training.txt")
        
        # Create databases if they don't exist
        self._create_databases()
        
        # Load settings
        self.settings_path = os.path.join(cargia_dir, "settings.json")
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from the settings file."""
        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r') as f:
                settings = json.load(f)
                self.current_user = settings.get('user', '')
                self.data_dir = settings.get('data_dir', 'data')
                self.order_map_type = settings.get('order_map_type', 'default')
                self.backup_dir = settings.get('backup_dir', 'data/backup')
        else:
            self.current_user = ''
            self.data_dir = 'data'
            self.order_map_type = 'default'
            self.backup_dir = 'data/backup'
    
    def save_settings(self):
        """Save current settings to the settings file."""
        settings = {
            'user': self.current_user,
            'data_dir': self.data_dir,
            'order_map_type': self.order_map_type
        }
        with open(self.settings_path, 'w') as f:
            json.dump(settings, f, indent=4)
    
    def _create_databases(self):
        """Create the solves and thoughts databases if they don't exist."""
        # Create solves database
        if not os.path.exists(self.solves_db_path):
            conn = sqlite3.connect(self.solves_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE solves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    order_map_type TEXT NOT NULL,
                    order_map TEXT NOT NULL,
                    color_map TEXT,
                    metadata_labels TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    UNIQUE(task_id, user_id, order_map)
                )
            """)
            conn.commit()
            conn.close()
        else:
            # Check if we need to add the new columns
            conn = sqlite3.connect(self.solves_db_path)
            cursor = conn.cursor()
            try:
                # Get table info
                cursor.execute("PRAGMA table_info(solves)")
                columns = {row[1] for row in cursor.fetchall()}
                
                # Add missing columns if needed
                if 'order_map_type' not in columns:
                    cursor.execute("ALTER TABLE solves ADD COLUMN order_map_type TEXT NOT NULL DEFAULT 'default'")
                if 'order_map' not in columns:
                    cursor.execute("ALTER TABLE solves ADD COLUMN order_map TEXT NOT NULL DEFAULT '{}'")
                if 'color_map' not in columns:
                    cursor.execute("ALTER TABLE solves ADD COLUMN color_map TEXT")
                if 'metadata_labels' not in columns:
                    cursor.execute("ALTER TABLE solves ADD COLUMN metadata_labels TEXT")
                if 'start_time' not in columns:
                    cursor.execute("ALTER TABLE solves ADD COLUMN start_time TEXT")
                if 'end_time' not in columns:
                    cursor.execute("ALTER TABLE solves ADD COLUMN end_time TEXT")
                
                conn.commit()
            finally:
                conn.close()
        
        # Create thoughts database
        if not os.path.exists(self.thoughts_db_path):
            conn = sqlite3.connect(self.thoughts_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE thoughts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    solve_id INTEGER NOT NULL,
                    pair_label TEXT NOT NULL,
                    pair_type TEXT NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    thought_text TEXT NOT NULL,
                    FOREIGN KEY (solve_id) REFERENCES solves(id)
                )
            """)
            conn.commit()
            conn.close()
    
    def get_next_task(self) -> Optional[Dict]:
        """Get the next unsolved task with a unique order map."""
        try:
            # Read training tasks from training.txt
            training_tasks_path = os.path.join(self.source_folder, "training.txt")
            if not os.path.exists(training_tasks_path):
                return None
            
            with open(training_tasks_path, 'r') as f:
                task_ids = [line.strip() for line in f if line.strip()]
            
            # For each task, try to find one that hasn't been solved with this order
            for task_id in task_ids:
                task_path = os.path.join(self.source_folder, "training", f"{task_id}.json")
                if not os.path.exists(task_path):
                    continue
                
                # Load task data
                with open(task_path, 'r') as f:
                    task_data = json.load(f)
                
                # Generate order map based on settings
                order_map = self._generate_order_map(task_data)
                
                # Check if this task+order combination exists for this user
                conn = sqlite3.connect(self.solves_db_path)
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT id FROM solves 
                        WHERE task_id = ? AND user_id = ? AND order_map = ?
                    """, (task_id, self.current_user, json.dumps(order_map)))
                    if cursor.fetchone() is None:
                        # Found a unique task+order combination
                        # Reformat task data to match expected structure
                        reformatted_task = {
                            'train': [task_data['pairs'][key] for key in order_map['train']],
                            'test': [task_data['pairs'][key] for key in order_map['test']]
                        }
                        return {
                            "task_id": task_id,
                            "task_data": reformatted_task,
                            "order_map": order_map
                        }
                finally:
                    conn.close()
            
            return None
        except Exception as e:
            log_error("Failed to get next task", e)
            return None
    
    def start_solve(self, task_id: str, user_id: str) -> Optional[int]:
        """Start a new solve for a task and return the solve ID."""
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO solves (task_id, user_id, status)
                VALUES (?, ?, 'in_progress')
            """, (task_id, user_id))
            solve_id = cursor.lastrowid
            conn.commit()
            return solve_id
        except sqlite3.IntegrityError:
            # Task already being solved by this user
            return None
        finally:
            conn.close()
    
    def save_thought(self, solve_id: int, pair_index: int, thought_text: str):
        """Save a thought for a specific pair in a solve."""
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO thoughts (solve_id, pair_index, thought_text)
            VALUES (?, ?, ?)
        """, (solve_id, pair_index, thought_text))
        conn.commit()
        conn.close()
    
    def _validate_backup_dir(self):
        """Validate that the backup directory is safe to use.
        Returns True if the directory is safe, False otherwise."""
        try:
            # Check if backup_dir is a subdirectory of data_dir
            backup_abs = os.path.abspath(self.backup_dir)
            data_abs = os.path.abspath(self.data_dir)
            if not backup_abs.startswith(data_abs):
                log_error(f"Backup directory {self.backup_dir} is not a subdirectory of data directory {self.data_dir}")
                return False
            
            # Check if backup_dir exists and is a directory
            if os.path.exists(self.backup_dir) and not os.path.isdir(self.backup_dir):
                log_error(f"Backup directory {self.backup_dir} exists but is not a directory")
                return False
            
            # If backup_dir exists, check its contents
            if os.path.exists(self.backup_dir):
                for item in os.listdir(self.backup_dir):
                    item_path = os.path.join(self.backup_dir, item)
                    # Only allow timestamped directories and .db files
                    if os.path.isdir(item_path):
                        if not (item.replace("_", "").isdigit() and len(item) == 15):  # YYYYMMDD_HHMMSS format
                            log_error(f"Found non-backup directory in backup folder: {item}")
                            return False
                    elif os.path.isfile(item_path):
                        if not (item.endswith('.db') and item in ['solves.db', 'thoughts.db']):
                            log_error(f"Found non-backup file in backup folder: {item}")
                            return False
            
            return True
        except Exception as e:
            log_error("Failed to validate backup directory", e)
            return False

    def backup_databases(self):
        """Create a backup of both databases in a timestamped folder.
        Maintains a maximum of 20 backups by removing the oldest ones when needed."""
        try:
            # Validate backup directory
            if not self._validate_backup_dir():
                log_error("Backup directory validation failed, aborting backup")
                return
            
            # Create backup directory if it doesn't exist
            os.makedirs(self.backup_dir, exist_ok=True)
            
            # Create timestamped backup folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder = os.path.join(self.backup_dir, timestamp)
            os.makedirs(backup_folder, exist_ok=True)
            
            # Copy both databases
            import shutil
            shutil.copy2(self.solves_db_path, os.path.join(backup_folder, "solves.db"))
            shutil.copy2(self.thoughts_db_path, os.path.join(backup_folder, "thoughts.db"))
            
            # Get list of all backup folders
            backup_folders = [d for d in os.listdir(self.backup_dir) 
                            if os.path.isdir(os.path.join(self.backup_dir, d)) 
                            and d.replace("_", "").isdigit()  # Only consider timestamped folders
                            and len(d) == 15]  # Ensure proper format (YYYYMMDD_HHMMSS)
            
            # Sort by timestamp (folder name)
            backup_folders.sort()
            
            # Remove oldest backups if we have more than 20
            while len(backup_folders) > 20:
                oldest_backup = backup_folders.pop(0)
                oldest_path = os.path.join(self.backup_dir, oldest_backup)
                try:
                    # Double check the folder only contains our backup files
                    contents = os.listdir(oldest_path)
                    if not all(f in ['solves.db', 'thoughts.db'] for f in contents):
                        log_error(f"Found unexpected files in backup folder {oldest_backup}, skipping deletion")
                        continue
                    
                    shutil.rmtree(oldest_path)
                    print(f"Removed oldest backup: {oldest_backup}")
                except Exception as e:
                    log_error(f"Failed to remove old backup {oldest_backup}", e)
            
            print(f"Created backup in {backup_folder}")
        except Exception as e:
            log_error("Failed to create database backup", e)
    
    def complete_solve(self, solve_id: int):
        """Mark a solve as completed by setting its end time."""
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE solves
            SET end_time = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), solve_id))
        conn.commit()
        conn.close()
        
        # Create backup after completing a solve
        self.backup_databases()
    
    def save_databases(self):
        """Save both databases to CSV files."""
        self.solves_df.to_csv(os.path.join(self.data_dir, "solves.csv"), index=False)
        self.thoughts_df.to_csv(os.path.join(self.data_dir, "thoughts.csv"), index=False)
    
    def set_current_user(self, username: str):
        """Set the current user for the session and save to settings."""
        self.current_user = username
        self.save_settings()
    
    def create_solve(
        self,
        task_id: str,
        order_map: Dict[str, List[str]],
        order_map_type: str,
        color_map: Dict[int, str],
        metadata_labels: Dict[str, bool]
    ) -> int:
        """Create a new solve entry and return its solve_id."""
        if not self.current_user:
            raise ValueError("No current user set. Call set_current_user first.")
        
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO solves (task_id, user_id, order_map_type, order_map, color_map, metadata_labels, start_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                self.current_user,
                order_map_type,
                json.dumps(order_map),
                json.dumps(color_map),
                json.dumps(metadata_labels),
                datetime.now().isoformat()
            ))
            solve_id = cursor.lastrowid
            conn.commit()
            return solve_id
        finally:
            conn.close()
    
    def add_thought(
        self,
        solve_id: int,
        pair_label: str,
        pair_type: str,
        sequence_index: int,
        thought_text: str
    ):
        """Add a thought entry to the thoughts database."""
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO thoughts (solve_id, pair_label, pair_type, sequence_index, thought_text)
                VALUES (?, ?, ?, ?, ?)
            """, (solve_id, pair_label, pair_type, sequence_index, thought_text))
            conn.commit()
        finally:
            conn.close()
    
    def get_solve(self, solve_id: int) -> Dict:
        """Get a solve entry by its solve_id."""
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT task_id, user_id, order_map, color_map, metadata_labels, start_time, end_time
                FROM solves
                WHERE id = ?
            """, (solve_id,))
            row = cursor.fetchone()
            
            if not row:
                raise ValueError(f"Solve not found with ID: {solve_id}")
            
            return {
                'solve_id': solve_id,
                'task_id': row[0],
                'user': row[1],
                'order_map': json.loads(row[2]),
                'color_map': json.loads(row[3]),
                'metadata_labels': json.loads(row[4]),
                'start_time': row[5],
                'end_time': row[6]
            }
        finally:
            conn.close()
    
    def get_thoughts(self, solve_id: int) -> List[Dict]:
        """Get all thoughts for a specific solve."""
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT pair_label, pair_type, sequence_index, thought_text
                FROM thoughts
                WHERE solve_id = ?
                ORDER BY sequence_index
            """, (solve_id,))
            
            thoughts = []
            for row in cursor.fetchall():
                thoughts.append({
                    'pair_label': row[0],
                    'pair_type': row[1],
                    'sequence_index': row[2],
                    'thought_text': row[3]
                })
            return thoughts
        finally:
            conn.close()
    
    def get_solves_by_task(self, task_id: str) -> List[Dict]:
        """Get all solves for a specific task."""
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, user_id, order_map, color_map, metadata_labels, start_time, end_time
                FROM solves
                WHERE task_id = ?
            """, (task_id,))
            
            solves = []
            for row in cursor.fetchall():
                solves.append({
                    'solve_id': row[0],
                    'user': row[1],
                    'order_map': json.loads(row[2]),
                    'color_map': json.loads(row[3]),
                    'metadata_labels': json.loads(row[4]),
                    'start_time': row[5],
                    'end_time': row[6]
                })
            return solves
        finally:
            conn.close()
    
    def get_solves_by_user(self, username: str) -> List[Dict]:
        """Get all solves by a specific user."""
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, task_id, order_map, color_map, metadata_labels, start_time, end_time
                FROM solves
                WHERE user_id = ?
            """, (username,))
            
            solves = []
            for row in cursor.fetchall():
                solves.append({
                    'solve_id': row[0],
                    'task_id': row[1],
                    'order_map': json.loads(row[2]),
                    'color_map': json.loads(row[3]),
                    'metadata_labels': json.loads(row[4]),
                    'start_time': row[5],
                    'end_time': row[6]
                })
            return solves
        finally:
            conn.close()
    
    def set_order_map_type(self, order_map_type: str):
        """Set the order map type and save to settings."""
        if order_map_type not in ['default', 'random']:
            raise ValueError("order_map_type must be either 'default' or 'random'")
        self.order_map_type = order_map_type
        self.save_settings()
    
    def _generate_order_map(self, task_data: Dict) -> Dict[str, List[str]]:
        """Generate order map based on settings."""
        try:
            if  self.order_map_type == 'default':
                # Use the default splits from the task
                return task_data['default_splits']
            else:  # 'random'
                # Get all available pair keys
                pair_keys = list(task_data['pairs'].keys())
                # Get the number of train/test pairs from default splits
                num_train = len(task_data['default_splits']['train'])
                num_test = len(task_data['default_splits']['test'])
                
                # Randomly shuffle the pair keys
                random.shuffle(pair_keys)
                
                # Split into train and test maintaining the same counts
                return {
                    'train': pair_keys[:num_train],
                    'test': pair_keys[num_train:num_train + num_test]
                }
        except Exception as e:
            log_error("Failed to generate order map", e)
            return {'train': [], 'test': []}
    
    def update_metadata_labels(self, solve_id: int, metadata_labels: Dict[str, bool]):
        """Update the metadata labels for a solve."""
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE solves
                SET metadata_labels = ?
                WHERE id = ?
            """, (json.dumps(metadata_labels), solve_id))
            conn.commit()
        finally:
            conn.close() 