import os
import sqlite3
import json
from cargia.data_manager import get_repo_root, log_error

def cleanup_incomplete_solves():
    """Remove all solves without an end_time and their associated thoughts."""
    try:
        # Get cargia directory
        cargia_dir = get_repo_root()
        
        # Load settings to get data directory
        settings_path = os.path.join(cargia_dir, "settings.json")
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        data_dir = os.path.abspath(settings.get('data_dir', os.path.join(cargia_dir, "data")))
        
        # Set database paths
        solves_db_path = os.path.join(data_dir, "solves.db")
        thoughts_db_path = os.path.join(data_dir, "thoughts.db")
        
        # Connect to databases
        solves_conn = sqlite3.connect(solves_db_path)
        thoughts_conn = sqlite3.connect(thoughts_db_path)
        solves_cursor = solves_conn.cursor()
        thoughts_cursor = thoughts_conn.cursor()
        
        try:
            # Get all solve IDs without end_time
            solves_cursor.execute("SELECT id FROM solves WHERE end_time IS NULL")
            incomplete_solve_ids = [row[0] for row in solves_cursor.fetchall()]
            
            if not incomplete_solve_ids:
                print("No incomplete solves found.")
                return
            
            print(f"Found {len(incomplete_solve_ids)} incomplete solves.")
            
            # Delete associated thoughts
            for solve_id in incomplete_solve_ids:
                thoughts_cursor.execute("DELETE FROM thoughts WHERE solve_id = ?", (solve_id,))
            
            # Delete incomplete solves
            solves_cursor.execute("DELETE FROM solves WHERE end_time IS NULL")
            
            # Commit changes
            solves_conn.commit()
            thoughts_conn.commit()
            
            print(f"Successfully removed {len(incomplete_solve_ids)} incomplete solves and their associated thoughts.")
            
        finally:
            solves_conn.close()
            thoughts_conn.close()
            
    except Exception as e:
        log_error("Failed to cleanup incomplete solves", e)
        raise

if __name__ == "__main__":
    cleanup_incomplete_solves() 