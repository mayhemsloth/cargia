import os
import sqlite3
import json
import sys
from cargia.data_manager import get_repo_root, log_error

def delete_solves(solve_ids):
    """Delete specific solves by ID and their associated thoughts.
    
    Args:
        solve_ids (list): List of solve IDs to delete
    """
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
            # Verify the solves exist
            placeholders = ','.join('?' * len(solve_ids))
            solves_cursor.execute(f"SELECT id FROM solves WHERE id IN ({placeholders})", solve_ids)
            existing_ids = [row[0] for row in solves_cursor.fetchall()]
            
            if not existing_ids:
                print("No matching solves found.")
                return
            
            print(f"Found {len(existing_ids)} matching solves.")
            
            # Delete associated thoughts
            for solve_id in existing_ids:
                thoughts_cursor.execute("DELETE FROM thoughts WHERE solve_id = ?", (solve_id,))
                print(f"Deleted thoughts for solve ID: {solve_id}")
            
            # Delete the solves
            solves_cursor.execute(f"DELETE FROM solves WHERE id IN ({placeholders})", solve_ids)
            
            # Commit changes
            solves_conn.commit()
            thoughts_conn.commit()
            
            print(f"Successfully removed {len(existing_ids)} solves and their associated thoughts.")
            
        finally:
            solves_conn.close()
            thoughts_conn.close()
            
    except Exception as e:
        log_error("Failed to delete solves", e)
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python delete_solves.py <solve_id1> [solve_id2 ...]")
        print("Example: python delete_solves.py 1 2 3")
        sys.exit(1)
    
    try:
        # Convert command line arguments to integers
        solve_ids = [int(arg) for arg in sys.argv[1:]]
        delete_solves(solve_ids)
    except ValueError:
        print("Error: All solve IDs must be integers")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 