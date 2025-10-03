#!/usr/bin/env python3
"""
Export unified solve data as individual JSON files.

This script consolidates data from multiple sources into single, comprehensive
JSON files that contain all information needed to understand and use each puzzle solve.

Data Sources Consolidated:
- solves table (metadata, timing, user info)
- thoughts table (step-by-step reasoning)
- ARC-AGI JSON files (original puzzle structure)
- Color maps and character mappings
- Spatial transformation metadata

Output: One JSON file per solve in the format:
solves_export/
├── solve_001_task_abc123.json
├── solve_002_task_def456.json
└── ...
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import argparse


class UnifiedSolveExporter:
    """Exports solve data into unified JSON files."""
    
    def __init__(self, data_dir: str, source_folder: str, output_dir: str):
        """
        Initialize the exporter.
        
        Args:
            data_dir: Directory containing solves.db and thoughts.db
            source_folder: Directory containing ARC-AGI JSON files
            output_dir: Directory to write unified JSON files
        """
        self.data_dir = Path(data_dir)
        self.source_folder = Path(source_folder)
        self.output_dir = Path(output_dir)
        
        # Database paths
        self.solves_db_path = self.data_dir / "solves.db"
        self.thoughts_db_path = self.data_dir / "thoughts.db"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_arc_agi_task(self, task_id: str) -> Optional[Dict]:
        """Load the original ARC-AGI task JSON file."""
        # Try different possible file extensions and locations
        candidate_bases = [
            self.source_folder,
            self.source_folder / "training",
            self.source_folder / "evaluation",
        ]

        # 1) Direct checks in common bases
        for base in candidate_bases:
            possible_paths = [
                base / f"{task_id}.json",
                base / f"task_{task_id}.json",
                base / f"{task_id}.txt",
                base / f"task_{task_id}.txt",
            ]
            for path in possible_paths:
                if path.exists():
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Warning: Could not parse {path}: {e}")
                        continue

        # 2) Recursive fallback search in bases for exact filename matches
        for base in candidate_bases:
            if base.exists():
                try:
                    # Prefer JSON, then TXT
                    target_names = {f"{task_id}.json", f"task_{task_id}.json", f"{task_id}.txt", f"task_{task_id}.txt"}
                    for p in base.rglob("*.*"):
                        if p.name in target_names and p.is_file():
                            try:
                                with open(p, 'r', encoding='utf-8') as f:
                                    return json.load(f)
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"Warning: Could not parse {p}: {e}")
                                continue
                except Exception as e:
                    print(f"Warning: Recursive search error under {base}: {e}")

        print(f"Warning: Could not find ARC-AGI task file for {task_id}")
        return None
    
    def load_solve_metadata(self) -> List[Dict]:
        """Load all solve metadata from the database."""
        if not self.solves_db_path.exists():
            print(f"Error: Database not found at {self.solves_db_path}")
            return []
        
        conn = sqlite3.connect(self.solves_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT id, task_id, user_id, order_map_type, order_map, 
                       color_map, metadata_labels, start_time, end_time
                FROM solves
                ORDER BY id
            """)
            
            solves = []
            for row in cursor.fetchall():
                solve = {
                    'solve_id': row[0],
                    'task_id': row[1],
                    'user_id': row[2],
                    'order_map_type': row[3],
                    'order_map': json.loads(row[4]) if row[4] else {},
                    'color_map': json.loads(row[5]) if row[5] else {},
                    'metadata_labels': json.loads(row[6]) if row[6] else [],
                    'start_time': row[7],
                    'end_time': row[8]
                }
                solves.append(solve)
            
            return solves
        finally:
            conn.close()
    
    def load_thoughts_for_solve(self, solve_id: int) -> List[Dict]:
        """Load all thoughts for a specific solve."""
        if not self.thoughts_db_path.exists():
            return []
        
        conn = sqlite3.connect(self.thoughts_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT pair_label, pair_type, sequence_index, thought_text, cleaned_thought_text
                FROM thoughts
                WHERE solve_id = ?
                ORDER BY sequence_index
            """, (solve_id,))
            
            thoughts = []
            for row in cursor.fetchall():
                thought = {
                    'pair_label': row[0],
                    'pair_type': row[1],  # 'train' or 'test'
                    'sequence_index': row[2],
                    'thought_text': row[3],
                    'cleaned_thought_text': row[4]
                }
                thoughts.append(thought)
            
            return thoughts
        finally:
            conn.close()
    
    def organize_thoughts_by_pairs(self, thoughts: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize thoughts by pair label for easier access."""
        organized = {}
        for thought in thoughts:
            pair_label = thought['pair_label']
            if pair_label not in organized:
                organized[pair_label] = []
            organized[pair_label].append(thought)
        
        # Sort by sequence index within each pair
        for pair_label in organized:
            organized[pair_label].sort(key=lambda x: x['sequence_index'])
        
        return organized
    
    def create_unified_solve(self, solve_metadata: Dict, thoughts: List[Dict], 
                           arc_agi_task: Optional[Dict]) -> Dict[str, Any]:
        """Create a unified solve representation."""
        
        # Organize thoughts by pairs
        thoughts_by_pairs = self.organize_thoughts_by_pairs(thoughts)
        
        # Calculate solve duration
        solve_duration = None
        if solve_metadata['start_time'] and solve_metadata['end_time']:
            try:
                start = datetime.fromisoformat(solve_metadata['start_time'])
                end = datetime.fromisoformat(solve_metadata['end_time'])
                solve_duration = int((end - start).total_seconds())
            except (ValueError, TypeError):
                pass
        
        # Create the unified structure
        unified_solve = {
            # Metadata
            'metadata': {
                'solve_id': solve_metadata['solve_id'],
                'task_id': solve_metadata['task_id'],
                'user_id': solve_metadata['user_id'],
                'solve_duration_seconds': solve_duration,
                'start_time': solve_metadata['start_time'],
                'end_time': solve_metadata['end_time'],
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0'
            },
            
            # Original ARC-AGI task structure
            'arc_agi_task': arc_agi_task or {},
            
            # Solve-specific metadata
            'solve_configuration': {
                'order_map_type': solve_metadata['order_map_type'],
                'order_map': solve_metadata['order_map'],
                'color_map': solve_metadata['color_map'],
                'metadata_labels': solve_metadata['metadata_labels']
            },
            
            # Training pairs with thoughts
            'training_pairs': [],
            
            # Test pairs with thoughts
            'test_pairs': [],
            
            # Thoughts organized by pair label
            'thoughts': {},
            
            # Summary statistics
            'summary': {
                'total_training_pairs': 0,
                'total_test_pairs': 0,
                'total_thoughts': len(thoughts),
                'has_cleaned_thoughts': any(t.get('cleaned_thought_text') for t in thoughts),
                'has_arc_agi_task': arc_agi_task is not None
            }
        }
        
        # Process training and test order as lists of pair labels
        train_order: list = []
        test_order: list = []

        # Derive order from thoughts sequence, if available
        if thoughts:
            # Sort thoughts globally by sequence_index
            sorted_thoughts = sorted(thoughts, key=lambda t: t.get('sequence_index', 0))
            seen_train = set()
            seen_test = set()
            for t in sorted_thoughts:
                label = t.get('pair_label')
                ptype = t.get('pair_type')
                if not label or not ptype:
                    continue
                if ptype == 'train' and label not in seen_train:
                    train_order.append(label)
                    seen_train.add(label)
                elif ptype == 'test' and label not in seen_test:
                    test_order.append(label)
                    seen_test.add(label)

        # Fallback: infer from arc_agi_task structure if no thoughts-based order
        if not train_order and not test_order and arc_agi_task:
            if 'pairs' in arc_agi_task:
                pairs = arc_agi_task['pairs']
                pair_labels = sorted(pairs.keys())
                train_count = arc_agi_task.get('train_count', len(pair_labels) - 1)
                train_order = pair_labels[:train_count]
                test_order = pair_labels[train_count:]
            else:
                # Legacy train/test lists
                train_count = len(arc_agi_task.get('train', []))
                test_count = len(arc_agi_task.get('test', []))
                train_order = [chr(ord('a') + i) for i in range(train_count)]
                test_order = [chr(ord('a') + i) for i in range(test_count)]

        # Assign simplified lists
        unified_solve['training_pairs'] = train_order
        unified_solve['test_pairs'] = test_order
        
        # Populate thoughts dictionary organized by pair label
        for pair_label, pair_thoughts in thoughts_by_pairs.items():
            # Concatenate raw thought_texts in sequence order into a single string
            ordered = sorted(pair_thoughts, key=lambda t: t.get('sequence_index', 0))
            texts = [t.get('thought_text', '') for t in ordered if t.get('thought_text')]
            unified_solve['thoughts'][pair_label] = "\n".join(texts)
        
        # Update summary
        unified_solve['summary']['total_training_pairs'] = len(unified_solve['training_pairs'])
        unified_solve['summary']['total_test_pairs'] = len(unified_solve['test_pairs'])
        
        return unified_solve
    
    def export_solve(self, solve_metadata: Dict, output_filename: str) -> bool:
        """Export a single solve to a JSON file."""
        try:
            # Load thoughts for this solve
            thoughts = self.load_thoughts_for_solve(solve_metadata['solve_id'])
            
            # Load ARC-AGI task
            arc_agi_task = self.load_arc_agi_task(solve_metadata['task_id'])
            
            # Create unified solve
            unified_solve = self.create_unified_solve(solve_metadata, thoughts, arc_agi_task)
            
            # Write to file
            output_path = self.output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(unified_solve, f, indent=2, ensure_ascii=False)
            
            print(f"Exported solve {solve_metadata['solve_id']} to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting solve {solve_metadata['solve_id']}: {e}")
            return False
    
    def export_all_solves(self) -> Dict[str, int]:
        """Export all solves to individual JSON files."""
        print(f"Starting export from {self.data_dir} to {self.output_dir}")
        
        # Load all solve metadata
        solves = self.load_solve_metadata()
        if not solves:
            print("No solves found in database")
            return {'total': 0, 'successful': 0, 'failed': 0}
        
        print(f"Found {len(solves)} solves to export")
        
        # Export each solve
        successful = 0
        failed = 0
        
        for solve in solves:
            # Create filename: solve_001_task_abc123.json
            solve_id_str = str(solve['solve_id']).zfill(3)
            task_id = solve['task_id']
            filename = f"solve_{solve_id_str}_task_{task_id}.json"
            
            if self.export_solve(solve, filename):
                successful += 1
            else:
                failed += 1
        
        results = {
            'total': len(solves),
            'successful': successful,
            'failed': failed
        }
        
        print(f"\nExport complete:")
        print(f"  Total solves: {results['total']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        
        return results
    
    def create_index_file(self) -> None:
        """Create an index file listing all exported solves."""
        json_files = list(self.output_dir.glob("solve_*.json"))
        
        index_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_solves': len(json_files),
                'export_version': '1.0'
            },
            'solves': []
        }
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    solve_data = json.load(f)
                
                solve_info = {
                    'filename': json_file.name,
                    'solve_id': solve_data['metadata']['solve_id'],
                    'task_id': solve_data['metadata']['task_id'],
                    'user_id': solve_data['metadata']['user_id'],
                    'solve_duration_seconds': solve_data['metadata']['solve_duration_seconds'],
                    'total_training_pairs': solve_data['summary']['total_training_pairs'],
                    'total_test_pairs': solve_data['summary']['total_test_pairs'],
                    'total_thoughts': solve_data['summary']['total_thoughts'],
                    'has_cleaned_thoughts': solve_data['summary']['has_cleaned_thoughts'],
                    'has_arc_agi_task': solve_data['summary']['has_arc_agi_task']
                }
                index_data['solves'].append(solve_info)
                
            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}")
        
        # Write index file
        index_path = self.output_dir / "index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created index file: {index_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Export unified solve data as JSON files')
    parser.add_argument('--data-dir', default='data/solves_and_thoughts',
                       help='Directory containing solves.db and thoughts.db')
    parser.add_argument('--source-folder', default='data/arc_agi_2_reformatted',
                       help='Directory containing ARC-AGI JSON files')
    parser.add_argument('--output-dir', default='solves_export',
                       help='Directory to write unified JSON files')
    parser.add_argument('--create-index', action='store_true',
                       help='Create an index.json file listing all exports')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = UnifiedSolveExporter(
        data_dir=args.data_dir,
        source_folder=args.source_folder,
        output_dir=args.output_dir
    )
    
    # Export all solves
    results = exporter.export_all_solves()
    
    # Create index if requested
    if args.create_index:
        exporter.create_index_file()
    
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
