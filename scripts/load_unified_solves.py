#!/usr/bin/env python3
"""
Load and work with unified solve JSON files.

This script provides utilities for loading, filtering, and analyzing
the unified solve data exported by export_unified_solves.py.

Example usage:
    python load_unified_solves.py --solves-dir solves_export --list-solves
    python load_unified_solves.py --solves-dir solves_export --filter-user "Thomas" --export-filtered filtered_solves
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime


class UnifiedSolveLoader:
    """Load and work with unified solve JSON files."""
    
    def __init__(self, solves_dir: str):
        """
        Initialize the loader.
        
        Args:
            solves_dir: Directory containing unified solve JSON files
        """
        self.solves_dir = Path(solves_dir)
        self.solves = []
        self.index_data = None
        
        # Load index if available
        index_path = self.solves_dir / "index.json"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)
    
    def load_all_solves(self) -> List[Dict]:
        """Load all solve files from the directory."""
        if self.solves:
            return self.solves
        
        json_files = list(self.solves_dir.glob("solve_*.json"))
        self.solves = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    solve_data = json.load(f)
                self.solves.append(solve_data)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        
        print(f"Loaded {len(self.solves)} solves from {self.solves_dir}")
        return self.solves
    
    def load_solve_by_id(self, solve_id: int) -> Optional[Dict]:
        """Load a specific solve by its ID."""
        solves = self.load_all_solves()
        for solve in solves:
            if solve['metadata']['solve_id'] == solve_id:
                return solve
        return None
    
    def load_solve_by_task_id(self, task_id: str) -> Optional[Dict]:
        """Load a specific solve by its task ID."""
        solves = self.load_all_solves()
        for solve in solves:
            if solve['metadata']['task_id'] == task_id:
                return solve
        return None
    
    def filter_solves(self, filter_func: Callable[[Dict], bool]) -> List[Dict]:
        """Filter solves using a custom function."""
        solves = self.load_all_solves()
        return [solve for solve in solves if filter_func(solve)]
    
    def filter_by_user(self, user_id: str) -> List[Dict]:
        """Filter solves by user ID."""
        return self.filter_solves(lambda s: s['metadata']['user_id'] == user_id)
    
    def filter_by_duration_range(self, min_seconds: int = None, max_seconds: int = None) -> List[Dict]:
        """Filter solves by solve duration."""
        def duration_filter(solve):
            duration = solve['metadata'].get('solve_duration_seconds')
            if duration is None:
                return False
            if min_seconds is not None and duration < min_seconds:
                return False
            if max_seconds is not None and duration > max_seconds:
                return False
            return True
        
        return self.filter_solves(duration_filter)
    
    def filter_by_thought_count(self, min_thoughts: int = None, max_thoughts: int = None) -> List[Dict]:
        """Filter solves by number of thoughts."""
        def thought_filter(solve):
            thought_count = solve['summary']['total_thoughts']
            if min_thoughts is not None and thought_count < min_thoughts:
                return False
            if max_thoughts is not None and thought_count > max_thoughts:
                return False
            return True
        
        return self.filter_solves(thought_filter)
    
    def filter_by_has_cleaned_thoughts(self, has_cleaned: bool = True) -> List[Dict]:
        """Filter solves by whether they have cleaned thoughts."""
        return self.filter_solves(lambda s: s['summary']['has_cleaned_thoughts'] == has_cleaned)
    
    def filter_by_has_arc_agi_task(self, has_task: bool = True) -> List[Dict]:
        """Filter solves by whether they have ARC-AGI task data."""
        return self.filter_solves(lambda s: s['summary']['has_arc_agi_task'] == has_task)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all solves."""
        solves = self.load_all_solves()
        if not solves:
            return {}
        
        # Basic counts
        total_solves = len(solves)
        users = set(solve['metadata']['user_id'] for solve in solves)
        tasks = set(solve['metadata']['task_id'] for solve in solves)
        
        # Duration statistics
        durations = [solve['metadata'].get('solve_duration_seconds') 
                    for solve in solves 
                    if solve['metadata'].get('solve_duration_seconds') is not None]
        
        # Thought statistics
        thought_counts = [solve['summary']['total_thoughts'] for solve in solves]
        
        # Pair statistics
        train_pair_counts = [solve['summary']['total_training_pairs'] for solve in solves]
        test_pair_counts = [solve['summary']['total_test_pairs'] for solve in solves]
        
        stats = {
            'total_solves': total_solves,
            'unique_users': len(users),
            'unique_tasks': len(tasks),
            'users': list(users),
            'tasks': list(tasks),
            'duration_stats': {
                'count': len(durations),
                'min': min(durations) if durations else None,
                'max': max(durations) if durations else None,
                'avg': sum(durations) / len(durations) if durations else None
            },
            'thought_stats': {
                'min': min(thought_counts) if thought_counts else 0,
                'max': max(thought_counts) if thought_counts else 0,
                'avg': sum(thought_counts) / len(thought_counts) if thought_counts else 0
            },
            'pair_stats': {
                'train_pairs': {
                    'min': min(train_pair_counts) if train_pair_counts else 0,
                    'max': max(train_pair_counts) if train_pair_counts else 0,
                    'avg': sum(train_pair_counts) / len(train_pair_counts) if train_pair_counts else 0
                },
                'test_pairs': {
                    'min': min(test_pair_counts) if test_pair_counts else 0,
                    'max': max(test_pair_counts) if test_pair_counts else 0,
                    'avg': sum(test_pair_counts) / len(test_pair_counts) if test_pair_counts else 0
                }
            },
            'data_quality': {
                'has_cleaned_thoughts': sum(1 for s in solves if s['summary']['has_cleaned_thoughts']),
                'has_arc_agi_task': sum(1 for s in solves if s['summary']['has_arc_agi_task']),
                'has_duration': len(durations)
            }
        }
        
        return stats
    
    def export_filtered_solves(self, filtered_solves: List[Dict], output_dir: str) -> None:
        """Export filtered solves to a new directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for solve in filtered_solves:
            solve_id = solve['metadata']['solve_id']
            task_id = solve['metadata']['task_id']
            filename = f"solve_{solve_id:03d}_task_{task_id}.json"
            
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                json.dump(solve, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(filtered_solves)} solves to {output_path}")
    
    def print_solve_summary(self, solve: Dict) -> None:
        """Print a summary of a single solve."""
        metadata = solve['metadata']
        summary = solve['summary']
        
        print(f"\nSolve ID: {metadata['solve_id']}")
        print(f"Task ID: {metadata['task_id']}")
        print(f"User: {metadata['user_id']}")
        print(f"Duration: {metadata.get('solve_duration_seconds', 'N/A')} seconds")
        print(f"Training pairs: {summary['total_training_pairs']}")
        print(f"Test pairs: {summary['total_test_pairs']}")
        print(f"Total thoughts: {summary['total_thoughts']}")
        print(f"Has cleaned thoughts: {summary['has_cleaned_thoughts']}")
        print(f"Has ARC-AGI task: {summary['has_arc_agi_task']}")
    
    def print_statistics(self) -> None:
        """Print summary statistics."""
        stats = self.get_statistics()
        if not stats:
            print("No solves found")
            return
        
        print(f"\n=== Solve Statistics ===")
        print(f"Total solves: {stats['total_solves']}")
        print(f"Unique users: {stats['unique_users']}")
        print(f"Unique tasks: {stats['unique_tasks']}")
        
        print(f"\nUsers: {', '.join(stats['users'])}")
        
        if stats['duration_stats']['count'] > 0:
            print(f"\nDuration (seconds):")
            print(f"  Min: {stats['duration_stats']['min']}")
            print(f"  Max: {stats['duration_stats']['max']}")
            print(f"  Avg: {stats['duration_stats']['avg']:.1f}")
        
        print(f"\nThoughts per solve:")
        print(f"  Min: {stats['thought_stats']['min']}")
        print(f"  Max: {stats['thought_stats']['max']}")
        print(f"  Avg: {stats['thought_stats']['avg']:.1f}")
        
        print(f"\nTraining pairs per solve:")
        print(f"  Min: {stats['pair_stats']['train_pairs']['min']}")
        print(f"  Max: {stats['pair_stats']['train_pairs']['max']}")
        print(f"  Avg: {stats['pair_stats']['train_pairs']['avg']:.1f}")
        
        print(f"\nData quality:")
        print(f"  Has cleaned thoughts: {stats['data_quality']['has_cleaned_thoughts']}/{stats['total_solves']}")
        print(f"  Has ARC-AGI task: {stats['data_quality']['has_arc_agi_task']}/{stats['total_solves']}")
        print(f"  Has duration: {stats['data_quality']['has_duration']}/{stats['total_solves']}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Load and work with unified solve JSON files')
    parser.add_argument('--solves-dir', required=True,
                       help='Directory containing unified solve JSON files')
    parser.add_argument('--list-solves', action='store_true',
                       help='List all solves with basic info')
    parser.add_argument('--statistics', action='store_true',
                       help='Show summary statistics')
    parser.add_argument('--filter-user', type=str,
                       help='Filter solves by user ID')
    parser.add_argument('--filter-duration-min', type=int,
                       help='Filter solves by minimum duration (seconds)')
    parser.add_argument('--filter-duration-max', type=int,
                       help='Filter solves by maximum duration (seconds)')
    parser.add_argument('--filter-thoughts-min', type=int,
                       help='Filter solves by minimum thought count')
    parser.add_argument('--filter-thoughts-max', type=int,
                       help='Filter solves by maximum thought count')
    parser.add_argument('--filter-has-cleaned', action='store_true',
                       help='Filter solves that have cleaned thoughts')
    parser.add_argument('--filter-has-task', action='store_true',
                       help='Filter solves that have ARC-AGI task data')
    parser.add_argument('--export-filtered', type=str,
                       help='Export filtered solves to specified directory')
    parser.add_argument('--solve-id', type=int,
                       help='Show details for specific solve ID')
    parser.add_argument('--task-id', type=str,
                       help='Show details for specific task ID')
    
    args = parser.parse_args()
    
    # Create loader
    loader = UnifiedSolveLoader(args.solves_dir)
    
    # Apply filters
    filtered_solves = loader.load_all_solves()
    
    if args.filter_user:
        filtered_solves = loader.filter_by_user(args.filter_user)
    
    if args.filter_duration_min is not None or args.filter_duration_max is not None:
        filtered_solves = loader.filter_by_duration_range(
            min_seconds=args.filter_duration_min,
            max_seconds=args.filter_duration_max
        )
    
    if args.filter_thoughts_min is not None or args.filter_thoughts_max is not None:
        filtered_solves = loader.filter_by_thought_count(
            min_thoughts=args.filter_thoughts_min,
            max_thoughts=args.filter_thoughts_max
        )
    
    if args.filter_has_cleaned:
        filtered_solves = loader.filter_by_has_cleaned_thoughts(True)
    
    if args.filter_has_task:
        filtered_solves = loader.filter_by_has_arc_agi_task(True)
    
    # Execute commands
    if args.statistics:
        loader.print_statistics()
    
    if args.list_solves:
        print(f"\n=== Filtered Solves ({len(filtered_solves)}) ===")
        for solve in filtered_solves:
            loader.print_solve_summary(solve)
    
    if args.solve_id:
        solve = loader.load_solve_by_id(args.solve_id)
        if solve:
            loader.print_solve_summary(solve)
        else:
            print(f"Solve ID {args.solve_id} not found")
    
    if args.task_id:
        solve = loader.load_solve_by_task_id(args.task_id)
        if solve:
            loader.print_solve_summary(solve)
        else:
            print(f"Task ID {args.task_id} not found")
    
    if args.export_filtered:
        loader.export_filtered_solves(filtered_solves, args.export_filtered)
    
    return 0


if __name__ == '__main__':
    exit(main())
