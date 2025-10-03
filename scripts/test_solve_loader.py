#!/usr/bin/env python3
"""
Test script for SolveLoader and SolveData classes.

This script verifies that the solve loading functionality works correctly
with actual data from the database and filesystem.
"""
import os
import sys
import json
from pathlib import Path

# Add the cargia package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cargia.training.solve_loader import SolveLoader, SolveData
from cargia.data_manager import get_repo_root


def test_solve_loader():
    """Test the SolveLoader functionality."""
    print("Testing SolveLoader...")
    
    try:
        # Get paths from the repository root
        cargia_dir = get_repo_root()
        
        # Load settings to get data directory
        settings_path = os.path.join(cargia_dir, "settings.json")
        if not os.path.exists(settings_path):
            print(f"Settings file not found: {settings_path}")
            return False
            
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        data_dir = os.path.abspath(settings.get('data_dir', os.path.join(cargia_dir, "data")))
        source_folder = os.path.abspath(settings.get('source_folder', os.path.join(cargia_dir, "data/arc_agi_2_reformatted")))
        
        print(f"Data directory: {data_dir}")
        print(f"Source folder: {source_folder}")
        
        # Initialize the solve loader
        loader = SolveLoader(data_dir, source_folder)
        print("✓ SolveLoader initialized successfully")
        
        # Test loading all solves
        print("\nLoading all solves...")
        all_solves = loader.load_all_solves()
        print(f"✓ Loaded {len(all_solves)} solves")
        
        if not all_solves:
            print("⚠ No solves found in database")
            return True
        
        # Test loading solves by user (use the first solve's user)
        first_user = all_solves[0].solve_metadata['user_id']
        print(f"\nLoading solves for user: {first_user}")
        user_solves = loader.load_solves_by_user(first_user)
        print(f"✓ Loaded {len(user_solves)} solves for user {first_user}")
        
        # Test loading solves by task ID (use the first solve's task)
        first_task = all_solves[0].task_id
        print(f"\nLoading solves for task: {first_task}")
        task_solves = loader.load_solves_by_task_id([first_task])
        print(f"✓ Loaded {len(task_solves)} solves for task {first_task}")
        
        # Test SolveData functionality
        print("\nTesting SolveData functionality...")
        test_solve_data(all_solves[0])
        
        # Test pretty_print functionality
        print("\nTesting pretty_print functionality...")
        test_pretty_print(all_solves[0])
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print_summary_statistics(all_solves)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solve_data(solve_data: SolveData):
    """Test the SolveData functionality."""
    print(f"  Testing solve: {solve_data}")
    
    # Test basic properties
    print(f"    Task ID: {solve_data.task_id}")
    print(f"    User: {solve_data.solve_metadata.get('user_id', 'unknown')}")
    print(f"    Train pairs: {len(solve_data.train_pairs)}")
    print(f"    Test pairs: {len(solve_data.test_pairs)}")
    
    # Test duration calculation
    duration = solve_data.get_solve_duration_seconds()
    if duration:
        print(f"    Solve duration: {duration} seconds ({duration//60}m {duration%60}s)")
    else:
        print(f"    Solve duration: Unknown (missing start/end time)")
    
    # Test pair access
    all_pairs = solve_data.get_all_pairs()
    print(f"    Total pairs: {len(all_pairs)}")
    
    # Test thought access
    train_thoughts = solve_data.get_thoughts_by_pair_type('train')
    test_thoughts = solve_data.get_thoughts_by_pair_type('test')
    print(f"    Train thoughts: {len(train_thoughts)}")
    print(f"    Test thoughts: {len(test_thoughts)}")
    
    # Test pair lookup
    if solve_data.train_pairs:
        first_pair_label = solve_data.train_pairs[0]['pair_label']
        found_pair = solve_data.get_pair_by_label(first_pair_label)
        if found_pair:
            print(f"    ✓ Found pair by label '{first_pair_label}'")
        else:
            print(f"    ❌ Failed to find pair by label '{first_pair_label}'")
    
    # Test raw task data
    if 'pairs' in solve_data.raw_task:
        print(f"    Raw task pairs: {len(solve_data.raw_task['pairs'])}")
    if 'default_splits' in solve_data.raw_task:
        print(f"    Default splits: {solve_data.raw_task['default_splits']}")
    
    print("  ✓ SolveData tests passed")


def test_pretty_print(solve_data: SolveData):
    """Test the pretty_print functionality."""
    print("  Testing pretty_print method...")
    
    # Test compact version first (without grids and thoughts)
    print("\n" + "="*60)
    print("COMPACT PRETTY PRINT (no grids, no thoughts)")
    print("="*60)
    compact_output = solve_data.pretty_print(include_grids=False, include_thoughts=False)
    print(compact_output)
    
    # Test version with grids but no thoughts
    print("\n" + "="*60)
    print("PRETTY PRINT WITH GRIDS (no thoughts)")
    print("="*60)
    grid_output = solve_data.pretty_print(include_grids=True, include_thoughts=False, max_grid_size=8)
    print(grid_output)
    
    # Test full version (if thoughts exist)
    train_thoughts = solve_data.get_thoughts_by_pair_type('train')
    test_thoughts = solve_data.get_thoughts_by_pair_type('test')
    has_thoughts = any(thought.strip() for thought in train_thoughts + test_thoughts)
    
    if has_thoughts:
        print("\n" + "="*60)
        print("FULL PRETTY PRINT (with grids and thoughts)")
        print("="*60)
        full_output = solve_data.pretty_print(include_grids=True, include_thoughts=True, max_grid_size=8)
        print(full_output)
    else:
        print("\n" + "="*60)
        print("SKIPPING FULL PRETTY PRINT (no thoughts available)")
        print("="*60)
    
    print("  ✓ pretty_print tests completed")


def print_summary_statistics(solves: list):
    """Print summary statistics about the loaded solves."""
    if not solves:
        print("No solves to analyze")
        return
    
    # Basic counts
    total_solves = len(solves)
    unique_tasks = len(set(solve.task_id for solve in solves))
    unique_users = len(set(solve.solve_metadata.get('user_id', 'unknown') for solve in solves))
    
    print(f"Total solves: {total_solves}")
    print(f"Unique tasks: {unique_tasks}")
    print(f"Unique users: {unique_users}")
    
    # Train/test pair statistics
    total_train_pairs = sum(len(solve.train_pairs) for solve in solves)
    total_test_pairs = sum(len(solve.test_pairs) for solve in solves)
    avg_train_pairs = total_train_pairs / total_solves if total_solves > 0 else 0
    avg_test_pairs = total_test_pairs / total_solves if total_solves > 0 else 0
    
    print(f"Total train pairs: {total_train_pairs}")
    print(f"Total test pairs: {total_test_pairs}")
    print(f"Average train pairs per solve: {avg_train_pairs:.1f}")
    print(f"Average test pairs per solve: {avg_test_pairs:.1f}")
    
    # Duration statistics
    durations = [solve.get_solve_duration_seconds() for solve in solves]
    durations = [d for d in durations if d is not None]
    
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        print(f"Average solve duration: {avg_duration:.1f} seconds ({avg_duration//60:.0f}m {avg_duration%60:.0f}s)")
        print(f"Min solve duration: {min_duration} seconds ({min_duration//60}m {min_duration%60}s)")
        print(f"Max solve duration: {max_duration} seconds ({max_duration//60}m {max_duration%60}s)")
    else:
        print("No duration data available")
    
    # Thought statistics
    total_thoughts = sum(len(solve.get_thoughts_by_pair_type('train')) + 
                       len(solve.get_thoughts_by_pair_type('test')) for solve in solves)
    non_empty_thoughts = sum(
        sum(1 for thought in solve.get_thoughts_by_pair_type('train') if thought.strip()) +
        sum(1 for thought in solve.get_thoughts_by_pair_type('test') if thought.strip())
        for solve in solves
    )
    
    print(f"Total thoughts: {total_thoughts}")
    print(f"Non-empty thoughts: {non_empty_thoughts}")
    print(f"Thought completion rate: {non_empty_thoughts/total_thoughts*100:.1f}%" if total_thoughts > 0 else "No thoughts")
    
    # Sample some solve metadata
    print("\nSample solve metadata:")
    for i, solve in enumerate(solves[:3]):  # Show first 3 solves
        metadata = solve.solve_metadata
        print(f"  Solve {i+1}: Task={solve.task_id}, User={metadata.get('user_id', 'unknown')}, "
              f"OrderMapType={metadata.get('order_map_type', 'unknown')}")


def main():
    """Main test function."""
    print("="*60)
    print("SOLVE LOADER TEST")
    print("="*60)
    
    success = test_solve_loader()
    
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 