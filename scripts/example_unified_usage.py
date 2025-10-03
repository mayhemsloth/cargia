#!/usr/bin/env python3
"""
Example usage of unified solve data.

This script demonstrates how to work with the unified JSON files
exported by export_unified_solves.py.
"""

import json
from pathlib import Path
from scripts.load_unified_solves import UnifiedSolveLoader


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Load all solves
    loader = UnifiedSolveLoader("solves_export")
    solves = loader.load_all_solves()
    
    if not solves:
        print("No solves found. Run export_unified_solves.py first.")
        return
    
    print(f"Loaded {len(solves)} solves")
    
    # Show statistics
    loader.print_statistics()
    
    # Show details of first solve
    if solves:
        print("\n=== First Solve Details ===")
        loader.print_solve_summary(solves[0])


def example_filtering():
    """Example of filtering solves."""
    print("\n=== Filtering Example ===")
    
    loader = UnifiedSolveLoader("solves_export")
    
    # Filter by user
    thomas_solves = loader.filter_by_user("Thomas")
    print(f"Thomas's solves: {len(thomas_solves)}")
    
    # Filter by duration (solves that took more than 5 minutes)
    long_solves = loader.filter_by_duration_range(min_seconds=300)
    print(f"Long solves (>5 min): {len(long_solves)}")
    
    # Filter by thought count
    detailed_solves = loader.filter_by_thought_count(min_thoughts=10)
    print(f"Detailed solves (>10 thoughts): {len(detailed_solves)}")


def example_working_with_solve_data():
    """Example of working with individual solve data."""
    print("\n=== Working with Solve Data ===")
    
    loader = UnifiedSolveLoader("solves_export")
    solves = loader.load_all_solves()
    
    if not solves:
        print("No solves found.")
        return
    
    # Get first solve
    solve = solves[0]
    
    print(f"Working with solve {solve['metadata']['solve_id']}")
    print(f"Task: {solve['metadata']['task_id']}")
    
    # Access ARC-AGI task data
    if solve['arc_agi_task']:
        print(f"ARC-AGI task has {len(solve['arc_agi_task'].get('train', []))} training pairs")
        print(f"ARC-AGI task has {len(solve['arc_agi_task'].get('test', []))} test pairs")
    
    # Access training pairs with thoughts
    print(f"\nTraining pairs with thoughts:")
    for pair in solve['training_pairs']:
        print(f"  Pair {pair['pair_label']}: {len(pair['thoughts'])} thoughts")
        
        # Show first thought
        if pair['thoughts']:
            first_thought = pair['thoughts'][0]
            print(f"    First thought: {first_thought['thought_text'][:100]}...")
    
    # Access test pairs
    print(f"\nTest pairs:")
    for pair in solve['test_pairs']:
        print(f"  Pair {pair['pair_label']}: {len(pair['thoughts'])} thoughts")
    
    # Access solve configuration
    config = solve['solve_configuration']
    print(f"\nSolve configuration:")
    print(f"  Order map type: {config['order_map_type']}")
    print(f"  Color map: {len(config['color_map'])} colors")
    print(f"  Metadata labels: {config['metadata_labels']}")


def example_export_filtered_data():
    """Example of exporting filtered data."""
    print("\n=== Export Filtered Data ===")
    
    loader = UnifiedSolveLoader("solves_export")
    
    # Filter for solves with cleaned thoughts
    cleaned_solves = loader.filter_by_has_cleaned_thoughts(True)
    print(f"Found {len(cleaned_solves)} solves with cleaned thoughts")
    
    # Export to new directory
    if cleaned_solves:
        loader.export_filtered_solves(cleaned_solves, "filtered_cleaned_solves")
        print("Exported to filtered_cleaned_solves/")


def example_analyzing_thoughts():
    """Example of analyzing thought patterns."""
    print("\n=== Analyzing Thoughts ===")
    
    loader = UnifiedSolveLoader("solves_export")
    solves = loader.load_all_solves()
    
    if not solves:
        print("No solves found.")
        return
    
    # Analyze thought patterns across all solves
    total_thoughts = 0
    cleaned_thoughts = 0
    thought_lengths = []
    
    for solve in solves:
        for pair in solve['training_pairs'] + solve['test_pairs']:
            for thought in pair['thoughts']:
                total_thoughts += 1
                thought_lengths.append(len(thought['thought_text']))
                
                if thought.get('cleaned_thought_text'):
                    cleaned_thoughts += 1
    
    if total_thoughts > 0:
        avg_length = sum(thought_lengths) / len(thought_lengths)
        print(f"Total thoughts: {total_thoughts}")
        print(f"Cleaned thoughts: {cleaned_thoughts} ({cleaned_thoughts/total_thoughts*100:.1f}%)")
        print(f"Average thought length: {avg_length:.1f} characters")
        print(f"Shortest thought: {min(thought_lengths)} characters")
        print(f"Longest thought: {max(thought_lengths)} characters")


def main():
    """Run all examples."""
    print("Unified Solve Data Usage Examples")
    print("=" * 40)
    
    # Check if export directory exists
    if not Path("solves_export").exists():
        print("Error: solves_export directory not found.")
        print("Please run: python scripts/export_unified_solves.py")
        return
    
    example_basic_usage()
    example_filtering()
    example_working_with_solve_data()
    example_export_filtered_data()
    example_analyzing_thoughts()
    
    print("\n" + "=" * 40)
    print("Examples complete!")


if __name__ == '__main__':
    main()
