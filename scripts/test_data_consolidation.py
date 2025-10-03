#!/usr/bin/env python3
"""
Test script for data consolidation system.

This script validates that the export and load functionality works correctly.
"""

import json
import tempfile
import shutil
from pathlib import Path
from scripts.export_unified_solves import UnifiedSolveExporter
from scripts.load_unified_solves import UnifiedSolveLoader


def create_test_data():
    """Create test data for validation."""
    # Create temporary directories
    data_dir = Path(tempfile.mkdtemp())
    source_dir = Path(tempfile.mkdtemp())
    
    # Create test ARC-AGI task
    test_task = {
        "train": [
            {
                "input": [[0, 1, 2], [3, 4, 5]],
                "output": [[1, 2, 3], [4, 5, 6]]
            }
        ],
        "test": [
            {
                "input": [[1, 2, 3], [4, 5, 6]],
                "output": [[2, 3, 4], [5, 6, 7]]
            }
        ]
    }
    
    # Save test task
    with open(source_dir / "test_task_001.json", 'w') as f:
        json.dump(test_task, f)
    
    # Create test databases (simplified)
    import sqlite3
    
    # Solves database
    solves_db = data_dir / "solves.db"
    conn = sqlite3.connect(solves_db)
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
            end_time TEXT
        )
    """)
    
    # Insert test solve
    cursor.execute("""
        INSERT INTO solves (task_id, user_id, order_map_type, order_map, color_map, metadata_labels, start_time, end_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "test_task_001",
        "test_user",
        "default",
        '{"0": "A", "1": "B", "2": "C"}',
        '{"0": [0, 0, 0], "1": [255, 0, 0], "2": [0, 255, 0]}',
        '["rotation", "color_invariance"]',
        "2024-01-01T10:00:00",
        "2024-01-01T10:20:00"
    ))
    conn.commit()
    conn.close()
    
    # Thoughts database
    thoughts_db = data_dir / "thoughts.db"
    conn = sqlite3.connect(thoughts_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            solve_id INTEGER NOT NULL,
            pair_label TEXT NOT NULL,
            pair_type TEXT NOT NULL,
            sequence_index INTEGER NOT NULL,
            thought_text TEXT NOT NULL,
            cleaned_thought_text TEXT
        )
    """)
    
    # Insert test thoughts
    test_thoughts = [
        (1, "a", "train", 0, "Looking at the first training pair...", "Looking at the first training pair, I can see the pattern."),
        (1, "a", "train", 1, "The input has numbers 0,1,2 in first row...", "The input has numbers 0,1,2 in the first row, and 3,4,5 in the second row."),
        (1, "a", "test", 0, "Now applying the pattern to test case...", "Now applying the pattern to the test case, I need to add 1 to each number.")
    ]
    
    for thought in test_thoughts:
        cursor.execute("""
            INSERT INTO thoughts (solve_id, pair_label, pair_type, sequence_index, thought_text, cleaned_thought_text)
            VALUES (?, ?, ?, ?, ?, ?)
        """, thought)
    
    conn.commit()
    conn.close()
    
    return data_dir, source_dir


def test_export():
    """Test the export functionality."""
    print("Testing export functionality...")
    
    # Create test data
    data_dir, source_dir = create_test_data()
    
    try:
        # Create exporter
        exporter = UnifiedSolveExporter(
            data_dir=str(data_dir),
            source_folder=str(source_dir),
            output_dir="test_export"
        )
        
        # Export solves
        results = exporter.export_all_solves()
        
        # Validate results
        assert results['total'] == 1, f"Expected 1 solve, got {results['total']}"
        assert results['successful'] == 1, f"Expected 1 successful export, got {results['successful']}"
        assert results['failed'] == 0, f"Expected 0 failed exports, got {results['failed']}"
        
        print("✓ Export test passed")
        return True
        
    except Exception as e:
        print(f"✗ Export test failed: {e}")
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(data_dir)
        shutil.rmtree(source_dir)


def test_load():
    """Test the load functionality."""
    print("Testing load functionality...")
    
    try:
        # Create loader
        loader = UnifiedSolveLoader("test_export")
        
        # Load all solves
        solves = loader.load_all_solves()
        
        # Validate
        assert len(solves) == 1, f"Expected 1 solve, got {len(solves)}"
        
        solve = solves[0]
        
        # Check metadata
        assert solve['metadata']['solve_id'] == 1
        assert solve['metadata']['task_id'] == "test_task_001"
        assert solve['metadata']['user_id'] == "test_user"
        assert solve['metadata']['solve_duration_seconds'] == 1200
        
        # Check ARC-AGI task
        assert solve['arc_agi_task']['train'][0]['input'] == [[0, 1, 2], [3, 4, 5]]
        assert solve['arc_agi_task']['test'][0]['input'] == [[1, 2, 3], [4, 5, 6]]
        
        # Check training pairs
        assert len(solve['training_pairs']) == 1
        assert solve['training_pairs'][0]['pair_label'] == "a"
        assert len(solve['training_pairs'][0]['thoughts']) == 2
        
        # Check test pairs
        assert len(solve['test_pairs']) == 1
        assert solve['test_pairs'][0]['pair_label'] == "a"
        assert len(solve['test_pairs'][0]['thoughts']) == 1
        
        # Check configuration
        assert solve['solve_configuration']['order_map_type'] == "default"
        assert solve['solve_configuration']['color_map']['0'] == [0, 0, 0]
        
        # Check summary
        assert solve['summary']['total_training_pairs'] == 1
        assert solve['summary']['total_test_pairs'] == 1
        assert solve['summary']['total_thoughts'] == 3
        assert solve['summary']['has_cleaned_thoughts'] == True
        assert solve['summary']['has_arc_agi_task'] == True
        
        print("✓ Load test passed")
        return True
        
    except Exception as e:
        print(f"✗ Load test failed: {e}")
        return False


def test_filtering():
    """Test the filtering functionality."""
    print("Testing filtering functionality...")
    
    try:
        loader = UnifiedSolveLoader("test_export")
        
        # Test user filtering
        user_solves = loader.filter_by_user("test_user")
        assert len(user_solves) == 1, f"Expected 1 solve for test_user, got {len(user_solves)}"
        
        # Test duration filtering
        long_solves = loader.filter_by_duration_range(min_seconds=1000)
        assert len(long_solves) == 1, f"Expected 1 long solve, got {len(long_solves)}"
        
        # Test thought count filtering
        detailed_solves = loader.filter_by_thought_count(min_thoughts=2)
        assert len(detailed_solves) == 1, f"Expected 1 detailed solve, got {len(detailed_solves)}"
        
        # Test cleaned thoughts filtering
        cleaned_solves = loader.filter_by_has_cleaned_thoughts(True)
        assert len(cleaned_solves) == 1, f"Expected 1 solve with cleaned thoughts, got {len(cleaned_solves)}"
        
        print("✓ Filtering test passed")
        return True
        
    except Exception as e:
        print(f"✗ Filtering test failed: {e}")
        return False


def test_statistics():
    """Test the statistics functionality."""
    print("Testing statistics functionality...")
    
    try:
        loader = UnifiedSolveLoader("test_export")
        stats = loader.get_statistics()
        
        # Validate statistics
        assert stats['total_solves'] == 1
        assert stats['unique_users'] == 1
        assert stats['unique_tasks'] == 1
        assert stats['duration_stats']['count'] == 1
        assert stats['duration_stats']['min'] == 1200
        assert stats['thought_stats']['min'] == 3
        assert stats['data_quality']['has_cleaned_thoughts'] == 1
        
        print("✓ Statistics test passed")
        return True
        
    except Exception as e:
        print(f"✗ Statistics test failed: {e}")
        return False


def cleanup():
    """Clean up test files."""
    try:
        shutil.rmtree("test_export")
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")


def main():
    """Run all tests."""
    print("Data Consolidation System Tests")
    print("=" * 40)
    
    tests = [
        test_export,
        test_load,
        test_filtering,
        test_statistics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    cleanup()
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    exit(main())
