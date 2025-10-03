# Data Consolidation System

This directory contains scripts for consolidating fragmented solve data into unified, easy-to-use JSON files.

## Problem

The original Cargia data structure is fragmented across multiple sources:
- **`solves` database table**: Task metadata, user info, timing, color maps
- **`thoughts` database table**: Step-by-step reasoning text for each pair
- **ARC-AGI JSON files**: Original puzzle structure with input/output grids
- **Scattered metadata**: Color maps, character mappings, spatial transforms

This makes it difficult for users to access complete solve information without writing custom code.

## Solution

The data consolidation system creates **one comprehensive JSON file per puzzle solve** that contains everything needed to understand and use the data.

## Scripts

### 1. `export_unified_solves.py`
Exports all solve data into individual JSON files.

**Usage:**
```bash
# Basic export
python scripts/export_unified_solves.py

# Custom paths
python scripts/export_unified_solves.py \
    --data-dir data/solves_and_thoughts \
    --source-folder data/arc_agi_2_reformatted \
    --output-dir solves_export

# Create index file
python scripts/export_unified_solves.py --create-index
```

**Output Structure:**
```
solves_export/
├── index.json                    # Index of all solves
├── solve_001_task_abc123.json   # Individual solve files
├── solve_002_task_def456.json
└── ...
```

### 2. `load_unified_solves.py`
Load and work with unified solve JSON files.

**Usage:**
```bash
# List all solves
python scripts/load_unified_solves.py --solves-dir solves_export --list-solves

# Show statistics
python scripts/load_unified_solves.py --solves-dir solves_export --statistics

# Filter by user
python scripts/load_unified_solves.py --solves-dir solves_export --filter-user "Thomas"

# Filter by duration
python scripts/load_unified_solves.py --solves-dir solves_export \
    --filter-duration-min 300 --filter-duration-max 1800

# Filter by thought count
python scripts/load_unified_solves.py --solves-dir solves_export \
    --filter-thoughts-min 10

# Export filtered results
python scripts/load_unified_solves.py --solves-dir solves_export \
    --filter-user "Thomas" --export-filtered thomas_solves

# Show specific solve
python scripts/load_unified_solves.py --solves-dir solves_export --solve-id 1
```

### 3. `example_unified_usage.py`
Example script showing how to use the unified data.

**Usage:**
```bash
python scripts/example_unified_usage.py
```

## Unified JSON Structure

Each solve JSON file contains:

```json
{
  "metadata": {
    "solve_id": 1,
    "task_id": "abc123",
    "user_id": "Thomas",
    "solve_duration_seconds": 1200,
    "start_time": "2024-01-01T10:00:00",
    "end_time": "2024-01-01T10:20:00",
    "export_timestamp": "2024-01-01T12:00:00",
    "export_version": "1.0"
  },
  "arc_agi_task": {
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
  },
  "solve_configuration": {
    "order_map_type": "default",
    "order_map": {"0": "A", "1": "B", "2": "C"},
    "color_map": {"0": [0, 0, 0], "1": [255, 0, 0]},
    "metadata_labels": ["rotation", "color_invariance"]
  },
  "training_pairs": [
    {
      "pair_label": "a",
      "input": [[0, 1, 2], [3, 4, 5]],
      "output": [[1, 2, 3], [4, 5, 6]],
      "thoughts": [
        {
          "pair_label": "a",
          "pair_type": "train",
          "sequence_index": 0,
          "thought_text": "Looking at the first training pair...",
          "cleaned_thought_text": "Looking at the first training pair, I can see..."
        }
      ]
    }
  ],
  "test_pairs": [
    {
      "pair_label": "a",
      "input": [[1, 2, 3], [4, 5, 6]],
      "expected_output": [[2, 3, 4], [5, 6, 7]],
      "thoughts": [
        {
          "pair_label": "a",
          "pair_type": "test",
          "sequence_index": 0,
          "thought_text": "Now applying the pattern to the test case...",
          "cleaned_thought_text": "Now applying the pattern to the test case, I need to..."
        }
      ]
    }
  ],
  "summary": {
    "total_training_pairs": 1,
    "total_test_pairs": 1,
    "total_thoughts": 2,
    "has_cleaned_thoughts": true,
    "has_arc_agi_task": true
  }
}
```

## Key Benefits

### 1. **Complete Data Access**
- All information in one file
- No need to query multiple databases
- No need to match data across sources

### 2. **Easy Analysis**
- Load individual solves or all solves
- Filter by any criteria
- Export filtered subsets

### 3. **Portable Format**
- Standard JSON format
- No database dependencies
- Easy to share and archive

### 4. **Rich Metadata**
- Solve timing and duration
- User information
- Data quality indicators
- Export versioning

## Use Cases

### Research Analysis
```python
from scripts.load_unified_solves import UnifiedSolveLoader

loader = UnifiedSolveLoader("solves_export")

# Analyze solve patterns
long_solves = loader.filter_by_duration_range(min_seconds=600)
detailed_solves = loader.filter_by_thought_count(min_thoughts=20)

# Export for analysis
loader.export_filtered_solves(long_solves, "analysis_data")
```

### Training Data Preparation
```python
# Get solves with cleaned thoughts
cleaned_solves = loader.filter_by_has_cleaned_thoughts(True)

# Extract training conversations
for solve in cleaned_solves:
    for pair in solve['training_pairs']:
        for thought in pair['thoughts']:
            if thought['cleaned_thought_text']:
                # Use cleaned thought for training
                pass
```

### Data Quality Assessment
```python
# Check data completeness
stats = loader.get_statistics()
print(f"Completeness: {stats['data_quality']['has_cleaned_thoughts']}/{stats['total_solves']}")
```

## Integration with Existing Code

The unified format is designed to be compatible with existing Cargia code:

```python
# Load unified solve
with open("solves_export/solve_001_task_abc123.json", 'r') as f:
    solve_data = json.load(f)

# Convert to SolveData object (if needed)
from cargia.training.solve_loader import SolveData

solve_data_obj = SolveData(
    task_id=solve_data['metadata']['task_id'],
    raw_task=solve_data['arc_agi_task'],
    solve_metadata=solve_data['metadata'],
    train_pairs=solve_data['training_pairs'],
    test_pairs=solve_data['test_pairs']
)
```

## File Naming Convention

Files are named: `solve_{ID:03d}_task_{TASK_ID}.json`

- `ID`: Zero-padded solve ID (001, 002, etc.)
- `TASK_ID`: Original ARC-AGI task identifier

This makes it easy to:
- Sort solves chronologically
- Find specific solves
- Maintain consistent naming

## Index File

The `index.json` file provides a quick overview of all exported solves:

```json
{
  "export_info": {
    "timestamp": "2024-01-01T12:00:00",
    "total_solves": 150,
    "export_version": "1.0"
  },
  "solves": [
    {
      "filename": "solve_001_task_abc123.json",
      "solve_id": 1,
      "task_id": "abc123",
      "user_id": "Thomas",
      "solve_duration_seconds": 1200,
      "total_training_pairs": 3,
      "total_test_pairs": 1,
      "total_thoughts": 15,
      "has_cleaned_thoughts": true,
      "has_arc_agi_task": true
    }
  ]
}
```

## Error Handling

The scripts include robust error handling:
- Missing files are logged as warnings
- Corrupted JSON files are skipped
- Database connection errors are handled gracefully
- Export continues even if individual solves fail

## Performance Considerations

- **Memory usage**: Loads one solve at a time during export
- **File size**: Individual JSON files are typically 10-100KB
- **Indexing**: Uses database indexes for efficient filtering
- **Caching**: Loader caches results to avoid re-reading files

## Future Enhancements

Potential improvements:
1. **Compression**: Gzip compression for large exports
2. **Streaming**: Stream processing for very large datasets
3. **Validation**: Schema validation for exported files
4. **Incremental**: Incremental export of new/changed solves
5. **Format conversion**: Export to other formats (CSV, Parquet, etc.)
