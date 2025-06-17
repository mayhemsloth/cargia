"""
Database interface for storing training run snapshots and examples.

This module provides a thin data access layer for:
- Storing training run metadata
- Saving example predictions during training
- Retrieving examples for visualization in the GUI
- Managing the relationship between raw data and augmented examples

TODO: Example Predictions Schema
The schema for storing example predictions will be defined after implementing:
1. Training loop and prediction format
2. GUI visualization requirements
3. Per-example metrics to track
4. Storage format for model outputs (grids, thoughts, etc.)

This will likely require additional tables and JSON fields to store:
- Input/output grid pairs
- Model's predicted grids
- Model's thought process
- Visualization metadata
- Performance metrics per example
""" 