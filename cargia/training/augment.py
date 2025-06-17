"""
Data augmentation module for ARC-AGI training data.

This module implements various augmentation strategies:
- Color invariance: Generating distinct color maps
- Character invariance: Random character substitutions
- Spatial transforms: Rotations and flips (with text rewriting)
- Helper functions for transforming both grid data and thought text
""" 