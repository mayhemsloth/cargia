"""
Trainer module for fine-tuning Gemma3 on ARC-AGI tasks.

This module handles the core training loop, including:
- Dataset preparation and loading
- Model initialization and training
- Loss computation (text + grid prediction)
- Checkpointing and model saving
- Integration with TensorBoard for monitoring
""" 