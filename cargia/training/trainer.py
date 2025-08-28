"""
Trainer module for fine-tuning Gemma3 on ARC-AGI tasks.

This module handles the core training loop, including:
- Dataset preparation and loading
- Model initialization and training
- Loss computation (text + grid prediction)
- Checkpointing and model saving
- Integration with TensorBoard for monitoring
""" 

# I'm going to generally be following this website 
# https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
# as it specifically is a fine-tuning for Gemma3, from Google, for a vision task.

import torch
import inspect
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, AutoProcessor, TrainerCallback, BitsAndBytesConfig
from cargia.training.training_config import TrainingConfig, TRAINING_CONFIG
from trl import SFTTrainer, SFTConfig
from cargia.training.solve_loader import SolveLoader, SolveData
from cargia.training.data_harness import DataHarness
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from peft import LoraConfig
from datasets import Dataset
from functools import partial
import numpy as np
import json
import torch.nn.functional as nn
import re

class CargiaGoogleGemma3Trainer:
    """
    Trainer class for Cargia, specifically for Gemma3 model architecture a model that can solve grid-based reasoning tasks.
    Code is based STRICTLY on the following website: https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        """
        self.config = config
        self.run_name = f"gemma3-sft-{datetime.now().strftime('%Y%m%d_%H%M%S')}"         # get current date and time
        self.run_dir = f"./runs/{self.run_name}"        # determine run name and create run directory
        os.makedirs(self.run_dir, exist_ok=True)

        # initialize the quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        # initialize the model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.start_checkpoint_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            quantization_config=quantization_config if config.use_bitsnbytes else None,
        )
        if self.config.verbose:
            print(f"Model initialized from {config.start_checkpoint_path}")
            if config.use_bitsnbytes:   # if bitsnbytes is used, print the quantization method
                print("bitsandbytes loaded:", hasattr(self.model, 'quantization_method'))
            print("first weight dtype:", next(self.model.parameters()).dtype)

        # initialize the processor
        self.processor = AutoProcessor.from_pretrained(
            config.start_checkpoint_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if self.config.verbose: print(f"Processor initialized from {config.start_checkpoint_path}")

        # Load all solves
        solves = SolveLoader(config.data_dir, config.source_folder).load_all_solves() # a list of SolveData objects
        if self.config.verbose: print(f"{len(solves)} SolveData objects loaded")
        
        # Apply data sampling limit if specified
        if config.data_sample_maximum_limit is not None:
            max_samples = min(config.data_sample_maximum_limit, len(solves))
            solves = solves[:max_samples]
            if self.config.verbose: print(f"Limited to {max_samples} samples due to data_sample_maximum_limit={config.data_sample_maximum_limit}")
        
        # split into train and eval sets
        train_solves, eval_solves = train_test_split(solves, test_size=0.2, random_state=1234)
        if self.config.verbose: print(f"{len(train_solves)} train solves and {len(eval_solves)} eval solves")
        
        # create the Dataset objects
        self.raw_train_ds = Dataset.from_list([{"task_raw": t.to_dict()} for t in train_solves])
        self.raw_eval_ds = Dataset.from_list([{"task_raw": t.to_dict()} for t in eval_solves])

        self.harness = DataHarness(self.config)  # reuse across epochs

        # build collate functions
        self.train_collate = partial(self._arc_collate, is_training=True, assistant_only_loss=self.config.assistant_only_loss)
        self.eval_collate  = partial(self._arc_collate, is_training=False, assistant_only_loss=self.config.assistant_only_loss)
        
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=2,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens",],
            # target_modules="all-linear",
            target_modules=[
                "q_proj",
                # "k_proj",
                # "v_proj",
                # "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
            ]
        )

        training_args = SFTConfig(
                output_dir                  = self.run_dir,
                per_device_train_batch_size = 1,
                per_device_eval_batch_size  = 1,
                gradient_accumulation_steps = 1,
                optim="adamw_torch_fused",                  # use fused adamw optimizer
                save_strategy="steps",                      # save checkpoint every some number of steps
                bf16=True,                                  # use bfloat16 precision
                max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
                warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
                num_train_epochs    = config.num_train_epochs,
                logging_steps       = 1,
                save_steps          = 500,
                eval_steps          = 500,
                learning_rate       = 2e-4,
                max_steps           = config.max_steps,  # use config value for training step limit
                remove_unused_columns=False, # ⚠️ must stay False for multimodal,
                assistant_only_loss=False,   # only use the assistant loss
                dataset_text_field=None,     # need a dummy field for collator
                dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                },  # use reentrant checkpointing
            )

        # initialize the SFTTrainer
        self.sft_trainer = SFTTrainer(
            model              = self.model,
            args               = training_args,
            peft_config        = peft_config,
            train_dataset      = self.raw_train_ds,
            eval_dataset       = self.raw_eval_ds,
            # processing_class   = self.processor, 
            # tokenizer          = self.processor.tokenizer,
            data_collator      = self.train_collate,
            compute_loss_func  = self.custom_grid_loss_function,
            # callbacks          = [AugmentSwitchCallback(self.train_collate, self.eval_collate)]
        )

    def custom_grid_loss_function(self, outputs, labels, num_items_in_batch=None):
        """
        Custom loss function that recreates standard LabelSmoother logic and adds
        higher weights for grid output tokens.
        
        Args:
            outputs: Model outputs containing logits
            labels: Ground truth labels
            num_items_in_batch: Number of items in batch (optional)
            
        Returns:
            Weighted loss tensor
        """
        # ------ START OF STANDARD LABEL SMOOTHER LOGIC ------
        # Extract logits from model outputs
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        
        # Apply causal LM shift: logits[..., :-1, :] and labels[..., 1:]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = -nn.log_softmax(logits, dim=-1)
        
        # Handle label dimensions
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        
        # Create padding mask (ignore_index is -100)
        ignore_index = -100
        padding_mask = labels.eq(ignore_index)
        
        # Clamp labels to avoid gather issues with -100
        labels = torch.clamp(labels, min=0)
        
        # Compute NLL loss
        nll_loss = log_probs.gather(dim=-1, index=labels)
        
        # Compute smoothed loss component (uniform distribution)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        
        # Apply padding mask
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        
        # Calculate number of active (non-padded) elements
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        
        # Compute standard loss components
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        
        # Apply label smoothing (epsilon = 0.1 as in standard LabelSmoother)
        epsilon = 0.1
        standard_loss = (1 - epsilon) * nll_loss + epsilon * smoothed_loss
        
        # ------ END OF STANDARD LABEL SMOOTHER LOGIC ------

        # ------ START OF GRID-SPECIFIC WEIGHTING ------
        # Now add grid-specific weighting
        grid_weighted_loss = self._add_grid_weighting(logits, labels, padding_mask, num_active_elements)
        
        # Add binary grid accuracy loss (all-or-nothing penalty)
        binary_grid_loss = self._add_binary_grid_accuracy_loss(logits, labels, padding_mask, num_active_elements)
        
        # Combine standard loss with grid-weighted components
        # You can adjust these weights to control the importance of grid accuracy
        grid_weight = getattr(self.config, 'grid_loss_weight', 10.0)  # Get from config or use default
        binary_grid_weight = getattr(self.config, 'binary_grid_loss_weight', 50.0)  # Get from config or use default
        total_loss = standard_loss + grid_weight * grid_weighted_loss + binary_grid_weight * binary_grid_loss
        
        # Debug logging (only in verbose mode)
        if self.config.verbose and hasattr(self, '_loss_debug_counter'):
            self._loss_debug_counter += 1
            if self._loss_debug_counter % 100 == 0:  # Log every 100 steps
                print(f"Loss components - Standard: {standard_loss:.4f}, Grid-weighted: {grid_weighted_loss:.4f}, Binary-grid: {binary_grid_loss:.4f}, Total: {total_loss:.4f}")
        elif self.config.verbose and not hasattr(self, '_loss_debug_counter'):
            self._loss_debug_counter = 0
        
        return total_loss
    
    def _add_grid_weighting(self, logits, labels, padding_mask, num_active_elements):
        """
        Add higher weights for grid output tokens.
        
        This provides PER-TOKEN gradients - each grid token gets individual
        gradient signals based on its correctness. This helps the model learn
        to predict each grid token accurately.
        
        Args:
            logits: Model logits (shifted)
            labels: Ground truth labels (shifted)
            padding_mask: Mask for padded tokens
            num_active_elements: Number of non-padded elements
            
        Returns:
            Grid-weighted loss component
        """
        # Decode the labels to find grid output tokens
        grid_token_mask = self._identify_grid_output_tokens(labels)
        
        if grid_token_mask.sum() == 0:
            # No grid tokens found, return zero loss
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Compute loss only for grid tokens
        log_probs = -nn.log_softmax(logits, dim=-1)
        
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        
        # Clamp labels to avoid gather issues
        labels = torch.clamp(labels, min=0)
        
        # Get NLL loss for grid tokens
        grid_nll_loss = log_probs.gather(dim=-1, index=labels)
        
        # Apply masks: only consider grid tokens that are not padded
        grid_and_active_mask = grid_token_mask & ~padding_mask
        grid_nll_loss.masked_fill_(~grid_and_active_mask, 0.0)
        
        # Count active grid tokens
        num_active_grid_tokens = grid_and_active_mask.sum()
        
        if num_active_grid_tokens == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Compute grid-weighted loss
        grid_loss = grid_nll_loss.sum() / num_active_grid_tokens
        
        return grid_loss
    
    def _add_binary_grid_accuracy_loss(self, logits, labels, padding_mask, num_active_elements):
        """
        Add binary loss that heavily penalizes any grid token error (all-or-nothing).
        
        This provides AGGREGATE gradients - the loss is based on whether ANY
        grid token is wrong, creating pressure for perfect grid accuracy.
        Combined with the per-token loss, this encourages both individual
        token accuracy AND perfect overall grid matches.
        
        Args:
            logits: Model logits (shifted)
            labels: Ground truth labels (shifted)
            padding_mask: Mask for padded tokens
            num_active_elements: Number of non-padded elements
            
        Returns:
            Binary grid accuracy loss component
        """
        # Identify grid output tokens
        grid_token_mask = self._identify_grid_output_tokens(labels)
        
        if grid_token_mask.sum() == 0:
            # No grid tokens found, return zero loss
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Get predictions for grid tokens
        predictions = logits.argmax(dim=-1)
        
        # Create mask for active grid tokens (not padded)
        grid_and_active_mask = grid_token_mask & ~padding_mask
        
        if grid_and_active_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Check if ALL grid tokens are correct
        grid_predictions = predictions[grid_and_active_mask]
        grid_labels = labels[grid_and_active_mask]
        
        # Create a differentiable binary loss
        # Instead of a hard binary decision, use a smooth approximation
        # that approaches 1.0 when any token is wrong and 0.0 when all are correct
        
        # Calculate per-token correctness (0.0 = correct, 1.0 = wrong)
        token_errors = (grid_predictions != grid_labels).float()
        
        # Use a much more aggressive approximation of "any wrong"
        # This creates a differentiable function that is very close to 0.0 when all tokens are correct
        # and quickly approaches 1.0 when any token is wrong
        total_errors = torch.sum(token_errors)
        
        # Get configuration parameters
        loss_type = getattr(self.config, 'binary_loss_type', 'exponential')
        sensitivity = getattr(self.config, 'binary_loss_sensitivity', 10.0)
        
        if loss_type == "exponential":
            # Exponential penalty: 1 - exp(-sensitivity * total_errors)
            # Very close to 0 when total_errors = 0, quickly approaches 1 when any errors
            binary_loss = 1.0 - torch.exp(-total_errors * sensitivity)
            
        elif loss_type == "sigmoid":
            # Scaled sigmoid: sigmoid((total_errors - threshold) * sensitivity)
            # Threshold allows for small tolerance, high sensitivity makes it steep
            threshold = 0.1
            binary_loss = torch.sigmoid((total_errors - threshold) * sensitivity)
            
        elif loss_type == "threshold":
            # Threshold-based: very close to binary but differentiable
            # Uses a very steep sigmoid around the threshold
            threshold = 0.1
            binary_loss = torch.sigmoid((total_errors - threshold) * sensitivity)
            
        else:
            # Default to exponential
            binary_loss = 1.0 - torch.exp(-total_errors * sensitivity)
        
        return binary_loss
    
    def _identify_grid_output_tokens(self, labels):
        """
        Identify which tokens correspond to grid output JSON strings.
        
        Args:
            labels: Ground truth labels tensor
            
        Returns:
            Boolean mask indicating grid output tokens
        """
        try:
            # Decode the labels to text
            decoded_text = self.processor.tokenizer.decode(labels[0], skip_special_tokens=False)
            
            # Create a mask for grid tokens
            grid_mask = torch.zeros_like(labels, dtype=torch.bool)
            
            # Method 1: Look for JSON array patterns that represent grids
            # Grid outputs are typically formatted as JSON arrays like [[0,1,2],[3,4,5]]
            grid_pattern = r'\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]'
            grid_matches = list(re.finditer(grid_pattern, decoded_text))
            
            for match in grid_matches:
                start_char = match.start()
                end_char = match.end()
                
                # Convert character positions to token positions more accurately
                start_tokens = self.processor.tokenizer.encode(decoded_text[:start_char], add_special_tokens=False)
                end_tokens = self.processor.tokenizer.encode(decoded_text[:end_char], add_special_tokens=False)
                
                start_token = len(start_tokens)
                end_token = len(end_tokens)
                
                # Mark the token range as grid tokens
                if start_token < labels.shape[1] and end_token <= labels.shape[1]:
                    grid_mask[0, start_token:end_token] = True
            
            # Method 2: Look for the LAST assistant response that contains grid-like content
            # This focuses on the model's output grid (not input grids)
            if grid_mask.sum() == 0:
                # Find the last occurrence of "assistant" role in the conversation
                assistant_tokens = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
                last_assistant_pos = -1
                
                for i in range(labels.shape[1] - len(assistant_tokens)):
                    if all(labels[0, i+j].item() == assistant_tokens[j] for j in range(len(assistant_tokens))):
                        last_assistant_pos = i
                
                if last_assistant_pos >= 0:
                    # Look for grid-like content after the last assistant token
                    # Grid content typically starts with '[' and contains nested arrays
                    start_pos = last_assistant_pos + len(assistant_tokens)
                    
                    # Find the start of grid content (look for opening bracket)
                    open_bracket_tokens = self.processor.tokenizer.encode("[", add_special_tokens=False)
                    grid_start = -1
                    
                    for i in range(start_pos, min(start_pos + 100, labels.shape[1])):
                        if labels[0, i].item() in open_bracket_tokens:
                            grid_start = i
                            break
                    
                    if grid_start >= 0:
                        # Mark tokens from grid start to end of sequence (or reasonable limit)
                        # Look for the end of the grid (matching closing bracket)
                        close_bracket_tokens = self.processor.tokenizer.encode("]", add_special_tokens=False)
                        grid_end = labels.shape[1]
                        
                        # Find the last closing bracket in the sequence
                        for i in range(grid_start, labels.shape[1]):
                            if labels[0, i].item() in close_bracket_tokens:
                                grid_end = i + 1
                        
                        # Mark the grid token range
                        for k in range(grid_start, grid_end):
                            if labels[0, k].item() != self.processor.tokenizer.pad_token_id:
                                grid_mask[0, k] = True
            
            # Method 3: Fallback - look for any JSON-like array structure
            # This handles character augmentations by looking for structural patterns
            if grid_mask.sum() == 0:
                # Look for any nested array structure that could be a grid
                # This works regardless of the specific characters used
                open_bracket_tokens = self.processor.tokenizer.encode("[", add_special_tokens=False)
                close_bracket_tokens = self.processor.tokenizer.encode("]", add_special_tokens=False)
                comma_tokens = self.processor.tokenizer.encode(",", add_special_tokens=False)
                
                # Find potential grid boundaries
                potential_grids = []
                bracket_stack = []
                
                for i in range(labels.shape[1]):
                    token_id = labels[0, i].item()
                    
                    if token_id in open_bracket_tokens:
                        bracket_stack.append(i)
                    elif token_id in close_bracket_tokens and bracket_stack:
                        start_pos = bracket_stack.pop()
                        if len(bracket_stack) == 0:  # Complete array
                            # Check if this looks like a grid (has commas and nested structure)
                            has_commas = any(labels[0, j].item() in comma_tokens for j in range(start_pos, i))
                            if has_commas:
                                potential_grids.append((start_pos, i + 1))
                
                # Use the last potential grid as the model's output
                if potential_grids:
                    last_grid_start, last_grid_end = potential_grids[-1]
                    for k in range(last_grid_start, last_grid_end):
                        if labels[0, k].item() != self.processor.tokenizer.pad_token_id:
                            grid_mask[0, k] = True
            
            return grid_mask
            
        except Exception as e:
            # If there's any error in identification, return no grid tokens
            print(f"Warning: Error identifying grid tokens: {e}")
            return torch.zeros_like(labels, dtype=torch.bool)

    def debug_grid_token_identification(self, max_samples=5):
        """
        Debug method to visualize which tokens are being identified as grid tokens.
        
        Args:
            max_samples: Number of samples to analyze for debugging
        """
        print("=== Debugging Grid Token Identification ===")
        
        # Get a few samples from the training dataset
        sample_indices = list(range(min(max_samples, len(self.raw_train_ds))))
        
        for idx in sample_indices:
            print(f"\n--- Sample {idx + 1} ---")
            
            # Get the sample
            sample = self.raw_train_ds[idx]
            batch_dict = [{"task_raw": sample["task_raw"]}]
            
            # Process through the collate function to get labels
            try:
                batch = self.train_collate(batch_dict)
                labels = batch["labels"]
                
                # Identify grid tokens
                grid_mask = self._identify_grid_output_tokens(labels)
                
                # Decode the full sequence
                full_text = self.processor.tokenizer.decode(labels[0], skip_special_tokens=False)
                
                # Show statistics
                total_tokens = (labels[0] != self.processor.tokenizer.pad_token_id).sum().item()
                grid_tokens = grid_mask.sum().item()
                
                print(f"Total tokens: {total_tokens}")
                print(f"Grid tokens identified: {grid_tokens}")
                print(f"Grid token percentage: {grid_tokens/total_tokens*100:.1f}%")
                
                # Show the identified grid tokens
                if grid_tokens > 0:
                    print("Grid tokens found at positions:")
                    grid_positions = torch.where(grid_mask[0])[0].tolist()
                    print(f"  Positions: {grid_positions}")
                    
                    # Show the actual tokens
                    grid_token_ids = labels[0][grid_mask[0]]
                    grid_text = self.processor.tokenizer.decode(grid_token_ids, skip_special_tokens=False)
                    print(f"  Grid text: {repr(grid_text)}")
                
                # Show a snippet of the full text around grid tokens
                if grid_tokens > 0:
                    start_pos = max(0, grid_positions[0] - 10)
                    end_pos = min(len(labels[0]), grid_positions[-1] + 10)
                    context_tokens = labels[0][start_pos:end_pos]
                    context_text = self.processor.tokenizer.decode(context_tokens, skip_special_tokens=False)
                    print(f"  Context: {repr(context_text)}")
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print("\n=== Debug Complete ===")

    def test_custom_loss_function(self, num_samples=3):
        """
        Test the custom loss function with sample data to ensure it's working correctly.
        
        Args:
            num_samples: Number of samples to test
        """
        print("=== Testing Custom Loss Function ===")
        
        # Get a few samples from the training dataset
        sample_indices = list(range(min(num_samples, len(self.raw_train_ds))))
        
        for idx in sample_indices:
            print(f"\n--- Testing Sample {idx + 1} ---")
            
            # Get the sample
            sample = self.raw_train_ds[idx]
            batch_dict = [{"task_raw": sample["task_raw"]}]
            
            try:
                # Process through the collate function to get the batch
                batch = self.train_collate(batch_dict)
                
                # Create dummy outputs (simulate model forward pass)
                batch_size, seq_len, vocab_size = batch["input_ids"].shape[0], batch["input_ids"].shape[1], self.processor.tokenizer.vocab_size
                dummy_logits = torch.randn(batch_size, seq_len, vocab_size, device=batch["input_ids"].device, dtype=torch.bfloat16)
                dummy_outputs = {"logits": dummy_logits}
                
                # Test the custom loss function
                loss = self.custom_grid_loss_function(dummy_outputs, batch["labels"])
                
                print(f"Loss value: {loss.item():.4f}")
                print(f"Loss shape: {loss.shape}")
                print(f"Loss requires grad: {loss.requires_grad}")
                
                # Test that gradients can be computed
                loss.backward()
                print("✓ Gradients computed successfully")
                
            except Exception as e:
                print(f"✗ Error testing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n=== Loss Function Test Complete ===")


    def analyze_dataset(self, max_samples=None):
        """
        Analyze token lengths for both train and eval datasets.
        
        Args:
            max_samples: If provided, only analyze this many samples (for quick testing)
        """

        # build analysis collate functions
        train_analyze_collate = partial(self._arc_collate, is_training=True, analyze_mode=True)
        eval_analyze_collate = partial(self._arc_collate, is_training=False, analyze_mode=True)

        print("=== Analyzing Training Dataset ===")
        train_lengths = self._analyze_split(self.raw_train_ds, train_analyze_collate, "train", max_samples)
        
        print("\n=== Analyzing Evaluation Dataset ===")
        eval_lengths = self._analyze_split(self.raw_eval_ds, eval_analyze_collate, "eval", max_samples)
        
        # Combine statistics
        all_lengths = train_lengths + eval_lengths
        
        # Calculate overall statistics
        overall_stats = self._calculate_statistics(all_lengths, "overall")
        
        # Save detailed results
        results = {
            "train": self._calculate_statistics(train_lengths, "train"),
            "eval": self._calculate_statistics(eval_lengths, "eval"),
            "overall": overall_stats,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "data_dir": self.config.data_dir,
                "source_folder": self.config.source_folder,
                "start_checkpoint_path": self.config.start_checkpoint_path,
                "max_samples_analyzed": max_samples
            }
        }
        
        # Save to file
        output_file = f"token_length_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {output_file}")
        
        return results
    
    def _analyze_split(self, dataset, collate_fn, split_name, max_samples=None):
        """
        Analyze a specific dataset split.
        """
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        print(f"Analyzing {len(dataset)} samples from {split_name} dataset...")
        
        all_lengths = []
        batch_size = 1  # Process one at a time to avoid memory issues
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            batch_dict = [{"task_raw": item["task_raw"]} for item in batch]
            
            try:
                lengths = collate_fn(batch_dict)
                all_lengths.extend(lengths)
                
                if (i + batch_size) % 100 == 0 or i + batch_size >= len(dataset):
                    print(f"  Processed {min(i + batch_size, len(dataset))}/{len(dataset)} samples")
                    
            except Exception as e:
                print(f"  Error processing batch starting at index {i}: {e}")
                continue
        
        return all_lengths
    
    def _calculate_statistics(self, lengths, split_name):
        """
        Calculate comprehensive statistics for a list of token lengths.
        """
        if not lengths:
            return {"error": "No valid lengths found"}
        
        lengths = np.array(lengths)
        
        stats = {
            "count": len(lengths),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "percentiles": {
                "25": float(np.percentile(lengths, 25)),
                "50": float(np.percentile(lengths, 50)),
                "75": float(np.percentile(lengths, 75)),
                "90": float(np.percentile(lengths, 90)),
                "95": float(np.percentile(lengths, 95)),
                "99": float(np.percentile(lengths, 99)),
            }
        }
        
        # Print summary
        print(f"\n{split_name.upper()} DATASET STATISTICS:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.1f} tokens")
        print(f"  Median: {stats['median']:.1f} tokens")
        print(f"  Std Dev: {stats['std']:.1f} tokens")
        print(f"  Min: {stats['min']} tokens")
        print(f"  Max: {stats['max']} tokens")
        print(f"  95th percentile: {stats['percentiles']['95']:.1f} tokens")
        print(f"  99th percentile: {stats['percentiles']['99']:.1f} tokens")
        
        return stats
    
    def estimate_memory_requirements(self, stats, batch_size=1):
        """
        Estimate memory requirements based on token length statistics.
        """
        print(f"\n=== MEMORY ESTIMATION (batch_size={batch_size}) ===")
        
        # Rough estimation: each token requires ~2 bytes in bfloat16
        bytes_per_token = 2
        
        # Calculate for different scenarios
        scenarios = {
            "mean": stats["overall"]["mean"],
            "median": stats["overall"]["median"],
            "95th_percentile": stats["overall"]["percentiles"]["95"],
            "99th_percentile": stats["overall"]["percentiles"]["99"],
            "max": stats["overall"]["max"]
        }
        
        for scenario, token_length in scenarios.items():
            memory_mb = (token_length * batch_size * bytes_per_token) / (1024 * 1024)
            print(f"  {scenario}: {token_length:.0f} tokens → ~{memory_mb:.1f} MB per batch")
        
        print(f"\n  Note: This is a rough estimate. Actual memory usage will be higher")
        print(f"  due to model parameters, gradients, optimizer states, etc.")

    def process_vision_info(self, messages):
        """Helper from the Gemma tutorial: returns list[Image.Image] in <boi> order."""
        imgs = []
        for m in messages:
            for part in m.get("content", []):
                if isinstance(part, dict) and part.get("type") == "image":
                    imgs.append(part["image"])
        return imgs
    
    def _arc_collate(self, examples, *, is_training: bool, assistant_only_loss: bool = False, analyze_mode: bool = False):
        texts, images = [], []

        for ex in examples:
            # convert the task_raw to a SolveData object for the DataHarness class
            if isinstance(ex["task_raw"], dict):
                task = SolveData(**ex["task_raw"]) # task variable is a SolveData object
            else:
                task = ex["task_raw"] # task variable is a SolveData object
           
            # ➊  stochastic augmentation on raw text every epoch / worker
            conv = self.harness.create_training_conversation(
                task, is_training=is_training
            )

            # ➋  build chat-template string (dynamic turns OK)
            txt = self.processor.apply_chat_template(
                    conv,
                    add_generation_prompt=False,
                    tokenize=False).strip()
            print(txt)
            texts.append(txt)

            # ➌  collect PIL images matching <boi> tokens
            images.append(self.process_vision_info(conv))

        # If in analyze mode, return token lengths instead of full batch
        if analyze_mode:
            token_lengths = []
            for i, text in enumerate(texts):
                # Process with images if available (same as training)
                if len(images) > 0 and i < len(images):
                    batch = self.processor(
                        text=[text],
                        images=images[i],
                        return_tensors="pt",
                        padding=False,  # No padding to get true length
                        truncation=False,  # No truncation to get true length
                        add_special_tokens=True
                    )
                else:
                    batch = self.processor(
                        text=[text],
                        return_tensors="pt",
                        padding=False,  # No padding to get true length
                        truncation=False,  # No truncation to get true length
                        add_special_tokens=True
                    )
                token_lengths.append(batch["input_ids"].shape[1])
            return token_lengths

        # ➍  processor → tensors (normal training mode)
        if len(images) > 0:
            batch = self.processor(text=texts, 
                                images=images,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=4096)
        else:
            batch = self.processor(text=texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=4096)

        # ➎  label masking (same as Google tutorial)
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        boi_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.special_tokens_map["boi_token"])
        labels[(labels == pad_id) | (labels == boi_id) | (labels == 262144)] = -100

        # TODO: add proper handling of assistant_only_loss by identifying the tokens for <start_of_turn>user and then subsequent tokens until <end_of_turn>
        
        batch["labels"] = labels

        return batch
    
class AugmentSwitchCallback(TrainerCallback):
    def __init__(self, train_collate, eval_collate):
        self.train_collate = train_collate
        self.eval_collate  = eval_collate

    # Will be called *inside* trainer.evaluate(...)
    def on_evaluate(self, args, state, control, **kwargs):
        kwargs["trainer"].data_collator = self.eval_collate   # disable aug

    # Called at the *very* beginning of every training step
    def on_step_begin(self, args, state, control, **kwargs):
        kwargs["trainer"].data_collator = self.train_collate  # enable aug


if __name__ == "__main__":
    print("===== Training Script Started =====")
    trainer = CargiaGoogleGemma3Trainer(TRAINING_CONFIG)
    print("----Trainer initialized----")
    
    # Debug grid token identification
    if TRAINING_CONFIG.verbose:
        trainer.debug_grid_token_identification(max_samples=1)
        
        print("----Testing custom loss function----")
        trainer.test_custom_loss_function(num_samples=1)
    
    print("----Training starting----")
    trainer.sft_trainer.train()
    print("===== Training Script Completed =====")