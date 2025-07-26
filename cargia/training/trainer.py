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
from cargia.training.dataset_builder import TaskDatasetBuilder  # this is the class that creates the HuggingFace IterableDataset
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
        solves= SolveLoader(config.data_dir, config.source_folder).load_all_solves() # a list of SolveData objects
        if self.config.verbose: print(f"{len(solves)} SolveData objects loaded")
        
        train_solves, eval_solves = train_test_split(solves, test_size=0.2, random_state=1234)
        if self.config.verbose: print(f"{len(train_solves)} train solves and {len(eval_solves)} eval solves")
        
        # create the Dataset objects
        self.raw_train_ds = Dataset.from_list([{"task_raw": t.to_dict()} for t in train_solves])
        self.raw_eval_ds = Dataset.from_list([{"task_raw": t.to_dict()} for t in eval_solves])

        self.harness = DataHarness(self.config)  # reuse across epochs

        # build collate functions
        self.train_collate = partial(self._arc_collate, is_training=True)
        self.eval_collate  = partial(self._arc_collate, is_training=False)
        
        # build analysis collate functions
        self.train_analyze_collate = partial(self._arc_collate, is_training=True, analyze_mode=True)
        self.eval_analyze_collate = partial(self._arc_collate, is_training=False, analyze_mode=True)
        
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
                output_dir          = self.run_dir,
                per_device_train_batch_size = 1,
                per_device_eval_batch_size  = 1,
                gradient_accumulation_steps = 1,
                optim="adamw_torch_fused",                  # use fused adamw optimizer
                save_strategy="steps",                      # save checkpoint every some number of steps
                bf16=True,                                  # use bfloat16 precision
                max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
                warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
                num_train_epochs    = 1,
                logging_steps       = 1,
                save_steps          = 500,
                eval_steps          = 500,
                learning_rate       = 2e-4,
                max_steps           = 1000,  # this is to limit the training to 1000 steps
                remove_unused_columns=False, # ⚠️ must stay False for multimodal,
                assistant_only_loss=False,   # only use the assistant loss
                dataset_text_field=None,     # need a dummy field for collator
                dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
                gradient_checkpointing_kwargs={
                    "use_reentrant": True
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
            # callbacks          = [AugmentSwitchCallback(self.train_collate, self.eval_collate)]
        )

    def analyze_dataset(self, max_samples=None):
        """
        Analyze token lengths for both train and eval datasets.
        
        Args:
            max_samples: If provided, only analyze this many samples (for quick testing)
        """
        print("=== Analyzing Training Dataset ===")
        train_lengths = self._analyze_split(self.raw_train_ds, self.train_analyze_collate, "train", max_samples)
        
        print("\n=== Analyzing Evaluation Dataset ===")
        eval_lengths = self._analyze_split(self.raw_eval_ds, self.eval_analyze_collate, "eval", max_samples)
        
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
    
    def _arc_collate(self, examples, *, is_training: bool, analyze_mode: bool = False):
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
                    conv, add_generation_prompt=False, tokenize=False).strip()
            texts.append(txt)
            # print(txt)

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
        batch["labels"] = labels
        # print(batch["input_ids"].size())
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

class CargiaOpenGemma3Trainer:
    pass



if __name__ == "__main__":
    print("===== Training Script Started =====")
    trainer = CargiaGoogleGemma3Trainer(TRAINING_CONFIG)
    print("----Trainer initialized----")
    print("----Training starting----")
    trainer.sft_trainer.train()
    print("===== Training Script Completed =====")