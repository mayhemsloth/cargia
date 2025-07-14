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
# Actually I'm going to not be using the quantized version, but combine it with the resource below.
# Note that this website https://www.datacamp.com/tutorial/fine-tune-gemma-3 could also be a supplementary resource

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from cargia.training.training_config import TrainingConfig, TRAINING_CONFIG
from cargia.training.dataset_builder import TaskDatasetBuilder  # this is the class that creates the HuggingFace IterableDataset
from trl import SFTTrainer
from cargia.training.solve_loader import SolveLoader
from cargia.training.data_harness import DataHarness
from sklearn.model_selection import train_test_split

from transformers import TrainingArguments
from peft import LoraConfig

class CargiaGemma3Trainer:
    """
    Trainer class for Cargia, specifically for Gemma3 model architecture a model that can solve grid-based reasoning tasks.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        """
        self.config = config
        
        # determine run name and create run directory

        # initialize the model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.start_checkpoint_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        if self.config.verbose: print(f"Model initialized from {config.start_checkpoint_path}")

        # initialize the processor
        self.processor = AutoTokenizer.from_pretrained(
            config.start_checkpoint_path,
            trust_remote_code=True,
        )
        if self.config.verbose: print(f"Processor initialized from {config.start_checkpoint_path}")

        # Load all solves
        solve_loader = SolveLoader(config.data_dir, config.source_folder)
        solves = solve_loader.load_all_solves()

        # Convert each solve to training format
        formatted_solves = [solve.to_data_harness_format(use_cleaned_thoughts=True) for solve in solves]
        if self.config.verbose: print(f"{len(formatted_solves)} Solves loaded and converted to training format")

        train_tasks, eval_tasks = train_test_split(formatted_solves, test_size=0.2, random_state=1234)
        if self.config.verbose: print(f"{len(train_tasks)} train tasks and {len(eval_tasks)} eval tasks")

        # creates the HuggingFace IterableDataset according to the config file
        self.train_ds = TaskDatasetBuilder(train_tasks, self.processor, self.config, is_training=True ).build()
        self.eval_ds  = TaskDatasetBuilder(eval_tasks,  self.processor, self.config, is_training=False).build()
        # NOTE that the processor is applying inside of the dataset yielding process

        # LoRA Configuration
        peft_config = LoraConfig(
            lora_alpha=16,                           # Scaling factor for LoRA
            lora_dropout=0.05,                       # Add slight dropout for regularization
            r=64,                                    # Rank of the LoRA update matrices
            bias="none",                             # No bias reparameterization
            task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Target modules for LoRA
        )

        training_args = TrainingArguments(
                output_dir          = "./runs/gemma3-sft",
                per_device_train_batch_size = 1,
                per_device_eval_batch_size  = 1,
                gradient_accumulation_steps = 2,
                optim="adamw_torch_fused",                  # use fused adamw optimizer
                save_strategy="epoch",                      # save checkpoint every epoch
                bf16=True,                                  # use bfloat16 precision
                num_train_epochs    = 1,
                logging_steps       = 20,
                save_steps          = 500,
                eval_steps          = 500,
                learning_rate       =2e-4,
                max_steps           = 1000,  # this is to limit the training to 1000 steps
                remove_unused_columns = False,      # ⚠️ must stay False for multimodal
            )

        # initialize the SFTTrainer
        self.trainer = SFTTrainer(
            model              = self.model,
            processing_class   = None, # not technically needed as the train_ds and eval_ds is doing the processor application via the DataHarness class
            train_dataset      = self.train_ds,
            eval_dataset       = self.eval_ds,
            peft_config        = peft_config,
            args               = training_args
        )

if __name__ == "__main__":
    print("===== Training Script Started =====")
    trainer = CargiaGemma3Trainer(TRAINING_CONFIG)
    trainer.trainer.train()
    print("===== Training Script Completed =====")