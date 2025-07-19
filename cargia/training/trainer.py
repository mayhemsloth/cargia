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
            quantization_config=quantization_config,
        )
        if self.config.verbose:
            print(f"Model initialized from {config.start_checkpoint_path}")
            print("first weight dtype:", next(self.model.parameters()).dtype)
            print("bitsandbytes loaded:", hasattr(self.model, 'quantization_method'))


        # initialize the processor
        self.processor = AutoProcessor.from_pretrained(
            config.start_checkpoint_path,
            trust_remote_code=True,
        )
        if self.config.verbose: print(f"Processor initialized from {config.start_checkpoint_path}")

        # Load all solves
        solves= SolveLoader(config.data_dir, config.source_folder).load_all_solves() # a list of SolveData objects
        if self.config.verbose: print(f"{len(solves)} SolveData objects loaded")
        
        train_solves, eval_solves = train_test_split(solves, test_size=0.2, random_state=1234)
        if self.config.verbose: print(f"{len(train_solves)} train solves and {len(eval_solves)} eval solves")

        # the helper class for augmenting and tokenising the data   
        # augment_tokenise_train = AugmentAndTokenise(harness=DataHarness(self.config),
        #                                       processor=self.processor,
        #                                       is_training=True) # train is True
        # augment_tokenise_eval = AugmentAndTokenise(harness=DataHarness(self.config),
        #                                       processor=self.processor,
        #                                       is_training=False) # train is False, so we won't apply augmentations on eval sets
        
        # create the Dataset objects
        self.raw_train_ds = Dataset.from_list([{"task_raw": t.to_dict()} for t in train_solves])
        self.raw_eval_ds = Dataset.from_list([{"task_raw": t.to_dict()} for t in eval_solves])

        # train_ds = (raw_train_ds
        #     .with_transform(augment_tokenise_train)   # lazy, per-sample
        #     .with_format("torch"))              # tensors all the way
        # eval_ds = (raw_eval_ds
        #     .with_transform(augment_tokenise_eval)   # lazy, per-sample
        #     .with_format("torch"))              # tensors all the way

        self.harness = DataHarness(self.config)  # reuse across epochs

        # build collate functions
        self.train_collate = partial(self._arc_collate, is_training=True)
        self.eval_collate  = partial(self._arc_collate, is_training=False)
        

        # LoRA Configuration
        # peft_config = LoraConfig(
        #     lora_alpha=16,                           # Scaling factor for LoRA
        #     lora_dropout=0.05,                       # Add slight dropout for regularization
        #     r=128,                                    # Rank of the LoRA update matrices
        #     bias="none",                             # No bias reparameterization
        #     task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
        #     target_modules=[
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj",
        #         "gate_proj",
        #         "up_proj",
        #         "down_proj",
        #     ],  # Target modules for LoRA
        # )
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            # modules_to_save=["lm_head", "embed_tokens",],
            target_modules="all-linear",
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
                remove_unused_columns=False,      # ⚠️ must stay False for multimodal,
                assistant_only_loss=False,           # only use the assistant loss
                dataset_text_field=None,                      # need a dummy field for collator
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

    def process_vision_info(self, messages):
        """Helper from the Gemma tutorial: returns list[Image.Image] in <boi> order."""
        imgs = []
        for m in messages:
            for part in m.get("content", []):
                if isinstance(part, dict) and part.get("type") == "image":
                    imgs.append(part["image"])
        return imgs
    
    def _arc_collate(self, examples, *,is_training: bool):
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

        # ➍  processor → tensors
        batch = self.processor(text=texts, 
                               images=images,
                               return_tensors="pt",
                               padding=True)

        # ➎  label masking (same as Google tutorial)
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        boi_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.tokenizer.special_tokens_map["boi_token"])
        labels[(labels == pad_id) | (labels == boi_id) | (labels == 262144)] = -100
        batch["labels"] = labels
        print(batch["input_ids"].size())
        print(batch["labels"].size())
        print("BLAHBLAHBLAH")
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

class CargiaGemma3Trainer:
    """
    Trainer class for Cargia, specifically for Gemma3 model architecture a model that can solve grid-based reasoning tasks.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        """
        self.config = config
        # get current date and time
        self.run_name = f"gemma3-sft-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # determine run name and create run directory
        self.run_dir = f"./runs/{self.run_name}"
        os.makedirs(self.run_dir, exist_ok=True)

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

        if self.config.verbose: print(f"{len(solves)} SolveData objects loaded")

        train_solves, eval_solves = train_test_split(solves, test_size=0.2, random_state=1234)
        if self.config.verbose: print(f"{len(train_solves)} train solves and {len(eval_solves)} eval solves")

        # creates the HuggingFace IterableDataset according to the config file
        self.train_ds = TaskDatasetBuilder(train_solves, self.processor, self.config, is_training=True ).build()
        self.eval_ds  = TaskDatasetBuilder(eval_solves,  self.processor, self.config, is_training=False).build()
        # NOTE that the processor is applying inside of the dataset yielding process

        # LoRA Configuration
        peft_config = LoraConfig(
            lora_alpha=16,                           # Scaling factor for LoRA
            lora_dropout=0.05,                       # Add slight dropout for regularization
            r=128,                                    # Rank of the LoRA update matrices
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

        training_args = SFTConfig(
                output_dir          = self.run_dir,
                per_device_train_batch_size = 1,
                per_device_eval_batch_size  = 1,
                gradient_accumulation_steps = 2,
                optim="adamw_torch_fused",                  # use fused adamw optimizer
                save_strategy="steps",                      # save checkpoint every some number of steps
                bf16=True,                                  # use bfloat16 precision
                num_train_epochs    = 1,
                logging_steps       = 20,
                save_steps          = 500,
                eval_steps          = 500,
                learning_rate       = 2e-4,
                max_steps           = 1000,  # this is to limit the training to 1000 steps
                remove_unused_columns = False,      # ⚠️ must stay False for multimodal
                assistant_only_loss=False,           # only use the assistant loss
            )

        # initialize the SFTTrainer
        self.sft_trainer = SFTTrainer(
            model              = self.model,
            processing_class   = None, # not technically needed as the train_ds and eval_ds is doing the processor application via the DataHarness class
            train_dataset      = self.train_ds,
            eval_dataset       = self.eval_ds,
            peft_config        = peft_config,
            args               = training_args
        )

if __name__ == "__main__":
    print("===== Training Script Started =====")
    trainer = CargiaGoogleGemma3Trainer(TRAINING_CONFIG)
    print("----Trainer initialized----")
    print("----Training starting----")
    trainer.sft_trainer.train()
    print("===== Training Script Completed =====")