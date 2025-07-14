# dataset_builder.py
from typing import Dict, Iterable, List
from datasets import IterableDataset
from cargia.training.data_harness import DataHarness 
from cargia.training.training_config import TrainingConfig

class TaskDatasetBuilder:
    """
    Lazily turns a list of raw ARC-AGI tasks into a HuggingFace Dataset whose
    rows are *already* tokenised for Gemma-3 (or any other multimodal processor).
    """

    def __init__(
        self,
        raw_tasks: List[Dict],
        processor,                         # Gemma3Processor / AutoProcessor
        config: TrainingConfig,
        is_training: bool = True,
    ):
        self.raw_tasks   = raw_tasks
        self.processor   = processor
        self.harness     = DataHarness(config)
        self.is_training = is_training

    # -------------------------------------------------
    # 1️⃣  generator that HF will call under the hood
    # -------------------------------------------------
    def _example_stream(self) -> Iterable[Dict]:
        """
        Yields *tokenised* rows; each row already contains the fields that
        SFTTrainer expects (`input_ids`, `attention_mask`, optionally
        `pixel_values`, etc., depending on the processor).
        """
        for task in self.raw_tasks:
            messages = self.harness.create_training_conversation(
                task, is_training=self.is_training
            )

            # Processor does text + images in one call
            model_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,      # let datasets keep lists / numpy
                padding="max_length",     # or 'longest' / False
                truncation=True,
                max_length=8192
            )
            # processor() returns a dict suitable for the model
            # (e.g. {"input_ids": [...], "attention_mask": [...], "pixel_values": [...]})
            yield model_inputs

    # -------------------------------------------------
    # 2️⃣  public method: returns an HF Dataset object
    # -------------------------------------------------
    def build(self) -> IterableDataset:
        return IterableDataset.from_generator(self._example_stream)
