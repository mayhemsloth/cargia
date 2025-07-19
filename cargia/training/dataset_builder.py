# dataset_builder.py
from typing import Dict, Iterable, List
from datasets import IterableDataset
from cargia.training.data_harness import DataHarness 
from cargia.training.training_config import TrainingConfig
from cargia.training.solve_loader import SolveData

class TaskDatasetBuilder:
    """
    Lazily turns a list of SolveData objects into a HuggingFace Dataset whose
    rows are *already* tokenised for Gemma-3 (or any other multimodal processor).
    """

    def __init__(
        self,
        solve_data_list: List[SolveData],
        processor,                         # Gemma3Processor / AutoProcessor
        config: TrainingConfig,
        is_training: bool = True,
    ):
        self.solve_data_list = solve_data_list
        self.processor       = processor
        self.harness         = DataHarness(config)
        self.is_training     = is_training

    # -------------------------------------------------
    # 1️⃣  generator that HF will call under the hood
    # -------------------------------------------------
    def _example_stream(self) -> Iterable[Dict]:
        """
        Yields *tokenised* rows; each row already contains the fields that
        SFTTrainer expects (`input_ids`, `attention_mask`, optionally
        `pixel_values`, etc., depending on the processor).
        """
        for solve_data in self.solve_data_list:
            messages = self.harness.create_training_conversation(
                solve_data, is_training=self.is_training
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
