"""
Gemma3 wrapper for easy model initialization and inference.
"""
import os
# Disable TorchDynamo to avoid Triton DLL issues on Windows
# os.environ["TORCHDYNAMO_DISABLE"] = "1"
# os.environ["TORCH_LOGS"] = ""

import torch
from typing import List, Dict, Any, Optional, Union
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from PIL import Image


class Gemma3Wrapper:
    """
    A wrapper class for Gemma3 models that handles initialization,
    message formatting, and inference.
    """
    
    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        """
        Initialize the Gemma3 wrapper.
        
        Args:
            model_id: Hugging Face model ID
            device_map: Device mapping strategy
            torch_dtype: Torch data type for model
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        
        self.model = None
        self.processor = None
        self._is_initialized = False
        
        # Configure PyTorch for better performance
        self._configure_torch()
    
    def _configure_torch(self):
        """Configure PyTorch settings for optimal performance."""
        try:
            # Disable TorchDynamo to avoid Triton DLL issues on Windows
            torch._dynamo.config.suppress_errors = True
            # Force eager mode to avoid compilation issues
            torch._dynamo.config.disable = True
        except Exception:
            pass  # Fall back to eager mode if inductor not available

        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass  # Fall back to default precision
    
    def _check_authentication(self):
        """Check and handle Hugging Face authentication."""
        if not os.getenv("HUGGINGFACE_TOKEN"):
            raise ValueError(
                "Please set your Hugging Face token as an environment variable: "
                "export HUGGINGFACE_TOKEN=your_token_here"
            )
        
        # Login to Hugging Face
        login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    def initialize(self):
        """Initialize the model and processor."""
        if self._is_initialized:
            return
        
        print(f"Initializing Gemma3 model: {self.model_id}")
        
        # Check authentication
        self._check_authentication()
        
        # Load model and processor
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype
        ).eval()

        self.model.generation_config.top_k = None # to get rid of annoying warning
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=False) # setting use_fast=False to get rid of annoying warning
        
        self._is_initialized = True
        print(f"Model initialized successfully on device: {self.model.device}")
    
    
    def single_generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate text response from formatted messages.
        
        Args:
            messages: Formatted messages list
            max_new_tokens: Override default max_new_tokens
            do_sample: Override default do_sample
            temperature: Override default temperature
            top_p: Override default top_p
            
        Returns:
            Generated text response
        """
        if not self._is_initialized:
            self.initialize()
        
        # Use provided parameters or defaults
        max_new_tokens = max_new_tokens or self.max_new_tokens
        do_sample = do_sample if do_sample is not None else self.do_sample
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        
        # Process input
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p
            )
            generation = generation[0][input_len:]
        
        # Decode and return response
        decoded_text = self.processor.decode(generation, skip_special_tokens=True)

        # extra strip of the <end_of_turn> token in case it slipped through
        decoded_text = decoded_text.replace("<end_of_turn>", "")
        
        return decoded_text
    


    def format_system_prompt_message(self, system_prompt: str) -> Dict[str, Any]:
        """
        Format the system prompt message.

        Args:
            system_prompt (str): The system prompt

        Returns:
            Dict[str, Any]: The formatted system prompt message
        """

        return {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
        }
        
    
    def format_user_message(self, text: str =None, image: Image.Image= None) -> Dict[str, Any]:
        """
        Format a single user message into the appropriate dictionary message style for Gemma3

        Args:
            text (str): The text to format
            image (Image.Image): The image to format if it exists

        Returns:
            Dict[str, Any]: The formatted user message, usually used in a sequential list of messages
        """

        if text is None and image is None:
            raise ValueError("Either text or image must be provided")   
        
        content = []

        # images, when provided, are always put into the content before the text
        if image is not None:   
            content.append({"type": "image", "image": image})

        if text is not None:
            content.append({"type": "text", "text": text})
        
        return {
            "role": "user",
            "content": content
        }

    def clean_thought_text(self, text: str) -> str:
        """
        The method to call to clean thought text.

        Args:
            text (str): The text to clean

        Returns:
            str: The cleaned text
        """

        clean_thought_system_prompt = """
        You are a text cleaning assistant specializing in cleaning transcribed thought text. Your task is to clean and normalize text while preserving its exact meaning and intent.

        CLEANING RULES:
        1. Fix common transcription errors:
        - "um", "uh", "er" → remove
        - "like", "you know", "sort of", "kind of" → remove if they don't add meaning
        - Repeated words → keep only once
        - Stutters (e.g., "the the" → "the")

        2. Fix punctuation and grammar:
        - Add missing periods, commas, and apostrophes
        - Fix capitalization at sentence beginnings
        - Ensure proper sentence structure
        - Remove excessive punctuation (e.g., "???" → "?")

        3. Remove transcription artifacts:
        - "Thank you" or "Thank you." → remove completely
        - "[inaudible]", "[unclear]", "[background noise]" → remove
        - Speaker labels like "Speaker:", "User:" → remove

        4. Preserve important content:
        - Keep all technical terms, numbers, and specific details
        - Maintain logical flow and reasoning
        - Keep all references to grid positions, colors, patterns
        - Preserve mathematical or spatial relationships

        5. Formatting:
        - Convert to clean, readable sentences
        - Remove extra spaces and line breaks

        Respond ONLY with the cleaned text. Do not add explanations, comments, or markdown formatting.
        """

        system_message = self.format_system_prompt_message(clean_thought_system_prompt)
        user_message = self.format_user_message(text=text)

        messages = [system_message, user_message]

        return self.single_generate(messages)
    
    
    def get_device(self) -> torch.device:
        """Get the device the model is running on."""
        if not self._is_initialized:
            self.initialize()
        return self.model.device
    
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self._is_initialized 