"""
Test script for cleaning a single thought text using Gemma3.
"""
import os
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login

def clean_text(text: str) -> str:
    """
    Clean a single thought text using Gemma3.
    
    Args:
        text: The text to clean
        
    Returns:
        The cleaned text
    """
    # Check for HF token
    if not os.getenv("HUGGINGFACE_TOKEN"):
        raise ValueError("Please set your Hugging Face token as an environment variable: export HUGGINGFACE_TOKEN=your_token_here")
    
    # Login to Hugging Face
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    # Load model and processor
    model_id = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Structure the input
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text", 
                "text": "You are a text cleaning assistant. Your task is to clean and normalize text while preserving its meaning. Remove any unnecessary formatting, fix typos, and ensure proper punctuation."
            }]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }
    ]
    
    # Process input
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate response
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    # Decode and return response
    return processor.decode(generation, skip_special_tokens=True)

if __name__ == "__main__":
    # Test with a sample text
    test_text = "This is a tesst text. with some typos and  extra   spaces. The message is. this sentence and the one before it."
    try:
        cleaned = clean_text(test_text)
        print("Original text:", test_text)
        print("Cleaned text:", cleaned)
    except Exception as e:
        print(f"Error: {str(e)}") 