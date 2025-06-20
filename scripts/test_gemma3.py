"""
Test script for Gemma3 installation and basic inference.
"""
import os
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login

def test_gemma3_installation():
    """Test basic Gemma3 functionality."""
    print("Testing Gemma3 installation...")
    
    # Check for HF token
    if not os.getenv("HUGGINGFACE_TOKEN"):
        print("Please set your Hugging Face token as an environment variable:")
        print("export HUGGINGFACE_TOKEN=your_token_here")
        return False
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    # Load model and processor
    model_id = "google/gemma-3-4b-it"
    print(f"Loading model: {model_id}")
    
    try:
        # Try to use inductor backend, fall back to eager if not available
        try:
            torch._dynamo.config.suppress_errors = True
            print("Using PyTorch inductor backend for better performance...")
        except Exception as e:
            print("Warning: Could not configure inductor backend, falling back to eager mode")
            print(f"Error: {str(e)}")
            print("\nTo enable better performance, install Triton:")
            print("pip install triton")
        
        # Load model and processor
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"Model and processor loaded successfully to device: {model.device}!")
        
        # Test basic inference
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Explain what a grid is in simple terms."}
                ]
            }
        ]
        
        print("\nTesting inference with prompt...")
        
        # Process input
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        # Decode and print response
        response = processor.decode(generation, skip_special_tokens=True)
        print("\nModel response:")
        print(response)
        
        return True
        
    except Exception as e:
        print(f"Error testing Gemma3: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemma3_installation() 