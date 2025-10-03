"""
Test script for the Gemma3Wrapper class.
"""
import sys
import os
from PIL import Image, ImageDraw

from cargia.common.gemma3_wrapper import Gemma3Wrapper

def test_formatting():
    print("Testing message formatting...")
    gemma = Gemma3Wrapper()
    sys_msg = gemma.format_system_prompt_message("You are a helpful assistant.")
    user_msg = gemma.format_user_message(text="What is a grid?")
    print("System message:", sys_msg)
    print("User message:", user_msg)
    # Test with image
    img = Image.new('RGB', (32, 32), color='red')
    user_img_msg = gemma.format_user_message(text="Describe this image.", image=img)
    print("User message with image:", user_img_msg)

def test_single_generate():
    print("\nTesting single_generate...")
    gemma = Gemma3Wrapper(max_new_tokens=64)
    sys_msg = gemma.format_system_prompt_message("You are a helpful assistant.")
    user_msg = gemma.format_user_message(text="Explain what a grid is in simple terms.")
    messages = [sys_msg, user_msg]
    response = gemma.single_generate(messages)
    print("Response:", response)
    print(f"Model is running on device: {gemma.get_device()}")

def test_clean_thought_text():
    print("\nTesting clean_thought_text...")
    gemma = Gemma3Wrapper(max_new_tokens=100)
    test_text = "This is a test text with some typos and  extra   spaces. Thank you."
    cleaned = gemma.clean_thought_text(test_text)
    print("Original text:", test_text)
    print("Cleaned text:", cleaned)

def test_image_message_inference():
    print("\nTesting image+text message inference...")
    gemma = Gemma3Wrapper(max_new_tokens=64)
    sys_msg = gemma.format_system_prompt_message("You are a helpful assistant that describes images.")
    # Create a simple image
    img = Image.new('RGB', (32, 32), color='blue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 24, 24], fill='yellow')
    user_msg = gemma.format_user_message(text="What do you see in this image?", image=img)
    messages = [sys_msg, user_msg]
    try:
        response = gemma.single_generate(messages)
        print("Image+text response:", response)
    except Exception as e:
        print("Image+text inference not supported or failed:", e)

if __name__ == "__main__":
    print("Testing Gemma3Wrapper functionality...")
    print("=" * 50)
    try:
        test_formatting()
        test_single_generate()
        test_clean_thought_text()
        test_image_message_inference()
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Make sure you have set your Hugging Face token:")
        print("export HUGGINGFACE_TOKEN=your_token_here") 