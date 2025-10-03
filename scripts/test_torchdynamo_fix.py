#!/usr/bin/env python3
"""
Test script to verify that TorchDynamo disabling fixes the Triton DLL issue.
"""
import sys

from cargia.common.gemma3_wrapper import Gemma3Wrapper

def test_gemma3_without_torchdynamo():
    """Test Gemma3 inference with TorchDynamo disabled."""
    try:
        print("Testing Gemma3 with TorchDynamo disabled...")
        
        # Initialize the wrapper
        wrapper = Gemma3Wrapper()
        
        # Test a simple text cleaning operation
        test_text = "This is a test text with some typos and bad grammer."
        print(f"Original text: {test_text}")
        
        # Clean the text
        cleaned_text = wrapper.clean_thought_text(test_text)
        print(f"Cleaned text: {cleaned_text}")
        
        print("✅ Test passed! TorchDynamo disabling works correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_gemma3_without_torchdynamo()
    sys.exit(0 if success else 1) 