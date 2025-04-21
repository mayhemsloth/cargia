import os
import sys
from transcription import TranscriptionManager
import time

def print_transcribed_text(text: str):
    """Callback function that prints transcribed text."""
    print(f"🎯 Transcribed: {text}")

def main():
    print("Testing Microphone and Transcription...")
    
    # Get the absolute path to the settings file
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    print(f"Using settings file at: {settings_path}")
    
    # Initialize the manager
    print("\nInitializing TranscriptionManager...")
    manager = TranscriptionManager(settings_path)
    
    # Check if settings were loaded
    print("\nChecking settings...")
    if not manager.settings:
        print("❌ Failed to load settings")
        return 1
    
    print("✅ Settings loaded successfully")
    print(f"Settings: {manager.settings}")
    
    # Check if model initialized
    print("\nChecking model initialization...")
    if not manager.is_initialized():
        print("❌ Model failed to initialize")
        return 1
    print("✅ Model initialized successfully")
    
    # Test audio settings
    print("\nAudio settings:")
    audio_settings = manager.get_audio_settings()
    print(f"Sample rate: {audio_settings['sample_rate']} Hz")
    print(f"Chunk size: {audio_settings['chunk_ms']} ms")
    print(f"Channels: {audio_settings['channels']}")
    print(f"Data type: {audio_settings['dtype']}")
    
    try:
        # Start transcription
        print("\n🎤 Starting transcription...")
        print("Speak into your microphone. Press Ctrl+C to stop.")
        success = manager.start_transcription(callback=print_transcribed_text)
        
        if not success:
            print("❌ Failed to start transcription")
            return 1
        
        # Keep the script running until Ctrl+C
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopping transcription...")
    finally:
        # Cleanup
        manager.stop_transcription()
        manager.cleanup()
        print("✅ Cleanup successful")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 