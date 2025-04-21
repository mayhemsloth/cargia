import os
import sys
from transcription import TranscriptionManager

def main():
    print("Testing TranscriptionManager...")
    
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
    
    # Check if model should be initialized
    if manager.settings.get("load_at_startup", False):
        print("\nChecking model initialization...")
        if manager.is_initialized():
            print("✅ Model initialized successfully")
        else:
            print("❌ Model failed to initialize")
            return 1
    else:
        print("\nSkipping model initialization (load_at_startup is false)")
    
    # Test audio settings
    print("\nChecking audio settings...")
    audio_settings = manager.get_audio_settings()
    print(f"Audio settings: {audio_settings}")
    
    # Cleanup
    print("\nCleaning up...")
    manager.cleanup()
    if not manager.is_initialized():
        print("✅ Cleanup successful")
    else:
        print("❌ Cleanup failed")
        return 1
    
    print("\nAll tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 