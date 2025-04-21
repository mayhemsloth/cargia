import json
import os
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any, Callable
import traceback
from huggingface_hub import snapshot_download
import queue
import threading
import time

class TranscriptionManager:
    def __init__(self, settings_path: str = "settings.json"):
        """Initialize the transcription manager with settings from the specified path."""
        self.settings_path = settings_path
        self.model: Optional[WhisperModel] = None
        self.settings: Dict[str, Any] = {}
        self.load_settings()
        
        # Add audio recording attributes
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.transcription_callback: Optional[Callable[[str], None]] = None
        self._model_lock = threading.Lock()  # Add thread lock for model access
        
        if self.settings.get("load_at_startup", False):
            self.initialize_model()
    
    def load_settings(self) -> None:
        """Load transcription settings from the settings file."""
        try:
            with open(self.settings_path, 'r') as f:
                all_settings = json.load(f)
                self.settings = all_settings.get("transcription", {})
        except Exception as e:
            print(f"Error loading transcription settings: {e}")
            self.settings = {}
        print(f"{self.settings=}")
    
    def initialize_model(self) -> bool:
        """Initialize the Whisper model with settings from the configuration."""
        try:
            if self.model is not None:
                return True  # Model already initialized
            
            # Get the model path from settings
            model_name = self.settings.get("model_path", "large-v3")
            device = self.settings.get("device", "cuda")
            compute_type = self.settings.get("compute_type", "float16")
            beam_size = self.settings.get("beam_size", 5)
            
            print(f"\nInitializing model with parameters:")
            print(f"Model name: {model_name}")
            print(f"Device: {device}")
            print(f"Compute type: {compute_type}")
            print(f"Beam size: {beam_size}")
            
            # Initialize the model with minimal parameters
            print("\nInitializing Whisper model...")
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type
            )
            print("✅ Model initialized successfully")
            return True
        except Exception as e:
            print(f"\n❌ Error initializing transcription model:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            self.model = None
            return False
    
    def is_initialized(self) -> bool:
        """Check if the transcription model is initialized."""
        return self.model is not None
    
    def get_audio_settings(self) -> Dict[str, Any]:
        """Get audio recording settings."""
        return {
            "sample_rate": 16000,
            "chunk_ms": 250,
            "channels": 1,
            "dtype": "float32"
        }
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input."""
        if status:
            print(f"Audio callback status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def _transcription_loop(self):
        """Main loop for audio recording and transcription."""
        try:
            audio_settings = self.get_audio_settings()
            with sd.InputStream(
                channels=audio_settings["channels"],
                samplerate=audio_settings["sample_rate"],
                dtype=audio_settings["dtype"],
                callback=self.audio_callback,
                blocksize=int(audio_settings["sample_rate"] * audio_settings["chunk_ms"] / 1000)
            ):
                print("🎤 Started recording...")
                while self.is_recording:
                    # Check if model is still initialized
                    if not self.is_initialized():
                        print("❌ Model not initialized, stopping transcription")
                        break
                    
                    # Collect audio chunks for 2 seconds
                    audio_chunks = []
                    timeout_counter = 0
                    chunk_duration = audio_settings["chunk_ms"] / 1000
                    max_chunks = int(2.0 / chunk_duration)
                    
                    while len(audio_chunks) < max_chunks and self.is_recording:
                        try:
                            chunk = self.audio_queue.get(timeout=0.1)
                            audio_chunks.append(chunk)
                            timeout_counter = 0
                        except queue.Empty:
                            timeout_counter += 1
                            if timeout_counter > 10:  # 1 second of silence
                                break
                    
                    if audio_chunks and self.is_recording:
                        # Convert chunks to numpy array
                        audio = np.concatenate(audio_chunks)
                        
                        # Safely access the model with lock
                        with self._model_lock:
                            if self.model is not None:
                                try:
                                    segments, _ = self.model.transcribe(
                                        audio,
                                        language=self.settings.get("language", "en"),
                                        beam_size=self.settings.get("beam_size", 5)
                                    )
                                    
                                    # Send transcribed text through callback
                                    text = " ".join(segment.text for segment in segments)
                                    if text.strip() and self.transcription_callback:
                                        self.transcription_callback(text.strip())
                                except Exception as e:
                                    print(f"Error during transcription: {e}")
                                    traceback.print_exc()
                            else:
                                print("❌ Model not available for transcription")
                                break
                
                print("🛑 Stopped recording")
        
        except Exception as e:
            print(f"Error in transcription loop: {e}")
            traceback.print_exc()
        finally:
            self.is_recording = False
    
    def start_transcription(self, callback: Callable[[str], None]) -> bool:
        """Start recording and transcribing audio."""
        # Verify model is initialized first
        if not self.is_initialized():
            print("Cannot start transcription: Model not initialized")
            return False
        
        if self.is_recording:
            print("Transcription already running")
            return True
        
        # Set up for new transcription session
        self.transcription_callback = callback
        self.is_recording = True
        self.audio_queue.queue.clear()  # Clear any old audio data
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._transcription_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Give the thread a moment to start and verify it's running
        time.sleep(0.5)
        if not self.recording_thread.is_alive():
            print("Failed to start recording thread")
            self.is_recording = False
            return False
        
        return True
    
    def stop_transcription(self):
        """Stop recording and transcribing audio."""
        self.is_recording = False
        if self.recording_thread:
            try:
                self.recording_thread.join(timeout=2.0)  # Wait up to 2 seconds
                if self.recording_thread.is_alive():
                    print("Warning: Recording thread did not stop cleanly")
            except Exception as e:
                print(f"Error stopping recording thread: {e}")
            self.recording_thread = None
        
        self.transcription_callback = None
        # Clear any remaining audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def cleanup(self) -> None:
        """Clean up resources used by the transcription model."""
        self.stop_transcription()  # Make sure recording is stopped
        self.model = None 