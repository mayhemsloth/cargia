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
import io
import wave
from PyQt6.QtCore import QTimer
import functools


SAMPLE_RATE = 16000
CHUNK_DURATION = 8
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


class TranscriptionManager:
    def __init__(self, settings_path: str = "settings.json"):
        """Initialize the transcription manager with settings from the specified path."""
        self.settings_path = settings_path
        self.model: Optional[WhisperModel] = None
        self.settings: Dict[str, Any] = {}
        self.load_settings()
        
        # Add audio recording attributes
        self.audio_queue = queue.Queue()
        self.audio_device = self.settings.get("audio_device", 43)
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.transcription_callback: Optional[Callable[[str], None]] = None
        self._model_lock = threading.Lock()  # Add thread lock for model access
        self._chunk_counter = 0

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
            print(f"⚠️ Audio status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())
            self._chunk_counter += 1
            if self._chunk_counter % 20 == 0:
                print(f"  → received {self._chunk_counter} chunks")
    
    def _record_audio(self):
        """Continuously read from the default mic into audio_queue."""
        def _cb(indata, frames, t, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_queue.put(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_cb):
            while self.is_recording:
                time.sleep(0.1)

    def _transcribe_loop(self):
        """Transciption loop"""
        model = self.model  # already initialized
        buffer = np.zeros((0,), dtype=np.int16)
        print("Transcriber started without overlap")
        while self.is_recording:
            # 1) Fill up 3 s
            while buffer.shape[0] < CHUNK_SIZE and self.is_recording:
                buffer = np.concatenate((buffer, np.squeeze(self.audio_queue.get())))
            if not self.is_recording:
                break
            # 2) Slice off exactly one chunk
            audio_chunk = buffer[:CHUNK_SIZE]
            buffer = buffer[CHUNK_SIZE:]
            # 3) Normalize and call whisper
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            segments, _ = model.transcribe(
                audio_float,
                beam_size=self.settings.get("beam_size", 5),
                language=self.settings.get("language", "en"),
                vad_filter=True
            )
            for seg in segments:
                text = seg.text.strip()
                if text and self.transcription_callback:
                    QTimer.singleShot(
                        0,
                        functools.partial(self.transcription_callback, text + " ")
                    )
    
    def start_transcription(self, callback: Callable[[str], None]) -> bool:
        if not self.initialize_model():
            return False

        if self.is_recording:
            return True

        self.transcription_callback = callback
        self.is_recording = True
        # clear any old audio
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # spawn both threads
        threading.Thread(target=self._record_audio, daemon=True).start()
        threading.Thread(target=self._transcribe_loop, daemon=True).start()

        return True
    
    def stop_transcription(self):
        self.is_recording = False
        self.transcription_callback = None
        # drain queue
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
    
    def cleanup(self) -> None:
        """Clean up resources used by the transcription model."""
        self.stop_transcription()  # Make sure recording is stopped
        self.model = None 