import os
import sys
import queue
import threading
import time

# Ensure ONNXRuntime DLLs are found (for Windows GPU VAD)
dll_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'onnxruntime', '.libs')
if os.path.isdir(dll_path):
    os.add_dll_directory(dll_path)

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QTextCursor

# Transcription parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 3          # seconds per chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MODEL_NAME = "large-v3-turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Queues for audio and text
audio_queue = queue.Queue()
text_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_callback):
        while True:
            time.sleep(1)

def transcribe_loop():
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    buffer = np.zeros((0,), dtype=np.int16)
    print("Transcriber started without overlap")
    while True:
        # Accumulate until one chunk ready
        while buffer.shape[0] < CHUNK_SIZE:
            buffer = np.concatenate((buffer, np.squeeze(audio_queue.get())))
        # Extract the chunk
        audio_chunk = buffer[:CHUNK_SIZE]
        # Remove used part
        buffer = buffer[CHUNK_SIZE:]
        # Normalize and transcribe
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        segments, info = model.transcribe(
            audio_float,
            beam_size=5,
            language="en",
            vad_filter=True
        )
        for segment in segments:
            # Push cleaned text
            text_queue.put(segment.text.strip())

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Whisper Transcription (No Overlap)")
        self.resize(800, 400)

        self.text_edit = QtWidgets.QTextEdit(self)
        self.text_edit.setAcceptRichText(False)
        self.setCentralWidget(self.text_edit)

        threading.Thread(target=record_audio, daemon=True).start()
        threading.Thread(target=transcribe_loop, daemon=True).start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_text)
        self.timer.start(100)

    def update_text(self):
        # Append text segments continuously
        while not text_queue.empty():
            segment = text_queue.get()
            doc = self.text_edit.document()
            cursor = QTextCursor(doc)
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(segment + " ")
            # Auto-scroll
            vsb = self.text_edit.verticalScrollBar()
            vsb.setValue(vsb.maximum())

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
