# src/audio_streamer.py

import sounddevice as sd
import numpy as np
import time

# --- 1. Define Parameters for the Audio Stream ---
# Standard sampling rate for high-quality speech (you must match your model's requirement!)
SAMPLE_RATE = 16000
# Duration of each audio chunk in seconds (small chunks = lower latency)
CHUNK_DURATION = 0.5
# Number of audio frames per chunk
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_DURATION)
# We only need one audio channel (mono)
CHANNELS = 1

# A list to store the captured audio chunks
audio_buffer = []


# --- 2. The Core Callback Function ---
# This function runs automatically every time a new chunk (BLOCKSIZE) of audio arrives.
def callback(indata, frames, time_info, status):
    """
    Called (from a separate thread) for each audio block.
    indata: NumPy array containing the new audio data.
    """
    if status:
        print(f"Audio Stream Warning: {status}", flush=True)

    # Crucial step: Convert the raw sounddevice data (indata) into a float array
    # and append it to our buffer for later processing.
    # We must ensure the data type is compatible (e.g., float32).
    # .copy() is used to ensure thread safety.
    audio_buffer.append(indata.copy())


# --- 3. The Main Streaming Function ---
def start_stream():
    """Starts the non-blocking audio recording stream."""
    print(f"Starting audio stream at {SAMPLE_RATE} Hz...")

    # We use sd.InputStream to process chunks in real-time.
    # The 'samplerate', 'blocksize', and 'channels' define the stream structure.
    # The 'callback' is the function executed on every new chunk.
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        channels=CHANNELS,
        callback=callback,
        dtype="float32",  # Defines the data type of the input NumPy array
    ):
        print("Listening... Press Ctrl+C to stop.")

        # Keep the main thread alive so the audio stream thread can continue running
        while True:
            # We can perform real-time ASR processing here, checking the audio_buffer
            # For now, we'll just wait.
            time.sleep(CHUNK_DURATION)


# --- 4. Execution ---
if __name__ == "__main__":
    try:
        start_stream()
    except KeyboardInterrupt:
        print("\nStopping stream.")
    except Exception as e:
        print(f"An error occurred: {e}")
