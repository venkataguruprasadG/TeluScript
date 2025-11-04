# src/live_asr.py

import sounddevice as sd
import numpy as np
import time

# --- 1. Import Project Component ---
# We import your transcription function from the other file you created
from transcriber import transcribe_chunk

# --- 2. Define Streaming Parameters ---
# These must match the sample rate expected by the Whisper model.
SAMPLE_RATE = 16000
# Chunk size: 0.5 seconds of audio for low latency
CHUNK_DURATION_SECONDS = 0.5
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_DURATION_SECONDS)
CHANNELS = 1

# List to store audio chunks received from the microphone
audio_buffer = []


# --- 3. The Stream Callback Function (Runs in a separate thread) ---
def callback(indata, frames, time_info, status):
    """
    Called automatically by sounddevice every time a new block of audio arrives.
    """
    if status:
        # Log any status warnings from the audio driver
        print(f"Audio Stream Status: {status}", flush=True)

    # Append the new audio chunk to the buffer for the main thread to process.
    # .copy() and .astype(np.float32) ensure thread safety and correct data format for Whisper.
    audio_buffer.append(indata.copy().flatten().astype(np.float32))


# --- 4. The Main Transcription Loop ---
def run_live_transcription():
    """Starts the audio stream and continuously checks the buffer for new data to transcribe."""
    print(f"Starting live Telugu ASR stream at {SAMPLE_RATE} Hz...")

    # sd.InputStream runs the `callback` function asynchronously.
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        channels=CHANNELS,
        callback=callback,
        dtype="float32",
    ):
        print("\n--- SPEAK IN TELUGU NOW (Press Ctrl+C to stop) ---\n")

        # This while loop runs continuously on the main thread
        while True:
            # Check if the separate audio thread has added any chunks to the buffer
            if audio_buffer:
                # 4a. Combine all chunks into one audio array
                audio_data = np.concatenate(audio_buffer, axis=0)

                # IMPORTANT: Clear the buffer immediately for the next incoming audio chunks
                audio_buffer.clear()

                # 4b. Transcribe the combined audio chunk
                try:
                    transcribed_text = transcribe_chunk(audio_data)

                    if transcribed_text:
                        # Print the result instantly!
                        print(f"LIVE TEXT: {transcribed_text}", flush=True)

                except Exception as e:
                    # Catch and report transcription errors without halting the audio stream
                    print(f"Transcription Error: {e}")

            # Pause briefly to prevent high CPU usage, optimizing for low latency
            time.sleep(CHUNK_DURATION_SECONDS / 4)


# --- 5. Execution ---
if __name__ == "__main__":
    try:
        run_live_transcription()
    except KeyboardInterrupt:
        print("\n--- Stopping Real-Time ASR. Goodbye! ---")
    except Exception as e:
        print(f"A fatal error occurred: {e}")
