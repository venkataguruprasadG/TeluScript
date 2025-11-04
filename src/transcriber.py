# src/transcriber.py

from faster_whisper import WhisperModel
import numpy as np

# --- 1. Model Parameters ---
# Choose a model size. 'small' or 'base' are good starting points for low-latency.
# 'medium' is more accurate but slower.
MODEL_SIZE = "small"  # You can change this to "small" or "medium" later

# --- 2. Load the Model and Tokenizer ---
# We load the model, which includes the ASR network AND the tokenizer.
# The 'device' and 'compute_type' are crucial for speed (CPU is default).
try:
    # Use 'cpu' initially. If you have an NVIDIA GPU, use 'cuda'.
    # Use 'int8' for faster inference on CPU.
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"Loaded Whisper model: {MODEL_SIZE} (int8 on CPU).")

except Exception as e:
    print(
        f"Error loading Whisper model. Did you install faster-whisper and have internet? Error: {e}"
    )
    # You may need to ensure a stable internet connection for the first run,
    # as the model weights are downloaded.


# --- 3. The Core Transcription Function ---
def transcribe_chunk(audio_chunk: np.ndarray) -> str:
    """
    Transcribes a raw audio chunk using the pre-trained Whisper model.

    Args:
        audio_chunk (np.ndarray): The raw audio array (must be 16000 Hz float32).

    Returns:
        str: The transcribed Telugu text.
    """

    # Whisper expects a flat 1D array of float32 data
    audio_data = audio_chunk.flatten().astype(np.float32)

    # 3a. Run the transcription. We specify the language (Telugu) and the task.
    # We use segments, even for a chunk, to get the full transcription data.
    segments, info = model.transcribe(
        audio_data,
        language="te",  # Crucial: Set the target language to Telugu
        task="transcribe",
        beam_size=5,  # Standard setting for better accuracy
    )

    # 3b. Compile the transcribed text
    full_text = []
    # Segments is a generator; we iterate through it once.
    for segment in segments:
        full_text.append(segment.text)

    return " ".join(full_text).strip()


# --- 4. Simple Test Block (Using dummy audio) ---
if __name__ == "__main__":
    # NOTE: Since we don't have a real Telugu speaker input yet,
    # this test will check if the model system works.

    # Create a 3-second silence chunk
    SAMPLE_RATE = 16000
    TEST_DURATION = 3.0
    TEST_FRAMES = int(SAMPLE_RATE * TEST_DURATION)
    dummy_audio = np.zeros((TEST_FRAMES, 1), dtype=np.float32)

    print(f"\nTesting transcription with {TEST_DURATION} seconds of silence...")

    # This will likely return empty or a transcription of background noise
    transcribed_text = transcribe_chunk(dummy_audio)

    print(f"Transcribed Text (Expected blank or noise): '{transcribed_text}'")

    # SUCCESS CHECK: If the script runs without crashing, the integration is successful.
