# src/features.py

import librosa
import numpy as np

# --- 1. Define Parameters (Must match the streaming parameters!) ---
SAMPLE_RATE = 16000
# The window size (N_FFT) determines the frequency resolution.
N_FFT = 512
# The step size (HOP_LENGTH) determines the time resolution.
HOP_LENGTH = 160
# The number of Mel bins (N_MELS) is the final feature dimension.
N_MELS = 80  # A common value for many ASR models (e.g., Whisper, Conformer)


# --- 2. The Feature Extraction Function ---
def extract_mel_spectrogram(audio_chunk: np.ndarray) -> np.ndarray:
    """
    Converts a raw audio NumPy array chunk into a Mel Spectrogram.

    Args:
        audio_chunk (np.ndarray): A raw audio array (e.g., from sounddevice)
                                   with shape (frames, 1).

    Returns:
        np.ndarray: A Mel Spectrogram feature matrix ready for the model.
    """

    # 2a. Reshape and ensure the audio is mono (if necessary, though sounddevice handles it)
    audio_chunk = audio_chunk.flatten()

    # 2b. Compute the Mel Spectrogram using librosa
    # sr=SAMPLE_RATE tells librosa the rate of the incoming audio.
    mel_features = librosa.feature.melspectrogram(
        y=audio_chunk, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    # 2c. Convert power to decibels (dB). Models prefer features in a log-scale.
    # This step is critical for normalizing the sound's volume.
    mel_db = librosa.power_to_db(mel_features, ref=np.max)

    # 2d. Transpose the matrix.
    # Librosa outputs (Mels, Frames), but ML models prefer (Frames, Mels).
    # The final shape will be (Time Steps, Feature Dimension), e.g., (X, 80)
    return mel_db.T


# --- 3. Simple Test Block ---
if __name__ == "__main__":
    # Create a dummy audio chunk (1 second of silence) for testing
    TEST_DURATION = 1.0
    TEST_FRAMES = int(SAMPLE_RATE * TEST_DURATION)
    dummy_audio = np.zeros((TEST_FRAMES, 1), dtype=np.float32)

    print(f"Input audio shape: {dummy_audio.shape}")

    # Perform the extraction
    features = extract_mel_spectrogram(dummy_audio)

    print(f"Output Mel Spectrogram shape (Time Steps, Features): {features.shape}")
    print(f"Feature Dimension (N_MELS): {features.shape[1]}")

    # Expected N_MELS should be 80. The Time Steps (frames) depend on the HOP_LENGTH.
