
import numpy as np
import soundfile as sf
from typing import Tuple, Any


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalise un signal audio entre -1 et 1."""

    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convertit un fichier audio en WAV 16kHz mono."""
    data, sr = sf.read(input_path)
    data = normalize_audio(data)
    sf.write(output_path, data, 16000, subtype='PCM_16')


def extract_features(audio: np.ndarray, sr: int) -> dict:
    """Extrait des features audio de base (RMS, ZCR, Spectral Centroid)."""
    import librosa
    features = {
        'rms': float(np.mean(librosa.feature.rms(y=audio))),
        'zcr': float(np.mean(librosa.feature.zero_crossing_rate(y=audio))),
        'centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    }
    return features
