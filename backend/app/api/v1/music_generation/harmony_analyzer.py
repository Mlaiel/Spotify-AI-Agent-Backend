import logging
from typing import Dict
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf

class HarmonyAnalysisRequest(BaseModel):
    audio_bytes: bytes = Field(..., description="Fichier audio (WAV)")

class HarmonyAnalyzer:
    """
    Analyse harmonique d'un fichier audio : clé, accords, tonalité, structure.
    Utilise librosa et/ou essentia pour l'analyse musicale.
    """
    def __init__(self):
        self.logger = logging.getLogger("HarmonyAnalyzer")

    def analyze(self, req: HarmonyAnalysisRequest) -> Dict:
        import io
        import librosa
        buf = io.BytesIO(req.audio_bytes)
        y, sr = sf.read(buf)
        if y.ndim > 1:
            y = y.mean(axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key = librosa.feature.tonnetz(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Analyse simple de la clé (majeur/mineur)
        chroma_mean = chroma.mean(axis=1)
        key_idx = np.argmax(chroma_mean)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_idx]
        self.logger.info(f"Analyse harmonique: clé détectée {detected_key}, tempo {tempo:.1f} BPM")
        return {
            "key": detected_key,
            "tempo": tempo,
            "chroma": chroma_mean.tolist(),
            "tonnetz": key.mean(axis=1).tolist()
        }

# Exemple d'utilisation
# analyzer = HarmonyAnalyzer()
# req = HarmonyAnalysisRequest(audio_bytes=...)
# result = analyzer.analyze(req)
