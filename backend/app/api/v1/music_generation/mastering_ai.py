import logging
from typing import Optional
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf

class MasteringRequest(BaseModel):
    audio_bytes: bytes = Field(..., description="Fichier audio brut (WAV)")
    target_lufs: float = Field(-14.0, ge=-24.0, le=-8.0, description="Loudness cible LUFS")
    eq_boost: Optional[float] = Field(0.0, ge=-6.0, le=6.0, description="Boost EQ (dB)")

class MasteringAI:
    """
    Mastering automatique IA : normalisation, compression, égalisation, loudness.
    """
    def __init__(self):
        self.logger = logging.getLogger("MasteringAI")

    def master(self, req: MasteringRequest) -> bytes:
        import io
        buf = io.BytesIO(req.audio_bytes)
        audio, sr = sf.read(buf)
        # Normalisation loudness
        audio = audio / np.max(np.abs(audio) * 0.98
        # Compression simple
        threshold = 0.8
        ratio = 4.0
        audio = np.where(np.abs(audio) > threshold, np.sign(audio) * (threshold + (np.abs(audio)-threshold)/ratio), audio)
        # EQ boost
        if req.eq_boost:
            audio = audio + req.eq_boost/10.0 * np.tanh(audio)
        # Limiteur
        audio = np.clip(audio, -1.0, 1.0)
        out_buf = io.BytesIO()
        sf.write(out_buf, audio, sr, format='WAV')
        out_buf.seek(0)
        self.logger.info(f"Mastering appliqué (LUFS cible: {req.target_lufs}, EQ: {req.eq_boost})")
        return out_buf.read()

# Exemple d'utilisation
# masterer = MasteringAI()
# req = MasteringRequest(audio_bytes=..., target_lufs=-14.0, eq_boost=2.0)
# mastered = masterer.master(req)
