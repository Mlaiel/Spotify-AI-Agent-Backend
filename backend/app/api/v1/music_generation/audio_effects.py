import logging
from typing import Optional, Literal
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf

class AudioEffectsRequest(BaseModel):
    effect: Literal["reverb", "eq", "pitch", "reverse", "fadein", "fadeout"] = Field(..., description="Type d'effet à appliquer")
    intensity: float = Field(0.5, ge=0.0, le=1.0, description="Intensité de l'effet")
    audio_bytes: bytes = Field(..., description="Fichier audio en bytes (WAV)")

class AudioEffects:
    """
    Application d'effets audio avancés sur des fichiers audio (WAV).
    """
    def __init__(self):
        self.logger = logging.getLogger("AudioEffects")

    def apply(self, req: AudioEffectsRequest) -> bytes:
        import io
        buf = io.BytesIO(req.audio_bytes)
        audio, sr = sf.read(buf)
        if req.effect == "reverse":
            audio = audio[::-1]
        elif req.effect == "fadein":
            fade_len = int(len(audio) * req.intensity)
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
        elif req.effect == "fadeout":
            fade_len = int(len(audio) * req.intensity)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        elif req.effect == "pitch":
            from librosa.effects import pitch_shift
            audio = pitch_shift(audio, sr, n_steps=2*req.intensity)
        elif req.effect == "reverb":
            audio = audio + np.convolve(audio, np.ones(int(1000*req.intensity)/1000, mode='full')[:len(audio)]
        elif req.effect == "eq":
            audio = audio * (1-req.intensity) + np.clip(audio, -0.2, 0.2) * req.intensity
        else:
            raise ValueError("Effet non supporté")
        # Normalisation
        audio = audio / np.max(np.abs(audio)
        out_buf = io.BytesIO()
        sf.write(out_buf, audio, sr, format='WAV')
        out_buf.seek(0)
        self.logger.info(f"Effet {req.effect} appliqué (intensité {req.intensity})")
        return out_buf.read()

# Exemple d'utilisation
# fx = AudioEffects()
# req = AudioEffectsRequest(effect="reverb", intensity=0.7, audio_bytes=...)
# audio_fx = fx.apply(req)
