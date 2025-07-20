import logging
from typing import Optional
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf

class BeatGenerationRequest(BaseModel):
    bpm: int = Field(90, ge=60, le=200, description="BPM du beat")
    style: str = Field("hiphop", description="Style du beat (hiphop, trap, pop, edm)")
    duration: int = Field(16, ge=4, le=64, description="Nombre de mesures")
    seed: Optional[int] = Field(None, description="Seed pour reproductibilité")

class BeatGenerator:
    """
    Générateur de beats IA avec patterns rythmiques paramétrables.
    """
    def __init__(self):
        self.logger = logging.getLogger("BeatGenerator")

    def generate(self, req: BeatGenerationRequest) -> bytes:
        np.random.seed(req.seed)
        sr = 44100
        beat_length = int(sr * 60 / req.bpm)
        total_length = beat_length * req.duration
        # Génération d'un pattern rythmique simple (kick, snare, hihat)
        audio = np.zeros(total_length)
        for i in range(req.duration):
            idx = i * beat_length
            audio[idx:idx+1000] += np.random.uniform(0.5, 1.0)  # Kick
            if i % 2 == 1:
                audio[idx+int(beat_length/2):idx+int(beat_length/2)+800] += np.random.uniform(0.3, 0.7)  # Snare
            audio[idx:idx+int(beat_length/8)] += np.random.uniform(0.05, 0.1)  # Hihat
        # Normalisation
        audio = audio / np.max(np.abs(audio))
        # Export WAV en mémoire
        import io
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        buf.seek(0)
        self.logger.info(f"Beat généré: {req.style}, {req.bpm} BPM, {req.duration} mesures")
        return buf.read()

# Exemple d'utilisation
# gen = BeatGenerator()
# req = BeatGenerationRequest(bpm=120, style="trap", duration=8)
# beat = gen.generate(req)
