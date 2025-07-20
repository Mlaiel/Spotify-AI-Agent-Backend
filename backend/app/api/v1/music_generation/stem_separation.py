import logging
from typing import List
from pydantic import BaseModel, Field

class StemSeparationRequest(BaseModel):
    audio_bytes: bytes = Field(..., description="Fichier audio (WAV)")
    stems: int = Field(4, ge=2, le=5, description="Nombre de stems à extraire (2, 4 ou 5)")

class StemSeparator:
    """
    Séparation de stems (voix, batterie, basse, etc.) à partir d'un fichier audio.
    Utilise Spleeter (Deezer) pour la séparation source.
    """
    def __init__(self):
        self.logger = logging.getLogger("StemSeparator")
        self._check_spleeter()

    def _check_spleeter(self):
        try:
            import spleeter
        except ImportError:
            self.logger.error("Spleeter n'est pas installé. Veuillez installer spleeter.")
            raise

    def separate(self, req: StemSeparationRequest) -> List[bytes]:
        import tempfile, os
        from spleeter.separator import Separator
        import soundfile as sf
        import io
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "input.wav")
            with open(in_path, "wb") as f:
                f.write(req.audio_bytes)
            separator = Separator(f"spleeter:{req.stems}stems")
            separator.separate_to_file(in_path, tmpdir)
            stems_dir = os.path.join(tmpdir, "input")
            stems = []
            for fname in sorted(os.listdir(stems_dir)):
                if fname.endswith(".wav"):
                    with open(os.path.join(stems_dir, fname), "rb") as f:
                        stems.append(f.read())
            self.logger.info(f"Séparation en {req.stems} stems réussie.")
            return stems

# Exemple d'utilisation
# sep = StemSeparator()
# req = StemSeparationRequest(audio_bytes=..., stems=4)
# stems = sep.separate(req)
