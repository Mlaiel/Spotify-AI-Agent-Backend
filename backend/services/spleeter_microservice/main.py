
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse
import tempfile
import os
from spleeter.separator import Separator
import logging
from . import monitoring
from . import utils
from . import security
from . import config
from .health import router as health_router

app = FastAPI(title="Spleeter Microservice", description="API de séparation de stems audio avec Spleeter", version="1.0")
app.include_router(health_router)

@app.post("/separate")
async def separate_audio(request: Request, file: UploadFile = File(...):
    await security.verify_api_key(request)
    utils.validate_audio_file(file)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = utils.save_temp_file(file)
        # Séparation 2 stems (voix + accompagnement)
        separator = Separator('spleeter:2stems')
        separator.separate_to_file(input_path, tmpdir)
        output_dir = os.path.join(tmpdir, file.filename.split('.')[0])
        vocals = os.path.join(output_dir, "vocals.wav")
        accompaniment = os.path.join(output_dir, "accompaniment.wav")
        # Nettoyage du fichier temporaire
        utils.cleanup_temp_file(input_path)
        # Retourne le chemin du fichier voix (exemple)
        return FileResponse(vocals, media_type="audio/wav", filename="vocals.wav")
