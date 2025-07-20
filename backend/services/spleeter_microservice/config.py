"""
Configuration centralisée du microservice Spleeter.
"""
import os

class Settings:
    API_KEY: str = os.getenv("SPLEETER_API_KEY", "changeme")
    ENV: str = os.getenv("SPLEETER_ENV", "production")
    MAX_FILE_SIZE_MB: int = int(os.getenv("SPLEETER_MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}

settings = Settings()
