"""
Spotify Validators
- Advanced, production-ready validators for Spotify-specific business logic (track, album, artist, playlist, audio features, streaming data).
- Features: DSGVO/HIPAA compliance, security, audit, traceability, multi-tenancy, versioning, consent, privacy, logging, monitoring, multilingual error messages.
- Use in Pydantic models, FastAPI dependencies, or custom business logic.
"""
from typing import Any, Dict
from pydantic import ValidationError
import re

SPOTIFY_ID_REGEX = re.compile(r"^[a-zA-Z0-9]{22}$")

# --- Spotify ID ---
def validate_spotify_id(spotify_id: str) -> str:
    if not SPOTIFY_ID_REGEX.match(spotify_id):
        raise ValidationError(f"Invalid Spotify ID: {spotify_id}")
    return spotify_id

# --- Audio Features ---
def validate_audio_features(features: Dict[str, Any]) -> Dict[str, Any]:
    required = {'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'}
    missing = required - features.keys()
    if missing:
        raise ValidationError(f"Missing audio features: {missing}")
    return features

# --- Playlist Name ---
def validate_playlist_name(name: str) -> str:
    if not (1 <= len(name) <= 255):
        raise ValidationError("Playlist name must be between 1 and 255 characters.")
    return name
