"""
Script d'extraction automatique des métadonnées audio et mise à jour de fixtures.json.
"""
import os
import json
import hashlib
import soundfile as sf
import mutagen
from mutagen.mp3 import MP3
from mutagen.flac import FLAC

AUDIO_EXTS = {"wav", "mp3", "flac"}


def get_audio_metadata(filepath):
    ext = filepath.split(".")[-1].lower()
    meta = {"type": ext}
    if ext == "wav":
        with sf.SoundFile(filepath) as f:
            meta["samplerate"] = f.samplerate
            meta["channels"] = f.channels
            meta["duration"] = round(len(f) / f.samplerate, 2)
    elif ext == "mp3":
        audio = MP3(filepath)
        meta["samplerate"] = audio.info.sample_rate
        meta["channels"] = audio.info.channels
        meta["duration"] = round(audio.info.length, 2)
        meta["bitrate"] = audio.info.bitrate
    elif ext == "flac":
        audio = FLAC(filepath)
        meta["samplerate"] = audio.info.sample_rate
        meta["channels"] = audio.info.channels
        meta["duration"] = round(audio.info.length, 2)
    return meta


def sha256sum(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    base = os.path.dirname(__file__)
    fixtures_path = os.path.join(base, "fixtures.json")
    files = [f for f in os.listdir(base) if f.split(".")[-1].lower() in AUDIO_EXTS]
    try:
        with open(fixtures_path, "r", encoding="utf-8") as f:
            fixtures = json.load(f)
    except Exception:
        fixtures = {}
    for fname in files:
        fpath = os.path.join(base, fname)
        meta = get_audio_metadata(fpath)
        meta["sha256"] = sha256sum(fpath)
        meta.setdefault("description", "")
        meta.setdefault("usage", "")
        meta.setdefault("copyright", "Libre de droits / démo")
        meta.setdefault("license", "CC0")
        fixtures[fname] = meta
    with open(fixtures_path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False)
    print(f"fixtures.json mis à jour avec {len(files)} fichiers audio.")

if __name__ == "__main__":
    main()
