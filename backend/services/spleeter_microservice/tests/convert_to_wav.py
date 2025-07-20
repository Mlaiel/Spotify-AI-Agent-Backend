"""
Script de conversion automatique de fichiers audio vers WAV (pour compatibilité Spleeter).
"""
import os
import sys
import soundfile as sf
import librosa

AUDIO_EXTS = {"mp3", "flac"}


def convert_to_wav(filepath, outdir=None):
    y, sr = librosa.load(filepath, sr=None, mono=False)
    if outdir is None:
        outdir = os.path.dirname(filepath)
    base = os.path.splitext(os.path.basename(filepath)[0]
    outpath = os.path.join(outdir, base + ".wav")
    sf.write(outpath, y.T if y.ndim > 1 else y, sr)
    print(f"Converti: {filepath} -> {outpath}")
    return outpath


def main():
    base = os.path.dirname(__file__)
    files = [f for f in os.listdir(base) if f.split(".")[-1].lower() in AUDIO_EXTS]
    for f in files:
        convert_to_wav(os.path.join(base, f)

if __name__ == "__main__":
    main()
