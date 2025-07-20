"""
Script de validation de fixtures.json et des fichiers audio associés.
"""
import os
import json

AUDIO_EXTS = {"wav", "mp3", "flac"}


def main():
    base = os.path.dirname(__file__)
    fixtures_path = os.path.join(base, "fixtures.json")
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)
    errors = []
    for fname, meta in fixtures.items():
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            errors.append(f"Fichier manquant: {fname}")
        if meta.get("type") != fname.split(".")[-1].lower():
            errors.append(f"Type incohérent pour {fname}")
        if "license" not in meta:
            errors.append(f"License manquante pour {fname}")
        if "sha256" not in meta:
            errors.append(f"SHA256 manquant pour {fname}")
    if errors:
        print("Erreurs de validation:")
        for e in errors:
            print("-", e)
    else:
        print("fixtures.json valide et cohérent.")

if __name__ == "__main__":
    main()
