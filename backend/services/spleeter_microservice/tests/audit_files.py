"""
Script d'audit des fichiers audio : logge chaque ajout/suppression, conserve un historique JSON.
"""
import os
import json
import time

AUDIO_EXTS = {"wav", "mp3", "flac"}
HIST_PATH = os.path.join(os.path.dirname(__file__), "audit_log.json")


def scan_files():
    base = os.path.dirname(__file__)
    return sorted([f for f in os.listdir(base) if f.split(".")[-1].lower() in AUDIO_EXTS])


def load_history():
    if not os.path.exists(HIST_PATH):
        return {"files": [], "events": []}
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(hist):
    with open(HIST_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

def main():
    hist = load_history()
    current = scan_files()
    prev = hist.get("files", [])
    added = [f for f in current if f not in prev]
    removed = [f for f in prev if f not in current]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for f in added:
        hist["events"].append({"event": "added", "file": f, "date": now})
    for f in removed:
        hist["events"].append({"event": "removed", "file": f, "date": now})
    hist["files"] = current
    save_history(hist)
    print(f"Audit terminé. Ajoutés: {added}, Supprimés: {removed}")

if __name__ == "__main__":
    main()
