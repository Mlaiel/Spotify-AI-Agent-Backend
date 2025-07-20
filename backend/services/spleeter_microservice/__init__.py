

"""
Spleeter Microservice – Spotify AI Agent

Ce package expose l'API FastAPI pour la séparation de stems audio via Spleeter.

Fonctionnalités clés :
- Sécurité avancée (API key, audit, logs)
- Monitoring Prometheus
- Prêt pour CI/CD, cloud, orchestration

Utilisation Python directe (hors HTTP) :
    from spleeter_microservice import separate_stems
    stems = separate_stems("/chemin/vers/fichier.wav", stems=2)

Voir README pour la documentation complète.
"""

from spleeter.separator import Separator
from .config import settings

__version__ = "1.0.0"

def separate_stems(filepath: str, stems: int = 2, output_dir: str = None) -> dict:
    """
    Sépare un fichier audio localement en utilisant Spleeter (sans API HTTP).
    Args:
        filepath (str): Chemin du fichier audio à séparer.
        stems (int): Nombre de stems (2, 4, 5).
        output_dir (str, optional): Dossier de sortie. Par défaut, un dossier temporaire est utilisé.
    Returns:
        dict: Chemins des stems générés.
    """
    import tempfile, os
    if output_dir is None:
        tmpdir = tempfile.TemporaryDirectory()
        outdir = tmpdir.name
    else:
        outdir = output_dir
        os.makedirs(outdir, exist_ok=True)
    model = f'spleeter:{stems}stems'
    separator = Separator(model)
    separator.separate_to_file(filepath, outdir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    stems_dir = os.path.join(outdir, base)
    result = {}
    for f in os.listdir(stems_dir):
        result[f] = os.path.join(stems_dir, f)
    return result
