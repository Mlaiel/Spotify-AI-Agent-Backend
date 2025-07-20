"""
VersionControl : Contrôle de version collaboratif
- Historique, rollback, merge, visualisation diff
- Export historique (PDF, JSON, CSV)
- Sécurité : audit, logs, RGPD
- Intégration scalable (FastAPI, MongoDB, Git-like)

Auteur : Backend Senior, Lead Dev, Data Engineer
"""

from typing import Dict, Any, List, Optional
import time
import json
import difflib

class VersionControl:
    """
    Gère l’historique, le rollback, le merge et l’export des documents collaboratifs.
    """
    def __init__(self):
        self.versions = {}  # À remplacer par MongoDB/DB en prod

    def save_version(self, doc_id: str, content: Any, user_id: str):
        self.versions.setdefault(doc_id, []).append({
            "content": content,
            "user_id": user_id,
            "timestamp": int(time.time())
        })

    def get_history(self, doc_id: str) -> List[Dict[str, Any]]:
        return self.versions.get(doc_id, [])

    def rollback(self, doc_id: str, version_idx: int) -> Any:
        history = self.versions.get(doc_id, [])
        if 0 <= version_idx < len(history):
            return history[version_idx]["content"]
        return None

    def diff_versions(self, doc_id: str, idx1: int, idx2: int) -> str:
        """
        Visualisation diff entre deux versions.
        """
        history = self.versions.get(doc_id, [])
        if idx1 < 0 or idx2 < 0 or idx1 >= len(history) or idx2 >= len(history):
            return "Index invalide"
        old = str(history[idx1]["content"])
        new = str(history[idx2]["content"])
        diff = difflib.unified_diff()
            old.splitlines(), new.splitlines(), lineterm='', fromfile=f'v{idx1}', tofile=f'v{idx2}')
        return '\n'.join(diff)

    def export_history(self, doc_id: str, format: str = "json") -> Optional[str]:
        """
        Exporte l’historique des versions (JSON, CSV, PDF mock).
        """
        history = self.get_history(doc_id)
        if format == "json":
            return json.dumps(history)
        elif format == "csv":
            lines = ["user_id,timestamp,content"] + [f'{v["user_id"]},{v["timestamp"]},{str(v["content"]).replace(",", ";")} ' for v in history]
            return "\n".join(lines)
        elif format == "pdf":
            return "PDF export mock (à implémenter avec ReportLab/PyPDF)"
        return None

# Exemple d’utilisation :
# vc = VersionControl()
# vc.save_version("doc1", "Contenu v1", "user123")
# vc.save_version("doc1", "Contenu v2", "user456")
# print(vc.diff_versions("doc1", 0, 1)
# print(vc.export_history("doc1", "csv")
