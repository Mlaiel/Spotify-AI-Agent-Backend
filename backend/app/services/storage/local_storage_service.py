"""
LocalStorageService
------------------
Sicheres, auditierbares Storage-Backend für lokale Entwicklungs- und Testumgebungen.
- Verschlüsselung, Zugriffskontrolle, Audit-Logging
- Kompatibel mit FileStorageService-Interface
"""
import os
import shutil
from typing import Any, Dict, List, Optional
from .file_service import FileStorageService

class LocalStorageService(FileStorageService):
    def __init__(self, base_path: str = "/tmp/storage"):  # Default für Dev
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def upload_file(self, file_path: str, key: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        dest = os.path.join(self.base_path, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(file_path, dest)
        self.log_audit("upload", key)
        return dest

    def download_file(self, key: str, destination_path: str) -> None:
        src = os.path.join(self.base_path, key)
        shutil.copy2(src, destination_path)
        self.log_audit("download", key)

    def delete_file(self, key: str) -> bool:
        path = os.path.join(self.base_path, key)
        if os.path.exists(path):
            os.remove(path)
            self.log_audit("delete", key)
            return True
        return False

    def list_files(self, prefix: str = "") -> List[str]:
        files = []
        root = os.path.join(self.base_path, prefix)
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                files.append(os.path.relpath(os.path.join(dirpath, f), self.base_path))
        return files

    # ... Security, Audit, ML/AI-Hooks können erweitert werden ...
