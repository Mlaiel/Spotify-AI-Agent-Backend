"""
🎵 Spotify AI Agent - Model Manager
==================================

Gestionnaire avancé des modèles de séparation audio avec support
de téléchargement, validation, versioning et optimisation.

🎖️ Développé par l'équipe d'experts enterprise
"""

import os
import json
import hashlib
import asyncio
import aiohttp
import aiofiles
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse
import logging
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import time

from .exceptions import ModelNotFoundError, SpleeterError
from .monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Informations sur un modèle Spleeter"""
    name: str
    version: str
    stems: int
    sample_rate: int
    size_mb: float
    checksum: str
    download_url: str
    description: str
    created_at: str
    tags: List[str]
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Crée depuis un dictionnaire"""
        return cls(**data)


class PretrainedModels:
    """Catalogue des modèles pré-entraînés disponibles"""
    
    # Modèles officiels Spleeter
    OFFICIAL_MODELS = {
        "spleeter:2stems-16kHz": ModelInfo(
            name="spleeter:2stems-16kHz",
            version="1.5.4",
            stems=2,
            sample_rate=16000,
            size_mb=85.2,
            checksum="a1b2c3d4e5f6",
            download_url="https://github.com/deezer/spleeter/releases/download/v1.5.4/2stems.tar.gz",
            description="Modèle 2 stems (vocals/accompaniment) 16kHz",
            created_at="2021-03-15T10:00:00Z",
            tags=["vocals", "accompaniment", "official"],
            performance_metrics={"accuracy": 0.85, "speed_factor": 2.3}
        ),
        "spleeter:2stems-8kHz": ModelInfo(
            name="spleeter:2stems-8kHz",
            version="1.5.4",
            stems=2,
            sample_rate=8000,
            size_mb=42.1,
            checksum="b2c3d4e5f6a1",
            download_url="https://github.com/deezer/spleeter/releases/download/v1.5.4/2stems-8kHz.tar.gz",
            description="Modèle 2 stems (vocals/accompaniment) 8kHz - version légère",
            created_at="2021-03-15T10:00:00Z",
            tags=["vocals", "accompaniment", "official", "lightweight"],
            performance_metrics={"accuracy": 0.78, "speed_factor": 4.1}
        ),
        "spleeter:4stems-16kHz": ModelInfo(
            name="spleeter:4stems-16kHz",
            version="1.5.4",
            stems=4,
            sample_rate=16000,
            size_mb=142.8,
            checksum="c3d4e5f6a1b2",
            download_url="https://github.com/deezer/spleeter/releases/download/v1.5.4/4stems.tar.gz",
            description="Modèle 4 stems (vocals/drums/bass/other) 16kHz",
            created_at="2021-03-15T10:00:00Z",
            tags=["vocals", "drums", "bass", "other", "official"],
            performance_metrics={"accuracy": 0.82, "speed_factor": 1.8}
        ),
        "spleeter:4stems-8kHz": ModelInfo(
            name="spleeter:4stems-8kHz",
            version="1.5.4",
            stems=4,
            sample_rate=8000,
            size_mb=71.4,
            checksum="d4e5f6a1b2c3",
            download_url="https://github.com/deezer/spleeter/releases/download/v1.5.4/4stems-8kHz.tar.gz",
            description="Modèle 4 stems (vocals/drums/bass/other) 8kHz - version légère",
            created_at="2021-03-15T10:00:00Z",
            tags=["vocals", "drums", "bass", "other", "official", "lightweight"],
            performance_metrics={"accuracy": 0.75, "speed_factor": 3.2}
        ),
        "spleeter:5stems-16kHz": ModelInfo(
            name="spleeter:5stems-16kHz",
            version="1.5.4",
            stems=5,
            sample_rate=16000,
            size_mb=178.5,
            checksum="e5f6a1b2c3d4",
            download_url="https://github.com/deezer/spleeter/releases/download/v1.5.4/5stems.tar.gz",
            description="Modèle 5 stems (vocals/drums/bass/piano/other) 16kHz",
            created_at="2021-03-15T10:00:00Z",
            tags=["vocals", "drums", "bass", "piano", "other", "official"],
            performance_metrics={"accuracy": 0.79, "speed_factor": 1.5}
        ),
        "spleeter:5stems-8kHz": ModelInfo(
            name="spleeter:5stems-8kHz",
            version="1.5.4",
            stems=5,
            sample_rate=8000,
            size_mb=89.3,
            checksum="f6a1b2c3d4e5",
            download_url="https://github.com/deezer/spleeter/releases/download/v1.5.4/5stems-8kHz.tar.gz",
            description="Modèle 5 stems (vocals/drums/bass/piano/other) 8kHz - version légère",
            created_at="2021-03-15T10:00:00Z",
            tags=["vocals", "drums", "bass", "piano", "other", "official", "lightweight"],
            performance_metrics={"accuracy": 0.72, "speed_factor": 2.8}
        )
    }
    
    # Modèles communautaires (exemples)
    COMMUNITY_MODELS = {
        "community:vocals-enhanced": ModelInfo(
            name="community:vocals-enhanced",
            version="2.0.1",
            stems=2,
            sample_rate=44100,
            size_mb=156.7,
            checksum="1a2b3c4d5e6f",
            download_url="https://example.com/models/vocals-enhanced.tar.gz",
            description="Modèle communautaire optimisé pour l'extraction vocale haute qualité",
            created_at="2024-01-15T14:30:00Z",
            tags=["vocals", "enhanced", "community", "high-quality"],
            performance_metrics={"accuracy": 0.91, "speed_factor": 1.2}
        ),
        "community:instrumental-pro": ModelInfo(
            name="community:instrumental-pro",
            version="1.8.2",
            stems=8,
            sample_rate=48000,
            size_mb=245.1,
            checksum="2b3c4d5e6f1a",
            download_url="https://example.com/models/instrumental-pro.tar.gz",
            description="Modèle communautaire pour séparation instrumentale avancée",
            created_at="2024-02-20T09:15:00Z",
            tags=["instrumental", "pro", "community", "8-stems"],
            performance_metrics={"accuracy": 0.87, "speed_factor": 0.8}
        )
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ModelInfo]:
        """Retourne tous les modèles disponibles"""
        return {**cls.OFFICIAL_MODELS, **cls.COMMUNITY_MODELS}
    
    @classmethod
    def get_official_models(cls) -> Dict[str, ModelInfo]:
        """Retourne seulement les modèles officiels"""
        return cls.OFFICIAL_MODELS.copy()
    
    @classmethod
    def get_community_models(cls) -> Dict[str, ModelInfo]:
        """Retourne seulement les modèles communautaires"""
        return cls.COMMUNITY_MODELS.copy()
    
    @classmethod
    def search_models(
        cls,
        query: str = "",
        stems: Optional[int] = None,
        sample_rate: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, ModelInfo]:
        """
        Recherche des modèles selon des critères
        
        Args:
            query: Texte de recherche
            stems: Nombre de stems
            sample_rate: Fréquence d'échantillonnage
            tags: Tags requis
            
        Returns:
            Dictionnaire des modèles correspondants
        """
        all_models = cls.get_all_models()
        results = {}
        
        for name, model in all_models.items():
            # Filtre par query
            if query and query.lower() not in model.name.lower() and query.lower() not in model.description.lower():
                continue
            
            # Filtre par stems
            if stems is not None and model.stems != stems:
                continue
            
            # Filtre par sample_rate
            if sample_rate is not None and model.sample_rate != sample_rate:
                continue
            
            # Filtre par tags
            if tags:
                if not all(tag in model.tags for tag in tags):
                    continue
            
            results[name] = model
        
        return results


class ModelManager:
    """
    Gestionnaire avancé des modèles Spleeter
    
    Features:
    - Téléchargement automatique des modèles
    - Validation et vérification d'intégrité
    - Cache local intelligent
    - Versioning des modèles
    - Métriques de performance
    """
    
    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True,
        auto_download: bool = True,
        max_cache_size_gb: float = 5.0
    ):
        """
        Initialise le gestionnaire de modèles
        
        Args:
            models_dir: Répertoire des modèles
            cache_enabled: Activer le cache
            auto_download: Téléchargement automatique
            max_cache_size_gb: Taille max du cache en GB
        """
        self.models_dir = Path(models_dir or os.path.expanduser("~/.spleeter/models"))
        self.cache_enabled = cache_enabled
        self.auto_download = auto_download
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Conversion en bytes
        
        # Créer le répertoire des modèles
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Fichier de métadonnées
        self.metadata_file = self.models_dir / "models_metadata.json"
        
        # État interne
        self._local_models = {}
        self._download_progress = {}
        self._session = None
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Monitoring
        self.monitor = PerformanceMonitor()
        
        # Charger les métadonnées existantes
        self._load_metadata()
        
        logger.info(f"ModelManager initialisé: {self.models_dir}")
    
    def _load_metadata(self):
        """Charge les métadonnées des modèles locaux"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                for name, data in metadata.items():
                    self._local_models[name] = ModelInfo.from_dict(data)
                
                logger.info(f"Métadonnées chargées: {len(self._local_models)} modèles")
                
            except Exception as e:
                logger.error(f"Erreur chargement métadonnées: {e}")
                self._local_models = {}
    
    def _save_metadata(self):
        """Sauvegarde les métadonnées des modèles"""
        try:
            metadata = {
                name: model.to_dict()
                for name, model in self._local_models.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug("Métadonnées sauvegardées")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde métadonnées: {e}")
    
    async def get_model_path(self, model_name: str) -> Path:
        """
        Retourne le chemin local d'un modèle
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Chemin vers le modèle local
            
        Raises:
            ModelNotFoundError: Si le modèle n'est pas disponible
        """
        # Vérifier si le modèle est déjà local
        model_path = self.models_dir / model_name
        
        if model_path.exists() and self._is_model_valid(model_name):
            return model_path
        
        # Télécharger le modèle si auto_download activé
        if self.auto_download:
            await self.download_model(model_name)
            return model_path
        
        raise ModelNotFoundError(f"Modèle non trouvé: {model_name}")
    
    def _is_model_valid(self, model_name: str) -> bool:
        """
        Vérifie la validité d'un modèle local
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            True si le modèle est valide
        """
        if model_name not in self._local_models:
            return False
        
        model_path = self.models_dir / model_name
        if not model_path.exists():
            return False
        
        # Vérification du checksum si disponible
        model_info = self._local_models[model_name]
        if model_info.checksum:
            calculated_checksum = self._calculate_checksum(model_path)
            if calculated_checksum != model_info.checksum:
                logger.warning(f"Checksum invalide pour {model_name}")
                return False
        
        return True
    
    def _calculate_checksum(self, model_path: Path) -> str:
        """
        Calcule le checksum d'un modèle
        
        Args:
            model_path: Chemin vers le modèle
            
        Returns:
            Checksum MD5 en hexadécimal
        """
        hasher = hashlib.md5()
        
        if model_path.is_file():
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        else:
            # Checksum du répertoire
            for file_path in sorted(model_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def download_model(
        self,
        model_name: str,
        force: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """
        Télécharge un modèle
        
        Args:
            model_name: Nom du modèle
            force: Forcer le téléchargement même si déjà présent
            progress_callback: Callback pour le progrès
            
        Returns:
            Chemin vers le modèle téléchargé
            
        Raises:
            ModelNotFoundError: Si le modèle n'existe pas
            SpleeterError: En cas d'erreur de téléchargement
        """
        # Vérifier si le modèle existe
        all_models = PretrainedModels.get_all_models()
        if model_name not in all_models:
            raise ModelNotFoundError(f"Modèle inconnu: {model_name}")
        
        model_info = all_models[model_name]
        model_path = self.models_dir / model_name
        
        # Vérifier si déjà présent et valide
        if not force and model_path.exists() and self._is_model_valid(model_name):
            logger.info(f"Modèle déjà présent: {model_name}")
            return model_path
        
        # Vérifier l'espace disque
        await self._ensure_disk_space(model_info.size_mb * 1024 * 1024)
        
        try:
            self.monitor.start_timer(f"download_{model_name}")
            
            logger.info(f"Téléchargement du modèle: {model_name}")
            
            # Téléchargement
            downloaded_file = await self._download_file(
                model_info.download_url,
                model_name,
                progress_callback
            )
            
            # Extraction si nécessaire
            if downloaded_file.suffix in ['.tar', '.gz', '.zip']:
                await self._extract_model(downloaded_file, model_path)
                downloaded_file.unlink()  # Supprimer l'archive
            else:
                downloaded_file.rename(model_path)
            
            # Validation du modèle téléchargé
            if model_info.checksum:
                calculated_checksum = self._calculate_checksum(model_path)
                if calculated_checksum != model_info.checksum:
                    model_path.unlink() if model_path.is_file() else shutil.rmtree(model_path)
                    raise SpleeterError(f"Checksum invalide pour {model_name}")
            
            # Mise à jour des métadonnées
            self._local_models[model_name] = model_info
            self._save_metadata()
            
            download_time = self.monitor.end_timer(f"download_{model_name}")
            logger.info(f"Modèle téléchargé en {download_time:.2f}s: {model_name}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Erreur téléchargement {model_name}: {e}")
            raise SpleeterError(f"Échec téléchargement: {e}")
    
    async def _download_file(
        self,
        url: str,
        model_name: str,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """
        Télécharge un fichier depuis une URL
        
        Args:
            url: URL du fichier
            model_name: Nom du modèle
            progress_callback: Callback pour le progrès
            
        Returns:
            Chemin vers le fichier téléchargé
        """
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=3600)  # 1 heure
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        # Nom du fichier temporaire
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name or f"{model_name}.tar.gz"
        temp_file = self.models_dir / f"downloading_{filename}"
        
        try:
            async with self._session.get(url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                self._download_progress[model_name] = {
                    'total': total_size,
                    'downloaded': 0,
                    'percentage': 0.0
                }
                
                async with aiofiles.open(temp_file, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Mise à jour du progrès
                        if total_size > 0:
                            percentage = (downloaded_size / total_size) * 100
                            self._download_progress[model_name].update({
                                'downloaded': downloaded_size,
                                'percentage': percentage
                            })
                            
                            if progress_callback:
                                progress_callback(downloaded_size, total_size, percentage)
                
                return temp_file
                
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    async def _extract_model(self, archive_path: Path, destination: Path):
        """
        Extrait une archive de modèle
        
        Args:
            archive_path: Chemin vers l'archive
            destination: Répertoire de destination
        """
        def _extract():
            if archive_path.suffix == '.zip':
                import zipfile
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(destination.parent)
            else:
                import tarfile
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(destination.parent)
        
        # Extraction dans un thread pour éviter le blocage
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._thread_pool, _extract)
    
    async def _ensure_disk_space(self, required_bytes: int):
        """
        S'assure qu'il y a assez d'espace disque
        
        Args:
            required_bytes: Espace requis en bytes
        """
        # Vérifier l'espace libre
        stat = os.statvfs(self.models_dir)
        free_bytes = stat.f_frsize * stat.f_avail
        
        if free_bytes < required_bytes:
            # Nettoyer le cache si nécessaire
            await self._cleanup_cache(required_bytes - free_bytes)
            
            # Revérifier
            stat = os.statvfs(self.models_dir)
            free_bytes = stat.f_frsize * stat.f_avail
            
            if free_bytes < required_bytes:
                raise SpleeterError(f"Espace disque insuffisant: {free_bytes} < {required_bytes}")
    
    async def _cleanup_cache(self, bytes_to_free: int):
        """
        Nettoie le cache pour libérer de l'espace
        
        Args:
            bytes_to_free: Nombre de bytes à libérer
        """
        if not self.cache_enabled:
            return
        
        # Obtenir les modèles triés par date d'accès
        models_by_access = []
        for name, model_info in self._local_models.items():
            model_path = self.models_dir / name
            if model_path.exists():
                access_time = model_path.stat().st_atime
                size = self._get_directory_size(model_path)
                models_by_access.append((access_time, name, size))
        
        # Trier par date d'accès (plus ancien en premier)
        models_by_access.sort()
        
        freed_bytes = 0
        for _, model_name, size in models_by_access:
            if freed_bytes >= bytes_to_free:
                break
            
            await self.remove_model(model_name)
            freed_bytes += size
            logger.info(f"Modèle supprimé du cache: {model_name} ({size} bytes)")
    
    def _get_directory_size(self, path: Path) -> int:
        """
        Calcule la taille d'un répertoire
        
        Args:
            path: Chemin vers le répertoire
            
        Returns:
            Taille en bytes
        """
        if path.is_file():
            return path.stat().st_size
        
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    async def remove_model(self, model_name: str):
        """
        Supprime un modèle du cache local
        
        Args:
            model_name: Nom du modèle à supprimer
        """
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            if model_path.is_file():
                model_path.unlink()
            else:
                shutil.rmtree(model_path)
            
            logger.info(f"Modèle supprimé: {model_name}")
        
        # Mise à jour des métadonnées
        if model_name in self._local_models:
            del self._local_models[model_name]
            self._save_metadata()
    
    def list_local_models(self) -> Dict[str, ModelInfo]:
        """
        Liste les modèles disponibles localement
        
        Returns:
            Dictionnaire des modèles locaux
        """
        return self._local_models.copy()
    
    def list_available_models(self) -> Dict[str, ModelInfo]:
        """
        Liste tous les modèles disponibles (locaux + distants)
        
        Returns:
            Dictionnaire de tous les modèles
        """
        return PretrainedModels.get_all_models()
    
    def get_download_progress(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retourne le progrès de téléchargement d'un modèle
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Dictionnaire du progrès ou None
        """
        return self._download_progress.get(model_name)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le cache
        
        Returns:
            Dictionnaire d'informations
        """
        total_size = sum(
            self._get_directory_size(self.models_dir / name)
            for name in self._local_models.keys()
            if (self.models_dir / name).exists()
        )
        
        return {
            "models_count": len(self._local_models),
            "total_size_mb": total_size / 1024 / 1024,
            "max_size_gb": self.max_cache_size / 1024 / 1024 / 1024,
            "usage_percentage": (total_size / self.max_cache_size) * 100,
            "cache_dir": str(self.models_dir)
        }
    
    async def validate_all_models(self) -> Dict[str, bool]:
        """
        Valide tous les modèles locaux
        
        Returns:
            Dictionnaire de validation par modèle
        """
        results = {}
        
        for model_name in self._local_models.keys():
            try:
                results[model_name] = self._is_model_valid(model_name)
            except Exception as e:
                logger.error(f"Erreur validation {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    async def update_model(self, model_name: str) -> bool:
        """
        Met à jour un modèle vers la dernière version
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            True si mis à jour, False sinon
        """
        if model_name not in self._local_models:
            return False
        
        # Vérifier s'il y a une nouvelle version
        available_models = PretrainedModels.get_all_models()
        if model_name not in available_models:
            return False
        
        local_model = self._local_models[model_name]
        available_model = available_models[model_name]
        
        if local_model.version >= available_model.version:
            logger.info(f"Modèle déjà à jour: {model_name}")
            return False
        
        # Télécharger la nouvelle version
        await self.download_model(model_name, force=True)
        logger.info(f"Modèle mis à jour: {model_name} v{local_model.version} -> v{available_model.version}")
        
        return True
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        if self._session:
            await self._session.close()
        
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
        
        await self.monitor.cleanup()
        logger.info("ModelManager nettoyé")
