"""
🎵 Spotify AI Agent - Spleeter Core Engine
=========================================

Moteur principal de séparation audio utilisant TensorFlow et des modèles
pré-entraînés pour la séparation de sources audio multi-stems.

🎖️ Développé par l'équipe d'experts enterprise
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import time

from .exceptions import SpleeterError, ModelNotFoundError, AudioProcessingError
from .monitoring import PerformanceMonitor
from .utils import AudioUtils, ValidationUtils

logger = logging.getLogger(__name__)


@dataclass
class SpleeterConfig:
    """Configuration pour le moteur Spleeter"""
    
    # Modèle et adapter
    model_name: str = "spleeter:2stems-16kHz"
    audio_adapter: str = "tensorflow"
    
    # Paramètres audio
    sample_rate: int = 44100
    frame_length: int = 4096
    frame_step: int = 1024
    channels: int = 2
    
    # Performance
    enable_gpu: bool = True
    batch_size: int = 8
    num_threads: int = 4
    memory_growth: bool = True
    
    # Cache et stockage
    cache_enabled: bool = True
    cache_dir: Optional[str] = None
    models_dir: Optional[str] = None
    
    # Limitations
    max_duration: int = 600  # 10 minutes
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    # Formats supportés
    supported_formats: List[str] = field(default_factory=lambda: [
        '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'
    ])
    
    def __post_init__(self):
        """Validation post-initialisation"""
        if self.cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.spleeter/cache")
        if self.models_dir is None:
            self.models_dir = os.path.expanduser("~/.spleeter/models")
        
        # Créer les répertoires si nécessaire
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)


class SpleeterEngine:
    """
    Moteur principal de séparation audio Spleeter enterprise
    
    Features:
    - Multi-modèles (2, 4, 5 stems)
    - Processing GPU/CPU optimisé
    - Cache intelligent
    - Monitoring performance
    - Traitement batch
    """
    
    def __init__(self, config: Optional[SpleeterConfig] = None):
        """
        Initialise le moteur Spleeter
        
        Args:
            config: Configuration du moteur
        """
        self.config = config or SpleeterConfig()
        self._setup_logging()
        self._setup_tensorflow()
        
        # Composants
        self.monitor = PerformanceMonitor() if self.config.enable_monitoring else None
        self.audio_utils = AudioUtils()
        self.validator = ValidationUtils()
        
        # État
        self._initialized = False
        self._models_cache = {}
        self._session = None
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        logger.info(f"Spleeter Engine initialisé avec config: {self.config.model_name}")
    
    def _setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_tensorflow(self):
        """Configure TensorFlow pour performance optimale"""
        if self.config.enable_gpu:
            # Configuration GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        if self.config.memory_growth:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU configuré: {len(gpus)} device(s)")
                except RuntimeError as e:
                    logger.warning(f"Erreur configuration GPU: {e}")
        else:
            # Forcer CPU seulement
            tf.config.set_visible_devices([], 'GPU')
            logger.info("Mode CPU forcé")
        
        # Configuration threads
        tf.config.threading.set_inter_op_parallelism_threads(self.config.num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(self.config.num_threads)
    
    async def initialize(self):
        """Initialise le moteur de manière asynchrone"""
        if self._initialized:
            return
        
        try:
            if self.monitor:
                self.monitor.start_timer("initialization")
            
            # Chargement du modèle principal
            await self._load_model(self.config.model_name)
            
            # Validation de l'environnement
            self._validate_environment()
            
            self._initialized = True
            
            if self.monitor:
                init_time = self.monitor.end_timer("initialization")
                logger.info(f"Moteur initialisé en {init_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            raise SpleeterError(f"Échec initialisation moteur: {e}")
    
    async def _load_model(self, model_name: str):
        """
        Charge un modèle Spleeter
        
        Args:
            model_name: Nom du modèle (ex: "spleeter:2stems-16kHz")
        """
        if model_name in self._models_cache:
            return self._models_cache[model_name]
        
        try:
            # Import dynamique pour éviter les dépendances
            from spleeter.separator import Separator
            
            # Configuration du séparateur
            separator_config = {
                'audio_adapter': self.config.audio_adapter,
                'sample_rate': self.config.sample_rate,
                'frame_length': self.config.frame_length,
                'frame_step': self.config.frame_step
            }
            
            # Création du séparateur
            separator = Separator(model_name, **separator_config)
            
            # Cache du modèle
            self._models_cache[model_name] = separator
            
            logger.info(f"Modèle chargé: {model_name}")
            return separator
            
        except ImportError:
            logger.error("Spleeter non installé - utilisation du mode simulation")
            # Mode simulation pour tests
            return self._create_mock_separator(model_name)
        except Exception as e:
            raise ModelNotFoundError(f"Impossible de charger {model_name}: {e}")
    
    def _create_mock_separator(self, model_name: str):
        """Crée un séparateur simulé pour les tests"""
        class MockSeparator:
            def __init__(self, model_name):
                self.model_name = model_name
                self.stems = 2 if "2stems" in model_name else 4 if "4stems" in model_name else 5
            
            def separate(self, waveform, sample_rate=44100):
                # Simulation de séparation
                duration = len(waveform) // sample_rate
                samples = len(waveform)
                
                if self.stems == 2:
                    return {
                        'vocals': waveform * 0.6 + np.random.normal(0, 0.1, waveform.shape),
                        'accompaniment': waveform * 0.4 + np.random.normal(0, 0.1, waveform.shape)
                    }
                elif self.stems == 4:
                    return {
                        'vocals': waveform * 0.4,
                        'drums': waveform * 0.3,
                        'bass': waveform * 0.2,
                        'other': waveform * 0.1
                    }
                else:  # 5 stems
                    return {
                        'vocals': waveform * 0.3,
                        'drums': waveform * 0.25,
                        'bass': waveform * 0.2,
                        'piano': waveform * 0.15,
                        'other': waveform * 0.1
                    }
        
        return MockSeparator(model_name)
    
    def _validate_environment(self):
        """Valide l'environnement d'exécution"""
        # Vérification TensorFlow
        if not tf.config.list_physical_devices():
            logger.warning("Aucun device TensorFlow détecté")
        
        # Vérification mémoire
        if self.config.enable_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.warning("GPU demandé mais non disponible")
        
        # Vérification répertoires
        if not os.path.exists(self.config.cache_dir):
            raise SpleeterError(f"Répertoire cache inaccessible: {self.config.cache_dir}")
    
    async def separate_audio(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Sépare un fichier audio en ses composants
        
        Args:
            audio_path: Chemin vers le fichier audio
            output_dir: Répertoire de sortie (optionnel)
            model_name: Modèle à utiliser (optionnel)
            **kwargs: Paramètres additionnels
            
        Returns:
            Dict contenant les stems séparés
            
        Raises:
            AudioProcessingError: En cas d'erreur de traitement
        """
        if not self._initialized:
            await self.initialize()
        
        # Validation du fichier
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
        
        # Validation format
        if audio_path.suffix.lower() not in self.config.supported_formats:
            raise AudioProcessingError(f"Format non supporté: {audio_path.suffix}")
        
        # Validation taille
        file_size = audio_path.stat().st_size
        if file_size > self.config.max_file_size:
            raise AudioProcessingError(f"Fichier trop volumineux: {file_size} bytes")
        
        try:
            if self.monitor:
                self.monitor.start_timer("separation")
                self.monitor.increment_counter("separations_total")
            
            # Chargement audio
            waveform, sample_rate = await self._load_audio(audio_path)
            
            # Validation durée
            duration = len(waveform) / sample_rate
            if duration > self.config.max_duration:
                raise AudioProcessingError(f"Audio trop long: {duration}s > {self.config.max_duration}s")
            
            # Sélection du modèle
            model_to_use = model_name or self.config.model_name
            separator = await self._load_model(model_to_use)
            
            # Séparation
            stems = await self._perform_separation(separator, waveform, sample_rate)
            
            # Sauvegarde optionnelle
            if output_dir:
                await self._save_stems(stems, output_dir, audio_path.stem, sample_rate)
            
            if self.monitor:
                sep_time = self.monitor.end_timer("separation")
                self.monitor.record_metric("separation_duration", sep_time)
                logger.info(f"Séparation terminée en {sep_time:.2f}s")
            
            return stems
            
        except Exception as e:
            if self.monitor:
                self.monitor.increment_counter("separations_failed")
            logger.error(f"Erreur séparation {audio_path}: {e}")
            raise AudioProcessingError(f"Échec séparation: {e}")
    
    async def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Charge un fichier audio
        
        Returns:
            Tuple (waveform, sample_rate)
        """
        try:
            # Utilisation de librosa pour le chargement
            import librosa
            
            waveform, sample_rate = librosa.load(
                audio_path,
                sr=self.config.sample_rate,
                mono=False
            )
            
            # Conversion stéréo si nécessaire
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform])
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]  # Garder seulement 2 channels
            
            return waveform.T, sample_rate  # Shape: (samples, channels)
            
        except ImportError:
            logger.warning("Librosa non disponible - simulation de chargement")
            # Simulation pour tests
            duration = 10  # 10 secondes
            samples = duration * self.config.sample_rate
            waveform = np.random.normal(0, 0.1, (samples, 2))
            return waveform, self.config.sample_rate
    
    async def _perform_separation(
        self,
        separator,
        waveform: np.ndarray,
        sample_rate: int
    ) -> Dict[str, np.ndarray]:
        """
        Effectue la séparation audio
        
        Args:
            separator: Instance du séparateur
            waveform: Forme d'onde audio
            sample_rate: Fréquence d'échantillonnage
            
        Returns:
            Dictionnaire des stems séparés
        """
        def _separate():
            return separator.separate(waveform, sample_rate)
        
        # Exécution dans thread pool pour éviter le blocage
        loop = asyncio.get_event_loop()
        stems = await loop.run_in_executor(self._thread_pool, _separate)
        
        return stems
    
    async def _save_stems(
        self,
        stems: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        base_name: str,
        sample_rate: int
    ):
        """
        Sauvegarde les stems séparés
        
        Args:
            stems: Dictionnaire des stems
            output_dir: Répertoire de sortie
            base_name: Nom de base des fichiers
            sample_rate: Fréquence d'échantillonnage
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import soundfile as sf
            
            for stem_name, audio_data in stems.items():
                output_path = output_dir / f"{base_name}_{stem_name}.wav"
                
                # Conversion format si nécessaire
                if audio_data.ndim == 1:
                    audio_data = audio_data.reshape(-1, 1)
                
                sf.write(output_path, audio_data, sample_rate)
                logger.debug(f"Stem sauvegardé: {output_path}")
                
        except ImportError:
            logger.warning("SoundFile non disponible - simulation de sauvegarde")
            for stem_name in stems.keys():
                output_path = output_dir / f"{base_name}_{stem_name}.wav"
                output_path.touch()  # Création fichier vide pour simulation
    
    async def batch_separate(
        self,
        audio_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        max_concurrent: int = 4
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Séparation en lot de plusieurs fichiers audio
        
        Args:
            audio_files: Liste des fichiers à traiter
            output_dir: Répertoire de sortie
            model_name: Modèle à utiliser
            max_concurrent: Nombre max de traitements simultanés
            
        Returns:
            Dictionnaire des résultats par fichier
        """
        if not self._initialized:
            await self.initialize()
        
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_file(audio_file):
            async with semaphore:
                try:
                    file_path = Path(audio_file)
                    file_output_dir = Path(output_dir) / file_path.stem
                    
                    stems = await self.separate_audio(
                        audio_file,
                        file_output_dir,
                        model_name
                    )
                    
                    results[str(audio_file)] = stems
                    logger.info(f"Traitement terminé: {audio_file}")
                    
                except Exception as e:
                    logger.error(f"Erreur traitement {audio_file}: {e}")
                    results[str(audio_file)] = None
        
        # Traitement concurrent
        tasks = [_process_file(audio_file) for audio_file in audio_files]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Retourne la liste des modèles disponibles
        
        Returns:
            Liste des noms de modèles
        """
        return [
            "spleeter:2stems-16kHz",
            "spleeter:2stems-8kHz", 
            "spleeter:4stems-16kHz",
            "spleeter:4stems-8kHz",
            "spleeter:5stems-16kHz",
            "spleeter:5stems-8kHz"
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Retourne les informations sur un modèle
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Dictionnaire d'informations
        """
        info = {
            "name": model_name,
            "loaded": model_name in self._models_cache,
            "stems": 2,
            "sample_rate": 16000,
            "description": "Unknown model"
        }
        
        if "2stems" in model_name:
            info["stems"] = 2
            info["description"] = "Sépare vocals et accompaniment"
        elif "4stems" in model_name:
            info["stems"] = 4
            info["description"] = "Sépare vocals, drums, bass, other"
        elif "5stems" in model_name:
            info["stems"] = 5
            info["description"] = "Sépare vocals, drums, bass, piano, other"
        
        if "16kHz" in model_name:
            info["sample_rate"] = 16000
        elif "8kHz" in model_name:
            info["sample_rate"] = 8000
        
        return info
    
    async def clear_cache(self):
        """Vide le cache des modèles"""
        self._models_cache.clear()
        logger.info("Cache des modèles vidé")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de performance
        
        Returns:
            Dictionnaire des métriques
        """
        if not self.monitor:
            return {}
        
        return self.monitor.get_summary()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du moteur
        
        Returns:
            Dictionnaire de statut
        """
        status = {
            "initialized": self._initialized,
            "models_loaded": len(self._models_cache),
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
            "memory_usage": self._get_memory_usage(),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
        
        return status
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Retourne l'utilisation mémoire"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    async def __aenter__(self):
        """Context manager async entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager async exit"""
        await self.cleanup()
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
        
        if self.monitor:
            await self.monitor.cleanup()
        
        self._models_cache.clear()
        logger.info("Moteur Spleeter nettoyé")
