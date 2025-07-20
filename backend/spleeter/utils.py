"""
🎵 Spotify AI Agent - Spleeter Utils
===================================

Utilitaires avancés pour le traitement audio, validation
et optimisations diverses du module Spleeter.

🎖️ Développé par l'équipe d'experts enterprise
"""

import os
import re
import asyncio
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
import time
import hashlib
import mimetypes
from dataclasses import dataclass
import json
import tempfile
import shutil

from .exceptions import AudioProcessingError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Métadonnées audio complètes"""
    filename: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    format: str
    codec: str
    file_size: int
    bitrate: Optional[int] = None
    
    # Métadonnées musicales
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    
    # Analyse audio
    rms_level: Optional[float] = None
    peak_level: Optional[float] = None
    dynamic_range: Optional[float] = None
    spectral_centroid: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


class AudioUtils:
    """
    Utilitaires pour le traitement et l'analyse audio
    
    Features:
    - Analyse de métadonnées
    - Validation de fichiers
    - Conversion de formats
    - Optimisations audio
    - Détection de caractéristiques
    """
    
    # Formats supportés
    SUPPORTED_FORMATS = {
        '.wav': {'codec': 'pcm', 'lossless': True},
        '.flac': {'codec': 'flac', 'lossless': True},
        '.mp3': {'codec': 'mp3', 'lossless': False},
        '.ogg': {'codec': 'vorbis', 'lossless': False},
        '.m4a': {'codec': 'aac', 'lossless': False},
        '.aac': {'codec': 'aac', 'lossless': False},
        '.wma': {'codec': 'wma', 'lossless': False}
    }
    
    # Échantillonnages standards
    STANDARD_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000, 96000, 192000]
    
    @classmethod
    def is_audio_file(cls, file_path: Union[str, Path]) -> bool:
        """
        Vérifie si un fichier est un fichier audio supporté
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            True si le fichier est audio
        """
        file_path = Path(file_path)
        
        # Vérification par extension
        if file_path.suffix.lower() in cls.SUPPORTED_FORMATS:
            return True
        
        # Vérification par MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith('audio/'):
            return True
        
        return False
    
    @classmethod
    def get_format_info(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Retourne les informations de format d'un fichier
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            Dictionnaire d'informations
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension in cls.SUPPORTED_FORMATS:
            return cls.SUPPORTED_FORMATS[extension].copy()
        
        return {'codec': 'unknown', 'lossless': False}
    
    @classmethod
    async def get_audio_metadata(cls, file_path: Union[str, Path]) -> AudioMetadata:
        """
        Extrait les métadonnées complètes d'un fichier audio
        
        Args:
            file_path: Chemin vers le fichier audio
            
        Returns:
            Métadonnées audio
            
        Raises:
            AudioProcessingError: En cas d'erreur
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        if not cls.is_audio_file(file_path):
            raise AudioProcessingError(f"Format non supporté: {file_path.suffix}")
        
        def _extract_metadata():
            try:
                # Tentative avec mutagen pour les métadonnées musicales
                metadata = cls._extract_with_mutagen(file_path)
                if metadata:
                    return metadata
            except ImportError:
                pass
            
            try:
                # Tentative avec librosa/soundfile pour l'audio
                metadata = cls._extract_with_librosa(file_path)
                if metadata:
                    return metadata
            except ImportError:
                pass
            
            # Fallback avec informations basiques
            return cls._extract_basic_metadata(file_path)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_metadata)
    
    @classmethod
    def _extract_with_mutagen(cls, file_path: Path) -> Optional[AudioMetadata]:
        """Extraction avec mutagen (métadonnées musicales)"""
        try:
            from mutagen import File
            from mutagen.id3 import ID3NoHeaderError
            
            audio_file = File(file_path)
            if not audio_file:
                return None
            
            # Informations de base
            info = audio_file.info
            file_info = cls.get_format_info(file_path)
            
            metadata = AudioMetadata(
                filename=file_path.name,
                duration=getattr(info, 'length', 0.0),
                sample_rate=getattr(info, 'sample_rate', 44100),
                channels=getattr(info, 'channels', 2),
                bit_depth=getattr(info, 'bits_per_sample', 16),
                format=file_path.suffix.lower()[1:],
                codec=file_info.get('codec', 'unknown'),
                file_size=file_path.stat().st_size,
                bitrate=getattr(info, 'bitrate', None)
            )
            
            # Métadonnées musicales
            if audio_file.tags:
                tags = audio_file.tags
                
                # Titre
                if 'TIT2' in tags:  # ID3v2
                    metadata.title = str(tags['TIT2'][0])
                elif 'TITLE' in tags:  # Vorbis
                    metadata.title = str(tags['TITLE'][0])
                
                # Artiste
                if 'TPE1' in tags:  # ID3v2
                    metadata.artist = str(tags['TPE1'][0])
                elif 'ARTIST' in tags:  # Vorbis
                    metadata.artist = str(tags['ARTIST'][0])
                
                # Album
                if 'TALB' in tags:  # ID3v2
                    metadata.album = str(tags['TALB'][0])
                elif 'ALBUM' in tags:  # Vorbis
                    metadata.album = str(tags['ALBUM'][0])
                
                # Genre
                if 'TCON' in tags:  # ID3v2
                    metadata.genre = str(tags['TCON'][0])
                elif 'GENRE' in tags:  # Vorbis
                    metadata.genre = str(tags['GENRE'][0])
                
                # Année
                if 'TDRC' in tags:  # ID3v2
                    try:
                        metadata.year = int(str(tags['TDRC'][0])[:4])
                    except (ValueError, IndexError):
                        pass
                elif 'DATE' in tags:  # Vorbis
                    try:
                        metadata.year = int(str(tags['DATE'][0])[:4])
                    except (ValueError, IndexError):
                        pass
            
            return metadata
            
        except Exception as e:
            logger.debug(f"Erreur extraction mutagen: {e}")
            return None
    
    @classmethod
    def _extract_with_librosa(cls, file_path: Path) -> Optional[AudioMetadata]:
        """Extraction avec librosa (analyse audio)"""
        try:
            import librosa
            import soundfile as sf
            
            # Informations du fichier
            info = sf.info(file_path)
            file_info = cls.get_format_info(file_path)
            
            metadata = AudioMetadata(
                filename=file_path.name,
                duration=info.duration,
                sample_rate=info.samplerate,
                channels=info.channels,
                bit_depth=16,  # Par défaut
                format=file_path.suffix.lower()[1:],
                codec=file_info.get('codec', 'unknown'),
                file_size=file_path.stat().st_size
            )
            
            # Analyse audio basique
            try:
                # Chargement d'un échantillon pour l'analyse
                y, sr = librosa.load(file_path, duration=30.0)  # 30 secondes max
                
                # RMS et peak
                rms = librosa.feature.rms(y=y)[0]
                metadata.rms_level = float(np.mean(rms))
                metadata.peak_level = float(np.max(np.abs(y)))
                
                # Centroïde spectral
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                metadata.spectral_centroid = float(np.mean(spectral_centroids))
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                metadata.zero_crossing_rate = float(np.mean(zcr))
                
                # Dynamic range approximatif
                if metadata.peak_level > 0 and metadata.rms_level > 0:
                    metadata.dynamic_range = float(
                        20 * np.log10(metadata.peak_level / metadata.rms_level)
                    )
                
            except Exception as e:
                logger.debug(f"Erreur analyse audio: {e}")
            
            return metadata
            
        except Exception as e:
            logger.debug(f"Erreur extraction librosa: {e}")
            return None
    
    @classmethod
    def _extract_basic_metadata(cls, file_path: Path) -> AudioMetadata:
        """Extraction basique (fallback)"""
        file_info = cls.get_format_info(file_path)
        file_size = file_path.stat().st_size
        
        # Estimation de durée basée sur la taille (très approximatif)
        estimated_duration = file_size / (44100 * 2 * 2)  # 44.1kHz stéréo 16-bit
        
        return AudioMetadata(
            filename=file_path.name,
            duration=estimated_duration,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
            format=file_path.suffix.lower()[1:],
            codec=file_info.get('codec', 'unknown'),
            file_size=file_size
        )
    
    @classmethod
    def calculate_audio_hash(cls, file_path: Union[str, Path]) -> str:
        """
        Calcule un hash audio basé sur le contenu
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            Hash hexadécimal
        """
        def _calculate():
            try:
                import librosa
                
                # Chargement d'un échantillon
                y, sr = librosa.load(file_path, duration=10.0, offset=30.0)
                
                # Features pour le hash
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                
                # Concaténation des features
                features = np.concatenate([
                    mfcc.flatten(),
                    spectral_centroid.flatten()
                ])
                
                # Hash des features
                features_bytes = features.tobytes()
                return hashlib.md5(features_bytes).hexdigest()
                
            except Exception:
                # Fallback: hash du fichier complet
                hasher = hashlib.md5()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()
        
        return _calculate()
    
    @classmethod
    def detect_silence(
        cls,
        audio_data: np.ndarray,
        sample_rate: int,
        threshold_db: float = -40.0,
        min_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Détecte les segments de silence dans l'audio
        
        Args:
            audio_data: Données audio
            sample_rate: Fréquence d'échantillonnage
            threshold_db: Seuil de silence en dB
            min_duration: Durée minimale du silence
            
        Returns:
            Liste des segments (start, end) en secondes
        """
        # Conversion du seuil en linéaire
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Calcul de l'énergie par fenêtre
        hop_length = 512
        frame_length = 1024
        
        if audio_data.ndim > 1:
            # Moyenne des canaux
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data
        
        # RMS par fenêtre
        rms = []
        for i in range(0, len(audio_mono) - frame_length, hop_length):
            frame = audio_mono[i:i + frame_length]
            rms_value = np.sqrt(np.mean(frame ** 2))
            rms.append(rms_value)
        
        rms = np.array(rms)
        
        # Détection des segments sous le seuil
        silence_mask = rms < threshold_linear
        
        # Conversion en segments temporels
        silence_segments = []
        in_silence = False
        start_time = 0
        
        time_per_frame = hop_length / sample_rate
        
        for i, is_silent in enumerate(silence_mask):
            current_time = i * time_per_frame
            
            if is_silent and not in_silence:
                # Début du silence
                start_time = current_time
                in_silence = True
            elif not is_silent and in_silence:
                # Fin du silence
                duration = current_time - start_time
                if duration >= min_duration:
                    silence_segments.append((start_time, current_time))
                in_silence = False
        
        # Dernier segment si se termine en silence
        if in_silence:
            final_time = len(silence_mask) * time_per_frame
            duration = final_time - start_time
            if duration >= min_duration:
                silence_segments.append((start_time, final_time))
        
        return silence_segments
    
    @classmethod
    def trim_silence(
        cls,
        audio_data: np.ndarray,
        sample_rate: int,
        threshold_db: float = -40.0
    ) -> np.ndarray:
        """
        Supprime le silence au début et à la fin
        
        Args:
            audio_data: Données audio
            sample_rate: Fréquence d'échantillonnage
            threshold_db: Seuil de silence
            
        Returns:
            Audio sans silence aux extrémités
        """
        threshold_linear = 10 ** (threshold_db / 20)
        
        if audio_data.ndim > 1:
            # Énergie moyenne des canaux
            energy = np.mean(np.abs(audio_data), axis=1)
        else:
            energy = np.abs(audio_data)
        
        # Trouver le début et la fin du contenu
        above_threshold = energy > threshold_linear
        
        if not np.any(above_threshold):
            # Tout est silence
            return audio_data[:sample_rate]  # Garder 1 seconde
        
        start_idx = np.argmax(above_threshold)
        end_idx = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1
        
        return audio_data[start_idx:end_idx + 1]
    
    @classmethod
    def normalize_loudness(
        cls,
        audio_data: np.ndarray,
        target_lufs: float = -23.0
    ) -> np.ndarray:
        """
        Normalise la sonie selon le standard LUFS
        
        Args:
            audio_data: Données audio
            target_lufs: Cible LUFS
            
        Returns:
            Audio normalisé
        """
        try:
            import pyloudnorm as pyln
            
            # Meter pour mesurer la sonie
            meter = pyln.Meter(44100)  # Fréquence de référence
            
            # Mesure de la sonie actuelle
            current_lufs = meter.integrated_loudness(audio_data)
            
            # Calcul du gain nécessaire
            gain_db = target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            
            # Application du gain
            normalized_audio = audio_data * gain_linear
            
            # Limitation pour éviter le clipping
            peak = np.max(np.abs(normalized_audio))
            if peak > 0.99:
                normalized_audio = normalized_audio * (0.99 / peak)
            
            return normalized_audio
            
        except ImportError:
            # Fallback: normalisation peak simple
            logger.warning("pyloudnorm non disponible - normalisation peak utilisée")
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                return audio_data * (0.7 / peak)  # -3dB environ
            return audio_data


class ValidationUtils:
    """
    Utilitaires de validation pour les fichiers et paramètres audio
    
    Features:
    - Validation de fichiers audio
    - Vérification de paramètres
    - Contrôles de cohérence
    - Validation de sécurité
    """
    
    # Limites de sécurité
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_DURATION = 3600  # 1 heure
    MAX_SAMPLE_RATE = 192000
    MIN_SAMPLE_RATE = 8000
    
    @classmethod
    def validate_audio_file(cls, file_path: Union[str, Path]) -> bool:
        """
        Valide un fichier audio
        
        Args:
            file_path: Chemin vers le fichier
            
        Returns:
            True si valide
            
        Raises:
            ValidationError: En cas de validation échouée
        """
        file_path = Path(file_path)
        
        # Existence
        if not file_path.exists():
            raise ValidationError(f"Fichier non trouvé: {file_path}")
        
        # Permissions de lecture
        if not os.access(file_path, os.R_OK):
            raise ValidationError(f"Fichier non lisible: {file_path}")
        
        # Taille du fichier
        file_size = file_path.stat().st_size
        if file_size > cls.MAX_FILE_SIZE:
            raise ValidationError(f"Fichier trop volumineux: {file_size} > {cls.MAX_FILE_SIZE}")
        
        if file_size == 0:
            raise ValidationError(f"Fichier vide: {file_path}")
        
        # Format supporté
        if not AudioUtils.is_audio_file(file_path):
            raise ValidationError(f"Format audio non supporté: {file_path.suffix}")
        
        # Validation du contenu (basique)
        try:
            cls._validate_audio_content(file_path)
        except Exception as e:
            raise ValidationError(f"Contenu audio invalide: {e}")
        
        return True
    
    @classmethod
    def _validate_audio_content(cls, file_path: Path):
        """Validation basique du contenu audio"""
        try:
            import soundfile as sf
            
            # Test de lecture
            info = sf.info(file_path)
            
            # Vérification des paramètres
            if info.samplerate < cls.MIN_SAMPLE_RATE or info.samplerate > cls.MAX_SAMPLE_RATE:
                raise ValidationError(f"Sample rate invalide: {info.samplerate}")
            
            if info.duration > cls.MAX_DURATION:
                raise ValidationError(f"Durée trop longue: {info.duration}s > {cls.MAX_DURATION}s")
            
            if info.channels < 1 or info.channels > 8:
                raise ValidationError(f"Nombre de canaux invalide: {info.channels}")
            
        except ImportError:
            # Validation minimale sans soundfile
            logger.warning("soundfile non disponible - validation limitée")
            
            # Vérification MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and not mime_type.startswith('audio/'):
                raise ValidationError(f"Type MIME invalide: {mime_type}")
    
    @classmethod
    def validate_sample_rate(cls, sample_rate: int) -> bool:
        """
        Valide une fréquence d'échantillonnage
        
        Args:
            sample_rate: Fréquence à valider
            
        Returns:
            True si valide
        """
        if not isinstance(sample_rate, int):
            raise ValidationError(f"Sample rate doit être un entier: {type(sample_rate)}")
        
        if sample_rate < cls.MIN_SAMPLE_RATE:
            raise ValidationError(f"Sample rate trop bas: {sample_rate} < {cls.MIN_SAMPLE_RATE}")
        
        if sample_rate > cls.MAX_SAMPLE_RATE:
            raise ValidationError(f"Sample rate trop élevé: {sample_rate} > {cls.MAX_SAMPLE_RATE}")
        
        # Recommandation pour les valeurs standards
        if sample_rate not in AudioUtils.STANDARD_SAMPLE_RATES:
            logger.warning(f"Sample rate non standard: {sample_rate}")
        
        return True
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """
        Valide un nom de modèle Spleeter
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            True si valide
        """
        if not isinstance(model_name, str):
            raise ValidationError(f"Nom de modèle doit être une chaîne: {type(model_name)}")
        
        if not model_name.strip():
            raise ValidationError("Nom de modèle vide")
        
        # Validation du format
        # Format attendu: "prefix:Nstems-SRkHz" ou "prefix:model-name"
        pattern = r'^[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+$'
        if not re.match(pattern, model_name):
            logger.warning(f"Format de nom de modèle non standard: {model_name}")
        
        # Caractères dangereux
        dangerous_chars = ['/', '\\', '..', '<', '>', '|', '*', '?', '"']
        for char in dangerous_chars:
            if char in model_name:
                raise ValidationError(f"Caractère dangereux dans le nom: '{char}'")
        
        return True
    
    @classmethod
    def validate_output_directory(cls, output_dir: Union[str, Path]) -> bool:
        """
        Valide un répertoire de sortie
        
        Args:
            output_dir: Répertoire à valider
            
        Returns:
            True si valide
        """
        output_dir = Path(output_dir)
        
        # Vérification des permissions si le répertoire existe
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValidationError(f"Le chemin n'est pas un répertoire: {output_dir}")
            
            if not os.access(output_dir, os.W_OK):
                raise ValidationError(f"Répertoire non accessible en écriture: {output_dir}")
        else:
            # Vérifier que le parent est accessible
            parent = output_dir.parent
            if not parent.exists():
                raise ValidationError(f"Répertoire parent non trouvé: {parent}")
            
            if not os.access(parent, os.W_OK):
                raise ValidationError(f"Répertoire parent non accessible en écriture: {parent}")
        
        # Vérification de sécurité pour les chemins
        try:
            resolved_path = output_dir.resolve()
            # Empêcher les path traversal
            if '..' in str(resolved_path):
                logger.warning(f"Chemin suspect détecté: {output_dir}")
        except Exception as e:
            raise ValidationError(f"Chemin invalide: {e}")
        
        return True
    
    @classmethod
    def validate_batch_size(cls, batch_size: int) -> bool:
        """
        Valide une taille de batch
        
        Args:
            batch_size: Taille à valider
            
        Returns:
            True si valide
        """
        if not isinstance(batch_size, int):
            raise ValidationError(f"Batch size doit être un entier: {type(batch_size)}")
        
        if batch_size < 1:
            raise ValidationError(f"Batch size doit être positif: {batch_size}")
        
        if batch_size > 100:
            logger.warning(f"Batch size très élevé: {batch_size}")
        
        return True
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Nettoie un nom de fichier pour la sécurité
        
        Args:
            filename: Nom à nettoyer
            
        Returns:
            Nom nettoyé
        """
        # Caractères dangereux à remplacer
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        
        sanitized = filename
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Supprimer les espaces multiples
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Supprimer les points au début/fin
        sanitized = sanitized.strip('. ')
        
        # Limiter la longueur
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            max_name_length = 255 - len(ext)
            sanitized = name[:max_name_length] + ext
        
        # Éviter les noms réservés Windows
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        name_without_ext = os.path.splitext(sanitized)[0].upper()
        if name_without_ext in reserved_names:
            sanitized = f"file_{sanitized}"
        
        return sanitized
    
    @classmethod
    def validate_disk_space(cls, required_bytes: int, target_dir: Union[str, Path]) -> bool:
        """
        Vérifie l'espace disque disponible
        
        Args:
            required_bytes: Espace requis
            target_dir: Répertoire cible
            
        Returns:
            True si assez d'espace
        """
        target_dir = Path(target_dir)
        
        try:
            # Utiliser shutil.disk_usage pour la compatibilité
            usage = shutil.disk_usage(target_dir if target_dir.exists() else target_dir.parent)
            free_bytes = usage.free
            
            if free_bytes < required_bytes:
                raise ValidationError(
                    f"Espace disque insuffisant: {free_bytes} < {required_bytes} "
                    f"({free_bytes / 1024 / 1024:.1f}MB < {required_bytes / 1024 / 1024:.1f}MB)"
                )
            
            # Avertissement si moins de 10% d'espace libre après opération
            remaining_after = free_bytes - required_bytes
            if remaining_after < (usage.total * 0.1):
                logger.warning(f"Peu d'espace restera après opération: {remaining_after / 1024 / 1024:.1f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification espace disque: {e}")
            raise ValidationError(f"Impossible de vérifier l'espace disque: {e}")


class PerformanceOptimizer:
    """
    Optimisateur de performance pour les opérations Spleeter
    
    Features:
    - Détection configuration optimale
    - Recommandations hardware
    - Optimisations automatiques
    - Profiling de performance
    """
    
    @classmethod
    def detect_optimal_config(cls) -> Dict[str, Any]:
        """
        Détecte la configuration optimale pour la machine actuelle
        
        Returns:
            Configuration recommandée
        """
        config = {
            'cpu_count': os.cpu_count(),
            'gpu_available': cls._detect_gpu(),
            'memory_gb': cls._get_memory_info(),
            'recommended_batch_size': 4,
            'recommended_workers': 4,
            'enable_gpu': False
        }
        
        # Recommandations basées sur les ressources
        if config['cpu_count'] >= 8:
            config['recommended_workers'] = min(config['cpu_count'] // 2, 8)
        
        if config['memory_gb'] >= 16:
            config['recommended_batch_size'] = 8
        elif config['memory_gb'] >= 8:
            config['recommended_batch_size'] = 4
        else:
            config['recommended_batch_size'] = 2
        
        if config['gpu_available']:
            config['enable_gpu'] = True
            config['recommended_batch_size'] *= 2  # GPU permet plus de parallélisme
        
        return config
    
    @classmethod
    def _detect_gpu(cls) -> bool:
        """Détecte la disponibilité GPU"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            return len(gpus) > 0
        except ImportError:
            return False
        except Exception:
            return False
    
    @classmethod
    def _get_memory_info(cls) -> float:
        """Retourne la mémoire disponible en GB"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.total / (1024 ** 3)  # Conversion en GB
        except ImportError:
            # Estimation basique
            return 8.0  # Valeur par défaut conservative
    
    @classmethod
    def estimate_processing_time(
        cls,
        audio_duration: float,
        model_complexity: str = "2stems",
        use_gpu: bool = False
    ) -> float:
        """
        Estime le temps de traitement
        
        Args:
            audio_duration: Durée audio en secondes
            model_complexity: Complexité du modèle
            use_gpu: Utilisation GPU
            
        Returns:
            Temps estimé en secondes
        """
        # Facteurs de base (temps de traitement / durée audio)
        base_factors = {
            "2stems": 2.0,
            "4stems": 3.5,
            "5stems": 4.0
        }
        
        base_factor = base_factors.get(model_complexity, 3.0)
        
        # Ajustement pour GPU
        if use_gpu:
            base_factor *= 0.4  # GPU ~2.5x plus rapide
        
        # Ajustement pour la durée (overhead fixe)
        overhead = 10.0  # 10 secondes d'overhead
        processing_time = (audio_duration * base_factor) + overhead
        
        return processing_time
    
    @classmethod
    def get_memory_requirements(
        cls,
        audio_duration: float,
        sample_rate: int = 44100,
        model_complexity: str = "2stems"
    ) -> Dict[str, float]:
        """
        Calcule les besoins mémoire estimés
        
        Args:
            audio_duration: Durée audio
            sample_rate: Fréquence d'échantillonnage
            model_complexity: Complexité du modèle
            
        Returns:
            Besoins mémoire en MB
        """
        # Mémoire pour l'audio brut (stéréo, float32)
        audio_samples = audio_duration * sample_rate * 2  # stéréo
        audio_memory_mb = (audio_samples * 4) / (1024 ** 2)  # float32 = 4 bytes
        
        # Mémoire pour le modèle
        model_memory = {
            "2stems": 200,  # MB
            "4stems": 350,
            "5stems": 450
        }
        
        model_mb = model_memory.get(model_complexity, 300)
        
        # Mémoire de traitement (spectrogrammes, etc.)
        processing_factor = {
            "2stems": 4,
            "4stems": 6,
            "5stems": 8
        }
        
        processing_mb = audio_memory_mb * processing_factor.get(model_complexity, 5)
        
        # Mémoire pour les résultats
        output_stems = int(model_complexity[0])
        output_mb = audio_memory_mb * output_stems
        
        total_mb = model_mb + processing_mb + output_mb + 100  # 100MB de marge
        
        return {
            "audio_mb": audio_memory_mb,
            "model_mb": model_mb,
            "processing_mb": processing_mb,
            "output_mb": output_mb,
            "total_mb": total_mb,
            "recommended_system_mb": total_mb * 1.5  # Marge de sécurité
        }
