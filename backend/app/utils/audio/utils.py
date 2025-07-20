"""
Audio Utilities - Enterprise Toolbox
===================================

Utilitaires audio industriels pour Spotify AI Agent.
Validation, métadonnées, conversion, et outils de développement.
"""

import asyncio
import logging
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import librosa
import soundfile as sf
import mutagen
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
import pyloudnorm as pyln
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# === Types et enums ===
class AudioHealth(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CORRUPTED = "corrupted"

class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    THOROUGH = "thorough"
    FORENSIC = "forensic"

@dataclass
class AudioMetadata:
    """Métadonnées audio complètes."""
    # === Fichier ===
    file_path: str
    file_size: int
    file_format: str
    creation_time: str
    modification_time: str
    checksum: str
    
    # === Audio technique ===
    duration: float
    sample_rate: int
    bit_depth: int
    channels: int
    bitrate: Optional[int]
    codec: Optional[str]
    
    # === Métadonnées tag ===
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    track_number: Optional[int] = None
    
    # === Qualité audio ===
    peak_level: float = 0.0
    rms_level: float = 0.0
    dynamic_range: float = 0.0
    lufs: float = 0.0
    true_peak: float = 0.0
    
    # === Analyse ===
    health_score: float = 0.0
    health_status: AudioHealth = AudioHealth.FAIR
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []

@dataclass
class ValidationResult:
    """Résultat de validation audio."""
    valid: bool
    health_score: float
    health_status: AudioHealth
    issues: List[str]
    warnings: List[str]
    metadata: Optional[AudioMetadata]
    processing_time: float
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []

# === Validateur audio ===
class AudioValidator:
    """
    Validateur audio industriel avec analyse forensique.
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Seuils de qualité
        self.quality_thresholds = {
            'min_duration': 1.0,     # Minimum 1 seconde
            'max_duration': 600.0,   # Maximum 10 minutes
            'min_sample_rate': 8000, # 8kHz minimum
            'max_sample_rate': 192000, # 192kHz maximum
            'min_dynamic_range': 6.0,  # 6dB minimum
            'max_peak_level': -0.1,    # -0.1dBFS maximum
            'min_lufs': -50.0,         # -50 LUFS minimum
            'max_lufs': 0.0,           # 0 LUFS maximum
        }
    
    async def validate_file(
        self,
        file_path: str,
        level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationResult:
        """
        Valide un fichier audio avec niveau de détail configurable.
        
        Args:
            file_path: Chemin vers le fichier audio
            level: Niveau de validation
            
        Returns:
            ValidationResult avec tous les détails
        """
        
        start_time = time.time()
        issues = []
        warnings = []
        
        try:
            # Vérification existence fichier
            path = Path(file_path)
            if not path.exists():
                return ValidationResult(
                    valid=False,
                    health_score=0.0,
                    health_status=AudioHealth.CORRUPTED,
                    issues=['File does not exist'],
                    warnings=[],
                    metadata=None,
                    processing_time=time.time() - start_time
                )
            
            # Extraction des métadonnées
            metadata = await self._extract_metadata(file_path)
            
            # Validation selon le niveau
            if level == ValidationLevel.BASIC:
                validation_score = await self._basic_validation(metadata, issues, warnings)
            elif level == ValidationLevel.STANDARD:
                validation_score = await self._standard_validation(metadata, issues, warnings)
            elif level == ValidationLevel.THOROUGH:
                validation_score = await self._thorough_validation(metadata, issues, warnings)
            else:  # FORENSIC
                validation_score = await self._forensic_validation(metadata, issues, warnings)
            
            # Détermination du statut de santé
            health_status = self._determine_health_status(validation_score)
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                valid=validation_score > 0.5,
                health_score=validation_score,
                health_status=health_status,
                issues=issues,
                warnings=warnings,
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            
            return ValidationResult(
                valid=False,
                health_score=0.0,
                health_status=AudioHealth.CORRUPTED,
                issues=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata=None,
                processing_time=time.time() - start_time
            )
    
    async def _extract_metadata(self, file_path: str) -> AudioMetadata:
        """Extrait les métadonnées complètes du fichier."""
        
        loop = asyncio.get_event_loop()
        
        def _extract():
            path = Path(file_path)
            
            # Informations fichier
            file_stats = path.stat()
            checksum = self._calculate_checksum(file_path)
            
            # Chargement audio pour analyse technique
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # Informations audio de base
            try:
                info = sf.info(file_path)
                sample_rate = info.samplerate
                channels = info.channels
                
                # Estimation de la profondeur de bits
                if info.subtype in ['PCM_16', 'ALAC_16']:
                    bit_depth = 16
                elif info.subtype in ['PCM_24', 'ALAC_24']:
                    bit_depth = 24
                elif info.subtype in ['PCM_32', 'ALAC_32']:
                    bit_depth = 32
                else:
                    bit_depth = 16  # Par défaut
                    
            except:
                sample_rate = sr
                channels = 1 if y.ndim == 1 else y.shape[0]
                bit_depth = 16
            
            # Métadonnées tag
            tag_metadata = self._extract_tag_metadata(file_path)
            
            # Analyse qualité audio
            peak_level = float(20 * np.log10(np.max(np.abs(y)) + 1e-10))
            rms_level = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10))
            
            # Dynamic range estimation
            frame_size = sr // 10  # 100ms frames
            rms_frames = []
            for i in range(0, len(y) - frame_size, frame_size):
                frame = y[i:i + frame_size]
                rms_frames.append(np.sqrt(np.mean(frame**2)))
            
            if rms_frames:
                dynamic_range = float(20 * np.log10(max(rms_frames) / (min(rms_frames) + 1e-10)))
            else:
                dynamic_range = 0.0
            
            # LUFS measurement (approximation)
            try:
                meter = pyln.Meter(sr)
                lufs = float(meter.integrated_loudness(y))
            except:
                lufs = rms_level  # Fallback
            
            return AudioMetadata(
                file_path=file_path,
                file_size=file_stats.st_size,
                file_format=path.suffix.lower(),
                creation_time=str(file_stats.st_ctime),
                modification_time=str(file_stats.st_mtime),
                checksum=checksum,
                duration=duration,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
                bitrate=None,  # Calculé séparément si nécessaire
                codec=None,
                peak_level=peak_level,
                rms_level=rms_level,
                dynamic_range=dynamic_range,
                lufs=lufs,
                true_peak=peak_level,  # Approximation
                **tag_metadata
            )
        
        return await loop.run_in_executor(self.executor, _extract)
    
    def _extract_tag_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extrait les métadonnées tag selon le format."""
        
        try:
            file_obj = mutagen.File(file_path)
            
            if file_obj is None:
                return {}
            
            metadata = {}
            
            # Mapping universel des tags
            if isinstance(file_obj, MP3):
                metadata.update({
                    'title': str(file_obj.get('TIT2', [''])[0]),
                    'artist': str(file_obj.get('TPE1', [''])[0]),
                    'album': str(file_obj.get('TALB', [''])[0]),
                    'genre': str(file_obj.get('TCON', [''])[0]),
                })
                
                # Année
                year_tag = file_obj.get('TDRC', [''])
                if year_tag:
                    try:
                        metadata['year'] = int(str(year_tag[0])[:4])
                    except:
                        metadata['year'] = None
                        
            elif isinstance(file_obj, FLAC):
                metadata.update({
                    'title': file_obj.get('title', [''])[0],
                    'artist': file_obj.get('artist', [''])[0],
                    'album': file_obj.get('album', [''])[0],
                    'genre': file_obj.get('genre', [''])[0],
                })
                
                # Année
                date = file_obj.get('date', [''])
                if date:
                    try:
                        metadata['year'] = int(date[0][:4])
                    except:
                        metadata['year'] = None
                        
            elif isinstance(file_obj, MP4):
                metadata.update({
                    'title': file_obj.get('©nam', [''])[0],
                    'artist': file_obj.get('©ART', [''])[0],
                    'album': file_obj.get('©alb', [''])[0],
                    'genre': file_obj.get('©gen', [''])[0],
                })
                
                # Année
                year = file_obj.get('©day', [''])
                if year:
                    try:
                        metadata['year'] = int(str(year[0])[:4])
                    except:
                        metadata['year'] = None
            
            # Nettoyage des chaînes vides
            for key, value in metadata.items():
                if isinstance(value, str) and not value.strip():
                    metadata[key] = None
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract tag metadata: {e}")
            return {}
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcule le checksum MD5 du fichier."""
        
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    async def _basic_validation(
        self,
        metadata: AudioMetadata,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """Validation de base - vérifications essentielles."""
        
        score = 1.0
        
        # Durée
        if metadata.duration < self.quality_thresholds['min_duration']:
            issues.append(f"Duration too short: {metadata.duration:.2f}s")
            score -= 0.3
        elif metadata.duration > self.quality_thresholds['max_duration']:
            warnings.append(f"Duration very long: {metadata.duration:.2f}s")
            score -= 0.1
        
        # Sample rate
        if metadata.sample_rate < self.quality_thresholds['min_sample_rate']:
            issues.append(f"Sample rate too low: {metadata.sample_rate}Hz")
            score -= 0.4
        elif metadata.sample_rate > self.quality_thresholds['max_sample_rate']:
            warnings.append(f"Sample rate very high: {metadata.sample_rate}Hz")
        
        # Peak level
        if metadata.peak_level > self.quality_thresholds['max_peak_level']:
            issues.append(f"Clipping detected: {metadata.peak_level:.1f}dBFS")
            score -= 0.3
        
        return max(0.0, score)
    
    async def _standard_validation(
        self,
        metadata: AudioMetadata,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """Validation standard - contrôles qualité étendus."""
        
        # Validation de base d'abord
        score = await self._basic_validation(metadata, issues, warnings)
        
        # Dynamic range
        if metadata.dynamic_range < self.quality_thresholds['min_dynamic_range']:
            issues.append(f"Low dynamic range: {metadata.dynamic_range:.1f}dB")
            score -= 0.2
        
        # LUFS
        if metadata.lufs < self.quality_thresholds['min_lufs']:
            warnings.append(f"Very low loudness: {metadata.lufs:.1f} LUFS")
            score -= 0.1
        elif metadata.lufs > self.quality_thresholds['max_lufs']:
            issues.append(f"Loudness too high: {metadata.lufs:.1f} LUFS")
            score -= 0.2
        
        # Vérification métadonnées
        if not metadata.title:
            warnings.append("Missing title metadata")
            score -= 0.05
        
        if not metadata.artist:
            warnings.append("Missing artist metadata")
            score -= 0.05
        
        return max(0.0, score)
    
    async def _thorough_validation(
        self,
        metadata: AudioMetadata,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """Validation approfondie - analyse détaillée."""
        
        # Validation standard d'abord
        score = await self._standard_validation(metadata, issues, warnings)
        
        # Analyse spectrale pour détecter les anomalies
        try:
            # Chargement pour analyse spectrale
            y, sr = librosa.load(metadata.file_path, sr=None, duration=30.0)  # 30s max
            
            # Détection de silence excessif
            silence_ratio = self._detect_silence_ratio(y)
            if silence_ratio > 0.3:  # Plus de 30% de silence
                warnings.append(f"High silence ratio: {silence_ratio*100:.1f}%")
                score -= 0.1
            
            # Détection de distorsion harmonique
            distortion_level = self._detect_harmonic_distortion(y, sr)
            if distortion_level > 0.1:  # Plus de 10% de distorsion
                issues.append(f"High harmonic distortion: {distortion_level*100:.1f}%")
                score -= 0.2
            
        except Exception as e:
            warnings.append(f"Spectral analysis failed: {str(e)}")
            score -= 0.1
        
        return max(0.0, score)
    
    async def _forensic_validation(
        self,
        metadata: AudioMetadata,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """Validation forensique - analyse complète."""
        
        # Validation approfondie d'abord
        score = await self._thorough_validation(metadata, issues, warnings)
        
        # Analyse forensique avancée
        try:
            y, sr = librosa.load(metadata.file_path, sr=None)
            
            # Détection de compression artifacts
            compression_artifacts = self._detect_compression_artifacts(y, sr)
            if compression_artifacts > 0.15:
                issues.append(f"Compression artifacts detected: {compression_artifacts*100:.1f}%")
                score -= 0.15
            
            # Analyse de cohérence temporelle
            temporal_consistency = self._analyze_temporal_consistency(y, sr)
            if temporal_consistency < 0.8:
                warnings.append(f"Temporal inconsistency: {temporal_consistency:.2f}")
                score -= 0.1
            
            # Détection de splicing/édition
            edit_detection = self._detect_audio_edits(y, sr)
            if edit_detection > 0.05:
                warnings.append(f"Potential audio edits detected: {edit_detection:.3f}")
            
        except Exception as e:
            warnings.append(f"Forensic analysis failed: {str(e)}")
            score -= 0.05
        
        return max(0.0, score)
    
    def _detect_silence_ratio(self, y: np.ndarray, threshold: float = 0.01) -> float:
        """Détecte le ratio de silence dans l'audio."""
        
        silent_samples = np.sum(np.abs(y) < threshold)
        return silent_samples / len(y)
    
    def _detect_harmonic_distortion(self, y: np.ndarray, sr: int) -> float:
        """Détecte la distorsion harmonique."""
        
        # Analyse spectrale
        D = librosa.stft(y, hop_length=512)
        magnitude = np.abs(D)
        
        # Recherche de pics harmoniques anormaux
        freqs = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0]*2-1)
        
        # Détection simplifiée de distorsion
        # (En réalité, cela nécessiterait une analyse plus sophistiquée)
        
        # Calcul de la variance spectrale comme proxy
        spectral_variance = np.var(magnitude, axis=1)
        high_variance_ratio = np.sum(spectral_variance > np.percentile(spectral_variance, 95)) / len(spectral_variance)
        
        return min(high_variance_ratio, 1.0)
    
    def _detect_compression_artifacts(self, y: np.ndarray, sr: int) -> float:
        """Détecte les artifacts de compression."""
        
        # Analyse de la réponse en fréquence
        D = librosa.stft(y)
        magnitude = np.abs(D)
        
        # Recherche de coupures fréquentielles typiques de la compression
        freq_bins = magnitude.shape[0]
        high_freq_energy = np.mean(magnitude[int(freq_bins*0.8):, :])
        total_energy = np.mean(magnitude)
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
            # Si le ratio est anormalement bas, cela peut indiquer une compression agressive
            if high_freq_ratio < 0.1:
                return 1.0 - high_freq_ratio * 10  # Conversion en score d'artifact
        
        return 0.0
    
    def _analyze_temporal_consistency(self, y: np.ndarray, sr: int) -> float:
        """Analyse la cohérence temporelle du signal."""
        
        # Division en segments
        segment_length = sr * 2  # 2 secondes par segment
        num_segments = len(y) // segment_length
        
        if num_segments < 2:
            return 1.0
        
        # Calcul des caractéristiques par segment
        segment_rms = []
        for i in range(num_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(y))
            segment = y[start:end]
            segment_rms.append(np.sqrt(np.mean(segment**2)))
        
        # Mesure de cohérence (variance normalisée)
        if len(segment_rms) > 1:
            mean_rms = np.mean(segment_rms)
            if mean_rms > 0:
                consistency = 1.0 - (np.std(segment_rms) / mean_rms)
                return max(0.0, min(1.0, consistency))
        
        return 1.0
    
    def _detect_audio_edits(self, y: np.ndarray, sr: int) -> float:
        """Détecte les éditions/coupures dans l'audio."""
        
        # Analyse des transitions brusques
        diff = np.diff(y)
        
        # Détection de discontinuités
        threshold = np.std(diff) * 5  # Seuil basé sur la variance
        discontinuities = np.sum(np.abs(diff) > threshold)
        
        # Normalisation par la longueur
        edit_ratio = discontinuities / len(diff)
        
        return min(edit_ratio * 100, 1.0)  # Conversion en score normalisé
    
    def _determine_health_status(self, score: float) -> AudioHealth:
        """Détermine le statut de santé basé sur le score."""
        
        if score >= 0.9:
            return AudioHealth.EXCELLENT
        elif score >= 0.7:
            return AudioHealth.GOOD
        elif score >= 0.5:
            return AudioHealth.FAIR
        elif score >= 0.3:
            return AudioHealth.POOR
        else:
            return AudioHealth.CORRUPTED
    
    async def batch_validate(
        self,
        file_paths: List[str],
        level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, ValidationResult]:
        """Valide plusieurs fichiers en parallèle."""
        
        results = {}
        
        # Limitation de la parallélisation
        semaphore = asyncio.Semaphore(3)
        
        async def validate_single(path: str) -> Tuple[str, ValidationResult]:
            async with semaphore:
                result = await self.validate_file(path, level)
                return path, result
        
        # Validation parallèle
        tasks = [validate_single(path) for path in file_paths]
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                logger.error(f"Validation task failed: {task_result}")
            else:
                path, result = task_result
                results[path] = result
        
        return results
    
    def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# === Extracteur de métadonnées ===
class MetadataExtractor:
    """
    Extracteur de métadonnées audio industriel.
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def extract_comprehensive_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extrait toutes les métadonnées possibles d'un fichier."""
        
        validator = AudioValidator()
        
        # Validation complète avec extraction de métadonnées
        validation_result = await validator.validate_file(file_path, ValidationLevel.THOROUGH)
        
        if validation_result.metadata:
            # Conversion en dictionnaire pour sérialisation
            metadata_dict = asdict(validation_result.metadata)
            
            # Ajout des informations de validation
            metadata_dict['validation'] = {
                'health_score': validation_result.health_score,
                'health_status': validation_result.health_status.value,
                'issues': validation_result.issues,
                'warnings': validation_result.warnings
            }
            
            return metadata_dict
        
        return {}
    
    async def extract_technical_info(self, file_path: str) -> Dict[str, Any]:
        """Extrait uniquement les informations techniques."""
        
        loop = asyncio.get_event_loop()
        
        def _extract_tech():
            try:
                # Informations LibROSA
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
                
                # Informations SoundFile
                info = sf.info(file_path)
                
                return {
                    'duration': duration,
                    'sample_rate': sr,
                    'channels': info.channels,
                    'frames': info.frames,
                    'format': info.format,
                    'subtype': info.subtype,
                    'bit_depth': self._estimate_bit_depth(info.subtype),
                    'file_size': Path(file_path).stat().st_size
                }
                
            except Exception as e:
                logger.error(f"Technical extraction failed: {e}")
                return {}
        
        return await loop.run_in_executor(self.executor, _extract_tech)
    
    def _estimate_bit_depth(self, subtype: str) -> int:
        """Estime la profondeur de bits à partir du sous-type."""
        
        if '16' in subtype:
            return 16
        elif '24' in subtype:
            return 24
        elif '32' in subtype:
            return 32
        elif 'FLOAT' in subtype:
            return 32
        elif 'DOUBLE' in subtype:
            return 64
        else:
            return 16  # Défaut
    
    def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# === Utilitaires généraux ===
class AudioUtils:
    """
    Utilitaires audio généraux.
    """
    
    @staticmethod
    def convert_duration(seconds: float) -> str:
        """Convertit une durée en format HH:MM:SS."""
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def db_to_linear(db: float) -> float:
        """Convertit dB en amplitude linéaire."""
        return 10 ** (db / 20)
    
    @staticmethod
    def linear_to_db(linear: float) -> float:
        """Convertit amplitude linéaire en dB."""
        return 20 * np.log10(max(linear, 1e-10))
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalise un chemin de fichier."""
        return str(Path(path).resolve())
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Retourne les formats audio supportés."""
        return ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.aiff']
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Vérifie si le format est supporté."""
        ext = Path(file_path).suffix.lower()
        return ext in AudioUtils.get_supported_formats()
    
    @staticmethod
    def estimate_processing_time(
        file_size_mb: float,
        operation: str = 'basic'
    ) -> float:
        """Estime le temps de traitement en secondes."""
        
        # Facteurs basés sur des benchmarks
        factors = {
            'basic': 0.1,      # Validation de base
            'standard': 0.3,   # Traitement standard
            'thorough': 0.8,   # Analyse approfondie
            'forensic': 2.0,   # Analyse forensique
            'ml_extraction': 1.5,  # Extraction features ML
            'classification': 0.5,  # Classification
            'effects': 1.0     # Application d'effets
        }
        
        factor = factors.get(operation, 0.5)
        return file_size_mb * factor

# === Factory functions ===
def create_validator() -> AudioValidator:
    """Factory pour créer un validateur audio."""
    return AudioValidator()

def create_metadata_extractor() -> MetadataExtractor:
    """Factory pour créer un extracteur de métadonnées."""
    return MetadataExtractor()

def get_audio_utils() -> AudioUtils:
    """Factory pour obtenir les utilitaires audio."""
    return AudioUtils()
