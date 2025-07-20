"""
Audio Feature Extractor - Enterprise Edition
==========================================

Extracteur de caractéristiques audio haute performance pour Spotify AI Agent.
Analyse spectrale avancée, métriques temporelles, et descripteurs sémantiques.
"""

import asyncio
import logging
import numpy as np
import librosa
import scipy.signal as sig
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# === Configuration de l'extraction ===
@dataclass
class ExtractionConfig:
    """Configuration pour l'extraction de features."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 13
    n_chroma: int = 12
    spectral_rolloff_percentile: float = 0.85
    zero_crossing_frame_length: int = 2048
    tempo_analysis: bool = True
    harmonic_analysis: bool = True
    onset_detection: bool = True

@dataclass
class AudioFeatures:
    """Structure complète des caractéristiques audio."""
    
    # === Métadonnées ===
    file_path: Optional[str] = None
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    
    # === Features temporelles ===
    rms_energy: Dict[str, float] = None
    zero_crossing_rate: Dict[str, float] = None
    tempo: float = 0.0
    beat_times: List[float] = None
    onset_times: List[float] = None
    
    # === Features spectrales ===
    spectral_centroid: Dict[str, float] = None
    spectral_rolloff: Dict[str, float] = None
    spectral_bandwidth: Dict[str, float] = None
    spectral_contrast: Dict[str, float] = None
    spectral_flatness: Dict[str, float] = None
    
    # === Features harmoniques ===
    chroma: np.ndarray = None
    chroma_stats: Dict[str, float] = None
    tonnetz: np.ndarray = None
    
    # === Features mel et mfcc ===
    mel_spectrogram: np.ndarray = None
    mfcc: np.ndarray = None
    mfcc_delta: np.ndarray = None
    mfcc_delta2: np.ndarray = None
    
    # === Features avancées ===
    pitch_confidence: float = 0.0
    fundamental_frequency: Dict[str, float] = None
    harmonic_ratio: float = 0.0
    noise_ratio: float = 0.0
    
    # === Métriques perceptuelles ===
    loudness_lufs: float = 0.0
    dynamic_range: float = 0.0
    stereo_width: float = 0.0
    
    # === Embeddings et représentations ===
    feature_vector: np.ndarray = None
    reduced_features: np.ndarray = None
    
    def __post_init__(self):
        """Initialise les dictionnaires vides."""
        if self.rms_energy is None:
            self.rms_energy = {}
        if self.zero_crossing_rate is None:
            self.zero_crossing_rate = {}
        if self.spectral_centroid is None:
            self.spectral_centroid = {}
        if self.spectral_rolloff is None:
            self.spectral_rolloff = {}
        if self.spectral_bandwidth is None:
            self.spectral_bandwidth = {}
        if self.spectral_contrast is None:
            self.spectral_contrast = {}
        if self.spectral_flatness is None:
            self.spectral_flatness = {}
        if self.chroma_stats is None:
            self.chroma_stats = {}
        if self.fundamental_frequency is None:
            self.fundamental_frequency = {}
        if self.beat_times is None:
            self.beat_times = []
        if self.onset_times is None:
            self.onset_times = []

# === Extracteur principal ===
class AudioFeatureExtractor:
    """
    Extracteur de caractéristiques audio industriel haute performance.
    """
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        
        # Cache pour optimiser les recalculs
        self._feature_cache = {}
        
    async def extract_features(
        self,
        audio_path: str,
        extract_all: bool = True
    ) -> AudioFeatures:
        """
        Extrait toutes les caractéristiques d'un fichier audio.
        
        Args:
            audio_path: Chemin vers le fichier audio
            extract_all: Si True, extrait toutes les features
            
        Returns:
            AudioFeatures: Structure complète des caractéristiques
        """
        try:
            # Chargement du fichier
            y, sr = librosa.load(
                audio_path, 
                sr=self.config.sample_rate,
                mono=True
            )
            
            # Obtention des métadonnées
            features = AudioFeatures(
                file_path=audio_path,
                duration=len(y) / sr,
                sample_rate=sr,
                channels=1
            )
            
            logger.info(f"Extracting features from: {audio_path}")
            logger.info(f"Duration: {features.duration:.2f}s, SR: {sr}Hz")
            
            if extract_all:
                # Extraction parallèle des différentes catégories
                tasks = [
                    self._extract_temporal_features(y, sr),
                    self._extract_spectral_features(y, sr),
                    self._extract_harmonic_features(y, sr),
                    self._extract_rhythm_features(y, sr),
                    self._extract_perceptual_features(y, sr),
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Fusion des résultats
                for result in results:
                    if result:
                        self._merge_features(features, result)
                
                # Génération du vecteur de features final
                features.feature_vector = await self._create_feature_vector(features)
                features.reduced_features = await self._reduce_dimensionality(features.feature_vector)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {audio_path}: {e}")
            return AudioFeatures()
    
    async def _extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extrait les caractéristiques temporelles."""
        loop = asyncio.get_event_loop()
        
        def _compute_temporal():
            features = {}
            
            # RMS Energy
            rms = librosa.feature.rms(
                y=y, 
                frame_length=self.config.n_fft,
                hop_length=self.config.hop_length
            )[0]
            
            features['rms_energy'] = {
                'mean': float(np.mean(rms)),
                'std': float(np.std(rms)),
                'min': float(np.min(rms)),
                'max': float(np.max(rms)),
                'median': float(np.median(rms)),
                'q25': float(np.percentile(rms, 25)),
                'q75': float(np.percentile(rms, 75))
            }
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(
                y,
                frame_length=self.config.zero_crossing_frame_length,
                hop_length=self.config.hop_length
            )[0]
            
            features['zero_crossing_rate'] = {
                'mean': float(np.mean(zcr)),
                'std': float(np.std(zcr)),
                'min': float(np.min(zcr)),
                'max': float(np.max(zcr))
            }
            
            return features
        
        return await loop.run_in_executor(self.executor, _compute_temporal)
    
    async def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extrait les caractéristiques spectrales."""
        loop = asyncio.get_event_loop()
        
        def _compute_spectral():
            features = {}
            
            # Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=y, sr=sr,
                hop_length=self.config.hop_length
            )[0]
            
            features['spectral_centroid'] = {
                'mean': float(np.mean(spectral_centroids)),
                'std': float(np.std(spectral_centroids)),
                'median': float(np.median(spectral_centroids))
            }
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr,
                roll_percent=self.config.spectral_rolloff_percentile,
                hop_length=self.config.hop_length
            )[0]
            
            features['spectral_rolloff'] = {
                'mean': float(np.mean(spectral_rolloff)),
                'std': float(np.std(spectral_rolloff)),
                'median': float(np.median(spectral_rolloff))
            }
            
            # Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr,
                hop_length=self.config.hop_length
            )[0]
            
            features['spectral_bandwidth'] = {
                'mean': float(np.mean(spectral_bandwidth)),
                'std': float(np.std(spectral_bandwidth))
            }
            
            # Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr,
                hop_length=self.config.hop_length
            )
            
            features['spectral_contrast'] = {
                'mean': float(np.mean(spectral_contrast)),
                'std': float(np.std(spectral_contrast)),
                'bands_mean': [float(np.mean(band)) for band in spectral_contrast]
            }
            
            # Spectral Flatness
            spectral_flatness = librosa.feature.spectral_flatness(
                y=y,
                hop_length=self.config.hop_length
            )[0]
            
            features['spectral_flatness'] = {
                'mean': float(np.mean(spectral_flatness)),
                'std': float(np.std(spectral_flatness))
            }
            
            return features
        
        return await loop.run_in_executor(self.executor, _compute_spectral)
    
    async def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extrait les caractéristiques harmoniques."""
        loop = asyncio.get_event_loop()
        
        def _compute_harmonic():
            features = {}
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(
                y=y, sr=sr,
                n_chroma=self.config.n_chroma,
                hop_length=self.config.hop_length
            )
            
            features['chroma'] = chroma
            features['chroma_stats'] = {
                'mean': float(np.mean(chroma)),
                'std': float(np.std(chroma)),
                'note_profiles': [float(np.mean(chroma[i, :])) for i in range(self.config.n_chroma)]
            }
            
            # Tonnetz (Harmonic Network)
            tonnetz = librosa.feature.tonnetz(
                y=librosa.effects.harmonic(y), sr=sr,
                hop_length=self.config.hop_length
            )
            features['tonnetz'] = tonnetz
            
            # MFCC
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr,
                n_mfcc=self.config.n_mfcc,
                hop_length=self.config.hop_length
            )
            features['mfcc'] = mfcc
            
            # MFCC Delta
            mfcc_delta = librosa.feature.delta(mfcc)
            features['mfcc_delta'] = mfcc_delta
            
            # MFCC Delta-Delta
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features['mfcc_delta2'] = mfcc_delta2
            
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_mels=self.config.n_mels,
                hop_length=self.config.hop_length
            )
            features['mel_spectrogram'] = mel_spec
            
            return features
        
        return await loop.run_in_executor(self.executor, _compute_harmonic)
    
    async def _extract_rhythm_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extrait les caractéristiques rythmiques."""
        loop = asyncio.get_event_loop()
        
        def _compute_rhythm():
            features = {}
            
            if self.config.tempo_analysis:
                # Tempo et beat tracking
                tempo, beats = librosa.beat.beat_track(
                    y=y, sr=sr,
                    hop_length=self.config.hop_length
                )
                
                features['tempo'] = float(tempo)
                features['beat_times'] = librosa.frames_to_time(beats, sr=sr).tolist()
            
            if self.config.onset_detection:
                # Détection d'onset
                onset_frames = librosa.onset.onset_detect(
                    y=y, sr=sr,
                    hop_length=self.config.hop_length
                )
                features['onset_times'] = librosa.frames_to_time(onset_frames, sr=sr).tolist()
            
            return features
        
        return await loop.run_in_executor(self.executor, _compute_rhythm)
    
    async def _extract_perceptual_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extrait les caractéristiques perceptuelles."""
        loop = asyncio.get_event_loop()
        
        def _compute_perceptual():
            features = {}
            
            # Séparation harmonique/percussive
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Ratio harmonique
            harmonic_energy = np.sum(y_harmonic**2)
            percussive_energy = np.sum(y_percussive**2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                features['harmonic_ratio'] = float(harmonic_energy / total_energy)
                features['noise_ratio'] = float(percussive_energy / total_energy)
            
            # Détection de pitch
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr,
                hop_length=self.config.hop_length
            )
            
            # Confiance moyenne de pitch
            pitch_confidence = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0.0
            features['pitch_confidence'] = float(pitch_confidence)
            
            # Fréquence fondamentale dominante
            if np.any(pitches > 0):
                dominant_pitches = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        dominant_pitches.append(pitch)
                
                if dominant_pitches:
                    features['fundamental_frequency'] = {
                        'mean': float(np.mean(dominant_pitches)),
                        'std': float(np.std(dominant_pitches)),
                        'median': float(np.median(dominant_pitches))
                    }
            
            # Dynamic Range (approximation)
            rms = librosa.feature.rms(y=y)[0]
            if len(rms) > 0:
                dynamic_range = 20 * np.log10(np.max(rms) / (np.min(rms[rms > 0]) + 1e-10))
                features['dynamic_range'] = float(dynamic_range)
            
            return features
        
        return await loop.run_in_executor(self.executor, _compute_perceptual)
    
    def _merge_features(self, main_features: AudioFeatures, new_features: Dict[str, Any]):
        """Fusionne les nouvelles features dans la structure principale."""
        for key, value in new_features.items():
            if hasattr(main_features, key):
                setattr(main_features, key, value)
    
    async def _create_feature_vector(self, features: AudioFeatures) -> np.ndarray:
        """Crée un vecteur de features unifié."""
        loop = asyncio.get_event_loop()
        
        def _vectorize():
            vector_parts = []
            
            # Features temporelles
            if features.rms_energy:
                vector_parts.extend([
                    features.rms_energy.get('mean', 0),
                    features.rms_energy.get('std', 0),
                    features.rms_energy.get('median', 0)
                ])
            
            if features.zero_crossing_rate:
                vector_parts.extend([
                    features.zero_crossing_rate.get('mean', 0),
                    features.zero_crossing_rate.get('std', 0)
                ])
            
            # Features spectrales
            if features.spectral_centroid:
                vector_parts.extend([
                    features.spectral_centroid.get('mean', 0),
                    features.spectral_centroid.get('std', 0)
                ])
            
            if features.spectral_rolloff:
                vector_parts.extend([
                    features.spectral_rolloff.get('mean', 0),
                    features.spectral_rolloff.get('std', 0)
                ])
            
            if features.spectral_bandwidth:
                vector_parts.extend([
                    features.spectral_bandwidth.get('mean', 0),
                    features.spectral_bandwidth.get('std', 0)
                ])
            
            # Features harmoniques
            if features.chroma_stats:
                vector_parts.extend(features.chroma_stats.get('note_profiles', [0]*12))
            
            # MFCC moyens
            if features.mfcc is not None:
                mfcc_means = np.mean(features.mfcc, axis=1)
                vector_parts.extend(mfcc_means.tolist())
            
            # Features rythmiques
            vector_parts.append(features.tempo or 0)
            
            # Features perceptuelles
            vector_parts.extend([
                features.harmonic_ratio or 0,
                features.noise_ratio or 0,
                features.pitch_confidence or 0,
                features.dynamic_range or 0
            ])
            
            return np.array(vector_parts, dtype=np.float32)
        
        return await loop.run_in_executor(self.executor, _vectorize)
    
    async def _reduce_dimensionality(self, feature_vector: np.ndarray) -> np.ndarray:
        """Réduit la dimensionalité du vecteur de features."""
        if len(feature_vector) > 50:
            try:
                # Normalisation
                normalized = self.scaler.fit_transform(feature_vector.reshape(1, -1))
                
                # Réduction PCA
                reduced = self.pca.fit_transform(normalized)
                return reduced.flatten()
            except:
                # Fallback: sélection des features les plus importantes
                return feature_vector[:50]
        
        return feature_vector
    
    async def extract_batch(
        self,
        audio_paths: List[str],
        output_format: str = 'dict'
    ) -> Union[List[AudioFeatures], pd.DataFrame, Dict[str, Any]]:
        """
        Extrait les features de plusieurs fichiers en parallèle.
        
        Args:
            audio_paths: Liste des chemins de fichiers
            output_format: Format de sortie ('dict', 'dataframe', 'json')
            
        Returns:
            Résultats selon le format spécifié
        """
        # Limitation de la parallélisation
        semaphore = asyncio.Semaphore(4)
        
        async def extract_single(path: str) -> AudioFeatures:
            async with semaphore:
                return await self.extract_features(path)
        
        # Extraction parallèle
        tasks = [extract_single(path) for path in audio_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrage des erreurs
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract features from {audio_paths[i]}: {result}")
            else:
                valid_results.append(result)
        
        # Conversion selon le format demandé
        if output_format == 'dataframe':
            return self._to_dataframe(valid_results)
        elif output_format == 'json':
            return self._to_json(valid_results)
        else:
            return valid_results
    
    def _to_dataframe(self, features_list: List[AudioFeatures]) -> pd.DataFrame:
        """Convertit les features en DataFrame pandas."""
        data = []
        for features in features_list:
            row = {}
            
            # Métadonnées
            row['file_path'] = features.file_path
            row['duration'] = features.duration
            row['tempo'] = features.tempo
            
            # Features statistiques
            if features.rms_energy:
                for stat, value in features.rms_energy.items():
                    row[f'rms_{stat}'] = value
            
            if features.spectral_centroid:
                for stat, value in features.spectral_centroid.items():
                    row[f'spectral_centroid_{stat}'] = value
            
            # Features moyennes
            if features.mfcc is not None:
                mfcc_means = np.mean(features.mfcc, axis=1)
                for i, value in enumerate(mfcc_means):
                    row[f'mfcc_{i}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _to_json(self, features_list: List[AudioFeatures]) -> Dict[str, Any]:
        """Convertit les features en format JSON."""
        result = {
            'extraction_info': {
                'timestamp': datetime.now().isoformat(),
                'total_files': len(features_list),
                'config': asdict(self.config)
            },
            'features': []
        }
        
        for features in features_list:
            # Conversion en dict avec sérialisation des numpy arrays
            features_dict = asdict(features)
            
            # Nettoyage des numpy arrays
            for key, value in features_dict.items():
                if isinstance(value, np.ndarray):
                    features_dict[key] = value.tolist()
            
            result['features'].append(features_dict)
        
        return result
    
    def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.executor.shutdown(wait=True)
            self._feature_cache.clear()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# === Analyseur de similarité ===
class AudioSimilarityAnalyzer:
    """
    Analyseur de similarité entre features audio.
    """
    
    def __init__(self):
        self.distance_metrics = {
            'euclidean': self._euclidean_distance,
            'cosine': self._cosine_similarity,
            'manhattan': self._manhattan_distance,
            'correlation': self._correlation_distance
        }
    
    def calculate_similarity(
        self,
        features1: AudioFeatures,
        features2: AudioFeatures,
        metric: str = 'cosine'
    ) -> float:
        """Calcule la similarité entre deux sets de features."""
        
        if metric not in self.distance_metrics:
            raise ValueError(f"Unknown metric: {metric}")
        
        if features1.feature_vector is None or features2.feature_vector is None:
            logger.warning("Missing feature vectors for similarity calculation")
            return 0.0
        
        return self.distance_metrics[metric](
            features1.feature_vector,
            features2.feature_vector
        )
    
    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Distance euclidienne (convertie en similarité)."""
        distance = np.linalg.norm(v1 - v2)
        return 1.0 / (1.0 + distance)
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Similarité cosinus."""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms == 0:
            return 0.0
        return dot_product / norms
    
    def _manhattan_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Distance de Manhattan (convertie en similarité)."""
        distance = np.sum(np.abs(v1 - v2))
        return 1.0 / (1.0 + distance)
    
    def _correlation_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Corrélation de Pearson."""
        if len(v1) < 2 or len(v2) < 2:
            return 0.0
        
        correlation, _ = stats.pearsonr(v1, v2)
        return correlation if not np.isnan(correlation) else 0.0

# === Factory functions ===
def create_extractor(
    sample_rate: int = 22050,
    extract_advanced: bool = True
) -> AudioFeatureExtractor:
    """Factory pour créer un extracteur configuré."""
    
    config = ExtractionConfig(
        sample_rate=sample_rate,
        tempo_analysis=extract_advanced,
        harmonic_analysis=extract_advanced,
        onset_detection=extract_advanced
    )
    
    return AudioFeatureExtractor(config)

def create_similarity_analyzer() -> AudioSimilarityAnalyzer:
    """Factory pour créer un analyseur de similarité."""
    return AudioSimilarityAnalyzer()
