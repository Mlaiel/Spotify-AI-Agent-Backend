"""
Audio Analysis Engine
====================

Moteur d'analyse audio avancé avec détection de caractéristiques musicales.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import librosa
import librosa.display
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    Analyseur audio principal avec extraction de caractéristiques avancées.
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = hop_length * 4
        
    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyse complète d'un fichier audio.
        
        Returns:
            Dict avec toutes les caractéristiques extraites
        """
        try:
            # Chargement audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Analyses en parallèle
            tasks = [
                self._extract_basic_features(y, sr),
                self._extract_spectral_features(y, sr),
                self._extract_rhythmic_features(y, sr),
                self._extract_harmonic_features(y, sr),
                self._extract_timbral_features(y, sr)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Fusion des résultats
            analysis = {}
            for result in results:
                analysis.update(result)
            
            # Métadonnées générales
            analysis.update({
                'file_path': file_path,
                'duration': len(y) / sr,
                'sample_rate': sr,
                'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024)
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Audio analysis failed for {file_path}: {e}")
            return {'error': str(e)}
    
    async def _extract_basic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extraction des caractéristiques de base."""
        features = {}
        
        # Caractéristiques temporelles
        features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features['signal_energy'] = float(np.sum(y**2))
        
        # Statistiques d'amplitude
        features['amplitude_mean'] = float(np.mean(np.abs(y)))
        features['amplitude_std'] = float(np.std(np.abs(y)))
        features['amplitude_max'] = float(np.max(np.abs(y)))
        features['amplitude_min'] = float(np.min(np.abs(y)))
        
        # Dynamic range
        features['dynamic_range'] = features['amplitude_max'] - features['amplitude_min']
        
        return features
    
    async def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extraction des caractéristiques spectrales."""
        features = {}
        
        # STFT
        stft = librosa.stft(y, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Centroïde spectral
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # Largeur de bande spectrale
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # Rolloff spectral
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # Contraste spectral
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
        features['spectral_contrast_std'] = float(np.std(spectral_contrast))
        
        # Flatness spectral
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
        features['spectral_flatness_std'] = float(np.std(spectral_flatness))
        
        return features
    
    async def _extract_rhythmic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extraction des caractéristiques rythmiques."""
        features = {}
        
        # Détection de tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        features['beat_count'] = len(beats)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        features['onset_count'] = len(onset_frames)
        features['onset_rate'] = len(onset_times) / (len(y) / sr)
        
        # Régularité du rythme
        if len(onset_times) > 1:
            onset_intervals = np.diff(onset_times)
            features['rhythm_regularity'] = float(1.0 / (np.std(onset_intervals) + 1e-7))
        else:
            features['rhythm_regularity'] = 0.0
        
        # Strength du rythme
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        features['rhythm_strength'] = float(np.mean(tempogram))
        
        return features
    
    async def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extraction des caractéristiques harmoniques."""
        features = {}
        
        # Séparation harmonique/percussive
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Ratio harmonique/percussive
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        total_energy = harmonic_energy + percussive_energy
        
        features['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-7))
        features['percussive_ratio'] = float(percussive_energy / (total_energy + 1e-7))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        features['chroma_energy'] = float(np.sum(chroma))
        
        # Détection de tonalité
        chroma_mean = np.mean(chroma, axis=1)
        dominant_pitch_class = np.argmax(chroma_mean)
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        features['dominant_pitch_class'] = pitch_names[dominant_pitch_class]
        features['tonal_clarity'] = float(np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-7))
        
        return features
    
    async def _extract_timbral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extraction des caractéristiques timbrales."""
        features = {}
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        features['mel_energy'] = float(np.sum(mel_spec))
        features['mel_mean'] = float(np.mean(mel_spec))
        features['mel_std'] = float(np.std(mel_spec))
        
        # Tonnetz (réseau tonal)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        features['tonnetz_mean'] = float(np.mean(tonnetz))
        features['tonnetz_std'] = float(np.std(tonnetz))
        
        return features

class SpectralAnalyzer:
    """
    Analyseur spectral spécialisé avec visualisations.
    """
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    async def create_spectral_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Crée une analyse spectrale complète avec visualisations."""
        
        # Calcul du spectrogramme
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Spectrogramme en dB
        db_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Mel spectrogramme
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        db_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Analyse des pics spectraux
        peaks_analysis = await self._analyze_spectral_peaks(magnitude, sr)
        
        # Stabilité spectrale
        stability = await self._calculate_spectral_stability(magnitude)
        
        return {
            'spectogram_shape': magnitude.shape,
            'frequency_resolution': sr / self.n_fft,
            'time_resolution': self.hop_length / sr,
            'mel_bands': mel_spec.shape[0],
            'spectral_peaks': peaks_analysis,
            'spectral_stability': stability,
            'db_range': {
                'min': float(np.min(db_magnitude)),
                'max': float(np.max(db_magnitude)),
                'mean': float(np.mean(db_magnitude))
            }
        }
    
    async def _analyze_spectral_peaks(self, magnitude: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyse les pics spectraux principaux."""
        
        # Moyennage temporel pour identifier les pics fréquentiels dominants
        avg_magnitude = np.mean(magnitude, axis=1)
        
        # Détection de pics
        peaks, properties = signal.find_peaks(
            avg_magnitude, 
            height=np.max(avg_magnitude) * 0.1,
            distance=10
        )
        
        # Conversion en fréquences
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=magnitude.shape[0]*2-1)
        peak_frequencies = frequencies[peaks]
        peak_magnitudes = avg_magnitude[peaks]
        
        # Tri par magnitude décroissante
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        
        return {
            'peak_count': len(peaks),
            'dominant_frequencies': peak_frequencies[sorted_indices[:10]].tolist(),
            'peak_magnitudes': peak_magnitudes[sorted_indices[:10]].tolist(),
            'frequency_spread': float(np.std(peak_frequencies)),
            'spectral_centroid_peaks': float(np.average(peak_frequencies, weights=peak_magnitudes))
        }
    
    async def _calculate_spectral_stability(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Calcule la stabilité spectrale dans le temps."""
        
        # Variance temporelle pour chaque bin fréquentiel
        temporal_variance = np.var(magnitude, axis=1)
        
        # Corrélation temporelle
        correlations = []
        for i in range(magnitude.shape[1] - 1):
            corr = np.corrcoef(magnitude[:, i], magnitude[:, i+1])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        # Mesures de stabilité
        stability_metrics = {
            'temporal_variance_mean': float(np.mean(temporal_variance)),
            'temporal_variance_std': float(np.std(temporal_variance)),
            'temporal_correlation_mean': float(np.mean(correlations)) if correlations else 0.0,
            'temporal_correlation_std': float(np.std(correlations)) if correlations else 0.0,
            'stability_score': float(np.mean(correlations)) if correlations else 0.0
        }
        
        return stability_metrics

class MoodAnalyzer:
    """
    Analyseur de mood/émotion musical basé sur les caractéristiques audio.
    """
    
    def __init__(self):
        self.mood_dimensions = ['valence', 'arousal', 'dominance']
        self.feature_weights = self._initialize_feature_weights()
    
    def _initialize_feature_weights(self) -> Dict[str, Dict[str, float]]:
        """Initialise les poids des caractéristiques pour chaque dimension."""
        return {
            'valence': {  # Positif/Négatif
                'spectral_centroid_mean': 0.3,
                'chroma_energy': 0.25,
                'tempo': 0.2,
                'harmonic_ratio': 0.15,
                'mfcc_0_mean': -0.1
            },
            'arousal': {  # Énergique/Calme
                'rms_energy': 0.4,
                'tempo': 0.3,
                'spectral_bandwidth_mean': 0.2,
                'onset_rate': 0.1
            },
            'dominance': {  # Dominant/Soumis
                'spectral_contrast_mean': 0.3,
                'dynamic_range': 0.25,
                'rhythm_strength': 0.2,
                'percussive_ratio': 0.25
            }
        }
    
    async def analyze_mood(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse le mood basé sur les caractéristiques audio.
        
        Args:
            audio_features: Caractéristiques extraites par AudioAnalyzer
            
        Returns:
            Dict avec scores de mood et classification
        """
        mood_scores = {}
        
        # Calcul des scores pour chaque dimension
        for dimension, weights in self.feature_weights.items():
            score = 0.0
            total_weight = 0.0
            
            for feature, weight in weights.items():
                if feature in audio_features:
                    # Normalisation de la valeur
                    normalized_value = self._normalize_feature_value(feature, audio_features[feature])
                    score += normalized_value * weight
                    total_weight += abs(weight)
            
            # Score final normalisé entre 0 et 1
            mood_scores[dimension] = max(0, min(1, score / total_weight if total_weight > 0 else 0.5))
        
        # Classification du mood
        mood_classification = self._classify_mood(mood_scores)
        
        # Confiance dans la classification
        confidence = self._calculate_confidence(mood_scores, audio_features)
        
        return {
            'mood_scores': mood_scores,
            'mood_classification': mood_classification,
            'confidence': confidence,
            'mood_description': self._get_mood_description(mood_classification),
            'recommended_genres': self._suggest_genres(mood_scores)
        }
    
    def _normalize_feature_value(self, feature_name: str, value: float) -> float:
        """Normalise une valeur de caractéristique entre 0 et 1."""
        
        # Ranges approximatifs pour différentes caractéristiques
        feature_ranges = {
            'tempo': (60, 200),
            'spectral_centroid_mean': (500, 8000),
            'rms_energy': (0, 0.5),
            'chroma_energy': (0, 100),
            'spectral_bandwidth_mean': (500, 4000),
            'spectral_contrast_mean': (0, 40),
            'dynamic_range': (0, 2),
            'harmonic_ratio': (0, 1),
            'percussive_ratio': (0, 1),
            'rhythm_strength': (0, 1),
            'onset_rate': (0, 10),
            'mfcc_0_mean': (-200, 0)
        }
        
        if feature_name in feature_ranges:
            min_val, max_val = feature_ranges[feature_name]
            return (value - min_val) / (max_val - min_val)
        else:
            # Normalisation générique
            return max(0, min(1, value))
    
    def _classify_mood(self, mood_scores: Dict[str, float]) -> str:
        """Classifie le mood basé sur les scores dimensionnels."""
        
        valence = mood_scores.get('valence', 0.5)
        arousal = mood_scores.get('arousal', 0.5)
        dominance = mood_scores.get('dominance', 0.5)
        
        # Classification basée sur le modèle VAD
        if valence > 0.6 and arousal > 0.6:
            return 'happy' if dominance > 0.5 else 'excited'
        elif valence > 0.6 and arousal < 0.4:
            return 'peaceful' if dominance > 0.5 else 'relaxed'
        elif valence < 0.4 and arousal > 0.6:
            return 'angry' if dominance > 0.5 else 'tense'
        elif valence < 0.4 and arousal < 0.4:
            return 'sad' if dominance < 0.5 else 'melancholic'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, mood_scores: Dict[str, float], audio_features: Dict[str, Any]) -> float:
        """Calcule la confiance dans la classification."""
        
        # Facteurs de confiance
        factors = []
        
        # Distance des scores par rapport au centre (0.5)
        center_distance = np.mean([abs(score - 0.5) for score in mood_scores.values()])
        factors.append(center_distance * 2)  # Normalisation
        
        # Présence des caractéristiques clés
        key_features = ['tempo', 'rms_energy', 'spectral_centroid_mean']
        feature_presence = sum(1 for feat in key_features if feat in audio_features)
        factors.append(feature_presence / len(key_features))
        
        # Cohérence des caractéristiques
        if 'tempo' in audio_features and 'rms_energy' in audio_features:
            tempo_energy_coherence = 1 - abs(
                self._normalize_feature_value('tempo', audio_features['tempo']) - 
                self._normalize_feature_value('rms_energy', audio_features['rms_energy'])
            )
            factors.append(tempo_energy_coherence)
        
        return float(np.mean(factors))
    
    def _get_mood_description(self, mood_classification: str) -> str:
        """Retourne une description textuelle du mood."""
        descriptions = {
            'happy': 'Joyful and uplifting music with positive energy',
            'excited': 'High-energy music that creates excitement and enthusiasm',
            'peaceful': 'Calm and serene music promoting relaxation',
            'relaxed': 'Gentle and soothing music for unwinding',
            'angry': 'Intense and aggressive music expressing strong emotions',
            'tense': 'Anxious and edgy music creating tension',
            'sad': 'Melancholic music expressing sorrow or loss',
            'melancholic': 'Bittersweet music with complex emotional depth',
            'neutral': 'Balanced music without strong emotional direction'
        }
        return descriptions.get(mood_classification, 'Unclassified emotional content')
    
    def _suggest_genres(self, mood_scores: Dict[str, float]) -> List[str]:
        """Suggère des genres musicaux basés sur le mood."""
        
        valence = mood_scores.get('valence', 0.5)
        arousal = mood_scores.get('arousal', 0.5)
        
        genres = []
        
        if valence > 0.7 and arousal > 0.7:
            genres.extend(['pop', 'dance', 'funk', 'disco'])
        elif valence > 0.6 and arousal < 0.4:
            genres.extend(['ambient', 'classical', 'new age', 'folk'])
        elif valence < 0.4 and arousal > 0.6:
            genres.extend(['metal', 'punk', 'hardcore', 'industrial'])
        elif valence < 0.4 and arousal < 0.4:
            genres.extend(['blues', 'ballad', 'downtempo', 'drone'])
        else:
            genres.extend(['indie', 'alternative', 'jazz', 'world'])
        
        return genres[:4]  # Limite à 4 suggestions
