"""
Configuration et utilitaires tests audio
========================================

Utilitaires réels pour configuration et helpers des tests audio.
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Tuple, Any
import json
import os


class AudioTestUtils:
    """Utilitaires réels pour tests audio."""
    
    @staticmethod
    def generate_test_signals(sample_rate=44100, duration=1.0):
        """Génère des signaux test réels et variés."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        signals = {}
        
        # 1. Sinusoïde pure
        signals['sine_440hz'] = {
            'signal': 0.5 * np.sin(2 * np.pi * 440 * t),
            'description': 'Sinusoïde 440Hz (A4)',
            'expected_freq': 440.0,
            'expected_amplitude': 0.5
        }
        
        # 2. Accord majeur
        signals['c_major_chord'] = {
            'signal': (
                0.3 * np.sin(2 * np.pi * 261.63 * t) +  # C4
                0.25 * np.sin(2 * np.pi * 329.63 * t) + # E4
                0.2 * np.sin(2 * np.pi * 392.00 * t)    # G4
            ),
            'description': 'Accord C majeur',
            'expected_freqs': [261.63, 329.63, 392.00],
            'harmonic_content': 'high'
        }
        
        # 3. Signal avec bruit
        clean_signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
        noise = 0.05 * np.random.randn(len(t))
        signals['noisy_signal'] = {
            'signal': clean_signal + noise,
            'description': 'Signal 1kHz + bruit blanc',
            'snr_db': 20.0,
            'noise_level': 0.05
        }
        
        # 4. Sweep fréquentiel
        f_start, f_end = 100, 8000
        signals['frequency_sweep'] = {
            'signal': 0.4 * np.sin(2 * np.pi * np.linspace(f_start, f_end, len(t)) * t),
            'description': f'Sweep {f_start}-{f_end}Hz',
            'freq_start': f_start,
            'freq_end': f_end
        }
        
        # 5. Signal percussif simulé
        envelope = np.exp(-5 * t)
        oscillation = np.sin(2 * np.pi * 80 * t)
        signals['kick_drum'] = {
            'signal': envelope * oscillation,
            'description': 'Simulation kick drum',
            'transient': True,
            'decay_time': 0.2
        }
        
        return signals
    
    @staticmethod
    def calculate_audio_metrics(signal, sample_rate=44100):
        """Calcule métriques audio réelles."""
        metrics = {}
        
        # Métriques temporelles
        metrics['rms_level'] = np.sqrt(np.mean(signal**2))
        metrics['peak_level'] = np.max(np.abs(signal))
        metrics['crest_factor'] = metrics['peak_level'] / (metrics['rms_level'] + 1e-8)
        metrics['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(signal))
        
        # Métriques spectrales
        stft = librosa.stft(signal)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sample_rate)
        
        # Centroïde spectral
        spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
        metrics['spectral_centroid'] = np.mean(spectral_centroids)
        
        # Rolloff spectral
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)
        metrics['spectral_rolloff'] = np.mean(rolloff)
        
        # Bandwidth spectrale
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
        metrics['spectral_bandwidth'] = np.mean(bandwidth)
        
        # Estimation tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=signal, sr=sample_rate)
            metrics['estimated_tempo'] = float(tempo)
        except:
            metrics['estimated_tempo'] = None
        
        return metrics
    
    @staticmethod
    def apply_audio_effects(signal, effect_type, **params):
        """Applique effets audio réels."""
        if effect_type == 'lowpass':
            cutoff = params.get('cutoff', 8000)
            nyquist = params.get('sample_rate', 44100) / 2
            b, a = scipy.signal.butter(4, cutoff/nyquist, 'lowpass')
            return scipy.signal.filtfilt(b, a, signal)
            
        elif effect_type == 'highpass':
            cutoff = params.get('cutoff', 80)
            nyquist = params.get('sample_rate', 44100) / 2
            b, a = scipy.signal.butter(4, cutoff/nyquist, 'highpass')
            return scipy.signal.filtfilt(b, a, signal)
            
        elif effect_type == 'distortion':
            gain = params.get('gain', 2.0)
            threshold = params.get('threshold', 0.7)
            amplified = signal * gain
            return np.where(
                np.abs(amplified) > threshold,
                np.sign(amplified) * threshold,
                amplified
            )
            
        elif effect_type == 'reverb_simple':
            delay_samples = int(params.get('delay_ms', 50) * params.get('sample_rate', 44100) / 1000)
            decay = params.get('decay', 0.3)
            delayed = np.zeros_like(signal)
            if delay_samples < len(signal):
                delayed[delay_samples:] = signal[:-delay_samples] * decay
            return signal + delayed
            
        else:
            return signal
    
    @staticmethod
    def validate_audio_quality(signal, expected_metrics=None):
        """Validation qualité audio réelle."""
        issues = []
        
        # Vérification clipping
        if np.any(np.abs(signal) >= 1.0):
            issues.append("Signal clipping détecté")
        
        # Vérification DC offset
        dc_offset = np.mean(signal)
        if abs(dc_offset) > 0.01:
            issues.append(f"DC offset significatif: {dc_offset:.4f}")
        
        # Vérification silence
        rms_level = np.sqrt(np.mean(signal**2))
        if rms_level < 1e-6:
            issues.append("Signal trop faible ou silence")
        
        # Vérification NaN/Inf
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            issues.append("Valeurs NaN ou infinies détectées")
        
        # Comparaison métriques attendues
        if expected_metrics:
            actual_metrics = AudioTestUtils.calculate_audio_metrics(signal)
            for key, expected_value in expected_metrics.items():
                if key in actual_metrics:
                    actual_value = actual_metrics[key]
                    if isinstance(expected_value, (int, float)):
                        tolerance = expected_value * 0.1  # 10% tolerance
                        if abs(actual_value - expected_value) > tolerance:
                            issues.append(f"{key}: attendu {expected_value}, obtenu {actual_value}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'quality_score': max(0, 1 - len(issues) * 0.2)
        }


class AudioTestData:
    """Gestionnaire de données test audio."""
    
    def __init__(self, base_path="/tmp/audio_test_data"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def create_test_dataset(self, num_samples=100):
        """Crée un dataset test réel."""
        dataset = {
            'samples': [],
            'labels': [],
            'metadata': {
                'num_samples': num_samples,
                'sample_rate': 44100,
                'duration_seconds': 1.0,
                'created_at': str(np.datetime64('now'))
            }
        }
        
        # Génération échantillons
        for i in range(num_samples):
            # Fréquence aléatoire entre 200-2000Hz
            freq = np.random.uniform(200, 2000)
            
            # Amplitude aléatoire
            amplitude = np.random.uniform(0.1, 0.8)
            
            # Signal
            t = np.linspace(0, 1.0, 44100)
            signal = amplitude * np.sin(2 * np.pi * freq * t)
            
            # Ajout bruit léger
            noise_level = np.random.uniform(0.01, 0.05)
            noise = noise_level * np.random.randn(len(signal))
            noisy_signal = signal + noise
            
            # Classification par bande de fréquence
            if freq < 500:
                label = 'low_freq'
            elif freq < 1000:
                label = 'mid_freq'
            else:
                label = 'high_freq'
            
            dataset['samples'].append(noisy_signal.tolist())
            dataset['labels'].append(label)
        
        # Sauvegarde
        dataset_file = os.path.join(self.base_path, 'test_dataset.json')
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f)
        
        return dataset_file
    
    def load_test_dataset(self, dataset_file):
        """Charge un dataset test."""
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        # Conversion en numpy arrays
        dataset['samples'] = [np.array(sample) for sample in dataset['samples']]
        
        return dataset


class PerformanceProfiler:
    """Profileur performance pour tests audio."""
    
    def __init__(self):
        self.measurements = {}
    
    def measure_function(self, func, *args, **kwargs):
        """Mesure performance d'une fonction."""
        import time
        import psutil
        import tracemalloc
        
        # Démarrage mesures
        start_time = time.perf_counter()
        tracemalloc.start()
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        
        # Exécution fonction
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Fin mesures
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cpu_after = process.cpu_percent()
        
        # Métriques
        metrics = {
            'execution_time_seconds': end_time - start_time,
            'memory_current_mb': current / (1024 * 1024),
            'memory_peak_mb': peak / (1024 * 1024),
            'cpu_usage_percent': max(cpu_after - cpu_before, 0),
            'success': success,
            'error': error
        }
        
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        self.measurements[func_name] = metrics
        
        return result, metrics
    
    def get_summary_report(self):
        """Génère rapport résumé performance."""
        if not self.measurements:
            return "Aucune mesure disponible"
        
        report = []
        report.append("📊 RAPPORT PERFORMANCE")
        report.append("=" * 40)
        
        total_time = sum(m['execution_time_seconds'] for m in self.measurements.values())
        total_memory = sum(m['memory_peak_mb'] for m in self.measurements.values())
        success_rate = sum(1 for m in self.measurements.values() if m['success']) / len(self.measurements)
        
        report.append(f"Fonctions testées: {len(self.measurements)}")
        report.append(f"Temps total: {total_time:.3f}s")
        report.append(f"Mémoire totale: {total_memory:.1f}MB")
        report.append(f"Taux succès: {success_rate:.1%}")
        report.append("")
        
        # Détail par fonction
        for func_name, metrics in self.measurements.items():
            status = "✅" if metrics['success'] else "❌"
            report.append(f"{status} {func_name}:")
            report.append(f"   Temps: {metrics['execution_time_seconds']:.3f}s")
            report.append(f"   Mémoire: {metrics['memory_peak_mb']:.1f}MB")
            if not metrics['success']:
                report.append(f"   Erreur: {metrics['error']}")
            report.append("")
        
        return "\n".join(report)
