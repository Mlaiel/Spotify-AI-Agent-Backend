"""
Audio Effects Engine - Enterprise Edition
========================================

Moteur d'effets audio professionnel pour Spotify AI Agent.
Effets temps réel, traitement créatif, et enhancement audio.
"""

import asyncio
import logging
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft, fftfreq
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import json
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# === Configuration et types ===
class EffectType(Enum):
    REVERB = "reverb"
    DELAY = "delay"
    CHORUS = "chorus"
    DISTORTION = "distortion"
    EQUALIZER = "equalizer"
    COMPRESSOR = "compressor"
    NOISE_GATE = "noise_gate"
    PITCH_SHIFT = "pitch_shift"
    TIME_STRETCH = "time_stretch"
    PHASER = "phaser"
    FLANGER = "flanger"
    TREMOLO = "tremolo"
    VIBRATO = "vibrato"
    LOW_PASS = "low_pass"
    HIGH_PASS = "high_pass"
    BAND_PASS = "band_pass"
    NOTCH = "notch"

class EffectQuality(Enum):
    DRAFT = "draft"       # Traitement rapide
    STANDARD = "standard" # Qualité équilibrée
    HIGH = "high"         # Haute qualité
    STUDIO = "studio"     # Qualité studio

@dataclass
class EffectParameters:
    """Paramètres génériques pour un effet."""
    effect_type: EffectType
    intensity: float = 0.5  # 0.0 à 1.0
    mix: float = 0.5        # Dry/Wet mix
    enabled: bool = True
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}

@dataclass
class EffectChain:
    """Chaîne d'effets audio."""
    name: str
    effects: List[EffectParameters]
    bypass: bool = False
    input_gain: float = 1.0
    output_gain: float = 1.0

# === Effets de base ===
class ReverbEffect:
    """Effet de réverbération convolutionnel."""
    
    def __init__(self, room_size: float = 0.5, damping: float = 0.5, wet_level: float = 0.3):
        self.room_size = np.clip(room_size, 0.0, 1.0)
        self.damping = np.clip(damping, 0.0, 1.0)
        self.wet_level = np.clip(wet_level, 0.0, 1.0)
        
        # Paramètres de réverbération
        self.delay_times = np.array([0.03, 0.05, 0.07, 0.11, 0.13, 0.17, 0.19])  # secondes
        self.feedback_gains = np.array([0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4])
        
    async def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique l'effet de réverbération."""
        
        loop = asyncio.get_event_loop()
        
        def _process_reverb():
            # Création des lignes de délai
            reverb_output = np.zeros_like(audio)
            
            for delay_time, feedback_gain in zip(self.delay_times, self.feedback_gains):
                # Ajustement selon la taille de la pièce
                adjusted_delay = delay_time * (1 + self.room_size)
                delay_samples = int(adjusted_delay * sample_rate)
                
                if delay_samples < len(audio):
                    # Ligne de délai avec feedback
                    delayed_signal = np.zeros_like(audio)
                    delayed_signal[delay_samples:] = audio[:-delay_samples]
                    
                    # Application du feedback avec amortissement
                    feedback_signal = delayed_signal * feedback_gain * (1 - self.damping)
                    reverb_output += feedback_signal
            
            # Normalisation et mélange
            if np.max(np.abs(reverb_output)) > 0:
                reverb_output = reverb_output / np.max(np.abs(reverb_output))
            
            # Mix dry/wet
            return (1 - self.wet_level) * audio + self.wet_level * reverb_output
        
        return await loop.run_in_executor(None, _process_reverb)

class DelayEffect:
    """Effet de délai/écho."""
    
    def __init__(self, delay_time: float = 0.3, feedback: float = 0.4, mix: float = 0.3):
        self.delay_time = delay_time  # secondes
        self.feedback = np.clip(feedback, 0.0, 0.95)
        self.mix = np.clip(mix, 0.0, 1.0)
    
    async def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique l'effet de délai."""
        
        delay_samples = int(self.delay_time * sample_rate)
        
        if delay_samples >= len(audio):
            return audio
        
        # Signal de sortie avec espace pour les échos
        output_length = len(audio) + delay_samples * 3
        output = np.zeros(output_length)
        output[:len(audio)] = audio
        
        # Application du délai avec feedback
        current_delay = delay_samples
        current_gain = self.feedback
        
        while current_delay < output_length and current_gain > 0.01:
            if current_delay + len(audio) <= output_length:
                output[current_delay:current_delay + len(audio)] += audio * current_gain
            
            current_delay += delay_samples
            current_gain *= self.feedback
        
        # Retour à la longueur originale avec mix
        processed = output[:len(audio)]
        return (1 - self.mix) * audio + self.mix * processed

class ChorusEffect:
    """Effet de chorus avec modulation LFO."""
    
    def __init__(self, rate: float = 1.5, depth: float = 0.3, mix: float = 0.5):
        self.rate = rate      # Hz
        self.depth = depth    # 0.0 à 1.0
        self.mix = mix        # 0.0 à 1.0
    
    async def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique l'effet de chorus."""
        
        loop = asyncio.get_event_loop()
        
        def _process_chorus():
            # Génération du LFO (Low Frequency Oscillator)
            t = np.arange(len(audio)) / sample_rate
            lfo = np.sin(2 * np.pi * self.rate * t)
            
            # Modulation du délai
            base_delay = 0.02  # 20ms de base
            max_delay = base_delay + (self.depth * 0.01)  # Jusqu'à 10ms de modulation
            
            modulated_delay = base_delay + (lfo * self.depth * 0.01)
            
            # Application du délai modulé
            output = np.zeros_like(audio)
            
            for i, delay_time in enumerate(modulated_delay):
                delay_samples = int(delay_time * sample_rate)
                
                if i + delay_samples < len(audio):
                    output[i] = audio[i + delay_samples] if i + delay_samples < len(audio) else 0
            
            # Mix avec le signal original
            return (1 - self.mix) * audio + self.mix * output
        
        return await loop.run_in_executor(None, _process_chorus)

class DistortionEffect:
    """Effet de distorsion/saturation."""
    
    def __init__(self, drive: float = 0.5, tone: float = 0.5, level: float = 0.8):
        self.drive = np.clip(drive, 0.0, 1.0)
        self.tone = np.clip(tone, 0.0, 1.0)
        self.level = np.clip(level, 0.0, 1.0)
    
    async def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique l'effet de distorsion."""
        
        # Amplification pré-distorsion
        driven_signal = audio * (1 + self.drive * 10)
        
        # Fonction de saturation
        distorted = np.tanh(driven_signal)
        
        # Filtre de tonalité (passe-bas simple)
        if self.tone < 1.0:
            # Fréquence de coupure basée sur le paramètre tone
            cutoff_freq = 2000 + (self.tone * 8000)  # 2kHz à 10kHz
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff_freq / nyquist
            
            if normalized_cutoff < 1.0:
                b, a = signal.butter(2, normalized_cutoff, btype='low')
                distorted = signal.filtfilt(b, a, distorted)
        
        # Ajustement du niveau de sortie
        return distorted * self.level

class EqualizerEffect:
    """Égaliseur graphique 5 bandes."""
    
    def __init__(self, bands: Dict[str, float] = None):
        # Bandes par défaut (dB)
        self.default_bands = {
            'low': 0.0,      # 100 Hz
            'low_mid': 0.0,  # 500 Hz
            'mid': 0.0,      # 1 kHz
            'high_mid': 0.0, # 3 kHz
            'high': 0.0      # 8 kHz
        }
        
        self.bands = bands or self.default_bands
        
        # Fréquences centrales des bandes
        self.frequencies = {
            'low': 100,
            'low_mid': 500,
            'mid': 1000,
            'high_mid': 3000,
            'high': 8000
        }
    
    async def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique l'égalisation."""
        
        loop = asyncio.get_event_loop()
        
        def _process_eq():
            processed = audio.copy()
            nyquist = sample_rate / 2
            
            for band, gain_db in self.bands.items():
                if abs(gain_db) > 0.1:  # Seulement si gain significatif
                    freq = self.frequencies[band]
                    
                    if freq < nyquist:
                        # Filtre peak/notch
                        Q = 2.0  # Facteur de qualité
                        gain_linear = 10 ** (gain_db / 20)
                        
                        # Calcul des coefficients du filtre
                        w0 = 2 * np.pi * freq / sample_rate
                        cos_w0 = np.cos(w0)
                        sin_w0 = np.sin(w0)
                        alpha = sin_w0 / (2 * Q)
                        
                        # Coefficients pour un filtre peaking
                        A = gain_linear
                        b0 = 1 + alpha * A
                        b1 = -2 * cos_w0
                        b2 = 1 - alpha * A
                        a0 = 1 + alpha / A
                        a1 = -2 * cos_w0
                        a2 = 1 - alpha / A
                        
                        # Normalisation
                        b = np.array([b0, b1, b2]) / a0
                        a = np.array([1, a1 / a0, a2 / a0])
                        
                        # Application du filtre
                        processed = signal.lfilter(b, a, processed)
            
            return processed
        
        return await loop.run_in_executor(None, _process_eq)

class CompressorEffect:
    """Compresseur dynamique."""
    
    def __init__(self, threshold: float = -20, ratio: float = 4, attack: float = 0.003, release: float = 0.1):
        self.threshold = threshold  # dB
        self.ratio = ratio         # 1:ratio
        self.attack = attack       # secondes
        self.release = release     # secondes
    
    async def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique la compression dynamique."""
        
        loop = asyncio.get_event_loop()
        
        def _process_compression():
            # Conversion en dB
            audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
            
            # Calcul des coefficients d'attaque et relâchement
            attack_coeff = np.exp(-1 / (self.attack * sample_rate))
            release_coeff = np.exp(-1 / (self.release * sample_rate))
            
            # Détection d'enveloppe
            envelope = np.zeros_like(audio_db)
            envelope[0] = audio_db[0]
            
            for i in range(1, len(audio_db)):
                if audio_db[i] > envelope[i-1]:
                    # Attaque
                    envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * audio_db[i]
                else:
                    # Relâchement
                    envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * audio_db[i]
            
            # Calcul de la réduction de gain
            gain_reduction = np.zeros_like(envelope)
            mask = envelope > self.threshold
            
            # Formule de compression
            over_threshold = envelope[mask] - self.threshold
            gain_reduction[mask] = over_threshold * (1 - 1/self.ratio)
            
            # Application de la réduction (en linéaire)
            gain_linear = 10 ** (-gain_reduction / 20)
            
            return audio * gain_linear
        
        return await loop.run_in_executor(None, _process_compression)

class PitchShiftEffect:
    """Effet de pitch shifting."""
    
    def __init__(self, semitones: float = 0):
        self.semitones = semitones
    
    async def apply(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Applique le pitch shifting."""
        
        if abs(self.semitones) < 0.01:
            return audio
        
        loop = asyncio.get_event_loop()
        
        def _process_pitch_shift():
            # Utilisation de la transformée PSOLA simplifiée
            # Facteur de pitch
            pitch_factor = 2 ** (self.semitones / 12)
            
            # Méthode par résynchèse STFT
            stft = librosa.stft(audio, hop_length=512)
            
            # Modification de la phase pour le pitch shifting
            stft_shifted = np.zeros_like(stft, dtype=complex)
            
            for i in range(stft.shape[1]):
                if i == 0:
                    stft_shifted[:, i] = stft[:, i]
                else:
                    # Calcul de la différence de phase
                    phase_diff = np.angle(stft[:, i]) - np.angle(stft[:, i-1])
                    
                    # Ajustement pour le pitch shifting
                    phase_diff_shifted = phase_diff * pitch_factor
                    
                    # Reconstruction
                    magnitude = np.abs(stft[:, i])
                    phase = np.angle(stft_shifted[:, i-1]) + phase_diff_shifted
                    
                    stft_shifted[:, i] = magnitude * np.exp(1j * phase)
            
            # Reconstruction du signal
            return librosa.istft(stft_shifted, hop_length=512, length=len(audio))
        
        return await loop.run_in_executor(None, _process_pitch_shift)

# === Moteur principal d'effets ===
class AudioEffectsEngine:
    """
    Moteur d'effets audio haute performance.
    """
    
    def __init__(self):
        self.effects_library = {
            EffectType.REVERB: ReverbEffect,
            EffectType.DELAY: DelayEffect,
            EffectType.CHORUS: ChorusEffect,
            EffectType.DISTORTION: DistortionEffect,
            EffectType.EQUALIZER: EqualizerEffect,
            EffectType.COMPRESSOR: CompressorEffect,
            EffectType.PITCH_SHIFT: PitchShiftEffect,
        }
        
        self.presets = self._load_presets()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _load_presets(self) -> Dict[str, EffectChain]:
        """Charge les presets d'effets prédéfinis."""
        
        presets = {
            'vocal_enhance': EffectChain(
                name='Vocal Enhancement',
                effects=[
                    EffectParameters(
                        effect_type=EffectType.COMPRESSOR,
                        intensity=0.6,
                        custom_params={'threshold': -18, 'ratio': 3, 'attack': 0.005}
                    ),
                    EffectParameters(
                        effect_type=EffectType.EQUALIZER,
                        intensity=0.7,
                        custom_params={'bands': {'low': -2, 'mid': 2, 'high': 3}}
                    ),
                    EffectParameters(
                        effect_type=EffectType.REVERB,
                        intensity=0.3,
                        mix=0.15
                    )
                ]
            ),
            
            'guitar_rock': EffectChain(
                name='Rock Guitar',
                effects=[
                    EffectParameters(
                        effect_type=EffectType.DISTORTION,
                        intensity=0.7,
                        custom_params={'drive': 0.8, 'tone': 0.6}
                    ),
                    EffectParameters(
                        effect_type=EffectType.DELAY,
                        intensity=0.4,
                        custom_params={'delay_time': 0.25, 'feedback': 0.3}
                    ),
                    EffectParameters(
                        effect_type=EffectType.REVERB,
                        intensity=0.5,
                        mix=0.2
                    )
                ]
            ),
            
            'ambient_space': EffectChain(
                name='Ambient Space',
                effects=[
                    EffectParameters(
                        effect_type=EffectType.REVERB,
                        intensity=0.9,
                        mix=0.6,
                        custom_params={'room_size': 0.9, 'damping': 0.3}
                    ),
                    EffectParameters(
                        effect_type=EffectType.CHORUS,
                        intensity=0.5,
                        custom_params={'rate': 0.8, 'depth': 0.4}
                    ),
                    EffectParameters(
                        effect_type=EffectType.DELAY,
                        intensity=0.6,
                        custom_params={'delay_time': 0.5, 'feedback': 0.5}
                    )
                ]
            ),
            
            'clean_master': EffectChain(
                name='Clean Master',
                effects=[
                    EffectParameters(
                        effect_type=EffectType.EQUALIZER,
                        intensity=0.5,
                        custom_params={'bands': {'low': 1, 'mid': 0, 'high': 2}}
                    ),
                    EffectParameters(
                        effect_type=EffectType.COMPRESSOR,
                        intensity=0.4,
                        custom_params={'threshold': -12, 'ratio': 2, 'attack': 0.01}
                    )
                ]
            )
        }
        
        return presets
    
    async def apply_effect_chain(
        self,
        audio: np.ndarray,
        sample_rate: int,
        effect_chain: EffectChain
    ) -> np.ndarray:
        """
        Applique une chaîne d'effets au signal audio.
        
        Args:
            audio: Signal audio d'entrée
            sample_rate: Taux d'échantillonnage
            effect_chain: Chaîne d'effets à appliquer
            
        Returns:
            Signal audio traité
        """
        
        if effect_chain.bypass:
            return audio
        
        # Application du gain d'entrée
        processed = audio * effect_chain.input_gain
        
        # Application séquentielle des effets
        for effect_params in effect_chain.effects:
            if effect_params.enabled:
                processed = await self._apply_single_effect(
                    processed, sample_rate, effect_params
                )
        
        # Application du gain de sortie
        processed = processed * effect_chain.output_gain
        
        return processed
    
    async def _apply_single_effect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        effect_params: EffectParameters
    ) -> np.ndarray:
        """Applique un effet unique."""
        
        if effect_params.effect_type not in self.effects_library:
            logger.warning(f"Effect {effect_params.effect_type} not available")
            return audio
        
        # Création de l'instance d'effet
        effect_class = self.effects_library[effect_params.effect_type]
        
        # Paramètres personnalisés ou par défaut
        if effect_params.custom_params:
            effect_instance = effect_class(**effect_params.custom_params)
        else:
            # Paramètres basés sur l'intensité
            effect_instance = self._create_effect_from_intensity(
                effect_class, effect_params.intensity
            )
        
        # Application de l'effet
        processed = await effect_instance.apply(audio, sample_rate)
        
        # Mix dry/wet
        return (1 - effect_params.mix) * audio + effect_params.mix * processed
    
    def _create_effect_from_intensity(self, effect_class, intensity: float):
        """Crée une instance d'effet basée sur l'intensité."""
        
        if effect_class == ReverbEffect:
            return ReverbEffect(
                room_size=intensity,
                damping=0.5,
                wet_level=intensity * 0.5
            )
        elif effect_class == DelayEffect:
            return DelayEffect(
                delay_time=0.1 + intensity * 0.4,
                feedback=intensity * 0.6,
                mix=intensity * 0.5
            )
        elif effect_class == ChorusEffect:
            return ChorusEffect(
                rate=0.5 + intensity * 2,
                depth=intensity * 0.5,
                mix=intensity * 0.6
            )
        elif effect_class == DistortionEffect:
            return DistortionEffect(
                drive=intensity,
                tone=0.5,
                level=1.0 - intensity * 0.3
            )
        elif effect_class == CompressorEffect:
            return CompressorEffect(
                threshold=-20 + intensity * 10,
                ratio=1 + intensity * 5,
                attack=0.001 + intensity * 0.01,
                release=0.05 + intensity * 0.2
            )
        else:
            return effect_class()
    
    async def apply_preset(
        self,
        audio: np.ndarray,
        sample_rate: int,
        preset_name: str
    ) -> np.ndarray:
        """Applique un preset d'effets prédéfini."""
        
        if preset_name not in self.presets:
            logger.error(f"Preset '{preset_name}' not found")
            return audio
        
        preset_chain = self.presets[preset_name]
        return await self.apply_effect_chain(audio, sample_rate, preset_chain)
    
    async def process_file(
        self,
        input_path: str,
        output_path: str,
        effect_chain: Union[EffectChain, str],
        quality: EffectQuality = EffectQuality.HIGH
    ) -> bool:
        """
        Traite un fichier audio avec une chaîne d'effets.
        
        Args:
            input_path: Chemin du fichier source
            output_path: Chemin du fichier destination
            effect_chain: Chaîne d'effets ou nom de preset
            quality: Qualité de traitement
            
        Returns:
            True si succès, False sinon
        """
        
        try:
            # Chargement du fichier
            audio, sr = librosa.load(input_path, sr=None)
            
            logger.info(f"Processing: {input_path} -> {output_path}")
            logger.info(f"Audio: {len(audio)} samples, {sr}Hz")
            
            # Détermination de la chaîne d'effets
            if isinstance(effect_chain, str):
                if effect_chain in self.presets:
                    chain = self.presets[effect_chain]
                else:
                    logger.error(f"Unknown preset: {effect_chain}")
                    return False
            else:
                chain = effect_chain
            
            # Traitement
            processed_audio = await self.apply_effect_chain(audio, sr, chain)
            
            # Sauvegarde avec qualité appropriée
            if quality == EffectQuality.STUDIO:
                # Haute résolution
                sf.write(output_path, processed_audio, sr, subtype='PCM_24')
            elif quality == EffectQuality.HIGH:
                sf.write(output_path, processed_audio, sr, subtype='PCM_16')
            else:
                # Qualité standard/draft
                sf.write(output_path, processed_audio, sr)
            
            logger.info(f"Effects processing completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Effects processing failed: {e}")
            return False
    
    async def batch_process(
        self,
        input_files: List[str],
        output_dir: str,
        effect_chain: Union[EffectChain, str],
        quality: EffectQuality = EffectQuality.HIGH
    ) -> Dict[str, bool]:
        """Traitement par lot avec effets."""
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Limitation de la parallélisation
        semaphore = asyncio.Semaphore(2)
        
        async def process_single(input_file: str) -> bool:
            async with semaphore:
                input_path = Path(input_file)
                output_file = output_path / f"{input_path.stem}_fx{input_path.suffix}"
                
                return await self.process_file(
                    str(input_path),
                    str(output_file),
                    effect_chain,
                    quality
                )
        
        # Traitement parallèle
        tasks = [(file, process_single(file)) for file in input_files]
        
        for input_file, task in tasks:
            try:
                success = await task
                results[input_file] = success
            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")
                results[input_file] = False
        
        return results
    
    def get_available_presets(self) -> List[str]:
        """Retourne la liste des presets disponibles."""
        return list(self.presets.keys())
    
    def get_available_effects(self) -> List[str]:
        """Retourne la liste des effets disponibles."""
        return [effect.value for effect in EffectType]
    
    def create_custom_chain(
        self,
        name: str,
        effects_config: List[Dict[str, Any]]
    ) -> EffectChain:
        """Crée une chaîne d'effets personnalisée."""
        
        effects = []
        for config in effects_config:
            effect_type = EffectType(config.get('type', 'reverb'))
            intensity = config.get('intensity', 0.5)
            mix = config.get('mix', 0.5)
            custom_params = config.get('params', {})
            
            effects.append(EffectParameters(
                effect_type=effect_type,
                intensity=intensity,
                mix=mix,
                custom_params=custom_params
            ))
        
        return EffectChain(name=name, effects=effects)
    
    def save_preset(self, preset_name: str, effect_chain: EffectChain):
        """Sauvegarde un nouveau preset."""
        self.presets[preset_name] = effect_chain
        logger.info(f"Preset '{preset_name}' saved")
    
    def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# === Factory functions ===
def create_effects_engine() -> AudioEffectsEngine:
    """Factory pour créer le moteur d'effets."""
    return AudioEffectsEngine()

def create_effect_chain(name: str, effects: List[Dict[str, Any]]) -> EffectChain:
    """Factory pour créer une chaîne d'effets."""
    engine = AudioEffectsEngine()
    return engine.create_custom_chain(name, effects)

def create_reverb_effect(
    room_size: float = 0.5,
    damping: float = 0.5,
    wet_level: float = 0.3
) -> ReverbEffect:
    """Factory pour créer un effet de réverbération."""
    return ReverbEffect(room_size, damping, wet_level)

def create_delay_effect(
    delay_time: float = 0.3,
    feedback: float = 0.4,
    mix: float = 0.3
) -> DelayEffect:
    """Factory pour créer un effet de délai."""
    return DelayEffect(delay_time, feedback, mix)

def create_compressor_effect(
    threshold: float = -20,
    ratio: float = 4,
    attack: float = 0.003,
    release: float = 0.1
) -> CompressorEffect:
    """Factory pour créer un compresseur."""
    return CompressorEffect(threshold, ratio, attack, release)
