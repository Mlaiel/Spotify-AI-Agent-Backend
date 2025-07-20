"""
Détecteurs de Seuils Adaptatifs et Statistiques Avancés
======================================================

Auteur: Fahed Mlaiel
Rôles: Lead Dev + Architecte IA, DBA & Data Engineer (PostgreSQL/Redis/MongoDB)

Ce module implémente des détecteurs de seuils intelligents avec apprentissage
adaptatif et analyse statistique avancée pour le monitoring en temps réel.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy import signal
import warnings
from collections import deque, defaultdict
import json
import redis
import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

Base = declarative_base()

class ThresholdHistory(Base):
    """Modèle pour l'historique des seuils"""
    __tablename__ = 'threshold_history'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), nullable=False)
    threshold_value = Column(Float, nullable=False)
    threshold_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    accuracy_score = Column(Float, default=0.0)
    false_positive_rate = Column(Float, default=0.0)
    false_negative_rate = Column(Float, default=0.0)

@dataclass
class ThresholdConfig:
    """Configuration des seuils"""
    metric_name: str
    initial_threshold: float
    adaptation_rate: float = 0.1
    min_threshold: float = 0.0
    max_threshold: float = float('inf')
    sensitivity: float = 1.0
    window_size: int = 100
    cooldown_period: int = 300  # secondes
    statistical_method: str = "zscore"  # zscore, iqr, mad, percentile

@dataclass
class StatisticalResult:
    """Résultat d'analyse statistique"""
    is_anomaly: bool
    confidence: float
    statistic_value: float
    p_value: float
    threshold_used: float
    method: str
    context: Dict[str, Any]

class StatisticalMethod(Enum):
    """Méthodes statistiques disponibles"""
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    IQR = "iqr"
    MAD = "mad"  # Median Absolute Deviation
    PERCENTILE = "percentile"
    GRUBBS = "grubbs"
    DIXON = "dixon"
    SHAPIRO_WILK = "shapiro_wilk"
    ANDERSON_DARLING = "anderson_darling"

class AdaptiveThresholdDetector:
    """Détecteur de seuils adaptatifs avec apprentissage automatique"""
    
    def __init__(self, config: ThresholdConfig, db_url: str = "sqlite:///thresholds.db"):
        self.config = config
        self.current_threshold = config.initial_threshold
        self.data_window = deque(maxlen=config.window_size)
        self.alert_history = deque(maxlen=1000)
        self.last_alert_time = None
        self.performance_metrics = defaultdict(list)
        
        # Base de données
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Redis pour cache
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        
        # Statistiques en cours
        self.stats_cache = {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'mad': 0.0,
            'q1': 0.0,
            'q3': 0.0,
            'iqr': 0.0
        }
        
        logger.info(f"Détecteur adaptatif initialisé pour {config.metric_name}")
    
    async def detect(self, value: float, timestamp: datetime = None) -> StatisticalResult:
        """Détecte les anomalies avec seuil adaptatif"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ajouter la valeur à la fenêtre
        self.data_window.append((value, timestamp))
        
        # Mettre à jour les statistiques
        await self._update_statistics()
        
        # Vérifier le cooldown
        if self._in_cooldown():
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=value,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method=self.config.statistical_method,
                context={'cooldown': True}
            )
        
        # Détection avec méthode statistique
        result = await self._detect_with_method(value, timestamp)
        
        # Adaptation du seuil si nécessaire
        if result.is_anomaly:
            await self._adapt_threshold(value, result)
            self.last_alert_time = timestamp
            self.alert_history.append((value, timestamp, result.confidence))
        
        # Enregistrer les performances
        await self._update_performance_metrics(result)
        
        return result
    
    async def _update_statistics(self):
        """Met à jour les statistiques de la fenêtre de données"""
        if len(self.data_window) < 10:
            return
        
        values = [item[0] for item in self.data_window]
        
        self.stats_cache['mean'] = np.mean(values)
        self.stats_cache['std'] = np.std(values, ddof=1)
        self.stats_cache['median'] = np.median(values)
        self.stats_cache['mad'] = stats.median_abs_deviation(values)
        
        q1, q3 = np.percentile(values, [25, 75])
        self.stats_cache['q1'] = q1
        self.stats_cache['q3'] = q3
        self.stats_cache['iqr'] = q3 - q1
    
    async def _detect_with_method(self, value: float, timestamp: datetime) -> StatisticalResult:
        """Détecte avec la méthode statistique configurée"""
        method = self.config.statistical_method
        
        if method == StatisticalMethod.ZSCORE.value:
            return await self._detect_zscore(value)
        elif method == StatisticalMethod.MODIFIED_ZSCORE.value:
            return await self._detect_modified_zscore(value)
        elif method == StatisticalMethod.IQR.value:
            return await self._detect_iqr(value)
        elif method == StatisticalMethod.MAD.value:
            return await self._detect_mad(value)
        elif method == StatisticalMethod.PERCENTILE.value:
            return await self._detect_percentile(value)
        elif method == StatisticalMethod.GRUBBS.value:
            return await self._detect_grubbs(value)
        else:
            return await self._detect_zscore(value)  # Fallback
    
    async def _detect_zscore(self, value: float) -> StatisticalResult:
        """Détection basée sur Z-Score"""
        if len(self.data_window) < 10:
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=value,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method="zscore",
                context={'insufficient_data': True}
            )
        
        mean = self.stats_cache['mean']
        std = self.stats_cache['std']
        
        if std == 0:
            z_score = 0
        else:
            z_score = abs(value - mean) / std
        
        # Calcul de la p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Seuil adaptatif basé sur la sensibilité
        threshold = self.current_threshold * self.config.sensitivity
        is_anomaly = z_score > threshold
        
        confidence = min(z_score / threshold, 1.0) if threshold > 0 else 0.0
        
        return StatisticalResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            statistic_value=z_score,
            p_value=p_value,
            threshold_used=threshold,
            method="zscore",
            context={
                'mean': mean,
                'std': std,
                'raw_value': value
            }
        )
    
    async def _detect_modified_zscore(self, value: float) -> StatisticalResult:
        """Détection basée sur Z-Score modifié (plus robuste)"""
        if len(self.data_window) < 10:
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=value,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method="modified_zscore",
                context={'insufficient_data': True}
            )
        
        median = self.stats_cache['median']
        mad = self.stats_cache['mad']
        
        # Z-Score modifié utilisant la médiane et MAD
        if mad == 0:
            modified_z = 0
        else:
            modified_z = 0.6745 * (value - median) / mad
        
        modified_z = abs(modified_z)
        
        # Seuil pour Z-Score modifié (généralement 3.5)
        threshold = 3.5 * self.config.sensitivity
        is_anomaly = modified_z > threshold
        
        # P-value approximative
        p_value = 2 * (1 - stats.norm.cdf(modified_z))
        
        confidence = min(modified_z / threshold, 1.0) if threshold > 0 else 0.0
        
        return StatisticalResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            statistic_value=modified_z,
            p_value=p_value,
            threshold_used=threshold,
            method="modified_zscore",
            context={
                'median': median,
                'mad': mad,
                'raw_value': value
            }
        )
    
    async def _detect_iqr(self, value: float) -> StatisticalResult:
        """Détection basée sur l'écart interquartile (IQR)"""
        if len(self.data_window) < 10:
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=value,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method="iqr",
                context={'insufficient_data': True}
            )
        
        q1 = self.stats_cache['q1']
        q3 = self.stats_cache['q3']
        iqr = self.stats_cache['iqr']
        
        # Bornes IQR avec facteur de sensibilité
        factor = 1.5 * self.config.sensitivity
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        is_anomaly = value < lower_bound or value > upper_bound
        
        # Distance normalisée aux bornes
        if value < lower_bound:
            distance = abs(value - lower_bound) / iqr if iqr > 0 else 0
        elif value > upper_bound:
            distance = abs(value - upper_bound) / iqr if iqr > 0 else 0
        else:
            distance = 0
        
        confidence = min(distance / factor, 1.0) if factor > 0 else 0.0
        
        return StatisticalResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            statistic_value=distance,
            p_value=0.1 if is_anomaly else 0.9,  # Approximation
            threshold_used=factor,
            method="iqr",
            context={
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'raw_value': value
            }
        )
    
    async def _detect_mad(self, value: float) -> StatisticalResult:
        """Détection basée sur la déviation absolue médiane (MAD)"""
        if len(self.data_window) < 10:
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=value,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method="mad",
                context={'insufficient_data': True}
            )
        
        median = self.stats_cache['median']
        mad = self.stats_cache['mad']
        
        if mad == 0:
            mad_score = 0
        else:
            mad_score = abs(value - median) / mad
        
        threshold = 3.0 * self.config.sensitivity  # Seuil MAD standard
        is_anomaly = mad_score > threshold
        
        confidence = min(mad_score / threshold, 1.0) if threshold > 0 else 0.0
        
        return StatisticalResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            statistic_value=mad_score,
            p_value=0.05 if is_anomaly else 0.95,  # Approximation
            threshold_used=threshold,
            method="mad",
            context={
                'median': median,
                'mad': mad,
                'raw_value': value
            }
        )
    
    async def _detect_percentile(self, value: float) -> StatisticalResult:
        """Détection basée sur les percentiles"""
        if len(self.data_window) < 20:
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=value,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method="percentile",
                context={'insufficient_data': True}
            )
        
        values = [item[0] for item in self.data_window]
        
        # Percentiles adaptatifs basés sur la sensibilité
        lower_percentile = 5 / self.config.sensitivity
        upper_percentile = 100 - lower_percentile
        
        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)
        
        is_anomaly = value < lower_bound or value > upper_bound
        
        # Calcul du percentile de la valeur
        percentile_rank = stats.percentileofscore(values, value)
        
        if percentile_rank < lower_percentile:
            confidence = (lower_percentile - percentile_rank) / lower_percentile
        elif percentile_rank > upper_percentile:
            confidence = (percentile_rank - upper_percentile) / (100 - upper_percentile)
        else:
            confidence = 0.0
        
        return StatisticalResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            statistic_value=percentile_rank,
            p_value=min(percentile_rank, 100 - percentile_rank) / 100,
            threshold_used=self.config.sensitivity,
            method="percentile",
            context={
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'percentile_rank': percentile_rank,
                'raw_value': value
            }
        )
    
    async def _detect_grubbs(self, value: float) -> StatisticalResult:
        """Test de Grubbs pour détection d'outliers"""
        if len(self.data_window) < 30:
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=value,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method="grubbs",
                context={'insufficient_data': True}
            )
        
        values = [item[0] for item in self.data_window]
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return StatisticalResult(
                is_anomaly=False,
                confidence=0.0,
                statistic_value=0.0,
                p_value=1.0,
                threshold_used=self.current_threshold,
                method="grubbs",
                context={'zero_variance': True}
            )
        
        # Statistique de Grubbs
        grubbs_stat = abs(value - mean) / std
        
        # Valeur critique de Grubbs
        alpha = 0.05 / self.config.sensitivity
        t_critical = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        grubbs_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
        
        is_anomaly = grubbs_stat > grubbs_critical
        confidence = min(grubbs_stat / grubbs_critical, 1.0) if grubbs_critical > 0 else 0.0
        
        return StatisticalResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            statistic_value=grubbs_stat,
            p_value=alpha if is_anomaly else 1 - alpha,
            threshold_used=grubbs_critical,
            method="grubbs",
            context={
                'mean': mean,
                'std': std,
                'n_samples': n,
                'raw_value': value
            }
        )
    
    def _in_cooldown(self) -> bool:
        """Vérifie si le détecteur est en période de cooldown"""
        if self.last_alert_time is None:
            return False
        
        time_diff = (datetime.now() - self.last_alert_time).total_seconds()
        return time_diff < self.config.cooldown_period
    
    async def _adapt_threshold(self, value: float, result: StatisticalResult):
        """Adapte le seuil basé sur les performances"""
        # Calcul du nouveau seuil basé sur le taux d'adaptation
        adaptation_factor = 1.0 + self.config.adaptation_rate * result.confidence
        
        if result.confidence > 0.8:  # Anomalie très probable
            # Légèrement augmenter le seuil pour réduire les faux positifs
            new_threshold = self.current_threshold * (1 + 0.1 * self.config.adaptation_rate)
        else:
            # Légèrement diminuer le seuil pour capturer plus d'anomalies
            new_threshold = self.current_threshold * (1 - 0.05 * self.config.adaptation_rate)
        
        # Appliquer les limites
        new_threshold = max(self.config.min_threshold, 
                           min(self.config.max_threshold, new_threshold))
        
        self.current_threshold = new_threshold
        
        # Enregistrer dans l'historique
        try:
            history_entry = ThresholdHistory(
                metric_name=self.config.metric_name,
                threshold_value=new_threshold,
                threshold_type=self.config.statistical_method,
                timestamp=datetime.now()
            )
            self.session.add(history_entry)
            self.session.commit()
        except Exception as e:
            logger.error(f"Erreur enregistrement historique: {e}")
            self.session.rollback()
    
    async def _update_performance_metrics(self, result: StatisticalResult):
        """Met à jour les métriques de performance"""
        current_time = datetime.now()
        
        # Stocker les métriques
        metrics_key = f"performance:{self.config.metric_name}:{current_time.hour}"
        
        metric_data = {
            'timestamp': current_time.isoformat(),
            'is_anomaly': result.is_anomaly,
            'confidence': result.confidence,
            'method': result.method,
            'threshold': result.threshold_used
        }
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.lpush,
                metrics_key,
                json.dumps(metric_data)
            )
            
            # Conserver seulement les 1000 dernières entrées
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.ltrim,
                metrics_key,
                0, 999
            )
        
        except Exception as e:
            logger.warning(f"Erreur mise à jour métriques: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance du détecteur"""
        try:
            # Récupérer les métriques récentes
            current_hour = datetime.now().hour
            metrics_keys = [f"performance:{self.config.metric_name}:{h}" 
                           for h in range(max(0, current_hour - 24), current_hour + 1)]
            
            all_metrics = []
            for key in metrics_keys:
                metrics = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.lrange, key, 0, -1
                )
                for metric in metrics:
                    try:
                        all_metrics.append(json.loads(metric))
                    except json.JSONDecodeError:
                        continue
            
            if not all_metrics:
                return {'status': 'no_data'}
            
            # Calculer les statistiques
            total_detections = len(all_metrics)
            anomaly_count = sum(1 for m in all_metrics if m['is_anomaly'])
            avg_confidence = np.mean([m['confidence'] for m in all_metrics])
            
            return {
                'total_detections': total_detections,
                'anomaly_count': anomaly_count,
                'anomaly_rate': anomaly_count / total_detections if total_detections > 0 else 0,
                'avg_confidence': avg_confidence,
                'current_threshold': self.current_threshold,
                'window_size': len(self.data_window),
                'method': self.config.statistical_method,
                'last_24h_summary': {
                    'peak_confidence': max([m['confidence'] for m in all_metrics], default=0),
                    'detection_frequency': anomaly_count / 24 if anomaly_count > 0 else 0
                }
            }
        
        except Exception as e:
            logger.error(f"Erreur génération rapport: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def reset_threshold(self):
        """Remet le seuil à sa valeur initiale"""
        self.current_threshold = self.config.initial_threshold
        logger.info(f"Seuil remis à zéro pour {self.config.metric_name}")
    
    def __del__(self):
        """Nettoyage des ressources"""
        try:
            self.session.close()
        except:
            pass

# Factory pour créer des détecteurs de seuils
class ThresholdDetectorFactory:
    """Factory pour créer des détecteurs de seuils spécialisés"""
    
    @staticmethod
    def create_cpu_detector() -> AdaptiveThresholdDetector:
        """Détecteur pour CPU"""
        config = ThresholdConfig(
            metric_name="cpu_usage",
            initial_threshold=2.0,  # Z-Score
            adaptation_rate=0.05,
            sensitivity=1.2,
            window_size=60,
            cooldown_period=180,
            statistical_method="zscore"
        )
        return AdaptiveThresholdDetector(config)
    
    @staticmethod
    def create_memory_detector() -> AdaptiveThresholdDetector:
        """Détecteur pour mémoire"""
        config = ThresholdConfig(
            metric_name="memory_usage",
            initial_threshold=3.0,
            adaptation_rate=0.03,
            sensitivity=1.5,
            window_size=100,
            cooldown_period=300,
            statistical_method="modified_zscore"
        )
        return AdaptiveThresholdDetector(config)
    
    @staticmethod
    def create_latency_detector() -> AdaptiveThresholdDetector:
        """Détecteur pour latence"""
        config = ThresholdConfig(
            metric_name="response_latency",
            initial_threshold=2.5,
            adaptation_rate=0.08,
            sensitivity=1.0,
            window_size=80,
            cooldown_period=120,
            statistical_method="iqr"
        )
        return AdaptiveThresholdDetector(config)
    
    @staticmethod
    def create_error_rate_detector() -> AdaptiveThresholdDetector:
        """Détecteur pour taux d'erreur"""
        config = ThresholdConfig(
            metric_name="error_rate",
            initial_threshold=3.5,
            adaptation_rate=0.12,
            sensitivity=0.8,
            window_size=50,
            cooldown_period=240,
            statistical_method="grubbs"
        )
        return AdaptiveThresholdDetector(config)
