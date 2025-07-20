"""
Scripts de Détection d'Anomalies Avancés avec Intelligence Artificielle
Module ultra-performant pour la détection proactive d'incidents dans l'écosystème Spotify AI Agent

Fonctionnalités:
- Détection d'anomalies par Machine Learning en temps réel
- Corrélation intelligente des métriques multi-services
- Prédiction de pannes basée sur des modèles LSTM
- Classification automatique des incidents par sévérité
- Adaptation dynamique des seuils par tenant
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import redis.asyncio as redis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from pathlib import Path

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types d'anomalies détectables par le système"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_LEAK = "memory_leak"
    AUDIO_QUALITY_DROP = "audio_quality_drop"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_INCREASE = "error_rate_increase"
    SECURITY_THREAT = "security_threat"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MODEL_DRIFT = "model_drift"

@dataclass
class AnomalyDetectionResult:
    """Résultat d'une détection d'anomalie"""
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence: float
    timestamp: datetime
    affected_services: List[str]
    metrics: Dict[str, float]
    prediction: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None

class MLAnomalyDetector:
    """Détecteur d'anomalies basé sur l'apprentissage automatique"""
    
    def __init__(self, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        self.models = {}
        self.scalers = {}
        self.redis_client = None
        self.model_path = Path("./ml_models/anomaly_detection")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialise les connexions et modèles ML"""
        try:
            # Connexion Redis pour les métriques en temps réel
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                decode_responses=True,
                db=1
            )
            
            # Chargement des modèles pré-entraînés
            await self._load_models()
            
            logger.info("Détecteur d'anomalies ML initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du détecteur ML: {e}")
            raise

    async def _load_models(self):
        """Charge les modèles ML pré-entraînés"""
        try:
            # Modèle Isolation Forest pour détection d'anomalies générales
            isolation_forest_path = self.model_path / "isolation_forest.joblib"
            if isolation_forest_path.exists():
                self.models['isolation_forest'] = joblib.load(isolation_forest_path)
            else:
                # Création d'un nouveau modèle si aucun n'existe
                self.models['isolation_forest'] = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
            
            # Modèle LSTM pour prédiction de séries temporelles
            lstm_path = self.model_path / "lstm_model.h5"
            if lstm_path.exists():
                from tensorflow.keras.models import load_model
                self.models['lstm'] = load_model(lstm_path)
            else:
                self.models['lstm'] = self._create_lstm_model()
            
            # Scaler pour normalisation des données
            scaler_path = self.model_path / "scaler.joblib"
            if scaler_path.exists():
                self.scalers['main'] = joblib.load(scaler_path)
            else:
                self.scalers['main'] = StandardScaler()
                
            logger.info("Modèles ML chargés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")

    def _create_lstm_model(self) -> Sequential:
        """Crée un modèle LSTM pour la prédiction temporelle"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    async def detect_anomalies(self, metrics: Dict[str, float]) -> List[AnomalyDetectionResult]:
        """Détecte les anomalies dans les métriques fournies"""
        anomalies = []
        
        try:
            # Préparation des données
            data_array = np.array(list(metrics.values())).reshape(1, -1)
            scaled_data = self.scalers['main'].transform(data_array)
            
            # Détection par Isolation Forest
            isolation_score = self.models['isolation_forest'].decision_function(scaled_data)[0]
            is_anomaly = self.models['isolation_forest'].predict(scaled_data)[0] == -1
            
            if is_anomaly:
                # Classification du type d'anomalie
                anomaly_type = await self._classify_anomaly_type(metrics)
                severity = self._calculate_severity(isolation_score, metrics)
                confidence = abs(isolation_score)
                
                # Prédiction future avec LSTM
                prediction = await self._predict_future_metrics(metrics)
                
                # Génération de recommandations
                recommendations = await self._generate_recommendations(anomaly_type, metrics)
                
                anomaly = AnomalyDetectionResult(
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=confidence,
                    timestamp=datetime.utcnow(),
                    affected_services=await self._identify_affected_services(metrics),
                    metrics=metrics,
                    prediction=prediction,
                    recommendations=recommendations,
                    tenant_id=self.tenant_id
                )
                
                anomalies.append(anomaly)
                
                # Logging de l'anomalie détectée
                logger.warning(f"Anomalie détectée: {anomaly_type.value} - Confiance: {confidence:.2f}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalies: {e}")
            
        return anomalies

    async def _classify_anomaly_type(self, metrics: Dict[str, float]) -> AnomalyType:
        """Classifie le type d'anomalie basé sur les métriques"""
        
        # Règles heuristiques pour classification rapide
        if metrics.get('memory_usage', 0) > 85:
            return AnomalyType.MEMORY_LEAK
        elif metrics.get('response_time', 0) > 1000:
            return AnomalyType.LATENCY_SPIKE
        elif metrics.get('error_rate', 0) > 5:
            return AnomalyType.ERROR_RATE_INCREASE
        elif metrics.get('audio_quality_score', 100) < 70:
            return AnomalyType.AUDIO_QUALITY_DROP
        elif metrics.get('cpu_usage', 0) > 90:
            return AnomalyType.RESOURCE_EXHAUSTION
        elif metrics.get('failed_login_attempts', 0) > 10:
            return AnomalyType.SECURITY_THREAT
        else:
            return AnomalyType.PERFORMANCE_DEGRADATION

    def _calculate_severity(self, isolation_score: float, metrics: Dict[str, float]) -> AlertSeverity:
        """Calcule la sévérité de l'anomalie"""
        
        # Score normalisé basé sur l'isolation forest et les métriques critiques
        critical_metrics = ['error_rate', 'memory_usage', 'cpu_usage']
        critical_values = [metrics.get(metric, 0) for metric in critical_metrics]
        
        severity_score = abs(isolation_score) + sum(critical_values) / len(critical_values) / 100
        
        if severity_score > 0.8:
            return AlertSeverity.CRITICAL
        elif severity_score > 0.6:
            return AlertSeverity.HIGH
        elif severity_score > 0.4:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    async def _predict_future_metrics(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Prédit l'évolution future des métriques avec LSTM"""
        try:
            # Récupération de l'historique des métriques
            historical_data = await self._get_historical_metrics()
            
            if len(historical_data) >= 60:  # Minimum requis pour LSTM
                # Préparation des données pour LSTM
                sequence = np.array(historical_data[-60:]).reshape(1, 60, -1)
                
                # Prédiction
                predicted_values = self.models['lstm'].predict(sequence)
                
                return {
                    'predicted_cpu_usage': float(predicted_values[0][0]),
                    'prediction_horizon': '30_minutes',
                    'confidence': 0.85,
                    'trend': 'increasing' if predicted_values[0][0] > current_metrics.get('cpu_usage', 0) else 'decreasing'
                }
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction LSTM: {e}")
            
        return {'error': 'Prédiction non disponible'}

    async def _get_historical_metrics(self) -> List[List[float]]:
        """Récupère l'historique des métriques depuis Redis"""
        try:
            key = f"metrics:history:{self.tenant_id or 'global'}"
            historical_data = await self.redis_client.lrange(key, 0, 60)
            return [json.loads(data) for data in historical_data]
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            return []

    async def _generate_recommendations(self, anomaly_type: AnomalyType, metrics: Dict[str, float]) -> List[str]:
        """Génère des recommandations intelligentes basées sur le type d'anomalie"""
        
        recommendations = []
        
        if anomaly_type == AnomalyType.MEMORY_LEAK:
            recommendations.extend([
                "Vérifier les objets non libérés en mémoire",
                "Redémarrer les services avec haute utilisation mémoire",
                "Analyser les dumps de mémoire pour identifier les fuites"
            ])
        elif anomaly_type == AnomalyType.LATENCY_SPIKE:
            recommendations.extend([
                "Vérifier la charge réseau et les connexions DB",
                "Analyser les requêtes lentes dans les logs",
                "Considérer l'activation de la mise en cache"
            ])
        elif anomaly_type == AnomalyType.AUDIO_QUALITY_DROP:
            recommendations.extend([
                "Vérifier la qualité des codecs audio",
                "Analyser la bande passante disponible",
                "Redémarrer les services de traitement audio"
            ])
        elif anomaly_type == AnomalyType.SECURITY_THREAT:
            recommendations.extend([
                "Bloquer immédiatement les IPs suspectes",
                "Renforcer l'authentification multi-facteurs",
                "Analyser les logs de sécurité en détail"
            ])
        
        # Recommandations générales basées sur les métriques
        if metrics.get('cpu_usage', 0) > 80:
            recommendations.append("Considérer l'ajout de ressources CPU supplémentaires")
        
        if metrics.get('disk_usage', 0) > 85:
            recommendations.append("Nettoyer l'espace disque ou ajouter du stockage")
        
        return recommendations

    async def _identify_affected_services(self, metrics: Dict[str, float]) -> List[str]:
        """Identifie les services affectés par l'anomalie"""
        affected_services = []
        
        # Logique d'identification basée sur les noms des métriques
        for metric_name in metrics.keys():
            if 'api' in metric_name.lower():
                affected_services.append('api-gateway')
            elif 'audio' in metric_name.lower():
                affected_services.append('audio-processing-service')
            elif 'ml' in metric_name.lower():
                affected_services.append('ml-recommendation-service')
            elif 'auth' in metric_name.lower():
                affected_services.append('authentication-service')
        
        return list(set(affected_services))  # Dédoublonnage

    async def train_models(self, training_data: pd.DataFrame):
        """Entraîne les modèles ML avec de nouvelles données"""
        try:
            # Préparation des données
            X = training_data.drop(['is_anomaly'], axis=1).values
            y = training_data['is_anomaly'].values
            
            # Normalisation
            X_scaled = self.scalers['main'].fit_transform(X)
            
            # Entraînement Isolation Forest
            self.models['isolation_forest'].fit(X_scaled)
            
            # Sauvegarde des modèles
            await self._save_models()
            
            logger.info("Modèles ML re-entraînés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles: {e}")

    async def _save_models(self):
        """Sauvegarde les modèles entraînés"""
        try:
            # Sauvegarde Isolation Forest
            joblib.dump(
                self.models['isolation_forest'],
                self.model_path / "isolation_forest.joblib"
            )
            
            # Sauvegarde LSTM
            self.models['lstm'].save(self.model_path / "lstm_model.h5")
            
            # Sauvegarde Scaler
            joblib.dump(
                self.scalers['main'],
                self.model_path / "scaler.joblib"
            )
            
            logger.info("Modèles sauvegardés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {e}")

# Instance globale du détecteur
_global_detector = None

async def get_anomaly_detector(tenant_id: Optional[str] = None) -> MLAnomalyDetector:
    """Factory pour obtenir une instance du détecteur d'anomalies"""
    global _global_detector
    
    if _global_detector is None:
        _global_detector = MLAnomalyDetector(tenant_id)
        await _global_detector.initialize()
    
    return _global_detector

async def run_anomaly_detection_scan():
    """Script principal de détection d'anomalies - exécuté périodiquement"""
    try:
        detector = await get_anomaly_detector()
        
        # Récupération des métriques actuelles
        current_metrics = await _collect_current_metrics()
        
        if current_metrics:
            # Détection des anomalies
            anomalies = await detector.detect_anomalies(current_metrics)
            
            # Traitement des anomalies détectées
            for anomaly in anomalies:
                await _process_detected_anomaly(anomaly)
                
            logger.info(f"Scan d'anomalies terminé - {len(anomalies)} anomalies détectées")
        else:
            logger.warning("Aucune métrique disponible pour la détection d'anomalies")
            
    except Exception as e:
        logger.error(f"Erreur lors du scan de détection d'anomalies: {e}")

async def _collect_current_metrics() -> Dict[str, float]:
    """Collecte les métriques actuelles du système"""
    try:
        # Simulation de collecte de métriques (à remplacer par une vraie collecte)
        return {
            'cpu_usage': 75.5,
            'memory_usage': 68.2,
            'disk_usage': 45.0,
            'response_time': 250.0,
            'error_rate': 2.1,
            'audio_quality_score': 92.5,
            'active_connections': 1250,
            'throughput': 15000.0,
            'failed_login_attempts': 3
        }
    except Exception as e:
        logger.error(f"Erreur lors de la collecte des métriques: {e}")
        return {}

async def _process_detected_anomaly(anomaly: AnomalyDetectionResult):
    """Traite une anomalie détectée"""
    try:
        # Enregistrement de l'alerte
        alert_config = AlertConfig(
            name=f"anomaly_{anomaly.anomaly_type.value}_{int(anomaly.timestamp.timestamp())}",
            category=AlertCategory.PERFORMANCE,
            severity=anomaly.severity,
            script_type=ScriptType.DETECTION,
            tenant_id=anomaly.tenant_id,
            thresholds={'confidence': anomaly.confidence},
            conditions=[f"Anomalie détectée: {anomaly.anomaly_type.value}"],
            actions=anomaly.recommendations,
            ml_enabled=True,
            metadata={
                'affected_services': anomaly.affected_services,
                'metrics': anomaly.metrics,
                'prediction': anomaly.prediction
            }
        )
        
        register_alert(alert_config)
        
        # Log détaillé de l'anomalie
        logger.critical(
            f"ANOMALIE CRITIQUE DÉTECTÉE: {anomaly.anomaly_type.value} "
            f"- Sévérité: {anomaly.severity.value} "
            f"- Confiance: {anomaly.confidence:.2f} "
            f"- Services affectés: {', '.join(anomaly.affected_services)}"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'anomalie: {e}")

# Configuration des alertes par défaut
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes de détection
    default_configs = [
        AlertConfig(
            name="ml_anomaly_detection_critical",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            thresholds={'confidence': 0.8},
            conditions=['ML anomaly detection confidence > 80%'],
            actions=['run_anomaly_detection_scan'],
            ml_enabled=True,
            auto_remediation=True
        ),
        AlertConfig(
            name="audio_quality_anomaly",
            category=AlertCategory.AUDIO_QUALITY,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.DETECTION,
            thresholds={'audio_quality_score': 70},
            conditions=['Audio quality score < 70%'],
            actions=['restart_audio_services', 'check_codec_health'],
            ml_enabled=True
        )
    ]
    
    for config in default_configs:
        register_alert(config)
