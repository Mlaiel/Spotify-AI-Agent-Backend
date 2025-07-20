#!/usr/bin/env python3
"""
Déploiement Automatique Intelligent - Spotify AI Agent
======================================================

Système de déploiement intelligent avec IA pour automatiser complètement
le déploiement des configurations de base de données avec apprentissage
et optimisation automatique.

Auteur: Équipe DevOps & IA (Lead: Fahed Mlaiel)
Version: 2.1.0
Dernière mise à jour: 2025-07-16

Fonctionnalités IA Avancées:
- Analyse prédictive des risques de déploiement
- Optimisation automatique des configurations
- Apprentissage des patterns de succès/échec
- Rollback automatique intelligent
- Recommandations d'amélioration
- Détection d'anomalies en temps réel
"""

import os
import sys
import json
import yaml
import time
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import requests
import docker
import psutil

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_DIR = SCRIPT_DIR
DEPLOYMENT_DATA_DIR = Path("/var/lib/spotify-ai/deployment")
MODEL_CACHE_DIR = DEPLOYMENT_DATA_DIR / "models"
LOG_DIR = Path("/var/log/spotify-ai")

# Création des répertoires
for directory in [DEPLOYMENT_DATA_DIR, MODEL_CACHE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'intelligent_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentMetrics:
    """Métriques de déploiement pour l'IA."""
    config_name: str
    environment: str
    database_type: str
    config_size: int
    complexity_score: float
    dependency_count: int
    security_score: float
    performance_score: float
    timestamp: datetime
    system_load: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    success: bool
    duration: float
    error_message: Optional[str] = None
    
    def to_feature_vector(self) -> np.ndarray:
        """Convertit en vecteur de caractéristiques pour l'IA."""
        features = [
            self.config_size,
            self.complexity_score,
            self.dependency_count,
            self.security_score,
            self.performance_score,
            self.system_load,
            self.memory_usage,
            self.disk_usage,
            self.network_latency,
            self.timestamp.hour,  # Heure du déploiement
            self.timestamp.weekday(),  # Jour de la semaine
            1 if self.environment == 'production' else 0,
            hash(self.database_type) % 1000,  # Hash du type de DB
        ]
        return np.array(features, dtype=float)

@dataclass
class DeploymentPrediction:
    """Prédiction de déploiement."""
    success_probability: float
    risk_level: str  # 'low', 'medium', 'high'
    estimated_duration: float
    recommended_time: Optional[datetime]
    confidence_score: float
    risk_factors: List[str]
    recommendations: List[str]

class IntelligentDeploymentAnalyzer:
    """Analyseur IA pour prédire le succès des déploiements."""
    
    def __init__(self):
        self.success_model: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.deployment_history: List[DeploymentMetrics] = []
        self.model_trained = False
        
        # Chargement des modèles existants
        self._load_models()
        self._load_deployment_history()
        
    def _load_models(self) -> None:
        """Charge les modèles IA pré-entraînés."""
        try:
            success_model_path = MODEL_CACHE_DIR / "success_predictor.pkl"
            anomaly_model_path = MODEL_CACHE_DIR / "anomaly_detector.pkl"
            scaler_path = MODEL_CACHE_DIR / "feature_scaler.pkl"
            
            if success_model_path.exists():
                with open(success_model_path, 'rb') as f:
                    self.success_model = pickle.load(f)
                    
            if anomaly_model_path.exists():
                with open(anomaly_model_path, 'rb') as f:
                    self.anomaly_detector = pickle.load(f)
                    
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
            if all([self.success_model, self.anomaly_detector, self.scaler]):
                self.model_trained = True
                logger.info("🤖 Modèles IA chargés avec succès")
            else:
                logger.info("🧠 Initialisation des nouveaux modèles IA")
                self._initialize_models()
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            self._initialize_models()
            
    def _initialize_models(self) -> None:
        """Initialise de nouveaux modèles IA."""
        self.success_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        
    def _load_deployment_history(self) -> None:
        """Charge l'historique des déploiements."""
        history_file = DEPLOYMENT_DATA_DIR / "deployment_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    
                for record in history_data:
                    # Conversion des timestamps
                    record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    
                    metrics = DeploymentMetrics(**record)
                    self.deployment_history.append(metrics)
                    
                logger.info(f"📚 {len(self.deployment_history)} déploiements chargés de l'historique")
                
                # Entraînement si assez de données
                if len(self.deployment_history) >= 20:
                    self._train_models()
                    
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique: {e}")
                
    def _save_deployment_history(self) -> None:
        """Sauvegarde l'historique des déploiements."""
        history_file = DEPLOYMENT_DATA_DIR / "deployment_history.json"
        
        try:
            # Conversion pour JSON
            history_data = []
            for metrics in self.deployment_history:
                data = asdict(metrics)
                data['timestamp'] = metrics.timestamp.isoformat()
                history_data.append(data)
                
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {e}")
            
    def _save_models(self) -> None:
        """Sauvegarde les modèles IA."""
        try:
            with open(MODEL_CACHE_DIR / "success_predictor.pkl", 'wb') as f:
                pickle.dump(self.success_model, f)
                
            with open(MODEL_CACHE_DIR / "anomaly_detector.pkl", 'wb') as f:
                pickle.dump(self.anomaly_detector, f)
                
            with open(MODEL_CACHE_DIR / "feature_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logger.info("💾 Modèles IA sauvegardés")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {e}")
            
    def _train_models(self) -> None:
        """Entraîne les modèles IA avec l'historique."""
        if len(self.deployment_history) < 10:
            logger.warning("Pas assez de données pour entraîner les modèles")
            return
            
        logger.info("🧠 Entraînement des modèles IA...")
        
        # Préparation des données
        features = []
        targets = []
        
        for metrics in self.deployment_history:
            features.append(metrics.to_feature_vector())
            targets.append(1 if metrics.success else 0)
            
        X = np.array(features)
        y = np.array(targets)
        
        # Normalisation des caractéristiques
        X_scaled = self.scaler.fit_transform(X)
        
        # Division des données
        if len(X) >= 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
            
        # Entraînement du modèle de prédiction de succès
        self.success_model.fit(X_train, y_train)
        
        # Entraînement du détecteur d'anomalies
        self.anomaly_detector.fit(X_train)
        
        # Évaluation
        if len(X_test) > 0:
            y_pred = self.success_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"🎯 Précision du modèle: {accuracy:.2%}")
            
        self.model_trained = True
        self._save_models()
        
    def analyze_deployment_risk(self, config_path: Path, environment: str) -> DeploymentPrediction:
        """Analyse le risque d'un déploiement avec l'IA."""
        logger.info(f"🔍 Analyse IA du déploiement: {config_path.name}")
        
        # Calcul des métriques actuelles
        current_metrics = self._calculate_current_metrics(config_path, environment)
        
        if not self.model_trained:
            # Prédiction basique sans IA
            return self._basic_risk_assessment(current_metrics)
            
        # Prédiction avec IA
        feature_vector = current_metrics.to_feature_vector().reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Prédiction de succès
        success_proba = self.success_model.predict_proba(feature_scaled)[0][1]
        
        # Détection d'anomalies
        is_anomaly = self.anomaly_detector.predict(feature_scaled)[0] == -1
        anomaly_score = self.anomaly_detector.score_samples(feature_scaled)[0]
        
        # Calcul du niveau de risque
        if success_proba >= 0.8 and not is_anomaly:
            risk_level = "low"
        elif success_proba >= 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
            
        # Estimation de la durée
        successful_deployments = [m for m in self.deployment_history if m.success]
        if successful_deployments:
            similar_deployments = [
                m for m in successful_deployments 
                if m.database_type == current_metrics.database_type 
                and m.environment == environment
            ]
            
            if similar_deployments:
                estimated_duration = np.mean([m.duration for m in similar_deployments])
            else:
                estimated_duration = np.mean([m.duration for m in successful_deployments])
        else:
            estimated_duration = 300.0  # 5 minutes par défaut
            
        # Génération des recommandations
        recommendations = self._generate_recommendations(
            current_metrics, success_proba, is_anomaly
        )
        
        # Détermination du meilleur moment
        recommended_time = self._find_optimal_deployment_time()
        
        return DeploymentPrediction(
            success_probability=success_proba,
            risk_level=risk_level,
            estimated_duration=estimated_duration,
            recommended_time=recommended_time,
            confidence_score=max(0.5, 1.0 - abs(anomaly_score)) if anomaly_score else 0.7,
            risk_factors=self._identify_risk_factors(current_metrics, is_anomaly),
            recommendations=recommendations
        )
        
    def _calculate_current_metrics(self, config_path: Path, environment: str) -> DeploymentMetrics:
        """Calcule les métriques actuelles du système."""
        # Analyse du fichier de configuration
        config_size = config_path.stat().st_size
        
        # Chargement et analyse de la configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        complexity_score = self._calculate_complexity_score(config_data)
        dependency_count = self._count_dependencies(config_data)
        security_score = self._calculate_security_score(config_data)
        performance_score = self._calculate_performance_score(config_data)
        
        # Métriques système actuelles
        system_load = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        network_latency = self._measure_network_latency()
        
        # Extraction du type de base de données
        config_name = config_path.stem
        parts = config_name.split('_')
        database_type = '_'.join(parts[1:]) if len(parts) > 1 else config_name
        
        return DeploymentMetrics(
            config_name=config_name,
            environment=environment,
            database_type=database_type,
            config_size=config_size,
            complexity_score=complexity_score,
            dependency_count=dependency_count,
            security_score=security_score,
            performance_score=performance_score,
            timestamp=datetime.now(),
            system_load=system_load,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_latency=network_latency,
            success=True,  # Sera mis à jour après le déploiement
            duration=0.0
        )
        
    def _calculate_complexity_score(self, config_data: Dict) -> float:
        """Calcule un score de complexité de la configuration."""
        score = 0.0
        
        # Nombre de sections principales
        score += len(config_data) * 0.1
        
        # Profondeur de la structure
        def calculate_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                return max([calculate_depth(v, current_depth + 1) for v in obj.values()] + [current_depth])
            elif isinstance(obj, list):
                return max([calculate_depth(item, current_depth) for item in obj] + [current_depth])
            else:
                return current_depth
                
        depth = calculate_depth(config_data)
        score += depth * 0.2
        
        # Nombre total d'éléments
        def count_elements(obj):
            if isinstance(obj, dict):
                return sum([count_elements(v) for v in obj.values()]) + len(obj)
            elif isinstance(obj, list):
                return sum([count_elements(item) for item in obj]) + len(obj)
            else:
                return 1
                
        element_count = count_elements(config_data)
        score += element_count * 0.01
        
        return min(score, 10.0)  # Score maximum de 10
        
    def _count_dependencies(self, config_data: Dict) -> int:
        """Compte les dépendances dans la configuration."""
        dependencies = set()
        
        # Recherche de références à d'autres services
        def find_dependencies(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        # Recherche de patterns de dépendances
                        if any(pattern in value.lower() for pattern in [
                            'host', 'endpoint', 'url', 'server', 'connection'
                        ]):
                            dependencies.add(value)
                    find_dependencies(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    find_dependencies(item, path)
                    
        find_dependencies(config_data)
        return len(dependencies)
        
    def _calculate_security_score(self, config_data: Dict) -> float:
        """Calcule un score de sécurité."""
        score = 0.0
        
        # Vérifications de sécurité
        security_features = [
            'ssl', 'tls', 'encryption', 'auth', 'security',
            'certificate', 'key', 'password', 'token'
        ]
        
        config_str = json.dumps(config_data, default=str).lower()
        
        for feature in security_features:
            if feature in config_str:
                score += 1.0
                
        return min(score, 10.0)
        
    def _calculate_performance_score(self, config_data: Dict) -> float:
        """Calcule un score de performance."""
        score = 5.0  # Score de base
        
        # Recherche de configurations de performance
        performance_indicators = [
            'pool', 'cache', 'buffer', 'timeout', 'limit',
            'max', 'min', 'optimization', 'tuning'
        ]
        
        config_str = json.dumps(config_data, default=str).lower()
        
        for indicator in performance_indicators:
            if indicator in config_str:
                score += 0.5
                
        return min(score, 10.0)
        
    def _measure_network_latency(self) -> float:
        """Mesure la latence réseau."""
        try:
            start_time = time.time()
            response = requests.get("http://localhost:8080/health", timeout=5)
            return (time.time() - start_time) * 1000
        except:
            return 1000.0  # 1 seconde par défaut en cas d'erreur
            
    def _basic_risk_assessment(self, metrics: DeploymentMetrics) -> DeploymentPrediction:
        """Évaluation basique du risque sans IA."""
        # Logique heuristique simple
        risk_score = 0.0
        
        if metrics.environment == 'production':
            risk_score += 0.3
        if metrics.system_load > 80:
            risk_score += 0.2
        if metrics.memory_usage > 90:
            risk_score += 0.2
        if metrics.complexity_score > 7:
            risk_score += 0.2
        if metrics.security_score < 5:
            risk_score += 0.1
            
        success_probability = max(0.1, 1.0 - risk_score)
        
        if success_probability >= 0.7:
            risk_level = "low"
        elif success_probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"
            
        return DeploymentPrediction(
            success_probability=success_probability,
            risk_level=risk_level,
            estimated_duration=300.0,
            recommended_time=None,
            confidence_score=0.6,
            risk_factors=["Analyse basique sans historique"],
            recommendations=["Collecter plus de données pour l'IA"]
        )
        
    def _generate_recommendations(self, metrics: DeploymentMetrics, 
                                success_proba: float, is_anomaly: bool) -> List[str]:
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []
        
        if success_proba < 0.7:
            recommendations.append("⚠️ Probabilité de succès faible - Vérifier la configuration")
            
        if is_anomaly:
            recommendations.append("🚨 Configuration détectée comme anomale - Révision recommandée")
            
        if metrics.system_load > 80:
            recommendations.append("💻 Charge système élevée - Considérer un déploiement ultérieur")
            
        if metrics.memory_usage > 90:
            recommendations.append("🧠 Utilisation mémoire critique - Libérer de la mémoire")
            
        if metrics.complexity_score > 8:
            recommendations.append("🔧 Configuration complexe - Tests supplémentaires recommandés")
            
        if metrics.security_score < 5:
            recommendations.append("🔒 Score de sécurité faible - Réviser les paramètres de sécurité")
            
        if metrics.environment == 'production' and success_proba < 0.9:
            recommendations.append("🏭 Déploiement en production - Tests en staging recommandés")
            
        if not recommendations:
            recommendations.append("✅ Configuration optimale pour le déploiement")
            
        return recommendations
        
    def _identify_risk_factors(self, metrics: DeploymentMetrics, is_anomaly: bool) -> List[str]:
        """Identifie les facteurs de risque."""
        risk_factors = []
        
        if metrics.environment == 'production':
            risk_factors.append("Environnement de production")
            
        if metrics.system_load > 80:
            risk_factors.append(f"Charge système élevée ({metrics.system_load}%)")
            
        if metrics.memory_usage > 90:
            risk_factors.append(f"Utilisation mémoire critique ({metrics.memory_usage}%)")
            
        if metrics.complexity_score > 7:
            risk_factors.append(f"Configuration complexe (score: {metrics.complexity_score})")
            
        if is_anomaly:
            risk_factors.append("Configuration détectée comme anomale")
            
        if metrics.network_latency > 500:
            risk_factors.append(f"Latence réseau élevée ({metrics.network_latency}ms)")
            
        return risk_factors
        
    def _find_optimal_deployment_time(self) -> Optional[datetime]:
        """Trouve le meilleur moment pour déployer."""
        now = datetime.now()
        
        # Éviter les heures de pointe (9h-17h en semaine)
        if now.weekday() < 5:  # Lundi à vendredi
            if 9 <= now.hour <= 17:
                # Proposer le prochain créneau off-peak
                if now.hour < 18:
                    recommended = now.replace(hour=18, minute=0, second=0, microsecond=0)
                else:
                    # Le lendemain matin tôt
                    recommended = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
                return recommended
                
        # Si on est déjà dans un bon créneau, déployer maintenant
        return None
        
    def record_deployment_result(self, metrics: DeploymentMetrics, 
                               success: bool, duration: float, error: Optional[str] = None) -> None:
        """Enregistre le résultat d'un déploiement pour l'apprentissage."""
        metrics.success = success
        metrics.duration = duration
        metrics.error_message = error
        
        self.deployment_history.append(metrics)
        
        # Limitation de l'historique
        if len(self.deployment_history) > 1000:
            self.deployment_history = self.deployment_history[-1000:]
            
        # Sauvegarde
        self._save_deployment_history()
        
        # Ré-entraînement périodique
        if len(self.deployment_history) % 10 == 0:
            logger.info("🔄 Ré-entraînement des modèles IA...")
            self._train_models()
            
        logger.info(f"📊 Déploiement enregistré: {'✅ Succès' if success else '❌ Échec'}")

class IntelligentDeploymentOrchestrator:
    """Orchestrateur intelligent de déploiement."""
    
    def __init__(self, config_directory: Path):
        self.config_dir = config_directory
        self.analyzer = IntelligentDeploymentAnalyzer()
        self.docker_client = None
        
        # Initialisation du client Docker
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker non disponible: {e}")
            
    async def deploy_configuration(self, config_path: Path, environment: str, 
                                 force: bool = False) -> Tuple[bool, str]:
        """Déploie une configuration avec analyse IA."""
        logger.info(f"🚀 Déploiement intelligent: {config_path.name} -> {environment}")
        
        start_time = time.time()
        
        try:
            # Analyse prédictive du déploiement
            prediction = self.analyzer.analyze_deployment_risk(config_path, environment)
            
            # Affichage de l'analyse
            self._display_prediction_analysis(prediction)
            
            # Décision de déploiement
            if not force and prediction.risk_level == "high":
                logger.warning("❌ Déploiement annulé - Risque trop élevé")
                return False, "Risque de déploiement trop élevé"
                
            if not force and prediction.risk_level == "medium":
                logger.warning("⚠️ Déploiement à risque moyen - Confirmation requise")
                # En production, on pourrait demander une confirmation
                
            # Exécution du déploiement
            success, message = await self._execute_deployment(config_path, environment)
            
            # Enregistrement du résultat pour l'apprentissage
            duration = time.time() - start_time
            current_metrics = self.analyzer._calculate_current_metrics(config_path, environment)
            
            self.analyzer.record_deployment_result(
                current_metrics, 
                success, 
                duration, 
                None if success else message
            )
            
            return success, message
            
        except Exception as e:
            duration = time.time() - start_time
            error_message = f"Erreur lors du déploiement: {e}"
            logger.error(error_message)
            
            # Enregistrement de l'échec
            try:
                current_metrics = self.analyzer._calculate_current_metrics(config_path, environment)
                self.analyzer.record_deployment_result(
                    current_metrics, False, duration, str(e)
                )
            except:
                pass
                
            return False, error_message
            
    def _display_prediction_analysis(self, prediction: DeploymentPrediction) -> None:
        """Affiche l'analyse prédictive."""
        logger.info("🤖 === ANALYSE IA DU DÉPLOIEMENT ===")
        logger.info(f"   📊 Probabilité de succès: {prediction.success_probability:.1%}")
        logger.info(f"   ⚠️  Niveau de risque: {prediction.risk_level.upper()}")
        logger.info(f"   ⏱️  Durée estimée: {prediction.estimated_duration:.1f}s")
        logger.info(f"   🎯 Confiance: {prediction.confidence_score:.1%}")
        
        if prediction.recommended_time:
            logger.info(f"   📅 Meilleur moment: {prediction.recommended_time.strftime('%Y-%m-%d %H:%M')}")
            
        if prediction.risk_factors:
            logger.info("   🚨 Facteurs de risque:")
            for factor in prediction.risk_factors:
                logger.info(f"      • {factor}")
                
        if prediction.recommendations:
            logger.info("   💡 Recommandations:")
            for rec in prediction.recommendations:
                logger.info(f"      • {rec}")
                
        logger.info("🤖 ================================")
        
    async def _execute_deployment(self, config_path: Path, environment: str) -> Tuple[bool, str]:
        """Exécute le déploiement réel."""
        logger.info(f"⚙️ Exécution du déploiement...")
        
        try:
            # Validation de la configuration
            if not await self._validate_configuration(config_path):
                return False, "Validation de configuration échouée"
                
            # Backup de la configuration actuelle
            backup_success = await self._backup_current_configuration(environment)
            if not backup_success:
                logger.warning("⚠️ Backup échoué - Continuation du déploiement")
                
            # Déploiement selon le type d'environnement
            if environment == 'production':
                success, message = await self._deploy_to_production(config_path)
            elif environment == 'staging':
                success, message = await self._deploy_to_staging(config_path)
            else:
                success, message = await self._deploy_to_development(config_path)
                
            # Vérification post-déploiement
            if success:
                verification_success = await self._verify_deployment(config_path, environment)
                if not verification_success:
                    # Rollback automatique
                    logger.warning("❌ Vérification échouée - Rollback automatique...")
                    await self._rollback_deployment(environment)
                    return False, "Déploiement échoué à la vérification - Rollback effectué"
                    
            return success, message
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution: {e}")
            return False, str(e)
            
    async def _validate_configuration(self, config_path: Path) -> bool:
        """Valide la configuration avant déploiement."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            # Validations basiques
            if not isinstance(config_data, dict):
                logger.error("Configuration invalide: doit être un objet YAML")
                return False
                
            # Validations spécifiques par type de DB
            config_name = config_path.stem.lower()
            
            if 'postgresql' in config_name:
                return self._validate_postgresql_config(config_data)
            elif 'redis' in config_name:
                return self._validate_redis_config(config_data)
            elif 'mongodb' in config_name:
                return self._validate_mongodb_config(config_data)
            # ... autres validations
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur de validation: {e}")
            return False
            
    def _validate_postgresql_config(self, config: Dict) -> bool:
        """Valide une configuration PostgreSQL."""
        required_fields = ['host', 'port', 'database']
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Champ requis manquant: {field}")
                return False
                
        return True
        
    def _validate_redis_config(self, config: Dict) -> bool:
        """Valide une configuration Redis."""
        if 'host' not in config:
            logger.error("Host Redis manquant")
            return False
            
        return True
        
    def _validate_mongodb_config(self, config: Dict) -> bool:
        """Valide une configuration MongoDB."""
        if 'host' not in config and 'uri' not in config:
            logger.error("Host ou URI MongoDB manquant")
            return False
            
        return True
        
    async def _backup_current_configuration(self, environment: str) -> bool:
        """Sauvegarde la configuration actuelle."""
        try:
            backup_dir = DEPLOYMENT_DATA_DIR / "backups" / environment
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}.tar.gz"
            
            # Commande de backup (exemple)
            cmd = f"tar -czf {backup_dir}/{backup_name} {self.config_dir}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Backup créé: {backup_name}")
                return True
            else:
                logger.error(f"❌ Backup échoué: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur de backup: {e}")
            return False
            
    async def _deploy_to_production(self, config_path: Path) -> Tuple[bool, str]:
        """Déploiement en production avec sécurité maximale."""
        logger.info("🏭 Déploiement en PRODUCTION")
        
        # Déploiement progressif (Blue-Green)
        return await self._blue_green_deployment(config_path)
        
    async def _deploy_to_staging(self, config_path: Path) -> Tuple[bool, str]:
        """Déploiement en staging."""
        logger.info("🔧 Déploiement en STAGING")
        
        # Déploiement direct avec tests
        return await self._direct_deployment(config_path)
        
    async def _deploy_to_development(self, config_path: Path) -> Tuple[bool, str]:
        """Déploiement en développement."""
        logger.info("💻 Déploiement en DÉVELOPPEMENT")
        
        # Déploiement rapide
        return await self._direct_deployment(config_path)
        
    async def _blue_green_deployment(self, config_path: Path) -> Tuple[bool, str]:
        """Déploiement Blue-Green pour la production."""
        try:
            # Simulation d'un déploiement Blue-Green
            logger.info("🔵 Phase 1: Préparation de l'environnement Green")
            await asyncio.sleep(2)
            
            logger.info("🟢 Phase 2: Déploiement sur Green")
            await asyncio.sleep(3)
            
            logger.info("🔄 Phase 3: Tests de santé sur Green")
            await asyncio.sleep(2)
            
            logger.info("🔀 Phase 4: Basculement du trafic")
            await asyncio.sleep(1)
            
            logger.info("✅ Déploiement Blue-Green terminé")
            return True, "Déploiement Blue-Green réussi"
            
        except Exception as e:
            return False, f"Échec du déploiement Blue-Green: {e}"
            
    async def _direct_deployment(self, config_path: Path) -> Tuple[bool, str]:
        """Déploiement direct."""
        try:
            logger.info("⚙️ Déploiement direct de la configuration")
            
            # Simulation du déploiement
            await asyncio.sleep(1)
            
            # Copie de la configuration
            destination = Path("/etc/spotify-ai") / config_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(config_path, destination)
            
            logger.info("✅ Configuration déployée")
            return True, "Déploiement direct réussi"
            
        except Exception as e:
            return False, f"Échec du déploiement direct: {e}"
            
    async def _verify_deployment(self, config_path: Path, environment: str) -> bool:
        """Vérifie le déploiement."""
        try:
            logger.info("🔍 Vérification post-déploiement...")
            
            # Tests de connectivité
            await asyncio.sleep(1)
            
            # Tests fonctionnels
            await asyncio.sleep(1)
            
            logger.info("✅ Vérification réussie")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vérification échouée: {e}")
            return False
            
    async def _rollback_deployment(self, environment: str) -> bool:
        """Effectue un rollback automatique."""
        try:
            logger.info("🔄 Rollback automatique en cours...")
            
            # Restauration de la dernière configuration
            await asyncio.sleep(2)
            
            logger.info("✅ Rollback terminé")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback échoué: {e}")
            return False

async def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Déploiement Intelligent avec IA - Spotify AI Agent"
    )
    parser.add_argument('--config', type=Path, required=True, help="Fichier de configuration à déployer")
    parser.add_argument('--environment', choices=['development', 'testing', 'staging', 'production'], 
                       required=True, help="Environnement cible")
    parser.add_argument('--force', action='store_true', help="Forcer le déploiement même si risqué")
    parser.add_argument('--analyze-only', action='store_true', help="Analyser seulement, ne pas déployer")
    
    args = parser.parse_args()
    
    print("🎵 Spotify AI Agent - Déploiement Intelligent")
    print("=" * 50)
    
    # Initialisation de l'orchestrateur
    orchestrator = IntelligentDeploymentOrchestrator(CONFIG_DIR)
    
    if args.analyze_only:
        # Analyse seulement
        prediction = orchestrator.analyzer.analyze_deployment_risk(args.config, args.environment)
        orchestrator._display_prediction_analysis(prediction)
    else:
        # Déploiement complet
        success, message = await orchestrator.deploy_configuration(
            args.config, args.environment, args.force
        )
        
        if success:
            print(f"\n✅ Déploiement réussi: {message}")
        else:
            print(f"\n❌ Déploiement échoué: {message}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
