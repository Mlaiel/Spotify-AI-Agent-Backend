"""
Contrôleur HPA (Horizontal Pod Autoscaler) avancé avec intelligence prédictive
Architecture microservices multi-tenant pour Spotify AI Agent
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from kubernetes import client, config as k8s_config
from prometheus_client.parser import text_string_to_metric_families
import aiohttp
import json
import math

from .config_manager import AutoscalingConfigManager, ScalingMode, TenantConfig
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class HPADecision:
    """Décision d'autoscaling HPA"""
    current_replicas: int
    desired_replicas: int
    scaling_reason: str
    confidence: float
    metrics_snapshot: Dict[str, float]
    timestamp: datetime

@dataclass
class PredictiveModel:
    """Modèle prédictif pour l'autoscaling"""
    service_name: str
    tenant_id: str
    historical_data: List[Tuple[datetime, Dict[str, float]]]
    prediction_horizon: int = 300  # 5 minutes
    accuracy_score: float = 0.0

class HorizontalPodAutoscaler:
    """Contrôleur HPA intelligent avec prédiction et optimisation multi-tenant"""
    
    def __init__(self, config_manager: AutoscalingConfigManager, metrics_collector: MetricsCollector):
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        self.k8s_apps_v1 = None
        self.k8s_autoscaling_v2 = None
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.scaling_history: Dict[str, List[HPADecision]] = {}
        self.cooldown_periods: Dict[str, datetime] = {}
        self._initialize_k8s_client()
    
    def _initialize_k8s_client(self):
        """Initialise le client Kubernetes"""
        try:
            k8s_config.load_incluster_config()
        except:
            try:
                k8s_config.load_kube_config()
            except Exception as e:
                logger.warning(f"Could not load Kubernetes config: {e}")
                return
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_autoscaling_v2 = client.AutoscalingV2Api()
    
    async def evaluate_scaling_decision(self, tenant_id: str, service_name: str, 
                                      namespace: str = "default") -> HPADecision:
        """Évalue une décision de scaling pour un service tenant"""
        
        # Récupération des métriques actuelles
        current_metrics = await self.metrics_collector.get_service_metrics(
            tenant_id, service_name, namespace
        )
        
        # Récupération de la configuration
        tenant_config = self.config_manager.get_tenant_config(tenant_id)
        scaling_policy = self.config_manager.get_scaling_policy(tenant_id, service_name)
        resource_limits = self.config_manager.get_resource_limits(tenant_id)
        
        # État actuel du déploiement
        current_replicas = await self._get_current_replicas(service_name, namespace)
        
        # Calcul du nombre de replicas désiré
        desired_replicas = self._calculate_desired_replicas(
            current_metrics, current_replicas, scaling_policy, resource_limits
        )
        
        # Application de l'intelligence prédictive
        if tenant_config and tenant_config.scaling_mode == ScalingMode.INTELLIGENT:
            predicted_replicas = await self._apply_predictive_scaling(
                tenant_id, service_name, current_metrics, desired_replicas
            )
            desired_replicas = predicted_replicas
        
        # Validation des contraintes
        desired_replicas = self._validate_scaling_constraints(
            tenant_id, service_name, current_replicas, desired_replicas, resource_limits
        )
        
        # Création de la décision
        scaling_reason = self._generate_scaling_reason(
            current_metrics, current_replicas, desired_replicas, scaling_policy
        )
        
        confidence = self._calculate_confidence(
            current_metrics, tenant_id, service_name
        )
        
        decision = HPADecision(
            current_replicas=current_replicas,
            desired_replicas=desired_replicas,
            scaling_reason=scaling_reason,
            confidence=confidence,
            metrics_snapshot=current_metrics,
            timestamp=datetime.utcnow()
        )
        
        # Enregistrement dans l'historique
        service_key = f"{tenant_id}:{service_name}"
        if service_key not in self.scaling_history:
            self.scaling_history[service_key] = []
        self.scaling_history[service_key].append(decision)
        
        # Nettoyage de l'historique (garde 24h)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.scaling_history[service_key] = [
            d for d in self.scaling_history[service_key] 
            if d.timestamp > cutoff_time
        ]
        
        return decision
    
    def _calculate_desired_replicas(self, metrics: Dict[str, float], current_replicas: int,
                                  policy, limits) -> int:
        """Calcule le nombre de replicas désiré basé sur les métriques"""
        
        cpu_utilization = metrics.get('cpu_utilization_percentage', 0)
        memory_utilization = metrics.get('memory_utilization_percentage', 0)
        request_rate = metrics.get('requests_per_second', 0)
        latency = metrics.get('avg_response_time_ms', 0)
        
        # Calcul basé sur CPU
        cpu_ratio = cpu_utilization / policy.target_cpu_utilization
        cpu_desired = math.ceil(current_replicas * cpu_ratio)
        
        # Calcul basé sur mémoire
        memory_ratio = memory_utilization / policy.target_memory_utilization
        memory_desired = math.ceil(current_replicas * memory_ratio)
        
        # Calcul basé sur le taux de requêtes (si applicable)
        request_desired = current_replicas
        if request_rate > 0:
            optimal_rps_per_pod = 100  # Configurable
            request_desired = math.ceil(request_rate / optimal_rps_per_pod)
        
        # Calcul basé sur la latence
        latency_desired = current_replicas
        if latency > 1000:  # Si latence > 1s
            latency_multiplier = min(latency / 500, 3.0)  # Max 3x scaling
            latency_desired = math.ceil(current_replicas * latency_multiplier)
        
        # Prendre le maximum des calculs
        desired_replicas = max(cpu_desired, memory_desired, request_desired, latency_desired)
        
        # Application des limites
        desired_replicas = max(limits.min_replicas, min(desired_replicas, limits.max_replicas))
        
        return desired_replicas
    
    async def _apply_predictive_scaling(self, tenant_id: str, service_name: str,
                                      current_metrics: Dict[str, float], 
                                      base_desired: int) -> int:
        """Applique l'intelligence prédictive pour ajuster le scaling"""
        
        model_key = f"{tenant_id}:{service_name}"
        
        # Mise à jour du modèle prédictif
        await self._update_predictive_model(model_key, current_metrics)
        
        if model_key not in self.predictive_models:
            return base_desired
        
        model = self.predictive_models[model_key]
        
        # Prédiction de la charge future
        predicted_load = self._predict_future_load(model)
        
        if predicted_load is None:
            return base_desired
        
        # Ajustement basé sur la prédiction
        load_increase_factor = predicted_load / current_metrics.get('cpu_utilization_percentage', 50)
        
        if load_increase_factor > 1.2:  # Augmentation prévue > 20%
            # Scale up proactivement
            predicted_desired = math.ceil(base_desired * load_increase_factor * 0.8)
            logger.info(f"Predictive scaling UP for {model_key}: {base_desired} -> {predicted_desired}")
            return predicted_desired
        elif load_increase_factor < 0.8:  # Diminution prévue > 20%
            # Scale down progressivement
            predicted_desired = max(math.floor(base_desired * load_increase_factor * 1.1), 1)
            logger.info(f"Predictive scaling DOWN for {model_key}: {base_desired} -> {predicted_desired}")
            return predicted_desired
        
        return base_desired
    
    async def _update_predictive_model(self, model_key: str, metrics: Dict[str, float]):
        """Met à jour le modèle prédictif avec de nouvelles données"""
        
        if model_key not in self.predictive_models:
            tenant_id, service_name = model_key.split(':')
            self.predictive_models[model_key] = PredictiveModel(
                service_name=service_name,
                tenant_id=tenant_id,
                historical_data=[]
            )
        
        model = self.predictive_models[model_key]
        
        # Ajout des nouvelles données
        model.historical_data.append((datetime.utcnow(), metrics.copy()))
        
        # Nettoyage des données anciennes (garde 24h)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        model.historical_data = [
            (timestamp, data) for timestamp, data in model.historical_data
            if timestamp > cutoff_time
        ]
        
        # Recalcul de la précision du modèle
        if len(model.historical_data) > 10:
            model.accuracy_score = self._calculate_model_accuracy(model)
    
    def _predict_future_load(self, model: PredictiveModel) -> Optional[float]:
        """Prédit la charge future basée sur l'historique"""
        
        if len(model.historical_data) < 5:
            return None
        
        # Extraction des valeurs de CPU des 2 dernières heures
        recent_data = []
        cutoff_time = datetime.utcnow() - timedelta(hours=2)
        
        for timestamp, metrics in model.historical_data:
            if timestamp > cutoff_time:
                cpu_util = metrics.get('cpu_utilization_percentage', 0)
                recent_data.append(cpu_util)
        
        if len(recent_data) < 3:
            return None
        
        # Moyenne mobile exponentielle pour la prédiction
        alpha = 0.3  # Facteur de lissage
        prediction = recent_data[0]
        
        for value in recent_data[1:]:
            prediction = alpha * value + (1 - alpha) * prediction
        
        # Détection de tendance
        if len(recent_data) >= 5:
            recent_trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            prediction += recent_trend * model.prediction_horizon / 60  # Extrapolation
        
        return max(0, min(prediction, 100))  # Borné entre 0 et 100%
    
    def _calculate_model_accuracy(self, model: PredictiveModel) -> float:
        """Calcule la précision du modèle prédictif"""
        
        if len(model.historical_data) < 10:
            return 0.0
        
        errors = []
        
        # Validation croisée sur les 10 derniers points
        for i in range(5, min(len(model.historical_data), 15)):
            # Utilise les données jusqu'à i-1 pour prédire i
            training_data = [metrics['cpu_utilization_percentage'] 
                           for _, metrics in model.historical_data[:i]]
            actual_value = model.historical_data[i][1]['cpu_utilization_percentage']
            
            # Prédiction simple par moyenne mobile
            if len(training_data) >= 3:
                predicted_value = np.mean(training_data[-3:])
                error = abs(actual_value - predicted_value)
                errors.append(error)
        
        if not errors:
            return 0.0
        
        # Précision basée sur l'erreur moyenne
        mean_error = np.mean(errors)
        accuracy = max(0, 1 - (mean_error / 100))  # Normalisation
        
        return accuracy
    
    def _validate_scaling_constraints(self, tenant_id: str, service_name: str,
                                    current_replicas: int, desired_replicas: int,
                                    limits) -> int:
        """Valide et applique les contraintes de scaling"""
        
        service_key = f"{tenant_id}:{service_name}"
        
        # Vérification du cooldown
        if service_key in self.cooldown_periods:
            if datetime.utcnow() < self.cooldown_periods[service_key]:
                logger.info(f"Scaling blocked for {service_key} due to cooldown")
                return current_replicas
        
        # Vérification du budget de coût
        if not self.config_manager.is_scaling_allowed(tenant_id, 0):  # TODO: coût réel
            logger.warning(f"Scaling blocked for {tenant_id} due to budget constraints")
            return current_replicas
        
        # Application des limites strictes
        validated_replicas = max(limits.min_replicas, min(desired_replicas, limits.max_replicas))
        
        # Limitation du taux de changement
        max_change = max(1, current_replicas // 2)  # Max 50% de changement
        if validated_replicas > current_replicas:
            validated_replicas = min(validated_replicas, current_replicas + max_change)
        elif validated_replicas < current_replicas:
            validated_replicas = max(validated_replicas, current_replicas - max_change)
        
        return validated_replicas
    
    def _generate_scaling_reason(self, metrics: Dict[str, float], current: int, 
                               desired: int, policy) -> str:
        """Génère une explication pour la décision de scaling"""
        
        if desired == current:
            return "No scaling needed - metrics within target ranges"
        
        reasons = []
        
        cpu_util = metrics.get('cpu_utilization_percentage', 0)
        if cpu_util > policy.target_cpu_utilization * 1.1:
            reasons.append(f"High CPU utilization ({cpu_util:.1f}%)")
        elif cpu_util < policy.target_cpu_utilization * 0.7:
            reasons.append(f"Low CPU utilization ({cpu_util:.1f}%)")
        
        memory_util = metrics.get('memory_utilization_percentage', 0)
        if memory_util > policy.target_memory_utilization * 1.1:
            reasons.append(f"High memory utilization ({memory_util:.1f}%)")
        
        latency = metrics.get('avg_response_time_ms', 0)
        if latency > 1000:
            reasons.append(f"High latency ({latency:.0f}ms)")
        
        action = "Scale up" if desired > current else "Scale down"
        reason_str = ", ".join(reasons) if reasons else "Preventive scaling"
        
        return f"{action} from {current} to {desired} replicas. Reason: {reason_str}"
    
    def _calculate_confidence(self, metrics: Dict[str, float], tenant_id: str, 
                            service_name: str) -> float:
        """Calcule le niveau de confiance de la décision"""
        
        base_confidence = 0.7
        
        # Bonus pour les métriques complètes
        metric_completeness = len(metrics) / 6  # 6 métriques attendues
        confidence = base_confidence + (metric_completeness * 0.2)
        
        # Bonus pour l'historique
        model_key = f"{tenant_id}:{service_name}"
        if model_key in self.predictive_models:
            model_accuracy = self.predictive_models[model_key].accuracy_score
            confidence += model_accuracy * 0.1
        
        return min(1.0, confidence)
    
    async def _get_current_replicas(self, service_name: str, namespace: str) -> int:
        """Récupère le nombre actuel de replicas"""
        
        if not self.k8s_apps_v1:
            return 1  # Fallback
        
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=service_name, namespace=namespace
            )
            return deployment.spec.replicas or 1
        except Exception as e:
            logger.warning(f"Could not get current replicas for {service_name}: {e}")
            return 1
    
    async def apply_scaling_decision(self, decision: HPADecision, service_name: str,
                                   namespace: str = "default") -> bool:
        """Applique la décision de scaling"""
        
        if decision.current_replicas == decision.desired_replicas:
            return True
        
        if not self.k8s_apps_v1:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            # Mise à jour du deployment
            body = {'spec': {'replicas': decision.desired_replicas}}
            self.k8s_apps_v1.patch_namespaced_deployment_scale(
                name=service_name,
                namespace=namespace,
                body=body
            )
            
            # Mise à jour du cooldown
            service_key = f"{service_name}:{namespace}"
            cooldown_duration = 300 if decision.desired_replicas > decision.current_replicas else 600
            self.cooldown_periods[service_key] = datetime.utcnow() + timedelta(seconds=cooldown_duration)
            
            logger.info(f"Applied scaling decision: {decision.scaling_reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply scaling decision: {e}")
            return False
    
    async def get_scaling_recommendations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Génère des recommandations de scaling pour un tenant"""
        
        recommendations = []
        
        # TODO: Récupérer la liste des services du tenant
        services = ["api-service", "ml-service", "audio-processor"]  # Exemple
        
        for service in services:
            try:
                decision = await self.evaluate_scaling_decision(tenant_id, service)
                
                if decision.current_replicas != decision.desired_replicas:
                    recommendations.append({
                        'service': service,
                        'current_replicas': decision.current_replicas,
                        'recommended_replicas': decision.desired_replicas,
                        'reason': decision.scaling_reason,
                        'confidence': decision.confidence,
                        'metrics': decision.metrics_snapshot
                    })
                    
            except Exception as e:
                logger.error(f"Error generating recommendation for {service}: {e}")
        
        return recommendations
