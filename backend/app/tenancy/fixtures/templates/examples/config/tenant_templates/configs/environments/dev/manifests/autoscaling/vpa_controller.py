"""
Contrôleur VPA (Vertical Pod Autoscaler) intelligent
Optimisation automatique des ressources CPU/Mémoire pour Spotify AI Agent
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import json
from kubernetes import client, config as k8s_config
from statistics import median, mean

from .config_manager import AutoscalingConfigManager, ScalingMode, ResourceLimits
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class ResourceRecommendation:
    """Recommandation de ressources VPA"""
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    confidence: float
    reasoning: str
    estimated_cost_impact: float
    timestamp: datetime

@dataclass
class ResourceUsagePattern:
    """Pattern d'utilisation des ressources"""
    service_name: str
    tenant_id: str
    cpu_percentiles: Dict[str, float]  # p50, p90, p95, p99
    memory_percentiles: Dict[str, float]
    peak_hours: List[int]  # Heures de pic
    baseline_usage: Dict[str, float]
    growth_trend: float

class VerticalPodAutoscaler:
    """Contrôleur VPA intelligent avec optimisation multi-tenant"""
    
    def __init__(self, config_manager: AutoscalingConfigManager, metrics_collector: MetricsCollector):
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        self.k8s_core_v1 = None
        self.k8s_apps_v1 = None
        self.usage_patterns: Dict[str, ResourceUsagePattern] = {}
        self.recommendation_history: Dict[str, List[ResourceRecommendation]] = {}
        self.cost_model = ResourceCostModel()
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
        
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_apps_v1 = client.AppsV1Api()
    
    async def analyze_resource_usage(self, tenant_id: str, service_name: str,
                                   analysis_period_hours: int = 24) -> ResourceUsagePattern:
        """Analyse les patterns d'utilisation des ressources"""
        
        # Collecte des métriques historiques
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=analysis_period_hours)
        
        historical_metrics = await self.metrics_collector.get_historical_metrics(
            tenant_id, service_name, start_time, end_time
        )
        
        if not historical_metrics:
            logger.warning(f"No historical data for {tenant_id}:{service_name}")
            return self._create_default_pattern(tenant_id, service_name)
        
        # Extraction des données CPU et mémoire
        cpu_values = []
        memory_values = []
        hourly_usage = {hour: {'cpu': [], 'memory': []} for hour in range(24)}
        
        for timestamp, metrics in historical_metrics:
            hour = timestamp.hour
            cpu_usage = metrics.get('cpu_usage_cores', 0)
            memory_usage = metrics.get('memory_usage_bytes', 0)
            
            cpu_values.append(cpu_usage)
            memory_values.append(memory_usage)
            hourly_usage[hour]['cpu'].append(cpu_usage)
            hourly_usage[hour]['memory'].append(memory_usage)
        
        # Calcul des percentiles
        cpu_percentiles = self._calculate_percentiles(cpu_values)
        memory_percentiles = self._calculate_percentiles(memory_values)
        
        # Identification des heures de pic
        peak_hours = self._identify_peak_hours(hourly_usage)
        
        # Calcul de l'usage baseline
        baseline_usage = {
            'cpu': cpu_percentiles['p50'],
            'memory': memory_percentiles['p50']
        }
        
        # Calcul de la tendance de croissance
        growth_trend = self._calculate_growth_trend(historical_metrics)
        
        pattern = ResourceUsagePattern(
            service_name=service_name,
            tenant_id=tenant_id,
            cpu_percentiles=cpu_percentiles,
            memory_percentiles=memory_percentiles,
            peak_hours=peak_hours,
            baseline_usage=baseline_usage,
            growth_trend=growth_trend
        )
        
        # Stockage du pattern
        pattern_key = f"{tenant_id}:{service_name}"
        self.usage_patterns[pattern_key] = pattern
        
        return pattern
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calcule les percentiles d'utilisation"""
        if not values:
            return {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'p50': sorted_values[int(n * 0.5)],
            'p90': sorted_values[int(n * 0.9)],
            'p95': sorted_values[int(n * 0.95)],
            'p99': sorted_values[int(n * 0.99)]
        }
    
    def _identify_peak_hours(self, hourly_usage: Dict[int, Dict[str, List[float]]]) -> List[int]:
        """Identifie les heures de pic d'utilisation"""
        hour_averages = {}
        
        for hour, usage_data in hourly_usage.items():
            cpu_avg = mean(usage_data['cpu']) if usage_data['cpu'] else 0
            memory_avg = mean(usage_data['memory']) if usage_data['memory'] else 0
            hour_averages[hour] = cpu_avg + memory_avg
        
        # Trouve les heures avec utilisation > 80% du maximum
        max_usage = max(hour_averages.values()) if hour_averages.values() else 0
        threshold = max_usage * 0.8
        
        peak_hours = [hour for hour, usage in hour_averages.items() if usage >= threshold]
        return sorted(peak_hours)
    
    def _calculate_growth_trend(self, historical_metrics: List[Tuple[datetime, Dict]]) -> float:
        """Calcule la tendance de croissance des ressources"""
        if len(historical_metrics) < 10:
            return 0.0
        
        # Groupement par jour
        daily_usage = {}
        for timestamp, metrics in historical_metrics:
            day_key = timestamp.date()
            if day_key not in daily_usage:
                daily_usage[day_key] = []
            
            cpu_usage = metrics.get('cpu_usage_cores', 0)
            memory_usage = metrics.get('memory_usage_bytes', 0)
            total_usage = cpu_usage + (memory_usage / 1e9)  # Normalisation
            daily_usage[day_key].append(total_usage)
        
        # Calcul de la moyenne par jour
        daily_averages = []
        for day, usage_list in sorted(daily_usage.items()):
            daily_averages.append(mean(usage_list))
        
        if len(daily_averages) < 3:
            return 0.0
        
        # Régression linéaire simple pour la tendance
        x = list(range(len(daily_averages)))
        slope = np.polyfit(x, daily_averages, 1)[0]
        
        # Conversion en pourcentage de croissance par jour
        if daily_averages[0] > 0:
            growth_rate = (slope / daily_averages[0]) * 100
        else:
            growth_rate = 0.0
        
        return max(-50, min(50, growth_rate))  # Bornage ±50%/jour
    
    def _create_default_pattern(self, tenant_id: str, service_name: str) -> ResourceUsagePattern:
        """Crée un pattern par défaut en l'absence de données"""
        return ResourceUsagePattern(
            service_name=service_name,
            tenant_id=tenant_id,
            cpu_percentiles={'p50': 0.1, 'p90': 0.3, 'p95': 0.5, 'p99': 0.8},
            memory_percentiles={'p50': 128e6, 'p90': 256e6, 'p95': 512e6, 'p99': 1e9},
            peak_hours=[9, 10, 11, 14, 15, 16],
            baseline_usage={'cpu': 0.1, 'memory': 128e6},
            growth_trend=0.0
        )
    
    async def generate_resource_recommendation(self, tenant_id: str, service_name: str) -> ResourceRecommendation:
        """Génère une recommandation de ressources optimisée"""
        
        # Analyse du pattern d'utilisation
        pattern = await self.analyze_resource_usage(tenant_id, service_name)
        
        # Configuration du tenant
        resource_limits = self.config_manager.get_resource_limits(tenant_id)
        scaling_mode = self.config_manager.get_scaling_mode(tenant_id)
        
        # Calcul des recommandations CPU
        cpu_recommendation = self._calculate_cpu_recommendation(pattern, resource_limits, scaling_mode)
        
        # Calcul des recommandations mémoire
        memory_recommendation = self._calculate_memory_recommendation(pattern, resource_limits, scaling_mode)
        
        # Évaluation de l'impact coût
        cost_impact = self.cost_model.calculate_cost_impact(
            cpu_recommendation, memory_recommendation, tenant_id
        )
        
        # Génération du raisonnement
        reasoning = self._generate_reasoning(pattern, cpu_recommendation, memory_recommendation)
        
        # Calcul de la confiance
        confidence = self._calculate_recommendation_confidence(pattern)
        
        recommendation = ResourceRecommendation(
            cpu_request=cpu_recommendation['request'],
            cpu_limit=cpu_recommendation['limit'],
            memory_request=memory_recommendation['request'],
            memory_limit=memory_recommendation['limit'],
            confidence=confidence,
            reasoning=reasoning,
            estimated_cost_impact=cost_impact,
            timestamp=datetime.utcnow()
        )
        
        # Stockage de la recommandation
        rec_key = f"{tenant_id}:{service_name}"
        if rec_key not in self.recommendation_history:
            self.recommendation_history[rec_key] = []
        self.recommendation_history[rec_key].append(recommendation)
        
        # Nettoyage de l'historique (garde 30 jours)
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        self.recommendation_history[rec_key] = [
            rec for rec in self.recommendation_history[rec_key]
            if rec.timestamp > cutoff_time
        ]
        
        return recommendation
    
    def _calculate_cpu_recommendation(self, pattern: ResourceUsagePattern, 
                                    limits: ResourceLimits, mode: ScalingMode) -> Dict[str, str]:
        """Calcule les recommandations CPU optimisées"""
        
        # Sélection du percentile selon le mode
        percentile_map = {
            ScalingMode.CONSERVATIVE: 'p99',
            ScalingMode.BALANCED: 'p95',
            ScalingMode.AGGRESSIVE: 'p90',
            ScalingMode.INTELLIGENT: 'p95',
            ScalingMode.COST_OPTIMIZED: 'p90'
        }
        
        target_percentile = percentile_map[mode]
        base_cpu = pattern.cpu_percentiles[target_percentile]
        
        # Ajustement pour la croissance
        if pattern.growth_trend > 0:
            growth_buffer = 1 + (pattern.growth_trend / 100) * 30  # 30 jours
            base_cpu *= growth_buffer
        
        # Buffer de sécurité selon le mode
        safety_buffers = {
            ScalingMode.CONSERVATIVE: 1.5,
            ScalingMode.BALANCED: 1.3,
            ScalingMode.AGGRESSIVE: 1.1,
            ScalingMode.INTELLIGENT: 1.2,
            ScalingMode.COST_OPTIMIZED: 1.1
        }
        
        safety_buffer = safety_buffers[mode]
        
        # CPU request (utilisation normale)
        cpu_request = base_cpu * 1.1  # 10% au-dessus du baseline
        cpu_request = max(0.1, min(cpu_request, limits.max_cpu))
        
        # CPU limit (pics de charge)
        cpu_limit = base_cpu * safety_buffer
        cpu_limit = max(cpu_request, min(cpu_limit, limits.max_cpu))
        
        return {
            'request': f"{cpu_request:.3f}",
            'limit': f"{cpu_limit:.3f}"
        }
    
    def _calculate_memory_recommendation(self, pattern: ResourceUsagePattern,
                                       limits: ResourceLimits, mode: ScalingMode) -> Dict[str, str]:
        """Calcule les recommandations mémoire optimisées"""
        
        # Sélection du percentile selon le mode
        percentile_map = {
            ScalingMode.CONSERVATIVE: 'p99',
            ScalingMode.BALANCED: 'p95',
            ScalingMode.AGGRESSIVE: 'p90',
            ScalingMode.INTELLIGENT: 'p95',
            ScalingMode.COST_OPTIMIZED: 'p90'
        }
        
        target_percentile = percentile_map[mode]
        base_memory = pattern.memory_percentiles[target_percentile]
        
        # Ajustement pour la croissance
        if pattern.growth_trend > 0:
            growth_buffer = 1 + (pattern.growth_trend / 100) * 30
            base_memory *= growth_buffer
        
        # Buffer de sécurité pour la mémoire (plus conservateur)
        safety_buffers = {
            ScalingMode.CONSERVATIVE: 1.8,
            ScalingMode.BALANCED: 1.5,
            ScalingMode.AGGRESSIVE: 1.3,
            ScalingMode.INTELLIGENT: 1.4,
            ScalingMode.COST_OPTIMIZED: 1.2
        }
        
        safety_buffer = safety_buffers[mode]
        
        # Memory request
        memory_request = base_memory * 1.2  # 20% au-dessus du baseline
        memory_request = max(64e6, memory_request)  # Min 64MB
        
        # Memory limit
        memory_limit = base_memory * safety_buffer
        memory_limit = max(memory_request, memory_limit)
        
        # Conversion des limites textuelles
        max_memory_bytes = self._parse_memory_string(limits.max_memory)
        memory_request = min(memory_request, max_memory_bytes)
        memory_limit = min(memory_limit, max_memory_bytes)
        
        return {
            'request': self._format_memory_string(memory_request),
            'limit': self._format_memory_string(memory_limit)
        }
    
    def _parse_memory_string(self, memory_str: str) -> float:
        """Parse une chaîne de mémoire (ex: '2Gi') en bytes"""
        if memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024**3
        elif memory_str.endswith('Mi'):
            return float(memory_str[:-2]) * 1024**2
        elif memory_str.endswith('Ki'):
            return float(memory_str[:-2]) * 1024
        else:
            return float(memory_str)
    
    def _format_memory_string(self, memory_bytes: float) -> str:
        """Formate les bytes en chaîne de mémoire lisible"""
        if memory_bytes >= 1024**3:
            return f"{memory_bytes / 1024**3:.2f}Gi"
        elif memory_bytes >= 1024**2:
            return f"{memory_bytes / 1024**2:.0f}Mi"
        elif memory_bytes >= 1024:
            return f"{memory_bytes / 1024:.0f}Ki"
        else:
            return f"{memory_bytes:.0f}"
    
    def _generate_reasoning(self, pattern: ResourceUsagePattern, 
                          cpu_rec: Dict[str, str], mem_rec: Dict[str, str]) -> str:
        """Génère l'explication de la recommandation"""
        
        reasons = []
        
        # Analyse CPU
        cpu_p95 = pattern.cpu_percentiles['p95']
        if cpu_p95 > 0.8:
            reasons.append(f"High CPU usage detected (P95: {cpu_p95:.2f} cores)")
        elif cpu_p95 < 0.1:
            reasons.append(f"Low CPU usage allows optimization (P95: {cpu_p95:.2f} cores)")
        
        # Analyse mémoire
        mem_p95_gb = pattern.memory_percentiles['p95'] / 1024**3
        if mem_p95_gb > 1.0:
            reasons.append(f"Significant memory usage (P95: {mem_p95_gb:.1f}GB)")
        
        # Tendance de croissance
        if abs(pattern.growth_trend) > 1:
            trend_direction = "increasing" if pattern.growth_trend > 0 else "decreasing"
            reasons.append(f"Resource usage {trend_direction} ({pattern.growth_trend:+.1f}%/day)")
        
        # Heures de pic
        if len(pattern.peak_hours) > 6:
            reasons.append(f"Multiple peak hours detected ({len(pattern.peak_hours)} hours)")
        
        base_msg = f"CPU: {cpu_rec['request']}-{cpu_rec['limit']}, Memory: {mem_rec['request']}-{mem_rec['limit']}"
        
        if reasons:
            return f"{base_msg}. Analysis: {'; '.join(reasons)}"
        else:
            return f"{base_msg}. Based on historical usage patterns"
    
    def _calculate_recommendation_confidence(self, pattern: ResourceUsagePattern) -> float:
        """Calcule le niveau de confiance de la recommandation"""
        
        confidence = 0.5  # Base
        
        # Bonus pour les données historiques
        if pattern.cpu_percentiles['p95'] > 0:
            confidence += 0.2
        
        # Bonus pour la stabilité des patterns
        cpu_variance = pattern.cpu_percentiles['p99'] - pattern.cpu_percentiles['p50']
        if cpu_variance < pattern.cpu_percentiles['p95']:
            confidence += 0.2
        
        # Bonus pour les tendances claires
        if abs(pattern.growth_trend) < 10:  # Tendance stable
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def apply_resource_recommendation(self, recommendation: ResourceRecommendation,
                                          service_name: str, namespace: str = "default") -> bool:
        """Applique la recommandation de ressources au déploiement"""
        
        if not self.k8s_apps_v1:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            # Récupération du déploiement actuel
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=service_name, namespace=namespace
            )
            
            # Mise à jour des ressources
            containers = deployment.spec.template.spec.containers
            for container in containers:
                if not container.resources:
                    container.resources = client.V1ResourceRequirements()
                
                # Mise à jour des requests
                if not container.resources.requests:
                    container.resources.requests = {}
                container.resources.requests['cpu'] = recommendation.cpu_request
                container.resources.requests['memory'] = recommendation.memory_request
                
                # Mise à jour des limits
                if not container.resources.limits:
                    container.resources.limits = {}
                container.resources.limits['cpu'] = recommendation.cpu_limit
                container.resources.limits['memory'] = recommendation.memory_limit
            
            # Application de la mise à jour
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=service_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Applied VPA recommendation for {service_name}: {recommendation.reasoning}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply VPA recommendation: {e}")
            return False
    
    async def get_optimization_recommendations(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Génère des recommandations d'optimisation pour un tenant"""
        
        recommendations = []
        
        # TODO: Récupérer la liste des services du tenant
        services = ["api-service", "ml-service", "audio-processor"]
        
        for service in services:
            try:
                recommendation = await self.generate_resource_recommendation(tenant_id, service)
                
                recommendations.append({
                    'service': service,
                    'current_resources': await self._get_current_resources(service),
                    'recommended_resources': {
                        'cpu_request': recommendation.cpu_request,
                        'cpu_limit': recommendation.cpu_limit,
                        'memory_request': recommendation.memory_request,
                        'memory_limit': recommendation.memory_limit
                    },
                    'reasoning': recommendation.reasoning,
                    'confidence': recommendation.confidence,
                    'cost_impact': recommendation.estimated_cost_impact
                })
                
            except Exception as e:
                logger.error(f"Error generating VPA recommendation for {service}: {e}")
        
        return recommendations
    
    async def _get_current_resources(self, service_name: str, namespace: str = "default") -> Dict[str, str]:
        """Récupère les ressources actuelles d'un service"""
        
        if not self.k8s_apps_v1:
            return {}
        
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=service_name, namespace=namespace
            )
            
            containers = deployment.spec.template.spec.containers
            if containers and containers[0].resources:
                resources = containers[0].resources
                return {
                    'cpu_request': resources.requests.get('cpu', 'Not set') if resources.requests else 'Not set',
                    'cpu_limit': resources.limits.get('cpu', 'Not set') if resources.limits else 'Not set',
                    'memory_request': resources.requests.get('memory', 'Not set') if resources.requests else 'Not set',
                    'memory_limit': resources.limits.get('memory', 'Not set') if resources.limits else 'Not set'
                }
        except Exception as e:
            logger.warning(f"Could not get current resources for {service_name}: {e}")
        
        return {}


class ResourceCostModel:
    """Modèle de calcul des coûts des ressources"""
    
    def __init__(self):
        # Prix par heure (estimations AWS/GCP)
        self.cpu_cost_per_core_hour = 0.04  # $0.04/core/hour
        self.memory_cost_per_gb_hour = 0.004  # $0.004/GB/hour
    
    def calculate_cost_impact(self, cpu_rec: Dict[str, str], 
                            memory_rec: Dict[str, str], tenant_id: str) -> float:
        """Calcule l'impact coût d'une recommandation"""
        
        try:
            # Coût CPU
            cpu_limit = float(cpu_rec['limit'])
            cpu_cost_monthly = cpu_limit * self.cpu_cost_per_core_hour * 24 * 30
            
            # Coût mémoire
            memory_limit_gb = self._parse_memory_to_gb(memory_rec['limit'])
            memory_cost_monthly = memory_limit_gb * self.memory_cost_per_gb_hour * 24 * 30
            
            total_cost_monthly = cpu_cost_monthly + memory_cost_monthly
            
            return round(total_cost_monthly, 2)
            
        except Exception as e:
            logger.warning(f"Could not calculate cost impact: {e}")
            return 0.0
    
    def _parse_memory_to_gb(self, memory_str: str) -> float:
        """Convertit une chaîne mémoire en GB"""
        try:
            if memory_str.endswith('Gi'):
                return float(memory_str[:-2])
            elif memory_str.endswith('Mi'):
                return float(memory_str[:-2]) / 1024
            elif memory_str.endswith('Ki'):
                return float(memory_str[:-2]) / (1024 * 1024)
            else:
                return float(memory_str) / (1024**3)
        except:
            return 0.0
