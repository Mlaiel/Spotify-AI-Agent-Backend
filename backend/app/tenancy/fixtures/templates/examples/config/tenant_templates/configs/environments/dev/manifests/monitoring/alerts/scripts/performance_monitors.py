"""
Moniteurs de Performance Ultra-Avancés
Surveillance intelligente des performances pour Spotify AI Agent

Fonctionnalités:
- Monitoring temps réel des performances système et applicatives
- Détection proactive des dégradations de performance
- Analyse des goulots d'étranglement par IA
- Optimisation automatique des performances
- Monitoring spécialisé pour streaming audio
- Surveillance des APIs et microservices
"""

import asyncio
import logging
import psutil
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
import statistics

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class PerformanceMetricType(Enum):
    """Types de métriques de performance"""
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    SYSTEM_DISK = "system_disk"
    SYSTEM_NETWORK = "system_network"
    APPLICATION_RESPONSE_TIME = "app_response_time"
    APPLICATION_THROUGHPUT = "app_throughput"
    APPLICATION_ERROR_RATE = "app_error_rate"
    DATABASE_PERFORMANCE = "db_performance"
    CACHE_PERFORMANCE = "cache_performance"
    AUDIO_PROCESSING = "audio_processing"
    ML_MODEL_INFERENCE = "ml_inference"
    API_LATENCY = "api_latency"

class PerformanceThreshold(Enum):
    """Seuils de performance prédéfinis"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Métrique de performance"""
    metric_type: PerformanceMetricType
    value: float
    unit: str
    timestamp: datetime
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_status: PerformanceThreshold = PerformanceThreshold.GOOD

@dataclass
class PerformanceProfile:
    """Profil de performance pour un service"""
    service_name: str
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    performance_score: float
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

class AdvancedPerformanceMonitor:
    """Moniteur de performance avancé avec IA"""
    
    def __init__(self):
        self.redis_client = None
        self.performance_history: Dict[str, List[PerformanceMetric]] = {}
        self.service_profiles: Dict[str, PerformanceProfile] = {}
        self.thresholds = self._initialize_thresholds()
        self.monitoring_active = False
        
    async def initialize(self):
        """Initialise le moniteur de performance"""
        try:
            self.redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                decode_responses=True,
                db=3
            )
            
            # Initialisation des profils de services
            await self._initialize_service_profiles()
            
            logger.info("Moniteur de performance initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")

    def _initialize_thresholds(self) -> Dict[PerformanceMetricType, Dict[str, float]]:
        """Initialise les seuils de performance"""
        return {
            PerformanceMetricType.SYSTEM_CPU: {
                "excellent": 30.0,
                "good": 60.0,
                "warning": 80.0,
                "critical": 90.0
            },
            PerformanceMetricType.SYSTEM_MEMORY: {
                "excellent": 40.0,
                "good": 70.0,
                "warning": 85.0,
                "critical": 95.0
            },
            PerformanceMetricType.APPLICATION_RESPONSE_TIME: {
                "excellent": 100.0,  # ms
                "good": 300.0,
                "warning": 1000.0,
                "critical": 3000.0
            },
            PerformanceMetricType.APPLICATION_ERROR_RATE: {
                "excellent": 0.1,  # %
                "good": 1.0,
                "warning": 5.0,
                "critical": 10.0
            },
            PerformanceMetricType.API_LATENCY: {
                "excellent": 50.0,  # ms
                "good": 200.0,
                "warning": 500.0,
                "critical": 1500.0
            },
            PerformanceMetricType.AUDIO_PROCESSING: {
                "excellent": 10.0,  # ms de latence audio
                "good": 50.0,
                "warning": 100.0,
                "critical": 200.0
            }
        }

    async def _initialize_service_profiles(self):
        """Initialise les profils de performance des services"""
        
        services = [
            "api-gateway",
            "authentication-service", 
            "audio-processing-service",
            "ml-recommendation-service",
            "user-management-service",
            "payment-service",
            "analytics-service"
        ]
        
        for service in services:
            baseline_metrics = await self._calculate_baseline_metrics(service)
            
            profile = PerformanceProfile(
                service_name=service,
                baseline_metrics=baseline_metrics,
                current_metrics={},
                performance_score=100.0
            )
            
            self.service_profiles[service] = profile

    async def _calculate_baseline_metrics(self, service_name: str) -> Dict[str, float]:
        """Calcule les métriques de base pour un service"""
        
        # En production, récupérer depuis l'historique des métriques
        # Ici, simulation de valeurs de baseline réalistes
        baseline_values = {
            "api-gateway": {
                "response_time": 150.0,
                "throughput": 1000.0,
                "error_rate": 0.5,
                "cpu_usage": 45.0,
                "memory_usage": 60.0
            },
            "authentication-service": {
                "response_time": 80.0,
                "throughput": 500.0,
                "error_rate": 0.2,
                "cpu_usage": 30.0,
                "memory_usage": 40.0
            },
            "audio-processing-service": {
                "response_time": 300.0,
                "throughput": 200.0,
                "error_rate": 1.0,
                "cpu_usage": 70.0,
                "memory_usage": 80.0,
                "audio_latency": 25.0
            },
            "ml-recommendation-service": {
                "response_time": 500.0,
                "throughput": 100.0,
                "error_rate": 2.0,
                "cpu_usage": 85.0,
                "memory_usage": 75.0,
                "inference_time": 150.0
            }
        }
        
        return baseline_values.get(service_name, {
            "response_time": 200.0,
            "throughput": 300.0,
            "error_rate": 1.0,
            "cpu_usage": 50.0,
            "memory_usage": 60.0
        })

    async def start_monitoring(self):
        """Démarre le monitoring en continu"""
        self.monitoring_active = True
        
        monitoring_tasks = [
            self._monitor_system_performance(),
            self._monitor_application_performance(),
            self._monitor_database_performance(),
            self._monitor_audio_processing_performance(),
            self._monitor_api_performance()
        ]
        
        await asyncio.gather(*monitoring_tasks)

    async def _monitor_system_performance(self):
        """Monitore les performances système"""
        while self.monitoring_active:
            try:
                # Collecte des métriques système
                cpu_metric = PerformanceMetric(
                    metric_type=PerformanceMetricType.SYSTEM_CPU,
                    value=psutil.cpu_percent(interval=1),
                    unit="%",
                    timestamp=datetime.utcnow(),
                    source="psutil"
                )
                
                memory_info = psutil.virtual_memory()
                memory_metric = PerformanceMetric(
                    metric_type=PerformanceMetricType.SYSTEM_MEMORY,
                    value=memory_info.percent,
                    unit="%",
                    timestamp=datetime.utcnow(),
                    source="psutil"
                )
                
                disk_info = psutil.disk_usage('/')
                disk_metric = PerformanceMetric(
                    metric_type=PerformanceMetricType.SYSTEM_DISK,
                    value=disk_info.percent,
                    unit="%",
                    timestamp=datetime.utcnow(),
                    source="psutil"
                )
                
                # Ajout des seuils
                for metric in [cpu_metric, memory_metric, disk_metric]:
                    metric.threshold_status = self._calculate_threshold_status(metric)
                    await self._store_metric(metric)
                    
                    # Alerte si seuil critique
                    if metric.threshold_status == PerformanceThreshold.CRITICAL:
                        await self._trigger_performance_alert(metric)
                
                await asyncio.sleep(30)  # Collecte toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur monitoring système: {e}")
                await asyncio.sleep(60)

    async def _monitor_application_performance(self):
        """Monitore les performances applicatives"""
        while self.monitoring_active:
            try:
                for service_name, profile in self.service_profiles.items():
                    # Simulation de collecte de métriques applicatives
                    current_metrics = await self._collect_application_metrics(service_name)
                    
                    # Mise à jour du profil
                    profile.current_metrics = current_metrics
                    profile.performance_score = await self._calculate_performance_score(profile)
                    profile.last_updated = datetime.utcnow()
                    
                    # Détection des dégradations
                    degradations = await self._detect_performance_degradation(profile)
                    if degradations:
                        await self._handle_performance_degradation(service_name, degradations)
                    
                    # Analyse des goulots d'étranglement
                    bottlenecks = await self._analyze_bottlenecks(profile)
                    profile.bottlenecks = bottlenecks
                    
                    # Génération de recommandations
                    recommendations = await self._generate_performance_recommendations(profile)
                    profile.recommendations = recommendations
                
                await asyncio.sleep(60)  # Analyse toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur monitoring application: {e}")
                await asyncio.sleep(120)

    async def _collect_application_metrics(self, service_name: str) -> Dict[str, float]:
        """Collecte les métriques d'une application"""
        
        try:
            # En production, collecter depuis les endpoints de métriques
            # Simulation de collecte avec variations réalistes
            baseline = self.service_profiles[service_name].baseline_metrics
            
            # Ajout de variations aléatoires autour de la baseline
            import random
            
            current_metrics = {}
            for metric_name, baseline_value in baseline.items():
                # Variation de ±20% autour de la baseline
                variation = random.uniform(-0.2, 0.2)
                current_value = baseline_value * (1 + variation)
                current_metrics[metric_name] = max(0, current_value)
            
            # Simulation de conditions spéciales
            current_hour = datetime.utcnow().hour
            if 19 <= current_hour <= 21:  # Pic du soir
                current_metrics["response_time"] *= 1.5
                current_metrics["cpu_usage"] *= 1.3
                current_metrics["error_rate"] *= 2.0
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques {service_name}: {e}")
            return {}

    async def _calculate_performance_score(self, profile: PerformanceProfile) -> float:
        """Calcule un score de performance global"""
        
        if not profile.current_metrics or not profile.baseline_metrics:
            return 100.0
        
        scores = []
        
        for metric_name in profile.baseline_metrics.keys():
            if metric_name in profile.current_metrics:
                baseline = profile.baseline_metrics[metric_name]
                current = profile.current_metrics[metric_name]
                
                # Calcul du score selon le type de métrique
                if metric_name in ["response_time", "error_rate", "cpu_usage", "memory_usage"]:
                    # Plus bas = mieux
                    if current <= baseline:
                        score = 100.0
                    else:
                        degradation = (current - baseline) / baseline
                        score = max(0, 100 - (degradation * 100))
                else:
                    # Plus haut = mieux (throughput, etc.)
                    if current >= baseline:
                        score = 100.0
                    else:
                        degradation = (baseline - current) / baseline
                        score = max(0, 100 - (degradation * 100))
                
                scores.append(score)
        
        return statistics.mean(scores) if scores else 100.0

    async def _detect_performance_degradation(self, profile: PerformanceProfile) -> List[str]:
        """Détecte les dégradations de performance"""
        
        degradations = []
        
        if not profile.current_metrics or not profile.baseline_metrics:
            return degradations
        
        for metric_name, baseline_value in profile.baseline_metrics.items():
            if metric_name in profile.current_metrics:
                current_value = profile.current_metrics[metric_name]
                
                # Seuils de dégradation
                if metric_name in ["response_time", "error_rate"]:
                    if current_value > baseline_value * 1.5:
                        degradations.append(f"{metric_name}: {current_value:.1f} (baseline: {baseline_value:.1f})")
                elif metric_name == "throughput":
                    if current_value < baseline_value * 0.7:
                        degradations.append(f"{metric_name}: {current_value:.1f} (baseline: {baseline_value:.1f})")
                elif metric_name in ["cpu_usage", "memory_usage"]:
                    if current_value > baseline_value * 1.3:
                        degradations.append(f"{metric_name}: {current_value:.1f}% (baseline: {baseline_value:.1f}%)")
        
        return degradations

    async def _analyze_bottlenecks(self, profile: PerformanceProfile) -> List[str]:
        """Analyse les goulots d'étranglement"""
        
        bottlenecks = []
        
        if not profile.current_metrics:
            return bottlenecks
        
        metrics = profile.current_metrics
        
        # Détection CPU
        if metrics.get("cpu_usage", 0) > 80:
            bottlenecks.append("CPU: Utilisation élevée détectée")
        
        # Détection mémoire
        if metrics.get("memory_usage", 0) > 85:
            bottlenecks.append("Mémoire: Saturation proche")
        
        # Détection latence réseau
        if metrics.get("response_time", 0) > 1000:
            bottlenecks.append("Réseau: Latence élevée")
        
        # Détection base de données
        if metrics.get("db_response_time", 0) > 500:
            bottlenecks.append("Base de données: Requêtes lentes")
        
        # Détection audio spécifique
        if metrics.get("audio_latency", 0) > 100:
            bottlenecks.append("Audio: Latence de traitement élevée")
        
        # Détection ML
        if metrics.get("inference_time", 0) > 300:
            bottlenecks.append("ML: Temps d'inférence élevé")
        
        return bottlenecks

    async def _generate_performance_recommendations(self, profile: PerformanceProfile) -> List[str]:
        """Génère des recommandations d'optimisation"""
        
        recommendations = []
        metrics = profile.current_metrics
        
        if not metrics:
            return recommendations
        
        # Recommandations CPU
        if metrics.get("cpu_usage", 0) > 80:
            recommendations.extend([
                "Optimiser les algorithmes gourmands en CPU",
                "Considérer le scaling horizontal",
                "Analyser les processus en arrière-plan"
            ])
        
        # Recommandations mémoire
        if metrics.get("memory_usage", 0) > 85:
            recommendations.extend([
                "Vérifier les fuites mémoire",
                "Optimiser le cache applicatif",
                "Augmenter la RAM disponible"
            ])
        
        # Recommandations latence
        if metrics.get("response_time", 0) > 1000:
            recommendations.extend([
                "Implémenter un cache distribué",
                "Optimiser les requêtes base de données",
                "Considérer un CDN pour les assets statiques"
            ])
        
        # Recommandations audio
        if metrics.get("audio_latency", 0) > 100:
            recommendations.extend([
                "Optimiser les buffers audio",
                "Vérifier la configuration des codecs",
                "Réduire la taille des chunks de traitement"
            ])
        
        # Recommandations ML
        if metrics.get("inference_time", 0) > 300:
            recommendations.extend([
                "Optimiser les modèles ML (quantization, pruning)",
                "Utiliser un cache pour les prédictions",
                "Considérer l'accélération GPU"
            ])
        
        return recommendations

    async def _monitor_database_performance(self):
        """Monitore les performances de base de données"""
        while self.monitoring_active:
            try:
                # Simulation de monitoring DB
                db_metrics = await self._collect_database_metrics()
                
                for metric in db_metrics:
                    metric.threshold_status = self._calculate_threshold_status(metric)
                    await self._store_metric(metric)
                    
                    if metric.threshold_status in [PerformanceThreshold.WARNING, PerformanceThreshold.CRITICAL]:
                        await self._trigger_database_alert(metric)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Erreur monitoring DB: {e}")
                await asyncio.sleep(120)

    async def _collect_database_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques de base de données"""
        
        # Simulation de métriques DB
        import random
        
        metrics = [
            PerformanceMetric(
                metric_type=PerformanceMetricType.DATABASE_PERFORMANCE,
                value=random.uniform(50, 200),  # Temps de réponse en ms
                unit="ms",
                timestamp=datetime.utcnow(),
                source="postgresql",
                tags={"metric": "query_time"}
            ),
            PerformanceMetric(
                metric_type=PerformanceMetricType.DATABASE_PERFORMANCE,
                value=random.uniform(0, 50),  # Connexions actives
                unit="connections",
                timestamp=datetime.utcnow(),
                source="postgresql",
                tags={"metric": "active_connections"}
            )
        ]
        
        return metrics

    async def _monitor_audio_processing_performance(self):
        """Monitore les performances de traitement audio"""
        while self.monitoring_active:
            try:
                audio_metrics = await self._collect_audio_metrics()
                
                for metric in audio_metrics:
                    metric.threshold_status = self._calculate_threshold_status(metric)
                    await self._store_metric(metric)
                    
                    if metric.threshold_status == PerformanceThreshold.CRITICAL:
                        await self._trigger_audio_alert(metric)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Erreur monitoring audio: {e}")
                await asyncio.sleep(60)

    async def _collect_audio_metrics(self) -> List[PerformanceMetric]:
        """Collecte les métriques de traitement audio"""
        
        import random
        
        return [
            PerformanceMetric(
                metric_type=PerformanceMetricType.AUDIO_PROCESSING,
                value=random.uniform(10, 150),  # Latence audio en ms
                unit="ms",
                timestamp=datetime.utcnow(),
                source="audio_processor",
                tags={"metric": "processing_latency"}
            ),
            PerformanceMetric(
                metric_type=PerformanceMetricType.AUDIO_PROCESSING,
                value=random.uniform(80, 100),  # Qualité audio score
                unit="score",
                timestamp=datetime.utcnow(),
                source="audio_processor",
                tags={"metric": "quality_score"}
            )
        ]

    async def _monitor_api_performance(self):
        """Monitore les performances des APIs"""
        while self.monitoring_active:
            try:
                api_endpoints = [
                    "http://localhost:8000/health",
                    "http://localhost:8000/api/v1/users",
                    "http://localhost:8000/api/v1/audio/process"
                ]
                
                for endpoint in api_endpoints:
                    latency = await self._measure_api_latency(endpoint)
                    
                    if latency:
                        metric = PerformanceMetric(
                            metric_type=PerformanceMetricType.API_LATENCY,
                            value=latency,
                            unit="ms",
                            timestamp=datetime.utcnow(),
                            source="api_monitor",
                            tags={"endpoint": endpoint}
                        )
                        
                        metric.threshold_status = self._calculate_threshold_status(metric)
                        await self._store_metric(metric)
                        
                        if metric.threshold_status == PerformanceThreshold.CRITICAL:
                            await self._trigger_api_alert(metric)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Erreur monitoring API: {e}")
                await asyncio.sleep(120)

    async def _measure_api_latency(self, endpoint: str) -> Optional[float]:
        """Mesure la latence d'un endpoint API"""
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    await response.read()
                    
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # en ms
            
            return latency
            
        except Exception as e:
            logger.error(f"Erreur mesure latence {endpoint}: {e}")
            return None

    def _calculate_threshold_status(self, metric: PerformanceMetric) -> PerformanceThreshold:
        """Calcule le status du seuil pour une métrique"""
        
        thresholds = self.thresholds.get(metric.metric_type)
        if not thresholds:
            return PerformanceThreshold.GOOD
        
        value = metric.value
        
        if value <= thresholds["excellent"]:
            return PerformanceThreshold.EXCELLENT
        elif value <= thresholds["good"]:
            return PerformanceThreshold.GOOD
        elif value <= thresholds["warning"]:
            return PerformanceThreshold.WARNING
        else:
            return PerformanceThreshold.CRITICAL

    async def _store_metric(self, metric: PerformanceMetric):
        """Stocke une métrique dans Redis"""
        
        try:
            if self.redis_client:
                key = f"performance:{metric.metric_type.value}:{metric.source}"
                
                data = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "threshold_status": metric.threshold_status.value,
                    "tags": metric.tags
                }
                
                await self.redis_client.lpush(key, json.dumps(data))
                await self.redis_client.ltrim(key, 0, 999)  # Garder seulement les 1000 dernières
                
        except Exception as e:
            logger.error(f"Erreur stockage métrique: {e}")

    async def _trigger_performance_alert(self, metric: PerformanceMetric):
        """Déclenche une alerte de performance"""
        
        logger.warning(
            f"ALERTE PERFORMANCE: {metric.metric_type.value} = {metric.value} {metric.unit} "
            f"(Seuil: {metric.threshold_status.value})"
        )

    async def _trigger_database_alert(self, metric: PerformanceMetric):
        """Déclenche une alerte de base de données"""
        
        logger.warning(
            f"ALERTE DATABASE: {metric.tags.get('metric', 'unknown')} = {metric.value} {metric.unit}"
        )

    async def _trigger_audio_alert(self, metric: PerformanceMetric):
        """Déclenche une alerte audio"""
        
        logger.warning(
            f"ALERTE AUDIO: {metric.tags.get('metric', 'unknown')} = {metric.value} {metric.unit}"
        )

    async def _trigger_api_alert(self, metric: PerformanceMetric):
        """Déclenche une alerte API"""
        
        logger.warning(
            f"ALERTE API: {metric.tags.get('endpoint', 'unknown')} latency = {metric.value} ms"
        )

    async def _handle_performance_degradation(self, service_name: str, degradations: List[str]):
        """Gère une dégradation de performance détectée"""
        
        logger.warning(f"Dégradation de performance détectée pour {service_name}:")
        for degradation in degradations:
            logger.warning(f"  - {degradation}")
        
        # En production, déclencher des actions de remédiation automatique

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances"""
        
        summary = {
            "services_monitored": len(self.service_profiles),
            "average_performance_score": 0.0,
            "critical_services": [],
            "top_bottlenecks": [],
            "monitoring_active": self.monitoring_active
        }
        
        if self.service_profiles:
            # Score moyen
            scores = [profile.performance_score for profile in self.service_profiles.values()]
            summary["average_performance_score"] = statistics.mean(scores)
            
            # Services critiques
            critical_services = [
                name for name, profile in self.service_profiles.items()
                if profile.performance_score < 70
            ]
            summary["critical_services"] = critical_services
            
            # Top des goulots d'étranglement
            all_bottlenecks = []
            for profile in self.service_profiles.values():
                all_bottlenecks.extend(profile.bottlenecks)
            
            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                bottleneck_type = bottleneck.split(":")[0]
                bottleneck_counts[bottleneck_type] = bottleneck_counts.get(bottleneck_type, 0) + 1
            
            summary["top_bottlenecks"] = sorted(
                bottleneck_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        
        return summary

    async def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitoring_active = False
        logger.info("Monitoring de performance arrêté")

# Instance globale du moniteur de performance
_performance_monitor = AdvancedPerformanceMonitor()

async def start_performance_monitoring():
    """Function helper pour démarrer le monitoring"""
    if not _performance_monitor.redis_client:
        await _performance_monitor.initialize()
    
    await _performance_monitor.start_monitoring()

async def get_performance_monitor() -> AdvancedPerformanceMonitor:
    """Retourne l'instance du moniteur de performance"""
    return _performance_monitor

# Configuration des alertes de performance
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes
    performance_configs = [
        AlertConfig(
            name="system_cpu_critical",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            conditions=['CPU usage > 90%'],
            actions=['scale_resources', 'notify_ops_team'],
            thresholds={"cpu_usage": 90},
            ml_enabled=False,
            auto_remediation=True
        ),
        AlertConfig(
            name="application_response_time_warning",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.DETECTION,
            conditions=['Response time > 1000ms'],
            actions=['analyze_performance', 'optimize_queries'],
            thresholds={"response_time": 1000},
            ml_enabled=True
        ),
        AlertConfig(
            name="audio_latency_critical",
            category=AlertCategory.AUDIO_QUALITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.DETECTION,
            conditions=['Audio processing latency > 200ms'],
            actions=['restart_audio_service', 'check_audio_pipeline'],
            thresholds={"audio_latency": 200},
            ml_enabled=False,
            auto_remediation=True
        )
    ]
    
    for config in performance_configs:
        register_alert(config)
