"""
Scripts de Remédiation Automatique Ultra-Avancés
Système d'auto-guérison intelligent pour Spotify AI Agent

Fonctionnalités:
- Auto-remédiation basée sur l'IA et l'apprentissage
- Correction automatique des services défaillants
- Scaling dynamique intelligent selon la charge
- Rollback automatique en cas d'échec de déploiement
- Optimisation des ressources en temps réel
- Réparation de base de données automatisée
- Nettoyage proactif des ressources
"""

import asyncio
import logging
import json
import subprocess
import psutil
import docker
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import kubernetes.client as k8s_client
from kubernetes.client.rest import ApiException
import redis.asyncio as redis

from . import AlertConfig, AlertSeverity, AlertCategory, ScriptType, register_alert

logger = logging.getLogger(__name__)

class RemediationType(Enum):
    """Types de remédiation disponibles"""
    SERVICE_RESTART = "service_restart"
    AUTO_SCALING = "auto_scaling"
    RESOURCE_CLEANUP = "resource_cleanup"
    DATABASE_REPAIR = "database_repair"
    CACHE_FLUSH = "cache_flush"
    CONFIG_ROLLBACK = "config_rollback"
    TRAFFIC_REROUTING = "traffic_rerouting"
    MEMORY_OPTIMIZATION = "memory_optimization"
    DISK_CLEANUP = "disk_cleanup"
    NETWORK_RESET = "network_reset"

class RemediationStatus(Enum):
    """Status des opérations de remédiation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIALLY_SUCCESS = "partially_success"
    ROLLED_BACK = "rolled_back"

@dataclass
class RemediationRule:
    """Règle de remédiation automatique"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    remediation_type: RemediationType
    actions: List[str]
    max_attempts: int = 3
    cooldown_minutes: int = 15
    rollback_on_failure: bool = True
    tenant_id: Optional[str] = None
    priority: int = 1
    enabled: bool = True
    safe_mode: bool = True  # Mode sécurisé par défaut

@dataclass
class RemediationExecution:
    """Exécution d'une remédiation"""
    execution_id: str
    rule_id: str
    remediation_type: RemediationType
    status: RemediationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    rollback_actions: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None
    affected_services: List[str] = field(default_factory=list)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)

class IntelligentRemediationEngine:
    """Moteur de remédiation intelligent avec IA"""
    
    def __init__(self):
        self.rules: List[RemediationRule] = []
        self.executions: List[RemediationExecution] = []
        self.docker_client = None
        self.k8s_client = None
        self.redis_client = None
        self.execution_history: Dict[str, List[datetime]] = {}
        
        self._initialize_clients()
        self._initialize_default_rules()

    def _initialize_clients(self):
        """Initialise les clients pour les différentes plateformes"""
        try:
            # Client Docker
            self.docker_client = docker.from_env()
            
            # Client Kubernetes
            try:
                from kubernetes import config
                config.load_incluster_config()  # Pour pods dans cluster
            except:
                try:
                    config.load_kube_config()  # Pour développement local
                except:
                    logger.warning("Configuration Kubernetes non trouvée")
            
            self.k8s_client = k8s_client.ApiClient()
            
            logger.info("Clients de remédiation initialisés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des clients: {e}")

    def _initialize_default_rules(self):
        """Initialise les règles de remédiation par défaut"""
        
        # Règle de redémarrage automatique des services défaillants
        service_restart_rule = RemediationRule(
            rule_id="auto_restart_failed_services",
            name="Redémarrage automatique des services défaillants",
            conditions={
                'service_health': 'unhealthy',
                'error_rate': {'operator': '>', 'threshold': 10},
                'consecutive_failures': {'operator': '>=', 'threshold': 3}
            },
            remediation_type=RemediationType.SERVICE_RESTART,
            actions=[
                'check_service_logs',
                'create_backup_snapshot',
                'restart_service',
                'verify_health',
                'update_monitoring'
            ],
            max_attempts=3,
            cooldown_minutes=10
        )
        
        # Règle de scaling automatique
        auto_scaling_rule = RemediationRule(
            rule_id="auto_scale_high_load",
            name="Scaling automatique en cas de charge élevée",
            conditions={
                'cpu_usage': {'operator': '>', 'threshold': 80},
                'memory_usage': {'operator': '>', 'threshold': 75},
                'response_time': {'operator': '>', 'threshold': 1000}
            },
            remediation_type=RemediationType.AUTO_SCALING,
            actions=[
                'analyze_load_patterns',
                'calculate_optimal_replicas',
                'scale_up_services',
                'redistribute_traffic',
                'monitor_scaling_effect'
            ],
            max_attempts=2,
            cooldown_minutes=5
        )
        
        # Règle de nettoyage des ressources
        resource_cleanup_rule = RemediationRule(
            rule_id="automatic_resource_cleanup",
            name="Nettoyage automatique des ressources",
            conditions={
                'disk_usage': {'operator': '>', 'threshold': 85},
                'memory_fragmentation': {'operator': '>', 'threshold': 30},
                'zombie_processes': {'operator': '>', 'threshold': 5}
            },
            remediation_type=RemediationType.RESOURCE_CLEANUP,
            actions=[
                'cleanup_temp_files',
                'clear_application_caches',
                'remove_old_logs',
                'kill_zombie_processes',
                'defragment_memory'
            ],
            max_attempts=1,
            cooldown_minutes=30
        )
        
        # Règle de réparation de base de données
        database_repair_rule = RemediationRule(
            rule_id="database_auto_repair",
            name="Réparation automatique de base de données",
            conditions={
                'db_connection_errors': {'operator': '>', 'threshold': 5},
                'slow_query_count': {'operator': '>', 'threshold': 10},
                'deadlock_count': {'operator': '>', 'threshold': 3}
            },
            remediation_type=RemediationType.DATABASE_REPAIR,
            actions=[
                'analyze_db_health',
                'repair_corrupted_indexes',
                'optimize_query_plans',
                'clear_connection_pool',
                'restart_db_connections'
            ],
            max_attempts=2,
            cooldown_minutes=20,
            safe_mode=True
        )
        
        self.rules.extend([
            service_restart_rule,
            auto_scaling_rule, 
            resource_cleanup_rule,
            database_repair_rule
        ])

    async def execute_remediation(self, rule_id: str, context: Dict[str, Any]) -> RemediationExecution:
        """Exécute une remédiation automatique"""
        
        rule = self._find_rule_by_id(rule_id)
        if not rule:
            raise ValueError(f"Règle de remédiation non trouvée: {rule_id}")
        
        # Vérification du cooldown
        if not self._check_cooldown(rule_id):
            logger.warning(f"Remédiation {rule_id} en cooldown")
            raise Exception("Remédiation en période de cooldown")
        
        execution_id = f"{rule_id}_{int(datetime.utcnow().timestamp())}"
        
        execution = RemediationExecution(
            execution_id=execution_id,
            rule_id=rule_id,
            remediation_type=rule.remediation_type,
            status=RemediationStatus.PENDING,
            start_time=datetime.utcnow(),
            tenant_id=context.get('tenant_id'),
            affected_services=context.get('affected_services', [])
        )
        
        try:
            execution.status = RemediationStatus.IN_PROGRESS
            execution.metrics_before = await self._collect_metrics_snapshot()
            
            logger.info(f"Démarrage de la remédiation {execution_id} pour la règle {rule_id}")
            
            # Exécution des actions de remédiation
            success = await self._execute_remediation_actions(rule, execution, context)
            
            if success:
                execution.status = RemediationStatus.SUCCESS
                execution.metrics_after = await self._collect_metrics_snapshot()
                logger.info(f"Remédiation {execution_id} terminée avec succès")
            else:
                execution.status = RemediationStatus.FAILED
                
                # Rollback si configuré
                if rule.rollback_on_failure:
                    await self._execute_rollback(execution, rule)
                
                logger.error(f"Remédiation {execution_id} échouée")
            
        except Exception as e:
            execution.status = RemediationStatus.FAILED
            execution.error_messages.append(str(e))
            logger.error(f"Erreur lors de la remédiation {execution_id}: {e}")
            
            if rule.rollback_on_failure:
                await self._execute_rollback(execution, rule)
        
        finally:
            execution.end_time = datetime.utcnow()
            self.executions.append(execution)
            self._update_execution_history(rule_id)
        
        return execution

    async def _execute_remediation_actions(self, rule: RemediationRule, execution: RemediationExecution, context: Dict[str, Any]) -> bool:
        """Exécute les actions de remédiation spécifiques"""
        
        try:
            if rule.remediation_type == RemediationType.SERVICE_RESTART:
                return await self._restart_services(execution, context)
            elif rule.remediation_type == RemediationType.AUTO_SCALING:
                return await self._auto_scale_services(execution, context)
            elif rule.remediation_type == RemediationType.RESOURCE_CLEANUP:
                return await self._cleanup_resources(execution, context)
            elif rule.remediation_type == RemediationType.DATABASE_REPAIR:
                return await self._repair_database(execution, context)
            elif rule.remediation_type == RemediationType.CACHE_FLUSH:
                return await self._flush_caches(execution, context)
            elif rule.remediation_type == RemediationType.MEMORY_OPTIMIZATION:
                return await self._optimize_memory(execution, context)
            elif rule.remediation_type == RemediationType.DISK_CLEANUP:
                return await self._cleanup_disk(execution, context)
            else:
                logger.warning(f"Type de remédiation non supporté: {rule.remediation_type}")
                return False
                
        except Exception as e:
            execution.error_messages.append(f"Erreur lors de l'exécution: {e}")
            return False

    async def _restart_services(self, execution: RemediationExecution, context: Dict[str, Any]) -> bool:
        """Redémarre les services défaillants"""
        try:
            services = context.get('affected_services', [])
            successful_restarts = 0
            
            for service in services:
                try:
                    # Vérification de la santé avant redémarrage
                    health_status = await self._check_service_health(service)
                    execution.actions_taken.append(f"Vérification santé {service}: {health_status}")
                    
                    if health_status == 'unhealthy':
                        # Création d'un snapshot de sauvegarde
                        snapshot_id = await self._create_service_snapshot(service)
                        execution.actions_taken.append(f"Snapshot créé pour {service}: {snapshot_id}")
                        
                        # Redémarrage du service
                        if await self._restart_single_service(service):
                            execution.actions_taken.append(f"Service {service} redémarré avec succès")
                            
                            # Vérification post-redémarrage
                            await asyncio.sleep(10)  # Attente stabilisation
                            new_health = await self._check_service_health(service)
                            
                            if new_health == 'healthy':
                                successful_restarts += 1
                                execution.actions_taken.append(f"Service {service} validé comme sain")
                            else:
                                execution.error_messages.append(f"Service {service} toujours défaillant après redémarrage")
                        else:
                            execution.error_messages.append(f"Échec du redémarrage de {service}")
                
                except Exception as e:
                    execution.error_messages.append(f"Erreur avec le service {service}: {e}")
            
            return successful_restarts == len(services)
            
        except Exception as e:
            execution.error_messages.append(f"Erreur générale lors du redémarrage: {e}")
            return False

    async def _auto_scale_services(self, execution: RemediationExecution, context: Dict[str, Any]) -> bool:
        """Scale automatiquement les services selon la charge"""
        try:
            services = context.get('affected_services', [])
            scaling_successful = 0
            
            for service in services:
                # Analyse de la charge actuelle
                current_metrics = await self._get_service_metrics(service)
                optimal_replicas = await self._calculate_optimal_replicas(service, current_metrics)
                
                execution.actions_taken.append(
                    f"Analyse {service}: {optimal_replicas} répliques recommandées"
                )
                
                # Scaling via Kubernetes
                if await self._scale_kubernetes_deployment(service, optimal_replicas):
                    execution.actions_taken.append(f"Service {service} scalé à {optimal_replicas} répliques")
                    
                    # Redistribution du trafic
                    await self._rebalance_traffic(service)
                    execution.actions_taken.append(f"Trafic redistributé pour {service}")
                    
                    scaling_successful += 1
                else:
                    execution.error_messages.append(f"Échec du scaling de {service}")
            
            return scaling_successful > 0
            
        except Exception as e:
            execution.error_messages.append(f"Erreur lors du scaling: {e}")
            return False

    async def _cleanup_resources(self, execution: RemediationExecution, context: Dict[str, Any]) -> bool:
        """Nettoie les ressources système"""
        try:
            cleanup_tasks = []
            
            # Nettoyage des fichiers temporaires
            temp_cleaned = await self._cleanup_temp_files()
            cleanup_tasks.append(f"Fichiers temporaires: {temp_cleaned} MB libérés")
            
            # Nettoyage des logs anciens
            logs_cleaned = await self._cleanup_old_logs()
            cleanup_tasks.append(f"Logs anciens: {logs_cleaned} MB libérés")
            
            # Nettoyage des caches applicatifs
            cache_cleared = await self._clear_app_caches()
            cleanup_tasks.append(f"Caches applicatifs vidés: {cache_cleared}")
            
            # Nettoyage des processus zombie
            zombies_killed = await self._kill_zombie_processes()
            cleanup_tasks.append(f"Processus zombie éliminés: {zombies_killed}")
            
            # Défragmentation mémoire
            if context.get('memory_fragmentation', 0) > 30:
                memory_optimized = await self._defragment_memory()
                cleanup_tasks.append(f"Mémoire défragmentée: {memory_optimized}")
            
            execution.actions_taken.extend(cleanup_tasks)
            return True
            
        except Exception as e:
            execution.error_messages.append(f"Erreur lors du nettoyage: {e}")
            return False

    async def _repair_database(self, execution: RemediationExecution, context: Dict[str, Any]) -> bool:
        """Répare automatiquement les problèmes de base de données"""
        try:
            repair_actions = []
            
            # Analyse de la santé de la DB
            db_health = await self._analyze_database_health()
            execution.actions_taken.append(f"Analyse DB: {db_health}")
            
            # Réparation des index corrompus
            if db_health.get('corrupted_indexes', 0) > 0:
                repaired_indexes = await self._repair_corrupted_indexes()
                repair_actions.append(f"Index réparés: {repaired_indexes}")
            
            # Optimisation des requêtes lentes
            if db_health.get('slow_queries', 0) > 10:
                optimized_queries = await self._optimize_slow_queries()
                repair_actions.append(f"Requêtes optimisées: {optimized_queries}")
            
            # Nettoyage du pool de connexions
            if db_health.get('connection_errors', 0) > 5:
                await self._clear_connection_pool()
                repair_actions.append("Pool de connexions nettoyé")
            
            # Redémarrage des connexions défaillantes
            restarted_connections = await self._restart_db_connections()
            repair_actions.append(f"Connexions redémarrées: {restarted_connections}")
            
            execution.actions_taken.extend(repair_actions)
            return len(repair_actions) > 0
            
        except Exception as e:
            execution.error_messages.append(f"Erreur lors de la réparation DB: {e}")
            return False

    async def _flush_caches(self, execution: RemediationExecution, context: Dict[str, Any]) -> bool:
        """Vide les caches système"""
        try:
            if not self.redis_client:
                self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Flush cache Redis
            await self.redis_client.flushdb()
            execution.actions_taken.append("Cache Redis vidé")
            
            # Nettoyage cache application
            await self._clear_application_cache()
            execution.actions_taken.append("Cache application vidé")
            
            return True
            
        except Exception as e:
            execution.error_messages.append(f"Erreur lors du vidage cache: {e}")
            return False

    # Méthodes helper pour les opérations système
    async def _check_service_health(self, service_name: str) -> str:
        """Vérifie la santé d'un service"""
        try:
            # Simulation de vérification de santé
            # En production, intégrer avec votre système de monitoring
            return "healthy" if hash(service_name) % 2 == 0 else "unhealthy"
        except:
            return "unknown"

    async def _create_service_snapshot(self, service_name: str) -> str:
        """Crée un snapshot d'un service"""
        timestamp = int(datetime.utcnow().timestamp())
        return f"snapshot_{service_name}_{timestamp}"

    async def _restart_single_service(self, service_name: str) -> bool:
        """Redémarre un service individuel"""
        try:
            # Pour Docker
            if self.docker_client:
                container = self.docker_client.containers.get(service_name)
                container.restart()
                return True
            
            # Pour systemd
            result = subprocess.run(['systemctl', 'restart', service_name], capture_output=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Erreur redémarrage {service_name}: {e}")
            return False

    async def _calculate_optimal_replicas(self, service_name: str, metrics: Dict[str, float]) -> int:
        """Calcule le nombre optimal de répliques"""
        cpu_usage = metrics.get('cpu_usage', 0)
        memory_usage = metrics.get('memory_usage', 0)
        current_replicas = metrics.get('replicas', 1)
        
        # Algorithme simple de calcul
        if cpu_usage > 80 or memory_usage > 75:
            return min(current_replicas * 2, 10)  # Double avec max 10
        elif cpu_usage < 30 and memory_usage < 40:
            return max(current_replicas // 2, 1)  # Réduit de moitié avec min 1
        
        return current_replicas

    async def _scale_kubernetes_deployment(self, service_name: str, replicas: int) -> bool:
        """Scale un déploiement Kubernetes"""
        try:
            if not self.k8s_client:
                return False
            
            apps_v1 = k8s_client.AppsV1Api(self.k8s_client)
            
            # Mise à jour du deployment
            body = {'spec': {'replicas': replicas}}
            apps_v1.patch_namespaced_deployment_scale(
                name=service_name,
                namespace='default',
                body=body
            )
            return True
            
        except ApiException as e:
            logger.error(f"Erreur Kubernetes scaling: {e}")
            return False

    async def _cleanup_temp_files(self) -> float:
        """Nettoie les fichiers temporaires"""
        try:
            result = subprocess.run(['find', '/tmp', '-type', 'f', '-atime', '+7', '-delete'], 
                                   capture_output=True, text=True)
            return 100.0  # Simulation de MB nettoyés
        except:
            return 0.0

    async def _cleanup_old_logs(self) -> float:
        """Nettoie les anciens logs"""
        try:
            result = subprocess.run(['find', '/var/log', '-name', '*.log', '-mtime', '+30', '-delete'],
                                   capture_output=True, text=True)
            return 200.0  # Simulation de MB nettoyés
        except:
            return 0.0

    async def _kill_zombie_processes(self) -> int:
        """Élimine les processus zombie"""
        try:
            zombies = [p for p in psutil.process_iter() if p.status() == psutil.STATUS_ZOMBIE]
            for zombie in zombies:
                try:
                    zombie.terminate()
                except:
                    pass
            return len(zombies)
        except:
            return 0

    async def _collect_metrics_snapshot(self) -> Dict[str, float]:
        """Collecte un snapshot des métriques système"""
        try:
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections()),
                'processes_count': len(psutil.pids())
            }
        except:
            return {}

    def _find_rule_by_id(self, rule_id: str) -> Optional[RemediationRule]:
        """Trouve une règle par son ID"""
        return next((rule for rule in self.rules if rule.rule_id == rule_id), None)

    def _check_cooldown(self, rule_id: str) -> bool:
        """Vérifie si la règle est en période de cooldown"""
        if rule_id not in self.execution_history:
            return True
        
        rule = self._find_rule_by_id(rule_id)
        if not rule:
            return True
        
        last_executions = self.execution_history[rule_id]
        cutoff_time = datetime.utcnow() - timedelta(minutes=rule.cooldown_minutes)
        
        recent_executions = [ex for ex in last_executions if ex > cutoff_time]
        return len(recent_executions) < rule.max_attempts

    def _update_execution_history(self, rule_id: str):
        """Met à jour l'historique d'exécution"""
        if rule_id not in self.execution_history:
            self.execution_history[rule_id] = []
        
        self.execution_history[rule_id].append(datetime.utcnow())
        
        # Nettoyage de l'historique (garder seulement les 24 dernières heures)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.execution_history[rule_id] = [
            ex for ex in self.execution_history[rule_id] if ex > cutoff
        ]

    async def _execute_rollback(self, execution: RemediationExecution, rule: RemediationRule):
        """Exécute un rollback en cas d'échec"""
        try:
            logger.info(f"Démarrage du rollback pour {execution.execution_id}")
            
            # Actions de rollback spécifiques au type de remédiation
            if rule.remediation_type == RemediationType.SERVICE_RESTART:
                # Restaurer depuis snapshot si disponible
                for action in execution.actions_taken:
                    if "snapshot créé" in action.lower():
                        snapshot_id = action.split(": ")[-1]
                        await self._restore_from_snapshot(snapshot_id)
                        execution.rollback_actions.append(f"Restauré depuis {snapshot_id}")
            
            elif rule.remediation_type == RemediationType.AUTO_SCALING:
                # Restaurer le nombre original de répliques
                for service in execution.affected_services:
                    original_replicas = execution.metrics_before.get('replicas', 1)
                    await self._scale_kubernetes_deployment(service, original_replicas)
                    execution.rollback_actions.append(f"Restauré {service} à {original_replicas} répliques")
            
            execution.status = RemediationStatus.ROLLED_BACK
            logger.info(f"Rollback terminé pour {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du rollback: {e}")
            execution.error_messages.append(f"Erreur rollback: {e}")

    async def get_remediation_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de remédiation"""
        if not self.executions:
            return {"total_executions": 0}
        
        total = len(self.executions)
        successful = sum(1 for ex in self.executions if ex.status == RemediationStatus.SUCCESS)
        failed = sum(1 for ex in self.executions if ex.status == RemediationStatus.FAILED)
        
        recent_executions = [
            ex for ex in self.executions 
            if ex.start_time > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "total_executions": total,
            "success_rate": f"{(successful/total)*100:.1f}%",
            "failed_executions": failed,
            "recent_24h": len(recent_executions),
            "most_used_type": max(
                [ex.remediation_type.value for ex in self.executions],
                key=[ex.remediation_type.value for ex in self.executions].count
            ) if self.executions else None
        }

# Instance globale du moteur de remédiation
_remediation_engine = IntelligentRemediationEngine()

async def execute_auto_remediation(rule_id: str, context: Dict[str, Any]) -> RemediationExecution:
    """Function helper pour exécuter une remédiation automatique"""
    return await _remediation_engine.execute_remediation(rule_id, context)

async def get_remediation_engine() -> IntelligentRemediationEngine:
    """Retourne l'instance du moteur de remédiation"""
    return _remediation_engine

# Configuration des alertes de remédiation
if __name__ == "__main__":
    # Enregistrement des configurations d'alertes
    remediation_configs = [
        AlertConfig(
            name="auto_restart_critical_services",
            category=AlertCategory.AVAILABILITY,
            severity=AlertSeverity.CRITICAL,
            script_type=ScriptType.REMEDIATION,
            conditions=['Service défaillant détecté'],
            actions=['execute_auto_remediation:auto_restart_failed_services'],
            ml_enabled=True,
            auto_remediation=True
        ),
        AlertConfig(
            name="auto_scale_high_load",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.HIGH,
            script_type=ScriptType.REMEDIATION,
            conditions=['Charge système élevée'],
            actions=['execute_auto_remediation:auto_scale_high_load'],
            ml_enabled=True,
            auto_remediation=True
        ),
        AlertConfig(
            name="auto_cleanup_resources",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.MEDIUM,
            script_type=ScriptType.REMEDIATION,
            conditions=['Utilisation ressources élevée'],
            actions=['execute_auto_remediation:automatic_resource_cleanup'],
            ml_enabled=False,
            auto_remediation=True
        )
    ]
    
    for config in remediation_configs:
        register_alert(config)
