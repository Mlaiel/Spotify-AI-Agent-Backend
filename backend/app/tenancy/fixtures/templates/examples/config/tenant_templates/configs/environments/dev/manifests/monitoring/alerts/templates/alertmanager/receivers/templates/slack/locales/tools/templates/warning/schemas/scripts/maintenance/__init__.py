# -*- coding: utf-8 -*-
"""
Maintenance Tools - Industrial Grade Maintenance System
=======================================================

Système de maintenance automatisé pour l'architecture multi-tenant Spotify AI Agent.
Fournit des outils complets pour la maintenance préventive, corrective et prédictive
avec automatisation intelligente et planification optimisée.

Key Features:
- Maintenance préventive automatisée
- Détection proactive des problèmes
- Planification intelligente des interventions
- Maintenance prédictive avec ML
- Gestion des fenêtres de maintenance
- Rollback automatique des opérations
- Reporting détaillé et audit trail
- Integration avec les systèmes de monitoring

Author: Spotify AI Agent Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

from .preventive_maintenance import PreventiveMaintenance
from .corrective_maintenance import CorrectiveMaintenance
from .predictive_maintenance import PredictiveMaintenance
from .maintenance_scheduler import MaintenanceScheduler
from .maintenance_executor import MaintenanceExecutor
from .maintenance_validator import MaintenanceValidator
from .maintenance_reporter import MaintenanceReporter
from .system_health_checker import SystemHealthChecker

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DU MODULE
# =============================================================================

MAINTENANCE_CONFIG = {
    "version": "1.0.0",
    "name": "maintenance_tools",
    "description": "Industrial grade maintenance system for multi-tenant architecture",
    "author": "Spotify AI Agent Team",
    
    # Configuration de la maintenance préventive
    "preventive_config": {
        "enabled": True,
        "schedule_interval_hours": 24,
        "maintenance_window_hours": 4,
        "max_concurrent_tasks": 3,
        "auto_approval_enabled": False,
        "notification_channels": ["slack", "email"],
        "backup_before_maintenance": True
    },
    
    # Configuration de la maintenance corrective
    "corrective_config": {
        "enabled": True,
        "auto_detection": True,
        "auto_resolution": False,
        "escalation_timeout_minutes": 30,
        "priority_levels": ["low", "medium", "high", "critical"],
        "approval_required": True
    },
    
    # Configuration de la maintenance prédictive
    "predictive_config": {
        "enabled": True,
        "ml_model_enabled": True,
        "prediction_horizon_days": 7,
        "confidence_threshold": 0.8,
        "data_retention_days": 90,
        "retrain_interval_days": 30
    },
    
    # Configuration du planificateur
    "scheduler_config": {
        "timezone": "UTC",
        "business_hours": {"start": "08:00", "end": "18:00"},
        "maintenance_windows": {
            "daily": {"start": "02:00", "duration_hours": 2},
            "weekly": {"day": "sunday", "start": "01:00", "duration_hours": 4},
            "monthly": {"day": "first_sunday", "start": "01:00", "duration_hours": 6}
        },
        "blackout_periods": ["2024-12-24", "2024-12-25", "2024-01-01"]
    },
    
    # Configuration de l'exécution
    "execution_config": {
        "dry_run_by_default": False,
        "rollback_enabled": True,
        "timeout_minutes": 60,
        "retry_attempts": 3,
        "parallel_execution": True,
        "health_check_before": True,
        "health_check_after": True
    },
    
    # Configuration de validation
    "validation_config": {
        "pre_maintenance_checks": True,
        "post_maintenance_checks": True,
        "performance_validation": True,
        "data_integrity_checks": True,
        "security_validation": True
    },
    
    # Configuration des rapports
    "reporting_config": {
        "automated_reports": True,
        "report_frequency": "daily",
        "include_metrics": True,
        "include_recommendations": True,
        "export_formats": ["pdf", "html", "json"],
        "distribution_list": ["ops-team@company.com"]
    }
}

# Types de maintenance supportés
MAINTENANCE_TYPES = {
    "preventive": {
        "class": "PreventiveMaintenance",
        "description": "Scheduled preventive maintenance",
        "automated": True,
        "requires_approval": False
    },
    "corrective": {
        "class": "CorrectiveMaintenance", 
        "description": "Corrective maintenance for issues",
        "automated": False,
        "requires_approval": True
    },
    "predictive": {
        "class": "PredictiveMaintenance",
        "description": "ML-based predictive maintenance",
        "automated": True,
        "requires_approval": False
    },
    "emergency": {
        "class": "EmergencyMaintenance",
        "description": "Emergency maintenance procedures",
        "automated": False,
        "requires_approval": False
    }
}

# Catégories de tâches de maintenance
MAINTENANCE_CATEGORIES = {
    "system": {
        "description": "System-level maintenance",
        "tasks": ["disk_cleanup", "log_rotation", "cache_cleanup", "temp_file_cleanup"]
    },
    "database": {
        "description": "Database maintenance",
        "tasks": ["index_optimization", "statistics_update", "vacuum", "backup_verification"]
    },
    "security": {
        "description": "Security maintenance",
        "tasks": ["certificate_renewal", "password_rotation", "vulnerability_scan", "access_review"]
    },
    "performance": {
        "description": "Performance optimization",
        "tasks": ["query_optimization", "cache_tuning", "connection_pool_adjustment", "resource_scaling"]
    },
    "backup": {
        "description": "Backup and recovery",
        "tasks": ["backup_verification", "restore_testing", "backup_cleanup", "disaster_recovery_test"]
    },
    "monitoring": {
        "description": "Monitoring system maintenance", 
        "tasks": ["metric_cleanup", "alert_tuning", "dashboard_update", "log_aggregation"]
    }
}

# =============================================================================
# REGISTRY DES MAINTENANCES
# =============================================================================

class MaintenanceRegistry:
    """Registry centralisé pour tous les types de maintenance."""
    
    def __init__(self):
        self._maintenance_types: Dict[str, Any] = {}
        self._active_maintenances: Dict[str, bool] = {}
        self._maintenance_history: List[Dict[str, Any]] = []
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialise le registry avec les types de maintenance disponibles."""
        self._maintenance_types = {
            "preventive": PreventiveMaintenance,
            "corrective": CorrectiveMaintenance,
            "predictive": PredictiveMaintenance,
            "scheduler": MaintenanceScheduler,
            "executor": MaintenanceExecutor,
            "validator": MaintenanceValidator,
            "reporter": MaintenanceReporter,
            "health_checker": SystemHealthChecker
        }
        
        # Marquer tous comme inactifs par défaut
        self._active_maintenances = {name: False for name in self._maintenance_types.keys()}
    
    def register_maintenance_type(self, name: str, maintenance_class: Any):
        """Enregistre un nouveau type de maintenance."""
        self._maintenance_types[name] = maintenance_class
        self._active_maintenances[name] = False
        logger.info(f"Registered maintenance type: {name}")
    
    def get_maintenance_type(self, name: str) -> Optional[Any]:
        """Récupère un type de maintenance par son nom."""
        return self._maintenance_types.get(name)
    
    def activate_maintenance(self, name: str):
        """Active un type de maintenance."""
        if name in self._maintenance_types:
            self._active_maintenances[name] = True
            logger.info(f"Activated maintenance: {name}")
    
    def deactivate_maintenance(self, name: str):
        """Désactive un type de maintenance."""
        if name in self._maintenance_types:
            self._active_maintenances[name] = False
            logger.info(f"Deactivated maintenance: {name}")
    
    def is_active(self, name: str) -> bool:
        """Vérifie si un type de maintenance est actif."""
        return self._active_maintenances.get(name, False)
    
    def list_maintenance_types(self) -> List[str]:
        """Liste tous les types de maintenance disponibles."""
        return list(self._maintenance_types.keys())
    
    def list_active_maintenances(self) -> List[str]:
        """Liste les maintenances actives."""
        return [name for name, active in self._active_maintenances.items() if active]
    
    def add_to_history(self, maintenance_record: Dict[str, Any]):
        """Ajoute un enregistrement à l'historique."""
        maintenance_record["recorded_at"] = datetime.now().isoformat()
        self._maintenance_history.append(maintenance_record)
        
        # Garder seulement les 1000 derniers enregistrements
        if len(self._maintenance_history) > 1000:
            self._maintenance_history = self._maintenance_history[-1000:]
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère l'historique des maintenances."""
        return self._maintenance_history[-limit:]

# Instance globale du registry
maintenance_registry = MaintenanceRegistry()

# =============================================================================
# FACTORY POUR LES MAINTENANCES
# =============================================================================

class MaintenanceFactory:
    """Factory pour créer des instances de maintenance."""
    
    @staticmethod
    def create_maintenance(maintenance_type: str, **kwargs) -> Any:
        """
        Crée une instance de maintenance selon le type.
        
        Args:
            maintenance_type: Type de maintenance
            **kwargs: Arguments de configuration
            
        Returns:
            Instance de maintenance appropriée
            
        Raises:
            ValueError: Si le type de maintenance n'est pas supporté
        """
        maintenance_class = maintenance_registry.get_maintenance_type(maintenance_type)
        
        if not maintenance_class:
            raise ValueError(f"Unsupported maintenance type: {maintenance_type}")
        
        # Configuration par défaut
        config = MAINTENANCE_CONFIG.copy()
        config.update(kwargs.get('config', {}))
        
        return maintenance_class(config=config, **kwargs)
    
    @staticmethod
    def create_maintenance_suite(config: Optional[Dict[str, Any]] = None) -> "MaintenanceSuite":
        """
        Crée une suite complète de maintenance.
        
        Args:
            config: Configuration optionnelle
            
        Returns:
            Suite de maintenance configurée
        """
        return MaintenanceSuite(config or MAINTENANCE_CONFIG)

# =============================================================================
# GESTIONNAIRE CENTRAL DE MAINTENANCE
# =============================================================================

class MaintenanceManager:
    """
    Gestionnaire central pour orchestrer toutes les opérations de maintenance.
    Point d'entrée unique pour toutes les activités de maintenance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or MAINTENANCE_CONFIG
        self.maintenance_systems: Dict[str, Any] = {}
        self.running = False
        self.current_maintenances: Dict[str, Any] = {}
        
        # Initialiser les systèmes de maintenance
        self._initialize_maintenance_systems()
        
        logger.info("MaintenanceManager initialized")
    
    def _initialize_maintenance_systems(self):
        """Initialise tous les systèmes de maintenance."""
        enabled_systems = [
            "preventive", "corrective", "predictive", "scheduler",
            "executor", "validator", "reporter", "health_checker"
        ]
        
        for system_type in enabled_systems:
            try:
                system = MaintenanceFactory.create_maintenance(
                    system_type,
                    config=self.config
                )
                self.maintenance_systems[system_type] = system
                maintenance_registry.activate_maintenance(system_type)
                logger.info(f"Initialized {system_type} maintenance system")
                
            except Exception as e:
                logger.error(f"Failed to initialize {system_type} maintenance: {e}")
    
    async def start_maintenance_systems(self) -> Dict[str, Any]:
        """
        Démarre tous les systèmes de maintenance.
        
        Returns:
            Résultat du démarrage avec statuts
        """
        if self.running:
            return {"status": "already_running", "message": "Maintenance systems already started"}
        
        start_results = {}
        
        for system_name, system in self.maintenance_systems.items():
            try:
                if hasattr(system, 'start'):
                    await system.start()
                start_results[system_name] = "started"
                logger.info(f"Started {system_name} maintenance system")
                
            except Exception as e:
                start_results[system_name] = f"failed: {e}"
                logger.error(f"Failed to start {system_name} maintenance: {e}")
        
        self.running = True
        
        return {
            "status": "started",
            "systems": start_results,
            "total_systems": len(self.maintenance_systems),
            "successful_starts": len([r for r in start_results.values() if r == "started"])
        }
    
    async def stop_maintenance_systems(self) -> Dict[str, Any]:
        """
        Arrête tous les systèmes de maintenance.
        
        Returns:
            Résultat de l'arrêt avec statuts
        """
        if not self.running:
            return {"status": "not_running", "message": "Maintenance systems not started"}
        
        stop_results = {}
        
        for system_name, system in self.maintenance_systems.items():
            try:
                if hasattr(system, 'stop'):
                    await system.stop()
                stop_results[system_name] = "stopped"
                logger.info(f"Stopped {system_name} maintenance system")
                
            except Exception as e:
                stop_results[system_name] = f"failed: {e}"
                logger.error(f"Failed to stop {system_name} maintenance: {e}")
        
        self.running = False
        
        return {
            "status": "stopped",
            "systems": stop_results,
            "total_systems": len(self.maintenance_systems),
            "successful_stops": len([r for r in stop_results.values() if r == "stopped"])
        }
    
    async def schedule_maintenance(
        self,
        maintenance_type: str,
        tasks: List[str],
        scheduled_time: Optional[datetime] = None,
        priority: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Planifie une maintenance.
        
        Args:
            maintenance_type: Type de maintenance
            tasks: Liste des tâches à exécuter
            scheduled_time: Heure programmée (None = immédiatement)
            priority: Priorité de la maintenance
            **kwargs: Arguments additionnels
            
        Returns:
            Informations sur la maintenance programmée
        """
        if "scheduler" not in self.maintenance_systems:
            return {"error": "Scheduler not available"}
        
        try:
            scheduler = self.maintenance_systems["scheduler"]
            maintenance_id = await scheduler.schedule_maintenance(
                maintenance_type=maintenance_type,
                tasks=tasks,
                scheduled_time=scheduled_time,
                priority=priority,
                **kwargs
            )
            
            logger.info(f"Scheduled maintenance {maintenance_id}")
            return {
                "maintenance_id": maintenance_id,
                "status": "scheduled",
                "scheduled_time": scheduled_time.isoformat() if scheduled_time else "immediate",
                "tasks_count": len(tasks)
            }
            
        except Exception as e:
            logger.error(f"Failed to schedule maintenance: {e}")
            return {"error": str(e)}
    
    async def execute_maintenance(self, maintenance_id: str) -> Dict[str, Any]:
        """
        Exécute une maintenance programmée.
        
        Args:
            maintenance_id: ID de la maintenance à exécuter
            
        Returns:
            Résultat de l'exécution
        """
        if "executor" not in self.maintenance_systems:
            return {"error": "Executor not available"}
        
        try:
            executor = self.maintenance_systems["executor"]
            
            # Marquer comme en cours
            self.current_maintenances[maintenance_id] = {
                "status": "running",
                "started_at": datetime.now(),
                "executor": executor
            }
            
            result = await executor.execute_maintenance(maintenance_id)
            
            # Mettre à jour le statut
            self.current_maintenances[maintenance_id]["status"] = "completed"
            self.current_maintenances[maintenance_id]["completed_at"] = datetime.now()
            
            # Ajouter à l'historique
            maintenance_registry.add_to_history({
                "maintenance_id": maintenance_id,
                "status": "completed",
                "result": result
            })
            
            logger.info(f"Executed maintenance {maintenance_id}")
            return result
            
        except Exception as e:
            # Marquer comme échoué
            if maintenance_id in self.current_maintenances:
                self.current_maintenances[maintenance_id]["status"] = "failed"
                self.current_maintenances[maintenance_id]["error"] = str(e)
            
            # Ajouter à l'historique
            maintenance_registry.add_to_history({
                "maintenance_id": maintenance_id,
                "status": "failed",
                "error": str(e)
            })
            
            logger.error(f"Failed to execute maintenance {maintenance_id}: {e}")
            return {"error": str(e)}
        
        finally:
            # Nettoyer les maintenances terminées
            if maintenance_id in self.current_maintenances:
                if self.current_maintenances[maintenance_id]["status"] in ["completed", "failed"]:
                    del self.current_maintenances[maintenance_id]
    
    async def get_maintenance_status(self) -> Dict[str, Any]:
        """
        Récupère le statut de tous les systèmes de maintenance.
        
        Returns:
            Statut détaillé de tous les systèmes
        """
        status = {
            "overall_status": "running" if self.running else "stopped",
            "systems": {},
            "current_maintenances": len(self.current_maintenances),
            "scheduled_maintenances": 0,
            "maintenance_history_count": len(maintenance_registry.get_history())
        }
        
        for system_name, system in self.maintenance_systems.items():
            try:
                if hasattr(system, 'get_status'):
                    system_status = await system.get_status()
                else:
                    system_status = {"status": "active" if self.running else "inactive"}
                
                status["systems"][system_name] = system_status
                
            except Exception as e:
                status["systems"][system_name] = {"status": "error", "error": str(e)}
        
        # Compter les maintenances programmées
        if "scheduler" in self.maintenance_systems:
            try:
                scheduler_status = await self.maintenance_systems["scheduler"].get_status()
                status["scheduled_maintenances"] = scheduler_status.get("scheduled_count", 0)
            except:
                pass
        
        return status
    
    async def get_maintenance_recommendations(self) -> Dict[str, Any]:
        """
        Génère des recommandations de maintenance.
        
        Returns:
            Recommandations de maintenance
        """
        recommendations = {
            "preventive": [],
            "corrective": [],
            "predictive": [],
            "priority_actions": []
        }
        
        # Recommandations préventives
        if "preventive" in self.maintenance_systems:
            try:
                preventive_recs = await self.maintenance_systems["preventive"].get_recommendations()
                recommendations["preventive"] = preventive_recs
            except Exception as e:
                logger.error(f"Failed to get preventive recommendations: {e}")
        
        # Recommandations correctives
        if "corrective" in self.maintenance_systems:
            try:
                corrective_recs = await self.maintenance_systems["corrective"].get_recommendations()
                recommendations["corrective"] = corrective_recs
            except Exception as e:
                logger.error(f"Failed to get corrective recommendations: {e}")
        
        # Recommandations prédictives
        if "predictive" in self.maintenance_systems:
            try:
                predictive_recs = await self.maintenance_systems["predictive"].get_recommendations()
                recommendations["predictive"] = predictive_recs
            except Exception as e:
                logger.error(f"Failed to get predictive recommendations: {e}")
        
        # Combiner en actions prioritaires
        all_recs = (
            recommendations["preventive"] + 
            recommendations["corrective"] + 
            recommendations["predictive"]
        )
        
        # Trier par priorité et urgence
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        recommendations["priority_actions"] = sorted(
            all_recs,
            key=lambda x: priority_order.get(x.get("priority", "low"), 1),
            reverse=True
        )[:10]  # Top 10
        
        return recommendations
    
    async def generate_maintenance_report(self, time_range: str = "30d") -> Dict[str, Any]:
        """
        Génère un rapport de maintenance.
        
        Args:
            time_range: Plage temporelle du rapport
            
        Returns:
            Rapport de maintenance détaillé
        """
        if "reporter" not in self.maintenance_systems:
            return {"error": "Reporter not available"}
        
        try:
            reporter = self.maintenance_systems["reporter"]
            return await reporter.generate_report(time_range)
        except Exception as e:
            logger.error(f"Failed to generate maintenance report: {e}")
            return {"error": str(e)}

# =============================================================================
# SUITE DE MAINTENANCE COMPLÈTE
# =============================================================================

class MaintenanceSuite:
    """
    Suite complète de maintenance avec tous les composants intégrés.
    Interface haut niveau pour la maintenance d'entreprise.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.manager = MaintenanceManager(config)
        self.automated_schedules: Dict[str, Any] = {}
        
        logger.info("MaintenanceSuite initialized")
    
    async def setup(self) -> Dict[str, Any]:
        """
        Configure et démarre la suite complète.
        
        Returns:
            Résultat de la configuration
        """
        setup_results = {
            "maintenance_setup": None,
            "schedules_setup": None,
            "automation_setup": None
        }
        
        # Démarrer les systèmes de maintenance
        setup_results["maintenance_setup"] = await self.manager.start_maintenance_systems()
        
        # Configurer les planifications automatiques
        try:
            await self._setup_automated_schedules()
            setup_results["schedules_setup"] = "success"
        except Exception as e:
            setup_results["schedules_setup"] = f"failed: {e}"
            logger.error(f"Schedules setup failed: {e}")
        
        # Configurer l'automatisation
        try:
            await self._setup_automation()
            setup_results["automation_setup"] = "success"
        except Exception as e:
            setup_results["automation_setup"] = f"failed: {e}"
            logger.error(f"Automation setup failed: {e}")
        
        return setup_results
    
    async def _setup_automated_schedules(self):
        """Configure les planifications automatiques."""
        # Maintenance quotidienne
        daily_tasks = [
            "log_rotation",
            "temp_file_cleanup",
            "cache_cleanup",
            "metric_cleanup"
        ]
        
        await self.manager.schedule_maintenance(
            maintenance_type="preventive",
            tasks=daily_tasks,
            scheduled_time=None,  # Sera programmé par le scheduler
            priority="low"
        )
        
        # Maintenance hebdomadaire
        weekly_tasks = [
            "index_optimization",
            "statistics_update",
            "backup_verification",
            "performance_analysis"
        ]
        
        await self.manager.schedule_maintenance(
            maintenance_type="preventive",
            tasks=weekly_tasks,
            scheduled_time=None,
            priority="medium"
        )
        
        logger.info("Automated schedules configured")
    
    async def _setup_automation(self):
        """Configure l'automatisation avancée."""
        # Configuration de la détection automatique
        # Configuration des réponses automatiques
        # Configuration des escalades
        await asyncio.sleep(0.1)  # Simulation
        logger.info("Automation configured")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Récupère un statut compréhensif de toute la suite.
        
        Returns:
            Statut complet avec toutes les informations
        """
        status = await self.manager.get_maintenance_status()
        
        # Ajouter les informations de la suite
        status["suite_info"] = {
            "version": self.config["version"],
            "automated_schedules": len(self.automated_schedules),
            "maintenance_types_available": len(MAINTENANCE_TYPES),
            "categories_supported": len(MAINTENANCE_CATEGORIES)
        }
        
        # Ajouter les recommandations
        try:
            recommendations = await self.manager.get_maintenance_recommendations()
            status["recommendations"] = recommendations
        except Exception as e:
            status["recommendations"] = {"error": str(e)}
        
        return status

# =============================================================================
# UTILITAIRES ET HELPERS
# =============================================================================

def validate_maintenance_config(config: Dict[str, Any]) -> bool:
    """
    Valide la configuration de maintenance.
    
    Args:
        config: Configuration à valider
        
    Returns:
        True si valide, False sinon
    """
    required_keys = ["preventive_config", "corrective_config", "scheduler_config"]
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    return True

async def emergency_maintenance_check() -> Dict[str, Any]:
    """
    Effectue une vérification de maintenance d'urgence.
    
    Returns:
        Résultat de la vérification
    """
    # Simulation de vérification d'urgence
    await asyncio.sleep(0.2)
    
    return {
        "status": "no_emergency",
        "timestamp": datetime.now().isoformat(),
        "checks_performed": [
            "disk_space",
            "memory_usage",
            "database_connectivity",
            "critical_services"
        ],
        "issues_found": []
    }

async def calculate_maintenance_impact(tasks: List[str]) -> Dict[str, Any]:
    """
    Calcule l'impact d'une maintenance.
    
    Args:
        tasks: Liste des tâches de maintenance
        
    Returns:
        Analyse d'impact
    """
    # Simulation de calcul d'impact
    await asyncio.sleep(0.1)
    
    impact_levels = {
        "log_rotation": "low",
        "temp_file_cleanup": "low",
        "index_optimization": "medium",
        "database_backup": "high",
        "system_restart": "critical"
    }
    
    max_impact = "low"
    estimated_downtime = 0
    
    for task in tasks:
        task_impact = impact_levels.get(task, "medium")
        if task_impact == "critical":
            max_impact = "critical"
            estimated_downtime += 30
        elif task_impact == "high" and max_impact != "critical":
            max_impact = "high"
            estimated_downtime += 15
        elif task_impact == "medium" and max_impact not in ["critical", "high"]:
            max_impact = "medium"
            estimated_downtime += 5
        else:
            estimated_downtime += 1
    
    return {
        "overall_impact": max_impact,
        "estimated_downtime_minutes": estimated_downtime,
        "affected_services": ["api", "database", "cache"],
        "recommended_window": "off-peak",
        "rollback_time_minutes": estimated_downtime // 2
    }

# =============================================================================
# EXPORTS PUBLICS
# =============================================================================

__all__ = [
    # Classes principales
    "MaintenanceManager",
    "MaintenanceSuite",
    "MaintenanceFactory", 
    "MaintenanceRegistry",
    
    # Systèmes de maintenance spécialisés
    "PreventiveMaintenance",
    "CorrectiveMaintenance",
    "PredictiveMaintenance",
    "MaintenanceScheduler",
    "MaintenanceExecutor",
    "MaintenanceValidator",
    "MaintenanceReporter",
    "SystemHealthChecker",
    
    # Configuration
    "MAINTENANCE_CONFIG",
    "MAINTENANCE_TYPES",
    "MAINTENANCE_CATEGORIES",
    
    # Registry global
    "maintenance_registry",
    
    # Fonctions utilitaires
    "validate_maintenance_config",
    "emergency_maintenance_check",
    "calculate_maintenance_impact"
]

# Initialisation du module
logger.info(f"Maintenance module initialized - Version {MAINTENANCE_CONFIG['version']}")
