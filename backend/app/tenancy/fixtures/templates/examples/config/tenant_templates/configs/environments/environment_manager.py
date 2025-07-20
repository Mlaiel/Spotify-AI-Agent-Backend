"""
Enterprise Environment Manager for Multi-Tenant Configuration

Ce module fournit un gestionnaire d'environnements de niveau enterprise
pour la gestion avancée des configurations multi-tenant.

Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager
import weakref

from .config_loader import ConfigLoader, get_config_loader
from .config_validator import ConfigValidator, ValidationResult

logger = logging.getLogger(__name__)


class EnvironmentStatus(str, Enum):
    """États possibles d'un environnement."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    FAILED = "failed"


class EnvironmentTransition(str, Enum):
    """Transitions possibles entre environnements."""
    DEPLOY = "deploy"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    MAINTENANCE_MODE = "maintenance_mode"
    RESTORE = "restore"


@dataclass
class EnvironmentMetrics:
    """Métriques d'un environnement."""
    
    last_deployed: Optional[datetime] = None
    last_validated: Optional[datetime] = None
    config_load_time: Optional[float] = None
    validation_score: float = 0.0
    active_connections: int = 0
    request_count: int = 0
    error_count: int = 0
    uptime: timedelta = field(default_factory=lambda: timedelta())
    
    def update_request_metrics(self, success: bool = True) -> None:
        """Met à jour les métriques de requêtes."""
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    @property
    def error_rate(self) -> float:
        """Calcule le taux d'erreur."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    @property
    def success_rate(self) -> float:
        """Calcule le taux de succès."""
        return 1.0 - self.error_rate


@dataclass
class Environment:
    """Représentation d'un environnement."""
    
    name: str
    type: str
    config: Dict[str, Any]
    status: EnvironmentStatus = EnvironmentStatus.INACTIVE
    metrics: EnvironmentMetrics = field(default_factory=EnvironmentMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialisation."""
        self.updated_at = datetime.now()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Met à jour la configuration."""
        self.config = new_config
        self.updated_at = datetime.now()
    
    def activate(self) -> None:
        """Active l'environnement."""
        self.status = EnvironmentStatus.ACTIVE
        self.updated_at = datetime.now()
    
    def deactivate(self) -> None:
        """Désactive l'environnement."""
        self.status = EnvironmentStatus.INACTIVE
        self.updated_at = datetime.now()
    
    def set_maintenance_mode(self, enabled: bool = True) -> None:
        """Active/désactive le mode maintenance."""
        if enabled:
            self.status = EnvironmentStatus.MAINTENANCE
        else:
            self.status = EnvironmentStatus.ACTIVE
        self.updated_at = datetime.now()
    
    def is_healthy(self) -> bool:
        """Vérifie si l'environnement est en bonne santé."""
        return (
            self.status == EnvironmentStatus.ACTIVE and
            self.metrics.error_rate < 0.05 and  # Moins de 5% d'erreurs
            self.metrics.validation_score > 80.0
        )


class EnvironmentManager:
    """Gestionnaire d'environnements enterprise."""
    
    def __init__(self, 
                 config_root: Union[str, Path],
                 cache_ttl: timedelta = timedelta(minutes=15),
                 enable_auto_validation: bool = True,
                 enable_metrics: bool = True):
        """
        Initialise le gestionnaire d'environnements.
        
        Args:
            config_root: Répertoire racine des configurations
            cache_ttl: Durée de vie du cache
            enable_auto_validation: Active la validation automatique
            enable_metrics: Active la collecte de métriques
        """
        self.config_root = Path(config_root)
        self.cache_ttl = cache_ttl
        self.enable_auto_validation = enable_auto_validation
        self.enable_metrics = enable_metrics
        
        # État interne
        self._environments: Dict[str, Environment] = {}
        self._config_loader = get_config_loader(
            cache_ttl_minutes=int(cache_ttl.total_seconds() / 60)
        )
        self._config_validator = ConfigValidator() if enable_auto_validation else None
        self._lock = threading.RLock()
        self._observers: List[Callable[[str, Environment], None]] = []
        self._transition_handlers: Dict[EnvironmentTransition, List[Callable]] = {}
        
        # Initialisation
        self._initialize_environments()
    
    def _initialize_environments(self) -> None:
        """Initialise tous les environnements disponibles."""
        if not self.config_root.exists():
            logger.warning(f"Répertoire de configuration non trouvé: {self.config_root}")
            return
        
        for env_dir in self.config_root.iterdir():
            if env_dir.is_dir() and not env_dir.name.startswith('.'):
                try:
                    self._load_environment(env_dir.name)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de l'environnement {env_dir.name}: {e}")
    
    def _load_environment(self, env_name: str) -> Environment:
        """Charge un environnement spécifique."""
        env_dir = self.config_root / env_name
        config_file = env_dir / f"{env_name}.yml"
        
        if not config_file.exists():
            config_file = env_dir / f"{env_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé pour {env_name}")
        
        # Chargement des overrides
        overrides = []
        overrides_dir = env_dir / "overrides"
        if overrides_dir.exists():
            for override_file in overrides_dir.glob("*.yml"):
                overrides.append(override_file)
            for override_file in overrides_dir.glob("*.yaml"):
                overrides.append(override_file)
        
        # Chargement de la configuration
        start_time = datetime.now()
        config = self._config_loader.load_config(config_file, overrides)
        load_time = (datetime.now() - start_time).total_seconds()
        
        # Validation si activée
        validation_score = 0.0
        if self.enable_auto_validation and self._config_validator:
            validation_result = self._config_validator.validate_config(config, env_name)
            validation_score = validation_result.score
            
            if not validation_result.is_valid:
                logger.warning(f"Validation échouée pour {env_name}: {validation_result.errors}")
        
        # Création de l'environnement
        env = Environment(
            name=env_name,
            type=config.get("metadata", {}).get("environment_type", env_name),
            config=config,
            status=EnvironmentStatus.ACTIVE
        )
        
        # Mise à jour des métriques
        if self.enable_metrics:
            env.metrics.config_load_time = load_time
            env.metrics.validation_score = validation_score
            env.metrics.last_validated = datetime.now()
        
        with self._lock:
            self._environments[env_name] = env
        
        # Notification des observers
        self._notify_observers(env_name, env)
        
        logger.info(f"Environnement chargé: {env_name} (score: {validation_score:.1f})")
        return env
    
    def get_environment(self, name: str) -> Optional[Environment]:
        """Récupère un environnement par nom."""
        with self._lock:
            return self._environments.get(name)
    
    def get_all_environments(self) -> Dict[str, Environment]:
        """Récupère tous les environnements."""
        with self._lock:
            return self._environments.copy()
    
    def get_active_environments(self) -> Dict[str, Environment]:
        """Récupère tous les environnements actifs."""
        with self._lock:
            return {
                name: env for name, env in self._environments.items()
                if env.status == EnvironmentStatus.ACTIVE
            }
    
    def reload_environment(self, name: str, force: bool = False) -> bool:
        """
        Recharge un environnement.
        
        Args:
            name: Nom de l'environnement
            force: Force le rechargement même si le cache est valide
            
        Returns:
            True si rechargé avec succès
        """
        try:
            if force:
                self._config_loader.clear_cache()
            
            old_env = self._environments.get(name)
            new_env = self._load_environment(name)
            
            # Comparaison des configurations
            if old_env and old_env.config != new_env.config:
                logger.info(f"Configuration mise à jour pour {name}")
                self._trigger_transition(name, EnvironmentTransition.DEPLOY)
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors du rechargement de {name}: {e}")
            return False
    
    def validate_environment(self, name: str) -> Optional[ValidationResult]:
        """Valide un environnement spécifique."""
        env = self.get_environment(name)
        if not env or not self._config_validator:
            return None
        
        result = self._config_validator.validate_config(env.config, name)
        
        # Mise à jour des métriques
        if self.enable_metrics:
            env.metrics.validation_score = result.score
            env.metrics.last_validated = datetime.now()
        
        return result
    
    def validate_all_environments(self) -> Dict[str, ValidationResult]:
        """Valide tous les environnements."""
        results = {}
        for name in self._environments.keys():
            result = self.validate_environment(name)
            if result:
                results[name] = result
        return results
    
    def promote_environment(self, source: str, target: str) -> bool:
        """
        Promeut une configuration d'un environnement vers un autre.
        
        Args:
            source: Environnement source
            target: Environnement cible
            
        Returns:
            True si promotion réussie
        """
        source_env = self.get_environment(source)
        target_env = self.get_environment(target)
        
        if not source_env or not target_env:
            logger.error(f"Environnement non trouvé: {source} ou {target}")
            return False
        
        try:
            # Sauvegarde de la configuration actuelle
            self._backup_environment(target)
            
            # Copie de la configuration
            target_env.update_config(source_env.config.copy())
            
            # Validation de la nouvelle configuration
            if self.enable_auto_validation and self._config_validator:
                result = self._config_validator.validate_config(target_env.config, target)
                if not result.is_valid:
                    # Rollback en cas d'échec
                    self._restore_environment_backup(target)
                    logger.error(f"Promotion échouée: validation invalide pour {target}")
                    return False
            
            # Trigger transition
            self._trigger_transition(target, EnvironmentTransition.PROMOTE)
            
            logger.info(f"Promotion réussie: {source} -> {target}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la promotion {source} -> {target}: {e}")
            # Tentative de rollback
            self._restore_environment_backup(target)
            return False
    
    def rollback_environment(self, name: str, to_version: str = None) -> bool:
        """
        Effectue un rollback d'un environnement.
        
        Args:
            name: Nom de l'environnement
            to_version: Version cible (optionnel)
            
        Returns:
            True si rollback réussi
        """
        try:
            success = self._restore_environment_backup(name, to_version)
            if success:
                self._trigger_transition(name, EnvironmentTransition.ROLLBACK)
                logger.info(f"Rollback réussi pour {name}")
            return success
        except Exception as e:
            logger.error(f"Erreur lors du rollback de {name}: {e}")
            return False
    
    def set_maintenance_mode(self, name: str, enabled: bool = True) -> bool:
        """
        Active/désactive le mode maintenance pour un environnement.
        
        Args:
            name: Nom de l'environnement
            enabled: Active le mode maintenance
            
        Returns:
            True si succès
        """
        env = self.get_environment(name)
        if not env:
            return False
        
        env.set_maintenance_mode(enabled)
        
        transition = EnvironmentTransition.MAINTENANCE_MODE if enabled else EnvironmentTransition.RESTORE
        self._trigger_transition(name, transition)
        
        action = "activé" if enabled else "désactivé"
        logger.info(f"Mode maintenance {action} pour {name}")
        return True
    
    def get_environment_health(self, name: str) -> Dict[str, Any]:
        """
        Récupère l'état de santé d'un environnement.
        
        Args:
            name: Nom de l'environnement
            
        Returns:
            Dictionnaire de l'état de santé
        """
        env = self.get_environment(name)
        if not env:
            return {"healthy": False, "reason": "Environment not found"}
        
        is_healthy = env.is_healthy()
        
        return {
            "healthy": is_healthy,
            "status": env.status.value,
            "metrics": {
                "validation_score": env.metrics.validation_score,
                "error_rate": env.metrics.error_rate,
                "success_rate": env.metrics.success_rate,
                "uptime": str(env.metrics.uptime),
                "last_validated": env.metrics.last_validated.isoformat() if env.metrics.last_validated else None
            },
            "last_updated": env.updated_at.isoformat()
        }
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Récupère l'état de santé de tous les environnements."""
        environments_health = {}
        total_environments = 0
        healthy_environments = 0
        
        for name, env in self._environments.items():
            health = self.get_environment_health(name)
            environments_health[name] = health
            total_environments += 1
            if health["healthy"]:
                healthy_environments += 1
        
        cluster_healthy = healthy_environments / max(total_environments, 1) >= 0.8
        
        return {
            "cluster_healthy": cluster_healthy,
            "total_environments": total_environments,
            "healthy_environments": healthy_environments,
            "health_ratio": healthy_environments / max(total_environments, 1),
            "environments": environments_health
        }
    
    def register_observer(self, callback: Callable[[str, Environment], None]) -> None:
        """Enregistre un observer pour les changements d'environnement."""
        self._observers.append(callback)
    
    def register_transition_handler(self, 
                                  transition: EnvironmentTransition,
                                  handler: Callable[[str, Environment], None]) -> None:
        """Enregistre un handler pour une transition spécifique."""
        if transition not in self._transition_handlers:
            self._transition_handlers[transition] = []
        self._transition_handlers[transition].append(handler)
    
    def _notify_observers(self, env_name: str, env: Environment) -> None:
        """Notifie tous les observers d'un changement."""
        for observer in self._observers:
            try:
                observer(env_name, env)
            except Exception as e:
                logger.error(f"Erreur dans observer: {e}")
    
    def _trigger_transition(self, env_name: str, transition: EnvironmentTransition) -> None:
        """Déclenche les handlers pour une transition."""
        env = self.get_environment(env_name)
        if not env:
            return
        
        handlers = self._transition_handlers.get(transition, [])
        for handler in handlers:
            try:
                handler(env_name, env)
            except Exception as e:
                logger.error(f"Erreur dans transition handler {transition}: {e}")
    
    def _backup_environment(self, name: str) -> None:
        """Sauvegarde la configuration d'un environnement."""
        # Implémentation de la sauvegarde
        # Pour l'instant, on simule
        logger.debug(f"Sauvegarde de l'environnement {name}")
    
    def _restore_environment_backup(self, name: str, version: str = None) -> bool:
        """Restaure une sauvegarde d'environnement."""
        # Implémentation de la restauration
        # Pour l'instant, on simule
        logger.debug(f"Restauration de l'environnement {name} (version: {version})")
        return True
    
    @contextmanager
    def temporary_environment(self, name: str, config: Dict[str, Any]):
        """
        Crée un environnement temporaire pour tests.
        
        Args:
            name: Nom de l'environnement temporaire
            config: Configuration à utiliser
        """
        temp_env = Environment(
            name=name,
            type="temporary",
            config=config,
            status=EnvironmentStatus.ACTIVE
        )
        
        with self._lock:
            original_env = self._environments.get(name)
            self._environments[name] = temp_env
        
        try:
            yield temp_env
        finally:
            with self._lock:
                if original_env:
                    self._environments[name] = original_env
                else:
                    del self._environments[name]
    
    async def async_reload_all_environments(self) -> Dict[str, bool]:
        """Recharge tous les environnements de manière asynchrone."""
        tasks = []
        env_names = list(self._environments.keys())
        
        async def reload_single(env_name: str) -> tuple[str, bool]:
            # Simulation asynchrone du rechargement
            await asyncio.sleep(0.1)  # Simule I/O
            return env_name, self.reload_environment(env_name)
        
        tasks = [reload_single(name) for name in env_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {name: success for name, success in results if not isinstance(success, Exception)}
    
    def export_environment_config(self, name: str, format: str = "yaml") -> Optional[str]:
        """
        Exporte la configuration d'un environnement.
        
        Args:
            name: Nom de l'environnement
            format: Format d'export (yaml, json)
            
        Returns:
            Configuration sérialisée
        """
        env = self.get_environment(name)
        if not env:
            return None
        
        if format.lower() == "yaml":
            import yaml
            return yaml.dump(env.config, default_flow_style=False, sort_keys=True)
        elif format.lower() == "json":
            import json
            return json.dumps(env.config, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Format non supporté: {format}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire."""
        cache_stats = self._config_loader.get_cache_stats()
        
        return {
            "total_environments": len(self._environments),
            "active_environments": len(self.get_active_environments()),
            "cache_stats": cache_stats,
            "validation_enabled": self.enable_auto_validation,
            "metrics_enabled": self.enable_metrics,
            "observers_count": len(self._observers),
            "transition_handlers_count": sum(len(handlers) for handlers in self._transition_handlers.values())
        }


# Instance globale du gestionnaire (singleton pattern)
_manager_instance: Optional[EnvironmentManager] = None
_manager_lock = threading.Lock()


def get_environment_manager(config_root: Union[str, Path] = None, **kwargs) -> EnvironmentManager:
    """
    Récupère l'instance globale du gestionnaire d'environnements.
    
    Args:
        config_root: Répertoire racine des configurations
        **kwargs: Arguments additionnels pour l'initialisation
        
    Returns:
        Instance du gestionnaire
    """
    global _manager_instance
    
    with _manager_lock:
        if _manager_instance is None:
            if config_root is None:
                config_root = Path(__file__).parent
            _manager_instance = EnvironmentManager(config_root, **kwargs)
        
        return _manager_instance


def reset_environment_manager() -> None:
    """Réinitialise l'instance globale du gestionnaire."""
    global _manager_instance
    with _manager_lock:
        _manager_instance = None


__all__ = [
    "EnvironmentStatus",
    "EnvironmentTransition",
    "EnvironmentMetrics",
    "Environment",
    "EnvironmentManager",
    "get_environment_manager",
    "reset_environment_manager"
]
