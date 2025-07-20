"""
Module Core - Système de Tenancy Avancé
Auteur: Fahed Mlaiel - Lead Dev & Architecte IA

Ce module fournit l'infrastructure centrale pour le système de tenancy multi-tenant
avec des fonctionnalités avancées de gestion, sécurité, monitoring et orchestration.

Composants principaux:
- Configuration et gestion centralisée
- Système d'alertes intelligent
- Moteur de templates avec localisation
- Gestionnaire de sécurité avancé
- Système de cache distribué
- Métriques et monitoring en temps réel
- Validation avancée des données
- Moteur de workflow
- Bus d'événements asynchrone
"""

from .config import (
    ConfigManager,
    TenantConfig,
    SecurityConfig,
    CacheConfig,
    MonitoringConfig,
    config_manager
)

from .alerts import (
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    AlertHandler,
    EmailAlertHandler,
    SlackAlertHandler,
    WebhookAlertHandler,
    alert_manager
)

from .templates import (
    TemplateEngine,
    Template,
    TemplateContext,
    TemplateType,
    LocalizationManager,
    template_engine,
    localization_manager
)

from .security import (
    SecurityManager,
    SecurityPolicy,
    PermissionLevel,
    AccessControl,
    EncryptionManager,
    AuditLogger,
    security_manager
)

from .cache import (
    CacheManager,
    CacheKey,
    CacheStrategy,
    CacheLevel,
    SerializationFormat,
    CacheEntry,
    CacheBackend,
    MemoryCacheBackend,
    RedisCacheBackend,
    DistributedCache,
    CacheSerializer,
    cache_manager,
    cache_result,
    cache_key_for_tenant,
    cache_key_for_user
)

from .metrics import (
    MetricsCollector,
    TenancyMetrics,
    SystemMetrics,
    MetricsAggregator,
    MetricsExporter,
    MetricDefinition,
    MetricValue,
    metrics_collector,
    tenancy_metrics,
    system_metrics
)

from .validation import (
    BaseValidator,
    TenantConfigValidator,
    SchemaValidator,
    ValidationRule,
    ValidationResult,
    ValidationError,
    ValidationType,
    ValidationSeverity,
    validate_with_decorator,
    tenant_validator,
    schema_validator
)

from .workflow import (
    Workflow,
    WorkflowEngine,
    BaseTask,
    TenantProvisioningTask,
    TenantValidationTask,
    NotificationTask,
    ConditionalTask,
    ParallelTask,
    WorkflowContext,
    TaskResult,
    WorkflowStatus,
    TaskStatus,
    workflow_engine
)

from .events import (
    Event,
    EventBus,
    EventHandler,
    TenantEventHandler,
    APIEventHandler,
    NotificationHandler,
    EventType,
    EventPriority,
    EventStatus,
    event_bus,
    publish_tenant_created,
    publish_api_request,
    publish_system_alert
)

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__description__ = "Système de tenancy multi-tenant avancé pour Spotify AI Agent"

# Configuration des composants principaux
__all__ = [
    # Configuration
    "ConfigManager", "TenantConfig", "SecurityConfig", "CacheConfig", 
    "MonitoringConfig", "config_manager",
    
    # Alertes
    "AlertManager", "Alert", "AlertRule", "AlertSeverity", "AlertStatus",
    "AlertHandler", "EmailAlertHandler", "SlackAlertHandler", "WebhookAlertHandler",
    "alert_manager",
    
    # Templates et localisation
    "TemplateEngine", "Template", "TemplateContext", "TemplateType",
    "LocalizationManager", "template_engine", "localization_manager",
    
    # Sécurité
    "SecurityManager", "SecurityPolicy", "PermissionLevel", "AccessControl",
    "EncryptionManager", "AuditLogger", "security_manager",
    
    # Cache
    "CacheManager", "CacheKey", "CacheStrategy", "CacheLevel", "SerializationFormat",
    "CacheEntry", "CacheBackend", "MemoryCacheBackend", "RedisCacheBackend", 
    "DistributedCache", "CacheSerializer", "cache_manager", "cache_result", 
    "cache_key_for_tenant", "cache_key_for_user",
    
    # Métriques
    "MetricsCollector", "TenancyMetrics", "SystemMetrics", "MetricsAggregator",
    "MetricsExporter", "MetricDefinition", "MetricValue",
    "metrics_collector", "tenancy_metrics", "system_metrics",
    
    # Validation
    "BaseValidator", "TenantConfigValidator", "SchemaValidator",
    "ValidationRule", "ValidationResult", "ValidationError",
    "ValidationType", "ValidationSeverity", "validate_with_decorator",
    "tenant_validator", "schema_validator",
    
    # Workflow
    "Workflow", "WorkflowEngine", "BaseTask", "TenantProvisioningTask",
    "TenantValidationTask", "NotificationTask", "ConditionalTask", "ParallelTask",
    "WorkflowContext", "TaskResult", "WorkflowStatus", "TaskStatus", "workflow_engine",
    
    # Événements
    "Event", "EventBus", "EventHandler", "TenantEventHandler", "APIEventHandler",
    "NotificationHandler", "EventType", "EventPriority", "EventStatus",
    "event_bus", "publish_tenant_created", "publish_api_request", "publish_system_alert"
]

# Initialisation du système
async def initialize_core_system():
    """Initialise le système core avec tous ses composants"""
    import structlog
    
    logger = structlog.get_logger(__name__)
    
    try:
        # Initialisation de la configuration
        await config_manager.load_configuration()
        
        # Initialisation du système de sécurité
        await security_manager.initialize()
        
        # Initialisation du cache
        await cache_manager.initialize()
        
        # Initialisation du système d'alertes
        await alert_manager.initialize()
        
        # Initialisation du moteur de templates
        await template_engine.initialize()
        
        # Initialisation du système de localisation
        await localization_manager.initialize()
        
        # Démarrage du bus d'événements
        await event_bus.start()
        
        logger.info("Système core initialisé avec succès")
        
    except Exception as e:
        logger.error("Erreur lors de l'initialisation du système core", error=str(e))
        raise

async def shutdown_core_system():
    """Arrête proprement le système core"""
    import structlog
    
    logger = structlog.get_logger(__name__)
    
    try:
        # Arrêt du bus d'événements
        await event_bus.stop()
        
        # Nettoyage du cache
        await cache_manager.cleanup()
        
        # Nettoyage du système de sécurité
        await security_manager.cleanup()
        
        logger.info("Système core arrêté proprement")
        
    except Exception as e:
        logger.error("Erreur lors de l'arrêt du système core", error=str(e))

# Configuration du logging structuré
def configure_structured_logging():
    """Configure le logging structuré pour le module core"""
    import structlog
    import logging
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=False,
    )

# Configuration automatique au import
configure_structured_logging()

# Métadonnées du module
CORE_MODULE_INFO = {
    "name": "tenancy_core",
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "components": [
        "Configuration Manager",
        "Alert System",
        "Template Engine", 
        "Security Manager",
        "Cache System",
        "Metrics Collection",
        "Validation Framework",
        "Workflow Engine",
        "Event Bus"
    ],
    "features": [
        "Multi-tenant configuration",
        "Real-time alerting",
        "Template rendering with i18n",
        "Advanced security policies",
        "Distributed caching",
        "Comprehensive metrics",
        "Data validation",
        "Workflow orchestration",
        "Event-driven architecture"
    ]
}

def get_module_info():
    """Retourne les informations du module core"""
    return CORE_MODULE_INFO.copy()

def get_component_status():
    """Retourne le statut de tous les composants"""
    return {
        "config_manager": config_manager.is_initialized if hasattr(config_manager, 'is_initialized') else False,
        "alert_manager": alert_manager.is_running if hasattr(alert_manager, 'is_running') else False,
        "template_engine": template_engine.is_initialized if hasattr(template_engine, 'is_initialized') else False,
        "security_manager": security_manager.is_initialized if hasattr(security_manager, 'is_initialized') else False,
        "cache_manager": cache_manager.is_connected if hasattr(cache_manager, 'is_connected') else False,
        "event_bus": event_bus.running,
        "workflow_engine": len(workflow_engine.workflows) > 0
    }
                ("automation", self._automation)
            ]
            
            for name, component in components:
                if component:
                    try:
                        component_health = await component.health_check()
                        health_status["components"][name] = component_health
                    except Exception as e:
                        health_status["components"][name] = {
                            "status": "unhealthy",
                            "error": str(e)
                        }
                        health_status["status"] = "degraded"
                else:
                    health_status["components"][name] = {
                        "status": "not_initialized"
                    }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Erreur lors du health check: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            return health_status


# Instance globale du système
_core_instance: Optional[CoreTenancySystem] = None


def get_core_instance() -> CoreTenancySystem:
    """
    Retourne l'instance globale du système core
    
    Returns:
        Instance globale du CoreTenancySystem
    """
    global _core_instance
    if _core_instance is None:
        _core_instance = CoreTenancySystem()
    return _core_instance


async def initialize_core_system(config: Optional[Dict[str, Any]] = None) -> CoreTenancySystem:
    """
    Initialise le système core avec une configuration
    
    Args:
        config: Configuration du système
        
    Returns:
        Instance initialisée du CoreTenancySystem
    """
    global _core_instance
    
    if _core_instance is None:
        _core_instance = CoreTenancySystem(config)
    
    if not _core_instance.is_initialized:
        await _core_instance.initialize()
    
    return _core_instance


# Hook de fermeture pour l'arrêt propre
import atexit

def _cleanup_core_system():
    """Nettoyage à la fermeture du processus"""
    global _core_instance
    if _core_instance and _core_instance.is_initialized:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_core_instance.shutdown())
            else:
                loop.run_until_complete(_core_instance.shutdown())
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")

atexit.register(_cleanup_core_system)
