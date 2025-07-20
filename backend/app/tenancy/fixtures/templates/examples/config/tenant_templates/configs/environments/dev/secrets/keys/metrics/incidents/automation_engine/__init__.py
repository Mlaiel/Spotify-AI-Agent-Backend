#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 MODULE D'AUTOMATION ULTRA-AVANCÉ - INTÉGRATION ENTERPRISE
Système d'automation intelligent avec orchestration distribuée et IA générative

Développé par l'équipe d'experts Achiri avec une architecture révolutionnaire
combinant l'automation intelligente, l'orchestration de workflows et le ML adaptatif.

Architecture Enterprise-Grade:
├── 🤖 Automation Engine (Réponse automatique intelligente)
├── 🔄 Workflow Engine (Orchestration distribuée) 
├── 🧠 ML Integration (Prédictions et optimisations)
├── 🛡️ Security Framework (Zero-Trust, Encryption)
├── 📊 Monitoring & Observability (Métriques temps réel)
├── 🌐 Multi-Cloud Support (AWS, Azure, GCP)
└── 🔧 DevOps Integration (CI/CD, Infrastructure as Code)

Fonctionnalités Ultra-Avancées:
- Auto-réponse intelligente avec ML
- Workflows distribués avec DAG
- Exécution multi-conteneurs (Docker/K8s)
- Prédictions comportementales
- Auto-scaling dynamique
- Sécurité adaptative
- Audit complet et compliance
- API REST/GraphQL
- Real-time monitoring
- Edge computing ready

Auteur: Fahed Mlaiel - Architecte Solutions d'Entreprise
Équipe: DevOps Experts, ML Engineers, Security Architects, Cloud Specialists
Version: 3.0.0 - Production Ready Enterprise
License: Enterprise Commercial License
"""

__version__ = "3.0.0"
__author__ = "Fahed Mlaiel & Achiri Expert Team"
__license__ = "Enterprise Commercial"
__status__ = "Production"

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

# Configuration du logging ultra-avancé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automation_engine.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# Imports des composants principaux
try:
    from .auto_response import (
        # Enums et types
        ActionType, ActionStatus, Priority, EscalationLevel, 
        LogicalOperator, ConditionOperator, MLModelType,
        
        # Modèles de données
        AutomationCondition, AutomationAction, AutomationRule,
        EscalationRule, ActionExecution, MLModel,
        
        # Exécuteurs spécialisés
        KubernetesActionExecutor, DockerActionExecutor,
        DatabaseActionExecutor, NotificationActionExecutor,
        
        # ML et IA
        MLAutomationPredictor,
        
        # Moteur principal
        AutomationEngine
    )
    
    from .workflow_engine import (
        # Enums workflow
        WorkflowStatus, TaskStatus, TaskType, ExecutionMode,
        Priority as WorkflowPriority, RetryStrategy,
        
        # Modèles workflow
        TaskConfiguration, TaskExecution, WorkflowTask,
        WorkflowDefinition, WorkflowExecution,
        
        # Exécuteurs de tâches
        TaskExecutor, ShellCommandExecutor, DockerTaskExecutor,
        KubernetesTaskExecutor, HTTPTaskExecutor, DatabaseTaskExecutor,
        NotificationTaskExecutor,
        
        # Moteur de workflows
        WorkflowEngine,
        
        # Utilitaires
        create_simple_workflow, load_workflow_from_yaml
    )
    
    AUTOMATION_AVAILABLE = True
    logger.info("Module d'automation chargé avec succès")
    
except ImportError as e:
    logger.error(f"Erreur chargement module automation: {e}")
    AUTOMATION_AVAILABLE = False

# =============================================================================
# FACTORY ET CONFIGURATION AVANCÉE
# =============================================================================

class AutomationEngineFactory:
    """Factory pour la création d'instances d'automation configurées"""
    
    @staticmethod
    def create_development_engine(config_override: Optional[Dict[str, Any]] = None) -> 'AutomationEngine':
        """Création d'un moteur pour développement"""
        default_config = {
            'ml': {
                'enabled': True,
                'cache_ttl': 300,
                'models_path': '/tmp/ml_models'
            },
            'kubernetes': {
                'in_cluster': False,
                'namespace': 'development',
                'auto_cleanup': True
            },
            'docker': {
                'network_mode': 'bridge',
                'auto_cleanup': True,
                'default_memory_limit': '512m'
            },
            'database': {
                'databases': {
                    'default': {
                        'connection_string': 'sqlite:///automation_dev.db'
                    }
                }
            },
            'notifications': {
                'channels': {
                    'email': {
                        'smtp_server': 'localhost',
                        'smtp_port': 1025  # MailHog pour dev
                    },
                    'slack': {
                        'webhook_url': 'http://localhost:3000/slack-webhook'
                    }
                }
            },
            'prometheus': {
                'enabled': True,
                'port': 8000
            }
        }
        
        if config_override:
            default_config.update(config_override)
        
        return AutomationEngine(default_config)
    
    @staticmethod
    def create_production_engine(config_path: str) -> 'AutomationEngine':
        """Création d'un moteur pour production"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validation de configuration production
            required_keys = ['kubernetes', 'database', 'notifications', 'ml']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Configuration production manquante: {key}")
            
            # Configuration sécurisée par défaut
            production_defaults = {
                'ml': {
                    'enabled': True,
                    'cache_ttl': 600,
                    'model_validation': True
                },
                'security': {
                    'encryption_enabled': True,
                    'audit_logging': True,
                    'rbac_enabled': True
                },
                'prometheus': {
                    'enabled': True,
                    'secure_metrics': True
                }
            }
            
            # Fusion avec les paramètres par défaut
            for key, value in production_defaults.items():
                if key not in config:
                    config[key] = value
                else:
                    config[key].update(value)
            
            return AutomationEngine(config)
            
        except Exception as e:
            logger.error(f"Erreur création moteur production: {e}")
            raise

class WorkflowEngineFactory:
    """Factory pour la création d'instances de workflow engine"""
    
    @staticmethod
    def create_development_engine(config_override: Optional[Dict[str, Any]] = None) -> 'WorkflowEngine':
        """Création d'un moteur de workflows pour développement"""
        default_config = {
            'shell': {
                'allowed_commands': ['echo', 'ls', 'cat', 'grep', 'head', 'tail'],
                'forbidden_patterns': ['rm -rf', 'format', 'del /'],
                'default_timeout': 300
            },
            'docker': {
                'auto_cleanup': True,
                'network_mode': 'bridge',
                'default_memory_limit': '512m',
                'default_cpu_limit': '0.5'
            },
            'kubernetes': {
                'in_cluster': False,
                'namespace': 'development',
                'auto_cleanup': True
            },
            'http': {
                'default_timeout': 60,
                'max_redirects': 5
            },
            'database': {
                'databases': {
                    'default': {
                        'connection_string': 'sqlite:///workflows_dev.db'
                    }
                }
            },
            'notifications': {
                'channels': {
                    'email': {'enabled': True},
                    'slack': {'enabled': True},
                    'webhook': {'enabled': True}
                }
            },
            'storage': {
                'type': 'memory'  # Pour le développement
            },
            'prometheus': {
                'enabled': True,
                'port': 8001
            },
            'max_workers': 5
        }
        
        if config_override:
            default_config.update(config_override)
        
        return WorkflowEngine(default_config)
    
    @staticmethod
    def create_production_engine(config_path: str) -> 'WorkflowEngine':
        """Création d'un moteur de workflows pour production"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Configuration sécurisée par défaut
            production_defaults = {
                'shell': {
                    'security_enabled': True,
                    'command_validation': True
                },
                'storage': {
                    'type': 'redis',
                    'persistence': True
                },
                'security': {
                    'encryption_enabled': True,
                    'audit_logging': True
                },
                'max_workers': 20
            }
            
            # Fusion avec les paramètres par défaut
            for key, value in production_defaults.items():
                if key not in config:
                    config[key] = value
                else:
                    config[key].update(value)
            
            return WorkflowEngine(config)
            
        except Exception as e:
            logger.error(f"Erreur création moteur workflows production: {e}")
            raise

# =============================================================================
# GESTIONNAIRE UNIFIÉ D'AUTOMATION
# =============================================================================

class UnifiedAutomationManager:
    """Gestionnaire unifié pour l'automation et les workflows"""
    
    def __init__(self, automation_config: Dict[str, Any], workflow_config: Dict[str, Any]):
        self.automation_engine: Optional[AutomationEngine] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        
        self.automation_config = automation_config
        self.workflow_config = workflow_config
        
        self.is_initialized = False
        self.metrics = {
            'automation_executions': 0,
            'workflow_executions': 0,
            'total_success_rate': 0.0,
            'uptime_seconds': 0
        }
        
        logger.info("Gestionnaire unifié d'automation créé")
    
    async def initialize(self) -> bool:
        """Initialisation des moteurs"""
        try:
            if not AUTOMATION_AVAILABLE:
                logger.error("Module d'automation non disponible")
                return False
            
            # Initialisation du moteur d'automation
            self.automation_engine = AutomationEngine(self.automation_config)
            
            # Initialisation du moteur de workflows
            self.workflow_engine = WorkflowEngine(self.workflow_config)
            
            self.is_initialized = True
            logger.info("Gestionnaire unifié initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur initialisation gestionnaire unifié: {e}")
            return False
    
    async def register_automation_rule(self, rule: 'AutomationRule') -> bool:
        """Enregistrement d'une règle d'automation"""
        if not self.automation_engine:
            logger.error("Moteur d'automation non initialisé")
            return False
        
        return await self.automation_engine.add_rule(rule)
    
    async def register_workflow(self, workflow: 'WorkflowDefinition') -> bool:
        """Enregistrement d'un workflow"""
        if not self.workflow_engine:
            logger.error("Moteur de workflows non initialisé")
            return False
        
        return await self.workflow_engine.register_workflow(workflow)
    
    async def process_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traitement d'un incident avec automation"""
        if not self.automation_engine:
            raise RuntimeError("Moteur d'automation non initialisé")
        
        result = await self.automation_engine.process_event(incident_data)
        self.metrics['automation_executions'] += 1
        return result
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None,
                             user_id: str = "", tenant_id: str = "") -> str:
        """Exécution d'un workflow"""
        if not self.workflow_engine:
            raise RuntimeError("Moteur de workflows non initialisé")
        
        execution_id = await self.workflow_engine.execute_workflow(
            workflow_id, input_data, user_id, tenant_id
        )
        self.metrics['workflow_executions'] += 1
        return execution_id
    
    async def get_unified_metrics(self) -> Dict[str, Any]:
        """Métriques unifiées des deux moteurs"""
        try:
            unified_metrics = self.metrics.copy()
            
            if self.automation_engine:
                automation_metrics = await self.automation_engine.get_metrics()
                unified_metrics['automation'] = automation_metrics
            
            if self.workflow_engine:
                workflow_metrics = await self.workflow_engine.get_metrics()
                unified_metrics['workflows'] = workflow_metrics
            
            # Calcul du taux de réussite global
            if self.automation_engine and self.workflow_engine:
                auto_success = automation_metrics.get('success_rate', 0)
                workflow_success = workflow_metrics.get('success_rate', 0)
                unified_metrics['total_success_rate'] = (auto_success + workflow_success) / 2
            
            return unified_metrics
            
        except Exception as e:
            logger.error(f"Erreur récupération métriques unifiées: {e}")
            return self.metrics.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé globale"""
        try:
            health = {
                'status': 'healthy',
                'components': {},
                'timestamp': None
            }
            
            if self.automation_engine:
                auto_health = await self.automation_engine.health_check()
                health['components']['automation_engine'] = auto_health
            
            if self.workflow_engine:
                workflow_health = await self.workflow_engine.health_check()
                health['components']['workflow_engine'] = workflow_health
            
            # Statut global
            component_statuses = [
                comp.get('status', 'unknown') 
                for comp in health['components'].values()
            ]
            
            if any(status == 'unhealthy' for status in component_statuses):
                health['status'] = 'unhealthy'
            elif any(status == 'degraded' for status in component_statuses):
                health['status'] = 'degraded'
            
            return health
            
        except Exception as e:
            logger.error(f"Erreur health check unifié: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def shutdown(self):
        """Arrêt propre des moteurs"""
        try:
            logger.info("Arrêt du gestionnaire unifié d'automation")
            
            if self.automation_engine:
                await self.automation_engine.shutdown()
            
            if self.workflow_engine:
                await self.workflow_engine.shutdown()
            
            self.is_initialized = False
            logger.info("Gestionnaire unifié arrêté")
            
        except Exception as e:
            logger.error(f"Erreur arrêt gestionnaire unifié: {e}")

# =============================================================================
# UTILITAIRES DE CONFIGURATION ET DÉPLOIEMENT
# =============================================================================

def create_default_automation_config() -> Dict[str, Any]:
    """Création d'une configuration d'automation par défaut"""
    return {
        'ml': {
            'enabled': True,
            'cache_ttl': 300,
            'models_path': './ml_models',
            'training_enabled': True
        },
        'kubernetes': {
            'enabled': True,
            'in_cluster': False,
            'namespace': 'automation',
            'auto_cleanup': True
        },
        'docker': {
            'enabled': True,
            'auto_cleanup': True,
            'network_mode': 'bridge'
        },
        'database': {
            'databases': {
                'default': {
                    'connection_string': 'postgresql://automation:password@localhost:5432/automation'
                }
            }
        },
        'notifications': {
            'channels': {
                'email': {'enabled': True},
                'slack': {'enabled': True},
                'webhook': {'enabled': True}
            }
        },
        'prometheus': {
            'enabled': True,
            'port': 8000
        },
        'security': {
            'encryption_enabled': True,
            'audit_logging': True
        }
    }

def create_default_workflow_config() -> Dict[str, Any]:
    """Création d'une configuration de workflows par défaut"""
    return {
        'shell': {
            'enabled': True,
            'security_enabled': True,
            'allowed_commands': ['echo', 'ls', 'cat', 'grep']
        },
        'docker': {
            'enabled': True,
            'auto_cleanup': True
        },
        'kubernetes': {
            'enabled': True,
            'in_cluster': False,
            'namespace': 'workflows'
        },
        'http': {
            'enabled': True,
            'default_timeout': 60
        },
        'database': {
            'databases': {
                'default': {
                    'connection_string': 'postgresql://workflows:password@localhost:5432/workflows'
                }
            }
        },
        'storage': {
            'type': 'redis',
            'url': 'redis://localhost:6379/1'
        },
        'prometheus': {
            'enabled': True,
            'port': 8001
        },
        'max_workers': 10
    }

async def create_unified_manager(automation_config: Optional[Dict[str, Any]] = None,
                               workflow_config: Optional[Dict[str, Any]] = None) -> UnifiedAutomationManager:
    """Création et initialisation d'un gestionnaire unifié"""
    auto_config = automation_config or create_default_automation_config()
    workflow_config = workflow_config or create_default_workflow_config()
    
    manager = UnifiedAutomationManager(auto_config, workflow_config)
    
    success = await manager.initialize()
    if not success:
        raise RuntimeError("Échec initialisation gestionnaire unifié")
    
    return manager

def save_config_template(output_dir: str = "./config"):
    """Sauvegarde des templates de configuration"""
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration automation
        auto_config = create_default_automation_config()
        with open(f"{output_dir}/automation_config.json", 'w', encoding='utf-8') as f:
            json.dump(auto_config, f, indent=2, ensure_ascii=False)
        
        # Configuration workflows
        workflow_config = create_default_workflow_config()
        with open(f"{output_dir}/workflow_config.json", 'w', encoding='utf-8') as f:
            json.dump(workflow_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Templates de configuration sauvegardés dans {output_dir}")
        
    except Exception as e:
        logger.error(f"Erreur sauvegarde templates: {e}")

# =============================================================================
# EXPORTS ET API PUBLIQUE
# =============================================================================

# Classes principales
__all__ = [
    # Factories
    'AutomationEngineFactory',
    'WorkflowEngineFactory',
    
    # Gestionnaire unifié
    'UnifiedAutomationManager',
    
    # Utilitaires
    'create_default_automation_config',
    'create_default_workflow_config',
    'create_unified_manager',
    'save_config_template',
    
    # État du module
    'AUTOMATION_AVAILABLE',
]

# Exports conditionnels si les modules sont disponibles
if AUTOMATION_AVAILABLE:
    __all__.extend([
        # Auto-response
        'ActionType', 'ActionStatus', 'Priority', 'EscalationLevel',
        'AutomationCondition', 'AutomationAction', 'AutomationRule',
        'AutomationEngine', 'MLAutomationPredictor',
        
        # Workflows
        'WorkflowStatus', 'TaskStatus', 'TaskType', 'ExecutionMode',
        'TaskConfiguration', 'WorkflowTask', 'WorkflowDefinition',
        'WorkflowEngine', 'create_simple_workflow', 'load_workflow_from_yaml'
    ])

# Informations sur le module
def get_module_info() -> Dict[str, Any]:
    """Informations sur le module d'automation"""
    return {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'status': __status__,
        'available': AUTOMATION_AVAILABLE,
        'components': {
            'automation_engine': AUTOMATION_AVAILABLE,
            'workflow_engine': AUTOMATION_AVAILABLE,
            'ml_integration': AUTOMATION_AVAILABLE,
            'unified_manager': True
        }
    }

# Message de bienvenue
logger.info(f"🚀 Module d'Automation Ultra-Avancé v{__version__} chargé")
logger.info(f"👨‍💻 Développé par {__author__}")
logger.info(f"✅ Statut: {__status__} - Disponible: {AUTOMATION_AVAILABLE}")

if AUTOMATION_AVAILABLE:
    logger.info("🎯 Fonctionnalités disponibles:")
    logger.info("   • Automation Engine avec ML")
    logger.info("   • Workflow Engine distribué")
    logger.info("   • Orchestration Kubernetes/Docker")
    logger.info("   • Monitoring temps réel")
    logger.info("   • Sécurité enterprise-grade")
else:
    logger.warning("⚠️  Module d'automation en mode dégradé - Vérifiez les dépendances")
