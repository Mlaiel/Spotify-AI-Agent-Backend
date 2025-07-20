#!/usr/bin/env python3
"""
Configuration Templates et Exemples - Règles d'Alertes Ultra-Performantes

Ce module contient des templates et exemples de configuration pour le système
de gestion des règles d'alertes avec patterns industriels éprouvés.

Templates Disponibles:
- Règles infrastructure (CPU, Memory, Disk, Network)
- Règles application (Performance, Erreurs, Availability)
- Règles sécurité (Intrusion, Anomalies, Compliance)
- Règles business (KPIs, SLAs, User Experience)
- Règles ML/AI (Anomaly Detection, Predictions)
- Configurations multi-tenant et multi-environnement

Équipe Engineering:
✅ Lead Dev + Architecte IA : Fahed Mlaiel
✅ Développeur Backend Senior (Python/FastAPI/Django)
✅ Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
✅ Spécialiste Sécurité Backend
✅ Architecte Microservices

License: Spotify Proprietary
Copyright: © 2025 Spotify Technology S.A.
"""

import json
import yaml
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path
import uuid


class AlertRuleTemplates:
    """Générateur de templates de règles d'alertes"""
    
    @staticmethod
    def infrastructure_cpu_high() -> Dict[str, Any]:
        """Template pour alerte CPU élevé"""
        return {
            "name": "High CPU Usage",
            "description": "Alert when CPU usage exceeds 80% for 5 minutes",
            "severity": "HIGH",
            "category": "infrastructure", 
            "enabled": True,
            "cooldown_period_seconds": 300,
            "max_executions_per_hour": 20,
            "conditions": [
                {
                    "type": "threshold",
                    "condition_id": f"cpu_threshold_{uuid.uuid4().hex[:8]}",
                    "metric_path": "current_metrics.cpu_usage",
                    "operator": ">",
                    "threshold": 80.0,
                    "weight": 1.0
                }
            ],
            "tags": {
                "team": "infrastructure",
                "priority": "high",
                "automation": "enabled"
            }
        }
    
    @staticmethod
    def infrastructure_memory_high() -> Dict[str, Any]:
        """Template pour alerte mémoire élevée"""
        return {
            "name": "High Memory Usage",
            "description": "Alert when memory usage exceeds 85%",
            "severity": "HIGH",
            "category": "infrastructure",
            "enabled": True,
            "cooldown_period_seconds": 300,
            "max_executions_per_hour": 15,
            "conditions": [
                {
                    "type": "threshold",
                    "condition_id": f"memory_threshold_{uuid.uuid4().hex[:8]}",
                    "metric_path": "current_metrics.memory_usage",
                    "operator": ">",
                    "threshold": 85.0,
                    "weight": 1.0
                }
            ],
            "tags": {
                "team": "infrastructure",
                "priority": "high",
                "automation": "enabled"
            }
        }
    
    @staticmethod
    def infrastructure_disk_high() -> Dict[str, Any]:
        """Template pour alerte disque plein"""
        return {
            "name": "High Disk Usage", 
            "description": "Alert when disk usage exceeds 90%",
            "severity": "CRITICAL",
            "category": "infrastructure",
            "enabled": True,
            "cooldown_period_seconds": 600,
            "max_executions_per_hour": 10,
            "conditions": [
                {
                    "type": "threshold",
                    "condition_id": f"disk_threshold_{uuid.uuid4().hex[:8]}",
                    "metric_path": "current_metrics.disk_usage",
                    "operator": ">",
                    "threshold": 90.0,
                    "weight": 1.0
                }
            ],
            "tags": {
                "team": "infrastructure",
                "priority": "critical",
                "automation": "enabled",
                "escalation": "immediate"
            }
        }
    
    @staticmethod
    def application_error_rate_high() -> Dict[str, Any]:
        """Template pour taux d'erreur élevé"""
        return {
            "name": "High Error Rate",
            "description": "Alert when application error rate exceeds 5%",
            "severity": "HIGH",
            "category": "application",
            "enabled": True,
            "cooldown_period_seconds": 180,
            "max_executions_per_hour": 30,
            "conditions": [
                {
                    "type": "threshold",
                    "condition_id": f"error_rate_threshold_{uuid.uuid4().hex[:8]}",
                    "metric_path": "current_metrics.error_rate",
                    "operator": ">",
                    "threshold": 5.0,
                    "weight": 1.0
                }
            ],
            "tags": {
                "team": "application",
                "priority": "high",
                "sla_impact": "high"
            }
        }
    
    @staticmethod
    def application_response_time_slow() -> Dict[str, Any]:
        """Template pour temps de réponse lent"""
        return {
            "name": "Slow Response Time",
            "description": "Alert when response time exceeds 2 seconds",
            "severity": "MEDIUM",
            "category": "performance",
            "enabled": True,
            "cooldown_period_seconds": 300,
            "max_executions_per_hour": 25,
            "conditions": [
                {
                    "type": "threshold",
                    "condition_id": f"response_time_threshold_{uuid.uuid4().hex[:8]}",
                    "metric_path": "current_metrics.response_time",
                    "operator": ">",
                    "threshold": 2000.0,  # en millisecondes
                    "weight": 1.0
                }
            ],
            "tags": {
                "team": "performance",
                "priority": "medium",
                "user_experience": "impacted"
            }
        }
    
    @staticmethod
    def ml_anomaly_detection() -> Dict[str, Any]:
        """Template pour détection d'anomalies ML"""
        return {
            "name": "ML Anomaly Detection",
            "description": "Machine learning based anomaly detection for system metrics",
            "severity": "MEDIUM",
            "category": "ml_anomaly",
            "enabled": True,
            "cooldown_period_seconds": 600,
            "max_executions_per_hour": 10,
            "conditions": [
                {
                    "type": "ml_anomaly",
                    "condition_id": f"ml_anomaly_{uuid.uuid4().hex[:8]}",
                    "model_name": "isolation_forest",
                    "contamination": 0.1,
                    "weight": 1.0
                }
            ],
            "tags": {
                "team": "ml",
                "priority": "medium",
                "automation": "ml_driven",
                "learning": "enabled"
            }
        }
    
    @staticmethod
    def security_suspicious_activity() -> Dict[str, Any]:
        """Template pour activité suspecte"""
        return {
            "name": "Suspicious Security Activity",
            "description": "Alert on suspicious patterns in system behavior",
            "severity": "HIGH",
            "category": "security",
            "enabled": True,
            "cooldown_period_seconds": 120,
            "max_executions_per_hour": 50,
            "conditions": [
                {
                    "type": "composite",
                    "condition_id": f"security_composite_{uuid.uuid4().hex[:8]}",
                    "logic_operator": "OR",
                    "weight": 1.0,
                    "conditions": [
                        {
                            "type": "threshold",
                            "condition_id": f"failed_requests_{uuid.uuid4().hex[:8]}",
                            "metric_path": "custom_metrics.failed_auth_requests",
                            "operator": ">",
                            "threshold": 10.0,
                            "weight": 0.7
                        },
                        {
                            "type": "ml_anomaly",
                            "condition_id": f"access_pattern_{uuid.uuid4().hex[:8]}",
                            "model_name": "isolation_forest",
                            "contamination": 0.05,
                            "weight": 0.8
                        }
                    ]
                }
            ],
            "tags": {
                "team": "security",
                "priority": "high",
                "compliance": "required",
                "escalation": "security_team"
            }
        }
    
    @staticmethod
    def business_user_experience_degraded() -> Dict[str, Any]:
        """Template pour dégradation de l'expérience utilisateur"""
        return {
            "name": "User Experience Degraded",
            "description": "Alert when user experience metrics indicate degradation",
            "severity": "HIGH",
            "category": "user_experience",
            "enabled": True,
            "cooldown_period_seconds": 240,
            "max_executions_per_hour": 20,
            "conditions": [
                {
                    "type": "composite",
                    "condition_id": f"ux_degraded_{uuid.uuid4().hex[:8]}",
                    "logic_operator": "AND",
                    "weight": 1.0,
                    "conditions": [
                        {
                            "type": "threshold",
                            "condition_id": f"bounce_rate_{uuid.uuid4().hex[:8]}",
                            "metric_path": "custom_metrics.bounce_rate",
                            "operator": ">",
                            "threshold": 60.0,
                            "weight": 0.6
                        },
                        {
                            "type": "threshold",
                            "condition_id": f"session_duration_{uuid.uuid4().hex[:8]}",
                            "metric_path": "custom_metrics.avg_session_duration",
                            "operator": "<",
                            "threshold": 120.0,  # secondes
                            "weight": 0.4
                        }
                    ]
                }
            ],
            "tags": {
                "team": "product",
                "priority": "high",
                "business_impact": "high",
                "kpi": "user_satisfaction"
            }
        }
    
    @staticmethod
    def business_sla_breach() -> Dict[str, Any]:
        """Template pour violation SLA"""
        return {
            "name": "SLA Breach Warning",
            "description": "Alert when SLA metrics approach breach thresholds",
            "severity": "CRITICAL",
            "category": "business",
            "enabled": True,
            "cooldown_period_seconds": 900,
            "max_executions_per_hour": 5,
            "conditions": [
                {
                    "type": "composite",
                    "condition_id": f"sla_breach_{uuid.uuid4().hex[:8]}",
                    "logic_operator": "OR",
                    "weight": 1.0,
                    "conditions": [
                        {
                            "type": "threshold",
                            "condition_id": f"availability_{uuid.uuid4().hex[:8]}",
                            "metric_path": "custom_metrics.availability_percentage",
                            "operator": "<",
                            "threshold": 99.5,
                            "weight": 1.0
                        },
                        {
                            "type": "threshold",
                            "condition_id": f"p99_latency_{uuid.uuid4().hex[:8]}",
                            "metric_path": "custom_metrics.p99_response_time",
                            "operator": ">",
                            "threshold": 5000.0,  # ms
                            "weight": 0.8
                        }
                    ]
                }
            ],
            "tags": {
                "team": "sre",
                "priority": "critical",
                "business_impact": "critical",
                "escalation": "immediate",
                "sla": "customer_facing"
            }
        }


class TenantConfigGenerator:
    """Générateur de configurations tenant-specific"""
    
    def __init__(self, tenant_id: str, environment: str = "dev"):
        self.tenant_id = tenant_id
        self.environment = environment
    
    def generate_tenant_config(
        self,
        rule_templates: List[str] = None,
        custom_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Génère une configuration complète pour un tenant"""
        
        if rule_templates is None:
            rule_templates = [
                "infrastructure_cpu_high",
                "infrastructure_memory_high", 
                "infrastructure_disk_high",
                "application_error_rate_high",
                "application_response_time_slow",
                "ml_anomaly_detection"
            ]
        
        if custom_thresholds is None:
            custom_thresholds = {}
        
        config = {
            "tenant_info": {
                "tenant_id": self.tenant_id,
                "environment": self.environment,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0.0"
            },
            "global_settings": {
                "default_cooldown_seconds": 300,
                "max_concurrent_evaluations": 50,
                "enable_ml_predictions": True,
                "cache_ttl_seconds": 60,
                "notification_channels": [
                    "email", "slack", "webhook", "sms"
                ]
            },
            "rules": []
        }
        
        # Génération des règles à partir des templates
        for template_name in rule_templates:
            if hasattr(AlertRuleTemplates, template_name):
                template_func = getattr(AlertRuleTemplates, template_name)
                rule = template_func()
                
                # Application des seuils personnalisés
                self._apply_custom_thresholds(rule, custom_thresholds)
                
                # Ajout du tenant_id et environment
                rule["tenant_id"] = self.tenant_id
                rule["environment"] = self.environment
                
                config["rules"].append(rule)
        
        return config
    
    def _apply_custom_thresholds(
        self,
        rule: Dict[str, Any],
        custom_thresholds: Dict[str, float]
    ):
        """Applique les seuils personnalisés à une règle"""
        
        def update_condition_thresholds(condition: Dict[str, Any]):
            if condition.get("type") == "threshold":
                metric_path = condition.get("metric_path", "")
                metric_name = metric_path.split(".")[-1] if "." in metric_path else metric_path
                
                if metric_name in custom_thresholds:
                    condition["threshold"] = custom_thresholds[metric_name]
            
            elif condition.get("type") == "composite":
                for sub_condition in condition.get("conditions", []):
                    update_condition_thresholds(sub_condition)
        
        for condition in rule.get("conditions", []):
            update_condition_thresholds(condition)
    
    def generate_environment_config(self) -> Dict[str, Any]:
        """Génère une configuration spécifique à l'environnement"""
        
        env_configs = {
            "dev": {
                "thresholds_multiplier": 1.5,  # Seuils plus élevés en dev
                "cooldown_multiplier": 0.5,    # Cooldowns plus courts
                "enable_debug_logging": True,
                "max_executions_per_hour": 200
            },
            "staging": {
                "thresholds_multiplier": 1.2,
                "cooldown_multiplier": 0.8,
                "enable_debug_logging": True,
                "max_executions_per_hour": 100
            },
            "prod": {
                "thresholds_multiplier": 1.0,
                "cooldown_multiplier": 1.0,
                "enable_debug_logging": False,
                "max_executions_per_hour": 50,
                "require_approval": True
            }
        }
        
        return env_configs.get(self.environment, env_configs["dev"])


class ExampleConfigurations:
    """Configurations d'exemple pour différents cas d'usage"""
    
    @staticmethod
    def spotify_streaming_service() -> Dict[str, Any]:
        """Configuration pour service de streaming Spotify"""
        generator = TenantConfigGenerator("spotify_streaming", "prod")
        
        custom_thresholds = {
            "cpu_usage": 75.0,          # CPU optimal pour streaming
            "memory_usage": 80.0,       # Mémoire pour cache audio
            "response_time": 1500.0,    # Latence critique pour UX
            "error_rate": 0.5,          # Très faible tolérance aux erreurs
            "availability_percentage": 99.9  # SLA premium
        }
        
        config = generator.generate_tenant_config(
            rule_templates=[
                "infrastructure_cpu_high",
                "infrastructure_memory_high",
                "application_error_rate_high", 
                "application_response_time_slow",
                "business_user_experience_degraded",
                "business_sla_breach",
                "ml_anomaly_detection",
                "security_suspicious_activity"
            ],
            custom_thresholds=custom_thresholds
        )
        
        # Règles spécifiques au streaming
        streaming_rules = [
            {
                "name": "Audio Buffer Underrun",
                "description": "Alert when audio buffer underruns exceed threshold",
                "severity": "HIGH",
                "category": "user_experience",
                "tenant_id": "spotify_streaming",
                "environment": "prod",
                "enabled": True,
                "cooldown_period_seconds": 120,
                "max_executions_per_hour": 30,
                "conditions": [
                    {
                        "type": "threshold",
                        "condition_id": f"buffer_underrun_{uuid.uuid4().hex[:8]}",
                        "metric_path": "custom_metrics.buffer_underrun_rate",
                        "operator": ">",
                        "threshold": 2.0,
                        "weight": 1.0
                    }
                ],
                "tags": {
                    "team": "streaming",
                    "priority": "high",
                    "user_impact": "audio_quality"
                }
            },
            {
                "name": "Transcoding Queue Backup",
                "description": "Alert when transcoding queue exceeds capacity",
                "severity": "MEDIUM",
                "category": "performance",
                "tenant_id": "spotify_streaming",
                "environment": "prod",
                "enabled": True,
                "cooldown_period_seconds": 300,
                "max_executions_per_hour": 20,
                "conditions": [
                    {
                        "type": "threshold",
                        "condition_id": f"transcoding_queue_{uuid.uuid4().hex[:8]}",
                        "metric_path": "custom_metrics.transcoding_queue_length",
                        "operator": ">",
                        "threshold": 1000.0,
                        "weight": 1.0
                    }
                ],
                "tags": {
                    "team": "media_processing",
                    "priority": "medium",
                    "automation": "scaling"
                }
            }
        ]
        
        config["rules"].extend(streaming_rules)
        return config
    
    @staticmethod
    def ecommerce_platform() -> Dict[str, Any]:
        """Configuration pour plateforme e-commerce"""
        generator = TenantConfigGenerator("ecommerce_platform", "prod")
        
        custom_thresholds = {
            "cpu_usage": 70.0,
            "memory_usage": 85.0,
            "response_time": 2000.0,
            "error_rate": 1.0,
            "availability_percentage": 99.5
        }
        
        config = generator.generate_tenant_config(
            rule_templates=[
                "infrastructure_cpu_high",
                "infrastructure_memory_high",
                "application_error_rate_high",
                "application_response_time_slow",
                "business_sla_breach",
                "security_suspicious_activity"
            ],
            custom_thresholds=custom_thresholds
        )
        
        # Règles spécifiques e-commerce
        ecommerce_rules = [
            {
                "name": "Cart Abandonment Rate High",
                "description": "Alert when cart abandonment exceeds normal rates",
                "severity": "MEDIUM",
                "category": "business",
                "tenant_id": "ecommerce_platform",
                "environment": "prod",
                "enabled": True,
                "cooldown_period_seconds": 600,
                "max_executions_per_hour": 10,
                "conditions": [
                    {
                        "type": "threshold",
                        "condition_id": f"cart_abandonment_{uuid.uuid4().hex[:8]}",
                        "metric_path": "custom_metrics.cart_abandonment_rate",
                        "operator": ">",
                        "threshold": 75.0,
                        "weight": 1.0
                    }
                ],
                "tags": {
                    "team": "product",
                    "priority": "medium",
                    "business_impact": "revenue"
                }
            },
            {
                "name": "Payment Gateway Error",
                "description": "Alert on payment processing errors",
                "severity": "CRITICAL",
                "category": "business",
                "tenant_id": "ecommerce_platform",
                "environment": "prod",
                "enabled": True,
                "cooldown_period_seconds": 60,
                "max_executions_per_hour": 100,
                "conditions": [
                    {
                        "type": "threshold",
                        "condition_id": f"payment_errors_{uuid.uuid4().hex[:8]}",
                        "metric_path": "custom_metrics.payment_error_rate",
                        "operator": ">",
                        "threshold": 5.0,
                        "weight": 1.0
                    }
                ],
                "tags": {
                    "team": "payments",
                    "priority": "critical",
                    "business_impact": "revenue_loss",
                    "escalation": "immediate"
                }
            }
        ]
        
        config["rules"].extend(ecommerce_rules)
        return config
    
    @staticmethod
    def financial_trading_system() -> Dict[str, Any]:
        """Configuration pour système de trading financier"""
        generator = TenantConfigGenerator("financial_trading", "prod")
        
        # Seuils très stricts pour la finance
        custom_thresholds = {
            "cpu_usage": 60.0,
            "memory_usage": 75.0,
            "response_time": 100.0,  # Ultra faible latence
            "error_rate": 0.1,       # Tolérance zéro aux erreurs
            "availability_percentage": 99.99
        }
        
        config = generator.generate_tenant_config(
            rule_templates=[
                "infrastructure_cpu_high",
                "infrastructure_memory_high",
                "application_error_rate_high",
                "application_response_time_slow",
                "business_sla_breach",
                "security_suspicious_activity",
                "ml_anomaly_detection"
            ],
            custom_thresholds=custom_thresholds
        )
        
        # Règles spécifiques au trading
        trading_rules = [
            {
                "name": "Market Data Feed Delay",
                "description": "Alert when market data feed experiences delays",
                "severity": "CRITICAL",
                "category": "business",
                "tenant_id": "financial_trading",
                "environment": "prod",
                "enabled": True,
                "cooldown_period_seconds": 30,
                "max_executions_per_hour": 200,
                "conditions": [
                    {
                        "type": "threshold",
                        "condition_id": f"market_data_delay_{uuid.uuid4().hex[:8]}",
                        "metric_path": "custom_metrics.market_data_delay_ms",
                        "operator": ">",
                        "threshold": 50.0,  # 50ms delay
                        "weight": 1.0
                    }
                ],
                "tags": {
                    "team": "trading",
                    "priority": "critical",
                    "business_impact": "trading_loss",
                    "compliance": "financial_regulation"
                }
            },
            {
                "name": "Order Execution Failure",
                "description": "Alert on order execution failures",
                "severity": "CRITICAL",
                "category": "business",
                "tenant_id": "financial_trading",
                "environment": "prod",
                "enabled": True,
                "cooldown_period_seconds": 10,
                "max_executions_per_hour": 500,
                "conditions": [
                    {
                        "type": "threshold",
                        "condition_id": f"order_failures_{uuid.uuid4().hex[:8]}",
                        "metric_path": "custom_metrics.order_failure_rate",
                        "operator": ">",
                        "threshold": 0.1,
                        "weight": 1.0
                    }
                ],
                "tags": {
                    "team": "trading",
                    "priority": "critical",
                    "business_impact": "financial_loss",
                    "escalation": "immediate",
                    "audit": "required"
                }
            }
        ]
        
        config["rules"].extend(trading_rules)
        return config


class ConfigurationValidator:
    """Validateur de configurations"""
    
    @staticmethod
    def validate_rule_config(config: Dict[str, Any]) -> List[str]:
        """Valide une configuration de règle"""
        errors = []
        
        # Champs obligatoires
        required_fields = ["name", "tenant_id", "severity", "category"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validation des valeurs
        if "severity" in config:
            valid_severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
            if config["severity"] not in valid_severities:
                errors.append(f"Invalid severity: {config['severity']}")
        
        if "category" in config:
            valid_categories = [
                "infrastructure", "application", "security", "business",
                "ml_anomaly", "performance", "user_experience"
            ]
            if config["category"] not in valid_categories:
                errors.append(f"Invalid category: {config['category']}")
        
        # Validation des conditions
        if "conditions" in config:
            for i, condition in enumerate(config["conditions"]):
                condition_errors = ConfigurationValidator._validate_condition(condition)
                for error in condition_errors:
                    errors.append(f"Condition {i}: {error}")
        
        return errors
    
    @staticmethod
    def _validate_condition(condition: Dict[str, Any]) -> List[str]:
        """Valide une condition"""
        errors = []
        
        if "type" not in condition:
            errors.append("Missing condition type")
            return errors
        
        condition_type = condition["type"]
        
        if condition_type == "threshold":
            required = ["metric_path", "operator", "threshold"]
            for field in required:
                if field not in condition:
                    errors.append(f"Missing field: {field}")
            
            if "operator" in condition:
                valid_operators = [">", "<", ">=", "<=", "==", "!="]
                if condition["operator"] not in valid_operators:
                    errors.append(f"Invalid operator: {condition['operator']}")
        
        elif condition_type == "ml_anomaly":
            if "contamination" in condition:
                contamination = condition["contamination"]
                if not (0 < contamination < 1):
                    errors.append("Contamination must be between 0 and 1")
        
        elif condition_type == "composite":
            if "conditions" not in condition:
                errors.append("Composite condition missing sub-conditions")
            elif not isinstance(condition["conditions"], list):
                errors.append("Sub-conditions must be a list")
        
        return errors
    
    @staticmethod
    def validate_tenant_config(config: Dict[str, Any]) -> List[str]:
        """Valide une configuration complète de tenant"""
        errors = []
        
        # Validation de la structure globale
        required_sections = ["tenant_info", "rules"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing section: {section}")
        
        # Validation des infos tenant
        if "tenant_info" in config:
            tenant_info = config["tenant_info"]
            if "tenant_id" not in tenant_info:
                errors.append("Missing tenant_id in tenant_info")
        
        # Validation des règles
        if "rules" in config:
            if not isinstance(config["rules"], list):
                errors.append("Rules must be a list")
            else:
                for i, rule in enumerate(config["rules"]):
                    rule_errors = ConfigurationValidator.validate_rule_config(rule)
                    for error in rule_errors:
                        errors.append(f"Rule {i}: {error}")
        
        return errors


class ConfigExporter:
    """Exporteur de configurations vers différents formats"""
    
    @staticmethod
    def export_to_yaml(config: Dict[str, Any], file_path: str):
        """Exporte en format YAML"""
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def export_to_json(config: Dict[str, Any], file_path: str):
        """Exporte en format JSON"""
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    @staticmethod
    def export_to_terraform(config: Dict[str, Any], file_path: str):
        """Exporte en format Terraform HCL"""
        terraform_content = ConfigExporter._generate_terraform_hcl(config)
        with open(file_path, 'w') as f:
            f.write(terraform_content)
    
    @staticmethod
    def _generate_terraform_hcl(config: Dict[str, Any]) -> str:
        """Génère le contenu Terraform HCL"""
        # Implementation simplifiée pour l'exemple
        hcl_content = f"""
# Generated Alert Rules Configuration
# Tenant: {config.get('tenant_info', {}).get('tenant_id', 'unknown')}
# Generated at: {datetime.utcnow().isoformat()}

terraform {{
  required_providers {{
    spotify_alerts = {{
      source  = "spotify/alerts"
      version = "~> 2.0"
    }}
  }}
}}

variable "tenant_id" {{
  description = "Tenant identifier"
  type        = string
  default     = "{config.get('tenant_info', {}).get('tenant_id', 'unknown')}"
}}

variable "environment" {{
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "{config.get('tenant_info', {}).get('environment', 'dev')}"
}}
"""
        
        # Ajout des règles
        for i, rule in enumerate(config.get("rules", [])):
            rule_name = rule.get("name", f"rule_{i}").lower().replace(" ", "_")
            hcl_content += f"""
resource "spotify_alerts_rule" "{rule_name}" {{
  name        = "{rule['name']}"
  description = "{rule.get('description', '')}"
  severity    = "{rule['severity']}"
  category    = "{rule['category']}"
  tenant_id   = var.tenant_id
  environment = var.environment
  enabled     = {str(rule.get('enabled', True)).lower()}
  
  cooldown_period_seconds = {rule.get('cooldown_period_seconds', 300)}
  max_executions_per_hour = {rule.get('max_executions_per_hour', 100)}
  
  tags = {{
"""
            for key, value in rule.get("tags", {}).items():
                hcl_content += f'    {key} = "{value}"\n'
            
            hcl_content += "  }\n}\n"
        
        return hcl_content


# Script principal pour génération de configurations
def main():
    """Script principal pour génération d'exemples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate alert rule configurations")
    parser.add_argument("--tenant-id", required=True, help="Tenant identifier")
    parser.add_argument("--environment", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--template", choices=["basic", "spotify", "ecommerce", "trading"], default="basic")
    parser.add_argument("--output-format", choices=["json", "yaml", "terraform"], default="yaml")
    parser.add_argument("--output-file", help="Output file path")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    
    args = parser.parse_args()
    
    # Génération de la configuration
    if args.template == "basic":
        generator = TenantConfigGenerator(args.tenant_id, args.environment)
        config = generator.generate_tenant_config()
    elif args.template == "spotify":
        config = ExampleConfigurations.spotify_streaming_service()
        config["tenant_info"]["tenant_id"] = args.tenant_id
        config["tenant_info"]["environment"] = args.environment
    elif args.template == "ecommerce":
        config = ExampleConfigurations.ecommerce_platform()
        config["tenant_info"]["tenant_id"] = args.tenant_id
        config["tenant_info"]["environment"] = args.environment
    elif args.template == "trading":
        config = ExampleConfigurations.financial_trading_system()
        config["tenant_info"]["tenant_id"] = args.tenant_id
        config["tenant_info"]["environment"] = args.environment
    
    # Validation si demandée
    if args.validate:
        errors = ConfigurationValidator.validate_tenant_config(config)
        if errors:
            print("Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("Configuration is valid!")
    
    # Export
    if args.output_file:
        output_path = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{args.tenant_id}_{args.environment}_{timestamp}.{args.output_format}"
    
    if args.output_format == "json":
        ConfigExporter.export_to_json(config, output_path)
    elif args.output_format == "yaml":
        ConfigExporter.export_to_yaml(config, output_path)
    elif args.output_format == "terraform":
        ConfigExporter.export_to_terraform(config, output_path)
    
    print(f"Configuration exported to: {output_path}")
    print(f"Number of rules generated: {len(config.get('rules', []))}")
    
    return 0


if __name__ == "__main__":
    exit(main())


# Exportation
__all__ = [
    'AlertRuleTemplates',
    'TenantConfigGenerator',
    'ExampleConfigurations',
    'ConfigurationValidator',
    'ConfigExporter'
]
