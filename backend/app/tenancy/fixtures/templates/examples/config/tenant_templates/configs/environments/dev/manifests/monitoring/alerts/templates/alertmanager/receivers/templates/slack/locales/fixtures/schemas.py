"""
Schémas de Validation Enterprise pour Système de Fixtures Multi-Tenant
=====================================================================

Ce module définit tous les schémas de validation JSON Schema avancés pour 
garantir l'intégrité, la sécurité et la conformité des données dans le système 
de fixtures d'alertes multi-tenant.

🎯 Architecture: JSON Schema + OpenAPI 3.0 + Validation stricte
🔒 Sécurité: Sanitization + Injection prevention + Data validation
⚡ Performance: Schema caching + Lazy validation + Optimized patterns
🧠 IA: Smart validation + Auto-correction + Pattern learning

Auteur: Fahed Mlaiel - Lead Developer & AI Architect
Équipe: DevOps/ML/Security/Backend Experts
Version: 3.0.0-enterprise
"""

import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# ============================================================================================
# PATTERNS ET CONSTANTES DE VALIDATION
# ============================================================================================

# Patterns de validation sécurisés
SECURITY_PATTERNS = {
    "tenant_id": r"^[a-zA-Z0-9_-]{3,50}$",
    "alert_id": r"^[a-zA-Z0-9_-]{8,64}$",
    "channel_name": r"^#[a-zA-Z0-9_-]{1,21}$",
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "url": r"^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?$",
    "webhook_url": r"^https://hooks\.slack\.com/services/[A-Z0-9]{9}/[A-Z0-9]{11}/[a-zA-Z0-9]{24}$",
    "version": r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$",
    "locale": r"^[a-z]{2}(?:_[A-Z]{2})?$",
    "timestamp": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z?$",
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "hash": r"^[a-f0-9]{32,128}$",
    "token": r"^[a-zA-Z0-9_-]{20,}$",
    "ip_address": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
    "phone": r"^\+?[1-9]\d{1,14}$",
    "safe_string": r"^[a-zA-Z0-9\s\-_.,:;!?()[\]{}\"']*$",
    "sql_injection": r"(?i)(select|insert|update|delete|drop|create|alter|exec|union|script|javascript|vbscript|onload|onerror)",
    "xss_patterns": r"(?i)(<script|javascript:|vbscript:|onload=|onerror=|<iframe|<object|<embed)"
}

# Limites de sécurité
SECURITY_LIMITS = {
    "max_string_length": 10000,
    "max_array_items": 1000,
    "max_object_properties": 100,
    "max_nesting_level": 10,
    "max_file_size_mb": 10,
    "max_template_size": 50000,
    "max_message_length": 4000,
    "max_title_length": 200
}

# ============================================================================================
# SCHÉMAS DE BASE ET RÉUTILISABLES
# ============================================================================================

# Définitions communes réutilisables
COMMON_DEFINITIONS = {
    "tenant_id": {
        "type": "string",
        "pattern": SECURITY_PATTERNS["tenant_id"],
        "minLength": 3,
        "maxLength": 50,
        "description": "Identifiant unique du tenant"
    },
    
    "timestamp": {
        "type": "string",
        "format": "date-time",
        "pattern": SECURITY_PATTERNS["timestamp"],
        "description": "Timestamp au format ISO 8601"
    },
    
    "uuid": {
        "type": "string",
        "pattern": SECURITY_PATTERNS["uuid"],
        "description": "UUID au format standard"
    },
    
    "locale": {
        "type": "string",
        "enum": ["fr", "en", "de", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"],
        "default": "fr",
        "description": "Code de locale ISO 639-1"
    },
    
    "environment": {
        "type": "string",
        "enum": ["dev", "test", "staging", "preprod", "prod", "dr"],
        "description": "Environnement de déploiement"
    },
    
    "alert_severity": {
        "type": "string",
        "enum": ["critical", "high", "medium", "low", "info", "debug"],
        "description": "Niveau de sévérité de l'alerte"
    },
    
    "alert_type": {
        "type": "string",
        "enum": [
            "system", "application", "security", "performance", 
            "business", "infrastructure", "compliance", "ml_model",
            "user_behavior", "financial"
        ],
        "description": "Type d'alerte métier"
    },
    
    "notification_channel": {
        "type": "string",
        "enum": [
            "slack", "teams", "discord", "email", "sms", "whatsapp",
            "pagerduty", "webhook", "push", "voice", "telegram"
        ],
        "description": "Canal de notification"
    },
    
    "security_level": {
        "type": "integer",
        "minimum": 0,
        "maximum": 4,
        "description": "Niveau de sécurité (0=public, 4=top secret)"
    },
    
    "safe_string": {
        "type": "string",
        "pattern": SECURITY_PATTERNS["safe_string"],
        "maxLength": SECURITY_LIMITS["max_string_length"],
        "description": "Chaîne sécurisée sans injection"
    },
    
    "metadata": {
        "type": "object",
        "maxProperties": SECURITY_LIMITS["max_object_properties"],
        "additionalProperties": {
            "anyOf": [
                {"type": "string", "maxLength": 1000},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"}
            ]
        },
        "description": "Métadonnées additionnelles"
    }
}

# ============================================================================================
# SCHÉMA PRINCIPAL ALERTE SLACK
# ============================================================================================

SLACK_ALERT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/slack-alert/v3.0.0",
    "title": "Slack Alert Schema Enterprise",
    "description": "Schéma complet pour les alertes Slack multi-tenant avec IA et sécurité avancée",
    "type": "object",
    "definitions": COMMON_DEFINITIONS,
    
    "required": ["tenant_id", "severity", "alert_type", "title", "message", "channel"],
    
    "properties": {
        # Identification et contexte
        "id": {"$ref": "#/definitions/uuid"},
        "tenant_id": {"$ref": "#/definitions/tenant_id"},
        "environment": {"$ref": "#/definitions/environment"},
        "correlation_id": {
            "type": "string",
            "pattern": r"^[a-zA-Z0-9_-]{8,64}$",
            "description": "ID de corrélation pour traçabilité"
        },
        
        # Classification de l'alerte
        "severity": {"$ref": "#/definitions/alert_severity"},
        "alert_type": {"$ref": "#/definitions/alert_type"},
        "security_level": {"$ref": "#/definitions/security_level"},
        
        # Contenu de l'alerte
        "title": {
            "type": "string",
            "minLength": 1,
            "maxLength": SECURITY_LIMITS["max_title_length"],
            "pattern": SECURITY_PATTERNS["safe_string"],
            "description": "Titre de l'alerte"
        },
        "message": {
            "type": "string",
            "minLength": 1,
            "maxLength": SECURITY_LIMITS["max_message_length"],
            "pattern": SECURITY_PATTERNS["safe_string"],
            "description": "Message détaillé de l'alerte"
        },
        
        # Configuration Slack
        "channel": {
            "type": "string",
            "pattern": SECURITY_PATTERNS["channel_name"],
            "description": "Canal Slack de destination"
        },
        "locale": {"$ref": "#/definitions/locale"},
        
        # Données techniques
        "source_system": {
            "type": "string",
            "pattern": r"^[a-zA-Z0-9_-]{1,50}$",
            "default": "achiri-ai-agent",
            "description": "Système source de l'alerte"
        },
        "raw_data": {
            "type": "object",
            "maxProperties": 50,
            "description": "Données brutes de l'alerte"
        },
        "enriched_data": {
            "type": "object",
            "maxProperties": 50,
            "description": "Données enrichies par l'IA"
        },
        
        # ML et prédictions
        "ml_predictions": {
            "type": "object",
            "properties": {
                "severity_prediction": {"type": "number", "minimum": 0, "maximum": 1},
                "escalation_probability": {"type": "number", "minimum": 0, "maximum": 1},
                "resolution_time_hours": {"type": "number", "minimum": 0},
                "false_positive_probability": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "description": "Prédictions générées par les modèles ML"
        },
        
        # Statut et lifecycle
        "status": {
            "type": "string",
            "enum": ["new", "acknowledged", "investigating", "resolved", "suppressed"],
            "default": "new",
            "description": "Statut actuel de l'alerte"
        },
        "acknowledged": {"type": "boolean", "default": false},
        "acknowledged_by": {"type": "string", "maxLength": 100},
        "acknowledged_at": {"$ref": "#/definitions/timestamp"},
        "resolved": {"type": "boolean", "default": false},
        "resolved_by": {"type": "string", "maxLength": 100},
        "resolved_at": {"$ref": "#/definitions/timestamp"},
        
        # SLA et escalation
        "sla_deadline": {"$ref": "#/definitions/timestamp"},
        "escalation_level": {"type": "integer", "minimum": 0, "maximum": 10},
        "escalation_history": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer"},
                    "timestamp": {"$ref": "#/definitions/timestamp"},
                    "escalated_to": {"type": "string"},
                    "reason": {"type": "string", "maxLength": 500}
                }
            }
        },
        
        # Métriques de performance
        "creation_latency_ms": {"type": "number", "minimum": 0},
        "delivery_latency_ms": {"type": "number", "minimum": 0},
        "processing_time_ms": {"type": "number", "minimum": 0},
        
        # Audit et métadonnées
        "created_at": {"$ref": "#/definitions/timestamp"},
        "updated_at": {"$ref": "#/definitions/timestamp"},
        "created_by": {"type": "string", "maxLength": 100},
        "updated_by": {"type": "string", "maxLength": 100},
        "version": {"type": "string", "pattern": SECURITY_PATTERNS["version"]},
        "tags": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "maxLength": 50}
        },
        "metadata": {"$ref": "#/definitions/metadata"}
    },
    
    "additionalProperties": false,
    
    # Validations conditionnelles
    "allOf": [
        {
            "if": {"properties": {"acknowledged": {"const": true}}},
            "then": {
                "required": ["acknowledged_by", "acknowledged_at"],
                "properties": {
                    "acknowledged_by": {"minLength": 1},
                    "acknowledged_at": {"type": "string"}
                }
            }
        },
        {
            "if": {"properties": {"resolved": {"const": true}}},
            "then": {
                "required": ["resolved_by", "resolved_at"],
                "properties": {
                    "resolved_by": {"minLength": 1},
                    "resolved_at": {"type": "string"}
                }
            }
        }
    ]
}

# ============================================================================================
# SCHÉMA WEBHOOK ALERTMANAGER
# ============================================================================================

ALERTMANAGER_WEBHOOK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/alertmanager-webhook/v3.0.0",
    "title": "Alertmanager Webhook Schema",
    "description": "Schéma pour les webhooks Alertmanager avec validation stricte",
    "type": "object",
    
    "required": ["receiver", "status", "alerts"],
    
    "properties": {
        "receiver": {
            "type": "string",
            "pattern": r"^[a-zA-Z0-9_-]{1,100}$",
            "description": "Nom du receiver Alertmanager"
        },
        "status": {
            "type": "string",
            "enum": ["firing", "resolved"],
            "description": "Statut du groupe d'alertes"
        },
        "alerts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 100,
            "items": {
                "type": "object",
                "required": ["status", "labels"],
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["firing", "resolved"]
                    },
                    "labels": {
                        "type": "object",
                        "required": ["alertname"],
                        "properties": {
                            "alertname": {"type": "string", "minLength": 1, "maxLength": 100},
                            "severity": {"$ref": "#/definitions/alert_severity"},
                            "tenant_id": {"$ref": "#/definitions/tenant_id"},
                            "instance": {"type": "string", "maxLength": 200},
                            "job": {"type": "string", "maxLength": 100}
                        },
                        "additionalProperties": {"type": "string", "maxLength": 500}
                    },
                    "annotations": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "maxLength": 500},
                            "description": {"type": "string", "maxLength": 2000},
                            "runbook_url": {"type": "string", "format": "uri"}
                        },
                        "additionalProperties": {"type": "string", "maxLength": 1000}
                    },
                    "startsAt": {"$ref": "#/definitions/timestamp"},
                    "endsAt": {"$ref": "#/definitions/timestamp"},
                    "generatorURL": {"type": "string", "format": "uri"},
                    "fingerprint": {"type": "string", "pattern": r"^[a-f0-9]{16}$"}
                }
            }
        },
        "groupLabels": {
            "type": "object",
            "additionalProperties": {"type": "string", "maxLength": 200}
        },
        "commonLabels": {
            "type": "object",
            "additionalProperties": {"type": "string", "maxLength": 200}
        },
        "commonAnnotations": {
            "type": "object",
            "additionalProperties": {"type": "string", "maxLength": 1000}
        },
        "externalURL": {"type": "string", "format": "uri"},
        "version": {"type": "string", "pattern": SECURITY_PATTERNS["version"]},
        "groupKey": {"type": "string", "maxLength": 100}
    },
    
    "definitions": COMMON_DEFINITIONS,
    "additionalProperties": false
}

# ============================================================================================
# SCHÉMA CONFIGURATION TENANT
# ============================================================================================

TENANT_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/tenant-config/v3.0.0",
    "title": "Tenant Configuration Schema",
    "description": "Schéma de configuration complète pour un tenant",
    "type": "object",
    
    "required": ["tenant_id", "environment", "alert_config"],
    
    "properties": {
        "tenant_id": {"$ref": "#/definitions/tenant_id"},
        "environment": {"$ref": "#/definitions/environment"},
        
        "alert_config": {
            "type": "object",
            "required": ["default_channel", "escalation_enabled"],
            "properties": {
                "default_channel": {
                    "type": "string",
                    "pattern": SECURITY_PATTERNS["channel_name"]
                },
                "escalation_enabled": {"type": "boolean"},
                "auto_resolve_enabled": {"type": "boolean", "default": true},
                "ml_predictions_enabled": {"type": "boolean", "default": true},
                "custom_templates": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-zA-Z0-9_-]+$": {
                            "type": "object",
                            "properties": {
                                "title_template": {"type": "string", "maxLength": 500},
                                "message_template": {"type": "string", "maxLength": 2000}
                            }
                        }
                    }
                }
            }
        },
        
        "notification_preferences": {
            "type": "object",
            "properties": {
                "channels": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/notification_channel"}
                },
                "quiet_hours": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "start_time": {"type": "string", "pattern": r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$"},
                        "end_time": {"type": "string", "pattern": r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$"},
                        "timezone": {"type": "string", "maxLength": 50}
                    }
                },
                "severity_filters": {
                    "type": "object",
                    "patternProperties": {
                        "^(critical|high|medium|low|info|debug)$": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/notification_channel"}
                        }
                    }
                }
            }
        },
        
        "escalation_matrix": {
            "type": "object",
            "properties": {
                "levels": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "required": ["level", "delay_minutes", "recipients"],
                        "properties": {
                            "level": {"type": "integer", "minimum": 1, "maximum": 10},
                            "delay_minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
                            "recipients": {
                                "type": "array",
                                "items": {"type": "string", "maxLength": 100}
                            },
                            "channels": {
                                "type": "array",
                                "items": {"$ref": "#/definitions/notification_channel"}
                            }
                        }
                    }
                }
            }
        },
        
        # Limites et quotas
        "quotas": {
            "type": "object",
            "properties": {
                "daily_alert_limit": {"type": "integer", "minimum": 1, "maximum": 100000},
                "rate_limit_per_minute": {"type": "integer", "minimum": 1, "maximum": 1000},
                "storage_quota_mb": {"type": "integer", "minimum": 1, "maximum": 10000},
                "retention_days": {"type": "integer", "minimum": 1, "maximum": 2555}
            }
        },
        
        # Sécurité et compliance
        "security_config": {
            "type": "object",
            "properties": {
                "encryption_required": {"type": "boolean", "default": true},
                "audit_enabled": {"type": "boolean", "default": true},
                "data_classification": {
                    "type": "string",
                    "enum": ["public", "internal", "confidential", "restricted", "top_secret"]
                },
                "compliance_standards": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["gdpr", "soc2", "iso27001", "pci_dss", "hipaa", "sox"]
                    }
                },
                "allowed_ips": {
                    "type": "array",
                    "items": {"type": "string", "pattern": SECURITY_PATTERNS["ip_address"]}
                }
            }
        },
        
        # Intégrations externes
        "integrations": {
            "type": "object",
            "properties": {
                "slack": {
                    "type": "object",
                    "properties": {
                        "workspace_id": {"type": "string", "pattern": r"^T[A-Z0-9]{10}$"},
                        "webhook_url": {"type": "string", "pattern": SECURITY_PATTERNS["webhook_url"]},
                        "bot_token": {"type": "string", "pattern": r"^xoxb-[a-zA-Z0-9-]+$"},
                        "default_channel": {"type": "string", "pattern": SECURITY_PATTERNS["channel_name"]}
                    }
                },
                "teams": {
                    "type": "object",
                    "properties": {
                        "webhook_url": {"type": "string", "format": "uri"},
                        "tenant_id": {"type": "string", "maxLength": 100}
                    }
                },
                "pagerduty": {
                    "type": "object",
                    "properties": {
                        "integration_key": {"type": "string", "pattern": r"^[a-f0-9]{32}$"},
                        "service_id": {"type": "string", "maxLength": 50}
                    }
                }
            }
        },
        
        "created_at": {"$ref": "#/definitions/timestamp"},
        "updated_at": {"$ref": "#/definitions/timestamp"},
        "version": {"type": "string", "pattern": SECURITY_PATTERNS["version"]},
        "metadata": {"$ref": "#/definitions/metadata"}
    },
    
    "definitions": COMMON_DEFINITIONS,
    "additionalProperties": false
}

# ============================================================================================
# SCHÉMAS SPÉCIALISÉS
# ============================================================================================

SECURITY_POLICY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/security-policy/v3.0.0",
    "title": "Security Policy Schema",
    "type": "object",
    
    "required": ["name", "policy_type", "rules"],
    
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "description": {"type": "string", "maxLength": 1000},
        "policy_type": {
            "type": "string",
            "enum": ["access_control", "data_classification", "encryption", "audit", "compliance"]
        },
        "severity": {"$ref": "#/definitions/alert_severity"},
        "rules": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["condition", "action"],
                "properties": {
                    "condition": {"type": "string", "maxLength": 500},
                    "action": {"type": "string", "maxLength": 200},
                    "parameters": {"type": "object"}
                }
            }
        },
        "enforcement_mode": {
            "type": "string",
            "enum": ["enforce", "warn", "audit_only"],
            "default": "enforce"
        }
    },
    
    "definitions": COMMON_DEFINITIONS
}

PERFORMANCE_METRIC_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/performance-metric/v3.0.0",
    "title": "Performance Metric Schema",
    "type": "object",
    
    "required": ["metric_name", "value", "unit", "timestamp"],
    
    "properties": {
        "metric_name": {"type": "string", "pattern": r"^[a-zA-Z0-9_.-]{1,100}$"},
        "metric_type": {
            "type": "string",
            "enum": ["counter", "gauge", "histogram", "summary"]
        },
        "value": {"type": "number"},
        "unit": {"type": "string", "maxLength": 20},
        "timestamp": {"$ref": "#/definitions/timestamp"},
        "labels": {
            "type": "object",
            "maxProperties": 20,
            "patternProperties": {
                "^[a-zA-Z_][a-zA-Z0-9_]*$": {"type": "string", "maxLength": 100}
            }
        },
        "thresholds": {
            "type": "object",
            "properties": {
                "warning": {"type": "number"},
                "critical": {"type": "number"}
            }
        }
    },
    
    "definitions": COMMON_DEFINITIONS
}

ML_MODEL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/ml-model/v3.0.0",
    "title": "ML Model Schema",
    "type": "object",
    
    "required": ["name", "model_type", "algorithm", "features"],
    
    "properties": {
        "name": {"type": "string", "pattern": r"^[a-zA-Z0-9_-]{1,100}$"},
        "model_type": {
            "type": "string",
            "enum": ["classification", "regression", "clustering", "anomaly_detection", "recommendation"]
        },
        "algorithm": {
            "type": "string",
            "enum": ["linear_regression", "random_forest", "xgboost", "neural_network", "lstm", "transformer"]
        },
        "features": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100}
        },
        "hyperparameters": {
            "type": "object",
            "maxProperties": 50
        },
        "performance_metrics": {
            "type": "object",
            "properties": {
                "accuracy": {"type": "number", "minimum": 0, "maximum": 1},
                "precision": {"type": "number", "minimum": 0, "maximum": 1},
                "recall": {"type": "number", "minimum": 0, "maximum": 1},
                "f1_score": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "training_config": {
            "type": "object",
            "properties": {
                "dataset_size": {"type": "integer", "minimum": 1},
                "training_time_hours": {"type": "number", "minimum": 0},
                "validation_split": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
    },
    
    "definitions": COMMON_DEFINITIONS
}

BUSINESS_RULE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/business-rule/v3.0.0",
    "title": "Business Rule Schema",
    "type": "object",
    
    "required": ["name", "rule_type", "conditions", "actions"],
    
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "description": {"type": "string", "maxLength": 1000},
        "rule_type": {
            "type": "string",
            "enum": ["alert_routing", "escalation", "suppression", "enrichment", "auto_resolve"]
        },
        "conditions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["field", "operator", "value"],
                "properties": {
                    "field": {"type": "string", "maxLength": 100},
                    "operator": {
                        "type": "string",
                        "enum": ["equals", "not_equals", "contains", "not_contains", "greater_than", "less_than", "regex"]
                    },
                    "value": {"type": ["string", "number", "boolean"]},
                    "logical_operator": {"type": "string", "enum": ["AND", "OR"], "default": "AND"}
                }
            }
        },
        "actions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["action_type"],
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["route", "escalate", "suppress", "enrich", "notify", "create_ticket"]
                    },
                    "parameters": {"type": "object"}
                }
            }
        },
        "priority": {"type": "integer", "minimum": 1, "maximum": 1000},
        "enabled": {"type": "boolean", "default": true}
    },
    
    "definitions": COMMON_DEFINITIONS
}

SLA_POLICY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/sla-policy/v3.0.0",
    "title": "SLA Policy Schema",
    "type": "object",
    
    "required": ["name", "targets"],
    
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "description": {"type": "string", "maxLength": 1000},
        "targets": {
            "type": "object",
            "properties": {
                "response_time_minutes": {
                    "type": "object",
                    "patternProperties": {
                        "^(critical|high|medium|low|info)$": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10080
                        }
                    }
                },
                "resolution_time_hours": {
                    "type": "object",
                    "patternProperties": {
                        "^(critical|high|medium|low|info)$": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8760
                        }
                    }
                },
                "availability_percentage": {"type": "number", "minimum": 90, "maximum": 100}
            }
        },
        "penalties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "violation_type": {"type": "string"},
                    "penalty_action": {"type": "string"},
                    "escalation_target": {"type": "string"}
                }
            }
        }
    },
    
    "definitions": COMMON_DEFINITIONS
}

# Schémas d'événements et d'audit
AUDIT_EVENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/audit-event/v3.0.0",
    "title": "Audit Event Schema",
    "type": "object",
    
    "required": ["event_type", "user_id", "action", "resource_type", "timestamp"],
    
    "properties": {
        "event_id": {"$ref": "#/definitions/uuid"},
        "event_type": {
            "type": "string",
            "enum": ["create", "update", "delete", "access", "login", "logout", "error"]
        },
        "user_id": {"type": "string", "maxLength": 100},
        "action": {"type": "string", "maxLength": 200},
        "resource_type": {"type": "string", "maxLength": 100},
        "resource_id": {"type": "string", "maxLength": 100},
        "timestamp": {"$ref": "#/definitions/timestamp"},
        "ip_address": {"type": "string", "pattern": SECURITY_PATTERNS["ip_address"]},
        "user_agent": {"type": "string", "maxLength": 500},
        "session_id": {"type": "string", "maxLength": 100},
        "event_data": {"type": "object", "maxProperties": 50},
        "risk_score": {"type": "number", "minimum": 0, "maximum": 10},
        "compliance_relevant": {"type": "boolean", "default": false}
    },
    
    "definitions": COMMON_DEFINITIONS
}

MONITORING_EVENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/monitoring-event/v3.0.0",
    "title": "Monitoring Event Schema",
    "type": "object",
    
    "required": ["source", "event_type", "severity", "timestamp"],
    
    "properties": {
        "event_id": {"$ref": "#/definitions/uuid"},
        "source": {"type": "string", "maxLength": 100},
        "event_type": {"type": "string", "maxLength": 100},
        "severity": {"$ref": "#/definitions/alert_severity"},
        "timestamp": {"$ref": "#/definitions/timestamp"},
        "tenant_id": {"$ref": "#/definitions/tenant_id"},
        "environment": {"$ref": "#/definitions/environment"},
        "service_name": {"type": "string", "maxLength": 100},
        "metrics": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9_.-]+$": {"type": "number"}
            }
        },
        "event_data": {"type": "object", "maxProperties": 100},
        "processed": {"type": "boolean", "default": false},
        "alert_generated": {"type": "boolean", "default": false}
    },
    
    "definitions": COMMON_DEFINITIONS
}

HEALTH_CHECK_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/health-check/v3.0.0",
    "title": "Health Check Schema",
    "type": "object",
    
    "required": ["component", "status", "timestamp"],
    
    "properties": {
        "check_id": {"$ref": "#/definitions/uuid"},
        "component": {"type": "string", "maxLength": 100},
        "check_type": {"type": "string", "maxLength": 50},
        "status": {
            "type": "string",
            "enum": ["healthy", "degraded", "unhealthy", "unknown"]
        },
        "timestamp": {"$ref": "#/definitions/timestamp"},
        "response_time_ms": {"type": "number", "minimum": 0},
        "error_message": {"type": "string", "maxLength": 1000},
        "additional_info": {"type": "object", "maxProperties": 20},
        "thresholds": {
            "type": "object",
            "properties": {
                "warning_ms": {"type": "number", "minimum": 0},
                "critical_ms": {"type": "number", "minimum": 0}
            }
        }
    },
    
    "definitions": COMMON_DEFINITIONS
}

BACKUP_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/backup/v3.0.0",
    "title": "Backup Schema",
    "type": "object",
    
    "required": ["backup_type", "file_path", "size_bytes", "timestamp"],
    
    "properties": {
        "backup_id": {"$ref": "#/definitions/uuid"},
        "backup_type": {
            "type": "string",
            "enum": ["full", "incremental", "differential", "configuration"]
        },
        "file_path": {"type": "string", "maxLength": 500},
        "size_bytes": {"type": "integer", "minimum": 0},
        "timestamp": {"$ref": "#/definitions/timestamp"},
        "compression_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "checksum": {"type": "string", "pattern": SECURITY_PATTERNS["hash"]},
        "encryption_enabled": {"type": "boolean", "default": true},
        "retention_days": {"type": "integer", "minimum": 1, "maximum": 2555},
        "integrity_verified": {"type": "boolean", "default": false},
        "restoration_tested": {"type": "boolean", "default": false}
    },
    
    "definitions": COMMON_DEFINITIONS
}

RECOVERY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/recovery/v3.0.0",
    "title": "Recovery Schema",
    "type": "object",
    
    "required": ["incident_id", "recovery_type", "started_at"],
    
    "properties": {
        "recovery_id": {"$ref": "#/definitions/uuid"},
        "incident_id": {"type": "string", "maxLength": 100},
        "recovery_type": {
            "type": "string",
            "enum": ["automatic", "manual", "partial", "full"]
        },
        "backup_used": {"type": "string", "maxLength": 100},
        "started_at": {"$ref": "#/definitions/timestamp"},
        "completed_at": {"$ref": "#/definitions/timestamp"},
        "success": {"type": "boolean"},
        "data_loss_bytes": {"type": "integer", "minimum": 0},
        "services_restored": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100}
        },
        "root_cause": {"type": "string", "maxLength": 1000},
        "lessons_learned": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500}
        }
    },
    
    "definitions": COMMON_DEFINITIONS
}

# ============================================================================================
# SCHÉMAS COMPOSÉS ET COMPLEXES
# ============================================================================================

COMPLIANCE_RULE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://achiri.ai/schemas/compliance-rule/v3.0.0",
    "title": "Compliance Rule Schema",
    "type": "object",
    
    "required": ["name", "standard", "rule_type", "validation_criteria"],
    
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "description": {"type": "string", "maxLength": 1000},
        "standard": {
            "type": "string",
            "enum": ["gdpr", "soc2", "iso27001", "pci_dss", "hipaa", "sox", "nist"]
        },
        "rule_type": {
            "type": "string",
            "enum": ["data_protection", "access_control", "audit_logging", "encryption", "retention"]
        },
        "validation_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["criterion", "requirement"],
                "properties": {
                    "criterion": {"type": "string", "maxLength": 200},
                    "requirement": {"type": "string", "maxLength": 500},
                    "validation_method": {"type": "string", "maxLength": 100},
                    "evidence_required": {"type": "boolean", "default": true}
                }
            }
        },
        "severity": {"$ref": "#/definitions/alert_severity"},
        "automated_check": {"type": "boolean", "default": false},
        "check_frequency": {
            "type": "string",
            "enum": ["continuous", "hourly", "daily", "weekly", "monthly"]
        }
    },
    
    "definitions": COMMON_DEFINITIONS
}

# ============================================================================================
# VALIDATEURS ET UTILITAIRES
# ============================================================================================

def validate_no_injection(data: Any) -> bool:
    """Valide qu'il n'y a pas d'injection SQL/XSS dans les données."""
    if isinstance(data, str):
        # Vérification SQL injection
        if re.search(SECURITY_PATTERNS["sql_injection"], data):
            return False
        # Vérification XSS
        if re.search(SECURITY_PATTERNS["xss_patterns"], data):
            return False
    elif isinstance(data, dict):
        return all(validate_no_injection(v) for v in data.values())
    elif isinstance(data, list):
        return all(validate_no_injection(item) for item in data)
    
    return True

def sanitize_string(value: str, max_length: int = None) -> str:
    """Nettoie et sécurise une chaîne de caractères."""
    if not isinstance(value, str):
        return str(value)
    
    # Suppression des caractères dangereux
    value = re.sub(r'[<>"\']', '', value)
    value = re.sub(r'(javascript:|vbscript:|data:)', '', value, flags=re.IGNORECASE)
    
    # Limitation de la longueur
    if max_length and len(value) > max_length:
        value = value[:max_length]
    
    return value.strip()

def get_schema_by_name(schema_name: str) -> Optional[Dict[str, Any]]:
    """Retourne un schéma par son nom."""
    schemas = {
        "slack_alert": SLACK_ALERT_SCHEMA,
        "alertmanager_webhook": ALERTMANAGER_WEBHOOK_SCHEMA,
        "tenant_config": TENANT_CONFIG_SCHEMA,
        "security_policy": SECURITY_POLICY_SCHEMA,
        "performance_metric": PERFORMANCE_METRIC_SCHEMA,
        "ml_model": ML_MODEL_SCHEMA,
        "business_rule": BUSINESS_RULE_SCHEMA,
        "sla_policy": SLA_POLICY_SCHEMA,
        "compliance_rule": COMPLIANCE_RULE_SCHEMA,
        "audit_event": AUDIT_EVENT_SCHEMA,
        "monitoring_event": MONITORING_EVENT_SCHEMA,
        "health_check": HEALTH_CHECK_SCHEMA,
        "backup": BACKUP_SCHEMA,
        "recovery": RECOVERY_SCHEMA
    }
    
    return schemas.get(schema_name)

def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """Retourne tous les schémas disponibles."""
    return {
        "slack_alert": SLACK_ALERT_SCHEMA,
        "alertmanager_webhook": ALERTMANAGER_WEBHOOK_SCHEMA,
        "tenant_config": TENANT_CONFIG_SCHEMA,
        "security_policy": SECURITY_POLICY_SCHEMA,
        "performance_metric": PERFORMANCE_METRIC_SCHEMA,
        "ml_model": ML_MODEL_SCHEMA,
        "business_rule": BUSINESS_RULE_SCHEMA,
        "sla_policy": SLA_POLICY_SCHEMA,
        "compliance_rule": COMPLIANCE_RULE_SCHEMA,
        "audit_event": AUDIT_EVENT_SCHEMA,
        "monitoring_event": MONITORING_EVENT_SCHEMA,
        "health_check": HEALTH_CHECK_SCHEMA,
        "backup": BACKUP_SCHEMA,
        "recovery": RECOVERY_SCHEMA
    }

# Export de tous les schémas et utilitaires
__all__ = [
    # Schémas principaux
    "SLACK_ALERT_SCHEMA",
    "ALERTMANAGER_WEBHOOK_SCHEMA", 
    "TENANT_CONFIG_SCHEMA",
    
    # Schémas spécialisés
    "SECURITY_POLICY_SCHEMA",
    "PERFORMANCE_METRIC_SCHEMA",
    "ML_MODEL_SCHEMA",
    "BUSINESS_RULE_SCHEMA",
    "SLA_POLICY_SCHEMA",
    "COMPLIANCE_RULE_SCHEMA",
    
    # Schémas d'événements
    "AUDIT_EVENT_SCHEMA",
    "MONITORING_EVENT_SCHEMA",
    "HEALTH_CHECK_SCHEMA",
    "BACKUP_SCHEMA",
    "RECOVERY_SCHEMA",
    
    # Patterns et constantes
    "SECURITY_PATTERNS",
    "SECURITY_LIMITS",
    "COMMON_DEFINITIONS",
    
    # Utilitaires
    "validate_no_injection",
    "sanitize_string",
    "get_schema_by_name",
    "get_all_schemas"
]
