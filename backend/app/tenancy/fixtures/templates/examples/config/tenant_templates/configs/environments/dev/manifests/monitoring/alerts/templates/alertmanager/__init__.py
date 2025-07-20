"""
AlertManager Ultra-Avancé pour Architecture Multi-Tenant
========================================================

Gestionnaire d'alertes intelligent avec routing sophistiqué, escalation automatique,
et intégration multi-canal pour le Spotify AI Agent.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)

class AlertManagerConfig:
    """Configuration ultra-avancée d'AlertManager"""
    
    def __init__(self):
        self.notification_channels = {}
        self.escalation_policies = {}
        self.routing_tree = {}
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialise la configuration AlertManager ultra-avancée"""
        logger.info("Initialisation AlertManager ultra-avancé")
    
    def generate_config(self) -> Dict[str, Any]:
        """Génère la configuration complète AlertManager"""
        return {
            "global": self._generate_global_config(),
            "route": self._generate_routing_tree(),
            "receivers": self._generate_receivers(),
            "inhibit_rules": self._generate_inhibit_rules(),
            "templates": self._generate_templates()
        }
    
    def _generate_global_config(self) -> Dict[str, Any]:
        """Configuration globale ultra-optimisée"""
        return {
            "smtp_smarthost": "${SMTP_HOST}:587",
            "smtp_from": "alerts@spotify-ai-agent.com",
            "smtp_auth_username": "${SMTP_USERNAME}",
            "smtp_auth_password": "${SMTP_PASSWORD}",
            "smtp_require_tls": True,
            "slack_api_url": "${SLACK_API_URL}",
            "pagerduty_url": "https://events.pagerduty.com/v2/enqueue",
            "resolve_timeout": "5m",
            "http_config": {
                "follow_redirects": True,
                "enable_http2": True
            }
        }
    
    def _generate_routing_tree(self) -> Dict[str, Any]:
        """Arbre de routage intelligent et hiérarchique"""
        return {
            "group_by": ["alertname", "cluster", "service", "tenant_id"],
            "group_wait": "10s",
            "group_interval": "10s",
            "repeat_interval": "12h",
            "receiver": "web.hook.default",
            "routes": [
                # ============================================================
                # ALERTES CRITIQUES - ESCALATION IMMÉDIATE
                # ============================================================
                {
                    "match": {
                        "severity": "critical"
                    },
                    "group_wait": "0s",
                    "group_interval": "5s",
                    "repeat_interval": "1h",
                    "receiver": "critical-alerts",
                    "routes": [
                        {
                            "match": {
                                "escalation_level": "L0"
                            },
                            "receiver": "l0-emergency",
                            "group_wait": "0s",
                            "repeat_interval": "15m"
                        },
                        {
                            "match": {
                                "escalation_level": "L1"
                            },
                            "receiver": "l1-critical",
                            "group_wait": "0s",
                            "repeat_interval": "30m"
                        },
                        {
                            "match": {
                                "escalation_level": "L2"
                            },
                            "receiver": "l2-high-priority",
                            "group_wait": "30s",
                            "repeat_interval": "1h"
                        }
                    ]
                },
                # ============================================================
                # ALERTES SÉCURITÉ - TRAITEMENT SPÉCIALISÉ
                # ============================================================
                {
                    "match": {
                        "category": "security"
                    },
                    "group_wait": "0s",
                    "group_interval": "1s",
                    "repeat_interval": "30m",
                    "receiver": "security-team",
                    "routes": [
                        {
                            "match": {
                                "service": "data_privacy"
                            },
                            "receiver": "data-protection-officer",
                            "group_wait": "0s",
                            "repeat_interval": "15m"
                        },
                        {
                            "match_re": {
                                "alertname": ".*Breach.*|.*Violation.*|.*Intrusion.*"
                            },
                            "receiver": "security-emergency",
                            "group_wait": "0s",
                            "repeat_interval": "10m"
                        }
                    ]
                },
                # ============================================================
                # ALERTES BUSINESS - ÉQUIPES MÉTIER
                # ============================================================
                {
                    "match": {
                        "category": "business"
                    },
                    "group_wait": "5m",
                    "group_interval": "30m",
                    "repeat_interval": "4h",
                    "receiver": "business-intelligence",
                    "routes": [
                        {
                            "match": {
                                "service": "billing"
                            },
                            "receiver": "finance-team"
                        },
                        {
                            "match": {
                                "service": "analytics"
                            },
                            "receiver": "product-team"
                        }
                    ]
                },
                # ============================================================
                # ALERTES ML/IA - ÉQUIPE DATA SCIENCE
                # ============================================================
                {
                    "match_re": {
                        "service": "ml_.*|.*_ml|ai_.*|.*_ai"
                    },
                    "group_wait": "2m",
                    "group_interval": "15m",
                    "repeat_interval": "2h",
                    "receiver": "ml-engineering",
                    "routes": [
                        {
                            "match": {
                                "category": "model_quality"
                            },
                            "receiver": "ml-model-ops"
                        },
                        {
                            "match": {
                                "category": "data_quality"
                            },
                            "receiver": "data-engineering"
                        }
                    ]
                },
                # ============================================================
                # ALERTES TENANT-SPÉCIFIQUES
                # ============================================================
                {
                    "match_re": {
                        "tenant_id": ".+"
                    },
                    "group_by": ["tenant_id", "severity", "service"],
                    "group_wait": "30s",
                    "group_interval": "5m",
                    "repeat_interval": "2h",
                    "receiver": "tenant-notifications",
                    "routes": [
                        {
                            "match": {
                                "sla_tier": "enterprise"
                            },
                            "receiver": "enterprise-support",
                            "group_wait": "10s",
                            "repeat_interval": "30m"
                        },
                        {
                            "match": {
                                "sla_tier": "premium"
                            },
                            "receiver": "premium-support",
                            "group_wait": "1m",
                            "repeat_interval": "1h"
                        }
                    ]
                },
                # ============================================================
                # ALERTES INFRASTRUCTURE - ÉQUIPE PLATFORM
                # ============================================================
                {
                    "match": {
                        "team": "platform"
                    },
                    "group_wait": "1m",
                    "group_interval": "10m",
                    "repeat_interval": "3h",
                    "receiver": "platform-engineering"
                },
                # ============================================================
                # ALERTES PRÉDICTIVES - PROACTIVES
                # ============================================================
                {
                    "match": {
                        "category": "predictive"
                    },
                    "group_wait": "5m",
                    "group_interval": "1h",
                    "repeat_interval": "24h",
                    "receiver": "predictive-alerts"
                }
            ]
        }
    
    def _generate_receivers(self) -> List[Dict[str, Any]]:
        """Générateurs de récepteurs multi-canaux ultra-avancés"""
        return [
            # ========================================================
            # RÉCEPTEUR CRITIQUE L0 - URGENCE MAXIMALE
            # ========================================================
            {
                "name": "l0-emergency",
                "pagerduty_configs": [
                    {
                        "routing_key": "${PAGERDUTY_L0_KEY}",
                        "description": "L0 EMERGENCY: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
                        "severity": "critical",
                        "class": "emergency",
                        "component": "spotify-ai-agent",
                        "group": "platform",
                        "custom_details": {
                            "alert_count": "{{ len .Alerts }}",
                            "affected_tenants": "{{ range .Alerts }}{{ .Labels.tenant_id }}{{ end }}",
                            "runbook": "{{ range .Alerts }}{{ .Annotations.runbook_url }}{{ end }}",
                            "auto_remediation": "{{ range .Alerts }}{{ .Annotations.auto_remediation }}{{ end }}"
                        }
                    }
                ],
                "slack_configs": [
                    {
                        "api_url": "${SLACK_EMERGENCY_WEBHOOK}",
                        "channel": "#emergency-alerts",
                        "username": "AlertManager Emergency",
                        "icon_emoji": ":rotating_light:",
                        "title": "🚨 L0 EMERGENCY ALERT 🚨",
                        "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
                        "color": "danger",
                        "fields": [
                            {
                                "title": "Severity",
                                "value": "{{ .CommonLabels.severity }}",
                                "short": True
                            },
                            {
                                "title": "Service",
                                "value": "{{ .CommonLabels.service }}",
                                "short": True
                            },
                            {
                                "title": "Tenant(s)",
                                "value": "{{ range .Alerts }}{{ .Labels.tenant_id }}{{ end }}",
                                "short": True
                            },
                            {
                                "title": "Auto-Remediation",
                                "value": "{{ range .Alerts }}{{ .Annotations.auto_remediation | default \"Manual intervention required\" }}{{ end }}",
                                "short": False
                            }
                        ],
                        "actions": [
                            {
                                "type": "button",
                                "text": "View Runbook",
                                "url": "{{ range .Alerts }}{{ .Annotations.runbook_url }}{{ end }}"
                            },
                            {
                                "type": "button",
                                "text": "Grafana Dashboard",
                                "url": "{{ range .Alerts }}{{ .Annotations.dashboard_url }}{{ end }}"
                            }
                        ]
                    }
                ],
                "email_configs": [
                    {
                        "to": "emergency@spotify-ai-agent.com",
                        "subject": "🚨 L0 EMERGENCY: {{ .CommonLabels.alertname }}",
                        "html": "{{ template \"email.emergency.html\" . }}"
                    }
                ]
            },
            # ========================================================
            # RÉCEPTEUR SÉCURITÉ - ÉQUIPE SPÉCIALISÉE
            # ========================================================
            {
                "name": "security-emergency",
                "slack_configs": [
                    {
                        "api_url": "${SLACK_SECURITY_WEBHOOK}",
                        "channel": "#security-alerts",
                        "username": "Security AlertManager",
                        "icon_emoji": ":shield:",
                        "title": "🛡️ SECURITY ALERT",
                        "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
                        "color": "warning",
                        "fields": [
                            {
                                "title": "Threat Level",
                                "value": "{{ .CommonLabels.severity }}",
                                "short": True
                            },
                            {
                                "title": "Affected Service",
                                "value": "{{ .CommonLabels.service }}",
                                "short": True
                            },
                            {
                                "title": "Tenant Impact",
                                "value": "{{ range .Alerts }}{{ .Labels.tenant_id | default \"System-wide\" }}{{ end }}",
                                "short": True
                            },
                            {
                                "title": "Threat Indicators",
                                "value": "{{ range .Alerts }}{{ .Annotations.threat_indicators | default \"Investigation required\" }}{{ end }}",
                                "short": False
                            }
                        ]
                    }
                ],
                "email_configs": [
                    {
                        "to": "security-team@spotify-ai-agent.com",
                        "subject": "🛡️ Security Alert: {{ .CommonLabels.alertname }}",
                        "html": "{{ template \"email.security.html\" . }}"
                    }
                ]
            },
            # ========================================================
            # RÉCEPTEUR ML/IA - DATA SCIENCE
            # ========================================================
            {
                "name": "ml-engineering",
                "slack_configs": [
                    {
                        "api_url": "${SLACK_ML_WEBHOOK}",
                        "channel": "#ml-alerts",
                        "username": "ML AlertManager",
                        "icon_emoji": ":robot_face:",
                        "title": "🤖 ML/AI Alert",
                        "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
                        "color": "{{ if eq .CommonLabels.severity \"critical\" }}danger{{ else }}warning{{ end }}",
                        "fields": [
                            {
                                "title": "Model/Pipeline",
                                "value": "{{ .CommonLabels.service }}",
                                "short": True
                            },
                            {
                                "title": "Tenant(s)",
                                "value": "{{ range .Alerts }}{{ .Labels.tenant_id }}{{ end }}",
                                "short": True
                            },
                            {
                                "title": "Impact",
                                "value": "{{ range .Alerts }}{{ .Annotations.business_impact | default \"Model performance affected\" }}{{ end }}",
                                "short": False
                            }
                        ]
                    }
                ]
            },
            # ========================================================
            # RÉCEPTEUR TENANT - NOTIFICATIONS CLIENTS
            # ========================================================
            {
                "name": "tenant-notifications",
                "webhook_configs": [
                    {
                        "url": "${TENANT_WEBHOOK_URL}",
                        "send_resolved": True,
                        "http_config": {
                            "bearer_token": "${TENANT_API_TOKEN}"
                        }
                    }
                ]
            },
            # ========================================================
            # RÉCEPTEUR DEFAULT
            # ========================================================
            {
                "name": "web.hook.default",
                "slack_configs": [
                    {
                        "api_url": "${SLACK_DEFAULT_WEBHOOK}",
                        "channel": "#general-alerts",
                        "username": "AlertManager",
                        "icon_emoji": ":warning:",
                        "title": "Spotify AI Agent Alert",
                        "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                    }
                ]
            }
        ]
    
    def _generate_inhibit_rules(self) -> List[Dict[str, Any]]:
        """Règles d'inhibition intelligentes pour éviter le spam"""
        return [
            {
                "source_match": {
                    "severity": "critical"
                },
                "target_match": {
                    "severity": "warning"
                },
                "equal": ["alertname", "cluster", "service", "tenant_id"]
            },
            {
                "source_match": {
                    "alertname": "ServiceDown"
                },
                "target_match_re": {
                    "alertname": ".*Latency.*|.*ErrorRate.*"
                },
                "equal": ["service", "instance"]
            },
            {
                "source_match": {
                    "category": "infrastructure"
                },
                "target_match": {
                    "category": "performance"
                },
                "equal": ["instance", "tenant_id"]
            }
        ]
    
    def _generate_templates(self) -> List[str]:
        """Templates personnalisés pour les notifications"""
        return [
            "/etc/alertmanager/templates/*.tmpl"
        ]

# Instance globale
alertmanager_config = AlertManagerConfig()
