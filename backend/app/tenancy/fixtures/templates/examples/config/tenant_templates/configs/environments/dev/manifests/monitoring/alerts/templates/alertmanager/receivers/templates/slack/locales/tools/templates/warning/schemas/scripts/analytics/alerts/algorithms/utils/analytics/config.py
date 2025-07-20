"""
Advanced Alert Configuration for Spotify AI Agent
===============================================

Comprehensive alert configuration with intelligent rules, escalation policies,
and multi-channel notification support.

Author: Fahed Mlaiel
Roles: Lead Dev + Architecte IA, Spécialiste Sécurité Backend
"""

from typing import Dict, List, Any
import yaml

# Alert Rule Templates
ALERT_RULE_TEMPLATES = {
    "high_error_rate": {
        "name": "High Error Rate",
        "description": "Alert when error rate exceeds threshold",
        "metric_query": "rate(spotify_ai_errors_total[5m])",
        "condition": "gt",
        "threshold": 0.05,
        "severity": "critical",
        "evaluation_interval_seconds": 60,
        "for_duration_seconds": 300,
        "labels": {
            "team": "platform",
            "category": "availability"
        },
        "annotations": {
            "summary": "High error rate detected",
            "description": "Error rate is {{ $value | humanizePercentage }} which is above the threshold",
            "runbook_url": "https://docs.spotify-ai.com/runbooks/high-error-rate"
        }
    },
    
    "slow_response_time": {
        "name": "Slow Response Time",
        "description": "Alert when response time exceeds threshold",
        "metric_query": "histogram_quantile(0.95, rate(spotify_ai_request_duration_seconds_bucket[5m]))",
        "condition": "gt",
        "threshold": 2.0,
        "severity": "high",
        "evaluation_interval_seconds": 60,
        "for_duration_seconds": 300,
        "labels": {
            "team": "platform",
            "category": "performance"
        },
        "annotations": {
            "summary": "Slow response time detected",
            "description": "95th percentile response time is {{ $value }}s which is above the threshold"
        }
    },
    
    "low_user_engagement": {
        "name": "Low User Engagement",
        "description": "Alert when user engagement drops significantly",
        "metric_query": "spotify_ai_user_engagement_rate",
        "condition": "lt",
        "threshold": 0.7,
        "severity": "medium",
        "evaluation_interval_seconds": 300,
        "for_duration_seconds": 900,
        "labels": {
            "team": "product",
            "category": "business"
        },
        "annotations": {
            "summary": "User engagement is low",
            "description": "User engagement rate is {{ $value | humanizePercentage }}"
        }
    },
    
    "ml_model_drift": {
        "name": "ML Model Drift Detected",
        "description": "Alert when ML model performance degrades",
        "metric_query": "spotify_ai_ml_model_accuracy",
        "condition": "lt",
        "threshold": 0.85,
        "severity": "high",
        "evaluation_interval_seconds": 300,
        "for_duration_seconds": 600,
        "labels": {
            "team": "ml",
            "category": "model_quality"
        },
        "annotations": {
            "summary": "ML model accuracy degraded",
            "description": "Model accuracy is {{ $value | humanizePercentage }} which is below threshold"
        }
    },
    
    "disk_space_low": {
        "name": "Low Disk Space",
        "description": "Alert when disk space is running low",
        "metric_query": "(1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100",
        "condition": "gt",
        "threshold": 85.0,
        "severity": "high",
        "evaluation_interval_seconds": 60,
        "for_duration_seconds": 300,
        "labels": {
            "team": "infrastructure",
            "category": "capacity"
        },
        "annotations": {
            "summary": "Disk space is running low",
            "description": "Disk usage is {{ $value }}% on {{ $labels.instance }}"
        }
    },
    
    "memory_usage_high": {
        "name": "High Memory Usage",
        "description": "Alert when memory usage is consistently high",
        "metric_query": "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100",
        "condition": "gt",
        "threshold": 90.0,
        "severity": "high",
        "evaluation_interval_seconds": 60,
        "for_duration_seconds": 600,
        "labels": {
            "team": "infrastructure",
            "category": "capacity"
        },
        "annotations": {
            "summary": "High memory usage detected",
            "description": "Memory usage is {{ $value }}% on {{ $labels.instance }}"
        }
    },
    
    "database_connections_high": {
        "name": "High Database Connections",
        "description": "Alert when database connection pool is nearly exhausted",
        "metric_query": "postgres_connections_active / postgres_connections_max * 100",
        "condition": "gt",
        "threshold": 80.0,
        "severity": "medium",
        "evaluation_interval_seconds": 60,
        "for_duration_seconds": 300,
        "labels": {
            "team": "database",
            "category": "capacity"
        },
        "annotations": {
            "summary": "Database connection usage is high",
            "description": "Connection pool usage is {{ $value }}%"
        }
    },
    
    "cache_hit_rate_low": {
        "name": "Low Cache Hit Rate",
        "description": "Alert when cache hit rate drops below optimal threshold",
        "metric_query": "rate(redis_cache_hits_total[5m]) / (rate(redis_cache_hits_total[5m]) + rate(redis_cache_misses_total[5m])) * 100",
        "condition": "lt",
        "threshold": 85.0,
        "severity": "medium",
        "evaluation_interval_seconds": 300,
        "for_duration_seconds": 600,
        "labels": {
            "team": "platform",
            "category": "performance"
        },
        "annotations": {
            "summary": "Cache hit rate is low",
            "description": "Cache hit rate is {{ $value }}% which may impact performance"
        }
    },
    
    "api_rate_limit_exceeded": {
        "name": "API Rate Limit Exceeded",
        "description": "Alert when API rate limits are frequently exceeded",
        "metric_query": "rate(spotify_ai_rate_limit_exceeded_total[5m])",
        "condition": "gt",
        "threshold": 10.0,
        "severity": "medium",
        "evaluation_interval_seconds": 60,
        "for_duration_seconds": 300,
        "labels": {
            "team": "platform",
            "category": "capacity"
        },
        "annotations": {
            "summary": "API rate limits being exceeded",
            "description": "Rate limit exceeded {{ $value }} times per second"
        }
    },
    
    "security_anomaly": {
        "name": "Security Anomaly Detected",
        "description": "Alert when security anomalies are detected",
        "metric_query": "spotify_ai_security_anomalies_total",
        "condition": "gt",
        "threshold": 0,
        "severity": "critical",
        "evaluation_interval_seconds": 60,
        "for_duration_seconds": 0,
        "labels": {
            "team": "security",
            "category": "security"
        },
        "annotations": {
            "summary": "Security anomaly detected",
            "description": "{{ $value }} security anomalies detected in the last minute"
        }
    }
}

# Notification Channel Configurations
NOTIFICATION_CHANNELS = {
    "slack_critical": {
        "type": "slack",
        "name": "Critical Alerts Slack",
        "config": {
            "webhook_url": "${SLACK_CRITICAL_WEBHOOK_URL}",
            "channel": "#alerts-critical",
            "username": "Spotify-AI-Alert",
            "icon_emoji": ":rotating_light:",
            "title_template": "🚨 Critical Alert: {{ .GroupLabels.alertname }}",
            "text_template": "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}"
        },
        "enabled": True
    },
    
    "slack_general": {
        "type": "slack",
        "name": "General Alerts Slack",
        "config": {
            "webhook_url": "${SLACK_GENERAL_WEBHOOK_URL}",
            "channel": "#alerts-general",
            "username": "Spotify-AI-Alert",
            "icon_emoji": ":warning:",
            "title_template": "⚠️ Alert: {{ .GroupLabels.alertname }}",
            "text_template": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
        },
        "enabled": True
    },
    
    "email_ops": {
        "type": "email",
        "name": "Operations Team Email",
        "config": {
            "smtp_server": "${SMTP_SERVER}",
            "smtp_port": 587,
            "smtp_username": "${SMTP_USERNAME}",
            "smtp_password": "${SMTP_PASSWORD}",
            "from_email": "alerts@spotify-ai.com",
            "to_emails": ["ops-team@spotify-ai.com"],
            "subject_template": "[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}",
            "body_template": """
Alert Details:
{{ range .Alerts }}
- Alert: {{ .Annotations.summary }}
- Description: {{ .Annotations.description }}
- Severity: {{ .Labels.severity }}
- Started: {{ .StartsAt }}
{{ if .GeneratorURL }}
- Dashboard: {{ .GeneratorURL }}
{{ end }}
{{ end }}
            """
        },
        "enabled": True
    },
    
    "pagerduty_critical": {
        "type": "pagerduty",
        "name": "PagerDuty Critical",
        "config": {
            "routing_key": "${PAGERDUTY_ROUTING_KEY}",
            "client": "Spotify-AI-Agent",
            "client_url": "https://monitoring.spotify-ai.com",
            "description_template": "{{ .GroupLabels.alertname }}: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
            "severity_mapping": {
                "critical": "critical",
                "high": "error",
                "medium": "warning",
                "low": "info"
            }
        },
        "enabled": True
    },
    
    "webhook_custom": {
        "type": "webhook",
        "name": "Custom Webhook",
        "config": {
            "url": "${CUSTOM_WEBHOOK_URL}",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer ${WEBHOOK_TOKEN}"
            },
            "payload_template": {
                "alert_name": "{{ .GroupLabels.alertname }}",
                "status": "{{ .Status }}",
                "alerts": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}",
                "timestamp": "{{ .Timestamp }}",
                "group_key": "{{ .GroupKey }}"
            }
        },
        "enabled": False
    }
}

# Escalation Policies
ESCALATION_POLICIES = {
    "critical_escalation": {
        "name": "Critical Alert Escalation",
        "rules": [
            {
                "level": 1,
                "delay_minutes": 0,
                "channels": ["slack_critical", "pagerduty_critical"]
            },
            {
                "level": 2,
                "delay_minutes": 15,
                "channels": ["email_ops"]
            },
            {
                "level": 3,
                "delay_minutes": 30,
                "channels": ["slack_critical", "pagerduty_critical"]
            }
        ]
    },
    
    "standard_escalation": {
        "name": "Standard Alert Escalation",
        "rules": [
            {
                "level": 1,
                "delay_minutes": 0,
                "channels": ["slack_general"]
            },
            {
                "level": 2,
                "delay_minutes": 30,
                "channels": ["email_ops"]
            }
        ]
    },
    
    "business_hours_only": {
        "name": "Business Hours Only",
        "rules": [
            {
                "level": 1,
                "delay_minutes": 0,
                "channels": ["slack_general"],
                "time_constraints": {
                    "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "hours": "09:00-17:00",
                    "timezone": "UTC"
                }
            }
        ]
    }
}

# Notification Rules
NOTIFICATION_RULES = {
    "critical_alerts": {
        "name": "Critical Alerts Routing",
        "severity_filter": ["critical"],
        "label_filters": {},
        "escalation_policy": "critical_escalation",
        "cooldown_minutes": 15,
        "enabled": True
    },
    
    "high_priority_alerts": {
        "name": "High Priority Alerts Routing", 
        "severity_filter": ["high"],
        "label_filters": {},
        "escalation_policy": "standard_escalation",
        "cooldown_minutes": 30,
        "enabled": True
    },
    
    "security_alerts": {
        "name": "Security Alerts Routing",
        "severity_filter": ["critical", "high", "medium"],
        "label_filters": {"category": "security"},
        "escalation_policy": "critical_escalation",
        "cooldown_minutes": 5,
        "enabled": True
    },
    
    "business_alerts": {
        "name": "Business Alerts Routing",
        "severity_filter": ["medium", "low"],
        "label_filters": {"category": "business"},
        "escalation_policy": "business_hours_only",
        "cooldown_minutes": 60,
        "enabled": True
    },
    
    "infrastructure_alerts": {
        "name": "Infrastructure Alerts Routing",
        "severity_filter": ["critical", "high"],
        "label_filters": {"team": "infrastructure"},
        "escalation_policy": "standard_escalation",
        "cooldown_minutes": 20,
        "enabled": True
    }
}

# Alert Grouping Rules
ALERT_GROUPING = {
    "by_service": {
        "group_by": ["service", "severity"],
        "group_wait": "10s",
        "group_interval": "5m",
        "repeat_interval": "12h"
    },
    
    "by_severity": {
        "group_by": ["severity"],
        "group_wait": "30s",
        "group_interval": "10m",
        "repeat_interval": "24h"
    },
    
    "by_team": {
        "group_by": ["team"],
        "group_wait": "1m",
        "group_interval": "15m",
        "repeat_interval": "24h"
    }
}

# Silence Rules
SILENCE_RULES = {
    "maintenance_window": {
        "name": "Maintenance Window",
        "matchers": [
            {"name": "alertname", "value": ".*", "is_regex": True}
        ],
        "start_time": "2024-01-01T02:00:00Z",
        "end_time": "2024-01-01T04:00:00Z",
        "created_by": "ops-team",
        "comment": "Scheduled maintenance window"
    },
    
    "known_issue": {
        "name": "Known Issue Silence",
        "matchers": [
            {"name": "alertname", "value": "HighErrorRate", "is_regex": False},
            {"name": "service", "value": "recommendation-api", "is_regex": False}
        ],
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-02T00:00:00Z",
        "created_by": "platform-team",
        "comment": "Known issue being tracked in JIRA-12345"
    }
}

# Export configuration as YAML for external tools
def export_alertmanager_config() -> str:
    """Export configuration in Alertmanager format."""
    config = {
        "global": {
            "smtp_smarthost": "${SMTP_SERVER}:587",
            "smtp_from": "alerts@spotify-ai.com"
        },
        "route": {
            "group_by": ["alertname", "severity"],
            "group_wait": "10s",
            "group_interval": "5m",
            "repeat_interval": "12h",
            "receiver": "default",
            "routes": [
                {
                    "match": {"severity": "critical"},
                    "receiver": "critical-alerts",
                    "group_wait": "0s",
                    "repeat_interval": "5m"
                },
                {
                    "match": {"category": "security"},
                    "receiver": "security-alerts",
                    "group_wait": "0s",
                    "repeat_interval": "1m"
                }
            ]
        },
        "receivers": [
            {
                "name": "default",
                "slack_configs": [
                    {
                        "api_url": "${SLACK_GENERAL_WEBHOOK_URL}",
                        "channel": "#alerts-general",
                        "title": "Alert: {{ .GroupLabels.alertname }}",
                        "text": "{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                    }
                ]
            },
            {
                "name": "critical-alerts",
                "slack_configs": [
                    {
                        "api_url": "${SLACK_CRITICAL_WEBHOOK_URL}",
                        "channel": "#alerts-critical",
                        "title": "🚨 CRITICAL: {{ .GroupLabels.alertname }}",
                        "text": "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}"
                    }
                ],
                "pagerduty_configs": [
                    {
                        "routing_key": "${PAGERDUTY_ROUTING_KEY}",
                        "description": "{{ .GroupLabels.alertname }}: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}"
                    }
                ]
            },
            {
                "name": "security-alerts",
                "slack_configs": [
                    {
                        "api_url": "${SLACK_CRITICAL_WEBHOOK_URL}",
                        "channel": "#security-alerts",
                        "title": "🔒 SECURITY ALERT: {{ .GroupLabels.alertname }}",
                        "text": "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}"
                    }
                ],
                "email_configs": [
                    {
                        "to": "security-team@spotify-ai.com",
                        "subject": "SECURITY ALERT: {{ .GroupLabels.alertname }}",
                        "body": "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
                    }
                ]
            }
        ]
    }
    
    return yaml.dump(config, default_flow_style=False)

def get_alert_template(template_name: str) -> Dict[str, Any]:
    """Get alert rule template by name."""
    if template_name not in ALERT_RULE_TEMPLATES:
        raise ValueError(f"Alert template not found: {template_name}")
    return ALERT_RULE_TEMPLATES[template_name].copy()

def get_notification_channel(channel_name: str) -> Dict[str, Any]:
    """Get notification channel configuration by name."""
    if channel_name not in NOTIFICATION_CHANNELS:
        raise ValueError(f"Notification channel not found: {channel_name}")
    return NOTIFICATION_CHANNELS[channel_name].copy()

# Export all configurations
__all__ = [
    'ALERT_RULE_TEMPLATES',
    'NOTIFICATION_CHANNELS', 
    'ESCALATION_POLICIES',
    'NOTIFICATION_RULES',
    'ALERT_GROUPING',
    'SILENCE_RULES',
    'export_alertmanager_config',
    'get_alert_template',
    'get_notification_channel'
]
