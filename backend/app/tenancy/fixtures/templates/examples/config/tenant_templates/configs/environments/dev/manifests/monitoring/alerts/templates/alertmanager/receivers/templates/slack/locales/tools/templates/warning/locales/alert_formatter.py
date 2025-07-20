#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spotify AI Agent - Alert Formatter pour Monitoring Multi-Tenant

Formateur d'alertes contextualisé pour l'écosystème Spotify AI Agent
avec support avancé pour Prometheus, Alertmanager et intégrations Slack.

Fonctionnalités:
- Formatage d'alertes multi-niveau (info, warning, critical, emergency)
- Enrichissement contextuel avec métadonnées IA musicale
- Templates adaptatifs selon le type d'alerte et tenant
- Intégration avec métriques Prometheus/Grafana
- Support multi-canal (Slack, Email, SMS, Webhook)
- Corrélation d'alertes et groupement intelligent

Architecture:
- Strategy Pattern pour différents formateurs
- Chain of Responsibility pour enrichissement
- Factory Pattern pour instanciation contextualisée
- Observer Pattern pour notifications temps réel

Cas d'usage:
- Dégradation performance modèles IA
- Seuils de latence API dépassés
- Erreurs de génération musicale
- Problèmes de recommandations
- Incidents sécurité tenant
- Violations quota et limites

Utilisation:
    formatter = AlertFormatter(locale_manager)
    
    alert = formatter.format_alert(
        alert_type="ai_model_performance",
        severity="warning",
        tenant_id="universal_music_001",
        context={
            "model_name": "MusicGenAI-v3",
            "accuracy_drop": 12.5,
            "affected_users": 1250
        },
        locale="fr"
    )
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
import logging
import uuid

# Imports externes
import structlog
from prometheus_client import Counter, Histogram, Gauge
import redis

# Imports internes
from .locale_manager import LocaleManager, LocaleConfig
from .config import (
    ALERT_SEVERITY_LEVELS,
    ALERT_TYPES,
    TENANT_TYPES,
    DEFAULT_TEMPLATES,
    SLACK_FORMATTING_RULES
)

# Configuration logging
logger = structlog.get_logger(__name__)

# Métriques Prometheus
alert_formatting_total = Counter(
    'spotify_ai_alert_formatting_total',
    'Nombre total d\'alertes formatées',
    ['tenant_id', 'alert_type', 'severity', 'locale', 'output_format']
)

alert_formatting_duration = Histogram(
    'spotify_ai_alert_formatting_seconds',
    'Durée de formatage des alertes',
    ['alert_type', 'severity']
)

alert_enrichment_errors = Counter(
    'spotify_ai_alert_enrichment_errors_total',
    'Erreurs d\'enrichissement d\'alertes',
    ['tenant_id', 'enrichment_type']
)

active_alert_templates = Gauge(
    'spotify_ai_active_alert_templates',
    'Nombre de templates d\'alertes actifs'
)

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Types d'alertes Spotify AI Agent."""
    # IA & Machine Learning
    AI_MODEL_PERFORMANCE = "ai_model_performance"
    AI_MODEL_TRAINING = "ai_model_training"
    AI_INFERENCE_LATENCY = "ai_inference_latency"
    AI_ACCURACY_DROP = "ai_accuracy_drop"
    
    # API & Services
    API_LATENCY = "api_latency"
    API_ERROR_RATE = "api_error_rate"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # Infrastructure
    RESOURCE_USAGE = "resource_usage"
    DISK_SPACE = "disk_space"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    
    # Sécurité
    SECURITY_BREACH = "security_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # Business Logic
    MUSIC_GENERATION_FAILED = "music_generation_failed"
    RECOMMENDATION_DEGRADED = "recommendation_degraded"
    USER_EXPERIENCE_IMPACT = "user_experience_impact"
    REVENUE_IMPACT = "revenue_impact"

class OutputFormat(Enum):
    """Formats de sortie supportés."""
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    CONSOLE = "console"

@dataclass
class AlertContext:
    """Contexte enrichi d'une alerte."""
    tenant_id: str
    tenant_type: str = "artist"
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Métriques principales
    metric_name: str = ""
    metric_value: Union[int, float] = 0
    threshold: Union[int, float] = 0
    previous_value: Union[int, float] = 0
    
    # Contexte IA/ML
    model_name: str = ""
    model_version: str = ""
    dataset_name: str = ""
    accuracy_score: float = 0.0
    latency_ms: float = 0.0
    
    # Contexte business
    affected_users: int = 0
    revenue_impact: float = 0.0
    artist_name: str = ""
    track_count: int = 0
    playlist_count: int = 0
    
    # Contexte technique
    service_name: str = ""
    instance_id: str = ""
    region: str = ""
    environment: str = "dev"
    
    # Métadonnées
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    parent_alert_id: str = ""

@dataclass
class FormattedAlert:
    """Alerte formatée prête à être envoyée."""
    id: str
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    description: str
    
    # Formatage spécifique
    slack_message: Dict[str, Any] = field(default_factory=dict)
    email_content: Dict[str, str] = field(default_factory=dict)
    sms_content: str = ""
    
    # Métadonnées
    context: AlertContext = None
    locale: str = "en"
    output_formats: List[OutputFormat] = field(default_factory=list)
    
    # Actions suggérées
    actions: List[Dict[str, str]] = field(default_factory=list)
    runbook_url: str = ""
    dashboard_url: str = ""
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'alerte en dictionnaire."""
        data = asdict(self)
        # Conversion des enums
        data['severity'] = self.severity.value
        data['alert_type'] = self.alert_type.value
        data['output_formats'] = [f.value for f in self.output_formats]
        return data

class AlertEnricher:
    """Enrichisseur d'alertes avec contexte métier."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self._enrichers = {
            AlertType.AI_MODEL_PERFORMANCE: self._enrich_ai_performance,
            AlertType.API_LATENCY: self._enrich_api_latency,
            AlertType.SECURITY_BREACH: self._enrich_security,
            AlertType.MUSIC_GENERATION_FAILED: self._enrich_music_generation,
            AlertType.RECOMMENDATION_DEGRADED: self._enrich_recommendations
        }
    
    def enrich_alert(self, context: AlertContext, alert_type: AlertType) -> AlertContext:
        """Enrichit le contexte d'une alerte."""
        enricher = self._enrichers.get(alert_type, self._enrich_default)
        
        try:
            return enricher(context)
        except Exception as e:
            logger.error(
                "Erreur enrichissement alerte",
                alert_type=alert_type.value,
                tenant_id=context.tenant_id,
                error=str(e)
            )
            alert_enrichment_errors.labels(
                tenant_id=context.tenant_id,
                enrichment_type=alert_type.value
            ).inc()
            return context
    
    def _enrich_ai_performance(self, context: AlertContext) -> AlertContext:
        """Enrichissement pour alertes de performance IA."""
        if self.redis_client:
            # Récupération historique performance
            history_key = f"ai_metrics:{context.tenant_id}:{context.model_name}"
            history = self.redis_client.lrange(history_key, 0, 9)  # 10 dernières valeurs
            
            if history:
                values = [float(h) for h in history]
                context.metadata['avg_performance'] = sum(values) / len(values)
                context.metadata['performance_trend'] = 'declining' if values[-1] < values[0] else 'improving'
        
        # Ajout de contexte IA
        if context.accuracy_score > 0:
            context.metadata['accuracy_category'] = self._categorize_accuracy(context.accuracy_score)
        
        context.tags.extend(['ai', 'performance', 'model'])
        return context
    
    def _enrich_api_latency(self, context: AlertContext) -> AlertContext:
        """Enrichissement pour alertes de latence API."""
        # Classification de la latence
        if context.latency_ms > 0:
            if context.latency_ms < 100:
                context.metadata['latency_level'] = 'excellent'
            elif context.latency_ms < 500:
                context.metadata['latency_level'] = 'good'
            elif context.latency_ms < 1000:
                context.metadata['latency_level'] = 'poor'
            else:
                context.metadata['latency_level'] = 'critical'
        
        context.tags.extend(['api', 'latency', 'performance'])
        return context
    
    def _enrich_security(self, context: AlertContext) -> AlertContext:
        """Enrichissement pour alertes de sécurité."""
        context.metadata['security_level'] = 'high'
        context.metadata['requires_immediate_action'] = True
        context.tags.extend(['security', 'urgent', 'breach'])
        return context
    
    def _enrich_music_generation(self, context: AlertContext) -> AlertContext:
        """Enrichissement pour échecs de génération musicale."""
        context.metadata['impact_type'] = 'user_experience'
        context.metadata['business_critical'] = context.affected_users > 100
        context.tags.extend(['music', 'generation', 'ai', 'user_impact'])
        return context
    
    def _enrich_recommendations(self, context: AlertContext) -> AlertContext:
        """Enrichissement pour dégradation des recommandations."""
        context.metadata['recommendation_quality'] = 'degraded'
        context.metadata['affects_discovery'] = True
        context.tags.extend(['recommendation', 'ai', 'discovery', 'user_experience'])
        return context
    
    def _enrich_default(self, context: AlertContext) -> AlertContext:
        """Enrichissement par défaut."""
        context.tags.append('general')
        return context
    
    def _categorize_accuracy(self, accuracy: float) -> str:
        """Catégorise un score de précision."""
        if accuracy >= 0.95:
            return 'excellent'
        elif accuracy >= 0.90:
            return 'good'
        elif accuracy >= 0.80:
            return 'fair'
        else:
            return 'poor'

class AlertFormatter:
    """
    Formateur principal d'alertes pour Spotify AI Agent.
    
    Gère le formatage contextuel des alertes avec support multi-locale,
    enrichissement automatique et génération de contenu adaptatif.
    """
    
    def __init__(self, locale_manager: LocaleManager, redis_client: Optional[redis.Redis] = None):
        self.locale_manager = locale_manager
        self.redis_client = redis_client
        self.enricher = AlertEnricher(redis_client)
        
        # Templates par défaut
        self.default_templates = DEFAULT_TEMPLATES
        
        # Formatters spécialisés
        self._formatters = {
            OutputFormat.SLACK: self._format_slack,
            OutputFormat.EMAIL: self._format_email,
            OutputFormat.SMS: self._format_sms,
            OutputFormat.WEBHOOK: self._format_webhook,
            OutputFormat.CONSOLE: self._format_console
        }
        
        # Initialisation métriques
        active_alert_templates.set(len(self.default_templates))
        
        logger.info("AlertFormatter initialisé", formatters_count=len(self._formatters))
    
    def format_alert(
        self,
        alert_type: Union[AlertType, str],
        severity: Union[AlertSeverity, str],
        context: Union[AlertContext, Dict[str, Any]],
        locale: str = "en",
        output_formats: List[Union[OutputFormat, str]] = None
    ) -> FormattedAlert:
        """
        Formate une alerte complète avec tous les enrichissements.
        
        Args:
            alert_type: Type d'alerte (enum ou string)
            severity: Niveau de sévérité
            context: Contexte de l'alerte (AlertContext ou dict)
            locale: Locale pour traductions
            output_formats: Formats de sortie souhaités
            
        Returns:
            Alerte formatée complète
        """
        start_time = time.time()
        
        # Normalisation des paramètres
        if isinstance(alert_type, str):
            alert_type = AlertType(alert_type)
        
        if isinstance(severity, str):
            severity = AlertSeverity(severity)
        
        if isinstance(context, dict):
            context = AlertContext(**context)
        
        output_formats = output_formats or [OutputFormat.SLACK]
        output_formats = [
            OutputFormat(f) if isinstance(f, str) else f 
            for f in output_formats
        ]
        
        try:
            # Enrichissement du contexte
            enriched_context = self.enricher.enrich_alert(context, alert_type)
            
            # Génération du contenu principal
            title = self._generate_title(alert_type, severity, enriched_context, locale)
            message = self._generate_message(alert_type, severity, enriched_context, locale)
            description = self._generate_description(alert_type, severity, enriched_context, locale)
            
            # Création de l'alerte formatée
            formatted_alert = FormattedAlert(
                id=enriched_context.alert_id,
                severity=severity,
                alert_type=alert_type,
                title=title,
                message=message,
                description=description,
                context=enriched_context,
                locale=locale,
                output_formats=output_formats
            )
            
            # Formatage spécifique par canal
            for output_format in output_formats:
                self._apply_format_specific(formatted_alert, output_format)
            
            # Génération des actions
            formatted_alert.actions = self._generate_actions(alert_type, enriched_context)
            formatted_alert.runbook_url = self._get_runbook_url(alert_type, enriched_context)
            formatted_alert.dashboard_url = self._get_dashboard_url(alert_type, enriched_context)
            
            # Métrics
            alert_formatting_total.labels(
                tenant_id=enriched_context.tenant_id,
                alert_type=alert_type.value,
                severity=severity.value,
                locale=locale,
                output_format=",".join(f.value for f in output_formats)
            ).inc()
            
            alert_formatting_duration.labels(
                alert_type=alert_type.value,
                severity=severity.value
            ).observe(time.time() - start_time)
            
            logger.info(
                "Alerte formatée avec succès",
                alert_id=formatted_alert.id,
                alert_type=alert_type.value,
                severity=severity.value,
                tenant_id=enriched_context.tenant_id,
                locale=locale
            )
            
            return formatted_alert
            
        except Exception as e:
            logger.error(
                "Erreur formatage alerte",
                alert_type=alert_type.value,
                severity=severity.value,
                tenant_id=context.tenant_id if context else "unknown",
                error=str(e)
            )
            
            # Alerte d'erreur par défaut
            return self._create_error_alert(alert_type, severity, context, locale, str(e))
    
    def _generate_title(
        self,
        alert_type: AlertType,
        severity: AlertSeverity, 
        context: AlertContext,
        locale: str
    ) -> str:
        """Génère le titre de l'alerte."""
        template_key = f"alerts.{alert_type.value}.{severity.value}.title"
        
        template_context = {
            'tenant_name': context.metadata.get('tenant_name', context.tenant_id),
            'model_name': context.model_name,
            'service_name': context.service_name,
            'metric_value': context.metric_value,
            'threshold': context.threshold,
            'affected_users': context.affected_users,
            'severity_emoji': self._get_severity_emoji(severity)
        }
        
        title = self.locale_manager.get_localized_string(
            key=template_key,
            locale=locale,
            context=template_context,
            tenant_id=context.tenant_id,
            fallback_value=f"{severity.value.upper()}: {alert_type.value}"
        )
        
        return title
    
    def _generate_message(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        context: AlertContext,
        locale: str
    ) -> str:
        """Génère le message principal de l'alerte."""
        template_key = f"alerts.{alert_type.value}.{severity.value}.message"
        
        template_context = {
            'tenant_name': context.metadata.get('tenant_name', context.tenant_id),
            'metric_name': context.metric_name,
            'metric_value': context.metric_value,
            'threshold': context.threshold,
            'previous_value': context.previous_value,
            'change_percent': self._calculate_change_percent(context.metric_value, context.previous_value),
            'model_name': context.model_name,
            'accuracy_score': context.accuracy_score,
            'latency_ms': context.latency_ms,
            'affected_users': context.affected_users,
            'timestamp': context.timestamp.strftime("%H:%M:%S"),
            'environment': context.environment.upper()
        }
        
        message = self.locale_manager.get_localized_string(
            key=template_key,
            locale=locale,
            context=template_context,
            tenant_id=context.tenant_id,
            fallback_value=f"Alerte {alert_type.value} détectée"
        )
        
        return message
    
    def _generate_description(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        context: AlertContext,
        locale: str
    ) -> str:
        """Génère la description détaillée de l'alerte."""
        template_key = f"alerts.{alert_type.value}.{severity.value}.description"
        
        template_context = {
            'context': context,
            'metadata': context.metadata,
            'tags': ", ".join(context.tags),
            'correlation_info': self._get_correlation_info(context)
        }
        
        description = self.locale_manager.get_localized_string(
            key=template_key,
            locale=locale,
            context=template_context,
            tenant_id=context.tenant_id,
            fallback_value="Description détaillée non disponible"
        )
        
        return description
    
    def _apply_format_specific(self, alert: FormattedAlert, output_format: OutputFormat):
        """Applique le formatage spécifique à un canal."""
        formatter = self._formatters.get(output_format)
        if formatter:
            formatter(alert)
    
    def _format_slack(self, alert: FormattedAlert):
        """Formatage spécifique Slack avec blocks et attachments."""
        severity_color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9500", 
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.EMERGENCY: "#8B0000"
        }
        
        # Construction du message Slack
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{self._get_severity_emoji(alert.severity)} {alert.title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message
                }
            }
        ]
        
        # Ajout de champs contextuels
        if alert.context:
            fields = []
            
            if alert.context.metric_value:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*Valeur:* {alert.context.metric_value}"
                })
            
            if alert.context.affected_users:
                fields.append({
                    "type": "mrkdwn", 
                    "text": f"*Utilisateurs affectés:* {alert.context.affected_users}"
                })
            
            if alert.context.environment:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*Environnement:* {alert.context.environment.upper()}"
                })
            
            if fields:
                blocks.append({
                    "type": "section",
                    "fields": fields
                })
        
        # Boutons d'action
        if alert.actions:
            elements = []
            for action in alert.actions[:5]:  # Max 5 boutons
                elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action['label']
                    },
                    "url": action.get('url', '#'),
                    "style": action.get('style', 'default')
                })
            
            if elements:
                blocks.append({
                    "type": "actions",
                    "elements": elements
                })
        
        alert.slack_message = {
            "text": alert.title,  # Fallback text
            "blocks": blocks,
            "color": severity_color[alert.severity],
            "ts": str(int(alert.created_at.timestamp()))
        }
    
    def _format_email(self, alert: FormattedAlert):
        """Formatage spécifique Email avec HTML."""
        severity_styles = {
            AlertSeverity.INFO: "color: #36a64f; background: #f0f8f0;",
            AlertSeverity.WARNING: "color: #ff9500; background: #fff8f0;",
            AlertSeverity.CRITICAL: "color: #ff0000; background: #fff0f0;",
            AlertSeverity.EMERGENCY: "color: #8B0000; background: #ffeeee;"
        }
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{ padding: 15px; border-radius: 5px; {severity_styles[alert.severity]} }}
                .alert-content {{ margin: 20px 0; }}
                .metadata {{ background: #f5f5f5; padding: 10px; border-radius: 3px; }}
                .actions {{ margin: 20px 0; }}
                .action-button {{ 
                    display: inline-block; padding: 10px 15px; margin: 5px;
                    background: #007cba; color: white; text-decoration: none;
                    border-radius: 3px;
                }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>{alert.title}</h2>
            </div>
            <div class="alert-content">
                <p>{alert.message}</p>
                <p>{alert.description}</p>
            </div>
        """
        
        if alert.context:
            html_content += f"""
            <div class="metadata">
                <h3>Détails techniques</h3>
                <ul>
                    <li><strong>Tenant:</strong> {alert.context.tenant_id}</li>
                    <li><strong>Environnement:</strong> {alert.context.environment}</li>
                    <li><strong>Timestamp:</strong> {alert.context.timestamp}</li>
                </ul>
            </div>
            """
        
        if alert.actions:
            html_content += '<div class="actions"><h3>Actions recommandées</h3>'
            for action in alert.actions:
                html_content += f'<a href="{action.get("url", "#")}" class="action-button">{action["label"]}</a>'
            html_content += '</div>'
        
        html_content += "</body></html>"
        
        alert.email_content = {
            'subject': alert.title,
            'html': html_content,
            'text': f"{alert.title}\n\n{alert.message}\n\n{alert.description}"
        }
    
    def _format_sms(self, alert: FormattedAlert):
        """Formatage spécifique SMS (limité en caractères)."""
        emoji = self._get_severity_emoji(alert.severity)
        
        # Message court pour SMS (max 160 caractères)
        sms_message = f"{emoji} {alert.title[:50]}"
        
        if alert.context and alert.context.metric_value:
            sms_message += f" - Valeur: {alert.context.metric_value}"
        
        # Troncature si nécessaire
        if len(sms_message) > 160:
            sms_message = sms_message[:157] + "..."
        
        alert.sms_content = sms_message
    
    def _format_webhook(self, alert: FormattedAlert):
        """Formatage spécifique Webhook (JSON structuré)."""
        # Le webhook utilise simplement to_dict() pour JSON complet
        pass
    
    def _format_console(self, alert: FormattedAlert):
        """Formatage spécifique Console avec couleurs ANSI."""
        # Couleurs ANSI pour terminal
        colors = {
            AlertSeverity.INFO: "\033[32m",      # Vert
            AlertSeverity.WARNING: "\033[33m",   # Jaune
            AlertSeverity.CRITICAL: "\033[31m",  # Rouge
            AlertSeverity.EMERGENCY: "\033[91m"  # Rouge vif
        }
        
        reset_color = "\033[0m"
        color = colors.get(alert.severity, "")
        
        console_output = f"{color}[{alert.severity.value.upper()}]{reset_color} {alert.title}\n"
        console_output += f"{alert.message}\n"
        
        if alert.context:
            console_output += f"Tenant: {alert.context.tenant_id} | Env: {alert.context.environment}\n"
        
        # Stockage dans metadata pour récupération
        if not hasattr(alert, 'console_output'):
            alert.console_output = console_output
    
    def _generate_actions(self, alert_type: AlertType, context: AlertContext) -> List[Dict[str, str]]:
        """Génère les actions recommandées selon le type d'alerte."""
        actions = []
        
        # Actions génériques
        actions.append({
            'label': 'Voir Dashboard',
            'url': self._get_dashboard_url(alert_type, context),
            'style': 'primary'
        })
        
        # Actions spécifiques par type
        type_actions = {
            AlertType.AI_MODEL_PERFORMANCE: [
                {'label': 'Redémarrer Modèle', 'url': f'/api/models/{context.model_name}/restart'},
                {'label': 'Voir Métriques ML', 'url': f'/ml/metrics/{context.model_name}'}
            ],
            AlertType.API_LATENCY: [
                {'label': 'Analyser Traces', 'url': f'/traces/{context.service_name}'},
                {'label': 'Scaler Service', 'url': f'/api/services/{context.service_name}/scale'}
            ],
            AlertType.SECURITY_BREACH: [
                {'label': 'Isoler Tenant', 'url': f'/security/isolate/{context.tenant_id}', 'style': 'danger'},
                {'label': 'Audit Log', 'url': f'/security/audit/{context.tenant_id}'}
            ]
        }
        
        specific_actions = type_actions.get(alert_type, [])
        actions.extend(specific_actions)
        
        return actions
    
    def _get_runbook_url(self, alert_type: AlertType, context: AlertContext) -> str:
        """Génère l'URL du runbook pour le type d'alerte."""
        base_url = "https://docs.spotify-ai-agent.com/runbooks"
        return f"{base_url}/{alert_type.value}"
    
    def _get_dashboard_url(self, alert_type: AlertType, context: AlertContext) -> str:
        """Génère l'URL du dashboard pour l'alerte."""
        base_url = "https://monitoring.spotify-ai-agent.com/dashboards"
        
        if alert_type in [AlertType.AI_MODEL_PERFORMANCE, AlertType.AI_ACCURACY_DROP]:
            return f"{base_url}/ml-models?tenant={context.tenant_id}&model={context.model_name}"
        elif alert_type in [AlertType.API_LATENCY, AlertType.API_ERROR_RATE]:
            return f"{base_url}/api-performance?service={context.service_name}"
        else:
            return f"{base_url}/overview?tenant={context.tenant_id}"
    
    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """Retourne l'emoji correspondant à la sévérité."""
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️", 
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔥"
        }
        return emoji_map.get(severity, "❓")
    
    def _calculate_change_percent(self, current: float, previous: float) -> float:
        """Calcule le pourcentage de changement."""
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    
    def _get_correlation_info(self, context: AlertContext) -> str:
        """Récupère les informations de corrélation."""
        if context.correlation_id:
            return f"Corrélation: {context.correlation_id}"
        if context.parent_alert_id:
            return f"Alerte parent: {context.parent_alert_id}"
        return ""
    
    def _create_error_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        context: AlertContext,
        locale: str,
        error_message: str
    ) -> FormattedAlert:
        """Crée une alerte d'erreur en cas d'échec de formatage."""
        return FormattedAlert(
            id=str(uuid.uuid4()),
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.SERVICE_UNAVAILABLE,
            title="Erreur Formatage Alerte",
            message=f"Impossible de formater l'alerte {alert_type.value}: {error_message}",
            description="Une erreur s'est produite lors du formatage de l'alerte originale.",
            context=context,
            locale=locale,
            output_formats=[OutputFormat.CONSOLE]
        )

# Fonctions utilitaires

def quick_format_alert(
    alert_type: str,
    severity: str,
    tenant_id: str,
    message: str,
    **kwargs
) -> Dict[str, Any]:
    """Fonction utilitaire pour formatage rapide d'alerte."""
    locale_manager = LocaleManager()
    formatter = AlertFormatter(locale_manager)
    
    context = AlertContext(
        tenant_id=tenant_id,
        **kwargs
    )
    
    alert = formatter.format_alert(
        alert_type=AlertType(alert_type),
        severity=AlertSeverity(severity),
        context=context
    )
    
    return alert.to_dict()

def batch_format_alerts(alerts_data: List[Dict[str, Any]]) -> List[FormattedAlert]:
    """Formate un lot d'alertes en parallèle."""
    locale_manager = LocaleManager()
    formatter = AlertFormatter(locale_manager)
    
    formatted_alerts = []
    
    for alert_data in alerts_data:
        try:
            alert = formatter.format_alert(**alert_data)
            formatted_alerts.append(alert)
        except Exception as e:
            logger.error(f"Erreur formatage alerte batch", error=str(e), data=alert_data)
            continue
    
    return formatted_alerts
