#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formateur d'Alertes Slack Intelligent et Contextuel

Ce module orchestre le formatage d'alertes Slack avec:
- Formatage contextuel par tenant et sévérité
- Intégration avec tous les composants (localisation, templates, cache)
- Agrégation intelligente des alertes similaires
- Enrichissement automatique des données
- Validation et sanitisation des données
- Support des alertes batch et temps réel
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
from prometheus_client import Counter, Histogram, Gauge
import hashlib
import re
from pathlib import Path

logger = structlog.get_logger(__name__)

# Métriques Prometheus
ALERT_FORMAT_REQUESTS = Counter(
    'slack_alert_format_requests_total',
    'Total alert format requests',
    ['tenant', 'severity', 'status', 'result']
)

ALERT_FORMAT_DURATION = Histogram(
    'slack_alert_format_duration_seconds',
    'Alert formatting duration',
    ['template', 'complexity']
)

ALERT_AGGREGATIONS = Counter(
    'slack_alert_aggregations_total',
    'Total alert aggregations',
    ['tenant', 'type']
)

ACTIVE_ALERT_CONTEXTS = Gauge(
    'slack_active_alert_contexts',
    'Number of active alert contexts'
)

class AlertStatus(Enum):
    """Statuts d'alerte possibles."""
    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"
    SILENCED = "silenced"

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    UNKNOWN = "unknown"

class AggregationType(Enum):
    """Types d'agrégation d'alertes."""
    NONE = "none"
    BY_SERVICE = "by_service"
    BY_SEVERITY = "by_severity"
    BY_TENANT = "by_tenant"
    SMART = "smart"

@dataclass
class AlertMetadata:
    """Métadonnées enrichies d'une alerte."""
    fingerprint: str
    tenant_id: str
    severity: AlertSeverity
    status: AlertStatus
    service: str
    environment: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    tags: Dict[str, str] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FormattedAlert:
    """Résultat du formatage d'une alerte."""
    slack_payload: str
    metadata: AlertMetadata
    template_used: str
    processing_time: float
    cached: bool = False
    aggregated: bool = False
    aggregation_count: int = 1

@dataclass
class AlertAggregation:
    """Agrégation d'alertes similaires."""
    key: str
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    count: int = 0
    
    def add_alert(self, alert: Dict[str, Any]):
        """Ajoute une alerte à l'agrégation."""
        self.alerts.append(alert)
        self.count += 1
        self.last_seen = datetime.utcnow()

class AlertFormatter:
    """
    Formateur d'alertes Slack intelligent avec contexte tenant.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        template_renderer=None,
        tenant_context=None,
        localization_engine=None,
        cache_manager=None
    ):
        self.config = config
        self.template_renderer = template_renderer
        self.tenant_context = tenant_context
        self.localization_engine = localization_engine
        self.cache_manager = cache_manager
        self.logger = logger.bind(component="alert_formatter")
        
        # Configuration
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_ttl = config.get("cache_ttl", 300)
        self.aggregation_enabled = config.get("aggregation_enabled", True)
        self.aggregation_window = config.get("aggregation_window", 300)  # 5 minutes
        self.max_aggregation_size = config.get("max_aggregation_size", 10)
        self.enrichment_enabled = config.get("enrichment_enabled", True)
        self.validation_enabled = config.get("validation_enabled", True)
        
        # Stockage des agrégations
        self._aggregations: Dict[str, AlertAggregation] = {}
        self._aggregation_lock = asyncio.Lock()
        
        # Cache des templates par sévérité
        self._severity_templates = {
            AlertSeverity.CRITICAL: "alert_critical.json.j2",
            AlertSeverity.WARNING: "alert_warning.json.j2",
            AlertSeverity.INFO: "alert_info.json.j2",
            AlertSeverity.UNKNOWN: "alert_info.json.j2"
        }
        
        # Cache des contextes actifs
        self._active_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Tâche de nettoyage des agrégations
        self._cleanup_task = None
        
        # Initialisation
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialise le formateur d'alertes."""
        try:
            await self._load_severity_templates()
            await self._start_cleanup_task()
            
            self.logger.info(
                "Formateur d'alertes initialisé",
                aggregation_enabled=self.aggregation_enabled,
                cache_enabled=self.cache_enabled
            )
            
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation", error=str(e))
            raise
    
    async def _load_severity_templates(self):
        """Charge les templates par sévérité depuis la configuration."""
        severity_config = self.config.get("severity_templates", {})
        
        # Fusionner avec les templates par défaut
        self._severity_templates.update({
            AlertSeverity(k): v for k, v in severity_config.items()
            if k in [s.value for s in AlertSeverity]
        })
    
    async def _start_cleanup_task(self):
        """Démarre la tâche de nettoyage des agrégations."""
        self._cleanup_task = asyncio.create_task(self._cleanup_aggregations())
    
    async def _cleanup_aggregations(self):
        """Nettoie périodiquement les agrégations expirées."""
        while True:
            try:
                await asyncio.sleep(60)  # Toutes les minutes
                
                async with self._aggregation_lock:
                    cutoff_time = datetime.utcnow() - timedelta(seconds=self.aggregation_window)
                    expired_keys = [
                        key for key, agg in self._aggregations.items()
                        if agg.last_seen < cutoff_time
                    ]
                    
                    for key in expired_keys:
                        del self._aggregations[key]
                    
                    if expired_keys:
                        self.logger.debug(
                            "Agrégations nettoyées",
                            expired_count=len(expired_keys),
                            active_count=len(self._aggregations)
                        )
                
                ACTIVE_ALERT_CONTEXTS.set(len(self._aggregations))
                
            except Exception as e:
                self.logger.error("Erreur nettoyage agrégations", error=str(e))
    
    async def format_alert(
        self,
        alert_data: Dict[str, Any],
        tenant_id: Optional[str] = None,
        locale: Optional[str] = None,
        template_override: Optional[str] = None,
        **kwargs
    ) -> FormattedAlert:
        """
        Formate une alerte pour Slack.
        
        Args:
            alert_data: Données de l'alerte
            tenant_id: ID du tenant
            locale: Locale pour la localisation
            template_override: Template spécifique à utiliser
            **kwargs: Options additionnelles
            
        Returns:
            Alerte formatée
        """
        start_time = datetime.utcnow()
        
        try:
            # Extraire et valider les métadonnées
            metadata = await self._extract_alert_metadata(alert_data, tenant_id)
            
            # Vérifier le cache
            if self.cache_enabled:
                cached_result = await self._get_cached_alert(metadata.fingerprint)
                if cached_result:
                    cached_result.cached = True
                    return cached_result
            
            # Récupérer le contexte tenant
            if not tenant_id:
                tenant_id = metadata.tenant_id
            
            tenant_context = await self._get_tenant_context(tenant_id, **kwargs)
            if not tenant_context:
                raise ValueError(f"Contexte tenant non trouvé: {tenant_id}")
            
            # Vérifier l'agrégation
            if self.aggregation_enabled:
                aggregation_result = await self._check_aggregation(alert_data, metadata)
                if aggregation_result:
                    return aggregation_result
            
            # Enrichir les données d'alerte
            if self.enrichment_enabled:
                alert_data = await self._enrich_alert_data(alert_data, metadata, tenant_context)
            
            # Déterminer le template à utiliser
            template_name = template_override or self._determine_template(metadata)
            
            # Préparer le contexte de rendu
            render_context = await self._prepare_render_context(
                alert_data, metadata, tenant_context, locale
            )
            
            # Rendre le template
            formatted_payload = await self._render_template(template_name, render_context)
            
            # Valider le résultat
            if self.validation_enabled:
                await self._validate_formatted_alert(formatted_payload)
            
            # Créer le résultat
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            formatted_alert = FormattedAlert(
                slack_payload=formatted_payload,
                metadata=metadata,
                template_used=template_name,
                processing_time=processing_time
            )
            
            # Mettre en cache
            if self.cache_enabled:
                await self._cache_alert(metadata.fingerprint, formatted_alert)
            
            # Métriques
            ALERT_FORMAT_REQUESTS.labels(
                tenant=tenant_id,
                severity=metadata.severity.value,
                status=metadata.status.value,
                result="success"
            ).inc()
            
            ALERT_FORMAT_DURATION.labels(
                template=template_name,
                complexity="single"
            ).observe(processing_time)
            
            return formatted_alert
            
        except Exception as e:
            self.logger.error(
                "Erreur formatage alerte",
                tenant_id=tenant_id,
                fingerprint=alert_data.get("fingerprint"),
                error=str(e)
            )
            
            ALERT_FORMAT_REQUESTS.labels(
                tenant=tenant_id or "unknown",
                severity="unknown",
                status="unknown",
                result="error"
            ).inc()
            
            raise
    
    async def _extract_alert_metadata(
        self,
        alert_data: Dict[str, Any],
        tenant_id: Optional[str]
    ) -> AlertMetadata:
        """Extrait les métadonnées d'une alerte."""
        labels = alert_data.get("labels", {})
        annotations = alert_data.get("annotations", {})
        
        # Déterminer le tenant
        if not tenant_id:
            tenant_id = labels.get("tenant") or labels.get("tenant_id") or "default"
        
        # Extraire la sévérité
        severity_str = labels.get("severity", "unknown").lower()
        try:
            severity = AlertSeverity(severity_str)
        except ValueError:
            severity = AlertSeverity.UNKNOWN
        
        # Extraire le statut
        status_str = alert_data.get("status", "firing").lower()
        try:
            status = AlertStatus(status_str)
        except ValueError:
            status = AlertStatus.FIRING
        
        # Calculer la durée si alerte résolue
        duration = None
        if status == AlertStatus.RESOLVED:
            starts_at = alert_data.get("startsAt")
            ends_at = alert_data.get("endsAt")
            if starts_at and ends_at:
                try:
                    start_time = datetime.fromisoformat(starts_at.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(ends_at.replace('Z', '+00:00'))
                    duration = end_time - start_time
                except:
                    pass
        
        return AlertMetadata(
            fingerprint=alert_data.get("fingerprint", self._generate_fingerprint(alert_data)),
            tenant_id=tenant_id,
            severity=severity,
            status=status,
            service=labels.get("service", labels.get("job", "unknown")),
            environment=labels.get("environment", labels.get("env", "unknown")),
            timestamp=datetime.utcnow(),
            duration=duration,
            tags={k: v for k, v in labels.items() if k not in ["severity", "tenant", "service"]},
            custom_fields=annotations
        )
    
    def _generate_fingerprint(self, alert_data: Dict[str, Any]) -> str:
        """Génère un fingerprint unique pour l'alerte."""
        labels = alert_data.get("labels", {})
        # Utiliser les labels principaux pour le fingerprint
        key_labels = ["alertname", "service", "instance", "tenant"]
        fingerprint_data = {k: labels.get(k, "") for k in key_labels}
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    async def _get_cached_alert(self, fingerprint: str) -> Optional[FormattedAlert]:
        """Récupère une alerte formatée depuis le cache."""
        if not self.cache_manager:
            return None
        
        try:
            cached_data = await self.cache_manager.get(f"formatted_alert:{fingerprint}")
            if cached_data:
                # Reconstruire l'objet FormattedAlert
                metadata = AlertMetadata(**cached_data["metadata"])
                return FormattedAlert(
                    slack_payload=cached_data["slack_payload"],
                    metadata=metadata,
                    template_used=cached_data["template_used"],
                    processing_time=cached_data["processing_time"],
                    cached=True
                )
        except Exception as e:
            self.logger.warning("Erreur récupération cache alerte", error=str(e))
        
        return None
    
    async def _cache_alert(self, fingerprint: str, formatted_alert: FormattedAlert):
        """Met une alerte formatée en cache."""
        if not self.cache_manager:
            return
        
        try:
            cache_data = {
                "slack_payload": formatted_alert.slack_payload,
                "metadata": {
                    "fingerprint": formatted_alert.metadata.fingerprint,
                    "tenant_id": formatted_alert.metadata.tenant_id,
                    "severity": formatted_alert.metadata.severity.value,
                    "status": formatted_alert.metadata.status.value,
                    "service": formatted_alert.metadata.service,
                    "environment": formatted_alert.metadata.environment,
                    "timestamp": formatted_alert.metadata.timestamp.isoformat(),
                    "duration": formatted_alert.metadata.duration.total_seconds() if formatted_alert.metadata.duration else None,
                    "tags": formatted_alert.metadata.tags,
                    "custom_fields": formatted_alert.metadata.custom_fields
                },
                "template_used": formatted_alert.template_used,
                "processing_time": formatted_alert.processing_time
            }
            
            await self.cache_manager.set(
                f"formatted_alert:{fingerprint}",
                cache_data,
                ttl=self.cache_ttl
            )
        except Exception as e:
            self.logger.warning("Erreur mise en cache alerte", error=str(e))
    
    async def _get_tenant_context(self, tenant_id: str, **kwargs):
        """Récupère le contexte tenant."""
        if not self.tenant_context:
            # Contexte par défaut
            return {
                "id": tenant_id,
                "name": tenant_id.title(),
                "locale": "fr_FR",
                "timezone": "UTC"
            }
        
        context = await self.tenant_context.get_tenant_context(
            tenant_id,
            **kwargs
        )
        
        return context.tenant if context else None
    
    async def _check_aggregation(
        self,
        alert_data: Dict[str, Any],
        metadata: AlertMetadata
    ) -> Optional[FormattedAlert]:
        """Vérifie si l'alerte doit être agrégée."""
        aggregation_key = self._generate_aggregation_key(metadata)
        
        async with self._aggregation_lock:
            aggregation = self._aggregations.get(aggregation_key)
            
            if aggregation is None:
                # Créer une nouvelle agrégation
                aggregation = AlertAggregation(key=aggregation_key)
                self._aggregations[aggregation_key] = aggregation
            
            # Ajouter l'alerte à l'agrégation
            aggregation.add_alert(alert_data)
            
            # Vérifier si on doit déclencher l'agrégation
            if aggregation.count >= self.max_aggregation_size:
                # Formater l'alerte agrégée
                return await self._format_aggregated_alert(aggregation, metadata)
            
            # Pas d'agrégation pour l'instant
            return None
    
    def _generate_aggregation_key(self, metadata: AlertMetadata) -> str:
        """Génère une clé d'agrégation basée sur les critères."""
        # Critères d'agrégation: tenant + service + sévérité
        key_parts = [
            metadata.tenant_id,
            metadata.service,
            metadata.severity.value
        ]
        return "|".join(key_parts)
    
    async def _format_aggregated_alert(
        self,
        aggregation: AlertAggregation,
        metadata: AlertMetadata
    ) -> FormattedAlert:
        """Formate une alerte agrégée."""
        # Créer des données d'alerte agrégées
        aggregated_data = {
            "status": "firing",
            "labels": {
                "tenant": metadata.tenant_id,
                "service": metadata.service,
                "severity": metadata.severity.value,
                "alertname": f"Alertes Multiples - {metadata.service}"
            },
            "annotations": {
                "summary": f"{aggregation.count} alertes similaires détectées",
                "description": f"Agrégation de {aggregation.count} alertes pour le service {metadata.service}",
                "aggregation_count": str(aggregation.count),
                "first_seen": aggregation.first_seen.isoformat(),
                "last_seen": aggregation.last_seen.isoformat()
            },
            "fingerprint": f"aggregated_{aggregation.key}_{aggregation.count}"
        }
        
        # Utiliser le template d'agrégation
        template_name = "alert_aggregated.json.j2"
        
        # Récupérer le contexte tenant
        tenant_context = await self._get_tenant_context(metadata.tenant_id)
        
        # Préparer le contexte de rendu
        render_context = await self._prepare_render_context(
            aggregated_data, metadata, tenant_context
        )
        
        # Ajouter les détails d'agrégation
        render_context.additional_context["aggregation"] = {
            "count": aggregation.count,
            "alerts": aggregation.alerts[:5],  # Limiter pour éviter surcharge
            "timespan": (aggregation.last_seen - aggregation.first_seen).total_seconds()
        }
        
        # Rendre le template
        formatted_payload = await self._render_template(template_name, render_context)
        
        # Métriques
        ALERT_AGGREGATIONS.labels(
            tenant=metadata.tenant_id,
            type="count_based"
        ).inc()
        
        # Nettoyer l'agrégation
        del self._aggregations[aggregation.key]
        
        return FormattedAlert(
            slack_payload=formatted_payload,
            metadata=metadata,
            template_used=template_name,
            processing_time=0.1,  # Estimation
            aggregated=True,
            aggregation_count=aggregation.count
        )
    
    async def _enrich_alert_data(
        self,
        alert_data: Dict[str, Any],
        metadata: AlertMetadata,
        tenant_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrichit les données d'alerte avec des informations contextuelles."""
        enriched_data = alert_data.copy()
        
        # Ajouter des informations de contexte
        enriched_data.setdefault("context", {}).update({
            "tenant_name": tenant_context.get("name", metadata.tenant_id),
            "environment_display": metadata.environment.title(),
            "service_display": metadata.service.replace("-", " ").title(),
            "severity_display": metadata.severity.value.title(),
            "status_display": metadata.status.value.title()
        })
        
        # Ajouter des URLs utiles
        base_url = tenant_context.get("base_url", "")
        if base_url:
            enriched_data["context"]["urls"] = {
                "dashboard": f"{base_url}/dashboard/{metadata.service}",
                "metrics": f"{base_url}/metrics/{metadata.service}",
                "logs": f"{base_url}/logs?service={metadata.service}",
                "runbook": f"{base_url}/runbooks/{metadata.service}"
            }
        
        # Ajouter des métadonnées temporelles
        enriched_data["context"]["timing"] = {
            "formatted_time": metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "relative_time": self._get_relative_time(metadata.timestamp),
            "duration_formatted": self._format_duration(metadata.duration) if metadata.duration else None
        }
        
        return enriched_data
    
    def _get_relative_time(self, timestamp: datetime) -> str:
        """Retourne le temps relatif (il y a X minutes)."""
        now = datetime.utcnow()
        diff = now - timestamp
        
        if diff.total_seconds() < 60:
            return "il y a quelques secondes"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() // 60)
            return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() // 3600)
            return f"il y a {hours} heure{'s' if hours > 1 else ''}"
        else:
            days = int(diff.total_seconds() // 86400)
            return f"il y a {days} jour{'s' if days > 1 else ''}"
    
    def _format_duration(self, duration: timedelta) -> str:
        """Formate une durée de manière lisible."""
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds} seconde{'s' if total_seconds > 1 else ''}"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''}"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            if minutes > 0:
                return f"{hours}h{minutes:02d}"
            else:
                return f"{hours} heure{'s' if hours > 1 else ''}"
    
    def _determine_template(self, metadata: AlertMetadata) -> str:
        """Détermine le template à utiliser selon la sévérité et le statut."""
        if metadata.status == AlertStatus.RESOLVED:
            return "alert_resolved.json.j2"
        
        return self._severity_templates.get(metadata.severity, "alert_info.json.j2")
    
    async def _prepare_render_context(
        self,
        alert_data: Dict[str, Any],
        metadata: AlertMetadata,
        tenant_context: Dict[str, Any],
        locale: Optional[str] = None
    ):
        """Prépare le contexte de rendu pour le template."""
        from .template_renderer import RenderContext
        
        # Déterminer la locale
        if not locale:
            locale = tenant_context.get("locale", "fr_FR")
        
        # Récupérer la configuration
        config = {
            "channel": tenant_context.get("slack_config", {}).get("channel", "#alerts"),
            "bot_name": tenant_context.get("slack_config", {}).get("bot_name", "Spotify AI Agent"),
            "base_url": tenant_context.get("base_url", ""),
            "ack_url": tenant_context.get("ack_url", "")
        }
        
        return RenderContext(
            alert=alert_data,
            tenant=tenant_context,
            config=config,
            locale=locale,
            timezone=tenant_context.get("timezone", "UTC")
        )
    
    async def _render_template(self, template_name: str, context) -> str:
        """Rend un template avec le contexte fourni."""
        if not self.template_renderer:
            # Fallback simple sans template
            return json.dumps({
                "text": f"Alerte: {context.alert.get('annotations', {}).get('summary', 'Alerte détectée')}",
                "channel": context.config.get("channel", "#alerts")
            })
        
        return await self.template_renderer.render_template(
            template_name,
            context,
            cache_enabled=self.cache_enabled
        )
    
    async def _validate_formatted_alert(self, formatted_payload: str):
        """Valide le payload Slack formaté."""
        try:
            # Vérifier que c'est du JSON valide
            payload_data = json.loads(formatted_payload)
            
            # Vérifications basiques Slack
            if "text" not in payload_data and "attachments" not in payload_data:
                raise ValueError("Payload Slack doit contenir 'text' ou 'attachments'")
            
            # Vérifier la taille
            if len(formatted_payload) > 40000:  # Limite Slack
                raise ValueError("Payload trop volumineux pour Slack")
            
            # Vérifier les champs obligatoires
            if "channel" not in payload_data:
                self.logger.warning("Canal Slack manquant dans le payload")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Payload JSON invalide: {str(e)}")
    
    async def format_batch_alerts(
        self,
        alerts_data: List[Dict[str, Any]],
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> List[FormattedAlert]:
        """
        Formate plusieurs alertes en lot.
        
        Args:
            alerts_data: Liste des données d'alertes
            tenant_id: ID du tenant
            **kwargs: Options additionnelles
            
        Returns:
            Liste des alertes formatées
        """
        start_time = datetime.utcnow()
        
        try:
            # Traitement en parallèle avec limite de concurrence
            semaphore = asyncio.Semaphore(10)  # Max 10 alertes simultanées
            
            async def format_single_alert(alert_data):
                async with semaphore:
                    return await self.format_alert(alert_data, tenant_id, **kwargs)
            
            tasks = [format_single_alert(alert_data) for alert_data in alerts_data]
            formatted_alerts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrer les erreurs
            successful_alerts = []
            errors = []
            
            for i, result in enumerate(formatted_alerts):
                if isinstance(result, Exception):
                    errors.append({
                        "index": i,
                        "alert": alerts_data[i],
                        "error": str(result)
                    })
                else:
                    successful_alerts.append(result)
            
            if errors:
                self.logger.warning(
                    "Erreurs lors du formatage batch",
                    error_count=len(errors),
                    total_count=len(alerts_data)
                )
            
            # Métriques
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            ALERT_FORMAT_DURATION.labels(
                template="batch",
                complexity="multiple"
            ).observe(processing_time)
            
            return successful_alerts
            
        except Exception as e:
            self.logger.error(
                "Erreur formatage batch alertes",
                alerts_count=len(alerts_data),
                error=str(e)
            )
            raise
    
    async def get_formatting_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de formatage."""
        return {
            "aggregations_active": len(self._aggregations),
            "aggregation_enabled": self.aggregation_enabled,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "aggregation_window": self.aggregation_window,
            "max_aggregation_size": self.max_aggregation_size,
            "severity_templates": {
                k.value: v for k, v in self._severity_templates.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du formateur d'alertes."""
        try:
            health = {
                "status": "healthy",
                "aggregations_count": len(self._aggregations),
                "cache_enabled": self.cache_enabled,
                "template_renderer_available": self.template_renderer is not None,
                "tenant_context_available": self.tenant_context is not None,
                "localization_available": self.localization_engine is not None,
                "last_check": datetime.utcnow().isoformat()
            }
            
            # Test de formatage simple
            test_alert = {
                "status": "firing",
                "labels": {"severity": "info", "service": "test"},
                "annotations": {"summary": "Test de santé"}
            }
            
            test_result = await self.format_alert(test_alert, tenant_id="test")
            health["test_formatting"] = "success"
            
            return health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }

# Factory function
def create_alert_formatter(
    config: Dict[str, Any],
    template_renderer=None,
    tenant_context=None,
    localization_engine=None,
    cache_manager=None
) -> AlertFormatter:
    """Crée une instance du formateur d'alertes."""
    return AlertFormatter(
        config,
        template_renderer,
        tenant_context,
        localization_engine,
        cache_manager
    )
