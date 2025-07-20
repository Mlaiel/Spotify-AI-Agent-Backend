"""
Slack Alert Manager - Gestionnaire central des alertes Slack
Architecture multi-tenant avec intelligence artificielle intégrée
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import aioredis
import asyncpg
from fastapi import HTTPException
from pydantic import BaseModel, Field, validator

from .slack_template_engine import SlackTemplateEngine
from .slack_webhook_handler import SlackWebhookHandler
from .slack_alert_formatter import SlackAlertFormatter
from .slack_channel_router import SlackChannelRouter
from .slack_rate_limiter import SlackRateLimiter
from .slack_escalation_manager import SlackEscalationManager


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """États des alertes"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


@dataclass
class AlertContext:
    """Contexte enrichi d'une alerte"""
    tenant_id: str
    environment: str
    service_name: str
    component: str
    instance_id: Optional[str] = None
    region: str = "eu-west-1"
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AlertData:
    """Structure de données d'une alerte"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    context: AlertContext
    created_at: datetime
    updated_at: datetime
    fingerprint: str
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    suppression_rules: List[str] = None
    ai_insights: Dict[str, Any] = None

    def __post_init__(self):
        if self.suppression_rules is None:
            self.suppression_rules = []
        if self.ai_insights is None:
            self.ai_insights = {}


class SlackAlertRequest(BaseModel):
    """Modèle de requête pour l'envoi d'alertes"""
    tenant_id: str = Field(..., description="Identifiant du tenant")
    alert_type: str = Field(..., description="Type d'alerte")
    severity: AlertSeverity = Field(..., description="Niveau de sévérité")
    title: str = Field(..., description="Titre de l'alerte")
    description: str = Field(..., description="Description détaillée")
    service_name: str = Field(..., description="Nom du service")
    component: str = Field(..., description="Composant concerné")
    environment: str = Field(default="dev", description="Environnement")
    instance_id: Optional[str] = Field(None, description="ID de l'instance")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags personnalisés")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")
    suppress_until: Optional[datetime] = Field(None, description="Suppression temporaire")
    escalation_policy: Optional[str] = Field(None, description="Politique d'escalade")

    @validator('severity')
    def validate_severity(cls, v):
        if v not in AlertSeverity:
            raise ValueError(f"Sévérité invalide: {v}")
        return v


class SlackAlertManager:
    """
    Gestionnaire central des alertes Slack avec fonctionnalités avancées:
    - Multi-tenant avec isolation complète
    - Intelligence artificielle pour la corrélation
    - Gestion des escalades automatiques
    - Rate limiting intelligent
    - Audit trail complet
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        postgres_url: str = "postgresql://localhost:5432/spotify_ai",
        config: Optional[Dict[str, Any]] = None
    ):
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        self.config = config or {}
        
        # Composants principaux
        self.template_engine = SlackTemplateEngine()
        self.webhook_handler = SlackWebhookHandler()
        self.alert_formatter = SlackAlertFormatter()
        self.channel_router = SlackChannelRouter()
        self.rate_limiter = SlackRateLimiter()
        self.escalation_manager = SlackEscalationManager()
        
        # Cache et base de données
        self.redis_pool = None
        self.postgres_pool = None
        
        # Métriques et monitoring
        self.metrics = {
            "alerts_sent": 0,
            "alerts_failed": 0,
            "escalations_triggered": 0,
            "rate_limited": 0
        }
        
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialise les connexions et composants"""
        try:
            # Connexion Redis
            self.redis_pool = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Connexion PostgreSQL
            self.postgres_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=5,
                max_size=20
            )
            
            # Initialisation des composants
            await self.template_engine.initialize()
            await self.webhook_handler.initialize()
            await self.channel_router.initialize(self.redis_pool)
            await self.rate_limiter.initialize(self.redis_pool)
            await self.escalation_manager.initialize(self.redis_pool, self.postgres_pool)
            
            # Création des tables si nécessaire
            await self._create_database_schema()
            
            self.logger.info("SlackAlertManager initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise

    async def send_alert(
        self,
        alert_request: Union[SlackAlertRequest, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Envoie une alerte Slack avec traitement intelligent
        
        Args:
            alert_request: Données de l'alerte
            
        Returns:
            Résultat de l'envoi avec métadonnées
        """
        try:
            # Conversion en modèle Pydantic si nécessaire
            if isinstance(alert_request, dict):
                alert_request = SlackAlertRequest(**alert_request)
            
            # Génération de l'ID unique
            alert_id = str(uuid.uuid4())
            fingerprint = self._generate_fingerprint(alert_request)
            
            # Création du contexte d'alerte
            context = AlertContext(
                tenant_id=alert_request.tenant_id,
                environment=alert_request.environment,
                service_name=alert_request.service_name,
                component=alert_request.component,
                instance_id=alert_request.instance_id,
                tags=alert_request.tags,
                metadata=alert_request.metadata
            )
            
            # Vérification du rate limiting
            is_rate_limited = await self.rate_limiter.check_rate_limit(
                tenant_id=alert_request.tenant_id,
                alert_type=alert_request.alert_type,
                fingerprint=fingerprint
            )
            
            if is_rate_limited:
                self.metrics["rate_limited"] += 1
                self.logger.warning(f"Alerte rate limitée: {alert_id}")
                return {
                    "alert_id": alert_id,
                    "status": "rate_limited",
                    "message": "Alerte supprimée par rate limiting"
                }
            
            # Création de l'objet AlertData
            alert_data = AlertData(
                alert_id=alert_id,
                title=alert_request.title,
                description=alert_request.description,
                severity=alert_request.severity,
                status=AlertStatus.ACTIVE,
                context=context,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                fingerprint=fingerprint
            )
            
            # Analyse IA pour corrélation et insights
            ai_insights = await self._analyze_alert_with_ai(alert_data)
            alert_data.ai_insights = ai_insights
            
            # Détection de duplication et corrélation
            correlation_result = await self._correlate_alert(alert_data)
            if correlation_result["is_duplicate"]:
                return await self._handle_duplicate_alert(alert_data, correlation_result)
            
            # Persistance en base de données
            await self._store_alert(alert_data)
            
            # Détermination du canal de destination
            target_channels = await self.channel_router.get_target_channels(
                tenant_id=alert_request.tenant_id,
                severity=alert_request.severity,
                environment=alert_request.environment,
                tags=alert_request.tags
            )
            
            # Formatage du message Slack
            slack_message = await self.alert_formatter.format_alert(
                alert_data=alert_data,
                template_type="standard",
                language=self._get_tenant_language(alert_request.tenant_id)
            )
            
            # Envoi via webhook
            webhook_results = []
            for channel in target_channels:
                result = await self.webhook_handler.send_message(
                    channel=channel,
                    message=slack_message,
                    metadata={
                        "alert_id": alert_id,
                        "tenant_id": alert_request.tenant_id,
                        "severity": alert_request.severity.value
                    }
                )
                webhook_results.append(result)
            
            # Configuration de l'escalade si nécessaire
            if alert_request.escalation_policy and alert_request.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                await self.escalation_manager.setup_escalation(
                    alert_id=alert_id,
                    policy=alert_request.escalation_policy,
                    tenant_id=alert_request.tenant_id
                )
            
            # Mise à jour des métriques
            self.metrics["alerts_sent"] += 1
            
            # Log de l'audit
            await self._log_audit_event(
                event_type="alert_sent",
                alert_id=alert_id,
                tenant_id=alert_request.tenant_id,
                details={
                    "channels": target_channels,
                    "webhook_results": webhook_results
                }
            )
            
            return {
                "alert_id": alert_id,
                "status": "sent",
                "channels": target_channels,
                "webhook_results": webhook_results,
                "ai_insights": ai_insights,
                "correlation": correlation_result
            }
            
        except Exception as e:
            self.metrics["alerts_failed"] += 1
            self.logger.error(f"Erreur lors de l'envoi d'alerte: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

    async def acknowledge_alert(
        self,
        alert_id: str,
        tenant_id: str,
        user_id: str,
        note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Acquitte une alerte"""
        try:
            alert_data = await self._get_alert(alert_id, tenant_id)
            if not alert_data:
                raise HTTPException(status_code=404, detail="Alerte non trouvée")
            
            # Mise à jour du statut
            alert_data.status = AlertStatus.ACKNOWLEDGED
            alert_data.updated_at = datetime.utcnow()
            
            await self._update_alert(alert_data)
            
            # Annulation de l'escalade
            await self.escalation_manager.cancel_escalation(alert_id)
            
            # Notification Slack
            ack_message = await self.alert_formatter.format_acknowledgment(
                alert_data, user_id, note
            )
            
            channels = await self.channel_router.get_target_channels(
                tenant_id=tenant_id,
                severity=alert_data.severity,
                environment=alert_data.context.environment
            )
            
            for channel in channels:
                await self.webhook_handler.send_message(channel, ack_message)
            
            await self._log_audit_event(
                event_type="alert_acknowledged",
                alert_id=alert_id,
                tenant_id=tenant_id,
                details={"user_id": user_id, "note": note}
            )
            
            return {"status": "acknowledged", "alert_id": alert_id}
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'acquittement: {e}")
            raise

    async def resolve_alert(
        self,
        alert_id: str,
        tenant_id: str,
        resolution_note: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Résout une alerte"""
        try:
            alert_data = await self._get_alert(alert_id, tenant_id)
            if not alert_data:
                raise HTTPException(status_code=404, detail="Alerte non trouvée")
            
            # Mise à jour du statut
            alert_data.status = AlertStatus.RESOLVED
            alert_data.updated_at = datetime.utcnow()
            
            await self._update_alert(alert_data)
            
            # Annulation de l'escalade
            await self.escalation_manager.cancel_escalation(alert_id)
            
            # Notification de résolution
            resolution_message = await self.alert_formatter.format_resolution(
                alert_data, user_id, resolution_note
            )
            
            channels = await self.channel_router.get_target_channels(
                tenant_id=tenant_id,
                severity=alert_data.severity,
                environment=alert_data.context.environment
            )
            
            for channel in channels:
                await self.webhook_handler.send_message(channel, resolution_message)
            
            await self._log_audit_event(
                event_type="alert_resolved",
                alert_id=alert_id,
                tenant_id=tenant_id,
                details={"user_id": user_id, "resolution_note": resolution_note}
            )
            
            return {"status": "resolved", "alert_id": alert_id}
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la résolution: {e}")
            raise

    async def get_alert_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Récupère les métriques d'alertes pour un tenant"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Métriques générales
                metrics_query = """
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(*) FILTER (WHERE status = 'active') as active_alerts,
                    COUNT(*) FILTER (WHERE status = 'resolved') as resolved_alerts,
                    COUNT(*) FILTER (WHERE severity = 'critical') as critical_alerts,
                    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_resolution_time
                FROM alerts 
                WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '24 hours'
                """
                
                metrics = await conn.fetchrow(metrics_query, tenant_id)
                
                # Top alertes par service
                top_services_query = """
                SELECT service_name, COUNT(*) as alert_count
                FROM alerts 
                WHERE tenant_id = $1 AND created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY service_name
                ORDER BY alert_count DESC
                LIMIT 10
                """
                
                top_services = await conn.fetch(top_services_query, tenant_id)
                
                return {
                    "metrics": dict(metrics),
                    "top_services": [dict(row) for row in top_services],
                    "system_metrics": self.metrics
                }
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des métriques: {e}")
            raise

    def _generate_fingerprint(self, alert_request: SlackAlertRequest) -> str:
        """Génère une empreinte unique pour l'alerte"""
        import hashlib
        
        data = f"{alert_request.tenant_id}:{alert_request.alert_type}:{alert_request.service_name}:{alert_request.component}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def _analyze_alert_with_ai(self, alert_data: AlertData) -> Dict[str, Any]:
        """Analyse l'alerte avec IA pour générer des insights"""
        # Simulation d'analyse IA - À remplacer par un vrai modèle
        insights = {
            "predicted_impact": "medium",
            "similar_incidents": [],
            "recommended_actions": [
                "Vérifier les métriques CPU",
                "Examiner les logs d'application",
                "Vérifier la connectivité réseau"
            ],
            "correlation_score": 0.8,
            "trend_analysis": "increasing"
        }
        
        return insights

    async def _correlate_alert(self, alert_data: AlertData) -> Dict[str, Any]:
        """Corrèle l'alerte avec les alertes existantes"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Recherche d'alertes similaires
                similar_alerts_query = """
                SELECT alert_id, fingerprint, created_at
                FROM alerts 
                WHERE tenant_id = $1 
                AND fingerprint = $2 
                AND status = 'active'
                AND created_at >= NOW() - INTERVAL '1 hour'
                """
                
                similar_alerts = await conn.fetch(
                    similar_alerts_query,
                    alert_data.context.tenant_id,
                    alert_data.fingerprint
                )
                
                is_duplicate = len(similar_alerts) > 0
                
                return {
                    "is_duplicate": is_duplicate,
                    "similar_alerts": [dict(row) for row in similar_alerts],
                    "correlation_count": len(similar_alerts)
                }
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la corrélation: {e}")
            return {"is_duplicate": False, "similar_alerts": [], "correlation_count": 0}

    async def _handle_duplicate_alert(self, alert_data: AlertData, correlation_result: Dict) -> Dict[str, Any]:
        """Gère les alertes dupliquées"""
        # Incrémente le compteur de duplicatas
        await self.rate_limiter.increment_duplicate_count(
            alert_data.context.tenant_id,
            alert_data.fingerprint
        )
        
        return {
            "alert_id": alert_data.alert_id,
            "status": "deduplicated",
            "message": "Alerte dédupliquée",
            "original_alerts": correlation_result["similar_alerts"]
        }

    def _get_tenant_language(self, tenant_id: str) -> str:
        """Récupère la langue préférée du tenant"""
        # À implémenter selon la configuration tenant
        return "fr"

    async def _create_database_schema(self):
        """Crée le schéma de base de données"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255) NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            severity VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            context JSONB NOT NULL,
            fingerprint VARCHAR(255) NOT NULL,
            correlation_id VARCHAR(255),
            ai_insights JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_alerts_tenant_id ON alerts(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint ON alerts(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
        CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
        
        CREATE TABLE IF NOT EXISTS alert_audit (
            id SERIAL PRIMARY KEY,
            alert_id VARCHAR(255),
            tenant_id VARCHAR(255) NOT NULL,
            event_type VARCHAR(100) NOT NULL,
            details JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_alert_id ON alert_audit(alert_id);
        CREATE INDEX IF NOT EXISTS idx_audit_tenant_id ON alert_audit(tenant_id);
        """
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def _store_alert(self, alert_data: AlertData):
        """Stocke l'alerte en base de données"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO alerts (
                    alert_id, tenant_id, title, description, severity, status,
                    context, fingerprint, correlation_id, ai_insights,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                alert_data.alert_id,
                alert_data.context.tenant_id,
                alert_data.title,
                alert_data.description,
                alert_data.severity.value,
                alert_data.status.value,
                json.dumps(asdict(alert_data.context)),
                alert_data.fingerprint,
                alert_data.correlation_id,
                json.dumps(alert_data.ai_insights),
                alert_data.created_at,
                alert_data.updated_at
            )

    async def _get_alert(self, alert_id: str, tenant_id: str) -> Optional[AlertData]:
        """Récupère une alerte depuis la base de données"""
        async with self.postgres_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM alerts 
                WHERE alert_id = $1 AND tenant_id = $2
            """, alert_id, tenant_id)
            
            if row:
                context_data = json.loads(row['context'])
                context = AlertContext(**context_data)
                
                return AlertData(
                    alert_id=row['alert_id'],
                    title=row['title'],
                    description=row['description'],
                    severity=AlertSeverity(row['severity']),
                    status=AlertStatus(row['status']),
                    context=context,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    fingerprint=row['fingerprint'],
                    correlation_id=row['correlation_id'],
                    ai_insights=json.loads(row['ai_insights'] or '{}')
                )
            return None

    async def _update_alert(self, alert_data: AlertData):
        """Met à jour une alerte en base de données"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                UPDATE alerts SET
                    status = $1, updated_at = $2, ai_insights = $3
                WHERE alert_id = $4
            """,
                alert_data.status.value,
                alert_data.updated_at,
                json.dumps(alert_data.ai_insights),
                alert_data.alert_id
            )

    async def _log_audit_event(
        self,
        event_type: str,
        alert_id: str,
        tenant_id: str,
        details: Dict[str, Any]
    ):
        """Log un événement d'audit"""
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO alert_audit (alert_id, tenant_id, event_type, details)
                VALUES ($1, $2, $3, $4)
            """, alert_id, tenant_id, event_type, json.dumps(details))

    async def cleanup(self):
        """Nettoyage des ressources"""
        if self.redis_pool:
            await self.redis_pool.close()
        if self.postgres_pool:
            await self.postgres_pool.close()
