"""
🏢 Tenant Manager - Gestionnaire Central Multi-Tenant
====================================================

Gestionnaire principal pour la création, configuration et gestion
des tenants dans l'architecture multi-tenant.

Features:
- Création/suppression de tenants
- Configuration des quotas et limites
- Gestion du cycle de vie
- Auto-provisioning et scaling
- Health checks et monitoring
- Intégrations tierces

Author: Lead Dev + Architecte IA Team
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from fastapi import HTTPException, status
from pydantic import BaseModel, validator
import redis.asyncio as redis

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.core.config import settings
from app.tenancy.models import Tenant, TenantSubscription, TenantConfiguration
from app.tenancy.utils import TenantEventDispatcher, TenantHealthChecker
from app.tenancy.security import TenantSecurityValidator
from app.tenancy.analytics import TenantMetricsCollector

logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    """Statuts possibles d'un tenant"""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    DELETED = "deleted"


class TenantPlan(str, Enum):
    """Plans d'abonnement disponibles"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class TenantQuotas:
    """Quotas et limites d'un tenant"""
    max_users: int = 10
    max_storage_gb: int = 5
    max_api_calls_per_hour: int = 1000
    max_concurrent_sessions: int = 5
    max_integrations: int = 3
    ai_processing_hours_per_month: int = 10
    max_projects: int = 5
    max_collaborators: int = 10
    backup_retention_days: int = 7
    support_level: str = "community"
    features: List[str] = field(default_factory=list)


class TenantCreateRequest(BaseModel):
    """Requête de création de tenant"""
    name: str
    domain: str
    plan: TenantPlan = TenantPlan.FREE
    admin_email: str
    admin_name: str
    organization_type: Optional[str] = "individual"
    country: Optional[str] = "US"
    timezone: str = "UTC"
    language: str = "en"
    features: List[str] = []
    custom_quotas: Optional[Dict[str, Any]] = None

    @validator('domain')
    def validate_domain(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Domain must be at least 3 characters')
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Domain must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()

    @validator('admin_email')
    def validate_email(cls, v):
        # Simple email validation
        if '@' not in v or '.' not in v.split('@')[1]:
            raise ValueError('Invalid email format')
        return v.lower()


class TenantUpdateRequest(BaseModel):
    """Requête de mise à jour de tenant"""
    name: Optional[str] = None
    status: Optional[TenantStatus] = None
    plan: Optional[TenantPlan] = None
    quotas: Optional[Dict[str, Any]] = None
    features: Optional[List[str]] = None
    configuration: Optional[Dict[str, Any]] = None


class TenantManager:
    """
    Gestionnaire central pour la gestion des tenants.
    
    Responsabilités:
    - CRUD des tenants
    - Gestion des quotas et limites
    - Provisioning automatisé
    - Health monitoring
    - Scaling automatique
    """

    def __init__(self):
        self.event_dispatcher = TenantEventDispatcher()
        self.health_checker = TenantHealthChecker()
        self.security_validator = TenantSecurityValidator()
        self.metrics_collector = TenantMetricsCollector()
        self._redis_client: Optional[redis.Redis] = None

    async def get_redis_client(self) -> redis.Redis:
        """Obtenir le client Redis pour le cache"""
        if not self._redis_client:
            self._redis_client = await get_redis_client()
        return self._redis_client

    async def create_tenant(
        self,
        request: TenantCreateRequest,
        db: AsyncSession = None
    ) -> Tenant:
        """
        Créer un nouveau tenant avec provisioning automatisé.
        
        Args:
            request: Données de création du tenant
            db: Session de base de données
            
        Returns:
            Tenant créé avec configuration complète
            
        Raises:
            HTTPException: Si la création échoue
        """
        try:
            # Validation des données
            await self.security_validator.validate_tenant_creation(request)
            
            # Vérification de l'unicité du domaine
            await self._validate_domain_uniqueness(request.domain, db)
            
            # Génération des identifiants
            tenant_id = str(uuid.uuid4())
            
            # Configuration des quotas selon le plan
            quotas = self._get_plan_quotas(request.plan)
            if request.custom_quotas:
                quotas.__dict__.update(request.custom_quotas)
            
            # Création du tenant en base
            tenant_data = {
                "id": tenant_id,
                "name": request.name,
                "domain": request.domain,
                "status": TenantStatus.PENDING,
                "plan": request.plan,
                "admin_email": request.admin_email,
                "admin_name": request.admin_name,
                "organization_type": request.organization_type,
                "country": request.country,
                "timezone": request.timezone,
                "language": request.language,
                "quotas": quotas.__dict__,
                "features": request.features,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            if not db:
                async with get_async_session() as db:
                    tenant = await self._create_tenant_in_db(tenant_data, db)
            else:
                tenant = await self._create_tenant_in_db(tenant_data, db)
            
            # Provisioning automatisé
            await self._provision_tenant(tenant)
            
            # Mise à jour du statut à ACTIVE
            tenant.status = TenantStatus.ACTIVE
            tenant.updated_at = datetime.utcnow()
            
            if not db:
                async with get_async_session() as db:
                    await db.commit()
            else:
                await db.commit()
            
            # Cache du tenant
            await self._cache_tenant(tenant)
            
            # Événement de création
            await self.event_dispatcher.dispatch("tenant.created", {
                "tenant_id": tenant.id,
                "domain": tenant.domain,
                "plan": tenant.plan,
                "admin_email": tenant.admin_email
            })
            
            # Démarrage du monitoring
            await self.health_checker.start_monitoring(tenant.id)
            
            logger.info(f"Tenant créé avec succès: {tenant.id} ({tenant.domain})")
            return tenant
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du tenant: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create tenant: {str(e)}"
            )

    async def get_tenant(
        self,
        tenant_id: str,
        use_cache: bool = True
    ) -> Optional[Tenant]:
        """
        Récupérer un tenant par son ID.
        
        Args:
            tenant_id: Identifiant du tenant
            use_cache: Utiliser le cache Redis
            
        Returns:
            Tenant trouvé ou None
        """
        try:
            # Tentative de récupération depuis le cache
            if use_cache:
                cached_tenant = await self._get_cached_tenant(tenant_id)
                if cached_tenant:
                    return cached_tenant
            
            # Récupération depuis la base de données
            async with get_async_session() as db:
                result = await db.execute(
                    select(Tenant).where(Tenant.id == tenant_id)
                )
                tenant = result.scalar_one_or_none()
                
                if tenant and use_cache:
                    await self._cache_tenant(tenant)
                
                return tenant
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du tenant {tenant_id}: {str(e)}")
            return None

    async def get_tenant_by_domain(
        self,
        domain: str,
        use_cache: bool = True
    ) -> Optional[Tenant]:
        """
        Récupérer un tenant par son domaine.
        
        Args:
            domain: Domaine du tenant
            use_cache: Utiliser le cache Redis
            
        Returns:
            Tenant trouvé ou None
        """
        try:
            # Cache key par domaine
            cache_key = f"tenant:domain:{domain.lower()}"
            
            if use_cache:
                redis_client = await self.get_redis_client()
                cached_id = await redis_client.get(cache_key)
                if cached_id:
                    return await self.get_tenant(cached_id.decode(), use_cache=True)
            
            # Récupération depuis la base
            async with get_async_session() as db:
                result = await db.execute(
                    select(Tenant).where(Tenant.domain == domain.lower())
                )
                tenant = result.scalar_one_or_none()
                
                if tenant and use_cache:
                    await self._cache_tenant(tenant)
                    # Cache du mapping domaine -> ID
                    redis_client = await self.get_redis_client()
                    await redis_client.setex(cache_key, 3600, tenant.id)
                
                return tenant
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du tenant par domaine {domain}: {str(e)}")
            return None

    async def update_tenant(
        self,
        tenant_id: str,
        request: TenantUpdateRequest
    ) -> Optional[Tenant]:
        """
        Mettre à jour un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            request: Données de mise à jour
            
        Returns:
            Tenant mis à jour ou None
        """
        try:
            async with get_async_session() as db:
                # Récupération du tenant existant
                result = await db.execute(
                    select(Tenant).where(Tenant.id == tenant_id)
                )
                tenant = result.scalar_one_or_none()
                
                if not tenant:
                    return None
                
                # Application des modifications
                update_data = {}
                if request.name is not None:
                    update_data["name"] = request.name
                if request.status is not None:
                    update_data["status"] = request.status
                if request.plan is not None:
                    update_data["plan"] = request.plan
                    # Mise à jour des quotas selon le nouveau plan
                    new_quotas = self._get_plan_quotas(request.plan)
                    update_data["quotas"] = new_quotas.__dict__
                if request.quotas is not None:
                    current_quotas = tenant.quotas or {}
                    current_quotas.update(request.quotas)
                    update_data["quotas"] = current_quotas
                if request.features is not None:
                    update_data["features"] = request.features
                if request.configuration is not None:
                    current_config = tenant.configuration or {}
                    current_config.update(request.configuration)
                    update_data["configuration"] = current_config
                
                update_data["updated_at"] = datetime.utcnow()
                
                # Mise à jour en base
                await db.execute(
                    update(Tenant)
                    .where(Tenant.id == tenant_id)
                    .values(**update_data)
                )
                await db.commit()
                
                # Récupération du tenant mis à jour
                result = await db.execute(
                    select(Tenant).where(Tenant.id == tenant_id)
                )
                updated_tenant = result.scalar_one_or_none()
                
                # Mise à jour du cache
                await self._cache_tenant(updated_tenant)
                
                # Événement de mise à jour
                await self.event_dispatcher.dispatch("tenant.updated", {
                    "tenant_id": tenant_id,
                    "changes": update_data
                })
                
                logger.info(f"Tenant mis à jour: {tenant_id}")
                return updated_tenant
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du tenant {tenant_id}: {str(e)}")
            return None

    async def delete_tenant(
        self,
        tenant_id: str,
        soft_delete: bool = True
    ) -> bool:
        """
        Supprimer un tenant (suppression douce par défaut).
        
        Args:
            tenant_id: Identifiant du tenant
            soft_delete: Suppression douce (status = deleted) ou définitive
            
        Returns:
            True si succès, False sinon
        """
        try:
            async with get_async_session() as db:
                if soft_delete:
                    # Suppression douce - changement de statut
                    await db.execute(
                        update(Tenant)
                        .where(Tenant.id == tenant_id)
                        .values(
                            status=TenantStatus.DELETED,
                            deleted_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                    )
                else:
                    # Suppression définitive
                    await db.execute(
                        delete(Tenant).where(Tenant.id == tenant_id)
                    )
                
                await db.commit()
                
                # Suppression du cache
                await self._remove_cached_tenant(tenant_id)
                
                # Arrêt du monitoring
                await self.health_checker.stop_monitoring(tenant_id)
                
                # Événement de suppression
                await self.event_dispatcher.dispatch("tenant.deleted", {
                    "tenant_id": tenant_id,
                    "soft_delete": soft_delete
                })
                
                logger.info(f"Tenant supprimé: {tenant_id} (soft_delete={soft_delete})")
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du tenant {tenant_id}: {str(e)}")
            return False

    async def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        plan: Optional[TenantPlan] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tenant]:
        """
        Lister les tenants avec filtres.
        
        Args:
            status: Filtrer par statut
            plan: Filtrer par plan
            limit: Nombre maximum de résultats
            offset: Décalage pour la pagination
            
        Returns:
            Liste des tenants
        """
        try:
            async with get_async_session() as db:
                query = select(Tenant)
                
                if status:
                    query = query.where(Tenant.status == status)
                if plan:
                    query = query.where(Tenant.plan == plan)
                
                query = query.offset(offset).limit(limit)
                query = query.order_by(Tenant.created_at.desc())
                
                result = await db.execute(query)
                tenants = result.scalars().all()
                
                return list(tenants)
                
        except Exception as e:
            logger.error(f"Erreur lors de la liste des tenants: {str(e)}")
            return []

    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """
        Récupérer les métriques d'un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            
        Returns:
            Dictionnaire des métriques
        """
        return await self.metrics_collector.get_tenant_metrics(tenant_id)

    async def check_tenant_health(self, tenant_id: str) -> Dict[str, Any]:
        """
        Vérifier la santé d'un tenant.
        
        Args:
            tenant_id: Identifiant du tenant
            
        Returns:
            Rapport de santé
        """
        return await self.health_checker.check_health(tenant_id)

    # Méthodes privées

    async def _validate_domain_uniqueness(
        self,
        domain: str,
        db: AsyncSession = None
    ) -> None:
        """Valider l'unicité du domaine"""
        if not db:
            async with get_async_session() as db:
                result = await db.execute(
                    select(Tenant.id).where(Tenant.domain == domain.lower())
                )
        else:
            result = await db.execute(
                select(Tenant.id).where(Tenant.domain == domain.lower())
            )
        
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Domain '{domain}' is already taken"
            )

    def _get_plan_quotas(self, plan: TenantPlan) -> TenantQuotas:
        """Obtenir les quotas selon le plan"""
        quotas_map = {
            TenantPlan.FREE: TenantQuotas(
                max_users=5,
                max_storage_gb=1,
                max_api_calls_per_hour=100,
                max_concurrent_sessions=2,
                max_integrations=1,
                ai_processing_hours_per_month=2,
                max_projects=1,
                max_collaborators=2,
                backup_retention_days=3,
                support_level="community",
                features=["basic_ai", "basic_analytics"]
            ),
            TenantPlan.STARTER: TenantQuotas(
                max_users=25,
                max_storage_gb=10,
                max_api_calls_per_hour=1000,
                max_concurrent_sessions=10,
                max_integrations=5,
                ai_processing_hours_per_month=20,
                max_projects=5,
                max_collaborators=10,
                backup_retention_days=7,
                support_level="email",
                features=["advanced_ai", "analytics", "collaboration"]
            ),
            TenantPlan.PROFESSIONAL: TenantQuotas(
                max_users=100,
                max_storage_gb=100,
                max_api_calls_per_hour=10000,
                max_concurrent_sessions=50,
                max_integrations=20,
                ai_processing_hours_per_month=100,
                max_projects=25,
                max_collaborators=50,
                backup_retention_days=30,
                support_level="priority",
                features=["premium_ai", "advanced_analytics", "white_label", "api_access"]
            ),
            TenantPlan.ENTERPRISE: TenantQuotas(
                max_users=1000,
                max_storage_gb=1000,
                max_api_calls_per_hour=100000,
                max_concurrent_sessions=500,
                max_integrations=100,
                ai_processing_hours_per_month=1000,
                max_projects=100,
                max_collaborators=500,
                backup_retention_days=90,
                support_level="dedicated",
                features=["enterprise_ai", "custom_analytics", "sso", "audit_logs", "compliance"]
            )
        }
        
        return quotas_map.get(plan, TenantQuotas())

    async def _create_tenant_in_db(
        self,
        tenant_data: Dict[str, Any],
        db: AsyncSession
    ) -> Tenant:
        """Créer le tenant en base de données"""
        tenant = Tenant(**tenant_data)
        db.add(tenant)
        await db.commit()
        await db.refresh(tenant)
        return tenant

    async def _provision_tenant(self, tenant: Tenant) -> None:
        """Provisioning automatisé du tenant"""
        try:
            # Création de l'infrastructure (DB schemas, caches, etc.)
            # Cette méthode sera implémentée par TenantProvisioningManager
            logger.info(f"Démarrage du provisioning pour le tenant {tenant.id}")
            
            # Simulation du provisioning
            await asyncio.sleep(0.1)
            
            logger.info(f"Provisioning terminé pour le tenant {tenant.id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du provisioning du tenant {tenant.id}: {str(e)}")
            raise

    async def _cache_tenant(self, tenant: Tenant) -> None:
        """Mettre en cache les données du tenant"""
        try:
            redis_client = await self.get_redis_client()
            
            # Cache principal par ID
            cache_key = f"tenant:{tenant.id}"
            tenant_data = {
                "id": tenant.id,
                "name": tenant.name,
                "domain": tenant.domain,
                "status": tenant.status,
                "plan": tenant.plan,
                "quotas": tenant.quotas,
                "features": tenant.features,
                "configuration": tenant.configuration
            }
            
            await redis_client.hset(cache_key, mapping=tenant_data)
            await redis_client.expire(cache_key, 3600)  # 1 heure
            
            # Cache par domaine
            domain_key = f"tenant:domain:{tenant.domain}"
            await redis_client.setex(domain_key, 3600, tenant.id)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise en cache du tenant {tenant.id}: {str(e)}")

    async def _get_cached_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Récupérer un tenant depuis le cache"""
        try:
            redis_client = await self.get_redis_client()
            cache_key = f"tenant:{tenant_id}"
            
            cached_data = await redis_client.hgetall(cache_key)
            if not cached_data:
                return None
            
            # Reconstruction de l'objet Tenant
            tenant_data = {k.decode(): v.decode() for k, v in cached_data.items()}
            
            # Conversion des types
            if 'quotas' in tenant_data and tenant_data['quotas']:
                import json
                tenant_data['quotas'] = json.loads(tenant_data['quotas'])
            if 'features' in tenant_data and tenant_data['features']:
                import json
                tenant_data['features'] = json.loads(tenant_data['features'])
            if 'configuration' in tenant_data and tenant_data['configuration']:
                import json
                tenant_data['configuration'] = json.loads(tenant_data['configuration'])
            
            return Tenant(**tenant_data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du cache pour le tenant {tenant_id}: {str(e)}")
            return None

    async def _remove_cached_tenant(self, tenant_id: str) -> None:
        """Supprimer un tenant du cache"""
        try:
            redis_client = await self.get_redis_client()
            
            # Récupération du domaine pour suppression du cache domain
            tenant = await self.get_tenant(tenant_id, use_cache=False)
            
            # Suppression des caches
            cache_keys = [f"tenant:{tenant_id}"]
            if tenant:
                cache_keys.append(f"tenant:domain:{tenant.domain}")
            
            await redis_client.delete(*cache_keys)
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du cache pour le tenant {tenant_id}: {str(e)}")


# Instance globale du gestionnaire
tenant_manager = TenantManager()
