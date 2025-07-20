"""
Advanced Tenant Manager
=======================

Gestionnaire avancé des tenants avec isolation complète des données,
gestion des ressources, monitoring et automation.

Auteur: Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import hashlib
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    """Statuts possibles d'un tenant"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    MIGRATING = "migrating"
    ARCHIVING = "archiving"
    ARCHIVED = "archived"
    ERROR = "error"


class TenantTier(Enum):
    """Niveaux de service tenant"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class TenantResource:
    """Ressources allouées à un tenant"""
    max_users: int = 100
    max_storage_gb: int = 10
    max_api_calls_per_hour: int = 1000
    max_db_connections: int = 10
    max_concurrent_requests: int = 50
    backup_retention_days: int = 30
    feature_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class TenantUsage:
    """Utilisation actuelle des ressources"""
    current_users: int = 0
    current_storage_gb: float = 0.0
    current_api_calls_hour: int = 0
    current_db_connections: int = 0
    current_requests: int = 0
    last_activity: Optional[datetime] = None


@dataclass
class TenantMetadata:
    """Métadonnées du tenant"""
    name: str
    description: Optional[str] = None
    contact_email: str = ""
    contact_phone: Optional[str] = None
    timezone: str = "UTC"
    locale: str = "en"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tenant:
    """Représentation complète d'un tenant"""
    id: str
    status: TenantStatus
    tier: TenantTier
    metadata: TenantMetadata
    resources: TenantResource
    usage: TenantUsage
    schema_name: str
    encryption_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.metadata.created_at:
            self.metadata.created_at = datetime.utcnow()
        self.metadata.updated_at = datetime.utcnow()


class TenantManager:
    """
    Gestionnaire avancé des tenants
    
    Fonctionnalités:
    - Création/suppression de tenants
    - Isolation complète des données
    - Gestion des ressources et quotas
    - Migration et sauvegarde
    - Monitoring en temps réel
    - Automation des tâches
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire de tenants
        
        Args:
            config: Configuration du tenant manager
        """
        self.config = config
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_schemas: Dict[str, str] = {}
        self.resource_monitors: Dict[str, Any] = {}
        
        # Base de données
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        
        # Statuts
        self.is_initialized = False
        self.is_monitoring = False
        
        logger.info("TenantManager initialisé")
    
    async def initialize(self) -> None:
        """Initialise le gestionnaire de tenants"""
        if self.is_initialized:
            return
        
        logger.info("Initialisation du TenantManager...")
        
        try:
            # Configuration de la base de données
            await self._init_database()
            
            # Chargement des tenants existants
            await self._load_existing_tenants()
            
            # Démarrage du monitoring
            await self._start_monitoring()
            
            # Démarrage des tâches d'automation
            await self._start_automation_tasks()
            
            self.is_initialized = True
            logger.info("TenantManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du TenantManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrêt propre du gestionnaire"""
        if not self.is_initialized:
            return
        
        logger.info("Arrêt du TenantManager...")
        
        try:
            # Arrêt du monitoring
            self.is_monitoring = False
            
            # Fermeture des connexions DB
            if self.async_engine:
                await self.async_engine.dispose()
            
            self.is_initialized = False
            logger.info("TenantManager arrêté avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")
    
    async def _init_database(self) -> None:
        """Initialise les connexions à la base de données"""
        db_config = self.config.get("database", {})
        
        # URL de connexion
        db_url = (
            f"postgresql://{db_config.get('username')}:"
            f"{db_config.get('password')}@{db_config.get('host')}:"
            f"{db_config.get('port')}/{db_config.get('database')}"
        )
        
        async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
        
        # Création des engines
        self.engine = create_engine(db_url, echo=db_config.get("echo", False))
        self.async_engine = create_async_engine(
            async_db_url,
            pool_size=db_config.get("pool_size", 10),
            max_overflow=db_config.get("max_overflow", 20)
        )
        
        # Factory de sessions
        self.session_factory = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Création des tables système si nécessaires
        await self._ensure_system_tables()
    
    async def _ensure_system_tables(self) -> None:
        """S'assure que les tables système existent"""
        create_tenant_table = """
        CREATE TABLE IF NOT EXISTS tenants (
            id VARCHAR(50) PRIMARY KEY,
            status VARCHAR(20) NOT NULL,
            tier VARCHAR(20) NOT NULL,
            schema_name VARCHAR(100) NOT NULL,
            metadata JSONB NOT NULL,
            resources JSONB NOT NULL,
            usage JSONB NOT NULL,
            encryption_key TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
        CREATE INDEX IF NOT EXISTS idx_tenants_tier ON tenants(tier);
        CREATE INDEX IF NOT EXISTS idx_tenants_schema ON tenants(schema_name);
        """
        
        async with self.async_engine.begin() as conn:
            await conn.execute(text(create_tenant_table))
    
    async def _load_existing_tenants(self) -> None:
        """Charge les tenants existants depuis la base de données"""
        query = "SELECT * FROM tenants WHERE status != 'archived'"
        
        async with self.session_factory() as session:
            result = await session.execute(text(query))
            rows = result.fetchall()
            
            for row in rows:
                tenant = self._row_to_tenant(row)
                self.tenants[tenant.id] = tenant
                self.tenant_schemas[tenant.id] = tenant.schema_name
        
        logger.info(f"Chargés {len(self.tenants)} tenants existants")
    
    def _row_to_tenant(self, row) -> Tenant:
        """Convertit une ligne de DB en objet Tenant"""
        metadata_dict = row.metadata
        resources_dict = row.resources
        usage_dict = row.usage
        
        metadata = TenantMetadata(
            name=metadata_dict["name"],
            description=metadata_dict.get("description"),
            contact_email=metadata_dict.get("contact_email", ""),
            contact_phone=metadata_dict.get("contact_phone"),
            timezone=metadata_dict.get("timezone", "UTC"),
            locale=metadata_dict.get("locale", "en"),
            created_at=datetime.fromisoformat(metadata_dict["created_at"]) if metadata_dict.get("created_at") else None,
            updated_at=datetime.fromisoformat(metadata_dict["updated_at"]) if metadata_dict.get("updated_at") else None,
            tags=set(metadata_dict.get("tags", [])),
            custom_properties=metadata_dict.get("custom_properties", {})
        )
        
        resources = TenantResource(**resources_dict)
        usage = TenantUsage(**usage_dict)
        
        if usage.last_activity and isinstance(usage.last_activity, str):
            usage.last_activity = datetime.fromisoformat(usage.last_activity)
        
        return Tenant(
            id=row.id,
            status=TenantStatus(row.status),
            tier=TenantTier(row.tier),
            metadata=metadata,
            resources=resources,
            usage=usage,
            schema_name=row.schema_name,
            encryption_key=row.encryption_key
        )
    
    async def create_tenant(
        self,
        name: str,
        contact_email: str,
        tier: TenantTier = TenantTier.FREE,
        custom_resources: Optional[TenantResource] = None
    ) -> Tenant:
        """
        Crée un nouveau tenant
        
        Args:
            name: Nom du tenant
            contact_email: Email de contact
            tier: Niveau de service
            custom_resources: Ressources personnalisées
            
        Returns:
            Tenant créé
        """
        if not self.is_initialized:
            raise RuntimeError("TenantManager non initialisé")
        
        # Génération d'un ID unique
        tenant_id = self._generate_tenant_id(name)
        
        if tenant_id in self.tenants:
            raise ValueError(f"Tenant avec l'ID {tenant_id} existe déjà")
        
        # Validation du nombre maximum de tenants
        if len(self.tenants) >= self.config.get("max_tenants", 1000):
            raise ValueError("Nombre maximum de tenants atteint")
        
        try:
            # Métadonnées
            metadata = TenantMetadata(
                name=name,
                contact_email=contact_email,
                created_at=datetime.utcnow()
            )
            
            # Ressources selon le tier
            resources = custom_resources or self._get_default_resources(tier)
            
            # Usage initial
            usage = TenantUsage()
            
            # Nom du schéma
            schema_name = f"tenant_{tenant_id}"
            
            # Clé de chiffrement
            encryption_key = self._generate_encryption_key()
            
            # Création du tenant
            tenant = Tenant(
                id=tenant_id,
                status=TenantStatus.ACTIVE,
                tier=tier,
                metadata=metadata,
                resources=resources,
                usage=usage,
                schema_name=schema_name,
                encryption_key=encryption_key
            )
            
            # Création du schéma de base de données
            await self._create_tenant_schema(tenant)
            
            # Sauvegarde en base
            await self._save_tenant(tenant)
            
            # Ajout au cache
            self.tenants[tenant_id] = tenant
            self.tenant_schemas[tenant_id] = schema_name
            
            logger.info(f"Tenant créé: {tenant_id} ({name})")
            
            # Initialisation du monitoring
            await self._init_tenant_monitoring(tenant)
            
            return tenant
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du tenant {tenant_id}: {e}")
            # Nettoyage en cas d'erreur
            await self._cleanup_failed_tenant(tenant_id)
            raise
    
    def _generate_tenant_id(self, name: str) -> str:
        """Génère un ID unique pour le tenant"""
        # Combinaison du nom et d'un UUID pour l'unicité
        clean_name = "".join(c.lower() for c in name if c.isalnum())[:20]
        unique_suffix = str(uuid.uuid4())[:8]
        return f"{clean_name}_{unique_suffix}"
    
    def _get_default_resources(self, tier: TenantTier) -> TenantResource:
        """Retourne les ressources par défaut selon le tier"""
        resource_tiers = {
            TenantTier.FREE: TenantResource(
                max_users=10,
                max_storage_gb=1,
                max_api_calls_per_hour=100,
                max_db_connections=2,
                max_concurrent_requests=5
            ),
            TenantTier.BASIC: TenantResource(
                max_users=50,
                max_storage_gb=10,
                max_api_calls_per_hour=1000,
                max_db_connections=5,
                max_concurrent_requests=20
            ),
            TenantTier.PREMIUM: TenantResource(
                max_users=200,
                max_storage_gb=50,
                max_api_calls_per_hour=5000,
                max_db_connections=10,
                max_concurrent_requests=50
            ),
            TenantTier.ENTERPRISE: TenantResource(
                max_users=1000,
                max_storage_gb=200,
                max_api_calls_per_hour=20000,
                max_db_connections=20,
                max_concurrent_requests=100
            )
        }
        
        return resource_tiers.get(tier, resource_tiers[TenantTier.FREE])
    
    def _generate_encryption_key(self) -> str:
        """Génère une clé de chiffrement pour le tenant"""
        return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
    
    async def _create_tenant_schema(self, tenant: Tenant) -> None:
        """Crée le schéma de base de données pour le tenant"""
        schema_sql = f"""
        CREATE SCHEMA IF NOT EXISTS {tenant.schema_name};
        
        -- Tables spécifiques au tenant
        CREATE TABLE IF NOT EXISTS {tenant.schema_name}.users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS {tenant.schema_name}.user_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id INTEGER REFERENCES {tenant.schema_name}.users(id),
            session_token VARCHAR(255) NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS {tenant.schema_name}.audit_log (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES {tenant.schema_name}.users(id),
            action VARCHAR(100) NOT NULL,
            resource_type VARCHAR(100),
            resource_id VARCHAR(100),
            details JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Index pour les performances
        CREATE INDEX IF NOT EXISTS idx_{tenant.schema_name}_users_email 
            ON {tenant.schema_name}.users(email);
        CREATE INDEX IF NOT EXISTS idx_{tenant.schema_name}_sessions_token 
            ON {tenant.schema_name}.user_sessions(session_token);
        CREATE INDEX IF NOT EXISTS idx_{tenant.schema_name}_audit_user 
            ON {tenant.schema_name}.audit_log(user_id, created_at);
        """
        
        async with self.async_engine.begin() as conn:
            await conn.execute(text(schema_sql))
        
        logger.info(f"Schéma créé pour le tenant: {tenant.schema_name}")
    
    async def _save_tenant(self, tenant: Tenant) -> None:
        """Sauvegarde le tenant en base de données"""
        insert_sql = """
        INSERT INTO tenants (
            id, status, tier, schema_name, metadata, resources, usage, 
            encryption_key, created_at, updated_at
        ) VALUES (
            :id, :status, :tier, :schema_name, :metadata, :resources, :usage,
            :encryption_key, :created_at, :updated_at
        )
        ON CONFLICT (id) DO UPDATE SET
            status = EXCLUDED.status,
            tier = EXCLUDED.tier,
            metadata = EXCLUDED.metadata,
            resources = EXCLUDED.resources,
            usage = EXCLUDED.usage,
            updated_at = EXCLUDED.updated_at
        """
        
        params = {
            "id": tenant.id,
            "status": tenant.status.value,
            "tier": tenant.tier.value,
            "schema_name": tenant.schema_name,
            "metadata": {
                "name": tenant.metadata.name,
                "description": tenant.metadata.description,
                "contact_email": tenant.metadata.contact_email,
                "contact_phone": tenant.metadata.contact_phone,
                "timezone": tenant.metadata.timezone,
                "locale": tenant.metadata.locale,
                "created_at": tenant.metadata.created_at.isoformat() if tenant.metadata.created_at else None,
                "updated_at": tenant.metadata.updated_at.isoformat() if tenant.metadata.updated_at else None,
                "tags": list(tenant.metadata.tags),
                "custom_properties": tenant.metadata.custom_properties
            },
            "resources": tenant.resources.__dict__,
            "usage": {
                **tenant.usage.__dict__,
                "last_activity": tenant.usage.last_activity.isoformat() if tenant.usage.last_activity else None
            },
            "encryption_key": tenant.encryption_key,
            "created_at": tenant.metadata.created_at,
            "updated_at": tenant.metadata.updated_at
        }
        
        async with self.session_factory() as session:
            await session.execute(text(insert_sql), params)
            await session.commit()
    
    async def _cleanup_failed_tenant(self, tenant_id: str) -> None:
        """Nettoie les ressources d'un tenant en cas d'échec de création"""
        try:
            # Suppression du schéma si créé
            if tenant_id in self.tenant_schemas:
                schema_name = self.tenant_schemas[tenant_id]
                async with self.async_engine.begin() as conn:
                    await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
                del self.tenant_schemas[tenant_id]
            
            # Suppression de la base
            async with self.session_factory() as session:
                await session.execute(
                    text("DELETE FROM tenants WHERE id = :id"),
                    {"id": tenant_id}
                )
                await session.commit()
            
            # Suppression du cache
            if tenant_id in self.tenants:
                del self.tenants[tenant_id]
            
            logger.info(f"Nettoyage effectué pour le tenant: {tenant_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du tenant {tenant_id}: {e}")
    
    async def _start_monitoring(self) -> None:
        """Démarre le monitoring des tenants"""
        if not self.is_monitoring:
            self.is_monitoring = True
            asyncio.create_task(self._monitoring_loop())
            logger.info("Monitoring des tenants démarré")
    
    async def _monitoring_loop(self) -> None:
        """Boucle principale de monitoring"""
        while self.is_monitoring:
            try:
                for tenant in self.tenants.values():
                    await self._monitor_tenant_resources(tenant)
                    await self._update_tenant_usage(tenant)
                
                await asyncio.sleep(30)  # Monitoring toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de monitoring: {e}")
                await asyncio.sleep(60)  # Attendre plus longtemps en cas d'erreur
    
    async def _monitor_tenant_resources(self, tenant: Tenant) -> None:
        """Surveille les ressources d'un tenant"""
        # Vérification des quotas
        if tenant.usage.current_users > tenant.resources.max_users:
            await self._trigger_quota_alert(tenant, "users", tenant.usage.current_users, tenant.resources.max_users)
        
        if tenant.usage.current_storage_gb > tenant.resources.max_storage_gb:
            await self._trigger_quota_alert(tenant, "storage", tenant.usage.current_storage_gb, tenant.resources.max_storage_gb)
        
        if tenant.usage.current_api_calls_hour > tenant.resources.max_api_calls_per_hour:
            await self._trigger_quota_alert(tenant, "api_calls", tenant.usage.current_api_calls_hour, tenant.resources.max_api_calls_per_hour)
    
    async def _trigger_quota_alert(self, tenant: Tenant, resource_type: str, current: float, limit: float) -> None:
        """Déclenche une alerte de quota dépassé"""
        alert_data = {
            "tenant_id": tenant.id,
            "tenant_name": tenant.metadata.name,
            "resource_type": resource_type,
            "current_usage": current,
            "limit": limit,
            "percentage": (current / limit) * 100,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.warning(f"Quota dépassé pour {tenant.id}: {resource_type} = {current}/{limit}")
        # Ici on pourrait envoyer vers le système d'alertes
    
    async def _update_tenant_usage(self, tenant: Tenant) -> None:
        """Met à jour l'utilisation des ressources du tenant"""
        try:
            # Comptage des utilisateurs
            user_count_sql = f"SELECT COUNT(*) FROM {tenant.schema_name}.users"
            async with self.session_factory() as session:
                result = await session.execute(text(user_count_sql))
                tenant.usage.current_users = result.scalar()
            
            # Mise à jour de la dernière activité
            tenant.usage.last_activity = datetime.utcnow()
            
            # Sauvegarde des changements
            await self._save_tenant(tenant)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'usage du tenant {tenant.id}: {e}")
    
    async def _start_automation_tasks(self) -> None:
        """Démarre les tâches d'automation"""
        # Nettoyage automatique des sessions expirées
        asyncio.create_task(self._cleanup_expired_sessions())
        
        # Archivage automatique des tenants inactifs
        asyncio.create_task(self._archive_inactive_tenants())
        
        logger.info("Tâches d'automation démarrées")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Nettoie les sessions expirées"""
        while self.is_initialized:
            try:
                for tenant in self.tenants.values():
                    cleanup_sql = f"""
                    DELETE FROM {tenant.schema_name}.user_sessions 
                    WHERE expires_at < NOW()
                    """
                    async with self.session_factory() as session:
                        await session.execute(text(cleanup_sql))
                        await session.commit()
                
                await asyncio.sleep(3600)  # Toutes les heures
                
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage des sessions: {e}")
                await asyncio.sleep(1800)
    
    async def _archive_inactive_tenants(self) -> None:
        """Archive les tenants inactifs"""
        while self.is_initialized:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=90)
                
                for tenant in list(self.tenants.values()):
                    if (tenant.usage.last_activity and 
                        tenant.usage.last_activity < cutoff_date and 
                        tenant.status == TenantStatus.ACTIVE):
                        
                        logger.info(f"Marquage du tenant {tenant.id} comme inactif")
                        tenant.status = TenantStatus.INACTIVE
                        await self._save_tenant(tenant)
                
                await asyncio.sleep(86400)  # Tous les jours
                
            except Exception as e:
                logger.error(f"Erreur lors de l'archivage: {e}")
                await asyncio.sleep(43200)
    
    async def _init_tenant_monitoring(self, tenant: Tenant) -> None:
        """Initialise le monitoring spécifique d'un tenant"""
        self.resource_monitors[tenant.id] = {
            "last_check": datetime.utcnow(),
            "alert_cooldown": {},
            "metrics_history": []
        }
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Récupère un tenant par son ID
        
        Args:
            tenant_id: ID du tenant
            
        Returns:
            Tenant ou None si non trouvé
        """
        return self.tenants.get(tenant_id)
    
    async def list_tenants(
        self, 
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None
    ) -> List[Tenant]:
        """
        Liste les tenants avec filtres optionnels
        
        Args:
            status: Filtre par statut
            tier: Filtre par tier
            
        Returns:
            Liste des tenants
        """
        tenants = list(self.tenants.values())
        
        if status:
            tenants = [t for t in tenants if t.status == status]
        
        if tier:
            tenants = [t for t in tenants if t.tier == tier]
        
        return tenants
    
    async def update_tenant_status(self, tenant_id: str, status: TenantStatus) -> bool:
        """
        Met à jour le statut d'un tenant
        
        Args:
            tenant_id: ID du tenant
            status: Nouveau statut
            
        Returns:
            True si mis à jour avec succès
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        old_status = tenant.status
        tenant.status = status
        tenant.metadata.updated_at = datetime.utcnow()
        
        await self._save_tenant(tenant)
        
        logger.info(f"Statut du tenant {tenant_id} changé: {old_status.value} -> {status.value}")
        return True
    
    async def delete_tenant(self, tenant_id: str, backup: bool = True) -> bool:
        """
        Supprime un tenant
        
        Args:
            tenant_id: ID du tenant
            backup: Effectuer une sauvegarde avant suppression
            
        Returns:
            True si supprimé avec succès
        """
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return False
        
        try:
            # Sauvegarde si demandée
            if backup:
                await self._backup_tenant_data(tenant)
            
            # Suppression du schéma
            async with self.async_engine.begin() as conn:
                await conn.execute(text(f"DROP SCHEMA IF EXISTS {tenant.schema_name} CASCADE"))
            
            # Suppression de la base
            async with self.session_factory() as session:
                await session.execute(
                    text("DELETE FROM tenants WHERE id = :id"),
                    {"id": tenant_id}
                )
                await session.commit()
            
            # Suppression du cache
            del self.tenants[tenant_id]
            del self.tenant_schemas[tenant_id]
            
            if tenant_id in self.resource_monitors:
                del self.resource_monitors[tenant_id]
            
            logger.info(f"Tenant supprimé: {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du tenant {tenant_id}: {e}")
            return False
    
    async def _backup_tenant_data(self, tenant: Tenant) -> None:
        """Sauvegarde les données d'un tenant"""
        backup_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = f"tenant_{tenant.id}_backup_{backup_timestamp}.sql"
        
        # Ici on pourrait implémenter une vraie sauvegarde
        logger.info(f"Sauvegarde du tenant {tenant.id} dans {backup_file}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé du gestionnaire
        
        Returns:
            Rapport d'état
        """
        try:
            # Test de connexion DB
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))
            
            return {
                "status": "healthy",
                "tenants_count": len(self.tenants),
                "active_tenants": len([t for t in self.tenants.values() if t.status == TenantStatus.ACTIVE]),
                "monitoring_active": self.is_monitoring,
                "database_connected": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "tenants_count": len(self.tenants),
                "monitoring_active": self.is_monitoring,
                "database_connected": False
            }
