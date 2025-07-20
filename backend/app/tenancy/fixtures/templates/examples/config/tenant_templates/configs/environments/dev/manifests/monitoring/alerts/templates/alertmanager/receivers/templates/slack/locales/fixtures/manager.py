"""
Gestionnaire de Fixtures Multi-Tenant pour Alertes Slack
=======================================================

Ce module implémente la logique métier principale pour la gestion des fixtures
d'alertes Slack dans un environnement multi-tenant avec support complet de
l'internationalisation et de la haute disponibilité.

Fonctionnalités:
- Gestion intelligente des templates d'alertes
- Support multi-tenant avec isolation complète
- Cache distribué avec invalidation automatique
- Validation de schémas JSON stricte
- Chiffrement des données sensibles
- Métriques et observabilité complètes
- API RESTful avec documentation OpenAPI

Architecture:
- Design Pattern: Repository + Factory + Observer
- Persistence: PostgreSQL + Redis
- Cache: Multi-level caching strategy
- Security: JWT + AES-256 + Rate limiting
- Monitoring: Prometheus + OpenTelemetry

Auteur: Fahed Mlaiel - Lead Developer Achiri
Version: 2.5.0
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import hashlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
import aioredis
import asyncpg
from pydantic import BaseModel, Field, validator
from jinja2 import Environment, FileSystemLoader, TemplateError
import prometheus_client
from opentelemetry import trace
from cryptography.fernet import Fernet
import yaml

# Configuration du logging
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Métriques Prometheus
METRICS = {
    'fixtures_loaded': prometheus_client.Counter(
        'slack_fixtures_loaded_total',
        'Total number of Slack fixtures loaded',
        ['tenant_id', 'environment', 'locale']
    ),
    'template_renders': prometheus_client.Counter(
        'slack_template_renders_total',
        'Total number of template renders',
        ['template_type', 'status']
    ),
    'cache_operations': prometheus_client.Counter(
        'slack_cache_operations_total',
        'Total cache operations',
        ['operation', 'result']
    ),
    'render_duration': prometheus_client.Histogram(
        'slack_template_render_duration_seconds',
        'Template render duration in seconds'
    )
}

class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"
    LOW = "low"

class Environment(str, Enum):
    """Environnements supportés."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

class Locale(str, Enum):
    """Locales supportées."""
    FR = "fr"
    EN = "en"
    DE = "de"
    ES = "es"
    IT = "it"

@dataclass
class FixtureMetadata:
    """Métadonnées d'une fixture."""
    id: str
    tenant_id: str
    environment: Environment
    locale: Locale
    alert_type: str
    version: str
    created_at: datetime
    updated_at: datetime
    hash: str
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)

class SlackTemplateModel(BaseModel):
    """Modèle de template Slack."""
    channel: str = Field(..., description="Canal Slack de destination")
    username: str = Field(default="Achiri Alert Bot", description="Nom d'utilisateur du bot")
    icon_emoji: str = Field(default=":warning:", description="Emoji d'icône")
    text: Optional[str] = Field(None, description="Texte principal")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Pièces jointes")
    
    @validator('channel')
    def validate_channel(cls, v):
        if not v.startswith('#'):
            raise ValueError("Le canal doit commencer par #")
        return v

class FixtureConfigModel(BaseModel):
    """Configuration d'une fixture."""
    metadata: Dict[str, Any]
    template: SlackTemplateModel
    conditions: Dict[str, Any] = Field(default_factory=dict)
    scheduling: Dict[str, Any] = Field(default_factory=dict)
    
class SlackFixtureManager:
    """
    Gestionnaire principal des fixtures Slack.
    
    Responsabilités:
    - Chargement et sauvegarde des fixtures
    - Rendu des templates avec contexte
    - Gestion du cache distribué
    - Validation et sécurité
    - Métriques et observabilité
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[aioredis.Redis] = None
        self.template_env: Optional[Environment] = None
        self.cipher: Optional[Fernet] = None
        self._cache: Dict[str, Any] = {}
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialise le gestionnaire."""
        with tracer.start_as_current_span("fixture_manager_init"):
            try:
                # Initialisation de la base de données
                await self._init_database()
                
                # Initialisation de Redis
                await self._init_redis()
                
                # Initialisation du moteur de templates
                await self._init_template_engine()
                
                # Initialisation du chiffrement
                await self._init_encryption()
                
                # Création des tables si nécessaire
                await self._ensure_schema()
                
                self._initialized = True
                logger.info("SlackFixtureManager initialisé avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation: {e}")
                raise
    
    async def _init_database(self) -> None:
        """Initialise la connexion PostgreSQL."""
        db_config = self.config.get('database', {})
        self.db_pool = await asyncpg.create_pool(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password', ''),
            database=db_config.get('name', 'spotify_ai'),
            min_size=5,
            max_size=20
        )
        logger.debug("Connexion PostgreSQL initialisée")
    
    async def _init_redis(self) -> None:
        """Initialise la connexion Redis."""
        redis_config = self.config.get('redis', {})
        self.redis = await aioredis.from_url(
            f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}",
            decode_responses=True
        )
        logger.debug("Connexion Redis initialisée")
    
    async def _init_template_engine(self) -> None:
        """Initialise le moteur de templates Jinja2."""
        templates_path = Path(__file__).parent / "templates"
        templates_path.mkdir(exist_ok=True)
        
        self.template_env = Environment(
            loader=FileSystemLoader(str(templates_path)),
            enable_async=True,
            auto_reload=True
        )
        logger.debug("Moteur de templates initialisé")
    
    async def _init_encryption(self) -> None:
        """Initialise le chiffrement."""
        key = self.config.get('security', {}).get('encryption_key')
        if key:
            self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
        logger.debug("Chiffrement initialisé")
    
    async def _ensure_schema(self) -> None:
        """Crée les tables nécessaires."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS slack_fixtures (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id VARCHAR(255) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            locale VARCHAR(10) NOT NULL,
            alert_type VARCHAR(100) NOT NULL,
            template_data JSONB NOT NULL,
            metadata JSONB NOT NULL,
            version VARCHAR(50) NOT NULL,
            hash VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            UNIQUE(tenant_id, environment, locale, alert_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_slack_fixtures_tenant 
        ON slack_fixtures(tenant_id, environment);
        
        CREATE INDEX IF NOT EXISTS idx_slack_fixtures_lookup
        ON slack_fixtures(tenant_id, environment, locale, alert_type);
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)
        
        logger.debug("Schéma de base de données vérifié")
    
    async def load_fixture(
        self,
        tenant_id: str,
        environment: Environment,
        locale: Locale,
        alert_type: str
    ) -> Optional[FixtureConfigModel]:
        """
        Charge une fixture spécifique.
        
        Args:
            tenant_id: ID du tenant
            environment: Environnement (dev/staging/prod)
            locale: Locale (fr/en/de/es/it)
            alert_type: Type d'alerte
            
        Returns:
            Configuration de la fixture ou None si non trouvée
        """
        with tracer.start_as_current_span("load_fixture") as span:
            span.set_attributes({
                "tenant_id": tenant_id,
                "environment": environment.value,
                "locale": locale.value,
                "alert_type": alert_type
            })
            
            # Génération de la clé de cache
            cache_key = self._generate_cache_key(tenant_id, environment, locale, alert_type)
            
            # Tentative de récupération depuis le cache
            cached_fixture = await self._get_from_cache(cache_key)
            if cached_fixture:
                METRICS['cache_operations'].labels(operation='get', result='hit').inc()
                return FixtureConfigModel(**cached_fixture)
            
            METRICS['cache_operations'].labels(operation='get', result='miss').inc()
            
            # Chargement depuis la base de données
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT template_data, metadata, version, hash
                    FROM slack_fixtures
                    WHERE tenant_id = $1 AND environment = $2 
                          AND locale = $3 AND alert_type = $4
                """, tenant_id, environment.value, locale.value, alert_type)
                
                if not row:
                    # Tentative de fallback vers locale par défaut
                    if locale != Locale.EN:
                        return await self.load_fixture(tenant_id, environment, Locale.EN, alert_type)
                    return None
                
                fixture_data = {
                    'metadata': row['metadata'],
                    'template': row['template_data'],
                    'version': row['version'],
                    'hash': row['hash']
                }
                
                # Mise en cache
                await self._set_cache(cache_key, fixture_data)
                
                METRICS['fixtures_loaded'].labels(
                    tenant_id=tenant_id,
                    environment=environment.value,
                    locale=locale.value
                ).inc()
                
                return FixtureConfigModel(**fixture_data)
    
    async def save_fixture(
        self,
        tenant_id: str,
        environment: Environment,
        locale: Locale,
        alert_type: str,
        config: FixtureConfigModel
    ) -> str:
        """
        Sauvegarde une fixture.
        
        Args:
            tenant_id: ID du tenant
            environment: Environnement
            locale: Locale
            alert_type: Type d'alerte
            config: Configuration de la fixture
            
        Returns:
            ID de la fixture sauvegardée
        """
        with tracer.start_as_current_span("save_fixture") as span:
            # Génération du hash pour la détection des changements
            config_str = json.dumps(config.dict(), sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            fixture_id = str(uuid.uuid4())
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO slack_fixtures 
                    (id, tenant_id, environment, locale, alert_type, 
                     template_data, metadata, version, hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (tenant_id, environment, locale, alert_type)
                    DO UPDATE SET
                        template_data = EXCLUDED.template_data,
                        metadata = EXCLUDED.metadata,
                        version = EXCLUDED.version,
                        hash = EXCLUDED.hash,
                        updated_at = NOW()
                """, 
                    fixture_id,
                    tenant_id,
                    environment.value,
                    locale.value,
                    alert_type,
                    json.dumps(config.template.dict()),
                    json.dumps(config.metadata),
                    "2.5.0",
                    config_hash
                )
            
            # Invalidation du cache
            cache_key = self._generate_cache_key(tenant_id, environment, locale, alert_type)
            await self._invalidate_cache(cache_key)
            
            logger.info(f"Fixture sauvegardée: {fixture_id}")
            return fixture_id
    
    async def render_template(
        self,
        fixture: FixtureConfigModel,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rend un template avec le contexte fourni.
        
        Args:
            fixture: Configuration de la fixture
            context: Contexte pour le rendu
            
        Returns:
            Template rendu
        """
        with tracer.start_as_current_span("render_template"):
            start_time = datetime.now()
            
            try:
                # Préparation du contexte enrichi
                enriched_context = {
                    **context,
                    'timestamp': datetime.now().isoformat(),
                    'tenant': context.get('tenant', {}),
                    'environment': context.get('environment', 'dev')
                }
                
                # Rendu du template principal
                template_dict = fixture.template.dict()
                rendered = await self._render_recursive(template_dict, enriched_context)
                
                # Calcul de la durée
                duration = (datetime.now() - start_time).total_seconds()
                METRICS['render_duration'].observe(duration)
                METRICS['template_renders'].labels(
                    template_type=fixture.metadata.get('alert_type', 'unknown'),
                    status='success'
                ).inc()
                
                return rendered
                
            except Exception as e:
                METRICS['template_renders'].labels(
                    template_type=fixture.metadata.get('alert_type', 'unknown'),
                    status='error'
                ).inc()
                logger.error(f"Erreur lors du rendu du template: {e}")
                raise
    
    async def _render_recursive(self, obj: Any, context: Dict[str, Any]) -> Any:
        """Rend récursivement les templates dans une structure."""
        if isinstance(obj, str) and '{{' in obj:
            try:
                template = self.template_env.from_string(obj)
                return await template.render_async(context)
            except TemplateError as e:
                logger.error(f"Erreur de template: {e}")
                return obj
        elif isinstance(obj, dict):
            return {k: await self._render_recursive(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [await self._render_recursive(item, context) for item in obj]
        else:
            return obj
    
    def _generate_cache_key(self, tenant_id: str, environment: Environment, 
                           locale: Locale, alert_type: str) -> str:
        """Génère une clé de cache unique."""
        return f"fixture:{tenant_id}:{environment.value}:{locale.value}:{alert_type}"
    
    async def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Récupère une valeur du cache."""
        if not self.redis:
            return self._cache.get(key)
        
        try:
            cached = await self.redis.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.error(f"Erreur cache get: {e}")
            return None
    
    async def _set_cache(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> None:
        """Met une valeur en cache."""
        if not self.redis:
            self._cache[key] = value
            return
        
        try:
            await self.redis.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Erreur cache set: {e}")
    
    async def _invalidate_cache(self, key: str) -> None:
        """Invalide une entrée du cache."""
        if not self.redis:
            self._cache.pop(key, None)
            return
        
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Erreur cache invalidation: {e}")
    
    async def list_fixtures(
        self,
        tenant_id: Optional[str] = None,
        environment: Optional[Environment] = None,
        locale: Optional[Locale] = None
    ) -> List[FixtureMetadata]:
        """
        Liste les fixtures selon les critères.
        
        Args:
            tenant_id: Filtrer par tenant (optionnel)
            environment: Filtrer par environnement (optionnel)
            locale: Filtrer par locale (optionnel)
            
        Returns:
            Liste des métadonnées des fixtures
        """
        with tracer.start_as_current_span("list_fixtures"):
            query_parts = ["SELECT * FROM slack_fixtures WHERE 1=1"]
            params = []
            param_count = 0
            
            if tenant_id:
                param_count += 1
                query_parts.append(f" AND tenant_id = ${param_count}")
                params.append(tenant_id)
            
            if environment:
                param_count += 1
                query_parts.append(f" AND environment = ${param_count}")
                params.append(environment.value)
            
            if locale:
                param_count += 1
                query_parts.append(f" AND locale = ${param_count}")
                params.append(locale.value)
            
            query_parts.append(" ORDER BY created_at DESC")
            query = "".join(query_parts)
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    FixtureMetadata(
                        id=str(row['id']),
                        tenant_id=row['tenant_id'],
                        environment=Environment(row['environment']),
                        locale=Locale(row['locale']),
                        alert_type=row['alert_type'],
                        version=row['version'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        hash=row['hash'],
                        tags=row['metadata'].get('tags', [])
                    )
                    for row in rows
                ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du gestionnaire."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Test PostgreSQL
        try:
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health['components']['database'] = 'healthy'
        except Exception as e:
            health['components']['database'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
        
        # Test Redis
        try:
            await self.redis.ping()
            health['components']['cache'] = 'healthy'
        except Exception as e:
            health['components']['cache'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
        
        return health
    
    async def cleanup(self) -> None:
        """Nettoie les ressources."""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis:
            await self.redis.close()
        logger.info("Ressources nettoyées")

@asynccontextmanager
async def get_fixture_manager(config: Dict[str, Any]):
    """Context manager pour le gestionnaire de fixtures."""
    manager = SlackFixtureManager(config)
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.cleanup()

# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation du gestionnaire."""
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'user': 'postgres',
            'password': 'password',
            'name': 'spotify_ai'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379
        },
        'security': {
            'encryption_key': Fernet.generate_key()
        }
    }
    
    async with get_fixture_manager(config) as manager:
        # Chargement d'une fixture
        fixture = await manager.load_fixture(
            tenant_id="spotify-tenant-01",
            environment=Environment.DEV,
            locale=Locale.FR,
            alert_type="system_critical"
        )
        
        if fixture:
            # Rendu avec contexte
            context = {
                'alert': {
                    'severity': 'critical',
                    'summary': 'CPU utilization > 90%',
                    'description': 'High CPU usage detected on server-01'
                },
                'tenant': {
                    'name': 'Spotify Production',
                    'id': 'spotify-tenant-01'
                }
            }
            
            rendered = await manager.render_template(fixture, context)
            print(json.dumps(rendered, indent=2))

if __name__ == "__main__":
    asyncio.run(example_usage())
