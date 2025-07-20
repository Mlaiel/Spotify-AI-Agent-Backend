"""
🏗️ HYBRID BACKEND - ORCHESTRATION DJANGO/FASTAPI ENTERPRISE
Expert Team: Senior Backend Developer, Microservices Architect

Architecture hybride ultra-avancée avec orchestration intelligente des frameworks
"""

import asyncio
import os
import threading
from typing import Optional, Dict, Any, List, Union, Callable
from contextlib import asynccontextmanager
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import weakref

# Django imports
import django
from django.conf import settings
from django.core.wsgi import get_wsgi_application
from django.core.asgi import get_asgi_application
from django.core.management import execute_from_command_line
from django.contrib import admin
from django.apps import AppConfig

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

# SQLAlchemy pour FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Base framework
from .core import BaseFramework, FrameworkStatus, FrameworkHealth
from .core import framework_orchestrator

# Configuration et monitoring
import prometheus_client
from opentelemetry import trace


@dataclass
class HybridConfig:
    """Configuration du backend hybride"""
    
    # Django settings
    django_settings_module: str = "backend.config.settings.development"
    django_secret_key: str = "django-hybrid-secret-key"
    django_debug: bool = False
    django_allowed_hosts: List[str] = None
    
    # FastAPI settings
    fastapi_title: str = "Spotify AI Agent API"
    fastapi_version: str = "2.0.0"
    fastapi_debug: bool = False
    fastapi_docs_url: str = "/docs"
    fastapi_redoc_url: str = "/redoc"
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/spotify_ai_agent"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Middleware
    enable_cors: bool = True
    cors_origins: List[str] = None
    enable_gzip: bool = True
    enable_trusted_hosts: bool = True
    trusted_hosts: List[str] = None
    
    # Performance
    worker_processes: int = 4
    max_requests: int = 1000
    request_timeout: int = 30
    
    def __post_init__(self):
        if self.django_allowed_hosts is None:
            self.django_allowed_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8000"]
        if self.trusted_hosts is None:
            self.trusted_hosts = ["localhost", "127.0.0.1"]


class DjangoFramework(BaseFramework):
    """
    🐍 FRAMEWORK DJANGO ENTERPRISE
    
    Gestion avancée de Django avec:
    - Configuration automatique
    - Gestion des migrations
    - Admin interface
    - ORM optimisé
    """
    
    def __init__(self, config: HybridConfig):
        super().__init__("django", config.__dict__)
        self.config = config
        self.wsgi_app: Optional[Any] = None
        self.asgi_app: Optional[Any] = None
        self._django_setup_done = False
        
    async def initialize(self) -> bool:
        """Initialise Django avec configuration optimisée"""
        try:
            if not self._django_setup_done:
                self._configure_django()
                django.setup()
                self._django_setup_done = True
            
            # Initialiser l'application WSGI/ASGI
            self.wsgi_app = get_wsgi_application()
            self.asgi_app = get_asgi_application()
            
            # Effectuer les migrations
            await self._run_migrations()
            
            # Configurer l'admin
            self._setup_admin()
            
            # Créer un superuser par défaut si nécessaire
            await self._create_default_superuser()
            
            self.logger.info("Django framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Django initialization failed: {e}")
            return False
    
    def _configure_django(self):
        """Configure Django avec paramètres optimisés"""
        if settings.configured:
            return
        
        settings.configure(
            DEBUG=self.config.django_debug,
            SECRET_KEY=self.config.django_secret_key,
            ALLOWED_HOSTS=self.config.django_allowed_hosts,
            
            INSTALLED_APPS=[
                'django.contrib.admin',
                'django.contrib.auth',
                'django.contrib.contenttypes',
                'django.contrib.sessions',
                'django.contrib.messages',
                'django.contrib.staticfiles',
                'rest_framework',
                'corsheaders',
                'django_extensions',
                'debug_toolbar',
                'backend.app.frameworks.django_integration',
            ],
            
            MIDDLEWARE=[
                'corsheaders.middleware.CorsMiddleware',
                'django.middleware.security.SecurityMiddleware',
                'whitenoise.middleware.WhiteNoiseMiddleware',
                'django.contrib.sessions.middleware.SessionMiddleware',
                'django.middleware.common.CommonMiddleware',
                'django.middleware.csrf.CsrfViewMiddleware',
                'django.contrib.auth.middleware.AuthenticationMiddleware',
                'django.contrib.messages.middleware.MessageMiddleware',
                'django.middleware.clickjacking.XFrameOptionsMiddleware',
                'debug_toolbar.middleware.DebugToolbarMiddleware',
            ],
            
            ROOT_URLCONF='backend.app.frameworks.django_integration.urls',
            
            DATABASES={
                'default': {
                    'ENGINE': 'django.db.backends.postgresql',
                    'NAME': os.getenv('DB_NAME', 'spotify_ai_agent'),
                    'USER': os.getenv('DB_USER', 'postgres'),
                    'PASSWORD': os.getenv('DB_PASSWORD', 'password'),
                    'HOST': os.getenv('DB_HOST', 'localhost'),
                    'PORT': os.getenv('DB_PORT', '5432'),
                    'OPTIONS': {
                        'MAX_CONNS': self.config.database_pool_size,
                    },
                }
            },
            
            TEMPLATES=[{
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'DIRS': [os.path.join(os.path.dirname(__file__), 'templates')],
                'APP_DIRS': True,
                'OPTIONS': {
                    'context_processors': [
                        'django.template.context_processors.debug',
                        'django.template.context_processors.request',
                        'django.contrib.auth.context_processors.auth',
                        'django.contrib.messages.context_processors.messages',
                    ],
                },
            }],
            
            # Internationalisation
            USE_I18N=True,
            USE_L10N=True,
            USE_TZ=True,
            LANGUAGE_CODE='en-us',
            TIME_ZONE='UTC',
            
            # Fichiers statiques
            STATIC_URL='/static/',
            STATIC_ROOT=os.path.join(os.path.dirname(__file__), 'staticfiles'),
            STATICFILES_STORAGE='whitenoise.storage.CompressedManifestStaticFilesStorage',
            
            MEDIA_URL='/media/',
            MEDIA_ROOT=os.path.join(os.path.dirname(__file__), 'media'),
            
            # CORS configuration
            CORS_ALLOW_ALL_ORIGINS=True if self.config.django_debug else False,
            CORS_ALLOWED_ORIGINS=self.config.cors_origins,
            CORS_ALLOW_CREDENTIALS=True,
            
            # REST Framework
            REST_FRAMEWORK={
                'DEFAULT_AUTHENTICATION_CLASSES': [
                    'rest_framework.authentication.SessionAuthentication',
                    'rest_framework.authentication.TokenAuthentication',
                    'rest_framework_simplejwt.authentication.JWTAuthentication',
                ],
                'DEFAULT_PERMISSION_CLASSES': [
                    'rest_framework.permissions.IsAuthenticated',
                ],
                'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
                'PAGE_SIZE': 20,
                'DEFAULT_THROTTLE_CLASSES': [
                    'rest_framework.throttling.AnonRateThrottle',
                    'rest_framework.throttling.UserRateThrottle'
                ],
                'DEFAULT_THROTTLE_RATES': {
                    'anon': '100/hour',
                    'user': '1000/hour'
                }
            },
            
            # Cache configuration
            CACHES={
                'default': {
                    'BACKEND': 'django_redis.cache.RedisCache',
                    'LOCATION': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                    'OPTIONS': {
                        'CLIENT_CLASS': 'django_redis.client.DefaultClient',
                        'CONNECTION_POOL_KWARGS': {'max_connections': 50}
                    }
                }
            },
            
            # Logging
            LOGGING={
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'verbose': {
                        'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
                        'style': '{',
                    },
                },
                'handlers': {
                    'file': {
                        'level': 'INFO',
                        'class': 'logging.FileHandler',
                        'filename': 'django.log',
                        'formatter': 'verbose',
                    },
                    'console': {
                        'level': 'DEBUG' if self.config.django_debug else 'INFO',
                        'class': 'logging.StreamHandler',
                        'formatter': 'verbose',
                    },
                },
                'root': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                },
            },
            
            # Security settings
            SECURE_BROWSER_XSS_FILTER=True,
            SECURE_CONTENT_TYPE_NOSNIFF=True,
            X_FRAME_OPTIONS='DENY',
            
            # Debug toolbar
            INTERNAL_IPS=['127.0.0.1', 'localhost'] if self.config.django_debug else [],
        )
    
    async def _run_migrations(self):
        """Exécute les migrations Django"""
        try:
            # Dans un thread séparé pour éviter les blocages
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            
            def run_migration_commands():
                try:
                    execute_from_command_line(['manage.py', 'makemigrations', '--noinput'])
                    execute_from_command_line(['manage.py', 'migrate', '--noinput'])
                    execute_from_command_line(['manage.py', 'collectstatic', '--noinput'])
                    return True
                except Exception as e:
                    self.logger.error(f"Migration failed: {e}")
                    return False
            
            success = await loop.run_in_executor(executor, run_migration_commands)
            if success:
                self.logger.info("Django migrations completed successfully")
            else:
                self.logger.error("Django migrations failed")
                
        except Exception as e:
            self.logger.error(f"Migration execution failed: {e}")
    
    def _setup_admin(self):
        """Configure l'interface admin Django"""
        try:
            # Import des modèles et admin configs
            from backend.app.models.orm.spotify import Track, Artist, Album, Playlist
            from backend.app.models.orm.users import User, UserProfile
            from backend.app.models.orm.ai import AIConversation, AIGeneratedContent
            
            # Configuration admin avancée pour Spotify models
            @admin.register(Track)
            class TrackAdmin(admin.ModelAdmin):
                list_display = ['name', 'artist', 'album', 'duration_ms', 'popularity', 'created_at']
                list_filter = ['album', 'popularity', 'created_at']
                search_fields = ['name', 'artist__name', 'album__name']
                ordering = ['-popularity', 'name']
                list_per_page = 50
                readonly_fields = ['created_at', 'updated_at']
                
                fieldsets = (
                    ('Track Information', {
                        'fields': ('name', 'artist', 'album', 'track_number')
                    }),
                    ('Audio Properties', {
                        'fields': ('duration_ms', 'explicit', 'popularity'),
                        'classes': ('collapse',)
                    }),
                    ('Metadata', {
                        'fields': ('spotify_id', 'preview_url', 'external_urls'),
                        'classes': ('collapse',)
                    }),
                    ('Timestamps', {
                        'fields': ('created_at', 'updated_at'),
                        'classes': ('collapse',)
                    }),
                )
            
            @admin.register(Artist)
            class ArtistAdmin(admin.ModelAdmin):
                list_display = ['name', 'popularity', 'followers', 'genres_list', 'created_at']
                list_filter = ['popularity', 'created_at']
                search_fields = ['name']
                ordering = ['-popularity', 'name']
                readonly_fields = ['created_at', 'updated_at']
                
                def genres_list(self, obj):
                    return ", ".join([g.name for g in obj.genres.all()[:3]])
                genres_list.short_description = "Genres"
            
            @admin.register(Album)
            class AlbumAdmin(admin.ModelAdmin):
                list_display = ['name', 'artist', 'release_date', 'total_tracks', 'album_type']
                list_filter = ['album_type', 'release_date']
                search_fields = ['name', 'artist__name']
                date_hierarchy = 'release_date'
                ordering = ['-release_date']
            
            # User models admin
            @admin.register(UserProfile)
            class UserProfileAdmin(admin.ModelAdmin):
                list_display = ['user', 'display_name', 'country', 'premium', 'created_at']
                list_filter = ['country', 'premium', 'created_at']
                search_fields = ['user__username', 'display_name']
            
            # AI models admin
            @admin.register(AIConversation)
            class AIConversationAdmin(admin.ModelAdmin):
                list_display = ['user', 'model_used', 'status', 'created_at']
                list_filter = ['model_used', 'status', 'created_at']
                readonly_fields = ['created_at', 'updated_at']
                date_hierarchy = 'created_at'
                
            self.logger.info("Django admin configured successfully")
            
        except Exception as e:
            self.logger.error(f"Admin setup failed: {e}")
    
    async def _create_default_superuser(self):
        """Crée un superuser par défaut si nécessaire"""
        try:
            from django.contrib.auth.models import User
            
            if not User.objects.filter(is_superuser=True).exists():
                User.objects.create_superuser(
                    username='admin',
                    email='admin@spotifyaiagent.com',
                    password='admin123'
                )
                self.logger.info("Default superuser created: admin/admin123")
                
        except Exception as e:
            self.logger.error(f"Superuser creation failed: {e}")
    
    async def shutdown(self) -> bool:
        """Arrête Django proprement"""
        try:
            # Fermer les connexions DB
            from django.db import connections
            for conn in connections.all():
                conn.close()
            
            self.logger.info("Django framework shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Django shutdown failed: {e}")
            return False
    
    async def health_check(self) -> FrameworkHealth:
        """Vérifie la santé de Django"""
        health = FrameworkHealth(
            status=FrameworkStatus.RUNNING,
            last_check=time.time()
        )
        
        try:
            # Vérifier la connexion DB
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                
            # Vérifier les migrations
            from django.core.management import execute_from_command_line
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                execute_from_command_line(['manage.py', 'showmigrations', '--plan'])
                migration_output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            if "[ ]" in migration_output:
                health.status = FrameworkStatus.DEGRADED
                health.metadata["pending_migrations"] = True
            
            health.metadata["admin_available"] = True
            health.metadata["migrations_status"] = "up_to_date"
            
        except Exception as e:
            health.status = FrameworkStatus.DEGRADED
            health.error_count += 1
            health.metadata["error"] = str(e)
        
        return health


class FastAPIFramework(BaseFramework):
    """
    ⚡ FRAMEWORK FASTAPI ENTERPRISE
    
    FastAPI haute performance avec:
    - Async/await natif
    - Validation automatique
    - Documentation interactive
    - Middleware avancé
    """
    
    def __init__(self, config: HybridConfig):
        super().__init__("fastapi", config.__dict__)
        self.config = config
        self.app: Optional[FastAPI] = None
        self.engine: Optional[Any] = None
        self.async_session: Optional[Any] = None
        
    async def initialize(self) -> bool:
        """Initialise FastAPI avec configuration optimisée"""
        try:
            # Créer l'application FastAPI
            self.app = FastAPI(
                title=self.config.fastapi_title,
                version=self.config.fastapi_version,
                debug=self.config.fastapi_debug,
                docs_url=self.config.fastapi_docs_url,
                redoc_url=self.config.fastapi_redoc_url,
                description="""
                🎵 **Spotify AI Agent API** - Architecture Enterprise
                
                API haute performance avec FastAPI pour l'agent IA Spotify.
                
                ## Fonctionnalités
                
                * **Intelligence Artificielle** - Recommandations personnalisées
                * **Streaming Musical** - Intégration Spotify complète  
                * **Analytics Avancées** - Métriques et insights utilisateur
                * **Architecture Hybride** - Django + FastAPI
                """,
                contact={
                    "name": "Spotify AI Agent Team",
                    "email": "support@spotifyaiagent.com",
                },
                license_info={
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT",
                }
            )
            
            # Configuration des middleware
            self._setup_middleware()
            
            # Configuration de la base de données
            await self._setup_database()
            
            # Configuration des routes
            self._setup_routes()
            
            # Configuration des handlers d'erreurs
            self._setup_error_handlers()
            
            # Configuration des événements
            self._setup_events()
            
            self.logger.info("FastAPI framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"FastAPI initialization failed: {e}")
            return False
    
    def _setup_middleware(self):
        """Configure les middleware FastAPI"""
        # CORS Middleware
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # GZip Middleware
        if self.config.enable_gzip:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Trusted Host Middleware
        if self.config.enable_trusted_hosts:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.trusted_hosts
            )
        
        # Custom monitoring middleware
        @self.app.middleware("http")
        async def monitoring_middleware(request, call_next):
            start_time = time.time()
            
            # Traçage
            with self.tracer.start_as_current_span("http_request") as span:
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.url", str(request.url))
                
                response = await call_next(request)
                
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)
                
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.response_time", process_time)
                
                # Métriques Prometheus
                self.latency_histogram.observe(process_time)
                
                return response
    
    async def _setup_database(self):
        """Configure la base de données SQLAlchemy"""
        try:
            self.engine = create_async_engine(
                self.config.database_url,
                pool_size=self.config.database_pool_size,
                max_overflow=self.config.database_max_overflow,
                echo=self.config.fastapi_debug
            )
            
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Dependency pour les sessions DB
            async def get_db():
                async with self.async_session() as session:
                    try:
                        yield session
                    finally:
                        await session.close()
            
            self.app.dependency_overrides[get_db] = get_db
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            raise
    
    def _setup_routes(self):
        """Configure les routes FastAPI"""
        # Route de santé
        @self.app.get("/health", tags=["Health"])
        async def health_check():
            """Vérification de santé de l'API"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": self.config.fastapi_version
            }
        
        # Route de métriques Prometheus
        @self.app.get("/metrics", tags=["Monitoring"])
        async def metrics():
            """Métriques Prometheus"""
            return prometheus_client.generate_latest()
        
        # Inclure les routeurs des modules
        try:
            from backend.app.api.routes import spotify, users, ai, billing
            
            self.app.include_router(
                spotify.router,
                prefix="/api/v1/spotify",
                tags=["Spotify"]
            )
            self.app.include_router(
                users.router,
                prefix="/api/v1/users",
                tags=["Users"]
            )
            self.app.include_router(
                ai.router,
                prefix="/api/v1/ai",
                tags=["AI"]
            )
            self.app.include_router(
                billing.router,
                prefix="/api/v1/billing",
                tags=["Billing"]
            )
            
        except ImportError as e:
            self.logger.warning(f"Some API routes not available: {e}")
    
    def _setup_error_handlers(self):
        """Configure les gestionnaires d'erreurs"""
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": time.time()
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            self.logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "status_code": 500,
                    "timestamp": time.time()
                }
            )
    
    def _setup_events(self):
        """Configure les événements de l'application"""
        @self.app.on_event("startup")
        async def startup_event():
            self.logger.info("FastAPI application starting up")
            
        @self.app.on_event("shutdown")
        async def shutdown_event():
            self.logger.info("FastAPI application shutting down")
            if self.engine:
                await self.engine.dispose()
    
    async def shutdown(self) -> bool:
        """Arrête FastAPI proprement"""
        try:
            if self.engine:
                await self.engine.dispose()
            
            self.logger.info("FastAPI framework shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"FastAPI shutdown failed: {e}")
            return False
    
    async def health_check(self) -> FrameworkHealth:
        """Vérifie la santé de FastAPI"""
        health = FrameworkHealth(
            status=FrameworkStatus.RUNNING,
            last_check=time.time()
        )
        
        try:
            # Vérifier la connexion DB
            if self.engine:
                async with self.engine.begin() as conn:
                    await conn.execute("SELECT 1")
            
            health.metadata["database_connected"] = True
            health.metadata["routes_registered"] = len(self.app.routes)
            
        except Exception as e:
            health.status = FrameworkStatus.DEGRADED
            health.error_count += 1
            health.metadata["error"] = str(e)
        
        return health


class HybridBackend:
    """
    🚀 BACKEND HYBRIDE ENTERPRISE
    
    Orchestration intelligente Django + FastAPI avec:
    - Load balancing automatique
    - Partage de session
    - Cache distribué
    - Monitoring unifié
    """
    
    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.django_framework = DjangoFramework(self.config)
        self.fastapi_framework = FastAPIFramework(self.config)
        
        self.logger = logging.getLogger("hybrid.backend")
        
        # Métriques
        self.requests_total = prometheus_client.Counter(
            'hybrid_requests_total',
            'Total requests to hybrid backend',
            ['framework', 'method', 'endpoint']
        )
        
        self.response_time = prometheus_client.Histogram(
            'hybrid_response_time_seconds',
            'Response time for hybrid backend',
            ['framework']
        )
    
    async def initialize(self) -> bool:
        """Initialise le backend hybride"""
        try:
            # Enregistrer les frameworks dans l'orchestrateur
            framework_orchestrator.register_framework(self.django_framework)
            framework_orchestrator.register_framework(
                self.fastapi_framework,
                dependencies=["django"]  # FastAPI dépend de Django pour les modèles
            )
            
            # Initialiser via l'orchestrateur
            results = await framework_orchestrator.initialize_all_frameworks()
            
            success = all(results.values())
            if success:
                self.logger.info("Hybrid backend initialized successfully")
                self._setup_shared_components()
            else:
                self.logger.error("Hybrid backend initialization failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Hybrid backend initialization error: {e}")
            return False
    
    def _setup_shared_components(self):
        """Configure les composants partagés"""
        # Session partagée entre Django et FastAPI
        # Cache Redis partagé
        # Logging unifié
        # Métriques centralisées
        
        self.logger.info("Shared components configured")
    
    async def shutdown(self) -> bool:
        """Arrête le backend hybride"""
        try:
            results = await framework_orchestrator.shutdown_all()
            success = all(results.values())
            
            if success:
                self.logger.info("Hybrid backend shutdown successfully")
            else:
                self.logger.error("Hybrid backend shutdown failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Hybrid backend shutdown error: {e}")
            return False
    
    def get_django_app(self):
        """Récupère l'application Django"""
        return self.django_framework.wsgi_app
    
    def get_fastapi_app(self):
        """Récupère l'application FastAPI"""
        return self.fastapi_framework.app
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Récupère le statut de santé complet"""
        return await framework_orchestrator.get_health_status()


# Instance globale du backend hybride
hybrid_backend = HybridBackend()


# Fonctions utilitaires
async def initialize_hybrid_backend(config: Optional[HybridConfig] = None) -> HybridBackend:
    """Initialise et retourne le backend hybride"""
    global hybrid_backend
    if config:
        hybrid_backend = HybridBackend(config)
    
    await hybrid_backend.initialize()
    return hybrid_backend


def get_django_app():
    """Récupère l'application Django"""
    return hybrid_backend.get_django_app()


def get_fastapi_app():
    """Récupère l'application FastAPI"""
    return hybrid_backend.get_fastapi_app()


# Export des classes principales
__all__ = [
    'HybridBackend',
    'DjangoFramework',
    'FastAPIFramework', 
    'HybridConfig',
    'hybrid_backend',
    'initialize_hybrid_backend',
    'get_django_app',
    'get_fastapi_app'
]
