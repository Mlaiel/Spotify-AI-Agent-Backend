"""
Constantes système - Spotify AI Agent
Définitions centralisées des constantes, limites et configurations
"""

import os
from datetime import timedelta
from decimal import Decimal
from typing import Dict, Any, List, Set, Tuple
from enum import Enum
import re

# =============================================================================
# CONSTANTES SYSTÈME GÉNÉRALES
# =============================================================================

# Version et informations de l'application
APP_VERSION = "1.0.0"
APP_NAME = "Spotify AI Agent"
APP_DESCRIPTION = "Agent IA avancé pour l'analyse et la manipulation audio via Spotify"
API_VERSION = "v1"
BUILD_NUMBER = os.getenv("BUILD_NUMBER", "dev")

# Encodage et formats
DEFAULT_ENCODING = "utf-8"
DEFAULT_LOCALE = "fr_FR"
DEFAULT_TIMEZONE = "UTC"
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIME_FORMAT = "%H:%M:%S"

# =============================================================================
# CONSTANTES DE LIMITE ET PERFORMANCE
# =============================================================================

# Limites de données
MAX_STRING_LENGTH = 10000
MAX_TEXT_LENGTH = 1000000
MAX_ARRAY_SIZE = 10000
MAX_OBJECT_DEPTH = 50
MAX_FILE_SIZE_MB = 100
MAX_BATCH_SIZE = 1000
MAX_QUERY_RESULTS = 10000
MAX_CONCURRENT_REQUESTS = 100

# Timeouts (en secondes)
DEFAULT_TIMEOUT = 30
SHORT_TIMEOUT = 5
MEDIUM_TIMEOUT = 60
LONG_TIMEOUT = 300
DATABASE_TIMEOUT = 30
API_TIMEOUT = 15
CACHE_TIMEOUT = 3600
SESSION_TIMEOUT = 1800

# Retry et circuit breaker
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60
EXPONENTIAL_BACKOFF_MULTIPLIER = 2
MAX_EXPONENTIAL_BACKOFF_DELAY = 60

# =============================================================================
# CONSTANTES SPOTIFY API
# =============================================================================

# URLs et endpoints
SPOTIFY_BASE_URL = "https://api.spotify.com/v1"
SPOTIFY_ACCOUNTS_URL = "https://accounts.spotify.com"
SPOTIFY_AUTH_URL = f"{SPOTIFY_ACCOUNTS_URL}/authorize"
SPOTIFY_TOKEN_URL = f"{SPOTIFY_ACCOUNTS_URL}/api/token"

# Scopes Spotify
SPOTIFY_SCOPES = {
    'READ_PRIVATE': 'user-read-private',
    'READ_EMAIL': 'user-read-email',
    'READ_PLAYBACK_STATE': 'user-read-playback-state',
    'MODIFY_PLAYBACK_STATE': 'user-modify-playback-state',
    'READ_CURRENTLY_PLAYING': 'user-read-currently-playing',
    'READ_RECENTLY_PLAYED': 'user-read-recently-played',
    'READ_TOP_TRACKS': 'user-top-read',
    'PLAYLIST_READ_PRIVATE': 'playlist-read-private',
    'PLAYLIST_READ_COLLABORATIVE': 'playlist-read-collaborative',
    'PLAYLIST_MODIFY_PRIVATE': 'playlist-modify-private',
    'PLAYLIST_MODIFY_PUBLIC': 'playlist-modify-public',
    'LIBRARY_READ': 'user-library-read',
    'LIBRARY_MODIFY': 'user-library-modify',
    'FOLLOW_READ': 'user-follow-read',
    'FOLLOW_MODIFY': 'user-follow-modify'
}

# Limites API Spotify
SPOTIFY_RATE_LIMITS = {
    'REQUESTS_PER_SECOND': 10,
    'REQUESTS_PER_MINUTE': 600,
    'REQUESTS_PER_HOUR': 36000,
    'MAX_TRACKS_PER_REQUEST': 50,
    'MAX_ARTISTS_PER_REQUEST': 50,
    'MAX_ALBUMS_PER_REQUEST': 20,
    'MAX_PLAYLIST_TRACKS': 10000,
    'MAX_SEARCH_RESULTS': 1000,
    'MAX_AUDIO_FEATURES_REQUEST': 100
}

# =============================================================================
# CONSTANTES AUDIO ET ML
# =============================================================================

# Formats audio supportés
SUPPORTED_AUDIO_FORMATS = {
    'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'wma'
}

# Spécifications audio
AUDIO_SAMPLE_RATES = {
    'LOW_QUALITY': 22050,
    'STANDARD': 44100,
    'HIGH_QUALITY': 48000,
    'STUDIO_QUALITY': 96000
}

AUDIO_BIT_DEPTHS = {
    'STANDARD': 16,
    'HIGH_QUALITY': 24,
    'STUDIO_QUALITY': 32
}

# Configuration Spleeter
SPLEETER_MODELS = {
    '2stems': '2stems-16kHz',
    '4stems': '4stems-16kHz', 
    '5stems': '5stems-16kHz'
}

SPLEETER_STEMS = {
    '2stems': ['vocals', 'accompaniment'],
    '4stems': ['vocals', 'drums', 'bass', 'other'],
    '5stems': ['vocals', 'drums', 'bass', 'piano', 'other']
}

# Paramètres ML
ML_MODEL_CACHE_SIZE = 1000
ML_BATCH_SIZE = 32
ML_MAX_SEQUENCE_LENGTH = 1024
ML_EMBEDDING_DIMENSIONS = 512
ML_LEARNING_RATE = 0.001
ML_DROPOUT_RATE = 0.1

# =============================================================================
# CONSTANTES DE SÉCURITÉ
# =============================================================================

# Hashing et cryptage
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
BCRYPT_ROUNDS = 12
JWT_ALGORITHM = "HS256"
AES_KEY_LENGTH = 32
RSA_KEY_LENGTH = 2048

# Tokens et sessions
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30
API_KEY_LENGTH = 32
SESSION_ID_LENGTH = 64
CSRF_TOKEN_LENGTH = 32

# Rate limiting par IP
RATE_LIMIT_PER_IP = {
    'REQUESTS_PER_MINUTE': 60,
    'REQUESTS_PER_HOUR': 1000,
    'REQUESTS_PER_DAY': 10000
}

# Patterns de sécurité
SECURITY_PATTERNS = {
    'SQL_INJECTION': [
        r"('|(\\'))+.*(or|union)",
        r"(union).*select",
        r"(insert|delete|update).*into",
        r"(drop|create|alter).*table"
    ],
    'XSS': [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*="
    ],
    'COMMAND_INJECTION': [
        r"(\||&|;|\$\(|\`)",
        r"(rm|mv|cp|cat|ls)\s+",
        r"(wget|curl)\s+"
    ]
}

# =============================================================================
# CONSTANTES DATABASE
# =============================================================================

# Configuration base de données
DB_CONNECTION_POOL_SIZE = 20
DB_MAX_OVERFLOW = 30
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
DB_ECHO_SQL = False

# Contraintes de données
MAX_VARCHAR_LENGTH = 255
MAX_TEXT_LENGTH = 65535
MAX_LONGTEXT_LENGTH = 16777215
MAX_INDEX_LENGTH = 767
MAX_FOREIGN_KEYS = 64

# Paramètres de pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 1000
MIN_PAGE_SIZE = 1

# =============================================================================
# CONSTANTES CACHE ET REDIS
# =============================================================================

# Configuration Redis
REDIS_DEFAULT_PORT = 6379
REDIS_CONNECTION_POOL_SIZE = 50
REDIS_SOCKET_TIMEOUT = 5
REDIS_SOCKET_CONNECT_TIMEOUT = 5
REDIS_HEALTH_CHECK_INTERVAL = 30

# TTL par type de cache (en secondes)
CACHE_TTL = {
    'USER_SESSION': 1800,      # 30 minutes
    'API_RESPONSE': 300,       # 5 minutes
    'DATABASE_QUERY': 600,     # 10 minutes
    'SPOTIFY_TOKEN': 3000,     # 50 minutes (les tokens Spotify durent 1h)
    'ML_MODEL': 86400,         # 24 heures
    'AUDIO_ANALYSIS': 7200,    # 2 heures
    'USER_PREFERENCES': 3600,  # 1 heure
    'PLAYLIST_DATA': 1800,     # 30 minutes
    'SEARCH_RESULTS': 900      # 15 minutes
}

# Préfixes de clés cache
CACHE_KEY_PREFIXES = {
    'USER': 'user:',
    'SESSION': 'session:',
    'API': 'api:',
    'SPOTIFY': 'spotify:',
    'ML': 'ml:',
    'AUDIO': 'audio:',
    'PLAYLIST': 'playlist:',
    'SEARCH': 'search:',
    'RATE_LIMIT': 'ratelimit:'
}

# =============================================================================
# CONSTANTES VALIDATION
# =============================================================================

# Patterns de validation
VALIDATION_PATTERNS = {
    'EMAIL': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'PHONE': r'^\+?1?\d{9,15}$',
    'UUID': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    'SLUG': r'^[a-z0-9]+(?:-[a-z0-9]+)*$',
    'USERNAME': r'^[a-zA-Z0-9_]{3,30}$',
    'PASSWORD': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]',
    'URL': r'^https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*)?(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?$',
    'IPV4': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
    'IPV6': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
}

# Limites de validation
VALIDATION_LIMITS = {
    'MIN_PASSWORD_LENGTH': 8,
    'MAX_PASSWORD_LENGTH': 128,
    'MIN_USERNAME_LENGTH': 3,
    'MAX_USERNAME_LENGTH': 30,
    'MIN_NAME_LENGTH': 1,
    'MAX_NAME_LENGTH': 100,
    'MIN_DESCRIPTION_LENGTH': 0,
    'MAX_DESCRIPTION_LENGTH': 1000,
    'MIN_TAGS': 0,
    'MAX_TAGS': 20,
    'MIN_TAG_LENGTH': 1,
    'MAX_TAG_LENGTH': 50
}

# =============================================================================
# CONSTANTES HTTP ET API
# =============================================================================

# Codes de statut HTTP personnalisés
HTTP_STATUS_CODES = {
    'CUSTOM_VALIDATION_ERROR': 422,
    'CUSTOM_BUSINESS_ERROR': 409,
    'CUSTOM_RATE_LIMIT': 429,
    'CUSTOM_MAINTENANCE': 503,
    'CUSTOM_FEATURE_DISABLED': 501
}

# Headers HTTP personnalisés
CUSTOM_HEADERS = {
    'REQUEST_ID': 'X-Request-ID',
    'TENANT_ID': 'X-Tenant-ID',
    'USER_ID': 'X-User-ID',
    'API_VERSION': 'X-API-Version',
    'RATE_LIMIT_REMAINING': 'X-RateLimit-Remaining',
    'RATE_LIMIT_RESET': 'X-RateLimit-Reset',
    'RESPONSE_TIME': 'X-Response-Time'
}

# Types de contenu
CONTENT_TYPES = {
    'JSON': 'application/json',
    'XML': 'application/xml',
    'YAML': 'application/x-yaml',
    'CSV': 'text/csv',
    'PLAIN_TEXT': 'text/plain',
    'HTML': 'text/html',
    'BINARY': 'application/octet-stream',
    'FORM_DATA': 'application/x-www-form-urlencoded',
    'MULTIPART': 'multipart/form-data'
}

# =============================================================================
# CONSTANTES MONITORING ET LOGS
# =============================================================================

# Niveaux de log
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# Formats de log
LOG_FORMATS = {
    'SIMPLE': '%(levelname)s - %(message)s',
    'DETAILED': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'JSON': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
    'PRODUCTION': '%(asctime)s [%(process)d] [%(levelname)s] %(name)s: %(message)s'
}

# Métriques et alertes
METRICS_THRESHOLDS = {
    'CPU_USAGE_WARNING': 70.0,
    'CPU_USAGE_CRITICAL': 90.0,
    'MEMORY_USAGE_WARNING': 80.0,
    'MEMORY_USAGE_CRITICAL': 95.0,
    'DISK_USAGE_WARNING': 85.0,
    'DISK_USAGE_CRITICAL': 95.0,
    'RESPONSE_TIME_WARNING': 2.0,
    'RESPONSE_TIME_CRITICAL': 5.0,
    'ERROR_RATE_WARNING': 5.0,
    'ERROR_RATE_CRITICAL': 10.0
}

# =============================================================================
# CONSTANTES MULTI-TENANT
# =============================================================================

# Configuration tenant
DEFAULT_TENANT_ID = "default"
SYSTEM_TENANT_ID = "system"
MAX_TENANTS = 1000
MAX_USERS_PER_TENANT = 10000

# Limites par tenant
TENANT_LIMITS = {
    'MAX_PLAYLISTS': 1000,
    'MAX_TRACKS_PER_PLAYLIST': 10000,
    'MAX_AUDIO_FILES': 10000,
    'MAX_STORAGE_MB': 10000,
    'MAX_API_CALLS_PER_DAY': 100000,
    'MAX_ML_MODELS': 10,
    'MAX_CONCURRENT_SESSIONS': 100
}

# =============================================================================
# CONSTANTES ENVIRONNEMENT
# =============================================================================

# Environnements supportés
ENVIRONMENTS = {
    'DEVELOPMENT': 'dev',
    'TESTING': 'test',
    'STAGING': 'staging',
    'PRODUCTION': 'prod'
}

# Configuration par environnement
ENV_CONFIGS = {
    'dev': {
        'DEBUG': True,
        'LOG_LEVEL': 'DEBUG',
        'CACHE_TTL_MULTIPLIER': 0.1,
        'RATE_LIMIT_MULTIPLIER': 10.0
    },
    'test': {
        'DEBUG': True,
        'LOG_LEVEL': 'WARNING',
        'CACHE_TTL_MULTIPLIER': 0.01,
        'RATE_LIMIT_MULTIPLIER': 100.0
    },
    'staging': {
        'DEBUG': False,
        'LOG_LEVEL': 'INFO',
        'CACHE_TTL_MULTIPLIER': 0.5,
        'RATE_LIMIT_MULTIPLIER': 2.0
    },
    'prod': {
        'DEBUG': False,
        'LOG_LEVEL': 'WARNING',
        'CACHE_TTL_MULTIPLIER': 1.0,
        'RATE_LIMIT_MULTIPLIER': 1.0
    }
}

# =============================================================================
# CONSTANTES DE LOCALISATION
# =============================================================================

# Langues supportées
SUPPORTED_LANGUAGES = {
    'fr': 'Français',
    'en': 'English',
    'es': 'Español',
    'de': 'Deutsch',
    'it': 'Italiano',
    'pt': 'Português'
}

# Codes de pays
SUPPORTED_COUNTRIES = {
    'FR': 'France',
    'US': 'United States',
    'GB': 'United Kingdom',
    'DE': 'Germany',
    'ES': 'Spain',
    'IT': 'Italy',
    'CA': 'Canada',
    'AU': 'Australia'
}

# Devises
SUPPORTED_CURRENCIES = {
    'EUR': 'Euro',
    'USD': 'US Dollar',
    'GBP': 'British Pound',
    'CAD': 'Canadian Dollar',
    'AUD': 'Australian Dollar'
}

# =============================================================================
# CONSTANTES DE CONFIGURATION
# =============================================================================

# Variables d'environnement requises
REQUIRED_ENV_VARS = {
    'DATABASE_URL',
    'REDIS_URL',
    'SECRET_KEY',
    'SPOTIFY_CLIENT_ID',
    'SPOTIFY_CLIENT_SECRET'
}

# Variables d'environnement optionnelles avec valeurs par défaut
OPTIONAL_ENV_VARS = {
    'DEBUG': 'False',
    'LOG_LEVEL': 'INFO',
    'PORT': '8000',
    'WORKERS': '4',
    'MAX_CONNECTIONS': '100',
    'TIMEZONE': 'UTC'
}

# =============================================================================
# EXPRESSIONS RÉGULIÈRES UTILES
# =============================================================================

# Compilation des patterns pour performance
COMPILED_PATTERNS = {
    'EMAIL': re.compile(VALIDATION_PATTERNS['EMAIL']),
    'PHONE': re.compile(VALIDATION_PATTERNS['PHONE']),
    'UUID': re.compile(VALIDATION_PATTERNS['UUID']),
    'SLUG': re.compile(VALIDATION_PATTERNS['SLUG']),
    'USERNAME': re.compile(VALIDATION_PATTERNS['USERNAME']),
    'URL': re.compile(VALIDATION_PATTERNS['URL']),
    'IPV4': re.compile(VALIDATION_PATTERNS['IPV4']),
    'IPV6': re.compile(VALIDATION_PATTERNS['IPV6'])
}

# =============================================================================
# DICTIONNAIRES DE MESSAGES
# =============================================================================

# Messages d'erreur standardisés
ERROR_MESSAGES = {
    'VALIDATION_FAILED': "Validation failed for field '{field}'",
    'REQUIRED_FIELD': "Field '{field}' is required",
    'INVALID_FORMAT': "Invalid format for field '{field}'",
    'VALUE_TOO_LONG': "Value too long for field '{field}' (max: {max_length})",
    'VALUE_TOO_SHORT': "Value too short for field '{field}' (min: {min_length})",
    'INVALID_CHOICE': "Invalid choice for field '{field}'. Valid choices: {choices}",
    'DUPLICATE_VALUE': "Value already exists for field '{field}'",
    'RESOURCE_NOT_FOUND': "Resource not found: {resource_type}",
    'ACCESS_DENIED': "Access denied to resource: {resource_type}",
    'RATE_LIMIT_EXCEEDED': "Rate limit exceeded. Try again in {retry_after} seconds",
    'SERVICE_UNAVAILABLE': "Service temporarily unavailable: {service_name}",
    'INTERNAL_ERROR': "Internal server error occurred"
}

# Messages de succès
SUCCESS_MESSAGES = {
    'CREATED': "Resource created successfully",
    'UPDATED': "Resource updated successfully", 
    'DELETED': "Resource deleted successfully",
    'PROCESSED': "Request processed successfully",
    'AUTHENTICATED': "Authentication successful",
    'AUTHORIZED': "Authorization successful"
}

# =============================================================================
# CONSTANTES DE TEST
# =============================================================================

# Configuration des tests
TEST_CONFIG = {
    'DATABASE_URL': 'sqlite:///test.db',
    'REDIS_URL': 'redis://localhost:6379/1',
    'SECRET_KEY': 'test-secret-key',
    'SPOTIFY_CLIENT_ID': 'test-client-id',
    'SPOTIFY_CLIENT_SECRET': 'test-client-secret',
    'DEBUG': True,
    'TESTING': True
}

# Données de test
TEST_DATA = {
    'VALID_EMAIL': 'test@example.com',
    'VALID_PASSWORD': 'TestPassword123!',
    'VALID_USERNAME': 'testuser',
    'VALID_UUID': '123e4567-e89b-12d3-a456-426614174000',
    'SAMPLE_AUDIO_URL': 'https://example.com/sample.mp3',
    'SAMPLE_PLAYLIST_ID': '37i9dQZF1DXcBWIGoYBM5M'
}


__all__ = [
    # Informations application
    'APP_VERSION', 'APP_NAME', 'APP_DESCRIPTION', 'API_VERSION', 'BUILD_NUMBER',
    
    # Encodage et formats
    'DEFAULT_ENCODING', 'DEFAULT_LOCALE', 'DEFAULT_TIMEZONE',
    'DEFAULT_DATE_FORMAT', 'DEFAULT_DATETIME_FORMAT', 'DEFAULT_TIME_FORMAT',
    
    # Limites système
    'MAX_STRING_LENGTH', 'MAX_TEXT_LENGTH', 'MAX_ARRAY_SIZE', 'MAX_OBJECT_DEPTH',
    'MAX_FILE_SIZE_MB', 'MAX_BATCH_SIZE', 'MAX_QUERY_RESULTS', 'MAX_CONCURRENT_REQUESTS',
    
    # Timeouts
    'DEFAULT_TIMEOUT', 'SHORT_TIMEOUT', 'MEDIUM_TIMEOUT', 'LONG_TIMEOUT',
    'DATABASE_TIMEOUT', 'API_TIMEOUT', 'CACHE_TIMEOUT', 'SESSION_TIMEOUT',
    
    # Retry et circuit breaker
    'MAX_RETRY_ATTEMPTS', 'RETRY_DELAY_SECONDS', 'CIRCUIT_BREAKER_FAILURE_THRESHOLD',
    'CIRCUIT_BREAKER_RECOVERY_TIMEOUT', 'EXPONENTIAL_BACKOFF_MULTIPLIER', 'MAX_EXPONENTIAL_BACKOFF_DELAY',
    
    # Spotify
    'SPOTIFY_BASE_URL', 'SPOTIFY_ACCOUNTS_URL', 'SPOTIFY_AUTH_URL', 'SPOTIFY_TOKEN_URL',
    'SPOTIFY_SCOPES', 'SPOTIFY_RATE_LIMITS',
    
    # Audio et ML
    'SUPPORTED_AUDIO_FORMATS', 'AUDIO_SAMPLE_RATES', 'AUDIO_BIT_DEPTHS',
    'SPLEETER_MODELS', 'SPLEETER_STEMS', 'ML_MODEL_CACHE_SIZE', 'ML_BATCH_SIZE',
    'ML_MAX_SEQUENCE_LENGTH', 'ML_EMBEDDING_DIMENSIONS', 'ML_LEARNING_RATE', 'ML_DROPOUT_RATE',
    
    # Sécurité
    'PASSWORD_MIN_LENGTH', 'PASSWORD_MAX_LENGTH', 'BCRYPT_ROUNDS', 'JWT_ALGORITHM',
    'AES_KEY_LENGTH', 'RSA_KEY_LENGTH', 'JWT_ACCESS_TOKEN_EXPIRE_MINUTES',
    'JWT_REFRESH_TOKEN_EXPIRE_DAYS', 'API_KEY_LENGTH', 'SESSION_ID_LENGTH',
    'CSRF_TOKEN_LENGTH', 'RATE_LIMIT_PER_IP', 'SECURITY_PATTERNS',
    
    # Database
    'DB_CONNECTION_POOL_SIZE', 'DB_MAX_OVERFLOW', 'DB_POOL_TIMEOUT',
    'DB_POOL_RECYCLE', 'DB_ECHO_SQL', 'MAX_VARCHAR_LENGTH', 'MAX_TEXT_LENGTH',
    'MAX_LONGTEXT_LENGTH', 'MAX_INDEX_LENGTH', 'MAX_FOREIGN_KEYS',
    'DEFAULT_PAGE_SIZE', 'MAX_PAGE_SIZE', 'MIN_PAGE_SIZE',
    
    # Cache et Redis
    'REDIS_DEFAULT_PORT', 'REDIS_CONNECTION_POOL_SIZE', 'REDIS_SOCKET_TIMEOUT',
    'REDIS_SOCKET_CONNECT_TIMEOUT', 'REDIS_HEALTH_CHECK_INTERVAL',
    'CACHE_TTL', 'CACHE_KEY_PREFIXES',
    
    # Validation
    'VALIDATION_PATTERNS', 'VALIDATION_LIMITS', 'COMPILED_PATTERNS',
    
    # HTTP et API
    'HTTP_STATUS_CODES', 'CUSTOM_HEADERS', 'CONTENT_TYPES',
    
    # Monitoring et logs
    'LOG_LEVELS', 'LOG_FORMATS', 'METRICS_THRESHOLDS',
    
    # Multi-tenant
    'DEFAULT_TENANT_ID', 'SYSTEM_TENANT_ID', 'MAX_TENANTS',
    'MAX_USERS_PER_TENANT', 'TENANT_LIMITS',
    
    # Environnement
    'ENVIRONMENTS', 'ENV_CONFIGS',
    
    # Localisation
    'SUPPORTED_LANGUAGES', 'SUPPORTED_COUNTRIES', 'SUPPORTED_CURRENCIES',
    
    # Configuration
    'REQUIRED_ENV_VARS', 'OPTIONAL_ENV_VARS',
    
    # Messages
    'ERROR_MESSAGES', 'SUCCESS_MESSAGES',
    
    # Tests
    'TEST_CONFIG', 'TEST_DATA'
]
