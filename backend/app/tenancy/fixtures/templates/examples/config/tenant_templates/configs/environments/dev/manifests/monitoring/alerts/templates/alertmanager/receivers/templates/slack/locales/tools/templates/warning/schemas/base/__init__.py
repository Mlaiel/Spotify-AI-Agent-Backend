"""
Module de schémas de base - Spotify AI Agent
Fondations réutilisables pour tous les schémas Pydantic

Ce module fournit:
- Types personnalisés avancés
- Modèles de base avec fonctionnalités communes
- Mixins réutilisables pour fonctionnalités spécialisées
- Énumérations intelligentes avec logique métier
- Validateurs spécialisés
- Sérialiseurs multi-format
- Système d'exceptions structuré
- Utilitaires et helpers
- Constantes système

Auteur: Fahed Mlaiel
"""

from .types import (
    # Types de base
    TenantId, UserId, SessionId, RequestId, 
    SqlQuery, JsonString, Base64String, HexString,
    PositiveInt, NonNegativeInt, PositiveFloat, NonNegativeFloat,
    StrictEmail, StrictUrl, StrictUUID,
    
    # Types complexes
    Coordinates, TimeRange, Threshold, Contact, Address, Money,
    
    # Modèles de base
    BaseSchema, TimestampedSchema, AuditableSchema, VersionedSchema,
    TenantAwareSchema, CacheableSchema, ValidatableSchema
)

from .enums import (
    # Énumérations de base
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment,
    NotificationChannel, EscalationLevel, CorrelationMethod,
    WorkflowStatus, IncidentStatus, ModelFramework, SecurityLevel,
    PermissionFlag, FeatureFlag,
    
    # Classe de base
    SmartEnum
)

from .mixins import (
    # Mixins fonctionnels
    VersionedMixin, CacheableMixin, ObservableMixin,
    WorkflowMixin, SecurityMixin, PerformanceMixin
)

from .validators import (
    # Classes de validation
    ValidationSeverity, ValidationScope, ValidationResult, BaseValidator,
    StringValidator, NumberValidator, EmailValidator, URLValidator,
    SQLValidator, BusinessRuleValidator, CompositeValidator
)

from .serializers import (
    # Formats et options
    SerializationFormat, CompressionType, SerializationOptions, SerializationResult,
    
    # Sérialiseurs
    BaseSerializer, JsonSerializer, YamlSerializer, XmlSerializer,
    MsgPackSerializer, PickleSerializer, SerializerFactory, MultiFormatSerializer,
    
    # Fonctions utilitaires
    serialize, deserialize, default_serializer
)

from .exceptions import (
    # Catégories et sévérité
    ErrorCategory, ErrorSeverity, ErrorContext,
    
    # Exception de base
    BaseSpotifyException,
    
    # Exceptions spécialisées
    ValidationError, SchemaValidationError, BusinessRuleViolationError,
    AuthenticationError, AuthorizationError, TokenExpiredError,
    DataIntegrityError, ResourceNotFoundError, DuplicateResourceError,
    ExternalServiceError, APIRateLimitError, SpotifyAPIError,
    ResourceExhaustionError, MemoryExhaustionError, StorageExhaustionError,
    ConfigurationError, MissingConfigurationError,
    SecurityError, InjectionAttemptError,
    
    # Gestionnaire
    ExceptionHandler, global_exception_handler
)

from .constants import (
    # Informations application
    APP_VERSION, APP_NAME, API_VERSION,
    
    # Limites système
    MAX_STRING_LENGTH, MAX_BATCH_SIZE, DEFAULT_TIMEOUT,
    
    # Spotify
    SPOTIFY_BASE_URL, SPOTIFY_SCOPES, SPOTIFY_RATE_LIMITS,
    
    # Audio et ML
    SUPPORTED_AUDIO_FORMATS, SPLEETER_MODELS,
    
    # Sécurité
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES, BCRYPT_ROUNDS,
    
    # Cache
    CACHE_TTL, CACHE_KEY_PREFIXES,
    
    # Validation
    VALIDATION_PATTERNS, COMPILED_PATTERNS,
    
    # Messages
    ERROR_MESSAGES, SUCCESS_MESSAGES
)

from .utils import (
    # Utilitaires de chaînes
    normalize_string, slugify, truncate_string, mask_sensitive_data,
    
    # Validation
    is_valid_email, is_valid_url, validate_password_strength,
    
    # Cryptographie
    generate_random_string, generate_api_key, hash_password,
    
    # Date/heure
    now_utc, format_duration, parse_datetime,
    
    # Formatage
    format_bytes, format_number, format_percentage,
    
    # Structures de données
    deep_merge, flatten_dict, chunk_list, group_by,
    
    # Performance
    timing_decorator, memoize, retry_on_exception,
    
    # Divers
    get_client_ip, detect_file_type, compress_string
)

# Configuration par défaut pour tous les schémas
DEFAULT_SCHEMA_CONFIG = {
    'validate_assignment': True,
    'use_enum_values': True,
    'allow_population_by_field_name': True,
    'str_strip_whitespace': True,
    'extra': 'forbid',
    'json_encoders': {
        # Encodeurs JSON personnalisés seront ajoutés ici
    }
}

# Métadonnées du module
__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__description__ = "Module de schémas de base pour Spotify AI Agent"

__all__ = [
    # Types personnalisés
    'TenantId', 'UserId', 'SessionId', 'RequestId',
    'SqlQuery', 'JsonString', 'Base64String', 'HexString',
    'PositiveInt', 'NonNegativeInt', 'PositiveFloat', 'NonNegativeFloat',
    'StrictEmail', 'StrictUrl', 'StrictUUID',
    
    # Types complexes
    'Coordinates', 'TimeRange', 'Threshold', 'Contact', 'Address', 'Money',
    
    # Modèles de base
    'BaseSchema', 'TimestampedSchema', 'AuditableSchema', 'VersionedSchema',
    'TenantAwareSchema', 'CacheableSchema', 'ValidatableSchema',
    
    # Énumérations
    'AlertLevel', 'AlertStatus', 'WarningCategory', 'Priority', 'Environment',
    'NotificationChannel', 'EscalationLevel', 'CorrelationMethod',
    'WorkflowStatus', 'IncidentStatus', 'ModelFramework', 'SecurityLevel',
    'PermissionFlag', 'FeatureFlag', 'SmartEnum',
    
    # Mixins
    'VersionedMixin', 'CacheableMixin', 'ObservableMixin',
    'WorkflowMixin', 'SecurityMixin', 'PerformanceMixin',
    
    # Validation
    'ValidationSeverity', 'ValidationScope', 'ValidationResult', 'BaseValidator',
    'StringValidator', 'NumberValidator', 'EmailValidator', 'URLValidator',
    'SQLValidator', 'BusinessRuleValidator', 'CompositeValidator',
    
    # Sérialisation
    'SerializationFormat', 'CompressionType', 'SerializationOptions', 'SerializationResult',
    'BaseSerializer', 'JsonSerializer', 'YamlSerializer', 'XmlSerializer',
    'MsgPackSerializer', 'PickleSerializer', 'SerializerFactory', 'MultiFormatSerializer',
    'serialize', 'deserialize', 'default_serializer',
    
    # Exceptions
    'ErrorCategory', 'ErrorSeverity', 'ErrorContext', 'BaseSpotifyException',
    'ValidationError', 'SchemaValidationError', 'BusinessRuleViolationError',
    'AuthenticationError', 'AuthorizationError', 'TokenExpiredError',
    'DataIntegrityError', 'ResourceNotFoundError', 'DuplicateResourceError',
    'ExternalServiceError', 'APIRateLimitError', 'SpotifyAPIError',
    'ResourceExhaustionError', 'MemoryExhaustionError', 'StorageExhaustionError',
    'ConfigurationError', 'MissingConfigurationError',
    'SecurityError', 'InjectionAttemptError',
    'ExceptionHandler', 'global_exception_handler',
    
    # Constantes
    'APP_VERSION', 'APP_NAME', 'API_VERSION', 'MAX_STRING_LENGTH', 'MAX_BATCH_SIZE',
    'DEFAULT_TIMEOUT', 'SPOTIFY_BASE_URL', 'SPOTIFY_SCOPES', 'SPOTIFY_RATE_LIMITS',
    'SUPPORTED_AUDIO_FORMATS', 'SPLEETER_MODELS', 'JWT_ACCESS_TOKEN_EXPIRE_MINUTES',
    'BCRYPT_ROUNDS', 'CACHE_TTL', 'CACHE_KEY_PREFIXES', 'VALIDATION_PATTERNS',
    'COMPILED_PATTERNS', 'ERROR_MESSAGES', 'SUCCESS_MESSAGES',
    
    # Utilitaires
    'normalize_string', 'slugify', 'truncate_string', 'mask_sensitive_data',
    'is_valid_email', 'is_valid_url', 'validate_password_strength',
    'generate_random_string', 'generate_api_key', 'hash_password',
    'now_utc', 'format_duration', 'parse_datetime',
    'format_bytes', 'format_number', 'format_percentage',
    'deep_merge', 'flatten_dict', 'chunk_list', 'group_by',
    'timing_decorator', 'memoize', 'retry_on_exception',
    'get_client_ip', 'detect_file_type', 'compress_string',
    
    # Configuration
    'DEFAULT_SCHEMA_CONFIG'
]
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date de création de l'entité"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date de dernière modification"
    )
    deleted_at: Optional[datetime] = Field(
        None,
        description="Date de suppression (soft delete)"
    )
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp de modification"""
        return datetime.now(timezone.utc)
    
    @root_validator
    def validate_timestamps(cls, values):
        """Valide la cohérence des timestamps"""
        created_at = values.get('created_at')
        updated_at = values.get('updated_at')
        deleted_at = values.get('deleted_at')
        
        if created_at and updated_at and updated_at < created_at:
            raise ValueError('updated_at cannot be before created_at')
        
        if deleted_at and created_at and deleted_at < created_at:
            raise ValueError('deleted_at cannot be before created_at')
        
        return values
    
    def mark_updated(self):
        """Marque l'entité comme modifiée"""
        self.updated_at = datetime.now(timezone.utc)
    
    def mark_deleted(self):
        """Marque l'entité comme supprimée (soft delete)"""
        self.deleted_at = datetime.now(timezone.utc)
    
    @computed_field
    @property
    def is_deleted(self) -> bool:
        """Indique si l'entité est supprimée"""
        return self.deleted_at is not None
    
    @computed_field
    @property
    def age_seconds(self) -> float:
        """Âge de l'entité en secondes"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


class TenantMixin(BaseModel):
    """Mixin pour le support multi-tenant avec isolation des données"""
    tenant_id: StrictStr = Field(
        ...,
        description="Identifiant du tenant",
        min_length=1,
        max_length=255
    )
    organization_id: Optional[StrictStr] = Field(
        None,
        description="Identifiant de l'organisation",
        max_length=255
    )
    workspace_id: Optional[StrictStr] = Field(
        None,
        description="Identifiant de l'espace de travail",
        max_length=255
    )
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Valide l'identifiant de tenant"""
        if not v or not v.strip():
            raise ValueError('tenant_id cannot be empty')
        # Validation format UUID ou slug
        if not (len(v) >= 3 and v.replace('-', '').replace('_', '').isalnum()):
            raise ValueError('tenant_id must be alphanumeric with optional hyphens/underscores')
        return v.strip()
    
    @computed_field
    @property
    def tenant_context(self) -> Dict[str, Optional[str]]:
        """Contexte complet du tenant"""
        return {
            'tenant_id': self.tenant_id,
            'organization_id': self.organization_id,
            'workspace_id': self.workspace_id
        }


class MetadataMixin(BaseModel):
    """Mixin pour les métadonnées et tags avec support de recherche"""
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées flexibles en format clé-valeur"
    )
    tags: Set[str] = Field(
        default_factory=set,
        description="Tags pour la classification et la recherche"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Labels structurés pour l'organisation"
    )
    
    @validator('tags')
    def validate_tags(cls, v):
        """Valide et normalise les tags"""
        if not isinstance(v, (set, list)):
            return set()
        # Normalisation : minuscules, suppression espaces
        normalized_tags = set()
        for tag in v:
            if isinstance(tag, str) and tag.strip():
                normalized_tag = tag.strip().lower()
                if len(normalized_tag) <= 50:  # Limite de longueur
                    normalized_tags.add(normalized_tag)
        return normalized_tags
    
    @validator('labels')
    def validate_labels(cls, v):
        """Valide les labels"""
        if not isinstance(v, dict):
            return {}
        # Validation des clés et valeurs
        validated_labels = {}
        for key, value in v.items():
            if isinstance(key, str) and isinstance(value, str):
                key = key.strip()
                value = value.strip()
                if key and value and len(key) <= 100 and len(value) <= 255:
                    validated_labels[key] = value
        return validated_labels
    
    def add_tag(self, tag: str) -> bool:
        """Ajoute un tag avec validation"""
        if not isinstance(tag, str) or not tag.strip():
            return False
        normalized_tag = tag.strip().lower()
        if len(normalized_tag) <= 50:
            self.tags.add(normalized_tag)
            return True
        return False
    
    def remove_tag(self, tag: str) -> bool:
        """Supprime un tag"""
        normalized_tag = tag.strip().lower() if isinstance(tag, str) else ""
        if normalized_tag in self.tags:
            self.tags.remove(normalized_tag)
            return True
        return False
    
    def has_tag(self, tag: str) -> bool:
        """Vérifie la présence d'un tag"""
        normalized_tag = tag.strip().lower() if isinstance(tag, str) else ""
        return normalized_tag in self.tags
    
    def set_metadata(self, key: str, value: Any) -> bool:
        """Définit une métadonnée avec validation"""
        if not isinstance(key, str) or not key.strip():
            return False
        key = key.strip()
        if len(key) <= 100:
            self.metadata[key] = value
            return True
        return False
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Récupère une métadonnée"""
        return self.metadata.get(key, default)
    
    def remove_metadata(self, key: str) -> bool:
        """Supprime une métadonnée"""
        if key in self.metadata:
            del self.metadata[key]
            return True
        return False
    
    def set_label(self, key: str, value: str) -> bool:
        """Définit un label avec validation"""
        if not isinstance(key, str) or not isinstance(value, str):
            return False
        key = key.strip()
        value = value.strip()
        if key and value and len(key) <= 100 and len(value) <= 255:
            self.labels[key] = value
            return True
        return False
    
    def get_label(self, key: str, default: str = "") -> str:
        """Récupère un label"""
        return self.labels.get(key, default)
    
    def remove_label(self, key: str) -> bool:
        """Supprime un label"""
        if key in self.labels:
            del self.labels[key]
            return True
        return False


class AuditMixin(BaseModel):
    """Mixin pour l'audit trail et la traçabilité"""
    created_by: Optional[UUID] = Field(
        None,
        description="ID de l'utilisateur créateur"
    )
    updated_by: Optional[UUID] = Field(
        None,
        description="ID du dernier utilisateur modificateur"
    )
    version: PositiveInt = Field(
        1,
        description="Version de l'entité pour la gestion des conflits"
    )
    checksum: Optional[str] = Field(
        None,
        description="Checksum pour l'intégrité des données"
    )
    
    def update_version(self, user_id: Optional[UUID] = None):
        """Met à jour la version et l'utilisateur modificateur"""
        self.version += 1
        self.updated_by = user_id
        self.mark_updated() if hasattr(self, 'mark_updated') else None
    
    def calculate_checksum(self) -> str:
        """Calcule le checksum de l'entité"""
        # Exclure les champs liés à l'audit du calcul
        data = self.dict(exclude={'checksum', 'updated_at', 'updated_by'})
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Vérifie l'intégrité de l'entité"""
        if not self.checksum:
            return True  # Pas de checksum à vérifier
        current_checksum = self.calculate_checksum()
        return current_checksum == self.checksum
    
    def update_checksum(self):
        """Met à jour le checksum"""
        self.checksum = self.calculate_checksum()


class SoftDeleteMixin(BaseModel):
    """Mixin pour la suppression douce avec restauration"""
    is_deleted: bool = Field(
        False,
        description="Indicateur de suppression douce"
    )
    deleted_at: Optional[datetime] = Field(
        None,
        description="Date de suppression"
    )
    deleted_by: Optional[UUID] = Field(
        None,
        description="ID de l'utilisateur qui a supprimé"
    )
    delete_reason: Optional[str] = Field(
        None,
        description="Raison de la suppression",
        max_length=500
    )
    
    def soft_delete(self, user_id: Optional[UUID] = None, reason: Optional[str] = None):
        """Effectue une suppression douce"""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        self.deleted_by = user_id
        self.delete_reason = reason
    
    def restore(self, user_id: Optional[UUID] = None):
        """Restaure l'entité supprimée"""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
        self.delete_reason = None
        if hasattr(self, 'update_version'):
            self.update_version(user_id)
    
    @computed_field
    @property
    def can_be_restored(self) -> bool:
        """Indique si l'entité peut être restaurée"""
        return self.is_deleted and self.deleted_at is not None


# Classes de base composites
class FullBaseModel(BaseEntity, TimestampMixin, TenantMixin, MetadataMixin, AuditMixin, SoftDeleteMixin):
    """Modèle de base complet avec toutes les fonctionnalités"""
    
    class Config(BaseConfig):
        """Configuration héritée"""
        pass
    
    def to_dict(self, exclude_none: bool = True, exclude_deleted: bool = False) -> Dict[str, Any]:
        """Convertit le modèle en dictionnaire avec options de filtrage"""
        data = self.dict(exclude_none=exclude_none)
        
        if exclude_deleted and hasattr(self, 'is_deleted') and self.is_deleted:
            return {}
        
        return data
    
    def clone(self, **overrides) -> 'FullBaseModel':
        """Crée une copie de l'entité avec de nouveaux ID et timestamps"""
        data = self.dict(exclude={'id', 'created_at', 'updated_at', 'version'})
        data.update(overrides)
        return self.__class__(**data)


# Utilitaires et helpers
class ModelRegistry:
    """Registre global des modèles pour la réflexion"""
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, model_class: type):
        """Enregistre un modèle"""
        cls._models[model_class.__name__] = model_class
    
    @classmethod
    def get_model(cls, name: str) -> Optional[type]:
        """Récupère un modèle par nom"""
        return cls._models.get(name)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Liste tous les modèles enregistrés"""
        return list(cls._models.keys())


# Décorateur pour l'enregistrement automatique
def register_model(cls):
    """Décorateur pour l'enregistrement automatique des modèles"""
    ModelRegistry.register(cls)
    return cls


__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__description__ = "Module de base pour les schémas d'alerte avec architecture enterprise avancée"


__all__ = [
    # Métadonnées du module
    '__version__', '__author__', '__description__',
    
    # Classes de base
    'BaseEntity', 'FullBaseModel',
    
    # Mixins
    'TimestampMixin', 'TenantMixin', 'MetadataMixin', 'AuditMixin', 'SoftDeleteMixin',
    
    # Utilitaires
    'ModelRegistry', 'register_model', 'BaseConfig',
    
    # Variables de type
    'T', 'K', 'V',
    
    # Toutes les exportations des modules spécialisés seront automatiquement
    # ajoutées via les imports avec '*' ci-dessus
]
        min_length=1,
        max_length=255,
        regex=r'^[a-zA-Z0-9_-]+$',
        description="Identifiant du tenant"
    )
    tenant_name: Optional[StrictStr] = Field(
        None,
        description="Nom du tenant"
    )
    
    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Valide l'ID du tenant"""
        if not v or v.isspace():
            raise ValueError('tenant_id cannot be empty')
        return v.strip().lower()


class MetadataMixin(BaseModel):
    """Mixin pour les métadonnées"""
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées flexibles"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Tags pour le filtrage"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Labels système"
    )
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Valide les métadonnées"""
        if v is None:
            return {}
        # Limite la taille des métadonnées
        if len(str(v)) > 10000:  # 10KB max
            raise ValueError('Metadata size cannot exceed 10KB')
        return v


class AuditMixin(BaseModel):
    """Mixin pour l'audit trail"""
    created_by: Optional[StrictStr] = Field(
        None,
        description="Créé par (utilisateur/système)"
    )
    updated_by: Optional[StrictStr] = Field(
        None,
        description="Modifié par"
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version du document"
    )
    revision_id: Optional[UUID] = Field(
        default_factory=uuid4,
        description="ID de révision"
    )
    
    @validator('version', pre=True, always=True)
    def increment_version(cls, v, values):
        """Incrémente automatiquement la version"""
        if 'updated_at' in values and values.get('updated_at'):
            return (v or 1) + 1
        return v or 1


class SoftDeleteMixin(BaseModel):
    """Mixin pour la suppression logique"""
    is_deleted: bool = Field(
        default=False,
        description="Marqué comme supprimé"
    )
    deleted_at: Optional[datetime] = Field(
        None,
        description="Date de suppression"
    )
    deleted_by: Optional[StrictStr] = Field(
        None,
        description="Supprimé par"
    )
    
    @root_validator
    def validate_deletion(cls, values):
        """Valide la cohérence de la suppression"""
        is_deleted = values.get('is_deleted', False)
        deleted_at = values.get('deleted_at')
        deleted_by = values.get('deleted_by')
        
        if is_deleted and not deleted_at:
            values['deleted_at'] = datetime.now(timezone.utc)
        
        if not is_deleted and (deleted_at or deleted_by):
            raise ValueError('Cannot have deletion metadata without is_deleted=True')
        
        return values


class SecurityMixin(BaseModel):
    """Mixin pour la sécurité et chiffrement"""
    security_level: Optional[str] = Field(
        None,
        regex=r'^(public|internal|confidential|restricted|secret)$',
        description="Niveau de sécurité"
    )
    encryption_key_id: Optional[UUID] = Field(
        None,
        description="ID de la clé de chiffrement"
    )
    is_encrypted: bool = Field(
        default=False,
        description="Données chiffrées"
    )
    access_permissions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Permissions d'accès"
    )


class GeolocationMixin(BaseModel):
    """Mixin pour la géolocalisation"""
    country_code: Optional[str] = Field(
        None,
        regex=r'^[A-Z]{2}$',
        description="Code pays ISO"
    )
    region: Optional[str] = Field(
        None,
        description="Région géographique"
    )
    city: Optional[str] = Field(
        None,
        description="Ville"
    )
    timezone: str = Field(
        default="UTC",
        description="Fuseau horaire"
    )
    latitude: Optional[Decimal] = Field(
        None,
        ge=-90,
        le=90,
        description="Latitude"
    )
    longitude: Optional[Decimal] = Field(
        None,
        ge=-180,
        le=180,
        description="Longitude"
    )


class CacheableMixin(BaseModel):
    """Mixin pour la mise en cache"""
    cache_key: Optional[str] = Field(
        None,
        description="Clé de cache"
    )
    cache_ttl_seconds: Optional[int] = Field(
        None,
        ge=0,
        le=86400,  # 24h max
        description="TTL du cache en secondes"
    )
    cache_tags: List[str] = Field(
        default_factory=list,
        description="Tags de cache pour invalidation"
    )
    is_cacheable: bool = Field(
        default=True,
        description="Peut être mis en cache"
    )


class PerformanceMixin(BaseModel):
    """Mixin pour les métriques de performance"""
    processing_time_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Temps de traitement en ms"
    )
    memory_usage_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Utilisation mémoire en bytes"
    )
    cpu_usage_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Utilisation CPU en %"
    )
    network_io_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="I/O réseau en bytes"
    )
    
    @property
    def performance_score(self) -> Optional[float]:
        """Calcule un score de performance"""
        if not any([self.processing_time_ms, self.cpu_usage_percent]):
            return None
        
        score = 100.0
        if self.processing_time_ms:
            # Pénalité pour temps de traitement élevé
            score -= min(self.processing_time_ms / 10, 50)
        
        if self.cpu_usage_percent:
            # Pénalité pour CPU élevé
            score -= min(self.cpu_usage_percent / 2, 25)
        
        return max(score, 0.0)


class BaseSchema(
    TimestampMixin,
    TenantMixin,
    MetadataMixin,
    AuditMixin,
    SoftDeleteMixin,
    CacheableMixin
):
    """Schéma de base avec tous les mixins essentiels"""
    id: UUID = Field(
        default_factory=uuid4,
        description="Identifiant unique"
    )
    
    class Config:
        """Configuration Pydantic"""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            Decimal: lambda v: float(v)
        }
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "tenant_id": "spotify_tenant_1",
                "metadata": {"source": "api", "priority": "high"},
                "tags": {"environment": "production", "team": "backend"},
                "version": 1
            }
        }
    
    def dict_for_cache(self) -> Dict[str, Any]:
        """Retourne une version optimisée pour le cache"""
        return self.dict(
            exclude={
                'created_at', 'updated_at', 'cache_key', 
                'cache_ttl_seconds', 'cache_tags'
            }
        )
    
    def dict_for_api(self) -> Dict[str, Any]:
        """Retourne une version optimisée pour l'API"""
        return self.dict(
            exclude_unset=True,
            exclude_none=True,
            exclude={
                'deleted_at', 'deleted_by', 'is_deleted',
                'encryption_key_id', 'security_level'
            }
        )
    
    def generate_cache_key(self, prefix: str = "") -> str:
        """Génère une clé de cache unique"""
        base_key = f"{prefix}:{self.tenant_id}:{self.id}"
        if self.version:
            base_key += f":v{self.version}"
        return base_key
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Crée une instance depuis un dictionnaire"""
        return cls(**data)
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Crée une instance depuis une chaîne JSON"""
        import json
        return cls(**json.loads(json_str))


class BaseResponse(BaseModel):
    """Schéma de base pour les réponses API"""
    success: bool = Field(..., description="Succès de l'opération")
    message: str = Field(..., description="Message de réponse")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp de la réponse"
    )
    request_id: Optional[UUID] = Field(
        None,
        description="ID de la requête"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class BaseError(BaseModel):
    """Schéma de base pour les erreurs"""
    error_code: str = Field(..., description="Code d'erreur")
    error_message: str = Field(..., description="Message d'erreur")
    error_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Détails de l'erreur"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp de l'erreur"
    )
    correlation_id: Optional[UUID] = Field(
        None,
        description="ID de corrélation"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ValidationError(BaseError):
    """Schéma pour les erreurs de validation"""
    field_errors: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Erreurs par champ"
    )
    validation_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Contexte de validation"
    )


class BusinessError(BaseError):
    """Schéma pour les erreurs métier"""
    business_rule: str = Field(..., description="Règle métier violée")
    suggested_action: Optional[str] = Field(
        None,
        description="Action suggérée"
    )
    retry_after_seconds: Optional[int] = Field(
        None,
        ge=0,
        description="Délai avant nouvelle tentative"
    )


class SystemError(BaseError):
    """Schéma pour les erreurs système"""
    component: str = Field(..., description="Composant en erreur")
    severity: str = Field(
        ...,
        regex=r'^(low|medium|high|critical)$',
        description="Sévérité de l'erreur"
    )
    stack_trace: Optional[str] = Field(
        None,
        description="Stack trace (en dev uniquement)"
    )
