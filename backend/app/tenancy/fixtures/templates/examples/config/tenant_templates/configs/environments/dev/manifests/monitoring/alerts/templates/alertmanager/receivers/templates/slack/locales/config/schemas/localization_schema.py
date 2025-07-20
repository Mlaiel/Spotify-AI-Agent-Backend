"""
Schémas de localisation - Module Python.

Ce module fournit les classes de validation pour la configuration
de localisation, internationalisation et traductions.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime


class TextDirection(str, Enum):
    """Direction d'écriture."""
    LTR = "ltr"  # Left to Right
    RTL = "rtl"  # Right to Left


class TranslatorRole(str, Enum):
    """Rôles de traducteurs."""
    TRANSLATOR = "translator"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"


class DetectionMethod(str, Enum):
    """Méthodes de détection de locale."""
    BROWSER = "browser"
    IP = "ip"
    USER_PREFERENCE = "user_preference"
    ACCEPT_LANGUAGE = "accept_language"


class FallbackStrategy(str, Enum):
    """Stratégies de fallback."""
    DEFAULT = "default"
    BROWSER = "browser"
    CLOSEST_MATCH = "closest_match"


class CurrencyPosition(str, Enum):
    """Position du symbole de devise."""
    BEFORE = "before"
    AFTER = "after"


class Translator(BaseModel):
    """Informations sur un traducteur."""
    name: str
    email: str
    role: TranslatorRole


class TranslationMetadata(BaseModel):
    """Métadonnées de traduction."""
    name: str
    english_name: str
    direction: TextDirection = TextDirection.LTR
    completion: float = Field(0.0, ge=0, le=100)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    translators: List[Translator] = Field(default_factory=list)


class CommonMessages(BaseModel):
    """Messages communs."""
    yes: str = "Yes"
    no: str = "No"
    ok: str = "OK"
    cancel: str = "Cancel"
    save: str = "Save"
    delete: str = "Delete"
    edit: str = "Edit"
    create: str = "Create"
    update: str = "Update"
    loading: str = "Loading..."
    error: str = "Error"
    success: str = "Success"
    warning: str = "Warning"
    info: str = "Information"


class NavigationMessages(BaseModel):
    """Messages de navigation."""
    home: str = "Home"
    dashboard: str = "Dashboard"
    profile: str = "Profile"
    settings: str = "Settings"
    help: str = "Help"
    logout: str = "Logout"
    back: str = "Back"
    next: str = "Next"
    previous: str = "Previous"


class AuthenticationMessages(BaseModel):
    """Messages d'authentification."""
    login: str = "Login"
    logout: str = "Logout"
    register: str = "Register"
    forgot_password: str = "Forgot Password"
    reset_password: str = "Reset Password"
    username: str = "Username"
    password: str = "Password"
    email: str = "Email"
    confirm_password: str = "Confirm Password"
    remember_me: str = "Remember me"
    invalid_credentials: str = "Invalid credentials"
    account_locked: str = "Account locked"
    password_too_weak: str = "Password too weak"


class SpotifyMessages(BaseModel):
    """Messages liés à Spotify."""
    connect: str = "Connect"
    disconnect: str = "Disconnect"
    playlist: str = "Playlist"
    track: str = "Track"
    artist: str = "Artist"
    album: str = "Album"
    play: str = "Play"
    pause: str = "Pause"
    skip: str = "Skip"
    previous: str = "Previous"
    shuffle: str = "Shuffle"
    repeat: str = "Repeat"
    volume: str = "Volume"
    search: str = "Search"
    recommendations: str = "Recommendations"
    top_tracks: str = "Top Tracks"
    recently_played: str = "Recently Played"


class AIMessages(BaseModel):
    """Messages liés à l'IA."""
    chat: str = "Chat"
    analyze: str = "Analyze"
    generate: str = "Generate"
    processing: str = "Processing..."
    thinking: str = "Thinking..."
    suggestion: str = "Suggestion"
    confidence: str = "Confidence"
    model: str = "Model"
    prompt: str = "Prompt"
    response: str = "Response"
    no_results: str = "No results found"
    rate_limit: str = "Rate limit exceeded"


class CollaborationMessages(BaseModel):
    """Messages de collaboration."""
    share: str = "Share"
    invite: str = "Invite"
    collaborate: str = "Collaborate"
    permissions: str = "Permissions"
    owner: str = "Owner"
    editor: str = "Editor"
    viewer: str = "Viewer"
    guest: str = "Guest"
    online: str = "Online"
    offline: str = "Offline"
    typing: str = "Typing..."


class ErrorMessages(BaseModel):
    """Messages d'erreur."""
    network_error: str = "Network error"
    server_error: str = "Server error"
    validation_error: str = "Validation error"
    permission_denied: str = "Permission denied"
    not_found: str = "Not found"
    timeout: str = "Timeout"
    unknown_error: str = "Unknown error"
    maintenance: str = "System under maintenance"


class NotificationMessages(BaseModel):
    """Messages de notification."""
    new_message: str = "New message"
    invitation_received: str = "Invitation received"
    file_shared: str = "File shared"
    task_completed: str = "Task completed"
    system_update: str = "System update"
    security_alert: str = "Security alert"
    mark_as_read: str = "Mark as read"
    clear_all: str = "Clear all"


class Messages(BaseModel):
    """Collection de tous les messages."""
    common: CommonMessages = Field(default_factory=CommonMessages)
    navigation: NavigationMessages = Field(default_factory=NavigationMessages)
    authentication: AuthenticationMessages = Field(default_factory=AuthenticationMessages)
    spotify: SpotifyMessages = Field(default_factory=SpotifyMessages)
    ai: AIMessages = Field(default_factory=AIMessages)
    collaboration: CollaborationMessages = Field(default_factory=CollaborationMessages)
    errors: ErrorMessages = Field(default_factory=ErrorMessages)
    notifications: NotificationMessages = Field(default_factory=NotificationMessages)


class DateFormats(BaseModel):
    """Formats de date."""
    short: str = "DD/MM/YYYY"
    medium: str = "DD MMM YYYY"
    long: str = "DD MMMM YYYY"
    full: str = "DDDD DD MMMM YYYY"


class TimeFormats(BaseModel):
    """Formats d'heure."""
    short: str = "HH:mm"
    medium: str = "HH:mm:ss"
    long: str = "HH:mm:ss Z"


class NumberFormats(BaseModel):
    """Formats de nombre."""
    decimal_separator: str = "."
    thousands_separator: str = ","
    currency_symbol: str = "$"
    currency_position: CurrencyPosition = CurrencyPosition.BEFORE


class Formats(BaseModel):
    """Formats de localisation."""
    date: DateFormats = Field(default_factory=DateFormats)
    time: TimeFormats = Field(default_factory=TimeFormats)
    number: NumberFormats = Field(default_factory=NumberFormats)


class PluralForm(BaseModel):
    """Forme plurielle."""
    zero: Optional[str] = None
    one: Optional[str] = None
    two: Optional[str] = None
    few: Optional[str] = None
    many: Optional[str] = None
    other: str


class PluralizationConfig(BaseModel):
    """Configuration de pluralisation."""
    rules: Dict[str, Any] = Field(default_factory=dict)
    examples: Dict[str, PluralForm] = Field(default_factory=dict)


class Translation(BaseModel):
    """Traduction complète pour une locale."""
    metadata: TranslationMetadata
    messages: Messages
    formats: Formats
    pluralization: PluralizationConfig = Field(default_factory=PluralizationConfig)


class LegalCompliance(BaseModel):
    """Conformité légale par région."""
    gdpr_applicable: bool = False
    ccpa_applicable: bool = False
    data_retention_days: int = 365
    cookie_consent_required: bool = False
    age_verification_required: bool = False
    minimum_age: int = 13


class RegionalConfig(BaseModel):
    """Configuration régionale."""
    timezone_mapping: Dict[str, List[str]] = Field(default_factory=dict)
    currency_mapping: Dict[str, str] = Field(default_factory=dict)
    legal_compliance: Dict[str, LegalCompliance] = Field(default_factory=dict)


class AutomaticDetection(BaseModel):
    """Configuration de détection automatique."""
    enabled: bool = True
    methods: List[DetectionMethod] = Field(
        default_factory=lambda: [DetectionMethod.BROWSER, DetectionMethod.ACCEPT_LANGUAGE]
    )
    fallback_strategy: FallbackStrategy = FallbackStrategy.DEFAULT


class DynamicLoading(BaseModel):
    """Configuration de chargement dynamique."""
    enabled: bool = True
    cache_duration: int = Field(3600, ge=300)  # secondes
    lazy_loading: bool = True


class RTLSupport(BaseModel):
    """Support RTL."""
    enabled: bool = True
    css_class: str = "rtl"
    mirror_layout: bool = True


class Interpolation(BaseModel):
    """Configuration d'interpolation."""
    enabled: bool = True
    escape_html: bool = True
    allowed_tags: List[str] = Field(default_factory=lambda: ["b", "i", "em", "strong"])


class LocalizationFeatures(BaseModel):
    """Fonctionnalités de localisation."""
    automatic_detection: AutomaticDetection = Field(default_factory=AutomaticDetection)
    dynamic_loading: DynamicLoading = Field(default_factory=DynamicLoading)
    rtl_support: RTLSupport = Field(default_factory=RTLSupport)
    interpolation: Interpolation = Field(default_factory=Interpolation)


class ConsistencyChecks(BaseModel):
    """Vérifications de cohérence."""
    check_placeholders: bool = True
    check_html_tags: bool = True
    check_special_chars: bool = True


class ValidationConfig(BaseModel):
    """Configuration de validation."""
    required_keys: List[str] = Field(default_factory=list)
    max_length: Dict[str, int] = Field(default_factory=dict)
    forbidden_patterns: List[str] = Field(default_factory=list)
    consistency_checks: ConsistencyChecks = Field(default_factory=ConsistencyChecks)


class LocalizationSchema(BaseModel):
    """Schéma complet de localisation."""
    default_locale: str = Field(..., regex=r"^[a-z]{2}(-[A-Z]{2})?$")
    supported_locales: List[str] = Field(..., min_items=1)
    fallback_locale: str = Field("en", regex=r"^[a-z]{2}(-[A-Z]{2})?$")
    translations: Dict[str, Translation] = Field(default_factory=dict)
    regions: RegionalConfig = Field(default_factory=RegionalConfig)
    features: LocalizationFeatures = Field(default_factory=LocalizationFeatures)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @validator('supported_locales')
    def validate_supported_locales(cls, v):
        """Valide le format des locales supportées."""
        import re
        pattern = re.compile(r"^[a-z]{2}(-[A-Z]{2})?$")
        for locale in v:
            if not pattern.match(locale):
                raise ValueError(f"Format de locale invalide: {locale}")
        return v

    @validator('fallback_locale')
    def validate_fallback_locale(cls, v, values):
        """Valide que la locale de fallback est dans les locales supportées."""
        if 'supported_locales' in values and v not in values['supported_locales']:
            raise ValueError("La locale de fallback doit être dans les locales supportées")
        return v

    @validator('translations')
    def validate_translations(cls, v, values):
        """Valide que chaque locale supportée a ses traductions."""
        if 'supported_locales' in values:
            missing_locales = set(values['supported_locales']) - set(v.keys())
            if missing_locales:
                raise ValueError(f"Traductions manquantes pour: {missing_locales}")
        return v

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
        validate_assignment = True


# Créateur de traductions par défaut
def create_default_translation(locale: str, name: str, english_name: str) -> Translation:
    """Crée une traduction par défaut pour une locale."""
    return Translation(
        metadata=TranslationMetadata(
            name=name,
            english_name=english_name,
            direction=TextDirection.RTL if locale in ['ar', 'he', 'fa'] else TextDirection.LTR,
            completion=0.0
        ),
        messages=Messages(),
        formats=Formats()
    )


# Locales par défaut supportées
DEFAULT_LOCALES = {
    'en': ('English', 'English'),
    'fr': ('Français', 'French'),
    'de': ('Deutsch', 'German'),
    'es': ('Español', 'Spanish'),
    'it': ('Italiano', 'Italian'),
    'pt': ('Português', 'Portuguese'),
    'ru': ('Русский', 'Russian'),
    'zh': ('中文', 'Chinese'),
    'ja': ('日本語', 'Japanese'),
    'ko': ('한국어', 'Korean'),
    'ar': ('العربية', 'Arabic'),
    'he': ('עברית', 'Hebrew')
}
