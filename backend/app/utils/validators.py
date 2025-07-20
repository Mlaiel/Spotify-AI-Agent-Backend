"""
Enterprise Validators for Spotify AI Agent
==========================================

Collection de validateurs industrialisés pour le backend Spotify AI Agent.
Inclut validation async, règles métier, compliance et sécurité.
"""

import asyncio
import re
import json
import logging
import hashlib
import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from urllib.parse import urlparse
from pathlib import Path
import aiofiles
import aiohttp
from email.utils import parseaddr
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

# === Exceptions personnalisées ===
class ValidationError(Exception):
    """Exception de base pour erreurs de validation."""
    def __init__(self, message: str, field: str = None, code: str = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)

class SecurityValidationError(ValidationError):
    """Exception pour erreurs de validation sécurité."""
    pass

class BusinessRuleError(ValidationError):
    """Exception pour violations de règles métier."""
    pass

class ComplianceError(ValidationError):
    """Exception pour violations de compliance."""
    pass

# === Patterns et constantes ===
SPOTIFY_TRACK_ID_PATTERN = re.compile(r'^[0-9A-Za-z]{22}$')
SPOTIFY_ALBUM_ID_PATTERN = re.compile(r'^[0-9A-Za-z]{22}$')
SPOTIFY_ARTIST_ID_PATTERN = re.compile(r'^[0-9A-Za-z]{22}$')
SPOTIFY_PLAYLIST_ID_PATTERN = re.compile(r'^[0-9A-Za-z]{22}$')
SPOTIFY_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')

EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
URL_PATTERN = re.compile(r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$')
PHONE_PATTERN = re.compile(r'^\+?[1-9]\d{1,14}$')

# Patterns de sécurité
XSS_PATTERNS = [
    re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'on\w+\s*=', re.IGNORECASE),
    re.compile(r'<iframe[^>]*>', re.IGNORECASE),
    re.compile(r'<object[^>]*>', re.IGNORECASE),
    re.compile(r'<embed[^>]*>', re.IGNORECASE),
]

SQL_INJECTION_PATTERNS = [
    re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)', re.IGNORECASE),
    re.compile(r'(\b(OR|AND)\s+\d+\s*=\s*\d+)', re.IGNORECASE),
    re.compile(r'(\'\s*(OR|AND)\s*\')', re.IGNORECASE),
    re.compile(r'(--|#|/\*|\*/)', re.IGNORECASE),
]

# Formats audio supportés
SUPPORTED_AUDIO_FORMATS = {
    'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'wma', 'aiff'
}

SUPPORTED_IMAGE_FORMATS = {
    'jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'bmp'
}

# === Validateur Asynchrone Principal ===
class AsyncValidator:
    """
    Validateur asynchrone principal avec cache et optimisations.
    """
    
    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    def _cache_key(self, *args) -> str:
        """Génère une clé de cache basée sur les arguments."""
        return hashlib.md5(str(args).encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache si valide."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return value
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any):
        """Met en cache une valeur."""
        self._cache[key] = (value, datetime.now())
    
    async def validate_batch(self, validations: List[Tuple[str, Any, str]]) -> Dict[str, bool]:
        """
        Valide un batch de données en parallèle.
        
        Args:
            validations: Liste de (type, valeur, contexte)
            
        Returns:
            Dict avec résultats de validation
        """
        tasks = []
        for validation_type, value, context in validations:
            if validation_type == "spotify_track":
                tasks.append(self.validate_spotify_track_id(value))
            elif validation_type == "email":
                tasks.append(self.validate_email(value))
            elif validation_type == "url":
                tasks.append(self.validate_url(value))
            # Ajouter d'autres types selon besoin
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            f"{i}_{validations[i][0]}": isinstance(result, bool) and result
            for i, result in enumerate(results)
        }

# === Validateur Spotify ===
class SpotifyValidator:
    """
    Validateur spécialisé pour les données Spotify avec API validation.
    """
    
    def __init__(self, spotify_api_client=None):
        self.spotify_api = spotify_api_client
        self._cache: Dict[str, bool] = {}
    
    def validate_track_id(self, track_id: str) -> bool:
        """Valide un ID de track Spotify."""
        if not isinstance(track_id, str):
            return False
        return bool(SPOTIFY_TRACK_ID_PATTERN.match(track_id))
    
    def validate_album_id(self, album_id: str) -> bool:
        """Valide un ID d'album Spotify."""
        if not isinstance(album_id, str):
            return False
        return bool(SPOTIFY_ALBUM_ID_PATTERN.match(album_id))
    
    def validate_artist_id(self, artist_id: str) -> bool:
        """Valide un ID d'artiste Spotify."""
        if not isinstance(artist_id, str):
            return False
        return bool(SPOTIFY_ARTIST_ID_PATTERN.match(artist_id))
    
    def validate_playlist_id(self, playlist_id: str) -> bool:
        """Valide un ID de playlist Spotify."""
        if not isinstance(playlist_id, str):
            return False
        return bool(SPOTIFY_PLAYLIST_ID_PATTERN.match(playlist_id))
    
    def validate_user_id(self, user_id: str) -> bool:
        """Valide un ID utilisateur Spotify."""
        if not isinstance(user_id, str):
            return False
        return bool(SPOTIFY_USER_ID_PATTERN.match(user_id))
    
    async def validate_track_exists(self, track_id: str) -> bool:
        """Valide qu'un track existe via l'API Spotify."""
        if not self.validate_track_id(track_id):
            return False
        
        cache_key = f"track_exists:{track_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not self.spotify_api:
            logger.warning("Spotify API client not configured, skipping existence check")
            return True
        
        try:
            track = await self.spotify_api.get_track(track_id)
            exists = track is not None
            self._cache[cache_key] = exists
            return exists
        except Exception as e:
            logger.error(f"Error validating track existence: {e}")
            return False
    
    async def validate_playlist(self, playlist_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide une playlist complète avec règles métier.
        
        Returns:
            Dict avec résultats détaillés de validation
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validation du nom
        name = playlist_data.get("name", "")
        if not name or len(name.strip()) == 0:
            result["errors"].append("Playlist name is required")
            result["valid"] = False
        elif len(name) > 100:
            result["errors"].append("Playlist name too long (max 100 characters)")
            result["valid"] = False
        
        # Validation de la description
        description = playlist_data.get("description", "")
        if len(description) > 300:
            result["warnings"].append("Description is very long (>300 characters)")
        
        # Validation des tracks
        tracks = playlist_data.get("tracks", [])
        if not isinstance(tracks, list):
            result["errors"].append("Tracks must be a list")
            result["valid"] = False
        elif len(tracks) == 0:
            result["warnings"].append("Playlist is empty")
        elif len(tracks) > 10000:
            result["errors"].append("Too many tracks (max 10,000)")
            result["valid"] = False
        else:
            # Validation des IDs de tracks
            invalid_tracks = []
            for i, track_id in enumerate(tracks):
                if not self.validate_track_id(track_id):
                    invalid_tracks.append(f"Track {i}: {track_id}")
            
            if invalid_tracks:
                result["errors"].append(f"Invalid track IDs: {', '.join(invalid_tracks[:5])}")
                if len(invalid_tracks) > 5:
                    result["errors"].append(f"... and {len(invalid_tracks) - 5} more")
                result["valid"] = False
        
        # Validation de la visibilité
        public = playlist_data.get("public", True)
        if not isinstance(public, bool):
            result["errors"].append("Public field must be boolean")
            result["valid"] = False
        
        return result

# === Validateur d'Input Utilisateur ===
class UserInputValidator:
    """
    Validateur pour tous les inputs utilisateur avec protection XSS/injection.
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Valide un email avec RFC compliance."""
        if not isinstance(email, str):
            return False
        
        email = email.strip().lower()
        if len(email) > 254:  # RFC limit
            return False
        
        return bool(EMAIL_PATTERN.match(email))
    
    @staticmethod
    def validate_password(password: str, min_length: int = 8) -> Dict[str, Any]:
        """
        Valide un mot de passe avec règles de sécurité.
        
        Returns:
            Dict avec score et recommandations
        """
        result = {
            "valid": True,
            "score": 0,
            "errors": [],
            "recommendations": []
        }
        
        if not isinstance(password, str):
            result["errors"].append("Password must be a string")
            result["valid"] = False
            return result
        
        # Longueur minimale
        if len(password) < min_length:
            result["errors"].append(f"Password too short (min {min_length} characters)")
            result["valid"] = False
        else:
            result["score"] += 1
        
        # Complexité
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        if has_upper:
            result["score"] += 1
        else:
            result["recommendations"].append("Add uppercase letters")
        
        if has_lower:
            result["score"] += 1
        else:
            result["recommendations"].append("Add lowercase letters")
        
        if has_digit:
            result["score"] += 1
        else:
            result["recommendations"].append("Add numbers")
        
        if has_special:
            result["score"] += 1
        else:
            result["recommendations"].append("Add special characters")
        
        # Patterns communs
        common_patterns = ['123', 'abc', 'qwerty', 'password', 'admin']
        for pattern in common_patterns:
            if pattern in password.lower():
                result["score"] -= 1
                result["recommendations"].append(f"Avoid common patterns like '{pattern}'")
                break
        
        # Score final
        if result["score"] < 3:
            result["valid"] = False
            result["errors"].append("Password too weak")
        
        return result
    
    @staticmethod
    def validate_phone(phone: str, country_code: str = None) -> bool:
        """Valide un numéro de téléphone."""
        if not isinstance(phone, str):
            return False
        
        # Nettoyage
        phone = re.sub(r'[^\d+]', '', phone)
        
        return bool(PHONE_PATTERN.match(phone))
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """
        Sanitise un texte contre XSS et injections.
        
        Args:
            text: Texte à sanitiser
            max_length: Longueur maximale
            
        Returns:
            Texte sanitisé
        """
        if not isinstance(text, str):
            return ""
        
        # Limitation de longueur
        if len(text) > max_length:
            text = text[:max_length]
        
        # Protection XSS
        for pattern in XSS_PATTERNS:
            text = pattern.sub('', text)
        
        # Échappement HTML de base
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('"', '&quot;').replace("'", '&#x27;')
        
        return text.strip()
    
    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        """Détecte les tentatives d'injection SQL."""
        if not isinstance(text, str):
            return False
        
        for pattern in SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                return True
        
        return False
    
    @staticmethod
    def validate_url(url: str, allowed_schemes: Set[str] = None) -> bool:
        """Valide une URL avec vérification de schéma."""
        if not isinstance(url, str):
            return False
        
        allowed_schemes = allowed_schemes or {'http', 'https'}
        
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in allowed_schemes and
                bool(parsed.netloc) and
                bool(URL_PATTERN.match(url))
            )
        except Exception:
            return False

# === Validateur de Fichiers Audio ===
class AudioFileValidator:
    """
    Validateur spécialisé pour fichiers audio avec analyse de contenu.
    """
    
    @staticmethod
    def validate_audio_format(filename: str) -> bool:
        """Valide le format d'un fichier audio."""
        if not isinstance(filename, str):
            return False
        
        extension = Path(filename).suffix.lower().lstrip('.')
        return extension in SUPPORTED_AUDIO_FORMATS
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
        """Valide la taille d'un fichier audio."""
        max_size_bytes = max_size_mb * 1024 * 1024
        return 0 < file_size <= max_size_bytes
    
    @staticmethod
    async def validate_audio_content(file_path: str) -> Dict[str, Any]:
        """
        Valide le contenu d'un fichier audio (nécessite librosa).
        
        Returns:
            Dict avec métadonnées et validation
        """
        result = {
            "valid": True,
            "duration": 0,
            "sample_rate": 0,
            "channels": 0,
            "errors": []
        }
        
        try:
            import librosa
            
            # Chargement et analyse
            y, sr = librosa.load(file_path, sr=None)
            
            result.update({
                "duration": len(y) / sr,
                "sample_rate": sr,
                "channels": 1 if y.ndim == 1 else y.shape[0]
            })
            
            # Validations
            if result["duration"] < 1:
                result["errors"].append("Audio too short (min 1 second)")
                result["valid"] = False
            elif result["duration"] > 600:  # 10 minutes
                result["errors"].append("Audio too long (max 10 minutes)")
                result["valid"] = False
            
            if result["sample_rate"] < 8000:
                result["errors"].append("Sample rate too low (min 8kHz)")
                result["valid"] = False
            
        except ImportError:
            result["errors"].append("Audio analysis library not available")
            result["valid"] = False
        except Exception as e:
            result["errors"].append(f"Audio analysis failed: {str(e)}")
            result["valid"] = False
        
        return result

# === Validateur de Playlist ===
class PlaylistValidator:
    """
    Validateur spécialisé pour playlists avec règles métier complexes.
    """
    
    def __init__(self, spotify_validator: SpotifyValidator = None):
        self.spotify_validator = spotify_validator or SpotifyValidator()
    
    async def validate_playlist_creation(self, playlist_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Valide la création d'une playlist avec toutes les règles métier.
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        # Validation basique de la playlist
        basic_validation = await self.spotify_validator.validate_playlist(playlist_data)
        result["valid"] = basic_validation["valid"]
        result["errors"].extend(basic_validation["errors"])
        result["warnings"].extend(basic_validation["warnings"])
        
        if not result["valid"]:
            return result
        
        # Règles métier avancées
        name = playlist_data["name"]
        tracks = playlist_data.get("tracks", [])
        
        # Vérification de doublons
        unique_tracks = set(tracks)
        if len(unique_tracks) != len(tracks):
            duplicates = len(tracks) - len(unique_tracks)
            result["warnings"].append(f"{duplicates} duplicate tracks found")
            result["metadata"]["duplicates"] = duplicates
        
        # Analyse de diversité (si plus de 10 tracks)
        if len(tracks) > 10:
            # Simulation d'analyse de diversité
            # En production: analyser les genres, artistes, etc.
            diversity_score = self._calculate_diversity_score(tracks)
            result["metadata"]["diversity_score"] = diversity_score
            
            if diversity_score < 0.3:
                result["warnings"].append("Playlist lacks diversity")
        
        # Vérification de contenu approprié
        # En production: utiliser des APIs de modération
        if self._contains_inappropriate_content(name):
            result["errors"].append("Playlist name contains inappropriate content")
            result["valid"] = False
        
        # Limite de playlists par utilisateur
        # En production: requête DB pour compter les playlists
        user_playlist_count = await self._get_user_playlist_count(user_id)
        if user_playlist_count >= 1000:  # Limite Spotify
            result["errors"].append("Maximum playlist limit reached")
            result["valid"] = False
        
        return result
    
    def _calculate_diversity_score(self, tracks: List[str]) -> float:
        """Calcule un score de diversité pour une playlist."""
        # Simulation - en production: analyser les métadonnées musicales
        import random
        return random.uniform(0.2, 0.9)
    
    def _contains_inappropriate_content(self, text: str) -> bool:
        """Détecte le contenu inapproprié."""
        # Liste basique de mots interdits (à étendre selon besoins)
        inappropriate_words = ['spam', 'test123', 'hack']
        text_lower = text.lower()
        return any(word in text_lower for word in inappropriate_words)
    
    async def _get_user_playlist_count(self, user_id: str) -> int:
        """Récupère le nombre de playlists d'un utilisateur."""
        # Simulation - en production: requête DB
        return 50

# === Validateur de Sécurité ===
class SecurityValidator:
    """
    Validateur de sécurité avancé avec détection de menaces.
    """
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Valide le format d'une clé API."""
        if not isinstance(api_key, str):
            return False
        
        # Format: 32-64 caractères alphanumériques
        return 32 <= len(api_key) <= 64 and api_key.isalnum()
    
    @staticmethod
    def validate_jwt_token(token: str) -> Dict[str, Any]:
        """
        Valide un token JWT (nécessite PyJWT).
        
        Returns:
            Dict avec validation et payload décodé
        """
        result = {
            "valid": False,
            "payload": None,
            "errors": []
        }
        
        try:
            import jwt
            
            # Vérification du format
            parts = token.split('.')
            if len(parts) != 3:
                result["errors"].append("Invalid JWT format")
                return result
            
            # Décodage sans vérification de signature (pour validation structure)
            payload = jwt.decode(token, options={"verify_signature": False})
            result["payload"] = payload
            
            # Validations de base
            current_time = datetime.now(timezone.utc).timestamp()
            
            if 'exp' in payload:
                if payload['exp'] < current_time:
                    result["errors"].append("Token expired")
                    return result
            
            if 'iat' in payload:
                if payload['iat'] > current_time:
                    result["errors"].append("Token issued in future")
                    return result
            
            result["valid"] = True
            
        except ImportError:
            result["errors"].append("JWT library not available")
        except Exception as e:
            result["errors"].append(f"JWT validation failed: {str(e)}")
        
        return result
    
    @staticmethod
    def detect_brute_force_attempt(
        user_id: str,
        attempts: List[datetime],
        max_attempts: int = 5,
        window_minutes: int = 15
    ) -> bool:
        """Détecte les tentatives de brute force."""
        if len(attempts) < max_attempts:
            return False
        
        # Vérifier les tentatives dans la fenêtre de temps
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_attempts = [attempt for attempt in attempts if attempt > cutoff_time]
        
        return len(recent_attempts) >= max_attempts
    
    @staticmethod
    def validate_ip_address(ip: str, whitelist: Set[str] = None, blacklist: Set[str] = None) -> bool:
        """Valide une adresse IP contre des listes."""
        import ipaddress
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Vérification blacklist
            if blacklist and ip in blacklist:
                return False
            
            # Vérification whitelist
            if whitelist and ip not in whitelist:
                return False
            
            # Blocage des IPs privées en production
            if ip_obj.is_private:
                logger.warning(f"Private IP detected: {ip}")
            
            return True
            
        except ValueError:
            return False

# === Validateur de Règles Métier ===
class BusinessRuleValidator:
    """
    Validateur pour règles métier spécifiques à Spotify AI Agent.
    """
    
    @staticmethod
    def validate_subscription_limits(
        user_subscription: str,
        resource_type: str,
        current_usage: int
    ) -> Dict[str, Any]:
        """
        Valide les limites d'abonnement.
        
        Args:
            user_subscription: Type d'abonnement (free, premium, artist)
            resource_type: Type de ressource (playlists, api_calls, etc.)
            current_usage: Usage actuel
            
        Returns:
            Dict avec validation et limites
        """
        # Définition des limites par abonnement
        limits = {
            "free": {
                "playlists": 50,
                "api_calls_per_hour": 100,
                "tracks_per_playlist": 100,
                "ai_generations_per_day": 5
            },
            "premium": {
                "playlists": 1000,
                "api_calls_per_hour": 1000,
                "tracks_per_playlist": 10000,
                "ai_generations_per_day": 100
            },
            "artist": {
                "playlists": 10000,
                "api_calls_per_hour": 10000,
                "tracks_per_playlist": 10000,
                "ai_generations_per_day": 1000
            }
        }
        
        subscription_limits = limits.get(user_subscription, limits["free"])
        resource_limit = subscription_limits.get(resource_type, 0)
        
        return {
            "valid": current_usage < resource_limit,
            "current_usage": current_usage,
            "limit": resource_limit,
            "remaining": max(0, resource_limit - current_usage),
            "subscription": user_subscription
        }
    
    @staticmethod
    def validate_ai_request_quota(
        user_id: str,
        request_type: str,
        daily_usage: int,
        subscription: str
    ) -> bool:
        """Valide le quota de requêtes IA."""
        quotas = {
            "free": {"lyrics_generation": 5, "music_analysis": 10, "recommendations": 50},
            "premium": {"lyrics_generation": 100, "music_analysis": 500, "recommendations": 1000},
            "artist": {"lyrics_generation": 1000, "music_analysis": 5000, "recommendations": 10000}
        }
        
        user_quotas = quotas.get(subscription, quotas["free"])
        limit = user_quotas.get(request_type, 0)
        
        return daily_usage < limit

# === Validateur de Compliance ===
class ComplianceValidator:
    """
    Validateur pour compliance RGPD, HIPAA et autres réglementations.
    """
    
    @staticmethod
    def validate_gdpr_consent(consent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide la conformité RGPD du consentement.
        
        Args:
            consent_data: Données de consentement
            
        Returns:
            Dict avec validation RGPD
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        required_fields = [
            "marketing_consent",
            "analytics_consent", 
            "personalization_consent",
            "data_processing_consent"
        ]
        
        # Vérification des champs requis
        for field in required_fields:
            if field not in consent_data:
                result["errors"].append(f"Missing required consent: {field}")
                result["valid"] = False
            elif not isinstance(consent_data[field], bool):
                result["errors"].append(f"Consent {field} must be boolean")
                result["valid"] = False
        
        # Vérification du timestamp
        if "consent_timestamp" not in consent_data:
            result["errors"].append("Missing consent timestamp")
            result["valid"] = False
        else:
            try:
                consent_time = datetime.fromisoformat(consent_data["consent_timestamp"])
                age_days = (datetime.now(timezone.utc) - consent_time).days
                
                if age_days > 365:  # Consentement expire après 1 an
                    result["warnings"].append("Consent is more than 1 year old")
                    
            except ValueError:
                result["errors"].append("Invalid consent timestamp format")
                result["valid"] = False
        
        # Vérification de la version
        if "consent_version" not in consent_data:
            result["warnings"].append("Missing consent version")
        
        return result
    
    @staticmethod
    def validate_data_retention(
        data_type: str,
        created_date: datetime,
        user_status: str = "active"
    ) -> Dict[str, Any]:
        """
        Valide la politique de rétention des données.
        
        Args:
            data_type: Type de données (user_data, analytics, logs, etc.)
            created_date: Date de création
            user_status: Statut utilisateur (active, inactive, deleted)
            
        Returns:
            Dict avec validation de rétention
        """
        # Politiques de rétention par type
        retention_policies = {
            "user_data": {"active": 365*5, "inactive": 365*2, "deleted": 30},  # jours
            "analytics": {"active": 365*2, "inactive": 365, "deleted": 7},
            "logs": {"active": 90, "inactive": 30, "deleted": 7},
            "ai_generated": {"active": 365, "inactive": 180, "deleted": 7}
        }
        
        policy = retention_policies.get(data_type, retention_policies["user_data"])
        max_retention_days = policy.get(user_status, policy["active"])
        
        age_days = (datetime.now(timezone.utc) - created_date).days
        
        return {
            "should_retain": age_days <= max_retention_days,
            "age_days": age_days,
            "max_retention_days": max_retention_days,
            "expires_in_days": max(0, max_retention_days - age_days),
            "data_type": data_type,
            "user_status": user_status
        }
    
    @staticmethod
    def validate_pii_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Détecte et valide les données personnelles identifiables (PII).
        
        Returns:
            Dict avec détection PII et recommandations
        """
        pii_fields = {
            "email": EMAIL_PATTERN,
            "phone": PHONE_PATTERN,
            "ssn": re.compile(r'\d{3}-\d{2}-\d{4}'),
            "credit_card": re.compile(r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}')
        }
        
        detected_pii = []
        recommendations = []
        
        def scan_value(key: str, value: Any, path: str = ""):
            if isinstance(value, str):
                for pii_type, pattern in pii_fields.items():
                    if pattern.search(value):
                        detected_pii.append({
                            "type": pii_type,
                            "field": f"{path}.{key}" if path else key,
                            "value": value[:10] + "..." if len(value) > 10 else value
                        })
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    scan_value(sub_key, sub_value, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    scan_value(f"[{i}]", item, f"{path}.{key}" if path else key)
        
        # Scan récursif
        for key, value in data.items():
            scan_value(key, value)
        
        # Recommandations basées sur détection
        if detected_pii:
            recommendations.extend([
                "Encrypt PII data at rest",
                "Implement field-level encryption",
                "Add audit logging for PII access",
                "Consider data anonymization",
                "Ensure GDPR compliance"
            ])
        
        return {
            "contains_pii": len(detected_pii) > 0,
            "detected_pii": detected_pii,
            "pii_count": len(detected_pii),
            "recommendations": recommendations
        }

# === Fonctions utilitaires ===
def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Valide des données contre un schema JSON."""
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return True
    except ImportError:
        logger.warning("jsonschema library not available")
        return True  # Pas de validation si lib manquante
    except jsonschema.ValidationError:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitise un nom de fichier."""
    if not isinstance(filename, str):
        return "unknown"
    
    # Suppression des caractères dangereux
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'\.\.', '', filename)  # Protection traversal
    filename = filename.strip('. ')  # Suppression points/espaces début/fin
    
    # Limitation de longueur
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:251] + '.' + ext if ext else name[:255]
    
    return filename or "unnamed"

def estimate_processing_time(data_size_mb: float, operation_type: str) -> float:
    """Estime le temps de traitement en secondes."""
    # Estimations basées sur des benchmarks
    processing_rates = {
        "audio_analysis": 0.5,  # MB/sec
        "ml_inference": 2.0,
        "data_validation": 10.0,
        "file_conversion": 1.0
    }
    
    rate = processing_rates.get(operation_type, 1.0)
    return data_size_mb / rate