"""
Utilitaires - Spotify AI Agent
Fonctions utilitaires et helpers pour transformation et manipulation de données
"""

import hashlib
import hmac
import secrets
import string
import unicodedata
import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, TypeVar, Generic
from uuid import UUID, uuid4
import json
import base64
import gzip
import zlib
from urllib.parse import urlparse, parse_qs, urlencode
from pathlib import Path
import mimetypes
import time
import functools
import asyncio
from collections import defaultdict, OrderedDict
import logging

try:
    import phonenumbers
    HAS_PHONENUMBERS = True
except ImportError:
    HAS_PHONENUMBERS = False

from .constants import (
    DEFAULT_ENCODING, VALIDATION_PATTERNS, COMPILED_PATTERNS,
    MAX_STRING_LENGTH, DEFAULT_TIMEZONE
)

T = TypeVar('T')


# =============================================================================
# UTILITAIRES DE CHAÎNES
# =============================================================================

def normalize_string(text: str, 
                    remove_accents: bool = True,
                    lowercase: bool = True,
                    remove_punctuation: bool = False,
                    remove_extra_spaces: bool = True) -> str:
    """
    Normalise une chaîne de caractères selon différents critères
    
    Args:
        text: Texte à normaliser
        remove_accents: Supprimer les accents
        lowercase: Convertir en minuscules
        remove_punctuation: Supprimer la ponctuation
        remove_extra_spaces: Supprimer les espaces supplémentaires
    
    Returns:
        Texte normalisé
    """
    if not text:
        return ""
    
    # Normaliser Unicode
    if remove_accents:
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Convertir en minuscules
    if lowercase:
        text = text.lower()
    
    # Supprimer la ponctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Supprimer les espaces supplémentaires
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def slugify(text: str, max_length: int = 50, separator: str = '-') -> str:
    """
    Convertit un texte en slug URL-friendly
    
    Args:
        text: Texte à convertir
        max_length: Longueur maximale du slug
        separator: Séparateur à utiliser
    
    Returns:
        Slug généré
    """
    # Normaliser le texte
    text = normalize_string(text, remove_punctuation=True)
    
    # Remplacer les espaces par le séparateur
    text = re.sub(r'\s+', separator, text)
    
    # Supprimer les caractères non alphanumériques
    text = re.sub(f'[^a-z0-9{re.escape(separator)}]', '', text)
    
    # Supprimer les séparateurs multiples
    text = re.sub(f'{re.escape(separator)}+', separator, text)
    
    # Supprimer les séparateurs en début/fin
    text = text.strip(separator)
    
    # Tronquer si nécessaire
    if len(text) > max_length:
        text = text[:max_length].rstrip(separator)
    
    return text


def truncate_string(text: str, 
                   max_length: int,
                   suffix: str = '...',
                   word_boundary: bool = True) -> str:
    """
    Tronque une chaîne en respectant optionnellement les mots
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter si tronqué
        word_boundary: Respecter les limites de mots
    
    Returns:
        Texte tronqué
    """
    if len(text) <= max_length:
        return text
    
    if word_boundary:
        # Trouver le dernier espace avant la limite
        truncated = text[:max_length - len(suffix)]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + suffix
    else:
        return text[:max_length - len(suffix)] + suffix


def mask_sensitive_data(text: str, mask_char: str = '*', reveal_chars: int = 4) -> str:
    """
    Masque les données sensibles en révélant seulement les derniers caractères
    
    Args:
        text: Texte à masquer
        mask_char: Caractère de masquage
        reveal_chars: Nombre de caractères à révéler à la fin
    
    Returns:
        Texte masqué
    """
    if len(text) <= reveal_chars:
        return mask_char * len(text)
    
    masked_length = len(text) - reveal_chars
    return mask_char * masked_length + text[-reveal_chars:]


def extract_mentions(text: str, prefix: str = '@') -> List[str]:
    """
    Extrait les mentions d'un texte (ex: @username)
    
    Args:
        text: Texte à analyser
        prefix: Préfixe des mentions
    
    Returns:
        Liste des mentions trouvées
    """
    pattern = rf'{re.escape(prefix)}([a-zA-Z0-9_]+)'
    return re.findall(pattern, text)


def extract_hashtags(text: str) -> List[str]:
    """
    Extrait les hashtags d'un texte
    
    Args:
        text: Texte à analyser
    
    Returns:
        Liste des hashtags trouvés
    """
    pattern = r'#([a-zA-Z0-9_]+)'
    return re.findall(pattern, text)


# =============================================================================
# UTILITAIRES DE VALIDATION
# =============================================================================

def is_valid_email(email: str) -> bool:
    """Valide une adresse email"""
    return COMPILED_PATTERNS['EMAIL'].match(email) is not None


def is_valid_phone(phone: str, region: str = None) -> bool:
    """
    Valide un numéro de téléphone
    
    Args:
        phone: Numéro à valider
        region: Code région (ex: 'FR', 'US')
    
    Returns:
        True si valide
    """
    if not HAS_PHONENUMBERS:
        # Validation basique si phonenumbers n'est pas disponible
        return COMPILED_PATTERNS['PHONE'].match(phone) is not None
    
    try:
        parsed = phonenumbers.parse(phone, region)
        return phonenumbers.is_valid_number(parsed)
    except:
        return False


def is_valid_url(url: str, schemes: List[str] = None) -> bool:
    """
    Valide une URL
    
    Args:
        url: URL à valider
        schemes: Schémas autorisés (par défaut: http, https)
    
    Returns:
        True si valide
    """
    if not COMPILED_PATTERNS['URL'].match(url):
        return False
    
    if schemes:
        parsed = urlparse(url)
        return parsed.scheme in schemes
    
    return True


def is_valid_uuid(uuid_string: str, version: int = None) -> bool:
    """
    Valide un UUID
    
    Args:
        uuid_string: UUID à valider
        version: Version spécifique à valider (1-5)
    
    Returns:
        True si valide
    """
    try:
        uuid_obj = UUID(uuid_string)
        if version and uuid_obj.version != version:
            return False
        return True
    except (ValueError, AttributeError):
        return False


def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Évalue la force d'un mot de passe
    
    Args:
        password: Mot de passe à évaluer
    
    Returns:
        Dictionnaire avec score et détails
    """
    score = 0
    details = {
        'length_ok': len(password) >= 8,
        'has_lowercase': bool(re.search(r'[a-z]', password)),
        'has_uppercase': bool(re.search(r'[A-Z]', password)),
        'has_digit': bool(re.search(r'\d', password)),
        'has_special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
        'no_common_patterns': not any(pattern in password.lower() for pattern in ['password', '123456', 'qwerty'])
    }
    
    score = sum(details.values())
    
    if score >= 6:
        strength = 'strong'
    elif score >= 4:
        strength = 'medium'
    elif score >= 2:
        strength = 'weak'
    else:
        strength = 'very_weak'
    
    return {
        'score': score,
        'max_score': 6,
        'strength': strength,
        'details': details
    }


# =============================================================================
# UTILITAIRES DE CRYPTOGRAPHIE
# =============================================================================

def generate_random_string(length: int = 32, 
                          charset: str = None,
                          exclude_ambiguous: bool = True) -> str:
    """
    Génère une chaîne aléatoire sécurisée
    
    Args:
        length: Longueur de la chaîne
        charset: Jeu de caractères à utiliser
        exclude_ambiguous: Exclure les caractères ambigus (0, O, l, I)
    
    Returns:
        Chaîne aléatoire
    """
    if charset is None:
        charset = string.ascii_letters + string.digits
        if exclude_ambiguous:
            charset = charset.replace('0', '').replace('O', '').replace('l', '').replace('I', '')
    
    return ''.join(secrets.choice(charset) for _ in range(length))


def generate_api_key(prefix: str = 'sk', length: int = 32) -> str:
    """
    Génère une clé API avec préfixe
    
    Args:
        prefix: Préfixe de la clé
        length: Longueur de la partie aléatoire
    
    Returns:
        Clé API formatée
    """
    random_part = generate_random_string(length)
    return f"{prefix}_{random_part}"


def hash_password(password: str, salt: bytes = None) -> Tuple[str, bytes]:
    """
    Hash un mot de passe avec salt
    
    Args:
        password: Mot de passe à hasher
        salt: Salt optionnel (généré automatiquement si non fourni)
    
    Returns:
        Tuple (hash, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(32)
    
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return hash_obj.hex(), salt


def verify_password(password: str, hashed: str, salt: bytes) -> bool:
    """
    Vérifie un mot de passe contre son hash
    
    Args:
        password: Mot de passe à vérifier
        hashed: Hash stocké
        salt: Salt utilisé
    
    Returns:
        True si le mot de passe correspond
    """
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, hashed)


def generate_csrf_token() -> str:
    """Génère un token CSRF sécurisé"""
    return secrets.token_urlsafe(32)


def generate_jwt_secret() -> str:
    """Génère un secret JWT sécurisé"""
    return secrets.token_urlsafe(64)


# =============================================================================
# UTILITAIRES DE DATE ET HEURE
# =============================================================================

def now_utc() -> datetime:
    """Retourne l'heure actuelle en UTC"""
    return datetime.now(timezone.utc)


def parse_datetime(dt_string: str, formats: List[str] = None) -> Optional[datetime]:
    """
    Parse une chaîne de date/heure avec plusieurs formats possibles
    
    Args:
        dt_string: Chaîne à parser
        formats: Formats à essayer
    
    Returns:
        datetime parsé ou None
    """
    if formats is None:
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%d-%m-%Y'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_string, fmt)
        except ValueError:
            continue
    
    return None


def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes en format lisible
    
    Args:
        seconds: Durée en secondes
    
    Returns:
        Durée formatée (ex: "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def get_age_from_birthdate(birthdate: datetime) -> int:
    """
    Calcule l'âge à partir d'une date de naissance
    
    Args:
        birthdate: Date de naissance
    
    Returns:
        Âge en années
    """
    today = datetime.now().date()
    birth_date = birthdate.date() if isinstance(birthdate, datetime) else birthdate
    
    age = today.year - birth_date.year
    if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
        age -= 1
    
    return age


def get_timezone_offset(tz_name: str) -> timedelta:
    """
    Obtient le décalage d'un fuseau horaire par rapport à UTC
    
    Args:
        tz_name: Nom du fuseau horaire
    
    Returns:
        Décalage par rapport à UTC
    """
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
        return datetime.now(tz).utcoffset()
    except ImportError:
        # Fallback pour Python < 3.9
        return timedelta(0)


# =============================================================================
# UTILITAIRES DE FORMATAGE ET CONVERSION
# =============================================================================

def format_bytes(bytes_value: int, decimal_places: int = 2) -> str:
    """
    Formate une taille en bytes en format lisible
    
    Args:
        bytes_value: Taille en bytes
        decimal_places: Nombre de décimales
    
    Returns:
        Taille formatée (ex: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.{decimal_places}f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.{decimal_places}f} EB"


def format_number(number: Union[int, float, Decimal], 
                 decimal_places: int = 2,
                 thousands_separator: str = ' ',
                 decimal_separator: str = ',') -> str:
    """
    Formate un nombre selon les conventions locales
    
    Args:
        number: Nombre à formater
        decimal_places: Nombre de décimales
        thousands_separator: Séparateur de milliers
        decimal_separator: Séparateur décimal
    
    Returns:
        Nombre formaté
    """
    if isinstance(number, Decimal):
        number = float(number)
    
    # Arrondir au nombre de décimales
    rounded = round(number, decimal_places)
    
    # Séparer partie entière et décimale
    integer_part = int(rounded)
    decimal_part = rounded - integer_part
    
    # Formater la partie entière avec séparateurs de milliers
    integer_str = f"{integer_part:,}".replace(',', thousands_separator)
    
    # Ajouter la partie décimale si nécessaire
    if decimal_places > 0 and decimal_part > 0:
        decimal_str = f"{decimal_part:.{decimal_places}f}"[2:]  # Enlever "0."
        return f"{integer_str}{decimal_separator}{decimal_str}"
    
    return integer_str


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Formate un pourcentage
    
    Args:
        value: Valeur (0.0 à 1.0)
        decimal_places: Nombre de décimales
    
    Returns:
        Pourcentage formaté
    """
    percentage = value * 100
    return f"{percentage:.{decimal_places}f}%"


def parse_boolean(value: Any) -> bool:
    """
    Parse une valeur en booléen de manière flexible
    
    Args:
        value: Valeur à convertir
    
    Returns:
        Booléen résultant
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value = value.lower().strip()
        return value in ['true', '1', 'yes', 'on', 'oui', 'vrai']
    
    if isinstance(value, (int, float)):
        return value != 0
    
    return bool(value)


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    """
    Cast sécurisé d'une valeur vers un type
    
    Args:
        value: Valeur à convertir
        target_type: Type cible
        default: Valeur par défaut si échec
    
    Returns:
        Valeur convertie ou valeur par défaut
    """
    try:
        if target_type == bool:
            return parse_boolean(value)
        return target_type(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# UTILITAIRES DE STRUCTURES DE DONNÉES
# =============================================================================

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusion profonde de deux dictionnaires
    
    Args:
        dict1: Premier dictionnaire
        dict2: Deuxième dictionnaire (prioritaire)
    
    Returns:
        Dictionnaire fusionné
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], 
                parent_key: str = '', 
                separator: str = '.') -> Dict[str, Any]:
    """
    Aplatit un dictionnaire imbriqué
    
    Args:
        d: Dictionnaire à aplatir
        parent_key: Clé parente pour la récursion
        separator: Séparateur de clés
    
    Returns:
        Dictionnaire aplati
    """
    items = []
    
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Reconstruit un dictionnaire imbriqué à partir d'un dictionnaire aplati
    
    Args:
        d: Dictionnaire aplati
        separator: Séparateur de clés
    
    Returns:
        Dictionnaire imbriqué
    """
    result = {}
    
    for key, value in d.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def remove_none_values(d: Dict[str, Any], recursive: bool = True) -> Dict[str, Any]:
    """
    Supprime les valeurs None d'un dictionnaire
    
    Args:
        d: Dictionnaire à nettoyer
        recursive: Nettoyer récursivement
    
    Returns:
        Dictionnaire nettoyé
    """
    result = {}
    
    for key, value in d.items():
        if value is None:
            continue
        
        if recursive and isinstance(value, dict):
            cleaned = remove_none_values(value, recursive)
            if cleaned:  # Ne pas ajouter de dictionnaires vides
                result[key] = cleaned
        else:
            result[key] = value
    
    return result


def chunk_list(lst: List[T], chunk_size: int) -> Iterator[List[T]]:
    """
    Divise une liste en chunks de taille fixe
    
    Args:
        lst: Liste à diviser
        chunk_size: Taille des chunks
    
    Yields:
        Chunks de la liste
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def group_by(items: List[T], key_func: Callable[[T], Any]) -> Dict[Any, List[T]]:
    """
    Groupe les éléments d'une liste par clé
    
    Args:
        items: Liste d'éléments
        key_func: Fonction pour extraire la clé
    
    Returns:
        Dictionnaire groupé
    """
    groups = defaultdict(list)
    for item in items:
        key = key_func(item)
        groups[key].append(item)
    return dict(groups)


# =============================================================================
# UTILITAIRES DE PERFORMANCE
# =============================================================================

def timing_decorator(func: Callable) -> Callable:
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction
    
    Args:
        func: Fonction à décorer
    
    Returns:
        Fonction décorée
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} executed in {execution_time:.4f}s")
        
        return result
    
    return wrapper


def memoize(maxsize: int = 128, ttl: Optional[int] = None) -> Callable:
    """
    Décorateur de mémoisation avec TTL optionnel
    
    Args:
        maxsize: Taille maximale du cache
        ttl: Durée de vie des entrées en secondes
    
    Returns:
        Décorateur
    """
    def decorator(func: Callable) -> Callable:
        cache = OrderedDict()
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Créer une clé de cache
            key = str(hash((args, tuple(sorted(kwargs.items())))))
            
            current_time = time.time()
            
            # Vérifier si l'entrée existe et n'est pas expirée
            if key in cache:
                if ttl is None or (current_time - cache_times[key]) < ttl:
                    # Déplacer vers la fin (LRU)
                    cache.move_to_end(key)
                    return cache[key]
                else:
                    # Supprimer l'entrée expirée
                    del cache[key]
                    del cache_times[key]
            
            # Calculer le résultat
            result = func(*args, **kwargs)
            
            # Ajouter au cache
            cache[key] = result
            cache_times[key] = current_time
            
            # Respecter la taille maximale
            while len(cache) > maxsize:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            return result
        
        wrapper.cache_clear = lambda: (cache.clear(), cache_times.clear())
        wrapper.cache_info = lambda: {
            'hits': len(cache),
            'maxsize': maxsize,
            'currsize': len(cache)
        }
        
        return wrapper
    
    return decorator


def retry_on_exception(max_attempts: int = 3,
                      delay: float = 1.0,
                      backoff_multiplier: float = 2.0,
                      exceptions: Tuple[type, ...] = (Exception,)) -> Callable:
    """
    Décorateur de retry avec backoff exponentiel
    
    Args:
        max_attempts: Nombre maximum de tentatives
        delay: Délai initial entre tentatives
        backoff_multiplier: Multiplicateur pour le backoff
        exceptions: Types d'exceptions à rattraper
    
    Returns:
        Décorateur
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_multiplier
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_multiplier
        
        # Retourner le wrapper approprié selon si la fonction est async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# UTILITAIRES DIVERS
# =============================================================================

def get_client_ip(request_headers: Dict[str, str]) -> str:
    """
    Extrait l'IP du client depuis les headers HTTP
    
    Args:
        request_headers: Headers de la requête
    
    Returns:
        Adresse IP du client
    """
    # Essayer différents headers dans l'ordre de priorité
    ip_headers = [
        'X-Forwarded-For',
        'X-Real-IP',
        'CF-Connecting-IP',
        'X-Client-IP',
        'Remote-Addr'
    ]
    
    for header in ip_headers:
        ip = request_headers.get(header)
        if ip:
            # Prendre la première IP si plusieurs (proxy chain)
            return ip.split(',')[0].strip()
    
    return '127.0.0.1'  # Fallback


def extract_domain(url: str) -> str:
    """
    Extrait le domaine d'une URL
    
    Args:
        url: URL à analyser
    
    Returns:
        Nom de domaine
    """
    parsed = urlparse(url)
    return parsed.netloc.lower()


def detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Détecte le type MIME d'un fichier
    
    Args:
        file_path: Chemin vers le fichier
    
    Returns:
        Type MIME détecté
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'


def generate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Génère un hash d'un fichier
    
    Args:
        file_path: Chemin vers le fichier
        algorithm: Algorithme de hash (md5, sha1, sha256, etc.)
    
    Returns:
        Hash hexadécimal du fichier
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def compress_string(text: str, method: str = 'gzip') -> bytes:
    """
    Compresse une chaîne de caractères
    
    Args:
        text: Texte à compresser
        method: Méthode de compression (gzip, zlib)
    
    Returns:
        Données compressées
    """
    data = text.encode(DEFAULT_ENCODING)
    
    if method == 'gzip':
        return gzip.compress(data)
    elif method == 'zlib':
        return zlib.compress(data)
    else:
        raise ValueError(f"Méthode de compression non supportée: {method}")


def decompress_string(data: bytes, method: str = 'gzip') -> str:
    """
    Décompresse des données en chaîne
    
    Args:
        data: Données compressées
        method: Méthode de décompression
    
    Returns:
        Texte décompressé
    """
    if method == 'gzip':
        decompressed = gzip.decompress(data)
    elif method == 'zlib':
        decompressed = zlib.decompress(data)
    else:
        raise ValueError(f"Méthode de décompression non supportée: {method}")
    
    return decompressed.decode(DEFAULT_ENCODING)


__all__ = [
    # String utilities
    'normalize_string', 'slugify', 'truncate_string', 'mask_sensitive_data',
    'extract_mentions', 'extract_hashtags',
    
    # Validation utilities
    'is_valid_email', 'is_valid_phone', 'is_valid_url', 'is_valid_uuid',
    'validate_password_strength',
    
    # Crypto utilities
    'generate_random_string', 'generate_api_key', 'hash_password', 
    'verify_password', 'generate_csrf_token', 'generate_jwt_secret',
    
    # Date/time utilities
    'now_utc', 'parse_datetime', 'format_duration', 'get_age_from_birthdate',
    'get_timezone_offset',
    
    # Formatting utilities
    'format_bytes', 'format_number', 'format_percentage', 'parse_boolean', 'safe_cast',
    
    # Data structure utilities
    'deep_merge', 'flatten_dict', 'unflatten_dict', 'remove_none_values',
    'chunk_list', 'group_by',
    
    # Performance utilities
    'timing_decorator', 'memoize', 'retry_on_exception',
    
    # Misc utilities
    'get_client_ip', 'extract_domain', 'detect_file_type', 'generate_file_hash',
    'compress_string', 'decompress_string'
]
