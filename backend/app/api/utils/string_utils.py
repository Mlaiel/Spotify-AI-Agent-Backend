"""
🎵 Spotify AI Agent - String Processing Utilities
================================================

Utilitaires enterprise pour la manipulation avancée de chaînes
avec optimisation des performances et sécurité renforcée.

Architecture:
- Manipulation et formatage de chaînes
- Extraction d'informations (emails, URLs, etc.)
- Conversion de casse et formatage
- Génération de slugs et identifiants
- Nettoyage et validation de texte
- Hachage et chiffrement de chaînes

🎖️ Développé par l'équipe d'experts enterprise
"""

import re
import hashlib
import secrets
import string
import unicodedata
from typing import List, Optional, Union, Pattern
from urllib.parse import urlparse


# =============================================================================
# MANIPULATION DE CHAÎNES
# =============================================================================

def slugify(text: str, max_length: int = 100, separator: str = '-') -> str:
    """
    Convertit une chaîne en slug URL-friendly
    
    Args:
        text: Texte à convertir
        max_length: Longueur maximale
        separator: Séparateur à utiliser
        
    Returns:
        Slug généré
    """
    # Normaliser le texte Unicode
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Convertir en minuscules et remplacer les espaces
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[\s_-]+', separator, text)
    
    # Supprimer les séparateurs en début/fin
    text = text.strip(separator)
    
    # Limiter la longueur
    if len(text) > max_length:
        text = text[:max_length].rstrip(separator)
    
    return text


def camel_to_snake(text: str) -> str:
    """
    Convertit camelCase en snake_case
    
    Args:
        text: Texte en camelCase
        
    Returns:
        Texte en snake_case
    """
    # Ajouter un underscore avant les majuscules
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
    """
    Convertit snake_case en camelCase
    
    Args:
        text: Texte en snake_case
        capitalize_first: Capitaliser la première lettre
        
    Returns:
        Texte en camelCase
    """
    components = text.split('_')
    if capitalize_first:
        return ''.join(word.capitalize() for word in components)
    else:
        return components[0] + ''.join(word.capitalize() for word in components[1:])


def truncate_text(text: str, max_length: int = 100, 
                 suffix: str = '...', word_boundary: bool = True) -> str:
    """
    Tronque un texte à une longueur maximale
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter
        word_boundary: Respecter les limites de mots
        
    Returns:
        Texte tronqué
    """
    if len(text) <= max_length:
        return text
    
    # Calculer la longueur effective
    effective_length = max_length - len(suffix)
    
    if word_boundary:
        # Trouver le dernier espace avant la limite
        truncated = text[:effective_length]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + suffix
    else:
        return text[:effective_length] + suffix


def clean_text(text: str, remove_extra_spaces: bool = True,
              remove_special_chars: bool = False,
              allowed_chars: Optional[str] = None) -> str:
    """
    Nettoie un texte selon différents critères
    
    Args:
        text: Texte à nettoyer
        remove_extra_spaces: Supprimer les espaces multiples
        remove_special_chars: Supprimer les caractères spéciaux
        allowed_chars: Caractères autorisés (si remove_special_chars=True)
        
    Returns:
        Texte nettoyé
    """
    cleaned = text.strip()
    
    if remove_extra_spaces:
        cleaned = re.sub(r'\s+', ' ', cleaned)
    
    if remove_special_chars:
        if allowed_chars is None:
            allowed_chars = string.ascii_letters + string.digits + ' .-_'
        
        cleaned = ''.join(char for char in cleaned if char in allowed_chars)
    
    return cleaned


# =============================================================================
# EXTRACTION D'INFORMATIONS
# =============================================================================

def extract_emails(text: str) -> List[str]:
    """
    Extrait toutes les adresses email d'un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Liste des emails trouvés
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def extract_urls(text: str) -> List[str]:
    """
    Extrait toutes les URLs d'un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Liste des URLs trouvées
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


def extract_phone_numbers(text: str, country_code: Optional[str] = None) -> List[str]:
    """
    Extrait les numéros de téléphone d'un texte
    
    Args:
        text: Texte à analyser
        country_code: Code pays pour filtrer
        
    Returns:
        Liste des numéros trouvés
    """
    # Pattern pour différents formats de numéros
    patterns = [
        r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',  # US/CA
        r'\+?33[-.\s]?([0-9])[-.\s]?([0-9]{2})[-.\s]?([0-9]{2})[-.\s]?([0-9]{2})[-.\s]?([0-9]{2})',  # FR
        r'\+?49[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{6,8})',  # DE
        r'\+?44[-.\s]?([0-9]{4})[-.\s]?([0-9]{6})',  # UK
    ]
    
    phone_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phone_numbers.extend([''.join(match) if isinstance(match, tuple) else match for match in matches])
    
    return phone_numbers


def extract_hashtags(text: str) -> List[str]:
    """
    Extrait les hashtags d'un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Liste des hashtags trouvés
    """
    hashtag_pattern = r'#\w+'
    return re.findall(hashtag_pattern, text)


def extract_mentions(text: str) -> List[str]:
    """
    Extrait les mentions (@username) d'un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Liste des mentions trouvées
    """
    mention_pattern = r'@\w+'
    return re.findall(mention_pattern, text)


# =============================================================================
# GÉNÉRATION ET HACHAGE
# =============================================================================

def generate_hash(text: str, algorithm: str = 'sha256') -> str:
    """
    Génère un hash d'une chaîne
    
    Args:
        text: Texte à hasher
        algorithm: Algorithme de hachage
        
    Returns:
        Hash hexadécimal
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


def generate_random_string(length: int = 32, 
                         include_symbols: bool = False,
                         exclude_ambiguous: bool = True) -> str:
    """
    Génère une chaîne aléatoire sécurisée
    
    Args:
        length: Longueur de la chaîne
        include_symbols: Inclure les symboles
        exclude_ambiguous: Exclure les caractères ambigus
        
    Returns:
        Chaîne aléatoire
    """
    chars = string.ascii_letters + string.digits
    
    if include_symbols:
        chars += '!@#$%^&*()_+-=[]{}|;:,.<>?'
    
    if exclude_ambiguous:
        # Supprimer les caractères ambigus
        ambiguous = '0O1lI'
        chars = ''.join(c for c in chars if c not in ambiguous)
    
    return ''.join(secrets.choice(chars) for _ in range(length))


def generate_uuid_string(version: int = 4, remove_hyphens: bool = False) -> str:
    """
    Génère un UUID sous forme de chaîne
    
    Args:
        version: Version de l'UUID
        remove_hyphens: Supprimer les tirets
        
    Returns:
        UUID sous forme de chaîne
    """
    import uuid
    
    if version == 1:
        uuid_obj = uuid.uuid1()
    elif version == 4:
        uuid_obj = uuid.uuid4()
    else:
        raise ValueError("Only UUID versions 1 and 4 are supported")
    
    uuid_str = str(uuid_obj)
    
    if remove_hyphens:
        uuid_str = uuid_str.replace('-', '')
    
    return uuid_str


# =============================================================================
# VALIDATION ET FORMATAGE
# =============================================================================

def is_valid_email(email: str) -> bool:
    """
    Valide une adresse email
    
    Args:
        email: Email à valider
        
    Returns:
        True si valide
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """
    Valide une URL
    
    Args:
        url: URL à valider
        
    Returns:
        True si valide
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def format_phone_number(phone: str, format_style: str = 'international') -> str:
    """
    Formate un numéro de téléphone
    
    Args:
        phone: Numéro à formater
        format_style: Style de formatage
        
    Returns:
        Numéro formaté
    """
    # Supprimer tous les caractères non numériques sauf le +
    digits = re.sub(r'[^\d+]', '', phone)
    
    if format_style == 'international':
        if not digits.startswith('+'):
            digits = '+' + digits
        return digits
    elif format_style == 'national':
        # Supprimer le code pays
        if digits.startswith('+'):
            digits = digits[1:]
        return digits
    elif format_style == 'e164':
        if not digits.startswith('+'):
            digits = '+' + digits
        return digits
    else:
        return phone


def mask_sensitive_data(text: str, mask_char: str = '*', 
                       preserve_first: int = 2, preserve_last: int = 2) -> str:
    """
    Masque les données sensibles dans une chaîne
    
    Args:
        text: Texte à masquer
        mask_char: Caractère de masquage
        preserve_first: Nombre de caractères à conserver au début
        preserve_last: Nombre de caractères à conserver à la fin
        
    Returns:
        Texte masqué
    """
    if len(text) <= preserve_first + preserve_last:
        return mask_char * len(text)
    
    start = text[:preserve_first]
    end = text[-preserve_last:] if preserve_last > 0 else ''
    middle_length = len(text) - preserve_first - preserve_last
    middle = mask_char * middle_length
    
    return start + middle + end


# =============================================================================
# ANALYSE DE TEXTE
# =============================================================================

def count_words(text: str, unique_only: bool = False) -> int:
    """
    Compte les mots dans un texte
    
    Args:
        text: Texte à analyser
        unique_only: Compter seulement les mots uniques
        
    Returns:
        Nombre de mots
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if unique_only:
        return len(set(words))
    else:
        return len(words)


def get_text_statistics(text: str) -> dict:
    """
    Retourne des statistiques sur un texte
    
    Args:
        text: Texte à analyser
        
    Returns:
        Dictionnaire de statistiques
    """
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    
    return {
        'characters': len(text),
        'characters_no_spaces': len(text.replace(' ', '')),
        'words': len(words),
        'unique_words': len(set(word.lower() for word in words)),
        'sentences': len([s for s in sentences if s.strip()]),
        'paragraphs': len([p for p in paragraphs if p.strip()]),
        'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'average_sentence_length': len(words) / len(sentences) if sentences else 0
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "slugify",
    "camel_to_snake",
    "snake_to_camel", 
    "truncate_text",
    "clean_text",
    "extract_emails",
    "extract_urls",
    "extract_phone_numbers",
    "extract_hashtags",
    "extract_mentions",
    "generate_hash",
    "generate_random_string",
    "generate_uuid_string",
    "is_valid_email",
    "is_valid_url",
    "format_phone_number",
    "mask_sensitive_data",
    "count_words",
    "get_text_statistics"
]
