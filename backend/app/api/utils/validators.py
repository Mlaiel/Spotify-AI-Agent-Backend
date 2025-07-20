"""
🎵 Spotify AI Agent - Enterprise Validators
===========================================

Validateurs enterprise complets pour toutes les données
avec sécurité renforcée et validation métier avancée.

Architecture:
- Validation de données utilisateur
- Validation de formats de fichiers
- Validation de métadonnées musicales
- Validation de sécurité
- Validation de modèles ML
- Validation d'APIs et endpoints

🎖️ Développé par l'équipe d'experts enterprise
"""

import re
import email_validator
import phonenumbers
import validators as external_validators
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, date
from urllib.parse import urlparse
import mimetypes
import magic
from pydantic import BaseModel, ValidationError, validator
from decimal import Decimal, InvalidOperation
import ipaddress


# =============================================================================
# VALIDATEURS DE FORMATS BASIQUES
# =============================================================================

def validate_email(email: str, check_deliverability: bool = False) -> Dict[str, Any]:
    """
    Valide une adresse email
    
    Args:
        email: Adresse email à valider
        check_deliverability: Vérifier la délivrabilité
        
    Returns:
        Résultat de validation avec détails
    """
    try:
        valid = email_validator.validate_email(
            email, 
            check_deliverability=check_deliverability
        )
        
        return {
            'valid': True,
            'email': valid.email,
            'local': valid.local,
            'domain': valid.domain,
            'ascii_email': valid.ascii_email,
            'ascii_local': valid.ascii_local,
            'ascii_domain': valid.ascii_domain,
            'smtputf8': valid.smtputf8
        }
    
    except email_validator.EmailNotValidError as e:
        return {
            'valid': False,
            'error': str(e),
            'code': e.code if hasattr(e, 'code') else None
        }


def validate_phone(phone: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Valide un numéro de téléphone
    
    Args:
        phone: Numéro de téléphone
        region: Code région (ISO 3166-1 alpha-2)
        
    Returns:
        Résultat de validation
    """
    try:
        parsed = phonenumbers.parse(phone, region)
        
        return {
            'valid': phonenumbers.is_valid_number(parsed),
            'possible': phonenumbers.is_possible_number(parsed),
            'country_code': parsed.country_code,
            'national_number': parsed.national_number,
            'extension': parsed.extension,
            'number_type': phonenumbers.number_type(parsed),
            'carrier': phonenumbers.carrier.name_for_number(parsed, 'en'),
            'timezone': list(phonenumbers.timezone.time_zones_for_number(parsed)),
            'formatted': {
                'international': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                'national': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL),
                'e164': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
                'rfc3966': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.RFC3966)
            }
        }
    
    except phonenumbers.phonenumberutil.NumberParseException as e:
        return {
            'valid': False,
            'error': str(e),
            'error_type': e.error_type
        }


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Valide une URL avec analyse détaillée
    
    Args:
        url: URL à valider
        allowed_schemes: Schémas autorisés
        
    Returns:
        Résultat de validation détaillé
    """
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']
    
    try:
        # Validation basique
        is_valid = external_validators.url(url)
        if not is_valid:
            return {'valid': False, 'error': 'Format URL invalide'}
        
        # Analyse détaillée
        parsed = urlparse(url)
        
        # Vérifier le schéma
        if parsed.scheme not in allowed_schemes:
            return {
                'valid': False,
                'error': f'Schéma non autorisé: {parsed.scheme}',
                'allowed_schemes': allowed_schemes
            }
        
        return {
            'valid': True,
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'hostname': parsed.hostname,
            'port': parsed.port,
            'path': parsed.path,
            'params': parsed.params,
            'query': parsed.query,
            'fragment': parsed.fragment,
            'is_secure': parsed.scheme == 'https'
        }
    
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def validate_ip_address(ip: str, version: Optional[int] = None) -> Dict[str, Any]:
    """
    Valide une adresse IP
    
    Args:
        ip: Adresse IP
        version: Version IP (4 ou 6)
        
    Returns:
        Résultat de validation
    """
    try:
        if version == 4:
            ip_obj = ipaddress.IPv4Address(ip)
        elif version == 6:
            ip_obj = ipaddress.IPv6Address(ip)
        else:
            ip_obj = ipaddress.ip_address(ip)
        
        return {
            'valid': True,
            'version': ip_obj.version,
            'compressed': str(ip_obj.compressed),
            'exploded': str(ip_obj.exploded),
            'is_private': ip_obj.is_private,
            'is_global': ip_obj.is_global,
            'is_loopback': ip_obj.is_loopback,
            'is_multicast': ip_obj.is_multicast,
            'is_reserved': ip_obj.is_reserved,
            'is_unspecified': ip_obj.is_unspecified
        }
    
    except ValueError as e:
        return {'valid': False, 'error': str(e)}


# =============================================================================
# VALIDATEURS DE DONNÉES MÉTIER
# =============================================================================

def validate_user_password(password: str, min_length: int = 8) -> Dict[str, Any]:
    """
    Valide un mot de passe avec critères de sécurité
    
    Args:
        password: Mot de passe à valider
        min_length: Longueur minimale
        
    Returns:
        Résultat de validation avec score de force
    """
    issues = []
    score = 0
    
    # Longueur
    if len(password) < min_length:
        issues.append(f'Doit contenir au moins {min_length} caractères')
    else:
        score += 1
        if len(password) >= 12:
            score += 1
    
    # Caractères minuscules
    if not re.search(r'[a-z]', password):
        issues.append('Doit contenir au moins une lettre minuscule')
    else:
        score += 1
    
    # Caractères majuscules
    if not re.search(r'[A-Z]', password):
        issues.append('Doit contenir au moins une lettre majuscule')
    else:
        score += 1
    
    # Chiffres
    if not re.search(r'\d', password):
        issues.append('Doit contenir au moins un chiffre')
    else:
        score += 1
    
    # Caractères spéciaux
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append('Doit contenir au moins un caractère spécial')
    else:
        score += 1
    
    # Motifs faibles
    if re.search(r'(.)\1{2,}', password):  # Répétitions
        issues.append('Éviter les répétitions de caractères')
        score -= 1
    
    if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
        issues.append('Éviter les séquences numériques')
        score -= 1
    
    if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
        issues.append('Éviter les séquences alphabétiques')
        score -= 1
    
    # Mots courants
    common_words = ['password', 'motdepasse', '123456', 'azerty', 'qwerty', 'admin']
    if any(word in password.lower() for word in common_words):
        issues.append('Éviter les mots courants')
        score -= 2
    
    # Score final
    max_score = 6
    strength_score = max(0, min(score, max_score))
    
    strength_levels = ['Très faible', 'Faible', 'Moyen', 'Bon', 'Fort', 'Très fort']
    strength = strength_levels[min(strength_score, len(strength_levels) - 1)]
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'score': strength_score,
        'max_score': max_score,
        'strength': strength,
        'percentage': (strength_score / max_score) * 100
    }


def validate_username(username: str, min_length: int = 3, max_length: int = 30) -> Dict[str, Any]:
    """
    Valide un nom d'utilisateur
    
    Args:
        username: Nom d'utilisateur
        min_length: Longueur minimale
        max_length: Longueur maximale
        
    Returns:
        Résultat de validation
    """
    issues = []
    
    # Longueur
    if len(username) < min_length:
        issues.append(f'Doit contenir au moins {min_length} caractères')
    
    if len(username) > max_length:
        issues.append(f'Ne doit pas dépasser {max_length} caractères')
    
    # Format
    if not re.match(r'^[a-zA-Z0-9_.-]+$', username):
        issues.append('Doit contenir uniquement des lettres, chiffres, _, . et -')
    
    # Début et fin
    if username.startswith(('.', '-', '_')):
        issues.append('Ne doit pas commencer par ., - ou _')
    
    if username.endswith(('.', '-', '_')):
        issues.append('Ne doit pas finir par ., - ou _')
    
    # Caractères consécutifs
    if '..' in username or '--' in username or '__' in username:
        issues.append('Éviter les caractères spéciaux consécutifs')
    
    # Mots réservés
    reserved_words = ['admin', 'root', 'user', 'test', 'null', 'undefined', 'system']
    if username.lower() in reserved_words:
        issues.append('Nom d\'utilisateur réservé')
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'username': username,
        'length': len(username)
    }


# =============================================================================
# VALIDATEURS MUSICAUX
# =============================================================================

def validate_audio_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide les métadonnées audio
    
    Args:
        metadata: Métadonnées à valider
        
    Returns:
        Résultat de validation
    """
    issues = []
    warnings = []
    
    # Champs obligatoires
    required_fields = ['title', 'artist', 'duration']
    for field in required_fields:
        if field not in metadata or not metadata[field]:
            issues.append(f'Champ obligatoire manquant: {field}')
    
    # Validation du titre
    if 'title' in metadata:
        title = metadata['title']
        if len(title) > 200:
            warnings.append('Titre très long (>200 caractères)')
        if len(title.strip()) == 0:
            issues.append('Titre ne peut pas être vide')
    
    # Validation de l'artiste
    if 'artist' in metadata:
        artist = metadata['artist']
        if len(artist) > 100:
            warnings.append('Nom d\'artiste très long (>100 caractères)')
        if len(artist.strip()) == 0:
            issues.append('Artiste ne peut pas être vide')
    
    # Validation de la durée
    if 'duration' in metadata:
        duration = metadata['duration']
        try:
            duration_float = float(duration)
            if duration_float <= 0:
                issues.append('Durée doit être positive')
            if duration_float > 3600:  # 1 heure
                warnings.append('Durée très longue (>1h)')
            if duration_float < 10:  # 10 secondes
                warnings.append('Durée très courte (<10s)')
        except (ValueError, TypeError):
            issues.append('Format de durée invalide')
    
    # Validation de l'album (optionnel)
    if 'album' in metadata and metadata['album']:
        album = metadata['album']
        if len(album) > 150:
            warnings.append('Nom d\'album très long (>150 caractères)')
    
    # Validation de l'année
    if 'year' in metadata and metadata['year']:
        try:
            year = int(metadata['year'])
            current_year = datetime.now().year
            if year < 1900 or year > current_year + 1:
                issues.append(f'Année invalide: {year}')
        except (ValueError, TypeError):
            issues.append('Format d\'année invalide')
    
    # Validation du genre
    if 'genre' in metadata and metadata['genre']:
        genre = metadata['genre']
        valid_genres = [
            'Rock', 'Pop', 'Hip-Hop', 'Jazz', 'Classical', 'Electronic',
            'Country', 'R&B', 'Folk', 'Blues', 'Reggae', 'Punk',
            'Metal', 'Alternative', 'Indie', 'Soul', 'Funk', 'Disco'
        ]
        if genre not in valid_genres:
            warnings.append(f'Genre non standard: {genre}')
    
    # Validation du bitrate
    if 'bitrate' in metadata and metadata['bitrate']:
        try:
            bitrate = int(metadata['bitrate'])
            if bitrate < 64:
                warnings.append('Bitrate très bas (<64 kbps)')
            if bitrate > 320:
                warnings.append('Bitrate très élevé (>320 kbps)')
        except (ValueError, TypeError):
            issues.append('Format de bitrate invalide')
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'metadata': metadata
    }


def validate_playlist_data(playlist: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide les données d'une playlist
    
    Args:
        playlist: Données de playlist
        
    Returns:
        Résultat de validation
    """
    issues = []
    warnings = []
    
    # Champs obligatoires
    if 'name' not in playlist or not playlist['name']:
        issues.append('Nom de playlist obligatoire')
    
    if 'tracks' not in playlist:
        issues.append('Liste des pistes obligatoire')
    
    # Validation du nom
    if 'name' in playlist:
        name = playlist['name']
        if len(name) > 100:
            issues.append('Nom de playlist trop long (>100 caractères)')
        if len(name.strip()) == 0:
            issues.append('Nom de playlist ne peut pas être vide')
        
        # Caractères interdits
        forbidden_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        if any(char in name for char in forbidden_chars):
            issues.append('Caractères interdits dans le nom de playlist')
    
    # Validation des pistes
    if 'tracks' in playlist:
        tracks = playlist['tracks']
        if not isinstance(tracks, list):
            issues.append('Les pistes doivent être une liste')
        else:
            if len(tracks) == 0:
                warnings.append('Playlist vide')
            elif len(tracks) > 1000:
                warnings.append('Playlist très longue (>1000 pistes)')
            
            # Validation de chaque piste
            for i, track in enumerate(tracks):
                if not isinstance(track, dict):
                    issues.append(f'Piste {i+1}: Format invalide')
                    continue
                
                if 'id' not in track and 'title' not in track:
                    issues.append(f'Piste {i+1}: ID ou titre requis')
    
    # Description
    if 'description' in playlist and playlist['description']:
        if len(playlist['description']) > 500:
            warnings.append('Description très longue (>500 caractères)')
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'track_count': len(playlist.get('tracks', [])),
        'playlist': playlist
    }


# =============================================================================
# VALIDATEURS DE FICHIERS
# =============================================================================

def validate_audio_file(file_path: str, max_size_mb: int = 50) -> Dict[str, Any]:
    """
    Valide un fichier audio
    
    Args:
        file_path: Chemin du fichier
        max_size_mb: Taille maximale en MB
        
    Returns:
        Résultat de validation
    """
    import os
    
    issues = []
    warnings = []
    
    # Existence du fichier
    if not os.path.exists(file_path):
        return {'valid': False, 'issues': ['Fichier inexistant']}
    
    # Taille du fichier
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        issues.append(f'Fichier trop volumineux: {file_size_mb:.1f}MB (max: {max_size_mb}MB)')
    
    if file_size_mb < 0.1:
        warnings.append('Fichier très petit (<0.1MB)')
    
    # Type MIME
    try:
        mime_type = magic.from_file(file_path, mime=True)
        
        valid_audio_types = [
            'audio/mpeg', 'audio/wav', 'audio/flac', 'audio/aac',
            'audio/ogg', 'audio/mp4', 'audio/x-ms-wma'
        ]
        
        if mime_type not in valid_audio_types:
            issues.append(f'Type de fichier non supporté: {mime_type}')
    
    except Exception as e:
        warnings.append(f'Impossible de détecter le type MIME: {str(e)}')
    
    # Extension
    _, ext = os.path.splitext(file_path)
    valid_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']
    
    if ext.lower() not in valid_extensions:
        issues.append(f'Extension non supportée: {ext}')
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'file_size_mb': file_size_mb,
        'mime_type': mime_type if 'mime_type' in locals() else None,
        'extension': ext
    }


def validate_image_file(file_path: str, max_size_mb: int = 10) -> Dict[str, Any]:
    """
    Valide un fichier image
    
    Args:
        file_path: Chemin du fichier
        max_size_mb: Taille maximale en MB
        
    Returns:
        Résultat de validation
    """
    import os
    from PIL import Image
    
    issues = []
    warnings = []
    
    # Existence du fichier
    if not os.path.exists(file_path):
        return {'valid': False, 'issues': ['Fichier inexistant']}
    
    # Taille du fichier
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        issues.append(f'Fichier trop volumineux: {file_size_mb:.1f}MB (max: {max_size_mb}MB)')
    
    # Validation avec PIL
    try:
        with Image.open(file_path) as img:
            # Dimensions
            width, height = img.size
            
            if width < 100 or height < 100:
                warnings.append('Image très petite (<100px)')
            
            if width > 5000 or height > 5000:
                warnings.append('Image très grande (>5000px)')
            
            # Format
            valid_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']
            if img.format not in valid_formats:
                issues.append(f'Format non supporté: {img.format}')
            
            # Mode couleur
            if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                warnings.append(f'Mode couleur inhabituel: {img.mode}')
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'width': width,
                'height': height,
                'format': img.format,
                'mode': img.mode,
                'file_size_mb': file_size_mb
            }
    
    except Exception as e:
        return {
            'valid': False,
            'issues': [f'Erreur lors de l\'ouverture de l\'image: {str(e)}'],
            'file_size_mb': file_size_mb
        }


# =============================================================================
# VALIDATEUR COMPOSITE
# =============================================================================

class DataValidator:
    """Validateur composite pour validation complexe"""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.results: List[Dict[str, Any]] = []
    
    def add_rule(self, field: str, validator_func: Callable, 
                 error_message: Optional[str] = None) -> None:
        """
        Ajoute une règle de validation
        
        Args:
            field: Nom du champ
            validator_func: Fonction de validation
            error_message: Message d'erreur personnalisé
        """
        if field not in self.validation_rules:
            self.validation_rules[field] = []
        
        rule = {
            'validator': validator_func,
            'error_message': error_message
        }
        
        self.validation_rules[field].append(rule)
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide un dictionnaire de données
        
        Args:
            data: Données à valider
            
        Returns:
            Résultat de validation complet
        """
        field_results = {}
        global_issues = []
        
        # Valider chaque champ
        for field, rules in self.validation_rules.items():
            field_issues = []
            field_warnings = []
            
            value = data.get(field)
            
            for rule in rules:
                try:
                    result = rule['validator'](value)
                    
                    if isinstance(result, dict):
                        if not result.get('valid', True):
                            message = rule['error_message'] or f'Validation échouée pour {field}'
                            field_issues.append(message)
                            if 'issues' in result:
                                field_issues.extend(result['issues'])
                        
                        if 'warnings' in result:
                            field_warnings.extend(result['warnings'])
                    
                    elif not result:  # Boolean False
                        message = rule['error_message'] or f'Validation échouée pour {field}'
                        field_issues.append(message)
                
                except Exception as e:
                    field_issues.append(f'Erreur de validation pour {field}: {str(e)}')
            
            field_results[field] = {
                'valid': len(field_issues) == 0,
                'issues': field_issues,
                'warnings': field_warnings,
                'value': value
            }
        
        # Résultat global
        all_issues = []
        all_warnings = []
        
        for field_result in field_results.values():
            all_issues.extend(field_result['issues'])
            all_warnings.extend(field_result['warnings'])
        
        result = {
            'valid': len(all_issues) == 0,
            'issues': all_issues,
            'warnings': all_warnings,
            'field_results': field_results,
            'data': data
        }
        
        self.results.append(result)
        return result
    
    def clear_rules(self) -> None:
        """Efface toutes les règles de validation"""
        self.validation_rules.clear()
    
    def clear_results(self) -> None:
        """Efface l'historique des résultats"""
        self.results.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "validate_email",
    "validate_phone",
    "validate_url",
    "validate_ip_address",
    "validate_user_password",
    "validate_username",
    "validate_audio_metadata",
    "validate_playlist_data",
    "validate_audio_file",
    "validate_image_file",
    "DataValidator"
]
