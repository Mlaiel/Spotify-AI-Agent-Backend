"""
🎵 Spotify AI Agent - Data Transformation Utilities
==================================================

Utilitaires enterprise pour la transformation, validation et 
normalisation des données avec sécurité et performance optimisées.

Architecture:
- Validation de structure de données
- Transformation et normalisation
- Sanitisation et nettoyage
- Sérialisation/Désérialisation sécurisée
- Manipulation de dictionnaires imbriqués
- Conversion de types avancée

🎖️ Développé par l'équipe d'experts enterprise
"""

import json
import copy
import re
import html
from typing import Any, Dict, List, Union, Optional, Callable, Type
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path

import bleach
from pydantic import BaseModel, ValidationError


# =============================================================================
# TRANSFORMATION DE DONNÉES
# =============================================================================

def transform_data(data: Any, schema: Optional[Dict] = None, 
                  transformers: Optional[Dict[str, Callable]] = None) -> Any:
    """
    Transforme les données selon un schéma et des transformateurs personnalisés
    
    Args:
        data: Données à transformer
        schema: Schéma de transformation
        transformers: Fonctions de transformation personnalisées
        
    Returns:
        Données transformées
    """
    if schema is None:
        return data
    
    if transformers is None:
        transformers = {}
    
    if isinstance(data, dict):
        return {
            key: transform_data(
                value, 
                schema.get(key, {}), 
                transformers
            ) for key, value in data.items()
        }
    elif isinstance(data, list):
        return [transform_data(item, schema, transformers) for item in data]
    else:
        # Appliquer les transformateurs personnalisés
        for transformer_name, transformer_func in transformers.items():
            if transformer_name in str(schema):
                return transformer_func(data)
        return data


def validate_data_structure(data: Any, expected_type: Type, 
                          strict: bool = True) -> bool:
    """
    Valide la structure des données par rapport au type attendu
    
    Args:
        data: Données à valider
        expected_type: Type attendu
        strict: Mode strict de validation
        
    Returns:
        True si valide, False sinon
    """
    try:
        if issubclass(expected_type, BaseModel):
            expected_type.model_validate(data)
            return True
        elif isinstance(data, expected_type):
            return True
        elif not strict:
            # Tentative de conversion
            expected_type(data)
            return True
        return False
    except (ValidationError, ValueError, TypeError):
        return False


def normalize_data(data: Dict[str, Any], 
                  normalization_rules: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Normalise les données selon des règles prédéfinies
    
    Args:
        data: Dictionnaire de données
        normalization_rules: Règles de normalisation
        
    Returns:
        Données normalisées
    """
    if normalization_rules is None:
        normalization_rules = {
            'string': 'strip_lower',
            'email': 'strip_lower',
            'phone': 'digits_only',
            'name': 'title_case'
        }
    
    normalized = copy.deepcopy(data)
    
    for key, value in normalized.items():
        if isinstance(value, str):
            rule = normalization_rules.get(key, 'strip')
            
            if rule == 'strip_lower':
                normalized[key] = value.strip().lower()
            elif rule == 'strip_upper':
                normalized[key] = value.strip().upper()
            elif rule == 'title_case':
                normalized[key] = value.strip().title()
            elif rule == 'digits_only':
                normalized[key] = re.sub(r'\D', '', value)
            elif rule == 'strip':
                normalized[key] = value.strip()
        elif isinstance(value, dict):
            normalized[key] = normalize_data(value, normalization_rules)
    
    return normalized


def sanitize_input(data: Any, allowed_tags: List[str] = None, 
                  strip_attributes: bool = True) -> Any:
    """
    Sanitise les données d'entrée pour prévenir les attaques XSS
    
    Args:
        data: Données à sanitiser
        allowed_tags: Tags HTML autorisés
        strip_attributes: Supprimer les attributs HTML
        
    Returns:
        Données sanitisées
    """
    if allowed_tags is None:
        allowed_tags = ['b', 'i', 'em', 'strong', 'p', 'br']
    
    if isinstance(data, str):
        # Échapper les caractères HTML
        sanitized = html.escape(data)
        
        # Nettoyer avec bleach si disponible
        try:
            sanitized = bleach.clean(
                sanitized, 
                tags=allowed_tags,
                strip=strip_attributes
            )
        except:
            pass
        
        return sanitized
    elif isinstance(data, dict):
        return {key: sanitize_input(value, allowed_tags, strip_attributes) 
                for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item, allowed_tags, strip_attributes) 
                for item in data]
    else:
        return data


# =============================================================================
# MANIPULATION DE DICTIONNAIRES
# =============================================================================

def deep_merge(dict1: Dict, dict2: Dict, 
               merge_lists: bool = False) -> Dict:
    """
    Fusionne récursivement deux dictionnaires
    
    Args:
        dict1: Premier dictionnaire
        dict2: Second dictionnaire
        merge_lists: Fusionner les listes au lieu de les remplacer
        
    Returns:
        Dictionnaire fusionné
    """
    result = copy.deepcopy(dict1)
    
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value, merge_lists)
            elif isinstance(result[key], list) and isinstance(value, list) and merge_lists:
                result[key] = result[key] + value
            else:
                result[key] = value
        else:
            result[key] = value
    
    return result


def flatten_dict(nested_dict: Dict, separator: str = '.', 
                prefix: str = '') -> Dict:
    """
    Aplatit un dictionnaire imbriqué
    
    Args:
        nested_dict: Dictionnaire imbriqué
        separator: Séparateur pour les clés
        prefix: Préfixe pour les clés
        
    Returns:
        Dictionnaire aplati
    """
    flattened = {}
    
    for key, value in nested_dict.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, separator, new_key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flattened.update(flatten_dict(item, separator, f"{new_key}[{i}]"))
                else:
                    flattened[f"{new_key}[{i}]"] = item
        else:
            flattened[new_key] = value
    
    return flattened


def unflatten_dict(flattened_dict: Dict, separator: str = '.') -> Dict:
    """
    Reconstruit un dictionnaire à partir d'un dictionnaire aplati
    
    Args:
        flattened_dict: Dictionnaire aplati
        separator: Séparateur utilisé
        
    Returns:
        Dictionnaire reconstruit
    """
    result = {}
    
    for key, value in flattened_dict.items():
        keys = key.split(separator)
        current = result
        
        for i, k in enumerate(keys[:-1]):
            # Gérer les indices de liste
            if '[' in k and ']' in k:
                base_key, index_str = k.split('[')
                index = int(index_str.rstrip(']'))
                
                if base_key not in current:
                    current[base_key] = []
                
                # Étendre la liste si nécessaire
                while len(current[base_key]) <= index:
                    current[base_key].append({})
                
                current = current[base_key][index]
            else:
                if k not in current:
                    current[k] = {}
                current = current[k]
        
        # Définir la valeur finale
        final_key = keys[-1]
        if '[' in final_key and ']' in final_key:
            base_key, index_str = final_key.split('[')
            index = int(index_str.rstrip(']'))
            
            if base_key not in current:
                current[base_key] = []
            
            while len(current[base_key]) <= index:
                current[base_key].append(None)
            
            current[base_key][index] = value
        else:
            current[final_key] = value
    
    return result


# =============================================================================
# CONVERSION DE TYPES
# =============================================================================

def safe_cast(value: Any, target_type: Type, default: Any = None) -> Any:
    """
    Conversion sécurisée de type avec valeur par défaut
    
    Args:
        value: Valeur à convertir
        target_type: Type cible
        default: Valeur par défaut en cas d'erreur
        
    Returns:
        Valeur convertie ou valeur par défaut
    """
    try:
        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            return bool(value)
        elif target_type == datetime:
            if isinstance(value, str):
                # Essayer différents formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d',
                    '%d/%m/%Y',
                    '%d-%m-%Y'
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue
                raise ValueError("Unable to parse datetime")
            return target_type(value)
        else:
            return target_type(value)
    except (ValueError, TypeError):
        return default


def serialize_for_json(obj: Any) -> Any:
    """
    Sérialise un objet pour JSON en gérant les types complexes
    
    Args:
        obj: Objet à sérialiser
        
    Returns:
        Objet sérialisable en JSON
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, BaseModel):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


# =============================================================================
# VALIDATION AVANCÉE
# =============================================================================

def validate_schema(data: Dict, schema: Dict, strict: bool = True) -> tuple[bool, List[str]]:
    """
    Valide les données contre un schéma personnalisé
    
    Args:
        data: Données à valider
        schema: Schéma de validation
        strict: Mode strict
        
    Returns:
        Tuple (is_valid, errors)
    """
    errors = []
    
    for field, rules in schema.items():
        if field not in data:
            if rules.get('required', False):
                errors.append(f"Field '{field}' is required")
            continue
        
        value = data[field]
        field_type = rules.get('type')
        
        # Validation du type
        if field_type and not isinstance(value, field_type):
            errors.append(f"Field '{field}' must be of type {field_type.__name__}")
            continue
        
        # Validation de la longueur
        if 'min_length' in rules and len(str(value)) < rules['min_length']:
            errors.append(f"Field '{field}' must have at least {rules['min_length']} characters")
        
        if 'max_length' in rules and len(str(value)) > rules['max_length']:
            errors.append(f"Field '{field}' must have at most {rules['max_length']} characters")
        
        # Validation de la valeur
        if 'min_value' in rules and value < rules['min_value']:
            errors.append(f"Field '{field}' must be at least {rules['min_value']}")
        
        if 'max_value' in rules and value > rules['max_value']:
            errors.append(f"Field '{field}' must be at most {rules['max_value']}")
        
        # Validation par pattern
        if 'pattern' in rules and isinstance(value, str):
            if not re.match(rules['pattern'], value):
                errors.append(f"Field '{field}' doesn't match required pattern")
        
        # Validation personnalisée
        if 'validator' in rules:
            try:
                if not rules['validator'](value):
                    errors.append(f"Field '{field}' failed custom validation")
            except Exception as e:
                errors.append(f"Field '{field}' validation error: {str(e)}")
    
    return len(errors) == 0, errors


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "transform_data",
    "validate_data_structure", 
    "normalize_data",
    "sanitize_input",
    "deep_merge",
    "flatten_dict",
    "unflatten_dict",
    "safe_cast",
    "serialize_for_json",
    "validate_schema"
]
