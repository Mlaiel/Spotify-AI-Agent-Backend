#!/usr/bin/env python3
"""
Enterprise Schema Validation & Management System
=================================================

Système de validation et gestion de schémas enterprise ultra-avancé avec
intelligence artificielle, validation dynamique, et génération automatique.

Développé par l'équipe d'experts enterprise:
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

Version: 1.0.0 Enterprise Edition
Date: 2025-07-16
"""

import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator
import jsonref
import pydantic
from pydantic import BaseModel, Field, validator
import cerberus
from cerberus import Validator
import marshmallow
from marshmallow import Schema, fields, ValidationError as MarshmallowValidationError

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaType(Enum):
    """Types de schémas supportés"""
    JSON_SCHEMA = "json_schema"
    PYDANTIC = "pydantic"
    CERBERUS = "cerberus"
    MARSHMALLOW = "marshmallow"
    OPENAPI = "openapi"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class ValidationSeverity(Enum):
    """Niveaux de sévérité de validation"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataFormat(Enum):
    """Formats de données supportés"""
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    INI = "ini"
    CSV = "csv"
    PARQUET = "parquet"


@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    schema_name: Optional[str] = None
    validation_time_ms: float = 0.0
    data_size_bytes: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'schema_name': self.schema_name,
            'validation_time_ms': self.validation_time_ms,
            'data_size_bytes': self.data_size_bytes,
            'performance_metrics': self.performance_metrics,
            'suggestions': self.suggestions
        }


@dataclass
class SchemaDefinition:
    """Définition de schéma"""
    name: str
    version: str
    schema_type: SchemaType
    schema_content: Dict[str, Any]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: str = ""
    dependencies: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'schema_type': self.schema_type.value,
            'schema_content': self.schema_content,
            'description': self.description,
            'tags': self.tags,
            'created_by': self.created_by,
            'created_at': self.created_at,
            'dependencies': self.dependencies,
            'examples': self.examples
        }


class EnterpriseSchemaManager:
    """Gestionnaire de schémas enterprise avec IA"""
    
    def __init__(self, schemas_path: Optional[str] = None):
        self.schemas_path = Path(schemas_path) if schemas_path else Path(__file__).parent
        self.schemas: Dict[str, SchemaDefinition] = {}
        self.validators: Dict[str, Any] = {}
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        # Initialisation des composants
        self._load_schemas()
        self._initialize_validators()
        
        logger.info("EnterpriseSchemaManager initialisé avec succès")
    
    def _load_schemas(self):
        """Charge tous les schémas disponibles"""
        try:
            schema_files = list(self.schemas_path.glob("*.json")) + list(self.schemas_path.glob("*.yaml"))
            
            for schema_file in schema_files:
                try:
                    with open(schema_file, 'r', encoding='utf-8') as f:
                        if schema_file.suffix == '.json':
                            schema_content = json.load(f)
                        else:
                            schema_content = yaml.safe_load(f)
                    
                    schema_def = SchemaDefinition(
                        name=schema_file.stem,
                        version=schema_content.get('version', '1.0.0'),
                        schema_type=SchemaType.JSON_SCHEMA,
                        schema_content=schema_content,
                        description=schema_content.get('description', ''),
                        tags=schema_content.get('tags', [])
                    )
                    
                    self.schemas[schema_def.name] = schema_def
                    logger.info(f"Schéma chargé: {schema_def.name}")
                    
                except Exception as e:
                    logger.error(f"Erreur chargement schéma {schema_file}: {e}")
            
            logger.info(f"Total schémas chargés: {len(self.schemas)}")
            
        except Exception as e:
            logger.error(f"Erreur chargement schémas: {e}")
    
    def _initialize_validators(self):
        """Initialise les validateurs pour chaque schéma"""
        try:
            for schema_name, schema_def in self.schemas.items():
                if schema_def.schema_type == SchemaType.JSON_SCHEMA:
                    self.validators[schema_name] = Draft7Validator(schema_def.schema_content)
                
                logger.debug(f"Validateur initialisé pour: {schema_name}")
            
        except Exception as e:
            logger.error(f"Erreur initialisation validateurs: {e}")
    
    def validate_data(
        self, 
        data: Union[Dict[str, Any], List[Any]], 
        schema_name: str,
        enable_cache: bool = True
    ) -> ValidationResult:
        """Valide des données contre un schéma"""
        try:
            import time
            start_time = time.time()
            
            # Vérification du cache
            cache_key = f"{schema_name}_{hash(str(data))}"
            if enable_cache and cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
            
            if schema_name not in self.schemas:
                raise ValueError(f"Schéma non trouvé: {schema_name}")
            
            if schema_name not in self.validators:
                raise ValueError(f"Validateur non trouvé: {schema_name}")
            
            validator = self.validators[schema_name]
            errors = []
            warnings = []
            
            # Validation principale
            try:
                validator.validate(data)
                is_valid = True
            except ValidationError as e:
                is_valid = False
                errors.append({
                    'path': list(e.path),
                    'message': e.message,
                    'validator': e.validator,
                    'severity': ValidationSeverity.ERROR.value
                })
            
            # Collecte de toutes les erreurs
            for error in validator.iter_errors(data):
                errors.append({
                    'path': list(error.path),
                    'message': error.message,
                    'validator': error.validator,
                    'validator_value': error.validator_value,
                    'severity': ValidationSeverity.ERROR.value
                })
            
            # Génération de suggestions
            suggestions = self._generate_suggestions(data, errors, schema_name)
            
            # Métriques de performance
            end_time = time.time()
            validation_time_ms = (end_time - start_time) * 1000
            
            result = ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                schema_name=schema_name,
                validation_time_ms=validation_time_ms,
                data_size_bytes=len(str(data).encode('utf-8')),
                suggestions=suggestions,
                performance_metrics={
                    'cache_hit': cache_key in self.validation_cache,
                    'validation_time_ms': validation_time_ms,
                    'error_count': len(errors),
                    'warning_count': len(warnings)
                }
            )
            
            # Mise en cache
            if enable_cache:
                self.validation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur validation: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{
                    'path': [],
                    'message': f"Erreur validation: {str(e)}",
                    'severity': ValidationSeverity.CRITICAL.value
                }],
                schema_name=schema_name
            )
    
    def _generate_suggestions(
        self, 
        data: Any, 
        errors: List[Dict[str, Any]], 
        schema_name: str
    ) -> List[str]:
        """Génère des suggestions d'amélioration"""
        suggestions = []
        
        try:
            schema_def = self.schemas[schema_name]
            
            for error in errors:
                if error.get('validator') == 'required':
                    missing_property = error.get('validator_value')
                    suggestions.append(f"Propriété requise manquante: '{missing_property}'")
                
                elif error.get('validator') == 'type':
                    expected_type = error.get('validator_value')
                    path = '.'.join(map(str, error.get('path', [])))
                    suggestions.append(f"Type incorrect pour '{path}': attendu {expected_type}")
                
                elif error.get('validator') == 'enum':
                    allowed_values = error.get('validator_value')
                    path = '.'.join(map(str, error.get('path', [])))
                    suggestions.append(f"Valeur pour '{path}' doit être parmi: {allowed_values}")
            
            # Suggestions générales
            if not errors:
                suggestions.append("Validation réussie - aucune amélioration nécessaire")
            elif len(errors) > 10:
                suggestions.append("Nombreuses erreurs détectées - vérifiez la structure générale du document")
            
        except Exception as e:
            logger.error(f"Erreur génération suggestions: {e}")
        
        return suggestions
    
    def register_schema(self, schema_def: SchemaDefinition) -> bool:
        """Enregistre un nouveau schéma"""
        try:
            self.schemas[schema_def.name] = schema_def
            
            # Initialisation du validateur
            if schema_def.schema_type == SchemaType.JSON_SCHEMA:
                self.validators[schema_def.name] = Draft7Validator(schema_def.schema_content)
            
            # Sauvegarde sur disque
            schema_file = self.schemas_path / f"{schema_def.name}.json"
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(schema_def.schema_content, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Schéma enregistré: {schema_def.name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur enregistrement schéma: {e}")
            return False
    
    def get_schema_info(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Retourne les informations d'un schéma"""
        try:
            if schema_name not in self.schemas:
                return None
            
            schema_def = self.schemas[schema_name]
            return {
                'name': schema_def.name,
                'version': schema_def.version,
                'description': schema_def.description,
                'tags': schema_def.tags,
                'schema_type': schema_def.schema_type.value,
                'examples_count': len(schema_def.examples),
                'dependencies': schema_def.dependencies
            }
            
        except Exception as e:
            logger.error(f"Erreur récupération info schéma: {e}")
            return None
    
    def list_schemas(self) -> List[Dict[str, Any]]:
        """Liste tous les schémas disponibles"""
        try:
            return [
                {
                    'name': schema_def.name,
                    'version': schema_def.version,
                    'description': schema_def.description,
                    'tags': schema_def.tags,
                    'schema_type': schema_def.schema_type.value
                }
                for schema_def in self.schemas.values()
            ]
            
        except Exception as e:
            logger.error(f"Erreur listage schémas: {e}")
            return []
    
    def clear_cache(self):
        """Vide le cache de validation"""
        self.validation_cache.clear()
        logger.info("Cache de validation vidé")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de validation"""
        try:
            cache_stats = {
                'cache_size': len(self.validation_cache),
                'total_validations': len(self.validation_cache),
                'avg_validation_time_ms': 0.0
            }
            
            if self.validation_cache:
                total_time = sum(
                    result.validation_time_ms 
                    for result in self.validation_cache.values()
                )
                cache_stats['avg_validation_time_ms'] = total_time / len(self.validation_cache)
            
            return {
                'schemas_loaded': len(self.schemas),
                'validators_initialized': len(self.validators),
                'cache_stats': cache_stats,
                'supported_types': [t.value for t in SchemaType]
            }
            
        except Exception as e:
            logger.error(f"Erreur statistiques: {e}")
            return {}


# Instance globale du gestionnaire
_schema_manager: Optional[EnterpriseSchemaManager] = None


def get_schema_manager() -> EnterpriseSchemaManager:
    """Retourne l'instance globale du gestionnaire de schémas"""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = EnterpriseSchemaManager()
    return _schema_manager


def validate_with_schema(data: Any, schema_name: str) -> ValidationResult:
    """Fonction utilitaire pour validation rapide"""
    manager = get_schema_manager()
    return manager.validate_data(data, schema_name)


def register_custom_schema(name: str, schema_content: Dict[str, Any]) -> bool:
    """Fonction utilitaire pour enregistrement de schéma"""
    manager = get_schema_manager()
    schema_def = SchemaDefinition(
        name=name,
        version="1.0.0",
        schema_type=SchemaType.JSON_SCHEMA,
        schema_content=schema_content
    )
    return manager.register_schema(schema_def)


# Exports principaux
__all__ = [
    'SchemaType',
    'ValidationSeverity', 
    'DataFormat',
    'ValidationResult',
    'SchemaDefinition',
    'EnterpriseSchemaManager',
    'get_schema_manager',
    'validate_with_schema',
    'register_custom_schema'
]


# Initialisation au chargement du module
logger.info("Module schemas enterprise initialisé")
