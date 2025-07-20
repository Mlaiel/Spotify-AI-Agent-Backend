"""
Gestionnaire de schémas et registre centralisé.

Ce module fournit un système de gestion centralisée des schémas
avec versioning, mise en cache et synchronisation.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Union
from datetime import datetime, timedelta
from pydantic import BaseModel
from ..schemas import *


class SchemaVersion:
    """Gestion des versions de schémas."""
    
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor  
        self.patch = patch
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __lt__(self, other: 'SchemaVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: 'SchemaVersion') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def is_compatible(self, other: 'SchemaVersion') -> bool:
        """Vérifie la compatibilité entre versions (même major)."""
        return self.major == other.major


class SchemaMetadata:
    """Métadonnées d'un schéma."""
    
    def __init__(
        self,
        name: str,
        schema_class: Type[BaseModel],
        version: SchemaVersion,
        description: str = "",
        tags: List[str] = None,
        deprecated: bool = False,
        migration_path: Optional[str] = None
    ):
        self.name = name
        self.schema_class = schema_class
        self.version = version
        self.description = description
        self.tags = tags or []
        self.deprecated = deprecated
        self.migration_path = migration_path
        self.created_at = datetime.now()
        self.updated_at = datetime.now()


class SchemaRegistry:
    """Registre centralisé des schémas avec versioning."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialise le registre de schémas."""
        self.schemas: Dict[str, Dict[str, SchemaMetadata]] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".schema_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache en mémoire
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=1)
        
        # Initialisation avec les schémas par défaut
        self._register_default_schemas()
    
    def register_schema(
        self,
        name: str,
        schema_class: Type[BaseModel],
        version: SchemaVersion,
        description: str = "",
        tags: List[str] = None,
        deprecated: bool = False,
        migration_path: Optional[str] = None
    ):
        """Enregistre un nouveau schéma."""
        metadata = SchemaMetadata(
            name=name,
            schema_class=schema_class,
            version=version,
            description=description,
            tags=tags,
            deprecated=deprecated,
            migration_path=migration_path
        )
        
        if name not in self.schemas:
            self.schemas[name] = {}
        
        version_key = str(version)
        self.schemas[name][version_key] = metadata
        
        # Sauvegarde en cache
        self._save_to_cache(name, version_key, metadata)
    
    def get_schema(
        self,
        name: str,
        version: Optional[Union[str, SchemaVersion]] = None
    ) -> Optional[Type[BaseModel]]:
        """Récupère un schéma par nom et version."""
        if name not in self.schemas:
            return None
        
        if version is None:
            # Récupère la dernière version non dépréciée
            latest_version = self._get_latest_version(name, include_deprecated=False)
            if latest_version:
                return self.schemas[name][latest_version].schema_class
            return None
        
        version_key = str(version)
        if version_key in self.schemas[name]:
            return self.schemas[name][version_key].schema_class
        
        return None
    
    def get_schema_metadata(
        self,
        name: str,
        version: Optional[Union[str, SchemaVersion]] = None
    ) -> Optional[SchemaMetadata]:
        """Récupère les métadonnées d'un schéma."""
        if name not in self.schemas:
            return None
        
        if version is None:
            latest_version = self._get_latest_version(name)
            if latest_version:
                return self.schemas[name][latest_version]
            return None
        
        version_key = str(version)
        return self.schemas[name].get(version_key)
    
    def list_schemas(
        self,
        tags: Optional[List[str]] = None,
        include_deprecated: bool = False
    ) -> List[SchemaMetadata]:
        """Liste tous les schémas avec filtrage optionnel."""
        results = []
        
        for schema_name, versions in self.schemas.items():
            latest_version = self._get_latest_version(schema_name, include_deprecated)
            if latest_version:
                metadata = versions[latest_version]
                
                # Filtrage par tags
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue
                
                # Filtrage par dépréciation
                if not include_deprecated and metadata.deprecated:
                    continue
                
                results.append(metadata)
        
        return sorted(results, key=lambda x: x.name)
    
    def list_versions(self, name: str) -> List[SchemaVersion]:
        """Liste toutes les versions d'un schéma."""
        if name not in self.schemas:
            return []
        
        versions = []
        for version_str in self.schemas[name].keys():
            major, minor, patch = map(int, version_str.split('.'))
            versions.append(SchemaVersion(major, minor, patch))
        
        return sorted(versions)
    
    def deprecate_schema(self, name: str, version: Union[str, SchemaVersion]):
        """Marque un schéma comme déprécié."""
        version_key = str(version)
        if name in self.schemas and version_key in self.schemas[name]:
            self.schemas[name][version_key].deprecated = True
            self.schemas[name][version_key].updated_at = datetime.now()
    
    def migrate_schema(
        self,
        data: Dict[str, Any],
        from_schema: str,
        to_schema: str,
        from_version: Optional[Union[str, SchemaVersion]] = None,
        to_version: Optional[Union[str, SchemaVersion]] = None
    ) -> Dict[str, Any]:
        """Migre des données d'un schéma vers un autre."""
        # Récupération des schémas source et cible
        source_metadata = self.get_schema_metadata(from_schema, from_version)
        target_metadata = self.get_schema_metadata(to_schema, to_version)
        
        if not source_metadata or not target_metadata:
            raise ValueError("Schéma source ou cible introuvable")
        
        # Validation des données source
        try:
            source_instance = source_metadata.schema_class(**data)
        except Exception as e:
            raise ValueError(f"Données invalides pour le schéma source: {e}")
        
        # Migration des données
        migrated_data = self._perform_migration(
            source_instance.dict(),
            source_metadata,
            target_metadata
        )
        
        # Validation des données migrées
        try:
            target_metadata.schema_class(**migrated_data)
        except Exception as e:
            raise ValueError(f"Données migrées invalides: {e}")
        
        return migrated_data
    
    def validate_compatibility(
        self,
        schema_name: str,
        from_version: Union[str, SchemaVersion],
        to_version: Union[str, SchemaVersion]
    ) -> bool:
        """Vérifie la compatibilité entre deux versions d'un schéma."""
        if isinstance(from_version, str):
            major, minor, patch = map(int, from_version.split('.'))
            from_version = SchemaVersion(major, minor, patch)
        
        if isinstance(to_version, str):
            major, minor, patch = map(int, to_version.split('.'))
            to_version = SchemaVersion(major, minor, patch)
        
        return from_version.is_compatible(to_version)
    
    def export_registry(self, file_path: str):
        """Exporte le registre vers un fichier."""
        export_data = {
            'schemas': {},
            'exported_at': datetime.now().isoformat()
        }
        
        for schema_name, versions in self.schemas.items():
            export_data['schemas'][schema_name] = {}
            for version_key, metadata in versions.items():
                export_data['schemas'][schema_name][version_key] = {
                    'name': metadata.name,
                    'version': str(metadata.version),
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'deprecated': metadata.deprecated,
                    'migration_path': metadata.migration_path,
                    'created_at': metadata.created_at.isoformat(),
                    'updated_at': metadata.updated_at.isoformat()
                }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _register_default_schemas(self):
        """Enregistre les schémas par défaut du système."""
        # Schémas d'alertes
        self.register_schema(
            "alert_rule",
            AlertRuleSchema,
            SchemaVersion(1, 0, 0),
            "Schéma de règle d'alerte",
            ["alerting", "monitoring"]
        )
        
        self.register_schema(
            "alert_manager_config",
            AlertManagerConfigSchema,
            SchemaVersion(1, 0, 0),
            "Configuration AlertManager",
            ["alerting", "configuration"]
        )
        
        # Schémas de monitoring
        self.register_schema(
            "monitoring_config",
            MonitoringConfigSchema,
            SchemaVersion(1, 0, 0),
            "Configuration de monitoring",
            ["monitoring", "configuration"]
        )
        
        self.register_schema(
            "prometheus_config",
            PrometheusConfigSchema,
            SchemaVersion(1, 0, 0),
            "Configuration Prometheus",
            ["monitoring", "prometheus"]
        )
        
        # Schémas Slack
        self.register_schema(
            "slack_config",
            SlackConfigSchema,
            SchemaVersion(1, 0, 0),
            "Configuration Slack",
            ["slack", "notification"]
        )
        
        self.register_schema(
            "slack_message",
            SlackMessageSchema,
            SchemaVersion(1, 0, 0),
            "Message Slack",
            ["slack", "messaging"]
        )
        
        # Schémas de tenant
        self.register_schema(
            "tenant_config",
            TenantConfigSchema,
            SchemaVersion(1, 0, 0),
            "Configuration tenant",
            ["tenant", "configuration"]
        )
        
        # Schémas de validation
        self.register_schema(
            "validation_result",
            ValidationResultSchema,
            SchemaVersion(1, 0, 0),
            "Résultat de validation",
            ["validation", "result"]
        )
    
    def _get_latest_version(
        self,
        name: str,
        include_deprecated: bool = True
    ) -> Optional[str]:
        """Récupère la dernière version d'un schéma."""
        if name not in self.schemas:
            return None
        
        versions = []
        for version_str, metadata in self.schemas[name].items():
            if not include_deprecated and metadata.deprecated:
                continue
            major, minor, patch = map(int, version_str.split('.'))
            versions.append((SchemaVersion(major, minor, patch), version_str))
        
        if not versions:
            return None
        
        return max(versions, key=lambda x: x[0])[1]
    
    def _perform_migration(
        self,
        data: Dict[str, Any],
        source_metadata: SchemaMetadata,
        target_metadata: SchemaMetadata
    ) -> Dict[str, Any]:
        """Effectue la migration entre deux schémas."""
        # Migration simple : copie des champs compatibles
        migrated_data = {}
        
        source_fields = set(source_metadata.schema_class.__fields__.keys())
        target_fields = set(target_metadata.schema_class.__fields__.keys())
        
        # Copie des champs communs
        common_fields = source_fields & target_fields
        for field in common_fields:
            if field in data:
                migrated_data[field] = data[field]
        
        # Application de règles de migration personnalisées
        if source_metadata.migration_path:
            # Ici on pourrait charger et appliquer un script de migration
            pass
        
        return migrated_data
    
    def _save_to_cache(self, name: str, version: str, metadata: SchemaMetadata):
        """Sauvegarde les métadonnées en cache."""
        cache_key = f"{name}_{version}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception:
            # Échec silencieux du cache
            pass


class SchemaManager:
    """Gestionnaire de schémas avec fonctionnalités avancées."""
    
    def __init__(self, registry: Optional[SchemaRegistry] = None):
        """Initialise le gestionnaire de schémas."""
        self.registry = registry or SchemaRegistry()
        self.validators = {}
        self.transformers = {}
    
    def register_validator(self, schema_name: str, validator_func: callable):
        """Enregistre un validateur personnalisé pour un schéma."""
        self.validators[schema_name] = validator_func
    
    def register_transformer(self, schema_name: str, transformer_func: callable):
        """Enregistre un transformateur pour un schéma."""
        self.transformers[schema_name] = transformer_func
    
    def validate_data(
        self,
        data: Dict[str, Any],
        schema_name: str,
        version: Optional[Union[str, SchemaVersion]] = None
    ) -> bool:
        """Valide des données contre un schéma."""
        schema_class = self.registry.get_schema(schema_name, version)
        if not schema_class:
            raise ValueError(f"Schéma non trouvé: {schema_name}")
        
        try:
            # Validation Pydantic
            schema_class(**data)
            
            # Validation personnalisée
            if schema_name in self.validators:
                return self.validators[schema_name](data)
            
            return True
        except Exception:
            return False
    
    def transform_data(
        self,
        data: Dict[str, Any],
        schema_name: str,
        version: Optional[Union[str, SchemaVersion]] = None
    ) -> Dict[str, Any]:
        """Transforme des données selon un schéma."""
        if schema_name in self.transformers:
            data = self.transformers[schema_name](data)
        
        # Validation après transformation
        if self.validate_data(data, schema_name, version):
            return data
        else:
            raise ValueError("Données invalides après transformation")
    
    def get_schema_diff(
        self,
        schema_name: str,
        from_version: Union[str, SchemaVersion],
        to_version: Union[str, SchemaVersion]
    ) -> Dict[str, Any]:
        """Compare deux versions d'un schéma."""
        from_schema = self.registry.get_schema(schema_name, from_version)
        to_schema = self.registry.get_schema(schema_name, to_version)
        
        if not from_schema or not to_schema:
            raise ValueError("Une des versions du schéma est introuvable")
        
        from_fields = set(from_schema.__fields__.keys())
        to_fields = set(to_schema.__fields__.keys())
        
        return {
            'added_fields': list(to_fields - from_fields),
            'removed_fields': list(from_fields - to_fields),
            'common_fields': list(from_fields & to_fields),
            'from_version': str(from_version),
            'to_version': str(to_version)
        }
    
    def generate_documentation(self, schema_name: str) -> str:
        """Génère la documentation d'un schéma."""
        metadata = self.registry.get_schema_metadata(schema_name)
        if not metadata:
            raise ValueError(f"Schéma non trouvé: {schema_name}")
        
        schema_class = metadata.schema_class
        
        doc = f"# Schéma: {metadata.name}\n\n"
        doc += f"**Version:** {metadata.version}\n\n"
        doc += f"**Description:** {metadata.description}\n\n"
        doc += f"**Tags:** {', '.join(metadata.tags)}\n\n"
        
        if metadata.deprecated:
            doc += "⚠️ **DÉPRÉCIÉ** - Ce schéma est déprécié\n\n"
        
        doc += "## Champs\n\n"
        
        for field_name, field in schema_class.__fields__.items():
            doc += f"### {field_name}\n"
            doc += f"- **Type:** {field.type_}\n"
            doc += f"- **Requis:** {'Oui' if field.required else 'Non'}\n"
            if field.default is not None:
                doc += f"- **Défaut:** {field.default}\n"
            if hasattr(field.field_info, 'description'):
                doc += f"- **Description:** {field.field_info.description}\n"
            doc += "\n"
        
        return doc


# Factory functions
def create_schema_registry(cache_dir: Optional[str] = None) -> SchemaRegistry:
    """Crée un registre de schémas."""
    return SchemaRegistry(cache_dir)


def create_schema_manager(registry: Optional[SchemaRegistry] = None) -> SchemaManager:
    """Crée un gestionnaire de schémas."""
    return SchemaManager(registry)


def get_default_schema_manager() -> SchemaManager:
    """Retourne un gestionnaire avec la configuration par défaut."""
    return SchemaManager()
