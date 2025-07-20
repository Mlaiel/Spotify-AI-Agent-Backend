"""
Générateur de configuration avancé avec moteur de templates.

Ce module fournit un système complet de génération de configuration
basé sur des templates Jinja2 avec validation intégrée.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from pydantic import BaseModel, ValidationError
from ..schemas.tenant_schemas import TenantConfigSchema
from ..schemas.monitoring_schemas import MonitoringConfigSchema
from ..schemas.alert_schemas import AlertManagerConfigSchema


class TemplateEngine:
    """Moteur de template Jinja2 avancé."""
    
    def __init__(self, template_dirs: List[str], enable_autoescape: bool = True):
        """Initialise le moteur de template."""
        self.template_dirs = template_dirs
        self.env = Environment(
            loader=FileSystemLoader(template_dirs),
            autoescape=select_autoescape(['html', 'xml']) if enable_autoescape else False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Ajout de filtres personnalisés
        self.env.filters.update({
            'to_yaml': self._to_yaml,
            'to_json': self._to_json,
            'merge_dict': self._merge_dict,
            'format_duration': self._format_duration,
            'generate_password': self._generate_password,
            'base64_encode': self._base64_encode,
            'base64_decode': self._base64_decode
        })
        
        # Ajout de fonctions globales
        self.env.globals.update({
            'now': datetime.now,
            'env_var': os.getenv,
            'file_exists': os.path.exists,
            'random_string': self._random_string
        })
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Rend un template avec les variables fournies."""
        try:
            template = self.env.get_template(template_name)
            return template.render(**variables)
        except Exception as e:
            raise RuntimeError(f"Erreur de rendu du template {template_name}: {e}")
    
    def render_string(self, template_string: str, variables: Dict[str, Any]) -> str:
        """Rend une chaîne template avec les variables fournies."""
        try:
            template = self.env.from_string(template_string)
            return template.render(**variables)
        except Exception as e:
            raise RuntimeError(f"Erreur de rendu de la chaîne template: {e}")
    
    @staticmethod
    def _to_yaml(value: Any) -> str:
        """Filtre pour convertir en YAML."""
        return yaml.dump(value, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def _to_json(value: Any, indent: int = 2) -> str:
        """Filtre pour convertir en JSON."""
        return json.dumps(value, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def _merge_dict(dict1: Dict, dict2: Dict) -> Dict:
        """Filtre pour fusionner deux dictionnaires."""
        result = dict1.copy()
        result.update(dict2)
        return result
    
    @staticmethod
    def _format_duration(seconds: int) -> str:
        """Filtre pour formater une durée."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        else:
            return f"{seconds // 3600}h"
    
    @staticmethod
    def _generate_password(length: int = 16) -> str:
        """Génère un mot de passe aléatoire."""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def _base64_encode(value: str) -> str:
        """Encode une chaîne en base64."""
        import base64
        return base64.b64encode(value.encode()).decode()
    
    @staticmethod
    def _base64_decode(value: str) -> str:
        """Décode une chaîne base64."""
        import base64
        return base64.b64decode(value.encode()).decode()
    
    @staticmethod
    def _random_string(length: int = 8) -> str:
        """Génère une chaîne aléatoire."""
        import secrets
        import string
        return ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(length))


class ConfigGenerator:
    """Générateur de configuration avec validation intégrée."""
    
    def __init__(
        self,
        template_engine: TemplateEngine,
        output_dir: str = "./generated_configs",
        validation_enabled: bool = True
    ):
        """Initialise le générateur de configuration."""
        self.template_engine = template_engine
        self.output_dir = Path(output_dir)
        self.validation_enabled = validation_enabled
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping des types de configuration vers leurs schémas
        self.schema_mapping = {
            'tenant': TenantConfigSchema,
            'monitoring': MonitoringConfigSchema,
            'alertmanager': AlertManagerConfigSchema
        }
    
    def generate_config(
        self,
        config_type: str,
        template_name: str,
        variables: Dict[str, Any],
        output_filename: Optional[str] = None,
        validate: bool = None
    ) -> Dict[str, Any]:
        """Génère une configuration complète."""
        validate = validate if validate is not None else self.validation_enabled
        
        # Rendu du template
        config_content = self.template_engine.render_template(template_name, variables)
        
        # Parsing du contenu généré
        if template_name.endswith('.yaml') or template_name.endswith('.yml'):
            config_data = yaml.safe_load(config_content)
        elif template_name.endswith('.json'):
            config_data = json.loads(config_content)
        else:
            raise ValueError(f"Format de template non supporté: {template_name}")
        
        # Validation si activée
        if validate and config_type in self.schema_mapping:
            self._validate_config(config_data, self.schema_mapping[config_type])
        
        # Sauvegarde si nom de fichier fourni
        if output_filename:
            self._save_config(config_data, output_filename)
        
        return config_data
    
    def generate_tenant_config(
        self,
        tenant_id: str,
        environment: str,
        variables: Dict[str, Any],
        template_name: str = "tenant/base.yaml.j2"
    ) -> TenantConfigSchema:
        """Génère une configuration tenant spécifique."""
        # Variables par défaut pour les tenants
        default_variables = {
            'tenant_id': tenant_id,
            'environment': environment,
            'timestamp': datetime.now().isoformat(),
            'generator_version': '1.0.0'
        }
        
        # Fusion des variables
        merged_variables = {**default_variables, **variables}
        
        # Génération
        config_data = self.generate_config(
            config_type='tenant',
            template_name=template_name,
            variables=merged_variables,
            output_filename=f"tenant_{tenant_id}_{environment}.yaml"
        )
        
        return TenantConfigSchema(**config_data)
    
    def generate_monitoring_config(
        self,
        tenant_id: str,
        environment: str,
        variables: Dict[str, Any],
        template_name: str = "monitoring/prometheus.yaml.j2"
    ) -> MonitoringConfigSchema:
        """Génère une configuration de monitoring."""
        default_variables = {
            'tenant_id': tenant_id,
            'environment': environment,
            'metrics_retention': '15d',
            'scrape_interval': '30s'
        }
        
        merged_variables = {**default_variables, **variables}
        
        config_data = self.generate_config(
            config_type='monitoring',
            template_name=template_name,
            variables=merged_variables,
            output_filename=f"monitoring_{tenant_id}_{environment}.yaml"
        )
        
        return MonitoringConfigSchema(**config_data)
    
    def generate_alertmanager_config(
        self,
        tenant_id: str,
        environment: str,
        variables: Dict[str, Any],
        template_name: str = "alerting/alertmanager.yaml.j2"
    ) -> AlertManagerConfigSchema:
        """Génère une configuration AlertManager."""
        default_variables = {
            'tenant_id': tenant_id,
            'environment': environment,
            'smtp_from': f'alerts-{environment}@company.com',
            'slack_api_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        }
        
        merged_variables = {**default_variables, **variables}
        
        config_data = self.generate_config(
            config_type='alertmanager',
            template_name=template_name,
            variables=merged_variables,
            output_filename=f"alertmanager_{tenant_id}_{environment}.yaml"
        )
        
        return AlertManagerConfigSchema(**config_data)
    
    def generate_bulk_configs(
        self,
        config_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Génère plusieurs configurations en lot."""
        results = {}
        
        for spec in config_specs:
            try:
                config_type = spec['type']
                method_name = f"generate_{config_type}_config"
                
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    result = method(
                        tenant_id=spec['tenant_id'],
                        environment=spec['environment'],
                        variables=spec.get('variables', {}),
                        template_name=spec.get('template_name')
                    )
                    results[f"{config_type}_{spec['tenant_id']}_{spec['environment']}"] = result
                else:
                    results[f"error_{spec['tenant_id']}"] = f"Type de config non supporté: {config_type}"
                    
            except Exception as e:
                results[f"error_{spec.get('tenant_id', 'unknown')}"] = str(e)
        
        return results
    
    def _validate_config(self, config_data: Dict[str, Any], schema_class: BaseModel):
        """Valide une configuration avec son schéma Pydantic."""
        try:
            schema_class(**config_data)
        except ValidationError as e:
            raise ValueError(f"Erreur de validation de configuration: {e}")
    
    def _save_config(self, config_data: Dict[str, Any], filename: str):
        """Sauvegarde une configuration dans un fichier."""
        file_path = self.output_dir / filename
        
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        elif filename.endswith('.json'):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Format de fichier non supporté: {filename}")
    
    def list_templates(self) -> List[str]:
        """Liste tous les templates disponibles."""
        templates = []
        for template_dir in self.template_engine.template_dirs:
            template_path = Path(template_dir)
            if template_path.exists():
                for template_file in template_path.rglob('*.j2'):
                    relative_path = template_file.relative_to(template_path)
                    templates.append(str(relative_path))
        return sorted(templates)
    
    def get_template_variables(self, template_name: str) -> List[str]:
        """Extrait les variables utilisées dans un template."""
        try:
            template_source = self.template_engine.env.get_template(template_name).source
            parsed = self.template_engine.env.parse(template_source)
            
            variables = set()
            for node in parsed.find_all():
                if hasattr(node, 'name') and isinstance(node.name, str):
                    variables.add(node.name)
            
            return sorted(list(variables))
        except Exception as e:
            raise RuntimeError(f"Erreur d'analyse du template {template_name}: {e}")


# Factory function pour créer un générateur configuré
def create_config_generator(
    template_dirs: List[str],
    output_dir: str = "./generated_configs",
    validation_enabled: bool = True
) -> ConfigGenerator:
    """Factory pour créer un générateur de configuration."""
    template_engine = TemplateEngine(template_dirs)
    return ConfigGenerator(template_engine, output_dir, validation_enabled)


# Configuration par défaut pour l'environnement de développement
DEFAULT_TEMPLATE_DIRS = [
    "/templates/tenant",
    "/templates/monitoring", 
    "/templates/alerting",
    "/templates/security",
    "/templates/common"
]


def get_default_generator() -> ConfigGenerator:
    """Retourne un générateur avec la configuration par défaut."""
    return create_config_generator(DEFAULT_TEMPLATE_DIRS)
