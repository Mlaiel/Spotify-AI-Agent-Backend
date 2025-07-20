#!/usr/bin/env python3
"""
Générateur de templates Slack pour Alertmanager
Auteur: Fahed Mlaiel
Date: 2025-07-19

Ce script génère des templates Slack optimisés pour Alertmanager
avec support multi-tenant et internationalisation.
"""

import os
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, Template


class SlackTemplateGenerator:
    """Générateur de templates Slack pour alertes multi-tenant."""
    
    def __init__(self, base_path: str, locale: str = 'en'):
        """
        Initialise le générateur de templates.
        
        Args:
            base_path: Chemin de base des templates
            locale: Locale pour l'internationalisation
        """
        self.base_path = Path(base_path)
        self.locale = locale
        self.templates_dir = self.base_path / 'templates'
        self.locales_dir = self.base_path / 'locales'
        self.output_dir = self.base_path / 'generated'
        
        # Configuration Jinja2
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Charger les messages localisés
        self.messages = self._load_messages()
        
        # Créer le répertoire de sortie
        self.output_dir.mkdir(exist_ok=True)
    
    def _load_messages(self) -> Dict[str, Any]:
        """Charge les messages localisés."""
        messages_file = self.locales_dir / self.locale / 'messages.yaml'
        
        if not messages_file.exists():
            raise FileNotFoundError(f"Messages file not found: {messages_file}")
        
        with open(messages_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def generate_critical_template(self, tenant_config: Dict[str, Any]) -> str:
        """Génère un template pour alertes critiques."""
        template = self.env.get_template('critical_alert.j2')
        
        context = {
            'messages': self.messages,
            'tenant': tenant_config,
            'timestamp': datetime.now().isoformat(),
            'locale': self.locale
        }
        
        return template.render(**context)
    
    def generate_warning_template(self, tenant_config: Dict[str, Any]) -> str:
        """Génère un template pour alertes d'avertissement."""
        template = self.env.get_template('warning_alert.j2')
        
        context = {
            'messages': self.messages,
            'tenant': tenant_config,
            'timestamp': datetime.now().isoformat(),
            'locale': self.locale
        }
        
        return template.render(**context)
    
    def generate_info_template(self, tenant_config: Dict[str, Any]) -> str:
        """Génère un template pour alertes d'information."""
        template = self.env.get_template('info_alert.j2')
        
        context = {
            'messages': self.messages,
            'tenant': tenant_config,
            'timestamp': datetime.now().isoformat(),
            'locale': self.locale
        }
        
        return template.render(**context)
    
    def generate_resolved_template(self, tenant_config: Dict[str, Any]) -> str:
        """Génère un template pour alertes résolues."""
        template = self.env.get_template('resolved_alert.j2')
        
        context = {
            'messages': self.messages,
            'tenant': tenant_config,
            'timestamp': datetime.now().isoformat(),
            'locale': self.locale
        }
        
        return template.render(**context)
    
    def generate_all_templates(self, tenant_configs: List[Dict[str, Any]]) -> None:
        """Génère tous les templates pour tous les tenants."""
        print(f"Génération des templates Slack en locale '{self.locale}'...")
        
        for tenant_config in tenant_configs:
            tenant_id = tenant_config.get('id', 'default')
            tenant_output_dir = self.output_dir / tenant_id
            tenant_output_dir.mkdir(exist_ok=True)
            
            print(f"  Génération pour tenant: {tenant_id}")
            
            # Générer chaque type de template
            templates = {
                'critical': self.generate_critical_template(tenant_config),
                'warning': self.generate_warning_template(tenant_config),
                'info': self.generate_info_template(tenant_config),
                'resolved': self.generate_resolved_template(tenant_config)
            }
            
            # Sauvegarder les templates
            for template_type, content in templates.items():
                output_file = tenant_output_dir / f'{template_type}_slack_template.yaml'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    ✅ {template_type} -> {output_file}")
    
    def validate_template(self, template_content: str) -> bool:
        """Valide un template généré."""
        try:
            # Vérifier que c'est du YAML valide
            yaml.safe_load(template_content)
            
            # Vérifications spécifiques
            required_fields = ['text', 'attachments']
            for field in required_fields:
                if field not in template_content:
                    print(f"❌ Champ requis manquant: {field}")
                    return False
            
            print("✅ Template valide")
            return True
            
        except yaml.YAMLError as e:
            print(f"❌ Erreur YAML: {e}")
            return False
        except Exception as e:
            print(f"❌ Erreur de validation: {e}")
            return False
    
    def preview_template(self, template_type: str, tenant_config: Dict[str, Any]) -> None:
        """Affiche un aperçu du template."""
        generators = {
            'critical': self.generate_critical_template,
            'warning': self.generate_warning_template,
            'info': self.generate_info_template,
            'resolved': self.generate_resolved_template
        }
        
        if template_type not in generators:
            print(f"❌ Type de template inconnu: {template_type}")
            return
        
        template_content = generators[template_type](tenant_config)
        
        print(f"\n📋 Aperçu du template '{template_type}':")
        print("=" * 50)
        print(template_content)
        print("=" * 50)


def load_tenant_configs(config_file: str) -> List[Dict[str, Any]]:
    """Charge les configurations des tenants."""
    if not os.path.exists(config_file):
        print(f"❌ Fichier de configuration tenant non trouvé: {config_file}")
        return []
    
    with open(config_file, 'r', encoding='utf-8') as f:
        if config_file.endswith('.json'):
            return json.load(f)
        else:
            return yaml.safe_load(f)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Générateur de templates Slack pour Alertmanager"
    )
    
    parser.add_argument(
        '--base-path',
        default='/workspaces/Achiri/spotify-ai-agent/backend/app/tenancy/fixtures/templates/examples/config/tenant_templates/configs/environments/dev/manifests/monitoring/alerts/templates/alertmanager/receivers/templates/slack',
        help='Chemin de base des templates'
    )
    
    parser.add_argument(
        '--locale',
        default='en',
        choices=['en', 'fr'],
        help='Locale pour l\'internationalisation'
    )
    
    parser.add_argument(
        '--tenant-config',
        default='tenant_config.yaml',
        help='Fichier de configuration des tenants'
    )
    
    parser.add_argument(
        '--preview',
        choices=['critical', 'warning', 'info', 'resolved'],
        help='Prévisualiser un type de template'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Valider les templates générés'
    )
    
    args = parser.parse_args()
    
    # Initialiser le générateur
    generator = SlackTemplateGenerator(args.base_path, args.locale)
    
    # Charger la configuration des tenants
    tenant_configs = load_tenant_configs(args.tenant_config)
    
    if not tenant_configs:
        # Configuration par défaut
        tenant_configs = [{
            'id': 'default',
            'name': 'Spotify AI Agent',
            'environment': 'dev',
            'region': 'eu-west-1',
            'cluster': 'main',
            'slack_channel': '#alerts-dev',
            'webhook_url': 'https://hooks.slack.com/services/...',
            'escalation_policy': 'dev-team'
        }]
    
    # Mode prévisualisation
    if args.preview:
        generator.preview_template(args.preview, tenant_configs[0])
        return
    
    # Générer tous les templates
    generator.generate_all_templates(tenant_configs)
    
    # Validation si demandée
    if args.validate:
        print("\n🔍 Validation des templates générés...")
        for tenant_config in tenant_configs:
            tenant_id = tenant_config.get('id', 'default')
            tenant_output_dir = generator.output_dir / tenant_id
            
            for template_file in tenant_output_dir.glob('*_slack_template.yaml'):
                print(f"  Validation: {template_file}")
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                generator.validate_template(content)
    
    print("\n✅ Génération terminée avec succès!")


if __name__ == '__main__':
    main()
