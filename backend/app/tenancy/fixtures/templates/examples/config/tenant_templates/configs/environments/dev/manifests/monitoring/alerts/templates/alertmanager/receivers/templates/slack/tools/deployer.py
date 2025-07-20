#!/usr/bin/env python3
"""
Déployeur de templates Slack pour Alertmanager
Auteur: Fahed Mlaiel
Date: 2025-07-19

Ce script déploie les templates Slack vers Alertmanager
avec gestion multi-tenant et validation.
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
from urllib.parse import urljoin


class SlackTemplateDeployer:
    """Déployeur de templates Slack pour Alertmanager."""
    
    def __init__(self, config_file: str):
        """
        Initialise le déployeur.
        
        Args:
            config_file: Fichier de configuration du déploiement
        """
        self.config = self._load_config(config_file)
        self.dry_run = False
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Charge la configuration de déploiement."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                return json.load(f)
            else:
                return yaml.safe_load(f)
    
    def validate_alertmanager_connection(self, alertmanager_url: str) -> bool:
        """Valide la connexion à Alertmanager."""
        try:
            health_url = urljoin(alertmanager_url, '/-/healthy')
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Connexion Alertmanager OK: {alertmanager_url}")
                return True
            else:
                print(f"❌ Alertmanager inaccessible: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur de connexion Alertmanager: {e}")
            return False
    
    def backup_current_config(self, alertmanager_url: str, tenant_id: str) -> Optional[str]:
        """Sauvegarde la configuration actuelle."""
        try:
            config_url = urljoin(alertmanager_url, '/api/v1/config')
            response = requests.get(config_url, timeout=10)
            
            if response.status_code == 200:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = self.backup_dir / f"alertmanager_config_{tenant_id}_{timestamp}.yaml"
                
                with open(backup_file, 'w', encoding='utf-8') as f:
                    yaml.dump(response.json(), f, default_flow_style=False)
                
                print(f"💾 Sauvegarde créée: {backup_file}")
                return str(backup_file)
            else:
                print(f"⚠️ Impossible de récupérer la config actuelle: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None
    
    def generate_alertmanager_config(self, templates_dir: str, tenant_config: Dict[str, Any]) -> Dict[str, Any]:
        """Génère la configuration Alertmanager avec les templates Slack."""
        template_files = list(Path(templates_dir).glob('*_slack_template.yaml'))
        
        # Configuration de base
        config = {
            'global': {
                'smtp_smarthost': 'localhost:587',
                'smtp_from': tenant_config.get('email_from', 'alertmanager@spotify-ai-agent.com'),
                'slack_api_url': 'https://hooks.slack.com/services/'
            },
            'route': {
                'group_by': ['alertname'],
                'group_wait': '10s',
                'group_interval': '10s',
                'repeat_interval': '1h',
                'receiver': 'web.hook'
            },
            'receivers': []
        }
        
        # Générer les receivers Slack
        for template_file in template_files:
            template_name = template_file.stem.replace('_slack_template', '')
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            receiver = {
                'name': f"slack-{template_name}-{tenant_config['id']}",
                'slack_configs': [{
                    'api_url': tenant_config.get('slack_webhook_url', ''),
                    'channel': tenant_config.get('slack_channel', '#alerts'),
                    'username': 'Alertmanager',
                    'icon_emoji': self._get_icon_for_template(template_name),
                    'title': f"Spotify AI Agent - {template_name.title()}",
                    'text': template_content,
                    'send_resolved': template_name == 'resolved'
                }]
            }
            
            config['receivers'].append(receiver)
        
        # Routes spécifiques par tenant
        config['route']['routes'] = self._generate_tenant_routes(tenant_config)
        
        return config
    
    def _get_icon_for_template(self, template_name: str) -> str:
        """Retourne l'icône appropriée pour le type de template."""
        icons = {
            'critical': ':rotating_light:',
            'warning': ':warning:',
            'info': ':information_source:',
            'resolved': ':white_check_mark:'
        }
        return icons.get(template_name, ':bell:')
    
    def _generate_tenant_routes(self, tenant_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Génère les routes spécifiques au tenant."""
        tenant_id = tenant_config['id']
        
        routes = [
            {
                'match': {
                    'tenant_id': tenant_id,
                    'severity': 'critical'
                },
                'receiver': f"slack-critical-{tenant_id}",
                'group_wait': '0s',
                'repeat_interval': '5m'
            },
            {
                'match': {
                    'tenant_id': tenant_id,
                    'severity': 'warning'
                },
                'receiver': f"slack-warning-{tenant_id}",
                'group_wait': '5s',
                'repeat_interval': '30m'
            },
            {
                'match': {
                    'tenant_id': tenant_id,
                    'severity': 'info'
                },
                'receiver': f"slack-info-{tenant_id}",
                'group_wait': '10s',
                'repeat_interval': '12h'
            }
        ]
        
        return routes
    
    def deploy_to_kubernetes(self, config: Dict[str, Any], namespace: str, dry_run: bool = False) -> bool:
        """Déploie la configuration vers Kubernetes."""
        try:
            # Créer le ConfigMap
            configmap_yaml = {
                'apiVersion': 'v1',
                'kind': 'ConfigMap',
                'metadata': {
                    'name': 'alertmanager-config',
                    'namespace': namespace,
                    'labels': {
                        'app': 'alertmanager',
                        'component': 'config'
                    }
                },
                'data': {
                    'alertmanager.yml': yaml.dump(config, default_flow_style=False)
                }
            }
            
            # Sauvegarder le ConfigMap
            configmap_file = f"alertmanager-configmap-{namespace}.yaml"
            with open(configmap_file, 'w', encoding='utf-8') as f:
                yaml.dump(configmap_yaml, f, default_flow_style=False)
            
            if dry_run:
                print(f"🔍 [DRY RUN] ConfigMap généré: {configmap_file}")
                return True
            
            # Appliquer le ConfigMap
            cmd = ['kubectl', 'apply', '-f', configmap_file, '-n', namespace]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ ConfigMap déployé dans le namespace: {namespace}")
                
                # Redémarrer Alertmanager pour recharger la config
                restart_cmd = ['kubectl', 'rollout', 'restart', 'deployment/alertmanager', '-n', namespace]
                restart_result = subprocess.run(restart_cmd, capture_output=True, text=True)
                
                if restart_result.returncode == 0:
                    print("🔄 Alertmanager redémarré")
                else:
                    print(f"⚠️ Erreur lors du redémarrage: {restart_result.stderr}")
                
                return True
            else:
                print(f"❌ Erreur de déploiement: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur de déploiement Kubernetes: {e}")
            return False
    
    def deploy_to_file(self, config: Dict[str, Any], output_file: str) -> bool:
        """Déploie la configuration vers un fichier."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"📁 Configuration sauvegardée: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def validate_deployment(self, alertmanager_url: str) -> bool:
        """Valide le déploiement."""
        try:
            # Vérifier le statut
            status_url = urljoin(alertmanager_url, '/api/v1/status')
            response = requests.get(status_url, timeout=10)
            
            if response.status_code == 200:
                status_data = response.json()
                print(f"✅ Alertmanager opérationnel")
                print(f"   Version: {status_data.get('data', {}).get('versionInfo', {}).get('version', 'inconnue')}")
                return True
            else:
                print(f"❌ Erreur de statut: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur de validation: {e}")
            return False
    
    def rollback(self, backup_file: str, alertmanager_url: str, namespace: str) -> bool:
        """Effectue un rollback vers une configuration précédente."""
        try:
            if not os.path.exists(backup_file):
                print(f"❌ Fichier de sauvegarde non trouvé: {backup_file}")
                return False
            
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_config = yaml.safe_load(f)
            
            print(f"🔄 Rollback vers: {backup_file}")
            
            # Redéployer la configuration de sauvegarde
            success = self.deploy_to_kubernetes(backup_config, namespace, dry_run=False)
            
            if success:
                print("✅ Rollback réussi")
            else:
                print("❌ Échec du rollback")
            
            return success
            
        except Exception as e:
            print(f"❌ Erreur lors du rollback: {e}")
            return False
    
    def deploy_templates(self, templates_dir: str, tenant_configs: List[Dict[str, Any]]) -> bool:
        """Déploie tous les templates pour tous les tenants."""
        print("🚀 Déploiement des templates Slack...")
        
        all_success = True
        
        for tenant_config in tenant_configs:
            tenant_id = tenant_config['id']
            print(f"\n📋 Déploiement pour tenant: {tenant_id}")
            
            # Vérifier la connexion
            alertmanager_url = tenant_config.get('alertmanager_url')
            if alertmanager_url and not self.validate_alertmanager_connection(alertmanager_url):
                print(f"⚠️ Alertmanager inaccessible pour {tenant_id}, passage au suivant")
                all_success = False
                continue
            
            # Sauvegarde
            if alertmanager_url:
                backup_file = self.backup_current_config(alertmanager_url, tenant_id)
            
            # Générer la configuration
            config = self.generate_alertmanager_config(templates_dir, tenant_config)
            
            # Déployer selon le mode configuré
            deployment_mode = tenant_config.get('deployment_mode', 'file')
            
            if deployment_mode == 'kubernetes':
                namespace = tenant_config.get('kubernetes_namespace', 'monitoring')
                success = self.deploy_to_kubernetes(config, namespace, self.dry_run)
            else:
                output_file = f"alertmanager-config-{tenant_id}.yaml"
                success = self.deploy_to_file(config, output_file)
            
            if success:
                print(f"✅ Déploiement réussi pour {tenant_id}")
                
                # Validation post-déploiement
                if alertmanager_url and not self.dry_run:
                    self.validate_deployment(alertmanager_url)
            else:
                print(f"❌ Échec du déploiement pour {tenant_id}")
                all_success = False
        
        return all_success


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Déployeur de templates Slack pour Alertmanager"
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Fichier de configuration du déploiement'
    )
    
    parser.add_argument(
        '--templates-dir',
        required=True,
        help='Répertoire contenant les templates générés'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Mode simulation (ne déploie pas réellement)'
    )
    
    parser.add_argument(
        '--rollback',
        help='Fichier de sauvegarde pour rollback'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Valider les connexions seulement'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialiser le déployeur
        deployer = SlackTemplateDeployer(args.config)
        deployer.dry_run = args.dry_run
        
        if args.dry_run:
            print("🔍 MODE SIMULATION ACTIVÉ")
        
        # Mode rollback
        if args.rollback:
            # Charger la config pour obtenir les infos de déploiement
            tenant_configs = deployer.config.get('tenants', [])
            if tenant_configs:
                tenant_config = tenant_configs[0]  # Premier tenant
                alertmanager_url = tenant_config.get('alertmanager_url')
                namespace = tenant_config.get('kubernetes_namespace', 'monitoring')
                
                success = deployer.rollback(args.rollback, alertmanager_url, namespace)
                sys.exit(0 if success else 1)
            else:
                print("❌ Aucune configuration tenant trouvée")
                sys.exit(1)
        
        # Charger les configurations des tenants
        tenant_configs = deployer.config.get('tenants', [])
        
        if not tenant_configs:
            print("❌ Aucune configuration tenant trouvée")
            sys.exit(1)
        
        # Mode validation seulement
        if args.validate_only:
            all_valid = True
            for tenant_config in tenant_configs:
                alertmanager_url = tenant_config.get('alertmanager_url')
                if alertmanager_url:
                    valid = deployer.validate_alertmanager_connection(alertmanager_url)
                    all_valid = all_valid and valid
            
            sys.exit(0 if all_valid else 1)
        
        # Déploiement normal
        success = deployer.deploy_templates(args.templates_dir, tenant_configs)
        
        if success:
            print("\n✅ Déploiement terminé avec succès!")
            sys.exit(0)
        else:
            print("\n❌ Des erreurs sont survenues lors du déploiement")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
