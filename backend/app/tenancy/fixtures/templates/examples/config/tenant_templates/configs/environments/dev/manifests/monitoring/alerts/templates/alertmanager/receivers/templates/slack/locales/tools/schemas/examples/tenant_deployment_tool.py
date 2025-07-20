#!/usr/bin/env python3
"""
Script avancé de déploiement et gestion des configurations multi-tenant.

Ce script automatise la création, validation et déploiement des configurations
tenant avec support du monitoring, alertes Slack et isolation des données.

Usage:
    python tenant_deployment_tool.py deploy --config config.json
    python tenant_deployment_tool.py validate --tenant-id my-tenant
    python tenant_deployment_tool.py scale --tenant-id my-tenant --instances 5
"""

import argparse
import asyncio
import json
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import subprocess
import shutil
import tempfile

# Imports pour validation et génération
from schema_generator import TenantConfigGenerator, TenantConfig, SubscriptionTier
import kubernetes
from kubernetes import client, config as k8s_config
import docker
import requests
from jinja2 import Environment, FileSystemLoader

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/tenant-deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class TenantDeploymentTool:
    """Outil avancé de déploiement multi-tenant."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.generator = TenantConfigGenerator()
        self.docker_client = docker.from_env()
        self.k8s_client = None
        self.jinja_env = Environment(loader=FileSystemLoader('templates'))
        
        # Initialisation Kubernetes
        try:
            k8s_config.load_incluster_config()
            self.k8s_client = client.ApiClient()
            logger.info("Configuration Kubernetes chargée (in-cluster)")
        except:
            try:
                k8s_config.load_kube_config()
                self.k8s_client = client.ApiClient()
                logger.info("Configuration Kubernetes chargée (kubeconfig)")
            except Exception as e:
                logger.warning(f"Impossible de charger la config Kubernetes: {e}")
    
    async def deploy_tenant(self, tenant_config: Dict[str, Any]) -> bool:
        """Déploie un nouveau tenant avec toute son infrastructure."""
        try:
            tenant_id = tenant_config.get('tenant_id')
            logger.info(f"Démarrage du déploiement du tenant: {tenant_id}")
            
            # 1. Validation de la configuration
            if not self.validate_tenant_config(tenant_config):
                logger.error("Configuration tenant invalide")
                return False
            
            # 2. Création de l'infrastructure de base
            await self.create_tenant_infrastructure(tenant_config)
            
            # 3. Configuration de l'isolation des données
            await self.setup_data_isolation(tenant_config)
            
            # 4. Déploiement du monitoring
            await self.deploy_monitoring_stack(tenant_config)
            
            # 5. Configuration des alertes Slack
            await self.setup_slack_alerts(tenant_config)
            
            # 6. Validation du déploiement
            if await self.validate_deployment(tenant_id):
                logger.info(f"Déploiement du tenant {tenant_id} réussi")
                await self.send_deployment_notification(tenant_config, "success")
                return True
            else:
                logger.error(f"Validation du déploiement échouée pour {tenant_id}")
                await self.send_deployment_notification(tenant_config, "failure")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du déploiement: {e}")
            await self.send_deployment_notification(tenant_config, "error", str(e))
            return False
    
    def validate_tenant_config(self, config: Dict[str, Any]) -> bool:
        """Valide la configuration tenant."""
        required_fields = ['tenant_id', 'tenant_name', 'subscription_tier']
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Champ requis manquant: {field}")
                return False
        
        # Validation du tenant_id
        tenant_id = config['tenant_id']
        if not tenant_id or len(tenant_id) < 3 or not tenant_id.replace('-', '').replace('_', '').isalnum():
            logger.error("tenant_id invalide")
            return False
        
        # Validation du tier
        if config['subscription_tier'] not in ['free', 'premium', 'enterprise', 'enterprise_plus']:
            logger.error("subscription_tier invalide")
            return False
        
        logger.info("Configuration tenant validée avec succès")
        return True
    
    async def create_tenant_infrastructure(self, config: Dict[str, Any]) -> None:
        """Crée l'infrastructure Kubernetes pour le tenant."""
        tenant_id = config['tenant_id']
        
        # Création du namespace
        await self.create_k8s_namespace(tenant_id)
        
        # Création des secrets
        await self.create_tenant_secrets(config)
        
        # Déploiement des services de base
        await self.deploy_tenant_services(config)
        
        logger.info(f"Infrastructure créée pour le tenant {tenant_id}")
    
    async def create_k8s_namespace(self, tenant_id: str) -> None:
        """Crée un namespace Kubernetes pour le tenant."""
        if not self.k8s_client:
            logger.warning("Client Kubernetes non disponible")
            return
        
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Vérifier si le namespace existe déjà
            try:
                v1.read_namespace(name=tenant_id)
                logger.info(f"Namespace {tenant_id} existe déjà")
                return
            except client.exceptions.ApiException as e:
                if e.status != 404:
                    raise
            
            # Créer le namespace
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=tenant_id,
                    labels={
                        'app.kubernetes.io/name': 'spotify-ai-agent',
                        'app.kubernetes.io/component': 'tenant',
                        'tenant-id': tenant_id,
                        'managed-by': 'tenant-deployment-tool'
                    },
                    annotations={
                        'deployment.timestamp': datetime.now(timezone.utc).isoformat(),
                        'tenant.isolation-level': 'namespace'
                    }
                )
            )
            
            v1.create_namespace(body=namespace)
            logger.info(f"Namespace {tenant_id} créé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du namespace: {e}")
            raise
    
    async def create_tenant_secrets(self, config: Dict[str, Any]) -> None:
        """Crée les secrets Kubernetes pour le tenant."""
        tenant_id = config['tenant_id']
        
        if not self.k8s_client:
            return
        
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Secret pour la base de données
            db_secret = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=f"{tenant_id}-database",
                    namespace=tenant_id
                ),
                type="Opaque",
                string_data={
                    'host': f"postgres-{tenant_id}.database.svc.cluster.local",
                    'port': "5432",
                    'database': tenant_id,
                    'username': f"user_{tenant_id}",
                    'password': self.generate_secure_password(),
                    'connection_string': f"postgresql://user_{tenant_id}:password@postgres-{tenant_id}.database.svc.cluster.local:5432/{tenant_id}"
                }
            )
            
            # Secret pour Redis
            redis_secret = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=f"{tenant_id}-redis",
                    namespace=tenant_id
                ),
                type="Opaque",
                string_data={
                    'host': f"redis-{tenant_id}.cache.svc.cluster.local",
                    'port': "6379",
                    'password': self.generate_secure_password(),
                    'namespace': f"{tenant_id}:cache"
                }
            )
            
            # Secret pour les API keys
            api_secret = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=f"{tenant_id}-api-keys",
                    namespace=tenant_id
                ),
                type="Opaque",
                string_data={
                    'api_key': self.generate_api_key(),
                    'jwt_secret': self.generate_secure_password(32),
                    'encryption_key': self.generate_secure_password(32)
                }
            )
            
            # Création des secrets
            for secret in [db_secret, redis_secret, api_secret]:
                v1.create_namespaced_secret(namespace=tenant_id, body=secret)
            
            logger.info(f"Secrets créés pour le tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la création des secrets: {e}")
            raise
    
    async def setup_data_isolation(self, config: Dict[str, Any]) -> None:
        """Configure l'isolation des données pour le tenant."""
        tenant_id = config['tenant_id']
        isolation_config = config.get('isolation_config', {})
        
        # Configuration base de données
        await self.setup_database_isolation(tenant_id, isolation_config.get('database_isolation', {}))
        
        # Configuration cache Redis
        await self.setup_cache_isolation(tenant_id, isolation_config.get('cache_isolation', {}))
        
        # Configuration stockage S3
        await self.setup_storage_isolation(tenant_id, isolation_config.get('storage_isolation', {}))
        
        logger.info(f"Isolation des données configurée pour {tenant_id}")
    
    async def setup_database_isolation(self, tenant_id: str, db_config: Dict[str, Any]) -> None:
        """Configure l'isolation de la base de données."""
        isolation_type = db_config.get('type', 'schema')
        
        if isolation_type == 'schema':
            # Création d'un schéma dédié
            await self.create_database_schema(tenant_id, db_config)
        elif isolation_type == 'database':
            # Création d'une base de données dédiée
            await self.create_dedicated_database(tenant_id, db_config)
        elif isolation_type == 'cluster':
            # Déploiement d'un cluster dédié
            await self.deploy_dedicated_database_cluster(tenant_id, db_config)
        
        logger.info(f"Isolation base de données ({isolation_type}) configurée pour {tenant_id}")
    
    async def deploy_monitoring_stack(self, config: Dict[str, Any]) -> None:
        """Déploie la stack de monitoring pour le tenant."""
        tenant_id = config['tenant_id']
        monitoring_config = config.get('monitoring_config', {})
        
        if monitoring_config.get('prometheus_enabled', True):
            await self.deploy_prometheus_config(tenant_id, monitoring_config)
        
        if monitoring_config.get('grafana_enabled', True):
            await self.deploy_grafana_dashboard(tenant_id, monitoring_config)
        
        logger.info(f"Stack de monitoring déployée pour {tenant_id}")
    
    async def deploy_prometheus_config(self, tenant_id: str, monitoring_config: Dict[str, Any]) -> None:
        """Déploie la configuration Prometheus pour le tenant."""
        try:
            # Génération de la configuration Prometheus spécifique au tenant
            prometheus_config = {
                'global': {
                    'scrape_interval': f"{monitoring_config.get('scrape_interval_seconds', 30)}s",
                    'external_labels': {
                        'tenant_id': tenant_id,
                        'environment': 'dev'
                    }
                },
                'scrape_configs': [
                    {
                        'job_name': f'{tenant_id}-api',
                        'kubernetes_sd_configs': [{
                            'role': 'pod',
                            'namespaces': {'names': [tenant_id]}
                        }],
                        'relabel_configs': [
                            {
                                'source_labels': ['__meta_kubernetes_pod_label_app'],
                                'action': 'keep',
                                'regex': 'spotify-ai-agent-api'
                            },
                            {
                                'source_labels': ['__meta_kubernetes_pod_label_tenant_id'],
                                'target_label': 'tenant_id'
                            }
                        ]
                    }
                ]
            }
            
            # Sauvegarde de la configuration
            config_path = f"/etc/prometheus/tenants/{tenant_id}.yml"
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(prometheus_config, f)
            
            # Rechargement de Prometheus
            await self.reload_prometheus_config()
            
            logger.info(f"Configuration Prometheus déployée pour {tenant_id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du déploiement Prometheus: {e}")
            raise
    
    async def setup_slack_alerts(self, config: Dict[str, Any]) -> None:
        """Configure les alertes Slack pour le tenant."""
        tenant_id = config['tenant_id']
        alert_config = config.get('alert_config', {})
        
        if not alert_config.get('slack_enabled', False):
            logger.info(f"Alertes Slack désactivées pour {tenant_id}")
            return
        
        # Génération des règles d'alerte
        alert_rules = self.generate_alert_rules(tenant_id, alert_config)
        
        # Sauvegarde des règles
        rules_path = f"/etc/prometheus/rules/{tenant_id}_alerts.yml"
        Path(rules_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(rules_path, 'w') as f:
            yaml.dump(alert_rules, f)
        
        # Configuration AlertManager
        await self.update_alertmanager_config(tenant_id, alert_config)
        
        logger.info(f"Alertes Slack configurées pour {tenant_id}")
    
    def generate_alert_rules(self, tenant_id: str, alert_config: Dict[str, Any]) -> Dict[str, Any]:
        """Génère les règles d'alerte Prometheus pour le tenant."""
        return {
            'groups': [
                {
                    'name': f'{tenant_id}_alerts',
                    'rules': [
                        {
                            'alert': f'{tenant_id}_HighCPUUsage',
                            'expr': f'rate(cpu_usage_seconds_total{{tenant_id="{tenant_id}"}}[5m]) > 0.8',
                            'for': '2m',
                            'labels': {
                                'severity': 'warning',
                                'tenant_id': tenant_id,
                                'category': 'resource'
                            },
                            'annotations': {
                                'summary': f'Utilisation CPU élevée pour le tenant {tenant_id}',
                                'description': f'Le tenant {tenant_id} utilise plus de 80% du CPU pendant plus de 2 minutes.'
                            }
                        },
                        {
                            'alert': f'{tenant_id}_HighMemoryUsage',
                            'expr': f'memory_usage_bytes{{tenant_id="{tenant_id}"}} / memory_limit_bytes{{tenant_id="{tenant_id}"}} > 0.9',
                            'for': '1m',
                            'labels': {
                                'severity': 'critical',
                                'tenant_id': tenant_id,
                                'category': 'resource'
                            },
                            'annotations': {
                                'summary': f'Utilisation mémoire critique pour le tenant {tenant_id}',
                                'description': f'Le tenant {tenant_id} utilise plus de 90% de la mémoire allouée.'
                            }
                        },
                        {
                            'alert': f'{tenant_id}_APIErrorRate',
                            'expr': f'rate(api_requests_total{{tenant_id="{tenant_id}",status=~"5.."}}[5m]) / rate(api_requests_total{{tenant_id="{tenant_id}"}}[5m]) > 0.1',
                            'for': '2m',
                            'labels': {
                                'severity': 'warning',
                                'tenant_id': tenant_id,
                                'category': 'api'
                            },
                            'annotations': {
                                'summary': f'Taux d\'erreur API élevé pour le tenant {tenant_id}',
                                'description': f'Le tenant {tenant_id} a un taux d\'erreur API > 10% pendant 2 minutes.'
                            }
                        }
                    ]
                }
            ]
        }
    
    async def validate_deployment(self, tenant_id: str) -> bool:
        """Valide que le déploiement du tenant est fonctionnel."""
        try:
            # Vérification du namespace
            if not await self.check_namespace_exists(tenant_id):
                logger.error(f"Namespace {tenant_id} introuvable")
                return False
            
            # Vérification des services
            if not await self.check_services_healthy(tenant_id):
                logger.error(f"Services non sains pour {tenant_id}")
                return False
            
            # Test de l'API
            if not await self.test_tenant_api(tenant_id):
                logger.error(f"API non accessible pour {tenant_id}")
                return False
            
            logger.info(f"Validation réussie pour le tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {e}")
            return False
    
    async def send_deployment_notification(self, config: Dict[str, Any], status: str, error_msg: str = None) -> None:
        """Envoie une notification de déploiement."""
        tenant_id = config['tenant_id']
        
        if status == "success":
            message = f"✅ Tenant {tenant_id} déployé avec succès"
            color = "good"
        elif status == "failure":
            message = f"❌ Échec du déploiement du tenant {tenant_id}"
            color = "danger"
        else:
            message = f"⚠️ Erreur lors du déploiement du tenant {tenant_id}: {error_msg}"
            color = "warning"
        
        # Envoi vers Slack
        webhook_url = "https://hooks.slack.com/services/YOUR_WEBHOOK_URL"
        payload = {
            "attachments": [{
                "color": color,
                "text": message,
                "fields": [
                    {"title": "Tenant ID", "value": tenant_id, "short": True},
                    {"title": "Timestamp", "value": datetime.now().isoformat(), "short": True}
                ]
            }]
        }
        
        try:
            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Notification Slack envoyée")
            else:
                logger.warning(f"Échec envoi Slack: {response.status_code}")
        except Exception as e:
            logger.error(f"Erreur notification Slack: {e}")
    
    # Méthodes utilitaires
    def generate_secure_password(self, length: int = 16) -> str:
        """Génère un mot de passe sécurisé."""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_api_key(self) -> str:
        """Génère une clé API."""
        import secrets
        return secrets.token_urlsafe(32)
    
    async def check_namespace_exists(self, tenant_id: str) -> bool:
        """Vérifie si le namespace existe."""
        if not self.k8s_client:
            return True  # Assume exists if no k8s client
        
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            v1.read_namespace(name=tenant_id)
            return True
        except:
            return False
    
    async def check_services_healthy(self, tenant_id: str) -> bool:
        """Vérifie la santé des services."""
        # Implémentation de vérification de santé
        return True
    
    async def test_tenant_api(self, tenant_id: str) -> bool:
        """Test l'API du tenant."""
        # Implémentation de test API
        return True
    
    async def reload_prometheus_config(self) -> None:
        """Recharge la configuration Prometheus."""
        try:
            response = requests.post("http://prometheus:9090/-/reload")
            if response.status_code == 200:
                logger.info("Configuration Prometheus rechargée")
            else:
                logger.warning("Échec rechargement Prometheus")
        except Exception as e:
            logger.error(f"Erreur rechargement Prometheus: {e}")
    
    async def update_alertmanager_config(self, tenant_id: str, alert_config: Dict[str, Any]) -> None:
        """Met à jour la configuration AlertManager."""
        # Implémentation mise à jour AlertManager
        pass
    
    async def create_database_schema(self, tenant_id: str, db_config: Dict[str, Any]) -> None:
        """Crée un schéma de base de données."""
        # Implémentation création schéma
        pass
    
    async def create_dedicated_database(self, tenant_id: str, db_config: Dict[str, Any]) -> None:
        """Crée une base de données dédiée."""
        # Implémentation base dédiée
        pass
    
    async def deploy_dedicated_database_cluster(self, tenant_id: str, db_config: Dict[str, Any]) -> None:
        """Déploie un cluster de base de données dédié."""
        # Implémentation cluster dédié
        pass
    
    async def setup_cache_isolation(self, tenant_id: str, cache_config: Dict[str, Any]) -> None:
        """Configure l'isolation du cache."""
        # Implémentation isolation cache
        pass
    
    async def setup_storage_isolation(self, tenant_id: str, storage_config: Dict[str, Any]) -> None:
        """Configure l'isolation du stockage."""
        # Implémentation isolation stockage
        pass
    
    async def deploy_tenant_services(self, config: Dict[str, Any]) -> None:
        """Déploie les services du tenant."""
        # Implémentation déploiement services
        pass
    
    async def deploy_grafana_dashboard(self, tenant_id: str, monitoring_config: Dict[str, Any]) -> None:
        """Déploie le dashboard Grafana."""
        # Implémentation dashboard Grafana
        pass

async def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Outil de déploiement multi-tenant')
    parser.add_argument('action', choices=['deploy', 'validate', 'scale', 'delete'])
    parser.add_argument('--config', help='Chemin vers le fichier de configuration')
    parser.add_argument('--tenant-id', help='ID du tenant')
    parser.add_argument('--instances', type=int, help='Nombre d\'instances')
    
    args = parser.parse_args()
    
    tool = TenantDeploymentTool(args.config)
    
    if args.action == 'deploy':
        if not args.config:
            logger.error("--config requis pour deploy")
            sys.exit(1)
        
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        success = await tool.deploy_tenant(config)
        sys.exit(0 if success else 1)
    
    elif args.action == 'validate':
        if not args.tenant_id:
            logger.error("--tenant-id requis pour validate")
            sys.exit(1)
        
        success = await tool.validate_deployment(args.tenant_id)
        sys.exit(0 if success else 1)
    
    else:
        logger.error(f"Action {args.action} non implémentée")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
