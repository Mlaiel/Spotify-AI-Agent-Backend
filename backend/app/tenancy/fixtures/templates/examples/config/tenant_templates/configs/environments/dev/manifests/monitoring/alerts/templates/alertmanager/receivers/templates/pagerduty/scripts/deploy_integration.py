#!/usr/bin/env python3
"""
Advanced Deployment Script for PagerDuty Integration

Ce script automatise le déploiement complet du système d'intégration PagerDuty
avec validation, migration, configuration, et vérification de santé.

Fonctionnalités:
- Déploiement automatisé avec rollback
- Validation de configuration avancée
- Migration de données intelligente
- Health checks complets
- Monitoring et alerting setup
- Configuration multi-environnement

Version: 4.0.0
Développé par l'équipe Spotify AI Agent
"""

import asyncio
import argparse
import json
import os
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import aiofiles
import aioredis
import aiohttp
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import structlog

console = Console()
logger = structlog.get_logger(__name__)

class DeploymentConfig:
    """Configuration de déploiement"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = {}
        self.load_config()
        
    def load_config(self):
        """Charge la configuration depuis le fichier"""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
                
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    self.config = yaml.safe_load(f)
                    
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            sys.exit(1)
            
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value

class PagerDutyDeployer:
    """Déployeur principal pour l'intégration PagerDuty"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.environment = config.get('environment', 'development')
        self.redis_url = config.get('redis.url', 'redis://localhost:6379')
        self.backup_path = Path(config.get('backup.path', './backups'))
        self.rollback_enabled = config.get('deployment.rollback_enabled', True)
        self.health_check_timeout = config.get('health_check.timeout', 300)
        
    async def deploy(self, force: bool = False) -> bool:
        """Lance le déploiement complet"""
        console.print(Panel.fit(
            f"🚀 Starting PagerDuty Integration Deployment\n"
            f"Environment: {self.environment}\n"
            f"Timestamp: {datetime.now().isoformat()}",
            title="Deployment Started",
            border_style="green"
        ))
        
        deployment_id = f"deploy_{int(time.time())}"
        success = False
        
        try:
            with Progress() as progress:
                # Étapes de déploiement
                tasks = [
                    ("🔍 Pre-deployment validation", self._validate_environment),
                    ("💾 Backup current state", self._backup_current_state),
                    ("📦 Deploy dependencies", self._deploy_dependencies),
                    ("⚙️ Deploy configuration", self._deploy_configuration),
                    ("🔧 Deploy application", self._deploy_application),
                    ("🗄️ Migrate data", self._migrate_data),
                    ("🔌 Setup integrations", self._setup_integrations),
                    ("📊 Configure monitoring", self._configure_monitoring),
                    ("✅ Health checks", self._run_health_checks),
                    ("📝 Post-deployment tasks", self._post_deployment_tasks)
                ]
                
                total_tasks = len(tasks)
                main_task = progress.add_task("Overall Progress", total=total_tasks)
                
                for i, (description, task_func) in enumerate(tasks):
                    task = progress.add_task(description, total=100)
                    
                    try:
                        await task_func(progress, task)
                        progress.update(task, completed=100)
                        progress.update(main_task, completed=i + 1)
                        
                    except Exception as e:
                        progress.update(task, completed=100, description=f"{description} ❌")
                        console.print(f"[red]Failed: {description} - {e}[/red]")
                        
                        if not force and self.rollback_enabled:
                            await self._rollback_deployment(deployment_id)
                            return False
                        else:
                            console.print("[yellow]Continuing despite error (force mode)[/yellow]")
                            
            success = True
            console.print(Panel.fit(
                "✅ Deployment completed successfully!",
                title="Success",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Deployment interrupted by user[/yellow]")
            if self.rollback_enabled:
                await self._rollback_deployment(deployment_id)
                
        except Exception as e:
            console.print(f"[red]Deployment failed: {e}[/red]")
            if self.rollback_enabled:
                await self._rollback_deployment(deployment_id)
                
        finally:
            await self._cleanup_deployment(deployment_id)
            
        return success
        
    async def _validate_environment(self, progress: Progress, task: TaskID):
        """Valide l'environnement de déploiement"""
        validations = [
            ("Python version", self._check_python_version),
            ("Required packages", self._check_required_packages),
            ("Environment variables", self._check_environment_variables),
            ("Network connectivity", self._check_network_connectivity),
            ("Redis connection", self._check_redis_connection),
            ("PagerDuty API access", self._check_pagerduty_access),
            ("Disk space", self._check_disk_space),
            ("Permissions", self._check_permissions)
        ]
        
        for i, (name, check_func) in enumerate(validations):
            try:
                await check_func()
                progress.update(task, completed=(i + 1) / len(validations) * 100,
                              description=f"🔍 Validating {name} ✅")
            except Exception as e:
                raise Exception(f"Validation failed for {name}: {e}")
                
    async def _check_python_version(self):
        """Vérifie la version Python"""
        import sys
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            raise Exception(f"Python 3.8+ required, found {version.major}.{version.minor}")
            
    async def _check_required_packages(self):
        """Vérifie les packages requis"""
        required_packages = [
            'aioredis', 'aiohttp', 'pydantic', 'structlog',
            'prometheus_client', 'psutil', 'numpy', 'pandas'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                raise Exception(f"Required package not found: {package}")
                
    async def _check_environment_variables(self):
        """Vérifie les variables d'environnement"""
        required_vars = [
            'PAGERDUTY_API_KEY',
            'PAGERDUTY_ROUTING_KEY',
            'REDIS_URL'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise Exception(f"Missing environment variables: {missing}")
            
    async def _check_network_connectivity(self):
        """Vérifie la connectivité réseau"""
        test_urls = [
            'https://api.pagerduty.com',
            'https://httpbin.org/status/200'
        ]
        
        async with aiohttp.ClientSession() as session:
            for url in test_urls:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status >= 400:
                            raise Exception(f"HTTP {response.status}")
                except Exception as e:
                    raise Exception(f"Cannot reach {url}: {e}")
                    
    async def _check_redis_connection(self):
        """Vérifie la connexion Redis"""
        try:
            redis = aioredis.from_url(self.redis_url)
            await redis.ping()
            await redis.close()
        except Exception as e:
            raise Exception(f"Redis connection failed: {e}")
            
    async def _check_pagerduty_access(self):
        """Vérifie l'accès à l'API PagerDuty"""
        api_key = os.getenv('PAGERDUTY_API_KEY')
        if not api_key:
            raise Exception("PAGERDUTY_API_KEY not set")
            
        headers = {
            'Authorization': f'Token token={api_key}',
            'Accept': 'application/vnd.pagerduty+json;version=2'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    'https://api.pagerduty.com/users',
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 401:
                        raise Exception("Invalid API key")
                    elif response.status >= 400:
                        raise Exception(f"API error: {response.status}")
            except Exception as e:
                raise Exception(f"PagerDuty API access failed: {e}")
                
    async def _check_disk_space(self):
        """Vérifie l'espace disque"""
        import shutil
        free_space = shutil.disk_usage('.').free
        required_space = 1024 * 1024 * 1024  # 1GB
        
        if free_space < required_space:
            raise Exception(f"Insufficient disk space: {free_space / 1024**3:.1f}GB available, 1GB required")
            
    async def _check_permissions(self):
        """Vérifie les permissions"""
        test_file = Path('./test_write_permission')
        try:
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            raise Exception(f"Write permission check failed: {e}")
            
    async def _backup_current_state(self, progress: Progress, task: TaskID):
        """Sauvegarde l'état actuel"""
        backup_dir = self.backup_path / f"backup_{int(time.time())}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_tasks = [
            ("Configuration files", self._backup_configs),
            ("Redis data", self._backup_redis_data),
            ("Application state", self._backup_app_state)
        ]
        
        for i, (name, backup_func) in enumerate(backup_tasks):
            await backup_func(backup_dir)
            progress.update(task, completed=(i + 1) / len(backup_tasks) * 100,
                          description=f"💾 Backing up {name}")
            
    async def _backup_configs(self, backup_dir: Path):
        """Sauvegarde les configurations"""
        config_files = [
            'config.yaml',
            'config.json',
            '.env'
        ]
        
        for config_file in config_files:
            src = Path(config_file)
            if src.exists():
                dst = backup_dir / src.name
                await self._copy_file(src, dst)
                
    async def _backup_redis_data(self, backup_dir: Path):
        """Sauvegarde les données Redis"""
        try:
            redis = aioredis.from_url(self.redis_url)
            
            # Sauvegarde des clés importantes
            important_keys = [
                'pagerduty:*',
                'notifications:*',
                'incidents:*',
                'user_prefs:*'
            ]
            
            backup_data = {}
            for pattern in important_keys:
                keys = await redis.keys(pattern)
                for key in keys:
                    value = await redis.get(key)
                    if value:
                        backup_data[key] = value
                        
            backup_file = backup_dir / 'redis_backup.json'
            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(backup_data, indent=2))
                
            await redis.close()
            
        except Exception as e:
            logger.warning(f"Redis backup failed: {e}")
            
    async def _backup_app_state(self, backup_dir: Path):
        """Sauvegarde l'état de l'application"""
        # Sauvegarde des logs, métriques, etc.
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'version': '4.0.0'
        }
        
        state_file = backup_dir / 'app_state.json'
        async with aiofiles.open(state_file, 'w') as f:
            await f.write(json.dumps(state_data, indent=2))
            
    async def _copy_file(self, src: Path, dst: Path):
        """Copie un fichier de manière asynchrone"""
        async with aiofiles.open(src, 'rb') as f_src:
            content = await f_src.read()
            
        async with aiofiles.open(dst, 'wb') as f_dst:
            await f_dst.write(content)
            
    async def _deploy_dependencies(self, progress: Progress, task: TaskID):
        """Déploie les dépendances"""
        dependencies = [
            ("Python packages", "pip install -r requirements.txt"),
            ("System packages", self._install_system_packages),
            ("External services", self._setup_external_services)
        ]
        
        for i, (name, cmd) in enumerate(dependencies):
            if isinstance(cmd, str):
                await self._run_command(cmd)
            else:
                await cmd()
                
            progress.update(task, completed=(i + 1) / len(dependencies) * 100,
                          description=f"📦 Installing {name}")
            
    async def _install_system_packages(self):
        """Installe les packages système requis"""
        # Exemple pour Ubuntu/Debian
        system_packages = [
            "redis-server",
            "python3-dev",
            "build-essential"
        ]
        
        for package in system_packages:
            try:
                await self._run_command(f"apt-get install -y {package}")
            except Exception as e:
                logger.warning(f"System package installation failed for {package}: {e}")
                
    async def _setup_external_services(self):
        """Configure les services externes"""
        # Configuration Redis si nécessaire
        try:
            await self._run_command("systemctl enable redis-server")
            await self._run_command("systemctl start redis-server")
        except Exception as e:
            logger.warning(f"Redis service setup failed: {e}")
            
    async def _deploy_configuration(self, progress: Progress, task: TaskID):
        """Déploie la configuration"""
        config_tasks = [
            ("Environment config", self._deploy_env_config),
            ("Application config", self._deploy_app_config),
            ("Security config", self._deploy_security_config),
            ("Monitoring config", self._deploy_monitoring_config)
        ]
        
        for i, (name, config_func) in enumerate(config_tasks):
            await config_func()
            progress.update(task, completed=(i + 1) / len(config_tasks) * 100,
                          description=f"⚙️ Configuring {name}")
            
    async def _deploy_env_config(self):
        """Déploie la configuration d'environnement"""
        env_vars = {
            'ENVIRONMENT': self.environment,
            'DEPLOYMENT_TIMESTAMP': datetime.now().isoformat(),
            'LOG_LEVEL': self.config.get('logging.level', 'INFO')
        }
        
        env_file = Path('.env.deployment')
        async with aiofiles.open(env_file, 'w') as f:
            for key, value in env_vars.items():
                await f.write(f"{key}={value}\n")
                
    async def _deploy_app_config(self):
        """Déploie la configuration application"""
        app_config = {
            'pagerduty': {
                'api_version': 'v2',
                'timeout': 30,
                'max_retries': 3
            },
            'redis': {
                'url': self.redis_url,
                'db': 0
            },
            'monitoring': {
                'enabled': True,
                'prometheus_port': 8000
            }
        }
        
        config_file = Path('config.yaml')
        async with aiofiles.open(config_file, 'w') as f:
            await f.write(yaml.dump(app_config, indent=2))
            
    async def _deploy_security_config(self):
        """Déploie la configuration de sécurité"""
        # Configuration des certificats, clés, etc.
        security_config = {
            'encryption': {
                'algorithm': 'AES-256-GCM',
                'key_rotation_interval': 3600
            },
            'authentication': {
                'jwt_expiry': 3600,
                'refresh_token_expiry': 86400
            }
        }
        
        security_file = Path('security.yaml')
        async with aiofiles.open(security_file, 'w') as f:
            await f.write(yaml.dump(security_config, indent=2))
            
    async def _deploy_monitoring_config(self):
        """Déploie la configuration de monitoring"""
        monitoring_config = {
            'prometheus': {
                'port': 8000,
                'path': '/metrics'
            },
            'grafana': {
                'dashboards_path': './dashboards'
            },
            'alerts': {
                'rules_path': './alert_rules'
            }
        }
        
        monitoring_file = Path('monitoring.yaml')
        async with aiofiles.open(monitoring_file, 'w') as f:
            await f.write(yaml.dump(monitoring_config, indent=2))
            
    async def _deploy_application(self, progress: Progress, task: TaskID):
        """Déploie l'application"""
        app_tasks = [
            ("Copy application files", self._copy_app_files),
            ("Install application", self._install_app),
            ("Configure services", self._configure_services),
            ("Start services", self._start_services)
        ]
        
        for i, (name, app_func) in enumerate(app_tasks):
            await app_func()
            progress.update(task, completed=(i + 1) / len(app_tasks) * 100,
                          description=f"🔧 {name}")
            
    async def _copy_app_files(self):
        """Copie les fichiers de l'application"""
        # Copie des fichiers sources
        src_files = [
            '__init__.py',
            'api_manager.py',
            'incident_manager.py',
            'escalation_manager.py',
            'oncall_manager.py',
            'ai_analyzer.py',
            'metrics_collector.py',
            'webhook_processor.py',
            'notification_engine.py'
        ]
        
        for src_file in src_files:
            src_path = Path(src_file)
            if src_path.exists():
                dst_path = Path(f'./deployed/{src_file}')
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                await self._copy_file(src_path, dst_path)
                
    async def _install_app(self):
        """Installe l'application"""
        await self._run_command("pip install -e .")
        
    async def _configure_services(self):
        """Configure les services"""
        # Configuration systemd ou autres services
        service_config = """
[Unit]
Description=PagerDuty Integration Service
After=network.target redis.service

[Service]
Type=simple
User=pagerduty
WorkingDirectory=/opt/pagerduty
ExecStart=/usr/bin/python3 -m pagerduty.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
        """
        
        service_file = Path('/etc/systemd/system/pagerduty.service')
        try:
            async with aiofiles.open(service_file, 'w') as f:
                await f.write(service_config)
                
            await self._run_command("systemctl daemon-reload")
            await self._run_command("systemctl enable pagerduty")
            
        except Exception as e:
            logger.warning(f"Service configuration failed: {e}")
            
    async def _start_services(self):
        """Démarre les services"""
        try:
            await self._run_command("systemctl start pagerduty")
        except Exception as e:
            logger.warning(f"Service start failed: {e}")
            
    async def _migrate_data(self, progress: Progress, task: TaskID):
        """Migre les données"""
        migration_tasks = [
            ("Database schema", self._migrate_database_schema),
            ("Redis data", self._migrate_redis_data),
            ("Configuration migration", self._migrate_configuration)
        ]
        
        for i, (name, migrate_func) in enumerate(migration_tasks):
            await migrate_func()
            progress.update(task, completed=(i + 1) / len(migration_tasks) * 100,
                          description=f"🗄️ Migrating {name}")
            
    async def _migrate_database_schema(self):
        """Migre le schéma de base de données"""
        # Pas de base de données relationnelle dans ce projet
        pass
        
    async def _migrate_redis_data(self):
        """Migre les données Redis"""
        try:
            redis = aioredis.from_url(self.redis_url)
            
            # Migration des anciennes clés vers le nouveau format si nécessaire
            old_keys = await redis.keys('old_format:*')
            for old_key in old_keys:
                value = await redis.get(old_key)
                if value:
                    new_key = old_key.replace('old_format:', 'pagerduty:')
                    await redis.set(new_key, value)
                    await redis.delete(old_key)
                    
            await redis.close()
            
        except Exception as e:
            logger.warning(f"Redis data migration failed: {e}")
            
    async def _migrate_configuration(self):
        """Migre la configuration"""
        # Migration de l'ancienne configuration vers la nouvelle
        old_config_file = Path('old_config.yaml')
        if old_config_file.exists():
            # Logique de migration spécifique
            pass
            
    async def _setup_integrations(self, progress: Progress, task: TaskID):
        """Configure les intégrations"""
        integration_tasks = [
            ("PagerDuty API", self._setup_pagerduty_integration),
            ("Prometheus", self._setup_prometheus_integration),
            ("Grafana", self._setup_grafana_integration),
            ("Slack", self._setup_slack_integration)
        ]
        
        for i, (name, setup_func) in enumerate(integration_tasks):
            try:
                await setup_func()
                progress.update(task, completed=(i + 1) / len(integration_tasks) * 100,
                              description=f"🔌 Setting up {name}")
            except Exception as e:
                logger.warning(f"Integration setup failed for {name}: {e}")
                
    async def _setup_pagerduty_integration(self):
        """Configure l'intégration PagerDuty"""
        # Test de l'API et configuration des webhooks
        api_key = os.getenv('PAGERDUTY_API_KEY')
        headers = {
            'Authorization': f'Token token={api_key}',
            'Accept': 'application/vnd.pagerduty+json;version=2'
        }
        
        async with aiohttp.ClientSession() as session:
            # Test de connexion
            async with session.get(
                'https://api.pagerduty.com/users',
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"PagerDuty API test failed: {response.status}")
                    
    async def _setup_prometheus_integration(self):
        """Configure l'intégration Prometheus"""
        prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pagerduty-integration'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
        """
        
        config_file = Path('prometheus.yml')
        async with aiofiles.open(config_file, 'w') as f:
            await f.write(prometheus_config)
            
    async def _setup_grafana_integration(self):
        """Configure l'intégration Grafana"""
        # Configuration des dashboards et datasources
        grafana_config = {
            'datasources': [
                {
                    'name': 'Prometheus',
                    'type': 'prometheus',
                    'url': 'http://localhost:9090'
                }
            ]
        }
        
        config_file = Path('grafana_datasources.json')
        async with aiofiles.open(config_file, 'w') as f:
            await f.write(json.dumps(grafana_config, indent=2))
            
    async def _setup_slack_integration(self):
        """Configure l'intégration Slack"""
        slack_token = os.getenv('SLACK_BOT_TOKEN')
        if slack_token:
            # Test de connexion Slack
            headers = {'Authorization': f'Bearer {slack_token}'}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://slack.com/api/auth.test',
                    headers=headers
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Slack API test failed: {response.status}")
                        
    async def _configure_monitoring(self, progress: Progress, task: TaskID):
        """Configure le monitoring"""
        monitoring_tasks = [
            ("Metrics collection", self._configure_metrics),
            ("Alert rules", self._configure_alerts),
            ("Dashboards", self._configure_dashboards),
            ("Log aggregation", self._configure_logging)
        ]
        
        for i, (name, config_func) in enumerate(monitoring_tasks):
            await config_func()
            progress.update(task, completed=(i + 1) / len(monitoring_tasks) * 100,
                          description=f"📊 Configuring {name}")
            
    async def _configure_metrics(self):
        """Configure la collecte de métriques"""
        # Démarrage du serveur de métriques Prometheus
        metrics_config = {
            'port': 8000,
            'path': '/metrics',
            'enabled': True
        }
        
        config_file = Path('metrics_config.json')
        async with aiofiles.open(config_file, 'w') as f:
            await f.write(json.dumps(metrics_config, indent=2))
            
    async def _configure_alerts(self):
        """Configure les règles d'alerte"""
        alert_rules = """
groups:
  - name: pagerduty_alerts
    rules:
      - alert: PagerDutyHighErrorRate
        expr: rate(pagerduty_api_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in PagerDuty integration"
          
      - alert: PagerDutyAPIDown
        expr: up{job="pagerduty-integration"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PagerDuty integration is down"
        """
        
        rules_file = Path('alert_rules.yml')
        async with aiofiles.open(rules_file, 'w') as f:
            await f.write(alert_rules)
            
    async def _configure_dashboards(self):
        """Configure les dashboards"""
        dashboard_config = {
            'dashboard': {
                'title': 'PagerDuty Integration',
                'panels': [
                    {
                        'title': 'API Requests',
                        'type': 'graph',
                        'targets': [
                            {'expr': 'rate(pagerduty_requests_total[5m])'}
                        ]
                    }
                ]
            }
        }
        
        dashboard_file = Path('pagerduty_dashboard.json')
        async with aiofiles.open(dashboard_file, 'w') as f:
            await f.write(json.dumps(dashboard_config, indent=2))
            
    async def _configure_logging(self):
        """Configure l'agrégation de logs"""
        logging_config = {
            'version': 1,
            'handlers': {
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': 'pagerduty.log',
                    'formatter': 'detailed'
                }
            },
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'loggers': {
                'pagerduty': {
                    'handlers': ['file'],
                    'level': 'INFO'
                }
            }
        }
        
        logging_file = Path('logging.json')
        async with aiofiles.open(logging_file, 'w') as f:
            await f.write(json.dumps(logging_config, indent=2))
            
    async def _run_health_checks(self, progress: Progress, task: TaskID):
        """Exécute les vérifications de santé"""
        health_checks = [
            ("Service status", self._check_service_status),
            ("API endpoints", self._check_api_endpoints),
            ("Database connectivity", self._check_database_connectivity),
            ("External integrations", self._check_external_integrations),
            ("Performance metrics", self._check_performance_metrics)
        ]
        
        for i, (name, check_func) in enumerate(health_checks):
            try:
                await check_func()
                progress.update(task, completed=(i + 1) / len(health_checks) * 100,
                              description=f"✅ Checking {name}")
            except Exception as e:
                raise Exception(f"Health check failed for {name}: {e}")
                
    async def _check_service_status(self):
        """Vérifie le statut du service"""
        try:
            result = await self._run_command("systemctl is-active pagerduty")
            if "active" not in result.stdout:
                raise Exception("Service is not active")
        except Exception as e:
            raise Exception(f"Service status check failed: {e}")
            
    async def _check_api_endpoints(self):
        """Vérifie les endpoints API"""
        endpoints = [
            'http://localhost:8080/webhook/health',
            'http://localhost:8000/metrics'
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status >= 400:
                            raise Exception(f"HTTP {response.status}")
                except Exception as e:
                    raise Exception(f"Endpoint {endpoint} check failed: {e}")
                    
    async def _check_database_connectivity(self):
        """Vérifie la connectivité base de données"""
        try:
            redis = aioredis.from_url(self.redis_url)
            await redis.ping()
            await redis.close()
        except Exception as e:
            raise Exception(f"Database connectivity check failed: {e}")
            
    async def _check_external_integrations(self):
        """Vérifie les intégrations externes"""
        # Vérification PagerDuty API
        api_key = os.getenv('PAGERDUTY_API_KEY')
        headers = {
            'Authorization': f'Token token={api_key}',
            'Accept': 'application/vnd.pagerduty+json;version=2'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    'https://api.pagerduty.com/users',
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status >= 400:
                        raise Exception(f"PagerDuty API: HTTP {response.status}")
            except Exception as e:
                raise Exception(f"PagerDuty integration check failed: {e}")
                
    async def _check_performance_metrics(self):
        """Vérifie les métriques de performance"""
        # Vérification que les métriques sont collectées
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    'http://localhost:8000/metrics',
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Metrics endpoint: HTTP {response.status}")
                        
                    content = await response.text()
                    if 'pagerduty_' not in content:
                        raise Exception("No PagerDuty metrics found")
                        
            except Exception as e:
                raise Exception(f"Performance metrics check failed: {e}")
                
    async def _post_deployment_tasks(self, progress: Progress, task: TaskID):
        """Tâches post-déploiement"""
        post_tasks = [
            ("Generate documentation", self._generate_documentation),
            ("Send notifications", self._send_deployment_notifications),
            ("Update monitoring", self._update_monitoring_config),
            ("Cleanup temporary files", self._cleanup_temp_files)
        ]
        
        for i, (name, task_func) in enumerate(post_tasks):
            await task_func()
            progress.update(task, completed=(i + 1) / len(post_tasks) * 100,
                          description=f"📝 {name}")
            
    async def _generate_documentation(self):
        """Génère la documentation de déploiement"""
        deployment_doc = {
            'deployment_id': f"deploy_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'version': '4.0.0',
            'components_deployed': [
                'PagerDuty Integration',
                'Metrics Collection',
                'Webhook Processing',
                'Notification Engine',
                'AI Analysis'
            ],
            'endpoints': [
                'http://localhost:8080/webhook/pagerduty',
                'http://localhost:8000/metrics'
            ],
            'health_check_url': 'http://localhost:8080/webhook/health'
        }
        
        doc_file = Path('deployment_info.json')
        async with aiofiles.open(doc_file, 'w') as f:
            await f.write(json.dumps(deployment_doc, indent=2))
            
    async def _send_deployment_notifications(self):
        """Envoie les notifications de déploiement"""
        # Notification Slack, email, etc.
        notification_data = {
            'message': f'✅ PagerDuty integration deployed successfully to {self.environment}',
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment
        }
        
        # Log pour démonstration
        logger.info("Deployment notification", **notification_data)
        
    async def _update_monitoring_config(self):
        """Met à jour la configuration de monitoring"""
        # Mise à jour des configs Prometheus, Grafana, etc.
        pass
        
    async def _cleanup_temp_files(self):
        """Nettoie les fichiers temporaires"""
        temp_files = [
            '.env.deployment',
            'temp_config.yaml'
        ]
        
        for temp_file in temp_files:
            temp_path = Path(temp_file)
            if temp_path.exists():
                temp_path.unlink()
                
    async def _rollback_deployment(self, deployment_id: str):
        """Effectue un rollback en cas d'échec"""
        console.print(Panel.fit(
            f"🔄 Starting rollback for deployment {deployment_id}",
            title="Rollback",
            border_style="yellow"
        ))
        
        try:
            # Arrêt des services
            await self._run_command("systemctl stop pagerduty")
            
            # Restauration depuis backup
            latest_backup = max(self.backup_path.glob('backup_*'))
            if latest_backup.exists():
                # Restauration des fichiers
                for backup_file in latest_backup.glob('*'):
                    if backup_file.is_file():
                        await self._copy_file(backup_file, Path(backup_file.name))
                        
            console.print("[green]Rollback completed successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Rollback failed: {e}[/red]")
            
    async def _cleanup_deployment(self, deployment_id: str):
        """Nettoie après déploiement"""
        # Nettoyage des fichiers temporaires, logs, etc.
        logger.info(f"Cleaning up deployment {deployment_id}")
        
    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        """Exécute une commande système"""
        try:
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise Exception(f"Command failed: {stderr.decode()}")
                
            return type('Result', (), {
                'stdout': stdout.decode(),
                'stderr': stderr.decode(),
                'returncode': result.returncode
            })()
            
        except Exception as e:
            raise Exception(f"Command execution failed: {e}")

async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Deploy PagerDuty Integration")
    parser.add_argument('--config', '-c', default='deployment_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--environment', '-e', default='development',
                       help='Deployment environment')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force deployment even on errors')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Perform a dry run without actual deployment')
    
    args = parser.parse_args()
    
    # Configuration du logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    try:
        # Chargement de la configuration
        config = DeploymentConfig(args.config)
        
        # Création du déployeur
        deployer = PagerDutyDeployer(config)
        
        if args.dry_run:
            console.print("[yellow]Dry run mode - no actual deployment will be performed[/yellow]")
            return
            
        # Déploiement
        success = await deployer.deploy(force=args.force)
        
        if success:
            console.print("[green]✅ Deployment completed successfully![/green]")
            sys.exit(0)
        else:
            console.print("[red]❌ Deployment failed![/red]")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Deployment interrupted by user[/yellow]")
        sys.exit(130)
        
    except Exception as e:
        console.print(f"[red]Deployment error: {e}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
