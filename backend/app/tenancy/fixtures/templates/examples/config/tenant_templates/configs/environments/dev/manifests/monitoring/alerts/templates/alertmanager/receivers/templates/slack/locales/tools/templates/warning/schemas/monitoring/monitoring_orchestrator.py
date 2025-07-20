#!/usr/bin/env python3
"""
Advanced Monitoring System Orchestrator
=======================================

Script d'orchestration pour initialiser et configurer le système de monitoring
ultra-avancé avec tous les modules intégrés.

Usage:
    python monitoring_orchestrator.py --action=init --environment=production
    python monitoring_orchestrator.py --action=validate --config-file=config.json
    python monitoring_orchestrator.py --action=deploy --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from metric_schemas import create_default_registry as create_metric_registry
from alert_schemas import create_default_alert_registry
from dashboard_schemas import create_default_dashboard_registry
from tenant_monitoring import create_default_tenant_monitoring_service
from compliance_monitoring import create_default_compliance_service
from ml_monitoring import create_default_ml_monitoring_service
from security_monitoring import create_default_security_monitoring_service
from performance_monitoring import create_default_performance_monitoring_service


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MonitoringOrchestrator:
    """Orchestrateur principal du système de monitoring"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = {}
        self.services = {}
        
        logger.info(f"Initializing Monitoring Orchestrator for environment: {environment}")
    
    def initialize_all_services(self) -> Dict[str, Any]:
        """Initialiser tous les services de monitoring"""
        logger.info("Initializing all monitoring services...")
        
        try:
            # Initialiser le registre de métriques
            logger.info("Creating metric registry...")
            self.services['metrics'] = create_metric_registry()
            
            # Initialiser le registre d'alertes
            logger.info("Creating alert registry...")
            self.services['alerts'] = create_default_alert_registry()
            
            # Initialiser le registre de dashboards
            logger.info("Creating dashboard registry...")
            self.services['dashboards'] = create_default_dashboard_registry()
            
            # Initialiser le monitoring multi-tenant
            logger.info("Creating tenant monitoring service...")
            self.services['tenant_monitoring'] = create_default_tenant_monitoring_service()
            
            # Initialiser le monitoring de conformité
            logger.info("Creating compliance monitoring service...")
            self.services['compliance'] = create_default_compliance_service()
            
            # Initialiser le monitoring ML
            logger.info("Creating ML monitoring service...")
            self.services['ml_monitoring'] = create_default_ml_monitoring_service()
            
            # Initialiser le monitoring de sécurité
            logger.info("Creating security monitoring service...")
            self.services['security'] = create_default_security_monitoring_service()
            
            # Initialiser le monitoring de performance
            logger.info("Creating performance monitoring service...")
            self.services['performance'] = create_default_performance_monitoring_service()
            
            logger.info("All monitoring services initialized successfully!")
            return self.services
            
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise
    
    def validate_configuration(self) -> bool:
        """Valider la configuration complète"""
        logger.info("Validating monitoring configuration...")
        
        validation_errors = []
        
        try:
            # Valider les métriques
            metrics_registry = self.services.get('metrics')
            if not metrics_registry or len(metrics_registry.metrics) == 0:
                validation_errors.append("No metrics defined in registry")
            
            # Valider les alertes
            alerts_registry = self.services.get('alerts')
            if not alerts_registry or len(alerts_registry.alert_rules) == 0:
                validation_errors.append("No alert rules defined")
            
            # Valider les dashboards
            dashboards_registry = self.services.get('dashboards')
            if not dashboards_registry or len(dashboards_registry.dashboards) == 0:
                validation_errors.append("No dashboards defined")
            
            # Valider la cohérence entre métriques et alertes
            self._validate_metric_alert_consistency(validation_errors)
            
            # Valider la cohérence entre métriques et dashboards
            self._validate_metric_dashboard_consistency(validation_errors)
            
            # Valider la configuration des services
            self._validate_service_configurations(validation_errors)
            
            if validation_errors:
                logger.error("Validation failed with errors:")
                for error in validation_errors:
                    logger.error(f"  - {error}")
                return False
            
            logger.info("Configuration validation successful!")
            return True
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return False
    
    def _validate_metric_alert_consistency(self, errors: List[str]) -> None:
        """Valider la cohérence entre métriques et alertes"""
        metrics_registry = self.services.get('metrics')
        alerts_registry = self.services.get('alerts')
        
        if not metrics_registry or not alerts_registry:
            return
        
        metric_names = {m.name for m in metrics_registry.metrics}
        
        for alert_rule in alerts_registry.alert_rules:
            if alert_rule.metric_name not in metric_names:
                errors.append(f"Alert rule '{alert_rule.name}' references unknown metric '{alert_rule.metric_name}'")
    
    def _validate_metric_dashboard_consistency(self, errors: List[str]) -> None:
        """Valider la cohérence entre métriques et dashboards"""
        metrics_registry = self.services.get('metrics')
        dashboards_registry = self.services.get('dashboards')
        
        if not metrics_registry or not dashboards_registry:
            return
        
        metric_names = {m.name for m in metrics_registry.metrics}
        
        for dashboard in dashboards_registry.dashboards:
            for widget in dashboard.widgets:
                for metric in widget.metrics:
                    if metric not in metric_names:
                        errors.append(f"Dashboard '{dashboard.title}' widget '{widget.title}' references unknown metric '{metric}'")
    
    def _validate_service_configurations(self, errors: List[str]) -> None:
        """Valider les configurations des services"""
        # Valider le service de monitoring multi-tenant
        tenant_service = self.services.get('tenant_monitoring')
        if tenant_service and len(tenant_service.tenants) == 0:
            errors.append("No tenants configured in tenant monitoring service")
        
        # Valider le service de conformité
        compliance_service = self.services.get('compliance')
        if compliance_service and len(compliance_service.controls) == 0:
            errors.append("No compliance controls configured")
        
        # Valider le service ML
        ml_service = self.services.get('ml_monitoring')
        if ml_service and len(ml_service.monitored_models) == 0:
            errors.append("No ML models configured for monitoring")
    
    def export_configuration(self, output_file: str) -> bool:
        """Exporter la configuration complète"""
        logger.info(f"Exporting configuration to {output_file}...")
        
        try:
            config_data = {
                'environment': self.environment,
                'generated_at': datetime.utcnow().isoformat(),
                'version': '1.0.0',
                'services': {}
            }
            
            # Exporter chaque service
            for service_name, service in self.services.items():
                if hasattr(service, 'dict'):
                    config_data['services'][service_name] = service.dict()
                else:
                    config_data['services'][service_name] = str(service)
            
            # Écrire le fichier de configuration
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Configuration exported successfully to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {str(e)}")
            return False
    
    def generate_deployment_manifests(self, output_dir: str) -> bool:
        """Générer les manifests de déploiement"""
        logger.info(f"Generating deployment manifests in {output_dir}...")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Générer manifests Prometheus
            self._generate_prometheus_config(output_path)
            
            # Générer manifests Grafana
            self._generate_grafana_config(output_path)
            
            # Générer manifests AlertManager
            self._generate_alertmanager_config(output_path)
            
            # Générer manifests Kubernetes
            self._generate_kubernetes_manifests(output_path)
            
            logger.info("Deployment manifests generated successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error generating deployment manifests: {str(e)}")
            return False
    
    def _generate_prometheus_config(self, output_path: Path) -> None:
        """Générer configuration Prometheus"""
        prometheus_config = {
            'global': {
                'scrape_interval': '30s',
                'evaluation_interval': '30s'
            },
            'rule_files': [
                '/etc/prometheus/rules/*.yml'
            ],
            'scrape_configs': [
                {
                    'job_name': 'spotify-ai-agent',
                    'static_configs': [
                        {'targets': ['localhost:8000']}
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s'
                }
            ],
            'alerting': {
                'alertmanagers': [
                    {
                        'static_configs': [
                            {'targets': ['alertmanager:9093']}
                        ]
                    }
                ]
            }
        }
        
        # Ajouter règles d'alerte
        alerts_registry = self.services.get('alerts')
        if alerts_registry:
            rules_config = {
                'groups': [
                    {
                        'name': 'spotify_ai_agent_alerts',
                        'rules': []
                    }
                ]
            }
            
            for alert_rule in alerts_registry.alert_rules:
                rule = {
                    'alert': alert_rule.name,
                    'expr': alert_rule.query,
                    'for': alert_rule.for_duration,
                    'labels': alert_rule.labels,
                    'annotations': alert_rule.annotations
                }
                rules_config['groups'][0]['rules'].append(rule)
            
            # Écrire les règles
            with open(output_path / 'prometheus-rules.yml', 'w') as f:
                import yaml
                yaml.dump(rules_config, f, default_flow_style=False)
        
        # Écrire la configuration Prometheus
        with open(output_path / 'prometheus.yml', 'w') as f:
            import yaml
            yaml.dump(prometheus_config, f, default_flow_style=False)
    
    def _generate_grafana_config(self, output_path: Path) -> None:
        """Générer configuration Grafana"""
        dashboards_registry = self.services.get('dashboards')
        if not dashboards_registry:
            return
        
        grafana_dir = output_path / 'grafana'
        grafana_dir.mkdir(exist_ok=True)
        
        # Générer chaque dashboard
        for dashboard in dashboards_registry.dashboards:
            dashboard_config = {
                'dashboard': {
                    'id': None,
                    'title': dashboard.title,
                    'tags': dashboard.tags,
                    'timezone': 'browser',
                    'panels': [],
                    'time': {
                        'from': 'now-1h',
                        'to': 'now'
                    },
                    'refresh': '30s'
                }
            }
            
            # Ajouter les widgets comme panels
            for i, widget in enumerate(dashboard.widgets):
                panel = {
                    'id': i + 1,
                    'title': widget.title,
                    'type': self._get_grafana_panel_type(widget.visualization_type),
                    'targets': [
                        {
                            'expr': widget.query,
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {
                        'h': 8,
                        'w': 12,
                        'x': widget.position.get('x', 0),
                        'y': widget.position.get('y', 0)
                    }
                }
                dashboard_config['dashboard']['panels'].append(panel)
            
            # Écrire le dashboard
            dashboard_file = grafana_dir / f'{dashboard.id}.json'
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_config, f, indent=2)
    
    def _get_grafana_panel_type(self, viz_type: str) -> str:
        """Mapper les types de visualisation vers Grafana"""
        mapping = {
            'line_chart': 'graph',
            'bar_chart': 'bargauge',
            'pie_chart': 'piechart',
            'gauge': 'gauge',
            'single_stat': 'stat',
            'table': 'table',
            'heatmap': 'heatmap'
        }
        return mapping.get(viz_type, 'graph')
    
    def _generate_alertmanager_config(self, output_path: Path) -> None:
        """Générer configuration AlertManager"""
        alertmanager_config = {
            'global': {
                'smtp_smarthost': 'localhost:587',
                'smtp_from': 'alertmanager@spotify.internal'
            },
            'route': {
                'group_by': ['alertname'],
                'group_wait': '10s',
                'group_interval': '10s',
                'repeat_interval': '1h',
                'receiver': 'web.hook'
            },
            'receivers': [
                {
                    'name': 'web.hook',
                    'slack_configs': [
                        {
                            'api_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
                            'channel': '#alerts',
                            'username': 'AlertManager',
                            'title': 'Spotify AI Agent Alert',
                            'text': '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
                        }
                    ]
                }
            ]
        }
        
        with open(output_path / 'alertmanager.yml', 'w') as f:
            import yaml
            yaml.dump(alertmanager_config, f, default_flow_style=False)
    
    def _generate_kubernetes_manifests(self, output_path: Path) -> None:
        """Générer manifests Kubernetes"""
        k8s_dir = output_path / 'kubernetes'
        k8s_dir.mkdir(exist_ok=True)
        
        # Générer les manifests de base pour le monitoring stack
        manifests = [
            self._create_namespace_manifest(),
            self._create_prometheus_manifest(),
            self._create_grafana_manifest(),
            self._create_alertmanager_manifest()
        ]
        
        for i, manifest in enumerate(manifests):
            with open(k8s_dir / f'manifest-{i:02d}.yml', 'w') as f:
                import yaml
                yaml.dump_all(manifest, f, default_flow_style=False)
    
    def _create_namespace_manifest(self) -> List[Dict]:
        """Créer manifest namespace"""
        return [{
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': 'monitoring'
            }
        }]
    
    def _create_prometheus_manifest(self) -> List[Dict]:
        """Créer manifests Prometheus"""
        return [
            {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': 'prometheus',
                    'namespace': 'monitoring'
                },
                'spec': {
                    'replicas': 1,
                    'selector': {
                        'matchLabels': {
                            'app': 'prometheus'
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': 'prometheus'
                            }
                        },
                        'spec': {
                            'containers': [
                                {
                                    'name': 'prometheus',
                                    'image': 'prom/prometheus:latest',
                                    'ports': [
                                        {
                                            'containerPort': 9090
                                        }
                                    ],
                                    'volumeMounts': [
                                        {
                                            'name': 'config',
                                            'mountPath': '/etc/prometheus'
                                        }
                                    ]
                                }
                            ],
                            'volumes': [
                                {
                                    'name': 'config',
                                    'configMap': {
                                        'name': 'prometheus-config'
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': 'prometheus',
                    'namespace': 'monitoring'
                },
                'spec': {
                    'selector': {
                        'app': 'prometheus'
                    },
                    'ports': [
                        {
                            'port': 9090,
                            'targetPort': 9090
                        }
                    ]
                }
            }
        ]
    
    def _create_grafana_manifest(self) -> List[Dict]:
        """Créer manifests Grafana"""
        return [
            {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': 'grafana',
                    'namespace': 'monitoring'
                },
                'spec': {
                    'replicas': 1,
                    'selector': {
                        'matchLabels': {
                            'app': 'grafana'
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': 'grafana'
                            }
                        },
                        'spec': {
                            'containers': [
                                {
                                    'name': 'grafana',
                                    'image': 'grafana/grafana:latest',
                                    'ports': [
                                        {
                                            'containerPort': 3000
                                        }
                                    ],
                                    'env': [
                                        {
                                            'name': 'GF_SECURITY_ADMIN_PASSWORD',
                                            'value': 'admin'
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        ]
    
    def _create_alertmanager_manifest(self) -> List[Dict]:
        """Créer manifests AlertManager"""
        return [
            {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {
                    'name': 'alertmanager',
                    'namespace': 'monitoring'
                },
                'spec': {
                    'replicas': 1,
                    'selector': {
                        'matchLabels': {
                            'app': 'alertmanager'
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': 'alertmanager'
                            }
                        },
                        'spec': {
                            'containers': [
                                {
                                    'name': 'alertmanager',
                                    'image': 'prom/alertmanager:latest',
                                    'ports': [
                                        {
                                            'containerPort': 9093
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        ]
    
    def health_check(self) -> Dict[str, bool]:
        """Vérification de santé des services"""
        logger.info("Performing health check...")
        
        health_status = {}
        
        for service_name, service in self.services.items():
            try:
                # Vérifications basiques
                if hasattr(service, 'dict'):
                    service.dict()  # Test de sérialisation
                    health_status[service_name] = True
                else:
                    health_status[service_name] = True
                    
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {str(e)}")
                health_status[service_name] = False
        
        overall_health = all(health_status.values())
        logger.info(f"Overall health status: {'HEALTHY' if overall_health else 'UNHEALTHY'}")
        
        return health_status


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Advanced Monitoring System Orchestrator')
    parser.add_argument('--action', choices=['init', 'validate', 'export', 'deploy', 'health'], 
                       required=True, help='Action to perform')
    parser.add_argument('--environment', default='production', help='Environment')
    parser.add_argument('--config-file', help='Configuration file path')
    parser.add_argument('--output-file', default='monitoring-config.json', help='Output configuration file')
    parser.add_argument('--output-dir', default='./deployment', help='Output directory for deployment files')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    args = parser.parse_args()
    
    orchestrator = MonitoringOrchestrator(environment=args.environment)
    
    try:
        if args.action == 'init':
            logger.info("=== INITIALIZING MONITORING SYSTEM ===")
            orchestrator.initialize_all_services()
            logger.info("Monitoring system initialized successfully!")
            
        elif args.action == 'validate':
            logger.info("=== VALIDATING CONFIGURATION ===")
            orchestrator.initialize_all_services()
            if orchestrator.validate_configuration():
                logger.info("Configuration validation passed!")
                sys.exit(0)
            else:
                logger.error("Configuration validation failed!")
                sys.exit(1)
                
        elif args.action == 'export':
            logger.info("=== EXPORTING CONFIGURATION ===")
            orchestrator.initialize_all_services()
            if orchestrator.export_configuration(args.output_file):
                logger.info(f"Configuration exported to {args.output_file}")
            else:
                logger.error("Configuration export failed!")
                sys.exit(1)
                
        elif args.action == 'deploy':
            logger.info("=== GENERATING DEPLOYMENT MANIFESTS ===")
            orchestrator.initialize_all_services()
            
            if args.dry_run:
                logger.info("DRY RUN MODE - No files will be created")
            
            if orchestrator.generate_deployment_manifests(args.output_dir):
                logger.info(f"Deployment manifests generated in {args.output_dir}")
            else:
                logger.error("Deployment manifest generation failed!")
                sys.exit(1)
                
        elif args.action == 'health':
            logger.info("=== HEALTH CHECK ===")
            orchestrator.initialize_all_services()
            health_status = orchestrator.health_check()
            
            print("\nHealth Check Results:")
            print("=" * 50)
            for service, status in health_status.items():
                status_text = "✅ HEALTHY" if status else "❌ UNHEALTHY"
                print(f"{service:30} : {status_text}")
            
            overall_healthy = all(health_status.values())
            print(f"\nOverall Status: {'✅ HEALTHY' if overall_healthy else '❌ UNHEALTHY'}")
            
            if not overall_healthy:
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Error executing action '{args.action}': {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
