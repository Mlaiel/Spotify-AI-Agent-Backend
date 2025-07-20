"""
Monitoring and Alerting Templates Module
========================================

Ce module fournit des templates d'alertes et de monitoring ultra-avancés 
pour l'architecture multi-tenant du Spotify AI Agent.

Fonctionnalités principales:
- Templates d'alertes Prometheus/Grafana
- Règles d'alerting intelligent
- Monitoring multi-tenant
- Escalation automatique
- Intégration Slack/Discord/Email
- Analytics prédictifs
- Auto-remédiation

Auteur: Fahed Mlaiel
Équipe: Lead Dev + Architecte IA, Développeur Backend Senior, 
        Ingénieur Machine Learning, Spécialiste Sécurité Backend,
        Architecte Microservices
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"

# Configuration des templates d'alertes
ALERT_TEMPLATES_CONFIG = {
    "version": "v1",
    "tenant_isolation": True,
    "auto_scaling": True,
    "predictive_alerts": True,
    "multi_channel_notifications": True,
    "self_healing": True
}

# Niveaux de criticité
ALERT_LEVELS = {
    "CRITICAL": {"priority": 1, "escalation_time": 60},
    "HIGH": {"priority": 2, "escalation_time": 300},
    "MEDIUM": {"priority": 3, "escalation_time": 900},
    "LOW": {"priority": 4, "escalation_time": 1800},
    "INFO": {"priority": 5, "escalation_time": 3600}
}

# Canaux de notification
NOTIFICATION_CHANNELS = {
    "slack": {"enabled": True, "webhook_url": "${SLACK_WEBHOOK_URL}"},
    "discord": {"enabled": True, "webhook_url": "${DISCORD_WEBHOOK_URL}"},
    "email": {"enabled": True, "smtp_config": "${EMAIL_CONFIG}"},
    "pagerduty": {"enabled": True, "api_key": "${PAGERDUTY_API_KEY}"},
    "teams": {"enabled": True, "webhook_url": "${TEAMS_WEBHOOK_URL}"}
}

class AlertTemplateManager:
    """Gestionnaire des templates d'alertes ultra-avancé"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Charge tous les templates d'alertes"""
        for template_file in self.config_path.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_name = template_file.stem
                    self.templates[template_name] = yaml.safe_load(f)
                    logger.info(f"Template d'alerte chargé: {template_name}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du template {template_file}: {e}")
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Récupère un template d'alerte par nom"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """Liste tous les templates disponibles"""
        return list(self.templates.keys())
    
    def validate_template(self, template: Dict[str, Any]) -> bool:
        """Valide la structure d'un template d'alerte"""
        required_fields = ["name", "description", "rules", "notifications"]
        return all(field in template for field in required_fields)

# Instance globale
alert_manager = AlertTemplateManager()
