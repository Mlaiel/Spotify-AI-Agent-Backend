"""
Gestionnaire de Localisation Avancé - Spotify AI Agent
====================================================

Module principal pour la gestion intelligente de la localisation des alertes,
formats de données et configurations régionales dans l'écosystème multi-tenant.

Fonctionnalités:
- Localisation intelligente des messages d'alerte
- Formatage adaptatif des données selon la culture
- Gestion des fuseaux horaires complexes
- Cache distribué haute performance
- Validation sécurisée des entrées

Author: Fahed Mlaiel
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import redis.asyncio as redis
from pathlib import Path
import aiofiles
import hashlib
from contextlib import asynccontextmanager

from . import LocaleType, LocaleDataConfig


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertCategory(Enum):
    """Catégories d'alertes"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"
    USER = "user"
    TENANT = "tenant"


@dataclass
class AlertTemplate:
    """Template d'alerte localisé"""
    id: str
    category: AlertCategory
    severity: AlertSeverity
    title_template: str
    message_template: str
    action_template: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizedAlert:
    """Alerte localisée générée"""
    alert_id: str
    locale: LocaleType
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    action: Optional[str]
    timestamp: datetime
    tenant_id: str
    parameters: Dict[str, Any]
    formatted_parameters: Dict[str, str]


class AlertLocalizer:
    """Localisateur intelligent d'alertes"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis_client
        self._templates_cache: Dict[Tuple[str, LocaleType], AlertTemplate] = {}
        self._locale_configs: Dict[LocaleType, LocaleDataConfig] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialise le localisateur"""
        if self._initialized:
            return
            
        try:
            await self._load_locale_configs()
            await self._load_alert_templates()
            self._initialized = True
            self.logger.info("AlertLocalizer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AlertLocalizer: {e}")
            raise

    async def _load_locale_configs(self):
        """Charge les configurations de locale"""
        # Configuration par défaut
        self._locale_configs = {
            LocaleType.EN_US: LocaleDataConfig(
                locale=LocaleType.EN_US,
                date_format="%Y-%m-%d %H:%M:%S UTC",
                time_format="%H:%M:%S",
                number_format="1,234.56",
                currency_symbol="$",
                decimal_separator=".",
                thousand_separator=","
            ),
            LocaleType.FR_FR: LocaleDataConfig(
                locale=LocaleType.FR_FR,
                date_format="%d/%m/%Y à %H:%M:%S UTC",
                time_format="%H:%M:%S",
                number_format="1 234,56",
                currency_symbol="€",
                decimal_separator=",",
                thousand_separator=" "
            ),
            LocaleType.DE_DE: LocaleDataConfig(
                locale=LocaleType.DE_DE,
                date_format="%d.%m.%Y um %H:%M:%S UTC",
                time_format="%H:%M:%S",
                number_format="1.234,56",
                currency_symbol="€",
                decimal_separator=",",
                thousand_separator="."
            )
        }

    async def _load_alert_templates(self):
        """Charge les templates d'alerte depuis les fichiers"""
        base_path = Path(__file__).parent / "locales"
        
        for locale in LocaleType:
            locale_path = base_path / locale.value
            if not locale_path.exists():
                continue
                
            alerts_file = locale_path / "alerts.json"
            if alerts_file.exists():
                await self._load_templates_for_locale(locale, alerts_file)

    async def _load_templates_for_locale(self, locale: LocaleType, alerts_file: Path):
        """Charge les templates pour une locale spécifique"""
        try:
            async with aiofiles.open(alerts_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                templates_data = json.loads(content)
                
            for template_id, template_data in templates_data.items():
                template = AlertTemplate(
                    id=template_id,
                    category=AlertCategory(template_data['category']),
                    severity=AlertSeverity(template_data['severity']),
                    title_template=template_data['title'],
                    message_template=template_data['message'],
                    action_template=template_data.get('action'),
                    metadata=template_data.get('metadata', {})
                )
                
                cache_key = (template_id, locale)
                self._templates_cache[cache_key] = template
                
        except Exception as e:
            self.logger.error(f"Failed to load templates for {locale.value}: {e}")

    async def generate_alert(
        self,
        alert_type: str,
        locale: LocaleType,
        tenant_id: str,
        parameters: Dict[str, Any],
        severity_override: Optional[AlertSeverity] = None
    ) -> LocalizedAlert:
        """Génère une alerte localisée"""
        if not self._initialized:
            await self.initialize()
            
        # Récupère le template
        template = await self._get_template(alert_type, locale)
        if not template:
            raise ValueError(f"Template not found for alert_type: {alert_type}, locale: {locale.value}")
        
        # Formate les paramètres selon la locale
        formatted_params = await self._format_parameters(parameters, locale)
        
        # Génère le contenu localisé
        severity = severity_override or template.severity
        title = await self._format_template_string(template.title_template, formatted_params, locale)
        message = await self._format_template_string(template.message_template, formatted_params, locale)
        action = None
        if template.action_template:
            action = await self._format_template_string(template.action_template, formatted_params, locale)
        
        # Crée l'alerte localisée
        alert = LocalizedAlert(
            alert_id=await self._generate_alert_id(alert_type, tenant_id),
            locale=locale,
            severity=severity,
            category=template.category,
            title=title,
            message=message,
            action=action,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            parameters=parameters,
            formatted_parameters=formatted_params
        )
        
        # Cache l'alerte si Redis est disponible
        if self.redis_client:
            await self._cache_alert(alert)
            
        return alert

    async def _get_template(self, alert_type: str, locale: LocaleType) -> Optional[AlertTemplate]:
        """Récupère un template d'alerte"""
        cache_key = (alert_type, locale)
        
        # Vérifie le cache local
        if cache_key in self._templates_cache:
            return self._templates_cache[cache_key]
        
        # Fallback vers locale par défaut
        fallback_key = (alert_type, LocaleType.EN_US)
        if fallback_key in self._templates_cache:
            self.logger.warning(f"Using fallback template for {alert_type} in {locale.value}")
            return self._templates_cache[fallback_key]
            
        return None

    async def _format_parameters(self, parameters: Dict[str, Any], locale: LocaleType) -> Dict[str, str]:
        """Formate les paramètres selon la locale"""
        config = self._locale_configs.get(locale, self._locale_configs[LocaleType.EN_US])
        formatted = {}
        
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                formatted[key] = self._format_number(value, config)
            elif isinstance(value, datetime):
                formatted[key] = value.strftime(config.date_format)
            elif key.endswith('_percentage') and isinstance(value, (int, float)):
                formatted[key] = f"{self._format_number(value, config)} %"
            elif key.endswith('_currency') and isinstance(value, (int, float)):
                formatted[key] = f"{self._format_number(value, config)} {config.currency_symbol}"
            else:
                formatted[key] = str(value)
                
        return formatted

    def _format_number(self, number: Union[int, float], config: LocaleDataConfig) -> str:
        """Formate un nombre selon la configuration locale"""
        if isinstance(number, int):
            formatted = f"{number:,}"
        else:
            formatted = f"{number:,.2f}"
            
        # Applique les séparateurs locaux
        formatted = formatted.replace(",", "TEMP_THOUSAND")
        formatted = formatted.replace(".", config.decimal_separator)
        formatted = formatted.replace("TEMP_THOUSAND", config.thousand_separator)
        
        return formatted

    async def _format_template_string(
        self, 
        template: str, 
        parameters: Dict[str, str], 
        locale: LocaleType
    ) -> str:
        """Formate une chaîne de template avec les paramètres"""
        try:
            # Utilise le formatage sécurisé des chaînes
            return template.format(**parameters)
        except KeyError as e:
            self.logger.error(f"Missing parameter {e} for template formatting")
            return template
        except Exception as e:
            self.logger.error(f"Template formatting error: {e}")
            return template

    async def _generate_alert_id(self, alert_type: str, tenant_id: str) -> str:
        """Génère un ID unique pour l'alerte"""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{alert_type}:{tenant_id}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _cache_alert(self, alert: LocalizedAlert):
        """Met en cache l'alerte dans Redis"""
        try:
            if self.redis_client:
                cache_key = f"alert:{alert.tenant_id}:{alert.alert_id}"
                alert_data = {
                    "alert_id": alert.alert_id,
                    "locale": alert.locale.value,
                    "severity": alert.severity.value,
                    "category": alert.category.value,
                    "title": alert.title,
                    "message": alert.message,
                    "action": alert.action,
                    "timestamp": alert.timestamp.isoformat(),
                    "tenant_id": alert.tenant_id,
                    "parameters": json.dumps(alert.parameters),
                    "formatted_parameters": json.dumps(alert.formatted_parameters)
                }
                
                await self.redis_client.hset(cache_key, mapping=alert_data)
                await self.redis_client.expire(cache_key, 86400)  # 24h TTL
                
        except Exception as e:
            self.logger.error(f"Failed to cache alert: {e}")

    async def get_cached_alert(self, tenant_id: str, alert_id: str) -> Optional[LocalizedAlert]:
        """Récupère une alerte depuis le cache"""
        if not self.redis_client:
            return None
            
        try:
            cache_key = f"alert:{tenant_id}:{alert_id}"
            alert_data = await self.redis_client.hgetall(cache_key)
            
            if not alert_data:
                return None
                
            return LocalizedAlert(
                alert_id=alert_data['alert_id'],
                locale=LocaleType(alert_data['locale']),
                severity=AlertSeverity(alert_data['severity']),
                category=AlertCategory(alert_data['category']),
                title=alert_data['title'],
                message=alert_data['message'],
                action=alert_data.get('action'),
                timestamp=datetime.fromisoformat(alert_data['timestamp']),
                tenant_id=alert_data['tenant_id'],
                parameters=json.loads(alert_data['parameters']),
                formatted_parameters=json.loads(alert_data['formatted_parameters'])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get cached alert: {e}")
            return None

    @asynccontextmanager
    async def batch_operations(self):
        """Context manager pour les opérations en lot"""
        try:
            yield self
        finally:
            # Nettoyage si nécessaire
            pass


# Instance globale du localisateur
alert_localizer = AlertLocalizer()

__all__ = [
    "AlertSeverity",
    "AlertCategory", 
    "AlertTemplate",
    "LocalizedAlert",
    "AlertLocalizer",
    "alert_localizer"
]
