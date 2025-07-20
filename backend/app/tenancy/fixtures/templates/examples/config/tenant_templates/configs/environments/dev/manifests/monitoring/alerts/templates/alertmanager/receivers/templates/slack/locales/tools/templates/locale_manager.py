"""
Locale Manager - Gestionnaire de localisation pour templates Slack
Système avancé de gestion multi-langue avec formatage contextuel
"""

import json
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
from pathlib import Path
import locale as system_locale
from babel import Locale, dates, numbers
from babel.support import Translations
import gettext
from dataclasses import dataclass
from enum import Enum


class SupportedLocale(Enum):
    """Locales supportées par le système"""
    FRENCH = "fr-FR"
    ENGLISH = "en-US"
    GERMAN = "de-DE"
    SPANISH = "es-ES"
    ITALIAN = "it-IT"
    PORTUGUESE = "pt-BR"
    JAPANESE = "ja-JP"
    KOREAN = "ko-KR"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"


@dataclass
class LocaleConfig:
    """Configuration d'une locale"""
    code: str
    name: str
    native_name: str
    direction: str  # 'ltr' ou 'rtl'
    date_format: str
    time_format: str
    number_format: str
    currency_symbol: str
    timezone: str
    emoji_support: bool = True


class LocaleManager:
    """
    Gestionnaire avancé de localisation pour templates Slack
    
    Fonctionnalités :
    - Support multi-langue complet
    - Formatage contextuel des dates/nombres
    - Gestion des fuseaux horaires
    - Templates de messages localisés
    - Pluralisation intelligente
    - Formatage des devises
    - Support RTL
    """

    def __init__(
        self,
        locales_dir: str = "/locales",
        default_locale: str = "fr-FR",
        fallback_locale: str = "en-US",
        timezone_aware: bool = True
    ):
        self.locales_dir = Path(locales_dir)
        self.default_locale = default_locale
        self.fallback_locale = fallback_locale
        self.timezone_aware = timezone_aware
        
        self.logger = logging.getLogger(__name__)
        
        # Cache des traductions
        self._translations_cache: Dict[str, Dict[str, Any]] = {}
        self._locale_configs: Dict[str, LocaleConfig] = {}
        
        # Initialisation
        self._init_locale_configs()
        self._load_translations()

    def _init_locale_configs(self):
        """Initialise les configurations des locales"""
        
        configs = {
            "fr-FR": LocaleConfig(
                code="fr-FR",
                name="French",
                native_name="Français",
                direction="ltr",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="#,##0.##",
                currency_symbol="€",
                timezone="Europe/Paris"
            ),
            "en-US": LocaleConfig(
                code="en-US",
                name="English",
                native_name="English",
                direction="ltr", 
                date_format="%m/%d/%Y",
                time_format="%I:%M:%S %p",
                number_format="#,##0.##",
                currency_symbol="$",
                timezone="America/New_York"
            ),
            "de-DE": LocaleConfig(
                code="de-DE",
                name="German",
                native_name="Deutsch",
                direction="ltr",
                date_format="%d.%m.%Y",
                time_format="%H:%M:%S",
                number_format="#.##0,##",
                currency_symbol="€",
                timezone="Europe/Berlin"
            ),
            "es-ES": LocaleConfig(
                code="es-ES",
                name="Spanish", 
                native_name="Español",
                direction="ltr",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="#.##0,##",
                currency_symbol="€",
                timezone="Europe/Madrid"
            ),
            "ja-JP": LocaleConfig(
                code="ja-JP",
                name="Japanese",
                native_name="日本語",
                direction="ltr",
                date_format="%Y/%m/%d",
                time_format="%H:%M:%S",
                number_format="#,##0.##",
                currency_symbol="¥",
                timezone="Asia/Tokyo"
            ),
            "zh-CN": LocaleConfig(
                code="zh-CN",
                name="Chinese Simplified",
                native_name="简体中文",
                direction="ltr",
                date_format="%Y/%m/%d",
                time_format="%H:%M:%S",
                number_format="#,##0.##",
                currency_symbol="¥",
                timezone="Asia/Shanghai"
            )
        }
        
        self._locale_configs.update(configs)

    def _load_translations(self):
        """Charge toutes les traductions disponibles"""
        
        if not self.locales_dir.exists():
            self.locales_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations()
        
        for locale_code in self._locale_configs.keys():
            self._load_locale_translations(locale_code)

    def _create_default_translations(self):
        """Crée les fichiers de traduction par défaut"""
        
        # Traductions françaises
        fr_translations = {
            "alerts": {
                "critical": "Critique",
                "warning": "Avertissement", 
                "info": "Information",
                "recovery": "Récupération",
                "resolved": "Résolu",
                "firing": "Déclenché",
                "silence": "Mis en silence"
            },
            "labels": {
                "severity": "Sévérité",
                "instance": "Instance",
                "job": "Tâche",
                "alertname": "Nom de l'alerte",
                "description": "Description",
                "summary": "Résumé",
                "runbook": "Guide d'intervention",
                "dashboard": "Tableau de bord"
            },
            "actions": {
                "view_details": "Voir les détails",
                "silence_alert": "Mettre en silence",
                "acknowledge": "Acquitter",
                "escalate": "Escalader",
                "view_logs": "Voir les logs",
                "view_metrics": "Voir les métriques"
            },
            "time": {
                "now": "maintenant",
                "minutes_ago": "il y a {count} minute(s)",
                "hours_ago": "il y a {count} heure(s)",
                "days_ago": "il y a {count} jour(s)",
                "duration_format": "{hours}h {minutes}m {seconds}s"
            },
            "messages": {
                "alert_fired": "🚨 Alerte déclenchée",
                "alert_resolved": "✅ Alerte résolue",
                "multiple_alerts": "{count} alertes en cours",
                "system_status": "État du système : {status}",
                "maintenance_window": "Fenêtre de maintenance en cours",
                "escalation_notice": "⚠️ Escalade après {duration}"
            }
        }
        
        # Traductions anglaises
        en_translations = {
            "alerts": {
                "critical": "Critical",
                "warning": "Warning",
                "info": "Info", 
                "recovery": "Recovery",
                "resolved": "Resolved",
                "firing": "Firing",
                "silence": "Silenced"
            },
            "labels": {
                "severity": "Severity",
                "instance": "Instance",
                "job": "Job",
                "alertname": "Alert Name",
                "description": "Description",
                "summary": "Summary",
                "runbook": "Runbook",
                "dashboard": "Dashboard"
            },
            "actions": {
                "view_details": "View Details",
                "silence_alert": "Silence Alert",
                "acknowledge": "Acknowledge",
                "escalate": "Escalate",
                "view_logs": "View Logs",
                "view_metrics": "View Metrics"
            },
            "time": {
                "now": "now",
                "minutes_ago": "{count} minute(s) ago",
                "hours_ago": "{count} hour(s) ago",
                "days_ago": "{count} day(s) ago",
                "duration_format": "{hours}h {minutes}m {seconds}s"
            },
            "messages": {
                "alert_fired": "🚨 Alert Fired",
                "alert_resolved": "✅ Alert Resolved",
                "multiple_alerts": "{count} alerts active",
                "system_status": "System Status: {status}",
                "maintenance_window": "Maintenance window in progress",
                "escalation_notice": "⚠️ Escalated after {duration}"
            }
        }
        
        # Traductions allemandes
        de_translations = {
            "alerts": {
                "critical": "Kritisch",
                "warning": "Warnung",
                "info": "Information",
                "recovery": "Wiederherstellung",
                "resolved": "Gelöst",
                "firing": "Ausgelöst",
                "silence": "Stumm geschaltet"
            },
            "labels": {
                "severity": "Schweregrad",
                "instance": "Instanz",
                "job": "Job",
                "alertname": "Alarmname",
                "description": "Beschreibung",
                "summary": "Zusammenfassung",
                "runbook": "Runbook",
                "dashboard": "Dashboard"
            },
            "actions": {
                "view_details": "Details anzeigen",
                "silence_alert": "Alarm stumm schalten",
                "acknowledge": "Bestätigen",
                "escalate": "Eskalieren",
                "view_logs": "Logs anzeigen",
                "view_metrics": "Metriken anzeigen"
            },
            "time": {
                "now": "jetzt",
                "minutes_ago": "vor {count} Minute(n)",
                "hours_ago": "vor {count} Stunde(n)",
                "days_ago": "vor {count} Tag(en)",
                "duration_format": "{hours}h {minutes}m {seconds}s"
            },
            "messages": {
                "alert_fired": "🚨 Alarm ausgelöst",
                "alert_resolved": "✅ Alarm gelöst",
                "multiple_alerts": "{count} aktive Alarme",
                "system_status": "Systemstatus: {status}",
                "maintenance_window": "Wartungsfenster läuft",
                "escalation_notice": "⚠️ Eskaliert nach {duration}"
            }
        }
        
        # Sauvegarde des fichiers
        translations_map = {
            "fr-FR": fr_translations,
            "en-US": en_translations,
            "de-DE": de_translations
        }
        
        for locale_code, translations in translations_map.items():
            locale_file = self.locales_dir / f"{locale_code}.json"
            with open(locale_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, indent=2, ensure_ascii=False)

    def _load_locale_translations(self, locale_code: str):
        """Charge les traductions pour une locale spécifique"""
        
        locale_file = self.locales_dir / f"{locale_code}.json"
        
        if locale_file.exists():
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                    self._translations_cache[locale_code] = translations
                    self.logger.debug(f"Traductions chargées pour {locale_code}")
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement des traductions {locale_code}: {e}")
                self._translations_cache[locale_code] = {}
        else:
            self.logger.warning(f"Fichier de traductions manquant: {locale_file}")
            self._translations_cache[locale_code] = {}

    def get_localized_text(
        self,
        key: str,
        locale: str,
        category: str = "messages",
        **kwargs
    ) -> str:
        """
        Récupère un texte localisé
        
        Args:
            key: Clé de traduction
            locale: Code locale (ex: 'fr-FR')
            category: Catégorie de traduction
            **kwargs: Variables pour le formatage
            
        Returns:
            str: Texte localisé formaté
        """
        
        # Normalisation de la locale
        normalized_locale = self._normalize_locale(locale)
        
        # Récupération de la traduction
        translations = self._translations_cache.get(normalized_locale, {})
        
        # Recherche dans la catégorie
        category_translations = translations.get(category, {})
        text = category_translations.get(key)
        
        # Fallback vers la locale par défaut
        if text is None and normalized_locale != self.fallback_locale:
            fallback_translations = self._translations_cache.get(self.fallback_locale, {})
            fallback_category = fallback_translations.get(category, {})
            text = fallback_category.get(key)
        
        # Fallback final
        if text is None:
            text = f"[{category}.{key}]"
            self.logger.warning(f"Traduction manquante: {category}.{key} pour {normalized_locale}")
        
        # Formatage avec les variables
        try:
            if kwargs:
                text = text.format(**kwargs)
        except (KeyError, ValueError) as e:
            self.logger.error(f"Erreur de formatage pour {category}.{key}: {e}")
        
        return text

    def get_localized_texts(self, category: str, locale: str) -> Dict[str, str]:
        """Récupère toutes les traductions d'une catégorie"""
        
        normalized_locale = self._normalize_locale(locale)
        translations = self._translations_cache.get(normalized_locale, {})
        return translations.get(category, {})

    def format_datetime(
        self,
        dt: datetime,
        locale: str,
        format_type: str = "medium",
        include_timezone: bool = True
    ) -> str:
        """
        Formate une date/heure selon la locale
        
        Args:
            dt: Objet datetime
            locale: Code locale
            format_type: Type de format ('short', 'medium', 'long', 'full')
            include_timezone: Inclure le fuseau horaire
            
        Returns:
            str: Date/heure formatée
        """
        
        try:
            # Normalisation de la locale pour Babel
            babel_locale = Locale.parse(locale.replace('-', '_'))
            
            # Configuration du fuseau horaire
            if self.timezone_aware and include_timezone:
                locale_config = self._locale_configs.get(locale)
                if locale_config and dt.tzinfo is None:
                    # Ajouter le fuseau horaire par défaut
                    import pytz
                    tz = pytz.timezone(locale_config.timezone)
                    dt = tz.localize(dt)
            
            # Formatage avec Babel
            if format_type == "relative":
                return self._format_relative_time(dt, locale)
            else:
                return dates.format_datetime(
                    dt,
                    format=format_type,
                    locale=babel_locale
                )
        
        except Exception as e:
            self.logger.error(f"Erreur de formatage datetime: {e}")
            # Fallback vers format ISO
            return dt.isoformat()

    def _format_relative_time(self, dt: datetime, locale: str) -> str:
        """Formate un temps relatif (il y a X minutes)"""
        
        now = datetime.now(dt.tzinfo if dt.tzinfo else timezone.utc)
        delta = now - dt
        
        total_seconds = delta.total_seconds()
        
        if total_seconds < 60:
            return self.get_localized_text("now", locale, "time")
        elif total_seconds < 3600:
            minutes = int(total_seconds // 60)
            return self.get_localized_text(
                "minutes_ago", locale, "time", count=minutes
            )
        elif total_seconds < 86400:
            hours = int(total_seconds // 3600)
            return self.get_localized_text(
                "hours_ago", locale, "time", count=hours
            )
        else:
            days = int(total_seconds // 86400)
            return self.get_localized_text(
                "days_ago", locale, "time", count=days
            )

    def format_number(
        self,
        number: float,
        locale: str,
        format_type: str = "decimal"
    ) -> str:
        """
        Formate un nombre selon la locale
        
        Args:
            number: Nombre à formater
            locale: Code locale
            format_type: Type de format ('decimal', 'currency', 'percent')
            
        Returns:
            str: Nombre formaté
        """
        
        try:
            babel_locale = Locale.parse(locale.replace('-', '_'))
            
            if format_type == "currency":
                locale_config = self._locale_configs.get(locale)
                currency = locale_config.currency_symbol if locale_config else "EUR"
                return numbers.format_currency(number, currency, locale=babel_locale)
            elif format_type == "percent":
                return numbers.format_percent(number, locale=babel_locale)
            else:
                return numbers.format_decimal(number, locale=babel_locale)
        
        except Exception as e:
            self.logger.error(f"Erreur de formatage nombre: {e}")
            return str(number)

    def format_duration(self, seconds: int, locale: str) -> str:
        """Formate une durée en heures/minutes/secondes"""
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        
        return self.get_localized_text(
            "duration_format",
            locale,
            "time",
            hours=hours,
            minutes=minutes,
            seconds=remaining_seconds
        )

    def _normalize_locale(self, locale: str) -> str:
        """Normalise un code locale"""
        
        # Conversion des formats courants
        locale_mapping = {
            "fr": "fr-FR",
            "en": "en-US", 
            "de": "de-DE",
            "es": "es-ES",
            "it": "it-IT",
            "pt": "pt-BR",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "zh": "zh-CN"
        }
        
        normalized = locale_mapping.get(locale, locale)
        
        # Vérification de l'existence
        if normalized not in self._locale_configs:
            self.logger.warning(f"Locale non supportée: {locale}, fallback vers {self.default_locale}")
            return self.default_locale
        
        return normalized

    def get_supported_locales(self) -> List[str]:
        """Retourne la liste des locales supportées"""
        return list(self._locale_configs.keys())

    def get_locale_config(self, locale: str) -> Optional[LocaleConfig]:
        """Récupère la configuration d'une locale"""
        normalized_locale = self._normalize_locale(locale)
        return self._locale_configs.get(normalized_locale)

    def is_rtl_locale(self, locale: str) -> bool:
        """Vérifie si une locale utilise l'écriture RTL"""
        config = self.get_locale_config(locale)
        return config.direction == "rtl" if config else False

    def get_locale_emoji_flag(self, locale: str) -> str:
        """Retourne l'emoji drapeau pour une locale"""
        
        emoji_mapping = {
            "fr-FR": "🇫🇷",
            "en-US": "🇺🇸",
            "de-DE": "🇩🇪", 
            "es-ES": "🇪🇸",
            "it-IT": "🇮🇹",
            "pt-BR": "🇧🇷",
            "ja-JP": "🇯🇵",
            "ko-KR": "🇰🇷",
            "zh-CN": "🇨🇳",
            "zh-TW": "🇹🇼"
        }
        
        normalized_locale = self._normalize_locale(locale)
        return emoji_mapping.get(normalized_locale, "🌐")

    def pluralize(
        self,
        count: int,
        singular: str,
        plural: str,
        locale: str
    ) -> str:
        """
        Gère la pluralisation selon les règles de la locale
        
        Args:
            count: Nombre
            singular: Forme singulière
            plural: Forme plurielle
            locale: Code locale
            
        Returns:
            str: Forme correcte selon le nombre
        """
        
        # Règles de pluralisation simplifiées
        # Pour une implémentation complète, utiliser Babel ou ICU
        
        if locale.startswith("fr"):
            # Français: pluriel si > 1
            return singular if count <= 1 else plural
        elif locale.startswith("en"):
            # Anglais: pluriel si != 1
            return singular if count == 1 else plural
        elif locale.startswith("de"):
            # Allemand: pluriel si != 1
            return singular if count == 1 else plural
        else:
            # Règle par défaut
            return singular if count == 1 else plural

    def add_custom_translation(
        self,
        locale: str,
        category: str,
        key: str,
        value: str
    ):
        """Ajoute une traduction personnalisée"""
        
        normalized_locale = self._normalize_locale(locale)
        
        if normalized_locale not in self._translations_cache:
            self._translations_cache[normalized_locale] = {}
        
        if category not in self._translations_cache[normalized_locale]:
            self._translations_cache[normalized_locale][category] = {}
        
        self._translations_cache[normalized_locale][category][key] = value
        
        # Sauvegarde immédiate
        self._save_locale_translations(normalized_locale)

    def _save_locale_translations(self, locale: str):
        """Sauvegarde les traductions d'une locale"""
        
        locale_file = self.locales_dir / f"{locale}.json"
        translations = self._translations_cache.get(locale, {})
        
        try:
            with open(locale_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des traductions {locale}: {e}")

    def reload_translations(self):
        """Recharge toutes les traductions depuis les fichiers"""
        self._translations_cache.clear()
        self._load_translations()

    def get_translation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des traductions"""
        
        stats = {
            "supported_locales": len(self._locale_configs),
            "loaded_translations": len(self._translations_cache),
            "translation_coverage": {}
        }
        
        # Calcul de la couverture par locale
        base_translations = self._translations_cache.get(self.fallback_locale, {})
        total_keys = sum(len(category.keys()) for category in base_translations.values())
        
        for locale, translations in self._translations_cache.items():
            locale_keys = sum(len(category.keys()) for category in translations.values())
            coverage = (locale_keys / total_keys * 100) if total_keys > 0 else 0
            stats["translation_coverage"][locale] = round(coverage, 2)
        
        return stats
