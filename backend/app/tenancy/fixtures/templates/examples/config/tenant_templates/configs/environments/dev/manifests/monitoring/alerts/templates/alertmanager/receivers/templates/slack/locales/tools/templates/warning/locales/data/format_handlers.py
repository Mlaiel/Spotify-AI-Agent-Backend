"""
Gestionnaires de Formatage de Données - Spotify AI Agent
======================================================

Module spécialisé pour le formatage intelligent des données selon les conventions
locales et culturelles. Prend en charge la formatage des nombres, devises, dates,
pourcentages et métriques de performance.

Fonctionnalités:
- Formatage numérique adaptatif selon la locale
- Conversion et formatage des devises en temps réel
- Formatage des dates et heures avec fuseaux horaires
- Formatage des métriques de performance
- Validation et nettoyage des données d'entrée

Author: Fahed Mlaiel
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import re
import logging
from pathlib import Path
import json
import aiofiles

from . import LocaleType, LocaleDataConfig


class UnitSystem(Enum):
    """Systèmes d'unités de mesure"""
    METRIC = "metric"
    IMPERIAL = "imperial"
    MIXED = "mixed"


class DataType(Enum):
    """Types de données formatables"""
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    BYTES = "bytes"
    DURATION = "duration"
    RATE = "rate"
    TIMESTAMP = "timestamp"


@dataclass
class FormatRule:
    """Règle de formatage pour un type de données"""
    data_type: DataType
    locale: LocaleType
    pattern: str
    precision: int = 2
    unit_system: UnitSystem = UnitSystem.METRIC
    prefix: str = ""
    suffix: str = ""
    scale_factor: float = 1.0
    metadata: Dict[str, Any] = None


class NumberFormatter:
    """Formateur de nombres intelligent"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._format_rules: Dict[Tuple[DataType, LocaleType], FormatRule] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialise les règles de formatage par défaut"""
        # Règles pour les nombres (EN_US)
        self._format_rules[(DataType.NUMBER, LocaleType.EN_US)] = FormatRule(
            data_type=DataType.NUMBER,
            locale=LocaleType.EN_US,
            pattern="{:,.{precision}f}",
            precision=2
        )
        
        # Règles pour les nombres (FR_FR)
        self._format_rules[(DataType.NUMBER, LocaleType.FR_FR)] = FormatRule(
            data_type=DataType.NUMBER,
            locale=LocaleType.FR_FR,
            pattern="{:,.{precision}f}",
            precision=2
        )
        
        # Règles pour les pourcentages
        for locale in [LocaleType.EN_US, LocaleType.FR_FR, LocaleType.DE_DE]:
            self._format_rules[(DataType.PERCENTAGE, locale)] = FormatRule(
                data_type=DataType.PERCENTAGE,
                locale=locale,
                pattern="{:,.{precision}f}",
                precision=1,
                suffix=" %",
                scale_factor=1.0
            )
    
    def format_number(
        self, 
        value: Union[int, float, Decimal], 
        locale: LocaleType,
        precision: Optional[int] = None
    ) -> str:
        """Formate un nombre selon la locale"""
        try:
            # Récupère la règle de formatage
            rule = self._get_format_rule(DataType.NUMBER, locale)
            actual_precision = precision if precision is not None else rule.precision
            
            # Convertit en Decimal pour la précision
            if not isinstance(value, Decimal):
                value = Decimal(str(value))
            
            # Applique le facteur d'échelle
            scaled_value = value * Decimal(str(rule.scale_factor))
            
            # Formate selon la locale
            formatted = self._apply_locale_formatting(scaled_value, locale, actual_precision)
            
            # Ajoute préfixe et suffixe
            return f"{rule.prefix}{formatted}{rule.suffix}"
            
        except Exception as e:
            self.logger.error(f"Number formatting error: {e}")
            return str(value)
    
    def format_percentage(
        self, 
        value: Union[int, float], 
        locale: LocaleType,
        precision: int = 1
    ) -> str:
        """Formate un pourcentage"""
        try:
            rule = self._get_format_rule(DataType.PERCENTAGE, locale)
            formatted_number = self.format_number(value, locale, precision)
            return f"{formatted_number}{rule.suffix}"
        except Exception as e:
            self.logger.error(f"Percentage formatting error: {e}")
            return f"{value}%"
    
    def format_bytes(
        self, 
        bytes_value: int, 
        locale: LocaleType,
        binary: bool = True
    ) -> str:
        """Formate une taille en octets"""
        try:
            base = 1024 if binary else 1000
            units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'] if not binary else ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
            
            if bytes_value == 0:
                return f"0 {units[0]}"
            
            # Calcule l'unité appropriée
            unit_index = min(len(units) - 1, int(bytes_value.bit_length() / (10 if binary else 9.966)))
            scaled_value = bytes_value / (base ** unit_index)
            
            # Formate selon la locale
            formatted_number = self.format_number(scaled_value, locale, 1 if scaled_value < 10 else 0)
            return f"{formatted_number} {units[unit_index]}"
            
        except Exception as e:
            self.logger.error(f"Bytes formatting error: {e}")
            return f"{bytes_value} B"
    
    def format_duration(
        self, 
        seconds: Union[int, float], 
        locale: LocaleType,
        short_format: bool = False
    ) -> str:
        """Formate une durée en secondes"""
        try:
            if seconds < 0:
                return "0s"
            
            # Définit les unités de temps
            time_units = [
                (86400, 'd' if short_format else ' jour(s)'),
                (3600, 'h' if short_format else ' heure(s)'),
                (60, 'm' if short_format else ' minute(s)'),
                (1, 's' if short_format else ' seconde(s)')
            ]
            
            # Ajuste les unités selon la locale
            if locale == LocaleType.EN_US:
                time_units = [
                    (86400, 'd' if short_format else ' day(s)'),
                    (3600, 'h' if short_format else ' hour(s)'),
                    (60, 'm' if short_format else ' minute(s)'),
                    (1, 's' if short_format else ' second(s)')
                ]
            elif locale == LocaleType.DE_DE:
                time_units = [
                    (86400, 'T' if short_format else ' Tag(e)'),
                    (3600, 'Std' if short_format else ' Stunde(n)'),
                    (60, 'Min' if short_format else ' Minute(n)'),
                    (1, 'Sek' if short_format else ' Sekunde(n)')
                ]
            
            # Construit la chaîne de durée
            parts = []
            remaining = int(seconds)
            
            for unit_seconds, unit_name in time_units:
                if remaining >= unit_seconds:
                    count = remaining // unit_seconds
                    remaining %= unit_seconds
                    parts.append(f"{count}{unit_name}")
                    
                if len(parts) >= 2:  # Limite à 2 unités principales
                    break
            
            return ' '.join(parts) if parts else f"0{time_units[-1][1]}"
            
        except Exception as e:
            self.logger.error(f"Duration formatting error: {e}")
            return f"{seconds}s"
    
    def format_rate(
        self, 
        value: Union[int, float], 
        unit: str, 
        time_unit: str,
        locale: LocaleType
    ) -> str:
        """Formate un taux (ex: requêtes/seconde)"""
        try:
            formatted_value = self.format_number(value, locale, 1)
            
            # Traduit les unités selon la locale
            unit_translations = {
                LocaleType.FR_FR: {
                    'requests': 'requêtes',
                    'users': 'utilisateurs',
                    'errors': 'erreurs',
                    'second': 'seconde',
                    'minute': 'minute',
                    'hour': 'heure'
                },
                LocaleType.DE_DE: {
                    'requests': 'Anfragen',
                    'users': 'Benutzer',
                    'errors': 'Fehler',
                    'second': 'Sekunde',
                    'minute': 'Minute',
                    'hour': 'Stunde'
                }
            }
            
            translated_unit = unit
            translated_time_unit = time_unit
            
            if locale in unit_translations:
                translations = unit_translations[locale]
                translated_unit = translations.get(unit, unit)
                translated_time_unit = translations.get(time_unit, time_unit)
            
            return f"{formatted_value} {translated_unit}/{translated_time_unit}"
            
        except Exception as e:
            self.logger.error(f"Rate formatting error: {e}")
            return f"{value} {unit}/{time_unit}"
    
    def _get_format_rule(self, data_type: DataType, locale: LocaleType) -> FormatRule:
        """Récupère la règle de formatage pour un type et une locale"""
        key = (data_type, locale)
        if key in self._format_rules:
            return self._format_rules[key]
        
        # Fallback vers EN_US
        fallback_key = (data_type, LocaleType.EN_US)
        return self._format_rules.get(fallback_key, FormatRule(
            data_type=data_type,
            locale=locale,
            pattern="{:.2f}"
        ))
    
    def _apply_locale_formatting(
        self, 
        value: Decimal, 
        locale: LocaleType, 
        precision: int
    ) -> str:
        """Applique le formatage spécifique à la locale"""
        # Arrondit à la précision demandée
        rounded_value = value.quantize(
            Decimal('0.' + '0' * precision),
            rounding=ROUND_HALF_UP
        )
        
        # Convertit en float pour le formatage
        float_value = float(rounded_value)
        
        # Formate avec séparateurs par défaut
        formatted = f"{float_value:,.{precision}f}"
        
        # Applique les conventions locales
        if locale == LocaleType.FR_FR:
            # Français: espace comme séparateur de milliers, virgule décimale
            formatted = formatted.replace(',', 'TEMP_THOUSAND')
            formatted = formatted.replace('.', ',')
            formatted = formatted.replace('TEMP_THOUSAND', ' ')
        elif locale == LocaleType.DE_DE:
            # Allemand: point comme séparateur de milliers, virgule décimale
            formatted = formatted.replace(',', 'TEMP_THOUSAND')
            formatted = formatted.replace('.', ',')
            formatted = formatted.replace('TEMP_THOUSAND', '.')
        
        return formatted


class DateTimeFormatter:
    """Formateur de dates et heures"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._datetime_patterns = {
            LocaleType.EN_US: {
                'short_date': '%m/%d/%Y',
                'long_date': '%B %d, %Y',
                'short_time': '%I:%M %p',
                'long_time': '%I:%M:%S %p',
                'datetime': '%m/%d/%Y %I:%M %p',
                'iso': '%Y-%m-%d %H:%M:%S UTC'
            },
            LocaleType.FR_FR: {
                'short_date': '%d/%m/%Y',
                'long_date': '%d %B %Y',
                'short_time': '%H:%M',
                'long_time': '%H:%M:%S',
                'datetime': '%d/%m/%Y à %H:%M',
                'iso': '%Y-%m-%d %H:%M:%S UTC'
            },
            LocaleType.DE_DE: {
                'short_date': '%d.%m.%Y',
                'long_date': '%d. %B %Y',
                'short_time': '%H:%M',
                'long_time': '%H:%M:%S',
                'datetime': '%d.%m.%Y um %H:%M',
                'iso': '%Y-%m-%d %H:%M:%S UTC'
            }
        }
    
    def format_datetime(
        self, 
        dt: datetime, 
        locale: LocaleType,
        format_type: str = 'datetime',
        timezone_info: bool = True
    ) -> str:
        """Formate une date/heure selon la locale"""
        try:
            patterns = self._datetime_patterns.get(locale, self._datetime_patterns[LocaleType.EN_US])
            pattern = patterns.get(format_type, patterns['datetime'])
            
            # Convertit en UTC si nécessaire
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            formatted = dt.strftime(pattern)
            
            # Traduit les noms de mois si nécessaire
            if locale == LocaleType.FR_FR:
                formatted = self._translate_month_names_fr(formatted)
            elif locale == LocaleType.DE_DE:
                formatted = self._translate_month_names_de(formatted)
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"DateTime formatting error: {e}")
            return dt.isoformat()
    
    def format_relative_time(self, dt: datetime, locale: LocaleType) -> str:
        """Formate un temps relatif (ex: "il y a 5 minutes")"""
        try:
            now = datetime.now(timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            delta = now - dt
            
            if locale == LocaleType.FR_FR:
                return self._format_relative_time_fr(delta)
            elif locale == LocaleType.DE_DE:
                return self._format_relative_time_de(delta)
            else:
                return self._format_relative_time_en(delta)
                
        except Exception as e:
            self.logger.error(f"Relative time formatting error: {e}")
            return "unknown"
    
    def _translate_month_names_fr(self, formatted: str) -> str:
        """Traduit les noms de mois en français"""
        translations = {
            'January': 'janvier', 'February': 'février', 'March': 'mars',
            'April': 'avril', 'May': 'mai', 'June': 'juin',
            'July': 'juillet', 'August': 'août', 'September': 'septembre',
            'October': 'octobre', 'November': 'novembre', 'December': 'décembre'
        }
        
        for en, fr in translations.items():
            formatted = formatted.replace(en, fr)
        
        return formatted
    
    def _translate_month_names_de(self, formatted: str) -> str:
        """Traduit les noms de mois en allemand"""
        translations = {
            'January': 'Januar', 'February': 'Februar', 'March': 'März',
            'April': 'April', 'May': 'Mai', 'June': 'Juni',
            'July': 'Juli', 'August': 'August', 'September': 'September',
            'October': 'Oktober', 'November': 'November', 'December': 'Dezember'
        }
        
        for en, de in translations.items():
            formatted = formatted.replace(en, de)
        
        return formatted
    
    def _format_relative_time_fr(self, delta: timedelta) -> str:
        """Formate le temps relatif en français"""
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return "à l'instant"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            return f"il y a {hours} heure{'s' if hours > 1 else ''}"
        else:
            days = total_seconds // 86400
            return f"il y a {days} jour{'s' if days > 1 else ''}"
    
    def _format_relative_time_de(self, delta: timedelta) -> str:
        """Formate le temps relatif en allemand"""
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return "gerade eben"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"vor {minutes} Minute{'n' if minutes > 1 else ''}"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            return f"vor {hours} Stunde{'n' if hours > 1 else ''}"
        else:
            days = total_seconds // 86400
            return f"vor {days} Tag{'en' if days > 1 else ''}"
    
    def _format_relative_time_en(self, delta: timedelta) -> str:
        """Formate le temps relatif en anglais"""
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return "just now"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = total_seconds // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"


# Instances globales des formateurs
number_formatter = NumberFormatter()
datetime_formatter = DateTimeFormatter()

__all__ = [
    "UnitSystem",
    "DataType",
    "FormatRule",
    "NumberFormatter",
    "DateTimeFormatter",
    "number_formatter",
    "datetime_formatter"
]
