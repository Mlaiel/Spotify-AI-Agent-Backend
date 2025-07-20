"""
Spotify AI Agent - Tenancy Data Locales Module
=============================================

Module de localisation des données pour le système d'alertes et de monitoring
du tenant multi-plateforme Spotify AI Agent.

Ce module fournit:
- Localisation des messages d'alerte
- Templates de données spécifiques par locale
- Configuration des formats de données régionaux
- Mapping des codes d'erreur localisés

Author: Fahed Mlaiel
Architecture: Multi-tenant monitoring system
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import os

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"


class LocaleType(Enum):
    """Types de locales supportées"""
    EN_US = "en_US"
    FR_FR = "fr_FR"
    DE_DE = "de_DE"
    ES_ES = "es_ES"
    IT_IT = "it_IT"
    PT_BR = "pt_BR"
    JA_JP = "ja_JP"
    KO_KR = "ko_KR"
    ZH_CN = "zh_CN"
    RU_RU = "ru_RU"
    AR_SA = "ar_SA"


class DataFormat(Enum):
    """Formats de données par région"""
    DATE_FORMAT = "date_format"
    TIME_FORMAT = "time_format"
    NUMBER_FORMAT = "number_format"
    CURRENCY_FORMAT = "currency_format"
    PERCENT_FORMAT = "percent_format"


@dataclass
class LocaleDataConfig:
    """Configuration des données de locale"""
    locale: LocaleType
    date_format: str
    time_format: str
    number_format: str
    currency_symbol: str
    decimal_separator: str
    thousand_separator: str
    rtl_support: bool = False
    encoding: str = "utf-8"


class LocaleDataManager:
    """Gestionnaire des données de localisation"""
    
    def __init__(self):
        self._locales: Dict[LocaleType, LocaleDataConfig] = {}
        self._current_locale = LocaleType.EN_US
        self._initialize_default_locales()
    
    def _initialize_default_locales(self):
        """Initialise les locales par défaut"""
        # Configuration US/International
        self._locales[LocaleType.EN_US] = LocaleDataConfig(
            locale=LocaleType.EN_US,
            date_format="%Y-%m-%d",
            time_format="%H:%M:%S",
            number_format="1,234.56",
            currency_symbol="$",
            decimal_separator=".",
            thousand_separator=","
        )
        
        # Configuration française
        self._locales[LocaleType.FR_FR] = LocaleDataConfig(
            locale=LocaleType.FR_FR,
            date_format="%d/%m/%Y",
            time_format="%H:%M:%S",
            number_format="1 234,56",
            currency_symbol="€",
            decimal_separator=",",
            thousand_separator=" "
        )
        
        # Configuration allemande
        self._locales[LocaleType.DE_DE] = LocaleDataConfig(
            locale=LocaleType.DE_DE,
            date_format="%d.%m.%Y",
            time_format="%H:%M:%S",
            number_format="1.234,56",
            currency_symbol="€",
            decimal_separator=",",
            thousand_separator="."
        )
    
    def get_locale_config(self, locale: LocaleType) -> LocaleDataConfig:
        """Récupère la configuration pour une locale"""
        return self._locales.get(locale, self._locales[LocaleType.EN_US])
    
    def set_current_locale(self, locale: LocaleType):
        """Définit la locale courante"""
        self._current_locale = locale
    
    def format_number(self, number: float, locale: Optional[LocaleType] = None) -> str:
        """Formate un nombre selon la locale"""
        config = self.get_locale_config(locale or self._current_locale)
        # Implémentation du formatage numérique
        return f"{number:,.2f}".replace(",", "TEMP").replace(".", config.decimal_separator).replace("TEMP", config.thousand_separator)


# Instance globale du gestionnaire
locale_manager = LocaleDataManager()

__all__ = [
    "LocaleType",
    "DataFormat", 
    "LocaleDataConfig",
    "LocaleDataManager",
    "locale_manager"
]
