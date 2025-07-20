#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formatage culturel adaptatif pour alertes Slack multilingues

Ce module fournit un système de formatage culturel avancé avec:
- Formatage automatique des dates selon la culture
- Formatage des nombres et devises par région
- Support des langues RTL (Right-to-Left)
- Adaptation des couleurs et emojis par culture
- Formatage intelligent des adresses et téléphones
- Gestion des fuseaux horaires adaptatifs
- Templates visuels culturellement appropriés
- Système de fallback multi-culturel

Auteur: Expert Team
Version: 2.0.0
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import locale
import calendar
from decimal import Decimal, ROUND_HALF_UP

from babel import Locale, dates, numbers, core
from babel.numbers import format_decimal, format_currency, format_percent
from babel.dates import format_datetime, format_date, format_time, format_timedelta
import pytz

logger = logging.getLogger(__name__)


class CulturalContext(Enum):
    """Contextes culturels pour l'adaptation"""
    BUSINESS_FORMAL = "business_formal"
    TECHNICAL_ALERT = "technical_alert"
    CASUAL_NOTIFICATION = "casual_notification"
    EMERGENCY_ALERT = "emergency_alert"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class FormattingPriority(Enum):
    """Priorités de formatage culturel"""
    USER_PREFERENCE = 1
    GEOGRAPHIC = 2
    ORGANIZATIONAL = 3
    DEFAULT = 4


@dataclass
class CulturalConfig:
    """Configuration culturelle pour une région/langue"""
    language_code: str
    country_code: Optional[str] = None
    locale_string: str = ""
    rtl_support: bool = False
    date_format_preference: str = "medium"
    time_format_24h: bool = True
    first_day_of_week: int = 1  # 0=Sunday, 1=Monday
    currency_symbol: str = "$"
    currency_position: str = "before"  # before, after
    decimal_separator: str = "."
    thousands_separator: str = ","
    number_grouping: List[int] = field(default_factory=lambda: [3])
    phone_format_pattern: str = ""
    address_format_order: List[str] = field(default_factory=lambda: ["street", "city", "country"])
    color_preferences: Dict[str, str] = field(default_factory=dict)
    emoji_style: str = "default"  # default, minimal, text_only
    urgency_escalation: Dict[str, str] = field(default_factory=dict)


@dataclass 
class FormattingRequest:
    """Requête de formatage culturel"""
    data: Any
    format_type: str
    language: str
    country: Optional[str] = None
    context: CulturalContext = CulturalContext.TECHNICAL_ALERT
    timezone: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormattingResult:
    """Résultat de formatage culturel"""
    formatted_value: str
    original_value: Any
    format_type: str
    language: str
    rtl_direction: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedCulturalFormatter:
    """Formateur culturel intelligent ultra-avancé"""
    
    def __init__(self):
        """Initialise le formateur culturel"""
        self._cultural_configs: Dict[str, CulturalConfig] = {}
        self._timezone_cache: Dict[str, pytz.BaseTzInfo] = {}
        self._locale_cache: Dict[str, Locale] = {}
        
        # Initialisation des configurations culturelles
        self._initialize_cultural_configs()
        
        # Cache des patterns de formatage
        self._format_patterns = {
            "phone_international": {
                "US": r"^\+1\s?(\d{3})\s?(\d{3})\s?(\d{4})$",
                "FR": r"^\+33\s?(\d{1})\s?(\d{2})\s?(\d{2})\s?(\d{2})\s?(\d{2})$",
                "DE": r"^\+49\s?(\d{3,4})\s?(\d{7,8})$",
                "GB": r"^\+44\s?(\d{4})\s?(\d{6})$"
            },
            "postal_code": {
                "US": r"^\d{5}(-\d{4})?$",
                "FR": r"^\d{5}$", 
                "DE": r"^\d{5}$",
                "GB": r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$",
                "CA": r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$"
            }
        }
        
        # Mapping des couleurs par culture et contexte
        self._cultural_colors = {
            "emergency": {
                "default": "#FF0000",
                "JP": "#FF4500",  # Orange-rouge au Japon
                "CN": "#DC143C",  # Rouge foncé en Chine
                "IN": "#B22222"   # Rouge brique en Inde
            },
            "warning": {
                "default": "#FFA500",
                "JP": "#FF8C00",
                "CN": "#FFD700",
                "IN": "#FF6347"
            },
            "success": {
                "default": "#008000",
                "JP": "#32CD32",  # Vert lime au Japon
                "CN": "#228B22",  # Vert forêt en Chine
                "IN": "#00FF00"   # Vert vif en Inde
            }
        }
        
        # Emojis culturellement appropriés
        self._cultural_emojis = {
            "celebration": {
                "default": "🎉",
                "JP": "🎊",
                "CN": "🧨",
                "IN": "🪔",
                "AR": "🌟"
            },
            "warning": {
                "default": "⚠️",
                "JP": "🚨",
                "CN": "⚡",
                "IN": "🔔",
                "AR": "⭐"
            },
            "success": {
                "default": "✅",
                "JP": "🆗",
                "CN": "👍",
                "IN": "🙏",
                "AR": "☪️"
            }
        }
        
        logger.info("Formateur culturel initialisé")
    
    def _initialize_cultural_configs(self) -> None:
        """Initialise les configurations culturelles par défaut"""
        
        # Configuration pour l'anglais (US)
        self._cultural_configs["en_US"] = CulturalConfig(
            language_code="en",
            country_code="US",
            locale_string="en_US",
            rtl_support=False,
            date_format_preference="medium",
            time_format_24h=False,
            first_day_of_week=0,  # Dimanche
            currency_symbol="$",
            currency_position="before",
            decimal_separator=".",
            thousands_separator=",",
            number_grouping=[3],
            phone_format_pattern="+1 (XXX) XXX-XXXX",
            address_format_order=["street", "city", "state", "zip", "country"],
            color_preferences={
                "emergency": "#FF0000",
                "warning": "#FFA500", 
                "info": "#0066CC",
                "success": "#008000"
            },
            emoji_style="default"
        )
        
        # Configuration pour le français (France)
        self._cultural_configs["fr_FR"] = CulturalConfig(
            language_code="fr",
            country_code="FR",
            locale_string="fr_FR",
            rtl_support=False,
            date_format_preference="medium",
            time_format_24h=True,
            first_day_of_week=1,  # Lundi
            currency_symbol="€",
            currency_position="after",
            decimal_separator=",",
            thousands_separator=" ",
            number_grouping=[3],
            phone_format_pattern="+33 X XX XX XX XX",
            address_format_order=["street", "zip", "city", "country"],
            color_preferences={
                "emergency": "#DC143C",
                "warning": "#FF8C00",
                "info": "#4169E1",
                "success": "#228B22"
            },
            emoji_style="default"
        )
        
        # Configuration pour l'allemand (Allemagne)
        self._cultural_configs["de_DE"] = CulturalConfig(
            language_code="de",
            country_code="DE",
            locale_string="de_DE",
            rtl_support=False,
            date_format_preference="medium",
            time_format_24h=True,
            first_day_of_week=1,  # Lundi
            currency_symbol="€",
            currency_position="after",
            decimal_separator=",",
            thousands_separator=".",
            number_grouping=[3],
            phone_format_pattern="+49 XXXX XXXXXXXX",
            address_format_order=["street", "zip", "city", "country"],
            color_preferences={
                "emergency": "#B22222",
                "warning": "#DAA520",
                "info": "#4682B4",
                "success": "#2E8B57"
            },
            emoji_style="minimal"
        )
        
        # Configuration pour l'espagnol (Espagne)
        self._cultural_configs["es_ES"] = CulturalConfig(
            language_code="es",
            country_code="ES",
            locale_string="es_ES",
            rtl_support=False,
            date_format_preference="medium",
            time_format_24h=True,
            first_day_of_week=1,  # Lundi
            currency_symbol="€",
            currency_position="after",
            decimal_separator=",",
            thousands_separator=".",
            number_grouping=[3],
            phone_format_pattern="+34 XXX XXX XXX",
            address_format_order=["street", "zip", "city", "province", "country"],
            emoji_style="default"
        )
        
        # Configuration pour le japonais (Japon)
        self._cultural_configs["ja_JP"] = CulturalConfig(
            language_code="ja",
            country_code="JP",
            locale_string="ja_JP",
            rtl_support=False,
            date_format_preference="full",
            time_format_24h=True,
            first_day_of_week=0,  # Dimanche
            currency_symbol="¥",
            currency_position="before",
            decimal_separator=".",
            thousands_separator=",",
            number_grouping=[4],  # Groupement par 10,000 (man)
            phone_format_pattern="+81 XX-XXXX-XXXX",
            address_format_order=["country", "prefecture", "city", "district", "street"],
            color_preferences={
                "emergency": "#FF4500",
                "warning": "#FF8C00",
                "info": "#4169E1",
                "success": "#32CD32"
            },
            emoji_style="kawaii"
        )
        
        # Configuration pour l'arabe (Arabie Saoudite)
        self._cultural_configs["ar_SA"] = CulturalConfig(
            language_code="ar",
            country_code="SA",
            locale_string="ar_SA",
            rtl_support=True,
            date_format_preference="full",
            time_format_24h=True,
            first_day_of_week=6,  # Samedi
            currency_symbol="ر.س",
            currency_position="after",
            decimal_separator=".",
            thousands_separator=",",
            number_grouping=[3],
            phone_format_pattern="+966 XX XXX XXXX",
            address_format_order=["country", "region", "city", "district", "street"],
            color_preferences={
                "emergency": "#8B0000",
                "warning": "#B8860B",
                "info": "#483D8B",
                "success": "#006400"
            },
            emoji_style="minimal"
        )
        
        # Configuration pour le chinois (Chine)
        self._cultural_configs["zh_CN"] = CulturalConfig(
            language_code="zh",
            country_code="CN",
            locale_string="zh_CN",
            rtl_support=False,
            date_format_preference="full",
            time_format_24h=True,
            first_day_of_week=1,  # Lundi
            currency_symbol="¥",
            currency_position="before",
            decimal_separator=".",
            thousands_separator=",",
            number_grouping=[4],  # Groupement par 10,000
            phone_format_pattern="+86 XXX XXXX XXXX",
            address_format_order=["country", "province", "city", "district", "street"],
            color_preferences={
                "emergency": "#DC143C",
                "warning": "#FFD700",
                "info": "#4682B4",
                "success": "#228B22"
            },
            emoji_style="default"
        )
    
    def get_cultural_config(self, language: str, country: Optional[str] = None) -> CulturalConfig:
        """Récupère la configuration culturelle pour une langue/pays"""
        
        # Tentative avec langue + pays
        if country:
            config_key = f"{language}_{country}"
            if config_key in self._cultural_configs:
                return self._cultural_configs[config_key]
        
        # Tentative avec langue uniquement
        for key, config in self._cultural_configs.items():
            if config.language_code == language:
                return config
        
        # Fallback vers anglais US
        return self._cultural_configs.get("en_US", CulturalConfig(language_code="en"))
    
    def format_datetime(self, 
                       dt: datetime,
                       language: str,
                       country: Optional[str] = None,
                       format_type: str = "medium",
                       timezone_name: Optional[str] = None,
                       context: CulturalContext = CulturalContext.TECHNICAL_ALERT) -> FormattingResult:
        """
        Formate une date/heure selon la culture
        
        Args:
            dt: DateTime à formater
            language: Code langue (ex: "fr")
            country: Code pays (ex: "FR")
            format_type: Type de formatage ("short", "medium", "long", "full", "relative")
            timezone_name: Nom du fuseau horaire (ex: "Europe/Paris")
            context: Contexte culturel
            
        Returns:
            Résultat de formatage culturel
        """
        config = self.get_cultural_config(language, country)
        
        try:
            # Gestion du fuseau horaire
            if timezone_name:
                target_tz = self._get_timezone(timezone_name)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt = dt.astimezone(target_tz)
            
            # Obtention de la locale
            babel_locale = self._get_babel_locale(config.locale_string)
            
            # Formatage selon le type
            if format_type == "relative":
                # Calcul du temps relatif
                now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
                delta = now - dt
                formatted = format_timedelta(delta, locale=babel_locale, add_direction=True)
                
            elif format_type == "time_only":
                formatted = format_time(dt, locale=babel_locale)
                
            elif format_type == "date_only":
                formatted = format_date(dt, locale=babel_locale)
                
            else:
                # Formatage datetime complet
                format_length = format_type if format_type in ["short", "medium", "long", "full"] else "medium"
                formatted = format_datetime(dt, format=format_length, locale=babel_locale)
            
            # Adaptations contextuelles
            if context == CulturalContext.EMERGENCY_ALERT:
                # Ajout d'indicateurs d'urgence
                if config.language_code == "ja":
                    formatted = f"⚡ {formatted} ⚡"
                elif config.language_code == "ar":
                    formatted = f"⭐ {formatted} ⭐"
                else:
                    formatted = f"🚨 {formatted}"
            
            return FormattingResult(
                formatted_value=formatted,
                original_value=dt,
                format_type=format_type,
                language=language,
                rtl_direction=config.rtl_support,
                metadata={
                    "timezone": timezone_name,
                    "context": context.value,
                    "locale": config.locale_string
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur formatage datetime: {e}")
            # Fallback simple
            return FormattingResult(
                formatted_value=dt.isoformat(),
                original_value=dt,
                format_type=format_type,
                language=language,
                rtl_direction=config.rtl_support
            )
    
    def format_number(self,
                     number: Union[int, float, Decimal],
                     language: str,
                     country: Optional[str] = None,
                     format_type: str = "decimal",
                     currency_code: Optional[str] = None,
                     precision: Optional[int] = None) -> FormattingResult:
        """
        Formate un nombre selon la culture
        
        Args:
            number: Nombre à formater
            language: Code langue
            country: Code pays
            format_type: Type ("decimal", "currency", "percent", "scientific")
            currency_code: Code devise (ex: "EUR", "USD")
            precision: Nombre de décimales
            
        Returns:
            Résultat de formatage culturel
        """
        config = self.get_cultural_config(language, country)
        
        try:
            babel_locale = self._get_babel_locale(config.locale_string)
            
            if format_type == "currency":
                # Formatage monétaire
                currency = currency_code or self._get_default_currency(country or config.country_code)
                formatted = format_currency(
                    number, 
                    currency, 
                    locale=babel_locale,
                    format=None
                )
                
            elif format_type == "percent":
                # Formatage en pourcentage
                formatted = format_percent(number, locale=babel_locale)
                
            elif format_type == "scientific":
                # Notation scientifique
                formatted = f"{number:.2e}"
                
            else:
                # Formatage décimal standard
                if precision is not None:
                    formatted = format_decimal(
                        round(Decimal(str(number)), precision),
                        locale=babel_locale
                    )
                else:
                    formatted = format_decimal(number, locale=babel_locale)
            
            # Adaptations culturelles spécifiques
            if config.language_code == "ja" and format_type == "decimal":
                # Groupement par 10,000 pour le japonais
                formatted = self._apply_japanese_grouping(formatted, number)
            
            elif config.language_code == "ar":
                # Adaptation pour l'arabe (chiffres arabes si configuré)
                formatted = self._apply_arabic_numerals(formatted, config)
            
            return FormattingResult(
                formatted_value=formatted,
                original_value=number,
                format_type=format_type,
                language=language,
                rtl_direction=config.rtl_support,
                metadata={
                    "currency": currency_code,
                    "precision": precision,
                    "locale": config.locale_string
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur formatage nombre: {e}")
            # Fallback simple
            return FormattingResult(
                formatted_value=str(number),
                original_value=number,
                format_type=format_type,
                language=language,
                rtl_direction=config.rtl_support
            )
    
    def format_address(self,
                      address_components: Dict[str, str],
                      language: str,
                      country: Optional[str] = None,
                      context: CulturalContext = CulturalContext.TECHNICAL_ALERT) -> FormattingResult:
        """
        Formate une adresse selon les conventions culturelles
        
        Args:
            address_components: Composants de l'adresse
            language: Code langue
            country: Code pays
            context: Contexte culturel
            
        Returns:
            Résultat de formatage culturel
        """
        config = self.get_cultural_config(language, country)
        
        try:
            # Ordre des composants selon la culture
            ordered_components = []
            
            for component_type in config.address_format_order:
                if component_type in address_components:
                    value = address_components[component_type]
                    
                    # Adaptations spécifiques
                    if component_type == "zip" and country:
                        value = self._format_postal_code(value, country)
                    
                    ordered_components.append(value)
            
            # Assemblage selon la culture
            if config.rtl_support:
                # Pour les langues RTL, inversion de l'ordre
                ordered_components.reverse()
                separator = " ،"  # Virgule arabe
            else:
                separator = ", "
            
            formatted = separator.join(ordered_components)
            
            # Adaptations contextuelles
            if context == CulturalContext.EMERGENCY_ALERT:
                if config.language_code == "ja":
                    formatted = f"📍 {formatted}"
                elif config.language_code == "ar": 
                    formatted = f"📍 {formatted}"
                else:
                    formatted = f"📍 {formatted}"
            
            return FormattingResult(
                formatted_value=formatted,
                original_value=address_components,
                format_type="address",
                language=language,
                rtl_direction=config.rtl_support,
                metadata={
                    "order": config.address_format_order,
                    "context": context.value
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur formatage adresse: {e}")
            # Fallback simple
            fallback = ", ".join(address_components.values())
            return FormattingResult(
                formatted_value=fallback,
                original_value=address_components,
                format_type="address",
                language=language,
                rtl_direction=config.rtl_support
            )
    
    def format_phone(self,
                    phone_number: str,
                    language: str,
                    country: Optional[str] = None,
                    format_style: str = "international") -> FormattingResult:
        """
        Formate un numéro de téléphone selon les conventions culturelles
        
        Args:
            phone_number: Numéro de téléphone
            language: Code langue
            country: Code pays
            format_style: Style ("international", "national", "local")
            
        Returns:
            Résultat de formatage culturel
        """
        config = self.get_cultural_config(language, country)
        
        try:
            # Nettoyage du numéro
            cleaned_number = re.sub(r'[^\d+]', '', phone_number)
            
            # Application du pattern culturel
            if country and country in self._format_patterns["phone_international"]:
                pattern = self._format_patterns["phone_international"][country]
                match = re.match(pattern, f"+{cleaned_number.lstrip('+')}")
                
                if match:
                    groups = match.groups()
                    
                    # Formatage selon le style demandé
                    if format_style == "international":
                        if country == "US":
                            formatted = f"+1 ({groups[0]}) {groups[1]}-{groups[2]}"
                        elif country == "FR":
                            formatted = f"+33 {groups[0]} {groups[1]} {groups[2]} {groups[3]} {groups[4]}"
                        elif country == "DE":
                            formatted = f"+49 {groups[0]} {groups[1]}"
                        else:
                            formatted = phone_number
                    else:
                        formatted = phone_number
                else:
                    formatted = phone_number
            else:
                formatted = phone_number
            
            return FormattingResult(
                formatted_value=formatted,
                original_value=phone_number,
                format_type="phone",
                language=language,
                rtl_direction=config.rtl_support,
                metadata={
                    "style": format_style,
                    "country": country
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur formatage téléphone: {e}")
            return FormattingResult(
                formatted_value=phone_number,
                original_value=phone_number,
                format_type="phone",
                language=language,
                rtl_direction=config.rtl_support
            )
    
    def get_cultural_color(self,
                          severity: str,
                          language: str,
                          country: Optional[str] = None,
                          context: CulturalContext = CulturalContext.TECHNICAL_ALERT) -> str:
        """
        Retourne une couleur culturellement appropriée
        
        Args:
            severity: Niveau de sévérité
            language: Code langue
            country: Code pays
            context: Contexte culturel
            
        Returns:
            Code couleur hexadécimal
        """
        country_key = country or "default"
        
        if severity in self._cultural_colors:
            return self._cultural_colors[severity].get(
                country_key,
                self._cultural_colors[severity]["default"]
            )
        
        return "#808080"  # Gris par défaut
    
    def get_cultural_emoji(self,
                          category: str,
                          language: str,
                          country: Optional[str] = None,
                          style: Optional[str] = None) -> str:
        """
        Retourne un emoji culturellement approprié
        
        Args:
            category: Catégorie d'emoji
            language: Code langue
            country: Code pays
            style: Style spécifique
            
        Returns:
            Emoji approprié
        """
        config = self.get_cultural_config(language, country)
        
        # Style par défaut selon la configuration
        if style is None:
            style = config.emoji_style
        
        # Pas d'emoji pour le style text_only
        if style == "text_only":
            return ""
        
        country_key = country or "default"
        
        if category in self._cultural_emojis:
            return self._cultural_emojis[category].get(
                country_key,
                self._cultural_emojis[category]["default"]
            )
        
        return ""
    
    def _get_timezone(self, timezone_name: str) -> pytz.BaseTzInfo:
        """Récupère un fuseau horaire avec cache"""
        if timezone_name not in self._timezone_cache:
            self._timezone_cache[timezone_name] = pytz.timezone(timezone_name)
        return self._timezone_cache[timezone_name]
    
    def _get_babel_locale(self, locale_string: str) -> Locale:
        """Récupère une locale Babel avec cache"""
        if locale_string not in self._locale_cache:
            try:
                self._locale_cache[locale_string] = Locale.parse(locale_string)
            except Exception:
                # Fallback vers locale par défaut
                self._locale_cache[locale_string] = Locale.parse("en_US")
        return self._locale_cache[locale_string]
    
    def _get_default_currency(self, country_code: Optional[str]) -> str:
        """Détermine la devise par défaut pour un pays"""
        currency_map = {
            "US": "USD", "CA": "CAD",
            "FR": "EUR", "DE": "EUR", "ES": "EUR", "IT": "EUR",
            "GB": "GBP",
            "JP": "JPY",
            "CN": "CNY",
            "SA": "SAR", "AE": "AED",
            "BR": "BRL", "MX": "MXN"
        }
        return currency_map.get(country_code, "USD")
    
    def _format_postal_code(self, postal_code: str, country: str) -> str:
        """Formate un code postal selon le pays"""
        if country in self._format_patterns["postal_code"]:
            pattern = self._format_patterns["postal_code"][country]
            if re.match(pattern, postal_code):
                # Code postal valide, formatage spécifique par pays
                if country == "GB":
                    # Format britannique avec espace
                    if len(postal_code) > 3 and ' ' not in postal_code:
                        return f"{postal_code[:-3]} {postal_code[-3:]}"
                elif country == "CA":
                    # Format canadien avec espace
                    if len(postal_code) == 6 and ' ' not in postal_code:
                        return f"{postal_code[:3]} {postal_code[3:]}"
        
        return postal_code
    
    def _apply_japanese_grouping(self, formatted: str, number: Union[int, float]) -> str:
        """Applique le groupement japonais par 10,000 (man)"""
        try:
            if abs(float(number)) >= 10000:
                # Conversion en unités japonaises (man = 10,000)
                man_value = int(number) // 10000
                remainder = int(number) % 10000
                
                if remainder == 0:
                    return f"{man_value}万"
                else:
                    return f"{man_value}万{remainder}"
        except (ValueError, TypeError):
            pass
        
        return formatted
    
    def _apply_arabic_numerals(self, formatted: str, config: CulturalConfig) -> str:
        """Applique les chiffres arabes si configuré"""
        # Mapping des chiffres occidentaux vers arabes
        arabic_digits = {
            '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
            '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
        }
        
        # Conversion uniquement si explicitement demandé
        if hasattr(config, 'use_arabic_numerals') and config.use_arabic_numerals:
            for western, arabic in arabic_digits.items():
                formatted = formatted.replace(western, arabic)
        
        return formatted
    
    def format_alert_template(self,
                             template_data: Dict[str, Any],
                             language: str,
                             country: Optional[str] = None,
                             context: CulturalContext = CulturalContext.TECHNICAL_ALERT) -> Dict[str, FormattingResult]:
        """
        Formate un template d'alerte complet selon la culture
        
        Args:
            template_data: Données du template
            language: Code langue
            country: Code pays
            context: Contexte culturel
            
        Returns:
            Dict avec tous les éléments formatés
        """
        results = {}
        
        for key, value in template_data.items():
            if isinstance(value, datetime):
                results[key] = self.format_datetime(value, language, country, context=context)
            elif isinstance(value, (int, float, Decimal)):
                results[key] = self.format_number(value, language, country)
            elif isinstance(value, dict) and 'street' in value:
                # Détection d'une adresse
                results[key] = self.format_address(value, language, country, context)
            elif isinstance(value, str) and re.match(r'^\+?\d', value):
                # Détection d'un numéro de téléphone
                results[key] = self.format_phone(value, language, country)
            else:
                # Formatage textuel simple
                results[key] = FormattingResult(
                    formatted_value=str(value),
                    original_value=value,
                    format_type="text",
                    language=language,
                    rtl_direction=self.get_cultural_config(language, country).rtl_support
                )
        
        return results


# Factory function
def create_cultural_formatter() -> AdvancedCulturalFormatter:
    """
    Factory pour créer un formateur culturel
    
    Returns:
        Formateur culturel initialisé
    """
    return AdvancedCulturalFormatter()


# Export des classes principales
__all__ = [
    "AdvancedCulturalFormatter",
    "CulturalConfig",
    "FormattingRequest",
    "FormattingResult",
    "CulturalContext",
    "FormattingPriority",
    "create_cultural_formatter"
]
