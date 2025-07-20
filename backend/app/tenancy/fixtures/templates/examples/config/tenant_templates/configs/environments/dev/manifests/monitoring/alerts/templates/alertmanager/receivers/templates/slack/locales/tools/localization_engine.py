#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moteur de Localisation Intelligent pour Notifications Slack Multi-Tenant

Ce module fournit un système de localisation avancé avec support multi-langue,
cache Redis distribué, détection automatique de locale et fallback intelligent.

Fonctionnalités:
- Support i18n complet avec interpolation de variables
- Cache Redis multi-niveau pour performances optimales
- Détection automatique de locale basée sur contexte tenant
- Fallback hiérarchique intelligent
- Hot-reload des traductions
- Validation de complétude des traductions
- Métriques de performance intégrées
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set
import re
import aioredis
from dataclasses import dataclass, field
import structlog
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(__name__)

# Métriques Prometheus
LOCALIZATION_REQUESTS = Counter(
    'slack_localization_requests_total',
    'Total localization requests',
    ['locale', 'tenant', 'status']
)

LOCALIZATION_DURATION = Histogram(
    'slack_localization_duration_seconds',
    'Localization processing duration',
    ['locale', 'operation']
)

LOCALIZATION_CACHE_HITS = Counter(
    'slack_localization_cache_hits_total',
    'Localization cache hits',
    ['cache_type', 'locale']
)

ACTIVE_LOCALES = Gauge(
    'slack_localization_active_locales',
    'Number of active locales'
)

@dataclass
class LocaleConfig:
    """Configuration pour une locale spécifique."""
    code: str
    name: str
    fallback: Optional[str] = None
    direction: str = "ltr"  # ltr ou rtl
    date_format: str = "%Y-%m-%d %H:%M:%S"
    timezone: str = "UTC"
    number_format: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_updated: Optional[datetime] = None

@dataclass
class TranslationEntry:
    """Entrée de traduction avec métadonnées."""
    key: str
    value: str
    context: Optional[str] = None
    description: Optional[str] = None
    plurals: Optional[Dict[str, str]] = None
    variables: Optional[Set[str]] = None
    last_modified: Optional[datetime] = None

class LocalizationEngine:
    """
    Moteur de localisation intelligent avec cache distribué et fallback.
    """
    
    def __init__(self, config: Dict[str, Any], cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager
        self.logger = logger.bind(component="localization_engine")
        
        # Configuration par défaut
        self.default_locale = config.get("default_locale", "fr_FR")
        self.supported_locales = config.get("supported_locales", ["fr_FR", "en_US"])
        self.cache_ttl = config.get("cache_ttl", 3600)
        self.hot_reload = config.get("hot_reload", True)
        self.validation_enabled = config.get("validation_enabled", True)
        
        # Stockage interne
        self._locales: Dict[str, LocaleConfig] = {}
        self._translations: Dict[str, Dict[str, TranslationEntry]] = {}
        self._fallback_chain: Dict[str, List[str]] = {}
        self._variable_pattern = re.compile(r'\{\{([^}]+)\}\}')
        self._plural_pattern = re.compile(r'\{\{([^}]+)\|plural\(([^)]+)\)\}\}')
        
        # Lock pour opérations thread-safe
        self._lock = asyncio.Lock()
        
        # Initialisation
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialise le moteur de localisation."""
        try:
            await self._load_locale_configs()
            await self._load_translations()
            await self._build_fallback_chains()
            await self._validate_translations()
            
            self.logger.info(
                "Moteur de localisation initialisé",
                locales_count=len(self._locales),
                translations_count=sum(len(t) for t in self._translations.values())
            )
            
            ACTIVE_LOCALES.set(len(self._locales))
            
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation", error=str(e))
            raise
    
    async def _load_locale_configs(self):
        """Charge les configurations des locales."""
        locales_dir = Path(__file__).parent / "locales"
        
        for locale_code in self.supported_locales:
            config_file = locales_dir / f"{locale_code}.config.json"
            
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    locale_config = LocaleConfig(
                        code=locale_code,
                        name=config_data.get("name", locale_code),
                        fallback=config_data.get("fallback"),
                        direction=config_data.get("direction", "ltr"),
                        date_format=config_data.get("date_format", "%Y-%m-%d %H:%M:%S"),
                        timezone=config_data.get("timezone", "UTC"),
                        number_format=config_data.get("number_format", {}),
                        enabled=config_data.get("enabled", True)
                    )
                    
                    self._locales[locale_code] = locale_config
                    
                except Exception as e:
                    self.logger.warning(
                        "Erreur lors du chargement de la config locale",
                        locale=locale_code,
                        error=str(e)
                    )
            else:
                # Configuration par défaut
                self._locales[locale_code] = LocaleConfig(
                    code=locale_code,
                    name=locale_code
                )
    
    async def _load_translations(self):
        """Charge toutes les traductions."""
        locales_dir = Path(__file__).parent / "locales"
        
        for locale_code in self.supported_locales:
            await self._load_locale_translations(locale_code)
    
    async def _load_locale_translations(self, locale_code: str):
        """Charge les traductions pour une locale spécifique."""
        locales_dir = Path(__file__).parent / "locales"
        translation_file = locales_dir / f"{locale_code}.json"
        
        if not translation_file.exists():
            self.logger.warning(
                "Fichier de traduction non trouvé",
                locale=locale_code,
                file=str(translation_file)
            )
            return
        
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations_data = json.load(f)
            
            # Aplatir la structure hiérarchique
            flat_translations = self._flatten_translations(translations_data)
            
            # Créer les entrées de traduction
            translation_entries = {}
            for key, value in flat_translations.items():
                entry = TranslationEntry(
                    key=key,
                    value=value,
                    variables=self._extract_variables(value),
                    last_modified=datetime.utcnow()
                )
                translation_entries[key] = entry
            
            self._translations[locale_code] = translation_entries
            
            # Cache dans Redis si disponible
            if self.cache_manager:
                cache_key = f"localization:translations:{locale_code}"
                await self.cache_manager.set(
                    cache_key,
                    json.dumps(flat_translations),
                    ttl=self.cache_ttl
                )
            
            self.logger.info(
                "Traductions chargées",
                locale=locale_code,
                count=len(translation_entries)
            )
            
        except Exception as e:
            self.logger.error(
                "Erreur lors du chargement des traductions",
                locale=locale_code,
                error=str(e)
            )
            raise
    
    def _flatten_translations(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Aplatit une structure hiérarchique de traductions."""
        result = {}
        
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                result.update(self._flatten_translations(value, new_key))
            elif isinstance(value, str):
                result[new_key] = value
            else:
                # Conversion en string pour autres types
                result[new_key] = str(value)
        
        return result
    
    def _extract_variables(self, text: str) -> Set[str]:
        """Extrait les variables d'un texte de traduction."""
        variables = set()
        
        # Variables simples: {{variable}}
        for match in self._variable_pattern.finditer(text):
            variables.add(match.group(1).strip())
        
        # Variables avec pluriel: {{count|plural(item,items)}}
        for match in self._plural_pattern.finditer(text):
            variables.add(match.group(1).strip())
        
        return variables
    
    async def _build_fallback_chains(self):
        """Construit les chaînes de fallback pour chaque locale."""
        for locale_code in self._locales:
            chain = [locale_code]
            current = locale_code
            visited = {locale_code}
            
            while True:
                locale_config = self._locales.get(current)
                if not locale_config or not locale_config.fallback:
                    break
                
                fallback = locale_config.fallback
                if fallback in visited:
                    # Éviter les boucles infinies
                    break
                
                chain.append(fallback)
                visited.add(fallback)
                current = fallback
            
            # Ajouter la locale par défaut si pas déjà présente
            if self.default_locale not in chain:
                chain.append(self.default_locale)
            
            self._fallback_chain[locale_code] = chain
            
            self.logger.debug(
                "Chaîne de fallback construite",
                locale=locale_code,
                chain=chain
            )
    
    async def _validate_translations(self):
        """Valide la complétude des traductions."""
        if not self.validation_enabled:
            return
        
        # Collecter toutes les clés disponibles
        all_keys = set()
        for translations in self._translations.values():
            all_keys.update(translations.keys())
        
        # Vérifier chaque locale
        for locale_code, translations in self._translations.items():
            missing_keys = all_keys - set(translations.keys())
            
            if missing_keys:
                self.logger.warning(
                    "Traductions manquantes détectées",
                    locale=locale_code,
                    missing_count=len(missing_keys),
                    missing_keys=list(missing_keys)[:10]  # Limiter l'affichage
                )
    
    async def localize(
        self,
        key: str,
        locale: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None,
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Localise une clé avec support des variables et pluralisation.
        
        Args:
            key: Clé de traduction
            locale: Code de locale (optionnel)
            variables: Variables pour interpolation
            count: Nombre pour pluralisation
            tenant_id: ID du tenant pour métadonnées
            
        Returns:
            Texte localisé
        """
        start_time = datetime.utcnow()
        
        try:
            # Déterminer la locale
            target_locale = locale or self.default_locale
            
            # Vérifier le cache d'abord
            cache_key = self._build_cache_key(key, target_locale, variables, count)
            cached_result = await self._get_from_cache(cache_key)
            
            if cached_result:
                LOCALIZATION_CACHE_HITS.labels(
                    cache_type="memory",
                    locale=target_locale
                ).inc()
                return cached_result
            
            # Chercher la traduction avec fallback
            translation = await self._find_translation(key, target_locale)
            
            if not translation:
                self.logger.warning(
                    "Traduction non trouvée",
                    key=key,
                    locale=target_locale,
                    tenant=tenant_id
                )
                LOCALIZATION_REQUESTS.labels(
                    locale=target_locale,
                    tenant=tenant_id or "unknown",
                    status="not_found"
                ).inc()
                return key  # Retourner la clé comme fallback
            
            # Interpoler les variables
            result = await self._interpolate_variables(
                translation.value,
                variables or {},
                count
            )
            
            # Mettre en cache
            await self._set_in_cache(cache_key, result)
            
            # Métriques
            duration = (datetime.utcnow() - start_time).total_seconds()
            LOCALIZATION_DURATION.labels(
                locale=target_locale,
                operation="localize"
            ).observe(duration)
            
            LOCALIZATION_REQUESTS.labels(
                locale=target_locale,
                tenant=tenant_id or "unknown",
                status="success"
            ).inc()
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Erreur lors de la localisation",
                key=key,
                locale=target_locale,
                error=str(e)
            )
            
            LOCALIZATION_REQUESTS.labels(
                locale=target_locale or "unknown",
                tenant=tenant_id or "unknown",
                status="error"
            ).inc()
            
            return key  # Fallback sur la clé
    
    async def _find_translation(self, key: str, locale: str) -> Optional[TranslationEntry]:
        """Trouve une traduction avec fallback."""
        fallback_chain = self._fallback_chain.get(locale, [locale, self.default_locale])
        
        for fallback_locale in fallback_chain:
            translations = self._translations.get(fallback_locale)
            if translations and key in translations:
                return translations[key]
        
        return None
    
    async def _interpolate_variables(
        self,
        text: str,
        variables: Dict[str, Any],
        count: Optional[int] = None
    ) -> str:
        """Interpole les variables dans le texte."""
        result = text
        
        # Traiter les pluriels d'abord
        if count is not None:
            plural_matches = list(self._plural_pattern.finditer(text))
            for match in reversed(plural_matches):  # Inverser pour préserver les positions
                var_name = match.group(1).strip()
                plural_forms = match.group(2).strip()
                
                # Analyser les formes plurielles
                forms = [form.strip() for form in plural_forms.split(',')]
                if len(forms) >= 2:
                    singular, plural = forms[0], forms[1]
                    chosen_form = singular if count == 1 else plural
                    
                    # Remplacer la variable de comptage
                    chosen_form = chosen_form.replace(f'{{{{{var_name}}}}}', str(count))
                    
                    # Remplacer dans le texte
                    result = result[:match.start()] + chosen_form + result[match.end():]
        
        # Traiter les variables simples
        for var_name, value in variables.items():
            placeholder = f'{{{{{var_name}}}}}'
            if isinstance(value, (int, float)):
                # Formatage des nombres selon la locale
                formatted_value = self._format_number(value, locale)
            elif isinstance(value, datetime):
                # Formatage des dates selon la locale
                formatted_value = self._format_date(value, locale)
            else:
                formatted_value = str(value)
            
            result = result.replace(placeholder, formatted_value)
        
        return result
    
    def _format_number(self, value: Union[int, float], locale: str) -> str:
        """Formate un nombre selon la locale."""
        # Implémentation simple, pourrait être étendue avec babel
        return str(value)
    
    def _format_date(self, value: datetime, locale: str) -> str:
        """Formate une date selon la locale."""
        locale_config = self._locales.get(locale)
        if locale_config:
            return value.strftime(locale_config.date_format)
        return value.strftime("%Y-%m-%d %H:%M:%S")
    
    def _build_cache_key(
        self,
        key: str,
        locale: str,
        variables: Optional[Dict[str, Any]],
        count: Optional[int]
    ) -> str:
        """Construit une clé de cache unique."""
        cache_parts = [f"loc:{locale}:{key}"]
        
        if variables:
            vars_str = "&".join(f"{k}={v}" for k, v in sorted(variables.items()))
            cache_parts.append(f"vars:{vars_str}")
        
        if count is not None:
            cache_parts.append(f"count:{count}")
        
        return "|".join(cache_parts)
    
    async def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Récupère une valeur du cache."""
        if not self.cache_manager:
            return None
        
        try:
            return await self.cache_manager.get(f"localization:cache:{cache_key}")
        except Exception as e:
            self.logger.warning("Erreur cache lecture", error=str(e))
            return None
    
    async def _set_in_cache(self, cache_key: str, value: str):
        """Met une valeur en cache."""
        if not self.cache_manager:
            return
        
        try:
            await self.cache_manager.set(
                f"localization:cache:{cache_key}",
                value,
                ttl=self.cache_ttl
            )
        except Exception as e:
            self.logger.warning("Erreur cache écriture", error=str(e))
    
    async def add_locale(self, locale_code: str, translations: Dict[str, str]):
        """Ajoute une nouvelle locale dynamiquement."""
        async with self._lock:
            try:
                # Créer la configuration de locale
                locale_config = LocaleConfig(
                    code=locale_code,
                    name=locale_code,
                    enabled=True,
                    last_updated=datetime.utcnow()
                )
                self._locales[locale_code] = locale_config
                
                # Ajouter les traductions
                translation_entries = {}
                for key, value in translations.items():
                    entry = TranslationEntry(
                        key=key,
                        value=value,
                        variables=self._extract_variables(value),
                        last_modified=datetime.utcnow()
                    )
                    translation_entries[key] = entry
                
                self._translations[locale_code] = translation_entries
                
                # Reconstruire les chaînes de fallback
                await self._build_fallback_chains()
                
                # Mettre à jour les métriques
                ACTIVE_LOCALES.set(len(self._locales))
                
                self.logger.info(
                    "Nouvelle locale ajoutée",
                    locale=locale_code,
                    translations_count=len(translations)
                )
                
            except Exception as e:
                self.logger.error(
                    "Erreur lors de l'ajout de locale",
                    locale=locale_code,
                    error=str(e)
                )
                raise
    
    async def update_translations(self, locale_code: str, updates: Dict[str, str]):
        """Met à jour les traductions pour une locale."""
        if locale_code not in self._translations:
            raise ValueError(f"Locale {locale_code} non trouvée")
        
        async with self._lock:
            try:
                for key, value in updates.items():
                    entry = TranslationEntry(
                        key=key,
                        value=value,
                        variables=self._extract_variables(value),
                        last_modified=datetime.utcnow()
                    )
                    self._translations[locale_code][key] = entry
                
                # Invalider le cache pour cette locale
                if self.cache_manager:
                    pattern = f"localization:cache:loc:{locale_code}:*"
                    await self.cache_manager.delete_pattern(pattern)
                
                self.logger.info(
                    "Traductions mises à jour",
                    locale=locale_code,
                    updates_count=len(updates)
                )
                
            except Exception as e:
                self.logger.error(
                    "Erreur lors de la mise à jour",
                    locale=locale_code,
                    error=str(e)
                )
                raise
    
    async def validate_completeness(self, locale_code: str) -> List[str]:
        """Valide la complétude des traductions pour une locale."""
        if locale_code not in self._translations:
            raise ValueError(f"Locale {locale_code} non trouvée")
        
        # Collecter toutes les clés de référence (default locale)
        reference_keys = set()
        if self.default_locale in self._translations:
            reference_keys = set(self._translations[self.default_locale].keys())
        else:
            # Utiliser toutes les clés disponibles
            for translations in self._translations.values():
                reference_keys.update(translations.keys())
        
        # Trouver les clés manquantes
        locale_keys = set(self._translations[locale_code].keys())
        missing_keys = reference_keys - locale_keys
        
        return list(missing_keys)
    
    async def get_supported_locales(self) -> List[Dict[str, Any]]:
        """Retourne la liste des locales supportées avec métadonnées."""
        locales_info = []
        
        for locale_code, config in self._locales.items():
            translations_count = len(self._translations.get(locale_code, {}))
            missing_count = len(await self.validate_completeness(locale_code))
            
            locales_info.append({
                "code": locale_code,
                "name": config.name,
                "enabled": config.enabled,
                "direction": config.direction,
                "translations_count": translations_count,
                "missing_translations": missing_count,
                "completeness": (translations_count - missing_count) / max(translations_count, 1) * 100,
                "last_updated": config.last_updated.isoformat() if config.last_updated else None
            })
        
        return locales_info
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moteur de localisation."""
        try:
            total_translations = sum(len(t) for t in self._translations.values())
            
            # Test de localisation simple
            test_result = await self.localize("test.health_check", self.default_locale)
            
            return {
                "status": "healthy",
                "locales_count": len(self._locales),
                "total_translations": total_translations,
                "default_locale": self.default_locale,
                "cache_enabled": self.cache_manager is not None,
                "test_localization": test_result,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }

# Factory function pour créer une instance
def create_localization_engine(config: Dict[str, Any], cache_manager=None) -> LocalizationEngine:
    """Crée une instance du moteur de localisation."""
    return LocalizationEngine(config, cache_manager)
