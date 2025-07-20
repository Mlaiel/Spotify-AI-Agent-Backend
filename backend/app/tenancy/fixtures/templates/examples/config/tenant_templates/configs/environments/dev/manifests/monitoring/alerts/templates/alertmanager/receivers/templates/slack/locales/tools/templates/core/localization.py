"""
Advanced Localization Manager
=============================

Gestionnaire de localisation avancé avec support multi-langues,
pluralisation, contextes, et cache intelligent.

Auteur: Fahed Mlaiel
"""

import asyncio
import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import aiofiles
import re
from babel import Locale, dates, numbers
from babel.messages import Catalog
from babel.messages.pofile import read_po, write_po
import weakref

logger = logging.getLogger(__name__)


@dataclass
class TranslationEntry:
    """Entrée de traduction"""
    key: str
    value: str
    context: Optional[str] = None
    description: Optional[str] = None
    plurals: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguagePack:
    """Pack de langue"""
    locale: Locale
    code: str
    name: str
    translations: Dict[str, TranslationEntry] = field(default_factory=dict)
    pluralization_rule: Optional[Callable] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    completion_percentage: float = 0.0


class LocalizationManager:
    """
    Gestionnaire de localisation avancé
    
    Fonctionnalités:
    - Support multi-langues avec fallback
    - Pluralisation intelligente
    - Contextes de traduction
    - Interpolation de variables
    - Cache intelligent
    - Hot-reload des traductions
    - Formats Babel, JSON, YAML
    - Métriques et monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire de localisation
        
        Args:
            config: Configuration de localisation
        """
        self.config = config
        self.is_initialized = False
        
        # Langues et traductions
        self.language_packs: Dict[str, LanguagePack] = {}
        self.default_language = config.get("default_language", "en")
        self.fallback_language = config.get("fallback_language", "en")
        self.supported_languages = config.get("supported_languages", ["en", "fr", "de", "es"])
        
        # Répertoires de traductions
        self.locale_dirs: List[Path] = []
        self.tenant_locale_dirs: Dict[str, Path] = {}
        
        # Cache des traductions
        self.translation_cache: Dict[str, Dict[str, str]] = {}
        self.cache_enabled = config.get("cache_translations", True)
        self.cache_ttl = config.get("cache_ttl", 3600)
        
        # Contextes
        self.contexts: Dict[str, Dict[str, Any]] = {}
        
        # Règles de pluralisation
        self.pluralization_rules: Dict[str, Callable] = {}
        
        # Métriques
        self.metrics = {
            "translations_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "translation_requests": 0,
            "missing_translations": 0,
            "fallback_used": 0
        }
        
        # Callbacks pour les traductions manquantes
        self.missing_translation_callbacks: List[Callable] = []
        
        logger.info("LocalizationManager initialisé")
    
    async def initialize(self) -> None:
        """Initialise le gestionnaire de localisation"""
        if self.is_initialized:
            return
        
        logger.info("Initialisation du LocalizationManager...")
        
        try:
            # Configuration des répertoires
            await self._setup_locale_directories()
            
            # Chargement des règles de pluralisation
            await self._setup_pluralization_rules()
            
            # Chargement des traductions
            await self._load_all_translations()
            
            # Configuration du cache
            if self.cache_enabled:
                await self._setup_translation_cache()
            
            # Validation des langues supportées
            await self._validate_supported_languages()
            
            self.is_initialized = True
            logger.info("LocalizationManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du LocalizationManager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrêt propre du gestionnaire de localisation"""
        if not self.is_initialized:
            return
        
        logger.info("Arrêt du LocalizationManager...")
        
        try:
            # Nettoyage du cache
            self.translation_cache.clear()
            self.language_packs.clear()
            
            self.is_initialized = False
            logger.info("LocalizationManager arrêté avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")
    
    async def _setup_locale_directories(self) -> None:
        """Configure les répertoires de locales"""
        # Répertoire principal
        main_locale_dir = Path(self.config.get("locale_dir", "locales"))
        if main_locale_dir.exists():
            self.locale_dirs.append(main_locale_dir)
        
        # Répertoires additionnels
        additional_dirs = self.config.get("additional_locale_dirs", [])
        for dir_path in additional_dirs:
            path = Path(dir_path)
            if path.exists():
                self.locale_dirs.append(path)
        
        # Répertoires tenant-spécifiques
        tenant_locale_base = Path(self.config.get("tenant_locale_dir", "locales/tenants"))
        if tenant_locale_base.exists():
            for tenant_dir in tenant_locale_base.iterdir():
                if tenant_dir.is_dir():
                    self.tenant_locale_dirs[tenant_dir.name] = tenant_dir
        
        logger.info(f"Configurés {len(self.locale_dirs)} répertoires de locales")
        logger.info(f"Configurés {len(self.tenant_locale_dirs)} répertoires tenant-spécifiques")
    
    async def _setup_pluralization_rules(self) -> None:
        """Configure les règles de pluralisation"""
        # Règles par défaut pour les langues courantes
        
        def english_plural_rule(n: int) -> str:
            """Règle de pluralisation anglaise"""
            return "one" if n == 1 else "other"
        
        def french_plural_rule(n: int) -> str:
            """Règle de pluralisation française"""
            return "one" if n <= 1 else "other"
        
        def german_plural_rule(n: int) -> str:
            """Règle de pluralisation allemande"""
            return "one" if n == 1 else "other"
        
        def spanish_plural_rule(n: int) -> str:
            """Règle de pluralisation espagnole"""
            return "one" if n == 1 else "other"
        
        def russian_plural_rule(n: int) -> str:
            """Règle de pluralisation russe"""
            if n % 10 == 1 and n % 100 != 11:
                return "one"
            elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
                return "few"
            else:
                return "many"
        
        def polish_plural_rule(n: int) -> str:
            """Règle de pluralisation polonaise"""
            if n == 1:
                return "one"
            elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
                return "few"
            else:
                return "many"
        
        # Attribution des règles par langue
        self.pluralization_rules.update({
            "en": english_plural_rule,
            "fr": french_plural_rule,
            "de": german_plural_rule,
            "es": spanish_plural_rule,
            "ru": russian_plural_rule,
            "pl": polish_plural_rule
        })
        
        # Chargement des règles personnalisées depuis la configuration
        custom_rules = self.config.get("custom_pluralization_rules", {})
        for lang_code, rule_func in custom_rules.items():
            if callable(rule_func):
                self.pluralization_rules[lang_code] = rule_func
        
        logger.info(f"Configurées {len(self.pluralization_rules)} règles de pluralisation")
    
    async def _load_all_translations(self) -> None:
        """Charge toutes les traductions"""
        for language_code in self.supported_languages:
            await self._load_language_pack(language_code)
        
        # Chargement des traductions tenant-spécifiques
        for tenant_id, tenant_dir in self.tenant_locale_dirs.items():
            for language_code in self.supported_languages:
                await self._load_tenant_translations(tenant_id, language_code, tenant_dir)
        
        logger.info(f"Chargés {len(self.language_packs)} packs de langues")
    
    async def _load_language_pack(self, language_code: str) -> None:
        """Charge un pack de langue"""
        try:
            # Création du locale Babel
            locale = Locale(language_code)
            
            # Création du pack de langue
            language_pack = LanguagePack(
                locale=locale,
                code=language_code,
                name=locale.display_name,
                pluralization_rule=self.pluralization_rules.get(language_code)
            )
            
            # Chargement des traductions depuis différents formats
            translations_loaded = 0
            
            for locale_dir in self.locale_dirs:
                lang_dir = locale_dir / language_code
                
                if lang_dir.exists():
                    # Fichiers .po (Babel)
                    po_file = lang_dir / "messages.po"
                    if po_file.exists():
                        translations_loaded += await self._load_po_file(po_file, language_pack)
                    
                    # Fichiers .json
                    json_file = lang_dir / "messages.json"
                    if json_file.exists():
                        translations_loaded += await self._load_json_file(json_file, language_pack)
                    
                    # Fichiers .yaml
                    yaml_file = lang_dir / "messages.yaml"
                    if yaml_file.exists():
                        translations_loaded += await self._load_yaml_file(yaml_file, language_pack)
                    
                    # Fichiers individuels
                    for file_path in lang_dir.glob("*.json"):
                        if file_path.name != "messages.json":
                            translations_loaded += await self._load_json_file(file_path, language_pack)
            
            # Calcul du pourcentage de complétude
            if self.default_language in self.language_packs:
                default_pack = self.language_packs[self.default_language]
                if len(default_pack.translations) > 0:
                    language_pack.completion_percentage = (
                        len(language_pack.translations) / len(default_pack.translations) * 100
                    )
            
            self.language_packs[language_code] = language_pack
            self.metrics["translations_loaded"] += translations_loaded
            
            logger.info(f"Pack de langue chargé: {language_code} ({translations_loaded} traductions)")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du pack de langue {language_code}: {e}")
    
    async def _load_po_file(self, po_file: Path, language_pack: LanguagePack) -> int:
        """Charge un fichier .po"""
        try:
            with open(po_file, 'rb') as f:
                catalog = read_po(f)
            
            translations_count = 0
            
            for message in catalog:
                if message.id and message.string:
                    # Clé principale
                    key = message.id if isinstance(message.id, str) else message.id[0]
                    
                    # Création de l'entrée
                    entry = TranslationEntry(
                        key=key,
                        value=message.string if isinstance(message.string, str) else message.string[0],
                        context=message.context,
                        description=message.comment
                    )
                    
                    # Gestion des pluriels
                    if isinstance(message.id, tuple) and isinstance(message.string, tuple):
                        entry.plurals = dict(zip(
                            ["one", "other"] if len(message.string) == 2 else [f"form_{i}" for i in range(len(message.string))],
                            message.string
                        ))
                    
                    language_pack.translations[key] = entry
                    translations_count += 1
            
            return translations_count
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier PO {po_file}: {e}")
            return 0
    
    async def _load_json_file(self, json_file: Path, language_pack: LanguagePack) -> int:
        """Charge un fichier .json"""
        try:
            async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
            return await self._process_translation_data(data, language_pack)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier JSON {json_file}: {e}")
            return 0
    
    async def _load_yaml_file(self, yaml_file: Path, language_pack: LanguagePack) -> int:
        """Charge un fichier .yaml"""
        try:
            async with aiofiles.open(yaml_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = yaml.safe_load(content)
            
            return await self._process_translation_data(data, language_pack)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier YAML {yaml_file}: {e}")
            return 0
    
    async def _process_translation_data(self, data: Dict[str, Any], language_pack: LanguagePack, prefix: str = "") -> int:
        """Traite les données de traduction (format hiérarchique)"""
        translations_count = 0
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Si c'est un dictionnaire, on vérifie s'il contient des pluriels ou des contextes
                if "_value" in value or "_one" in value or "_other" in value:
                    # Format spécial avec métadonnées
                    entry = TranslationEntry(
                        key=full_key,
                        value=value.get("_value", ""),
                        context=value.get("_context"),
                        description=value.get("_description")
                    )
                    
                    # Extraction des pluriels
                    plurals = {}
                    for plural_key, plural_value in value.items():
                        if plural_key.startswith("_") and plural_key not in ["_value", "_context", "_description"]:
                            plural_form = plural_key[1:]  # Suppression du préfixe _
                            plurals[plural_form] = plural_value
                    
                    if plurals:
                        entry.plurals = plurals
                    
                    language_pack.translations[full_key] = entry
                    translations_count += 1
                else:
                    # Dictionnaire hiérarchique standard
                    translations_count += await self._process_translation_data(value, language_pack, full_key)
            elif isinstance(value, str):
                # Traduction simple
                entry = TranslationEntry(key=full_key, value=value)
                language_pack.translations[full_key] = entry
                translations_count += 1
        
        return translations_count
    
    async def _load_tenant_translations(self, tenant_id: str, language_code: str, tenant_dir: Path) -> None:
        """Charge les traductions spécifiques à un tenant"""
        try:
            lang_dir = tenant_dir / language_code
            
            if not lang_dir.exists():
                return
            
            # Clé du pack tenant-spécifique
            tenant_pack_key = f"{tenant_id}_{language_code}"
            
            # Copie du pack de langue de base
            if language_code in self.language_packs:
                base_pack = self.language_packs[language_code]
                tenant_pack = LanguagePack(
                    locale=base_pack.locale,
                    code=f"{tenant_id}_{language_code}",
                    name=f"{base_pack.name} (Tenant {tenant_id})",
                    translations=base_pack.translations.copy(),
                    pluralization_rule=base_pack.pluralization_rule
                )
            else:
                locale = Locale(language_code)
                tenant_pack = LanguagePack(
                    locale=locale,
                    code=f"{tenant_id}_{language_code}",
                    name=f"{locale.display_name} (Tenant {tenant_id})",
                    pluralization_rule=self.pluralization_rules.get(language_code)
                )
            
            # Chargement des traductions tenant-spécifiques
            for file_path in lang_dir.glob("*.json"):
                await self._load_json_file(file_path, tenant_pack)
            
            for file_path in lang_dir.glob("*.yaml"):
                await self._load_yaml_file(file_path, tenant_pack)
            
            self.language_packs[tenant_pack_key] = tenant_pack
            
            logger.debug(f"Traductions tenant chargées: {tenant_id}/{language_code}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des traductions tenant {tenant_id}/{language_code}: {e}")
    
    async def _setup_translation_cache(self) -> None:
        """Configure le cache des traductions"""
        # Le cache est géré en mémoire pour les performances
        # Dans une version production, on pourrait utiliser Redis
        logger.info("Cache de traductions configuré")
    
    async def _validate_supported_languages(self) -> None:
        """Valide que toutes les langues supportées sont disponibles"""
        missing_languages = []
        
        for lang_code in self.supported_languages:
            if lang_code not in self.language_packs:
                missing_languages.append(lang_code)
        
        if missing_languages:
            logger.warning(f"Langues supportées manquantes: {missing_languages}")
        
        # Vérification que la langue par défaut est disponible
        if self.default_language not in self.language_packs:
            logger.error(f"Langue par défaut manquante: {self.default_language}")
            raise ValueError(f"Langue par défaut {self.default_language} non disponible")
        
        # Vérification que la langue de fallback est disponible
        if self.fallback_language not in self.language_packs:
            logger.warning(f"Langue de fallback manquante: {self.fallback_language}")
    
    # API publique
    
    async def translate(
        self,
        key: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        tenant_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None
    ) -> str:
        """
        Traduit une clé
        
        Args:
            key: Clé de traduction
            language: Code de langue (défaut si non spécifié)
            context: Contexte de traduction
            tenant_id: ID du tenant pour les traductions spécifiques
            variables: Variables pour l'interpolation
            count: Nombre pour la pluralisation
            
        Returns:
            Texte traduit
        """
        if not self.is_initialized:
            return key
        
        self.metrics["translation_requests"] += 1
        
        # Langue à utiliser
        target_language = language or self.default_language
        
        # Clé de cache
        cache_key = f"{target_language}:{tenant_id or 'global'}:{context or 'default'}:{key}"
        
        # Vérification du cache
        if self.cache_enabled and cache_key in self.translation_cache:
            self.metrics["cache_hits"] += 1
            translation = self.translation_cache[cache_key]
        else:
            self.metrics["cache_misses"] += 1
            translation = await self._get_translation(
                key, target_language, context, tenant_id, count
            )
            
            # Mise en cache
            if self.cache_enabled:
                self.translation_cache[cache_key] = translation
        
        # Interpolation des variables
        if variables and translation:
            translation = await self._interpolate_variables(translation, variables)
        
        return translation
    
    async def _get_translation(
        self,
        key: str,
        language: str,
        context: Optional[str],
        tenant_id: Optional[str],
        count: Optional[int]
    ) -> str:
        """Récupère une traduction"""
        # Recherche dans les traductions tenant-spécifiques
        if tenant_id:
            tenant_pack_key = f"{tenant_id}_{language}"
            if tenant_pack_key in self.language_packs:
                translation = await self._find_translation_in_pack(
                    self.language_packs[tenant_pack_key], key, context, count
                )
                if translation:
                    return translation
        
        # Recherche dans les traductions générales
        if language in self.language_packs:
            translation = await self._find_translation_in_pack(
                self.language_packs[language], key, context, count
            )
            if translation:
                return translation
        
        # Fallback vers la langue de fallback
        if language != self.fallback_language and self.fallback_language in self.language_packs:
            self.metrics["fallback_used"] += 1
            translation = await self._find_translation_in_pack(
                self.language_packs[self.fallback_language], key, context, count
            )
            if translation:
                return translation
        
        # Traduction manquante
        self.metrics["missing_translations"] += 1
        await self._handle_missing_translation(key, language, context, tenant_id)
        
        return key  # Retour de la clé comme fallback final
    
    async def _find_translation_in_pack(
        self,
        language_pack: LanguagePack,
        key: str,
        context: Optional[str],
        count: Optional[int]
    ) -> Optional[str]:
        """Trouve une traduction dans un pack de langue"""
        if key not in language_pack.translations:
            return None
        
        entry = language_pack.translations[key]
        
        # Vérification du contexte
        if context and entry.context and entry.context != context:
            return None
        
        # Gestion de la pluralisation
        if count is not None and entry.plurals:
            plural_form = await self._get_plural_form(language_pack, count)
            return entry.plurals.get(plural_form, entry.value)
        
        return entry.value
    
    async def _get_plural_form(self, language_pack: LanguagePack, count: int) -> str:
        """Détermine la forme plurielle"""
        if language_pack.pluralization_rule:
            return language_pack.pluralization_rule(count)
        else:
            # Règle par défaut simple
            return "one" if count == 1 else "other"
    
    async def _interpolate_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Interpole les variables dans le texte"""
        # Support de différents formats d'interpolation
        
        # Format {variable}
        for var_name, var_value in variables.items():
            text = text.replace(f"{{{var_name}}}", str(var_value))
        
        # Format %(variable)s
        try:
            text = text % variables
        except (KeyError, TypeError, ValueError):
            # Ignore les erreurs d'interpolation
            pass
        
        # Format ${variable} (optionnel)
        pattern = r'\$\{([^}]+)\}'
        
        def replace_match(match):
            var_name = match.group(1)
            return str(variables.get(var_name, match.group(0)))
        
        text = re.sub(pattern, replace_match, text)
        
        return text
    
    async def _handle_missing_translation(
        self,
        key: str,
        language: str,
        context: Optional[str],
        tenant_id: Optional[str]
    ) -> None:
        """Gère une traduction manquante"""
        missing_info = {
            "key": key,
            "language": language,
            "context": context,
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.debug(f"Traduction manquante: {missing_info}")
        
        # Appel des callbacks
        for callback in self.missing_translation_callbacks:
            try:
                await callback(missing_info)
            except Exception as e:
                logger.error(f"Erreur dans le callback de traduction manquante: {e}")
    
    async def translate_multiple(
        self,
        keys: List[str],
        language: Optional[str] = None,
        context: Optional[str] = None,
        tenant_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Traduit plusieurs clés
        
        Args:
            keys: Liste des clés à traduire
            language: Code de langue
            context: Contexte
            tenant_id: ID du tenant
            variables: Variables pour l'interpolation
            
        Returns:
            Dictionnaire clé -> traduction
        """
        translations = {}
        
        for key in keys:
            translations[key] = await self.translate(
                key, language, context, tenant_id, variables
            )
        
        return translations
    
    async def get_available_languages(self, tenant_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Récupère les langues disponibles
        
        Args:
            tenant_id: ID du tenant
            
        Returns:
            Liste des langues avec métadonnées
        """
        languages = []
        
        for lang_code, pack in self.language_packs.items():
            # Filtrage pour les tenants
            if tenant_id:
                if "_" in lang_code and not lang_code.startswith(f"{tenant_id}_"):
                    continue
                # Suppression du préfixe tenant pour l'affichage
                display_code = lang_code.replace(f"{tenant_id}_", "") if "_" in lang_code else lang_code
            else:
                # Exclusion des langues tenant-spécifiques
                if "_" in lang_code:
                    continue
                display_code = lang_code
            
            languages.append({
                "code": display_code,
                "name": pack.name,
                "completion": pack.completion_percentage,
                "translations_count": len(pack.translations)
            })
        
        return sorted(languages, key=lambda x: x["name"])
    
    async def format_date(
        self,
        date_obj: datetime,
        language: Optional[str] = None,
        format: str = "medium"
    ) -> str:
        """
        Formate une date selon la locale
        
        Args:
            date_obj: Date à formater
            language: Code de langue
            format: Format (short, medium, long, full)
            
        Returns:
            Date formatée
        """
        target_language = language or self.default_language
        
        if target_language in self.language_packs:
            locale = self.language_packs[target_language].locale
            return dates.format_date(date_obj, format=format, locale=locale)
        
        return date_obj.strftime("%Y-%m-%d")
    
    async def format_datetime(
        self,
        datetime_obj: datetime,
        language: Optional[str] = None,
        format: str = "medium"
    ) -> str:
        """
        Formate une datetime selon la locale
        
        Args:
            datetime_obj: Datetime à formater
            language: Code de langue
            format: Format
            
        Returns:
            Datetime formatée
        """
        target_language = language or self.default_language
        
        if target_language in self.language_packs:
            locale = self.language_packs[target_language].locale
            return dates.format_datetime(datetime_obj, format=format, locale=locale)
        
        return datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    async def format_number(
        self,
        number: Union[int, float],
        language: Optional[str] = None
    ) -> str:
        """
        Formate un nombre selon la locale
        
        Args:
            number: Nombre à formater
            language: Code de langue
            
        Returns:
            Nombre formaté
        """
        target_language = language or self.default_language
        
        if target_language in self.language_packs:
            locale = self.language_packs[target_language].locale
            return numbers.format_number(number, locale=locale)
        
        return str(number)
    
    async def format_currency(
        self,
        amount: Union[int, float],
        currency: str,
        language: Optional[str] = None
    ) -> str:
        """
        Formate un montant monétaire selon la locale
        
        Args:
            amount: Montant
            currency: Code de devise (EUR, USD, etc.)
            language: Code de langue
            
        Returns:
            Montant formaté
        """
        target_language = language or self.default_language
        
        if target_language in self.language_packs:
            locale = self.language_packs[target_language].locale
            return numbers.format_currency(amount, currency, locale=locale)
        
        return f"{amount} {currency}"
    
    def add_missing_translation_callback(self, callback: Callable) -> None:
        """
        Ajoute un callback pour les traductions manquantes
        
        Args:
            callback: Fonction appelée pour les traductions manquantes
        """
        self.missing_translation_callbacks.append(callback)
    
    def remove_missing_translation_callback(self, callback: Callable) -> None:
        """
        Supprime un callback de traduction manquante
        
        Args:
            callback: Fonction à supprimer
        """
        if callback in self.missing_translation_callbacks:
            self.missing_translation_callbacks.remove(callback)
    
    async def add_translation(
        self,
        key: str,
        value: str,
        language: str,
        context: Optional[str] = None,
        description: Optional[str] = None,
        plurals: Optional[Dict[str, str]] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Ajoute une traduction
        
        Args:
            key: Clé de traduction
            value: Valeur traduite
            language: Code de langue
            context: Contexte optionnel
            description: Description optionnelle
            plurals: Formes plurielles optionnelles
            tenant_id: ID du tenant
            
        Returns:
            True si ajouté avec succès
        """
        try:
            # Détermination du pack de langue
            pack_key = f"{tenant_id}_{language}" if tenant_id else language
            
            if pack_key not in self.language_packs:
                logger.error(f"Pack de langue non trouvé: {pack_key}")
                return False
            
            # Création de l'entrée
            entry = TranslationEntry(
                key=key,
                value=value,
                context=context,
                description=description,
                plurals=plurals or {}
            )
            
            # Ajout au pack
            self.language_packs[pack_key].translations[key] = entry
            
            # Invalidation du cache
            if self.cache_enabled:
                cache_keys_to_remove = [
                    ck for ck in self.translation_cache.keys()
                    if ck.endswith(f":{key}")
                ]
                for cache_key in cache_keys_to_remove:
                    del self.translation_cache[cache_key]
            
            logger.info(f"Traduction ajoutée: {key} ({language})")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la traduction {key}: {e}")
            return False
    
    async def remove_translation(
        self,
        key: str,
        language: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Supprime une traduction
        
        Args:
            key: Clé de traduction
            language: Code de langue
            tenant_id: ID du tenant
            
        Returns:
            True si supprimé avec succès
        """
        try:
            pack_key = f"{tenant_id}_{language}" if tenant_id else language
            
            if pack_key not in self.language_packs:
                return False
            
            if key in self.language_packs[pack_key].translations:
                del self.language_packs[pack_key].translations[key]
                
                # Invalidation du cache
                if self.cache_enabled:
                    cache_keys_to_remove = [
                        ck for ck in self.translation_cache.keys()
                        if ck.endswith(f":{key}")
                    ]
                    for cache_key in cache_keys_to_remove:
                        del self.translation_cache[cache_key]
                
                logger.info(f"Traduction supprimée: {key} ({language})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la traduction {key}: {e}")
            return False
    
    async def get_completion_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les statistiques de complétude
        
        Args:
            tenant_id: ID du tenant
            
        Returns:
            Statistiques de complétude
        """
        stats = {}
        
        reference_pack = None
        if self.default_language in self.language_packs:
            reference_pack = self.language_packs[self.default_language]
        
        for lang_code, pack in self.language_packs.items():
            # Filtrage pour les tenants
            if tenant_id:
                if "_" in lang_code and not lang_code.startswith(f"{tenant_id}_"):
                    continue
                display_code = lang_code.replace(f"{tenant_id}_", "") if "_" in lang_code else lang_code
            else:
                if "_" in lang_code:
                    continue
                display_code = lang_code
            
            total_translations = len(pack.translations)
            missing_translations = 0
            
            if reference_pack and pack != reference_pack:
                missing_translations = len(reference_pack.translations) - total_translations
            
            completion_percentage = pack.completion_percentage
            if reference_pack and len(reference_pack.translations) > 0:
                completion_percentage = (total_translations / len(reference_pack.translations)) * 100
            
            stats[display_code] = {
                "total_translations": total_translations,
                "missing_translations": max(0, missing_translations),
                "completion_percentage": round(completion_percentage, 2),
                "last_updated": pack.last_updated.isoformat()
            }
        
        return stats
    
    async def clear_cache(self) -> None:
        """Vide le cache des traductions"""
        self.translation_cache.clear()
        logger.info("Cache de traductions vidé")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du gestionnaire
        
        Returns:
            Métriques
        """
        return {
            **self.metrics,
            "language_packs": len(self.language_packs),
            "cache_size": len(self.translation_cache),
            "supported_languages": len(self.supported_languages),
            "tenant_locales": len(self.tenant_locale_dirs)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé du gestionnaire
        
        Returns:
            Rapport d'état
        """
        try:
            return {
                "status": "healthy",
                "is_initialized": self.is_initialized,
                "language_packs_loaded": len(self.language_packs),
                "default_language_available": self.default_language in self.language_packs,
                "fallback_language_available": self.fallback_language in self.language_packs,
                "cache_enabled": self.cache_enabled,
                "cache_size": len(self.translation_cache)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_initialized": self.is_initialized
            }
