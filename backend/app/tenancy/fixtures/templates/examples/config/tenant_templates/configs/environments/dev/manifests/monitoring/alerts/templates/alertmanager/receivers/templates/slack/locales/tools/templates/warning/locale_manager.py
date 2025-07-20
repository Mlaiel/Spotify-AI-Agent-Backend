"""
Locale Manager - Gestionnaire de Localisation Avancé pour Spotify AI Agent
Support multilingue complet avec traduction dynamique et adaptation culturelle
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import aiofiles
from pathlib import Path

import aioredis
from googletrans import Translator
from babel.dates import format_datetime
from babel.numbers import format_decimal
from babel import Locale
import yaml


class SupportedLocale(Enum):
    """Locales supportées par le système"""
    EN_US = "en_US"  # Anglais (États-Unis)
    FR_FR = "fr_FR"  # Français (France)
    DE_DE = "de_DE"  # Allemand (Allemagne)
    ES_ES = "es_ES"  # Espagnol (Espagne)
    IT_IT = "it_IT"  # Italien (Italie)
    PT_BR = "pt_BR"  # Portugais (Brésil)
    JA_JP = "ja_JP"  # Japonais (Japon)
    KO_KR = "ko_KR"  # Coréen (Corée du Sud)
    ZH_CN = "zh_CN"  # Chinois simplifié (Chine)
    ZH_TW = "zh_TW"  # Chinois traditionnel (Taïwan)
    RU_RU = "ru_RU"  # Russe (Russie)
    AR_SA = "ar_SA"  # Arabe (Arabie Saoudite)
    HI_IN = "hi_IN"  # Hindi (Inde)
    NL_NL = "nl_NL"  # Néerlandais (Pays-Bas)
    SV_SE = "sv_SE"  # Suédois (Suède)


@dataclass
class LocalizedMessage:
    """Message localisé avec métadonnées"""
    original_text: str
    localized_text: str
    locale: str
    translation_method: str
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class CulturalContext:
    """Contexte culturel pour adaptation des messages"""
    locale: str
    date_format: str
    time_format: str
    number_format: str
    currency_symbol: str
    formal_address: bool
    cultural_notes: List[str]


class LocaleManager:
    """
    Gestionnaire de localisation avancé avec fonctionnalités :
    - Traduction automatique multi-moteur
    - Cache intelligent des traductions
    - Adaptation culturelle des messages
    - Formatage localisé des dates/nombres
    - Détection automatique de langue
    - Templates multilingues
    - Fallback gracieux
    - Support RTL (Right-to-Left)
    """
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        translations_path: str,
        default_locale: str = "en_US",
        config: Dict[str, Any] = None,
        tenant_id: str = ""
    ):
        self.redis_client = redis_client
        self.translations_path = Path(translations_path)
        self.default_locale = default_locale
        self.config = config or {}
        self.tenant_id = tenant_id
        
        # Logger avec contexte
        self.logger = logging.getLogger(f"locale_manager.{tenant_id}")
        
        # Traducteurs
        self.google_translator = Translator()
        
        # Cache des traductions
        self.translation_cache = {}
        self.template_cache = {}
        
        # Mapping des codes de langue
        self.locale_mapping = {
            "en": "en_US", "fr": "fr_FR", "de": "de_DE", "es": "es_ES",
            "it": "it_IT", "pt": "pt_BR", "ja": "ja_JP", "ko": "ko_KR",
            "zh": "zh_CN", "ru": "ru_RU", "ar": "ar_SA", "hi": "hi_IN",
            "nl": "nl_NL", "sv": "sv_SE"
        }
        
        # Configuration culturelle
        self.cultural_contexts = {}
        
        # Templates de messages par catégorie
        self.message_templates = {}
        
        # Patterns de détection de langue
        self.language_patterns = {
            'en': r'[a-zA-Z\s.,!?]+',
            'fr': r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ\s.,!?]+',
            'de': r'[a-zA-ZäöüßÄÖÜ\s.,!?]+',
            'es': r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s.,!?]+',
            'it': r'[a-zA-ZàèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ\s.,!?]+',
            'ja': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\s.,!?]+',
            'ko': r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\s.,!?]+',
            'zh': r'[\u4E00-\u9FFF\s.,!?]+',
            'ar': r'[\u0600-\u06FF\u0750-\u077F\s.,!?]+',
            'ru': r'[а-яёА-ЯЁ\s.,!?]+'
        }
        
        # Initialisation asynchrone
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialisation asynchrone du gestionnaire de locale"""
        if self._initialized:
            return
        
        try:
            # Chargement des traductions
            await self._load_translations()
            
            # Chargement des contextes culturels
            await self._load_cultural_contexts()
            
            # Chargement des templates de messages
            await self._load_message_templates()
            
            # Initialisation du cache Redis
            await self._initialize_cache()
            
            self._initialized = True
            self.logger.info("Locale manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize locale manager: {str(e)}")
            raise
    
    async def localize_message(
        self,
        message: str,
        locale: str,
        context: Optional[Dict[str, Any]] = None,
        force_translation: bool = False
    ) -> str:
        """
        Localise un message pour la locale spécifiée
        
        Args:
            message: Message à localiser
            locale: Locale cible
            context: Contexte additionnel pour la traduction
            force_translation: Force la traduction même si déjà en cache
            
        Returns:
            str: Message localisé
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Normalisation de la locale
            normalized_locale = self._normalize_locale(locale)
            
            # Si c'est déjà la locale par défaut, retour direct
            if normalized_locale == self.default_locale:
                return message
            
            # Vérification du cache
            cache_key = self._generate_cache_key(message, normalized_locale)
            
            if not force_translation:
                cached_translation = await self._get_cached_translation(cache_key)
                if cached_translation:
                    return cached_translation
            
            # Détection de la langue source
            source_lang = await self._detect_language(message)
            
            # Traduction
            translated_message = await self._translate_message(
                message=message,
                source_lang=source_lang,
                target_locale=normalized_locale,
                context=context
            )
            
            # Adaptation culturelle
            culturally_adapted = await self._adapt_culturally(
                translated_message, normalized_locale, context
            )
            
            # Mise en cache
            await self._cache_translation(cache_key, culturally_adapted)
            
            # Log de l'activité
            self.logger.info(
                f"Message localized",
                extra={
                    "source_lang": source_lang,
                    "target_locale": normalized_locale,
                    "message_length": len(message),
                    "tenant_id": self.tenant_id
                }
            )
            
            return culturally_adapted
            
        except Exception as e:
            self.logger.error(f"Error localizing message: {str(e)}")
            return message  # Fallback vers le message original
    
    async def localize_template(
        self,
        template_name: str,
        locale: str,
        variables: Dict[str, Any],
        category: str = "general"
    ) -> str:
        """
        Localise un template avec variables
        
        Args:
            template_name: Nom du template
            locale: Locale cible
            variables: Variables pour le template
            category: Catégorie du template
            
        Returns:
            str: Template localisé et rendu
        """
        try:
            # Normalisation de la locale
            normalized_locale = self._normalize_locale(locale)
            
            # Récupération du template localisé
            template = await self._get_localized_template(
                template_name, normalized_locale, category
            )
            
            if not template:
                # Fallback vers le template par défaut
                template = await self._get_localized_template(
                    template_name, self.default_locale, category
                )
            
            if not template:
                return f"Template '{template_name}' not found"
            
            # Localisation des variables
            localized_variables = await self._localize_variables(
                variables, normalized_locale
            )
            
            # Rendu du template
            rendered = await self._render_template(template, localized_variables)
            
            return rendered
            
        except Exception as e:
            self.logger.error(f"Error localizing template: {str(e)}")
            return f"Error rendering template: {template_name}"
    
    async def format_datetime(
        self,
        dt: datetime,
        locale: str,
        format_type: str = "medium"
    ) -> str:
        """
        Formate une date selon la locale
        
        Args:
            dt: Date à formater
            locale: Locale pour le formatage
            format_type: Type de format (short, medium, long, full)
            
        Returns:
            str: Date formatée
        """
        try:
            normalized_locale = self._normalize_locale(locale)
            babel_locale = Locale.parse(normalized_locale.replace('_', '-'))
            
            return format_datetime(dt, format=format_type, locale=babel_locale)
            
        except Exception as e:
            self.logger.error(f"Error formatting datetime: {str(e)}")
            return dt.strftime('%Y-%m-%d %H:%M:%S')  # Fallback
    
    async def format_number(
        self,
        number: Union[int, float],
        locale: str,
        decimal_places: Optional[int] = None
    ) -> str:
        """
        Formate un nombre selon la locale
        
        Args:
            number: Nombre à formater
            locale: Locale pour le formatage
            decimal_places: Nombre de décimales
            
        Returns:
            str: Nombre formaté
        """
        try:
            normalized_locale = self._normalize_locale(locale)
            babel_locale = Locale.parse(normalized_locale.replace('_', '-'))
            
            if decimal_places is not None:
                return format_decimal(number, locale=babel_locale, decimal_quantization=False)
            else:
                return format_decimal(number, locale=babel_locale)
            
        except Exception as e:
            self.logger.error(f"Error formatting number: {str(e)}")
            return str(number)  # Fallback
    
    async def detect_locale_from_context(
        self,
        context: Dict[str, Any]
    ) -> str:
        """
        Détecte la locale appropriée à partir du contexte
        
        Args:
            context: Contexte contenant des indices de locale
            
        Returns:
            str: Locale détectée
        """
        try:
            # Priorité 1: Locale explicite dans le contexte
            if 'locale' in context:
                return self._normalize_locale(context['locale'])
            
            # Priorité 2: User agent / Accept-Language
            if 'user_agent' in context:
                detected = await self._detect_locale_from_user_agent(context['user_agent'])
                if detected:
                    return detected
            
            # Priorité 3: Géolocalisation
            if 'country_code' in context:
                detected = await self._locale_from_country(context['country_code'])
                if detected:
                    return detected
            
            # Priorité 4: Tenant configuration
            if self.tenant_id:
                tenant_locale = await self._get_tenant_default_locale()
                if tenant_locale:
                    return tenant_locale
            
            return self.default_locale
            
        except Exception as e:
            self.logger.error(f"Error detecting locale: {str(e)}")
            return self.default_locale
    
    async def get_supported_locales(self) -> List[Dict[str, str]]:
        """
        Retourne la liste des locales supportées
        
        Returns:
            List[Dict]: Liste des locales avec métadonnées
        """
        locales = []
        
        for locale_enum in SupportedLocale:
            locale_code = locale_enum.value
            locale_info = {
                'code': locale_code,
                'name': await self._get_locale_native_name(locale_code),
                'english_name': await self._get_locale_english_name(locale_code),
                'rtl': await self._is_rtl_locale(locale_code),
                'supported_features': await self._get_locale_features(locale_code)
            }
            locales.append(locale_info)
        
        return locales
    
    async def get_translation_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de traduction
        
        Returns:
            Dict: Statistiques détaillées
        """
        try:
            stats_key = f"translation_stats:{self.tenant_id}"
            cached_stats = await self.redis_client.get(stats_key)
            
            if cached_stats:
                return json.loads(cached_stats)
            
            # Calcul des statistiques
            total_translations = await self._count_cached_translations()
            locale_distribution = await self._get_locale_distribution()
            cache_hit_rate = await self._calculate_cache_hit_rate()
            
            stats = {
                'total_translations': total_translations,
                'locale_distribution': locale_distribution,
                'cache_hit_rate': cache_hit_rate,
                'supported_locales_count': len(SupportedLocale),
                'last_updated': datetime.utcnow().isoformat(),
                'tenant_id': self.tenant_id
            }
            
            # Cache pour 10 minutes
            await self.redis_client.setex(stats_key, 600, json.dumps(stats))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting translation stats: {str(e)}")
            return {'error': str(e)}
    
    # Méthodes privées
    
    async def _load_translations(self) -> None:
        """Charge les traductions depuis les fichiers"""
        try:
            if not self.translations_path.exists():
                self.logger.warning(f"Translations path not found: {self.translations_path}")
                return
            
            for locale_file in self.translations_path.glob("*.yml"):
                locale_code = locale_file.stem
                
                async with aiofiles.open(locale_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    translations = yaml.safe_load(content)
                    self.translation_cache[locale_code] = translations
            
            self.logger.info(f"Loaded translations for {len(self.translation_cache)} locales")
            
        except Exception as e:
            self.logger.error(f"Error loading translations: {str(e)}")
    
    async def _load_cultural_contexts(self) -> None:
        """Charge les contextes culturels"""
        self.cultural_contexts = {
            "en_US": CulturalContext(
                locale="en_US",
                date_format="MM/dd/yyyy",
                time_format="h:mm a",
                number_format="1,234.56",
                currency_symbol="$",
                formal_address=False,
                cultural_notes=["Direct communication style", "Time-conscious"]
            ),
            "fr_FR": CulturalContext(
                locale="fr_FR",
                date_format="dd/MM/yyyy",
                time_format="HH:mm",
                number_format="1 234,56",
                currency_symbol="€",
                formal_address=True,
                cultural_notes=["Formal address preferred", "Hierarchical respect"]
            ),
            "de_DE": CulturalContext(
                locale="de_DE",
                date_format="dd.MM.yyyy",
                time_format="HH:mm",
                number_format="1.234,56",
                currency_symbol="€",
                formal_address=True,
                cultural_notes=["Precision valued", "Formal communication"]
            ),
            "ja_JP": CulturalContext(
                locale="ja_JP",
                date_format="yyyy/MM/dd",
                time_format="HH:mm",
                number_format="1,234.56",
                currency_symbol="¥",
                formal_address=True,
                cultural_notes=["Respect and humility", "Indirect communication"]
            ),
            "ar_SA": CulturalContext(
                locale="ar_SA",
                date_format="dd/MM/yyyy",
                time_format="HH:mm",
                number_format="1,234.56",
                currency_symbol="ر.س",
                formal_address=True,
                cultural_notes=["Right-to-left text", "Respectful tone"]
            )
        }
    
    async def _load_message_templates(self) -> None:
        """Charge les templates de messages"""
        self.message_templates = {
            "alerts": {
                "en_US": {
                    "critical_alert": "🚨 CRITICAL: {message} - Immediate action required",
                    "warning_alert": "⚠️ WARNING: {message} - Please review",
                    "info_alert": "ℹ️ INFO: {message}",
                    "resolved_alert": "✅ RESOLVED: {message}"
                },
                "fr_FR": {
                    "critical_alert": "🚨 CRITIQUE: {message} - Action immédiate requise",
                    "warning_alert": "⚠️ ATTENTION: {message} - Veuillez vérifier",
                    "info_alert": "ℹ️ INFO: {message}",
                    "resolved_alert": "✅ RÉSOLU: {message}"
                },
                "de_DE": {
                    "critical_alert": "🚨 KRITISCH: {message} - Sofortige Maßnahme erforderlich",
                    "warning_alert": "⚠️ WARNUNG: {message} - Bitte überprüfen",
                    "info_alert": "ℹ️ INFO: {message}",
                    "resolved_alert": "✅ GELÖST: {message}"
                }
            },
            "notifications": {
                "en_US": {
                    "welcome": "Welcome to Spotify AI Agent, {name}!",
                    "alert_summary": "You have {count} active alerts",
                    "system_status": "System status: {status}"
                },
                "fr_FR": {
                    "welcome": "Bienvenue dans l'Agent IA Spotify, {name}!",
                    "alert_summary": "Vous avez {count} alertes actives",
                    "system_status": "État du système: {status}"
                },
                "de_DE": {
                    "welcome": "Willkommen beim Spotify KI-Agent, {name}!",
                    "alert_summary": "Sie haben {count} aktive Warnungen",
                    "system_status": "Systemstatus: {status}"
                }
            }
        }
    
    async def _initialize_cache(self) -> None:
        """Initialise le cache Redis pour les traductions"""
        cache_key = f"locale_cache_initialized:{self.tenant_id}"
        is_initialized = await self.redis_client.get(cache_key)
        
        if not is_initialized:
            # Pré-chargement des traductions fréquentes
            await self._preload_common_translations()
            await self.redis_client.setex(cache_key, 3600, "true")
    
    def _normalize_locale(self, locale: str) -> str:
        """Normalise une locale vers le format standard"""
        if not locale:
            return self.default_locale
        
        # Gestion des codes courts (en -> en_US)
        if len(locale) == 2:
            return self.locale_mapping.get(locale.lower(), self.default_locale)
        
        # Gestion des formats avec tiret (en-US -> en_US)
        if '-' in locale:
            locale = locale.replace('-', '_')
        
        # Vérification de la validité
        if locale in [l.value for l in SupportedLocale]:
            return locale
        
        # Fallback vers la langue principale
        lang_code = locale.split('_')[0]
        return self.locale_mapping.get(lang_code, self.default_locale)
    
    async def _detect_language(self, text: str) -> str:
        """Détecte la langue d'un texte"""
        try:
            # Tentative avec Google Translate
            detection = self.google_translator.detect(text)
            if detection.confidence > 0.8:
                return detection.lang
        except:
            pass
        
        # Fallback avec patterns de caractères
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, text):
                return lang
        
        return 'en'  # Défaut
    
    async def _translate_message(
        self,
        message: str,
        source_lang: str,
        target_locale: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Traduit un message vers la locale cible"""
        try:
            target_lang = target_locale.split('_')[0]
            
            if source_lang == target_lang:
                return message
            
            # Traduction avec Google Translate
            translation = self.google_translator.translate(
                message,
                src=source_lang,
                dest=target_lang
            )
            
            return translation.text
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return message
    
    async def _adapt_culturally(
        self,
        message: str,
        locale: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Adapte un message selon le contexte culturel"""
        cultural_context = self.cultural_contexts.get(locale)
        if not cultural_context:
            return message
        
        adapted_message = message
        
        # Adaptation du niveau de formalité
        if cultural_context.formal_address and context:
            # Ajout de formules de politesse si nécessaire
            if locale.startswith('fr_'):
                adapted_message = f"Veuillez noter: {adapted_message}"
            elif locale.startswith('de_'):
                adapted_message = f"Bitte beachten Sie: {adapted_message}"
            elif locale.startswith('ja_'):
                adapted_message = f"ご注意ください: {adapted_message}"
        
        return adapted_message
    
    def _generate_cache_key(self, message: str, locale: str) -> str:
        """Génère une clé de cache pour une traduction"""
        import hashlib
        message_hash = hashlib.md5(message.encode()).hexdigest()[:8]
        return f"translation:{self.tenant_id}:{locale}:{message_hash}"
    
    async def _get_cached_translation(self, cache_key: str) -> Optional[str]:
        """Récupère une traduction du cache"""
        try:
            cached = await self.redis_client.get(cache_key)
            return cached.decode() if cached else None
        except:
            return None
    
    async def _cache_translation(self, cache_key: str, translation: str) -> None:
        """Met en cache une traduction"""
        try:
            await self.redis_client.setex(cache_key, 3600, translation)
        except Exception as e:
            self.logger.error(f"Error caching translation: {str(e)}")
    
    async def _get_localized_template(
        self,
        template_name: str,
        locale: str,
        category: str
    ) -> Optional[str]:
        """Récupère un template localisé"""
        templates = self.message_templates.get(category, {})
        locale_templates = templates.get(locale, {})
        return locale_templates.get(template_name)
    
    async def _localize_variables(
        self,
        variables: Dict[str, Any],
        locale: str
    ) -> Dict[str, Any]:
        """Localise les variables d'un template"""
        localized = {}
        
        for key, value in variables.items():
            if isinstance(value, str):
                localized[key] = await self.localize_message(value, locale)
            elif isinstance(value, datetime):
                localized[key] = await self.format_datetime(value, locale)
            elif isinstance(value, (int, float)):
                localized[key] = await self.format_number(value, locale)
            else:
                localized[key] = value
        
        return localized
    
    async def _render_template(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """Rend un template avec les variables"""
        try:
            return template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {str(e)}")
            return template
        except Exception as e:
            self.logger.error(f"Template rendering error: {str(e)}")
            return template
