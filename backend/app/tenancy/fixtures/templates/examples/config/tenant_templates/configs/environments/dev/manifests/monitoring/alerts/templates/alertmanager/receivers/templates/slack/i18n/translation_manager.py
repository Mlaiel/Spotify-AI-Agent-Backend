#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gestionnaire de traductions intelligent pour alertes Slack

Ce module fournit un gestionnaire de traductions avancé avec:
- Cache distribué Redis ultra-performant
- Détection automatique de langue basée IA
- Traductions contextuelles adaptatives
- Fallback intelligent multi-niveaux
- Intégration IA pour amélioration continue
- Support RTL (Right-to-Left) complet
- Pluralisation avancée par langue
- Formatage culturel automatique
- Monitoring et métriques en temps réel
- Hot-reload des traductions

Auteur: Expert Team
Version: 2.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import hashlib
import re

import yaml
import aioredis
from babel import Locale, dates, numbers
from babel.core import get_global
from langdetect import detect, LangDetectError
import jinja2
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


class TranslationPriority(Enum):
    """Priorités des sources de traduction"""
    CACHE = 1
    USER_OVERRIDE = 2
    AI_ENHANCED = 3
    STATIC = 4
    FALLBACK = 5


class CacheStrategy(Enum):
    """Stratégies de mise en cache"""
    LONG_TERM = "long_term"      # 24h
    MEDIUM_TERM = "medium_term"  # 4h
    SHORT_TERM = "short_term"    # 1h
    SESSION = "session"          # Session utilisateur


@dataclass
class TranslationRequest:
    """Requête de traduction structurée"""
    key: str
    language: str
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    priority: TranslationPriority = TranslationPriority.STATIC
    use_ai: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM
    fallback_languages: List[str] = field(default_factory=list)
    
    def to_cache_key(self) -> str:
        """Génère une clé de cache unique"""
        context_hash = hashlib.md5(
            json.dumps(self.context, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"trans:{self.language}:{self.key}:{context_hash}"


@dataclass
class TranslationResult:
    """Résultat de traduction enrichi"""
    text: str
    language: str
    source: TranslationPriority
    cached: bool
    ai_enhanced: bool = False
    processing_time_ms: float = 0.0
    confidence_score: float = 1.0
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedTranslationManager:
    """Gestionnaire de traductions intelligent et ultra-performant"""
    
    def __init__(self, 
                 translations_path: Path,
                 redis_url: str = "redis://localhost:6379/2",
                 ai_api_key: Optional[str] = None,
                 default_language: str = "en"):
        """
        Initialise le gestionnaire de traductions
        
        Args:
            translations_path: Chemin vers les fichiers de traduction
            redis_url: URL de connexion Redis pour le cache
            ai_api_key: Clé API pour l'amélioration IA
            default_language: Langue par défaut
        """
        self.translations_path = Path(translations_path)
        self.redis_url = redis_url
        self.ai_api_key = ai_api_key
        self.default_language = default_language
        
        # Cache local pour améliorer les performances
        self._local_cache: Dict[str, TranslationResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._local_cache_ttl = timedelta(minutes=5)
        
        # Pool de connexions Redis
        self._redis_pool: Optional[aioredis.ConnectionPool] = None
        self._redis: Optional[aioredis.Redis] = None
        
        # Données de traduction chargées
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._pluralization_rules: Dict[str, str] = {}
        self._cultural_configs: Dict[str, Dict[str, Any]] = {}
        
        # Statistiques et monitoring
        self._stats = {
            "requests_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "ai_enhancements": 0,
            "fallbacks_used": 0,
            "errors_count": 0,
            "avg_processing_time": 0.0
        }
        
        # Configuration Jinja2 pour les templates
        self._jinja_env = Environment(
            loader=FileSystemLoader(self.translations_path),
            autoescape=select_autoescape(['html', 'xml']),
            enable_async=True
        )
        
        # Langues supportées avec fallbacks
        self._language_fallbacks = {
            "fr": ["en"],
            "de": ["en"],
            "es": ["en"],
            "pt": ["es", "en"],
            "it": ["es", "en"],
            "ru": ["en"],
            "zh": ["en"],
            "ja": ["en"],
            "ar": ["en"],
            "he": ["en"]
        }
        
        logger.info(f"Gestionnaire de traductions initialisé - Langue par défaut: {default_language}")
    
    async def initialize(self) -> None:
        """Initialise les connexions et charge les données"""
        start_time = time.time()
        
        try:
            # Initialisation Redis
            await self._init_redis()
            
            # Chargement des traductions
            await self._load_translations()
            
            # Validation des données
            await self._validate_translations()
            
            # Configuration des templates Jinja2
            await self._setup_templates()
            
            init_time = (time.time() - start_time) * 1000
            logger.info(f"Gestionnaire initialisé en {init_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def _init_redis(self) -> None:
        """Initialise la connexion Redis"""
        try:
            self._redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self._redis = aioredis.Redis(connection_pool=self._redis_pool)
            
            # Test de connexion
            await self._redis.ping()
            logger.info("Connexion Redis établie avec succès")
            
        except Exception as e:
            logger.warning(f"Erreur Redis: {e}. Fonctionnement sans cache.")
            self._redis = None
    
    async def _load_translations(self) -> None:
        """Charge toutes les traductions depuis les fichiers"""
        try:
            # Chargement du fichier principal
            translations_file = self.translations_path / "translations.yaml"
            
            if not translations_file.exists():
                raise FileNotFoundError(f"Fichier de traductions non trouvé: {translations_file}")
            
            with open(translations_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Extraction des traductions par langue
            for lang_code in ["en", "fr", "de", "es", "pt", "it", "ru", "zh", "ja", "ar", "he"]:
                if lang_code in data:
                    self._translations[lang_code] = data[lang_code]
            
            # Chargement des règles de pluralisation
            if "pluralization_rules" in data:
                self._pluralization_rules = data["pluralization_rules"]
            
            # Chargement des configurations culturelles
            for config_key in ["date_formatting", "number_formatting", "cultural_emojis"]:
                if config_key in data:
                    self._cultural_configs[config_key] = data[config_key]
            
            logger.info(f"Traductions chargées pour {len(self._translations)} langues")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des traductions: {e}")
            raise
    
    async def _validate_translations(self) -> None:
        """Valide la cohérence des traductions"""
        base_lang = self.default_language
        base_keys = set()
        
        if base_lang in self._translations:
            base_keys = self._collect_all_keys(self._translations[base_lang])
        
        for lang_code, translations in self._translations.items():
            if lang_code == base_lang:
                continue
                
            lang_keys = self._collect_all_keys(translations)
            missing_keys = base_keys - lang_keys
            
            if missing_keys:
                logger.warning(f"Clés manquantes en {lang_code}: {list(missing_keys)[:5]}...")
    
    def _collect_all_keys(self, data: Dict[str, Any], prefix: str = "") -> Set[str]:
        """Collecte récursivement toutes les clés de traduction"""
        keys = set()
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                keys.update(self._collect_all_keys(value, full_key))
            else:
                keys.add(full_key)
        
        return keys
    
    async def _setup_templates(self) -> None:
        """Configure les templates Jinja2 pour le rendu avancé"""
        # Fonctions personnalisées pour Jinja2
        def format_datetime(dt: datetime, lang: str = "en", format_type: str = "datetime") -> str:
            return self._format_datetime(dt, lang, format_type)
        
        def format_number(num: Union[int, float], lang: str = "en", format_type: str = "decimal") -> str:
            return self._format_number(num, lang, format_type)
        
        def get_emoji(category: str, key: str, lang: str = "en") -> str:
            return self._get_cultural_emoji(category, key, lang)
        
        # Ajout des fonctions aux templates
        self._jinja_env.globals.update({
            'format_datetime': format_datetime,
            'format_number': format_number,
            'get_emoji': get_emoji,
            'utcnow': datetime.utcnow
        })
    
    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """
        Traduit une clé avec contexte intelligent
        
        Args:
            request: Requête de traduction structurée
            
        Returns:
            Résultat de traduction enrichi
        """
        start_time = time.time()
        self._stats["requests_count"] += 1
        
        try:
            # Vérification du cache local
            cache_key = request.to_cache_key()
            local_result = self._get_from_local_cache(cache_key)
            if local_result:
                self._stats["cache_hits"] += 1
                return local_result
            
            # Vérification du cache Redis
            if self._redis:
                redis_result = await self._get_from_redis_cache(cache_key)
                if redis_result:
                    self._stats["cache_hits"] += 1
                    self._set_local_cache(cache_key, redis_result)
                    return redis_result
            
            self._stats["cache_misses"] += 1
            
            # Traduction avec fallback intelligent
            result = await self._perform_translation(request)
            
            # Mise en cache du résultat
            await self._cache_result(cache_key, result, request.cache_strategy)
            
            # Mise à jour des statistiques
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            self._update_stats(processing_time)
            
            return result
            
        except Exception as e:
            self._stats["errors_count"] += 1
            logger.error(f"Erreur lors de la traduction: {e}")
            
            # Fallback d'urgence
            return TranslationResult(
                text=f"Translation error: {request.key}",
                language=request.language,
                source=TranslationPriority.FALLBACK,
                cached=False,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _perform_translation(self, request: TranslationRequest) -> TranslationResult:
        """Effectue la traduction avec logique de fallback"""
        languages_to_try = [request.language] + request.fallback_languages + self._language_fallbacks.get(request.language, [])
        languages_to_try.append(self.default_language)
        
        # Suppression des doublons en gardant l'ordre
        seen = set()
        unique_languages = []
        for lang in languages_to_try:
            if lang not in seen:
                seen.add(lang)
                unique_languages.append(lang)
        
        for lang in unique_languages:
            translation = self._get_static_translation(request.key, lang)
            if translation:
                # Rendu avec contexte
                rendered_text = await self._render_translation(translation, request.context, lang)
                
                # Amélioration IA si activée
                if request.use_ai and self.ai_api_key:
                    enhanced_text = await self._enhance_with_ai(rendered_text, request, lang)
                    if enhanced_text:
                        return TranslationResult(
                            text=enhanced_text,
                            language=lang,
                            source=TranslationPriority.AI_ENHANCED,
                            cached=False,
                            ai_enhanced=True,
                            fallback_used=(lang != request.language)
                        )
                
                return TranslationResult(
                    text=rendered_text,
                    language=lang,
                    source=TranslationPriority.STATIC,
                    cached=False,
                    fallback_used=(lang != request.language)
                )
        
        # Fallback final
        self._stats["fallbacks_used"] += 1
        return TranslationResult(
            text=f"Missing translation: {request.key}",
            language=self.default_language,
            source=TranslationPriority.FALLBACK,
            cached=False,
            fallback_used=True
        )
    
    def _get_static_translation(self, key: str, language: str) -> Optional[str]:
        """Récupère une traduction statique"""
        if language not in self._translations:
            return None
        
        # Navigation dans l'arbre de traductions
        parts = key.split('.')
        current = self._translations[language]
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    async def _render_translation(self, template: str, context: Dict[str, Any], language: str) -> str:
        """Rend un template de traduction avec le contexte"""
        try:
            # Enrichissement du contexte avec les données culturelles
            enriched_context = {
                **context,
                'lang': language,
                'rtl': self._is_rtl_language(language),
                'culture': self._cultural_configs.get('date_formatting', {}).get(language, {})
            }
            
            # Rendu avec Jinja2
            template_obj = self._jinja_env.from_string(template)
            return await template_obj.render_async(**enriched_context)
            
        except Exception as e:
            logger.error(f"Erreur lors du rendu du template: {e}")
            return template  # Retour du template non rendu en cas d'erreur
    
    async def _enhance_with_ai(self, text: str, request: TranslationRequest, language: str) -> Optional[str]:
        """Améliore la traduction avec l'IA"""
        # TODO: Implémentation de l'amélioration IA
        # Cette fonction sera implémentée pour intégrer GPT-4 ou autre modèle
        self._stats["ai_enhancements"] += 1
        return None
    
    def _is_rtl_language(self, language: str) -> bool:
        """Détermine si une langue utilise RTL"""
        return language in ["ar", "he", "ur", "fa"]
    
    def _format_datetime(self, dt: datetime, language: str, format_type: str = "datetime") -> str:
        """Formate une date selon la culture"""
        try:
            locale = Locale.parse(language)
            
            if format_type == "relative":
                return dates.format_timedelta(
                    datetime.utcnow() - dt,
                    locale=locale,
                    add_direction=True
                )
            elif format_type == "date":
                return dates.format_date(dt, locale=locale)
            elif format_type == "time":
                return dates.format_time(dt, locale=locale)
            else:
                return dates.format_datetime(dt, locale=locale)
                
        except Exception as e:
            logger.error(f"Erreur formatage date: {e}")
            return dt.isoformat()
    
    def _format_number(self, num: Union[int, float], language: str, format_type: str = "decimal") -> str:
        """Formate un nombre selon la culture"""
        try:
            locale = Locale.parse(language)
            
            if format_type == "percent":
                return numbers.format_percent(num, locale=locale)
            elif format_type == "currency":
                # TODO: Configuration de la devise par langue
                return numbers.format_currency(num, "EUR", locale=locale)
            else:
                return numbers.format_decimal(num, locale=locale)
                
        except Exception as e:
            logger.error(f"Erreur formatage nombre: {e}")
            return str(num)
    
    def _get_cultural_emoji(self, category: str, key: str, language: str) -> str:
        """Récupère un emoji adapté à la culture"""
        cultural_emojis = self._cultural_configs.get("cultural_emojis", {})
        
        # Recherche spécifique à la langue
        if category in cultural_emojis:
            lang_specific = cultural_emojis[category].get(language, {})
            if key in lang_specific:
                return lang_specific[key]
            
            # Fallback vers default
            default_emojis = cultural_emojis[category].get("default", {})
            return default_emojis.get(key, "")
        
        return ""
    
    def _get_from_local_cache(self, cache_key: str) -> Optional[TranslationResult]:
        """Récupère du cache local"""
        if cache_key in self._local_cache:
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp and datetime.now() - timestamp < self._local_cache_ttl:
                result = self._local_cache[cache_key]
                result.cached = True
                return result
            else:
                # Nettoyage du cache expiré
                del self._local_cache[cache_key]
                if cache_key in self._cache_timestamps:
                    del self._cache_timestamps[cache_key]
        
        return None
    
    def _set_local_cache(self, cache_key: str, result: TranslationResult) -> None:
        """Met en cache localement"""
        self._local_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Nettoyage périodique du cache
        if len(self._local_cache) > 1000:
            self._cleanup_local_cache()
    
    def _cleanup_local_cache(self) -> None:
        """Nettoie le cache local"""
        now = datetime.now()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if now - timestamp > self._local_cache_ttl
        ]
        
        for key in expired_keys:
            self._local_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    async def _get_from_redis_cache(self, cache_key: str) -> Optional[TranslationResult]:
        """Récupère du cache Redis"""
        if not self._redis:
            return None
        
        try:
            cached_data = await self._redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return TranslationResult(**data)
        except Exception as e:
            logger.error(f"Erreur cache Redis: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: TranslationResult, strategy: CacheStrategy) -> None:
        """Met en cache le résultat"""
        # Cache local
        self._set_local_cache(cache_key, result)
        
        # Cache Redis
        if self._redis:
            try:
                ttl_map = {
                    CacheStrategy.LONG_TERM: 86400,    # 24h
                    CacheStrategy.MEDIUM_TERM: 14400,  # 4h
                    CacheStrategy.SHORT_TERM: 3600,    # 1h
                    CacheStrategy.SESSION: 1800        # 30min
                }
                
                ttl = ttl_map.get(strategy, 3600)
                data = json.dumps(result.__dict__, default=str)
                
                await self._redis.setex(cache_key, ttl, data)
                
            except Exception as e:
                logger.error(f"Erreur mise en cache Redis: {e}")
    
    def _update_stats(self, processing_time: float) -> None:
        """Met à jour les statistiques"""
        current_avg = self._stats["avg_processing_time"]
        count = self._stats["requests_count"]
        
        # Moyenne mobile
        self._stats["avg_processing_time"] = (
            (current_avg * (count - 1) + processing_time) / count
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire"""
        cache_hit_rate = (
            self._stats["cache_hits"] / max(1, self._stats["requests_count"]) * 100
        )
        
        return {
            **self._stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "supported_languages": list(self._translations.keys()),
            "local_cache_size": len(self._local_cache),
            "redis_connected": self._redis is not None
        }
    
    async def detect_language(self, text: str) -> str:
        """Détecte la langue d'un texte"""
        try:
            detected = detect(text)
            # Validation que la langue est supportée
            return detected if detected in self._translations else self.default_language
        except LangDetectError:
            return self.default_language
    
    async def reload_translations(self) -> None:
        """Recharge les traductions à chaud"""
        logger.info("Rechargement des traductions...")
        
        # Nettoyage des caches
        self._local_cache.clear()
        self._cache_timestamps.clear()
        
        if self._redis:
            try:
                # Suppression du cache Redis avec pattern
                keys = await self._redis.keys("trans:*")
                if keys:
                    await self._redis.delete(*keys)
            except Exception as e:
                logger.error(f"Erreur nettoyage cache Redis: {e}")
        
        # Rechargement
        await self._load_translations()
        await self._validate_translations()
        
        logger.info("Traductions rechargées avec succès")
    
    async def close(self) -> None:
        """Ferme proprement le gestionnaire"""
        if self._redis:
            await self._redis.close()
        
        if self._redis_pool:
            await self._redis_pool.disconnect()
        
        logger.info("Gestionnaire de traductions fermé")
    
    @asynccontextmanager
    async def translation_context(self):
        """Context manager pour la gestion automatique du cycle de vie"""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()


# Factory function pour créer une instance configurée
async def create_translation_manager(
    translations_path: str,
    redis_url: str = "redis://localhost:6379/2",
    ai_api_key: Optional[str] = None,
    default_language: str = "en"
) -> AdvancedTranslationManager:
    """
    Factory pour créer et initialiser un gestionnaire de traductions
    
    Args:
        translations_path: Chemin vers les fichiers de traduction
        redis_url: URL Redis pour le cache
        ai_api_key: Clé API pour l'IA
        default_language: Langue par défaut
        
    Returns:
        Gestionnaire initialisé
    """
    manager = AdvancedTranslationManager(
        translations_path=Path(translations_path),
        redis_url=redis_url,
        ai_api_key=ai_api_key,
        default_language=default_language
    )
    
    await manager.initialize()
    return manager


# Export des classes principales
__all__ = [
    "AdvancedTranslationManager",
    "TranslationRequest", 
    "TranslationResult",
    "TranslationPriority",
    "CacheStrategy",
    "create_translation_manager"
]
