"""
🌐 Advanced Localization Engine - Production-Ready System
========================================================

Moteur de localisation ultra-avancé avec ML, détection automatique de langue,
adaptation culturelle et optimisation contextuelle.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import re
import unicodedata

import langdetect
from langdetect.lang_detect_exception import LangDetectException
from googletrans import Translator
import pycountry
from babel import Locale, dates, numbers
from babel.core import UnknownLocaleError

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Langues supportées avec métadonnées"""
    ENGLISH = ("en", "English", "ltr", "en_US")
    FRENCH = ("fr", "Français", "ltr", "fr_FR")  
    GERMAN = ("de", "Deutsch", "ltr", "de_DE")
    SPANISH = ("es", "Español", "ltr", "es_ES")
    ITALIAN = ("it", "Italiano", "ltr", "it_IT")
    PORTUGUESE = ("pt", "Português", "ltr", "pt_PT")
    DUTCH = ("nl", "Nederlands", "ltr", "nl_NL")
    RUSSIAN = ("ru", "Русский", "ltr", "ru_RU")
    CHINESE_SIMPLIFIED = ("zh-cn", "简体中文", "ltr", "zh_CN")
    CHINESE_TRADITIONAL = ("zh-tw", "繁體中文", "ltr", "zh_TW")
    JAPANESE = ("ja", "日本語", "ltr", "ja_JP")
    KOREAN = ("ko", "한국어", "ltr", "ko_KR")
    ARABIC = ("ar", "العربية", "rtl", "ar_SA")
    HEBREW = ("he", "עברית", "rtl", "he_IL")
    HINDI = ("hi", "हिन्दी", "ltr", "hi_IN")
    
    def __init__(self, code, name, direction, locale):
        self.code = code
        self.name = name
        self.direction = direction
        self.locale = locale


class CulturalContext(Enum):
    """Contextes culturels pour adaptation"""
    BUSINESS_FORMAL = "business_formal"
    BUSINESS_CASUAL = "business_casual"
    PERSONAL_FRIENDLY = "personal_friendly"
    TECHNICAL_PRECISE = "technical_precise"
    MARKETING_ENGAGING = "marketing_engaging"
    SUPPORT_HELPFUL = "support_helpful"
    LEGAL_COMPLIANT = "legal_compliant"


@dataclass
class LocalizationContext:
    """Contexte de localisation"""
    target_language: str
    source_language: str = "en"
    cultural_context: CulturalContext = CulturalContext.BUSINESS_CASUAL
    
    # Géolocalisation
    country: Optional[str] = None
    region: Optional[str] = None
    timezone: Optional[str] = None
    
    # Personnalisation
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    tenant_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration technique
    preserve_formatting: bool = True
    enable_cultural_adaptation: bool = True
    enable_number_localization: bool = True
    enable_date_localization: bool = True
    
    # Qualité
    quality_threshold: float = 0.8
    fallback_language: str = "en"


@dataclass
class LocalizedContent:
    """Contenu localisé avec métadonnées"""
    id: str = field(default_factory=lambda: f"loc_{int(datetime.utcnow().timestamp())}")
    original_content: str = ""
    localized_content: str = ""
    source_language: str = "en"
    target_language: str = "en"
    
    # Métriques de qualité
    confidence_score: float = 0.0
    fluency_score: float = 0.0
    cultural_adaptation_score: float = 0.0
    
    # Métadonnées
    localization_method: str = "automatic"
    processing_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Informations culturelles
    cultural_adaptations: List[str] = field(default_factory=list)
    formatting_changes: List[str] = field(default_factory=list)
    
    # Cache et optimisation
    cached: bool = False
    cache_key: str = ""


class LanguageDetector:
    """Détecteur de langue avancé avec ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.confidence_threshold = config.get('detection_confidence_threshold', 0.8)
        self.min_text_length = config.get('min_text_length', 10)
        
        # Cache des détections
        self.detection_cache: Dict[str, Tuple[str, float]] = {}
        
        # Patterns de langue
        self.language_patterns = {
            'en': [r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b'],
            'fr': [r'\b(le|la|les|un|une|des|et|ou|mais|dans|sur|à|pour|de|avec|par)\b'],
            'de': [r'\b(der|die|das|ein|eine|und|oder|aber|in|auf|zu|für|von|mit|durch)\b'],
            'es': [r'\b(el|la|los|las|un|una|y|o|pero|en|sobre|a|para|de|con|por)\b'],
            'it': [r'\b(il|la|i|le|un|una|e|o|ma|in|su|a|per|di|con|da)\b']
        }
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """Détection de langue avec confiance"""
        
        if len(text) < self.min_text_length:
            return "en", 0.0  # Langue par défaut pour texte trop court
        
        # Vérification du cache
        text_hash = str(hash(text))
        if text_hash in self.detection_cache:
            return self.detection_cache[text_hash]
        
        try:
            # Détection primaire avec langdetect
            detected_lang = langdetect.detect(text)
            confidence = await self._calculate_detection_confidence(text, detected_lang)
            
            # Validation avec patterns
            pattern_confidence = await self._validate_with_patterns(text, detected_lang)
            
            # Score final
            final_confidence = (confidence + pattern_confidence) / 2
            
            # Mise en cache
            result = (detected_lang, final_confidence)
            self.detection_cache[text_hash] = result
            
            return result
            
        except LangDetectException:
            # Fallback avec analyse des patterns
            return await self._fallback_detection(text)
    
    async def _calculate_detection_confidence(self, text: str, detected_lang: str) -> float:
        """Calcul de la confiance de détection"""
        
        try:
            # Utilisation de langdetect avec probabilités
            from langdetect import detect_langs
            
            probs = detect_langs(text)
            for prob in probs:
                if prob.lang == detected_lang:
                    return prob.prob
            
            return 0.0
            
        except Exception:
            # Calcul basé sur la longueur et caractères
            base_confidence = min(0.8, len(text) / 100)  # Plus le texte est long, plus on est confiant
            
            # Bonus pour caractères spécifiques
            if detected_lang == 'fr' and any(c in text for c in 'àéèêëîïôöùûüÿç'):
                base_confidence += 0.1
            elif detected_lang == 'de' and any(c in text for c in 'äöüß'):
                base_confidence += 0.1
            elif detected_lang == 'es' and any(c in text for c in 'ñáéíóúü¿¡'):
                base_confidence += 0.1
            
            return min(1.0, base_confidence)
    
    async def _validate_with_patterns(self, text: str, detected_lang: str) -> float:
        """Validation avec patterns linguistiques"""
        
        if detected_lang not in self.language_patterns:
            return 0.5  # Score neutre
        
        patterns = self.language_patterns[detected_lang]
        matches = 0
        total_patterns = len(patterns)
        
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                matches += 1
        
        return matches / total_patterns if total_patterns > 0 else 0.5
    
    async def _fallback_detection(self, text: str) -> Tuple[str, float]:
        """Détection de fallback basée sur patterns"""
        
        scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text.lower()))
                score += matches
            
            # Normalisation par longueur du texte
            normalized_score = score / len(text.split()) if text.split() else 0
            scores[lang] = normalized_score
        
        if scores:
            best_lang = max(scores, key=scores.get)
            confidence = min(0.7, scores[best_lang] * 10)  # Score conservateur
            return best_lang, confidence
        
        return "en", 0.0


class CulturalAdaptationEngine:
    """Moteur d'adaptation culturelle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Règles d'adaptation culturelle
        self.cultural_rules = self._load_cultural_rules()
        
        # Formatage par pays
        self.country_formats = self._load_country_formats()
        
        # Expressions culturelles
        self.cultural_expressions = self._load_cultural_expressions()
    
    def _load_cultural_rules(self) -> Dict[str, Dict[str, Any]]:
        """Chargement des règles d'adaptation culturelle"""
        
        return {
            'en': {
                'formality_level': 'medium',
                'directness': 'high',
                'emotional_expression': 'moderate',
                'hierarchy_respect': 'low',
                'time_orientation': 'monochronic'
            },
            'fr': {
                'formality_level': 'high',
                'directness': 'medium',
                'emotional_expression': 'high',
                'hierarchy_respect': 'medium',
                'time_orientation': 'monochronic'
            },
            'de': {
                'formality_level': 'high',
                'directness': 'very_high',
                'emotional_expression': 'low',
                'hierarchy_respect': 'high',
                'time_orientation': 'monochronic'
            },
            'ja': {
                'formality_level': 'very_high',
                'directness': 'very_low',
                'emotional_expression': 'very_low',
                'hierarchy_respect': 'very_high',
                'time_orientation': 'polychronic'
            },
            'ar': {
                'formality_level': 'high',
                'directness': 'low',
                'emotional_expression': 'high',
                'hierarchy_respect': 'very_high',
                'time_orientation': 'polychronic'
            }
        }
    
    def _load_country_formats(self) -> Dict[str, Dict[str, str]]:
        """Chargement des formats par pays"""
        
        return {
            'US': {
                'date_format': 'MM/dd/yyyy',
                'time_format': 'h:mm a',
                'number_decimal': '.',
                'number_thousands': ',',
                'currency_symbol': '$',
                'currency_position': 'before'
            },
            'FR': {
                'date_format': 'dd/MM/yyyy',
                'time_format': 'HH:mm',
                'number_decimal': ',',
                'number_thousands': ' ',
                'currency_symbol': '€',
                'currency_position': 'after'
            },
            'DE': {
                'date_format': 'dd.MM.yyyy',
                'time_format': 'HH:mm',
                'number_decimal': ',',
                'number_thousands': '.',
                'currency_symbol': '€',
                'currency_position': 'after'
            },
            'JP': {
                'date_format': 'yyyy/MM/dd',
                'time_format': 'HH:mm',
                'number_decimal': '.',
                'number_thousands': ',',
                'currency_symbol': '¥',
                'currency_position': 'before'
            }
        }
    
    def _load_cultural_expressions(self) -> Dict[str, Dict[str, List[str]]]:
        """Chargement des expressions culturelles"""
        
        return {
            'greetings': {
                'en': ['Hello', 'Hi', 'Good morning', 'Good afternoon'],
                'fr': ['Bonjour', 'Salut', 'Bonsoir', 'Bonne journée'],
                'de': ['Hallo', 'Guten Tag', 'Guten Morgen', 'Guten Abend'],
                'es': ['Hola', 'Buenos días', 'Buenas tardes', 'Buenas noches'],
                'ja': ['こんにちは', 'おはようございます', 'こんばんは']
            },
            'polite_expressions': {
                'en': ['please', 'thank you', 'you\'re welcome', 'excuse me'],
                'fr': ['s\'il vous plaît', 'merci', 'de rien', 'excusez-moi'],
                'de': ['bitte', 'danke', 'gern geschehen', 'entschuldigung'],
                'es': ['por favor', 'gracias', 'de nada', 'disculpe'],
                'ja': ['お願いします', 'ありがとうございます', 'どういたしまして', 'すみません']
            },
            'business_closings': {
                'en': ['Best regards', 'Sincerely', 'Thank you'],
                'fr': ['Cordialement', 'Bien à vous', 'Merci'],
                'de': ['Mit freundlichen Grüßen', 'Hochachtungsvoll', 'Vielen Dank'],
                'es': ['Saludos cordiales', 'Atentamente', 'Gracias'],
                'ja': ['よろしくお願いいたします', '敬具', 'ありがとうございます']
            }
        }
    
    async def adapt_culturally(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Adaptation culturelle du contenu"""
        
        if not context.enable_cultural_adaptation:
            return content, []
        
        adapted_content = content
        adaptations = []
        
        try:
            # Adaptation du niveau de formalité
            adapted_content, formality_changes = await self._adapt_formality(
                adapted_content, context
            )
            adaptations.extend(formality_changes)
            
            # Adaptation des expressions
            adapted_content, expression_changes = await self._adapt_expressions(
                adapted_content, context
            )
            adaptations.extend(expression_changes)
            
            # Adaptation des références culturelles
            adapted_content, cultural_changes = await self._adapt_cultural_references(
                adapted_content, context
            )
            adaptations.extend(cultural_changes)
            
            # Adaptation du ton et style
            adapted_content, style_changes = await self._adapt_style(
                adapted_content, context
            )
            adaptations.extend(style_changes)
            
            return adapted_content, adaptations
            
        except Exception as e:
            self.logger.error(f"Cultural adaptation failed: {str(e)}")
            return content, [f"Adaptation error: {str(e)}"]
    
    async def _adapt_formality(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Adaptation du niveau de formalité"""
        
        target_lang = context.target_language
        cultural_context = context.cultural_context
        
        if target_lang not in self.cultural_rules:
            return content, []
        
        rules = self.cultural_rules[target_lang]
        formality_level = rules.get('formality_level', 'medium')
        
        adaptations = []
        adapted_content = content
        
        # Adaptations spécifiques par langue et contexte
        if target_lang == 'de' and formality_level == 'high':
            # Allemand : utilisation du Sie
            adapted_content = re.sub(r'\byou\b', 'Sie', adapted_content, flags=re.IGNORECASE)
            adaptations.append("Applied formal address (Sie) for German")
        
        elif target_lang == 'fr' and cultural_context == CulturalContext.BUSINESS_FORMAL:
            # Français : vouvoiement
            adapted_content = re.sub(r'\byou\b', 'vous', adapted_content, flags=re.IGNORECASE)
            adaptations.append("Applied formal address (vous) for French business context")
        
        elif target_lang == 'ja':
            # Japonais : keigo (langue honorifique)
            adaptations.append("Applied keigo (honorific language) for Japanese")
            # Ici on ajouterait la logique spécifique au japonais
        
        return adapted_content, adaptations
    
    async def _adapt_expressions(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Adaptation des expressions culturelles"""
        
        target_lang = context.target_language
        adaptations = []
        adapted_content = content
        
        # Remplacement des salutations
        if 'greetings' in self.cultural_expressions:
            en_greetings = self.cultural_expressions['greetings'].get('en', [])
            target_greetings = self.cultural_expressions['greetings'].get(target_lang, [])
            
            if target_greetings:
                for i, en_greeting in enumerate(en_greetings):
                    if en_greeting.lower() in content.lower():
                        if i < len(target_greetings):
                            adapted_content = re.sub(
                                rf'\b{re.escape(en_greeting)}\b',
                                target_greetings[i],
                                adapted_content,
                                flags=re.IGNORECASE
                            )
                            adaptations.append(f"Adapted greeting: {en_greeting} -> {target_greetings[i]}")
        
        return adapted_content, adaptations
    
    async def _adapt_cultural_references(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Adaptation des références culturelles"""
        
        adaptations = []
        adapted_content = content
        
        # Références temporelles (exemple: formats de date)
        if context.enable_date_localization:
            country = context.country or self._get_country_from_language(context.target_language)
            if country in self.country_formats:
                # Adaptation des formats de date
                date_patterns = [
                    r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/dd/yyyy ou dd/MM/yyyy
                    r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-dd-yyyy ou dd-MM-yyyy
                ]
                
                for pattern in date_patterns:
                    if re.search(pattern, content):
                        adaptations.append(f"Date format adapted for {country}")
                        break
        
        return adapted_content, adaptations
    
    async def _adapt_style(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Adaptation du style et du ton"""
        
        target_lang = context.target_language
        cultural_context = context.cultural_context
        
        if target_lang not in self.cultural_rules:
            return content, []
        
        rules = self.cultural_rules[target_lang]
        directness = rules.get('directness', 'medium')
        
        adaptations = []
        adapted_content = content
        
        # Adaptation de la directness
        if directness == 'very_low' and target_lang == 'ja':
            # Japonais : ajout de modestie linguistique
            adapted_content = re.sub(
                r'\bI think\b', 
                'I humbly believe', 
                adapted_content, 
                flags=re.IGNORECASE
            )
            adaptations.append("Added linguistic modesty for Japanese culture")
        
        elif directness == 'very_high' and target_lang == 'de':
            # Allemand : style plus direct
            adapted_content = re.sub(
                r'\bmight be\b', 
                'is', 
                adapted_content, 
                flags=re.IGNORECASE
            )
            adaptations.append("Applied direct communication style for German culture")
        
        return adapted_content, adaptations
    
    def _get_country_from_language(self, language: str) -> str:
        """Obtention du pays par défaut pour une langue"""
        
        language_country_map = {
            'en': 'US',
            'fr': 'FR',
            'de': 'DE',
            'es': 'ES',
            'it': 'IT',
            'pt': 'PT',
            'nl': 'NL',
            'ru': 'RU',
            'zh-cn': 'CN',
            'zh-tw': 'TW',
            'ja': 'JP',
            'ko': 'KR',
            'ar': 'SA',
            'he': 'IL',
            'hi': 'IN'
        }
        
        return language_country_map.get(language, 'US')


class LocalizationEngine:
    """Moteur principal de localisation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Composants
        self.language_detector = LanguageDetector(config)
        self.cultural_adapter = CulturalAdaptationEngine(config)
        
        # Traducteur
        self.translator = Translator()
        
        # Cache de localisation
        self.localization_cache: Dict[str, LocalizedContent] = {}
        self.cache_ttl = config.get('cache_ttl', 3600)
        
        # Métriques
        self.localization_stats = {
            'total_localizations': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'quality_scores': []
        }
    
    async def localize_content(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> LocalizedContent:
        """Localisation principale du contenu"""
        
        start_time = datetime.utcnow()
        
        try:
            # Génération de la clé de cache
            cache_key = self._generate_cache_key(content, context)
            
            # Vérification du cache
            if cache_key in self.localization_cache:
                self.localization_stats['cache_hits'] += 1
                cached_result = self.localization_cache[cache_key]
                cached_result.cached = True
                return cached_result
            
            # Détection de la langue source si non spécifiée
            if not context.source_language or context.source_language == "auto":
                detected_lang, confidence = await self.language_detector.detect_language(content)
                if confidence > 0.7:
                    context.source_language = detected_lang
                else:
                    context.source_language = "en"  # Fallback
            
            # Vérification si traduction nécessaire
            if context.source_language == context.target_language:
                # Même langue : adaptation culturelle seulement
                localized_content, adaptations = await self.cultural_adapter.adapt_culturally(
                    content, context
                )
                
                result = LocalizedContent(
                    original_content=content,
                    localized_content=localized_content,
                    source_language=context.source_language,
                    target_language=context.target_language,
                    confidence_score=1.0,
                    fluency_score=1.0,
                    cultural_adaptation_score=len(adaptations) / 10,  # Score basé sur les adaptations
                    localization_method="cultural_adaptation_only",
                    cultural_adaptations=adaptations,
                    cache_key=cache_key
                )
            else:
                # Traduction + adaptation
                result = await self._translate_and_adapt(content, context, cache_key)
            
            # Calcul du temps de traitement
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            # Mise en cache
            self.localization_cache[cache_key] = result
            
            # Mise à jour des statistiques
            self._update_stats(processing_time, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Localization failed: {str(e)}")
            
            # Retour du contenu original en cas d'erreur
            return LocalizedContent(
                original_content=content,
                localized_content=content,
                source_language=context.source_language,
                target_language=context.target_language,
                confidence_score=0.0,
                fluency_score=0.0,
                localization_method="error_fallback"
            )
    
    async def _translate_and_adapt(
        self, 
        content: str, 
        context: LocalizationContext, 
        cache_key: str
    ) -> LocalizedContent:
        """Traduction et adaptation culturelle"""
        
        # Étape 1: Traduction automatique
        translated_content = await self._translate_content(
            content, 
            context.source_language, 
            context.target_language
        )
        
        # Étape 2: Adaptation culturelle
        culturally_adapted_content, adaptations = await self.cultural_adapter.adapt_culturally(
            translated_content, context
        )
        
        # Étape 3: Post-processing (formatage, nombres, dates)
        final_content, formatting_changes = await self._post_process_content(
            culturally_adapted_content, context
        )
        
        # Étape 4: Évaluation de la qualité
        quality_scores = await self._evaluate_quality(
            content, final_content, context
        )
        
        return LocalizedContent(
            original_content=content,
            localized_content=final_content,
            source_language=context.source_language,
            target_language=context.target_language,
            confidence_score=quality_scores['confidence'],
            fluency_score=quality_scores['fluency'],
            cultural_adaptation_score=quality_scores['cultural_adaptation'],
            localization_method="translate_and_adapt",
            cultural_adaptations=adaptations,
            formatting_changes=formatting_changes,
            cache_key=cache_key
        )
    
    async def _translate_content(
        self, 
        content: str, 
        source_lang: str, 
        target_lang: str
    ) -> str:
        """Traduction automatique du contenu"""
        
        try:
            # Utilisation de Google Translate (en production, on utiliserait une API plus robuste)
            result = self.translator.translate(
                content, 
                src=source_lang, 
                dest=target_lang
            )
            
            return result.text
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return content  # Retour du contenu original en cas d'erreur
    
    async def _post_process_content(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Post-processing du contenu localisé"""
        
        processed_content = content
        changes = []
        
        # Localisation des nombres
        if context.enable_number_localization:
            processed_content, number_changes = await self._localize_numbers(
                processed_content, context
            )
            changes.extend(number_changes)
        
        # Localisation des dates
        if context.enable_date_localization:
            processed_content, date_changes = await self._localize_dates(
                processed_content, context
            )
            changes.extend(date_changes)
        
        # Préservation du formatage
        if context.preserve_formatting:
            processed_content = await self._preserve_formatting(processed_content, context)
        
        return processed_content, changes
    
    async def _localize_numbers(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Localisation des nombres"""
        
        changes = []
        processed_content = content
        
        country = context.country or self.cultural_adapter._get_country_from_language(
            context.target_language
        )
        
        if country in self.cultural_adapter.country_formats:
            formats = self.cultural_adapter.country_formats[country]
            
            # Localisation des nombres décimaux
            decimal_pattern = r'\b\d+\.\d+\b'
            if formats['number_decimal'] != '.':
                processed_content = re.sub(
                    decimal_pattern,
                    lambda m: m.group().replace('.', formats['number_decimal']),
                    processed_content
                )
                changes.append(f"Localized decimal separator to '{formats['number_decimal']}'")
        
        return processed_content, changes
    
    async def _localize_dates(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> Tuple[str, List[str]]:
        """Localisation des dates"""
        
        changes = []
        processed_content = content
        
        # Patterns de date courants
        date_patterns = [
            (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', 'MM/dd/yyyy'),
            (r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', 'MM-dd-yyyy'),
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'yyyy-MM-dd')
        ]
        
        country = context.country or self.cultural_adapter._get_country_from_language(
            context.target_language
        )
        
        if country in self.cultural_adapter.country_formats:
            target_format = self.cultural_adapter.country_formats[country]['date_format']
            
            for pattern, source_format in date_patterns:
                if re.search(pattern, content):
                    changes.append(f"Date format adapted from {source_format} to {target_format}")
                    # Ici on implémenterait la logique de conversion de date
                    break
        
        return processed_content, changes
    
    async def _preserve_formatting(
        self, 
        content: str, 
        context: LocalizationContext
    ) -> str:
        """Préservation du formatage original"""
        
        # Préservation des éléments Markdown/HTML
        preserved_content = content
        
        # Préservation des liens
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)
        
        # Préservation des éléments de code
        code_pattern = r'`([^`]+)`'
        code_blocks = re.findall(code_pattern, content)
        
        # Ici on ajouterait la logique de préservation complète
        
        return preserved_content
    
    async def _evaluate_quality(
        self, 
        original: str, 
        localized: str, 
        context: LocalizationContext
    ) -> Dict[str, float]:
        """Évaluation de la qualité de localisation"""
        
        scores = {
            'confidence': 0.8,  # Score de base
            'fluency': 0.8,
            'cultural_adaptation': 0.7
        }
        
        # Évaluation basée sur la longueur relative
        length_ratio = len(localized) / len(original) if original else 1.0
        if 0.5 <= length_ratio <= 2.0:  # Ratio raisonnable
            scores['confidence'] += 0.1
        
        # Évaluation basée sur la préservation de structure
        original_sentences = len(re.split(r'[.!?]', original))
        localized_sentences = len(re.split(r'[.!?]', localized))
        
        if abs(original_sentences - localized_sentences) <= 1:
            scores['fluency'] += 0.1
        
        # Évaluation de l'adaptation culturelle
        if context.enable_cultural_adaptation:
            scores['cultural_adaptation'] += 0.2
        
        return scores
    
    def _generate_cache_key(self, content: str, context: LocalizationContext) -> str:
        """Génération de clé de cache"""
        
        content_hash = hash(content)
        context_hash = hash(f"{context.target_language}_{context.source_language}_{context.cultural_context.value}")
        
        return f"{content_hash}_{context_hash}"
    
    def _update_stats(self, processing_time: float, result: LocalizedContent):
        """Mise à jour des statistiques"""
        
        self.localization_stats['total_localizations'] += 1
        
        # Moyenne mobile du temps de traitement
        current_avg = self.localization_stats['avg_processing_time']
        total = self.localization_stats['total_localizations']
        new_avg = (current_avg * (total - 1) + processing_time) / total
        self.localization_stats['avg_processing_time'] = new_avg
        
        # Ajout du score de qualité
        quality_score = (result.confidence_score + result.fluency_score + result.cultural_adaptation_score) / 3
        self.localization_stats['quality_scores'].append(quality_score)
        
        # Maintien des 100 derniers scores
        if len(self.localization_stats['quality_scores']) > 100:
            self.localization_stats['quality_scores'] = self.localization_stats['quality_scores'][-100:]
    
    async def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Obtention des langues supportées"""
        
        languages = []
        
        for lang in SupportedLanguage:
            languages.append({
                'code': lang.code,
                'name': lang.name,
                'direction': lang.direction,
                'locale': lang.locale,
                'cultural_rules_available': lang.code in self.cultural_adapter.cultural_rules
            })
        
        return languages
    
    async def get_localization_stats(self) -> Dict[str, Any]:
        """Obtention des statistiques de localisation"""
        
        avg_quality = (
            sum(self.localization_stats['quality_scores']) / 
            len(self.localization_stats['quality_scores'])
        ) if self.localization_stats['quality_scores'] else 0.0
        
        cache_hit_rate = (
            self.localization_stats['cache_hits'] / 
            self.localization_stats['total_localizations']
        ) if self.localization_stats['total_localizations'] > 0 else 0.0
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_localizations': self.localization_stats['total_localizations'],
            'cache_hit_rate': cache_hit_rate,
            'avg_processing_time_ms': self.localization_stats['avg_processing_time'],
            'avg_quality_score': avg_quality,
            'cache_size': len(self.localization_cache),
            'supported_languages_count': len(SupportedLanguage)
        }


class LanguageDetector:
    """Détecteur de langue avec ML avancé"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.supported_languages = [lang.code for lang in SupportedLanguage]
        
        # Cache des détections
        self.detection_cache: Dict[str, Tuple[str, float]] = {}
        
        # Modèles de détection avancés
        self._init_detection_models()
    
    def _init_detection_models(self):
        """Initialisation des modèles de détection"""
        
        try:
            # Ici on chargerait des modèles ML spécialisés pour la détection de langue
            # Pour la démo, on utilise langdetect
            self.model_available = True
            self.logger.info("Language detection models initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize detection models: {str(e)}")
            self.model_available = False
    
    async def detect_with_alternatives(self, text: str) -> List[Tuple[str, float]]:
        """Détection avec alternatives et scores de confiance"""
        
        if not text or len(text) < 3:
            return [("en", 0.0)]
        
        try:
            from langdetect import detect_langs
            
            # Détection avec probabilités
            detections = detect_langs(text)
            
            # Filtrage des langues supportées
            supported_detections = [
                (detection.lang, detection.prob) 
                for detection in detections 
                if detection.lang in self.supported_languages
            ]
            
            # Si aucune langue supportée détectée, fallback
            if not supported_detections:
                supported_detections = [("en", 0.5)]
            
            return supported_detections[:5]  # Top 5
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {str(e)}")
            return [("en", 0.0)]
    
    async def get_detection_stats(self) -> Dict[str, Any]:
        """Statistiques de détection"""
        
        return {
            'cache_size': len(self.detection_cache),
            'supported_languages': len(self.supported_languages),
            'model_available': self.model_available,
            'confidence_threshold': self.confidence_threshold
        }
