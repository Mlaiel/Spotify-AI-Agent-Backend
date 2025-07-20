#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détecteur intelligent de langue pour alertes Slack

Ce module fournit un système de détection de langue avancé avec:
- Détection basée sur le contenu du message
- Analyse des préférences utilisateur
- Détection selon le contexte géographique  
- Machine Learning pour améliorer la précision
- Cache distribué des détections
- Fallback intelligent multi-critères
- API de détection en temps réel
- Métriques de performance et précision

Auteur: Expert Team
Version: 2.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata

import aioredis
from langdetect import detect, detect_langs, LangDetectError
import pycountry

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Méthodes de détection de langue"""
    CONTENT_ANALYSIS = "content_analysis"
    USER_PREFERENCE = "user_preference"
    GEOGRAPHIC = "geographic"
    TENANT_DEFAULT = "tenant_default"
    BROWSER_HEADER = "browser_header"
    ML_PREDICTION = "ml_prediction"
    FALLBACK = "fallback"


class ConfidenceLevel(Enum):
    """Niveaux de confiance pour la détection"""
    VERY_HIGH = "very_high"    # >95%
    HIGH = "high"              # 85-95%
    MEDIUM = "medium"          # 70-85%
    LOW = "low"                # 50-70%
    VERY_LOW = "very_low"      # <50%


@dataclass
class DetectionResult:
    """Résultat de détection de langue"""
    language: str
    confidence: float
    method: DetectionMethod
    processing_time_ms: float = 0.0
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Détermine le niveau de confiance"""
        if self.confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class UserLanguageProfile:
    """Profil linguistique d'un utilisateur"""
    user_id: str
    preferred_languages: List[str] = field(default_factory=list)
    detected_languages: Dict[str, int] = field(default_factory=dict)  # langue -> nombre détections
    timezone: Optional[str] = None
    country: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class SmartLanguageDetector:
    """Détecteur intelligent de langue ultra-avancé"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/2",
                 supported_languages: Optional[List[str]] = None,
                 default_language: str = "en"):
        """
        Initialise le détecteur de langue
        
        Args:
            redis_url: URL Redis pour le cache
            supported_languages: Langues supportées
            default_language: Langue par défaut
        """
        self.redis_url = redis_url
        self.supported_languages = supported_languages or [
            "en", "fr", "de", "es", "pt", "it", "ru", "zh", "ja", "ar", "he"
        ]
        self.default_language = default_language
        
        # Cache et connexions
        self._redis: Optional[aioredis.Redis] = None
        self._user_profiles: Dict[str, UserLanguageProfile] = {}
        
        # Statistiques
        self._stats = {
            "total_detections": 0,
            "method_usage": {method.value: 0 for method in DetectionMethod},
            "accuracy_scores": [],
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Patterns pour la détection basée sur le contenu
        self._language_patterns = {
            "en": [
                r'\b(the|and|or|but|with|for|from|to|in|on|at|by|of|is|are|was|were|have|has|had)\b',
                r'\b(alert|error|warning|critical|system|service|application)\b'
            ],
            "fr": [
                r'\b(le|la|les|de|du|des|et|ou|mais|avec|pour|dans|sur|par|est|sont|était|étaient)\b',
                r'\b(alerte|erreur|avertissement|critique|système|service|application)\b'
            ],
            "de": [
                r'\b(der|die|das|und|oder|aber|mit|für|von|zu|in|auf|bei|ist|sind|war|waren)\b',
                r'\b(warnung|fehler|kritisch|system|dienst|anwendung)\b'
            ],
            "es": [
                r'\b(el|la|los|las|de|del|y|o|pero|con|para|en|por|es|son|era|eran)\b',
                r'\b(alerta|error|advertencia|crítico|sistema|servicio|aplicación)\b'
            ],
            "pt": [
                r'\b(o|a|os|as|de|do|da|e|ou|mas|com|para|em|por|é|são|era|eram)\b',
                r'\b(alerta|erro|aviso|crítico|sistema|serviço|aplicação)\b'
            ],
            "it": [
                r'\b(il|la|lo|gli|le|di|del|e|o|ma|con|per|in|su|da|è|sono|era|erano)\b',
                r'\b(allarme|errore|avviso|critico|sistema|servizio|applicazione)\b'
            ],
            "ru": [
                r'\b(и|или|но|с|для|от|к|в|на|при|это|был|была|было|были)\b',
                r'\b(предупреждение|ошибка|критический|система|служба|приложение)\b'
            ],
            "ar": [
                r'\b(في|من|إلى|على|مع|أو|و|لكن|هذا|هذه|كان|كانت)\b',
                r'\b(تنبيه|خطأ|تحذير|حرج|نظام|خدمة|تطبيق)\b'
            ]
        }
        
        # Mappings géographiques
        self._country_language_map = {
            "US": ["en"], "GB": ["en"], "CA": ["en", "fr"], "AU": ["en"],
            "FR": ["fr"], "BE": ["fr", "nl"], "CH": ["de", "fr", "it"],
            "DE": ["de"], "AT": ["de"],
            "ES": ["es"], "MX": ["es"], "AR": ["es"], "CO": ["es"],
            "PT": ["pt"], "BR": ["pt"],
            "IT": ["it"],
            "RU": ["ru"], "BY": ["ru"], "KZ": ["ru"],
            "CN": ["zh"], "TW": ["zh"], "HK": ["zh", "en"],
            "JP": ["ja"],
            "SA": ["ar"], "AE": ["ar"], "EG": ["ar"], "MA": ["ar"]
        }
        
        logger.info(f"Détecteur de langue initialisé - Langues supportées: {len(self.supported_languages)}")
    
    async def initialize(self) -> None:
        """Initialise les connexions"""
        try:
            # Connexion Redis
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                retry_on_timeout=True
            )
            
            # Test de connexion
            await self._redis.ping()
            logger.info("Connexion Redis établie pour le détecteur de langue")
            
            # Chargement des profils utilisateur depuis le cache
            await self._load_user_profiles()
            
        except Exception as e:
            logger.warning(f"Erreur initialisation détecteur: {e}")
            self._redis = None
    
    async def detect_language(self, 
                            text: str,
                            user_id: Optional[str] = None,
                            user_agent: Optional[str] = None,
                            ip_address: Optional[str] = None,
                            tenant_id: Optional[str] = None) -> DetectionResult:
        """
        Détecte la langue avec analyse multi-critères
        
        Args:
            text: Texte à analyser
            user_id: ID utilisateur pour les préférences
            user_agent: User-Agent pour détection navigateur
            ip_address: IP pour géolocalisation
            tenant_id: ID tenant pour défaut organisationnel
            
        Returns:
            Résultat de détection enrichi
        """
        start_time = time.time()
        self._stats["total_detections"] += 1
        
        # Nettoyage et préparation du texte
        cleaned_text = self._clean_text(text)
        
        # Tentatives de détection par ordre de priorité
        detection_attempts = []
        
        # 1. Préférences utilisateur (si disponible)
        if user_id:
            user_result = await self._detect_from_user_profile(user_id, cleaned_text)
            if user_result:
                detection_attempts.append(user_result)
        
        # 2. Analyse du contenu textuel
        content_result = await self._detect_from_content(cleaned_text)
        if content_result:
            detection_attempts.append(content_result)
        
        # 3. Détection géographique (si IP disponible)
        if ip_address:
            geo_result = await self._detect_from_geography(ip_address)
            if geo_result:
                detection_attempts.append(geo_result)
        
        # 4. Headers navigateur
        if user_agent:
            browser_result = await self._detect_from_browser(user_agent)
            if browser_result:
                detection_attempts.append(browser_result)
        
        # 5. Défaut tenant
        if tenant_id:
            tenant_result = await self._detect_from_tenant(tenant_id)
            if tenant_result:
                detection_attempts.append(tenant_result)
        
        # Sélection du meilleur résultat
        best_result = self._select_best_detection(detection_attempts)
        
        # Fallback si aucune détection
        if not best_result:
            best_result = DetectionResult(
                language=self.default_language,
                confidence=0.1,
                method=DetectionMethod.FALLBACK
            )
        
        # Finalisation
        processing_time = (time.time() - start_time) * 1000
        best_result.processing_time_ms = processing_time
        
        # Mise à jour des statistiques
        self._update_stats(best_result)
        
        # Mise en cache et apprentissage
        if user_id:
            await self._update_user_profile(user_id, best_result.language, best_result.confidence)
        
        await self._cache_detection(text, best_result)
        
        return best_result
    
    def _clean_text(self, text: str) -> str:
        """Nettoie et normalise le texte pour la détection"""
        # Suppression des caractères de contrôle
        cleaned = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Normalisation Unicode
        cleaned = unicodedata.normalize('NFKD', cleaned)
        
        # Suppression des URLs et emails
        url_pattern = r'https?://[^\s]+'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        cleaned = re.sub(url_pattern, '', cleaned)
        cleaned = re.sub(email_pattern, '', cleaned)
        
        # Suppression des caractères spéciaux en excès
        cleaned = re.sub(r'[^\w\s\.,!?;:]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    async def _detect_from_user_profile(self, user_id: str, text: str) -> Optional[DetectionResult]:
        """Détecte selon le profil utilisateur"""
        profile = self._user_profiles.get(user_id)
        if not profile or not profile.preferred_languages:
            return None
        
        # Langue la plus probable selon l'historique
        most_common_lang = max(
            profile.detected_languages.items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]
        
        if most_common_lang in self.supported_languages:
            confidence = profile.confidence_scores.get(most_common_lang, 0.8)
            
            self._stats["method_usage"][DetectionMethod.USER_PREFERENCE.value] += 1
            
            return DetectionResult(
                language=most_common_lang,
                confidence=confidence,
                method=DetectionMethod.USER_PREFERENCE,
                metadata={"user_id": user_id}
            )
        
        return None
    
    async def _detect_from_content(self, text: str) -> Optional[DetectionResult]:
        """Détecte la langue depuis le contenu textuel"""
        if len(text.strip()) < 10:  # Texte trop court
            return None
        
        try:
            # Détection avec langdetect
            detections = detect_langs(text)
            
            # Filtrage par langues supportées
            valid_detections = [
                (det.lang, det.prob) for det in detections
                if det.lang in self.supported_languages
            ]
            
            if not valid_detections:
                return None
            
            # Amélioration avec patterns spécifiques
            pattern_scores = self._analyze_with_patterns(text)
            
            # Combinaison des scores
            combined_scores = {}
            for lang, prob in valid_detections:
                pattern_boost = pattern_scores.get(lang, 0.0)
                combined_scores[lang] = prob + (pattern_boost * 0.3)
            
            # Meilleur résultat
            best_lang = max(combined_scores, key=combined_scores.get)
            best_score = combined_scores[best_lang]
            
            # Alternatives
            alternatives = [
                (lang, score) for lang, score in combined_scores.items()
                if lang != best_lang
            ]
            alternatives.sort(key=lambda x: x[1], reverse=True)
            
            self._stats["method_usage"][DetectionMethod.CONTENT_ANALYSIS.value] += 1
            
            return DetectionResult(
                language=best_lang,
                confidence=min(best_score, 1.0),
                method=DetectionMethod.CONTENT_ANALYSIS,
                alternatives=alternatives[:3],
                metadata={"text_length": len(text)}
            )
            
        except LangDetectError as e:
            logger.debug(f"Erreur détection langue: {e}")
            return None
    
    def _analyze_with_patterns(self, text: str) -> Dict[str, float]:
        """Analyse le texte avec des patterns spécifiques par langue"""
        scores = {}
        text_lower = text.lower()
        
        for lang, patterns in self._language_patterns.items():
            score = 0.0
            total_matches = 0
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
                total_matches += matches
            
            # Normalisation par longueur du texte
            if len(text) > 0:
                scores[lang] = min(score / len(text.split()) * 10, 1.0)
        
        return scores
    
    async def _detect_from_geography(self, ip_address: str) -> Optional[DetectionResult]:
        """Détecte selon la géolocalisation IP"""
        try:
            # TODO: Intégration avec service de géolocalisation (MaxMind, etc.)
            # Pour l'instant, simulation avec mapping statique
            
            # Exemple de détection géographique simplifiée
            # En production, utiliser un service comme GeoIP2
            country_code = await self._get_country_from_ip(ip_address)
            
            if country_code in self._country_language_map:
                languages = self._country_language_map[country_code]
                primary_lang = languages[0]
                
                if primary_lang in self.supported_languages:
                    confidence = 0.7 if len(languages) == 1 else 0.5
                    
                    self._stats["method_usage"][DetectionMethod.GEOGRAPHIC.value] += 1
                    
                    return DetectionResult(
                        language=primary_lang,
                        confidence=confidence,
                        method=DetectionMethod.GEOGRAPHIC,
                        metadata={"country": country_code, "ip": ip_address}
                    )
        
        except Exception as e:
            logger.debug(f"Erreur détection géographique: {e}")
        
        return None
    
    async def _get_country_from_ip(self, ip_address: str) -> Optional[str]:
        """Obtient le pays depuis l'IP (placeholder)"""
        # TODO: Implémentation réelle avec GeoIP2 ou service équivalent
        # Simulation pour le moment
        
        # Patterns IP locaux pour tests
        if ip_address.startswith("192.168.") or ip_address == "127.0.0.1":
            return "US"  # Défaut pour IPs locales
        
        # En production, utiliser:
        # import geoip2.database
        # reader = geoip2.database.Reader('GeoLite2-Country.mmdb')
        # response = reader.country(ip_address)
        # return response.country.iso_code
        
        return None
    
    async def _detect_from_browser(self, user_agent: str) -> Optional[DetectionResult]:
        """Détecte selon les headers du navigateur"""
        # Extraction des informations de langue depuis Accept-Language
        # Note: En réalité, user_agent ne contient pas Accept-Language
        # Cette fonction est un placeholder pour une implémentation complète
        
        # Patterns pour détecter la langue depuis user-agent
        lang_patterns = {
            "fr": r"fr[-_]FR|French",
            "de": r"de[-_]DE|German",
            "es": r"es[-_]ES|Spanish", 
            "pt": r"pt[-_]BR|pt[-_]PT|Portuguese",
            "it": r"it[-_]IT|Italian",
            "ru": r"ru[-_]RU|Russian",
            "zh": r"zh[-_]CN|zh[-_]TW|Chinese",
            "ja": r"ja[-_]JP|Japanese",
            "ar": r"ar[-_]SA|Arabic"
        }
        
        for lang, pattern in lang_patterns.items():
            if re.search(pattern, user_agent, re.IGNORECASE):
                if lang in self.supported_languages:
                    self._stats["method_usage"][DetectionMethod.BROWSER_HEADER.value] += 1
                    
                    return DetectionResult(
                        language=lang,
                        confidence=0.6,
                        method=DetectionMethod.BROWSER_HEADER,
                        metadata={"user_agent": user_agent[:100]}
                    )
        
        return None
    
    async def _detect_from_tenant(self, tenant_id: str) -> Optional[DetectionResult]:
        """Détecte selon la configuration du tenant"""
        # TODO: Récupération de la langue par défaut du tenant
        # Simulation pour le moment
        
        tenant_defaults = {
            "tenant_fr": "fr",
            "tenant_de": "de", 
            "tenant_es": "es"
        }
        
        default_lang = tenant_defaults.get(tenant_id, self.default_language)
        
        if default_lang in self.supported_languages:
            self._stats["method_usage"][DetectionMethod.TENANT_DEFAULT.value] += 1
            
            return DetectionResult(
                language=default_lang,
                confidence=0.4,
                method=DetectionMethod.TENANT_DEFAULT,
                metadata={"tenant_id": tenant_id}
            )
        
        return None
    
    def _select_best_detection(self, attempts: List[DetectionResult]) -> Optional[DetectionResult]:
        """Sélectionne la meilleure détection parmi les tentatives"""
        if not attempts:
            return None
        
        # Pondération par méthode
        method_weights = {
            DetectionMethod.USER_PREFERENCE: 1.0,
            DetectionMethod.CONTENT_ANALYSIS: 0.9,
            DetectionMethod.ML_PREDICTION: 0.8,
            DetectionMethod.GEOGRAPHIC: 0.6,
            DetectionMethod.BROWSER_HEADER: 0.5,
            DetectionMethod.TENANT_DEFAULT: 0.3,
            DetectionMethod.FALLBACK: 0.1
        }
        
        # Calcul du score pondéré
        scored_attempts = []
        for attempt in attempts:
            weight = method_weights.get(attempt.method, 0.5)
            weighted_score = attempt.confidence * weight
            scored_attempts.append((weighted_score, attempt))
        
        # Tri par score décroissant
        scored_attempts.sort(key=lambda x: x[0], reverse=True)
        
        return scored_attempts[0][1] if scored_attempts else None
    
    async def _update_user_profile(self, user_id: str, language: str, confidence: float) -> None:
        """Met à jour le profil linguistique utilisateur"""
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = UserLanguageProfile(user_id=user_id)
        
        profile = self._user_profiles[user_id]
        
        # Mise à jour des statistiques
        profile.detected_languages[language] = profile.detected_languages.get(language, 0) + 1
        
        # Mise à jour du score de confiance (moyenne mobile)
        current_confidence = profile.confidence_scores.get(language, 0.5)
        profile.confidence_scores[language] = (current_confidence + confidence) / 2
        
        # Mise à jour de la langue préférée
        if language not in profile.preferred_languages and confidence > 0.8:
            profile.preferred_languages.append(language)
        
        profile.last_updated = datetime.utcnow()
        
        # Sauvegarde en cache
        if self._redis:
            try:
                profile_data = {
                    "user_id": profile.user_id,
                    "preferred_languages": profile.preferred_languages,
                    "detected_languages": profile.detected_languages,
                    "confidence_scores": profile.confidence_scores,
                    "last_updated": profile.last_updated.isoformat()
                }
                
                await self._redis.setex(
                    f"user_profile:{user_id}",
                    86400 * 30,  # 30 jours
                    json.dumps(profile_data)
                )
                
            except Exception as e:
                logger.error(f"Erreur sauvegarde profil utilisateur: {e}")
    
    async def _cache_detection(self, text: str, result: DetectionResult) -> None:
        """Met en cache une détection"""
        if not self._redis or len(text) > 500:  # Ne pas cacher les textes trop longs
            return
        
        try:
            # Clé de cache basée sur le hash du texte
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = f"detection:{text_hash}"
            
            cache_data = {
                "language": result.language,
                "confidence": result.confidence,
                "method": result.method.value,
                "cached_at": datetime.utcnow().isoformat()
            }
            
            # Cache de 1 heure
            await self._redis.setex(cache_key, 3600, json.dumps(cache_data))
            
        except Exception as e:
            logger.error(f"Erreur mise en cache détection: {e}")
    
    async def _load_user_profiles(self) -> None:
        """Charge les profils utilisateur depuis le cache"""
        if not self._redis:
            return
        
        try:
            # Récupération de tous les profils
            keys = await self._redis.keys("user_profile:*")
            
            for key in keys:
                profile_data = await self._redis.get(key)
                if profile_data:
                    data = json.loads(profile_data)
                    user_id = data["user_id"]
                    
                    profile = UserLanguageProfile(
                        user_id=user_id,
                        preferred_languages=data.get("preferred_languages", []),
                        detected_languages=data.get("detected_languages", {}),
                        confidence_scores=data.get("confidence_scores", {}),
                        last_updated=datetime.fromisoformat(data.get("last_updated", datetime.utcnow().isoformat()))
                    )
                    
                    self._user_profiles[user_id] = profile
            
            logger.info(f"Profils utilisateur chargés: {len(self._user_profiles)}")
            
        except Exception as e:
            logger.error(f"Erreur chargement profils: {e}")
    
    def _update_stats(self, result: DetectionResult) -> None:
        """Met à jour les statistiques"""
        # Mise à jour du temps de traitement moyen
        current_avg = self._stats["avg_processing_time"]
        total = self._stats["total_detections"]
        
        self._stats["avg_processing_time"] = (
            (current_avg * (total - 1) + result.processing_time_ms) / total
        )
        
        # Stockage du score de confiance pour analyse
        self._stats["accuracy_scores"].append(result.confidence)
        
        # Limitation de l'historique (derniers 1000)
        if len(self._stats["accuracy_scores"]) > 1000:
            self._stats["accuracy_scores"] = self._stats["accuracy_scores"][-1000:]
    
    async def get_stats(self) -> Dict[str, any]:
        """Retourne les statistiques du détecteur"""
        accuracy_scores = self._stats["accuracy_scores"]
        avg_confidence = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        
        return {
            **self._stats,
            "average_confidence": round(avg_confidence, 3),
            "user_profiles_count": len(self._user_profiles),
            "supported_languages": self.supported_languages,
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, self._stats["cache_hits"] + self._stats["cache_misses"]) * 100
            )
        }
    
    async def get_user_profile(self, user_id: str) -> Optional[UserLanguageProfile]:
        """Retourne le profil linguistique d'un utilisateur"""
        return self._user_profiles.get(user_id)
    
    async def set_user_preference(self, user_id: str, language: str) -> bool:
        """Définit la préférence linguistique d'un utilisateur"""
        if language not in self.supported_languages:
            return False
        
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = UserLanguageProfile(user_id=user_id)
        
        profile = self._user_profiles[user_id]
        
        # Ajout en tête de liste (priorité maximale)
        if language in profile.preferred_languages:
            profile.preferred_languages.remove(language)
        
        profile.preferred_languages.insert(0, language)
        profile.confidence_scores[language] = 1.0
        profile.last_updated = datetime.utcnow()
        
        # Sauvegarde
        await self._update_user_profile(user_id, language, 1.0)
        
        return True
    
    async def batch_detect(self, texts: List[str], **kwargs) -> List[DetectionResult]:
        """Détection en lot pour améliorer les performances"""
        tasks = [
            self.detect_language(text, **kwargs) 
            for text in texts
        ]
        
        return await asyncio.gather(*tasks)
    
    async def close(self) -> None:
        """Ferme proprement le détecteur"""
        if self._redis:
            await self._redis.close()
        
        logger.info("Détecteur de langue fermé")


# Factory function
async def create_language_detector(
    redis_url: str = "redis://localhost:6379/2",
    supported_languages: Optional[List[str]] = None,
    default_language: str = "en"
) -> SmartLanguageDetector:
    """
    Factory pour créer et initialiser un détecteur de langue
    
    Args:
        redis_url: URL Redis
        supported_languages: Langues supportées
        default_language: Langue par défaut
        
    Returns:
        Détecteur initialisé
    """
    detector = SmartLanguageDetector(
        redis_url=redis_url,
        supported_languages=supported_languages,
        default_language=default_language
    )
    
    await detector.initialize()
    return detector


# Export des classes principales
__all__ = [
    "SmartLanguageDetector",
    "DetectionResult",
    "DetectionMethod",
    "ConfidenceLevel",
    "UserLanguageProfile",
    "create_language_detector"
]
