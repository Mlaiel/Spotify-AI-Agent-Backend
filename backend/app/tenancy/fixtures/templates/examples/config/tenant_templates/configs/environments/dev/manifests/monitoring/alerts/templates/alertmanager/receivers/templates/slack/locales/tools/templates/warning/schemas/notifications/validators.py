"""
Validateurs avancés pour le système de notifications
===================================================

Validation sophistiquée avec rules engine, ML, et sécurité.
"""

import re
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio

import dns.resolver
import phonenumbers
from phonenumbers import PhoneNumberFormat, NumberParseException
from email_validator import validate_email, EmailNotValidError
import validators
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import aioredis
from textblob import TextBlob
import langdetect
from profanity_check import predict as is_profane

from .models import *
from .schemas import *
from .config import NotificationSettings


class ValidationLevel(str, Enum):
    """Niveaux de validation"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationError(Exception):
    """Exception de validation"""
    def __init__(self, message: str, field: str = None, code: str = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)


@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    errors: List[ValidationError] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class BaseValidator(ABC):
    """Validateur de base"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Valider une valeur"""
        pass
    
    def _create_error(self, message: str, field: str = None, code: str = None) -> ValidationError:
        """Créer une erreur de validation"""
        return ValidationError(message, field, code)


class EmailValidator(BaseValidator):
    """Validateur email avancé"""
    
    async def validate(self, email: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Valider une adresse email"""
        
        result = ValidationResult(is_valid=True)
        
        if not email or not isinstance(email, str):
            result.is_valid = False
            result.errors.append(self._create_error("Email requis", "email", "REQUIRED"))
            return result
        
        # Validation basique du format
        try:
            validation_result = validate_email(
                email,
                check_deliverability=self.config.get('check_deliverability', True)
            )
            normalized_email = validation_result.email
            result.metadata['normalized_email'] = normalized_email
            
        except EmailNotValidError as e:
            result.is_valid = False
            result.errors.append(self._create_error(str(e), "email", "INVALID_FORMAT"))
            return result
        
        # Validation du domaine
        if self.config.get('validate_domain', True):
            domain = email.split('@')[1].lower()
            
            # Vérifier les domaines interdits
            blocked_domains = self.config.get('blocked_domains', [])
            if domain in blocked_domains:
                result.is_valid = False
                result.errors.append(
                    self._create_error(f"Domaine interdit: {domain}", "email", "BLOCKED_DOMAIN")
                )
                return result
            
            # Vérifier les domaines temporaires
            if self.config.get('block_disposable', False):
                if await self._is_disposable_domain(domain):
                    result.is_valid = False
                    result.errors.append(
                        self._create_error(f"Domaine temporaire non autorisé: {domain}", "email", "DISPOSABLE_DOMAIN")
                    )
                    return result
            
            # Vérifier l'existence du domaine (DNS MX)
            if self.config.get('check_mx_record', True):
                if not await self._check_mx_record(domain):
                    result.warnings.append(f"Aucun enregistrement MX trouvé pour {domain}")
        
        # Validation de la longueur
        max_length = self.config.get('max_length', 254)
        if len(email) > max_length:
            result.is_valid = False
            result.errors.append(
                self._create_error(f"Email trop long (max {max_length})", "email", "TOO_LONG")
            )
        
        return result
    
    async def _is_disposable_domain(self, domain: str) -> bool:
        """Vérifier si le domaine est temporaire"""
        # Liste de domaines temporaires connus
        disposable_domains = {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'temp-mail.org', 'throwaway.email',
            # Ajouter d'autres domaines selon les besoins
        }
        
        return domain in disposable_domains
    
    async def _check_mx_record(self, domain: str) -> bool:
        """Vérifier l'existence d'un enregistrement MX"""
        try:
            mx_records = dns.resolver.resolve(domain, 'MX')
            return len(mx_records) > 0
        except:
            return False


class PhoneValidator(BaseValidator):
    """Validateur de numéro de téléphone"""
    
    async def validate(self, phone: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Valider un numéro de téléphone"""
        
        result = ValidationResult(is_valid=True)
        
        if not phone or not isinstance(phone, str):
            result.is_valid = False
            result.errors.append(self._create_error("Numéro de téléphone requis", "phone", "REQUIRED"))
            return result
        
        # Nettoyer le numéro
        clean_phone = re.sub(r'[^\d+]', '', phone)
        
        try:
            # Parser avec phonenumbers
            default_region = context.get('country_code', 'US') if context else 'US'
            parsed_number = phonenumbers.parse(clean_phone, default_region)
            
            # Vérifier la validité
            if not phonenumbers.is_valid_number(parsed_number):
                result.is_valid = False
                result.errors.append(
                    self._create_error("Numéro de téléphone invalide", "phone", "INVALID_NUMBER")
                )
                return result
            
            # Formater le numéro
            formatted_international = phonenumbers.format_number(parsed_number, PhoneNumberFormat.E164)
            formatted_national = phonenumbers.format_number(parsed_number, PhoneNumberFormat.NATIONAL)
            
            result.metadata.update({
                'formatted_international': formatted_international,
                'formatted_national': formatted_national,
                'country_code': parsed_number.country_code,
                'national_number': parsed_number.national_number,
                'region': phonenumbers.region_code_for_number(parsed_number)
            })
            
            # Vérifier le type de numéro
            number_type = phonenumbers.number_type(parsed_number)
            allowed_types = self.config.get('allowed_types', [
                phonenumbers.PhoneNumberType.MOBILE,
                phonenumbers.PhoneNumberType.FIXED_LINE,
                phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE
            ])
            
            if number_type not in allowed_types:
                result.warnings.append(f"Type de numéro inhabituel: {number_type}")
            
            # Vérifier les régions autorisées
            region = phonenumbers.region_code_for_number(parsed_number)
            allowed_regions = self.config.get('allowed_regions')
            if allowed_regions and region not in allowed_regions:
                result.is_valid = False
                result.errors.append(
                    self._create_error(f"Région non autorisée: {region}", "phone", "BLOCKED_REGION")
                )
            
        except NumberParseException as e:
            result.is_valid = False
            result.errors.append(
                self._create_error(f"Erreur parsing numéro: {e}", "phone", "PARSE_ERROR")
            )
        
        return result


class ContentValidator(BaseValidator):
    """Validateur de contenu avancé"""
    
    async def validate(self, content: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Valider le contenu d'un message"""
        
        result = ValidationResult(is_valid=True)
        
        if not content or not isinstance(content, str):
            result.is_valid = False
            result.errors.append(self._create_error("Contenu requis", "content", "REQUIRED"))
            return result
        
        # Validation de la longueur
        min_length = self.config.get('min_length', 1)
        max_length = self.config.get('max_length', 10000)
        
        if len(content) < min_length:
            result.is_valid = False
            result.errors.append(
                self._create_error(f"Contenu trop court (min {min_length})", "content", "TOO_SHORT")
            )
        
        if len(content) > max_length:
            result.is_valid = False
            result.errors.append(
                self._create_error(f"Contenu trop long (max {max_length})", "content", "TOO_LONG")
            )
        
        # Détection de la langue
        if self.config.get('detect_language', True):
            try:
                detected_lang = langdetect.detect(content)
                result.metadata['detected_language'] = detected_lang
                
                allowed_languages = self.config.get('allowed_languages')
                if allowed_languages and detected_lang not in allowed_languages:
                    result.warnings.append(f"Langue détectée non autorisée: {detected_lang}")
            except:
                result.warnings.append("Impossible de détecter la langue")
        
        # Détection de profanité
        if self.config.get('check_profanity', True):
            try:
                if is_profane(content):
                    if self.config.get('block_profanity', False):
                        result.is_valid = False
                        result.errors.append(
                            self._create_error("Contenu inapproprié détecté", "content", "PROFANITY")
                        )
                    else:
                        result.warnings.append("Contenu potentiellement inapproprié détecté")
            except:
                pass
        
        # Analyse de sentiment
        if self.config.get('analyze_sentiment', False):
            try:
                blob = TextBlob(content)
                sentiment = blob.sentiment
                result.metadata['sentiment'] = {
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity
                }
                
                # Alerter sur sentiment très négatif
                if sentiment.polarity < -0.5:
                    result.warnings.append("Sentiment très négatif détecté")
            except:
                pass
        
        # Détection de spam patterns
        if self.config.get('check_spam', True):
            spam_score = await self._calculate_spam_score(content)
            result.metadata['spam_score'] = spam_score
            
            spam_threshold = self.config.get('spam_threshold', 0.7)
            if spam_score > spam_threshold:
                if self.config.get('block_spam', False):
                    result.is_valid = False
                    result.errors.append(
                        self._create_error("Contenu identifié comme spam", "content", "SPAM")
                    )
                else:
                    result.warnings.append(f"Score de spam élevé: {spam_score:.2f}")
        
        # Validation des URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        if urls:
            result.metadata['urls'] = urls
            
            # Vérifier les domaines d'URL
            blocked_domains = self.config.get('blocked_url_domains', [])
            for url in urls:
                try:
                    domain = validators.domain(url.split('/')[2])
                    if domain in blocked_domains:
                        result.is_valid = False
                        result.errors.append(
                            self._create_error(f"URL avec domaine interdit: {domain}", "content", "BLOCKED_URL")
                        )
                except:
                    pass
        
        return result
    
    async def _calculate_spam_score(self, content: str) -> float:
        """Calculer un score de spam basique"""
        score = 0.0
        content_lower = content.lower()
        
        # Patterns de spam communs
        spam_patterns = [
            r'urgent[!]*',
            r'act now',
            r'limited time',
            r'free money',
            r'click here',
            r'buy now',
            r'guarantee[d]*',
            r'no risk',
            r'winner',
            r'congratulations',
            r'\$+\d+',  # Montants d'argent
            r'[A-Z]{5,}',  # Mots en majuscules
        ]
        
        for pattern in spam_patterns:
            matches = len(re.findall(pattern, content_lower))
            score += matches * 0.1
        
        # Répétition excessive de caractères
        repeated_chars = len(re.findall(r'(.)\1{3,}', content))
        score += repeated_chars * 0.05
        
        # Ratio majuscules/minuscules
        if len(content) > 10:
            upper_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if upper_ratio > 0.5:
                score += 0.3
        
        return min(score, 1.0)


class RateLimitValidator(BaseValidator):
    """Validateur de limite de taux"""
    
    def __init__(self, config: Dict[str, Any] = None, redis_client: aioredis.Redis = None):
        super().__init__(config)
        self.redis = redis_client
    
    async def validate(self, key: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Valider les limites de taux"""
        
        result = ValidationResult(is_valid=True)
        
        if not self.redis:
            result.warnings.append("Redis non disponible, limite de taux non vérifiée")
            return result
        
        # Configuration des limites
        limits = self.config.get('limits', {
            'per_minute': 60,
            'per_hour': 1000,
            'per_day': 10000
        })
        
        # Vérifier chaque limite
        for period, limit in limits.items():
            window_seconds = self._get_window_seconds(period)
            current_count = await self._get_current_count(key, window_seconds)
            
            result.metadata[f'count_{period}'] = current_count
            result.metadata[f'limit_{period}'] = limit
            
            if current_count >= limit:
                result.is_valid = False
                result.errors.append(
                    self._create_error(
                        f"Limite de taux dépassée: {current_count}/{limit} {period}",
                        "rate_limit",
                        f"RATE_LIMIT_{period.upper()}"
                    )
                )
                break
        
        return result
    
    def _get_window_seconds(self, period: str) -> int:
        """Obtenir la fenêtre en secondes"""
        windows = {
            'per_minute': 60,
            'per_hour': 3600,
            'per_day': 86400
        }
        return windows.get(period, 60)
    
    async def _get_current_count(self, key: str, window_seconds: int) -> int:
        """Obtenir le comptage actuel dans la fenêtre"""
        now = datetime.now(timezone.utc).timestamp()
        
        # Utiliser une fenêtre glissante avec Redis sorted set
        rate_key = f"rate_limit:{key}:{window_seconds}"
        
        # Nettoyer les entrées expirées
        await self.redis.zremrangebyscore(rate_key, 0, now - window_seconds)
        
        # Compter les entrées dans la fenêtre
        count = await self.redis.zcard(rate_key)
        
        return count
    
    async def increment_count(self, key: str):
        """Incrémenter le compteur"""
        if not self.redis:
            return
        
        now = datetime.now(timezone.utc).timestamp()
        
        # Incrémenter pour toutes les fenêtres configurées
        limits = self.config.get('limits', {})
        for period in limits.keys():
            window_seconds = self._get_window_seconds(period)
            rate_key = f"rate_limit:{key}:{window_seconds}"
            
            # Ajouter une entrée
            await self.redis.zadd(rate_key, {str(now): now})
            
            # Expirer la clé
            await self.redis.expire(rate_key, window_seconds)


class SecurityValidator(BaseValidator):
    """Validateur de sécurité"""
    
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Valider les aspects sécuritaires"""
        
        result = ValidationResult(is_valid=True)
        
        # Vérifier les injections SQL
        if self.config.get('check_sql_injection', True):
            sql_patterns = [
                r"(\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bdrop\b|\bunion\b)",
                r"(\bor\b|\band\b)\s+\d+\s*=\s*\d+",
                r"['\"];?\s*--",
                r"\bexec\b|\bexecute\b"
            ]
            
            for field_name, field_value in data.items():
                if isinstance(field_value, str):
                    for pattern in sql_patterns:
                        if re.search(pattern, field_value.lower()):
                            result.is_valid = False
                            result.errors.append(
                                self._create_error(
                                    f"Injection SQL potentielle détectée dans {field_name}",
                                    field_name,
                                    "SQL_INJECTION"
                                )
                            )
                            break
        
        # Vérifier les injections XSS
        if self.config.get('check_xss', True):
            xss_patterns = [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"eval\s*\(",
                r"document\.",
                r"window\."
            ]
            
            for field_name, field_value in data.items():
                if isinstance(field_value, str):
                    for pattern in xss_patterns:
                        if re.search(pattern, field_value.lower()):
                            result.is_valid = False
                            result.errors.append(
                                self._create_error(
                                    f"XSS potentiel détecté dans {field_name}",
                                    field_name,
                                    "XSS"
                                )
                            )
                            break
        
        # Vérifier la taille des données
        max_data_size = self.config.get('max_data_size', 1024 * 1024)  # 1MB
        data_size = len(json.dumps(data, default=str))
        
        if data_size > max_data_size:
            result.is_valid = False
            result.errors.append(
                self._create_error(
                    f"Données trop volumineuses: {data_size} bytes",
                    "data_size",
                    "TOO_LARGE"
                )
            )
        
        # Vérifier les caractères suspects
        if self.config.get('check_suspicious_chars', True):
            suspicious_chars = set(['\x00', '\x01', '\x02', '\x03', '\x04', '\x05'])
            
            for field_name, field_value in data.items():
                if isinstance(field_value, str):
                    found_suspicious = suspicious_chars.intersection(set(field_value))
                    if found_suspicious:
                        result.warnings.append(
                            f"Caractères suspects dans {field_name}: {found_suspicious}"
                        )
        
        return result


class NotificationValidator:
    """Validateur principal pour les notifications"""
    
    def __init__(
        self,
        settings: NotificationSettings,
        db_session: AsyncSession,
        redis_client: aioredis.Redis
    ):
        self.settings = settings
        self.db = db_session
        self.redis = redis_client
        self.logger = logging.getLogger("NotificationValidator")
        
        # Initialiser les validateurs spécialisés
        self.email_validator = EmailValidator({
            'check_deliverability': True,
            'validate_domain': True,
            'block_disposable': True,
            'check_mx_record': True,
            'max_length': 254,
            'blocked_domains': ['example.com', 'test.com']
        })
        
        self.phone_validator = PhoneValidator({
            'allowed_types': [
                phonenumbers.PhoneNumberType.MOBILE,
                phonenumbers.PhoneNumberType.FIXED_LINE,
                phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE
            ]
        })
        
        self.content_validator = ContentValidator({
            'min_length': 1,
            'max_length': 10000,
            'detect_language': True,
            'allowed_languages': ['en', 'fr', 'es', 'de'],
            'check_profanity': True,
            'block_profanity': False,
            'analyze_sentiment': True,
            'check_spam': True,
            'spam_threshold': 0.7,
            'block_spam': False
        })
        
        self.rate_limit_validator = RateLimitValidator(
            {
                'limits': {
                    'per_minute': settings.get_rate_limit('per_user_per_minute') or 60,
                    'per_hour': settings.get_rate_limit('per_user_per_hour') or 600
                }
            },
            redis_client
        )
        
        self.security_validator = SecurityValidator({
            'check_sql_injection': True,
            'check_xss': True,
            'max_data_size': 1024 * 1024,  # 1MB
            'check_suspicious_chars': True
        })
    
    async def validate_notification(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str,
        user_id: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationResult:
        """Valider une notification complète"""
        
        result = ValidationResult(is_valid=True)
        
        try:
            # Validation de sécurité globale
            security_result = await self.security_validator.validate(
                notification.dict(),
                {'tenant_id': tenant_id, 'user_id': user_id}
            )
            
            if not security_result.is_valid:
                result.is_valid = False
                result.errors.extend(security_result.errors)
                return result  # Arrêter immédiatement si sécurité échoue
            
            result.warnings.extend(security_result.warnings)
            
            # Validation du contenu
            content_result = await self.content_validator.validate(
                notification.message,
                {'title': notification.title}
            )
            
            if not content_result.is_valid:
                result.is_valid = False
                result.errors.extend(content_result.errors)
            
            result.warnings.extend(content_result.warnings)
            result.metadata.update(content_result.metadata)
            
            # Validation des destinataires
            for i, recipient in enumerate(notification.recipients):
                recipient_result = await self._validate_recipient(
                    recipient, 
                    validation_level,
                    f"recipients[{i}]"
                )
                
                if not recipient_result.is_valid:
                    result.is_valid = False
                    result.errors.extend(recipient_result.errors)
                
                result.warnings.extend(recipient_result.warnings)
            
            # Validation des canaux
            for i, channel in enumerate(notification.channels):
                channel_result = await self._validate_channel(
                    channel,
                    validation_level,
                    f"channels[{i}]"
                )
                
                if not channel_result.is_valid:
                    result.is_valid = False
                    result.errors.extend(channel_result.errors)
                
                result.warnings.extend(channel_result.warnings)
            
            # Validation des limites tenant
            tenant_result = await self._validate_tenant_limits(
                notification,
                tenant_id,
                user_id
            )
            
            if not tenant_result.is_valid:
                result.is_valid = False
                result.errors.extend(tenant_result.errors)
            
            # Validation des limites de taux
            if user_id and validation_level != ValidationLevel.BASIC:
                rate_limit_key = f"user:{user_id}"
                rate_result = await self.rate_limit_validator.validate(rate_limit_key)
                
                if not rate_result.is_valid:
                    result.is_valid = False
                    result.errors.extend(rate_result.errors)
                else:
                    # Incrémenter le compteur si validation réussie
                    await self.rate_limit_validator.increment_count(rate_limit_key)
            
            # Validation avancée selon le niveau
            if validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                advanced_result = await self._validate_advanced(notification, tenant_id)
                
                if not advanced_result.is_valid:
                    result.is_valid = False
                    result.errors.extend(advanced_result.errors)
                
                result.warnings.extend(advanced_result.warnings)
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation: {e}")
            result.is_valid = False
            result.errors.append(
                ValidationError(f"Erreur interne de validation: {e}", None, "INTERNAL_ERROR")
            )
        
        return result
    
    async def _validate_recipient(
        self,
        recipient: RecipientSchema,
        validation_level: ValidationLevel,
        field_prefix: str
    ) -> ValidationResult:
        """Valider un destinataire"""
        
        result = ValidationResult(is_valid=True)
        
        # Validation email si présent
        if recipient.email:
            email_result = await self.email_validator.validate(recipient.email)
            if not email_result.is_valid:
                for error in email_result.errors:
                    error.field = f"{field_prefix}.email"
                result.is_valid = False
                result.errors.extend(email_result.errors)
            result.warnings.extend(email_result.warnings)
        
        # Validation téléphone si présent
        if recipient.phone:
            phone_result = await self.phone_validator.validate(recipient.phone)
            if not phone_result.is_valid:
                for error in phone_result.errors:
                    error.field = f"{field_prefix}.phone"
                result.is_valid = False
                result.errors.extend(phone_result.errors)
            result.warnings.extend(phone_result.warnings)
        
        # Validation des métadonnées
        if recipient.metadata:
            metadata_size = len(json.dumps(recipient.metadata))
            if metadata_size > 10240:  # 10KB
                result.warnings.append(f"Métadonnées volumineuses: {metadata_size} bytes")
        
        return result
    
    async def _validate_channel(
        self,
        channel: ChannelConfigSchema,
        validation_level: ValidationLevel,
        field_prefix: str
    ) -> ValidationResult:
        """Valider une configuration de canal"""
        
        result = ValidationResult(is_valid=True)
        
        # Vérifier que le canal est activé dans la configuration
        channel_config = self.settings.get_channel_config(channel.type.value)
        if not channel_config or not channel_config.get('enabled', False):
            result.is_valid = False
            result.errors.append(
                ValidationError(
                    f"Canal {channel.type.value} non activé",
                    f"{field_prefix}.type",
                    "CHANNEL_DISABLED"
                )
            )
            return result
        
        # Validation spécifique par type de canal
        if channel.type == ChannelTypeEnum.EMAIL:
            if channel.email_reply_to:
                email_result = await self.email_validator.validate(channel.email_reply_to)
                if not email_result.is_valid:
                    result.is_valid = False
                    result.errors.append(
                        ValidationError(
                            "Adresse reply-to invalide",
                            f"{field_prefix}.email_reply_to",
                            "INVALID_REPLY_TO"
                        )
                    )
        
        elif channel.type == ChannelTypeEnum.WEBHOOK:
            if channel.webhook_url:
                if not validators.url(str(channel.webhook_url)):
                    result.is_valid = False
                    result.errors.append(
                        ValidationError(
                            "URL webhook invalide",
                            f"{field_prefix}.webhook_url",
                            "INVALID_WEBHOOK_URL"
                        )
                    )
        
        # Validation des limites de taux du canal
        if channel.rate_limit_per_minute:
            global_limit = channel_config.get('rate_limit_per_minute', 1000)
            if channel.rate_limit_per_minute > global_limit:
                result.warnings.append(
                    f"Limite de taux canal supérieure à la limite globale: {channel.rate_limit_per_minute} > {global_limit}"
                )
        
        return result
    
    async def _validate_tenant_limits(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str,
        user_id: Optional[str]
    ) -> ValidationResult:
        """Valider les limites du tenant"""
        
        result = ValidationResult(is_valid=True)
        
        # Vérifier le nombre de destinataires
        max_recipients = self.settings.get_tenant_limit('max_recipients_per_notification')
        if max_recipients and len(notification.recipients) > max_recipients:
            result.is_valid = False
            result.errors.append(
                ValidationError(
                    f"Trop de destinataires: {len(notification.recipients)} > {max_recipients}",
                    "recipients",
                    "TOO_MANY_RECIPIENTS"
                )
            )
        
        # Vérifier la limite horaire du tenant
        tenant_hourly_limit = self.settings.get_tenant_limit('max_notifications_per_hour')
        if tenant_hourly_limit:
            # Compter les notifications de la dernière heure
            hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            
            query = select(func.count(Notification.id)).where(
                and_(
                    Notification.tenant_id == tenant_id,
                    Notification.created_at >= hour_ago
                )
            )
            
            result_count = await self.db.execute(query)
            current_count = result_count.scalar() or 0
            
            if current_count >= tenant_hourly_limit:
                result.is_valid = False
                result.errors.append(
                    ValidationError(
                        f"Limite horaire tenant dépassée: {current_count}/{tenant_hourly_limit}",
                        None,
                        "TENANT_HOURLY_LIMIT"
                    )
                )
        
        return result
    
    async def _validate_advanced(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str
    ) -> ValidationResult:
        """Validation avancée avec ML et rules engine"""
        
        result = ValidationResult(is_valid=True)
        
        # Analyser les patterns suspects
        if await self._detect_suspicious_patterns(notification):
            result.warnings.append("Patterns suspects détectés dans la notification")
        
        # Vérifier la cohérence avec l'historique
        if await self._check_historical_consistency(notification, tenant_id):
            result.warnings.append("Notification incohérente avec l'historique")
        
        return result
    
    async def _detect_suspicious_patterns(self, notification: NotificationCreateSchema) -> bool:
        """Détecter des patterns suspects avec ML"""
        # Implémentation basique - à améliorer avec des modèles ML
        
        # Pattern: même message à de nombreux destinataires
        if len(notification.recipients) > 100 and len(notification.message) < 50:
            return True
        
        # Pattern: urgence excessive
        urgency_keywords = ['urgent', 'immédiat', 'maintenant', 'vite', 'rapidement']
        message_lower = notification.message.lower()
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in message_lower)
        
        if urgency_count > 2:
            return True
        
        return False
    
    async def _check_historical_consistency(
        self,
        notification: NotificationCreateSchema,
        tenant_id: str
    ) -> bool:
        """Vérifier la cohérence avec l'historique"""
        
        # Vérifier les pics d'activité inhabituelle
        hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        day_ago = datetime.now(timezone.utc) - timedelta(days=1)
        
        # Compter les notifications récentes
        recent_query = select(func.count(Notification.id)).where(
            and_(
                Notification.tenant_id == tenant_id,
                Notification.created_at >= hour_ago
            )
        )
        
        daily_query = select(func.count(Notification.id)).where(
            and_(
                Notification.tenant_id == tenant_id,
                Notification.created_at >= day_ago
            )
        )
        
        recent_result = await self.db.execute(recent_query)
        daily_result = await self.db.execute(daily_query)
        
        recent_count = recent_result.scalar() or 0
        daily_count = daily_result.scalar() or 0
        
        # Détecter des pics inhabituels
        daily_avg = daily_count / 24  # Moyenne par heure
        
        if daily_avg > 0 and recent_count > daily_avg * 5:  # 5x la moyenne
            return True
        
        return False
