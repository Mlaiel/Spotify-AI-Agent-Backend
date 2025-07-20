"""
Processeur de Locales Avancé pour Spotify AI Agent
Système de traitement et formatage intelligent des messages localisés
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict
import threading
from enum import Enum
import unicodedata
import locale as system_locale

logger = logging.getLogger(__name__)


class MessageFormat(Enum):
    """Formats de messages supportés"""
    SIMPLE = "simple"
    ICU = "icu"
    GETTEXT = "gettext"
    FLUENT = "fluent"
    REACT_INTL = "react_intl"


@dataclass
class ProcessingConfig:
    """Configuration du processeur"""
    default_format: MessageFormat = MessageFormat.ICU
    enable_interpolation: bool = True
    enable_pluralization: bool = True
    enable_formatting: bool = True
    enable_fallback: bool = True
    strict_mode: bool = False
    max_recursion_depth: int = 5
    cache_processed: bool = True
    auto_escape_html: bool = True
    allow_rich_text: bool = True


@dataclass
class ProcessingContext:
    """Contexte de traitement"""
    locale_code: str
    tenant_id: Optional[str]
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    format_type: MessageFormat = MessageFormat.ICU
    timezone: Optional[str] = None
    currency: Optional[str] = None
    number_format: Optional[str] = None


class MessageProcessor(ABC):
    """Interface pour les processeurs de messages"""
    
    @abstractmethod
    async def process_message(
        self,
        message: str,
        context: ProcessingContext
    ) -> str:
        """Traite un message"""
        pass
    
    @property
    @abstractmethod
    def supported_format(self) -> MessageFormat:
        """Format supporté par ce processeur"""
        pass


class ICUMessageProcessor(MessageProcessor):
    """Processeur de messages au format ICU"""
    
    @property
    def supported_format(self) -> MessageFormat:
        return MessageFormat.ICU
    
    async def process_message(
        self,
        message: str,
        context: ProcessingContext
    ) -> str:
        """Traite un message ICU"""
        try:
            # Traitement des variables simples {variable}
            processed = await self._process_variables(message, context)
            
            # Traitement de la pluralisation
            processed = await self._process_plural(processed, context)
            
            # Traitement des sélections
            processed = await self._process_select(processed, context)
            
            # Traitement des formats de nombre et date
            processed = await self._process_formatting(processed, context)
            
            return processed
            
        except Exception as e:
            logger.error(f"ICU message processing error: {e}")
            return message  # Retourner le message original en cas d'erreur
    
    async def _process_variables(self, message: str, context: ProcessingContext) -> str:
        """Traite les variables simples"""
        pattern = r'\{([^}]+)\}'
        
        def replace_variable(match):
            var_name = match.group(1).strip()
            if var_name in context.variables:
                value = context.variables[var_name]
                return str(value)
            return match.group(0)  # Garder l'original si non trouvé
        
        return re.sub(pattern, replace_variable, message)
    
    async def _process_plural(self, message: str, context: ProcessingContext) -> str:
        """Traite la pluralisation ICU"""
        pattern = r'\{([^,}]+),\s*plural,\s*([^}]+)\}'
        
        def replace_plural(match):
            var_name = match.group(1).strip()
            plural_rules = match.group(2)
            
            if var_name not in context.variables:
                return match.group(0)
            
            count = context.variables[var_name]
            if not isinstance(count, (int, float)):
                return match.group(0)
            
            # Parser les règles de pluralisation
            rules = {}
            for rule in plural_rules.split():
                if '=' in rule:
                    # Règle exacte: =0 {aucun} =1 {un}
                    parts = rule.split('{', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].rstrip('}')
                        if key.startswith('='):
                            exact_value = int(key[1:])
                            if count == exact_value:
                                return value.format(**context.variables)
                        else:
                            rules[key] = value
            
            # Règles générales (zero, one, two, few, many, other)
            plural_form = self._get_plural_form(count, context.locale_code)
            if plural_form in rules:
                return rules[plural_form].format(**context.variables)
            elif 'other' in rules:
                return rules['other'].format(**context.variables)
            
            return match.group(0)
        
        return re.sub(pattern, replace_plural, message)
    
    async def _process_select(self, message: str, context: ProcessingContext) -> str:
        """Traite les sélections ICU"""
        pattern = r'\{([^,}]+),\s*select,\s*([^}]+)\}'
        
        def replace_select(match):
            var_name = match.group(1).strip()
            select_rules = match.group(2)
            
            if var_name not in context.variables:
                return match.group(0)
            
            value = str(context.variables[var_name])
            
            # Parser les règles de sélection
            rules = {}
            current_key = None
            current_value = ""
            in_braces = 0
            
            i = 0
            while i < len(select_rules):
                char = select_rules[i]
                
                if char == '{':
                    in_braces += 1
                    if in_braces == 1 and current_key:
                        # Début de la valeur
                        current_value = ""
                    else:
                        current_value += char
                elif char == '}':
                    in_braces -= 1
                    if in_braces == 0 and current_key:
                        # Fin de la valeur
                        rules[current_key.strip()] = current_value
                        current_key = None
                        current_value = ""
                    else:
                        current_value += char
                elif in_braces == 0 and char.isspace():
                    if current_key is None:
                        # Chercher la prochaine clé
                        j = i + 1
                        while j < len(select_rules) and not select_rules[j].isspace() and select_rules[j] != '{':
                            j += 1
                        if j > i + 1:
                            current_key = select_rules[i+1:j]
                            i = j - 1
                else:
                    if in_braces == 0 and current_key is None:
                        # Construire la clé
                        if not char.isspace():
                            j = i
                            while j < len(select_rules) and not select_rules[j].isspace() and select_rules[j] != '{':
                                j += 1
                            current_key = select_rules[i:j]
                            i = j - 1
                    else:
                        current_value += char
                
                i += 1
            
            # Appliquer la sélection
            if value in rules:
                return rules[value].format(**context.variables)
            elif 'other' in rules:
                return rules['other'].format(**context.variables)
            
            return match.group(0)
        
        return re.sub(pattern, replace_select, message)
    
    async def _process_formatting(self, message: str, context: ProcessingContext) -> str:
        """Traite le formatage des nombres et dates"""
        # Formatage des nombres
        number_pattern = r'\{([^,}]+),\s*number(?:,\s*([^}]+))?\}'
        
        def replace_number(match):
            var_name = match.group(1).strip()
            format_type = match.group(2) or "decimal"
            
            if var_name not in context.variables:
                return match.group(0)
            
            value = context.variables[var_name]
            if not isinstance(value, (int, float)):
                return str(value)
            
            return self._format_number(value, format_type, context)
        
        message = re.sub(number_pattern, replace_number, message)
        
        # Formatage des dates
        date_pattern = r'\{([^,}]+),\s*date(?:,\s*([^}]+))?\}'
        
        def replace_date(match):
            var_name = match.group(1).strip()
            format_type = match.group(2) or "medium"
            
            if var_name not in context.variables:
                return match.group(0)
            
            value = context.variables[var_name]
            return self._format_date(value, format_type, context)
        
        message = re.sub(date_pattern, replace_date, message)
        
        return message
    
    def _get_plural_form(self, count: Union[int, float], locale_code: str) -> str:
        """Détermine la forme plurielle selon la locale"""
        # Implémentation simplifiée des règles CLDR
        if locale_code.startswith('en'):
            return 'one' if count == 1 else 'other'
        elif locale_code.startswith('fr'):
            return 'one' if count <= 1 else 'other'
        elif locale_code.startswith('ru'):
            if count % 10 == 1 and count % 100 != 11:
                return 'one'
            elif count % 10 in [2, 3, 4] and count % 100 not in [12, 13, 14]:
                return 'few'
            else:
                return 'many'
        else:
            return 'other'
    
    def _format_number(self, value: Union[int, float], format_type: str, context: ProcessingContext) -> str:
        """Formate un nombre selon la locale"""
        try:
            if format_type == "percent":
                return f"{value:.1%}"
            elif format_type == "currency":
                currency = context.currency or "USD"
                return f"{value:.2f} {currency}"
            elif format_type == "integer":
                return f"{int(value):,}"
            else:  # decimal
                return f"{value:,.2f}"
        except Exception as e:
            logger.warning(f"Number formatting error: {e}")
            return str(value)
    
    def _format_date(self, value: Any, format_type: str, context: ProcessingContext) -> str:
        """Formate une date selon la locale"""
        try:
            if isinstance(value, str):
                # Essayer de parser la date
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            elif isinstance(value, (int, float)):
                value = datetime.fromtimestamp(value)
            elif not isinstance(value, datetime):
                return str(value)
            
            if format_type == "short":
                return value.strftime("%d/%m/%Y")
            elif format_type == "medium":
                return value.strftime("%d %b %Y")
            elif format_type == "long":
                return value.strftime("%d %B %Y")
            elif format_type == "full":
                return value.strftime("%A %d %B %Y")
            else:
                return value.strftime(format_type)
                
        except Exception as e:
            logger.warning(f"Date formatting error: {e}")
            return str(value)


class SimpleMessageProcessor(MessageProcessor):
    """Processeur de messages simple avec interpolation de variables"""
    
    @property
    def supported_format(self) -> MessageFormat:
        return MessageFormat.SIMPLE
    
    async def process_message(
        self,
        message: str,
        context: ProcessingContext
    ) -> str:
        """Traite un message simple"""
        try:
            # Remplacement simple des variables
            for key, value in context.variables.items():
                placeholder = f"{{{key}}}"
                message = message.replace(placeholder, str(value))
            
            return message
            
        except Exception as e:
            logger.error(f"Simple message processing error: {e}")
            return message


class GettextMessageProcessor(MessageProcessor):
    """Processeur de messages Gettext"""
    
    @property
    def supported_format(self) -> MessageFormat:
        return MessageFormat.GETTEXT
    
    async def process_message(
        self,
        message: str,
        context: ProcessingContext
    ) -> str:
        """Traite un message Gettext"""
        try:
            # Traitement des variables de style printf
            if '%' in message:
                # Remplacement des variables %(name)s
                pattern = r'%\(([^)]+)\)s'
                
                def replace_var(match):
                    var_name = match.group(1)
                    if var_name in context.variables:
                        return str(context.variables[var_name])
                    return match.group(0)
                
                message = re.sub(pattern, replace_var, message)
            
            return message
            
        except Exception as e:
            logger.error(f"Gettext message processing error: {e}")
            return message


class LocaleProcessor:
    """Processeur principal de locales"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._processors = {
            MessageFormat.ICU: ICUMessageProcessor(),
            MessageFormat.SIMPLE: SimpleMessageProcessor(),
            MessageFormat.GETTEXT: GettextMessageProcessor()
        }
        self._cache = {}
        self._lock = threading.RLock()
        self._stats = defaultdict(int)
    
    async def process_message(
        self,
        key: str,
        message: str,
        context: ProcessingContext,
        fallback_message: Optional[str] = None
    ) -> str:
        """Traite un message avec le contexte donné"""
        try:
            self._stats['total_processing'] += 1
            
            # Vérifier le cache
            if self.config.cache_processed:
                cache_key = self._get_cache_key(key, message, context)
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    self._stats['cache_hits'] += 1
                    return cached_result
            
            self._stats['cache_misses'] += 1
            
            # Obtenir le processeur approprié
            processor = self._processors.get(context.format_type)
            if not processor:
                processor = self._processors[self.config.default_format]
            
            # Traiter le message
            try:
                result = await processor.process_message(message, context)
                
                # Post-traitement
                result = await self._post_process(result, context)
                
                # Mettre en cache
                if self.config.cache_processed:
                    await self._cache_result(cache_key, result)
                
                self._stats['successful_processing'] += 1
                return result
                
            except Exception as e:
                logger.error(f"Message processing error for key {key}: {e}")
                self._stats['processing_errors'] += 1
                
                # Fallback
                if self.config.enable_fallback and fallback_message:
                    return await self.process_message(
                        f"{key}_fallback",
                        fallback_message,
                        context
                    )
                
                return message  # Retourner le message original
            
        except Exception as e:
            logger.error(f"Locale processing error: {e}")
            return message
    
    async def process_batch(
        self,
        messages: Dict[str, str],
        context: ProcessingContext,
        fallback_messages: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Traite un lot de messages"""
        try:
            results = {}
            
            for key, message in messages.items():
                fallback = fallback_messages.get(key) if fallback_messages else None
                result = await self.process_message(key, message, context, fallback)
                results[key] = result
            
            self._stats['batch_processing'] += 1
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return messages
    
    async def validate_message_syntax(
        self,
        message: str,
        format_type: MessageFormat
    ) -> Dict[str, Any]:
        """Valide la syntaxe d'un message"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'variables': set(),
                'plural_rules': [],
                'select_rules': []
            }
            
            if format_type == MessageFormat.ICU:
                await self._validate_icu_syntax(message, validation_result)
            elif format_type == MessageFormat.GETTEXT:
                await self._validate_gettext_syntax(message, validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'variables': set(),
                'plural_rules': [],
                'select_rules': []
            }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de traitement"""
        with self._lock:
            cache_efficiency = 0
            total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
            if total_requests > 0:
                cache_efficiency = self._stats['cache_hits'] / total_requests
            
            return {
                'stats': dict(self._stats),
                'cache_size': len(self._cache),
                'cache_efficiency': cache_efficiency,
                'supported_formats': list(self._processors.keys()),
                'config': {
                    'default_format': self.config.default_format.value,
                    'enable_interpolation': self.config.enable_interpolation,
                    'enable_pluralization': self.config.enable_pluralization,
                    'cache_processed': self.config.cache_processed
                }
            }
    
    async def clear_cache(self):
        """Vide le cache de traitement"""
        with self._lock:
            self._cache.clear()
            logger.info("Processing cache cleared")
    
    async def _post_process(self, message: str, context: ProcessingContext) -> str:
        """Post-traitement du message"""
        try:
            result = message
            
            # Échappement HTML si activé
            if self.config.auto_escape_html and not self.config.allow_rich_text:
                result = self._escape_html(result)
            
            # Normalisation Unicode
            result = unicodedata.normalize('NFC', result)
            
            # Trim des espaces
            result = result.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            return message
    
    def _escape_html(self, text: str) -> str:
        """Échappe les caractères HTML"""
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#x27;",
            ">": "&gt;",
            "<": "&lt;",
        }
        return "".join(html_escape_table.get(c, c) for c in text)
    
    def _get_cache_key(
        self,
        key: str,
        message: str,
        context: ProcessingContext
    ) -> str:
        """Génère une clé de cache"""
        import hashlib
        
        cache_data = {
            'key': key,
            'message': message,
            'locale': context.locale_code,
            'format': context.format_type.value,
            'variables': sorted(context.variables.items()) if context.variables else []
        }
        
        cache_str = str(cache_data)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Récupère depuis le cache"""
        with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                
                # Vérifier l'expiration (1 heure)
                if datetime.now() - timestamp < timedelta(hours=1):
                    return result
                else:
                    del self._cache[cache_key]
            
            return None
    
    async def _cache_result(self, cache_key: str, result: str):
        """Met en cache un résultat"""
        with self._lock:
            self._cache[cache_key] = (result, datetime.now())
            
            # Limiter la taille du cache
            if len(self._cache) > 1000:
                # Supprimer les entrées les plus anciennes
                oldest_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )[:100]
                
                for key in oldest_keys:
                    del self._cache[key]
    
    async def _validate_icu_syntax(self, message: str, result: Dict[str, Any]):
        """Valide la syntaxe ICU"""
        try:
            # Vérifier les accolades équilibrées
            brace_count = 0
            for char in message:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count < 0:
                        result['errors'].append("Accolade fermante sans ouverture")
                        result['is_valid'] = False
                        return
            
            if brace_count != 0:
                result['errors'].append("Accolades non équilibrées")
                result['is_valid'] = False
            
            # Extraire les variables
            variables = re.findall(r'\{([^,}]+)(?:,[^}]*)?\}', message)
            result['variables'] = set(var.strip() for var in variables)
            
            # Vérifier les règles de pluralisation
            plural_matches = re.findall(r'\{[^,}]+,\s*plural,\s*([^}]+)\}', message)
            for match in plural_matches:
                result['plural_rules'].append(match)
            
            # Vérifier les règles de sélection
            select_matches = re.findall(r'\{[^,}]+,\s*select,\s*([^}]+)\}', message)
            for match in select_matches:
                result['select_rules'].append(match)
            
        except Exception as e:
            result['errors'].append(f"Erreur de validation ICU: {e}")
            result['is_valid'] = False
    
    async def _validate_gettext_syntax(self, message: str, result: Dict[str, Any]):
        """Valide la syntaxe Gettext"""
        try:
            # Vérifier les variables printf
            variables = re.findall(r'%\(([^)]+)\)s', message)
            result['variables'] = set(variables)
            
            # Vérifier les placeholders positionnels
            positional = re.findall(r'%[sd]', message)
            if positional:
                result['warnings'].append("Placeholders positionnels détectés (non recommandés)")
            
        except Exception as e:
            result['errors'].append(f"Erreur de validation Gettext: {e}")
            result['is_valid'] = False


class MessageFormatter:
    """Formateur de messages avec support avancé"""
    
    def __init__(self, processor: LocaleProcessor):
        self.processor = processor
    
    async def format_message(
        self,
        template: str,
        variables: Dict[str, Any],
        locale_code: str,
        format_type: MessageFormat = MessageFormat.ICU
    ) -> str:
        """Formate un message avec variables"""
        context = ProcessingContext(
            locale_code=locale_code,
            tenant_id=None,
            variables=variables,
            format_type=format_type
        )
        
        return await self.processor.process_message(
            "format_message",
            template,
            context
        )
    
    async def format_plural(
        self,
        count: int,
        singular: str,
        plural: str,
        locale_code: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Formate un message avec pluralisation"""
        variables = variables or {}
        variables['count'] = count
        
        # Construire le template ICU
        template = f"{{count, plural, one{{{singular}}} other{{{plural}}}}}"
        
        return await self.format_message(
            template,
            variables,
            locale_code,
            MessageFormat.ICU
        )
    
    async def format_currency(
        self,
        amount: float,
        currency: str,
        locale_code: str
    ) -> str:
        """Formate un montant en devise"""
        variables = {'amount': amount}
        template = "{amount, number, currency}"
        
        context = ProcessingContext(
            locale_code=locale_code,
            tenant_id=None,
            variables=variables,
            format_type=MessageFormat.ICU,
            currency=currency
        )
        
        return await self.processor.process_message(
            "format_currency",
            template,
            context
        )
    
    async def format_date_relative(
        self,
        date: datetime,
        locale_code: str,
        reference_date: Optional[datetime] = None
    ) -> str:
        """Formate une date de manière relative"""
        if reference_date is None:
            reference_date = datetime.now()
        
        delta = reference_date - date
        
        if delta.days == 0:
            if delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
            else:
                hours = delta.seconds // 3600
                return f"il y a {hours} heure{'s' if hours > 1 else ''}"
        elif delta.days == 1:
            return "hier"
        elif delta.days < 7:
            return f"il y a {delta.days} jour{'s' if delta.days > 1 else ''}"
        else:
            return await self.format_message(
                "{date, date, medium}",
                {'date': date},
                locale_code,
                MessageFormat.ICU
            )
