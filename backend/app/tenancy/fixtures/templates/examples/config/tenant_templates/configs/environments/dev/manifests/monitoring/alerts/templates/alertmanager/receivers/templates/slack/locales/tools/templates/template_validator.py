"""
Slack Template Validator - Validateur de templates et payloads Slack
Validation avancée avec règles métier et sécurité
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError
from dataclasses import dataclass
from enum import Enum

import bleach
from urllib.parse import urlparse


class ValidationLevel(Enum):
    """Niveaux de validation"""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecurityLevel(Enum):
    """Niveaux de sécurité"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    security_issues: List[str]
    performance_warnings: List[str]


class SlackTemplateValidator:
    """
    Validateur avancé pour templates et payloads Slack
    
    Fonctionnalités :
    - Validation de schéma JSON
    - Validation de sécurité
    - Validation de performance
    - Règles métier personnalisées
    - Détection d'injection
    - Validation de contenu
    """

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STRICT,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        max_payload_size: int = 10000,
        max_attachments: int = 20,
        allowed_domains: Optional[Set[str]] = None
    ):
        self.validation_level = validation_level
        self.security_level = security_level
        self.max_payload_size = max_payload_size
        self.max_attachments = max_attachments
        self.allowed_domains = allowed_domains or {
            "spotify-ai-agent.com",
            "grafana.spotify-ai-agent.com", 
            "prometheus.spotify-ai-agent.com",
            "alertmanager.spotify-ai-agent.com"
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Chargement des schémas de validation
        self._load_schemas()
        
        # Patterns de sécurité
        self._init_security_patterns()

    def _load_schemas(self):
        """Charge les schémas JSON pour validation"""
        
        # Schéma Slack de base
        self.slack_message_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "maxLength": 4000},
                "username": {"type": "string", "maxLength": 80},
                "icon_emoji": {"type": "string", "pattern": r"^:[a-z0-9_+-]+:$"},
                "icon_url": {"type": "string", "format": "uri"},
                "channel": {"type": "string", "pattern": r"^[#@]?[a-z0-9_-]+$"},
                "attachments": {
                    "type": "array",
                    "maxItems": self.max_attachments,
                    "items": {"$ref": "#/definitions/attachment"}
                },
                "blocks": {
                    "type": "array",
                    "maxItems": 50,
                    "items": {"$ref": "#/definitions/block"}
                }
            },
            "definitions": {
                "attachment": {
                    "type": "object",
                    "properties": {
                        "fallback": {"type": "string", "maxLength": 1000},
                        "color": {"type": "string", "pattern": r"^(good|warning|danger|#[0-9A-Fa-f]{6})$"},
                        "pretext": {"type": "string", "maxLength": 2000},
                        "author_name": {"type": "string", "maxLength": 256},
                        "author_link": {"type": "string", "format": "uri"},
                        "author_icon": {"type": "string", "format": "uri"},
                        "title": {"type": "string", "maxLength": 2000},
                        "title_link": {"type": "string", "format": "uri"},
                        "text": {"type": "string", "maxLength": 8000},
                        "fields": {
                            "type": "array",
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string", "maxLength": 300},
                                    "value": {"type": "string", "maxLength": 2000},
                                    "short": {"type": "boolean"}
                                },
                                "required": ["title", "value"]
                            }
                        },
                        "image_url": {"type": "string", "format": "uri"},
                        "thumb_url": {"type": "string", "format": "uri"},
                        "footer": {"type": "string", "maxLength": 300},
                        "footer_icon": {"type": "string", "format": "uri"},
                        "ts": {"type": "integer"}
                    }
                },
                "block": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["section", "divider", "image", "actions", "context", "input"]},
                        "text": {"$ref": "#/definitions/text_object"},
                        "fields": {
                            "type": "array",
                            "maxItems": 10,
                            "items": {"$ref": "#/definitions/text_object"}
                        },
                        "accessory": {"type": "object"},
                        "elements": {"type": "array", "maxItems": 10}
                    },
                    "required": ["type"]
                },
                "text_object": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["plain_text", "mrkdwn"]},
                        "text": {"type": "string", "maxLength": 3000},
                        "emoji": {"type": "boolean"},
                        "verbatim": {"type": "boolean"}
                    },
                    "required": ["type", "text"]
                }
            }
        }

    def _init_security_patterns(self):
        """Initialise les patterns de sécurité"""
        
        # Patterns d'injection potentiels
        self.injection_patterns = [
            r"<script[^>]*>.*?</script>",  # Scripts JavaScript
            r"javascript:",                # URLs JavaScript
            r"data:",                     # URLs data
            r"vbscript:",                 # VBScript
            r"on\w+\s*=",                # Event handlers
            r"<iframe[^>]*>",            # iframes
            r"<object[^>]*>",            # objects
            r"<embed[^>]*>",             # embeds
            r"<form[^>]*>",              # formulaires
            r"<input[^>]*>",             # inputs
        ]
        
        # Patterns de tokens/secrets potentiels
        self.secret_patterns = [
            r"xox[baprs]-[0-9a-zA-Z\-]+",     # Tokens Slack
            r"AKIA[0-9A-Z]{16}",              # AWS Access Keys
            r"sk-[a-zA-Z0-9]{48}",            # OpenAI API Keys
            r"ghp_[a-zA-Z0-9]{36}",           # GitHub Personal Access Tokens
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",  # UUIDs potentiels
        ]
        
        # HTML autorisé pour Slack
        self.allowed_html_tags = {
            'b', 'i', 'strong', 'em', 'code', 'pre', 'blockquote'
        }
        
        self.allowed_html_attrs = {}

    def validate_slack_payload(
        self,
        payload: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Valide un payload Slack complet
        
        Args:
            payload: Payload Slack à valider
            tenant_id: ID du tenant pour validation contextuelle
            
        Returns:
            ValidationResult: Résultat de la validation
        """
        
        errors = []
        warnings = []
        suggestions = []
        security_issues = []
        performance_warnings = []
        
        try:
            # 1. Validation de la taille
            payload_size = len(json.dumps(payload))
            if payload_size > self.max_payload_size:
                errors.append(f"Payload trop volumineux: {payload_size} > {self.max_payload_size}")
            elif payload_size > self.max_payload_size * 0.8:
                performance_warnings.append(f"Payload proche de la limite: {payload_size}")
            
            # 2. Validation du schéma JSON
            try:
                validate(instance=payload, schema=self.slack_message_schema)
            except ValidationError as e:
                errors.append(f"Erreur de schéma: {e.message}")
            
            # 3. Validation de sécurité
            security_result = self._validate_security(payload)
            security_issues.extend(security_result)
            
            # 4. Validation du contenu
            content_result = self._validate_content(payload)
            warnings.extend(content_result.get('warnings', []))
            suggestions.extend(content_result.get('suggestions', []))
            
            # 5. Validation des URLs
            url_result = self._validate_urls(payload)
            errors.extend(url_result.get('errors', []))
            warnings.extend(url_result.get('warnings', []))
            
            # 6. Validation métier
            if tenant_id:
                business_result = self._validate_business_rules(payload, tenant_id)
                warnings.extend(business_result.get('warnings', []))
                suggestions.extend(business_result.get('suggestions', []))
            
            # 7. Validation de performance
            perf_result = self._validate_performance(payload)
            performance_warnings.extend(perf_result)
            
        except Exception as e:
            errors.append(f"Erreur lors de la validation: {str(e)}")
            self.logger.error(f"Erreur de validation: {e}", exc_info=True)
        
        # Détermination du statut final
        is_valid = len(errors) == 0 and len(security_issues) == 0
        
        # En mode strict, les warnings deviennent des erreurs
        if self.validation_level == ValidationLevel.STRICT and warnings:
            errors.extend(warnings)
            warnings = []
            is_valid = False
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            security_issues=security_issues,
            performance_warnings=performance_warnings
        )

    def _validate_security(self, payload: Dict[str, Any]) -> List[str]:
        """Valide les aspects sécuritaires du payload"""
        
        security_issues = []
        payload_str = json.dumps(payload)
        
        # 1. Détection d'injection de code
        for pattern in self.injection_patterns:
            if re.search(pattern, payload_str, re.IGNORECASE):
                security_issues.append(f"Injection potentielle détectée: {pattern}")
        
        # 2. Détection de secrets/tokens
        for pattern in self.secret_patterns:
            if re.search(pattern, payload_str):
                security_issues.append("Token/secret potentiel détecté dans le payload")
        
        # 3. Validation des champs texte
        text_fields = self._extract_text_fields(payload)
        for field_path, text in text_fields:
            # Nettoyage HTML
            cleaned_text = bleach.clean(
                text,
                tags=self.allowed_html_tags,
                attributes=self.allowed_html_attrs,
                strip=True
            )
            
            if cleaned_text != text and self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                security_issues.append(f"HTML non autorisé dans le champ: {field_path}")
        
        # 4. Validation de la longueur des chaînes
        if self.security_level == SecurityLevel.CRITICAL:
            for field_path, text in text_fields:
                if len(text) > 10000:  # Limite arbitraire pour éviter les DoS
                    security_issues.append(f"Champ trop long (DoS potentiel): {field_path}")
        
        return security_issues

    def _extract_text_fields(self, payload: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str]]:
        """Extrait tous les champs texte d'un payload"""
        
        text_fields = []
        
        def extract_recursive(obj, path):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    extract_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    extract_recursive(item, new_path)
            elif isinstance(obj, str):
                text_fields.append((path, obj))
        
        extract_recursive(payload, prefix)
        return text_fields

    def _validate_content(self, payload: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valide le contenu du payload"""
        
        warnings = []
        suggestions = []
        
        # 1. Vérification de la présence de texte principal
        if not payload.get('text') and not payload.get('attachments') and not payload.get('blocks'):
            warnings.append("Aucun contenu principal détecté")
        
        # 2. Vérification de la longueur du texte principal
        main_text = payload.get('text', '')
        if main_text:
            if len(main_text) < 10:
                suggestions.append("Le texte principal est très court")
            elif len(main_text) > 1000:
                suggestions.append("Le texte principal est très long, considérez utiliser des attachments")
        
        # 3. Vérification des attachments
        attachments = payload.get('attachments', [])
        if len(attachments) > 10:
            warnings.append(f"Nombre élevé d'attachments: {len(attachments)}")
        
        for i, attachment in enumerate(attachments):
            if not attachment.get('fallback'):
                warnings.append(f"Attachment {i}: fallback manquant (accessibilité)")
            
            if not attachment.get('color'):
                suggestions.append(f"Attachment {i}: couleur recommandée pour une meilleure visibilité")
        
        # 4. Vérification des blocks
        blocks = payload.get('blocks', [])
        if len(blocks) > 25:
            warnings.append(f"Nombre élevé de blocks: {len(blocks)}")
        
        return {
            'warnings': warnings,
            'suggestions': suggestions
        }

    def _validate_urls(self, payload: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valide les URLs présentes dans le payload"""
        
        errors = []
        warnings = []
        
        # Extraction de toutes les URLs
        urls = self._extract_urls(payload)
        
        for field_path, url in urls:
            try:
                parsed_url = urlparse(url)
                
                # 1. Validation du schéma
                if parsed_url.scheme not in ['http', 'https']:
                    errors.append(f"Schéma URL non autorisé dans {field_path}: {parsed_url.scheme}")
                
                # 2. Validation du domaine si configuré
                if self.allowed_domains and parsed_url.netloc:
                    domain_allowed = False
                    for allowed_domain in self.allowed_domains:
                        if parsed_url.netloc == allowed_domain or parsed_url.netloc.endswith(f'.{allowed_domain}'):
                            domain_allowed = True
                            break
                    
                    if not domain_allowed and self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                        warnings.append(f"Domaine non autorisé dans {field_path}: {parsed_url.netloc}")
                
                # 3. Validation de la longueur
                if len(url) > 2048:
                    warnings.append(f"URL très longue dans {field_path}")
                
            except Exception as e:
                errors.append(f"URL invalide dans {field_path}: {url}")
        
        return {
            'errors': errors,
            'warnings': warnings
        }

    def _extract_urls(self, payload: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extrait toutes les URLs d'un payload"""
        
        urls = []
        url_fields = [
            'icon_url', 'author_link', 'author_icon', 'title_link',
            'image_url', 'thumb_url', 'footer_icon'
        ]
        
        def extract_recursive(obj, path):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if key in url_fields and isinstance(value, str):
                        urls.append((new_path, value))
                    else:
                        extract_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    extract_recursive(item, new_path)
        
        extract_recursive(payload, "")
        return urls

    def _validate_business_rules(self, payload: Dict[str, Any], tenant_id: str) -> Dict[str, List[str]]:
        """Valide les règles métier spécifiques"""
        
        warnings = []
        suggestions = []
        
        # 1. Vérification du branding
        username = payload.get('username', '')
        if username and 'spotify' not in username.lower():
            suggestions.append("Considérez inclure 'Spotify' dans le username pour le branding")
        
        # 2. Vérification du canal
        channel = payload.get('channel', '')
        if channel and not channel.startswith('#'):
            warnings.append("Le canal devrait commencer par '#' pour éviter les envois directs")
        
        # 3. Vérification de l'accessibilité
        attachments = payload.get('attachments', [])
        for i, attachment in enumerate(attachments):
            if attachment.get('color') and not attachment.get('fallback'):
                warnings.append(f"Attachment {i}: fallback requis pour l'accessibilité")
        
        return {
            'warnings': warnings,
            'suggestions': suggestions
        }

    def _validate_performance(self, payload: Dict[str, Any]) -> List[str]:
        """Valide les aspects performance du payload"""
        
        performance_warnings = []
        
        # 1. Nombre d'éléments
        attachments_count = len(payload.get('attachments', []))
        blocks_count = len(payload.get('blocks', []))
        
        if attachments_count > 5:
            performance_warnings.append(f"Nombre élevé d'attachments: {attachments_count}")
        
        if blocks_count > 10:
            performance_warnings.append(f"Nombre élevé de blocks: {blocks_count}")
        
        # 2. Complexité des blocks
        total_elements = 0
        for block in payload.get('blocks', []):
            elements = block.get('elements', [])
            fields = block.get('fields', [])
            total_elements += len(elements) + len(fields)
        
        if total_elements > 50:
            performance_warnings.append(f"Complexité élevée des blocks: {total_elements} éléments")
        
        # 3. Taille des images
        image_urls = []
        for attachment in payload.get('attachments', []):
            if attachment.get('image_url'):
                image_urls.append(attachment['image_url'])
            if attachment.get('thumb_url'):
                image_urls.append(attachment['thumb_url'])
        
        if len(image_urls) > 3:
            performance_warnings.append(f"Nombre élevé d'images: {len(image_urls)}")
        
        return performance_warnings

    def validate_template_content(self, template_content: Dict[str, Any]) -> bool:
        """Valide le contenu d'un template"""
        
        try:
            # Validation basique de la structure
            if not isinstance(template_content, dict):
                return False
            
            # Vérification de la présence de champs obligatoires pour un template
            required_fields = ['text', 'attachments', 'blocks']
            if not any(field in template_content for field in required_fields):
                return False
            
            # Validation en tant que payload Slack
            result = self.validate_slack_payload(template_content)
            
            return result.is_valid
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la validation du template: {e}")
            return False

    def get_validation_suggestions(self, payload: Dict[str, Any]) -> List[str]:
        """Retourne des suggestions d'amélioration pour un payload"""
        
        suggestions = []
        
        # Suggestions de structure
        if not payload.get('username'):
            suggestions.append("Ajoutez un username personnalisé pour identifier la source")
        
        if not payload.get('icon_emoji') and not payload.get('icon_url'):
            suggestions.append("Ajoutez une icône (emoji ou URL) pour améliorer la visibilité")
        
        # Suggestions pour les attachments
        attachments = payload.get('attachments', [])
        for i, attachment in enumerate(attachments):
            if not attachment.get('color'):
                suggestions.append(f"Attachment {i}: Ajoutez une couleur pour la sévérité")
            
            if not attachment.get('footer'):
                suggestions.append(f"Attachment {i}: Ajoutez un footer avec timestamp/source")
        
        # Suggestions pour l'accessibilité
        if attachments and not all(att.get('fallback') for att in attachments):
            suggestions.append("Ajoutez des fallbacks pour tous les attachments (accessibilité)")
        
        return suggestions

    def sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Nettoie et sécurise un payload Slack"""
        
        sanitized = payload.copy()
        
        # Nettoyage récursif des champs texte
        def sanitize_recursive(obj):
            if isinstance(obj, dict):
                return {key: sanitize_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Nettoyage HTML
                cleaned = bleach.clean(
                    obj,
                    tags=self.allowed_html_tags,
                    attributes=self.allowed_html_attrs,
                    strip=True
                )
                # Limitation de longueur
                if len(cleaned) > 5000:
                    cleaned = cleaned[:4997] + "..."
                return cleaned
            else:
                return obj
        
        return sanitize_recursive(sanitized)
