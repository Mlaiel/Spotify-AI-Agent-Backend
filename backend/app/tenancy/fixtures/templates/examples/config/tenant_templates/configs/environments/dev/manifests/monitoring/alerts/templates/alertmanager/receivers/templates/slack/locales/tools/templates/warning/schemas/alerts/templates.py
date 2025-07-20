"""
Schémas de templates d'alertes - Spotify AI Agent
Gestion avancée des templates de contenu et de formatage
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from enum import Enum
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class TemplateType(str, Enum):
    """Types de templates"""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    DISCORD = "discord"
    PAGERDUTY = "pagerduty"
    JIRA = "jira"
    SERVICENOW = "servicenow"
    CUSTOM = "custom"


class TemplateFormat(str, Enum):
    """Formats de template"""
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    JINJA2 = "jinja2"
    MUSTACHE = "mustache"
    HANDLEBARS = "handlebars"


class TemplateCategory(str, Enum):
    """Catégories de templates"""
    ALERT_NOTIFICATION = "alert_notification"
    ESCALATION = "escalation"
    RESOLUTION = "resolution"
    SUMMARY = "summary"
    REPORT = "report"
    REMINDER = "reminder"
    ACKNOWLEDGMENT = "acknowledgment"
    DIGEST = "digest"


class LocalizationSupport(str, Enum):
    """Support de localisation"""
    NONE = "none"
    BASIC = "basic"
    FULL = "full"
    ADVANCED = "advanced"


class TemplateVariable(BaseModel):
    """Variable de template"""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    data_type: str = Field("string")  # string, number, boolean, datetime, object, array
    required: bool = Field(False)
    default_value: Optional[Any] = Field(None)
    
    # Validation
    pattern: Optional[str] = Field(None)  # Regex pattern
    min_length: Optional[int] = Field(None, ge=0)
    max_length: Optional[int] = Field(None, ge=0)
    min_value: Optional[Union[int, float]] = Field(None)
    max_value: Optional[Union[int, float]] = Field(None)
    
    # Formatage
    format_string: Optional[str] = Field(None)
    transformations: List[str] = Field(default_factory=list)
    
    # Exemples et aide
    example_value: Optional[Any] = Field(None)
    help_text: Optional[str] = Field(None, max_length=1000)


class TemplateCondition(BaseModel):
    """Condition d'affichage de template"""
    
    variable_name: str = Field(..., min_length=1, max_length=100)
    operator: str = Field(...)  # eq, ne, gt, lt, contains, regex, etc.
    value: Any = Field(...)
    negate: bool = Field(False)
    
    # Conditions composées
    logical_operator: Optional[Literal["AND", "OR"]] = Field(None)
    sub_conditions: List['TemplateCondition'] = Field(default_factory=list)


class TemplateContent(BaseModel):
    """Contenu de template"""
    
    subject: Optional[str] = Field(None, max_length=1000)
    body: str = Field(..., min_length=1)
    footer: Optional[str] = Field(None, max_length=500)
    
    # Contenu conditionnel
    conditional_blocks: Dict[str, str] = Field(default_factory=dict)
    conditions: List[TemplateCondition] = Field(default_factory=list)
    
    # Métadonnées de formatage
    css_styles: Optional[str] = Field(None)
    inline_styles: Dict[str, str] = Field(default_factory=dict)
    
    # Pièces jointes et ressources
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    embedded_images: Dict[str, str] = Field(default_factory=dict)


class AlertTemplate(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Template d'alerte avancé"""
    
    # Informations de base
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field("1.0.0")
    
    # Configuration du template
    template_type: TemplateType = Field(...)
    format: TemplateFormat = Field(...)
    category: TemplateCategory = Field(...)
    
    # Contenu principal
    content: TemplateContent = Field(...)
    
    # Variables et paramètres
    variables: List[TemplateVariable] = Field(default_factory=list)
    global_variables: Dict[str, Any] = Field(default_factory=dict)
    
    # Localisation
    localization_support: LocalizationSupport = Field(LocalizationSupport.NONE)
    default_locale: str = Field("en_US")
    localized_content: Dict[str, TemplateContent] = Field(default_factory=dict)
    
    # Filtres d'application
    severity_filter: List[AlertLevel] = Field(default_factory=list)
    category_filter: List[WarningCategory] = Field(default_factory=list)
    environment_filter: List[Environment] = Field(default_factory=list)
    
    # Configuration de rendu
    render_engine: str = Field("jinja2")
    render_options: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_rules: List[str] = Field(default_factory=list)
    postprocessing_rules: List[str] = Field(default_factory=list)
    
    # Validation et test
    validated: bool = Field(False)
    validation_errors: List[str] = Field(default_factory=list)
    test_data: Optional[Dict[str, Any]] = Field(None)
    
    # État et usage
    enabled: bool = Field(True)
    usage_count: int = Field(0, ge=0)
    last_used: Optional[datetime] = Field(None)
    success_rate: Optional[float] = Field(None, ge=0, le=100)
    
    # Versioning et héritage
    parent_template_id: Optional[UUID] = Field(None)
    child_templates: List[UUID] = Field(default_factory=list)
    is_system_template: bool = Field(False)
    
    # Audit et approval
    created_by: Optional[UUID] = Field(None)
    approved_by: Optional[UUID] = Field(None)
    approved_at: Optional[datetime] = Field(None)
    approval_required: bool = Field(False)
    
    # Tags et organisation
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @validator('variables')
    def validate_variables(cls, v):
        """Valide les variables du template"""
        variable_names = [var.name for var in v]
        
        # Vérifier l'unicité des noms
        if len(variable_names) != len(set(variable_names)):
            raise ValueError('Variable names must be unique')
        
        # Vérifier les noms de variables (format valide)
        for name in variable_names:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
                raise ValueError(f'Invalid variable name: {name}')
        
        return v

    @validator('content')
    def validate_content(cls, v, values):
        """Valide le contenu du template"""
        # Vérifier la syntaxe du template selon le format
        template_format = values.get('format')
        
        if template_format == TemplateFormat.JINJA2:
            try:
                import jinja2
                jinja2.Template(v.body)
                if v.subject:
                    jinja2.Template(v.subject)
            except Exception as e:
                raise ValueError(f'Invalid Jinja2 template syntax: {str(e)}')
        
        return v

    @computed_field
    @property
    def required_variables(self) -> List[str]:
        """Liste des variables requises"""
        return [var.name for var in self.variables if var.required]

    @computed_field
    @property
    def variable_count(self) -> int:
        """Nombre de variables définies"""
        return len(self.variables)

    @computed_field
    @property
    def locales_supported(self) -> List[str]:
        """Langues supportées"""
        locales = [self.default_locale]
        locales.extend(self.localized_content.keys())
        return list(set(locales))

    def get_variable(self, name: str) -> Optional[TemplateVariable]:
        """Obtient une variable par nom"""
        for variable in self.variables:
            if variable.name == name:
                return variable
        return None

    def add_variable(self, variable: TemplateVariable):
        """Ajoute une variable au template"""
        # Vérifier l'unicité
        if any(var.name == variable.name for var in self.variables):
            raise ValueError(f'Variable {variable.name} already exists')
        
        self.variables.append(variable)

    def render(self, context: Dict[str, Any], locale: Optional[str] = None) -> Dict[str, str]:
        """Rend le template avec le contexte donné"""
        # Sélectionner le contenu approprié
        if locale and locale in self.localized_content:
            content = self.localized_content[locale]
        else:
            content = self.content
        
        # Préparer le contexte de rendu
        render_context = {**self.global_variables, **context}
        
        # Ajouter les variables par défaut
        for variable in self.variables:
            if variable.name not in render_context and variable.default_value is not None:
                render_context[variable.name] = variable.default_value
        
        # Vérifier les variables requises
        missing_vars = [
            var.name for var in self.variables
            if var.required and var.name not in render_context
        ]
        if missing_vars:
            raise ValueError(f'Missing required variables: {missing_vars}')
        
        # Rendre le template
        if self.render_engine == "jinja2":
            return self._render_jinja2(content, render_context)
        else:
            return self._render_default(content, render_context)

    def _render_jinja2(self, content: TemplateContent, context: Dict[str, Any]) -> Dict[str, str]:
        """Rend avec Jinja2"""
        import jinja2
        
        result = {}
        
        # Rendre le sujet
        if content.subject:
            subject_template = jinja2.Template(content.subject)
            result['subject'] = subject_template.render(**context)
        
        # Rendre le corps
        body_template = jinja2.Template(content.body)
        result['body'] = body_template.render(**context)
        
        # Rendre le pied de page
        if content.footer:
            footer_template = jinja2.Template(content.footer)
            result['footer'] = footer_template.render(**context)
        
        return result

    def _render_default(self, content: TemplateContent, context: Dict[str, Any]) -> Dict[str, str]:
        """Rendu par défaut avec substitution simple"""
        result = {}
        
        # Substitution simple des variables
        def substitute(text: str) -> str:
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                text = text.replace(placeholder, str(value))
            return text
        
        if content.subject:
            result['subject'] = substitute(content.subject)
        
        result['body'] = substitute(content.body)
        
        if content.footer:
            result['footer'] = substitute(content.footer)
        
        return result

    def validate_template(self, test_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Valide le template avec des données de test"""
        errors = []
        
        # Utiliser les données de test ou un contexte par défaut
        context = test_context or self.test_data or {}
        
        # Ajouter des valeurs par défaut pour les variables requises
        for variable in self.variables:
            if variable.required and variable.name not in context:
                if variable.example_value is not None:
                    context[variable.name] = variable.example_value
                else:
                    context[variable.name] = f"test_{variable.name}"
        
        try:
            # Tenter le rendu
            rendered = self.render(context)
            
            # Vérifications de base
            if not rendered.get('body'):
                errors.append('Template body is empty after rendering')
            
            # Vérifier la longueur du sujet si présent
            if rendered.get('subject') and len(rendered['subject']) > 1000:
                errors.append('Subject is too long (>1000 characters)')
            
        except Exception as e:
            errors.append(f'Template rendering failed: {str(e)}')
        
        self.validation_errors = errors
        self.validated = len(errors) == 0
        
        return errors

    def clone(self, new_name: str, new_version: str = "1.0.0") -> 'AlertTemplate':
        """Clone le template avec un nouveau nom"""
        cloned_data = self.model_dump(exclude={'id', 'created_at', 'updated_at'})
        cloned_data.update({
            'name': new_name,
            'version': new_version,
            'parent_template_id': self.id,
            'usage_count': 0,
            'last_used': None,
            'success_rate': None,
            'validated': False,
            'validation_errors': []
        })
        
        return AlertTemplate(**cloned_data)


class TemplateLibrary(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Bibliothèque de templates"""
    
    library_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration
    version: str = Field("1.0.0")
    is_system_library: bool = Field(False)
    public: bool = Field(False)
    
    # Templates inclus
    template_ids: List[UUID] = Field(default_factory=list)
    template_categories: Dict[str, List[UUID]] = Field(default_factory=dict)
    
    # Organisation
    tags: Set[str] = Field(default_factory=set)
    maintainer_ids: List[UUID] = Field(default_factory=list)
    
    # État
    enabled: bool = Field(True)
    total_templates: int = Field(0, ge=0)
    active_templates: int = Field(0, ge=0)
    
    @computed_field
    @property
    def utilization_rate(self) -> float:
        """Taux d'utilisation de la bibliothèque"""
        if self.total_templates == 0:
            return 0.0
        return (self.active_templates / self.total_templates) * 100

    def add_template(self, template_id: UUID, category: Optional[str] = None):
        """Ajoute un template à la bibliothèque"""
        if template_id not in self.template_ids:
            self.template_ids.append(template_id)
            self.total_templates += 1
            
            if category:
                if category not in self.template_categories:
                    self.template_categories[category] = []
                self.template_categories[category].append(template_id)

    def remove_template(self, template_id: UUID):
        """Retire un template de la bibliothèque"""
        if template_id in self.template_ids:
            self.template_ids.remove(template_id)
            self.total_templates -= 1
            
            # Retirer des catégories
            for category, templates in self.template_categories.items():
                if template_id in templates:
                    templates.remove(template_id)


class TemplateUsage(BaseSchema, TimestampMixin, TenantMixin):
    """Usage d'un template"""
    
    usage_id: UUID = Field(default_factory=uuid4)
    template_id: UUID = Field(...)
    alert_id: UUID = Field(...)
    
    # Contexte d'usage
    render_context: Dict[str, Any] = Field(...)
    locale_used: Optional[str] = Field(None)
    
    # Résultat du rendu
    rendered_content: Dict[str, str] = Field(...)
    render_duration_ms: Optional[float] = Field(None, ge=0)
    
    # Livraison
    delivery_successful: bool = Field(False)
    delivery_attempts: int = Field(1, ge=1)
    delivery_errors: List[str] = Field(default_factory=list)
    
    # Feedback
    user_feedback: Optional[str] = Field(None)
    satisfaction_score: Optional[int] = Field(None, ge=1, le=5)


# Conditions de template avec forward reference
TemplateCondition.model_rebuild()


__all__ = [
    'TemplateType', 'TemplateFormat', 'TemplateCategory', 'LocalizationSupport',
    'TemplateVariable', 'TemplateCondition', 'TemplateContent', 'AlertTemplate',
    'TemplateLibrary', 'TemplateUsage'
]
