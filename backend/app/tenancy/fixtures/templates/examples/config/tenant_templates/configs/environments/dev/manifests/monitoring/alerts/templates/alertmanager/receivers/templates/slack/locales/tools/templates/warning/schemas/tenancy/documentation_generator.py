#!/usr/bin/env python3
"""
Documentation Management System
==============================

Système complet de gestion de documentation pour les schémas tenancy
avec génération automatique, versioning et support multi-langue.

Créé par: Fahed Mlaiel
Rôles: Architecte Principal, Expert ML, Consultant DevOps, Spécialiste Sécurité,
       Expert Cloud, Analyste Performance, Consultant Compliance
"""

import json
import yaml
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import inspect

# Import des schémas pour documentation
from .tenant_config_schema import TenantConfigSchema, TenantType, TenantStatus
from .alert_schema import AlertSchema, TenantAlertSchema, AlertSeverity
from .warning_schema import WarningSchema, TenantWarningSchema, WarningSeverity
from .notification_schema import NotificationSchema, NotificationChannel
from .monitoring_schema import MonitoringConfigSchema, MonitoringMetric
from .compliance_schema import ComplianceSchema, ComplianceStandard
from .performance_schema import PerformanceMetricsSchema, PerformanceBaseline


class DocumentationType(Enum):
    """Types de documentation disponibles."""
    API = "api"
    SCHEMA = "schema"
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    REFERENCE = "reference"
    CHANGELOG = "changelog"
    TROUBLESHOOTING = "troubleshooting"


class OutputFormat(Enum):
    """Formats de sortie pour la documentation."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    OPENAPI = "openapi"
    CONFLUENCE = "confluence"


@dataclass
class DocumentationConfig:
    """Configuration pour la génération de documentation."""
    output_formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.MARKDOWN])
    languages: List[str] = field(default_factory=lambda: ["en", "fr", "de"])
    include_examples: bool = True
    include_diagrams: bool = True
    include_schemas: bool = True
    include_validation_rules: bool = True
    version: str = "1.0.0"
    author: str = "Fahed Mlaiel"
    organization: str = "Spotify AI Agent Team"


@dataclass
class DocumentSection:
    """Section d'un document."""
    title: str
    content: str
    level: int = 1
    section_type: str = "content"
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['DocumentSection'] = field(default_factory=list)


@dataclass
class SchemaDocumentation:
    """Documentation d'un schéma."""
    schema_name: str
    schema_class: type
    description: str
    fields: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    validation_rules: List[str]
    relationships: List[str]
    use_cases: List[str]
    best_practices: List[str]


class BaseDocumentGenerator(ABC):
    """Classe de base pour les générateurs de documentation."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate(self, content: Any, output_path: Path) -> str:
        """Génère la documentation dans le format spécifique."""
        pass


class MarkdownGenerator(BaseDocumentGenerator):
    """Générateur de documentation Markdown."""
    
    def generate(self, sections: List[DocumentSection], output_path: Path) -> str:
        """Génère un document Markdown."""
        
        markdown_content = []
        
        # En-tête du document
        markdown_content.append(f"# Tenancy Schema Documentation")
        markdown_content.append(f"")
        markdown_content.append(f"**Version:** {self.config.version}")
        markdown_content.append(f"**Author:** {self.config.author}")
        markdown_content.append(f"**Organization:** {self.config.organization}")
        markdown_content.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        markdown_content.append(f"")
        
        # Table des matières
        markdown_content.append("## Table of Contents")
        markdown_content.append("")
        for section in sections:
            indent = "  " * (section.level - 1)
            markdown_content.append(f"{indent}- [{section.title}](#{self._slugify(section.title)})")
            for subsection in section.subsections:
                sub_indent = "  " * section.level
                markdown_content.append(f"{sub_indent}- [{subsection.title}](#{self._slugify(subsection.title)})")
        markdown_content.append("")
        
        # Contenu des sections
        for section in sections:
            markdown_content.extend(self._render_section(section))
        
        final_content = "\n".join(markdown_content)
        
        # Sauvegarder si chemin fourni
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
        
        return final_content
    
    def _render_section(self, section: DocumentSection) -> List[str]:
        """Rend une section en Markdown."""
        content = []
        
        # Titre de section
        header_prefix = "#" * (section.level + 1)
        content.append(f"{header_prefix} {section.title}")
        content.append("")
        
        # Contenu principal
        if section.content:
            content.append(section.content)
            content.append("")
        
        # Métadonnées spéciales
        if section.section_type == "schema" and section.metadata:
            content.extend(self._render_schema_metadata(section.metadata))
        elif section.section_type == "api" and section.metadata:
            content.extend(self._render_api_metadata(section.metadata))
        
        # Sous-sections
        for subsection in section.subsections:
            content.extend(self._render_section(subsection))
        
        return content
    
    def _render_schema_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Rend les métadonnées d'un schéma."""
        content = []
        
        # Champs du schéma
        if "fields" in metadata:
            content.append("### Fields")
            content.append("")
            content.append("| Field | Type | Required | Description |")
            content.append("|-------|------|----------|-------------|")
            
            for field in metadata["fields"]:
                name = field.get("name", "")
                field_type = field.get("type", "")
                required = "✅" if field.get("required", False) else "❌"
                description = field.get("description", "")
                content.append(f"| `{name}` | `{field_type}` | {required} | {description} |")
            content.append("")
        
        # Exemples
        if "examples" in metadata:
            content.append("### Examples")
            content.append("")
            for i, example in enumerate(metadata["examples"], 1):
                content.append(f"#### Example {i}")
                content.append("")
                content.append("```json")
                content.append(json.dumps(example, indent=2))
                content.append("```")
                content.append("")
        
        # Règles de validation
        if "validation_rules" in metadata:
            content.append("### Validation Rules")
            content.append("")
            for rule in metadata["validation_rules"]:
                content.append(f"- {rule}")
            content.append("")
        
        return content
    
    def _render_api_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Rend les métadonnées d'une API."""
        content = []
        
        # Endpoints
        if "endpoints" in metadata:
            content.append("### Endpoints")
            content.append("")
            
            for endpoint in metadata["endpoints"]:
                method = endpoint.get("method", "GET")
                path = endpoint.get("path", "")
                description = endpoint.get("description", "")
                
                content.append(f"#### `{method} {path}`")
                content.append("")
                content.append(description)
                content.append("")
                
                # Paramètres
                if "parameters" in endpoint:
                    content.append("**Parameters:**")
                    content.append("")
                    content.append("| Name | Type | Required | Description |")
                    content.append("|------|------|----------|-------------|")
                    
                    for param in endpoint["parameters"]:
                        name = param.get("name", "")
                        param_type = param.get("type", "")
                        required = "✅" if param.get("required", False) else "❌"
                        description = param.get("description", "")
                        content.append(f"| `{name}` | `{param_type}` | {required} | {description} |")
                    content.append("")
                
                # Réponses
                if "responses" in endpoint:
                    content.append("**Responses:**")
                    content.append("")
                    
                    for response in endpoint["responses"]:
                        status = response.get("status", "200")
                        description = response.get("description", "")
                        content.append(f"- **{status}**: {description}")
                    content.append("")
        
        return content
    
    def _slugify(self, text: str) -> str:
        """Convertit un texte en slug pour les liens Markdown."""
        return re.sub(r'[^\w\s-]', '', text).strip().lower().replace(' ', '-')


class HTMLGenerator(BaseDocumentGenerator):
    """Générateur de documentation HTML."""
    
    def generate(self, sections: List[DocumentSection], output_path: Path) -> str:
        """Génère un document HTML."""
        
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tenancy Schema Documentation</title>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .toc {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    border-left: 4px solid #667eea;
                    background-color: #fafafa;
                }}
                .schema-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .schema-table th, .schema-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .schema-table th {{
                    background-color: #667eea;
                    color: white;
                }}
                .code-block {{
                    background-color: #f4f4f4;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 15px;
                    overflow-x: auto;
                    font-family: 'Courier New', monospace;
                }}
                .example {{
                    background-color: #e8f5e8;
                    border-left: 4px solid #28a745;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .warning {{
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .nav-link {{
                    color: #667eea;
                    text-decoration: none;
                }}
                .nav-link:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Tenancy Schema Documentation</h1>
                <p><strong>Version:</strong> {version}</p>
                <p><strong>Author:</strong> {author}</p>
                <p><strong>Organization:</strong> {organization}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
            
            <div class="toc">
                <h2>Table of Contents</h2>
                {toc_html}
            </div>
            
            {sections_html}
        </body>
        </html>
        """
        
        # Générer la table des matières
        toc_html = self._generate_toc_html(sections)
        
        # Générer le contenu des sections
        sections_html = ""
        for section in sections:
            sections_html += self._render_section_html(section)
        
        # Remplir le template
        html_content = html_template.format(
            version=self.config.version,
            author=self.config.author,
            organization=self.config.organization,
            timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            toc_html=toc_html,
            sections_html=sections_html
        )
        
        # Sauvegarder si chemin fourni
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    def _generate_toc_html(self, sections: List[DocumentSection]) -> str:
        """Génère la table des matières HTML."""
        toc = ["<ul>"]
        
        for section in sections:
            section_id = self._slugify(section.title)
            toc.append(f'<li><a href="#{section_id}" class="nav-link">{section.title}</a>')
            
            if section.subsections:
                toc.append("<ul>")
                for subsection in section.subsections:
                    subsection_id = self._slugify(subsection.title)
                    toc.append(f'<li><a href="#{subsection_id}" class="nav-link">{subsection.title}</a></li>')
                toc.append("</ul>")
            
            toc.append("</li>")
        
        toc.append("</ul>")
        return "\n".join(toc)
    
    def _render_section_html(self, section: DocumentSection) -> str:
        """Rend une section en HTML."""
        section_id = self._slugify(section.title)
        
        html = f"""
        <div class="section" id="{section_id}">
            <h{section.level + 1}>{section.title}</h{section.level + 1}>
            {self._process_content_html(section.content)}
        """
        
        # Métadonnées spéciales
        if section.section_type == "schema" and section.metadata:
            html += self._render_schema_metadata_html(section.metadata)
        
        # Sous-sections
        for subsection in section.subsections:
            html += self._render_section_html(subsection)
        
        html += "</div>"
        return html
    
    def _process_content_html(self, content: str) -> str:
        """Traite le contenu pour l'HTML."""
        if not content:
            return ""
        
        # Conversion simple Markdown vers HTML
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
        content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
        
        # Convertir les listes
        lines = content.split('\n')
        processed_lines = []
        in_list = False
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    processed_lines.append('<ul>')
                    in_list = True
                processed_lines.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    processed_lines.append('</ul>')
                    in_list = False
                if line.strip():
                    processed_lines.append(f'<p>{line}</p>')
        
        if in_list:
            processed_lines.append('</ul>')
        
        return '\n'.join(processed_lines)
    
    def _render_schema_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """Rend les métadonnées d'un schéma en HTML."""
        html = ""
        
        # Champs du schéma
        if "fields" in metadata:
            html += "<h3>Fields</h3>"
            html += '<table class="schema-table">'
            html += '<thead><tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr></thead>'
            html += '<tbody>'
            
            for field in metadata["fields"]:
                name = field.get("name", "")
                field_type = field.get("type", "")
                required = "✅" if field.get("required", False) else "❌"
                description = field.get("description", "")
                html += f'<tr><td><code>{name}</code></td><td><code>{field_type}</code></td><td>{required}</td><td>{description}</td></tr>'
            
            html += '</tbody></table>'
        
        # Exemples
        if "examples" in metadata:
            html += "<h3>Examples</h3>"
            for i, example in enumerate(metadata["examples"], 1):
                html += f"<h4>Example {i}</h4>"
                html += '<div class="code-block">'
                html += f'<pre>{json.dumps(example, indent=2)}</pre>'
                html += '</div>'
        
        return html
    
    def _slugify(self, text: str) -> str:
        """Convertit un texte en slug pour les IDs HTML."""
        return re.sub(r'[^\w\s-]', '', text).strip().lower().replace(' ', '-')


class SchemaIntrospector:
    """Introspecteur pour analyser les schémas et extraire la documentation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_schema_documentation(self, schema_class: type) -> SchemaDocumentation:
        """Extrait la documentation d'un schéma Pydantic."""
        
        try:
            # Informations de base
            schema_name = schema_class.__name__
            description = self._extract_description(schema_class)
            
            # Analyser les champs
            fields = self._extract_fields(schema_class)
            
            # Générer des exemples
            examples = self._generate_examples(schema_class)
            
            # Extraire les règles de validation
            validation_rules = self._extract_validation_rules(schema_class)
            
            # Analyser les relations
            relationships = self._extract_relationships(schema_class)
            
            # Cas d'usage et bonnes pratiques
            use_cases = self._extract_use_cases(schema_class)
            best_practices = self._extract_best_practices(schema_class)
            
            return SchemaDocumentation(
                schema_name=schema_name,
                schema_class=schema_class,
                description=description,
                fields=fields,
                examples=examples,
                validation_rules=validation_rules,
                relationships=relationships,
                use_cases=use_cases,
                best_practices=best_practices
            )
        
        except Exception as e:
            self.logger.error(f"Error extracting documentation for {schema_class}: {e}")
            return SchemaDocumentation(
                schema_name=schema_class.__name__,
                schema_class=schema_class,
                description="Documentation extraction failed",
                fields=[],
                examples=[],
                validation_rules=[],
                relationships=[],
                use_cases=[],
                best_practices=[]
            )
    
    def _extract_description(self, schema_class: type) -> str:
        """Extrait la description d'un schéma."""
        docstring = inspect.getdoc(schema_class)
        if docstring:
            # Prendre la première ligne comme description
            return docstring.split('\n')[0].strip()
        return f"Schema for {schema_class.__name__}"
    
    def _extract_fields(self, schema_class: type) -> List[Dict[str, Any]]:
        """Extrait les champs d'un schéma Pydantic."""
        fields = []
        
        try:
            # Utiliser le schéma Pydantic pour obtenir les champs
            schema_dict = schema_class.schema()
            properties = schema_dict.get("properties", {})
            required_fields = schema_dict.get("required", [])
            
            for field_name, field_info in properties.items():
                field_doc = {
                    "name": field_name,
                    "type": field_info.get("type", "unknown"),
                    "required": field_name in required_fields,
                    "description": field_info.get("description", ""),
                    "default": field_info.get("default"),
                    "format": field_info.get("format"),
                    "enum": field_info.get("enum"),
                    "minimum": field_info.get("minimum"),
                    "maximum": field_info.get("maximum")
                }
                
                # Nettoyer les valeurs None
                field_doc = {k: v for k, v in field_doc.items() if v is not None}
                fields.append(field_doc)
        
        except Exception as e:
            self.logger.error(f"Error extracting fields from {schema_class}: {e}")
        
        return fields
    
    def _generate_examples(self, schema_class: type) -> List[Dict[str, Any]]:
        """Génère des exemples pour un schéma."""
        examples = []
        
        try:
            # Exemple minimal
            minimal_example = self._create_minimal_example(schema_class)
            if minimal_example:
                examples.append({
                    "name": "minimal",
                    "description": "Minimal required configuration",
                    "data": minimal_example
                })
            
            # Exemple complet
            complete_example = self._create_complete_example(schema_class)
            if complete_example:
                examples.append({
                    "name": "complete",
                    "description": "Complete configuration with all optional fields",
                    "data": complete_example
                })
        
        except Exception as e:
            self.logger.error(f"Error generating examples for {schema_class}: {e}")
        
        return examples
    
    def _create_minimal_example(self, schema_class: type) -> Optional[Dict[str, Any]]:
        """Crée un exemple minimal avec uniquement les champs requis."""
        try:
            schema_dict = schema_class.schema()
            properties = schema_dict.get("properties", {})
            required_fields = schema_dict.get("required", [])
            
            example = {}
            
            for field_name in required_fields:
                if field_name in properties:
                    field_info = properties[field_name]
                    example[field_name] = self._generate_field_value(field_name, field_info)
            
            return example if example else None
        
        except Exception:
            return None
    
    def _create_complete_example(self, schema_class: type) -> Optional[Dict[str, Any]]:
        """Crée un exemple complet avec tous les champs."""
        try:
            schema_dict = schema_class.schema()
            properties = schema_dict.get("properties", {})
            
            example = {}
            
            for field_name, field_info in properties.items():
                example[field_name] = self._generate_field_value(field_name, field_info)
            
            return example if example else None
        
        except Exception:
            return None
    
    def _generate_field_value(self, field_name: str, field_info: Dict[str, Any]) -> Any:
        """Génère une valeur d'exemple pour un champ."""
        field_type = field_info.get("type", "string")
        field_format = field_info.get("format")
        enum_values = field_info.get("enum")
        default = field_info.get("default")
        
        # Utiliser la valeur par défaut si disponible
        if default is not None:
            return default
        
        # Utiliser les valeurs enum si disponibles
        if enum_values:
            return enum_values[0]
        
        # Génération basée sur le nom du champ
        field_name_lower = field_name.lower()
        
        if "id" in field_name_lower:
            return f"example_{field_name_lower}"
        elif "email" in field_name_lower:
            return "user@example.com"
        elif "name" in field_name_lower:
            return f"Example {field_name.replace('_', ' ').title()}"
        elif "url" in field_name_lower:
            return "https://example.com"
        elif "date" in field_name_lower or "time" in field_name_lower:
            return datetime.now(timezone.utc).isoformat()
        
        # Génération basée sur le type
        if field_type == "string":
            if field_format == "email":
                return "user@example.com"
            elif field_format == "uri":
                return "https://example.com"
            elif field_format == "date-time":
                return datetime.now(timezone.utc).isoformat()
            else:
                return f"example_{field_name}"
        elif field_type == "integer":
            return 100
        elif field_type == "number":
            return 99.5
        elif field_type == "boolean":
            return True
        elif field_type == "array":
            return ["example_item"]
        elif field_type == "object":
            return {"key": "value"}
        else:
            return f"example_{field_name}"
    
    def _extract_validation_rules(self, schema_class: type) -> List[str]:
        """Extrait les règles de validation d'un schéma."""
        rules = []
        
        try:
            schema_dict = schema_class.schema()
            properties = schema_dict.get("properties", {})
            required_fields = schema_dict.get("required", [])
            
            # Champs requis
            if required_fields:
                rules.append(f"Required fields: {', '.join(required_fields)}")
            
            # Contraintes par champ
            for field_name, field_info in properties.items():
                field_rules = []
                
                if field_info.get("minimum") is not None:
                    field_rules.append(f"minimum: {field_info['minimum']}")
                
                if field_info.get("maximum") is not None:
                    field_rules.append(f"maximum: {field_info['maximum']}")
                
                if field_info.get("minLength") is not None:
                    field_rules.append(f"min length: {field_info['minLength']}")
                
                if field_info.get("maxLength") is not None:
                    field_rules.append(f"max length: {field_info['maxLength']}")
                
                if field_info.get("pattern"):
                    field_rules.append(f"pattern: {field_info['pattern']}")
                
                if field_info.get("enum"):
                    field_rules.append(f"allowed values: {', '.join(map(str, field_info['enum']))}")
                
                if field_rules:
                    rules.append(f"{field_name}: {', '.join(field_rules)}")
        
        except Exception as e:
            self.logger.error(f"Error extracting validation rules from {schema_class}: {e}")
        
        return rules
    
    def _extract_relationships(self, schema_class: type) -> List[str]:
        """Extrait les relations avec d'autres schémas."""
        relationships = []
        
        try:
            # Analyser les annotations de type pour trouver les références
            annotations = getattr(schema_class, '__annotations__', {})
            
            for field_name, field_type in annotations.items():
                type_str = str(field_type)
                
                # Rechercher les références à d'autres schémas
                if "Schema" in type_str:
                    relationships.append(f"{field_name} references {type_str}")
                elif "List" in type_str and "Schema" in type_str:
                    relationships.append(f"{field_name} contains a list of {type_str}")
        
        except Exception as e:
            self.logger.error(f"Error extracting relationships from {schema_class}: {e}")
        
        return relationships
    
    def _extract_use_cases(self, schema_class: type) -> List[str]:
        """Extrait les cas d'usage d'un schéma."""
        schema_name = schema_class.__name__.lower()
        use_cases = []
        
        if "tenant" in schema_name:
            use_cases.extend([
                "Multi-tenant application configuration",
                "SaaS platform tenant management",
                "Customer-specific customization settings"
            ])
        
        if "alert" in schema_name:
            use_cases.extend([
                "System monitoring and alerting",
                "Performance threshold management",
                "Incident response automation"
            ])
        
        if "monitoring" in schema_name:
            use_cases.extend([
                "Application performance monitoring",
                "Infrastructure health tracking",
                "Business metrics collection"
            ])
        
        if "compliance" in schema_name:
            use_cases.extend([
                "Regulatory compliance management",
                "Data governance enforcement",
                "Audit trail maintenance"
            ])
        
        return use_cases or ["General purpose schema"]
    
    def _extract_best_practices(self, schema_class: type) -> List[str]:
        """Extrait les bonnes pratiques pour un schéma."""
        best_practices = [
            "Always validate input data before processing",
            "Use appropriate data types for each field",
            "Provide meaningful descriptions for all fields",
            "Include examples in your documentation",
            "Test schema validation with edge cases",
            "Version your schemas for backward compatibility",
            "Use consistent naming conventions",
            "Document any business rules or constraints"
        ]
        
        schema_name = schema_class.__name__.lower()
        
        if "tenant" in schema_name:
            best_practices.extend([
                "Ensure tenant isolation at all levels",
                "Implement proper access controls",
                "Consider data residency requirements",
                "Plan for tenant-specific customizations"
            ])
        
        if "alert" in schema_name:
            best_practices.extend([
                "Configure appropriate escalation policies",
                "Avoid alert fatigue with proper thresholds",
                "Include actionable information in alerts",
                "Test alert delivery mechanisms regularly"
            ])
        
        return best_practices


class DocumentationManager:
    """Gestionnaire principal de documentation."""
    
    def __init__(self, config: DocumentationConfig):
        self.config = config
        self.introspector = SchemaIntrospector()
        self.generators = {
            OutputFormat.MARKDOWN: MarkdownGenerator(config),
            OutputFormat.HTML: HTMLGenerator(config)
        }
        self.logger = logging.getLogger(__name__)
    
    async def generate_complete_documentation(self, output_dir: Path) -> Dict[str, List[Path]]:
        """Génère la documentation complète pour tous les schémas."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = {format.value: [] for format in self.config.output_formats}
        
        try:
            # Schémas à documenter
            schemas = [
                TenantConfigSchema,
                AlertSchema,
                WarningSchema,
                NotificationSchema,
                MonitoringConfigSchema,
                ComplianceSchema,
                PerformanceMetricsSchema
            ]
            
            # Générer la documentation pour chaque schéma
            for schema_class in schemas:
                schema_doc = self.introspector.extract_schema_documentation(schema_class)
                sections = self._create_schema_sections(schema_doc)
                
                # Générer dans tous les formats demandés
                for format in self.config.output_formats:
                    if format in self.generators:
                        schema_name = schema_class.__name__.lower()
                        output_file = output_dir / f"{schema_name}.{format.value}"
                        
                        self.generators[format].generate(sections, output_file)
                        generated_files[format.value].append(output_file)
            
            # Générer l'index principal
            await self._generate_index(schemas, output_dir, generated_files)
            
            # Générer la documentation API
            await self._generate_api_documentation(output_dir, generated_files)
            
            # Générer le guide de démarrage rapide
            await self._generate_quick_start_guide(output_dir, generated_files)
        
        except Exception as e:
            self.logger.error(f"Error generating complete documentation: {e}")
        
        return generated_files
    
    def _create_schema_sections(self, schema_doc: SchemaDocumentation) -> List[DocumentSection]:
        """Crée les sections de documentation pour un schéma."""
        
        sections = []
        
        # Section principale
        main_section = DocumentSection(
            title=f"{schema_doc.schema_name} Schema",
            content=schema_doc.description,
            level=1,
            section_type="schema",
            metadata={
                "fields": schema_doc.fields,
                "examples": [ex["data"] for ex in schema_doc.examples],
                "validation_rules": schema_doc.validation_rules
            }
        )
        
        # Sous-sections
        if schema_doc.use_cases:
            use_cases_section = DocumentSection(
                title="Use Cases",
                content="\n".join([f"- {use_case}" for use_case in schema_doc.use_cases]),
                level=2
            )
            main_section.subsections.append(use_cases_section)
        
        if schema_doc.best_practices:
            best_practices_section = DocumentSection(
                title="Best Practices",
                content="\n".join([f"- {practice}" for practice in schema_doc.best_practices]),
                level=2
            )
            main_section.subsections.append(best_practices_section)
        
        if schema_doc.relationships:
            relationships_section = DocumentSection(
                title="Relationships",
                content="\n".join([f"- {relationship}" for relationship in schema_doc.relationships]),
                level=2
            )
            main_section.subsections.append(relationships_section)
        
        sections.append(main_section)
        return sections
    
    async def _generate_index(self, schemas: List[type], output_dir: Path, 
                            generated_files: Dict[str, List[Path]]) -> None:
        """Génère l'index principal de la documentation."""
        
        index_sections = []
        
        # Section d'introduction
        intro_section = DocumentSection(
            title="Introduction",
            content="""
This documentation covers the complete tenancy schema system for the Spotify AI Agent platform.
The schemas provide a robust foundation for multi-tenant applications with enterprise-grade
features including monitoring, alerting, compliance, and performance management.

## Key Features

- **Multi-tenant Architecture**: Complete tenant isolation and management
- **Enterprise Compliance**: GDPR, SOC2, HIPAA, ISO27001 support
- **Advanced Monitoring**: Comprehensive metrics and alerting
- **Machine Learning Integration**: Predictive analytics and anomaly detection
- **High Performance**: Optimized for scale and reliability
- **Security First**: Built-in security controls and audit trails

## Getting Started

For a quick introduction, see the [Quick Start Guide](quick_start_guide.md).
For detailed API documentation, see the [API Reference](api_documentation.md).
            """,
            level=1
        )
        index_sections.append(intro_section)
        
        # Section des schémas
        schemas_section = DocumentSection(
            title="Available Schemas",
            content="The following schemas are available in this system:",
            level=1
        )
        
        for schema_class in schemas:
            schema_name = schema_class.__name__
            schema_file = f"{schema_name.lower()}.md"
            
            schema_subsection = DocumentSection(
                title=schema_name,
                content=f"""
Schema for {schema_name.replace('Schema', '').replace('Config', ' Configuration')}.

**Documentation**: [{schema_file}]({schema_file})

**Key Features**:
- Industry-standard validation
- Comprehensive field coverage
- Built-in security controls
- Performance optimized
                """,
                level=2
            )
            schemas_section.subsections.append(schema_subsection)
        
        index_sections.append(schemas_section)
        
        # Générer l'index dans tous les formats
        for format in self.config.output_formats:
            if format in self.generators:
                index_file = output_dir / f"index.{format.value}"
                self.generators[format].generate(index_sections, index_file)
                generated_files[format.value].append(index_file)
    
    async def _generate_api_documentation(self, output_dir: Path, 
                                        generated_files: Dict[str, List[Path]]) -> None:
        """Génère la documentation API."""
        
        api_sections = []
        
        # Section principale API
        api_section = DocumentSection(
            title="API Reference",
            content="""
This section provides detailed API documentation for the tenancy schema system.
All endpoints support JSON request/response format with comprehensive validation.

## Authentication

All API endpoints require authentication using JWT tokens or API keys.
Include the authentication token in the Authorization header:

```
Authorization: Bearer <your-token>
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid field value",
    "details": {
      "field": "tenant_id",
      "reason": "Required field missing"
    }
  }
}
```
            """,
            level=1,
            section_type="api"
        )
        
        # Endpoints principaux
        endpoints_data = [
            {
                "method": "POST",
                "path": "/api/v1/tenants",
                "description": "Create a new tenant configuration",
                "parameters": [
                    {"name": "tenant_id", "type": "string", "required": True, "description": "Unique tenant identifier"},
                    {"name": "tenant_name", "type": "string", "required": True, "description": "Display name for the tenant"},
                    {"name": "admin_email", "type": "string", "required": True, "description": "Administrator email address"}
                ],
                "responses": [
                    {"status": "201", "description": "Tenant created successfully"},
                    {"status": "400", "description": "Invalid request data"},
                    {"status": "409", "description": "Tenant already exists"}
                ]
            },
            {
                "method": "GET",
                "path": "/api/v1/tenants/{tenant_id}",
                "description": "Retrieve tenant configuration",
                "parameters": [
                    {"name": "tenant_id", "type": "string", "required": True, "description": "Tenant identifier"}
                ],
                "responses": [
                    {"status": "200", "description": "Tenant configuration returned"},
                    {"status": "404", "description": "Tenant not found"}
                ]
            },
            {
                "method": "POST",
                "path": "/api/v1/tenants/{tenant_id}/alerts",
                "description": "Create a new alert configuration",
                "parameters": [
                    {"name": "tenant_id", "type": "string", "required": True, "description": "Tenant identifier"},
                    {"name": "alert_config", "type": "object", "required": True, "description": "Alert configuration object"}
                ],
                "responses": [
                    {"status": "201", "description": "Alert created successfully"},
                    {"status": "400", "description": "Invalid alert configuration"}
                ]
            }
        ]
        
        api_section.metadata = {"endpoints": endpoints_data}
        api_sections.append(api_section)
        
        # Générer la documentation API
        for format in self.config.output_formats:
            if format in self.generators:
                api_file = output_dir / f"api_documentation.{format.value}"
                self.generators[format].generate(api_sections, api_file)
                generated_files[format.value].append(api_file)
    
    async def _generate_quick_start_guide(self, output_dir: Path, 
                                        generated_files: Dict[str, List[Path]]) -> None:
        """Génère le guide de démarrage rapide."""
        
        guide_sections = []
        
        # Guide principal
        guide_section = DocumentSection(
            title="Quick Start Guide",
            content="""
This guide will help you get started with the tenancy schema system in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- Pydantic 2.0 or higher
- FastAPI (optional, for API integration)

## Installation

Install the required dependencies:

```bash
pip install pydantic fastapi uvicorn
```

## Basic Usage

### 1. Create a Tenant Configuration

```python
from tenancy.schemas import TenantConfigSchema, TenantType

# Create a basic tenant configuration
tenant_config = TenantConfigSchema(
    tenant_id="my_first_tenant",
    tenant_name="My First Tenant",
    tenant_type=TenantType.PROFESSIONAL,
    admin_email="admin@example.com",
    country_code="US"
)

print(f"Created tenant: {tenant_config.tenant_name}")
```

### 2. Set Up Monitoring

```python
from tenancy.schemas import MonitoringConfigSchema, MonitoringMetric

# Create monitoring configuration
monitoring_config = MonitoringConfigSchema(
    tenant_id="my_first_tenant",
    name="Basic Monitoring",
    metrics=[
        MonitoringMetric(
            metric_id="cpu_usage",
            name="cpu_usage_percent",
            display_name="CPU Usage",
            metric_type="gauge",
            source="system",
            warning_threshold=70.0,
            critical_threshold=90.0
        )
    ]
)
```

### 3. Configure Alerts

```python
from tenancy.schemas import AlertSchema, AlertSeverity

# Create an alert
alert_config = AlertSchema(
    tenant_id="my_first_tenant",
    name="high_cpu_alert",
    title="High CPU Usage Alert",
    severity=AlertSeverity.HIGH,
    category="performance",
    conditions=[{
        "metric_name": "cpu_usage_percent",
        "operator": "gt",
        "threshold": 80.0,
        "duration_minutes": 5
    }],
    notification_channels=["email"],
    recipients=["admin@example.com"]
)
```

## Advanced Features

### Schema Factory

Use the Schema Factory for automated configuration generation:

```python
from tenancy.schemas import SchemaFactory, SchemaBuilderConfig, TenantType

# Configure the factory
config = SchemaBuilderConfig(
    tenant_type=TenantType.ENTERPRISE,
    compliance_standards=["gdpr", "soc2"],
    auto_optimize=True
)

# Create the factory
factory = SchemaFactory(config)

# Generate complete tenant configuration
tenant_config = factory.create_tenant_config(
    tenant_id="enterprise_tenant",
    tenant_name="Enterprise Customer",
    admin_email="admin@enterprise.com"
)
```

### Analytics Engine

Enable advanced analytics and insights:

```python
from tenancy.schemas import AnalyticsEngine, AnalyticsType

# Create analytics engine
engine = AnalyticsEngine()

# Run analysis on your metrics data
results = await engine.run_complete_analysis(
    metrics_data=your_metrics_data,
    tenant_config=tenant_config,
    analysis_types=[AnalyticsType.DESCRIPTIVE, AnalyticsType.PREDICTIVE]
)

# Get insights and recommendations
insights = results["insights"]
recommendations = results["recommendations"]
```

## Next Steps

- Explore the [complete schema documentation](index.md)
- Check out the [API reference](api_documentation.md)
- Learn about [compliance features](compliance_schema.md)
- Set up [advanced monitoring](monitoring_schema.md)

## Support

For questions and support:
- Email: support@spotify-ai-agent.com
- Documentation: [Full Documentation](index.md)
- Issues: GitHub Issues (if available)
            """,
            level=1
        )
        
        guide_sections.append(guide_section)
        
        # Générer le guide
        for format in self.config.output_formats:
            if format in self.generators:
                guide_file = output_dir / f"quick_start_guide.{format.value}"
                self.generators[format].generate(guide_sections, guide_file)
                generated_files[format.value].append(guide_file)


# Exemple d'utilisation
async def main():
    """Exemple d'utilisation du système de documentation."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Configuration de la documentation
    config = DocumentationConfig(
        output_formats=[OutputFormat.MARKDOWN, OutputFormat.HTML],
        languages=["en", "fr", "de"],
        include_examples=True,
        include_diagrams=True,
        version="1.0.0",
        author="Fahed Mlaiel"
    )
    
    # Gestionnaire de documentation
    doc_manager = DocumentationManager(config)
    
    # Générer la documentation complète
    output_dir = Path("/tmp/tenancy_documentation")
    generated_files = await doc_manager.generate_complete_documentation(output_dir)
    
    print("📚 Documentation generation complete!")
    print(f"📁 Output directory: {output_dir}")
    
    for format, files in generated_files.items():
        print(f"📄 {format.upper()} files: {len(files)}")
        for file in files:
            print(f"   - {file.name}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
