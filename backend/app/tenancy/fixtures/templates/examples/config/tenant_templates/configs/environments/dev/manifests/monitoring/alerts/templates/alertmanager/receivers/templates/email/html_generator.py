"""
Advanced HTML Email Template Generator

This module provides sophisticated HTML email template generation with
responsive design, AI-powered content optimization, and cross-client compatibility.

Version: 3.0.0
Developed by Spotify AI Agent Team
"""

import re
import base64
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import structlog
import aiofiles
from jinja2 import Template
from bs4 import BeautifulSoup, Tag
import cssutils
import premailer
from PIL import Image
import requests

logger = structlog.get_logger(__name__)

# ============================================================================
# HTML Template Configuration
# ============================================================================

@dataclass
class ResponsiveConfig:
    """Configuration pour le design responsive"""
    mobile_breakpoint: int = 600
    tablet_breakpoint: int = 768
    desktop_breakpoint: int = 1200
    enable_dark_mode: bool = True
    fluid_layout: bool = True

@dataclass
class EmailClientConfig:
    """Configuration de compatibilité email"""
    outlook_compatibility: bool = True
    gmail_compatibility: bool = True
    apple_mail_compatibility: bool = True
    yahoo_compatibility: bool = True
    thunderbird_compatibility: bool = True
    webmail_optimization: bool = True

@dataclass
class TemplateTheme:
    """Thème de template"""
    name: str
    primary_color: str = "#007bff"
    secondary_color: str = "#6c757d"
    success_color: str = "#28a745"
    warning_color: str = "#ffc107"
    danger_color: str = "#dc3545"
    info_color: str = "#17a2b8"
    background_color: str = "#ffffff"
    text_color: str = "#212529"
    font_family: str = "Arial, sans-serif"
    border_radius: str = "4px"
    box_shadow: str = "0 2px 4px rgba(0,0,0,0.1)"

# ============================================================================
# Advanced HTML Template Generator
# ============================================================================

class AdvancedHTMLTemplateGenerator:
    """Générateur de templates HTML avancés pour emails"""
    
    def __init__(self,
                 assets_dir: str,
                 responsive_config: Optional[ResponsiveConfig] = None,
                 client_config: Optional[EmailClientConfig] = None,
                 default_theme: Optional[TemplateTheme] = None):
        
        self.assets_dir = Path(assets_dir)
        self.responsive_config = responsive_config or ResponsiveConfig()
        self.client_config = client_config or EmailClientConfig()
        self.default_theme = default_theme or TemplateTheme("default")
        
        # Template components
        self.base_templates: Dict[str, str] = {}
        self.component_library: Dict[str, str] = {}
        self.css_frameworks: Dict[str, str] = {}
        
        # Image processing
        self.image_cache: Dict[str, str] = {}
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.svg']
        
        # CSS optimization
        self.css_optimizer = cssutils.CSSParser(raiseExceptions=False)
        
        # Initialize
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced HTML Template Generator initialized")
    
    async def _initialize(self):
        """Initialisation du générateur"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Chargement des templates de base
        await self._load_base_templates()
        
        # Chargement de la bibliothèque de composants
        await self._load_component_library()
        
        # Initialisation des frameworks CSS
        await self._initialize_css_frameworks()
        
        logger.info("HTML template generator initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.assets_dir,
            self.assets_dir / "templates",
            self.assets_dir / "components",
            self.assets_dir / "css",
            self.assets_dir / "images",
            self.assets_dir / "fonts",
            self.assets_dir / "icons"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_base_templates(self):
        """Charge les templates de base"""
        
        # Template minimal responsive
        self.base_templates["minimal"] = """
<!DOCTYPE html>
<html lang="{{ language|default('en') }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>{{ title|default('Email') }}</title>
    <style>
        {{ base_css }}
        {{ custom_css }}
    </style>
    <!--[if mso]>
    <style>
        {{ outlook_css }}
    </style>
    <![endif]-->
</head>
<body>
    <div class="email-container">
        {{ content }}
    </div>
</body>
</html>
        """
        
        # Template avec header et footer
        self.base_templates["standard"] = """
<!DOCTYPE html>
<html lang="{{ language|default('en') }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>{{ title|default('Email') }}</title>
    <style>
        {{ base_css }}
        {{ responsive_css }}
        {{ custom_css }}
    </style>
    <!--[if mso]>
    <style>
        {{ outlook_css }}
    </style>
    <![endif]-->
</head>
<body>
    <div class="email-wrapper">
        {% if header %}
        <div class="email-header">
            {{ header }}
        </div>
        {% endif %}
        
        <div class="email-content">
            {{ content }}
        </div>
        
        {% if footer %}
        <div class="email-footer">
            {{ footer }}
        </div>
        {% endif %}
    </div>
</body>
</html>
        """
        
        # Template avancé avec sidebar
        self.base_templates["advanced"] = """
<!DOCTYPE html>
<html lang="{{ language|default('en') }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>{{ title|default('Email') }}</title>
    <style>
        {{ base_css }}
        {{ responsive_css }}
        {{ grid_css }}
        {{ custom_css }}
    </style>
    <!--[if mso]>
    <style>
        {{ outlook_css }}
    </style>
    <![endif]-->
</head>
<body>
    <div class="email-container">
        {% if header %}
        <header class="email-header">
            {{ header }}
        </header>
        {% endif %}
        
        <div class="email-main">
            {% if sidebar %}
            <aside class="email-sidebar">
                {{ sidebar }}
            </aside>
            {% endif %}
            
            <main class="email-content">
                {{ content }}
            </main>
        </div>
        
        {% if footer %}
        <footer class="email-footer">
            {{ footer }}
        </footer>
        {% endif %}
    </div>
</body>
</html>
        """
    
    async def _load_component_library(self):
        """Charge la bibliothèque de composants"""
        
        # Composant bouton
        self.component_library["button"] = """
<div class="btn-container" style="text-align: {{ align|default('center') }}; margin: {{ margin|default('20px 0') }};">
    <a href="{{ url }}" class="btn btn-{{ style|default('primary') }}" style="
        display: inline-block;
        padding: {{ padding|default('12px 24px') }};
        background-color: {{ bg_color|default(theme.primary_color) }};
        color: {{ text_color|default('#ffffff') }};
        text-decoration: none;
        border-radius: {{ border_radius|default(theme.border_radius) }};
        font-weight: bold;
        font-family: {{ theme.font_family }};
        ">
        {{ text }}
    </a>
</div>
        """
        
        # Composant alerte
        self.component_library["alert"] = """
<div class="alert alert-{{ type|default('info') }}" style="
    padding: {{ padding|default('15px') }};
    margin: {{ margin|default('20px 0') }};
    border: 1px solid {{ border_color }};
    border-radius: {{ theme.border_radius }};
    background-color: {{ bg_color }};
    color: {{ text_color }};
    font-family: {{ theme.font_family }};
    ">
    {% if icon %}
    <span class="alert-icon" style="margin-right: 10px;">{{ icon }}</span>
    {% endif %}
    <div class="alert-content">
        {% if title %}
        <h4 style="margin: 0 0 10px 0; font-weight: bold;">{{ title }}</h4>
        {% endif %}
        {{ content }}
    </div>
</div>
        """
        
        # Composant card
        self.component_library["card"] = """
<div class="card" style="
    background-color: {{ bg_color|default('#ffffff') }};
    border: {{ border|default('1px solid #e0e0e0') }};
    border-radius: {{ theme.border_radius }};
    box-shadow: {{ theme.box_shadow }};
    margin: {{ margin|default('20px 0') }};
    overflow: hidden;
    ">
    {% if image %}
    <div class="card-image">
        <img src="{{ image }}" alt="{{ image_alt|default('') }}" style="width: 100%; height: auto; display: block;">
    </div>
    {% endif %}
    
    <div class="card-body" style="padding: {{ padding|default('20px') }};">
        {% if title %}
        <h3 class="card-title" style="margin: 0 0 15px 0; color: {{ title_color|default(theme.text_color) }}; font-family: {{ theme.font_family }};">
            {{ title }}
        </h3>
        {% endif %}
        
        {% if subtitle %}
        <h4 class="card-subtitle" style="margin: 0 0 15px 0; color: {{ subtitle_color|default(theme.secondary_color) }}; font-family: {{ theme.font_family }};">
            {{ subtitle }}
        </h4>
        {% endif %}
        
        <div class="card-content" style="color: {{ text_color|default(theme.text_color) }}; font-family: {{ theme.font_family }};">
            {{ content }}
        </div>
        
        {% if actions %}
        <div class="card-actions" style="margin-top: 20px;">
            {{ actions }}
        </div>
        {% endif %}
    </div>
</div>
        """
        
        # Composant liste
        self.component_library["list"] = """
<div class="list-container" style="margin: {{ margin|default('20px 0') }};">
    {% if title %}
    <h3 style="margin: 0 0 15px 0; color: {{ title_color|default(theme.text_color) }}; font-family: {{ theme.font_family }};">
        {{ title }}
    </h3>
    {% endif %}
    
    <ul class="list" style="
        list-style: {{ list_style|default('none') }};
        padding: 0;
        margin: 0;
        ">
        {% for item in items %}
        <li class="list-item" style="
            padding: {{ item_padding|default('10px 0') }};
            border-bottom: {{ item_border|default('1px solid #e0e0e0') }};
            font-family: {{ theme.font_family }};
            color: {{ item_color|default(theme.text_color) }};
            ">
            {% if item.icon %}
            <span class="item-icon" style="margin-right: 10px;">{{ item.icon }}</span>
            {% endif %}
            
            {% if item.title %}
            <strong>{{ item.title }}</strong>{% if item.content %}: {% endif %}
            {% endif %}
            
            {% if item.content %}
            {{ item.content }}
            {% endif %}
        </li>
        {% endfor %}
    </ul>
</div>
        """
        
        # Composant tableau
        self.component_library["table"] = """
<div class="table-container" style="margin: {{ margin|default('20px 0') }}; overflow-x: auto;">
    {% if title %}
    <h3 style="margin: 0 0 15px 0; color: {{ title_color|default(theme.text_color) }}; font-family: {{ theme.font_family }};">
        {{ title }}
    </h3>
    {% endif %}
    
    <table class="table" style="
        width: 100%;
        border-collapse: collapse;
        font-family: {{ theme.font_family }};
        background-color: {{ bg_color|default('#ffffff') }};
        ">
        {% if headers %}
        <thead>
            <tr style="background-color: {{ header_bg|default('#f8f9fa') }};">
                {% for header in headers %}
                <th style="
                    padding: {{ header_padding|default('12px') }};
                    text-align: {{ header_align|default('left') }};
                    border: {{ border|default('1px solid #dee2e6') }};
                    color: {{ header_color|default(theme.text_color) }};
                    font-weight: bold;
                    ">
                    {{ header }}
                </th>
                {% endfor %}
            </tr>
        </thead>
        {% endif %}
        
        <tbody>
            {% for row in rows %}
            <tr style="{% if loop.index % 2 == 0 %}background-color: {{ stripe_color|default('#f8f9fa') }};{% endif %}">
                {% for cell in row %}
                <td style="
                    padding: {{ cell_padding|default('12px') }};
                    border: {{ border|default('1px solid #dee2e6') }};
                    color: {{ cell_color|default(theme.text_color) }};
                    ">
                    {{ cell }}
                </td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
        """
    
    async def _initialize_css_frameworks(self):
        """Initialise les frameworks CSS"""
        
        # CSS de base responsive
        self.css_frameworks["base"] = """
/* Reset et normalisation */
* { box-sizing: border-box; }
body, table, td, div, p, a { -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }
body { margin: 0; padding: 0; background-color: #f4f4f4; font-family: Arial, sans-serif; }
table { border-collapse: collapse; mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
img { border: 0; height: auto; line-height: 100%; outline: none; text-decoration: none; }

/* Container principal */
.email-container {
    max-width: 600px;
    margin: 0 auto;
    background-color: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.email-wrapper {
    width: 100%;
    background-color: #f4f4f4;
    padding: 20px 0;
}

/* Layout responsive */
.email-header,
.email-content,
.email-footer {
    padding: 20px;
}

.email-header {
    background-color: #007bff;
    color: #ffffff;
    text-align: center;
}

.email-footer {
    background-color: #f8f9fa;
    color: #6c757d;
    font-size: 14px;
    text-align: center;
}

/* Utilitaires */
.text-center { text-align: center !important; }
.text-left { text-align: left !important; }
.text-right { text-align: right !important; }

.d-block { display: block !important; }
.d-inline { display: inline !important; }
.d-inline-block { display: inline-block !important; }

.mb-0 { margin-bottom: 0 !important; }
.mb-1 { margin-bottom: 10px !important; }
.mb-2 { margin-bottom: 20px !important; }
.mb-3 { margin-bottom: 30px !important; }

.mt-0 { margin-top: 0 !important; }
.mt-1 { margin-top: 10px !important; }
.mt-2 { margin-top: 20px !important; }
.mt-3 { margin-top: 30px !important; }
        """
        
        # CSS responsive
        self.css_frameworks["responsive"] = f"""
/* Media queries pour responsive */
@media only screen and (max-width: {self.responsive_config.mobile_breakpoint}px) {{
    .email-container {{
        width: 100% !important;
        margin: 0 !important;
        border-radius: 0 !important;
    }}
    
    .email-header,
    .email-content,
    .email-footer {{
        padding: 15px !important;
    }}
    
    .card {{
        margin: 10px 0 !important;
    }}
    
    .btn {{
        display: block !important;
        width: 100% !important;
        text-align: center !important;
    }}
    
    .table-container {{
        overflow-x: auto !important;
    }}
    
    .email-main {{
        flex-direction: column !important;
    }}
    
    .email-sidebar {{
        width: 100% !important;
        margin-bottom: 20px !important;
    }}
}}

@media only screen and (max-width: {self.responsive_config.tablet_breakpoint}px) {{
    .email-container {{
        max-width: 100% !important;
    }}
}}

/* Dark mode support */
@media (prefers-color-scheme: dark) {{
    .email-container {{
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }}
    
    .email-footer {{
        background-color: #2a2a2a !important;
        color: #cccccc !important;
    }}
    
    .card {{
        background-color: #2a2a2a !important;
        border-color: #444444 !important;
    }}
}}
        """
        
        # CSS pour Outlook
        self.css_frameworks["outlook"] = """
/* Styles spécifiques à Outlook */
.outlook-table {
    width: 100%;
    border-collapse: collapse;
}

.outlook-cell {
    padding: 0;
    vertical-align: top;
}

/* Fixes pour Outlook */
table { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
a[x-apple-data-detectors] { color: inherit !important; text-decoration: none !important; }
        """
        
        # CSS de grille
        self.css_frameworks["grid"] = """
/* Système de grille */
.email-main {
    display: flex;
    max-width: 600px;
    margin: 0 auto;
}

.email-sidebar {
    width: 200px;
    background-color: #f8f9fa;
    padding: 20px;
}

.email-content {
    flex: 1;
    padding: 20px;
}

.row {
    display: flex;
    flex-wrap: wrap;
    margin: 0 -10px;
}

.col {
    flex: 1;
    padding: 0 10px;
}

.col-2 { flex: 0 0 16.666667%; max-width: 16.666667%; }
.col-3 { flex: 0 0 25%; max-width: 25%; }
.col-4 { flex: 0 0 33.333333%; max-width: 33.333333%; }
.col-6 { flex: 0 0 50%; max-width: 50%; }
.col-8 { flex: 0 0 66.666667%; max-width: 66.666667%; }
.col-9 { flex: 0 0 75%; max-width: 75%; }
.col-12 { flex: 0 0 100%; max-width: 100%; }
        """
    
    async def generate_template(self,
                              template_type: str,
                              content: Dict[str, Any],
                              theme: Optional[TemplateTheme] = None,
                              custom_css: str = "") -> str:
        """Génère un template HTML complet"""
        
        theme = theme or self.default_theme
        
        # Sélection du template de base
        base_template = self.base_templates.get(template_type, self.base_templates["standard"])
        
        # Génération du CSS
        css_parts = []
        css_parts.append(self.css_frameworks["base"])
        
        if self.responsive_config:
            css_parts.append(self.css_frameworks["responsive"])
        
        if self.client_config.outlook_compatibility:
            css_parts.append(self.css_frameworks["outlook"])
        
        if template_type == "advanced":
            css_parts.append(self.css_frameworks["grid"])
        
        # CSS du thème
        theme_css = self._generate_theme_css(theme)
        css_parts.append(theme_css)
        
        # CSS personnalisé
        if custom_css:
            css_parts.append(custom_css)
        
        # Rendu du template
        template = Template(base_template)
        
        context = {
            "base_css": css_parts[0],
            "responsive_css": css_parts[1] if len(css_parts) > 1 else "",
            "outlook_css": self.css_frameworks["outlook"] if self.client_config.outlook_compatibility else "",
            "grid_css": self.css_frameworks.get("grid", "") if template_type == "advanced" else "",
            "custom_css": "\n".join(css_parts[1:]),
            "theme": theme,
            **content
        }
        
        html_content = template.render(**context)
        
        # Post-traitement
        html_content = await self._post_process_html(html_content)
        
        return html_content
    
    async def render_component(self,
                             component_name: str,
                             data: Dict[str, Any],
                             theme: Optional[TemplateTheme] = None) -> str:
        """Rend un composant spécifique"""
        
        if component_name not in self.component_library:
            raise ValueError(f"Unknown component: {component_name}")
        
        theme = theme or self.default_theme
        component_template = self.component_library[component_name]
        
        template = Template(component_template)
        
        context = {
            "theme": theme,
            **data
        }
        
        # Traitement spécial pour certains composants
        if component_name == "alert":
            context = self._prepare_alert_context(context)
        elif component_name == "button":
            context = self._prepare_button_context(context)
        
        return template.render(**context)
    
    async def optimize_for_client(self,
                                html_content: str,
                                client: str = "all") -> str:
        """Optimise le HTML pour des clients email spécifiques"""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if client in ["outlook", "all"]:
            soup = self._optimize_for_outlook(soup)
        
        if client in ["gmail", "all"]:
            soup = self._optimize_for_gmail(soup)
        
        if client in ["apple", "all"]:
            soup = self._optimize_for_apple_mail(soup)
        
        return str(soup)
    
    async def inline_css(self, html_content: str) -> str:
        """Inline le CSS dans le HTML"""
        
        try:
            p = premailer.Premailer(
                html_content,
                keep_style_tags=True,
                strip_important=False,
                remove_classes=False
            )
            return p.transform()
        except Exception as e:
            logger.error(f"CSS inlining failed: {e}")
            return html_content
    
    async def validate_html(self, html_content: str) -> Dict[str, Any]:
        """Valide le HTML pour les emails"""
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Vérifications de base
        if not soup.find('html'):
            validation_results["errors"].append("Missing <html> tag")
            validation_results["valid"] = False
        
        if not soup.find('head'):
            validation_results["warnings"].append("Missing <head> tag")
        
        if not soup.find('body'):
            validation_results["errors"].append("Missing <body> tag")
            validation_results["valid"] = False
        
        # Vérification des méta tags
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        if not viewport_meta:
            validation_results["warnings"].append("Missing viewport meta tag")
        
        # Vérification des images
        images = soup.find_all('img')
        for img in images:
            if not img.get('alt'):
                validation_results["warnings"].append(f"Image missing alt attribute: {img.get('src', 'unknown')}")
            
            if not img.get('width') or not img.get('height'):
                validation_results["suggestions"].append(f"Consider adding width/height to image: {img.get('src', 'unknown')}")
        
        # Vérification des liens
        links = soup.find_all('a')
        for link in links:
            href = link.get('href')
            if href and href.startswith('mailto:'):
                continue
            elif href and not href.startswith(('http://', 'https://', '#')):
                validation_results["warnings"].append(f"Relative URL found: {href}")
        
        # Vérification des styles inline vs CSS
        style_tags = soup.find_all('style')
        inline_styles = soup.find_all(attrs={'style': True})
        
        if len(style_tags) > 0 and len(inline_styles) == 0:
            validation_results["suggestions"].append("Consider inlining CSS for better email client compatibility")
        
        return validation_results
    
    async def generate_plain_text_version(self, html_content: str) -> str:
        """Génère une version texte à partir du HTML"""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Suppression des éléments non désirés
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # Conversion des liens
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                link_text = link.get_text()
                link.replace_with(f"{link_text} ({href})")
        
        # Conversion des images
        for img in soup.find_all('img'):
            alt_text = img.get('alt', '[Image]')
            img.replace_with(f"[{alt_text}]")
        
        # Extraction du texte
        text = soup.get_text()
        
        # Nettoyage
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Suppression des lignes vides
        
        return '\n\n'.join(lines)
    
    def _generate_theme_css(self, theme: TemplateTheme) -> str:
        """Génère le CSS du thème"""
        
        return f"""
/* Thème: {theme.name} */
:root {{
    --primary-color: {theme.primary_color};
    --secondary-color: {theme.secondary_color};
    --success-color: {theme.success_color};
    --warning-color: {theme.warning_color};
    --danger-color: {theme.danger_color};
    --info-color: {theme.info_color};
    --background-color: {theme.background_color};
    --text-color: {theme.text_color};
    --font-family: {theme.font_family};
    --border-radius: {theme.border_radius};
    --box-shadow: {theme.box_shadow};
}}

.btn-primary {{ background-color: {theme.primary_color}; }}
.btn-secondary {{ background-color: {theme.secondary_color}; }}
.btn-success {{ background-color: {theme.success_color}; }}
.btn-warning {{ background-color: {theme.warning_color}; }}
.btn-danger {{ background-color: {theme.danger_color}; }}
.btn-info {{ background-color: {theme.info_color}; }}

.text-primary {{ color: {theme.primary_color}; }}
.text-secondary {{ color: {theme.secondary_color}; }}
.text-success {{ color: {theme.success_color}; }}
.text-warning {{ color: {theme.warning_color}; }}
.text-danger {{ color: {theme.danger_color}; }}
.text-info {{ color: {theme.info_color}; }}

.bg-primary {{ background-color: {theme.primary_color}; }}
.bg-secondary {{ background-color: {theme.secondary_color}; }}
.bg-success {{ background-color: {theme.success_color}; }}
.bg-warning {{ background-color: {theme.warning_color}; }}
.bg-danger {{ background-color: {theme.danger_color}; }}
.bg-info {{ background-color: {theme.info_color}; }}
        """
    
    def _prepare_alert_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare le contexte pour les alertes"""
        
        alert_type = context.get('type', 'info')
        theme = context.get('theme')
        
        color_map = {
            'success': (theme.success_color, '#d4edda', '#155724'),
            'warning': (theme.warning_color, '#fff3cd', '#856404'),
            'danger': (theme.danger_color, '#f8d7da', '#721c24'),
            'info': (theme.info_color, '#d1ecf1', '#0c5460')
        }
        
        if alert_type in color_map:
            border_color, bg_color, text_color = color_map[alert_type]
            context['border_color'] = border_color
            context['bg_color'] = bg_color
            context['text_color'] = text_color
        
        return context
    
    def _prepare_button_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare le contexte pour les boutons"""
        
        style = context.get('style', 'primary')
        theme = context.get('theme')
        
        if style == 'primary' and 'bg_color' not in context:
            context['bg_color'] = theme.primary_color
        elif style == 'secondary' and 'bg_color' not in context:
            context['bg_color'] = theme.secondary_color
        
        return context
    
    def _optimize_for_outlook(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Optimise pour Outlook"""
        
        # Ajout de conditional comments pour Outlook
        head = soup.find('head')
        if head:
            outlook_css = soup.new_tag('style')
            outlook_css.string = """
            <!--[if mso]>
            <style>
                table { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
                .outlook-hide { display: none !important; }
            </style>
            <![endif]-->
            """
            head.append(outlook_css)
        
        return soup
    
    def _optimize_for_gmail(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Optimise pour Gmail"""
        
        # Ajout de classes spécifiques pour Gmail
        body = soup.find('body')
        if body:
            body['class'] = body.get('class', []) + ['gmail-mobile']
        
        return soup
    
    def _optimize_for_apple_mail(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Optimise pour Apple Mail"""
        
        # Ajout d'attributs pour Apple Mail
        links = soup.find_all('a')
        for link in links:
            link['x-apple-data-detectors'] = 'false'
        
        return soup
    
    async def _post_process_html(self, html_content: str) -> str:
        """Post-traitement du HTML"""
        
        # Nettoyage des espaces
        html_content = re.sub(r'\n\s*\n', '\n', html_content)
        html_content = re.sub(r'>\s+<', '><', html_content)
        
        # Optimisation pour les emails
        if self.client_config.outlook_compatibility:
            html_content = await self.optimize_for_client(html_content, "outlook")
        
        return html_content

# ============================================================================
# Factory Functions
# ============================================================================

def create_html_generator(
    assets_dir: str,
    enable_responsive: bool = True,
    enable_dark_mode: bool = True
) -> AdvancedHTMLTemplateGenerator:
    """Factory pour créer un générateur HTML"""
    
    responsive_config = ResponsiveConfig(enable_dark_mode=enable_dark_mode) if enable_responsive else None
    
    return AdvancedHTMLTemplateGenerator(
        assets_dir=assets_dir,
        responsive_config=responsive_config
    )

def create_default_theme(name: str, primary_color: str) -> TemplateTheme:
    """Crée un thème par défaut avec couleur primaire"""
    
    return TemplateTheme(
        name=name,
        primary_color=primary_color
    )

# Export des classes principales
__all__ = [
    "AdvancedHTMLTemplateGenerator",
    "ResponsiveConfig",
    "EmailClientConfig",
    "TemplateTheme",
    "create_html_generator",
    "create_default_theme"
]
