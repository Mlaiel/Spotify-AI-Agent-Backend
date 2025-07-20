"""
Advanced CSS Style Manager

This module provides sophisticated CSS management for email templates including
responsive design, dark mode support, cross-client compatibility, and AI-powered
style optimization.

Version: 3.0.0
Developed by Spotify AI Agent Team
"""

import re
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aiofiles
import cssutils
import sass
from colorsys import rgb_to_hls, hls_to_rgb

logger = structlog.get_logger(__name__)

# ============================================================================
# CSS Configuration Classes
# ============================================================================

class CSSFramework(Enum):
    """Types de frameworks CSS supportés"""
    FOUNDATION = "foundation"
    BOOTSTRAP = "bootstrap"
    TAILWIND = "tailwind"
    CUSTOM = "custom"

class ColorScheme(Enum):
    """Schémas de couleurs"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"

@dataclass
class CSSVariable:
    """Variable CSS"""
    name: str
    value: str
    category: str = "general"
    description: str = ""
    fallback: Optional[str] = None

@dataclass
class ResponsiveBreakpoint:
    """Point de rupture responsive"""
    name: str
    min_width: Optional[int] = None
    max_width: Optional[int] = None
    orientation: Optional[str] = None

@dataclass
class StyleRule:
    """Règle de style CSS"""
    selector: str
    properties: Dict[str, str]
    media_query: Optional[str] = None
    pseudo_class: Optional[str] = None
    importance: int = 0

@dataclass
class ColorPalette:
    """Palette de couleurs"""
    name: str
    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    text: str
    text_secondary: str
    border: str
    success: str
    warning: str
    error: str
    info: str

# ============================================================================
# Advanced CSS Style Manager
# ============================================================================

class AdvancedCSSStyleManager:
    """Gestionnaire de styles CSS avancé pour emails"""
    
    def __init__(self,
                 assets_dir: str,
                 framework: CSSFramework = CSSFramework.CUSTOM,
                 enable_sass: bool = True,
                 enable_autoprefixer: bool = True):
        
        self.assets_dir = Path(assets_dir)
        self.framework = framework
        self.enable_sass = enable_sass
        self.enable_autoprefixer = enable_autoprefixer
        
        # Configuration
        self.css_variables: Dict[str, CSSVariable] = {}
        self.color_palettes: Dict[str, ColorPalette] = {}
        self.breakpoints: Dict[str, ResponsiveBreakpoint] = {}
        self.style_rules: List[StyleRule] = []
        
        # Cache
        self.compiled_css_cache: Dict[str, str] = {}
        self.minified_css_cache: Dict[str, str] = {}
        
        # CSS Parser
        self.css_parser = cssutils.CSSParser(raiseExceptions=False)
        cssutils.log.setLevel('ERROR')
        
        # Initialize
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced CSS Style Manager initialized")
    
    async def _initialize(self):
        """Initialisation du gestionnaire"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Chargement des palettes par défaut
        await self._load_default_palettes()
        
        # Configuration des breakpoints
        await self._setup_default_breakpoints()
        
        # Variables CSS de base
        await self._setup_base_variables()
        
        # Styles de base selon le framework
        await self._setup_framework_styles()
        
        logger.info("CSS Style Manager initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.assets_dir / "css",
            self.assets_dir / "css" / "components",
            self.assets_dir / "css" / "themes",
            self.assets_dir / "css" / "frameworks",
            self.assets_dir / "css" / "responsive",
            self.assets_dir / "scss" if self.enable_sass else None
        ]
        
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_default_palettes(self):
        """Charge les palettes de couleurs par défaut"""
        
        # Palette moderne
        self.color_palettes["modern"] = ColorPalette(
            name="modern",
            primary="#007bff",
            secondary="#6c757d",
            accent="#17a2b8",
            background="#ffffff",
            surface="#f8f9fa",
            text="#212529",
            text_secondary="#6c757d",
            border="#dee2e6",
            success="#28a745",
            warning="#ffc107",
            error="#dc3545",
            info="#17a2b8"
        )
        
        # Palette sombre
        self.color_palettes["dark"] = ColorPalette(
            name="dark",
            primary="#0d6efd",
            secondary="#6c757d",
            accent="#20c997",
            background="#121212",
            surface="#1e1e1e",
            text="#ffffff",
            text_secondary="#adb5bd",
            border="#495057",
            success="#198754",
            warning="#fd7e14",
            error="#dc3545",
            info="#0dcaf0"
        )
        
        # Palette haute accessibilité
        self.color_palettes["accessible"] = ColorPalette(
            name="accessible",
            primary="#0056b3",
            secondary="#495057",
            accent="#117a8b",
            background="#ffffff",
            surface="#f8f9fa",
            text="#000000",
            text_secondary="#495057",
            border="#6c757d",
            success="#155724",
            warning="#856404",
            error="#721c24",
            info="#0c5460"
        )
        
        # Palette Spotify
        self.color_palettes["spotify"] = ColorPalette(
            name="spotify",
            primary="#1db954",
            secondary="#191414",
            accent="#1ed760",
            background="#000000",
            surface="#121212",
            text="#ffffff",
            text_secondary="#b3b3b3",
            border="#282828",
            success="#1db954",
            warning="#ff9500",
            error="#e22134",
            info="#00d4ff"
        )
    
    async def _setup_default_breakpoints(self):
        """Configure les breakpoints par défaut"""
        
        self.breakpoints = {
            "xs": ResponsiveBreakpoint("xs", max_width=575),
            "sm": ResponsiveBreakpoint("sm", min_width=576, max_width=767),
            "md": ResponsiveBreakpoint("md", min_width=768, max_width=991),
            "lg": ResponsiveBreakpoint("lg", min_width=992, max_width=1199),
            "xl": ResponsiveBreakpoint("xl", min_width=1200)
        }
    
    async def _setup_base_variables(self):
        """Configure les variables CSS de base"""
        
        base_variables = [
            CSSVariable("font-family-sans", "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", "typography"),
            CSSVariable("font-family-serif", "Georgia, 'Times New Roman', serif", "typography"),
            CSSVariable("font-family-mono", "'SFMono-Regular', Consolas, 'Liberation Mono', monospace", "typography"),
            
            CSSVariable("font-size-xs", "0.75rem", "typography"),
            CSSVariable("font-size-sm", "0.875rem", "typography"),
            CSSVariable("font-size-base", "1rem", "typography"),
            CSSVariable("font-size-lg", "1.125rem", "typography"),
            CSSVariable("font-size-xl", "1.25rem", "typography"),
            
            CSSVariable("line-height-tight", "1.25", "typography"),
            CSSVariable("line-height-normal", "1.5", "typography"),
            CSSVariable("line-height-relaxed", "1.75", "typography"),
            
            CSSVariable("spacing-xs", "0.25rem", "spacing"),
            CSSVariable("spacing-sm", "0.5rem", "spacing"),
            CSSVariable("spacing-md", "1rem", "spacing"),
            CSSVariable("spacing-lg", "1.5rem", "spacing"),
            CSSVariable("spacing-xl", "2rem", "spacing"),
            
            CSSVariable("border-radius-sm", "0.125rem", "borders"),
            CSSVariable("border-radius-md", "0.25rem", "borders"),
            CSSVariable("border-radius-lg", "0.5rem", "borders"),
            CSSVariable("border-radius-xl", "1rem", "borders"),
            
            CSSVariable("shadow-sm", "0 1px 2px 0 rgba(0, 0, 0, 0.05)", "shadows"),
            CSSVariable("shadow-md", "0 4px 6px -1px rgba(0, 0, 0, 0.1)", "shadows"),
            CSSVariable("shadow-lg", "0 10px 15px -3px rgba(0, 0, 0, 0.1)", "shadows"),
            
            CSSVariable("transition-fast", "0.15s ease-in-out", "transitions"),
            CSSVariable("transition-normal", "0.3s ease-in-out", "transitions"),
            CSSVariable("transition-slow", "0.5s ease-in-out", "transitions"),
        ]
        
        for var in base_variables:
            self.css_variables[var.name] = var
    
    async def _setup_framework_styles(self):
        """Configure les styles selon le framework"""
        
        if self.framework == CSSFramework.BOOTSTRAP:
            await self._setup_bootstrap_styles()
        elif self.framework == CSSFramework.FOUNDATION:
            await self._setup_foundation_styles()
        elif self.framework == CSSFramework.TAILWIND:
            await self._setup_tailwind_styles()
        else:
            await self._setup_custom_styles()
    
    async def _setup_bootstrap_styles(self):
        """Configure les styles Bootstrap"""
        
        bootstrap_rules = [
            StyleRule(".container", {
                "width": "100%",
                "max-width": "600px",
                "margin": "0 auto",
                "padding": "0 15px"
            }),
            
            StyleRule(".row", {
                "display": "flex",
                "flex-wrap": "wrap",
                "margin": "0 -15px"
            }),
            
            StyleRule(".col", {
                "flex": "1",
                "padding": "0 15px"
            }),
            
            StyleRule(".btn", {
                "display": "inline-block",
                "padding": "8px 16px",
                "text-decoration": "none",
                "border-radius": "4px",
                "font-weight": "400",
                "text-align": "center",
                "vertical-align": "middle",
                "cursor": "pointer",
                "border": "1px solid transparent"
            }),
            
            StyleRule(".card", {
                "background-color": "var(--color-surface)",
                "border": "1px solid var(--color-border)",
                "border-radius": "var(--border-radius-md)",
                "box-shadow": "var(--shadow-sm)"
            })
        ]
        
        self.style_rules.extend(bootstrap_rules)
    
    async def _setup_foundation_styles(self):
        """Configure les styles Foundation"""
        
        foundation_rules = [
            StyleRule(".grid-container", {
                "max-width": "600px",
                "margin": "0 auto",
                "padding": "0 1rem"
            }),
            
            StyleRule(".grid-x", {
                "display": "flex",
                "flex-flow": "row wrap"
            }),
            
            StyleRule(".cell", {
                "flex": "0 0 auto",
                "min-height": "0px",
                "min-width": "0px"
            }),
            
            StyleRule(".button", {
                "display": "inline-block",
                "vertical-align": "middle",
                "margin": "0 0 1rem 0",
                "padding": "0.85em 1em",
                "border": "1px solid transparent",
                "border-radius": "var(--border-radius-md)",
                "transition": "var(--transition-fast)",
                "text-decoration": "none",
                "text-align": "center",
                "cursor": "pointer"
            })
        ]
        
        self.style_rules.extend(foundation_rules)
    
    async def _setup_tailwind_styles(self):
        """Configure les styles Tailwind"""
        
        tailwind_rules = [
            # Utility classes
            StyleRule(".flex", {"display": "flex"}),
            StyleRule(".inline-block", {"display": "inline-block"}),
            StyleRule(".block", {"display": "block"}),
            
            # Spacing
            StyleRule(".p-1", {"padding": "var(--spacing-xs)"}),
            StyleRule(".p-2", {"padding": "var(--spacing-sm)"}),
            StyleRule(".p-4", {"padding": "var(--spacing-md)"}),
            StyleRule(".p-6", {"padding": "var(--spacing-lg)"}),
            StyleRule(".p-8", {"padding": "var(--spacing-xl)"}),
            
            StyleRule(".m-1", {"margin": "var(--spacing-xs)"}),
            StyleRule(".m-2", {"margin": "var(--spacing-sm)"}),
            StyleRule(".m-4", {"margin": "var(--spacing-md)"}),
            StyleRule(".m-6", {"margin": "var(--spacing-lg)"}),
            StyleRule(".m-8", {"margin": "var(--spacing-xl)"}),
            
            # Text alignment
            StyleRule(".text-left", {"text-align": "left"}),
            StyleRule(".text-center", {"text-align": "center"}),
            StyleRule(".text-right", {"text-align": "right"}),
            
            # Colors
            StyleRule(".text-primary", {"color": "var(--color-primary)"}),
            StyleRule(".text-secondary", {"color": "var(--color-secondary)"}),
            StyleRule(".bg-primary", {"background-color": "var(--color-primary)"}),
            StyleRule(".bg-secondary", {"background-color": "var(--color-secondary)"}),
            
            # Border radius
            StyleRule(".rounded", {"border-radius": "var(--border-radius-md)"}),
            StyleRule(".rounded-sm", {"border-radius": "var(--border-radius-sm)"}),
            StyleRule(".rounded-lg", {"border-radius": "var(--border-radius-lg)"}),
        ]
        
        self.style_rules.extend(tailwind_rules)
    
    async def _setup_custom_styles(self):
        """Configure les styles personnalisés"""
        
        custom_rules = [
            # Layout
            StyleRule(".email-container", {
                "max-width": "600px",
                "margin": "0 auto",
                "background-color": "var(--color-background)",
                "font-family": "var(--font-family-sans)",
                "color": "var(--color-text)"
            }),
            
            # Typography
            StyleRule("h1, h2, h3, h4, h5, h6", {
                "margin": "0 0 var(--spacing-md) 0",
                "font-weight": "bold",
                "line-height": "var(--line-height-tight)"
            }),
            
            StyleRule("p", {
                "margin": "0 0 var(--spacing-md) 0",
                "line-height": "var(--line-height-normal)"
            }),
            
            # Components
            StyleRule(".btn", {
                "display": "inline-block",
                "padding": "var(--spacing-sm) var(--spacing-md)",
                "background-color": "var(--color-primary)",
                "color": "var(--color-background)",
                "text-decoration": "none",
                "border-radius": "var(--border-radius-md)",
                "font-weight": "bold",
                "text-align": "center",
                "transition": "var(--transition-fast)"
            }),
            
            StyleRule(".card", {
                "background-color": "var(--color-surface)",
                "border": "1px solid var(--color-border)",
                "border-radius": "var(--border-radius-lg)",
                "padding": "var(--spacing-lg)",
                "margin": "var(--spacing-md) 0",
                "box-shadow": "var(--shadow-md)"
            }),
            
            StyleRule(".alert", {
                "padding": "var(--spacing-md)",
                "border-radius": "var(--border-radius-md)",
                "margin": "var(--spacing-md) 0",
                "border": "1px solid transparent"
            })
        ]
        
        self.style_rules.extend(custom_rules)
    
    async def generate_css(self,
                          palette_name: str = "modern",
                          color_scheme: ColorScheme = ColorScheme.LIGHT,
                          responsive: bool = True,
                          include_dark_mode: bool = True) -> str:
        """Génère le CSS complet"""
        
        css_parts = []
        
        # Variables CSS
        css_parts.append(await self._generate_css_variables(palette_name, color_scheme))
        
        # Styles de base
        css_parts.append(await self._generate_base_styles())
        
        # Styles du framework
        css_parts.append(await self._generate_framework_css())
        
        # Styles responsive
        if responsive:
            css_parts.append(await self._generate_responsive_css())
        
        # Dark mode
        if include_dark_mode:
            css_parts.append(await self._generate_dark_mode_css())
        
        # Styles pour clients email
        css_parts.append(await self._generate_email_client_css())
        
        return "\n\n".join(css_parts)
    
    async def _generate_css_variables(self,
                                    palette_name: str,
                                    color_scheme: ColorScheme) -> str:
        """Génère les variables CSS"""
        
        palette = self.color_palettes.get(palette_name, self.color_palettes["modern"])
        
        # Ajustement selon le schéma de couleurs
        if color_scheme == ColorScheme.DARK:
            palette = await self._convert_to_dark_palette(palette)
        elif color_scheme == ColorScheme.HIGH_CONTRAST:
            palette = await self._convert_to_high_contrast_palette(palette)
        
        variables = [":root {"]
        
        # Variables de couleurs
        variables.extend([
            f"  --color-primary: {palette.primary};",
            f"  --color-secondary: {palette.secondary};",
            f"  --color-accent: {palette.accent};",
            f"  --color-background: {palette.background};",
            f"  --color-surface: {palette.surface};",
            f"  --color-text: {palette.text};",
            f"  --color-text-secondary: {palette.text_secondary};",
            f"  --color-border: {palette.border};",
            f"  --color-success: {palette.success};",
            f"  --color-warning: {palette.warning};",
            f"  --color-error: {palette.error};",
            f"  --color-info: {palette.info};"
        ])
        
        # Autres variables
        for var in self.css_variables.values():
            variables.append(f"  --{var.name}: {var.value};")
        
        variables.append("}")
        
        return "\n".join(variables)
    
    async def _generate_base_styles(self) -> str:
        """Génère les styles de base"""
        
        base_css = """
/* Reset et normalisation */
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: var(--font-family-sans);
  font-size: var(--font-size-base);
  line-height: var(--line-height-normal);
  color: var(--color-text);
  background-color: var(--color-background);
  -webkit-text-size-adjust: 100%;
  -ms-text-size-adjust: 100%;
}

/* Tables pour email */
table {
  border-collapse: collapse;
  mso-table-lspace: 0pt;
  mso-table-rspace: 0pt;
  width: 100%;
}

td {
  vertical-align: top;
}

/* Images */
img {
  border: 0;
  height: auto;
  line-height: 100%;
  outline: none;
  text-decoration: none;
  max-width: 100%;
  display: block;
}

/* Liens */
a {
  color: var(--color-primary);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

/* Typography */
h1 { font-size: calc(var(--font-size-base) * 2.5); }
h2 { font-size: calc(var(--font-size-base) * 2); }
h3 { font-size: calc(var(--font-size-base) * 1.75); }
h4 { font-size: calc(var(--font-size-base) * 1.5); }
h5 { font-size: calc(var(--font-size-base) * 1.25); }
h6 { font-size: var(--font-size-base); }
        """
        
        return base_css
    
    async def _generate_framework_css(self) -> str:
        """Génère le CSS du framework"""
        
        css_rules = []
        
        for rule in self.style_rules:
            rule_css = f"{rule.selector} {{"
            
            for prop, value in rule.properties.items():
                rule_css += f"\n  {prop}: {value};"
            
            rule_css += "\n}"
            
            if rule.media_query:
                rule_css = f"@media {rule.media_query} {{\n{rule_css}\n}}"
            
            css_rules.append(rule_css)
        
        return "\n\n".join(css_rules)
    
    async def _generate_responsive_css(self) -> str:
        """Génère le CSS responsive"""
        
        responsive_css = []
        
        for name, breakpoint in self.breakpoints.items():
            media_query_parts = []
            
            if breakpoint.min_width:
                media_query_parts.append(f"(min-width: {breakpoint.min_width}px)")
            
            if breakpoint.max_width:
                media_query_parts.append(f"(max-width: {breakpoint.max_width}px)")
            
            if breakpoint.orientation:
                media_query_parts.append(f"(orientation: {breakpoint.orientation})")
            
            if media_query_parts:
                media_query = " and ".join(media_query_parts)
                
                # Styles spécifiques au breakpoint
                breakpoint_styles = await self._get_breakpoint_styles(name)
                
                if breakpoint_styles:
                    responsive_css.append(f"@media {media_query} {{\n{breakpoint_styles}\n}}")
        
        return "\n\n".join(responsive_css)
    
    async def _generate_dark_mode_css(self) -> str:
        """Génère le CSS pour le mode sombre"""
        
        dark_palette = self.color_palettes.get("dark", self.color_palettes["modern"])
        
        dark_css = f"""
@media (prefers-color-scheme: dark) {{
  :root {{
    --color-primary: {dark_palette.primary};
    --color-secondary: {dark_palette.secondary};
    --color-accent: {dark_palette.accent};
    --color-background: {dark_palette.background};
    --color-surface: {dark_palette.surface};
    --color-text: {dark_palette.text};
    --color-text-secondary: {dark_palette.text_secondary};
    --color-border: {dark_palette.border};
    --color-success: {dark_palette.success};
    --color-warning: {dark_palette.warning};
    --color-error: {dark_palette.error};
    --color-info: {dark_palette.info};
  }}
  
  .email-container {{
    background-color: var(--color-background);
    color: var(--color-text);
  }}
  
  .card {{
    background-color: var(--color-surface);
    border-color: var(--color-border);
  }}
  
  .btn {{
    background-color: var(--color-primary);
    color: var(--color-background);
  }}
}}
        """
        
        return dark_css
    
    async def _generate_email_client_css(self) -> str:
        """Génère le CSS pour les clients email"""
        
        client_css = """
/* Outlook support */
<!--[if mso]>
<style>
  table { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
  .outlook-hide { display: none !important; }
  .outlook-only { display: block !important; }
</style>
<![endif]-->

/* Gmail support */
.gmail-mobile {
  width: 100% !important;
  min-width: 100% !important;
}

/* Apple Mail support */
a[x-apple-data-detectors] {
  color: inherit !important;
  text-decoration: none !important;
  font-size: inherit !important;
  font-family: inherit !important;
  font-weight: inherit !important;
  line-height: inherit !important;
}

/* Yahoo Mail support */
.yahoo-hide {
  display: none !important;
}

/* Thunderbird support */
.thunderbird-fix {
  width: 100% !important;
}
        """
        
        return client_css
    
    async def _get_breakpoint_styles(self, breakpoint_name: str) -> str:
        """Obtient les styles pour un breakpoint spécifique"""
        
        styles = {
            "xs": """
  .email-container {
    width: 100% !important;
    margin: 0 !important;
    padding: var(--spacing-sm) !important;
  }
  
  .btn {
    width: 100% !important;
    display: block !important;
  }
  
  .card {
    margin: var(--spacing-sm) 0 !important;
  }
            """,
            
            "sm": """
  .container {
    max-width: 540px !important;
  }
            """,
            
            "md": """
  .container {
    max-width: 720px !important;
  }
            """,
            
            "lg": """
  .container {
    max-width: 960px !important;
  }
            """,
            
            "xl": """
  .container {
    max-width: 1140px !important;
  }
            """
        }
        
        return styles.get(breakpoint_name, "")
    
    async def _convert_to_dark_palette(self, palette: ColorPalette) -> ColorPalette:
        """Convertit une palette en version sombre"""
        
        # Si on a déjà une palette sombre, l'utiliser
        if "dark" in self.color_palettes:
            return self.color_palettes["dark"]
        
        # Sinon, ajuster les couleurs
        return ColorPalette(
            name=f"{palette.name}_dark",
            primary=await self._adjust_color_brightness(palette.primary, 0.1),
            secondary=palette.secondary,
            accent=await self._adjust_color_brightness(palette.accent, 0.1),
            background="#121212",
            surface="#1e1e1e",
            text="#ffffff",
            text_secondary="#adb5bd",
            border="#495057",
            success=await self._adjust_color_brightness(palette.success, -0.1),
            warning=await self._adjust_color_brightness(palette.warning, -0.1),
            error=await self._adjust_color_brightness(palette.error, -0.1),
            info=await self._adjust_color_brightness(palette.info, -0.1)
        )
    
    async def _convert_to_high_contrast_palette(self, palette: ColorPalette) -> ColorPalette:
        """Convertit en palette haute accessibilité"""
        
        return ColorPalette(
            name=f"{palette.name}_accessible",
            primary="#0056b3",
            secondary="#495057",
            accent="#117a8b",
            background="#ffffff",
            surface="#f8f9fa",
            text="#000000",
            text_secondary="#495057",
            border="#6c757d",
            success="#155724",
            warning="#856404",
            error="#721c24",
            info="#0c5460"
        )
    
    async def _adjust_color_brightness(self, hex_color: str, adjustment: float) -> str:
        """Ajuste la luminosité d'une couleur"""
        
        # Conversion hex vers RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Conversion RGB vers HLS
        h, l, s = rgb_to_hls(r/255.0, g/255.0, b/255.0)
        
        # Ajustement de la luminosité
        l = max(0, min(1, l + adjustment))
        
        # Conversion HLS vers RGB
        r, g, b = hls_to_rgb(h, l, s)
        
        # Conversion RGB vers hex
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    async def minify_css(self, css_content: str) -> str:
        """Minifie le CSS"""
        
        # Suppression des commentaires
        css_content = re.sub(r'/\*.*?\*/', '', css_content, flags=re.DOTALL)
        
        # Suppression des espaces inutiles
        css_content = re.sub(r'\s+', ' ', css_content)
        css_content = re.sub(r';\s*}', '}', css_content)
        css_content = re.sub(r'{\s*', '{', css_content)
        css_content = re.sub(r'}\s*', '}', css_content)
        css_content = re.sub(r':\s*', ':', css_content)
        css_content = re.sub(r';\s*', ';', css_content)
        
        return css_content.strip()
    
    async def compile_sass(self, scss_content: str) -> str:
        """Compile SCSS vers CSS"""
        
        if not self.enable_sass:
            return scss_content
        
        try:
            compiled = sass.compile(string=scss_content, output_style='expanded')
            return compiled
        except Exception as e:
            logger.error(f"SASS compilation failed: {e}")
            return scss_content
    
    async def validate_css(self, css_content: str) -> Dict[str, Any]:
        """Valide le CSS"""
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            sheet = self.css_parser.parseString(css_content)
            
            # Vérification des erreurs de parsing
            if sheet.cssRules is None:
                validation_results["valid"] = False
                validation_results["errors"].append("Invalid CSS syntax")
            
            # Vérification des propriétés non supportées par email
            unsupported_properties = [
                'position', 'float', 'z-index', 'transform', 'animation',
                'flexbox', 'grid', 'calc()', 'vh', 'vw'
            ]
            
            for property_name in unsupported_properties:
                if property_name in css_content:
                    validation_results["warnings"].append(
                        f"Property '{property_name}' may not be supported in email clients"
                    )
            
            # Suggestions d'optimisation
            if '@media' in css_content:
                validation_results["suggestions"].append(
                    "Consider inlining critical CSS for better email client support"
                )
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"CSS parsing error: {str(e)}")
        
        return validation_results
    
    async def add_custom_palette(self, palette: ColorPalette):
        """Ajoute une palette personnalisée"""
        
        self.color_palettes[palette.name] = palette
        logger.info(f"Added custom palette: {palette.name}")
    
    async def add_custom_breakpoint(self, breakpoint: ResponsiveBreakpoint):
        """Ajoute un breakpoint personnalisé"""
        
        self.breakpoints[breakpoint.name] = breakpoint
        logger.info(f"Added custom breakpoint: {breakpoint.name}")
    
    async def add_style_rule(self, rule: StyleRule):
        """Ajoute une règle de style"""
        
        self.style_rules.append(rule)
        logger.info(f"Added style rule: {rule.selector}")

# ============================================================================
# Factory Functions
# ============================================================================

def create_css_manager(
    assets_dir: str,
    framework: CSSFramework = CSSFramework.CUSTOM,
    enable_sass: bool = True
) -> AdvancedCSSStyleManager:
    """Factory pour créer un gestionnaire CSS"""
    
    return AdvancedCSSStyleManager(
        assets_dir=assets_dir,
        framework=framework,
        enable_sass=enable_sass
    )

def create_spotify_theme() -> ColorPalette:
    """Crée le thème Spotify"""
    
    return ColorPalette(
        name="spotify",
        primary="#1db954",
        secondary="#191414",
        accent="#1ed760",
        background="#000000",
        surface="#121212",
        text="#ffffff",
        text_secondary="#b3b3b3",
        border="#282828",
        success="#1db954",
        warning="#ff9500",
        error="#e22134",
        info="#00d4ff"
    )

# Export des classes principales
__all__ = [
    "AdvancedCSSStyleManager",
    "CSSFramework",
    "ColorScheme",
    "ColorPalette",
    "ResponsiveBreakpoint",
    "StyleRule",
    "create_css_manager",
    "create_spotify_theme"
]
