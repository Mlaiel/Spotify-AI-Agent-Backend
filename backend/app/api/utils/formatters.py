"""
🎵 Spotify AI Agent - Formatting Utilities
==========================================

Utilitaires enterprise pour le formatage et la présentation
avec templates avancés, transformations et export multi-formats.

Architecture:
- Formatage de données complexes
- Templates dynamiques
- Export multi-formats (JSON, XML, CSV, PDF)
- Formatage de réponses API
- Beautification de code
- Génération de rapports

🎖️ Développé par l'équipe d'experts enterprise
"""

import json
import csv
import xml.etree.ElementTree as ET
import xml.dom.minidom
from typing import Any, Dict, List, Optional, Union, TextIO
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from io import StringIO, BytesIO
import base64
import pandas as pd
from jinja2 import Template, Environment, BaseLoader
import markdown
import yaml


# =============================================================================
# FORMATAGE JSON AVANCÉ
# =============================================================================

class EnterpriseJSONEncoder(json.JSONEncoder):
    """Encodeur JSON enterprise avec support types avancés"""
    
    def default(self, obj):
        """Sérialise les objets non-standard"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


def format_json(data: Any, pretty: bool = True, sort_keys: bool = True, 
               ensure_ascii: bool = False) -> str:
    """
    Formate des données en JSON
    
    Args:
        data: Données à formater
        pretty: Indentation pour la lisibilité
        sort_keys: Trier les clés
        ensure_ascii: Forcer l'ASCII
        
    Returns:
        JSON formaté
    """
    kwargs = {
        'cls': EnterpriseJSONEncoder,
        'sort_keys': sort_keys,
        'ensure_ascii': ensure_ascii
    }
    
    if pretty:
        kwargs['indent'] = 2
        kwargs['separators'] = (',', ': ')
    else:
        kwargs['separators'] = (',', ':')
    
    return json.dumps(data, **kwargs)


def format_json_response(data: Any, status: str = 'success', 
                        message: Optional[str] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Formate une réponse API standardisée
    
    Args:
        data: Données de la réponse
        status: Statut de la réponse
        message: Message optionnel
        metadata: Métadonnées additionnelles
        
    Returns:
        Réponse JSON formatée
    """
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    
    if message:
        response['message'] = message
    
    if metadata:
        response['metadata'] = metadata
    
    return format_json(response)


# =============================================================================
# FORMATAGE XML
# =============================================================================

def dict_to_xml(data: Dict[str, Any], root_name: str = 'root') -> str:
    """
    Convertit un dictionnaire en XML
    
    Args:
        data: Dictionnaire à convertir
        root_name: Nom de l'élément racine
        
    Returns:
        XML formaté
    """
    def _build_element(parent: ET.Element, key: str, value: Any) -> None:
        """Construit récursivement les éléments XML"""
        if isinstance(value, dict):
            element = ET.SubElement(parent, key)
            for k, v in value.items():
                _build_element(element, k, v)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    element = ET.SubElement(parent, key)
                    for k, v in item.items():
                        _build_element(element, k, v)
                else:
                    element = ET.SubElement(parent, key)
                    element.text = str(item)
        else:
            element = ET.SubElement(parent, key)
            element.text = str(value) if value is not None else ''
    
    root = ET.Element(root_name)
    
    for key, value in data.items():
        _build_element(root, key, value)
    
    # Formatage avec indentation
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def format_xml_response(data: Any, root_name: str = 'response') -> str:
    """
    Formate une réponse en XML
    
    Args:
        data: Données à formater
        root_name: Nom de l'élément racine
        
    Returns:
        XML formaté
    """
    response_data = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'data': data
    }
    
    return dict_to_xml(response_data, root_name)


# =============================================================================
# FORMATAGE CSV
# =============================================================================

def format_csv(data: List[Dict[str, Any]], delimiter: str = ',', 
              quoting: int = csv.QUOTE_MINIMAL) -> str:
    """
    Formate des données en CSV
    
    Args:
        data: Liste de dictionnaires
        delimiter: Délimiteur de colonnes
        quoting: Style de quotation
        
    Returns:
        CSV formaté
    """
    if not data:
        return ''
    
    output = StringIO()
    
    # Obtenir toutes les clés possibles
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    
    fieldnames = sorted(list(fieldnames))
    
    writer = csv.DictWriter(
        output, 
        fieldnames=fieldnames,
        delimiter=delimiter,
        quoting=quoting
    )
    
    writer.writeheader()
    
    for item in data:
        # Convertir les valeurs non-string
        row = {}
        for key, value in item.items():
            if isinstance(value, (datetime, date)):
                row[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                row[key] = json.dumps(value)
            else:
                row[key] = str(value) if value is not None else ''
        
        writer.writerow(row)
    
    return output.getvalue()


def format_csv_from_dataframe(df: pd.DataFrame, **kwargs) -> str:
    """
    Formate un DataFrame pandas en CSV
    
    Args:
        df: DataFrame à formater
        **kwargs: Arguments pour to_csv
        
    Returns:
        CSV formaté
    """
    return df.to_csv(index=False, **kwargs)


# =============================================================================
# FORMATAGE DE TEXTE
# =============================================================================

def format_table(data: List[Dict[str, Any]], headers: Optional[List[str]] = None,
                alignment: str = 'left', max_width: int = 80) -> str:
    """
    Formate des données en tableau texte
    
    Args:
        data: Données tabulaires
        headers: En-têtes de colonnes
        alignment: Alignement (left, center, right)
        max_width: Largeur maximale
        
    Returns:
        Tableau formaté
    """
    if not data:
        return ''
    
    # Déterminer les colonnes
    if headers is None:
        headers = list(data[0].keys())
    
    # Calculer les largeurs de colonnes
    col_widths = {header: len(header) for header in headers}
    
    for row in data:
        for header in headers:
            value = str(row.get(header, ''))
            col_widths[header] = max(col_widths[header], len(value))
    
    # Ajuster si nécessaire
    total_width = sum(col_widths.values()) + len(headers) * 3 + 1
    if total_width > max_width:
        # Réduire proportionnellement
        available_width = max_width - len(headers) * 3 - 1
        total_original = sum(col_widths.values())
        for header in headers:
            col_widths[header] = int(col_widths[header] * available_width / total_original)
    
    # Fonction d'alignement
    def align_text(text: str, width: int) -> str:
        text = text[:width]  # Tronquer si nécessaire
        if alignment == 'center':
            return text.center(width)
        elif alignment == 'right':
            return text.rjust(width)
        else:
            return text.ljust(width)
    
    # Construire le tableau
    lines = []
    
    # Ligne de séparation
    sep_line = '+' + '+'.join('-' * (col_widths[h] + 2) for h in headers) + '+'
    lines.append(sep_line)
    
    # En-têtes
    header_line = '|' + '|'.join(f' {align_text(h, col_widths[h])} ' for h in headers) + '|'
    lines.append(header_line)
    lines.append(sep_line)
    
    # Données
    for row in data:
        row_line = '|' + '|'.join(
            f' {align_text(str(row.get(h, "")), col_widths[h])} ' 
            for h in headers
        ) + '|'
        lines.append(row_line)
    
    lines.append(sep_line)
    
    return '\n'.join(lines)


def format_list(items: List[Any], bullet: str = '•', indent: str = '  ') -> str:
    """
    Formate une liste avec des puces
    
    Args:
        items: Éléments de la liste
        bullet: Caractère de puce
        indent: Indentation
        
    Returns:
        Liste formatée
    """
    return '\n'.join(f'{indent}{bullet} {item}' for item in items)


def format_key_value(data: Dict[str, Any], separator: str = ': ', 
                    indent: str = '', max_key_width: Optional[int] = None) -> str:
    """
    Formate un dictionnaire clé-valeur
    
    Args:
        data: Dictionnaire à formater
        separator: Séparateur clé-valeur
        indent: Indentation
        max_key_width: Largeur maximale des clés
        
    Returns:
        Texte formaté
    """
    if max_key_width is None:
        max_key_width = max(len(str(key)) for key in data.keys()) if data else 0
    
    lines = []
    for key, value in data.items():
        key_str = str(key).ljust(max_key_width)
        
        if isinstance(value, dict):
            lines.append(f'{indent}{key_str}{separator}')
            lines.append(format_key_value(value, separator, indent + '  ', max_key_width))
        elif isinstance(value, list):
            lines.append(f'{indent}{key_str}{separator}')
            for item in value:
                lines.append(f'{indent}  - {item}')
        else:
            lines.append(f'{indent}{key_str}{separator}{value}')
    
    return '\n'.join(lines)


# =============================================================================
# TEMPLATES DYNAMIQUES
# =============================================================================

class TemplateFormatter:
    """Formateur de templates dynamiques"""
    
    def __init__(self):
        self.env = Environment(loader=BaseLoader())
        self.templates: Dict[str, Template] = {}
    
    def add_template(self, name: str, template_string: str) -> None:
        """
        Ajoute un template
        
        Args:
            name: Nom du template
            template_string: Contenu du template Jinja2
        """
        self.templates[name] = self.env.from_string(template_string)
    
    def render_template(self, name: str, data: Dict[str, Any]) -> str:
        """
        Rend un template avec des données
        
        Args:
            name: Nom du template
            data: Données à injecter
            
        Returns:
            Template rendu
        """
        if name not in self.templates:
            raise ValueError(f'Template {name} non trouvé')
        
        return self.templates[name].render(**data)
    
    def render_string(self, template_string: str, data: Dict[str, Any]) -> str:
        """
        Rend un template à partir d'une chaîne
        
        Args:
            template_string: Template Jinja2
            data: Données à injecter
            
        Returns:
            Template rendu
        """
        template = self.env.from_string(template_string)
        return template.render(**data)


# Templates prédéfinis
EMAIL_TEMPLATE = """
Bonjour {{ name }},

{{ message }}

{% if items %}
Détails:
{% for item in items %}
- {{ item }}
{% endfor %}
{% endif %}

Cordialement,
{{ sender }}
"""

REPORT_TEMPLATE = """
# Rapport {{ title }}

**Généré le**: {{ timestamp }}
**Auteur**: {{ author }}

## Résumé

{{ summary }}

## Données

{% for section in sections %}
### {{ section.title }}

{{ section.content }}

{% if section.data %}
| Métrique | Valeur |
|----------|---------|
{% for key, value in section.data.items() %}
| {{ key }} | {{ value }} |
{% endfor %}
{% endif %}

{% endfor %}

## Conclusion

{{ conclusion }}
"""


# =============================================================================
# FORMATAGE SPÉCIALISÉ
# =============================================================================

def format_file_size(size_bytes: int, decimal_places: int = 2) -> str:
    """
    Formate une taille de fichier
    
    Args:
        size_bytes: Taille en bytes
        decimal_places: Nombre de décimales
        
    Returns:
        Taille formatée (ex: "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.{decimal_places}f} {units[unit_index]}"


def format_duration(seconds: float, precision: str = 'auto') -> str:
    """
    Formate une durée
    
    Args:
        seconds: Durée en secondes
        precision: Précision (auto, hours, minutes, seconds)
        
    Returns:
        Durée formatée
    """
    if seconds < 0:
        return "0:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if precision == 'hours' or (precision == 'auto' and hours > 0):
        return f"{hours}:{minutes:02d}:{secs:02d}"
    elif precision == 'minutes' or (precision == 'auto' and minutes > 0):
        return f"{minutes}:{secs:02d}"
    else:
        return f"0:{secs:02d}"


def format_percentage(value: float, decimal_places: int = 1, 
                     include_symbol: bool = True) -> str:
    """
    Formate un pourcentage
    
    Args:
        value: Valeur (0-1 ou 0-100)
        decimal_places: Nombre de décimales
        include_symbol: Inclure le symbole %
        
    Returns:
        Pourcentage formaté
    """
    # Détecter si la valeur est entre 0-1 ou 0-100
    if value <= 1.0:
        percentage = value * 100
    else:
        percentage = value
    
    formatted = f"{percentage:.{decimal_places}f}"
    
    if include_symbol:
        formatted += "%"
    
    return formatted


def format_currency(amount: float, currency: str = 'EUR', 
                   decimal_places: int = 2) -> str:
    """
    Formate un montant monétaire
    
    Args:
        amount: Montant
        currency: Code devise
        decimal_places: Nombre de décimales
        
    Returns:
        Montant formaté
    """
    formatted_amount = f"{amount:,.{decimal_places}f}"
    
    # Symboles de devises
    currency_symbols = {
        'EUR': '€',
        'USD': '$',
        'GBP': '£',
        'JPY': '¥',
        'CHF': 'CHF',
        'CAD': 'C$'
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    if currency in ['USD', 'CAD']:
        return f"{symbol}{formatted_amount}"
    else:
        return f"{formatted_amount} {symbol}"


# =============================================================================
# EXPORT MULTI-FORMATS
# =============================================================================

class MultiFormatExporter:
    """Exporteur multi-formats"""
    
    def __init__(self):
        self.template_formatter = TemplateFormatter()
    
    def export_data(self, data: Any, format: str, **options) -> Union[str, bytes]:
        """
        Exporte des données dans le format spécifié
        
        Args:
            data: Données à exporter
            format: Format de sortie
            **options: Options spécifiques au format
            
        Returns:
            Données exportées
        """
        format = format.lower()
        
        if format == 'json':
            return format_json(data, **options)
        
        elif format == 'xml':
            root_name = options.get('root_name', 'data')
            return dict_to_xml(data, root_name)
        
        elif format == 'csv':
            if isinstance(data, list):
                return format_csv(data, **options)
            else:
                raise ValueError('CSV nécessite une liste de dictionnaires')
        
        elif format == 'yaml':
            return yaml.dump(data, default_flow_style=False, **options)
        
        elif format == 'markdown':
            if isinstance(data, str):
                return markdown.markdown(data)
            else:
                # Convertir en format lisible
                return self._data_to_markdown(data)
        
        elif format == 'html':
            if isinstance(data, str):
                return markdown.markdown(data)
            else:
                return self._data_to_html(data)
        
        else:
            raise ValueError(f'Format non supporté: {format}')
    
    def _data_to_markdown(self, data: Any) -> str:
        """Convertit des données en Markdown"""
        if isinstance(data, dict):
            lines = ['# Données\n']
            for key, value in data.items():
                lines.append(f"## {key}\n")
                if isinstance(value, (list, dict)):
                    lines.append(f"```json\n{format_json(value)}\n```\n")
                else:
                    lines.append(f"{value}\n")
            return '\n'.join(lines)
        
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                # Table markdown
                if not data:
                    return 'Aucune donnée'
                
                headers = list(data[0].keys())
                lines = ['| ' + ' | '.join(headers) + ' |']
                lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
                
                for item in data:
                    row = [str(item.get(h, '')) for h in headers]
                    lines.append('| ' + ' | '.join(row) + ' |')
                
                return '\n'.join(lines)
            else:
                return '\n'.join(f"- {item}" for item in data)
        
        else:
            return str(data)
    
    def _data_to_html(self, data: Any) -> str:
        """Convertit des données en HTML"""
        md_content = self._data_to_markdown(data)
        return markdown.markdown(md_content, extensions=['tables'])


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnterpriseJSONEncoder",
    "format_json",
    "format_json_response",
    "dict_to_xml",
    "format_xml_response",
    "format_csv",
    "format_csv_from_dataframe",
    "format_table",
    "format_list",
    "format_key_value",
    "TemplateFormatter",
    "EMAIL_TEMPLATE",
    "REPORT_TEMPLATE",
    "format_file_size",
    "format_duration",
    "format_percentage",
    "format_currency",
    "MultiFormatExporter"
]
