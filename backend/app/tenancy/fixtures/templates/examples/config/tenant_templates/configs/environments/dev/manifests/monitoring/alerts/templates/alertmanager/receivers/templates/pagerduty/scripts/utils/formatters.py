#!/usr/bin/env python3
"""
Advanced Formatters for PagerDuty Integration

Module de formatage avancé pour les messages, données, et rapports PagerDuty.
Fournit des formateurs sophistiqués pour différents types de sorties,
avec support multi-format et templates personnalisables.

Fonctionnalités:
- Formatage de messages d'alerte
- Templates de notification personnalisables
- Formatage de données structurées
- Formatage de rapports et métriques
- Support multi-format (JSON, YAML, HTML, Markdown)
- Formatage adaptatif selon le contexte
- Internationalisation basique

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from string import Template
import yaml
import html

class MessageFormatter:
    """Formateur de messages pour PagerDuty"""
    
    # Templates par défaut pour les messages
    DEFAULT_TEMPLATES = {
        "alert": "🚨 ALERT: {severity} - {summary}",
        "incident_trigger": "🔥 New incident: {title} (ID: {incident_id})",
        "incident_acknowledge": "👀 Incident acknowledged: {title}",
        "incident_resolve": "✅ Incident resolved: {title}",
        "escalation": "⚠️ Incident escalated: {title} → {escalation_level}",
        "notification": "📢 {type}: {message}",
        "status_update": "📊 Status: {service} is {status}",
        "metric_alert": "📈 Metric Alert: {metric_name} = {value} {operator} {threshold}"
    }
    
    # Mapping des sévérités vers des emojis
    SEVERITY_EMOJIS = {
        "critical": "🔴",
        "high": "🟠", 
        "medium": "🟡",
        "low": "🟢",
        "info": "🔵"
    }
    
    # Mapping des statuts vers des emojis
    STATUS_EMOJIS = {
        "up": "✅",
        "down": "❌",
        "degraded": "⚠️",
        "maintenance": "🔧",
        "unknown": "❓"
    }
    
    def __init__(self, templates: Optional[Dict[str, str]] = None):
        self.templates = {**self.DEFAULT_TEMPLATES, **(templates or {})}
    
    def format_alert_message(
        self,
        severity: str,
        summary: str,
        source: str = "",
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Formate un message d'alerte"""
        
        emoji = self.SEVERITY_EMOJIS.get(severity.lower(), "🔔")
        timestamp_str = ""
        
        if timestamp:
            timestamp_str = f" at {timestamp.strftime('%H:%M:%S UTC')}"
        
        base_message = f"{emoji} **{severity.upper()}**: {summary}"
        
        if source:
            base_message += f" (from {source})"
        
        if timestamp_str:
            base_message += timestamp_str
        
        if details:
            formatted_details = self._format_details(details)
            if formatted_details:
                base_message += f"\n\n**Details:**\n{formatted_details}"
        
        return base_message
    
    def format_incident_message(
        self,
        action: str,
        incident_id: str,
        title: str,
        status: str = "",
        assigned_to: str = "",
        escalation_level: Optional[int] = None
    ) -> str:
        """Formate un message d'incident"""
        
        action_emojis = {
            "trigger": "🔥",
            "acknowledge": "👀",
            "resolve": "✅",
            "escalate": "⚠️",
            "assign": "👤",
            "note": "💬"
        }
        
        emoji = action_emojis.get(action, "📢")
        message = f"{emoji} **Incident {action.title()}**\n"
        message += f"**ID:** {incident_id}\n"
        message += f"**Title:** {title}\n"
        
        if status:
            message += f"**Status:** {status}\n"
        
        if assigned_to:
            message += f"**Assigned to:** {assigned_to}\n"
        
        if escalation_level is not None:
            message += f"**Escalation Level:** {escalation_level}\n"
        
        return message.strip()
    
    def format_status_message(
        self,
        service: str,
        status: str,
        previous_status: str = "",
        uptime: str = "",
        response_time: Optional[float] = None
    ) -> str:
        """Formate un message de statut de service"""
        
        emoji = self.STATUS_EMOJIS.get(status.lower(), "❓")
        message = f"{emoji} **{service}** is **{status.upper()}**"
        
        if previous_status and previous_status != status:
            prev_emoji = self.STATUS_EMOJIS.get(previous_status.lower(), "❓")
            message += f" (was {prev_emoji} {previous_status})"
        
        details = []
        if uptime:
            details.append(f"Uptime: {uptime}")
        
        if response_time is not None:
            details.append(f"Response time: {response_time:.2f}ms")
        
        if details:
            message += f"\n{' | '.join(details)}"
        
        return message
    
    def format_metric_alert(
        self,
        metric_name: str,
        current_value: Union[int, float],
        threshold: Union[int, float],
        operator: str,
        unit: str = "",
        trend: Optional[str] = None
    ) -> str:
        """Formate une alerte de métrique"""
        
        operator_symbols = {
            "gt": ">",
            "gte": "≥", 
            "lt": "<",
            "lte": "≤",
            "eq": "=",
            "ne": "≠"
        }
        
        symbol = operator_symbols.get(operator, operator)
        unit_str = f" {unit}" if unit else ""
        
        message = f"📈 **Metric Alert: {metric_name}**\n"
        message += f"Current value: **{current_value}{unit_str}**\n"
        message += f"Threshold: {symbol} {threshold}{unit_str}\n"
        
        if trend:
            trend_emojis = {
                "increasing": "📈",
                "decreasing": "📉",
                "stable": "➡️"
            }
            trend_emoji = trend_emojis.get(trend, "")
            message += f"Trend: {trend_emoji} {trend}"
        
        return message
    
    def format_notification_summary(
        self,
        total_alerts: int,
        critical_count: int = 0,
        high_count: int = 0,
        services_affected: Optional[List[str]] = None,
        time_range: str = "last hour"
    ) -> str:
        """Formate un résumé de notifications"""
        
        message = f"📊 **Alert Summary** ({time_range})\n\n"
        message += f"Total alerts: **{total_alerts}**\n"
        
        if critical_count > 0:
            message += f"🔴 Critical: {critical_count}\n"
        
        if high_count > 0:
            message += f"🟠 High: {high_count}\n"
        
        if services_affected:
            message += f"\n**Services affected:** {', '.join(services_affected[:5])}"
            if len(services_affected) > 5:
                message += f" and {len(services_affected) - 5} more"
        
        return message
    
    def _format_details(self, details: Dict[str, Any], max_depth: int = 2) -> str:
        """Formate les détails sous forme de liste"""
        lines = []
        
        for key, value in details.items():
            if isinstance(value, dict) and max_depth > 0:
                lines.append(f"**{key}:**")
                sub_details = self._format_details(value, max_depth - 1)
                for line in sub_details.split('\n'):
                    lines.append(f"  {line}")
            elif isinstance(value, list):
                lines.append(f"**{key}:** {', '.join(map(str, value[:3]))}")
                if len(value) > 3:
                    lines.append(f"  ... and {len(value) - 3} more")
            else:
                # Tronquer les valeurs très longues
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:97] + "..."
                lines.append(f"**{key}:** {str_value}")
        
        return '\n'.join(lines)
    
    def format_custom_message(self, template_name: str, **kwargs) -> str:
        """Formate un message avec un template personnalisé"""
        
        if template_name not in self.templates:
            return f"Template '{template_name}' not found"
        
        try:
            template = Template(self.templates[template_name])
            return template.safe_substitute(**kwargs)
        except Exception as e:
            return f"Error formatting template: {str(e)}"

class DataFormatter:
    """Formateur de données structurées"""
    
    @staticmethod
    def format_json(data: Any, indent: int = 2, sort_keys: bool = True) -> str:
        """Formate des données en JSON"""
        try:
            return json.dumps(
                data,
                indent=indent,
                sort_keys=sort_keys,
                default=DataFormatter._json_serializer,
                ensure_ascii=False
            )
        except Exception as e:
            return f"Error formatting JSON: {str(e)}"
    
    @staticmethod
    def format_yaml(data: Any, default_flow_style: bool = False) -> str:
        """Formate des données en YAML"""
        try:
            return yaml.dump(
                data,
                default_flow_style=default_flow_style,
                allow_unicode=True,
                sort_keys=True
            )
        except Exception as e:
            return f"Error formatting YAML: {str(e)}"
    
    @staticmethod
    def format_table(
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        max_width: int = 80
    ) -> str:
        """Formate des données en tableau texte"""
        
        if not data:
            return "No data available"
        
        # Détecter les colonnes si non spécifiées
        if columns is None:
            columns = list(data[0].keys())
        
        # Calculer la largeur des colonnes
        col_widths = {}
        for col in columns:
            max_len = len(col)
            for row in data:
                cell_value = str(row.get(col, ""))
                max_len = max(max_len, len(cell_value))
            col_widths[col] = min(max_len, max_width // len(columns))
        
        # Créer l'en-tête
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        separator = "-" * len(header)
        
        # Créer les lignes de données
        rows = []
        for row in data:
            row_parts = []
            for col in columns:
                cell_value = str(row.get(col, ""))
                if len(cell_value) > col_widths[col]:
                    cell_value = cell_value[:col_widths[col]-3] + "..."
                row_parts.append(cell_value.ljust(col_widths[col]))
            rows.append(" | ".join(row_parts))
        
        return f"{header}\n{separator}\n" + "\n".join(rows)
    
    @staticmethod
    def format_key_value_pairs(
        data: Dict[str, Any],
        separator: str = ": ",
        max_key_width: int = 20
    ) -> str:
        """Formate des données en paires clé-valeur"""
        
        lines = []
        for key, value in data.items():
            key_formatted = key.ljust(max_key_width)
            value_str = DataFormatter._format_value(value)
            lines.append(f"{key_formatted}{separator}{value_str}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_list(
        items: List[Any],
        bullet: str = "•",
        numbered: bool = False,
        max_items: Optional[int] = None
    ) -> str:
        """Formate une liste d'éléments"""
        
        if not items:
            return "No items"
        
        if max_items and len(items) > max_items:
            display_items = items[:max_items]
            truncated = True
        else:
            display_items = items
            truncated = False
        
        lines = []
        for i, item in enumerate(display_items, 1):
            if numbered:
                prefix = f"{i}."
            else:
                prefix = bullet
            
            value_str = DataFormatter._format_value(item)
            lines.append(f"{prefix} {value_str}")
        
        if truncated:
            lines.append(f"... and {len(items) - max_items} more items")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_value(value: Any) -> str:
        """Formate une valeur individuelle"""
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M:%S UTC")
        elif isinstance(value, (dict, list)):
            return json.dumps(value, default=DataFormatter._json_serializer)
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif value is None:
            return "N/A"
        else:
            return str(value)
    
    @staticmethod
    def _json_serializer(obj: Any) -> str:
        """Sérialiseur personnalisé pour JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

class ReportFormatter:
    """Formateur de rapports"""
    
    def __init__(self, title: str = "Report"):
        self.title = title
        self.sections = []
    
    def add_section(self, title: str, content: str, level: int = 1):
        """Ajoute une section au rapport"""
        self.sections.append({
            "title": title,
            "content": content,
            "level": level
        })
    
    def add_summary(self, data: Dict[str, Any]):
        """Ajoute une section de résumé"""
        summary_content = DataFormatter.format_key_value_pairs(data)
        self.add_section("Summary", summary_content, 1)
    
    def add_table(self, title: str, data: List[Dict[str, Any]], columns: Optional[List[str]] = None):
        """Ajoute un tableau au rapport"""
        table_content = DataFormatter.format_table(data, columns)
        self.add_section(title, table_content, 2)
    
    def add_metrics(self, title: str, metrics: Dict[str, Union[int, float]]):
        """Ajoute des métriques au rapport"""
        metrics_lines = []
        for name, value in metrics.items():
            if isinstance(value, float):
                metrics_lines.append(f"• {name}: {value:.2f}")
            else:
                metrics_lines.append(f"• {name}: {value}")
        
        self.add_section(title, "\n".join(metrics_lines), 2)
    
    def generate_markdown(self) -> str:
        """Génère le rapport en Markdown"""
        lines = [f"# {self.title}\n"]
        
        # Ajouter la date de génération
        now = datetime.now(timezone.utc)
        lines.append(f"*Generated on {now.strftime('%Y-%m-%d at %H:%M:%S UTC')}*\n")
        
        for section in self.sections:
            level = "#" * (section["level"] + 1)
            lines.append(f"{level} {section['title']}\n")
            lines.append(f"{section['content']}\n")
        
        return "\n".join(lines)
    
    def generate_html(self) -> str:
        """Génère le rapport en HTML"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{html.escape(self.title)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        pre {{ background-color: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{html.escape(self.title)}</h1>
    <p><em>Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d at %H:%M:%S UTC')}</em></p>
"""
        
        for section in self.sections:
            level = section["level"] + 1
            title_escaped = html.escape(section["title"])
            content_escaped = html.escape(section["content"])
            
            html_content += f"    <h{level}>{title_escaped}</h{level}>\n"
            html_content += f"    <pre>{content_escaped}</pre>\n"
        
        html_content += """
</body>
</html>
"""
        return html_content
    
    def generate_text(self) -> str:
        """Génère le rapport en texte brut"""
        lines = [
            "=" * len(self.title),
            self.title,
            "=" * len(self.title),
            "",
            f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d at %H:%M:%S UTC')}",
            ""
        ]
        
        for section in self.sections:
            # Titre de section
            title_line = section["title"]
            if section["level"] == 1:
                lines.extend([title_line, "-" * len(title_line)])
            else:
                lines.append(f"{'  ' * (section['level'] - 1)}{title_line}")
            
            # Contenu
            content_lines = section["content"].split('\n')
            for line in content_lines:
                lines.append(f"{'  ' * section['level']}{line}")
            
            lines.append("")  # Ligne vide
        
        return "\n".join(lines)

class TemplateFormatter:
    """Formateur basé sur des templates"""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = Path(template_dir) if template_dir else None
        self.templates = {}
    
    def load_template(self, name: str, content: Optional[str] = None) -> bool:
        """Charge un template"""
        try:
            if content:
                self.templates[name] = Template(content)
            elif self.template_dir:
                template_file = self.template_dir / f"{name}.txt"
                if template_file.exists():
                    with open(template_file, 'r', encoding='utf-8') as f:
                        self.templates[name] = Template(f.read())
                    return True
            return False
        except Exception:
            return False
    
    def format_with_template(self, template_name: str, **kwargs) -> Optional[str]:
        """Formate avec un template"""
        if template_name not in self.templates:
            return None
        
        try:
            return self.templates[template_name].safe_substitute(**kwargs)
        except Exception:
            return None
    
    def list_templates(self) -> List[str]:
        """Liste les templates disponibles"""
        return list(self.templates.keys())

# Fonctions utilitaires
def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Tronque un texte si nécessaire"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_duration(seconds: float) -> str:
    """Formate une durée en format lisible"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_percentage(value: float, total: float, decimals: int = 1) -> str:
    """Formate un pourcentage"""
    if total == 0:
        return "0%"
    percentage = (value / total) * 100
    return f"{percentage:.{decimals}f}%"

def format_file_size(size_bytes: int) -> str:
    """Formate une taille de fichier"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

# Export des classes principales
__all__ = [
    "MessageFormatter",
    "DataFormatter", 
    "ReportFormatter",
    "TemplateFormatter",
    "truncate_text",
    "format_duration",
    "format_percentage",
    "format_file_size"
]
