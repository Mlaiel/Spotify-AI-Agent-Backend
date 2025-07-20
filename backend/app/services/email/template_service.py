import logging
from typing import Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os

logger = logging.getLogger("template_service")

class TemplateService:
    """
    Service de templates avancé : Jinja2, multilingue, dynamique, sécurité, hooks, logique métier.
    Utilisé pour emails transactionnels, campagnes, notifications, IA, Spotify, etc.
    """
    def __init__(self):
        template_dir = os.getenv("EMAIL_TEMPLATE_DIR", "./templates/email")
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"])
        )
        self.hooks = []
    def register_hook(self, hook):
        self.hooks.append(hook)
        logger.info(f"Template hook enregistré: {hook}")
    def render(self, template_name: str, context: Dict) -> str:
        template = self.env.get_template(template_name)
        result = template.render(**context)
        for hook in self.hooks:
            hook(template_name, context, result)
        logger.info(f"Template rendu: {template_name}")
        return result
