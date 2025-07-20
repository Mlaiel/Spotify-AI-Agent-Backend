import logging
from typing import List, Dict, Optional
from .smtp_service import SMTPService
from .template_service import TemplateService
from .email_analytics import EmailAnalytics

logger = logging.getLogger("email_service")

class EmailService:
    """
    Service d’email avancé : orchestration, audit, sécurité, hooks, logique métier, observabilité.
    Utilisé pour notifications, campagnes, transactionnels, IA, analytics, Spotify, etc.
    """
    def __init__(self):
        self.smtp = SMTPService()
        self.templates = TemplateService()
        self.analytics = EmailAnalytics()
        self.hooks = []
    def register_hook(self, hook):
        self.hooks.append(hook)
        logger.info(f"Email hook enregistré: {hook}")
    def send_mail(self, to: List[str], subject: str, template_name: str, context: Dict, attachments: Optional[List[str]] = None, metadata: Optional[Dict] = None):
        body = self.templates.render(template_name, context)
        result = self.smtp.send(to, subject, body, attachments)
        self.analytics.track_send(to, subject, metadata)
        self.audit(to, subject, template_name, result)
        for hook in self.hooks:
            hook(to, subject, template_name, context, result)
        return result
    def audit(self, to, subject, template_name, result):
        logger.info(f"[AUDIT] Email envoyé à {to} | Sujet: {subject} | Template: {template_name} | Résultat: {result}")
