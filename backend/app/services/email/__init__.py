"""
Spotify AI Agent – Email Module

Created by: Achiri AI Engineering Team
Roles: Lead Dev + Architecte IA, Développeur Backend Senior, Ingénieur ML, DBA/Data Engineer, Spécialiste Sécurité, Architecte Microservices
"""
from .email_service import EmailService
from .smtp_service import SMTPService
from .template_service import TemplateService
from .email_analytics import EmailAnalytics

__version__ = "1.0.0"
__all__ = [
    "EmailService",
    "SMTPService",
    "TemplateService",
    "EmailAnalytics",
]
