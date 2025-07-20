"""
Enterprise Template Collection for Warning System
Comprehensive template library for notifications, alerts, reports and documentation
Multi-format support with advanced customization and localization
"""

__version__ = "1.0.0"
__author__ = "Enterprise Development Team"

from .email import EmailTemplates
from .slack import SlackTemplates  
from .sms import SMSTemplates
from .dashboard import DashboardTemplates
from .reports import ReportTemplates
from .documentation import DocumentationTemplates

__all__ = [
    "EmailTemplates",
    "SlackTemplates", 
    "SMSTemplates",
    "DashboardTemplates",
    "ReportTemplates",
    "DocumentationTemplates"
]
