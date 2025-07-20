#!/usr/bin/env python3
"""
Spotify AI Agent - Advanced Tenancy Tools & Scripts Package
===========================================================

Comprehensive industrialized solution for multi-tenant monitoring,
alerting, and notification management with Slack integration.

This package provides enterprise-grade tools for:
- Advanced alertmanager configurations and templating
- Multi-locale Slack notification management  
- Sophisticated monitoring and alerting workflows
- Production-ready automation scripts
- Security-first implementation with enterprise standards
- Scalable multi-tenant architecture

Architecture Components:
- Alertmanager integration and configuration management
- Slack receivers with advanced templating
- Multi-language localization support
- DevOps automation tools and scripts
- Monitoring manifests for Kubernetes/Docker environments
- Enterprise security and compliance features

Author: Fahed Mlaiel (Lead Developer + AI Architect)
Contributors: Expert Development Team
License: Enterprise - All Rights Reserved
Version: 1.0.0 (Production Ready)
"""

from .monitoring_manager import (
    MonitoringManager,
    AlertmanagerConfigManager,
    SlackNotificationManager
)
from .script_executor import (
    ScriptExecutor,
    AutomationEngine,
    TaskScheduler
)
from .locale_manager import (
    LocaleManager,
    MultiLanguageHandler,
    TranslationEngine
)
from .security_manager import (
    SecurityManager,
    ComplianceChecker,
    AuditLogger
)

__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__team__ = "Expert Development Team"
__status__ = "Production"

# Package configuration
PACKAGE_CONFIG = {
    "name": "tenancy-monitoring-tools",
    "version": __version__,
    "description": "Enterprise tenancy monitoring and alerting solution",
    "author": __author__,
    "team": __team__,
    "status": __status__,
    "features": [
        "Advanced Alertmanager Integration",
        "Multi-tenant Slack Notifications", 
        "Real-time Monitoring Dashboards",
        "Automated DevOps Workflows",
        "Enterprise Security & Compliance",
        "Multi-language Support",
        "Production-ready Automation"
    ],
    "components": {
        "monitoring": MonitoringManager,
        "scripts": ScriptExecutor,
        "locales": LocaleManager,
        "security": SecurityManager
    }
}

# Export all public interfaces
__all__ = [
    "MonitoringManager",
    "AlertmanagerConfigManager", 
    "SlackNotificationManager",
    "ScriptExecutor",
    "AutomationEngine",
    "TaskScheduler",
    "LocaleManager",
    "MultiLanguageHandler",
    "TranslationEngine",
    "SecurityManager",
    "ComplianceChecker",
    "AuditLogger",
    "PACKAGE_CONFIG"
]
