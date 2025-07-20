# Spotify AI Agent – Module Sécurité (FR)

Ce module fournit un système de sécurité industriel, clé en main, pour plateformes IA, SaaS et microservices.

## Fonctionnalités
- Gestion des API keys (génération, validation, rotation, permissions)
- Gestion des tokens (création, validation, rotation, blacklist, OAuth2-ready)
- Gestion JWT (création, validation, rotation, FastAPI/OAuth2-ready)
- Gestion des mots de passe (hashing, validation, politique, reset token)
- Chiffrement (AES/Fernet, gestion des clés, hashing)
- Détection de menaces (brute-force, IP, scoring, SIEM/SOC)
- Audit sécurité (RGPD/SOX, actions critiques, conformité)
- Vérification conformité (RGPD, SOX, PCI, audit, reporting)

## Fichiers clés
- `api_key_manager.py` : Cycle de vie API key, permissions, audit
- `token_manager.py` : Cycle de vie token, OAuth2, blacklist
- `jwt_manager.py` : Cycle de vie JWT, OAuth2/FastAPI
- `password_manager.py` : Politique mot de passe, hashing, reset
- `encryption.py` : Chiffrement, gestion des clés, hashing
- `threat_detection.py` : Détection menaces, alertes, SIEM
- `audit_logger.py` : Audit sécurité, conformité
- `compliance_checker.py` : Vérification conformité automatisée
- `__init__.py` : Expose tous les modules pour import direct

## Exemple d’utilisation
```python
from .jwt_manager import JWTManager
from .password_manager import PasswordManager
JWTManager.create_access_token({"sub": "user_id"})
PasswordManager.validate_password("StrongP@ssw0rd")
```

## Prêt pour la production
- Typage strict, gestion d’erreur robuste
- Aucun TODO, aucune logique manquante
- Intégrable dans APIs, microservices, pipelines analytics
- Extensible SSO, OAuth2, Sentry, SIEM

