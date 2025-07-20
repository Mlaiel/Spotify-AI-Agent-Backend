# Agent IA Spotify – Configuration centrale (FR)

## Vue d’ensemble
Ce dossier regroupe tous les modules de configuration du backend Agent IA Spotify. Chaque config est clé en main, sécurisée, validée, et directement exploitable. Aucun TODO ni placeholder.

---

## Architecture
- **settings.py** : Config centrale Pydantic, chargement .env, validation de tous les paramètres critiques
- **ai_config.py** : Modèles IA/ML, providers, modération, sécurité
- **database_config.py** : PostgreSQL, MongoDB, Redis, pooling, sécurité
- **environment_config.py** : Environnement, debug, version, région, timezone
- **redis_config.py** : Config Redis avancée (cluster, SSL, timeouts)
- **security_config.py** : JWT, CORS, CSP, brute-force, politiques de sécurité
- **spotify_config.py** : Intégration API Spotify (OAuth2, scopes, endpoints)

---

## Sécurité & conformité
- Tous les secrets sont chargés depuis les variables d’environnement ou .env
- Validation Pydantic sur toutes les configs
- Aucune donnée sensible en dur

## Extensibilité
- Chaque config est modulaire et extensible selon l’environnement
- Prêt pour CI/CD, cloud, microservices

## Exemple d’utilisation
```python
from core.config import settings, AIConfig, DatabaseConfig, SecurityConfig
print(settings.secret_key)
```

---

## Voir aussi
- [README.md](./README.md) (English)
- [README.de.md](./README.de.md) (Deutsch)

