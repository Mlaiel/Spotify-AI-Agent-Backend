# Spotify AI Agent – Zentrale Konfiguration (DE)

## Übersicht
Dieses Verzeichnis enthält alle Konfigurationsmodule für das Backend des Spotify AI Agenten. Jede Konfiguration ist produktionsreif, sicher, validiert und direkt nutzbar. Keine TODOs oder Platzhalter.

---

## Architektur
- **settings.py**: Zentrale Pydantic-Konfiguration, lädt aus .env, validiert alle kritischen Einstellungen
- **ai_config.py**: KI/ML-Modelle, Provider, Moderation, Sicherheit
- **database_config.py**: PostgreSQL, MongoDB, Redis, Pooling, Sicherheit
- **environment_config.py**: Umgebung, Debug, Version, Region, Zeitzone
- **redis_config.py**: Erweiterte Redis-Konfiguration (Cluster, SSL, Timeouts)
- **security_config.py**: JWT, CORS, CSP, Brute-Force, Sicherheitsrichtlinien
- **spotify_config.py**: Spotify-API-Integration (OAuth2, Scopes, Endpunkte)

---

## Sicherheit & Compliance
- Alle Secrets werden aus Umgebungsvariablen oder .env geladen
- Pydantic-Validierung für alle Konfigurationen
- Keine sensiblen Daten im Code

## Erweiterbarkeit
- Jede Konfiguration ist modular und kann je nach Umgebung erweitert werden
- Bereit für CI/CD, Cloud und Microservices

## Beispielnutzung
```python
from core.config import settings, AIConfig, DatabaseConfig, SecurityConfig
print(settings.secret_key)
```

---

## Siehe auch
- [README.md](./README.md) (English)
- [README.fr.md](./README.fr.md) (Français)

