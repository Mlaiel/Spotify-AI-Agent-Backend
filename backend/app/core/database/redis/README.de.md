# Redis Modul für Spotify AI Agent (DE)

Dieses Modul bietet eine sichere, skalierbare und industrialisierte Redis-Integration für KI-gestützte Musik- und Analyseanwendungen.

## Creator Team (Rollen)
- Lead Dev & KI-Architekt: [Ergänzen]
- Senior Backend Entwickler (Python/FastAPI/Django): [Ergänzen]
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face): [Ergänzen]
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB): [Ergänzen]
- Backend Security Spezialist: [Ergänzen]
- Microservices Architekt: [Ergänzen]

## Features:
- Sicherer Cache Manager, TTL, Invalidation, Logs, DI ready
- Cluster Manager (Auto-Discovery, Failover, Monitoring)
- Pub/Sub Manager (Channels, Events, Hooks, Logs)
- Rate Limiter (Token Bucket, Sliding Window, Anti-Abuse, Logs)
- Session Store (Verschlüsselung, Ablauf, Audit, Sicherheit)

**Beispiel:**
```python
from .redis import get_cache
cache = get_cache()
cache.set("user:1", {"name": "Alice"})
```

## Sicherheit:
- Niemals Zugangsdaten hardcoden
- TLS & Auth immer aktivieren
- Monitoring, Audit, Schlüsselrotation

## Troubleshooting:
- Logs bei Verbindungs- oder Clusterproblemen prüfen
- Integrierte Health-Checks nutzen

