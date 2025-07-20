# Erweiterte WebSocket-Middlewares

Dieses Verzeichnis enthält kritische Middlewares für Sicherheit, Compliance und Robustheit von WebSockets:

- **auth_jwt.py**: JWT-Validierung beim Verbindungsaufbau, Logging, Fehlerbehandlung, DSGVO-Konformität.
- **rate_limiter.py**: Ratenbegrenzung pro Nutzer/IP, Anti-Flood, Logging, Redis-Empfehlung für Produktion.
- **audit_logger.py**: Audit-Logging aller Verbindungen, Trennungen und sensibler Aktionen, erweiterbar für SIEM oder Datenbank.

## Nutzung
```python
from middleware.auth_jwt import require_jwt
from middleware.rate_limiter import InMemoryRateLimiter
from middleware.audit_logger import AuditLogger

rate_limiter = InMemoryRateLimiter()
audit_logger = AuditLogger()

async def websocket_endpoint(websocket, user_id, room=None):
    payload = await require_jwt(websocket)()
    rate_limiter.check(user_id)
    audit_logger.log_event("connect", user_id, room, "Verbindung akzeptiert")
    # ... weitere Handler-Logik ...
```

## Sicherheit & Compliance
- Keine WebSocket-Verbindung ohne gültiges JWT akzeptieren
- Ratenbegrenzung gegen Spam/Flood (Redis in Produktion empfohlen)
- Alle kritischen Aktionen für Audit und DSGVO loggen
- Secret Key in der Konfiguration auslagern

## Erweiterbarkeit
- `audit_logger` an PostgreSQL, MongoDB oder SIEM anbinden
- `InMemoryRateLimiter` durch Redis-Version ersetzen
- Weitere Middlewares (z.B. Anti-Abuse, i18n) nach Bedarf ergänzen
