# Middlewares WebSocket avancés

Ce dossier contient des middlewares critiques pour la sécurité, la conformité et la robustesse des WebSocket :

- **auth_jwt.py** : validation du JWT à l’ouverture de la connexion, logs, gestion des erreurs, conformité RGPD.
- **rate_limiter.py** : limitation de débit par utilisateur/IP, protection anti-flood, logs, recommandation Redis pour la prod.
- **audit_logger.py** : audit logging de toutes les connexions, déconnexions et actions sensibles, extensible vers SIEM ou base de données.

## Utilisation
```python
from middleware.auth_jwt import require_jwt
from middleware.rate_limiter import InMemoryRateLimiter
from middleware.audit_logger import AuditLogger

rate_limiter = InMemoryRateLimiter()
audit_logger = AuditLogger()

async def websocket_endpoint(websocket, user_id, room=None):
    payload = await require_jwt(websocket)()
    rate_limiter.check(user_id)
    audit_logger.log_event("connect", user_id, room, "Connexion acceptée")
    # ... suite du handler ...
```

## Sécurité & conformité
- Ne jamais accepter de connexion WebSocket sans JWT valide
- Limiter le débit pour éviter le spam/flood (Redis recommandé en prod)
- Logger toutes les actions critiques pour l’audit et la conformité RGPD
- Externaliser la clé secrète dans la configuration

## Extensibilité
- Brancher `audit_logger` sur PostgreSQL, MongoDB ou SIEM pour l’audit centralisé
- Remplacer `InMemoryRateLimiter` par une version Redis pour la scalabilité
- Ajouter d’autres middlewares (ex : anti-abus, i18n, etc.) selon les besoins
