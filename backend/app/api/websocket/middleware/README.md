# Advanced WebSocket Middlewares

This folder contains critical middlewares for security, compliance, and robustness of WebSocket connections:

- **auth_jwt.py**: JWT validation on connection, logging, error handling, GDPR compliance.
- **rate_limiter.py**: Per-user/IP rate limiting, anti-flood protection, logging, Redis recommended for production.
- **audit_logger.py**: Audit logging of all connections, disconnections, and sensitive actions, extendable to SIEM or database.

## Usage
```python
from middleware.auth_jwt import require_jwt
from middleware.rate_limiter import InMemoryRateLimiter
from middleware.audit_logger import AuditLogger

rate_limiter = InMemoryRateLimiter()
audit_logger = AuditLogger()

async def websocket_endpoint(websocket, user_id, room=None):
    payload = await require_jwt(websocket)()
    rate_limiter.check(user_id)
    audit_logger.log_event("connect", user_id, room, "Connection accepted")
    # ... handler logic ...
```

## Security & Compliance
- Never accept a WebSocket connection without a valid JWT
- Enforce rate limiting to prevent spam/flood (Redis recommended in production)
- Log all critical actions for audit and GDPR compliance
- Store the secret key in configuration, never in code

## Extensibility
- Connect `audit_logger` to PostgreSQL, MongoDB, or SIEM for centralized audit
- Replace `InMemoryRateLimiter` with a Redis-based version for scalability
- Add other middlewares (e.g., anti-abuse, i18n) as needed
