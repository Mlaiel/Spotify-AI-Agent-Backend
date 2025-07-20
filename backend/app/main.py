"""
Main Application Entrypoint
--------------------------
- Startet die Spotify AI Agent Backend-App (ASGI/WSGI)
- CLI, Dev/Prod-Start, Security, Observability, Autoren & Rollen

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

import os
import uvicorn
from backend.app.asgi import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", reload=os.getenv("DEV_MODE", "0") == "1")
