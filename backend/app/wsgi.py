"""
WSGI Application Entry Point
---------------------------
- Startet die WSGI-App für Spotify AI Agent (z.B. für Gunicorn, uWSGI)
- Integriert Security, Observability, Health, Sentry

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

import os
import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette_exporter import PrometheusMiddleware, handle_metrics
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from backend.app.api import router as api_router
from mangum import Mangum

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN", ""))

app = FastAPI(title="Spotify AI Agent Backend", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)
FastAPIInstrumentor.instrument_app(app)
app.include_router(api_router)

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

@app.get("/ready", tags=["System"])
def ready():
    return {"status": "ready"}

# WSGI-Handler für Gunicorn, uWSGI, AWS Lambda (via Mangum)
handler = Mangum(app)
