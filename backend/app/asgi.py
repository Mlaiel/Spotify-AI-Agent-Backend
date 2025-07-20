"""
ASGI Application Entry Point
---------------------------
- Startet die FastAPI-ASGI-App für Spotify AI Agent
- Integriert Security, CORS, Observability, Health, Multilingual, Sentry, OpenTelemetry

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

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN", ""))

app = FastAPI(title="Spotify AI Agent Backend", version="1.0.0", docs_url="/docs")

# Security: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Observability: Prometheus
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

# Observability: OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# API Routing
app.include_router(api_router)

# Health Endpoint
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

# Readiness Endpoint
@app.get("/ready", tags=["System"])
def ready():
    return {"status": "ready"}
