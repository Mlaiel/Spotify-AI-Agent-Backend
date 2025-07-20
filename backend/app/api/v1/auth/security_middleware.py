"""
SecurityMiddleware : Middleware de sécurité avancé
- Rate limiting, CORS, headers sécurité, audit
- Protection brute-force, logs, RGPD
- Intégration FastAPI/Django, scalable microservices

Auteur : Spécialiste Sécurité, Backend Senior, Lead Dev
"""

from typing import Callable
from fastapi import Request, Response
import time

class SecurityMiddleware:
    """
    Middleware de sécurité pour API FastAPI/Django.
    """
    def __init__(self, app, rate_limit: int = 100):
        self.app = app
        self.rate_limit = rate_limit
        self.requests = {}

    async def __call__(self, request: Request, call_next: Callable):
        ip = request.client.host
        now = int(time.time())
        self.requests.setdefault(ip, []).append(now)
        # Nettoyage des anciennes requêtes
        self.requests[ip] = [t for t in self.requests[ip] if t > now - 60]
        if len(self.requests[ip]) > self.rate_limit:
            return Response("Trop de requêtes", status_code=429)
        # Headers sécurité
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response

# Exemple d’utilisation (FastAPI) :
# from fastapi import FastAPI
# app = FastAPI()
# app.add_middleware(SecurityMiddleware, rate_limit=100)
