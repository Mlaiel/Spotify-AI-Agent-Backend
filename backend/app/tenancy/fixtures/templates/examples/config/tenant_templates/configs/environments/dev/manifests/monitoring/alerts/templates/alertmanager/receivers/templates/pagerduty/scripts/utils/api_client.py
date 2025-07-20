#!/usr/bin/env python3
"""
Advanced API Client for PagerDuty Integration

Client API robuste et optimisé pour les intégrations PagerDuty.
Fournit une interface complète pour interagir avec l'API PagerDuty
avec gestion avancée des erreurs, retry, rate limiting, et monitoring.

Fonctionnalités:
- Client API complet pour PagerDuty
- Gestion automatique du rate limiting
- Retry intelligent avec backoff exponentiel
- Pool de connexions optimisé
- Cache intelligent des réponses
- Monitoring et métriques
- Support de l'authentication
- Gestion des webhooks

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from urllib.parse import urljoin, urlencode
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import structlog

logger = structlog.get_logger(__name__)

class PagerDutyEventAction(Enum):
    """Actions d'événement PagerDuty"""
    TRIGGER = "trigger"
    ACKNOWLEDGE = "acknowledge"
    RESOLVE = "resolve"

class PagerDutySeverity(Enum):
    """Niveaux de sévérité PagerDuty"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class APIResponse:
    """Réponse API standardisée"""
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    request_id: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None

@dataclass
class RateLimitInfo:
    """Informations de rate limiting"""
    limit: int
    remaining: int
    reset_time: datetime
    window_seconds: int

class APIError(Exception):
    """Exception pour les erreurs API"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class RateLimitExceeded(APIError):
    """Exception pour dépassement de rate limit"""
    
    def __init__(self, reset_time: datetime):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Resets at {reset_time}")

class ConnectionPool:
    """Pool de connexions optimisé"""
    
    def __init__(
        self,
        max_connections: int = 100,
        max_connections_per_host: int = 30,
        timeout: int = 30
    ):
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.timeout = timeout
        self._session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Récupère ou crée une session"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Spotify-AI-Agent/1.0 PagerDuty-Client",
                    "Accept": "application/vnd.pagerduty+json;version=2",
                    "Content-Type": "application/json"
                }
            )
        
        return self._session
    
    async def close(self):
        """Ferme la session"""
        if self._session and not self._session.closed:
            await self._session.close()

class RetryHandler:
    """Gestionnaire de retry avec backoff exponentiel"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def should_retry(self, status_code: int, attempt: int) -> bool:
        """Détermine s'il faut retry"""
        if attempt >= self.max_retries:
            return False
        
        # Retry pour les erreurs temporaires
        return status_code in [429, 500, 502, 503, 504, 408, 409]
    
    def get_delay(self, attempt: int) -> float:
        """Calcule le délai avant retry"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Jitter de ±25%
        
        return delay

class RequestCache:
    """Cache intelligent pour les requêtes"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def _make_key(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Crée une clé de cache"""
        import hashlib
        
        key_data = f"{method}:{url}"
        if params:
            key_data += f":{json.dumps(params, sort_keys=True)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, method: str, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Récupère du cache"""
        key = self._make_key(method, url, params)
        
        if key in self.cache:
            data, expiry = self.cache[key]
            if datetime.now() < expiry:
                return data
            else:
                del self.cache[key]
        
        return None
    
    def set(
        self,
        method: str,
        url: str,
        data: Dict,
        params: Optional[Dict] = None,
        ttl: Optional[int] = None
    ):
        """Stocke en cache"""
        key = self._make_key(method, url, params)
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = (data, expiry)
    
    def clear(self):
        """Vide le cache"""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if now >= expiry
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

class PagerDutyAPIClient:
    """Client API principal pour PagerDuty"""
    
    BASE_URL = "https://api.pagerduty.com"
    EVENTS_URL = "https://events.pagerduty.com"
    
    def __init__(
        self,
        api_key: str,
        integration_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        enable_cache: bool = True,
        cache_ttl: int = 300
    ):
        self.api_key = api_key
        self.integration_key = integration_key
        
        self.connection_pool = ConnectionPool(timeout=timeout)
        self.retry_handler = RetryHandler(max_retries=max_retries)
        self.cache = RequestCache(cache_ttl) if enable_cache else None
        
        self.rate_limit_info = None
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "requests_cached": 0,
            "rate_limit_hits": 0
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Ferme les connexions"""
        await self.connection_pool.close()
    
    def _parse_rate_limit_headers(self, headers: Dict[str, str]) -> Optional[RateLimitInfo]:
        """Parse les headers de rate limiting"""
        try:
            limit = int(headers.get("X-Rate-Limit-Limit", 0))
            remaining = int(headers.get("X-Rate-Limit-Remaining", 0))
            reset_timestamp = int(headers.get("X-Rate-Limit-Reset", 0))
            
            if limit and reset_timestamp:
                reset_time = datetime.fromtimestamp(reset_timestamp, timezone.utc)
                return RateLimitInfo(
                    limit=limit,
                    remaining=remaining,
                    reset_time=reset_time,
                    window_seconds=3600  # PagerDuty utilise 1 heure
                )
        except (ValueError, TypeError):
            pass
        
        return None
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        base_url: Optional[str] = None,
        use_cache: bool = True
    ) -> APIResponse:
        """Effectue une requête API avec retry et rate limiting"""
        
        url = urljoin(base_url or self.BASE_URL, endpoint)
        request_headers = {"Authorization": f"Token token={self.api_key}"}
        
        if headers:
            request_headers.update(headers)
        
        # Vérifier le cache pour les requêtes GET
        if method == "GET" and use_cache and self.cache:
            cached_response = self.cache.get(method, url, params)
            if cached_response:
                self.metrics["requests_cached"] += 1
                return APIResponse(
                    status_code=200,
                    data=cached_response
                )
        
        # Vérifier le rate limiting
        if self.rate_limit_info and self.rate_limit_info.remaining <= 0:
            if datetime.now(timezone.utc) < self.rate_limit_info.reset_time:
                self.metrics["rate_limit_hits"] += 1
                raise RateLimitExceeded(self.rate_limit_info.reset_time)
        
        session = await self.connection_pool.get_session()
        
        for attempt in range(self.retry_handler.max_retries + 1):
            try:
                self.metrics["requests_total"] += 1
                
                # Préparer les paramètres de requête
                request_kwargs = {
                    "headers": request_headers,
                    "params": params
                }
                
                if data:
                    request_kwargs["json"] = data
                
                # Effectuer la requête
                async with session.request(method, url, **request_kwargs) as response:
                    response_headers = dict(response.headers)
                    
                    # Mettre à jour les informations de rate limiting
                    rate_limit_info = self._parse_rate_limit_headers(response_headers)
                    if rate_limit_info:
                        self.rate_limit_info = rate_limit_info
                    
                    # Lire le contenu de la réponse
                    try:
                        response_data = await response.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        response_data = {"message": await response.text()}
                    
                    # Créer la réponse API
                    api_response = APIResponse(
                        status_code=response.status,
                        data=response_data if response.status < 400 else None,
                        error=response_data.get("error", {}).get("message") if response.status >= 400 else None,
                        headers=response_headers,
                        request_id=response_headers.get("X-Request-Id"),
                        rate_limit_remaining=rate_limit_info.remaining if rate_limit_info else None,
                        rate_limit_reset=rate_limit_info.reset_time if rate_limit_info else None
                    )
                    
                    # Gestion des erreurs
                    if response.status >= 400:
                        self.metrics["requests_error"] += 1
                        
                        if response.status == 429:
                            self.metrics["rate_limit_hits"] += 1
                            if rate_limit_info:
                                raise RateLimitExceeded(rate_limit_info.reset_time)
                        
                        if not self.retry_handler.should_retry(response.status, attempt):
                            raise APIError(
                                f"API request failed: {api_response.error or 'Unknown error'}",
                                response.status,
                                response_data
                            )
                        
                        # Attendre avant retry
                        if attempt < self.retry_handler.max_retries:
                            delay = self.retry_handler.get_delay(attempt)
                            logger.warning(f"Request failed, retrying in {delay:.2f}s (attempt {attempt + 1})")
                            await asyncio.sleep(delay)
                            continue
                    
                    # Succès
                    self.metrics["requests_success"] += 1
                    
                    # Mettre en cache si approprié
                    if method == "GET" and response.status == 200 and self.cache and use_cache:
                        self.cache.set(method, url, response_data, params)
                    
                    return api_response
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.metrics["requests_error"] += 1
                
                if not self.retry_handler.should_retry(500, attempt):
                    raise APIError(f"Network error: {str(e)}")
                
                if attempt < self.retry_handler.max_retries:
                    delay = self.retry_handler.get_delay(attempt)
                    logger.warning(f"Network error, retrying in {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)
        
        raise APIError("Max retries exceeded")
    
    # Méthodes pour les événements
    async def send_event(
        self,
        action: PagerDutyEventAction,
        summary: str,
        source: str,
        severity: PagerDutySeverity,
        dedup_key: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        component: Optional[str] = None,
        group: Optional[str] = None,
        class_name: Optional[str] = None,
        custom_details: Optional[Dict[str, Any]] = None,
        images: Optional[List[Dict[str, str]]] = None,
        links: Optional[List[Dict[str, str]]] = None
    ) -> APIResponse:
        """Envoie un événement à PagerDuty"""
        
        if not self.integration_key:
            raise APIError("Integration key required for sending events")
        
        event_data = {
            "routing_key": self.integration_key,
            "event_action": action.value,
            "payload": {
                "summary": summary,
                "source": source,
                "severity": severity.value
            }
        }
        
        if dedup_key:
            event_data["dedup_key"] = dedup_key
        
        if timestamp:
            event_data["payload"]["timestamp"] = timestamp.isoformat()
        
        if component:
            event_data["payload"]["component"] = component
        
        if group:
            event_data["payload"]["group"] = group
        
        if class_name:
            event_data["payload"]["class"] = class_name
        
        if custom_details:
            event_data["payload"]["custom_details"] = custom_details
        
        if images:
            event_data["images"] = images
        
        if links:
            event_data["links"] = links
        
        return await self._make_request(
            "POST",
            "/v2/enqueue",
            data=event_data,
            base_url=self.EVENTS_URL,
            use_cache=False
        )
    
    # Méthodes pour l'API REST
    async def get_services(self, limit: int = 25, offset: int = 0, query: Optional[str] = None) -> APIResponse:
        """Récupère la liste des services"""
        params = {"limit": limit, "offset": offset}
        if query:
            params["query"] = query
        
        return await self._make_request("GET", "/services", params=params)
    
    async def get_service(self, service_id: str) -> APIResponse:
        """Récupère un service spécifique"""
        return await self._make_request("GET", f"/services/{service_id}")
    
    async def get_incidents(
        self,
        limit: int = 25,
        offset: int = 0,
        service_ids: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> APIResponse:
        """Récupère la liste des incidents"""
        params = {"limit": limit, "offset": offset}
        
        if service_ids:
            params["service_ids[]"] = service_ids
        
        if statuses:
            params["statuses[]"] = statuses
        
        if since:
            params["since"] = since.isoformat()
        
        if until:
            params["until"] = until.isoformat()
        
        return await self._make_request("GET", "/incidents", params=params)
    
    async def get_incident(self, incident_id: str) -> APIResponse:
        """Récupère un incident spécifique"""
        return await self._make_request("GET", f"/incidents/{incident_id}")
    
    async def acknowledge_incident(self, incident_id: str, from_email: str) -> APIResponse:
        """Acknowledge un incident"""
        data = {
            "incidents": [
                {
                    "id": incident_id,
                    "type": "incident",
                    "status": "acknowledged"
                }
            ]
        }
        
        headers = {"From": from_email}
        
        return await self._make_request(
            "PUT",
            f"/incidents/{incident_id}",
            data=data,
            headers=headers,
            use_cache=False
        )
    
    async def resolve_incident(self, incident_id: str, from_email: str) -> APIResponse:
        """Résout un incident"""
        data = {
            "incidents": [
                {
                    "id": incident_id,
                    "type": "incident", 
                    "status": "resolved"
                }
            ]
        }
        
        headers = {"From": from_email}
        
        return await self._make_request(
            "PUT",
            f"/incidents/{incident_id}",
            data=data,
            headers=headers,
            use_cache=False
        )
    
    async def get_escalation_policies(self, limit: int = 25, offset: int = 0) -> APIResponse:
        """Récupère les politiques d'escalade"""
        params = {"limit": limit, "offset": offset}
        return await self._make_request("GET", "/escalation_policies", params=params)
    
    async def get_users(self, limit: int = 25, offset: int = 0, query: Optional[str] = None) -> APIResponse:
        """Récupère la liste des utilisateurs"""
        params = {"limit": limit, "offset": offset}
        if query:
            params["query"] = query
        
        return await self._make_request("GET", "/users", params=params)
    
    async def get_on_calls(self, escalation_policy_ids: Optional[List[str]] = None) -> APIResponse:
        """Récupère les personnes d'astreinte"""
        params = {}
        if escalation_policy_ids:
            params["escalation_policy_ids[]"] = escalation_policy_ids
        
        return await self._make_request("GET", "/oncalls", params=params)
    
    async def create_note(self, incident_id: str, content: str, from_email: str) -> APIResponse:
        """Ajoute une note à un incident"""
        data = {
            "note": {
                "content": content
            }
        }
        
        headers = {"From": from_email}
        
        return await self._make_request(
            "POST",
            f"/incidents/{incident_id}/notes",
            data=data,
            headers=headers,
            use_cache=False
        )
    
    # Méthodes utilitaires
    async def test_connection(self) -> bool:
        """Teste la connexion à l'API"""
        try:
            response = await self._make_request("GET", "/abilities")
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_current_user(self) -> APIResponse:
        """Récupère l'utilisateur actuel"""
        return await self._make_request("GET", "/users/me")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du client"""
        return {
            **self.metrics,
            "rate_limit_info": asdict(self.rate_limit_info) if self.rate_limit_info else None,
            "cache_size": len(self.cache.cache) if self.cache else 0
        }
    
    def clear_cache(self):
        """Vide le cache"""
        if self.cache:
            self.cache.clear()

# Fonctions utilitaires
async def create_test_incident(
    client: PagerDutyAPIClient,
    summary: str = "Test incident from API client",
    auto_resolve: bool = True,
    resolve_delay: int = 30
) -> str:
    """Crée un incident de test"""
    
    # Créer l'incident
    response = await client.send_event(
        action=PagerDutyEventAction.TRIGGER,
        summary=summary,
        source="api-client-test",
        severity=PagerDutySeverity.INFO,
        dedup_key=f"test-{int(time.time())}",
        custom_details={
            "test": True,
            "created_by": "PagerDuty API Client",
            "auto_resolve": auto_resolve
        }
    )
    
    if response.status_code != 202:
        raise APIError(f"Failed to create test incident: {response.error}")
    
    dedup_key = response.data.get("dedup_key")
    
    # Résoudre automatiquement si demandé
    if auto_resolve and dedup_key:
        await asyncio.sleep(resolve_delay)
        
        await client.send_event(
            action=PagerDutyEventAction.RESOLVE,
            summary=summary,
            source="api-client-test",
            severity=PagerDutySeverity.INFO,
            dedup_key=dedup_key
        )
    
    return dedup_key

# Export des classes principales
__all__ = [
    "PagerDutyAPIClient",
    "PagerDutyEventAction",
    "PagerDutySeverity",
    "APIResponse",
    "APIError",
    "RateLimitExceeded",
    "create_test_incident"
]
