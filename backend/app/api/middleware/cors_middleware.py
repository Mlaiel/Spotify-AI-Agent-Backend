"""
CORS Middleware Ultra-Avancé - Spotify AI Agent
========================================
Gestion complète CORS avec sécurité avancée, analytics et optimisations
Auteur: Équipe Lead Dev + Architecte IA + Spécialiste Sécurité Backend

Fonctionnalités:
- CORS dynamique et contextuel
- Whitelist/blacklist intelligente
- Rate limiting par origine
- Analytics d'origine
- Optimisation preflight
- Support GeoIP
- Sécurité renforcée
"""

import json
import time
import asyncio
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse
from dataclasses import dataclass, field
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator
import geoip2.database
import geoip2.errors

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.metrics_manager import MetricsManager
from app.core.security import SecurityUtils

logger = get_logger(__name__)
metrics = MetricsManager()
security_utils = SecurityUtils()


class CORSSecurityLevel(Enum):
    """Niveaux de sécurité CORS"""
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"
    DEVELOPMENT = "development"


class OriginType(Enum):
    """Types d'origine"""
    DOMAIN = "domain"
    SUBDOMAIN = "subdomain"
    WILDCARD = "wildcard"
    REGEX = "regex"
    IP = "ip"
    LOCALHOST = "localhost"


@dataclass
class OriginConfig:
    """Configuration d'une origine"""
    pattern: str
    type: OriginType
    allowed_methods: Set[str] = field(default_factory=lambda: {"GET", "POST", "PUT", "DELETE", "OPTIONS"})
    allowed_headers: Set[str] = field(default_factory=set)
    exposed_headers: Set[str] = field(default_factory=set)
    max_age: int = 86400  # 24 heures
    allow_credentials: bool = True
    rate_limit_per_minute: int = 300
    require_auth: bool = False
    security_level: CORSSecurityLevel = CORSSecurityLevel.MODERATE
    geo_restrictions: Optional[List[str]] = None  # Codes pays ISO
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0


class CORSMetrics(BaseModel):
    """Métriques CORS"""
    origin: str
    method: str
    endpoint: str
    status_code: int
    response_time: float
    user_agent: str
    referer: Optional[str]
    geo_country: Optional[str]
    geo_city: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_preflight: bool = False
    cache_hit: bool = False


class CORSViolation(BaseModel):
    """Violation CORS détectée"""
    origin: str
    requested_method: str
    requested_headers: List[str]
    violation_type: str
    severity: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_agent: str
    ip_address: str
    geo_data: Optional[Dict] = None


class ContextPropagator:
    """Propagateur de contexte pour CORS avancé"""
    
    def __init__(self):
        """Initialize context propagator."""
        self._context_store: Dict[str, Dict] = {}
        self.logger = get_logger(__name__)
    
    async def propagate_context(self, request: Request, context_data: Dict) -> None:
        """Propager le contexte de requête pour CORS."""
        try:
            request_id = getattr(request.state, 'request_id', None)
            if request_id:
                self._context_store[request_id] = {
                    'origin': request.headers.get('origin'),
                    'user_agent': request.headers.get('user-agent'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'context': context_data
                }
        except Exception as e:
            self.logger.warning(f"Failed to propagate context: {e}")
    
    async def get_context(self, request_id: str) -> Optional[Dict]:
        """Récupérer le contexte propagé."""
        return self._context_store.get(request_id)
    
    async def cleanup_context(self, request_id: str) -> None:
        """Nettoyer le contexte après traitement."""
        if request_id in self._context_store:
            del self._context_store[request_id]


# Instance globale pour l'export
context_propagator = ContextPropagator()


class AdvancedCORSMiddleware(BaseHTTPMiddleware):
    """
    Middleware CORS ultra-avancé avec:
    - Configuration dynamique
    - Analytics et métriques
    - Sécurité renforcée
    - Optimisations performance
    - Support GeoIP
    """
    
    def __init__(
        self,
        app: ASGIApp,
        redis_client: Optional[redis.Redis] = None,
        geoip_db_path: Optional[str] = None,
        enable_analytics: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 300,
        default_security_level: CORSSecurityLevel = CORSSecurityLevel.MODERATE
    ):
        super().__init__(app)
        self.redis_client = redis_client or redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        self.enable_analytics = enable_analytics
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.default_security_level = default_security_level
        
        # Cache en mémoire pour les configurations fréquentes
        self._config_cache: Dict[str, OriginConfig] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # GeoIP pour la géolocalisation
        self.geoip_reader = None
        if geoip_db_path:
            try:
                self.geoip_reader = geoip2.database.Reader(geoip_db_path)
            except Exception as e:
                logger.warning(f"Impossible de charger la base GeoIP: {e}")
        
        # Initialisation des configurations par défaut
        asyncio.create_task(self._initialize_default_configs())
        
        # Statistiques en temps réel
        self.stats = {
            "total_requests": 0,
            "preflight_requests": 0,
            "blocked_requests": 0,
            "cache_hits": 0,
            "origins_seen": set(),
            "last_reset": datetime.utcnow()
        }

    async def _initialize_default_configs(self):
        """Initialiser les configurations par défaut"""
        default_configs = [
            OriginConfig(
                pattern="https://spotify.com",
                type=OriginType.DOMAIN,
                security_level=CORSSecurityLevel.STRICT,
                description="Spotify officiel"
            ),
            OriginConfig(
                pattern="https://*.spotify.com",
                type=OriginType.SUBDOMAIN,
                security_level=CORSSecurityLevel.STRICT,
                description="Sous-domaines Spotify"
            ),
            OriginConfig(
                pattern="http://localhost*",
                type=OriginType.LOCALHOST,
                security_level=CORSSecurityLevel.DEVELOPMENT,
                rate_limit_per_minute=1000,
                description="Développement local"
            ),
            OriginConfig(
                pattern="https://app.musicai.com",
                type=OriginType.DOMAIN,
                security_level=CORSSecurityLevel.MODERATE,
                description="Application principale"
            )
        ]
        
        for config in default_configs:
            await self._store_origin_config(config)

    async def dispatch(self, request: Request, call_next):
        """Point d'entrée principal du middleware CORS"""
        start_time = time.time()
        origin = request.headers.get("origin")
        
        # Statistiques globales
        self.stats["total_requests"] += 1
        if origin:
            self.stats["origins_seen"].add(origin)
        
        try:
            # Vérification de l'origine
            if not origin:
                # Pas d'origine = requête directe (non CORS)
                response = await call_next(request)
                await self._record_non_cors_request(request, response, start_time)
                return response
            
            # Gestion requête preflight OPTIONS
            if request.method == "OPTIONS":
                self.stats["preflight_requests"] += 1
                return await self._handle_preflight_request(request, origin, start_time)
            
            # Vérification et validation de l'origine
            origin_config = await self._get_origin_config(origin)
            if not origin_config:
                self.stats["blocked_requests"] += 1
                await self._record_cors_violation(
                    request, origin, "ORIGIN_NOT_ALLOWED", "HIGH"
                )
                return await self._create_cors_error_response(
                    "Origin not allowed", status.HTTP_403_FORBIDDEN
                )
            
            # Vérification des restrictions géographiques
            if not await self._check_geo_restrictions(request, origin_config):
                self.stats["blocked_requests"] += 1
                await self._record_cors_violation(
                    request, origin, "GEO_RESTRICTED", "MEDIUM"
                )
                return await self._create_cors_error_response(
                    "Geographic restriction", status.HTTP_403_FORBIDDEN
                )
            
            # Vérification du rate limiting
            if not await self._check_rate_limit(origin, origin_config):
                self.stats["blocked_requests"] += 1
                await self._record_cors_violation(
                    request, origin, "RATE_LIMIT_EXCEEDED", "MEDIUM"
                )
                return await self._create_cors_error_response(
                    "Rate limit exceeded", status.HTTP_429_TOO_MANY_REQUESTS
                )
            
            # Vérification de la méthode
            if request.method not in origin_config.allowed_methods:
                self.stats["blocked_requests"] += 1
                await self._record_cors_violation(
                    request, origin, "METHOD_NOT_ALLOWED", "MEDIUM"
                )
                return await self._create_cors_error_response(
                    "Method not allowed", status.HTTP_405_METHOD_NOT_ALLOWED
                )
            
            # Traitement de la requête
            response = await call_next(request)
            
            # Application des headers CORS
            self._apply_cors_headers(response, origin_config, origin)
            
            # Enregistrement des métriques
            await self._record_cors_metrics(
                request, response, origin, origin_config, start_time
            )
            
            # Mise à jour des statistiques d'utilisation
            await self._update_origin_usage(origin_config)
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur CORS middleware: {str(e)}")
            await self._record_cors_violation(
                request, origin or "unknown", "INTERNAL_ERROR", "HIGH"
            )
            # Ne pas bloquer la requête en cas d'erreur interne
            response = await call_next(request)
            return response

    async def _handle_preflight_request(
        self, request: Request, origin: str, start_time: float
    ) -> Response:
        """Gérer les requêtes preflight OPTIONS"""
        
        # Vérification de l'origine
        origin_config = await self._get_origin_config(origin)
        if not origin_config:
            await self._record_cors_violation(
                request, origin, "PREFLIGHT_ORIGIN_DENIED", "HIGH"
            )
            return await self._create_cors_error_response(
                "Origin not allowed", status.HTTP_403_FORBIDDEN
            )
        
        # Vérification de la méthode demandée
        requested_method = request.headers.get("access-control-request-method")
        if requested_method and requested_method not in origin_config.allowed_methods:
            await self._record_cors_violation(
                request, origin, "PREFLIGHT_METHOD_DENIED", "MEDIUM"
            )
            return await self._create_cors_error_response(
                "Method not allowed", status.HTTP_405_METHOD_NOT_ALLOWED
            )
        
        # Vérification des headers demandés
        requested_headers = request.headers.get("access-control-request-headers", "")
        if requested_headers and origin_config.allowed_headers:
            requested_header_list = [h.strip().lower() for h in requested_headers.split(",")]
            allowed_header_list = [h.lower() for h in origin_config.allowed_headers]
            
            for header in requested_header_list:
                if header not in allowed_header_list and not self._is_simple_header(header):
                    await self._record_cors_violation(
                        request, origin, "PREFLIGHT_HEADER_DENIED", "LOW"
                    )
                    return await self._create_cors_error_response(
                        "Header not allowed", status.HTTP_403_FORBIDDEN
                    )
        
        # Création de la réponse preflight
        response = Response(status_code=200)
        self._apply_cors_headers(response, origin_config, origin, is_preflight=True)
        
        # Enregistrement des métriques preflight
        await self._record_preflight_metrics(request, origin, origin_config, start_time)
        
        return response

    async def _get_origin_config(self, origin: str) -> Optional[OriginConfig]:
        """Récupérer la configuration pour une origine"""
        
        # Vérification du cache en mémoire
        if self.enable_caching and origin in self._config_cache:
            cache_time = self._cache_timestamps.get(origin, 0)
            if time.time() - cache_time < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return self._config_cache[origin]
        
        # Recherche dans Redis
        config = await self._find_matching_origin_config(origin)
        
        # Mise en cache
        if config and self.enable_caching:
            self._config_cache[origin] = config
            self._cache_timestamps[origin] = time.time()
        
        return config

    async def _find_matching_origin_config(self, origin: str) -> Optional[OriginConfig]:
        """Trouver la configuration correspondante à l'origine"""
        
        # Récupération de toutes les configurations
        configs_key = "cors_configs:*"
        config_keys = await self.redis_client.keys(configs_key)
        
        for config_key in config_keys:
            config_data = await self.redis_client.get(config_key)
            if not config_data:
                continue
            
            try:
                config_dict = json.loads(config_data)
                config = OriginConfig(**config_dict)
                
                if self._origin_matches_pattern(origin, config):
                    return config
            except Exception as e:
                logger.error(f"Erreur parsing config CORS: {e}")
                continue
        
        return None

    def _origin_matches_pattern(self, origin: str, config: OriginConfig) -> bool:
        """Vérifier si l'origine correspond au pattern"""
        pattern = config.pattern
        origin_type = config.type
        
        try:
            if origin_type == OriginType.DOMAIN:
                return origin.lower() == pattern.lower()
            
            elif origin_type == OriginType.SUBDOMAIN:
                # Convertir pattern wildcard en regex
                regex_pattern = pattern.replace("*", ".*")
                return bool(re.match(regex_pattern, origin, re.IGNORECASE))
            
            elif origin_type == OriginType.WILDCARD:
                # Support wildcard complet
                regex_pattern = pattern.replace("*", ".*").replace("?", ".")
                return bool(re.match(regex_pattern, origin, re.IGNORECASE))
            
            elif origin_type == OriginType.REGEX:
                return bool(re.match(pattern, origin, re.IGNORECASE))
            
            elif origin_type == OriginType.IP:
                parsed_origin = urlparse(origin)
                return parsed_origin.hostname == pattern
            
            elif origin_type == OriginType.LOCALHOST:
                parsed_origin = urlparse(origin)
                hostname = parsed_origin.hostname
                return (
                    hostname in ["localhost", "127.0.0.1", "::1"] or
                    hostname.startswith("192.168.") or
                    hostname.startswith("10.") or
                    (hostname.startswith("172.") and 
                     16 <= int(hostname.split(".")[1]) <= 31)
                )
            
        except Exception as e:
            logger.error(f"Erreur matching origine {origin} avec pattern {pattern}: {e}")
            return False
        
        return False

    async def _check_geo_restrictions(
        self, request: Request, config: OriginConfig
    ) -> bool:
        """Vérifier les restrictions géographiques"""
        
        if not config.geo_restrictions or not self.geoip_reader:
            return True
        
        client_ip = self._get_client_ip(request)
        if not client_ip or client_ip == "unknown":
            return True
        
        try:
            response = self.geoip_reader.country(client_ip)
            country_code = response.country.iso_code
            
            # Vérifier si le pays est autorisé
            return country_code in config.geo_restrictions
            
        except (geoip2.errors.AddressNotFoundError, ValueError):
            # En cas d'erreur, autoriser par défaut
            return True
        except Exception as e:
            logger.error(f"Erreur GeoIP: {e}")
            return True

    async def _check_rate_limit(self, origin: str, config: OriginConfig) -> bool:
        """Vérifier le rate limiting par origine"""
        
        current_minute = int(time.time() // 60)
        rate_key = f"cors_rate_limit:{origin}:{current_minute}"
        
        try:
            current_count = await self.redis_client.get(rate_key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= config.rate_limit_per_minute:
                return False
            
            # Incrémenter le compteur
            await self.redis_client.incr(rate_key)
            await self.redis_client.expire(rate_key, 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur rate limiting: {e}")
            return True

    def _apply_cors_headers(
        self,
        response: Response,
        config: OriginConfig,
        origin: str,
        is_preflight: bool = False
    ):
        """Appliquer les headers CORS à la réponse"""
        
        # Header d'origine
        response.headers["Access-Control-Allow-Origin"] = origin
        
        # Credentials
        if config.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Headers pour preflight
        if is_preflight:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(config.allowed_methods)
            
            if config.allowed_headers:
                response.headers["Access-Control-Allow-Headers"] = ", ".join(config.allowed_headers)
            
            response.headers["Access-Control-Max-Age"] = str(config.max_age)
        
        # Headers exposés
        if config.exposed_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(config.exposed_headers)
        
        # Headers de sécurité supplémentaires selon le niveau
        if config.security_level == CORSSecurityLevel.STRICT:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"

    async def _record_cors_metrics(
        self,
        request: Request,
        response: Response,
        origin: str,
        config: OriginConfig,
        start_time: float
    ):
        """Enregistrer les métriques CORS"""
        
        if not self.enable_analytics:
            return
        
        response_time = time.time() - start_time
        geo_data = await self._get_geo_data(request)
        
        metrics_data = CORSMetrics(
            origin=origin,
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            response_time=response_time,
            user_agent=request.headers.get("user-agent", ""),
            referer=request.headers.get("referer"),
            geo_country=geo_data.get("country") if geo_data else None,
            geo_city=geo_data.get("city") if geo_data else None,
            is_preflight=False,
            cache_hit=origin in self._config_cache
        )
        
        # Enregistrement dans Redis
        metrics_key = f"cors_metrics:{datetime.utcnow().strftime('%Y%m%d%H')}"
        await self.redis_client.lpush(
            metrics_key,
            metrics_data.json()
        )
        await self.redis_client.expire(metrics_key, timedelta(days=7))
        
        # Métriques Prometheus
        metrics.get_or_create_counter(
            "cors_requests_total",
            "Total CORS requests"
        ).labels(
            origin=origin,
            method=request.method,
            status_code=response.status_code
        ).inc()
        
        metrics.get_or_create_histogram(
            "cors_request_duration_seconds",
            "CORS request duration"
        ).labels(
            origin=origin,
            method=request.method
        ).observe(response_time)

    async def _record_preflight_metrics(
        self,
        request: Request,
        origin: str,
        config: OriginConfig,
        start_time: float
    ):
        """Enregistrer les métriques preflight"""
        
        response_time = time.time() - start_time
        
        metrics.get_or_create_counter(
            "cors_preflight_requests_total",
            "Total CORS preflight requests"
        ).labels(origin=origin).inc()
        
        metrics.get_or_create_histogram(
            "cors_preflight_duration_seconds",
            "CORS preflight duration"
        ).labels(origin=origin).observe(response_time)

    async def _record_cors_violation(
        self,
        request: Request,
        origin: str,
        violation_type: str,
        severity: str
    ):
        """Enregistrer une violation CORS"""
        
        geo_data = await self._get_geo_data(request)
        
        violation = CORSViolation(
            origin=origin,
            requested_method=request.method,
            requested_headers=list(request.headers.keys()),
            violation_type=violation_type,
            severity=severity,
            user_agent=request.headers.get("user-agent", ""),
            ip_address=self._get_client_ip(request),
            geo_data=geo_data
        )
        
        # Enregistrement dans Redis
        violations_key = f"cors_violations:{datetime.utcnow().strftime('%Y%m%d')}"
        await self.redis_client.lpush(
            violations_key,
            violation.json()
        )
        await self.redis_client.expire(violations_key, timedelta(days=30))
        
        # Log selon la sévérité
        if severity == "HIGH":
            logger.error(f"CORS violation HIGH: {violation_type} from {origin}")
        elif severity == "MEDIUM":
            logger.warning(f"CORS violation MEDIUM: {violation_type} from {origin}")
        else:
            logger.info(f"CORS violation LOW: {violation_type} from {origin}")
        
        # Métriques Prometheus
        metrics.get_or_create_counter(
            "cors_violations_total",
            "Total CORS violations"
        ).labels(
            origin=origin,
            violation_type=violation_type,
            severity=severity
        ).inc()

    async def _get_geo_data(self, request: Request) -> Optional[Dict]:
        """Obtenir les données géographiques du client"""
        
        if not self.geoip_reader:
            return None
        
        client_ip = self._get_client_ip(request)
        if not client_ip or client_ip == "unknown":
            return None
        
        try:
            response = self.geoip_reader.city(client_ip)
            return {
                "country": response.country.iso_code,
                "country_name": response.country.name,
                "city": response.city.name,
                "latitude": float(response.location.latitude) if response.location.latitude else None,
                "longitude": float(response.location.longitude) if response.location.longitude else None
            }
        except Exception:
            return None

    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'IP réelle du client"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

    def _is_simple_header(self, header: str) -> bool:
        """Vérifier si c'est un header simple CORS"""
        simple_headers = {
            "accept", "accept-language", "content-language", "content-type"
        }
        return header.lower() in simple_headers

    async def _create_cors_error_response(self, message: str, status_code: int) -> Response:
        """Créer une réponse d'erreur CORS"""
        return Response(
            content=json.dumps({"error": message, "code": "CORS_ERROR"}),
            status_code=status_code,
            media_type="application/json",
            headers={
                "X-CORS-Error": message,
                "Access-Control-Allow-Origin": "*",  # Pour afficher l'erreur
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

    async def _store_origin_config(self, config: OriginConfig):
        """Stocker une configuration d'origine"""
        config_key = f"cors_configs:{config.pattern}"
        config_data = {
            "pattern": config.pattern,
            "type": config.type.value,
            "allowed_methods": list(config.allowed_methods),
            "allowed_headers": list(config.allowed_headers),
            "exposed_headers": list(config.exposed_headers),
            "max_age": config.max_age,
            "allow_credentials": config.allow_credentials,
            "rate_limit_per_minute": config.rate_limit_per_minute,
            "require_auth": config.require_auth,
            "security_level": config.security_level.value,
            "geo_restrictions": config.geo_restrictions,
            "description": config.description,
            "created_at": config.created_at.isoformat(),
            "last_used": config.last_used.isoformat() if config.last_used else None,
            "usage_count": config.usage_count
        }
        
        await self.redis_client.set(
            config_key,
            json.dumps(config_data)
        )

    async def _update_origin_usage(self, config: OriginConfig):
        """Mettre à jour les statistiques d'utilisation"""
        config.usage_count += 1
        config.last_used = datetime.utcnow()
        await self._store_origin_config(config)

    async def _record_non_cors_request(
        self, request: Request, response: Response, start_time: float
    ):
        """Enregistrer une requête non-CORS"""
        response_time = time.time() - start_time
        
        metrics.get_or_create_counter(
            "non_cors_requests_total",
            "Total non-CORS requests"
        ).labels(
            method=request.method,
            status_code=response.status_code
        ).inc()

    async def get_cors_statistics(self) -> Dict:
        """Récupérer les statistiques CORS"""
        return {
            **self.stats,
            "origins_seen": list(self.stats["origins_seen"]),
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(self.stats["total_requests"], 1) * 100
            ),
            "block_rate": (
                self.stats["blocked_requests"] / max(self.stats["total_requests"], 1) * 100
            )
        }

    async def add_origin_config(
        self,
        pattern: str,
        origin_type: OriginType,
        allowed_methods: Set[str] = None,
        allowed_headers: Set[str] = None,
        **kwargs
    ) -> OriginConfig:
        """Ajouter une nouvelle configuration d'origine"""
        
        config = OriginConfig(
            pattern=pattern,
            type=origin_type,
            allowed_methods=allowed_methods or {"GET", "POST", "PUT", "DELETE", "OPTIONS"},
            allowed_headers=allowed_headers or set(),
            **kwargs
        )
        
        await self._store_origin_config(config)
        
        # Invalider le cache
        if pattern in self._config_cache:
            del self._config_cache[pattern]
            del self._cache_timestamps[pattern]
        
        logger.info(f"Configuration CORS ajoutée pour: {pattern}")
        return config

    async def remove_origin_config(self, pattern: str) -> bool:
        """Supprimer une configuration d'origine"""
        config_key = f"cors_configs:{pattern}"
        
        result = await self.redis_client.delete(config_key)
        
        # Invalider le cache
        if pattern in self._config_cache:
            del self._config_cache[pattern]
            del self._cache_timestamps[pattern]
        
        if result:
            logger.info(f"Configuration CORS supprimée pour: {pattern}")
        
        return bool(result)


class CORSConfigManager:
    """Gestionnaire de configuration CORS"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

    async def bulk_import_configs(self, configs: List[Dict]) -> int:
        """Importer en masse des configurations"""
        imported = 0
        
        for config_data in configs:
            try:
                config = OriginConfig(**config_data)
                config_key = f"cors_configs:{config.pattern}"
                
                await self.redis_client.set(
                    config_key,
                    json.dumps(config_data)
                )
                imported += 1
                
            except Exception as e:
                logger.error(f"Erreur import config CORS: {e}")
                continue
        
        logger.info(f"Importé {imported} configurations CORS")
        return imported

    async def export_configs(self) -> List[Dict]:
        """Exporter toutes les configurations"""
        configs = []
        config_keys = await self.redis_client.keys("cors_configs:*")
        
        for key in config_keys:
            config_data = await self.redis_client.get(key)
            if config_data:
                configs.append(json.loads(config_data))
        
        return configs

    async def backup_configs(self, backup_name: str) -> bool:
        """Sauvegarder les configurations"""
        try:
            configs = await self.export_configs()
            backup_key = f"cors_backup:{backup_name}:{datetime.utcnow().isoformat()}"
            
            await self.redis_client.set(
                backup_key,
                json.dumps(configs)
            )
            
            # Garder seulement les 10 dernières sauvegardes
            backup_keys = await self.redis_client.keys(f"cors_backup:{backup_name}:*")
            if len(backup_keys) > 10:
                oldest_keys = sorted(backup_keys)[:-10]
                await self.redis_client.delete(*oldest_keys)
            
            logger.info(f"Sauvegarde CORS créée: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde CORS: {e}")
            return False

    async def restore_configs(self, backup_name: str) -> bool:
        """Restaurer des configurations depuis une sauvegarde"""
        try:
            backup_keys = await self.redis_client.keys(f"cors_backup:{backup_name}:*")
            if not backup_keys:
                return False
            
            # Prendre la sauvegarde la plus récente
            latest_backup = sorted(backup_keys)[-1]
            backup_data = await self.redis_client.get(latest_backup)
            
            if backup_data:
                configs = json.loads(backup_data)
                return await self.bulk_import_configs(configs) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur restauration CORS: {e}")
            return False


# Factory functions pour différents environnements
def create_development_cors_middleware(app: ASGIApp) -> AdvancedCORSMiddleware:
    """Créer middleware CORS pour développement"""
    return AdvancedCORSMiddleware(
        app=app,
        default_security_level=CORSSecurityLevel.DEVELOPMENT,
        enable_analytics=True,
        enable_caching=False,
        cache_ttl=60
    )


def create_production_cors_middleware(
    app: ASGIApp,
    geoip_db_path: str = None
) -> AdvancedCORSMiddleware:
    """Créer middleware CORS pour production"""
    return AdvancedCORSMiddleware(
        app=app,
        geoip_db_path=geoip_db_path,
        default_security_level=CORSSecurityLevel.STRICT,
        enable_analytics=True,
        enable_caching=True,
        cache_ttl=300
    )


def create_testing_cors_middleware(app: ASGIApp) -> AdvancedCORSMiddleware:
    """Créer middleware CORS pour tests"""
    return AdvancedCORSMiddleware(
        app=app,
        default_security_level=CORSSecurityLevel.RELAXED,
        enable_analytics=False,
        enable_caching=False
    )
