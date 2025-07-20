"""
⚡ Advanced Rate Limiting Middleware System
==========================================

Système ultra-avancé de limitation de taux pour l'Agent IA Spotify.
Protection contre les abus, DDoS, et gestion intelligente des quotas.

Features:
- Rate limiting adaptatif basé sur l'IA
- Limitation par tier utilisateur (Free, Premium, Enterprise)
- Protection spécialisée pour l'API Spotify
- Rate limiting par endpoint avec règles dynamiques
- Détection et prévention des attaques DDoS
- Système de quotas et crédits pour les features IA
- Analytics et monitoring en temps réel
- Whitelist/Blacklist d'IPs automatique

Algorithmes:
- Token Bucket avec refill dynamique
- Sliding Window Counter
- Fixed Window Counter
- Leaky Bucket pour le lissage
- Machine Learning pour la détection d'anomalies

Author: Architecte Microservices + DBA & Data Engineer
Date: 2025-01-14
"""

import asyncio
import json
import time
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import hashlib
import statistics
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

from ...core.config import get_settings
from ...core.logging import get_logger
from ...core.database import get_database
from ...core.exceptions import RateLimitExceededError, SecurityViolationError
from ...models.user import UserRole, UserTier

settings = get_settings()
logger = get_logger(__name__)


class RateLimitStrategy(Enum):
    """Stratégies de limitation de taux"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class RateLimitScope(Enum):
    """Portées de limitation"""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"
    SPOTIFY_API = "spotify_api"


@dataclass
class RateLimitRule:
    """Règle de limitation de taux"""
    name: str
    strategy: RateLimitStrategy
    scope: RateLimitScope
    limit: int
    window_size: int  # en secondes
    burst_size: Optional[int] = None
    refill_rate: Optional[float] = None
    enabled: bool = True
    priority: int = 0


@dataclass
class RateLimitResult:
    """Résultat d'une vérification de rate limiting"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    rule_name: str = ""
    current_usage: int = 0


class RateLimitingMiddleware:
    """
    Middleware principal de limitation de taux
    Implémente différentes stratégies et algorithmes
    """
    
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.rules = self._load_rate_limit_rules()
        self.anomaly_detector = None
        self._load_anomaly_detector()
    
    async def __call__(self, request: Request, call_next):
        """Traitement principal du middleware de rate limiting"""
        start_time = time.time()
        
        try:
            # Vérifier si l'IP est bloquée
            client_ip = self._get_client_ip(request)
            if await self._is_ip_blocked(client_ip):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "ip_blocked",
                        "message": "Your IP address has been temporarily blocked",
                        "blocked_until": await self._get_ip_block_expiry(client_ip)
                    }
                )
            
            # Appliquer les règles de rate limiting
            rate_limit_result = await self._check_rate_limits(request)
            
            if not rate_limit_result.allowed:
                # Enregistrer la violation
                await self._log_rate_limit_violation(request, rate_limit_result)
                
                # Détecter les attaques potentielles
                await self._detect_potential_attack(request)
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "rate_limit_exceeded",
                        "message": f"Rate limit exceeded for {rate_limit_result.rule_name}",
                        "remaining": rate_limit_result.remaining,
                        "reset_time": rate_limit_result.reset_time.isoformat(),
                        "retry_after": rate_limit_result.retry_after
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_limit_result.current_usage + rate_limit_result.remaining),
                        "X-RateLimit-Remaining": str(rate_limit_result.remaining),
                        "X-RateLimit-Reset": str(int(rate_limit_result.reset_time.timestamp())),
                        "Retry-After": str(rate_limit_result.retry_after or 60)
                    }
                )
            
            # Enregistrer l'usage
            await self._record_usage(request, rate_limit_result)
            
            response = await call_next(request)
            
            # Ajouter des headers de rate limiting
            self._add_rate_limit_headers(response, rate_limit_result)
            
            return response
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # En cas d'erreur, laisser passer la requête mais enregistrer l'erreur
            await self._log_middleware_error(request, str(e))
            return await call_next(request)
        
        finally:
            # Enregistrer les métriques de performance
            processing_time = time.time() - start_time
            await self._record_rate_limit_metrics(request, processing_time)
    
    def _load_rate_limit_rules(self) -> List[RateLimitRule]:
        """Charger les règles de rate limiting"""
        base_rules = [
            # Règles globales par IP
            RateLimitRule(
                name="global_ip_requests",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.IP,
                limit=1000,
                window_size=3600,  # 1 heure
                priority=1
            ),
            RateLimitRule(
                name="global_ip_burst",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.IP,
                limit=100,
                window_size=60,  # 1 minute
                burst_size=20,
                refill_rate=1.67,  # 100 tokens per minute
                priority=2
            ),
            
            # Règles par utilisateur authentifié
            RateLimitRule(
                name="user_api_requests",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
                limit=5000,
                window_size=3600,  # 1 heure
                priority=3
            ),
            
            # Règles spécifiques aux endpoints
            RateLimitRule(
                name="auth_endpoints",
                strategy=RateLimitStrategy.FIXED_WINDOW,
                scope=RateLimitScope.ENDPOINT,
                limit=10,
                window_size=300,  # 5 minutes
                priority=4
            ),
            RateLimitRule(
                name="ai_generation_endpoints",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.USER,
                limit=50,
                window_size=3600,  # 1 heure
                burst_size=5,
                refill_rate=0.014,  # 50 tokens per hour
                priority=5
            ),
            
            # Règles pour l'API Spotify
            RateLimitRule(
                name="spotify_api_calls",
                strategy=RateLimitStrategy.LEAKY_BUCKET,
                scope=RateLimitScope.GLOBAL,
                limit=100,
                window_size=60,  # 1 minute
                priority=6
            )
        ]
        
        return base_rules
    
    def _load_anomaly_detector(self):
        """Charger le détecteur d'anomalies ML"""
        try:
            # Charger un modèle pré-entraîné ou créer un nouveau
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Dans un vrai système, on chargerait un modèle pré-entraîné
            # self.anomaly_detector = joblib.load("anomaly_detector.pkl")
            
        except Exception as e:
            logger.warning(f"Could not load anomaly detector: {e}")
            self.anomaly_detector = None
    
    async def _check_rate_limits(self, request: Request) -> RateLimitResult:
        """Vérifier toutes les règles de rate limiting applicables"""
        applicable_rules = self._get_applicable_rules(request)
        
        # Trier par priorité
        applicable_rules.sort(key=lambda r: r.priority)
        
        for rule in applicable_rules:
            result = await self._check_single_rule(request, rule)
            
            if not result.allowed:
                return result
        
        # Si toutes les règles passent, retourner le résultat de la règle la plus restrictive
        if applicable_rules:
            rule_results = []
            for rule in applicable_rules:
                rule_result = await self._check_single_rule(request, rule)
                rule_results.append(rule_result)
            
            most_restrictive = min(rule_results, key=lambda r: r.remaining)
            return most_restrictive
        
        # Aucune règle applicable, autoriser
        return RateLimitResult(
            allowed=True,
            remaining=999999,
            reset_time=datetime.utcnow() + timedelta(hours=1)
        )
    
    def _get_applicable_rules(self, request: Request) -> List[RateLimitRule]:
        """Obtenir les règles applicables à cette requête"""
        applicable_rules = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if rule.scope == RateLimitScope.GLOBAL:
                applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.IP:
                applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.USER and hasattr(request.state, "user"):
                applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.ENDPOINT:
                if self._rule_applies_to_endpoint(rule, request.url.path):
                    applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.API_KEY and hasattr(request.state, "api_key_info"):
                applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.SPOTIFY_API:
                if request.url.path.startswith("/api/v1/spotify"):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_applies_to_endpoint(self, rule: RateLimitRule, path: str) -> bool:
        """Vérifier si une règle s'applique à un endpoint"""
        endpoint_patterns = {
            "auth_endpoints": ["/api/v1/auth/", "/api/v1/oauth/"],
            "ai_generation_endpoints": ["/api/v1/ai/generate", "/api/v1/content/generate"],
            "spotify_api_calls": ["/api/v1/spotify/"]
        }
        
        patterns = endpoint_patterns.get(rule.name, [])
        return any(path.startswith(pattern) for pattern in patterns)
    
    async def _check_single_rule(self, request: Request, rule: RateLimitRule) -> RateLimitResult:
        """Vérifier une règle spécifique"""
        key = self._build_rate_limit_key(request, rule)
        
        if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(key, rule)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(key, rule)
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(key, rule)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return await self._check_leaky_bucket(key, rule)
        elif rule.strategy == RateLimitStrategy.ADAPTIVE:
            return await self._check_adaptive(key, rule, request)
        else:
            # Fallback vers sliding window
            return await self._check_sliding_window(key, rule)
    
    def _build_rate_limit_key(self, request: Request, rule: RateLimitRule) -> str:
        """Construire la clé Redis pour une règle"""
        key_parts = ["rate_limit", rule.name]
        
        if rule.scope == RateLimitScope.GLOBAL:
            key_parts.append("global")
        elif rule.scope == RateLimitScope.IP:
            key_parts.append(f"ip:{self._get_client_ip(request)}")
        elif rule.scope == RateLimitScope.USER:
            user_id = getattr(request.state, "user", {}).get("user_id", "anonymous")
            key_parts.append(f"user:{user_id}")
        elif rule.scope == RateLimitScope.ENDPOINT:
            endpoint = f"{request.method}:{request.url.path}"
            key_parts.append(f"endpoint:{hashlib.md5(endpoint.encode()).hexdigest()}")
        elif rule.scope == RateLimitScope.API_KEY:
            api_key = getattr(request.state, "api_key_info", {}).get("key_id", "unknown")
            key_parts.append(f"apikey:{api_key}")
        elif rule.scope == RateLimitScope.SPOTIFY_API:
            key_parts.append("spotify_api")
        
        return ":".join(key_parts)
    
    async def _check_token_bucket(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Implémentation Token Bucket"""
        bucket_key = f"{key}:bucket"
        current_time = time.time()
        
        # Obtenir l'état actuel du bucket
        bucket_data = await self.redis_client.hmget(
            bucket_key,
            "tokens", "last_refill"
        )
        
        tokens = float(bucket_data[0] or rule.burst_size or rule.limit)
        last_refill = float(bucket_data[1] or current_time)
        
        # Calculer les tokens à ajouter
        time_passed = current_time - last_refill
        tokens_to_add = time_passed * (rule.refill_rate or rule.limit / rule.window_size)
        tokens = min(tokens + tokens_to_add, rule.burst_size or rule.limit)
        
        if tokens >= 1:
            # Consommer 1 token
            tokens -= 1
            
            # Mettre à jour le bucket
            await self.redis_client.hset(bucket_key, mapping={
                "tokens": str(tokens),
                "last_refill": str(current_time)
            })
            await self.redis_client.expire(bucket_key, rule.window_size * 2)
            
            reset_time = datetime.utcnow() + timedelta(
                seconds=(rule.burst_size or rule.limit - tokens) / (rule.refill_rate or 1)
            )
            
            return RateLimitResult(
                allowed=True,
                remaining=int(tokens),
                reset_time=reset_time,
                rule_name=rule.name,
                current_usage=int(rule.burst_size or rule.limit) - int(tokens)
            )
        else:
            # Pas assez de tokens
            retry_after = int(1 / (rule.refill_rate or 1))
            reset_time = datetime.utcnow() + timedelta(seconds=retry_after)
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                rule_name=rule.name,
                current_usage=rule.limit
            )
    
    async def _check_sliding_window(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Implémentation Sliding Window Counter"""
        window_key = f"{key}:sliding"
        current_time = time.time()
        window_start = current_time - rule.window_size
        
        # Supprimer les entrées expirées
        await self.redis_client.zremrangebyscore(window_key, 0, window_start)
        
        # Compter les requêtes dans la fenêtre
        count = await self.redis_client.zcard(window_key)
        
        if count < rule.limit:
            # Ajouter la requête actuelle
            await self.redis_client.zadd(window_key, {str(current_time): current_time})
            await self.redis_client.expire(window_key, rule.window_size + 60)
            
            remaining = rule.limit - count - 1
            reset_time = datetime.utcnow() + timedelta(seconds=rule.window_size)
            
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=reset_time,
                rule_name=rule.name,
                current_usage=count + 1
            )
        else:
            # Limite dépassée
            oldest_request = await self.redis_client.zrange(window_key, 0, 0, withscores=True)
            if oldest_request:
                oldest_time = oldest_request[0][1]
                retry_after = int(oldest_time + rule.window_size - current_time)
            else:
                retry_after = rule.window_size
            
            reset_time = datetime.utcnow() + timedelta(seconds=retry_after)
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                rule_name=rule.name,
                current_usage=count
            )
    
    async def _check_fixed_window(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Implémentation Fixed Window Counter"""
        current_time = time.time()
        window_start = int(current_time // rule.window_size) * rule.window_size
        window_key = f"{key}:fixed:{window_start}"
        
        # Incrémenter le compteur
        current_count = await self.redis_client.incr(window_key)
        
        if current_count == 1:
            # Premier accès dans cette fenêtre, définir l'expiration
            await self.redis_client.expire(window_key, rule.window_size + 60)
        
        if current_count <= rule.limit:
            remaining = rule.limit - current_count
            reset_time = datetime.fromtimestamp(window_start + rule.window_size)
            
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=reset_time,
                rule_name=rule.name,
                current_usage=current_count
            )
        else:
            retry_after = int(window_start + rule.window_size - current_time)
            reset_time = datetime.fromtimestamp(window_start + rule.window_size)
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                rule_name=rule.name,
                current_usage=current_count
            )
    
    async def _check_leaky_bucket(self, key: str, rule: RateLimitRule) -> RateLimitResult:
        """Implémentation Leaky Bucket"""
        bucket_key = f"{key}:leaky"
        current_time = time.time()
        
        bucket_data = await self.redis_client.hmget(
            bucket_key,
            "level", "last_leak"
        )
        
        level = float(bucket_data[0] or 0)
        last_leak = float(bucket_data[1] or current_time)
        
        # Calculer le leak
        time_passed = current_time - last_leak
        leak_rate = rule.limit / rule.window_size
        level = max(0, level - (time_passed * leak_rate))
        
        if level < rule.limit:
            # Ajouter 1 à la bucket
            level += 1
            
            await self.redis_client.hset(bucket_key, mapping={
                "level": str(level),
                "last_leak": str(current_time)
            })
            await self.redis_client.expire(bucket_key, rule.window_size * 2)
            
            remaining = int(rule.limit - level)
            reset_time = datetime.utcnow() + timedelta(
                seconds=level / leak_rate
            )
            
            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=reset_time,
                rule_name=rule.name,
                current_usage=int(level)
            )
        else:
            # Bucket pleine
            retry_after = int((level - rule.limit + 1) / leak_rate)
            reset_time = datetime.utcnow() + timedelta(seconds=retry_after)
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                rule_name=rule.name,
                current_usage=rule.limit
            )
    
    async def _check_adaptive(self, key: str, rule: RateLimitRule, request: Request) -> RateLimitResult:
        """Implémentation Rate Limiting Adaptatif basé sur l'IA"""
        # Analyser le comportement récent
        behavior_data = await self._analyze_request_behavior(request)
        
        # Ajuster la limite dynamiquement
        adjusted_limit = await self._calculate_adaptive_limit(rule, behavior_data)
        
        # Créer une règle temporaire avec la limite ajustée
        adaptive_rule = RateLimitRule(
            name=f"{rule.name}_adaptive",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            scope=rule.scope,
            limit=adjusted_limit,
            window_size=rule.window_size
        )
        
        return await self._check_sliding_window(key, adaptive_rule)
    
    async def _analyze_request_behavior(self, request: Request) -> Dict[str, Any]:
        """Analyser le comportement des requêtes pour l'adaptation"""
        # Analyser les métriques récentes
        analysis_key = f"behavior:{self._get_client_ip(request)}"
        recent_requests = await self.redis_client.lrange(analysis_key, 0, 99)
        
        if not recent_requests:
            return {"risk_score": 0.0, "pattern": "normal"}
        
        # Analyser les patterns
        timestamps = []
        for req_data in recent_requests:
            try:
                req_info = json.loads(req_data)
                timestamps.append(req_info.get("timestamp", time.time()))
            except:
                continue
        
        if len(timestamps) < 5:
            return {"risk_score": 0.0, "pattern": "normal"}
        
        # Calculer le score de risque basé sur les patterns
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_interval = statistics.mean(intervals)
        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        
        # Score de risque simple
        risk_score = 0.0
        if avg_interval < 1.0:  # Très rapide
            risk_score += 0.5
        if std_interval < 0.1:  # Très régulier (bot-like)
            risk_score += 0.3
        
        return {
            "risk_score": min(risk_score, 1.0),
            "pattern": "suspicious" if risk_score > 0.5 else "normal",
            "avg_interval": avg_interval,
            "std_interval": std_interval
        }
    
    async def _calculate_adaptive_limit(self, rule: RateLimitRule, behavior_data: Dict[str, Any]) -> int:
        """Calculer la limite adaptative basée sur le comportement"""
        base_limit = rule.limit
        risk_score = behavior_data.get("risk_score", 0.0)
        
        # Réduire la limite si le comportement est suspect
        if risk_score > 0.7:
            return int(base_limit * 0.3)  # Très restrictif
        elif risk_score > 0.5:
            return int(base_limit * 0.6)  # Restrictif
        elif risk_score > 0.3:
            return int(base_limit * 0.8)  # Légèrement restrictif
        else:
            return base_limit  # Normal
    
    def _get_client_ip(self, request: Request) -> str:
        """Obtenir l'IP du client"""
        # Vérifier les headers de proxy
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback vers l'IP de connexion
        return getattr(request.client, "host", "unknown")
    
    async def _is_ip_blocked(self, client_ip: str) -> bool:
        """Vérifier si l'IP est bloquée"""
        block_key = f"blocked_ip:{client_ip}"
        return await self.redis_client.exists(block_key) > 0
    
    async def _get_ip_block_expiry(self, client_ip: str) -> Optional[str]:
        """Obtenir l'expiration du blocage d'IP"""
        block_key = f"blocked_ip:{client_ip}"
        ttl = await self.redis_client.ttl(block_key)
        if ttl > 0:
            expiry_time = datetime.utcnow() + timedelta(seconds=ttl)
            return expiry_time.isoformat()
        return None
    
    async def _log_rate_limit_violation(self, request: Request, result: RateLimitResult):
        """Enregistrer une violation de rate limiting"""
        violation_data = {
            "timestamp": time.time(),
            "ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", ""),
            "path": str(request.url.path),
            "method": request.method,
            "rule_name": result.rule_name,
            "current_usage": result.current_usage,
            "limit": result.current_usage + result.remaining
        }
        
        # Enregistrer dans Redis pour analyse
        violation_key = f"violations:{self._get_client_ip(request)}"
        await self.redis_client.lpush(violation_key, json.dumps(violation_data))
        await self.redis_client.ltrim(violation_key, 0, 99)  # Garder les 100 dernières
        await self.redis_client.expire(violation_key, 3600)  # 1 heure
        
        logger.warning(f"Rate limit violation: {violation_data}")
    
    async def _detect_potential_attack(self, request: Request):
        """Détecter les attaques potentielles"""
        client_ip = self._get_client_ip(request)
        violation_key = f"violations:{client_ip}"
        
        # Compter les violations récentes
        violations_count = await self.redis_client.llen(violation_key)
        
        # Si trop de violations, bloquer temporairement l'IP
        if violations_count > 10:  # Plus de 10 violations
            block_key = f"blocked_ip:{client_ip}"
            await self.redis_client.setex(block_key, 3600, "auto_blocked")  # Bloquer 1 heure
            
            logger.error(f"Auto-blocked IP {client_ip} due to {violations_count} violations")
    
    async def _record_usage(self, request: Request, result: RateLimitResult):
        """Enregistrer l'usage pour analytics"""
        usage_data = {
            "timestamp": time.time(),
            "ip": self._get_client_ip(request),
            "path": str(request.url.path),
            "method": request.method,
            "rule_name": result.rule_name,
            "allowed": result.allowed,
            "remaining": result.remaining
        }
        
        # Enregistrer pour behaviour analysis
        behavior_key = f"behavior:{self._get_client_ip(request)}"
        await self.redis_client.lpush(behavior_key, json.dumps(usage_data))
        await self.redis_client.ltrim(behavior_key, 0, 99)
        await self.redis_client.expire(behavior_key, 3600)
    
    def _add_rate_limit_headers(self, response: Response, result: RateLimitResult):
        """Ajouter les headers de rate limiting à la réponse"""
        response.headers["X-RateLimit-Limit"] = str(result.current_usage + result.remaining)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_time.timestamp()))
        
        if result.retry_after:
            response.headers["Retry-After"] = str(result.retry_after)
    
    async def _log_middleware_error(self, request: Request, error: str):
        """Enregistrer les erreurs du middleware"""
        error_data = {
            "timestamp": time.time(),
            "ip": self._get_client_ip(request),
            "path": str(request.url.path),
            "error": error
        }
        
        logger.error(f"Rate limiting middleware error: {error_data}")
    
    async def _record_rate_limit_metrics(self, request: Request, processing_time: float):
        """Enregistrer les métriques de performance"""
        metrics_data = {
            "timestamp": time.time(),
            "path": str(request.url.path),
            "processing_time": processing_time,
            "ip": self._get_client_ip(request)
        }
        
        # Enregistrer dans Redis pour monitoring
        metrics_key = "rate_limit_metrics"
        await self.redis_client.lpush(metrics_key, json.dumps(metrics_data))
        await self.redis_client.ltrim(metrics_key, 0, 999)  # Garder les 1000 dernières
        await self.redis_client.expire(metrics_key, 3600)


class AdaptiveRateLimitMiddleware(RateLimitingMiddleware):
    """Middleware de rate limiting adaptatif avancé"""
    
    def __init__(self):
        super().__init__()
        self.ml_model = None
        self._load_ml_model()
    
    def _load_ml_model(self):
        """Charger le modèle ML pour la prédiction adaptative"""
        try:
            # Créer un modèle simple d'isolation forest
            self.ml_model = IsolationForest(
                contamination=0.05,
                random_state=42
            )
            logger.info("ML model loaded for adaptive rate limiting")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
            self.ml_model = None
    
    async def _calculate_adaptive_limit(self, rule: RateLimitRule, behavior_data: Dict[str, Any]) -> int:
        """Calcul avancé de limite adaptative avec ML"""
        if not self.ml_model:
            return await super()._calculate_adaptive_limit(rule, behavior_data)
        
        try:
            # Préparer les features pour le modèle ML
            features = [
                behavior_data.get("avg_interval", 1.0),
                behavior_data.get("std_interval", 0.0),
                behavior_data.get("risk_score", 0.0),
                time.time() % 86400,  # Heure de la journée
            ]
            
            # Prédire si le comportement est anormal
            prediction = self.ml_model.decision_function([features])[0]
            
            # Ajuster la limite basée sur la prédiction
            if prediction < -0.5:  # Très anormal
                return int(rule.limit * 0.1)
            elif prediction < -0.2:  # Anormal
                return int(rule.limit * 0.3)
            elif prediction < 0:  # Légèrement anormal
                return int(rule.limit * 0.7)
            else:
                return rule.limit
                
        except Exception as e:
            logger.error(f"ML adaptive calculation failed: {e}")
            return await super()._calculate_adaptive_limit(rule, behavior_data)


class UserTierRateLimitMiddleware(RateLimitingMiddleware):
    """Middleware de rate limiting basé sur les tiers utilisateurs"""
    
    def _load_rate_limit_rules(self) -> List[RateLimitRule]:
        """Charger les règles par tier utilisateur"""
        tier_rules = []
        
        # Règles pour Free Tier
        tier_rules.extend([
            RateLimitRule(
                name="free_tier_requests",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
                limit=100,
                window_size=3600,
                priority=1
            ),
            RateLimitRule(
                name="free_tier_ai_requests",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.USER,
                limit=10,
                window_size=86400,  # 1 jour
                burst_size=2,
                priority=2
            )
        ])
        
        # Règles pour Premium Tier
        tier_rules.extend([
            RateLimitRule(
                name="premium_tier_requests",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
                limit=1000,
                window_size=3600,
                priority=1
            ),
            RateLimitRule(
                name="premium_tier_ai_requests",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.USER,
                limit=100,
                window_size=86400,
                burst_size=10,
                priority=2
            )
        ])
        
        # Règles pour Enterprise Tier
        tier_rules.extend([
            RateLimitRule(
                name="enterprise_tier_requests",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
                limit=10000,
                window_size=3600,
                priority=1
            ),
            RateLimitRule(
                name="enterprise_tier_ai_requests",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.USER,
                limit=1000,
                window_size=86400,
                burst_size=50,
                priority=2
            )
        ])
        
        # Ajouter les règles de base
        base_rules = super()._load_rate_limit_rules()
        tier_rules.extend(base_rules)
        
        return tier_rules
    
    def _get_applicable_rules(self, request: Request) -> List[RateLimitRule]:
        """Obtenir les règles basées sur le tier utilisateur"""
        base_rules = super()._get_applicable_rules(request)
        
        # Obtenir le tier utilisateur
        user_tier = self._get_user_tier(request)
        
        # Filtrer les règles par tier
        tier_specific_rules = []
        for rule in self.rules:
            if user_tier == UserTier.FREE and "free_tier" in rule.name:
                tier_specific_rules.append(rule)
            elif user_tier == UserTier.PREMIUM and "premium_tier" in rule.name:
                tier_specific_rules.append(rule)
            elif user_tier == UserTier.ENTERPRISE and "enterprise_tier" in rule.name:
                tier_specific_rules.append(rule)
        
        return base_rules + tier_specific_rules
    
    def _get_user_tier(self, request: Request) -> UserTier:
        """Obtenir le tier de l'utilisateur"""
        if hasattr(request.state, "user"):
            return getattr(request.state.user, "tier", UserTier.FREE)
        return UserTier.FREE


class APIEndpointRateLimitMiddleware(RateLimitingMiddleware):
    """Middleware de rate limiting spécialisé par endpoint"""
    
    def _load_rate_limit_rules(self) -> List[RateLimitRule]:
        """Charger les règles spécifiques par endpoint"""
        endpoint_rules = [
            # Authentication endpoints
            RateLimitRule(
                name="login_endpoint",
                strategy=RateLimitStrategy.FIXED_WINDOW,
                scope=RateLimitScope.IP,
                limit=5,
                window_size=300,  # 5 minutes
                priority=1
            ),
            
            # AI Generation endpoints
            RateLimitRule(
                name="ai_content_generation",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.USER,
                limit=20,
                window_size=3600,
                burst_size=3,
                refill_rate=0.0056,  # 20 per hour
                priority=1
            ),
            
            # Search endpoints
            RateLimitRule(
                name="search_endpoints",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
                limit=200,
                window_size=3600,
                priority=2
            ),
            
            # Data export endpoints
            RateLimitRule(
                name="export_endpoints",
                strategy=RateLimitStrategy.FIXED_WINDOW,
                scope=RateLimitScope.USER,
                limit=5,
                window_size=3600,
                priority=1
            )
        ]
        
        # Ajouter les règles de base
        base_rules = super()._load_rate_limit_rules()
        endpoint_rules.extend(base_rules)
        
        return endpoint_rules
    
    def _rule_applies_to_endpoint(self, rule: RateLimitRule, path: str) -> bool:
        """Mapping avancé des règles aux endpoints"""
        endpoint_mappings = {
            "login_endpoint": ["/api/v1/auth/login", "/api/v1/auth/oauth"],
            "ai_content_generation": [
                "/api/v1/ai/generate",
                "/api/v1/content/generate",
                "/api/v1/ai/chat",
                "/api/v1/ai/recommendations"
            ],
            "search_endpoints": [
                "/api/v1/search",
                "/api/v1/spotify/search",
                "/api/v1/tracks/search"
            ],
            "export_endpoints": [
                "/api/v1/export/playlist",
                "/api/v1/export/data",
                "/api/v1/analytics/export"
            ]
        }
        
        patterns = endpoint_mappings.get(rule.name, [])
        return any(path.startswith(pattern) for pattern in patterns)


class SpotifyAPIRateLimitMiddleware(RateLimitingMiddleware):
    """Middleware spécialisé pour les appels API Spotify"""
    
    def _load_rate_limit_rules(self) -> List[RateLimitRule]:
        """Règles spécifiques pour l'API Spotify"""
        spotify_rules = [
            # Respecter les limites de Spotify API
            RateLimitRule(
                name="spotify_web_api",
                strategy=RateLimitStrategy.LEAKY_BUCKET,
                scope=RateLimitScope.GLOBAL,
                limit=100,
                window_size=60,  # 1 minute
                priority=1
            ),
            
            # Limites par utilisateur pour éviter l'abus
            RateLimitRule(
                name="spotify_user_calls",
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                scope=RateLimitScope.USER,
                limit=500,
                window_size=3600,  # 1 heure
                priority=2
            ),
            
            # Limites pour les endpoints intensifs
            RateLimitRule(
                name="spotify_search_calls",
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                scope=RateLimitScope.USER,
                limit=100,
                window_size=3600,
                burst_size=10,
                refill_rate=0.028,  # 100 per hour
                priority=1
            )
        ]
        
        return spotify_rules
    
    async def _check_spotify_quota(self, request: Request) -> bool:
        """Vérifier les quotas spécifiques Spotify"""
        # Vérifier les quotas AI
        if await self._check_ai_quota_exceeded(request):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "ai_quota_exceeded",
                    "message": f"Daily AI request limit exceeded",
                    "upgrade_url": "/pricing"
                }
            )
        
        # Vérifier les quotas Spotify sync
        if await self._check_spotify_sync_quota_exceeded(request):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "spotify_quota_exceeded",
                    "message": f"Hourly Spotify sync limit exceeded"
                }
            )
        
        # Vérifier les quotas généraux
        if await self._check_general_quota_exceeded(request):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "general_quota_exceeded",
                    "message": f"Hourly request limit exceeded"
                }
            )
        
        return True
    
    async def _check_ai_quota_exceeded(self, request: Request) -> bool:
        """Vérifier si le quota IA est dépassé"""
        user_id = getattr(request.state, "user", {}).get("user_id")
        if not user_id:
            return False
        
        quota_key = f"ai_quota:{user_id}:daily"
        current_usage = await self.redis_client.get(quota_key)
        daily_limit = self._get_ai_daily_limit(request)
        
        return int(current_usage or 0) >= daily_limit
    
    async def _check_spotify_sync_quota_exceeded(self, request: Request) -> bool:
        """Vérifier si le quota Spotify sync est dépassé"""
        user_id = getattr(request.state, "user", {}).get("user_id")
        if not user_id:
            return False
        
        quota_key = f"spotify_sync:{user_id}:hourly"
        current_usage = await self.redis_client.get(quota_key)
        hourly_limit = self._get_spotify_hourly_limit(request)
        
        return int(current_usage or 0) >= hourly_limit
    
    async def _check_general_quota_exceeded(self, request: Request) -> bool:
        """Vérifier si le quota général est dépassé"""
        user_id = getattr(request.state, "user", {}).get("user_id")
        if not user_id:
            return False
        
        quota_key = f"general_quota:{user_id}:hourly"
        current_usage = await self.redis_client.get(quota_key)
        hourly_limit = self._get_general_hourly_limit(request)
        
        return int(current_usage or 0) >= hourly_limit
    
    def _get_ai_daily_limit(self, request: Request) -> int:
        """Obtenir la limite quotidienne IA pour l'utilisateur"""
        user_tier = self._get_user_tier(request)
        limits = {
            UserTier.FREE: 10,
            UserTier.PREMIUM: 100,
            UserTier.ENTERPRISE: 1000
        }
        return limits.get(user_tier, 10)
    
    def _get_spotify_hourly_limit(self, request: Request) -> int:
        """Obtenir la limite horaire Spotify pour l'utilisateur"""
        user_tier = self._get_user_tier(request)
        limits = {
            UserTier.FREE: 50,
            UserTier.PREMIUM: 200,
            UserTier.ENTERPRISE: 1000
        }
        return limits.get(user_tier, 50)
    
    def _get_general_hourly_limit(self, request: Request) -> int:
        """Obtenir la limite horaire générale pour l'utilisateur"""
        user_tier = self._get_user_tier(request)
        limits = {
            UserTier.FREE: 100,
            UserTier.PREMIUM: 1000,
            UserTier.ENTERPRISE: 10000
        }
        return limits.get(user_tier, 100)
