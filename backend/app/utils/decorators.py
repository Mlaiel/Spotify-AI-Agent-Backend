"""
Advanced Decorators for Enterprise Applications
==============================================

Collection de décorateurs industrialisés pour le backend Spotify AI Agent.
Inclut retry, cache, audit, rate limiting, circuit breakers et monitoring.
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
import weakref
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, ParamSpec
from enum import Enum
import inspect

# Type hints avancés
P = ParamSpec('P')
T = TypeVar('T')

logger = logging.getLogger(__name__)

# === Exceptions personnalisées ===
class RetryExhausted(Exception):
    """Exception levée quand toutes les tentatives de retry sont épuisées."""
    pass

class RateLimitExceeded(Exception):
    """Exception levée quand la limite de taux est dépassée."""
    pass

class CircuitBreakerOpen(Exception):
    """Exception levée quand le circuit breaker est ouvert."""
    pass

class ValidationError(Exception):
    """Exception levée lors d'erreurs de validation."""
    pass

# === Enums ===
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

# === Storage global pour les états ===
_circuit_breakers: Dict[str, Dict] = {}
_rate_limiters: Dict[str, deque] = defaultdict(deque)
_cache_storage: Dict[str, Any] = {}
_performance_metrics: Dict[str, List[float]] = defaultdict(list)

# === Décorateur Retry Avancé ===
def retry_async(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    max_backoff: float = 60.0,
    exceptions: tuple = (Exception,),
    jitter: bool = True,
    on_retry: Optional[Callable] = None
):
    """
    Décorateur de retry avancé avec backoff exponentiel et jitter.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        backoff_factor: Facteur multiplicateur pour le backoff
        max_backoff: Délai maximum entre tentatives
        exceptions: Types d'exceptions à retry
        jitter: Ajouter du bruit pour éviter thundering herd
        on_retry: Callback appelé à chaque retry
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"Retry exhausted for {func.__name__} after {max_attempts} attempts")
                        break
                    
                    # Calcul du délai avec backoff exponentiel
                    delay = min(backoff_factor ** attempt, max_backoff)
                    if jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}, retrying in {delay:.2f}s")
                    
                    if on_retry:
                        await on_retry(attempt, e, delay)
                    
                    await asyncio.sleep(delay)
            
            raise RetryExhausted(f"Failed after {max_attempts} attempts: {last_exception}")
        
        return wrapper
    return decorator

# === Décorateur Cache Avancé ===
def cache_result(
    ttl: int = 300,
    key_prefix: str = "",
    exclude_args: List[str] = None,
    cache_condition: Optional[Callable] = None,
    serializer: str = "json"
):
    """
    Décorateur de cache avancé avec TTL et conditions.
    
    Args:
        ttl: Time to live en secondes
        key_prefix: Préfixe pour les clés de cache
        exclude_args: Arguments à exclure de la clé
        cache_condition: Fonction pour déterminer si on doit cacher
        serializer: Type de sérialisation (json, pickle)
    """
    exclude_args = exclude_args or []
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Génération de la clé de cache
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Exclusion des arguments spécifiés
            cache_args = {k: v for k, v in bound_args.arguments.items() 
                         if k not in exclude_args}
            
            cache_key = f"{key_prefix}:{func.__name__}:{hashlib.md5(str(cache_args).encode()).hexdigest()}"
            
            # Vérification du cache
            if cache_key in _cache_storage:
                cached_data, timestamp = _cache_storage[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_data
                else:
                    del _cache_storage[cache_key]
            
            # Exécution de la fonction
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Mise en cache conditionnelle
            if cache_condition is None or cache_condition(result):
                _cache_storage[cache_key] = (result, time.time())
                logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator

# === Décorateur Audit Trail ===
def audit_trail(
    action: str,
    sensitive: bool = False,
    include_args: bool = True,
    include_result: bool = False,
    user_id_arg: str = "user_id"
):
    """
    Décorateur d'audit trail pour tracking des actions sensibles.
    
    Args:
        action: Nom de l'action à auditer
        sensitive: Si true, masque les données sensibles
        include_args: Inclure les arguments dans l'audit
        include_result: Inclure le résultat dans l'audit
        user_id_arg: Nom de l'argument contenant l'ID utilisateur
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            audit_data = {
                "action": action,
                "function": func.__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": kwargs.get(user_id_arg, "unknown"),
                "sensitive": sensitive
            }
            
            if include_args and not sensitive:
                audit_data["arguments"] = {
                    "args": args,
                    "kwargs": kwargs
                }
            elif include_args and sensitive:
                audit_data["arguments"] = "<REDACTED>"
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                audit_data.update({
                    "status": "success",
                    "duration_ms": (time.time() - start_time) * 1000
                })
                
                if include_result and not sensitive:
                    audit_data["result"] = result
                
                logger.info(f"AUDIT: {json.dumps(audit_data)}")
                return result
                
            except Exception as e:
                audit_data.update({
                    "status": "error",
                    "error": str(e),
                    "duration_ms": (time.time() - start_time) * 1000
                })
                
                logger.error(f"AUDIT: {json.dumps(audit_data)}")
                raise
        
        return wrapper
    return decorator

# === Décorateur Rate Limiting ===
def rate_limit(
    calls: int = 100,
    period: int = 60,
    per_user: bool = True,
    user_id_arg: str = "user_id",
    key_func: Optional[Callable] = None
):
    """
    Décorateur de rate limiting avancé.
    
    Args:
        calls: Nombre d'appels autorisés
        period: Période en secondes
        per_user: Rate limiting par utilisateur
        user_id_arg: Nom de l'argument contenant l'ID utilisateur
        key_func: Fonction custom pour générer la clé
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Génération de la clé de rate limiting
            if key_func:
                rl_key = key_func(*args, **kwargs)
            elif per_user:
                rl_key = f"{func.__name__}:user:{kwargs.get(user_id_arg, 'anonymous')}"
            else:
                rl_key = f"{func.__name__}:global"
            
            now = time.time()
            window_start = now - period
            
            # Nettoyage des anciennes entrées
            call_times = _rate_limiters[rl_key]
            while call_times and call_times[0] < window_start:
                call_times.popleft()
            
            # Vérification de la limite
            if len(call_times) >= calls:
                logger.warning(f"Rate limit exceeded for {rl_key}")
                raise RateLimitExceeded(f"Rate limit of {calls} calls per {period}s exceeded")
            
            # Enregistrement de l'appel
            call_times.append(now)
            
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator

# === Décorateur Circuit Breaker ===
def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: tuple = (Exception,),
    fallback_func: Optional[Callable] = None
):
    """
    Décorateur circuit breaker pour protection contre les failures en cascade.
    
    Args:
        failure_threshold: Nombre d'échecs avant ouverture
        recovery_timeout: Temps avant tentative de fermeture
        expected_exception: Types d'exceptions considérées comme échecs
        fallback_func: Fonction de fallback en cas d'ouverture
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        circuit_key = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Initialisation du circuit breaker
            if circuit_key not in _circuit_breakers:
                _circuit_breakers[circuit_key] = {
                    "state": CircuitState.CLOSED,
                    "failure_count": 0,
                    "last_failure_time": None,
                    "success_count": 0
                }
            
            circuit = _circuit_breakers[circuit_key]
            now = time.time()
            
            # Logique du circuit breaker
            if circuit["state"] == CircuitState.OPEN:
                if now - circuit["last_failure_time"] > recovery_timeout:
                    circuit["state"] = CircuitState.HALF_OPEN
                    circuit["success_count"] = 0
                    logger.info(f"Circuit breaker for {func.__name__} moved to HALF_OPEN")
                else:
                    if fallback_func:
                        return await fallback_func(*args, **kwargs)
                    raise CircuitBreakerOpen(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Succès
                if circuit["state"] == CircuitState.HALF_OPEN:
                    circuit["success_count"] += 1
                    if circuit["success_count"] >= 3:  # 3 succès consécutifs
                        circuit["state"] = CircuitState.CLOSED
                        circuit["failure_count"] = 0
                        logger.info(f"Circuit breaker for {func.__name__} moved to CLOSED")
                
                return result
                
            except expected_exception as e:
                circuit["failure_count"] += 1
                circuit["last_failure_time"] = now
                
                if circuit["failure_count"] >= failure_threshold:
                    circuit["state"] = CircuitState.OPEN
                    logger.warning(f"Circuit breaker for {func.__name__} moved to OPEN")
                
                raise
        
        return wrapper
    return decorator

# === Décorateur Performance Monitoring ===
def measure_performance(
    track_memory: bool = False,
    alert_threshold_ms: Optional[float] = None,
    store_metrics: bool = True
):
    """
    Décorateur de mesure de performance avec alertes.
    
    Args:
        track_memory: Suivre l'usage mémoire
        alert_threshold_ms: Seuil d'alerte en millisecondes
        store_metrics: Stocker les métriques pour analyse
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import psutil
            import os
            
            # Mesures initiales
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss if track_memory else 0
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Mesures finales
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                memory_delta = 0
                if track_memory:
                    end_memory = psutil.Process(os.getpid()).memory_info().rss
                    memory_delta = end_memory - start_memory
                
                # Stockage des métriques
                if store_metrics:
                    metrics_key = f"{func.__module__}.{func.__name__}"
                    _performance_metrics[metrics_key].append(duration_ms)
                    
                    # Gardez seulement les 1000 dernières mesures
                    if len(_performance_metrics[metrics_key]) > 1000:
                        _performance_metrics[metrics_key] = _performance_metrics[metrics_key][-1000:]
                
                # Alerte si nécessaire
                if alert_threshold_ms and duration_ms > alert_threshold_ms:
                    logger.warning(
                        f"Performance alert: {func.__name__} took {duration_ms:.2f}ms "
                        f"(threshold: {alert_threshold_ms}ms)"
                    )
                
                # Log des métriques
                log_data = {
                    "function": func.__name__,
                    "duration_ms": round(duration_ms, 2),
                    "status": "success"
                }
                
                if track_memory:
                    log_data["memory_delta_mb"] = round(memory_delta / 1024 / 1024, 2)
                
                logger.debug(f"PERFORMANCE: {json.dumps(log_data)}")
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"PERFORMANCE: {func.__name__} failed after {duration_ms:.2f}ms: {e}"
                )
                raise
        
        return wrapper
    return decorator

# === Décorateur Validation d'Input ===
def validate_input(
    schema: Optional[Dict] = None,
    validator_func: Optional[Callable] = None,
    sanitize: bool = True
):
    """
    Décorateur de validation d'input avec schemas JSON et sanitisation.
    
    Args:
        schema: Schema JSON pour validation
        validator_func: Fonction de validation personnalisée
        sanitize: Sanitiser les inputs string
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Validation avec schema JSON
            if schema:
                import jsonschema
                try:
                    # Convertir les args en dict pour validation
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    jsonschema.validate(bound_args.arguments, schema)
                except jsonschema.ValidationError as e:
                    raise ValidationError(f"Input validation failed: {e.message}")
            
            # Validation personnalisée
            if validator_func:
                validation_result = validator_func(*args, **kwargs)
                if not validation_result:
                    raise ValidationError("Custom validation failed")
            
            # Sanitisation des strings
            if sanitize:
                sanitized_args = []
                sanitized_kwargs = {}
                
                for arg in args:
                    if isinstance(arg, str):
                        # Sanitisation basique XSS
                        arg = arg.replace('<script', '&lt;script')
                        arg = arg.replace('javascript:', '')
                    sanitized_args.append(arg)
                
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        value = value.replace('<script', '&lt;script')
                        value = value.replace('javascript:', '')
                    sanitized_kwargs[key] = value
                
                args = tuple(sanitized_args)
                kwargs = sanitized_kwargs
            
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator

# === Décorateur Timeout ===
def timeout_async(seconds: float):
    """
    Décorateur de timeout pour fonctions asynchrones.
    
    Args:
        seconds: Timeout en secondes
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds}s")
                raise
        
        return wrapper
    return decorator

# === Décorateur Log Execution ===
def log_execution(
    log_level: str = "INFO",
    include_args: bool = False,
    include_result: bool = False,
    mask_sensitive: List[str] = None
):
    """
    Décorateur de logging d'exécution de fonction.
    
    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        include_args: Inclure les arguments
        include_result: Inclure le résultat
        mask_sensitive: Liste des noms d'arguments à masquer
    """
    mask_sensitive = mask_sensitive or ['password', 'token', 'secret', 'key']
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            
            # Préparation des données de log
            log_data = {
                "function": func.__name__,
                "module": func.__module__,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if include_args:
                # Masquage des données sensibles
                safe_kwargs = {}
                for key, value in kwargs.items():
                    if any(sensitive in key.lower() for sensitive in mask_sensitive):
                        safe_kwargs[key] = "***MASKED***"
                    else:
                        safe_kwargs[key] = value
                
                log_data.update({
                    "args_count": len(args),
                    "kwargs": safe_kwargs
                })
            
            # Log d'entrée
            getattr(logger, log_level.lower())(f"EXECUTING: {json.dumps(log_data)}")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Log de sortie
                duration_ms = (time.time() - start_time) * 1000
                exit_log = {
                    "function": func.__name__,
                    "status": "success",
                    "duration_ms": round(duration_ms, 2)
                }
                
                if include_result:
                    # Limiter la taille du résultat loggé
                    result_str = str(result)
                    if len(result_str) > 1000:
                        result_str = result_str[:1000] + "... (truncated)"
                    exit_log["result"] = result_str
                
                getattr(logger, log_level.lower())(f"COMPLETED: {json.dumps(exit_log)}")
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_log = {
                    "function": func.__name__,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration_ms": round(duration_ms, 2)
                }
                
                logger.error(f"FAILED: {json.dumps(error_log)}")
                raise
        
        return wrapper
    return decorator

# === Décorateur Authentification ===
def require_auth(
    roles: Optional[List[str]] = None,
    permissions: Optional[List[str]] = None,
    user_id_arg: str = "user_id",
    token_arg: str = "token"
):
    """
    Décorateur d'authentification et autorisation.
    
    Args:
        roles: Rôles requis
        permissions: Permissions requises
        user_id_arg: Nom de l'argument contenant l'ID utilisateur
        token_arg: Nom de l'argument contenant le token
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Récupération du token et user_id
            token = kwargs.get(token_arg)
            user_id = kwargs.get(user_id_arg)
            
            if not token:
                raise PermissionError("Authentication token required")
            
            # Simulation de validation de token (à remplacer par vraie logique)
            # En production, utiliser JWT validation, vérification DB, etc.
            if not _validate_token(token, user_id):
                raise PermissionError("Invalid authentication token")
            
            # Vérification des rôles
            if roles:
                user_roles = _get_user_roles(user_id)  # À implémenter
                if not any(role in user_roles for role in roles):
                    raise PermissionError(f"Required roles: {roles}")
            
            # Vérification des permissions
            if permissions:
                user_permissions = _get_user_permissions(user_id)  # À implémenter
                if not any(perm in user_permissions for perm in permissions):
                    raise PermissionError(f"Required permissions: {permissions}")
            
            # Log d'audit pour l'authentification
            logger.info(f"AUTH: User {user_id} accessed {func.__name__}")
            
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator

# === Fonctions utilitaires ===
def _validate_token(token: str, user_id: str) -> bool:
    """Validation simulée de token - à remplacer par vraie logique JWT."""
    # En production: décoder JWT, vérifier signature, expiration, etc.
    return len(token) > 10  # Validation basique pour démo

def _get_user_roles(user_id: str) -> List[str]:
    """Récupération simulée des rôles - à remplacer par requête DB."""
    # En production: requête DB pour récupérer les rôles
    return ["user", "premium"]  # Rôles par défaut pour démo

def _get_user_permissions(user_id: str) -> List[str]:
    """Récupération simulée des permissions - à remplacer par requête DB."""
    # En production: requête DB pour récupérer les permissions
    return ["read_tracks", "create_playlist"]  # Permissions par défaut pour démo

# === Fonctions utilitaires pour métriques ===
def get_performance_stats(function_name: str = None) -> Dict[str, Any]:
    """Récupère les statistiques de performance."""
    if function_name:
        if function_name in _performance_metrics:
            durations = _performance_metrics[function_name]
            return {
                "function": function_name,
                "calls": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0
            }
        return {"error": "Function not found"}
    
    # Stats globales
    return {
        "total_functions": len(_performance_metrics),
        "total_calls": sum(len(durations) for durations in _performance_metrics.values()),
        "functions": list(_performance_metrics.keys())
    }

def clear_cache():
    """Vide le cache global."""
    _cache_storage.clear()
    logger.info("Global cache cleared")

def get_circuit_breaker_status() -> Dict[str, Any]:
    """Récupère l'état des circuit breakers."""
    return {
        name: {
            "state": circuit["state"].value,
            "failure_count": circuit["failure_count"],
            "last_failure": circuit["last_failure_time"]
        }
        for name, circuit in _circuit_breakers.items()
    }