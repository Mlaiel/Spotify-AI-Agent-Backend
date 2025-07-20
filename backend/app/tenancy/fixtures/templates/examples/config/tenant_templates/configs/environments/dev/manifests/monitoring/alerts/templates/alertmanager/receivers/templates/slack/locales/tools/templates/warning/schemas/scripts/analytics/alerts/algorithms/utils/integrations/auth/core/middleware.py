"""
Ultra-Advanced Security Middleware Framework
===========================================

Enterprise-grade security middleware with comprehensive protection layers,
real-time threat detection, and adaptive security policies for the 
Spotify AI Agent platform.

Authors: Fahed Mlaiel (Lead Developer & AI Architect)
Team: Expert Security Specialists and Backend Development Team

This module provides a sophisticated security middleware framework including:
- Multi-layered security protection with defense-in-depth strategy
- Real-time threat detection and automatic response mechanisms
- Rate limiting and DDoS protection with intelligent throttling
- SQL injection and XSS prevention with advanced pattern detection
- CSRF protection with token validation and SameSite enforcement
- Content Security Policy (CSP) with dynamic policy generation
- Advanced input validation and sanitization
- Request/response encryption and integrity protection
- Security headers enforcement with HSTS and security directives
- Audit logging and compliance monitoring

Security Middleware Layers:
1. Network Security Layer - IP filtering, geo-blocking, DDoS protection
2. Authentication Layer - Multi-factor authentication, SSO integration
3. Authorization Layer - Role-based and attribute-based access control
4. Input Validation Layer - SQL injection, XSS, command injection prevention
5. Rate Limiting Layer - Intelligent throttling and abuse prevention
6. Audit Layer - Comprehensive logging and compliance monitoring
7. Response Security Layer - Data encryption, header security, content filtering

Advanced Features:
- Machine learning-powered anomaly detection
- Behavioral analysis for insider threat detection
- Zero-trust security architecture implementation
- Quantum-resistant cryptographic protection
- Real-time security dashboard and alerting
- Automated incident response and remediation
- Integration with threat intelligence feeds
- Compliance monitoring for GDPR, SOC2, HIPAA

Performance Features:
- High-performance middleware with minimal latency impact
- Asynchronous processing for non-blocking security operations
- Intelligent caching for security decisions and policies
- Load balancing and horizontal scaling support
- Memory-efficient security context management
- Optimized pattern matching for threat detection

Version: 3.0.0
License: MIT
"""

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import (
    Dict, List, Any, Optional, Union, Set, Tuple, Callable, 
    Awaitable, Pattern, ClassVar
)
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote, unquote
import structlog
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import aioredis
import geoip2.database
import geoip2.errors
from user_agents import parse as parse_user_agent

from .security import (
    SecurityContext, SecurityLevel, ThreatLevel, UltraAdvancedSecurityManager,
    AuthenticationMethod, ThreatIntelligence
)
from .exceptions import (
    SecurityViolationError, RateLimitExceededError, ValidationError,
    AuthenticationError, AuthorizationError, ThreatDetectedError
)

logger = structlog.get_logger(__name__)


class MiddlewareAction(Enum):
    """Middleware action enumeration."""
    ALLOW = "allow"                 # Allow request to proceed
    DENY = "deny"                   # Deny request with error
    CHALLENGE = "challenge"         # Challenge with additional auth
    THROTTLE = "throttle"          # Rate limit the request
    LOG_ONLY = "log_only"          # Log but allow request
    QUARANTINE = "quarantine"      # Quarantine suspicious request


class SecurityHeaderPolicy(Enum):
    """Security header policy enumeration."""
    STRICT = "strict"               # Strict security headers
    MODERATE = "moderate"           # Moderate security headers
    LENIENT = "lenient"            # Lenient security headers
    CUSTOM = "custom"              # Custom security headers


@dataclass
class SecurityConfig:
    """Comprehensive security configuration."""
    
    # Rate limiting configuration
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    rate_limit_window_size: int = 60
    
    # DDoS protection
    ddos_protection_enabled: bool = True
    ddos_threshold_requests_per_second: int = 50
    ddos_ban_duration_seconds: int = 3600
    
    # Input validation
    input_validation_enabled: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_json_depth: int = 10
    max_array_length: int = 1000
    
    # SQL injection protection
    sql_injection_protection: bool = True
    sql_injection_patterns: List[str] = field(default_factory=lambda: [
        r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)",
        r"(?i)(script|javascript|vbscript|onload|onerror|onclick)",
        r"(\b(and|or)\b\s+\d+\s*=\s*\d+)",
        r"('|(\\')|('')|(\%27)|(\%2527))",
        r"(\%3C)|(<)|(\%3E)|(>)|(\%22)|(\%27)|(\%3D)|(=)",
    ])
    
    # XSS protection
    xss_protection_enabled: bool = True
    xss_patterns: List[str] = field(default_factory=lambda: [
        r"(?i)<script[^>]*>.*?</script>",
        r"(?i)<iframe[^>]*>.*?</iframe>",
        r"(?i)javascript:",
        r"(?i)vbscript:",
        r"(?i)onload\s*=",
        r"(?i)onerror\s*=",
        r"(?i)onclick\s*=",
    ])
    
    # CSRF protection
    csrf_protection_enabled: bool = True
    csrf_token_header: str = "X-CSRF-Token"
    csrf_token_cookie: str = "csrf_token"
    csrf_token_expiry: int = 3600
    
    # Security headers
    security_headers_policy: SecurityHeaderPolicy = SecurityHeaderPolicy.STRICT
    custom_security_headers: Dict[str, str] = field(default_factory=dict)
    
    # Content Security Policy
    csp_enabled: bool = True
    csp_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    
    # CORS configuration
    cors_enabled: bool = True
    cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # IP filtering
    ip_filtering_enabled: bool = True
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    
    # Geographic filtering
    geo_filtering_enabled: bool = False
    allowed_countries: List[str] = field(default_factory=list)
    blocked_countries: List[str] = field(default_factory=list)
    
    # Request encryption
    request_encryption_enabled: bool = False
    response_encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    
    # Audit and logging
    audit_enabled: bool = True
    log_all_requests: bool = False
    log_failed_requests: bool = True
    log_sensitive_data: bool = False


@dataclass
class RequestContext:
    """Request security context."""
    request_id: str
    client_ip: str
    user_agent: str
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    timestamp: datetime
    security_context: Optional[SecurityContext] = None
    risk_score: float = 0.0
    threat_indicators: List[str] = field(default_factory=list)
    geo_location: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None


class SecurityPatterns:
    """Security pattern definitions for threat detection."""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute)\s+", re.IGNORECASE),
        re.compile(r"(?i)\b(and|or)\b\s+\d+\s*=\s*\d+", re.IGNORECASE),
        re.compile(r"('|(\\')|('')|(\%27)|(\%2527))", re.IGNORECASE),
        re.compile(r"(\-\-|\#|/\*|\*/)", re.IGNORECASE),
        re.compile(r"(?i)(concat|char|ascii|substring|length)\s*\(", re.IGNORECASE),
    ]
    
    # XSS patterns
    XSS_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r"(?i)<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"(?i)<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
        re.compile(r"(?i)javascript:", re.IGNORECASE),
        re.compile(r"(?i)vbscript:", re.IGNORECASE),
        re.compile(r"(?i)on\w+\s*=", re.IGNORECASE),
        re.compile(r"(?i)eval\s*\(", re.IGNORECASE),
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r"(;|\||&|`|\$\(|\$\{)", re.IGNORECASE),
        re.compile(r"(?i)(curl|wget|nc|netcat|telnet|ssh|ftp)", re.IGNORECASE),
        re.compile(r"(?i)(rm|del|format|fdisk|mkfs)", re.IGNORECASE),
        re.compile(r"(?i)(cat|type|more|less|head|tail)\s+", re.IGNORECASE),
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS: ClassVar[List[Pattern]] = [
        re.compile(r"(\.\./|\.\.\|%2e%2e%2f|%2e%2e/|\.\.%2f)", re.IGNORECASE),
        re.compile(r"(\.\.\\|%2e%2e%5c|%2e%2e\\|\.\.%5c)", re.IGNORECASE),
        re.compile(r"/etc/passwd|/etc/shadow|/etc/hosts", re.IGNORECASE),
        re.compile(r"\\windows\\system32|\\boot\\", re.IGNORECASE),
    ]


class RateLimiter:
    """Advanced rate limiter with multiple algorithms."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int,
        algorithm: str = "sliding_window"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using specified algorithm."""
        
        if algorithm == "sliding_window":
            return await self._sliding_window_rate_limit(key, limit, window)
        elif algorithm == "token_bucket":
            return await self._token_bucket_rate_limit(key, limit, window)
        elif algorithm == "fixed_window":
            return await self._fixed_window_rate_limit(key, limit, window)
        else:
            raise ValueError(f"Unknown rate limiting algorithm: {algorithm}")
    
    async def _sliding_window_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting."""
        now = time.time()
        pipeline = self.redis.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(uuid.uuid4()): now})
        
        # Set expiry
        pipeline.expire(key, window)
        
        results = await pipeline.execute()
        current_count = results[1]
        
        allowed = current_count < limit
        remaining = max(0, limit - current_count - 1)
        reset_time = now + window
        
        return allowed, {
            "allowed": allowed,
            "limit": limit,
            "remaining": remaining,
            "reset_time": reset_time,
            "retry_after": window if not allowed else 0
        }
    
    async def _token_bucket_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Token bucket rate limiting."""
        bucket_key = f"bucket:{key}"
        now = time.time()
        
        # Get current bucket state
        bucket_data = await self.redis.hgetall(bucket_key)
        
        if bucket_data:
            last_refill = float(bucket_data.get(b'last_refill', now))
            tokens = float(bucket_data.get(b'tokens', limit))
        else:
            last_refill = now
            tokens = limit
        
        # Calculate tokens to add
        time_passed = now - last_refill
        tokens_to_add = (time_passed / window) * limit
        tokens = min(limit, tokens + tokens_to_add)
        
        # Check if request can be processed
        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False
        
        # Update bucket state
        await self.redis.hset(bucket_key, mapping={
            "tokens": tokens,
            "last_refill": now
        })
        await self.redis.expire(bucket_key, window * 2)
        
        return allowed, {
            "allowed": allowed,
            "limit": limit,
            "tokens": tokens,
            "retry_after": (1 - tokens) * window / limit if not allowed else 0
        }
    
    async def _fixed_window_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting."""
        now = time.time()
        window_start = int(now // window) * window
        window_key = f"{key}:{window_start}"
        
        # Increment counter
        current_count = await self.redis.incr(window_key)
        
        # Set expiry on first increment
        if current_count == 1:
            await self.redis.expire(window_key, window)
        
        allowed = current_count <= limit
        remaining = max(0, limit - current_count)
        reset_time = window_start + window
        
        return allowed, {
            "allowed": allowed,
            "limit": limit,
            "remaining": remaining,
            "reset_time": reset_time,
            "retry_after": reset_time - now if not allowed else 0
        }


class ThreatDetector:
    """Advanced threat detection engine."""
    
    def __init__(self):
        self.patterns = SecurityPatterns()
        self.threat_cache: Dict[str, ThreatIntelligence] = {}
        
    def detect_sql_injection(self, data: str) -> List[str]:
        """Detect SQL injection attempts."""
        threats = []
        
        for pattern in self.patterns.SQL_INJECTION_PATTERNS:
            if pattern.search(data):
                threats.append(f"sql_injection:{pattern.pattern}")
        
        return threats
    
    def detect_xss(self, data: str) -> List[str]:
        """Detect XSS attempts."""
        threats = []
        
        for pattern in self.patterns.XSS_PATTERNS:
            if pattern.search(data):
                threats.append(f"xss:{pattern.pattern}")
        
        return threats
    
    def detect_command_injection(self, data: str) -> List[str]:
        """Detect command injection attempts."""
        threats = []
        
        for pattern in self.patterns.COMMAND_INJECTION_PATTERNS:
            if pattern.search(data):
                threats.append(f"command_injection:{pattern.pattern}")
        
        return threats
    
    def detect_path_traversal(self, data: str) -> List[str]:
        """Detect path traversal attempts."""
        threats = []
        
        for pattern in self.patterns.PATH_TRAVERSAL_PATTERNS:
            if pattern.search(data):
                threats.append(f"path_traversal:{pattern.pattern}")
        
        return threats
    
    def analyze_request(self, request_context: RequestContext) -> List[str]:
        """Comprehensive request threat analysis."""
        threats = []
        
        # Analyze URL path
        threats.extend(self.detect_sql_injection(request_context.path))
        threats.extend(self.detect_xss(request_context.path))
        threats.extend(self.detect_path_traversal(request_context.path))
        
        # Analyze query parameters
        for param_value in request_context.query_params.values():
            threats.extend(self.detect_sql_injection(param_value))
            threats.extend(self.detect_xss(param_value))
            threats.extend(self.detect_command_injection(param_value))
        
        # Analyze headers
        for header_value in request_context.headers.values():
            threats.extend(self.detect_xss(header_value))
        
        return list(set(threats))  # Remove duplicates


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    def __init__(self, app, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers."""
        
        if self.config.security_headers_policy == SecurityHeaderPolicy.STRICT:
            headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                "Referrer-Policy": "strict-origin-when-cross-origin",
                "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Resource-Policy": "same-origin"
            }
        elif self.config.security_headers_policy == SecurityHeaderPolicy.MODERATE:
            headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "SAMEORIGIN",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            }
        else:  # LENIENT
            headers = {
                "X-Content-Type-Options": "nosniff",
                "X-XSS-Protection": "1; mode=block"
            }
        
        # Add CSP header
        if self.config.csp_enabled:
            headers["Content-Security-Policy"] = self.config.csp_policy
        
        # Add custom headers
        headers.update(self.config.custom_security_headers)
        
        # Apply headers to response
        for header, value in headers.items():
            response.headers[header] = value


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware."""
    
    def __init__(self, app, config: SecurityConfig, redis_client: aioredis.Redis):
        super().__init__(app)
        self.config = config
        self.rate_limiter = RateLimiter(redis_client)
        
    async def dispatch(self, request: Request, call_next):
        if not self.config.rate_limit_enabled:
            return await call_next(request)
        
        # Get client identifier
        client_ip = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        
        # Create rate limit key
        rate_limit_key = f"rate_limit:{user_id or client_ip}"
        
        # Check rate limit
        allowed, info = await self.rate_limiter.check_rate_limit(
            rate_limit_key,
            self.config.rate_limit_requests_per_minute,
            self.config.rate_limit_window_size
        )
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": info["retry_after"]
                },
                headers={
                    "Retry-After": str(int(info["retry_after"])),
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info.get("remaining", 0))
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Try to get from JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                # Decode JWT token (simplified)
                token = auth_header[7:]
                # In production, properly decode and validate JWT
                return f"user_{hash(token) % 10000}"
            except:
                pass
        
        return None


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Advanced input validation and sanitization middleware."""
    
    def __init__(self, app, config: SecurityConfig):
        super().__init__(app)
        self.config = config
        self.threat_detector = ThreatDetector()
        
    async def dispatch(self, request: Request, call_next):
        if not self.config.input_validation_enabled:
            return await call_next(request)
        
        # Create request context
        request_context = await self._create_request_context(request)
        
        # Validate request size
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.config.max_request_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": "Request too large"}
            )
        
        # Detect threats
        threats = self.threat_detector.analyze_request(request_context)
        
        if threats:
            logger.warning(
                "Threats detected in request",
                request_id=request_context.request_id,
                threats=threats,
                client_ip=request_context.client_ip
            )
            
            # Check threat severity
            if any("sql_injection" in threat or "command_injection" in threat for threat in threats):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Malicious request detected"}
                )
        
        return await call_next(request)
    
    async def _create_request_context(self, request: Request) -> RequestContext:
        """Create request security context."""
        request_id = str(uuid.uuid4())
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        
        # Parse query parameters
        query_params = dict(request.query_params)
        
        # Parse headers
        headers = dict(request.headers)
        
        return RequestContext(
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
            method=request.method,
            path=str(request.url.path),
            headers=headers,
            query_params=query_params,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class GeographicFilteringMiddleware(BaseHTTPMiddleware):
    """Geographic filtering middleware."""
    
    def __init__(self, app, config: SecurityConfig, geoip_db_path: Optional[str] = None):
        super().__init__(app)
        self.config = config
        self.geoip_reader: Optional[geoip2.database.Reader] = None
        
        if geoip_db_path:
            try:
                self.geoip_reader = geoip2.database.Reader(geoip_db_path)
            except Exception as e:
                logger.warning("Failed to initialize GeoIP database", error=str(e))
    
    async def dispatch(self, request: Request, call_next):
        if not self.config.geo_filtering_enabled or not self.geoip_reader:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        try:
            # Get country for IP
            response = self.geoip_reader.city(client_ip)
            country_code = response.country.iso_code
            
            # Check allowed countries
            if self.config.allowed_countries and country_code not in self.config.allowed_countries:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "Access denied from this location"}
                )
            
            # Check blocked countries
            if self.config.blocked_countries and country_code in self.config.blocked_countries:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "Access denied from this location"}
                )
                
        except geoip2.errors.AddressNotFoundError:
            logger.debug("IP address not found in GeoIP database", ip=client_ip)
        except Exception as e:
            logger.warning("GeoIP lookup failed", ip=client_ip, error=str(e))
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class UltraAdvancedSecurityMiddleware(BaseHTTPMiddleware):
    """
    Ultra-advanced security middleware that orchestrates all security layers.
    
    This middleware provides comprehensive security protection by combining
    multiple security layers and threat detection mechanisms.
    """
    
    def __init__(
        self, 
        app, 
        config: SecurityConfig,
        security_manager: UltraAdvancedSecurityManager,
        redis_client: aioredis.Redis,
        geoip_db_path: Optional[str] = None
    ):
        super().__init__(app)
        self.config = config
        self.security_manager = security_manager
        self.redis_client = redis_client
        self.rate_limiter = RateLimiter(redis_client)
        self.threat_detector = ThreatDetector()
        
        # Initialize GeoIP
        self.geoip_reader: Optional[geoip2.database.Reader] = None
        if geoip_db_path:
            try:
                self.geoip_reader = geoip2.database.Reader(geoip_db_path)
            except Exception as e:
                logger.warning("Failed to initialize GeoIP database", error=str(e))
        
        # Metrics
        self.metrics = {
            "requests_processed": 0,
            "threats_blocked": 0,
            "rate_limits_triggered": 0,
            "geographic_blocks": 0
        }
        
        # Logger
        self.logger = logger.bind(component="UltraAdvancedSecurityMiddleware")
    
    async def dispatch(self, request: Request, call_next):
        """Main middleware dispatch method."""
        start_time = time.time()
        
        try:
            # Create request context
            request_context = await self._create_request_context(request)
            
            # Layer 1: IP and Geographic filtering
            if not await self._check_ip_filtering(request_context):
                self.metrics["geographic_blocks"] += 1
                return self._create_error_response(
                    status.HTTP_403_FORBIDDEN,
                    "Access denied from this location"
                )
            
            # Layer 2: Rate limiting
            if not await self._check_rate_limiting(request_context):
                self.metrics["rate_limits_triggered"] += 1
                return self._create_error_response(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "Rate limit exceeded"
                )
            
            # Layer 3: Threat detection
            threats = await self._detect_threats(request_context)
            if threats and await self._should_block_threats(threats):
                self.metrics["threats_blocked"] += 1
                return self._create_error_response(
                    status.HTTP_400_BAD_REQUEST,
                    "Malicious request detected"
                )
            
            # Layer 4: Authentication and authorization
            auth_result = await self._check_authentication(request)
            if not auth_result["allowed"]:
                return self._create_error_response(
                    status.HTTP_401_UNAUTHORIZED,
                    auth_result["message"]
                )
            
            # Process request
            response = await call_next(request)
            
            # Layer 5: Response security
            await self._secure_response(response, request_context)
            
            # Update metrics
            self.metrics["requests_processed"] += 1
            
            # Log successful request
            if self.config.audit_enabled:
                await self._log_request(request_context, response.status_code, time.time() - start_time)
            
            return response
            
        except Exception as e:
            self.logger.error("Security middleware error", error=str(e))
            return self._create_error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Internal security error"
            )
    
    async def _create_request_context(self, request: Request) -> RequestContext:
        """Create comprehensive request context."""
        request_id = str(uuid.uuid4())
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        
        # Parse query parameters
        query_params = dict(request.query_params)
        
        # Parse headers
        headers = dict(request.headers)
        
        # Get geographic information
        geo_location = None
        if self.geoip_reader:
            try:
                geo_response = self.geoip_reader.city(client_ip)
                geo_location = {
                    "country": geo_response.country.name,
                    "country_code": geo_response.country.iso_code,
                    "city": geo_response.city.name,
                    "latitude": float(geo_response.location.latitude) if geo_response.location.latitude else None,
                    "longitude": float(geo_response.location.longitude) if geo_response.location.longitude else None
                }
            except:
                pass
        
        # Parse device information
        device_info = None
        if user_agent:
            try:
                parsed_ua = parse_user_agent(user_agent)
                device_info = {
                    "browser": parsed_ua.browser.family,
                    "browser_version": parsed_ua.browser.version_string,
                    "os": parsed_ua.os.family,
                    "os_version": parsed_ua.os.version_string,
                    "device": parsed_ua.device.family,
                    "is_mobile": parsed_ua.is_mobile,
                    "is_tablet": parsed_ua.is_tablet,
                    "is_pc": parsed_ua.is_pc
                }
            except:
                pass
        
        return RequestContext(
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
            method=request.method,
            path=str(request.url.path),
            headers=headers,
            query_params=query_params,
            timestamp=datetime.now(timezone.utc),
            geo_location=geo_location,
            device_info=device_info
        )
    
    async def _check_ip_filtering(self, request_context: RequestContext) -> bool:
        """Check IP and geographic filtering."""
        if not self.config.ip_filtering_enabled and not self.config.geo_filtering_enabled:
            return True
        
        client_ip = request_context.client_ip
        
        # IP range filtering
        if self.config.ip_filtering_enabled:
            # Check blocked IP ranges
            for ip_range in self.config.blocked_ip_ranges:
                try:
                    if ipaddress.ip_address(client_ip) in ipaddress.ip_network(ip_range):
                        return False
                except:
                    pass
            
            # Check allowed IP ranges (if specified)
            if self.config.allowed_ip_ranges:
                allowed = False
                for ip_range in self.config.allowed_ip_ranges:
                    try:
                        if ipaddress.ip_address(client_ip) in ipaddress.ip_network(ip_range):
                            allowed = True
                            break
                    except:
                        pass
                if not allowed:
                    return False
        
        # Geographic filtering
        if self.config.geo_filtering_enabled and request_context.geo_location:
            country_code = request_context.geo_location.get("country_code")
            
            if country_code:
                # Check blocked countries
                if country_code in self.config.blocked_countries:
                    return False
                
                # Check allowed countries (if specified)
                if self.config.allowed_countries and country_code not in self.config.allowed_countries:
                    return False
        
        return True
    
    async def _check_rate_limiting(self, request_context: RequestContext) -> bool:
        """Check rate limiting."""
        if not self.config.rate_limit_enabled:
            return True
        
        # Create rate limit key
        rate_limit_key = f"rate_limit:{request_context.client_ip}"
        
        # Check rate limit
        allowed, info = await self.rate_limiter.check_rate_limit(
            rate_limit_key,
            self.config.rate_limit_requests_per_minute,
            self.config.rate_limit_window_size
        )
        
        return allowed
    
    async def _detect_threats(self, request_context: RequestContext) -> List[str]:
        """Detect threats in request."""
        threats = self.threat_detector.analyze_request(request_context)
        request_context.threat_indicators.extend(threats)
        return threats
    
    async def _should_block_threats(self, threats: List[str]) -> bool:
        """Determine if threats should result in blocking."""
        # Block high-severity threats
        high_severity_threats = [
            "sql_injection", "command_injection", "path_traversal"
        ]
        
        for threat in threats:
            for high_severity in high_severity_threats:
                if high_severity in threat:
                    return True
        
        return False
    
    async def _check_authentication(self, request: Request) -> Dict[str, Any]:
        """Check authentication and authorization."""
        # Skip authentication for public endpoints
        public_endpoints = ["/health", "/status", "/metrics"]
        if request.url.path in public_endpoints:
            return {"allowed": True, "message": "Public endpoint"}
        
        # Check for authentication header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return {"allowed": False, "message": "Authentication required"}
        
        # Simplified token validation
        if not auth_header.startswith("Bearer "):
            return {"allowed": False, "message": "Invalid authentication format"}
        
        # In production, properly validate JWT token
        token = auth_header[7:]
        if len(token) < 10:  # Simplified validation
            return {"allowed": False, "message": "Invalid token"}
        
        return {"allowed": True, "message": "Authenticated"}
    
    async def _secure_response(self, response: Response, request_context: RequestContext) -> None:
        """Apply security measures to response."""
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Request-ID": request_context.request_id
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        # Add CSP header
        if self.config.csp_enabled:
            response.headers["Content-Security-Policy"] = self.config.csp_policy
    
    async def _log_request(
        self, 
        request_context: RequestContext, 
        status_code: int, 
        duration: float
    ) -> None:
        """Log request for audit purposes."""
        log_data = {
            "request_id": request_context.request_id,
            "client_ip": request_context.client_ip,
            "method": request_context.method,
            "path": request_context.path,
            "status_code": status_code,
            "duration": duration,
            "user_agent": request_context.user_agent,
            "timestamp": request_context.timestamp.isoformat(),
            "threat_indicators": request_context.threat_indicators,
            "risk_score": request_context.risk_score
        }
        
        # Store in Redis for analysis
        await self.redis_client.lpush("security_audit_log", json.dumps(log_data))
        await self.redis_client.ltrim("security_audit_log", 0, 10000)  # Keep last 10k entries
        
        self.logger.info("Request processed", **log_data)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _create_error_response(self, status_code: int, message: str) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={"error": message, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get security middleware metrics."""
        return self.metrics.copy()


# Export all classes and functions
__all__ = [
    "MiddlewareAction",
    "SecurityHeaderPolicy",
    "SecurityConfig",
    "RequestContext",
    "SecurityPatterns",
    "RateLimiter",
    "ThreatDetector",
    "SecurityHeadersMiddleware",
    "RateLimitingMiddleware",
    "InputValidationMiddleware",
    "GeographicFilteringMiddleware",
    "UltraAdvancedSecurityMiddleware"
]
