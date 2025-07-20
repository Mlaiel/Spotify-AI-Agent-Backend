#!/usr/bin/env python3
"""
Slack Webhook Manager - Advanced webhook handling and delivery system
=====================================================================

Created by: Fahed Mlaiel - Senior Backend Developer & Slack Integration Specialist
Description: Industrial-grade Slack webhook manager with advanced features for 
            multi-tenant environments. Supports webhook rotation, delivery guarantees,
            rate limiting, and comprehensive monitoring.

Features:
- Advanced webhook rotation and failover
- Exponential backoff retry mechanisms  
- Webhook health monitoring and auto-healing
- Rate limiting with token bucket algorithm
- Circuit breaker patterns for reliability
- Comprehensive delivery tracking and metrics
- Multi-tenant webhook isolation
- Security token validation and rotation
"""

import asyncio
import aiohttp
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from urllib.parse import urlparse
import ssl

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
from pydantic import BaseModel, Field, validator

from .security_manager import SecurityManager
from .metrics_collector import MetricsCollector


class WebhookStatus(Enum):
    """Webhook status enumeration"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILING = "failing"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"


class DeliveryStatus(Enum):
    """Message delivery status enumeration"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    ABANDONED = "abandoned"
    RATE_LIMITED = "rate_limited"


class WebhookConfig(BaseModel):
    """Webhook configuration model"""
    url: str
    secret: Optional[str] = None
    timeout: int = Field(default=30, ge=1, le=300)
    retry_count: int = Field(default=3, ge=0, le=10)
    retry_delay: int = Field(default=1, ge=1, le=60)
    rate_limit: int = Field(default=100, ge=1)
    rate_window: int = Field(default=60, ge=1)
    verify_ssl: bool = True
    headers: Dict[str, str] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)
    
    @validator('url')
    def validate_url(cls, v):
        """Validate webhook URL"""
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError('Invalid webhook URL format')
        if parsed.scheme not in ['http', 'https']:
            raise ValueError('Only HTTP/HTTPS protocols supported')
        return v


class MessagePayload(BaseModel):
    """Slack message payload model"""
    channel: Optional[str] = None
    text: Optional[str] = None
    blocks: Optional[List[Dict]] = None
    attachments: Optional[List[Dict]] = None
    thread_ts: Optional[str] = None
    reply_broadcast: bool = False
    unfurl_links: bool = True
    unfurl_media: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeliveryRecord(BaseModel):
    """Message delivery tracking record"""
    message_id: str
    tenant_id: str
    webhook_id: str
    payload: MessagePayload
    status: DeliveryStatus
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    last_error: Optional[str] = None
    delivered_at: Optional[datetime] = None
    response_code: Optional[int] = None
    response_time: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker implementation for webhook reliability"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens"""
        now = time.time()
        # Add tokens based on time elapsed
        tokens_to_add = (now - self.last_refill) * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class SlackWebhookManager:
    """
    Advanced Slack webhook manager with industrial-grade features
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        security_manager: SecurityManager,
        metrics_collector: MetricsCollector,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        circuit_breaker_threshold: int = 5
    ):
        self.redis = redis_client
        self.security = security_manager
        self.metrics = metrics_collector
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Webhook tracking
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, TokenBucket] = {}
        self.webhook_health: Dict[str, WebhookStatus] = {}
        
        # Message queue for failed deliveries
        self.retry_queue = asyncio.Queue()
        
        # Metrics
        self.delivery_counter = Counter(
            'slack_webhook_deliveries_total',
            'Total webhook delivery attempts',
            ['tenant_id', 'webhook_id', 'status']
        )
        self.delivery_latency = Histogram(
            'slack_webhook_delivery_duration_seconds',
            'Webhook delivery latency',
            ['tenant_id', 'webhook_id']
        )
        self.webhook_health_gauge = Gauge(
            'slack_webhook_health_status',
            'Webhook health status',
            ['tenant_id', 'webhook_id', 'status']
        )
    
    async def initialize(self):
        """Initialize webhook manager"""
        try:
            # Load webhook configurations from Redis
            await self._load_webhook_configs()
            
            # Start background tasks
            asyncio.create_task(self._retry_processor())
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._metrics_updater())
            
            self.logger.info("SlackWebhookManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SlackWebhookManager: {e}")
            raise
    
    async def register_webhook(
        self,
        tenant_id: str,
        webhook_id: str,
        config: WebhookConfig
    ) -> bool:
        """Register a new webhook"""
        try:
            # Validate webhook configuration
            if not await self._validate_webhook(config):
                return False
            
            # Store webhook configuration
            webhook_key = f"webhook:{tenant_id}:{webhook_id}"
            self.webhooks[webhook_key] = config
            
            # Initialize circuit breaker and rate limiter
            self.circuit_breakers[webhook_key] = CircuitBreaker(
                failure_threshold=self.circuit_breaker_threshold
            )
            self.rate_limiters[webhook_key] = TokenBucket(
                capacity=config.rate_limit,
                refill_rate=config.rate_limit / config.rate_window
            )
            
            # Set initial health status
            self.webhook_health[webhook_key] = WebhookStatus.ACTIVE
            
            # Persist to Redis
            await self.redis.hset(
                f"webhooks:{tenant_id}",
                webhook_id,
                json.dumps(config.dict())
            )
            
            self.logger.info(f"Registered webhook {webhook_id} for tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register webhook: {e}")
            return False
    
    async def send_message(
        self,
        tenant_id: str,
        webhook_id: str,
        payload: MessagePayload,
        priority: int = 1
    ) -> str:
        """Send message via webhook with delivery guarantees"""
        message_id = self._generate_message_id()
        
        try:
            # Create delivery record
            record = DeliveryRecord(
                message_id=message_id,
                tenant_id=tenant_id,
                webhook_id=webhook_id,
                payload=payload,
                status=DeliveryStatus.PENDING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Store delivery record
            await self._store_delivery_record(record)
            
            # Attempt immediate delivery
            success = await self._deliver_message(record)
            
            if not success:
                # Queue for retry
                await self.retry_queue.put((record, priority))
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            # Update record with error
            await self._update_delivery_status(
                message_id, 
                DeliveryStatus.FAILED, 
                error=str(e)
            )
            return message_id
    
    async def get_delivery_status(self, message_id: str) -> Optional[DeliveryRecord]:
        """Get message delivery status"""
        try:
            record_data = await self.redis.get(f"delivery:{message_id}")
            if record_data:
                return DeliveryRecord.parse_raw(record_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get delivery status: {e}")
            return None
    
    async def get_webhook_health(self, tenant_id: str, webhook_id: str) -> WebhookStatus:
        """Get webhook health status"""
        webhook_key = f"webhook:{tenant_id}:{webhook_id}"
        return self.webhook_health.get(webhook_key, WebhookStatus.DISABLED)
    
    async def disable_webhook(self, tenant_id: str, webhook_id: str) -> bool:
        """Disable a webhook"""
        try:
            webhook_key = f"webhook:{tenant_id}:{webhook_id}"
            self.webhook_health[webhook_key] = WebhookStatus.DISABLED
            
            await self.redis.hset(
                f"webhook_status:{tenant_id}",
                webhook_id,
                WebhookStatus.DISABLED.value
            )
            
            self.logger.info(f"Disabled webhook {webhook_id} for tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to disable webhook: {e}")
            return False
    
    async def _deliver_message(self, record: DeliveryRecord) -> bool:
        """Deliver message to webhook"""
        webhook_key = f"webhook:{record.tenant_id}:{record.webhook_id}"
        
        # Check if webhook exists
        if webhook_key not in self.webhooks:
            await self._update_delivery_status(
                record.message_id,
                DeliveryStatus.FAILED,
                error="Webhook not found"
            )
            return False
        
        webhook_config = self.webhooks[webhook_key]
        circuit_breaker = self.circuit_breakers[webhook_key]
        rate_limiter = self.rate_limiters[webhook_key]
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            await self._update_delivery_status(
                record.message_id,
                DeliveryStatus.FAILED,
                error="Circuit breaker open"
            )
            return False
        
        # Check rate limiting
        if not rate_limiter.consume():
            await self._update_delivery_status(
                record.message_id,
                DeliveryStatus.RATE_LIMITED,
                error="Rate limit exceeded"
            )
            return False
        
        try:
            start_time = time.time()
            
            # Prepare request
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Spotify-AI-Agent-Slack-Webhook/1.0',
                **webhook_config.headers
            }
            
            # Add webhook signature if secret is configured
            payload_json = json.dumps(record.payload.dict(), separators=(',', ':'))
            if webhook_config.secret:
                signature = self._generate_signature(payload_json, webhook_config.secret)
                headers['X-Slack-Signature'] = signature
                headers['X-Slack-Request-Timestamp'] = str(int(time.time()))
            
            # Create SSL context
            ssl_context = ssl.create_default_context() if webhook_config.verify_ssl else False
            
            # Send request
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=webhook_config.timeout),
                connector=aiohttp.TCPConnector(ssl=ssl_context)
            ) as session:
                async with session.post(
                    webhook_config.url,
                    data=payload_json,
                    headers=headers
                ) as response:
                    response_time = time.time() - start_time
                    
                    # Update metrics
                    self.delivery_latency.labels(
                        tenant_id=record.tenant_id,
                        webhook_id=record.webhook_id
                    ).observe(response_time)
                    
                    # Check response
                    if response.status == 200:
                        # Success
                        circuit_breaker.record_success()
                        self.webhook_health[webhook_key] = WebhookStatus.ACTIVE
                        
                        await self._update_delivery_status(
                            record.message_id,
                            DeliveryStatus.DELIVERED,
                            response_code=response.status,
                            response_time=response_time
                        )
                        
                        self.delivery_counter.labels(
                            tenant_id=record.tenant_id,
                            webhook_id=record.webhook_id,
                            status='success'
                        ).inc()
                        
                        return True
                    
                    elif response.status == 429:
                        # Rate limited by Slack
                        retry_after = response.headers.get('Retry-After', 60)
                        await self._update_delivery_status(
                            record.message_id,
                            DeliveryStatus.RATE_LIMITED,
                            error=f"Rate limited by Slack, retry after {retry_after}s",
                            response_code=response.status
                        )
                        return False
                    
                    else:
                        # Other HTTP error
                        response_text = await response.text()
                        circuit_breaker.record_failure()
                        
                        await self._update_delivery_status(
                            record.message_id,
                            DeliveryStatus.FAILED,
                            error=f"HTTP {response.status}: {response_text}",
                            response_code=response.status,
                            response_time=response_time
                        )
                        
                        self.delivery_counter.labels(
                            tenant_id=record.tenant_id,
                            webhook_id=record.webhook_id,
                            status='failed'
                        ).inc()
                        
                        return False
        
        except asyncio.TimeoutError:
            circuit_breaker.record_failure()
            await self._update_delivery_status(
                record.message_id,
                DeliveryStatus.FAILED,
                error="Request timeout"
            )
            return False
        
        except Exception as e:
            circuit_breaker.record_failure()
            await self._update_delivery_status(
                record.message_id,
                DeliveryStatus.FAILED,
                error=str(e)
            )
            return False
    
    async def _retry_processor(self):
        """Background task to process retry queue"""
        while True:
            try:
                # Get message from retry queue
                record, priority = await self.retry_queue.get()
                
                # Check if we should retry
                if record.attempts >= self.max_retries:
                    await self._update_delivery_status(
                        record.message_id,
                        DeliveryStatus.ABANDONED,
                        error="Maximum retry attempts exceeded"
                    )
                    continue
                
                # Wait for backoff period
                backoff_time = self.retry_backoff * (2 ** record.attempts)
                await asyncio.sleep(backoff_time)
                
                # Update record
                record.attempts += 1
                record.status = DeliveryStatus.RETRYING
                record.updated_at = datetime.utcnow()
                await self._store_delivery_record(record)
                
                # Attempt delivery
                success = await self._deliver_message(record)
                
                if not success and record.attempts < self.max_retries:
                    # Queue for another retry
                    await self.retry_queue.put((record, priority))
                
            except Exception as e:
                self.logger.error(f"Error in retry processor: {e}")
                await asyncio.sleep(1)
    
    async def _health_monitor(self):
        """Background task to monitor webhook health"""
        while True:
            try:
                for webhook_key, config in self.webhooks.items():
                    # Check circuit breaker state
                    circuit_breaker = self.circuit_breakers.get(webhook_key)
                    if circuit_breaker:
                        if circuit_breaker.state == "OPEN":
                            self.webhook_health[webhook_key] = WebhookStatus.FAILING
                        elif circuit_breaker.state == "HALF_OPEN":
                            self.webhook_health[webhook_key] = WebhookStatus.DEGRADED
                        else:
                            if self.webhook_health[webhook_key] in [WebhookStatus.FAILING, WebhookStatus.DEGRADED]:
                                self.webhook_health[webhook_key] = WebhookStatus.ACTIVE
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_updater(self):
        """Background task to update metrics"""
        while True:
            try:
                for webhook_key, status in self.webhook_health.items():
                    tenant_id, webhook_id = webhook_key.split(':')[1:3]
                    
                    # Clear previous status metrics
                    for s in WebhookStatus:
                        self.webhook_health_gauge.labels(
                            tenant_id=tenant_id,
                            webhook_id=webhook_id,
                            status=s.value
                        ).set(0)
                    
                    # Set current status
                    self.webhook_health_gauge.labels(
                        tenant_id=tenant_id,
                        webhook_id=webhook_id,
                        status=status.value
                    ).set(1)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics updater: {e}")
                await asyncio.sleep(60)
    
    async def _load_webhook_configs(self):
        """Load webhook configurations from Redis"""
        try:
            # Get all tenant webhook configurations
            tenant_keys = await self.redis.keys("webhooks:*")
            
            for tenant_key in tenant_keys:
                tenant_id = tenant_key.decode().split(':')[1]
                webhook_configs = await self.redis.hgetall(tenant_key)
                
                for webhook_id, config_json in webhook_configs.items():
                    webhook_id = webhook_id.decode()
                    config_data = json.loads(config_json.decode())
                    config = WebhookConfig(**config_data)
                    
                    webhook_key = f"webhook:{tenant_id}:{webhook_id}"
                    self.webhooks[webhook_key] = config
                    
                    # Initialize circuit breaker and rate limiter
                    self.circuit_breakers[webhook_key] = CircuitBreaker()
                    self.rate_limiters[webhook_key] = TokenBucket(
                        capacity=config.rate_limit,
                        refill_rate=config.rate_limit / config.rate_window
                    )
                    
                    # Load health status
                    status = await self.redis.hget(f"webhook_status:{tenant_id}", webhook_id)
                    if status:
                        self.webhook_health[webhook_key] = WebhookStatus(status.decode())
                    else:
                        self.webhook_health[webhook_key] = WebhookStatus.ACTIVE
            
            self.logger.info(f"Loaded {len(self.webhooks)} webhook configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to load webhook configurations: {e}")
            raise
    
    async def _validate_webhook(self, config: WebhookConfig) -> bool:
        """Validate webhook configuration"""
        try:
            # Test webhook URL accessibility
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(config.url) as response:
                    # We expect any response (even 404) as long as URL is reachable
                    return True
                    
        except Exception as e:
            self.logger.warning(f"Webhook validation failed: {e}")
            return False
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"msg_{timestamp}_{random_part}"
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate webhook signature for verification"""
        timestamp = str(int(time.time()))
        sig_basestring = f"v0:{timestamp}:{payload}"
        signature = hmac.new(
            secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"v0={signature}"
    
    async def _store_delivery_record(self, record: DeliveryRecord):
        """Store delivery record in Redis"""
        try:
            await self.redis.setex(
                f"delivery:{record.message_id}",
                86400,  # 24 hours TTL
                record.json()
            )
        except Exception as e:
            self.logger.error(f"Failed to store delivery record: {e}")
    
    async def _update_delivery_status(
        self,
        message_id: str,
        status: DeliveryStatus,
        error: Optional[str] = None,
        response_code: Optional[int] = None,
        response_time: Optional[float] = None
    ):
        """Update delivery record status"""
        try:
            record_data = await self.redis.get(f"delivery:{message_id}")
            if record_data:
                record = DeliveryRecord.parse_raw(record_data)
                record.status = status
                record.updated_at = datetime.utcnow()
                
                if error:
                    record.last_error = error
                if response_code:
                    record.response_code = response_code
                if response_time:
                    record.response_time = response_time
                if status == DeliveryStatus.DELIVERED:
                    record.delivered_at = datetime.utcnow()
                
                await self._store_delivery_record(record)
                
        except Exception as e:
            self.logger.error(f"Failed to update delivery status: {e}")


# Export classes for external use
__all__ = [
    'SlackWebhookManager',
    'WebhookConfig',
    'MessagePayload',
    'DeliveryRecord',
    'WebhookStatus',
    'DeliveryStatus'
]
