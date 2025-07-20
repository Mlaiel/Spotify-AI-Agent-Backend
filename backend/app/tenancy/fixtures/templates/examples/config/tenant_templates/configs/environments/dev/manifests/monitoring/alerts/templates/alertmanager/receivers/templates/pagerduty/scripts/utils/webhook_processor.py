#!/usr/bin/env python3
"""
Webhook Processor for PagerDuty Integration.

Advanced webhook processing system for handling incoming webhooks from
PagerDuty and other monitoring systems with validation, parsing, and routing.

Features:
- Webhook signature validation and security
- Multi-format payload parsing (JSON, XML, form-data)
- Event routing and filtering
- Asynchronous processing and queuing
- Rate limiting and abuse protection
- Webhook replay and debugging
- Custom event handlers and transformers
- Monitoring and analytics
"""

import asyncio
import hmac
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import urllib.parse
import base64

from .metrics_collector import MetricsCollector
from .audit_logger import AuditLogger
from .rate_limiter import RateLimiter
from .data_transformer import DataTransformer

logger = logging.getLogger(__name__)


class WebhookValidationError(Exception):
    """Exception raised for webhook validation errors."""
    pass


class WebhookProcessingError(Exception):
    """Exception raised for webhook processing errors."""
    pass


class WebhookFormat(Enum):
    """Supported webhook payload formats."""
    JSON = "json"
    XML = "xml"
    FORM_DATA = "form_data"
    PLAIN_TEXT = "plain_text"


class WebhookEventType(Enum):
    """PagerDuty webhook event types."""
    INCIDENT_TRIGGER = "incident.trigger"
    INCIDENT_ACKNOWLEDGE = "incident.acknowledge"
    INCIDENT_RESOLVE = "incident.resolve"
    INCIDENT_ASSIGN = "incident.assign"
    INCIDENT_ESCALATE = "incident.escalate"
    INCIDENT_DELEGATE = "incident.delegate"
    INCIDENT_ANNOTATE = "incident.annotate"
    SERVICE_CREATE = "service.create"
    SERVICE_UPDATE = "service.update"
    SERVICE_DELETE = "service.delete"
    LOG_ENTRY_CREATE = "log_entry.create"


@dataclass
class WebhookRequest:
    """Webhook request data."""
    method: str
    url: str
    headers: Dict[str, str]
    body: bytes
    query_params: Dict[str, str] = field(default_factory=dict)
    remote_addr: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WebhookEvent:
    """Parsed webhook event."""
    event_type: str
    event_id: str
    webhook_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WebhookConfig:
    """Webhook endpoint configuration."""
    name: str
    endpoint_path: str
    secret: Optional[str] = None
    signature_header: str = "X-PagerDuty-Signature"
    timestamp_header: Optional[str] = "X-PagerDuty-Timestamp"
    max_age_seconds: int = 300  # 5 minutes
    allowed_ips: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 1000
    format: WebhookFormat = WebhookFormat.JSON
    custom_headers: Dict[str, str] = field(default_factory=dict)


class WebhookProcessor:
    """
    Advanced webhook processor with security and reliability features.
    
    Features:
    - Signature validation and security
    - Rate limiting and IP filtering
    - Multi-format payload parsing
    - Event routing and processing
    - Async processing with queues
    - Comprehensive monitoring
    """
    
    def __init__(self,
                 default_secret: Optional[str] = None,
                 enable_rate_limiting: bool = True,
                 enable_monitoring: bool = True):
        """
        Initialize webhook processor.
        
        Args:
            default_secret: Default webhook secret for validation
            enable_rate_limiting: Enable rate limiting protection
            enable_monitoring: Enable metrics and monitoring
        """
        self.default_secret = default_secret
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_monitoring = enable_monitoring
        
        # Webhook configurations
        self.webhooks: Dict[str, WebhookConfig] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.global_handlers: List[Callable] = []
        
        # Security and rate limiting
        self.rate_limiter = RateLimiter(requests_per_minute=1000) if enable_rate_limiting else None
        
        # Monitoring
        self.metrics = MetricsCollector() if enable_monitoring else None
        self.audit_logger = AuditLogger()
        
        # Data transformer for payload processing
        self.data_transformer = DataTransformer()
        
        # Processing queue
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []
        self.max_workers = 5
        
        logger.info("Webhook processor initialized")
    
    def register_webhook(self, config: WebhookConfig):
        """Register a webhook endpoint configuration."""
        self.webhooks[config.endpoint_path] = config
        logger.info(f"Registered webhook endpoint: {config.endpoint_path}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """
        Register an event handler for specific event type.
        
        Args:
            event_type: Event type to handle (e.g., 'incident.trigger')
            handler: Async function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Registered event handler for: {event_type}")
    
    def register_global_handler(self, handler: Callable):
        """Register a global event handler that processes all events."""
        self.global_handlers.append(handler)
        logger.debug("Registered global event handler")
    
    async def process_webhook(self, request: WebhookRequest) -> Dict[str, Any]:
        """
        Process incoming webhook request.
        
        Args:
            request: Webhook request data
            
        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        
        try:
            # Find webhook configuration
            webhook_config = self._find_webhook_config(request.url)
            if not webhook_config:
                raise WebhookValidationError(f"Unknown webhook endpoint: {request.url}")
            
            # Apply rate limiting
            if self.rate_limiter:
                key = request.remote_addr or "unknown"
                await self.rate_limiter.wait_if_needed(key)
            
            # Validate request
            await self._validate_request(request, webhook_config)
            
            # Parse payload
            payload = await self._parse_payload(request, webhook_config)
            
            # Extract events
            events = await self._extract_events(payload, webhook_config)
            
            # Queue events for processing
            for event in events:
                await self.processing_queue.put((event, webhook_config))
            
            # Record metrics
            if self.metrics:
                self.metrics.increment('webhook_requests_total')
                self.metrics.increment('webhook_requests_success')
                self.metrics.record_histogram(
                    'webhook_processing_duration',
                    time.time() - start_time
                )
            
            # Audit log
            self.audit_logger.log_webhook_processed(
                endpoint=webhook_config.endpoint_path,
                remote_addr=request.remote_addr,
                event_count=len(events),
                success=True
            )
            
            return {
                'status': 'success',
                'events_processed': len(events),
                'webhook_id': webhook_config.name
            }
            
        except Exception as e:
            # Record error metrics
            if self.metrics:
                self.metrics.increment('webhook_requests_total')
                self.metrics.increment('webhook_requests_failed')
            
            # Audit log error
            self.audit_logger.log_webhook_processed(
                endpoint=request.url,
                remote_addr=request.remote_addr,
                error=str(e),
                success=False
            )
            
            logger.error(f"Webhook processing failed: {e}")
            raise WebhookProcessingError(f"Webhook processing failed: {e}")
    
    def _find_webhook_config(self, url: str) -> Optional[WebhookConfig]:
        """Find webhook configuration by URL path."""
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        
        return self.webhooks.get(path)
    
    async def _validate_request(self, request: WebhookRequest, config: WebhookConfig):
        """Validate webhook request security and format."""
        # Check IP whitelist
        if config.allowed_ips and request.remote_addr:
            if request.remote_addr not in config.allowed_ips:
                raise WebhookValidationError(f"IP address not allowed: {request.remote_addr}")
        
        # Validate signature if secret is configured
        secret = config.secret or self.default_secret
        if secret:
            await self._validate_signature(request, secret, config)
        
        # Validate timestamp to prevent replay attacks
        if config.timestamp_header:
            await self._validate_timestamp(request, config)
        
        # Validate content type
        content_type = request.headers.get('Content-Type', '').lower()
        if config.format == WebhookFormat.JSON and 'application/json' not in content_type:
            raise WebhookValidationError("Expected JSON content type")
        elif config.format == WebhookFormat.XML and 'application/xml' not in content_type:
            raise WebhookValidationError("Expected XML content type")
    
    async def _validate_signature(self, request: WebhookRequest, secret: str, config: WebhookConfig):
        """Validate webhook signature."""
        signature_header = request.headers.get(config.signature_header)
        if not signature_header:
            raise WebhookValidationError(f"Missing signature header: {config.signature_header}")
        
        # Extract signature (format: v1=<signature>)
        try:
            version, signature = signature_header.split('=', 1)
        except ValueError:
            raise WebhookValidationError("Invalid signature format")
        
        if version != 'v1':
            raise WebhookValidationError(f"Unsupported signature version: {version}")
        
        # Calculate expected signature
        if config.timestamp_header and config.timestamp_header in request.headers:
            # Include timestamp in signature calculation (PagerDuty v3 format)
            timestamp = request.headers[config.timestamp_header]
            payload = f"{timestamp}.{request.body.decode('utf-8')}"
        else:
            payload = request.body.decode('utf-8')
        
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        if not hmac.compare_digest(signature, expected_signature):
            raise WebhookValidationError("Invalid signature")
    
    async def _validate_timestamp(self, request: WebhookRequest, config: WebhookConfig):
        """Validate request timestamp to prevent replay attacks."""
        timestamp_header = request.headers.get(config.timestamp_header)
        if not timestamp_header:
            raise WebhookValidationError(f"Missing timestamp header: {config.timestamp_header}")
        
        try:
            webhook_timestamp = int(timestamp_header)
            current_timestamp = int(time.time())
            
            if abs(current_timestamp - webhook_timestamp) > config.max_age_seconds:
                raise WebhookValidationError("Request timestamp too old")
                
        except ValueError:
            raise WebhookValidationError("Invalid timestamp format")
    
    async def _parse_payload(self, request: WebhookRequest, config: WebhookConfig) -> Dict[str, Any]:
        """Parse webhook payload based on format."""
        try:
            if config.format == WebhookFormat.JSON:
                return json.loads(request.body.decode('utf-8'))
            
            elif config.format == WebhookFormat.XML:
                # Use data transformer for XML parsing
                return self.data_transformer.transform(
                    request.body.decode('utf-8'),
                    source_format=self.data_transformer.TransformationFormat.XML,
                    target_format=self.data_transformer.TransformationFormat.JSON
                )
            
            elif config.format == WebhookFormat.FORM_DATA:
                return dict(urllib.parse.parse_qsl(request.body.decode('utf-8')))
            
            elif config.format == WebhookFormat.PLAIN_TEXT:
                return {'body': request.body.decode('utf-8')}
            
            else:
                raise WebhookProcessingError(f"Unsupported payload format: {config.format}")
                
        except Exception as e:
            raise WebhookProcessingError(f"Failed to parse payload: {e}")
    
    async def _extract_events(self, payload: Dict[str, Any], config: WebhookConfig) -> List[WebhookEvent]:
        """Extract events from webhook payload."""
        events = []
        
        # Handle PagerDuty v3 webhook format
        if 'messages' in payload:
            for message in payload['messages']:
                event = self._create_pagerduty_event(message)
                if event:
                    events.append(event)
        
        # Handle PagerDuty v2 webhook format
        elif 'type' in payload and 'data' in payload:
            event = self._create_pagerduty_event(payload)
            if event:
                events.append(event)
        
        # Handle custom format
        else:
            event = WebhookEvent(
                event_type='custom',
                event_id=payload.get('id', f"event_{int(time.time())}"),
                data=payload,
                metadata={'webhook_config': config.name}
            )
            events.append(event)
        
        return events
    
    def _create_pagerduty_event(self, message: Dict[str, Any]) -> Optional[WebhookEvent]:
        """Create webhook event from PagerDuty message."""
        try:
            event_type = message.get('event')
            if not event_type:
                return None
            
            # Extract resource information
            resource_type = None
            resource_id = None
            data = message.get('data', {})
            
            if 'incident' in data:
                resource_type = 'incident'
                resource_id = data['incident'].get('id')
            elif 'service' in data:
                resource_type = 'service'
                resource_id = data['service'].get('id')
            elif 'log_entry' in data:
                resource_type = 'log_entry'
                resource_id = data['log_entry'].get('id')
            
            return WebhookEvent(
                event_type=event_type,
                event_id=message.get('id', f"pd_event_{int(time.time())}"),
                webhook_id=message.get('webhook', {}).get('id'),
                resource_type=resource_type,
                resource_id=resource_id,
                data=data,
                metadata={
                    'created_on': message.get('created_on'),
                    'webhook_summary': message.get('summary')
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create PagerDuty event: {e}")
            return None
    
    async def _process_event(self, event: WebhookEvent, config: WebhookConfig):
        """Process a single webhook event."""
        try:
            # Call global handlers first
            for handler in self.global_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Global handler error: {e}")
            
            # Call specific event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Event handler error for {event.event_type}: {e}")
            
            # Record success metrics
            if self.metrics:
                self.metrics.increment('webhook_events_processed')
                self.metrics.increment(f'webhook_events_{event.event_type}_processed')
            
            logger.debug(f"Processed event: {event.event_type} ({event.event_id})")
            
        except Exception as e:
            # Record error metrics
            if self.metrics:
                self.metrics.increment('webhook_events_failed')
                self.metrics.increment(f'webhook_events_{event.event_type}_failed')
            
            logger.error(f"Failed to process event {event.event_id}: {e}")
            raise
    
    async def _worker(self):
        """Background worker to process events from queue."""
        while True:
            try:
                # Get event from queue
                event, config = await self.processing_queue.get()
                
                # Process event
                await self._process_event(event, config)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                self.processing_queue.task_done()
    
    def start_workers(self):
        """Start background workers for event processing."""
        if self.worker_tasks:
            return  # Already started
        
        loop = asyncio.get_event_loop()
        for i in range(self.max_workers):
            task = loop.create_task(self._worker())
            self.worker_tasks.append(task)
        
        logger.info(f"Started {self.max_workers} webhook processing workers")
    
    async def stop_workers(self):
        """Stop background workers."""
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for cancellation
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Stopped webhook processing workers")
    
    async def drain_queue(self, timeout: float = 30.0):
        """Wait for all queued events to be processed."""
        try:
            await asyncio.wait_for(self.processing_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Queue drain timed out after {timeout} seconds")
    
    def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook processing statistics."""
        stats = {
            'registered_webhooks': len(self.webhooks),
            'event_handlers': {
                event_type: len(handlers)
                for event_type, handlers in self.event_handlers.items()
            },
            'global_handlers': len(self.global_handlers),
            'queue_size': self.processing_queue.qsize(),
            'active_workers': len(self.worker_tasks)
        }
        
        if self.metrics:
            stats['metrics'] = self.metrics.get_all_metrics()
        
        return stats
    
    def create_pagerduty_webhook_config(self, 
                                       name: str,
                                       endpoint_path: str,
                                       secret: str) -> WebhookConfig:
        """Create standard PagerDuty webhook configuration."""
        return WebhookConfig(
            name=name,
            endpoint_path=endpoint_path,
            secret=secret,
            signature_header="X-PagerDuty-Signature",
            timestamp_header="X-PagerDuty-Timestamp",
            max_age_seconds=300,
            rate_limit_per_minute=1000,
            format=WebhookFormat.JSON,
            custom_headers={}
        )


# Global webhook processor instance
_global_webhook_processor = None

def get_webhook_processor() -> WebhookProcessor:
    """Get global webhook processor instance."""
    global _global_webhook_processor
    if _global_webhook_processor is None:
        _global_webhook_processor = WebhookProcessor()
    return _global_webhook_processor


# Convenience functions for common PagerDuty events

async def handle_incident_trigger(event: WebhookEvent):
    """Example handler for incident trigger events."""
    incident_data = event.data.get('incident', {})
    logger.info(f"Incident triggered: {incident_data.get('id')} - {incident_data.get('title')}")


async def handle_incident_resolve(event: WebhookEvent):
    """Example handler for incident resolve events."""
    incident_data = event.data.get('incident', {})
    logger.info(f"Incident resolved: {incident_data.get('id')} - {incident_data.get('title')}")


def setup_default_pagerduty_handlers():
    """Setup default PagerDuty event handlers."""
    processor = get_webhook_processor()
    
    processor.register_event_handler('incident.trigger', handle_incident_trigger)
    processor.register_event_handler('incident.resolve', handle_incident_resolve)
    
    logger.info("Default PagerDuty handlers registered")
