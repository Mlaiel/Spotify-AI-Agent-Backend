"""
Slack Webhook Handler - Gestionnaire robuste des webhooks Slack
Gestion avancée des retry, rate limiting et monitoring des performances
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac

import aiohttp
import aioredis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Histogram, Gauge
import backoff


class WebhookStatus(str, Enum):
    """États des webhooks"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    RETRYING = "retrying"


class SlackAPIError(Exception):
    """Exception pour les erreurs API Slack"""
    def __init__(self, status_code: int, response: Dict[str, Any]):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Slack API Error {status_code}: {response}")


@dataclass
class WebhookRequest:
    """Requête webhook structurée"""
    webhook_id: str
    url: str
    payload: Dict[str, Any]
    headers: Dict[str, str]
    tenant_id: str
    channel: str
    priority: int = 1
    max_retries: int = 3
    timeout: int = 30
    created_at: datetime = None
    scheduled_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at


@dataclass
class WebhookResponse:
    """Réponse webhook enrichie"""
    webhook_id: str
    status: WebhookStatus
    status_code: Optional[int] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class SlackWebhookHandler:
    """
    Gestionnaire avancé des webhooks Slack avec:
    - Retry automatique avec backoff exponentiel
    - Rate limiting intelligent par canal
    - Circuit breaker pour la résilience
    - Monitoring et métriques détaillées
    - Queue de priorité pour l'optimisation
    - Validation des signatures Slack
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_concurrent_requests: int = 50,
        default_timeout: int = 30
    ):
        self.redis_url = redis_url
        self.max_concurrent_requests = max_concurrent_requests
        self.default_timeout = default_timeout
        
        # Connexions et pools
        self.redis_pool = None
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Configuration
        self.config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "backoff_multiplier": 2.0,
            "rate_limit_window": 60,
            "rate_limit_default": 100,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 300
        }
        
        # Circuit breakers par channel
        self.circuit_breakers = {}
        
        # Métriques Prometheus
        self.metrics = {
            "webhooks_sent": Counter("slack_webhooks_sent_total", "Total webhooks sent", ["tenant_id", "channel", "status"]),
            "webhook_duration": Histogram("slack_webhook_duration_seconds", "Webhook execution time", ["tenant_id", "channel"]),
            "webhook_retries": Counter("slack_webhook_retries_total", "Webhook retries", ["tenant_id", "channel"]),
            "rate_limit_hits": Counter("slack_rate_limit_hits_total", "Rate limit hits", ["tenant_id", "channel"]),
            "circuit_breaker_trips": Counter("slack_circuit_breaker_trips_total", "Circuit breaker trips", ["channel"]),
            "active_webhooks": Gauge("slack_active_webhooks", "Currently active webhooks")
        }
        
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialise le gestionnaire de webhooks"""
        try:
            # Connexion Redis
            self.redis_pool = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Session HTTP avec configuration optimisée
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=50,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Spotify-AI-Agent-Webhook/2.1.0",
                    "Content-Type": "application/json"
                }
            )
            
            # Démarrage du worker de traitement
            asyncio.create_task(self._webhook_worker())
            
            # Démarrage du nettoyage périodique
            asyncio.create_task(self._cleanup_worker())
            
            self.logger.info("SlackWebhookHandler initialisé avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation: {e}")
            raise

    async def send_message(
        self,
        channel: str,
        message: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        webhook_url: Optional[str] = None
    ) -> WebhookResponse:
        """
        Envoie un message Slack via webhook
        
        Args:
            channel: Canal de destination
            message: Message Slack formaté
            metadata: Métadonnées additionnelles
            priority: Priorité du message (1-5)
            webhook_url: URL webhook spécifique
            
        Returns:
            Réponse du webhook
        """
        webhook_id = str(uuid.uuid4())
        
        try:
            # Validation du message
            await self._validate_message(message)
            
            # Récupération de l'URL webhook
            if not webhook_url:
                webhook_url = await self._get_webhook_url(channel)
            
            # Vérification du circuit breaker
            if await self._is_circuit_breaker_open(channel):
                return WebhookResponse(
                    webhook_id=webhook_id,
                    status=WebhookStatus.FAILED,
                    error_message="Circuit breaker ouvert"
                )
            
            # Vérification du rate limiting
            if await self._is_rate_limited(channel):
                self.metrics["rate_limit_hits"].labels(
                    tenant_id=metadata.get("tenant_id", "unknown"),
                    channel=channel
                ).inc()
                
                return WebhookResponse(
                    webhook_id=webhook_id,
                    status=WebhookStatus.RATE_LIMITED,
                    error_message="Rate limit atteint"
                )
            
            # Enrichissement du message
            enriched_message = await self._enrich_message(message, metadata)
            
            # Préparation de la requête
            webhook_request = WebhookRequest(
                webhook_id=webhook_id,
                url=webhook_url,
                payload=enriched_message,
                headers=await self._prepare_headers(channel, metadata),
                tenant_id=metadata.get("tenant_id", "unknown"),
                channel=channel,
                priority=priority
            )
            
            # Envoi avec retry automatique
            response = await self._send_with_retry(webhook_request)
            
            # Mise à jour des métriques
            self.metrics["webhooks_sent"].labels(
                tenant_id=webhook_request.tenant_id,
                channel=channel,
                status=response.status.value
            ).inc()
            
            # Mise à jour du circuit breaker
            await self._update_circuit_breaker(channel, response.status == WebhookStatus.SENT)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi du webhook {webhook_id}: {e}")
            
            return WebhookResponse(
                webhook_id=webhook_id,
                status=WebhookStatus.FAILED,
                error_message=str(e)
            )

    async def send_batch(
        self,
        messages: List[Dict[str, Any]],
        delay_between: float = 0.1
    ) -> List[WebhookResponse]:
        """Envoie plusieurs messages en lot avec délai"""
        responses = []
        
        for i, msg_data in enumerate(messages):
            if i > 0:
                await asyncio.sleep(delay_between)
            
            response = await self.send_message(**msg_data)
            responses.append(response)
        
        return responses

    async def get_webhook_stats(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """Récupère les statistiques des webhooks"""
        try:
            stats = {}
            
            if channel:
                # Stats pour un canal spécifique
                stats[channel] = await self._get_channel_stats(channel)
            else:
                # Stats globales
                channels = await self.redis_pool.smembers("slack:channels")
                for ch in channels:
                    stats[ch] = await self._get_channel_stats(ch)
            
            # Ajout des métriques système
            stats["system"] = {
                "active_connections": len(self.session._connector._conns) if self.session else 0,
                "circuit_breakers": {ch: await self._get_circuit_breaker_status(ch) for ch in self.circuit_breakers},
                "queue_size": await self.redis_pool.llen("slack:webhook_queue")
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {"error": str(e)}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, SlackAPIError))
    )
    async def _send_with_retry(self, webhook_request: WebhookRequest) -> WebhookResponse:
        """Envoie un webhook avec retry automatique"""
        start_time = time.time()
        
        try:
            async with self.semaphore:
                self.metrics["active_webhooks"].inc()
                
                try:
                    async with self.session.post(
                        webhook_request.url,
                        json=webhook_request.payload,
                        headers=webhook_request.headers,
                        timeout=aiohttp.ClientTimeout(total=webhook_request.timeout)
                    ) as response:
                        
                        execution_time = time.time() - start_time
                        
                        # Mise à jour des métriques de durée
                        self.metrics["webhook_duration"].labels(
                            tenant_id=webhook_request.tenant_id,
                            channel=webhook_request.channel
                        ).observe(execution_time)
                        
                        response_data = None
                        try:
                            response_data = await response.json()
                        except:
                            response_data = {"text": await response.text()}
                        
                        if response.status == 200:
                            # Mise à jour du rate limiting
                            await self._update_rate_limit(webhook_request.channel)
                            
                            return WebhookResponse(
                                webhook_id=webhook_request.webhook_id,
                                status=WebhookStatus.SENT,
                                status_code=response.status,
                                response_data=response_data,
                                execution_time=execution_time
                            )
                        
                        elif response.status == 429:
                            # Rate limiting Slack
                            retry_after = int(response.headers.get("Retry-After", 60))
                            await self._set_rate_limit(webhook_request.channel, retry_after)
                            
                            raise SlackAPIError(response.status, response_data)
                        
                        else:
                            # Autres erreurs
                            raise SlackAPIError(response.status, response_data)
                
                finally:
                    self.metrics["active_webhooks"].dec()
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Comptage des retries
            self.metrics["webhook_retries"].labels(
                tenant_id=webhook_request.tenant_id,
                channel=webhook_request.channel
            ).inc()
            
            if isinstance(e, SlackAPIError):
                return WebhookResponse(
                    webhook_id=webhook_request.webhook_id,
                    status=WebhookStatus.FAILED,
                    status_code=e.status_code,
                    response_data=e.response,
                    error_message=str(e),
                    execution_time=execution_time
                )
            else:
                return WebhookResponse(
                    webhook_id=webhook_request.webhook_id,
                    status=WebhookStatus.FAILED,
                    error_message=str(e),
                    execution_time=execution_time
                )

    async def _validate_message(self, message: Dict[str, Any]):
        """Valide un message Slack"""
        if not isinstance(message, dict):
            raise ValueError("Le message doit être un dictionnaire")
        
        # Vérification des champs requis
        if "text" not in message and "blocks" not in message and "attachments" not in message:
            raise ValueError("Le message doit contenir au moins 'text', 'blocks' ou 'attachments'")
        
        # Limites Slack
        if "text" in message and len(message["text"]) > 40000:
            raise ValueError("Le texte dépasse la limite de 40000 caractères")
        
        if "blocks" in message and len(message["blocks"]) > 50:
            raise ValueError("Trop de blocs (maximum 50)")

    async def _enrich_message(self, message: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrichit le message avec des métadonnées"""
        enriched = message.copy()
        
        if metadata:
            # Ajout d'un footer avec les métadonnées
            if "blocks" in enriched:
                footer_block = {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"🤖 Spotify AI Agent | Alert ID: {metadata.get('alert_id', 'N/A')} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                        }
                    ]
                }
                enriched["blocks"].append(footer_block)
        
        return enriched

    async def _prepare_headers(self, channel: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Prépare les headers HTTP"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Spotify-AI-Agent-Webhook/2.1.0",
            "X-Slack-Channel": channel
        }
        
        if metadata:
            if "webhook_secret" in metadata:
                # Signature HMAC pour validation
                timestamp = str(int(time.time()))
                headers["X-Slack-Request-Timestamp"] = timestamp
                
                sig_basestring = f"v0:{timestamp}:{json.dumps(metadata)}"
                signature = hmac.new(
                    metadata["webhook_secret"].encode(),
                    sig_basestring.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Slack-Signature"] = f"v0={signature}"
        
        return headers

    async def _get_webhook_url(self, channel: str) -> str:
        """Récupère l'URL webhook pour un canal"""
        # Récupération depuis Redis ou configuration
        webhook_url = await self.redis_pool.hget("slack:webhooks", channel)
        
        if not webhook_url:
            # Fallback vers webhook par défaut
            webhook_url = await self.redis_pool.get("slack:default_webhook")
            
        if not webhook_url:
            raise ValueError(f"Aucun webhook configuré pour le canal {channel}")
        
        return webhook_url

    async def _is_rate_limited(self, channel: str) -> bool:
        """Vérifie si un canal est rate limité"""
        rate_limit_key = f"slack:rate_limit:{channel}"
        current_count = await self.redis_pool.get(rate_limit_key)
        
        if current_count is None:
            return False
        
        limit = self.config["rate_limit_default"]
        return int(current_count) >= limit

    async def _update_rate_limit(self, channel: str):
        """Met à jour le compteur de rate limiting"""
        rate_limit_key = f"slack:rate_limit:{channel}"
        pipe = self.redis_pool.pipeline()
        pipe.incr(rate_limit_key)
        pipe.expire(rate_limit_key, self.config["rate_limit_window"])
        await pipe.execute()

    async def _set_rate_limit(self, channel: str, duration: int):
        """Définit un rate limit temporaire"""
        rate_limit_key = f"slack:rate_limit:{channel}"
        await self.redis_pool.setex(rate_limit_key, duration, self.config["rate_limit_default"])

    async def _is_circuit_breaker_open(self, channel: str) -> bool:
        """Vérifie l'état du circuit breaker"""
        cb_key = f"slack:circuit_breaker:{channel}"
        cb_data = await self.redis_pool.hgetall(cb_key)
        
        if not cb_data:
            return False
        
        failure_count = int(cb_data.get("failures", 0))
        last_failure = datetime.fromisoformat(cb_data.get("last_failure", "1970-01-01"))
        
        # Circuit ouvert si trop d'échecs récents
        if failure_count >= self.config["circuit_breaker_threshold"]:
            if datetime.utcnow() - last_failure < timedelta(seconds=self.config["circuit_breaker_timeout"]):
                return True
            else:
                # Reset du circuit breaker après timeout
                await self.redis_pool.delete(cb_key)
        
        return False

    async def _update_circuit_breaker(self, channel: str, success: bool):
        """Met à jour l'état du circuit breaker"""
        cb_key = f"slack:circuit_breaker:{channel}"
        
        if success:
            # Reset en cas de succès
            await self.redis_pool.delete(cb_key)
        else:
            # Incrémentation des échecs
            pipe = self.redis_pool.pipeline()
            pipe.hincrby(cb_key, "failures", 1)
            pipe.hset(cb_key, "last_failure", datetime.utcnow().isoformat())
            pipe.expire(cb_key, self.config["circuit_breaker_timeout"] * 2)
            await pipe.execute()
            
            # Métriques
            self.metrics["circuit_breaker_trips"].labels(channel=channel).inc()

    async def _get_circuit_breaker_status(self, channel: str) -> Dict[str, Any]:
        """Récupère le statut du circuit breaker"""
        cb_key = f"slack:circuit_breaker:{channel}"
        cb_data = await self.redis_pool.hgetall(cb_key)
        
        if not cb_data:
            return {"status": "closed", "failures": 0}
        
        return {
            "status": "open" if await self._is_circuit_breaker_open(channel) else "closed",
            "failures": int(cb_data.get("failures", 0)),
            "last_failure": cb_data.get("last_failure")
        }

    async def _get_channel_stats(self, channel: str) -> Dict[str, Any]:
        """Récupère les stats d'un canal"""
        stats = {}
        
        # Rate limiting
        rate_limit_key = f"slack:rate_limit:{channel}"
        current_count = await self.redis_pool.get(rate_limit_key)
        stats["rate_limit"] = {
            "current": int(current_count) if current_count else 0,
            "limit": self.config["rate_limit_default"]
        }
        
        # Circuit breaker
        stats["circuit_breaker"] = await self._get_circuit_breaker_status(channel)
        
        return stats

    async def _webhook_worker(self):
        """Worker pour traiter la queue des webhooks"""
        while True:
            try:
                # Traitement de la queue prioritaire
                webhook_data = await self.redis_pool.blpop("slack:webhook_queue", timeout=1)
                
                if webhook_data:
                    queue_name, webhook_json = webhook_data
                    webhook_request = WebhookRequest(**json.loads(webhook_json))
                    
                    response = await self._send_with_retry(webhook_request)
                    
                    # Log du résultat
                    self.logger.info(f"Webhook {webhook_request.webhook_id} traité: {response.status}")
                
            except Exception as e:
                self.logger.error(f"Erreur dans le webhook worker: {e}")
                await asyncio.sleep(1)

    async def _cleanup_worker(self):
        """Worker de nettoyage périodique"""
        while True:
            try:
                # Nettoyage des données expirées
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Nettoyage des métriques anciennes
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Ici on pourrait nettoyer les anciennes données Redis
                # selon les besoins de rétention
                
            except Exception as e:
                self.logger.error(f"Erreur dans le cleanup worker: {e}")

    async def cleanup(self):
        """Nettoyage des ressources"""
        if self.session:
            await self.session.close()
        if self.redis_pool:
            await self.redis_pool.close()
