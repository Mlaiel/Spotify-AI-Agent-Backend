"""
Gestionnaire de Webhooks Slack Ultra-Avancé
===========================================

Module de gestion avancée des webhooks Slack pour le système AlertManager
du Spotify AI Agent. Fournit une gestion robuste, sécurisée et performante
des webhooks avec retry automatique, rate limiting et monitoring complet.

Développé par l'équipe Backend Senior sous la direction de Fahed Mlaiel.
"""

import asyncio
import logging
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse
import aiohttp
import ssl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import backoff

from . import SlackSeverity, SlackChannelType, SlackNotificationStatus
from .utils import SlackUtils

logger = logging.getLogger(__name__)

# Métriques Prometheus
webhook_requests_total = Counter(
    'slack_webhook_requests_total',
    'Nombre total de requêtes webhook Slack',
    ['tenant_id', 'severity', 'status']
)

webhook_duration_seconds = Histogram(
    'slack_webhook_duration_seconds',
    'Durée des requêtes webhook Slack',
    ['tenant_id', 'severity']
)

webhook_queue_size = Gauge(
    'slack_webhook_queue_size',
    'Taille de la queue des webhooks',
    ['tenant_id']
)

@dataclass
class WebhookRequest:
    """Représente une requête webhook Slack."""
    
    id: str = field(default_factory=lambda: SlackUtils.generate_id())
    tenant_id: str = ""
    webhook_url: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    severity: SlackSeverity = SlackSeverity.INFO
    channel_type: SlackChannelType = SlackChannelType.ALERTS
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 30
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    status: SlackNotificationStatus = SlackNotificationStatus.PENDING
    error_message: Optional[str] = None
    response_code: Optional[int] = None
    response_time: Optional[float] = None

@dataclass 
class WebhookResponse:
    """Représente la réponse d'un webhook Slack."""
    
    request_id: str
    status_code: int
    response_body: str
    headers: Dict[str, str]
    response_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = field(init=False)
    
    def __post_init__(self):
        self.success = 200 <= self.status_code < 300

@dataclass
class WebhookConfig:
    """Configuration d'un webhook Slack."""
    
    url: str
    tenant_id: str
    signing_secret: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    rate_limit: int = 50  # requêtes par minute
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class SlackWebhookManager:
    """
    Gestionnaire ultra-avancé des webhooks Slack.
    
    Fonctionnalités:
    - Pool de connexions HTTP réutilisables
    - Rate limiting intelligent par tenant
    - Retry automatique avec backoff exponentiel
    - Queue de priorité pour les alertes critiques
    - Validation de signature Slack
    - Métriques et monitoring complets
    - Circuit breaker pattern
    - Audit trail détaillé
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 max_concurrent_requests: int = 100,
                 default_timeout: int = 30,
                 rate_limit_per_minute: int = 100):
        """
        Initialise le gestionnaire de webhooks.
        
        Args:
            redis_client: Client Redis pour la queue et le cache
            max_concurrent_requests: Nombre max de requêtes concurrentes
            default_timeout: Timeout par défaut en secondes
            rate_limit_per_minute: Limite de taux par minute
        """
        self.redis_client = redis_client
        self.max_concurrent_requests = max_concurrent_requests
        self.default_timeout = default_timeout
        self.rate_limit_per_minute = rate_limit_per_minute
        
        # Semaphore pour limiter les requêtes concurrentes
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Session HTTP avec pool de connexions
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Queues par priorité
        self.priority_queues = {
            SlackSeverity.CRITICAL: asyncio.Queue(),
            SlackSeverity.HIGH: asyncio.Queue(),
            SlackSeverity.MEDIUM: asyncio.Queue(),
            SlackSeverity.LOW: asyncio.Queue(),
            SlackSeverity.INFO: asyncio.Queue()
        }
        
        # Configuration des webhooks par tenant
        self.webhook_configs: Dict[str, Dict[str, WebhookConfig]] = {}
        
        # Rate limiting par tenant
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breakers par webhook
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Métriques internes
        self.metrics = {
            'requests_sent': 0,
            'requests_failed': 0,
            'requests_retried': 0,
            'rate_limited': 0,
            'circuit_breaker_opened': 0,
            'queue_overflow': 0
        }
        
        # Workers actifs
        self.workers_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        logger.info("SlackWebhookManager initialisé")
    
    async def __aenter__(self):
        """Contexte manager - entrée."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Contexte manager - sortie."""
        await self.stop()
    
    async def start(self):
        """Démarre le gestionnaire de webhooks."""
        try:
            # Créer la session HTTP
            connector = aiohttp.TCPConnector(
                limit=200,
                limit_per_host=50,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=ssl.create_default_context()
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.default_timeout,
                connect=10,
                sock_read=self.default_timeout
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Spotify-AI-Agent-WebhookManager/2.1.0',
                    'Content-Type': 'application/json'
                }
            )
            
            # Démarrer les workers
            await self._start_workers()
            
            logger.info("SlackWebhookManager démarré avec succès")
            
        except Exception as e:
            logger.error(f"Erreur démarrage SlackWebhookManager: {e}")
            raise
    
    async def stop(self):
        """Arrête le gestionnaire de webhooks."""
        try:
            # Arrêter les workers
            await self._stop_workers()
            
            # Fermer la session HTTP
            if self.session:
                await self.session.close()
                self.session = None
            
            logger.info("SlackWebhookManager arrêté")
            
        except Exception as e:
            logger.error(f"Erreur arrêt SlackWebhookManager: {e}")
    
    async def _start_workers(self):
        """Démarre les workers de traitement des queues."""
        if self.workers_running:
            return
        
        self.workers_running = True
        
        # Worker par niveau de priorité
        for severity in SlackSeverity:
            worker_task = asyncio.create_task(
                self._queue_worker(severity)
            )
            self.worker_tasks.append(worker_task)
        
        # Worker de maintenance
        maintenance_task = asyncio.create_task(self._maintenance_worker())
        self.worker_tasks.append(maintenance_task)
        
        logger.info(f"Démarré {len(self.worker_tasks)} workers")
    
    async def _stop_workers(self):
        """Arrête les workers."""
        self.workers_running = False
        
        # Annuler toutes les tâches
        for task in self.worker_tasks:
            task.cancel()
        
        # Attendre la fin des tâches
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Workers arrêtés")
    
    async def _queue_worker(self, severity: SlackSeverity):
        """Worker pour traiter une queue de priorité."""
        queue = self.priority_queues[severity]
        
        while self.workers_running:
            try:
                # Attendre une requête avec timeout
                request = await asyncio.wait_for(
                    queue.get(),
                    timeout=1.0
                )
                
                # Traiter la requête
                await self._process_webhook_request(request)
                
                # Marquer la tâche comme terminée
                queue.task_done()
                
            except asyncio.TimeoutError:
                # Pas de requête en attente
                continue
            except Exception as e:
                logger.error(f"Erreur worker {severity.value}: {e}")
                await asyncio.sleep(1)
    
    async def _maintenance_worker(self):
        """Worker de maintenance périodique."""
        while self.workers_running:
            try:
                await asyncio.sleep(60)  # Maintenance toutes les minutes
                
                # Nettoyer les rate limiters expirés
                await self._cleanup_rate_limiters()
                
                # Réinitialiser les circuit breakers si nécessaire
                await self._reset_circuit_breakers()
                
                # Mettre à jour les métriques
                await self._update_queue_metrics()
                
            except Exception as e:
                logger.error(f"Erreur worker maintenance: {e}")
    
    async def register_webhook(self, 
                             tenant_id: str,
                             webhook_url: str,
                             config: Optional[WebhookConfig] = None) -> bool:
        """
        Enregistre un nouveau webhook pour un tenant.
        
        Args:
            tenant_id: ID du tenant
            webhook_url: URL du webhook Slack
            config: Configuration optionnelle
            
        Returns:
            True si succès, False sinon
        """
        try:
            # Valider l'URL
            if not self._validate_webhook_url(webhook_url):
                raise ValueError(f"URL webhook invalide: {webhook_url}")
            
            # Créer la configuration par défaut si nécessaire
            if config is None:
                config = WebhookConfig(
                    url=webhook_url,
                    tenant_id=tenant_id,
                    timeout=self.default_timeout,
                    rate_limit=self.rate_limit_per_minute
                )
            
            # Stocker la configuration
            if tenant_id not in self.webhook_configs:
                self.webhook_configs[tenant_id] = {}
            
            webhook_key = self._get_webhook_key(webhook_url)
            self.webhook_configs[tenant_id][webhook_key] = config
            
            # Initialiser le rate limiter
            await self._init_rate_limiter(tenant_id, webhook_key, config.rate_limit)
            
            # Initialiser le circuit breaker
            await self._init_circuit_breaker(tenant_id, webhook_key)
            
            # Persister en Redis si disponible
            if self.redis_client:
                await self._persist_webhook_config(tenant_id, webhook_key, config)
            
            logger.info(f"Webhook enregistré: {tenant_id}/{webhook_key}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur enregistrement webhook: {e}")
            return False
    
    def _validate_webhook_url(self, url: str) -> bool:
        """Valide une URL de webhook Slack."""
        try:
            parsed = urlparse(url)
            
            # Vérifier le schéma
            if parsed.scheme != 'https':
                return False
            
            # Vérifier le domaine
            if not parsed.netloc.endswith('slack.com'):
                return False
            
            # Vérifier le chemin
            if not parsed.path.startswith('/services/'):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_webhook_key(self, webhook_url: str) -> str:
        """Génère une clé unique pour un webhook."""
        return hashlib.md5(webhook_url.encode()).hexdigest()[:16]
    
    async def _init_rate_limiter(self, tenant_id: str, webhook_key: str, rate_limit: int):
        """Initialise le rate limiter pour un webhook."""
        if tenant_id not in self.rate_limiters:
            self.rate_limiters[tenant_id] = {}
        
        self.rate_limiters[tenant_id][webhook_key] = {
            'limit': rate_limit,
            'current': 0,
            'reset_time': datetime.utcnow() + timedelta(minutes=1),
            'blocked_until': None
        }
    
    async def _init_circuit_breaker(self, tenant_id: str, webhook_key: str):
        """Initialise le circuit breaker pour un webhook."""
        if tenant_id not in self.circuit_breakers:
            self.circuit_breakers[tenant_id] = {}
        
        self.circuit_breakers[tenant_id][webhook_key] = {
            'state': 'closed',  # closed, open, half-open
            'failure_count': 0,
            'failure_threshold': 5,
            'recovery_timeout': 300,  # 5 minutes
            'last_failure': None,
            'next_attempt': None
        }
    
    async def send_webhook(self,
                          tenant_id: str,
                          webhook_url: str,
                          payload: Dict[str, Any],
                          severity: SlackSeverity = SlackSeverity.INFO,
                          priority: bool = False) -> str:
        """
        Envoie un webhook Slack de manière asynchrone.
        
        Args:
            tenant_id: ID du tenant
            webhook_url: URL du webhook
            payload: Données à envoyer
            severity: Niveau de sévérité
            priority: Si True, traite en priorité
            
        Returns:
            ID de la requête
        """
        try:
            # Créer la requête
            request = WebhookRequest(
                tenant_id=tenant_id,
                webhook_url=webhook_url,
                payload=payload,
                severity=severity,
                max_retries=3,
                timeout=self.default_timeout
            )
            
            # Ajouter à la queue appropriée
            if priority or severity in [SlackSeverity.CRITICAL, SlackSeverity.HIGH]:
                queue = self.priority_queues[SlackSeverity.CRITICAL]
            else:
                queue = self.priority_queues[severity]
            
            # Vérifier la taille de la queue
            if queue.qsize() > 1000:
                self.metrics['queue_overflow'] += 1
                logger.warning(f"Queue overflow pour {severity.value}")
                
                # En cas de débordement, traiter immédiatement les requêtes critiques
                if severity == SlackSeverity.CRITICAL:
                    await self._process_webhook_request(request)
                    return request.id
            
            await queue.put(request)
            
            # Mettre à jour les métriques
            webhook_queue_size.labels(tenant_id=tenant_id).set(queue.qsize())
            
            logger.debug(f"Webhook {request.id} ajouté à la queue {severity.value}")
            return request.id
            
        except Exception as e:
            logger.error(f"Erreur envoi webhook: {e}")
            raise
    
    async def _process_webhook_request(self, request: WebhookRequest):
        """Traite une requête webhook."""
        start_time = datetime.utcnow()
        
        try:
            # Vérifier le circuit breaker
            if not await self._check_circuit_breaker(request.tenant_id, request.webhook_url):
                request.status = SlackNotificationStatus.FAILED
                request.error_message = "Circuit breaker ouvert"
                self.metrics['circuit_breaker_opened'] += 1
                return
            
            # Vérifier le rate limiting
            if not await self._check_rate_limit(request.tenant_id, request.webhook_url):
                request.status = SlackNotificationStatus.FAILED
                request.error_message = "Rate limit dépassé"
                self.metrics['rate_limited'] += 1
                return
            
            # Traiter la requête avec retry
            response = await self._send_webhook_with_retry(request)
            
            # Traiter la réponse
            if response.success:
                request.status = SlackNotificationStatus.SENT
                await self._record_success(request.tenant_id, request.webhook_url)
                self.metrics['requests_sent'] += 1
            else:
                request.status = SlackNotificationStatus.FAILED
                await self._record_failure(request.tenant_id, request.webhook_url)
                self.metrics['requests_failed'] += 1
            
            request.response_code = response.status_code
            request.response_time = response.response_time
            
        except Exception as e:
            request.status = SlackNotificationStatus.FAILED
            request.error_message = str(e)
            await self._record_failure(request.tenant_id, request.webhook_url)
            self.metrics['requests_failed'] += 1
            logger.error(f"Erreur traitement webhook {request.id}: {e}")
        
        finally:
            # Enregistrer les métriques
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            webhook_requests_total.labels(
                tenant_id=request.tenant_id,
                severity=request.severity.value,
                status=request.status.name.lower()
            ).inc()
            
            webhook_duration_seconds.labels(
                tenant_id=request.tenant_id,
                severity=request.severity.value
            ).observe(duration)
            
            # Persister le résultat si Redis disponible
            if self.redis_client:
                await self._persist_request_result(request)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _send_webhook_with_retry(self, request: WebhookRequest) -> WebhookResponse:
        """Envoie un webhook avec retry automatique."""
        async with self.semaphore:
            start_time = datetime.utcnow()
            
            try:
                # Préparer les headers
                headers = {
                    'Content-Type': 'application/json',
                    'User-Agent': 'Spotify-AI-Agent/2.1.0',
                    **request.headers
                }
                
                # Ajouter la signature si configurée
                webhook_config = await self._get_webhook_config(request.tenant_id, request.webhook_url)
                if webhook_config and webhook_config.signing_secret:
                    timestamp = str(int(datetime.utcnow().timestamp()))
                    signature = self._calculate_signature(
                        webhook_config.signing_secret,
                        timestamp,
                        json.dumps(request.payload)
                    )
                    headers['X-Slack-Request-Timestamp'] = timestamp
                    headers['X-Slack-Signature'] = signature
                
                # Envoyer la requête
                async with self.session.post(
                    request.webhook_url,
                    json=request.payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    
                    response_body = await response.text()
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    return WebhookResponse(
                        request_id=request.id,
                        status_code=response.status,
                        response_body=response_body,
                        headers=dict(response.headers),
                        response_time=response_time
                    )
                    
            except Exception as e:
                request.retry_count += 1
                self.metrics['requests_retried'] += 1
                logger.warning(f"Retry {request.retry_count} pour webhook {request.id}: {e}")
                raise
    
    def _calculate_signature(self, signing_secret: str, timestamp: str, body: str) -> str:
        """Calcule la signature Slack pour la vérification."""
        sig_basestring = f"v0:{timestamp}:{body}"
        signature = hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"v0={signature}"
    
    async def _check_circuit_breaker(self, tenant_id: str, webhook_url: str) -> bool:
        """Vérifie l'état du circuit breaker."""
        webhook_key = self._get_webhook_key(webhook_url)
        
        if tenant_id not in self.circuit_breakers:
            return True
        
        if webhook_key not in self.circuit_breakers[tenant_id]:
            return True
        
        breaker = self.circuit_breakers[tenant_id][webhook_key]
        now = datetime.utcnow()
        
        if breaker['state'] == 'open':
            # Vérifier si on peut passer en half-open
            if breaker['next_attempt'] and now >= breaker['next_attempt']:
                breaker['state'] = 'half-open'
                logger.info(f"Circuit breaker half-open: {tenant_id}/{webhook_key}")
                return True
            return False
        
        return True
    
    async def _check_rate_limit(self, tenant_id: str, webhook_url: str) -> bool:
        """Vérifie le rate limiting."""
        webhook_key = self._get_webhook_key(webhook_url)
        
        if tenant_id not in self.rate_limiters:
            return True
        
        if webhook_key not in self.rate_limiters[tenant_id]:
            return True
        
        limiter = self.rate_limiters[tenant_id][webhook_key]
        now = datetime.utcnow()
        
        # Vérifier si on est bloqué
        if limiter['blocked_until'] and now < limiter['blocked_until']:
            return False
        
        # Réinitialiser si nécessaire
        if now >= limiter['reset_time']:
            limiter['current'] = 0
            limiter['reset_time'] = now + timedelta(minutes=1)
            limiter['blocked_until'] = None
        
        # Vérifier la limite
        if limiter['current'] >= limiter['limit']:
            # Bloquer jusqu'à la prochaine fenêtre
            limiter['blocked_until'] = limiter['reset_time']
            return False
        
        limiter['current'] += 1
        return True
    
    async def _record_success(self, tenant_id: str, webhook_url: str):
        """Enregistre un succès pour le circuit breaker."""
        webhook_key = self._get_webhook_key(webhook_url)
        
        if tenant_id in self.circuit_breakers and webhook_key in self.circuit_breakers[tenant_id]:
            breaker = self.circuit_breakers[tenant_id][webhook_key]
            breaker['failure_count'] = 0
            
            if breaker['state'] == 'half-open':
                breaker['state'] = 'closed'
                logger.info(f"Circuit breaker fermé: {tenant_id}/{webhook_key}")
    
    async def _record_failure(self, tenant_id: str, webhook_url: str):
        """Enregistre un échec pour le circuit breaker."""
        webhook_key = self._get_webhook_key(webhook_url)
        
        if tenant_id not in self.circuit_breakers:
            await self._init_circuit_breaker(tenant_id, webhook_key)
        
        if webhook_key not in self.circuit_breakers[tenant_id]:
            await self._init_circuit_breaker(tenant_id, webhook_key)
        
        breaker = self.circuit_breakers[tenant_id][webhook_key]
        breaker['failure_count'] += 1
        breaker['last_failure'] = datetime.utcnow()
        
        if breaker['failure_count'] >= breaker['failure_threshold']:
            breaker['state'] = 'open'
            breaker['next_attempt'] = datetime.utcnow() + timedelta(seconds=breaker['recovery_timeout'])
            logger.warning(f"Circuit breaker ouvert: {tenant_id}/{webhook_key}")
    
    async def _get_webhook_config(self, tenant_id: str, webhook_url: str) -> Optional[WebhookConfig]:
        """Récupère la configuration d'un webhook."""
        webhook_key = self._get_webhook_key(webhook_url)
        
        if tenant_id in self.webhook_configs and webhook_key in self.webhook_configs[tenant_id]:
            return self.webhook_configs[tenant_id][webhook_key]
        
        return None
    
    async def _cleanup_rate_limiters(self):
        """Nettoie les rate limiters expirés."""
        now = datetime.utcnow()
        
        for tenant_id, webhooks in self.rate_limiters.items():
            for webhook_key, limiter in list(webhooks.items()):
                # Supprimer les limiters inactifs depuis plus d'une heure
                if now > limiter['reset_time'] + timedelta(hours=1):
                    del webhooks[webhook_key]
    
    async def _reset_circuit_breakers(self):
        """Réinitialise les circuit breakers si nécessaire."""
        now = datetime.utcnow()
        
        for tenant_id, webhooks in self.circuit_breakers.items():
            for webhook_key, breaker in webhooks.items():
                # Réinitialiser les breakers ouverts depuis trop longtemps
                if (breaker['state'] == 'open' and 
                    breaker['last_failure'] and 
                    now > breaker['last_failure'] + timedelta(hours=1)):
                    
                    breaker['state'] = 'closed'
                    breaker['failure_count'] = 0
                    logger.info(f"Circuit breaker réinitialisé: {tenant_id}/{webhook_key}")
    
    async def _update_queue_metrics(self):
        """Met à jour les métriques des queues."""
        for severity, queue in self.priority_queues.items():
            webhook_queue_size.labels(tenant_id='global').set(queue.qsize())
    
    async def _persist_webhook_config(self, tenant_id: str, webhook_key: str, config: WebhookConfig):
        """Persiste la configuration webhook en Redis."""
        try:
            if self.redis_client:
                key = f"webhook_config:{tenant_id}:{webhook_key}"
                data = {
                    'url': config.url,
                    'tenant_id': config.tenant_id,
                    'timeout': config.timeout,
                    'max_retries': config.max_retries,
                    'rate_limit': config.rate_limit,
                    'enabled': config.enabled,
                    'created_at': config.created_at.isoformat()
                }
                await self.redis_client.hset(key, mapping=data)
                await self.redis_client.expire(key, 86400)  # 24h
                
        except Exception as e:
            logger.error(f"Erreur persistance config webhook: {e}")
    
    async def _persist_request_result(self, request: WebhookRequest):
        """Persiste le résultat d'une requête en Redis."""
        try:
            if self.redis_client:
                key = f"webhook_result:{request.tenant_id}:{request.id}"
                data = {
                    'id': request.id,
                    'tenant_id': request.tenant_id,
                    'webhook_url': request.webhook_url,
                    'status': request.status.name,
                    'response_code': request.response_code or 0,
                    'response_time': request.response_time or 0,
                    'error_message': request.error_message or '',
                    'created_at': request.created_at.isoformat(),
                    'processed_at': datetime.utcnow().isoformat()
                }
                await self.redis_client.hset(key, mapping=data)
                await self.redis_client.expire(key, 3600)  # 1h
                
        except Exception as e:
            logger.error(f"Erreur persistance résultat: {e}")
    
    async def get_webhook_status(self, tenant_id: str, request_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'une requête webhook."""
        try:
            if self.redis_client:
                key = f"webhook_result:{tenant_id}:{request_id}"
                result = await self.redis_client.hgetall(key)
                
                if result:
                    return {k.decode(): v.decode() for k, v in result.items()}
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération statut webhook: {e}")
            return None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du gestionnaire."""
        queue_sizes = {
            severity.value: queue.qsize()
            for severity, queue in self.priority_queues.items()
        }
        
        return {
            **self.metrics,
            'queue_sizes': queue_sizes,
            'total_webhooks_registered': sum(
                len(webhooks) for webhooks in self.webhook_configs.values()
            ),
            'active_rate_limiters': sum(
                len(limiters) for limiters in self.rate_limiters.values()
            ),
            'circuit_breakers_open': sum(
                1 for webhooks in self.circuit_breakers.values()
                for breaker in webhooks.values()
                if breaker['state'] == 'open'
            ),
            'session_active': self.session is not None and not self.session.closed,
            'workers_running': self.workers_running
        }
    
    def __repr__(self) -> str:
        return f"SlackWebhookManager(concurrent_limit={self.max_concurrent_requests}, rate_limit={self.rate_limit_per_minute})"
