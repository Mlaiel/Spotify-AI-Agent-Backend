"""
🌐 MICROSERVICES FRAMEWORK - ARCHITECTURE DISTRIBUÉE ENTERPRISE
Expert Team: Microservices Architect, DBA & Data Engineer

Architecture microservices complète avec service mesh, découverte et orchestration
"""

import asyncio
import os
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

# Service Discovery et Configuration
import consul
import etcd3
from kubernetes import client, config as k8s_config

# Circuit Breaker et Resilience
from circuitbreaker import circuit
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Load Balancing
import httpx
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry

# Message Queue et Event Streaming
import aio_pika
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import redis.asyncio as aioredis

# Service Mesh (Istio simulation)
from grpcio import server as grpc_server
import grpc

# Base framework
from .core import BaseFramework, FrameworkStatus, FrameworkHealth
from .core import framework_orchestrator

# Configuration
from pydantic import BaseSettings, Field


class ServiceType(Enum):
    """Types de services"""
    WEB_API = "web_api"
    WORKER = "worker"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    GATEWAY = "gateway"
    AUTH = "auth"
    MONITORING = "monitoring"


class LoadBalancingStrategy(Enum):
    """Stratégies de load balancing"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"
    RANDOM = "random"


class ServiceDiscoveryBackend(Enum):
    """Backends de service discovery"""
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"
    REDIS = "redis"


@dataclass
class ServiceConfig:
    """Configuration d'un microservice"""
    name: str
    service_type: ServiceType
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8000
    health_check_path: str = "/health"
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Scaling
    min_instances: int = 1
    max_instances: int = 10
    cpu_limit: str = "500m"
    memory_limit: str = "512Mi"
    
    # Circuit Breaker
    failure_threshold: int = 5
    recovery_timeout: int = 60
    timeout: float = 30.0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    def service_id(self) -> str:
        """Génère un ID unique pour le service"""
        return f"{self.name}-{uuid.uuid4().hex[:8]}"


@dataclass
class ServiceInstance:
    """Instance d'un microservice"""
    service_id: str
    config: ServiceConfig
    status: str = "starting"
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Vérifie si l'instance est en bonne santé"""
        if not self.last_health_check:
            return False
        
        # Health check dans les 2 dernières minutes
        return (
            datetime.utcnow() - self.last_health_check < timedelta(minutes=2)
            and self.health_status == "healthy"
        )


class ServiceRegistry:
    """
    📋 REGISTRE DE SERVICES
    
    Gestion centralisée des services:
    - Enregistrement/désenregistrement
    - Découverte de services
    - Health monitoring
    - Load balancing
    """
    
    def __init__(self, backend: ServiceDiscoveryBackend = ServiceDiscoveryBackend.CONSUL):
        self.backend = backend
        self.services: Dict[str, ServiceInstance] = {}
        self.logger = logging.getLogger("microservices.registry")
        
        # Clients pour les backends
        self.consul_client: Optional[consul.Consul] = None
        self.etcd_client: Optional[etcd3.Etcd3Client] = None
        self.k8s_client: Optional[Any] = None
        self.redis_client: Optional[Any] = None
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialise le backend de service discovery"""
        try:
            if self.backend == ServiceDiscoveryBackend.CONSUL:
                self.consul_client = consul.Consul(
                    host=os.getenv('CONSUL_HOST', 'localhost'),
                    port=int(os.getenv('CONSUL_PORT', '8500'))
                )
            elif self.backend == ServiceDiscoveryBackend.ETCD:
                self.etcd_client = etcd3.client(
                    host=os.getenv('ETCD_HOST', 'localhost'),
                    port=int(os.getenv('ETCD_PORT', '2379'))
                )
            elif self.backend == ServiceDiscoveryBackend.KUBERNETES:
                k8s_config.load_incluster_config()
                self.k8s_client = client.CoreV1Api()
            elif self.backend == ServiceDiscoveryBackend.REDIS:
                self.redis_client = aioredis.from_url(
                    os.getenv('REDIS_URL', 'redis://localhost:6379')
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.backend.value} backend: {e}")
    
    async def register_service(self, config: ServiceConfig) -> str:
        """Enregistre un service"""
        try:
            service_id = config.service_id()
            instance = ServiceInstance(
                service_id=service_id,
                config=config,
                status="registered"
            )
            
            self.services[service_id] = instance
            
            # Enregistrer dans le backend
            await self._register_in_backend(instance)
            
            self.logger.info(f"Service registered: {config.name} ({service_id})")
            return service_id
            
        except Exception as e:
            self.logger.error(f"Service registration failed: {e}")
            raise
    
    async def _register_in_backend(self, instance: ServiceInstance):
        """Enregistre dans le backend de service discovery"""
        if self.backend == ServiceDiscoveryBackend.CONSUL:
            await self._register_consul(instance)
        elif self.backend == ServiceDiscoveryBackend.ETCD:
            await self._register_etcd(instance)
        elif self.backend == ServiceDiscoveryBackend.KUBERNETES:
            await self._register_k8s(instance)
        elif self.backend == ServiceDiscoveryBackend.REDIS:
            await self._register_redis(instance)
    
    async def _register_consul(self, instance: ServiceInstance):
        """Enregistre dans Consul"""
        try:
            self.consul_client.agent.service.register(
                name=instance.config.name,
                service_id=instance.service_id,
                address=instance.config.host,
                port=instance.config.port,
                tags=instance.config.tags,
                meta=instance.config.metadata,
                check=consul.Check.http(
                    f"http://{instance.config.host}:{instance.config.port}{instance.config.health_check_path}",
                    interval="10s",
                    timeout="5s"
                )
            )
        except Exception as e:
            self.logger.error(f"Consul registration failed: {e}")
            raise
    
    async def _register_etcd(self, instance: ServiceInstance):
        """Enregistre dans etcd"""
        try:
            service_key = f"/services/{instance.config.name}/{instance.service_id}"
            service_data = {
                "host": instance.config.host,
                "port": instance.config.port,
                "version": instance.config.version,
                "metadata": instance.config.metadata
            }
            
            self.etcd_client.put(service_key, json.dumps(service_data))
        except Exception as e:
            self.logger.error(f"etcd registration failed: {e}")
            raise
    
    async def _register_k8s(self, instance: ServiceInstance):
        """Enregistre dans Kubernetes"""
        # Simulation d'enregistrement K8s
        # En réalité, les services K8s sont gérés par des manifests
        pass
    
    async def _register_redis(self, instance: ServiceInstance):
        """Enregistre dans Redis"""
        try:
            service_key = f"services:{instance.config.name}:{instance.service_id}"
            service_data = {
                "host": instance.config.host,
                "port": instance.config.port,
                "version": instance.config.version,
                "status": instance.status,
                "registered_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.hset(service_key, mapping=service_data)
            await self.redis_client.expire(service_key, 300)  # TTL 5 minutes
        except Exception as e:
            self.logger.error(f"Redis registration failed: {e}")
            raise
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Découvre les instances d'un service"""
        try:
            if self.backend == ServiceDiscoveryBackend.CONSUL:
                return await self._discover_consul(service_name)
            elif self.backend == ServiceDiscoveryBackend.ETCD:
                return await self._discover_etcd(service_name)
            elif self.backend == ServiceDiscoveryBackend.KUBERNETES:
                return await self._discover_k8s(service_name)
            elif self.backend == ServiceDiscoveryBackend.REDIS:
                return await self._discover_redis(service_name)
            else:
                return []
        except Exception as e:
            self.logger.error(f"Service discovery failed for {service_name}: {e}")
            return []
    
    async def _discover_consul(self, service_name: str) -> List[ServiceInstance]:
        """Découvre via Consul"""
        try:
            _, services = self.consul_client.health.service(service_name, passing=True)
            
            instances = []
            for service in services:
                service_info = service['Service']
                config = ServiceConfig(
                    name=service_info['Service'],
                    service_type=ServiceType.WEB_API,  # Default
                    host=service_info['Address'],
                    port=service_info['Port'],
                    tags=service_info.get('Tags', []),
                    metadata=service_info.get('Meta', {})
                )
                
                instance = ServiceInstance(
                    service_id=service_info['ID'],
                    config=config,
                    status="running",
                    health_status="healthy"
                )
                instances.append(instance)
            
            return instances
        except Exception as e:
            self.logger.error(f"Consul discovery failed: {e}")
            return []
    
    async def _discover_etcd(self, service_name: str) -> List[ServiceInstance]:
        """Découvre via etcd"""
        try:
            services = []
            prefix = f"/services/{service_name}/"
            
            for value, metadata in self.etcd_client.get_prefix(prefix):
                service_data = json.loads(value.decode())
                
                config = ServiceConfig(
                    name=service_name,
                    service_type=ServiceType.WEB_API,
                    host=service_data['host'],
                    port=service_data['port'],
                    version=service_data.get('version', '1.0.0'),
                    metadata=service_data.get('metadata', {})
                )
                
                service_id = metadata.key.decode().split('/')[-1]
                instance = ServiceInstance(
                    service_id=service_id,
                    config=config,
                    status="running"
                )
                services.append(instance)
            
            return services
        except Exception as e:
            self.logger.error(f"etcd discovery failed: {e}")
            return []
    
    async def _discover_k8s(self, service_name: str) -> List[ServiceInstance]:
        """Découvre via Kubernetes"""
        # Simulation de découverte K8s
        return []
    
    async def _discover_redis(self, service_name: str) -> List[ServiceInstance]:
        """Découvre via Redis"""
        try:
            pattern = f"services:{service_name}:*"
            keys = await self.redis_client.keys(pattern)
            
            instances = []
            for key in keys:
                service_data = await self.redis_client.hgetall(key)
                
                if service_data:
                    config = ServiceConfig(
                        name=service_name,
                        service_type=ServiceType.WEB_API,
                        host=service_data.get('host', ''),
                        port=int(service_data.get('port', 8000)),
                        version=service_data.get('version', '1.0.0')
                    )
                    
                    service_id = key.decode().split(':')[-1]
                    instance = ServiceInstance(
                        service_id=service_id,
                        config=config,
                        status=service_data.get('status', 'unknown')
                    )
                    instances.append(instance)
            
            return instances
        except Exception as e:
            self.logger.error(f"Redis discovery failed: {e}")
            return []
    
    async def unregister_service(self, service_id: str):
        """Désenregistre un service"""
        try:
            if service_id in self.services:
                instance = self.services[service_id]
                
                # Désenregistrer du backend
                await self._unregister_from_backend(instance)
                
                # Supprimer localement
                del self.services[service_id]
                
                self.logger.info(f"Service unregistered: {service_id}")
        except Exception as e:
            self.logger.error(f"Service unregistration failed: {e}")
    
    async def _unregister_from_backend(self, instance: ServiceInstance):
        """Désenregistre du backend"""
        if self.backend == ServiceDiscoveryBackend.CONSUL:
            self.consul_client.agent.service.deregister(instance.service_id)
        elif self.backend == ServiceDiscoveryBackend.ETCD:
            service_key = f"/services/{instance.config.name}/{instance.service_id}"
            self.etcd_client.delete(service_key)
        elif self.backend == ServiceDiscoveryBackend.REDIS:
            service_key = f"services:{instance.config.name}:{instance.service_id}"
            await self.redis_client.delete(service_key)


class LoadBalancer:
    """
    ⚖️ LOAD BALANCER INTELLIGENT
    
    Distribution intelligente des requêtes:
    - Algorithmes de load balancing
    - Health-aware routing
    - Circuit breaker intégré
    - Métriques de performance
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.logger = logging.getLogger("microservices.loadbalancer")
        
        # État pour round robin
        self._round_robin_index = 0
        
        # Métriques de connexions
        self._connection_counts: Dict[str, int] = {}
        
        # Poids pour weighted round robin
        self._service_weights: Dict[str, int] = {}
    
    async def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Sélectionne une instance selon la stratégie"""
        if not instances:
            return None
        
        # Filtrer les instances saines
        healthy_instances = [inst for inst in instances if inst.is_healthy()]
        
        if not healthy_instances:
            self.logger.warning("No healthy instances available")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_select(healthy_instances)
        else:
            return healthy_instances[0]
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Sélection round robin"""
        instance = instances[self._round_robin_index % len(instances)]
        self._round_robin_index += 1
        return instance
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Sélection par moins de connexions"""
        min_connections = float('inf')
        selected_instance = instances[0]
        
        for instance in instances:
            connections = self._connection_counts.get(instance.service_id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance
        
        return selected_instance
    
    def _weighted_round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Sélection weighted round robin"""
        # Implémentation simplifiée
        # En réalité, il faudrait maintenir des compteurs par poids
        return self._round_robin_select(instances)
    
    def _health_based_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Sélection basée sur la santé"""
        # Trier par health score (simulé)
        instances_sorted = sorted(
            instances,
            key=lambda x: x.metrics.get('health_score', 1.0),
            reverse=True
        )
        return instances_sorted[0]
    
    def _random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Sélection aléatoire"""
        import random
        return random.choice(instances)
    
    def increment_connections(self, service_id: str):
        """Incrémente le compteur de connexions"""
        self._connection_counts[service_id] = self._connection_counts.get(service_id, 0) + 1
    
    def decrement_connections(self, service_id: str):
        """Décrémente le compteur de connexions"""
        if service_id in self._connection_counts:
            self._connection_counts[service_id] = max(0, self._connection_counts[service_id] - 1)


class ServiceMesh:
    """
    🕸️ SERVICE MESH
    
    Communication inter-services sécurisée:
    - Proxy sidecar simulation
    - mTLS entre services
    - Traffic policies
    - Observabilité
    """
    
    def __init__(self):
        self.logger = logging.getLogger("microservices.servicemesh")
        
        # Policies de trafic
        self.traffic_policies: Dict[str, Dict[str, Any]] = {}
        
        # Certificats pour mTLS (simulation)
        self.service_certificates: Dict[str, Dict[str, str]] = {}
    
    async def setup_service_proxy(self, service_name: str, config: Dict[str, Any]):
        """Configure un proxy sidecar pour un service"""
        try:
            # Configuration du proxy (simulation Envoy/Istio)
            proxy_config = {
                "service_name": service_name,
                "upstream_services": config.get("upstream_services", []),
                "load_balancing": config.get("load_balancing", "round_robin"),
                "circuit_breaker": config.get("circuit_breaker", {}),
                "retry_policy": config.get("retry_policy", {}),
                "timeout": config.get("timeout", 30)
            }
            
            self.logger.info(f"Service proxy configured for {service_name}")
            return proxy_config
            
        except Exception as e:
            self.logger.error(f"Service proxy setup failed for {service_name}: {e}")
            raise
    
    async def setup_mtls(self, service_name: str) -> Dict[str, str]:
        """Configure mTLS pour un service"""
        try:
            # Génération de certificats (simulation)
            certificates = {
                "ca_cert": f"ca-cert-for-{service_name}",
                "service_cert": f"service-cert-for-{service_name}",
                "service_key": f"service-key-for-{service_name}"
            }
            
            self.service_certificates[service_name] = certificates
            
            self.logger.info(f"mTLS configured for {service_name}")
            return certificates
            
        except Exception as e:
            self.logger.error(f"mTLS setup failed for {service_name}: {e}")
            raise
    
    def set_traffic_policy(self, service_name: str, policy: Dict[str, Any]):
        """Définit une politique de trafic"""
        self.traffic_policies[service_name] = policy
        self.logger.info(f"Traffic policy set for {service_name}")
    
    async def route_request(
        self, 
        from_service: str, 
        to_service: str, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route une requête selon les policies"""
        try:
            # Vérifier les policies
            policy = self.traffic_policies.get(to_service, {})
            
            # Appliquer les rules de routing
            if "version_routing" in policy:
                version_rules = policy["version_routing"]
                # Logique de routing par version
            
            # Appliquer le circuit breaker
            if "circuit_breaker" in policy:
                cb_config = policy["circuit_breaker"]
                # Logique de circuit breaker
            
            # Simulation d'envoi de requête
            response = {
                "status": "success",
                "from_service": from_service,
                "to_service": to_service,
                "routed_at": datetime.utcnow().isoformat(),
                "data": request_data
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Request routing failed: {e}")
            raise


class MessageBroker:
    """
    📨 MESSAGE BROKER
    
    Communication asynchrone entre services:
    - RabbitMQ integration
    - Kafka integration
    - Redis Streams
    - Event sourcing
    """
    
    def __init__(self, broker_type: str = "rabbitmq"):
        self.broker_type = broker_type
        self.logger = logging.getLogger("microservices.messagebroker")
        
        # Connections
        self.rabbitmq_connection: Optional[Any] = None
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None
        self.redis_client: Optional[Any] = None
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    async def initialize(self):
        """Initialise le message broker"""
        try:
            if self.broker_type == "rabbitmq":
                await self._init_rabbitmq()
            elif self.broker_type == "kafka":
                await self._init_kafka()
            elif self.broker_type == "redis":
                await self._init_redis()
            
            self.logger.info(f"Message broker initialized: {self.broker_type}")
            
        except Exception as e:
            self.logger.error(f"Message broker initialization failed: {e}")
            raise
    
    async def _init_rabbitmq(self):
        """Initialise RabbitMQ"""
        rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')
        self.rabbitmq_connection = await aio_pika.connect_robust(rabbitmq_url)
    
    async def _init_kafka(self):
        """Initialise Kafka"""
        kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode()
        )
        await self.kafka_producer.start()
        
        self.kafka_consumer = AIOKafkaConsumer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode())
        )
        await self.kafka_consumer.start()
    
    async def _init_redis(self):
        """Initialise Redis Streams"""
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = aioredis.from_url(redis_url)
    
    async def publish_event(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        routing_key: str = ""
    ):
        """Publie un événement"""
        try:
            event = {
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "event_id": str(uuid.uuid4())
            }
            
            if self.broker_type == "rabbitmq":
                await self._publish_rabbitmq(event, routing_key)
            elif self.broker_type == "kafka":
                await self._publish_kafka(event, event_type)
            elif self.broker_type == "redis":
                await self._publish_redis(event, event_type)
            
            self.logger.info(f"Event published: {event_type}")
            
        except Exception as e:
            self.logger.error(f"Event publishing failed: {e}")
            raise
    
    async def _publish_rabbitmq(self, event: Dict[str, Any], routing_key: str):
        """Publie sur RabbitMQ"""
        if not self.rabbitmq_connection:
            raise RuntimeError("RabbitMQ not initialized")
        
        channel = await self.rabbitmq_connection.channel()
        
        # Déclarer l'exchange
        exchange = await channel.declare_exchange(
            "microservices_events",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        # Publier le message
        await exchange.publish(
            aio_pika.Message(
                json.dumps(event).encode(),
                content_type="application/json"
            ),
            routing_key=routing_key or event["event_type"]
        )
        
        await channel.close()
    
    async def _publish_kafka(self, event: Dict[str, Any], topic: str):
        """Publie sur Kafka"""
        if not self.kafka_producer:
            raise RuntimeError("Kafka producer not initialized")
        
        await self.kafka_producer.send(topic, event)
    
    async def _publish_redis(self, event: Dict[str, Any], stream: str):
        """Publie sur Redis Streams"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        stream_name = f"events:{stream}"
        await self.redis_client.xadd(stream_name, event)
    
    def subscribe_to_event(self, event_type: str, handler: Callable):
        """S'abonne à un type d'événement"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Handler subscribed to event: {event_type}")
    
    async def start_consuming(self):
        """Démarre la consommation d'événements"""
        try:
            if self.broker_type == "rabbitmq":
                await self._consume_rabbitmq()
            elif self.broker_type == "kafka":
                await self._consume_kafka()
            elif self.broker_type == "redis":
                await self._consume_redis()
                
        except Exception as e:
            self.logger.error(f"Event consumption failed: {e}")
    
    async def _consume_rabbitmq(self):
        """Consomme depuis RabbitMQ"""
        if not self.rabbitmq_connection:
            return
        
        channel = await self.rabbitmq_connection.channel()
        exchange = await channel.declare_exchange(
            "microservices_events",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        queue = await channel.declare_queue("microservice_consumer", durable=True)
        
        # Bind toutes les routes d'événements
        for event_type in self.event_handlers.keys():
            await queue.bind(exchange, routing_key=event_type)
        
        async def process_message(message):
            async with message.process():
                event = json.loads(message.body.decode())
                await self._handle_event(event)
        
        await queue.consume(process_message)
    
    async def _consume_kafka(self):
        """Consomme depuis Kafka"""
        if not self.kafka_consumer:
            return
        
        # S'abonner aux topics
        topics = list(self.event_handlers.keys())
        self.kafka_consumer.subscribe(topics)
        
        async for message in self.kafka_consumer:
            event = message.value
            await self._handle_event(event)
    
    async def _consume_redis(self):
        """Consomme depuis Redis Streams"""
        if not self.redis_client:
            return
        
        streams = {f"events:{event_type}": "$" for event_type in self.event_handlers.keys()}
        
        while True:
            try:
                messages = await self.redis_client.xread(streams, block=1000)
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        event = {k.decode(): v.decode() for k, v in fields.items()}
                        await self._handle_event(event)
                        
            except Exception as e:
                self.logger.error(f"Redis consumption error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Traite un événement reçu"""
        event_type = event.get("event_type")
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler failed for {event_type}: {e}")


class MicroserviceFramework(BaseFramework):
    """
    🌐 FRAMEWORK MICROSERVICES ENTERPRISE
    
    Orchestration complète des microservices avec:
    - Service discovery automatique
    - Load balancing intelligent
    - Service mesh intégré
    - Communication asynchrone
    - Resilience patterns
    """
    
    def __init__(self):
        super().__init__("microservices", {})
        
        # Composants
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer()
        self.service_mesh = ServiceMesh()
        self.message_broker = MessageBroker()
        
        # Services enregistrés
        self.managed_services: Dict[str, ServiceInstance] = {}
        
    async def initialize(self) -> bool:
        """Initialise le framework microservices"""
        try:
            # Initialiser le message broker
            await self.message_broker.initialize()
            
            # Enregistrer les services par défaut
            await self._register_default_services()
            
            # Démarrer la consommation d'événements
            asyncio.create_task(self.message_broker.start_consuming())
            
            self.logger.info("Microservices Framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Microservices framework initialization failed: {e}")
            return False
    
    async def _register_default_services(self):
        """Enregistre les services par défaut"""
        # Service principal API
        api_config = ServiceConfig(
            name="spotify-ai-api",
            service_type=ServiceType.WEB_API,
            host="localhost",
            port=8000,
            tags=["api", "main"],
            dependencies=["spotify-ai-db", "spotify-ai-cache"]
        )
        await self.register_service(api_config)
        
        # Service de base de données
        db_config = ServiceConfig(
            name="spotify-ai-db",
            service_type=ServiceType.DATABASE,
            host="localhost",
            port=5432,
            tags=["database", "postgresql"]
        )
        await self.register_service(db_config)
        
        # Service de cache
        cache_config = ServiceConfig(
            name="spotify-ai-cache",
            service_type=ServiceType.CACHE,
            host="localhost",
            port=6379,
            tags=["cache", "redis"]
        )
        await self.register_service(cache_config)
    
    async def register_service(self, config: ServiceConfig) -> str:
        """Enregistre un nouveau service"""
        try:
            service_id = await self.service_registry.register_service(config)
            
            # Configurer le service mesh
            await self.service_mesh.setup_service_proxy(
                config.name,
                {
                    "upstream_services": config.dependencies,
                    "circuit_breaker": {
                        "failure_threshold": config.failure_threshold,
                        "recovery_timeout": config.recovery_timeout
                    },
                    "timeout": config.timeout
                }
            )
            
            # Configurer mTLS
            await self.service_mesh.setup_mtls(config.name)
            
            # Publier l'événement d'enregistrement
            await self.message_broker.publish_event(
                "service.registered",
                {
                    "service_id": service_id,
                    "service_name": config.name,
                    "service_type": config.service_type.value,
                    "host": config.host,
                    "port": config.port
                }
            )
            
            return service_id
            
        except Exception as e:
            self.logger.error(f"Service registration failed: {e}")
            raise
    
    async def call_service(
        self, 
        service_name: str, 
        endpoint: str, 
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        from_service: str = "unknown"
    ) -> Dict[str, Any]:
        """Appelle un service avec load balancing et resilience"""
        try:
            # Découvrir les instances du service
            instances = await self.service_registry.discover_services(service_name)
            
            if not instances:
                raise RuntimeError(f"No instances found for service: {service_name}")
            
            # Sélectionner une instance via load balancer
            instance = await self.load_balancer.select_instance(instances)
            
            if not instance:
                raise RuntimeError(f"No healthy instances for service: {service_name}")
            
            # Incrémenter le compteur de connexions
            self.load_balancer.increment_connections(instance.service_id)
            
            try:
                # Router la requête via service mesh
                mesh_response = await self.service_mesh.route_request(
                    from_service,
                    service_name,
                    {"endpoint": endpoint, "method": method, "data": data}
                )
                
                # Effectuer l'appel HTTP (simulation)
                url = f"http://{instance.config.host}:{instance.config.port}{endpoint}"
                
                async with httpx.AsyncClient() as client:
                    if method.upper() == "GET":
                        response = await client.get(url, timeout=instance.config.timeout)
                    elif method.upper() == "POST":
                        response = await client.post(url, json=data, timeout=instance.config.timeout)
                    elif method.upper() == "PUT":
                        response = await client.put(url, json=data, timeout=instance.config.timeout)
                    elif method.upper() == "DELETE":
                        response = await client.delete(url, timeout=instance.config.timeout)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Publier l'événement d'appel de service
                await self.message_broker.publish_event(
                    "service.called",
                    {
                        "from_service": from_service,
                        "to_service": service_name,
                        "endpoint": endpoint,
                        "method": method,
                        "status_code": response.status_code,
                        "instance_id": instance.service_id
                    }
                )
                
                return {
                    "status_code": response.status_code,
                    "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "instance_id": instance.service_id,
                    "service_mesh_response": mesh_response
                }
                
            finally:
                # Décrémenter le compteur de connexions
                self.load_balancer.decrement_connections(instance.service_id)
            
        except Exception as e:
            # Publier l'événement d'erreur
            await self.message_broker.publish_event(
                "service.call_failed",
                {
                    "from_service": from_service,
                    "to_service": service_name,
                    "endpoint": endpoint,
                    "error": str(e)
                }
            )
            
            self.logger.error(f"Service call failed: {e}")
            raise
    
    async def shutdown(self) -> bool:
        """Arrête le framework microservices"""
        try:
            # Désenregistrer tous les services
            for service_id in list(self.managed_services.keys()):
                await self.service_registry.unregister_service(service_id)
            
            # Fermer les connections du message broker
            if self.message_broker.rabbitmq_connection:
                await self.message_broker.rabbitmq_connection.close()
            
            if self.message_broker.kafka_producer:
                await self.message_broker.kafka_producer.stop()
            
            if self.message_broker.kafka_consumer:
                await self.message_broker.kafka_consumer.stop()
            
            if self.message_broker.redis_client:
                await self.message_broker.redis_client.close()
            
            self.logger.info("Microservices Framework shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Microservices framework shutdown failed: {e}")
            return False
    
    async def health_check(self) -> FrameworkHealth:
        """Vérifie la santé du framework microservices"""
        health = FrameworkHealth(
            status=FrameworkStatus.RUNNING,
            last_check=time.time()
        )
        
        try:
            # Vérifier les services enregistrés
            total_services = len(self.managed_services)
            healthy_services = sum(
                1 for service in self.managed_services.values() 
                if service.is_healthy()
            )
            
            health.metadata = {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "service_registry_backend": self.service_registry.backend.value,
                "message_broker_type": self.message_broker.broker_type,
                "load_balancing_strategy": self.load_balancer.strategy.value
            }
            
            if healthy_services < total_services * 0.8:  # 80% de services sains minimum
                health.status = FrameworkStatus.DEGRADED
            
        except Exception as e:
            health.status = FrameworkStatus.DEGRADED
            health.error_count += 1
            health.metadata["error"] = str(e)
        
        return health


# Instance globale du framework microservices
microservice_manager = MicroserviceFramework()


# Export des classes principales
__all__ = [
    'MicroserviceFramework',
    'ServiceRegistry',
    'LoadBalancer',
    'ServiceMesh',
    'MessageBroker',
    'ServiceConfig',
    'ServiceInstance',
    'ServiceType',
    'LoadBalancingStrategy',
    'ServiceDiscoveryBackend',
    'microservice_manager'
]
