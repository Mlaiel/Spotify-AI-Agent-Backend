"""
🧪 Tests Microservices Framework - Distributed Architecture
==========================================================

Tests complets du framework microservices avec:
- Service Discovery (Consul/etcd/K8s)
- Load Balancing intelligent
- Service Mesh avec mTLS
- Message Broker (RabbitMQ/Kafka)
- Circuit Breaker distribué

Développé par: Microservices Architect
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

from backend.app.frameworks.microservices import (
    MicroservicesFramework,
    ServiceRegistry,
    LoadBalancer,
    ServiceMesh,
    MessageBroker,
    ServiceConfig,
    ServiceType,
    ServiceHealth,
    LoadBalancingStrategy,
    MessagePattern,
    CircuitBreakerConfig,
    ServiceDiscoveryBackend
)
from backend.app.frameworks import TEST_CONFIG, clean_frameworks, logger


@pytest.fixture
def sample_service_config():
    """Configuration service d'exemple."""
    return ServiceConfig(
        name="spotify-recommendation-service",
        service_type=ServiceType.WEB_API,
        host="localhost",
        port=8001,
        health_check_path="/health",
        tags=["ml", "recommendation", "v1"],
        metadata={
            "version": "1.0.0",
            "environment": "test",
            "region": "us-east-1"
        }
    )


@pytest.fixture
def sample_service_instances():
    """Instances de service d'exemple."""
    return [
        ServiceConfig("service-1", ServiceType.WEB_API, "192.168.1.10", 8001),
        ServiceConfig("service-1", ServiceType.WEB_API, "192.168.1.11", 8001),
        ServiceConfig("service-1", ServiceType.WEB_API, "192.168.1.12", 8001)
    ]


@pytest.fixture
def mock_consul_client():
    """Mock client Consul."""
    client = Mock()
    client.agent = Mock()
    client.health = Mock()
    client.kv = Mock()
    
    # Mock responses
    client.agent.service.register.return_value = True
    client.agent.service.deregister.return_value = True
    client.health.service.return_value = (None, [
        {
            'Service': {
                'ID': 'service-1-instance-1',
                'Service': 'service-1',
                'Address': '192.168.1.10',
                'Port': 8001,
                'Tags': ['v1', 'ml']
            },
            'Checks': [{'Status': 'passing'}]
        }
    ])
    
    return client


@pytest.fixture
def mock_etcd_client():
    """Mock client etcd."""
    client = AsyncMock()
    
    # Mock responses
    client.put.return_value = True
    client.get.return_value = (b'{"host": "192.168.1.10", "port": 8001}', Mock())
    client.delete.return_value = True
    client.watch_prefix.return_value = AsyncMock()
    
    return client


@pytest.mark.microservices
class TestServiceConfig:
    """Tests de la configuration service."""
    
    def test_service_config_creation(self):
        """Test création configuration service."""
        config = ServiceConfig(
            name="test-service",
            service_type=ServiceType.WEB_API,
            host="localhost",
            port=8080
        )
        
        assert config.name == "test-service"
        assert config.service_type == ServiceType.WEB_API
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.health_check_path == "/health"
        assert config.tags == []
        
    def test_service_config_validation(self):
        """Test validation configuration service."""
        # Port invalide
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            ServiceConfig("test", ServiceType.WEB_API, "localhost", 0)
            
        # Host vide
        with pytest.raises(ValueError, match="Host cannot be empty"):
            ServiceConfig("test", ServiceType.WEB_API, "", 8080)
            
    def test_service_config_url_generation(self):
        """Test génération URL service."""
        config = ServiceConfig(
            name="api-service",
            service_type=ServiceType.WEB_API,
            host="api.example.com",
            port=443,
            use_tls=True
        )
        
        url = config.get_base_url()
        assert url == "https://api.example.com:443"
        
        health_url = config.get_health_check_url()
        assert health_url == "https://api.example.com:443/health"
        
    def test_service_config_serialization(self, sample_service_config):
        """Test sérialisation configuration."""
        serialized = sample_service_config.to_dict()
        
        assert serialized["name"] == sample_service_config.name
        assert serialized["service_type"] == sample_service_config.service_type.value
        assert serialized["host"] == sample_service_config.host
        assert serialized["port"] == sample_service_config.port
        
        # Désérialisation
        deserialized = ServiceConfig.from_dict(serialized)
        assert deserialized.name == sample_service_config.name
        assert deserialized.service_type == sample_service_config.service_type


@pytest.mark.microservices
class TestServiceRegistry:
    """Tests du registre de services."""
    
    @pytest.mark.asyncio
    async def test_service_registry_consul_backend(self, mock_consul_client):
        """Test backend Consul."""
        with patch('consul.Consul', return_value=mock_consul_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.CONSUL)
            await registry.initialize()
            
            assert registry.backend == ServiceDiscoveryBackend.CONSUL
            assert registry.consul_client is not None
            
    @pytest.mark.asyncio
    async def test_service_registry_etcd_backend(self, mock_etcd_client):
        """Test backend etcd."""
        with patch('etcd3.client', return_value=mock_etcd_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.ETCD)
            await registry.initialize()
            
            assert registry.backend == ServiceDiscoveryBackend.ETCD
            assert registry.etcd_client is not None
            
    @pytest.mark.asyncio
    async def test_register_service_consul(self, sample_service_config, mock_consul_client):
        """Test enregistrement service avec Consul."""
        with patch('consul.Consul', return_value=mock_consul_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.CONSUL)
            await registry.initialize()
            
            service_id = await registry.register_service(sample_service_config)
            
            assert service_id is not None
            mock_consul_client.agent.service.register.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_register_service_etcd(self, sample_service_config, mock_etcd_client):
        """Test enregistrement service avec etcd."""
        with patch('etcd3.client', return_value=mock_etcd_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.ETCD)
            await registry.initialize()
            
            service_id = await registry.register_service(sample_service_config)
            
            assert service_id is not None
            mock_etcd_client.put.assert_called()
            
    @pytest.mark.asyncio
    async def test_discover_services_consul(self, mock_consul_client):
        """Test découverte services avec Consul."""
        with patch('consul.Consul', return_value=mock_consul_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.CONSUL)
            await registry.initialize()
            
            services = await registry.discover_services("service-1")
            
            assert len(services) == 1
            assert services[0].host == "192.168.1.10"
            assert services[0].port == 8001
            
    @pytest.mark.asyncio
    async def test_deregister_service(self, sample_service_config, mock_consul_client):
        """Test désenregistrement service."""
        with patch('consul.Consul', return_value=mock_consul_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.CONSUL)
            await registry.initialize()
            
            # Enregistrer puis désenregistrer
            service_id = await registry.register_service(sample_service_config)
            result = await registry.deregister_service(service_id)
            
            assert result is True
            mock_consul_client.agent.service.deregister.assert_called_with(service_id)
            
    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, sample_service_config, mock_consul_client):
        """Test monitoring santé services."""
        with patch('consul.Consul', return_value=mock_consul_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.CONSUL)
            await registry.initialize()
            
            service_id = await registry.register_service(sample_service_config)
            
            # Mock health check response
            with patch('httpx.AsyncClient.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "healthy"}
                mock_get.return_value = mock_response
                
                health = await registry.check_service_health(service_id)
                
            assert health.status == "healthy"
            assert health.response_time is not None
            
    @pytest.mark.asyncio
    async def test_service_watch_changes(self, mock_consul_client):
        """Test surveillance changements services."""
        with patch('consul.Consul', return_value=mock_consul_client):
            registry = ServiceRegistry(ServiceDiscoveryBackend.CONSUL)
            await registry.initialize()
            
            # Mock watch
            changes = []
            
            async def on_service_change(event):
                changes.append(event)
                
            # Simuler changement
            await registry.watch_service_changes("test-service", on_service_change)
            
            # Note: Test simplifié - en réalité, Consul/etcd enverraient des événements


@pytest.mark.microservices
class TestLoadBalancer:
    """Tests du load balancer."""
    
    def test_load_balancer_creation(self, sample_service_instances):
        """Test création load balancer."""
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        assert lb.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert lb.services == {}
        assert lb.service_weights == {}
        
    def test_add_service_instances(self, sample_service_instances):
        """Test ajout instances service."""
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        for instance in sample_service_instances:
            lb.add_service_instance("service-1", instance)
            
        assert len(lb.services["service-1"]) == 3
        
    def test_round_robin_strategy(self, sample_service_instances):
        """Test stratégie round robin."""
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        for instance in sample_service_instances:
            lb.add_service_instance("service-1", instance)
            
        # Sélections successives doivent tourner
        selected1 = lb.select_service_instance("service-1")
        selected2 = lb.select_service_instance("service-1")
        selected3 = lb.select_service_instance("service-1")
        selected4 = lb.select_service_instance("service-1")  # Retour au début
        
        assert selected1.host != selected2.host
        assert selected2.host != selected3.host
        assert selected1.host == selected4.host  # Cycle complet
        
    def test_weighted_round_robin_strategy(self, sample_service_instances):
        """Test stratégie weighted round robin."""
        lb = LoadBalancer(LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
        
        # Ajouter instances avec poids différents
        lb.add_service_instance("service-1", sample_service_instances[0], weight=3)
        lb.add_service_instance("service-1", sample_service_instances[1], weight=1)
        lb.add_service_instance("service-1", sample_service_instances[2], weight=2)
        
        # Compter sélections sur 60 tentatives
        selections = {}
        for _ in range(60):
            selected = lb.select_service_instance("service-1")
            host = selected.host
            selections[host] = selections.get(host, 0) + 1
            
        # Vérifier répartition selon poids (approximative)
        total_weight = 6
        expected_1 = (3 / total_weight) * 60  # 30
        expected_2 = (1 / total_weight) * 60  # 10
        expected_3 = (2 / total_weight) * 60  # 20
        
        # Tolérance de 20%
        assert abs(selections[sample_service_instances[0].host] - expected_1) < 12
        assert abs(selections[sample_service_instances[1].host] - expected_2) < 6
        assert abs(selections[sample_service_instances[2].host] - expected_3) < 8
        
    def test_least_connections_strategy(self, sample_service_instances):
        """Test stratégie least connections."""
        lb = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        for instance in sample_service_instances:
            lb.add_service_instance("service-1", instance)
            
        # Simuler connexions actives
        lb.active_connections = {
            sample_service_instances[0].host: 5,
            sample_service_instances[1].host: 2,
            sample_service_instances[2].host: 8
        }
        
        # Sélection doit choisir l'instance avec le moins de connexions
        selected = lb.select_service_instance("service-1")
        assert selected.host == sample_service_instances[1].host  # 2 connexions
        
    def test_health_aware_load_balancing(self, sample_service_instances):
        """Test load balancing tenant compte de la santé."""
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        for instance in sample_service_instances:
            lb.add_service_instance("service-1", instance)
            
        # Marquer une instance comme non saine
        lb.mark_service_unhealthy("service-1", sample_service_instances[1].host)
        
        # Sélections ne doivent pas inclure l'instance non saine
        for _ in range(10):
            selected = lb.select_service_instance("service-1")
            assert selected.host != sample_service_instances[1].host
            
    def test_sticky_sessions(self, sample_service_instances):
        """Test sessions collantes."""
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN, enable_sticky_sessions=True)
        
        for instance in sample_service_instances:
            lb.add_service_instance("service-1", instance)
            
        # Première sélection pour session
        session_id = "user_session_123"
        selected1 = lb.select_service_instance("service-1", session_id=session_id)
        
        # Sélections suivantes doivent retourner la même instance
        selected2 = lb.select_service_instance("service-1", session_id=session_id)
        selected3 = lb.select_service_instance("service-1", session_id=session_id)
        
        assert selected1.host == selected2.host == selected3.host


@pytest.mark.microservices
class TestServiceMesh:
    """Tests du service mesh."""
    
    @pytest.mark.asyncio
    async def test_service_mesh_creation(self):
        """Test création service mesh."""
        mesh = ServiceMesh()
        await mesh.initialize()
        
        assert mesh.services == {}
        assert mesh.traffic_policies == {}
        assert mesh.security_policies == {}
        
    @pytest.mark.asyncio
    async def test_service_registration_in_mesh(self, sample_service_config):
        """Test enregistrement service dans mesh."""
        mesh = ServiceMesh()
        await mesh.initialize()
        
        await mesh.register_service(sample_service_config)
        
        assert sample_service_config.name in mesh.services
        
    @pytest.mark.asyncio
    async def test_mtls_configuration(self, sample_service_config):
        """Test configuration mTLS."""
        mesh = ServiceMesh()
        await mesh.initialize()
        
        await mesh.register_service(sample_service_config)
        
        # Configurer mTLS
        await mesh.configure_mtls(
            service_name=sample_service_config.name,
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem",
            ca_path="/path/to/ca.pem"
        )
        
        policy = mesh.security_policies[sample_service_config.name]
        assert policy["mtls_enabled"] is True
        assert policy["cert_path"] == "/path/to/cert.pem"
        
    @pytest.mark.asyncio
    async def test_traffic_routing_rules(self, sample_service_config):
        """Test règles de routage trafic."""
        mesh = ServiceMesh()
        await mesh.initialize()
        
        await mesh.register_service(sample_service_config)
        
        # Configurer règles de routage
        routing_rules = {
            "weight_based": [
                {"destination": "v1", "weight": 80},
                {"destination": "v2", "weight": 20}
            ],
            "header_based": {
                "x-version": "beta",
                "destination": "v2"
            }
        }
        
        await mesh.configure_traffic_routing(sample_service_config.name, routing_rules)
        
        policy = mesh.traffic_policies[sample_service_config.name]
        assert "weight_based" in policy
        assert "header_based" in policy
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_in_mesh(self, sample_service_config):
        """Test circuit breaker dans mesh."""
        mesh = ServiceMesh()
        await mesh.initialize()
        
        await mesh.register_service(sample_service_config)
        
        # Configurer circuit breaker
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_duration=30,
            max_requests=10
        )
        
        await mesh.configure_circuit_breaker(sample_service_config.name, circuit_config)
        
        policy = mesh.traffic_policies[sample_service_config.name]
        assert "circuit_breaker" in policy
        assert policy["circuit_breaker"]["failure_threshold"] == 5
        
    @pytest.mark.asyncio
    async def test_service_mesh_observability(self, sample_service_config):
        """Test observabilité service mesh."""
        mesh = ServiceMesh()
        await mesh.initialize()
        
        await mesh.register_service(sample_service_config)
        
        # Enregistrer métrique trafic
        await mesh.record_request_metrics(
            source_service="api-gateway",
            destination_service=sample_service_config.name,
            response_code=200,
            duration=0.25
        )
        
        # Récupérer métriques
        metrics = mesh.get_service_metrics(sample_service_config.name)
        
        assert metrics["total_requests"] == 1
        assert metrics["success_rate"] == 100.0
        assert metrics["avg_response_time"] == 0.25


@pytest.mark.microservices
class TestMessageBroker:
    """Tests du message broker."""
    
    @pytest.mark.asyncio
    async def test_message_broker_rabbitmq(self):
        """Test message broker RabbitMQ."""
        # Mock connexion RabbitMQ
        with patch('aio_pika.connect_robust') as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            
            broker = MessageBroker("amqp://localhost")
            await broker.initialize()
            
            assert broker.connection is not None
            assert broker.channel is not None
            
    @pytest.mark.asyncio
    async def test_publish_message(self):
        """Test publication message."""
        with patch('aio_pika.connect_robust') as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            
            broker = MessageBroker("amqp://localhost")
            await broker.initialize()
            
            # Publier message
            message = {
                "user_id": "user_123",
                "event": "playlist_created",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await broker.publish_message(
                exchange="spotify_events",
                routing_key="playlist.created",
                message=message
            )
            
            mock_exchange.publish.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_subscribe_to_messages(self):
        """Test abonnement aux messages."""
        with patch('aio_pika.connect_robust') as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_queue = AsyncMock()
            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_queue.return_value = mock_queue
            
            broker = MessageBroker("amqp://localhost")
            await broker.initialize()
            
            # Handler de message
            messages_received = []
            
            async def message_handler(message_body):
                messages_received.append(message_body)
                
            # S'abonner
            await broker.subscribe_to_queue(
                queue_name="recommendation_queue",
                handler=message_handler
            )
            
            mock_queue.consume.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_request_response_pattern(self):
        """Test pattern request/response."""
        with patch('aio_pika.connect_robust') as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            
            broker = MessageBroker("amqp://localhost")
            await broker.initialize()
            
            # Mock réponse
            mock_response = {"result": "recommendation_data"}
            
            with patch.object(broker, '_wait_for_response', return_value=mock_response):
                request_data = {"user_id": "user_123", "limit": 10}
                
                response = await broker.send_request(
                    queue="ml_service_queue",
                    message=request_data,
                    timeout=30
                )
                
            assert response == mock_response
            
    @pytest.mark.asyncio
    async def test_message_patterns(self):
        """Test différents patterns de messagerie."""
        with patch('aio_pika.connect_robust'):
            broker = MessageBroker("amqp://localhost")
            await broker.initialize()
            
            # Test pattern pub/sub
            await broker.setup_pubsub_pattern(
                topic="user_events",
                subscribers=["analytics_service", "notification_service"]
            )
            
            # Test pattern work queue
            await broker.setup_work_queue_pattern(
                queue="image_processing",
                workers=["worker1", "worker2", "worker3"]
            )
            
            # Test pattern RPC
            await broker.setup_rpc_pattern(
                service="recommendation_service"
            )
            
            # Vérifier que les patterns sont configurés
            assert "user_events" in broker.pubsub_topics
            assert "image_processing" in broker.work_queues
            assert "recommendation_service" in broker.rpc_services


@pytest.mark.microservices
class TestMicroservicesFramework:
    """Tests du framework microservices complet."""
    
    @pytest.mark.asyncio
    async def test_microservices_framework_initialization(self, clean_frameworks):
        """Test initialisation framework microservices."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            result = await framework.initialize()
            
        assert result is True
        assert framework.status.name == "RUNNING"
        assert framework.service_registry is not None
        assert framework.load_balancer is not None
        assert framework.service_mesh is not None
        assert framework.message_broker is not None
        
    @pytest.mark.asyncio
    async def test_register_service_full_workflow(self, sample_service_config, clean_frameworks):
        """Test workflow complet enregistrement service."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul') as mock_consul, patch('aio_pika.connect_robust'):
            mock_client = Mock()
            mock_consul.return_value = mock_client
            mock_client.agent.service.register.return_value = True
            
            await framework.initialize()
            
            # Enregistrer service
            service_id = await framework.register_service(sample_service_config)
            
            assert service_id is not None
            
            # Vérifier que le service est dans tous les composants
            assert sample_service_config.name in framework.service_mesh.services
            assert len(framework.load_balancer.services.get(sample_service_config.name, [])) > 0
            
    @pytest.mark.asyncio
    async def test_service_call_with_load_balancing(self, sample_service_instances, clean_frameworks):
        """Test appel service avec load balancing."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            # Ajouter instances
            for instance in sample_service_instances:
                framework.load_balancer.add_service_instance("service-1", instance)
                
            # Mock réponse HTTP
            with patch('httpx.AsyncClient.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"result": "success"}
                mock_request.return_value = mock_response
                
                response = await framework.call_service(
                    service_name="service-1",
                    path="/api/data",
                    method="GET",
                    params={"user_id": "123"}
                )
                
            assert response.status_code == 200
            assert response.json() == {"result": "success"}
            
    @pytest.mark.asyncio
    async def test_message_publishing_and_consumption(self, clean_frameworks):
        """Test publication et consommation messages."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            # Mock publication
            with patch.object(framework.message_broker, 'publish_message') as mock_publish:
                await framework.publish_event(
                    event_type="user_registered",
                    data={"user_id": "new_user_123", "email": "user@example.com"}
                )
                
                mock_publish.assert_called_once()
                
            # Mock consommation
            messages_handled = []
            
            async def event_handler(event_data):
                messages_handled.append(event_data)
                
            with patch.object(framework.message_broker, 'subscribe_to_queue'):
                await framework.subscribe_to_events("user_events", event_handler)
                
    @pytest.mark.asyncio
    async def test_service_health_monitoring_integration(self, sample_service_config, clean_frameworks):
        """Test intégration monitoring santé services."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            service_id = await framework.register_service(sample_service_config)
            
            # Mock health check
            with patch('httpx.AsyncClient.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "healthy"}
                mock_get.return_value = mock_response
                
                health_status = await framework.check_service_health(service_id)
                
            assert health_status.status == "healthy"
            
    @pytest.mark.asyncio
    async def test_microservices_framework_health_check(self, clean_frameworks):
        """Test health check framework microservices."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            health = await framework.health_check()
            
            assert health.status.name == "RUNNING"
            assert "Microservices framework" in health.message
            assert "service_registry" in health.details
            assert "load_balancer" in health.details
            assert "service_mesh" in health.details
            assert "message_broker" in health.details


@pytest.mark.microservices
@pytest.mark.integration
class TestMicroservicesFrameworkIntegration:
    """Tests d'intégration framework microservices."""
    
    @pytest.mark.asyncio
    async def test_full_microservices_scenario(self, clean_frameworks):
        """Test scénario microservices complet."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            # 1. Enregistrer plusieurs services
            services = [
                ServiceConfig("user-service", ServiceType.WEB_API, "localhost", 8001),
                ServiceConfig("playlist-service", ServiceType.WEB_API, "localhost", 8002),
                ServiceConfig("recommendation-service", ServiceType.WEB_API, "localhost", 8003)
            ]
            
            service_ids = []
            for service_config in services:
                with patch.object(framework.service_registry, 'register_service', return_value=f"{service_config.name}-id"):
                    service_id = await framework.register_service(service_config)
                    service_ids.append(service_id)
                    
            # 2. Configurer load balancing
            for service_config in services:
                framework.load_balancer.add_service_instance(service_config.name, service_config)
                
            # 3. Appel inter-services
            with patch('httpx.AsyncClient.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"user_id": "123", "preferences": ["rock", "pop"]}
                mock_request.return_value = mock_response
                
                user_data = await framework.call_service(
                    "user-service",
                    "/api/users/123",
                    "GET"
                )
                
                assert user_data.json()["user_id"] == "123"
                
            # 4. Publication événement
            with patch.object(framework.message_broker, 'publish_message'):
                await framework.publish_event(
                    "playlist_created",
                    {"user_id": "123", "playlist_id": "playlist_456"}
                )
                
            # 5. Vérifier santé de tous les services
            all_healthy = True
            for service_id in service_ids:
                with patch.object(framework.service_registry, 'check_service_health') as mock_health:
                    mock_health.return_value = ServiceHealth("healthy", 0.05)
                    health = await framework.check_service_health(service_id)
                    if health.status != "healthy":
                        all_healthy = False
                        
            assert all_healthy is True
            
    @pytest.mark.asyncio
    async def test_service_mesh_traffic_management(self, clean_frameworks):
        """Test gestion trafic service mesh."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            # Enregistrer service avec plusieurs versions
            service_v1 = ServiceConfig("api-service", ServiceType.WEB_API, "localhost", 8001, tags=["v1"])
            service_v2 = ServiceConfig("api-service", ServiceType.WEB_API, "localhost", 8002, tags=["v2"])
            
            await framework.register_service(service_v1)
            await framework.register_service(service_v2)
            
            # Configurer répartition trafic 80/20
            await framework.service_mesh.configure_traffic_routing(
                "api-service",
                {
                    "weight_based": [
                        {"destination": "v1", "weight": 80},
                        {"destination": "v2", "weight": 20}
                    ]
                }
            )
            
            # Simuler 100 appels et vérifier répartition
            v1_calls = 0
            v2_calls = 0
            
            with patch('httpx.AsyncClient.request') as mock_request:
                def side_effect(*args, **kwargs):
                    nonlocal v1_calls, v2_calls
                    url = kwargs.get('url', args[1] if len(args) > 1 else '')
                    if '8001' in str(url):
                        v1_calls += 1
                    elif '8002' in str(url):
                        v2_calls += 1
                    
                    mock_response = Mock()
                    mock_response.status_code = 200
                    return mock_response
                    
                mock_request.side_effect = side_effect
                
                # Simuler logique de routage (simplifié)
                for _ in range(100):
                    # 80% vers v1, 20% vers v2
                    import random
                    if random.randint(1, 100) <= 80:
                        await framework.call_service("api-service", "/test", "GET", version="v1")
                    else:
                        await framework.call_service("api-service", "/test", "GET", version="v2")
                        
            # Vérifier répartition approximative (tolérance 15%)
            assert abs(v1_calls - 80) < 15
            assert abs(v2_calls - 20) < 15


@pytest.mark.microservices
@pytest.mark.performance
class TestMicroservicesFrameworkPerformance:
    """Tests de performance framework microservices."""
    
    @pytest.mark.asyncio
    async def test_high_volume_service_calls(self, sample_service_instances, clean_frameworks):
        """Test appels services haut volume."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            # Ajouter instances
            for instance in sample_service_instances:
                framework.load_balancer.add_service_instance("high-volume-service", instance)
                
            # Mock réponses rapides
            with patch('httpx.AsyncClient.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "ok"}
                mock_request.return_value = mock_response
                
                # Lancer 1000 appels concurrents
                async def make_call():
                    return await framework.call_service(
                        "high-volume-service",
                        "/api/fast-endpoint",
                        "GET"
                    )
                    
                start_time = time.time()
                tasks = [make_call() for _ in range(1000)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                # Vérifier performance
                duration = end_time - start_time
                successful_calls = len([r for r in results if not isinstance(r, Exception)])
                
                assert duration < 10.0  # Moins de 10 secondes pour 1000 appels
                assert successful_calls == 1000  # Tous les appels réussissent
                
    @pytest.mark.asyncio
    async def test_message_throughput(self, clean_frameworks):
        """Test débit messages."""
        framework = MicroservicesFramework()
        
        with patch('consul.Consul'), patch('aio_pika.connect_robust'):
            await framework.initialize()
            
            # Mock publication rapide
            with patch.object(framework.message_broker, 'publish_message') as mock_publish:
                mock_publish.return_value = True
                
                # Publier 10000 messages
                start_time = time.time()
                
                tasks = []
                for i in range(10000):
                    task = framework.publish_event(
                        f"test_event_{i % 10}",
                        {"message_id": i, "data": f"test_data_{i}"}
                    )
                    tasks.append(task)
                    
                await asyncio.gather(*tasks)
                end_time = time.time()
                
                duration = end_time - start_time
                throughput = 10000 / duration
                
                # Débit doit être > 1000 messages/seconde
                assert throughput > 1000
                assert mock_publish.call_count == 10000
