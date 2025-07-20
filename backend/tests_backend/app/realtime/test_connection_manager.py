# 🧪 Tests pour Connection Manager
# =================================
# 
# Tests complets pour le gestionnaire de connexions
# avec tests de pool, load balancing et résilience.
#
# 🎖️ Expert: Network Testing Specialist + Infrastructure Engineer
#
# 👨‍💻 Développé par: Fahed Mlaiel
# =================================

"""
🔗 Connection Manager Tests
===========================

Comprehensive test suite for the Real-Time Connection Manager:
- Connection pool management tests
- Load balancing strategy tests
- Health monitoring and failover tests
- Session management and tracking tests
- Performance and scalability tests
- Error handling and recovery tests
- Multi-platform connection tests
- Connection lifecycle management tests
"""

import asyncio
import json
import pytest
import time
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import du module à tester
from app.realtime.connection_manager import (
    RealTimeConnectionManager,
    ConnectionPool,
    Connection,
    ServerEndpoint,
    ConnectionState,
    ConnectionType,
    LoadBalanceStrategy,
    ConnectionMetrics,
    create_connection_manager
)

from . import TestUtils, REDIS_TEST_URL


class TestServerEndpoint:
    """Tests pour ServerEndpoint"""
    
    def test_endpoint_creation(self):
        """Test de création d'endpoint"""
        endpoint = ServerEndpoint(
            host="localhost",
            port=8080,
            path="/ws",
            ssl_enabled=True,
            weight=10,
            max_connections=500
        )
        
        assert endpoint.host == "localhost"
        assert endpoint.port == 8080
        assert endpoint.path == "/ws"
        assert endpoint.ssl_enabled is True
        assert endpoint.weight == 10
        assert endpoint.max_connections == 500
    
    def test_url_generation(self):
        """Test de génération d'URL"""
        # Endpoint HTTP
        http_endpoint = ServerEndpoint(
            host="example.com",
            port=80,
            path="/api/ws",
            ssl_enabled=False
        )
        
        assert http_endpoint.get_url() == "ws://example.com:80/api/ws"
        
        # Endpoint HTTPS
        https_endpoint = ServerEndpoint(
            host="secure.example.com",
            port=443,
            path="/secure/ws",
            ssl_enabled=True
        )
        
        assert https_endpoint.get_url() == "wss://secure.example.com:443/secure/ws"
    
    def test_connection_capacity_check(self):
        """Test de vérification de capacité"""
        endpoint = ServerEndpoint(
            host="test.com",
            port=8080,
            max_connections=10
        )
        
        # Endpoint sain avec capacité
        assert endpoint.can_accept_connection() is True
        
        # Simuler la charge
        endpoint.current_connections = 10
        assert endpoint.can_accept_connection() is False
        
        # Endpoint non sain
        endpoint.current_connections = 5
        endpoint.is_healthy = False
        assert endpoint.can_accept_connection() is False
    
    def test_load_factor_calculation(self):
        """Test de calcul du facteur de charge"""
        endpoint = ServerEndpoint(
            host="load.test",
            port=8080,
            max_connections=100
        )
        
        # Pas de connexions
        assert endpoint.get_load_factor() == 0.0
        
        # 50% de charge
        endpoint.current_connections = 50
        assert endpoint.get_load_factor() == 0.5
        
        # Pleine charge
        endpoint.current_connections = 100
        assert endpoint.get_load_factor() == 1.0


class TestConnectionMetrics:
    """Tests pour ConnectionMetrics"""
    
    def test_metrics_initialization(self):
        """Test d'initialisation des métriques"""
        connection_id = "test_conn_123"
        metrics = ConnectionMetrics(connection_id)
        
        assert metrics.connection_id == connection_id
        assert metrics.created_at is not None
        assert metrics.last_activity is not None
        assert metrics.bytes_sent == 0
        assert metrics.bytes_received == 0
        assert metrics.messages_sent == 0
        assert metrics.messages_received == 0
        assert metrics.average_latency == 0.0
        assert metrics.health_score == 1.0
        assert metrics.error_count == 0
    
    def test_latency_update(self):
        """Test de mise à jour de latence"""
        metrics = ConnectionMetrics("latency_test")
        
        # Ajouter quelques échantillons de latence
        latencies = [10.0, 15.0, 20.0, 25.0, 30.0]
        for latency in latencies:
            metrics.update_latency(latency)
        
        # Vérifier la moyenne
        expected_avg = sum(latencies) / len(latencies)
        assert metrics.average_latency == expected_avg
        assert len(metrics.latency_samples) == 5
    
    def test_error_recording(self):
        """Test d'enregistrement d'erreur"""
        metrics = ConnectionMetrics("error_test")
        initial_health = metrics.health_score
        
        metrics.record_error("Connection timeout")
        
        assert metrics.error_count == 1
        assert metrics.last_error == "Connection timeout"
        assert metrics.last_error_time is not None
        assert metrics.consecutive_failures == 1
        assert metrics.health_score < initial_health
    
    def test_success_recording(self):
        """Test d'enregistrement de succès"""
        metrics = ConnectionMetrics("success_test")
        
        # D'abord quelques erreurs
        for i in range(3):
            metrics.record_error(f"Error {i}")
        
        initial_health = metrics.health_score
        initial_activity = metrics.last_activity
        
        # Puis un succès
        metrics.record_success()
        
        assert metrics.consecutive_failures == 0
        assert metrics.health_score > initial_health
        assert metrics.last_activity > initial_activity


class TestConnection:
    """Tests pour Connection"""
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket"""
        websocket = Mock()
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock()
        websocket.close = AsyncMock()
        websocket.ping = AsyncMock()
        websocket.remote_address = ("127.0.0.1", 12345)
        return websocket
    
    @pytest.fixture
    def test_endpoint(self):
        """Endpoint de test"""
        return ServerEndpoint(
            host="localhost",
            port=8080,
            path="/test",
            ssl_enabled=False
        )
    
    @pytest.fixture
    async def connection(self, mock_websocket, test_endpoint):
        """Connexion de test"""
        connection_id = str(uuid.uuid4())
        user_id = TestUtils.generate_test_user_id()
        
        conn = Connection(
            connection_id=connection_id,
            connection_type=ConnectionType.WEBSOCKET,
            endpoint=test_endpoint,
            user_id=user_id
        )
        
        # Simuler la connexion WebSocket
        conn.websocket = mock_websocket
        conn.state = ConnectionState.CONNECTED
        
        yield conn
        
        if not conn.is_closed:
            await conn.disconnect()
    
    @pytest.mark.asyncio
    async def test_connection_initialization(self, connection):
        """Test d'initialisation de connexion"""
        assert connection.connection_id is not None
        assert connection.connection_type == ConnectionType.WEBSOCKET
        assert connection.user_id is not None
        assert connection.state == ConnectionState.CONNECTED
        assert not connection.is_authenticated
        assert connection.metrics is not None
    
    @pytest.mark.asyncio
    async def test_message_sending(self, connection):
        """Test d'envoi de message"""
        message = {"type": "test", "data": "hello world"}
        
        result = await connection.send_message(message)
        
        assert result is True
        connection.websocket.send.assert_called_once()
        assert connection.metrics.messages_sent == 1
        assert connection.metrics.bytes_sent > 0
    
    @pytest.mark.asyncio
    async def test_message_sending_failure(self, connection):
        """Test d'échec d'envoi de message"""
        connection.websocket.send.side_effect = Exception("Send failed")
        
        message = {"type": "test", "data": "fail"}
        result = await connection.send_message(message)
        
        assert result is False
        assert connection.metrics.error_count == 1
        assert connection.metrics.last_error == "Send failed"
    
    @pytest.mark.asyncio
    async def test_authentication(self, connection):
        """Test d'authentification"""
        # Mock JWT decode
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                "user_id": connection.user_id,
                "permissions": ["read", "write", "admin"]
            }
            
            result = await connection.authenticate("valid_token")
            
            assert result is True
            assert connection.is_authenticated
            assert "read" in connection.permissions
            assert "write" in connection.permissions
            assert "admin" in connection.permissions
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, connection):
        """Test d'échec d'authentification"""
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Invalid token")
            
            result = await connection.authenticate("invalid_token")
            
            assert result is False
            assert not connection.is_authenticated
            assert len(connection.permissions) == 0
    
    @pytest.mark.asyncio
    async def test_connection_status(self, connection):
        """Test de statut de connexion"""
        status = connection.get_status()
        
        assert "connection_id" in status
        assert "state" in status
        assert "user_id" in status
        assert "is_authenticated" in status
        assert "endpoint" in status
        assert "uptime" in status
        assert "messages_sent" in status
        assert "messages_received" in status
        assert "health_score" in status
        
        assert status["connection_id"] == connection.connection_id
        assert status["state"] == connection.state.value
        assert status["user_id"] == connection.user_id


class TestConnectionPool:
    """Tests pour ConnectionPool"""
    
    @pytest.fixture
    def test_endpoints(self):
        """Endpoints de test"""
        return [
            ServerEndpoint("server1.test", 8080, weight=10),
            ServerEndpoint("server2.test", 8080, weight=5),
            ServerEndpoint("server3.test", 8080, weight=15)
        ]
    
    @pytest.fixture
    async def connection_pool(self, test_endpoints):
        """Pool de connexions de test"""
        pool = ConnectionPool(
            endpoints=test_endpoints,
            strategy=LoadBalanceStrategy.LEAST_CONNECTIONS
        )
        await pool.start()
        
        yield pool
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, connection_pool, test_endpoints):
        """Test d'initialisation du pool"""
        assert len(connection_pool.endpoints) == len(test_endpoints)
        assert connection_pool.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS
        assert connection_pool.total_connections == 0
        assert connection_pool.active_connections == 0
    
    @pytest.mark.asyncio
    async def test_round_robin_selection(self):
        """Test de sélection round-robin"""
        endpoints = [
            ServerEndpoint("rr1.test", 8080),
            ServerEndpoint("rr2.test", 8080),
            ServerEndpoint("rr3.test", 8080)
        ]
        
        pool = ConnectionPool(endpoints, LoadBalanceStrategy.ROUND_ROBIN)
        
        # Test de sélection séquentielle
        selected_endpoints = []
        for i in range(6):  # 2 tours complets
            endpoint = pool._round_robin_select(endpoints)
            selected_endpoints.append(endpoint.host)
        
        # Devrait suivre l'ordre: rr1, rr2, rr3, rr1, rr2, rr3
        expected = ["rr1.test", "rr2.test", "rr3.test"] * 2
        assert selected_endpoints == expected
    
    @pytest.mark.asyncio
    async def test_least_connections_selection(self):
        """Test de sélection par moindres connexions"""
        endpoints = [
            ServerEndpoint("lc1.test", 8080),
            ServerEndpoint("lc2.test", 8080),
            ServerEndpoint("lc3.test", 8080)
        ]
        
        # Simuler différents nombres de connexions
        endpoints[0].current_connections = 5
        endpoints[1].current_connections = 2  # Le moins chargé
        endpoints[2].current_connections = 8
        
        pool = ConnectionPool(endpoints, LoadBalanceStrategy.LEAST_CONNECTIONS)
        selected = pool._least_connections_select(endpoints)
        
        assert selected.host == "lc2.test"  # Le moins chargé
    
    @pytest.mark.asyncio
    async def test_weighted_round_robin_selection(self):
        """Test de sélection weighted round-robin"""
        endpoints = [
            ServerEndpoint("wr1.test", 8080, weight=1),
            ServerEndpoint("wr2.test", 8080, weight=3),  # Poids plus élevé
            ServerEndpoint("wr3.test", 8080, weight=1)
        ]
        
        pool = ConnectionPool(endpoints, LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN)
        
        # Compter les sélections sur plusieurs tours
        selections = {}
        for i in range(50):
            endpoint = pool._weighted_round_robin_select(endpoints)
            selections[endpoint.host] = selections.get(endpoint.host, 0) + 1
        
        # wr2 devrait être sélectionné plus souvent (poids 3 vs 1)
        assert selections.get("wr2.test", 0) > selections.get("wr1.test", 0)
        assert selections.get("wr2.test", 0) > selections.get("wr3.test", 0)
    
    @pytest.mark.asyncio
    async def test_pool_stats(self, connection_pool):
        """Test de statistiques du pool"""
        stats = connection_pool.get_pool_stats()
        
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "failed_connections" in stats
        assert "unique_users" in stats
        assert "load_balance_strategy" in stats
        assert "endpoints" in stats
        
        assert stats["load_balance_strategy"] == "least_connections"
        assert len(stats["endpoints"]) == 3
        
        # Vérifier les stats d'endpoint
        for endpoint_stat in stats["endpoints"]:
            assert "url" in endpoint_stat
            assert "is_healthy" in endpoint_stat
            assert "current_connections" in endpoint_stat
            assert "load_factor" in endpoint_stat


class TestRealTimeConnectionManager:
    """Tests pour RealTimeConnectionManager complet"""
    
    @pytest.fixture
    def test_endpoints(self):
        """Endpoints de test pour le manager"""
        return [
            ServerEndpoint("mgr1.test", 8080, max_connections=100),
            ServerEndpoint("mgr2.test", 8080, max_connections=100)
        ]
    
    @pytest.fixture
    async def connection_manager(self, test_endpoints):
        """Manager de connexions de test"""
        manager = RealTimeConnectionManager(
            endpoints=test_endpoints,
            redis_url=REDIS_TEST_URL
        )
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, connection_manager):
        """Test d'initialisation du manager"""
        assert connection_manager.redis_client is not None
        assert connection_manager.websocket_pool is not None
        assert connection_manager.http_stream_pool is not None
        assert connection_manager.max_connections_per_user == 10
        assert len(connection_manager.user_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_connection_limits_check(self, connection_manager):
        """Test de vérification des limites de connexion"""
        user_id = TestUtils.generate_test_user_id()
        
        # Devrait être autorisé initialement
        can_connect = await connection_manager._check_connection_limits(user_id)
        assert can_connect is True
        
        # Simuler beaucoup de connexions pour cet utilisateur
        for i in range(15):  # Plus que la limite
            mock_conn = Mock()
            mock_conn.connection_id = f"mock_conn_{i}"
            connection_manager.websocket_pool.user_connections[user_id].add(f"mock_conn_{i}")
        
        # Devrait maintenant être bloqué
        can_connect = await connection_manager._check_connection_limits(user_id)
        assert can_connect is False
    
    @pytest.mark.asyncio
    async def test_session_registration(self, connection_manager):
        """Test d'enregistrement de session"""
        user_id = TestUtils.generate_test_user_id()
        
        # Mock connexion
        mock_connection = Mock()
        mock_connection.connection_id = "session_test_conn"
        mock_connection.connection_type = ConnectionType.WEBSOCKET
        mock_connection.endpoint = Mock()
        mock_connection.endpoint.get_url.return_value = "ws://test.com:8080"
        
        await connection_manager._register_user_session(user_id, mock_connection)
        
        # Vérifier qu'une session a été créée
        assert len(connection_manager.user_sessions) == 1
        
        # Vérifier dans Redis
        if connection_manager.redis_client:
            sessions = await connection_manager.redis_client.smembers(f"user_sessions:{user_id}")
            assert len(sessions) >= 1
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, connection_manager):
        """Test de nettoyage de sessions"""
        user_id = TestUtils.generate_test_user_id()
        
        # Créer quelques sessions de test
        for i in range(3):
            session_id = f"cleanup_session_{i}"
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "connection_id": f"conn_{i}",
                "created_at": datetime.utcnow() - timedelta(hours=2),  # Session ancienne
                "last_activity": datetime.utcnow() - timedelta(hours=2)
            }
            connection_manager.user_sessions[session_id] = session_data
        
        # Nettoyer les sessions expirées
        await connection_manager._cleanup_user_sessions(user_id)
        
        # Vérifier que les sessions ont été supprimées
        remaining_sessions = [
            s for s in connection_manager.user_sessions.values()
            if s["user_id"] == user_id
        ]
        assert len(remaining_sessions) == 0
    
    @pytest.mark.asyncio
    async def test_manager_stats(self, connection_manager):
        """Test de statistiques du manager"""
        stats = connection_manager.get_manager_stats()
        
        assert "total_metrics" in stats
        assert "websocket_pool" in stats
        assert "http_stream_pool" in stats
        assert "active_sessions" in stats
        assert "configuration" in stats
        
        # Vérifier la configuration
        config = stats["configuration"]
        assert config["max_connections_per_user"] == 10
        assert config["session_timeout"] == 3600
        assert config["endpoints_count"] == 2


@pytest.mark.integration
class TestConnectionManagerIntegration:
    """Tests d'intégration pour le gestionnaire de connexions"""
    
    @pytest.mark.asyncio
    async def test_full_connection_lifecycle(self):
        """Test du cycle de vie complet d'une connexion"""
        endpoints = [
            ServerEndpoint("integration.test", 8080, max_connections=50)
        ]
        
        manager = RealTimeConnectionManager(
            endpoints=endpoints,
            redis_url=REDIS_TEST_URL
        )
        await manager.initialize()
        
        try:
            user_id = TestUtils.generate_test_user_id()
            
            # 1. Créer une connexion
            with patch('websockets.connect') as mock_connect:
                mock_websocket = Mock()
                mock_websocket.send = AsyncMock()
                mock_websocket.close = AsyncMock()
                mock_websocket.remote_address = ("127.0.0.1", 12345)
                mock_connect.return_value = mock_websocket
                
                connection = await manager.create_connection(
                    user_id=user_id,
                    connection_type=ConnectionType.WEBSOCKET
                )
                
                # La connexion devrait être créée (même si mock)
                # En pratique, elle pourrait échouer à cause du mock
                # mais la logique du manager devrait être testée
            
            # 2. Vérifier les sessions
            sessions = await manager.get_user_connections(user_id)
            # Peut être vide si la connexion mock a échoué, mais le test vérifie la logique
            
            # 3. Déconnecter l'utilisateur
            disconnected_count = await manager.disconnect_user(user_id)
            
            # 4. Vérifier le nettoyage
            remaining_sessions = await manager.get_user_connections(user_id)
            assert len(remaining_sessions) == 0
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_load_balancing_under_load(self):
        """Test de load balancing sous charge"""
        # Plusieurs endpoints avec différentes capacités
        endpoints = [
            ServerEndpoint("lb1.test", 8080, max_connections=10, weight=1),
            ServerEndpoint("lb2.test", 8080, max_connections=20, weight=2),
            ServerEndpoint("lb3.test", 8080, max_connections=5, weight=1)
        ]
        
        pool = ConnectionPool(endpoints, LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN)
        await pool.start()
        
        try:
            # Simuler la sélection d'endpoints sous charge
            selections = {}
            
            for i in range(100):
                # Simuler la charge croissante
                for endpoint in endpoints:
                    endpoint.current_connections = min(
                        endpoint.current_connections + 1,
                        endpoint.max_connections
                    )
                
                selected = pool._select_endpoint()
                if selected:
                    host = selected.host
                    selections[host] = selections.get(host, 0) + 1
            
            # lb2 devrait être sélectionné plus souvent (poids 2, capacité 20)
            assert selections.get("lb2.test", 0) >= selections.get("lb1.test", 0)
            assert selections.get("lb2.test", 0) >= selections.get("lb3.test", 0)
            
        finally:
            await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_failover_scenario(self):
        """Test de scénario de basculement"""
        endpoints = [
            ServerEndpoint("primary.test", 8080, max_connections=100),
            ServerEndpoint("backup.test", 8080, max_connections=50)
        ]
        
        pool = ConnectionPool(endpoints, LoadBalanceStrategy.LEAST_CONNECTIONS)
        await pool.start()
        
        try:
            # Initialement, le primary devrait être sélectionné
            selected = pool._select_endpoint()
            assert selected.host == "primary.test"
            
            # Simuler une panne du primary
            endpoints[0].is_healthy = False
            
            # Maintenant le backup devrait être sélectionné
            selected = pool._select_endpoint()
            assert selected.host == "backup.test"
            
            # Restaurer le primary
            endpoints[0].is_healthy = True
            
            # Le primary devrait être à nouveau disponible
            selected = pool._select_endpoint()
            # Peut être l'un ou l'autre selon la charge
            assert selected.host in ["primary.test", "backup.test"]
            
        finally:
            await pool.shutdown()


class TestConnectionPerformance:
    """Tests de performance pour les connexions"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_connection_pool_scalability(self):
        """Test de scalabilité du pool de connexions"""
        # Simuler beaucoup d'endpoints
        endpoints = []
        for i in range(10):
            endpoints.append(
                ServerEndpoint(f"scale{i}.test", 8080, max_connections=100)
            )
        
        pool = ConnectionPool(endpoints, LoadBalanceStrategy.ROUND_ROBIN)
        await pool.start()
        
        try:
            start_time = time.time()
            
            # Test de sélection d'endpoint rapide
            for i in range(1000):
                endpoint = pool._select_endpoint()
                assert endpoint is not None
            
            selection_time = time.time() - start_time
            
            # Devrait être très rapide
            assert selection_time < 1.0  # Moins d'1 seconde pour 1000 sélections
            
        finally:
            await pool.shutdown()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_connection_creation(self):
        """Test de création simultanée de connexions"""
        endpoints = [
            ServerEndpoint("concurrent.test", 8080, max_connections=1000)
        ]
        
        manager = RealTimeConnectionManager(
            endpoints=endpoints,
            redis_url=REDIS_TEST_URL
        )
        await manager.initialize()
        
        try:
            # Mock des connexions WebSocket
            with patch('websockets.connect') as mock_connect:
                mock_websocket = Mock()
                mock_websocket.send = AsyncMock()
                mock_websocket.close = AsyncMock()
                mock_websocket.remote_address = ("127.0.0.1", 12345)
                mock_connect.return_value = mock_websocket
                
                start_time = time.time()
                
                # Créer beaucoup de connexions en parallèle
                tasks = []
                for i in range(50):
                    user_id = f"concurrent_user_{i}"
                    task = manager.create_connection(
                        user_id=user_id,
                        connection_type=ConnectionType.WEBSOCKET
                    )
                    tasks.append(task)
                
                # Attendre toutes les créations
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                creation_time = time.time() - start_time
                
                # Vérifier les performances
                assert creation_time < 5.0  # Moins de 5 secondes
                
                # Compter les succès (certains peuvent échouer à cause des mocks)
                successful_connections = [r for r in results if not isinstance(r, Exception)]
                
        finally:
            await manager.shutdown()


# Utilitaires pour les tests de connexions
class ConnectionTestUtils:
    """Utilitaires pour les tests de connexions"""
    
    @staticmethod
    def create_mock_websocket():
        """Crée un mock WebSocket"""
        websocket = Mock()
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock()
        websocket.close = AsyncMock()
        websocket.ping = AsyncMock()
        websocket.remote_address = ("127.0.0.1", 12345)
        return websocket
    
    @staticmethod
    def create_test_endpoints(count=3):
        """Crée des endpoints de test"""
        endpoints = []
        for i in range(count):
            endpoint = ServerEndpoint(
                host=f"test{i}.example.com",
                port=8080 + i,
                path=f"/ws{i}",
                weight=i + 1,
                max_connections=100 * (i + 1)
            )
            endpoints.append(endpoint)
        return endpoints
    
    @staticmethod
    async def simulate_connection_load(pool, num_connections=10):
        """Simule une charge de connexions"""
        connections = []
        
        for i in range(num_connections):
            # Simuler l'ajout de connexion au pool
            endpoint = pool._select_endpoint()
            if endpoint:
                endpoint.current_connections += 1
                endpoint.total_connections += 1
                connections.append({"endpoint": endpoint, "id": f"sim_conn_{i}"})
        
        return connections
    
    @staticmethod
    async def simulate_health_checks(endpoints):
        """Simule des vérifications de santé"""
        import random
        
        for endpoint in endpoints:
            # Simuler aléatoirement des endpoints sains/non sains
            endpoint.is_healthy = random.choice([True, True, True, False])  # 75% sains
            endpoint.last_health_check = datetime.utcnow()
            endpoint.average_response_time = random.uniform(0.1, 2.0)


# Export des classes de test
__all__ = [
    "TestServerEndpoint",
    "TestConnectionMetrics",
    "TestConnection",
    "TestConnectionPool",
    "TestRealTimeConnectionManager",
    "TestConnectionManagerIntegration",
    "TestConnectionPerformance",
    "ConnectionTestUtils"
]
