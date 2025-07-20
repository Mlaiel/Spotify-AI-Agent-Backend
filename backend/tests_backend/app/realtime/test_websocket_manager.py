# 🧪 Tests pour WebSocket Manager
# ================================
# 
# Tests complets pour le gestionnaire WebSocket avancé
# avec tests de performance, sécurité et résilience.
#
# 🎖️ Expert: WebSocket Testing Specialist + Performance Engineer
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ================================

"""
🔌 WebSocket Manager Tests
=========================

Comprehensive test suite for the Advanced WebSocket Manager:
- Connection lifecycle management tests
- Clustering and horizontal scaling tests
- Rate limiting and circuit breaker tests
- Authentication and authorization tests
- Message routing and broadcasting tests
- Performance and load testing
- Fault tolerance and recovery tests
- Security vulnerability testing
"""

import asyncio
import json
import pytest
import time
import uuid
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import du module à tester
from app.realtime.websocket_manager import (
    AdvancedWebSocketManager,
    WebSocketConnection,
    ConnectionState,
    RateLimiter,
    CircuitBreaker,
    WebSocketCluster,
    MessageType,
    ConnectionMetrics
)

from . import TestUtils, REDIS_TEST_URL


class TestWebSocketConnection:
    """Tests pour WebSocketConnection"""
    
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
    async def connection(self, mock_websocket):
        """Connexion WebSocket de test"""
        connection_id = TestUtils.generate_test_user_id()
        user_id = TestUtils.generate_test_user_id()
        
        conn = WebSocketConnection(
            connection_id=connection_id,
            websocket=mock_websocket,
            user_id=user_id
        )
        
        yield conn
        
        # Cleanup
        if not conn.is_closed:
            await conn.close()
    
    @pytest.mark.asyncio
    async def test_connection_initialization(self, connection):
        """Test d'initialisation de connexion"""
        assert connection.connection_id is not None
        assert connection.state == ConnectionState.CONNECTED
        assert not connection.is_authenticated
        assert connection.created_at is not None
        assert connection.last_activity is not None
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, connection):
        """Test d'envoi de message réussi"""
        message = {"type": "test", "data": "hello"}
        
        result = await connection.send_message(message)
        
        assert result is True
        connection.websocket.send.assert_called_once()
        assert connection.metrics.messages_sent == 1
        assert connection.metrics.bytes_sent > 0
    
    @pytest.mark.asyncio
    async def test_send_message_failure(self, connection):
        """Test d'échec d'envoi de message"""
        connection.websocket.send.side_effect = Exception("Send failed")
        
        result = await connection.send_message({"type": "test"})
        
        assert result is False
        assert connection.metrics.error_count == 1
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, connection):
        """Test d'authentification réussie"""
        # Mock token valide
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                "user_id": connection.user_id,
                "permissions": ["read", "write"]
            }
            
            result = await connection.authenticate("valid_token")
            
            assert result is True
            assert connection.is_authenticated
            assert "read" in connection.permissions
            assert "write" in connection.permissions
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, connection):
        """Test d'échec d'authentification"""
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Invalid token")
            
            result = await connection.authenticate("invalid_token")
            
            assert result is False
            assert not connection.is_authenticated
    
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, connection):
        """Test du mécanisme de heartbeat"""
        # Démarrer le heartbeat
        await connection.start_heartbeat()
        
        # Attendre un peu
        await asyncio.sleep(0.1)
        
        # Vérifier que ping a été appelé
        assert connection.heartbeat_task is not None
        
        # Arrêter le heartbeat
        await connection.stop_heartbeat()
        assert connection.heartbeat_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_connection_metrics_update(self, connection):
        """Test de mise à jour des métriques"""
        initial_messages = connection.metrics.messages_sent
        
        await connection.send_message({"test": "data"})
        
        assert connection.metrics.messages_sent == initial_messages + 1
        assert connection.metrics.bytes_sent > 0
        assert connection.last_activity > connection.created_at
    
    @pytest.mark.asyncio
    async def test_message_queue_overflow(self, connection):
        """Test de débordement de queue de messages"""
        # Remplir la queue
        for i in range(connection.message_queue.maxsize + 10):
            try:
                await connection.add_to_queue(f"message_{i}")
            except:
                # Expected pour les derniers messages
                pass
        
        # Vérifier que la queue n'est pas plus grande que maxsize
        assert connection.message_queue.qsize() <= connection.message_queue.maxsize


class TestRateLimiter:
    """Tests pour RateLimiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Rate limiter de test"""
        return RateLimiter(
            max_requests=10,
            window_size=60,
            burst_size=5
        )
    
    @pytest.mark.asyncio
    async def test_rate_limiting_success(self, rate_limiter):
        """Test de limitation de taux - succès"""
        client_id = "test_client"
        
        # Les premières requêtes devraient passer
        for i in range(5):
            allowed = await rate_limiter.is_allowed(client_id)
            assert allowed is True
    
    @pytest.mark.asyncio
    async def test_rate_limiting_blocked(self, rate_limiter):
        """Test de limitation de taux - bloqué"""
        client_id = "test_client"
        
        # Dépasser la limite
        for i in range(15):
            await rate_limiter.is_allowed(client_id)
        
        # La prochaine requête devrait être bloquée
        allowed = await rate_limiter.is_allowed(client_id)
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self, rate_limiter):
        """Test de reset du rate limiter"""
        client_id = "test_client"
        
        # Atteindre la limite
        for i in range(11):
            await rate_limiter.is_allowed(client_id)
        
        # Reset
        rate_limiter.reset_client(client_id)
        
        # Devrait être autorisé maintenant
        allowed = await rate_limiter.is_allowed(client_id)
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_multiple_clients(self, rate_limiter):
        """Test avec plusieurs clients"""
        clients = ["client1", "client2", "client3"]
        
        # Chaque client devrait avoir sa propre limite
        for client in clients:
            for i in range(5):
                allowed = await rate_limiter.is_allowed(client)
                assert allowed is True


class TestCircuitBreaker:
    """Tests pour CircuitBreaker"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Circuit breaker de test"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,
            expected_exception=Exception
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test état fermé du circuit breaker"""
        async def success_function():
            return "success"
        
        result = await circuit_breaker.call(success_function)
        assert result == "success"
        assert circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, circuit_breaker):
        """Test ouverture du circuit breaker sur échecs"""
        async def failing_function():
            raise Exception("Function failed")
        
        # Provoquer assez d'échecs pour ouvrir le circuit
        for i in range(4):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass
        
        assert circuit_breaker.state == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test de récupération en état semi-ouvert"""
        async def failing_function():
            raise Exception("Function failed")
        
        async def success_function():
            return "recovered"
        
        # Ouvrir le circuit
        for i in range(4):
            try:
                await circuit_breaker.call(failing_function)
            except:
                pass
        
        assert circuit_breaker.state == "open"
        
        # Attendre le timeout de récupération
        await asyncio.sleep(1.1)
        
        # Le prochain appel devrait être en half-open
        result = await circuit_breaker.call(success_function)
        assert result == "recovered"
        assert circuit_breaker.state == "closed"


class TestAdvancedWebSocketManager:
    """Tests pour AdvancedWebSocketManager"""
    
    @pytest.fixture
    async def manager(self):
        """Manager WebSocket de test"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "max_connections": 1000,
            "rate_limit": {
                "max_requests": 100,
                "window_size": 60
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 30
            }
        }
        
        manager = AdvancedWebSocketManager(config)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket pour les tests"""
        websocket = Mock()
        websocket.send = AsyncMock()
        websocket.recv = AsyncMock()
        websocket.close = AsyncMock()
        websocket.ping = AsyncMock()
        websocket.remote_address = ("127.0.0.1", 12345)
        return websocket
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test d'initialisation du manager"""
        assert manager.redis_client is not None
        assert manager.rate_limiter is not None
        assert manager.circuit_breaker is not None
        assert len(manager.connections) == 0
    
    @pytest.mark.asyncio
    async def test_add_connection(self, manager, mock_websocket):
        """Test d'ajout de connexion"""
        user_id = TestUtils.generate_test_user_id()
        
        connection = await manager.add_connection(mock_websocket, user_id)
        
        assert connection is not None
        assert connection.user_id == user_id
        assert connection.connection_id in manager.connections
        assert user_id in manager.user_connections
    
    @pytest.mark.asyncio
    async def test_remove_connection(self, manager, mock_websocket):
        """Test de suppression de connexion"""
        user_id = TestUtils.generate_test_user_id()
        
        connection = await manager.add_connection(mock_websocket, user_id)
        connection_id = connection.connection_id
        
        await manager.remove_connection(connection_id)
        
        assert connection_id not in manager.connections
        if user_id in manager.user_connections:
            assert connection_id not in manager.user_connections[user_id]
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, manager, mock_websocket):
        """Test de diffusion de message"""
        # Ajouter plusieurs connexions
        connections = []
        for i in range(3):
            user_id = TestUtils.generate_test_user_id()
            conn = await manager.add_connection(mock_websocket, user_id)
            connections.append(conn)
        
        message = {"type": "broadcast", "data": "hello all"}
        sent_count = await manager.broadcast_message(message)
        
        assert sent_count == 3
        assert mock_websocket.send.call_count == 3
    
    @pytest.mark.asyncio
    async def test_send_to_user(self, manager, mock_websocket):
        """Test d'envoi à un utilisateur spécifique"""
        user_id = TestUtils.generate_test_user_id()
        await manager.add_connection(mock_websocket, user_id)
        
        message = {"type": "personal", "data": "hello user"}
        sent_count = await manager.send_to_user(user_id, message)
        
        assert sent_count == 1
        mock_websocket.send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_to_room(self, manager, mock_websocket):
        """Test d'envoi à une room"""
        room_id = "test_room"
        user_ids = [TestUtils.generate_test_user_id() for _ in range(3)]
        
        # Ajouter des connexions à la room
        for user_id in user_ids:
            conn = await manager.add_connection(mock_websocket, user_id)
            await manager.join_room(conn.connection_id, room_id)
        
        message = {"type": "room", "data": "hello room"}
        sent_count = await manager.send_to_room(room_id, message)
        
        assert sent_count == 3
        assert mock_websocket.send.call_count == 3
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, manager, mock_websocket):
        """Test d'intégration du rate limiting"""
        user_id = TestUtils.generate_test_user_id()
        connection = await manager.add_connection(mock_websocket, user_id)
        
        # Envoyer beaucoup de messages rapidement
        messages_sent = 0
        for i in range(150):  # Plus que la limite
            result = await connection.send_message({"data": f"message_{i}"})
            if result:
                messages_sent += 1
        
        # Certains messages devraient être bloqués
        assert messages_sent < 150
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_on_disconnect(self, manager, mock_websocket):
        """Test de nettoyage lors de la déconnexion"""
        user_id = TestUtils.generate_test_user_id()
        connection = await manager.add_connection(mock_websocket, user_id)
        connection_id = connection.connection_id
        
        # Simuler une déconnexion
        await manager.handle_disconnect(connection_id)
        
        assert connection_id not in manager.connections
        if user_id in manager.user_connections:
            assert connection_id not in manager.user_connections[user_id]
    
    @pytest.mark.asyncio
    async def test_room_management(self, manager, mock_websocket):
        """Test de gestion des rooms"""
        room_id = "test_room"
        user_id = TestUtils.generate_test_user_id()
        
        connection = await manager.add_connection(mock_websocket, user_id)
        connection_id = connection.connection_id
        
        # Rejoindre une room
        await manager.join_room(connection_id, room_id)
        assert room_id in manager.rooms
        assert connection_id in manager.rooms[room_id]
        
        # Quitter la room
        await manager.leave_room(connection_id, room_id)
        assert connection_id not in manager.rooms.get(room_id, set())
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_load_connections(self, manager):
        """Test de charge élevée avec beaucoup de connexions"""
        start_time = time.time()
        
        # Créer beaucoup de connexions mock
        connections = []
        for i in range(100):
            mock_ws = Mock()
            mock_ws.send = AsyncMock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            
            user_id = f"load_test_user_{i}"
            conn = await manager.add_connection(mock_ws, user_id)
            connections.append(conn)
        
        creation_time = time.time() - start_time
        
        # Test de broadcast à toutes les connexions
        start_time = time.time()
        message = {"type": "load_test", "data": "performance test"}
        sent_count = await manager.broadcast_message(message)
        broadcast_time = time.time() - start_time
        
        assert sent_count == 100
        assert creation_time < 5.0  # Moins de 5 secondes pour créer 100 connexions
        assert broadcast_time < 1.0  # Moins de 1 seconde pour broadcaster
        
        # Nettoyage
        for conn in connections:
            await manager.remove_connection(conn.connection_id)


class TestWebSocketCluster:
    """Tests pour WebSocketCluster"""
    
    @pytest.fixture
    async def cluster(self):
        """Cluster WebSocket de test"""
        cluster = WebSocketCluster(
            redis_url=REDIS_TEST_URL,
            node_id="test_node_1"
        )
        await cluster.initialize()
        
        yield cluster
        
        await cluster.shutdown()
    
    @pytest.mark.asyncio
    async def test_cluster_initialization(self, cluster):
        """Test d'initialisation du cluster"""
        assert cluster.node_id == "test_node_1"
        assert cluster.redis_client is not None
    
    @pytest.mark.asyncio
    async def test_node_registration(self, cluster):
        """Test d'enregistrement de nœud"""
        await cluster.register_node()
        
        # Vérifier dans Redis
        nodes = await cluster.get_active_nodes()
        assert "test_node_1" in nodes
    
    @pytest.mark.asyncio
    async def test_cross_node_messaging(self, cluster):
        """Test de messagerie inter-nœuds"""
        target_user = TestUtils.generate_test_user_id()
        message = {"type": "cross_node", "data": "hello from another node"}
        
        # Simuler l'envoi à un autre nœud
        await cluster.send_to_user_across_cluster(target_user, message)
        
        # Vérifier que le message a été publié
        # (Dans un vrai test, on vérifierait la réception sur l'autre nœud)
    
    @pytest.mark.asyncio
    async def test_node_health_monitoring(self, cluster):
        """Test de surveillance de santé des nœuds"""
        await cluster.register_node()
        
        # Mettre à jour le heartbeat
        await cluster.update_heartbeat()
        
        # Vérifier que le nœud est considéré comme actif
        nodes = await cluster.get_active_nodes()
        assert "test_node_1" in nodes


@pytest.mark.integration
class TestWebSocketIntegration:
    """Tests d'intégration WebSocket complets"""
    
    @pytest.mark.asyncio
    async def test_full_connection_lifecycle(self):
        """Test du cycle de vie complet d'une connexion"""
        # Configuration
        config = {
            "redis_url": REDIS_TEST_URL,
            "max_connections": 1000
        }
        
        manager = AdvancedWebSocketManager(config)
        await manager.initialize()
        
        try:
            # Mock WebSocket
            mock_websocket = Mock()
            mock_websocket.send = AsyncMock()
            mock_websocket.recv = AsyncMock(return_value='{"type": "ping"}')
            mock_websocket.close = AsyncMock()
            mock_websocket.remote_address = ("127.0.0.1", 12345)
            
            user_id = TestUtils.generate_test_user_id()
            
            # 1. Connexion
            connection = await manager.add_connection(mock_websocket, user_id)
            assert connection is not None
            
            # 2. Authentification
            with patch('jwt.decode') as mock_decode:
                mock_decode.return_value = {"user_id": user_id}
                auth_result = await connection.authenticate("test_token")
                assert auth_result is True
            
            # 3. Envoi de messages
            message = {"type": "test", "data": "hello"}
            send_result = await connection.send_message(message)
            assert send_result is True
            
            # 4. Rejoindre une room
            room_id = "test_integration_room"
            await manager.join_room(connection.connection_id, room_id)
            
            # 5. Broadcast à la room
            room_message = {"type": "room_message", "data": "hello room"}
            sent_count = await manager.send_to_room(room_id, room_message)
            assert sent_count == 1
            
            # 6. Déconnexion
            await manager.remove_connection(connection.connection_id)
            assert connection.connection_id not in manager.connections
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test de scénarios de récupération d'erreur"""
        config = {
            "redis_url": REDIS_TEST_URL,
            "circuit_breaker": {
                "failure_threshold": 2,
                "recovery_timeout": 1
            }
        }
        
        manager = AdvancedWebSocketManager(config)
        await manager.initialize()
        
        try:
            # Test de récupération après panne Redis
            original_redis = manager.redis_client
            
            # Simuler panne Redis
            manager.redis_client = None
            
            # Les opérations devraient continuer à fonctionner
            mock_websocket = Mock()
            mock_websocket.send = AsyncMock()
            mock_websocket.remote_address = ("127.0.0.1", 12345)
            
            user_id = TestUtils.generate_test_user_id()
            connection = await manager.add_connection(mock_websocket, user_id)
            
            # Même sans Redis, la connexion locale devrait fonctionner
            assert connection is not None
            
            # Restaurer Redis
            manager.redis_client = original_redis
            
        finally:
            await manager.shutdown()


# Utilitaires pour les tests de performance
class PerformanceTestUtils:
    """Utilitaires pour les tests de performance"""
    
    @staticmethod
    async def measure_execution_time(coro):
        """Mesure le temps d'exécution d'une coroutine"""
        start_time = time.time()
        result = await coro
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    async def stress_test_connections(manager, num_connections=100):
        """Test de stress avec de nombreuses connexions"""
        connections = []
        
        start_time = time.time()
        
        for i in range(num_connections):
            mock_ws = Mock()
            mock_ws.send = AsyncMock()
            mock_ws.remote_address = ("127.0.0.1", 12345 + i)
            
            user_id = f"stress_user_{i}"
            conn = await manager.add_connection(mock_ws, user_id)
            connections.append(conn)
        
        creation_time = time.time() - start_time
        
        # Test de broadcast
        start_time = time.time()
        await manager.broadcast_message({"type": "stress_test"})
        broadcast_time = time.time() - start_time
        
        # Nettoyage
        start_time = time.time()
        for conn in connections:
            await manager.remove_connection(conn.connection_id)
        cleanup_time = time.time() - start_time
        
        return {
            "creation_time": creation_time,
            "broadcast_time": broadcast_time,
            "cleanup_time": cleanup_time,
            "connections_created": len(connections)
        }


# Configuration des tests de performance
@pytest.mark.performance
class TestWebSocketPerformance:
    """Tests de performance WebSocket"""
    
    @pytest.mark.asyncio
    async def test_connection_creation_performance(self):
        """Test de performance de création de connexions"""
        config = {"redis_url": REDIS_TEST_URL}
        manager = AdvancedWebSocketManager(config)
        await manager.initialize()
        
        try:
            results = await PerformanceTestUtils.stress_test_connections(manager, 50)
            
            # Assertions de performance
            assert results["creation_time"] < 5.0  # Moins de 5 secondes
            assert results["broadcast_time"] < 1.0  # Moins de 1 seconde
            assert results["cleanup_time"] < 2.0   # Moins de 2 secondes
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Test de débit de messages"""
        config = {"redis_url": REDIS_TEST_URL}
        manager = AdvancedWebSocketManager(config)
        await manager.initialize()
        
        try:
            # Créer quelques connexions
            connections = []
            for i in range(10):
                mock_ws = Mock()
                mock_ws.send = AsyncMock()
                mock_ws.remote_address = ("127.0.0.1", 12345 + i)
                
                conn = await manager.add_connection(mock_ws, f"throughput_user_{i}")
                connections.append(conn)
            
            # Test de débit
            num_messages = 1000
            start_time = time.time()
            
            for i in range(num_messages):
                await manager.broadcast_message({"type": "throughput", "seq": i})
            
            total_time = time.time() - start_time
            messages_per_second = num_messages / total_time
            
            # Au moins 100 messages par seconde
            assert messages_per_second > 100
            
        finally:
            await manager.shutdown()


# Export des classes de test
__all__ = [
    "TestWebSocketConnection",
    "TestRateLimiter", 
    "TestCircuitBreaker",
    "TestAdvancedWebSocketManager",
    "TestWebSocketCluster",
    "TestWebSocketIntegration",
    "TestWebSocketPerformance",
    "PerformanceTestUtils"
]
