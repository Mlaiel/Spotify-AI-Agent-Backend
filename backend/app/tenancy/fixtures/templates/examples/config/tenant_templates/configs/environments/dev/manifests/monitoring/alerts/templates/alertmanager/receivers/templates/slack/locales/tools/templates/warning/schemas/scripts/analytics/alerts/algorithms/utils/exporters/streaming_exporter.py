"""
Advanced Streaming Metrics Exporter
===================================

Exportateur en temps réel pour streaming haute fréquence avec
support WebSocket, Server-Sent Events et MQTT.

Fonctionnalités:
- Streaming temps réel multi-protocole
- Backpressure handling
- Reconnection automatique
- Load balancing
- Monitoring de latence
- Support multi-tenant
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import websockets
import aiohttp
import aiomqtt
from aiohttp_sse import sse_response
import structlog
from custom_exporter import MetricPayload

logger = structlog.get_logger(__name__)


class StreamingProtocol(Enum):
    """Protocoles de streaming supportés."""
    WEBSOCKET = "websocket"
    SSE = "sse"
    MQTT = "mqtt"
    HTTP_STREAM = "http_stream"


@dataclass
class StreamingConfig:
    """Configuration pour le streaming."""
    protocol: StreamingProtocol
    endpoint_url: str
    tenant_id: str
    max_connections: int = 100
    max_queue_size: int = 10000
    heartbeat_interval: float = 30.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    enable_compression: bool = True
    enable_batching: bool = False
    batch_size: int = 10
    batch_timeout: float = 1.0
    auth_token: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class StreamConnection:
    """Représente une connexion de streaming."""
    connection_id: str
    protocol: StreamingProtocol
    endpoint: str
    connected_at: datetime
    last_heartbeat: datetime
    metrics_sent: int = 0
    reconnect_count: int = 0
    is_healthy: bool = True
    latency_ms: float = 0.0


class StreamingMetricsExporter:
    """
    Exportateur streaming haute performance.
    
    Support multiple protocoles avec failover automatique
    et monitoring de performance en temps réel.
    """
    
    def __init__(self, configs: List[StreamingConfig]):
        self.configs = configs
        self.connections: Dict[str, StreamConnection] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.is_running = False
        
        # Statistiques
        self.stats = {
            'total_metrics_streamed': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'avg_latency_ms': 0,
            'throughput_per_second': 0,
            'backpressure_events': 0
        }
        
        # Monitoring de débit
        self.throughput_counter = 0
        self.last_throughput_reset = time.time()
        
        # Load balancing
        self.connection_weights: Dict[str, float] = {}
        
    async def initialize(self):
        """Initialise l'exportateur streaming."""
        try:
            # Créer les connexions pour chaque configuration
            for config in self.configs:
                await self._create_connection(config)
                
            # Démarrer les tâches de monitoring
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._throughput_monitor())
            asyncio.create_task(self._health_monitor())
            
            self.is_running = True
            
            logger.info(
                "StreamingMetricsExporter initialized",
                connections=len(self.connections),
                protocols=[config.protocol.value for config in self.configs]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize streaming exporter: {e}")
            raise
            
    async def _create_connection(self, config: StreamingConfig):
        """Crée une connexion selon le protocole."""
        connection_id = f"{config.protocol.value}_{config.tenant_id}_{uuid.uuid4().hex[:8]}"
        
        # Créer la queue de messages
        self.message_queues[connection_id] = asyncio.Queue(maxsize=config.max_queue_size)
        
        # Créer l'objet connection
        connection = StreamConnection(
            connection_id=connection_id,
            protocol=config.protocol,
            endpoint=config.endpoint_url,
            connected_at=datetime.now(),
            last_heartbeat=datetime.now()
        )
        
        self.connections[connection_id] = connection
        self.connection_weights[connection_id] = 1.0
        
        # Démarrer la tâche de streaming selon le protocole
        if config.protocol == StreamingProtocol.WEBSOCKET:
            asyncio.create_task(self._websocket_stream_worker(connection_id, config))
        elif config.protocol == StreamingProtocol.SSE:
            asyncio.create_task(self._sse_stream_worker(connection_id, config))
        elif config.protocol == StreamingProtocol.MQTT:
            asyncio.create_task(self._mqtt_stream_worker(connection_id, config))
        elif config.protocol == StreamingProtocol.HTTP_STREAM:
            asyncio.create_task(self._http_stream_worker(connection_id, config))
            
    async def _websocket_stream_worker(self, connection_id: str, config: StreamingConfig):
        """Worker pour WebSocket streaming."""
        connection = self.connections[connection_id]
        queue = self.message_queues[connection_id]
        reconnect_count = 0
        
        while self.is_running and reconnect_count < config.max_reconnect_attempts:
            try:
                # Headers d'authentification
                headers = config.custom_headers.copy()
                if config.auth_token:
                    headers['Authorization'] = f'Bearer {config.auth_token}'
                    
                # Connexion WebSocket
                async with websockets.connect(
                    config.endpoint_url,
                    extra_headers=headers,
                    compression='deflate' if config.enable_compression else None
                ) as websocket:
                    
                    connection.is_healthy = True
                    connection.reconnect_count = reconnect_count
                    self.stats['active_connections'] += 1
                    
                    logger.info(f"WebSocket connected: {connection_id}")
                    
                    # Boucle de streaming
                    while self.is_running:
                        try:
                            # Gérer le batching si activé
                            if config.enable_batching:
                                messages = await self._collect_batch(queue, config.batch_size, config.batch_timeout)
                                if messages:
                                    batch_payload = {
                                        'type': 'batch',
                                        'tenant_id': config.tenant_id,
                                        'timestamp': datetime.now().isoformat(),
                                        'metrics': messages
                                    }
                                    await self._send_websocket_message(websocket, batch_payload, connection)
                            else:
                                # Message unique
                                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                                await self._send_websocket_message(websocket, message, connection)
                                
                        except asyncio.TimeoutError:
                            # Envoyer un heartbeat
                            await self._send_heartbeat(websocket, config.tenant_id)
                            connection.last_heartbeat = datetime.now()
                            
            except Exception as e:
                connection.is_healthy = False
                self.stats['failed_connections'] += 1
                reconnect_count += 1
                
                logger.warning(
                    f"WebSocket connection failed: {e}",
                    connection_id=connection_id,
                    reconnect_count=reconnect_count
                )
                
                if reconnect_count < config.max_reconnect_attempts:
                    await asyncio.sleep(config.reconnect_delay * reconnect_count)
                    
        # Connexion fermée
        if connection_id in self.connections:
            self.connections[connection_id].is_healthy = False
            self.stats['active_connections'] = max(0, self.stats['active_connections'] - 1)
            
    async def _send_websocket_message(self, websocket, message: Dict[str, Any], connection: StreamConnection):
        """Envoie un message WebSocket avec mesure de latence."""
        start_time = time.time()
        
        try:
            await websocket.send(json.dumps(message))
            
            # Mesurer la latence
            latency = (time.time() - start_time) * 1000
            connection.latency_ms = latency
            connection.metrics_sent += 1
            
            self.stats['total_metrics_streamed'] += 1
            self.throughput_counter += 1
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise
            
    async def _send_heartbeat(self, websocket, tenant_id: str):
        """Envoie un heartbeat WebSocket."""
        heartbeat = {
            'type': 'heartbeat',
            'tenant_id': tenant_id,
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send(json.dumps(heartbeat))
        
    async def _sse_stream_worker(self, connection_id: str, config: StreamingConfig):
        """Worker pour Server-Sent Events."""
        connection = self.connections[connection_id]
        queue = self.message_queues[connection_id]
        reconnect_count = 0
        
        while self.is_running and reconnect_count < config.max_reconnect_attempts:
            try:
                headers = config.custom_headers.copy()
                if config.auth_token:
                    headers['Authorization'] = f'Bearer {config.auth_token}'
                    
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        config.endpoint_url,
                        headers=headers
                    ) as response:
                        
                        if response.status == 200:
                            connection.is_healthy = True
                            self.stats['active_connections'] += 1
                            
                            # Stream SSE
                            async for line in response.content:
                                if not self.is_running:
                                    break
                                    
                                # Traiter les messages de la queue
                                try:
                                    message = queue.get_nowait()
                                    await self._send_sse_message(session, config.endpoint_url, message, connection)
                                except asyncio.QueueEmpty:
                                    await asyncio.sleep(0.1)
                                    
            except Exception as e:
                connection.is_healthy = False
                self.stats['failed_connections'] += 1
                reconnect_count += 1
                
                logger.warning(f"SSE connection failed: {e}")
                await asyncio.sleep(config.reconnect_delay * reconnect_count)
                
    async def _send_sse_message(self, session, url: str, message: Dict[str, Any], connection: StreamConnection):
        """Envoie un message SSE."""
        start_time = time.time()
        
        try:
            sse_data = f"data: {json.dumps(message)}\n\n"
            
            async with session.post(url, data=sse_data) as response:
                if response.status == 200:
                    latency = (time.time() - start_time) * 1000
                    connection.latency_ms = latency
                    connection.metrics_sent += 1
                    self.stats['total_metrics_streamed'] += 1
                    self.throughput_counter += 1
                    
        except Exception as e:
            logger.error(f"Failed to send SSE message: {e}")
            raise
            
    async def _mqtt_stream_worker(self, connection_id: str, config: StreamingConfig):
        """Worker pour MQTT streaming."""
        connection = self.connections[connection_id]
        queue = self.message_queues[connection_id]
        reconnect_count = 0
        
        while self.is_running and reconnect_count < config.max_reconnect_attempts:
            try:
                # Extraire les paramètres MQTT de l'URL
                mqtt_host = config.endpoint_url.split('://')[1].split(':')[0]
                mqtt_port = int(config.endpoint_url.split(':')[-1]) if ':' in config.endpoint_url.split('://')[1] else 1883
                
                async with aiomqtt.Client(
                    hostname=mqtt_host,
                    port=mqtt_port,
                    username=config.auth_token.split(':')[0] if config.auth_token and ':' in config.auth_token else None,
                    password=config.auth_token.split(':')[1] if config.auth_token and ':' in config.auth_token else None
                ) as client:
                    
                    connection.is_healthy = True
                    self.stats['active_connections'] += 1
                    
                    logger.info(f"MQTT connected: {connection_id}")
                    
                    # Topic basé sur le tenant
                    topic = f"spotify/ai/metrics/{config.tenant_id}"
                    
                    while self.is_running:
                        try:
                            message = await asyncio.wait_for(queue.get(), timeout=1.0)
                            await self._send_mqtt_message(client, topic, message, connection)
                            
                        except asyncio.TimeoutError:
                            # Heartbeat MQTT
                            heartbeat = {
                                'type': 'heartbeat',
                                'tenant_id': config.tenant_id,
                                'timestamp': datetime.now().isoformat()
                            }
                            await client.publish(f"{topic}/heartbeat", json.dumps(heartbeat))
                            connection.last_heartbeat = datetime.now()
                            
            except Exception as e:
                connection.is_healthy = False
                self.stats['failed_connections'] += 1
                reconnect_count += 1
                
                logger.warning(f"MQTT connection failed: {e}")
                await asyncio.sleep(config.reconnect_delay * reconnect_count)
                
    async def _send_mqtt_message(self, client, topic: str, message: Dict[str, Any], connection: StreamConnection):
        """Envoie un message MQTT."""
        start_time = time.time()
        
        try:
            payload = json.dumps(message)
            await client.publish(topic, payload)
            
            latency = (time.time() - start_time) * 1000
            connection.latency_ms = latency
            connection.metrics_sent += 1
            self.stats['total_metrics_streamed'] += 1
            self.throughput_counter += 1
            
        except Exception as e:
            logger.error(f"Failed to send MQTT message: {e}")
            raise
            
    async def _http_stream_worker(self, connection_id: str, config: StreamingConfig):
        """Worker pour HTTP streaming."""
        connection = self.connections[connection_id]
        queue = self.message_queues[connection_id]
        
        while self.is_running:
            try:
                headers = config.custom_headers.copy()
                if config.auth_token:
                    headers['Authorization'] = f'Bearer {config.auth_token}'
                headers['Content-Type'] = 'application/x-ndjson'  # Newline Delimited JSON
                
                async with aiohttp.ClientSession() as session:
                    # Collecter un batch de messages
                    messages = await self._collect_batch(queue, config.batch_size, config.batch_timeout)
                    
                    if messages:
                        # Formatter en NDJSON
                        ndjson_data = '\n'.join(json.dumps(msg) for msg in messages)
                        
                        async with session.post(
                            config.endpoint_url,
                            data=ndjson_data,
                            headers=headers
                        ) as response:
                            
                            if response.status == 200:
                                connection.metrics_sent += len(messages)
                                self.stats['total_metrics_streamed'] += len(messages)
                                self.throughput_counter += len(messages)
                                connection.is_healthy = True
                            else:
                                connection.is_healthy = False
                                logger.warning(f"HTTP stream failed: {response.status}")
                                
            except Exception as e:
                connection.is_healthy = False
                logger.error(f"HTTP stream worker error: {e}")
                await asyncio.sleep(5.0)
                
    async def _collect_batch(self, queue: asyncio.Queue, batch_size: int, timeout: float) -> List[Dict[str, Any]]:
        """Collecte un batch de messages avec timeout."""
        messages = []
        end_time = time.time() + timeout
        
        while len(messages) < batch_size and time.time() < end_time:
            try:
                remaining_time = max(0.1, end_time - time.time())
                message = await asyncio.wait_for(queue.get(), timeout=remaining_time)
                messages.append(message)
            except asyncio.TimeoutError:
                break
                
        return messages
        
    async def stream_metric(self, metric: MetricPayload):
        """Streame une métrique vers toutes les connexions actives."""
        if not self.is_running:
            return
            
        # Formatter le message
        message = {
            'type': 'metric',
            'tenant_id': metric.tenant_id,
            'metric_name': metric.metric_name,
            'metric_value': metric.metric_value,
            'metric_type': metric.metric_type,
            'timestamp': metric.timestamp.isoformat(),
            'labels': metric.labels,
            'metadata': metric.metadata,
            'source': metric.source
        }
        
        # Load balancing : sélectionner les meilleures connexions
        healthy_connections = [
            conn_id for conn_id, conn in self.connections.items()
            if conn.is_healthy
        ]
        
        if not healthy_connections:
            logger.warning("No healthy streaming connections available")
            return
            
        # Distribuer selon les poids
        selected_connections = self._select_connections_for_load_balancing(healthy_connections)
        
        # Envoyer vers les connexions sélectionnées
        for conn_id in selected_connections:
            try:
                queue = self.message_queues[conn_id]
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Backpressure : connexion surchargée
                self.stats['backpressure_events'] += 1
                self.connection_weights[conn_id] *= 0.9  # Réduire le poids
                logger.warning(f"Backpressure on connection {conn_id}")
                
    def _select_connections_for_load_balancing(self, healthy_connections: List[str]) -> List[str]:
        """Sélectionne les connexions pour load balancing."""
        # Tri par poids et latence
        scored_connections = []
        
        for conn_id in healthy_connections:
            connection = self.connections[conn_id]
            weight = self.connection_weights[conn_id]
            
            # Score basé sur latence et poids
            score = weight / max(connection.latency_ms, 1.0)
            scored_connections.append((conn_id, score))
            
        # Trier par score décroissant
        scored_connections.sort(key=lambda x: x[1], reverse=True)
        
        # Sélectionner le top 50% des connexions
        num_selected = max(1, len(scored_connections) // 2)
        return [conn_id for conn_id, _ in scored_connections[:num_selected]]
        
    async def _heartbeat_monitor(self):
        """Surveille les heartbeats des connexions."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for connection in self.connections.values():
                    if connection.is_healthy:
                        # Vérifier le dernier heartbeat
                        time_since_heartbeat = (current_time - connection.last_heartbeat).total_seconds()
                        
                        if time_since_heartbeat > 60:  # 1 minute sans heartbeat
                            connection.is_healthy = False
                            logger.warning(f"Connection {connection.connection_id} marked unhealthy")
                            
                await asyncio.sleep(30)  # Vérifier toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _throughput_monitor(self):
        """Surveille le débit de streaming."""
        while self.is_running:
            try:
                current_time = time.time()
                time_elapsed = current_time - self.last_throughput_reset
                
                if time_elapsed >= 1.0:  # Calculer toutes les secondes
                    self.stats['throughput_per_second'] = self.throughput_counter / time_elapsed
                    self.throughput_counter = 0
                    self.last_throughput_reset = current_time
                    
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Throughput monitor error: {e}")
                await asyncio.sleep(5)
                
    async def _health_monitor(self):
        """Surveille la santé globale du système."""
        while self.is_running:
            try:
                # Mettre à jour les statistiques
                healthy_connections = sum(1 for conn in self.connections.values() if conn.is_healthy)
                self.stats['active_connections'] = healthy_connections
                
                # Calculer la latence moyenne
                latencies = [conn.latency_ms for conn in self.connections.values() if conn.is_healthy]
                if latencies:
                    self.stats['avg_latency_ms'] = sum(latencies) / len(latencies)
                    
                # Ajuster les poids des connexions
                for conn_id, connection in self.connections.items():
                    if connection.is_healthy:
                        # Augmenter le poids pour les connexions performantes
                        if connection.latency_ms < 50:  # Latence faible
                            self.connection_weights[conn_id] = min(2.0, self.connection_weights[conn_id] * 1.01)
                    else:
                        # Réduire le poids pour les connexions malsaines
                        self.connection_weights[conn_id] = max(0.1, self.connection_weights[conn_id] * 0.95)
                        
                await asyncio.sleep(10)  # Vérifier toutes les 10 secondes
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
                
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de streaming détaillées."""
        connection_stats = []
        
        for conn_id, connection in self.connections.items():
            stats = {
                'connection_id': conn_id,
                'protocol': connection.protocol.value,
                'endpoint': connection.endpoint,
                'is_healthy': connection.is_healthy,
                'connected_at': connection.connected_at.isoformat(),
                'metrics_sent': connection.metrics_sent,
                'latency_ms': connection.latency_ms,
                'reconnect_count': connection.reconnect_count,
                'weight': self.connection_weights.get(conn_id, 1.0),
                'queue_size': self.message_queues[conn_id].qsize()
            }
            connection_stats.append(stats)
            
        return {
            'overall_stats': self.stats,
            'connections': connection_stats,
            'health_summary': {
                'healthy_connections': sum(1 for conn in self.connections.values() if conn.is_healthy),
                'total_connections': len(self.connections),
                'total_queue_size': sum(queue.qsize() for queue in self.message_queues.values())
            }
        }
        
    async def stop(self):
        """Arrête l'exportateur streaming."""
        self.is_running = False
        
        # Attendre un peu pour que les workers se terminent proprement
        await asyncio.sleep(2)
        
        logger.info("StreamingMetricsExporter stopped")


# Factory pour créer des exportateurs streaming
class StreamingExporterFactory:
    """Factory pour créer des exportateurs streaming."""
    
    @staticmethod
    def create_multi_protocol_exporter(
        tenant_id: str,
        websocket_url: Optional[str] = None,
        sse_url: Optional[str] = None,
        mqtt_url: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> StreamingMetricsExporter:
        """Crée un exportateur multi-protocole."""
        configs = []
        
        if websocket_url:
            configs.append(StreamingConfig(
                protocol=StreamingProtocol.WEBSOCKET,
                endpoint_url=websocket_url,
                tenant_id=tenant_id,
                auth_token=auth_token,
                enable_compression=True,
                enable_batching=False
            ))
            
        if sse_url:
            configs.append(StreamingConfig(
                protocol=StreamingProtocol.SSE,
                endpoint_url=sse_url,
                tenant_id=tenant_id,
                auth_token=auth_token,
                enable_batching=True,
                batch_size=5
            ))
            
        if mqtt_url:
            configs.append(StreamingConfig(
                protocol=StreamingProtocol.MQTT,
                endpoint_url=mqtt_url,
                tenant_id=tenant_id,
                auth_token=auth_token,
                enable_compression=False
            ))
            
        return StreamingMetricsExporter(configs)


# Usage example
if __name__ == "__main__":
    async def main():
        # Créer l'exportateur streaming multi-protocole
        exporter = StreamingExporterFactory.create_multi_protocol_exporter(
            tenant_id="spotify_artist_daft_punk",
            websocket_url="ws://localhost:8080/metrics/stream",
            sse_url="http://localhost:8080/metrics/sse",
            mqtt_url="mqtt://localhost:1883",
            auth_token="your-auth-token"
        )
        
        await exporter.initialize()
        
        # Simuler du streaming de métriques
        for i in range(100):
            metric = MetricPayload(
                tenant_id="spotify_artist_daft_punk",
                metric_name=f"streaming_test_{i % 5}",
                metric_value=i * 0.1,
                metric_type="gauge",
                timestamp=datetime.now(),
                labels={'test': 'streaming', 'sequence': str(i)}
            )
            
            await exporter.stream_metric(metric)
            await asyncio.sleep(0.1)  # 10 métriques/seconde
            
        # Statistiques
        stats = await exporter.get_streaming_stats()
        print(f"Streaming stats: {stats}")
        
        await exporter.stop()
        
    asyncio.run(main())
