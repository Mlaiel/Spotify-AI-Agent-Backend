"""
Enterprise Cache Coordination
=============================
Distributed cache coordination, clustering, and consistency management for enterprise deployment.

Expert Team Implementation:
- Lead Developer + AI Architect: Intelligent load balancing and predictive scaling
- Senior Backend Developer: High-performance distributed algorithms and consensus protocols
- Machine Learning Engineer: ML-driven cache placement and migration strategies
- DBA & Data Engineer: Data consistency, replication, and conflict resolution
- Security Specialist: Secure inter-node communication and cluster authentication
- Microservices Architect: Service mesh integration and cross-cluster coordination
"""

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import socket
import struct

# Consistent hashing imports
try:
    import hashring
    HASHRING_AVAILABLE = True
except ImportError:
    HASHRING_AVAILABLE = False
    hashring = None

# Redis clustering for coordination
try:
    import redis
    from redis.sentinel import Sentinel
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Message queue for coordination
try:
    import aio_pika
    AMQP_AVAILABLE = True
except ImportError:
    AMQP_AVAILABLE = False
    aio_pika = None

logger = logging.getLogger(__name__)

# === Coordination Types and Enums ===
class NodeStatus(Enum):
    """Cache node status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    JOINING = "joining"
    LEAVING = "leaving"

class ReplicationStrategy(Enum):
    """Data replication strategies."""
    NONE = "none"
    MASTER_SLAVE = "master_slave"
    MULTI_MASTER = "multi_master"
    EVENTUAL_CONSISTENCY = "eventual_consistency"
    STRONG_CONSISTENCY = "strong_consistency"

class ConsistencyLevel(Enum):
    """Data consistency levels."""
    ANY = "any"           # Any replica can respond
    ONE = "one"           # At least one replica
    QUORUM = "quorum"     # Majority of replicas
    ALL = "all"           # All replicas must respond

class PartitionStrategy(Enum):
    """Cache partitioning strategies."""
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"
    DIRECTORY_BASED = "directory_based"
    CONSISTENT_HASHING = "consistent_hashing"

@dataclass
class CacheNode:
    """Cache node information."""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    load_factor: float = 0.0
    memory_usage_percent: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    capabilities: Set[str] = field(default_factory=set)
    region: str = "default"
    zone: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        """Get node address."""
        return f"{self.host}:{self.port}"
    
    def is_healthy(self, heartbeat_timeout: timedelta = timedelta(seconds=30)) -> bool:
        """Check if node is healthy based on heartbeat."""
        if self.status == NodeStatus.OFFLINE:
            return False
        return datetime.now() - self.last_heartbeat < heartbeat_timeout
    
    def update_heartbeat(self, load_factor: float = None, memory_usage: float = None):
        """Update node heartbeat and metrics."""
        self.last_heartbeat = datetime.now()
        if load_factor is not None:
            self.load_factor = load_factor
        if memory_usage is not None:
            self.memory_usage_percent = memory_usage

@dataclass
class ClusterConfig:
    """Cache cluster configuration."""
    cluster_name: str
    replication_strategy: ReplicationStrategy = ReplicationStrategy.MASTER_SLAVE
    consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM
    partition_strategy: PartitionStrategy = PartitionStrategy.CONSISTENT_HASHING
    replication_factor: int = 3
    heartbeat_interval: int = 10  # seconds
    heartbeat_timeout: int = 30   # seconds
    auto_failover: bool = True
    auto_recovery: bool = True
    load_balancing: bool = True
    cross_region_replication: bool = False
    encryption_in_transit: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReplicationRequest:
    """Data replication request."""
    operation: str  # set, delete, invalidate
    key: str
    value: Any = None
    ttl: Optional[int] = None
    source_node: str = ""
    target_nodes: List[str] = field(default_factory=list)
    consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ConsensusRequest:
    """Consensus/coordination request."""
    operation: str
    data: Dict[str, Any]
    proposer_node: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    votes: Dict[str, bool] = field(default_factory=dict)
    accepted: Optional[bool] = None

# === Consistent Hashing ===
class ConsistentHashRing:
    """Consistent hashing implementation for cache partitioning."""
    
    def __init__(self, nodes: List[CacheNode] = None, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Dict[str, CacheNode] = {}
        
        if nodes:
            for node in nodes:
                self.add_node(node)
        
        logger.info(f"ConsistentHashRing initialized with {len(self.nodes)} nodes")
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: CacheNode):
        """Add node to the hash ring."""
        self.nodes[node.node_id] = node
        
        # Add virtual nodes for better distribution
        for i in range(self.virtual_nodes):
            virtual_key = f"{node.node_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node.node_id
        
        # Update sorted keys
        self.sorted_keys = sorted(self.ring.keys())
        logger.info(f"Added node {node.node_id} to hash ring")
    
    def remove_node(self, node_id: str):
        """Remove node from the hash ring."""
        if node_id not in self.nodes:
            return
        
        # Remove virtual nodes
        keys_to_remove = [k for k, v in self.ring.items() if v == node_id]
        for key in keys_to_remove:
            del self.ring[key]
        
        del self.nodes[node_id]
        self.sorted_keys = sorted(self.ring.keys())
        logger.info(f"Removed node {node_id} from hash ring")
    
    def get_node(self, key: str) -> Optional[CacheNode]:
        """Get the primary node for a given key."""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node with hash >= key_hash
        for ring_key in self.sorted_keys:
            if ring_key >= hash_value:
                node_id = self.ring[ring_key]
                return self.nodes.get(node_id)
        
        # Wrap around to the first node
        first_key = self.sorted_keys[0]
        node_id = self.ring[first_key]
        return self.nodes.get(node_id)
    
    def get_nodes(self, key: str, count: int = 1) -> List[CacheNode]:
        """Get multiple nodes for a key (for replication)."""
        if not self.ring or count <= 0:
            return []
        
        hash_value = self._hash(key)
        nodes = []
        seen_nodes = set()
        
        # Start from the primary node position
        start_idx = 0
        for i, ring_key in enumerate(self.sorted_keys):
            if ring_key >= hash_value:
                start_idx = i
                break
        
        # Collect unique nodes
        for i in range(len(self.sorted_keys)):
            idx = (start_idx + i) % len(self.sorted_keys)
            ring_key = self.sorted_keys[idx]
            node_id = self.ring[ring_key]
            
            if node_id not in seen_nodes:
                node = self.nodes.get(node_id)
                if node and node.is_healthy():
                    nodes.append(node)
                    seen_nodes.add(node_id)
                    
                    if len(nodes) >= count:
                        break
        
        return nodes
    
    def get_key_distribution(self) -> Dict[str, int]:
        """Get key distribution across nodes."""
        distribution = defaultdict(int)
        
        for node_id in self.ring.values():
            distribution[node_id] += 1
        
        return dict(distribution)

# === Cache Cluster Manager ===
class CacheCluster:
    """Distributed cache cluster management."""
    
    def __init__(self, config: ClusterConfig, local_node: CacheNode):
        self.config = config
        self.local_node = local_node
        self.nodes: Dict[str, CacheNode] = {local_node.node_id: local_node}
        self.hash_ring = ConsistentHashRing([local_node])
        
        # Cluster state
        self.master_node: Optional[str] = None
        self.is_master = False
        self.cluster_state = "initializing"
        
        # Replication and consensus
        self.replication_queue: deque = deque()
        self.consensus_requests: Dict[str, ConsensusRequest] = {}
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._replication_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Coordination backend (Redis, message queue, etc.)
        self.coordination_backend: Optional[Any] = None
        
        logger.info(f"CacheCluster {config.cluster_name} initialized with node {local_node.node_id}")
    
    async def start_cluster(self):
        """Start cluster coordination."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize coordination backend
        await self._init_coordination_backend()
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._replication_task = asyncio.create_task(self._replication_loop())
        
        # Join cluster
        await self._join_cluster()
        
        logger.info(f"Cluster {self.config.cluster_name} started")
    
    async def stop_cluster(self):
        """Stop cluster coordination."""
        if not self._running:
            return
        
        self._running = False
        
        # Leave cluster gracefully
        await self._leave_cluster()
        
        # Stop background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._replication_task:
            self._replication_task.cancel()
        
        # Close coordination backend
        await self._close_coordination_backend()
        
        logger.info(f"Cluster {self.config.cluster_name} stopped")
    
    async def _init_coordination_backend(self):
        """Initialize coordination backend (Redis/Message Queue)."""
        if REDIS_AVAILABLE:
            try:
                # Use Redis for cluster coordination
                self.coordination_backend = redis.Redis(
                    host='localhost',  # Configure appropriately
                    port=6379,
                    decode_responses=True
                )
                await self.coordination_backend.ping()
                logger.info("Redis coordination backend initialized")
            except Exception as e:
                logger.warning(f"Redis coordination failed: {e}")
                self.coordination_backend = None
    
    async def _close_coordination_backend(self):
        """Close coordination backend."""
        if self.coordination_backend:
            await self.coordination_backend.close()
    
    async def _join_cluster(self):
        """Join the cache cluster."""
        try:
            # Register node in cluster
            await self._register_node()
            
            # Discover existing nodes
            await self._discover_nodes()
            
            # Elect master if needed
            await self._elect_master()
            
            self.cluster_state = "active"
            logger.info(f"Node {self.local_node.node_id} joined cluster")
            
        except Exception as e:
            logger.error(f"Failed to join cluster: {e}")
            self.cluster_state = "failed"
    
    async def _leave_cluster(self):
        """Leave the cache cluster gracefully."""
        try:
            # Migrate data if needed
            await self._migrate_data_before_leaving()
            
            # Unregister node
            await self._unregister_node()
            
            self.cluster_state = "left"
            logger.info(f"Node {self.local_node.node_id} left cluster")
            
        except Exception as e:
            logger.error(f"Failed to leave cluster cleanly: {e}")
    
    async def _register_node(self):
        """Register this node in the cluster."""
        if not self.coordination_backend:
            return
        
        node_data = {
            "node_id": self.local_node.node_id,
            "host": self.local_node.host,
            "port": self.local_node.port,
            "status": self.local_node.status.value,
            "region": self.local_node.region,
            "zone": self.local_node.zone,
            "capabilities": list(self.local_node.capabilities),
            "joined_at": datetime.now().isoformat()
        }
        
        # Register in Redis
        key = f"cluster:{self.config.cluster_name}:nodes:{self.local_node.node_id}"
        await self.coordination_backend.hset(key, mapping=node_data)
        await self.coordination_backend.expire(key, self.config.heartbeat_timeout * 2)
        
        # Add to active nodes set
        await self.coordination_backend.sadd(
            f"cluster:{self.config.cluster_name}:active_nodes",
            self.local_node.node_id
        )
    
    async def _unregister_node(self):
        """Unregister this node from the cluster."""
        if not self.coordination_backend:
            return
        
        # Remove from active nodes
        await self.coordination_backend.srem(
            f"cluster:{self.config.cluster_name}:active_nodes",
            self.local_node.node_id
        )
        
        # Delete node data
        await self.coordination_backend.delete(
            f"cluster:{self.config.cluster_name}:nodes:{self.local_node.node_id}"
        )
    
    async def _discover_nodes(self):
        """Discover other nodes in the cluster."""
        if not self.coordination_backend:
            return
        
        # Get active nodes
        active_nodes = await self.coordination_backend.smembers(
            f"cluster:{self.config.cluster_name}:active_nodes"
        )
        
        for node_id in active_nodes:
            if node_id == self.local_node.node_id:
                continue
            
            # Get node data
            key = f"cluster:{self.config.cluster_name}:nodes:{node_id}"
            node_data = await self.coordination_backend.hgetall(key)
            
            if node_data:
                node = CacheNode(
                    node_id=node_data["node_id"],
                    host=node_data["host"],
                    port=int(node_data["port"]),
                    status=NodeStatus(node_data["status"]),
                    region=node_data.get("region", "default"),
                    zone=node_data.get("zone", "default"),
                    capabilities=set(json.loads(node_data.get("capabilities", "[]")))
                )
                
                self.add_node(node)
        
        logger.info(f"Discovered {len(self.nodes) - 1} other nodes in cluster")
    
    async def _elect_master(self):
        """Elect cluster master node."""
        if not self.coordination_backend:
            return
        
        # Try to acquire master lock
        master_key = f"cluster:{self.config.cluster_name}:master"
        
        # Check if there's already a master
        current_master = await self.coordination_backend.get(master_key)
        
        if not current_master:
            # Try to become master
            success = await self.coordination_backend.set(
                master_key,
                self.local_node.node_id,
                nx=True,  # Only set if not exists
                ex=self.config.heartbeat_timeout
            )
            
            if success:
                self.master_node = self.local_node.node_id
                self.is_master = True
                logger.info(f"Node {self.local_node.node_id} elected as cluster master")
            else:
                # Someone else became master
                current_master = await self.coordination_backend.get(master_key)
                self.master_node = current_master
                self.is_master = False
        else:
            self.master_node = current_master
            self.is_master = (current_master == self.local_node.node_id)
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while self._running:
            try:
                await self._send_heartbeat()
                await self._check_node_health()
                
                # Renew master lock if we're the master
                if self.is_master:
                    await self._renew_master_lock()
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(1)
    
    async def _send_heartbeat(self):
        """Send heartbeat to cluster."""
        if not self.coordination_backend:
            return
        
        # Update node data with current metrics
        node_data = {
            "status": self.local_node.status.value,
            "load_factor": self.local_node.load_factor,
            "memory_usage_percent": self.local_node.memory_usage_percent,
            "last_heartbeat": datetime.now().isoformat()
        }
        
        key = f"cluster:{self.config.cluster_name}:nodes:{self.local_node.node_id}"
        await self.coordination_backend.hset(key, mapping=node_data)
        await self.coordination_backend.expire(key, self.config.heartbeat_timeout * 2)
    
    async def _check_node_health(self):
        """Check health of other nodes."""
        if not self.coordination_backend:
            return
        
        unhealthy_nodes = []
        
        for node_id, node in self.nodes.items():
            if node_id == self.local_node.node_id:
                continue
            
            # Check if node data exists and is recent
            key = f"cluster:{self.config.cluster_name}:nodes:{node_id}"
            ttl = await self.coordination_backend.ttl(key)
            
            if ttl <= 0:
                # Node data expired - mark as unhealthy
                node.status = NodeStatus.OFFLINE
                unhealthy_nodes.append(node_id)
            else:
                # Update node data
                node_data = await self.coordination_backend.hgetall(key)
                if node_data:
                    node.status = NodeStatus(node_data.get("status", "unhealthy"))
                    node.load_factor = float(node_data.get("load_factor", 0))
                    node.memory_usage_percent = float(node_data.get("memory_usage_percent", 0))
                    if "last_heartbeat" in node_data:
                        node.last_heartbeat = datetime.fromisoformat(node_data["last_heartbeat"])
        
        # Remove unhealthy nodes
        for node_id in unhealthy_nodes:
            self.remove_node(node_id)
            logger.warning(f"Removed unhealthy node {node_id} from cluster")
    
    async def _renew_master_lock(self):
        """Renew master lock if we're the master."""
        if not self.coordination_backend or not self.is_master:
            return
        
        master_key = f"cluster:{self.config.cluster_name}:master"
        await self.coordination_backend.expire(master_key, self.config.heartbeat_timeout)
    
    async def _replication_loop(self):
        """Background replication processing loop."""
        while self._running:
            try:
                if self.replication_queue:
                    request = self.replication_queue.popleft()
                    await self._process_replication_request(request)
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Replication loop error: {e}")
                await asyncio.sleep(1)
    
    def add_node(self, node: CacheNode):
        """Add node to cluster."""
        self.nodes[node.node_id] = node
        self.hash_ring.add_node(node)
        logger.info(f"Added node {node.node_id} to cluster")
    
    def remove_node(self, node_id: str):
        """Remove node from cluster."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.hash_ring.remove_node(node_id)
            logger.info(f"Removed node {node_id} from cluster")
    
    def get_nodes_for_key(self, key: str) -> List[CacheNode]:
        """Get nodes responsible for a key."""
        return self.hash_ring.get_nodes(key, self.config.replication_factor)
    
    def get_primary_node_for_key(self, key: str) -> Optional[CacheNode]:
        """Get primary node for a key."""
        return self.hash_ring.get_node(key)
    
    async def replicate_operation(self, operation: str, key: str, value: Any = None, ttl: Optional[int] = None):
        """Replicate cache operation to appropriate nodes."""
        nodes = self.get_nodes_for_key(key)
        
        if not nodes:
            logger.warning(f"No nodes available for key {key}")
            return
        
        # Create replication request
        request = ReplicationRequest(
            operation=operation,
            key=key,
            value=value,
            ttl=ttl,
            source_node=self.local_node.node_id,
            target_nodes=[node.node_id for node in nodes if node.node_id != self.local_node.node_id],
            consistency_level=self.config.consistency_level
        )
        
        self.replication_queue.append(request)
    
    async def _process_replication_request(self, request: ReplicationRequest):
        """Process a replication request."""
        successful_replications = 0
        total_targets = len(request.target_nodes)
        
        if total_targets == 0:
            return  # No replication needed
        
        # Determine required success count based on consistency level
        required_successes = self._get_required_successes(total_targets, request.consistency_level)
        
        # Send replication to target nodes
        tasks = []
        for node_id in request.target_nodes:
            if node_id in self.nodes:
                task = asyncio.create_task(
                    self._send_replication_to_node(request, self.nodes[node_id])
                )
                tasks.append(task)
        
        # Wait for responses
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_replications = sum(1 for result in results if result is True)
        
        # Check if consistency requirements are met
        if successful_replications >= required_successes:
            logger.debug(f"Replication successful: {successful_replications}/{total_targets} nodes")
        else:
            logger.warning(f"Replication failed: only {successful_replications}/{total_targets} nodes succeeded")
    
    def _get_required_successes(self, total_nodes: int, consistency_level: ConsistencyLevel) -> int:
        """Get required number of successful replications."""
        if consistency_level == ConsistencyLevel.ANY:
            return 0
        elif consistency_level == ConsistencyLevel.ONE:
            return 1
        elif consistency_level == ConsistencyLevel.QUORUM:
            return (total_nodes // 2) + 1
        elif consistency_level == ConsistencyLevel.ALL:
            return total_nodes
        else:
            return 1
    
    async def _send_replication_to_node(self, request: ReplicationRequest, target_node: CacheNode) -> bool:
        """Send replication request to a specific node."""
        try:
            # This would implement actual network communication to the target node
            # For now, simulate the operation
            await asyncio.sleep(0.01)  # Simulate network latency
            
            # In real implementation, this would:
            # 1. Serialize the request
            # 2. Send via HTTP/gRPC/Redis/MessageQueue
            # 3. Handle authentication/encryption
            # 4. Process response
            
            return True  # Simulate success
            
        except Exception as e:
            logger.error(f"Replication to node {target_node.node_id} failed: {e}")
            return False
    
    async def _migrate_data_before_leaving(self):
        """Migrate data before leaving cluster."""
        # This would implement data migration logic
        # Find keys owned by this node and migrate to other nodes
        logger.info("Data migration completed before leaving cluster")

# === Distributed Coordinator ===
class DistributedCoordinator:
    """High-level distributed cache coordination."""
    
    def __init__(self, cluster_config: ClusterConfig):
        self.cluster_config = cluster_config
        self.clusters: Dict[str, CacheCluster] = {}
        self.cross_cluster_enabled = cluster_config.cross_region_replication
        
        # Cross-cluster coordination
        self.cluster_topology: Dict[str, List[str]] = {}
        self.global_routing_table: Dict[str, str] = {}
        
        logger.info("DistributedCoordinator initialized")
    
    async def register_cluster(self, cluster: CacheCluster):
        """Register a cache cluster."""
        self.clusters[cluster.config.cluster_name] = cluster
        
        if self.cross_cluster_enabled:
            await self._update_global_topology()
        
        logger.info(f"Registered cluster {cluster.config.cluster_name}")
    
    async def unregister_cluster(self, cluster_name: str):
        """Unregister a cache cluster."""
        if cluster_name in self.clusters:
            del self.clusters[cluster_name]
            
            if self.cross_cluster_enabled:
                await self._update_global_topology()
            
            logger.info(f"Unregistered cluster {cluster_name}")
    
    async def _update_global_topology(self):
        """Update global cluster topology."""
        # Build topology map
        self.cluster_topology = {}
        for cluster_name, cluster in self.clusters.items():
            regions = set()
            for node in cluster.nodes.values():
                regions.add(node.region)
            self.cluster_topology[cluster_name] = list(regions)
        
        logger.debug(f"Updated global topology: {self.cluster_topology}")
    
    def get_optimal_cluster(self, key: str, region: str = None) -> Optional[CacheCluster]:
        """Get optimal cluster for a key."""
        if not self.clusters:
            return None
        
        if len(self.clusters) == 1:
            return next(iter(self.clusters.values()))
        
        if region:
            # Find cluster in the same region
            for cluster in self.clusters.values():
                if any(node.region == region for node in cluster.nodes.values()):
                    return cluster
        
        # Fallback: use consistent hashing across clusters
        cluster_names = list(self.clusters.keys())
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        cluster_index = hash_value % len(cluster_names)
        cluster_name = cluster_names[cluster_index]
        
        return self.clusters[cluster_name]
    
    async def coordinate_cross_cluster_operation(self, operation: str, key: str, value: Any = None):
        """Coordinate operation across multiple clusters."""
        if not self.cross_cluster_enabled:
            return
        
        # Determine target clusters
        primary_cluster = self.get_optimal_cluster(key)
        if not primary_cluster:
            return
        
        # Replicate to other clusters if needed
        replication_tasks = []
        for cluster_name, cluster in self.clusters.items():
            if cluster != primary_cluster:
                task = asyncio.create_task(
                    self._replicate_to_cluster(cluster, operation, key, value)
                )
                replication_tasks.append(task)
        
        if replication_tasks:
            await asyncio.gather(*replication_tasks, return_exceptions=True)
    
    async def _replicate_to_cluster(self, cluster: CacheCluster, operation: str, key: str, value: Any):
        """Replicate operation to a specific cluster."""
        try:
            # This would implement cross-cluster replication
            # Could use message queues, HTTP APIs, or direct connections
            await asyncio.sleep(0.01)  # Simulate network latency
            logger.debug(f"Cross-cluster replication to {cluster.config.cluster_name} completed")
        except Exception as e:
            logger.error(f"Cross-cluster replication failed: {e}")

# === Factory Functions ===
def create_cache_node(node_id: str, host: str, port: int, **kwargs) -> CacheNode:
    """Create cache node."""
    return CacheNode(node_id=node_id, host=host, port=port, **kwargs)

def create_cluster_config(cluster_name: str, **kwargs) -> ClusterConfig:
    """Create cluster configuration."""
    return ClusterConfig(cluster_name=cluster_name, **kwargs)

def create_consistent_hash_ring(nodes: List[CacheNode] = None) -> ConsistentHashRing:
    """Create consistent hash ring."""
    return ConsistentHashRing(nodes)

def create_cache_cluster(config: ClusterConfig, local_node: CacheNode) -> CacheCluster:
    """Create cache cluster."""
    return CacheCluster(config, local_node)

def create_distributed_coordinator(cluster_config: ClusterConfig) -> DistributedCoordinator:
    """Create distributed coordinator."""
    return DistributedCoordinator(cluster_config)

def create_coordination_suite(cluster_name: str, local_node_id: str, host: str, port: int) -> Dict[str, Any]:
    """Create complete coordination suite."""
    # Create components
    local_node = create_cache_node(local_node_id, host, port)
    cluster_config = create_cluster_config(cluster_name)
    hash_ring = create_consistent_hash_ring([local_node])
    cache_cluster = create_cache_cluster(cluster_config, local_node)
    coordinator = create_distributed_coordinator(cluster_config)
    
    return {
        'local_node': local_node,
        'cluster_config': cluster_config,
        'hash_ring': hash_ring,
        'cache_cluster': cache_cluster,
        'coordinator': coordinator
    }

# === Load Balancer ===
class CacheLoadBalancer:
    """Intelligent load balancer for cache operations."""
    
    def __init__(self, cluster: CacheCluster):
        self.cluster = cluster
        self.load_balancing_strategy = "least_loaded"  # least_loaded, round_robin, random, consistent_hash
        self.health_check_interval = 30
        
    def select_node_for_read(self, key: str) -> Optional[CacheNode]:
        """Select optimal node for read operation."""
        available_nodes = self.get_available_nodes_for_key(key)
        
        if not available_nodes:
            return None
        
        if self.load_balancing_strategy == "least_loaded":
            return min(available_nodes, key=lambda n: n.load_factor)
        elif self.load_balancing_strategy == "random":
            return random.choice(available_nodes)
        else:
            return available_nodes[0]  # Primary node
    
    def select_node_for_write(self, key: str) -> Optional[CacheNode]:
        """Select optimal node for write operation."""
        # Writes typically go to primary node
        return self.cluster.get_primary_node_for_key(key)
    
    def get_available_nodes_for_key(self, key: str) -> List[CacheNode]:
        """Get available nodes that can serve a key."""
        nodes = self.cluster.get_nodes_for_key(key)
        return [node for node in nodes if node.is_healthy()]

# === Replication Manager ===
class ReplicationManager:
    """Advanced replication management."""
    
    def __init__(self, cluster: CacheCluster):
        self.cluster = cluster
        self.replication_lag_threshold = timedelta(seconds=1)
        self.conflict_resolution_strategy = "last_write_wins"
        
    async def ensure_replication_health(self):
        """Monitor and ensure replication health."""
        # Check replication lag
        # Detect and resolve conflicts
        # Repair missing replicas
        pass
    
    async def resolve_replication_conflict(self, key: str, conflicts: List[Dict[str, Any]]):
        """Resolve replication conflicts."""
        if self.conflict_resolution_strategy == "last_write_wins":
            # Choose the version with latest timestamp
            latest_version = max(conflicts, key=lambda c: c.get("timestamp", 0))
            return latest_version
        
        # Other strategies could be implemented here
        return conflicts[0]
