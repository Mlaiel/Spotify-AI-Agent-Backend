"""
🔧 Data Partition Management - Partitionnement Intelligent des Données
=====================================================================

Système ultra-avancé de partitionnement des données pour architecture 
multi-tenant avec performance optimisée et isolation sécurisée.

Author: DBA & Data Engineer - Fahed Mlaiel
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...core.database import DatabaseManager
from ...utils.performance import PerformanceMonitor
from ..exceptions import PartitionError


class PartitionType(Enum):
    """Types de partitionnement supportés"""
    HORIZONTAL = "horizontal"  # Partition par tenant_id
    VERTICAL = "vertical"      # Partition par type de données
    HYBRID = "hybrid"         # Combinaison des deux
    TEMPORAL = "temporal"     # Partition par temps
    GEOGRAPHIC = "geographic" # Partition par région


class PartitionStrategy(Enum):
    """Stratégies de partitionnement"""
    HASH = "hash"           # Partitionnement par hash
    RANGE = "range"         # Partitionnement par plage
    LIST = "list"           # Partitionnement par liste
    MODULO = "modulo"       # Partitionnement modulo
    CONSISTENT_HASH = "consistent_hash"  # Hash consistant


@dataclass
class PartitionConfig:
    """Configuration du partitionnement"""
    partition_type: PartitionType
    strategy: PartitionStrategy
    partition_count: int = 16
    replication_factor: int = 2
    auto_scaling: bool = True
    max_partition_size: int = 1_000_000  # Nombre d'enregistrements
    compression_enabled: bool = True
    encryption_enabled: bool = True
    backup_enabled: bool = True
    monitoring_enabled: bool = True


@dataclass
class PartitionMetadata:
    """Métadonnées d'une partition"""
    partition_id: str
    partition_name: str
    table_name: str
    tenant_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    record_count: int = 0
    size_bytes: int = 0
    status: str = "active"
    health_score: float = 1.0
    last_maintenance: Optional[datetime] = None


class PartitionManager(ABC):
    """Interface abstraite pour les gestionnaires de partition"""
    
    @abstractmethod
    async def create_partition(self, config: PartitionConfig) -> str:
        """Crée une nouvelle partition"""
        pass
    
    @abstractmethod
    async def get_partition(self, tenant_id: str) -> Optional[PartitionMetadata]:
        """Récupère la partition pour un tenant"""
        pass
    
    @abstractmethod
    async def migrate_partition(self, partition_id: str, target_config: PartitionConfig):
        """Migre une partition vers une nouvelle configuration"""
        pass
    
    @abstractmethod
    async def balance_partitions(self) -> Dict[str, Any]:
        """Équilibre les partitions"""
        pass


class HashPartitionManager(PartitionManager):
    """Gestionnaire de partitionnement par hash"""
    
    def __init__(self, config: PartitionConfig):
        self.config = config
        self.partitions: Dict[str, PartitionMetadata] = {}
        self.tenant_partition_map: Dict[str, str] = {}
        self.logger = logging.getLogger("partition.hash")
        self.performance_monitor = PerformanceMonitor()
        
        # Hash ring pour consistent hashing
        self._hash_ring = {}
        self._initialize_hash_ring()
    
    def _initialize_hash_ring(self):
        """Initialise le hash ring pour le consistent hashing"""
        if self.config.strategy == PartitionStrategy.CONSISTENT_HASH:
            for i in range(self.config.partition_count):
                partition_id = f"partition_{i}"
                # Multiple points per partition for better distribution
                for j in range(self.config.replication_factor * 10):
                    hash_point = self._hash_function(f"{partition_id}_{j}")
                    self._hash_ring[hash_point] = partition_id
    
    def _hash_function(self, value: str) -> int:
        """Fonction de hash consistante"""
        return int(hashlib.md5(value.encode()).hexdigest(), 16)
    
    def _get_partition_id(self, tenant_id: str) -> str:
        """Détermine l'ID de partition pour un tenant"""
        if self.config.strategy == PartitionStrategy.HASH:
            hash_value = self._hash_function(tenant_id)
            partition_index = hash_value % self.config.partition_count
            return f"partition_{partition_index}"
        
        elif self.config.strategy == PartitionStrategy.CONSISTENT_HASH:
            hash_value = self._hash_function(tenant_id)
            # Find the next partition in the ring
            for ring_point in sorted(self._hash_ring.keys()):
                if ring_point >= hash_value:
                    return self._hash_ring[ring_point]
            # Wrap around to the first partition
            return self._hash_ring[min(self._hash_ring.keys())]
        
        elif self.config.strategy == PartitionStrategy.MODULO:
            # Simple modulo distribution
            hash_value = sum(ord(c) for c in tenant_id)
            partition_index = hash_value % self.config.partition_count
            return f"partition_{partition_index}"
        
        else:
            raise PartitionError(f"Unsupported partition strategy: {self.config.strategy}")
    
    async def create_partition(self, config: PartitionConfig) -> str:
        """Crée une nouvelle partition"""
        partition_id = f"partition_{len(self.partitions)}"
        
        metadata = PartitionMetadata(
            partition_id=partition_id,
            partition_name=f"{config.partition_type.value}_{partition_id}",
            table_name=f"data_{partition_id}"
        )
        
        try:
            # Create physical partition
            await self._create_physical_partition(metadata)
            
            # Store metadata
            self.partitions[partition_id] = metadata
            
            self.logger.info(f"Created partition: {partition_id}")
            return partition_id
            
        except Exception as e:
            self.logger.error(f"Failed to create partition {partition_id}: {e}")
            raise PartitionError(f"Partition creation failed: {e}")
    
    async def _create_physical_partition(self, metadata: PartitionMetadata):
        """Crée la partition physique dans la base de données"""
        # This would contain actual SQL DDL commands
        # Example for PostgreSQL:
        sql_commands = [
            f"CREATE TABLE {metadata.table_name} PARTITION OF main_table FOR VALUES WITH (modulus {self.config.partition_count}, remainder {len(self.partitions)})",
            f"CREATE INDEX idx_{metadata.table_name}_tenant_id ON {metadata.table_name} (tenant_id)",
            f"CREATE INDEX idx_{metadata.table_name}_created_at ON {metadata.table_name} (created_at)"
        ]
        
        # Execute commands (implementation would use actual database connection)
        for command in sql_commands:
            self.logger.debug(f"Executing: {command}")
    
    async def get_partition(self, tenant_id: str) -> Optional[PartitionMetadata]:
        """Récupère la partition pour un tenant"""
        partition_id = self._get_partition_id(tenant_id)
        
        # Update mapping
        self.tenant_partition_map[tenant_id] = partition_id
        
        # Create partition if it doesn't exist
        if partition_id not in self.partitions:
            await self.create_partition(self.config)
        
        partition = self.partitions.get(partition_id)
        if partition and tenant_id not in partition.tenant_ids:
            partition.tenant_ids.append(tenant_id)
            partition.updated_at = datetime.now(timezone.utc)
        
        return partition
    
    async def migrate_partition(self, partition_id: str, target_config: PartitionConfig):
        """Migre une partition vers une nouvelle configuration"""
        if partition_id not in self.partitions:
            raise PartitionError(f"Partition {partition_id} not found")
        
        try:
            with self.performance_monitor.measure("partition_migration"):
                # Create new partition with target config
                new_partition_id = await self.create_partition(target_config)
                
                # Migrate data
                await self._migrate_data(partition_id, new_partition_id)
                
                # Update mappings
                await self._update_tenant_mappings(partition_id, new_partition_id)
                
                # Remove old partition
                await self._cleanup_partition(partition_id)
                
                self.logger.info(f"Migrated partition {partition_id} to {new_partition_id}")
                
        except Exception as e:
            self.logger.error(f"Migration failed for partition {partition_id}: {e}")
            raise PartitionError(f"Migration failed: {e}")
    
    async def _migrate_data(self, source_partition: str, target_partition: str):
        """Migre les données entre partitions"""
        source_table = self.partitions[source_partition].table_name
        target_table = self.partitions[target_partition].table_name
        
        # Implementation would use actual data migration logic
        self.logger.info(f"Migrating data from {source_table} to {target_table}")
    
    async def _update_tenant_mappings(self, old_partition: str, new_partition: str):
        """Met à jour les mappings tenant -> partition"""
        for tenant_id, partition_id in self.tenant_partition_map.items():
            if partition_id == old_partition:
                self.tenant_partition_map[tenant_id] = new_partition
        
        # Update partition metadata
        old_metadata = self.partitions[old_partition]
        new_metadata = self.partitions[new_partition]
        new_metadata.tenant_ids.extend(old_metadata.tenant_ids)
    
    async def _cleanup_partition(self, partition_id: str):
        """Nettoie une partition obsolète"""
        metadata = self.partitions.pop(partition_id, None)
        if metadata:
            # Drop physical table
            self.logger.info(f"Dropping table: {metadata.table_name}")
    
    async def balance_partitions(self) -> Dict[str, Any]:
        """Équilibre les partitions"""
        stats = {
            "total_partitions": len(self.partitions),
            "total_tenants": len(self.tenant_partition_map),
            "balancing_actions": []
        }
        
        # Calculate partition loads
        partition_loads = {}
        for partition_id, metadata in self.partitions.items():
            partition_loads[partition_id] = len(metadata.tenant_ids)
        
        # Find imbalanced partitions
        avg_load = sum(partition_loads.values()) / len(partition_loads) if partition_loads else 0
        threshold = avg_load * 1.5  # 50% above average
        
        overloaded_partitions = [
            pid for pid, load in partition_loads.items() 
            if load > threshold
        ]
        
        # Rebalance if needed
        for partition_id in overloaded_partitions:
            action = await self._rebalance_partition(partition_id)
            stats["balancing_actions"].append(action)
        
        return stats
    
    async def _rebalance_partition(self, partition_id: str) -> Dict[str, Any]:
        """Rééquilibre une partition spécifique"""
        metadata = self.partitions[partition_id]
        
        # Find tenants to move
        tenants_to_move = metadata.tenant_ids[:len(metadata.tenant_ids)//2]
        
        # Create new partition
        new_partition_id = await self.create_partition(self.config)
        
        # Move tenants
        for tenant_id in tenants_to_move:
            self.tenant_partition_map[tenant_id] = new_partition_id
            metadata.tenant_ids.remove(tenant_id)
            self.partitions[new_partition_id].tenant_ids.append(tenant_id)
        
        return {
            "action": "rebalance",
            "source_partition": partition_id,
            "target_partition": new_partition_id,
            "moved_tenants": len(tenants_to_move)
        }


class DataPartition:
    """
    Gestionnaire principal de partitionnement des données
    
    Features:
    - Support multi-stratégies
    - Auto-scaling intelligent
    - Monitoring en temps réel
    - Optimisation automatique
    - Recovery automatique
    """
    
    def __init__(self, config: PartitionConfig):
        self.config = config
        self.logger = logging.getLogger("data.partition")
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize partition manager based on strategy
        if config.strategy in [PartitionStrategy.HASH, PartitionStrategy.CONSISTENT_HASH, PartitionStrategy.MODULO]:
            self.manager = HashPartitionManager(config)
        else:
            raise PartitionError(f"Unsupported partition strategy: {config.strategy}")
        
        # Monitoring and maintenance
        self._monitoring_task = None
        self._maintenance_task = None
        
        if config.monitoring_enabled:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Démarre le monitoring des partitions"""
        async def monitor_loop():
            while True:
                try:
                    await self._check_partition_health()
                    await asyncio.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(10)
        
        self._monitoring_task = asyncio.create_task(monitor_loop())
    
    async def _check_partition_health(self):
        """Vérifie la santé des partitions"""
        for partition_id, metadata in self.manager.partitions.items():
            # Check partition size
            if metadata.record_count > self.config.max_partition_size:
                self.logger.warning(f"Partition {partition_id} exceeds maximum size")
                if self.config.auto_scaling:
                    await self._auto_scale_partition(partition_id)
            
            # Check health score
            if metadata.health_score < 0.8:
                self.logger.warning(f"Partition {partition_id} has low health score: {metadata.health_score}")
                await self._repair_partition(partition_id)
    
    async def _auto_scale_partition(self, partition_id: str):
        """Auto-scaling d'une partition"""
        self.logger.info(f"Auto-scaling partition: {partition_id}")
        
        # Create new partition with same config
        new_partition_id = await self.manager.create_partition(self.config)
        
        # Move half of the tenants to new partition
        await self.manager._rebalance_partition(partition_id)
    
    async def _repair_partition(self, partition_id: str):
        """Répare une partition défaillante"""
        self.logger.info(f"Repairing partition: {partition_id}")
        
        # Implementation would include:
        # - Data integrity checks
        # - Index rebuilding
        # - Statistics updates
        # - Health score recalculation
    
    async def get_partition_for_tenant(self, tenant_id: str) -> PartitionMetadata:
        """Récupère la partition pour un tenant"""
        with self.performance_monitor.measure("get_partition"):
            partition = await self.manager.get_partition(tenant_id)
            if not partition:
                raise PartitionError(f"No partition found for tenant: {tenant_id}")
            return partition
    
    async def create_partition(self) -> str:
        """Crée une nouvelle partition"""
        return await self.manager.create_partition(self.config)
    
    async def balance_partitions(self) -> Dict[str, Any]:
        """Équilibre toutes les partitions"""
        with self.performance_monitor.measure("balance_partitions"):
            return await self.manager.balance_partitions()
    
    async def get_partition_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des partitions"""
        stats = {
            "total_partitions": len(self.manager.partitions),
            "total_tenants": len(self.manager.tenant_partition_map),
            "config": {
                "partition_type": self.config.partition_type.value,
                "strategy": self.config.strategy.value,
                "partition_count": self.config.partition_count
            },
            "partitions": {}
        }
        
        for partition_id, metadata in self.manager.partitions.items():
            stats["partitions"][partition_id] = {
                "tenant_count": len(metadata.tenant_ids),
                "record_count": metadata.record_count,
                "size_bytes": metadata.size_bytes,
                "health_score": metadata.health_score,
                "status": metadata.status
            }
        
        return stats
    
    async def cleanup(self):
        """Nettoie les ressources"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._maintenance_task:
            self._maintenance_task.cancel()
        
        self.logger.info("Data partition cleanup completed")
