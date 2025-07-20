"""
Cluster Manager Redis
====================
- Gestion cluster, auto-discovery, failover, monitoring
- Sécurité, logs, audit
"""

import os
import logging
from redis.cluster import RedisCluster

class RedisClusterManager:
    def __init__(self, startup_nodes=None):
        if os.getenv("TESTING", "0") == "1":
            raise RuntimeError("Redis désactivé en mode test (TESTING=1)")
        # Correction du format pour RedisCluster : liste de dicts avec 'host' et 'port'
        self.startup_nodes = startup_nodes or [
            {"host": os.getenv("REDIS_HOST", "localhost"), "port": int(os.getenv("REDIS_PORT", 6379))}
        ]
        # RedisCluster attend une liste de dicts, mais chaque dict doit avoir 'host' et 'port' (pas 'name')
        self.client = RedisCluster(startup_nodes=self.startup_nodes, decode_responses=True, skip_full_coverage_check=True)

    def get_client(self):
        return self.client

    def health_check(self):
        try:
            pong = self.client.ping()
            logging.info(f"Redis Cluster health: {pong}")
            return pong
        except Exception as e:
            logging.error(f"Redis Cluster health check failed: {e}")
            return False

    def get_cluster_info(self):
        try:
            info = self.client.cluster('info')
            logging.info(f"Cluster info: {info}")
            return info
        except Exception as e:
            logging.error(f"Cluster info failed: {e}")
            return None

# Factory pour DI

def get_cluster_manager():
    if os.getenv("TESTING", "0") == "1":
        raise RuntimeError("Redis désactivé en mode test (TESTING=1)")
    return RedisClusterManager()

def get_cluster():
    return get_cluster_manager()
