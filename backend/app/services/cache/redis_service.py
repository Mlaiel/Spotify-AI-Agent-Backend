import redis
import logging
import os

logger = logging.getLogger("redis_service")

class RedisCacheService:
    """
    Service Redis sécurisé, pooling, scripts Lua, monitoring, backup, failover, clustering.
    Utilisé pour le cache IA, analytics, Spotify, scoring, etc.
    """
    def __init__(self, url: str = None, db: int = 0, password: str = None, ssl: bool = False, cluster: bool = False):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.ssl = ssl
        self.cluster = cluster
        if cluster:
            from redis.cluster import RedisCluster
            self.client = RedisCluster.from_url(self.url, password=self.password, decode_responses=True, ssl=self.ssl)
        else:
            self.pool = redis.ConnectionPool.from_url(self.url, db=self.db, password=self.password, decode_responses=True, ssl=self.ssl)
            self.client = redis.Redis(connection_pool=self.pool)
        logger.info(f"RedisCacheService connecté à {self.url} (db={self.db}, ssl={self.ssl}, cluster={self.cluster})")
    def get(self, key):
        return self.client.get(key)
    def set(self, key, value, ex=None):
        self.client.set(key, value, ex=ex)
    def delete(self, key):
        self.client.delete(key)
    def flushdb(self):
        self.client.flushdb()
    def run_lua_script(self, script: str, keys=None, args=None):
        return self.client.eval(script, len(keys or []), *(keys or []), *(args or []))
    def info(self):
        return self.client.info()
    def backup(self, path: str = "/tmp/redis_backup.rdb"):
        # Sauvegarde manuelle (exemple)
        self.client.save()
        os.system(f"cp /data/dump.rdb {path}")
        logger.info(f"Backup Redis sauvegardé à {path}")
    def monitor(self):
        # Monitoring avancé (exemple)
        return self.client.monitor()
