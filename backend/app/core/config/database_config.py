from pydantic import BaseModel, Field, SecretStr, AnyUrl
from typing import Optional

class PostgresConfig(BaseModel):
    dsn: AnyUrl = Field(..., description="PostgreSQL DSN")
    pool_size: int = Field(10, description="Connection pool size")
    timeout: int = Field(30, description="Connection timeout (s)")
    ssl_mode: Optional[str] = Field(None, description="SSL mode (require, disable, etc.)")

class MongoDBConfig(BaseModel):
    url: AnyUrl = Field(..., description="MongoDB connection URL")
    db_name: str = Field("spotify_ai", description="Default database name")
    analytics_collection: str = Field("analytics", description="Analytics collection name")

class RedisConfig(BaseModel):
    url: AnyUrl = Field(..., description="Redis connection URL")
    db: int = Field(0, description="Redis DB index")
    socket_timeout: int = Field(5, description="Socket timeout (s)")

class DatabaseConfig(BaseModel):
    postgres: PostgresConfig
    mongodb: MongoDBConfig
    redis: RedisConfig

# Example usage:
# from .database_config import DatabaseConfig, PostgresConfig, MongoDBConfig, RedisConfig
# db_conf = DatabaseConfig()
#     postgres=PostgresConfig(dsn="postgresql://user:pass@localhost/db"),
#     mongodb=MongoDBConfig(url="mongodb://localhost:27017"),
#     redis=RedisConfig(url="redis://localhost:6379/0")
# )
