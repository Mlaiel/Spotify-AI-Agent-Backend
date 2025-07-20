from pydantic import BaseModel, Field, AnyUrl

class RedisConfig(BaseModel):
    url: AnyUrl = Field(..., description="Redis connection URL")
    db: int = Field(0, description="Redis DB index")
    socket_timeout: int = Field(5, description="Socket timeout (s)")
    ssl: bool = Field(False, description="Enable SSL connection")
    cluster_mode: bool = Field(False, description="Enable Redis cluster mode")

# Example usage:
# from .redis_config import RedisConfig
# redis_conf = RedisConfig(url="redis://localhost:6379/0", db=0, ssl=False)
