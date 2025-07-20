from pydantic import BaseModel, Field
from typing import Optional

class EnvironmentConfig(BaseModel):
    """
    Environment configuration for Spotify AI Agent (dev, staging, prod, etc.)
    """
    env: str = Field("development", description="Environment name (development, staging, production)")
    debug: bool = Field(True, description="Enable debug mode")
    version: str = Field("1.0.0", description="App version")
    build: Optional[str] = Field(None, description="Build identifier (git SHA, CI/CD tag, etc.)")
    region: Optional[str] = Field(None, description="Deployment region (eu-west-1, us-east-1, etc.)")
    timezone: str = Field("UTC", description="Default timezone")

# Example usage:
# from .environment_config import EnvironmentConfig
# env_conf = EnvironmentConfig(env="production", debug=False, version="1.2.3", build="sha256:...", region="eu-west-1")
