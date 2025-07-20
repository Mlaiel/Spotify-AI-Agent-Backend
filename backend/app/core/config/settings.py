"""
Configuration settings for Spotify AI Agent
"""
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Main application settings"""
    
    # Security
    secret_key: str = Field(
        "spotify-ai-agent-secret-key-change-in-production",
        validation_alias="SECRET_KEY"
    )
    algorithm: str = Field(
        "HS256",
        validation_alias="ALGORITHM"
    )

    # Database
    postgres_dsn: str = Field(
        "postgresql://user:pass@localhost/dbname",
        validation_alias="DATABASE_URL"
    )
    
    # Redis
    redis_url: str = Field(
        "redis://localhost:6379",
        validation_alias="REDIS_URL"
    )
    
    # Spotify API
    spotify_client_id: str = Field(
        "",
        validation_alias="SPOTIFY_CLIENT_ID"
    )
    spotify_client_secret: str = Field(
        "",
        validation_alias="SPOTIFY_CLIENT_SECRET"
    )
    
    # Application
    app_name: str = "Spotify AI Agent"
    app_version: str = "1.0.0"
    debug: bool = Field(False, validation_alias="DEBUG")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        validation_alias="CORS_ORIGINS"
    )
    
    # JWT
    access_token_expire_minutes: int = Field(
        30,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings

# Example usage:
# from .settings import settings
# db_url = settings.postgres_dsn
