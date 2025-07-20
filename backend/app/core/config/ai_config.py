from pydantic import BaseModel, Field, SecretStr
from typing import Optional, List

class AIModelConfig(BaseModel):
    name: str = Field(..., description="Model name (e.g. gpt-4, musicgen, custom-ml)")
    provider: str = Field(..., description="Provider (huggingface, openai, custom, etc.)")
    api_url: Optional[str] = Field(None, description="API endpoint for inference")
    api_key: Optional[SecretStr] = Field(None, description="API key/token (if needed)")
    enabled: bool = Field(True, description="Is this model active?")
    max_tokens: Optional[int] = Field(4096, description="Max tokens for generation")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature")

class AIConfig(BaseModel):
    """
    Advanced AI/ML configuration for the Spotify AI Agent (models, providers, security, etc.)
    """
    default_model: str = Field("musicgen", description="Default AI model")
    models: List[AIModelConfig] = Field(default_factory=list, description="List of available AI/ML models")
    moderation_enabled: bool = Field(True, description="Enable AI moderation for chat/content")
    moderation_threshold: float = Field(0.7, description="Toxicity threshold for moderation")

# Example usage:
# from .ai_config import AIConfig, AIModelConfig
# ai_conf = AIConfig(
#     default_model="musicgen",)
#     models=[AIModelConfig(name="musicgen", provider="huggingface", api_url="https://api.hf.com/...", api_key=...), ...]
# )
