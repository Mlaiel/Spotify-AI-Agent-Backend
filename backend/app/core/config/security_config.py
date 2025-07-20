from pydantic import BaseModel, Field, SecretStr
from typing import List, Optional

class SecurityConfig(BaseModel):
    secret_key: SecretStr = Field(..., description="Main secret key for JWT and encryption")
    algorithm: str = Field("HS256", description="JWT algorithm")
    allowed_origins: List[str] = Field(["http://localhost", "http://127.0.0.1"], description="CORS allowed origins")
    cors_enabled: bool = Field(True, description="Enable CORS middleware")
    csp_policy: Optional[str] = Field(None, description="Content Security Policy header value")
    brute_force_protection: bool = Field(True, description="Enable brute-force protection")
    max_login_attempts: int = Field(5, description="Max login attempts before lockout")
    lockout_time: int = Field(600, description="Lockout time in seconds")

# Example usage:
# from .security_config import SecurityConfig
# sec_conf = SecurityConfig(secret_key="...", allowed_origins=[...], cors_enabled=True)
