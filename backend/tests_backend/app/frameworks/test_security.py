"""
🧪 Tests Security Framework - Enterprise Security
===============================================

Tests complets du framework de sécurité avec:
- JWT/OAuth2 Authentication
- Cryptographie RSA/Fernet
- Rate Limiting Redis
- Audit Logging
- Security Monitoring

Développé par: Security Specialist
"""

import pytest
import asyncio
import jwt
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.app.frameworks.security import (
    SecurityFramework,
    JWTManager,
    CryptoManager,
    RateLimiter,
    AuditLogger,
    OAuth2Manager,
    SecurityConfig,
    UserCredentials,
    SecurityEvent,
    SecurityLevel
)
from backend.app.frameworks import TEST_CONFIG, clean_frameworks, logger


@pytest.fixture
def security_config():
    """Configuration sécurité pour les tests."""
    return SecurityConfig(
        jwt_secret_key=TEST_CONFIG["test_jwt_secret"],
        jwt_access_token_expire_minutes=30,
        jwt_refresh_token_expire_days=7,
        redis_url=TEST_CONFIG["test_redis_url"],
        rate_limit_requests=100,
        rate_limit_window_seconds=60,
        enable_audit_logging=True,
        audit_log_level=SecurityLevel.INFO,
        encryption_key=Fernet.generate_key(),
        oauth2_spotify_client_id="test_spotify_client",
        oauth2_spotify_client_secret="test_spotify_secret"
    )


@pytest.fixture
def sample_user_credentials():
    """Identifiants utilisateur d'exemple."""
    return UserCredentials(
        user_id="test_user_123",
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$test_hashed_password",
        roles=["user", "premium"],
        permissions=["read:profile", "write:playlists"]
    )


@pytest.fixture
def mock_redis():
    """Mock Redis pour les tests."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    redis_mock.delete.return_value = True
    return redis_mock


@pytest.mark.security
class TestSecurityConfig:
    """Tests de la configuration sécurité."""
    
    def test_security_config_creation(self):
        """Test création configuration sécurité."""
        config = SecurityConfig(
            jwt_secret_key="test-secret",
            redis_url="redis://localhost:6379/0"
        )
        
        assert config.jwt_secret_key == "test-secret"
        assert config.jwt_access_token_expire_minutes == 60
        assert config.rate_limit_requests == 1000
        assert config.enable_audit_logging is True
        
    def test_security_config_validation(self):
        """Test validation configuration."""
        # Secret JWT trop court
        with pytest.raises(ValueError, match="JWT secret key too short"):
            SecurityConfig(jwt_secret_key="short")
            
        # URL Redis invalide
        with pytest.raises(ValueError, match="Invalid Redis URL"):
            SecurityConfig(
                jwt_secret_key="valid-secret-key-long-enough",
                redis_url="invalid-redis-url"
            )
            
    def test_security_config_encryption_key_generation(self):
        """Test génération clé de chiffrement."""
        config = SecurityConfig(jwt_secret_key="test-secret-key-long-enough")
        
        # Si aucune clé fournie, elle doit être générée
        assert config.encryption_key is not None
        assert len(config.encryption_key) == 44  # Longueur clé Fernet base64
        
    def test_security_config_oauth2_validation(self):
        """Test validation OAuth2."""
        config = SecurityConfig(
            jwt_secret_key="test-secret-key-long-enough",
            oauth2_spotify_client_id="test_id",
            oauth2_spotify_client_secret="test_secret"
        )
        
        assert config.oauth2_spotify_client_id == "test_id"
        assert config.oauth2_spotify_client_secret == "test_secret"


@pytest.mark.security
class TestJWTManager:
    """Tests du gestionnaire JWT."""
    
    def test_jwt_manager_creation(self, security_config):
        """Test création gestionnaire JWT."""
        jwt_manager = JWTManager(security_config)
        
        assert jwt_manager.secret_key == security_config.jwt_secret_key
        assert jwt_manager.access_token_expire_minutes == 30
        assert jwt_manager.refresh_token_expire_days == 7
        
    def test_jwt_token_generation(self, security_config, sample_user_credentials):
        """Test génération tokens JWT."""
        jwt_manager = JWTManager(security_config)
        
        # Générer access token
        access_token = jwt_manager.create_access_token(
            user_id=sample_user_credentials.user_id,
            additional_claims={
                "username": sample_user_credentials.username,
                "roles": sample_user_credentials.roles
            }
        )
        
        assert access_token is not None
        assert isinstance(access_token, str)
        
        # Vérifier le token
        payload = jwt.decode(
            access_token, 
            security_config.jwt_secret_key, 
            algorithms=["HS256"]
        )
        
        assert payload["sub"] == sample_user_credentials.user_id
        assert payload["username"] == sample_user_credentials.username
        assert payload["roles"] == sample_user_credentials.roles
        assert "exp" in payload
        
    def test_jwt_token_validation(self, security_config, sample_user_credentials):
        """Test validation tokens JWT."""
        jwt_manager = JWTManager(security_config)
        
        # Créer token valide
        access_token = jwt_manager.create_access_token(
            user_id=sample_user_credentials.user_id
        )
        
        # Valider token
        payload = jwt_manager.validate_token(access_token)
        
        assert payload is not None
        assert payload["sub"] == sample_user_credentials.user_id
        
    def test_jwt_token_expiration(self, security_config):
        """Test expiration tokens JWT."""
        jwt_manager = JWTManager(security_config)
        
        # Créer token expiré
        expired_token = jwt.encode({
            "sub": "test_user",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }, security_config.jwt_secret_key, algorithm="HS256")
        
        # Validation doit échouer
        payload = jwt_manager.validate_token(expired_token)
        assert payload is None
        
    def test_jwt_refresh_token(self, security_config, sample_user_credentials):
        """Test refresh token."""
        jwt_manager = JWTManager(security_config)
        
        # Générer refresh token
        refresh_token = jwt_manager.create_refresh_token(
            user_id=sample_user_credentials.user_id
        )
        
        assert refresh_token is not None
        
        # Créer nouveau access token depuis refresh token
        new_access_token = jwt_manager.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        
        # Valider le nouveau token
        payload = jwt_manager.validate_token(new_access_token)
        assert payload["sub"] == sample_user_credentials.user_id
        
    def test_jwt_token_blacklist(self, security_config):
        """Test blacklist tokens."""
        jwt_manager = JWTManager(security_config)
        
        access_token = jwt_manager.create_access_token(user_id="test_user")
        
        # Ajouter à la blacklist
        jwt_manager.blacklist_token(access_token)
        
        # Validation doit échouer
        payload = jwt_manager.validate_token(access_token)
        assert payload is None


@pytest.mark.security
class TestCryptoManager:
    """Tests du gestionnaire cryptographique."""
    
    def test_crypto_manager_creation(self, security_config):
        """Test création gestionnaire crypto."""
        crypto_manager = CryptoManager(security_config)
        
        assert crypto_manager.fernet is not None
        assert crypto_manager.rsa_private_key is not None
        assert crypto_manager.rsa_public_key is not None
        
    def test_symmetric_encryption_decryption(self, security_config):
        """Test chiffrement/déchiffrement symétrique."""
        crypto_manager = CryptoManager(security_config)
        
        # Données à chiffrer
        sensitive_data = "Données sensibles utilisateur"
        
        # Chiffrement
        encrypted_data = crypto_manager.encrypt_data(sensitive_data)
        assert encrypted_data != sensitive_data
        assert isinstance(encrypted_data, bytes)
        
        # Déchiffrement
        decrypted_data = crypto_manager.decrypt_data(encrypted_data)
        assert decrypted_data == sensitive_data
        
    def test_asymmetric_encryption_decryption(self, security_config):
        """Test chiffrement/déchiffrement asymétrique."""
        crypto_manager = CryptoManager(security_config)
        
        # Données à chiffrer
        message = "Message secret"
        
        # Chiffrement avec clé publique
        encrypted_message = crypto_manager.encrypt_rsa(message.encode())
        assert encrypted_message != message.encode()
        
        # Déchiffrement avec clé privée
        decrypted_message = crypto_manager.decrypt_rsa(encrypted_message)
        assert decrypted_message.decode() == message
        
    def test_password_hashing_verification(self, security_config):
        """Test hachage/vérification mots de passe."""
        crypto_manager = CryptoManager(security_config)
        
        password = "mon_mot_de_passe_secret"
        
        # Hachage
        hashed_password = crypto_manager.hash_password(password)
        assert hashed_password != password
        assert hashed_password.startswith("$2b$")
        
        # Vérification
        is_valid = crypto_manager.verify_password(password, hashed_password)
        assert is_valid is True
        
        # Vérification avec mauvais mot de passe
        is_invalid = crypto_manager.verify_password("mauvais_password", hashed_password)
        assert is_invalid is False
        
    def test_digital_signature(self, security_config):
        """Test signature/vérification numérique."""
        crypto_manager = CryptoManager(security_config)
        
        # Message à signer
        message = "Document important à signer"
        
        # Signature
        signature = crypto_manager.sign_data(message.encode())
        assert signature is not None
        
        # Vérification signature valide
        is_valid = crypto_manager.verify_signature(message.encode(), signature)
        assert is_valid is True
        
        # Vérification signature invalide (message modifié)
        is_invalid = crypto_manager.verify_signature(b"Message modifie", signature)
        assert is_invalid is False


@pytest.mark.security
class TestRateLimiter:
    """Tests du limiteur de débit."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_creation(self, security_config):
        """Test création limiteur de débit."""
        with patch('redis.asyncio.from_url') as mock_redis_factory:
            mock_redis_factory.return_value = AsyncMock()
            
            rate_limiter = RateLimiter(security_config)
            await rate_limiter.initialize()
            
            assert rate_limiter.redis is not None
            assert rate_limiter.max_requests == security_config.rate_limit_requests
            assert rate_limiter.window_seconds == security_config.rate_limit_window_seconds
            
    @pytest.mark.asyncio
    async def test_rate_limiting_allow(self, security_config, mock_redis):
        """Test autorisation requête sous limite."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            rate_limiter = RateLimiter(security_config)
            await rate_limiter.initialize()
            
            # Simuler 5 requêtes existantes
            mock_redis.get.return_value = "5"
            
            # Vérifier autorisation (sous la limite)
            is_allowed = await rate_limiter.is_request_allowed("user_123")
            assert is_allowed is True
            
    @pytest.mark.asyncio
    async def test_rate_limiting_deny(self, security_config, mock_redis):
        """Test refus requête au-dessus limite."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            rate_limiter = RateLimiter(security_config)
            await rate_limiter.initialize()
            
            # Simuler limite atteinte
            mock_redis.get.return_value = str(security_config.rate_limit_requests)
            
            # Vérifier refus (limite atteinte)
            is_allowed = await rate_limiter.is_request_allowed("user_123")
            assert is_allowed is False
            
    @pytest.mark.asyncio
    async def test_rate_limiting_record_request(self, security_config, mock_redis):
        """Test enregistrement requête."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            rate_limiter = RateLimiter(security_config)
            await rate_limiter.initialize()
            
            mock_redis.incr.return_value = 1
            
            # Enregistrer requête
            await rate_limiter.record_request("user_123")
            
            # Vérifier appels Redis
            mock_redis.incr.assert_called()
            mock_redis.expire.assert_called()
            
    @pytest.mark.asyncio
    async def test_rate_limiting_get_remaining(self, security_config, mock_redis):
        """Test obtention requêtes restantes."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            rate_limiter = RateLimiter(security_config)
            await rate_limiter.initialize()
            
            mock_redis.get.return_value = "25"
            
            remaining = await rate_limiter.get_remaining_requests("user_123")
            assert remaining == 75  # 100 - 25


@pytest.mark.security
class TestAuditLogger:
    """Tests du logger d'audit."""
    
    @pytest.mark.asyncio
    async def test_audit_logger_creation(self, security_config):
        """Test création logger d'audit."""
        audit_logger = AuditLogger(security_config)
        
        assert audit_logger.log_level == security_config.audit_log_level
        assert audit_logger.events == []
        
    @pytest.mark.asyncio
    async def test_audit_log_security_event(self, security_config):
        """Test log événement sécurité."""
        audit_logger = AuditLogger(security_config)
        
        event = SecurityEvent(
            event_type="login_attempt",
            user_id="user_123",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 TestAgent",
            details={"success": True, "method": "password"},
            level=SecurityLevel.INFO
        )
        
        await audit_logger.log_event(event)
        
        assert len(audit_logger.events) == 1
        logged_event = audit_logger.events[0]
        assert logged_event.event_type == "login_attempt"
        assert logged_event.user_id == "user_123"
        
    @pytest.mark.asyncio
    async def test_audit_log_suspicious_activity(self, security_config):
        """Test log activité suspecte."""
        audit_logger = AuditLogger(security_config)
        
        # Événement suspect
        suspicious_event = SecurityEvent(
            event_type="multiple_login_failures",
            user_id="user_456",
            ip_address="10.0.0.1",
            details={"attempts": 10, "timespan": "5_minutes"},
            level=SecurityLevel.WARNING
        )
        
        await audit_logger.log_event(suspicious_event)
        
        # Vérifier alerte générée
        assert len(audit_logger.events) == 1
        event = audit_logger.events[0]
        assert event.level == SecurityLevel.WARNING
        
    @pytest.mark.asyncio
    async def test_audit_log_filtering_by_level(self, security_config):
        """Test filtrage par niveau."""
        # Logger avec niveau ERROR seulement
        security_config.audit_log_level = SecurityLevel.ERROR
        audit_logger = AuditLogger(security_config)
        
        # Événement INFO (ne devrait pas être loggé)
        info_event = SecurityEvent(
            event_type="user_action",
            user_id="user_123",
            level=SecurityLevel.INFO
        )
        
        await audit_logger.log_event(info_event)
        assert len(audit_logger.events) == 0
        
        # Événement ERROR (devrait être loggé)
        error_event = SecurityEvent(
            event_type="security_breach",
            user_id="user_123",
            level=SecurityLevel.ERROR
        )
        
        await audit_logger.log_event(error_event)
        assert len(audit_logger.events) == 1
        
    @pytest.mark.asyncio
    async def test_audit_log_export(self, security_config):
        """Test export logs d'audit."""
        audit_logger = AuditLogger(security_config)
        
        # Ajouter plusieurs événements
        events = [
            SecurityEvent("login", "user1", level=SecurityLevel.INFO),
            SecurityEvent("logout", "user1", level=SecurityLevel.INFO),
            SecurityEvent("failed_login", "user2", level=SecurityLevel.WARNING)
        ]
        
        for event in events:
            await audit_logger.log_event(event)
            
        # Export JSON
        export_data = audit_logger.export_logs(format="json")
        parsed_data = json.loads(export_data)
        
        assert len(parsed_data) == 3
        assert parsed_data[0]["event_type"] == "login"


@pytest.mark.security
class TestOAuth2Manager:
    """Tests du gestionnaire OAuth2."""
    
    def test_oauth2_manager_creation(self, security_config):
        """Test création gestionnaire OAuth2."""
        oauth2_manager = OAuth2Manager(security_config)
        
        assert oauth2_manager.spotify_client_id == security_config.oauth2_spotify_client_id
        assert oauth2_manager.spotify_client_secret == security_config.oauth2_spotify_client_secret
        
    def test_spotify_authorization_url(self, security_config):
        """Test génération URL autorisation Spotify."""
        oauth2_manager = OAuth2Manager(security_config)
        
        scopes = ["user-read-private", "playlist-read-private"]
        state = "random_state_string"
        
        auth_url = oauth2_manager.get_spotify_authorization_url(scopes, state)
        
        assert "https://accounts.spotify.com/authorize" in auth_url
        assert security_config.oauth2_spotify_client_id in auth_url
        assert "user-read-private" in auth_url
        assert "playlist-read-private" in auth_url
        assert state in auth_url
        
    @pytest.mark.asyncio
    async def test_spotify_token_exchange(self, security_config):
        """Test échange code contre token Spotify."""
        oauth2_manager = OAuth2Manager(security_config)
        
        # Mock réponse Spotify
        mock_response = {
            "access_token": "BQC4fUDa1...",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "AQBk7g...",
            "scope": "user-read-private playlist-read-private"
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.status_code = 200
            
            token_data = await oauth2_manager.exchange_spotify_code(
                code="authorization_code_from_callback",
                redirect_uri="http://localhost:8000/callback"
            )
            
        assert token_data["access_token"] == mock_response["access_token"]
        assert token_data["refresh_token"] == mock_response["refresh_token"]
        
    @pytest.mark.asyncio
    async def test_spotify_token_refresh(self, security_config):
        """Test refresh token Spotify."""
        oauth2_manager = OAuth2Manager(security_config)
        
        # Mock réponse refresh
        mock_response = {
            "access_token": "BQC4fUDa2...",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "user-read-private playlist-read-private"
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.status_code = 200
            
            new_token_data = await oauth2_manager.refresh_spotify_token(
                refresh_token="existing_refresh_token"
            )
            
        assert new_token_data["access_token"] == mock_response["access_token"]
        
    @pytest.mark.asyncio
    async def test_google_oauth2_flow(self, security_config):
        """Test flux OAuth2 Google."""
        oauth2_manager = OAuth2Manager(security_config)
        
        # Configuration Google (mock)
        oauth2_manager.google_client_id = "test_google_client_id"
        oauth2_manager.google_client_secret = "test_google_client_secret"
        
        # URL autorisation Google
        auth_url = oauth2_manager.get_google_authorization_url(
            scopes=["openid", "email", "profile"],
            state="google_state"
        )
        
        assert "https://accounts.google.com/o/oauth2/auth" in auth_url
        assert "test_google_client_id" in auth_url


@pytest.mark.security
class TestSecurityFramework:
    """Tests du framework de sécurité complet."""
    
    @pytest.mark.asyncio
    async def test_security_framework_initialization(self, security_config, clean_frameworks):
        """Test initialisation framework sécurité."""
        security_framework = SecurityFramework(security_config)
        
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            result = await security_framework.initialize()
            
        assert result is True
        assert security_framework.status.name == "RUNNING"
        assert security_framework.jwt_manager is not None
        assert security_framework.crypto_manager is not None
        assert security_framework.rate_limiter is not None
        assert security_framework.audit_logger is not None
        assert security_framework.oauth2_manager is not None
        
    @pytest.mark.asyncio
    async def test_security_framework_authenticate_user(self, security_config, sample_user_credentials, clean_frameworks):
        """Test authentification utilisateur."""
        security_framework = SecurityFramework(security_config)
        
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            await security_framework.initialize()
            
        # Mock validation JWT
        with patch.object(security_framework.jwt_manager, 'validate_token') as mock_validate:
            mock_validate.return_value = {
                "sub": sample_user_credentials.user_id,
                "username": sample_user_credentials.username,
                "roles": sample_user_credentials.roles
            }
            
            # Authentifier avec token valide
            access_token = "valid_jwt_token"
            user = await security_framework.authenticate_user(access_token)
            
        assert user is not None
        assert user["user_id"] == sample_user_credentials.user_id
        assert user["username"] == sample_user_credentials.username
        
    @pytest.mark.asyncio
    async def test_security_framework_login_workflow(self, security_config, sample_user_credentials, clean_frameworks):
        """Test workflow complet de login."""
        security_framework = SecurityFramework(security_config)
        
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            await security_framework.initialize()
            
        # Mock vérification mot de passe
        with patch.object(security_framework.crypto_manager, 'verify_password', return_value=True):
            # Login
            login_result = await security_framework.login_user(
                username=sample_user_credentials.username,
                password="correct_password",
                ip_address="192.168.1.100",
                user_agent="Test Agent"
            )
            
        assert login_result["success"] is True
        assert "access_token" in login_result
        assert "refresh_token" in login_result
        
        # Vérifier événement d'audit loggé
        assert len(security_framework.audit_logger.events) > 0
        login_event = security_framework.audit_logger.events[-1]
        assert login_event.event_type == "login_success"
        
    @pytest.mark.asyncio
    async def test_security_framework_rate_limiting_integration(self, security_config, clean_frameworks):
        """Test intégration rate limiting."""
        security_framework = SecurityFramework(security_config)
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "99"  # Sous la limite
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            await security_framework.initialize()
            
        # Vérifier autorisation requête
        is_allowed = await security_framework.check_rate_limit("user_123")
        assert is_allowed is True
        
        # Simuler limite atteinte
        mock_redis.get.return_value = str(security_config.rate_limit_requests)
        is_denied = await security_framework.check_rate_limit("user_123")
        assert is_denied is False
        
    @pytest.mark.asyncio
    async def test_security_framework_health_check(self, security_config, clean_frameworks):
        """Test health check framework sécurité."""
        security_framework = SecurityFramework(security_config)
        
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            await security_framework.initialize()
            
        health = await security_framework.health_check()
        
        assert health.status.name == "RUNNING"
        assert "Security framework" in health.message
        assert "jwt_manager" in health.details
        assert "crypto_manager" in health.details
        assert "rate_limiter" in health.details


@pytest.mark.security
@pytest.mark.integration
class TestSecurityFrameworkIntegration:
    """Tests d'intégration framework sécurité."""
    
    @pytest.mark.asyncio
    async def test_full_authentication_flow(self, security_config, clean_frameworks):
        """Test flux d'authentification complet."""
        security_framework = SecurityFramework(security_config)
        
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            await security_framework.initialize()
            
        # 1. Inscription utilisateur
        password = "secure_password_123"
        hashed_password = security_framework.crypto_manager.hash_password(password)
        
        user_data = {
            "user_id": "new_user_123",
            "username": "newuser",
            "email": "newuser@example.com",
            "hashed_password": hashed_password
        }
        
        # 2. Login
        with patch.object(security_framework.crypto_manager, 'verify_password', return_value=True):
            login_result = await security_framework.login_user(
                username=user_data["username"],
                password=password,
                ip_address="192.168.1.100"
            )
            
        assert login_result["success"] is True
        access_token = login_result["access_token"]
        
        # 3. Utilisation token pour authentification
        with patch.object(security_framework.jwt_manager, 'validate_token') as mock_validate:
            mock_validate.return_value = {
                "sub": user_data["user_id"],
                "username": user_data["username"]
            }
            
            authenticated_user = await security_framework.authenticate_user(access_token)
            
        assert authenticated_user["user_id"] == user_data["user_id"]
        
        # 4. Logout
        await security_framework.logout_user(access_token)
        
        # Vérifier événements d'audit
        events = security_framework.audit_logger.events
        assert len(events) >= 2  # Login + Logout
        assert any(event.event_type == "login_success" for event in events)
        assert any(event.event_type == "logout" for event in events)


@pytest.mark.security
@pytest.mark.performance
class TestSecurityFrameworkPerformance:
    """Tests de performance framework sécurité."""
    
    @pytest.mark.asyncio
    async def test_concurrent_authentication(self, security_config, clean_frameworks):
        """Test authentifications concurrentes."""
        security_framework = SecurityFramework(security_config)
        
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            await security_framework.initialize()
            
        # Mock validation JWT réussie
        with patch.object(security_framework.jwt_manager, 'validate_token') as mock_validate:
            mock_validate.return_value = {"sub": "user_123", "username": "testuser"}
            
            # Lancer authentifications concurrentes
            async def authenticate():
                return await security_framework.authenticate_user("valid_token")
                
            tasks = [authenticate() for _ in range(20)]
            results = await asyncio.gather(*tasks)
            
        # Toutes les authentifications doivent réussir
        assert len(results) == 20
        assert all(result is not None for result in results)
        
    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self, security_config, clean_frameworks):
        """Test performance rate limiting."""
        security_framework = SecurityFramework(security_config)
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "50"  # Sous la limite
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            await security_framework.initialize()
            
        # Test nombreuses vérifications de rate limit
        async def check_limit():
            return await security_framework.check_rate_limit("user_123")
            
        tasks = [check_limit() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        # Toutes les vérifications doivent être rapides
        assert len(results) == 100
        assert all(result is True for result in results)
