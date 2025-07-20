# 🧪 Tests TokenManager Ultra-Avancés
# =================================

import pytest
import pytest_asyncio
import asyncio
import time
import json
import jwt
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.security.auth.token_manager import (
    TokenManager, AccessToken, RefreshToken, APIKey, TokenType, 
    TokenStatus, TokenScope, JWTClaims, TokenRotationPolicy
)

from conftest import (
    TestDataFactory, TestUtils, PerformanceTestUtils, SecurityValidators,
    pytest_markers
)


@pytest_markers["unit"]
@pytest_markers["token"]
class TestTokenManager:
    """Tests unitaires pour TokenManager"""
    
    @pytest.mark.asyncio
    async def test_generate_access_token(self, token_manager):
        """Test génération de token d'accès"""
        user_data = TestDataFactory.create_test_user()
        
        with patch.object(token_manager, '_store_token') as mock_store:
            mock_store.return_value = True
            
            token = await token_manager.generate_access_token(
                user_id=user_data["user_id"],
                scopes=["read", "write"],
                expires_in=3600,
                client_id="test_client"
            )
        
        assert token.token_id is not None
        assert token.user_id == user_data["user_id"]
        assert token.token_type == TokenType.ACCESS
        assert token.scopes == ["read", "write"]
        assert token.expires_at > datetime.utcnow()
        assert token.status == TokenStatus.ACTIVE
        assert token.client_id == "test_client"
        
        # Vérifier que le token JWT est valide
        decoded = jwt.decode(
            token.jwt_token, 
            options={"verify_signature": False}
        )
        assert decoded["sub"] == user_data["user_id"]
        assert decoded["scope"] == "read write"
        assert decoded["exp"] > int(time.time())
    
    @pytest.mark.asyncio
    async def test_generate_refresh_token(self, token_manager):
        """Test génération de token de rafraîchissement"""
        user_data = TestDataFactory.create_test_user()
        
        with patch.object(token_manager, '_store_token') as mock_store:
            mock_store.return_value = True
            
            refresh_token = await token_manager.generate_refresh_token(
                user_id=user_data["user_id"],
                client_id="test_client",
                access_token_id="access_token_123"
            )
        
        assert refresh_token.token_id is not None
        assert refresh_token.user_id == user_data["user_id"]
        assert refresh_token.token_type == TokenType.REFRESH
        assert refresh_token.linked_access_token == "access_token_123"
        assert refresh_token.expires_at > datetime.utcnow() + timedelta(days=29)  # ~30 jours
        assert refresh_token.status == TokenStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_validate_token(self, token_manager):
        """Test validation de token"""
        token_data = TestDataFactory.create_access_token()
        
        # Token valide
        with patch.object(token_manager, '_get_token') as mock_get:
            mock_get.return_value = token_data
            
            result = await token_manager.validate_token(
                token=token_data["jwt_token"],
                required_scopes=["read"]
            )
        
        assert result["valid"] is True
        assert result["token"]["user_id"] == token_data["user_id"]
        assert result["token"]["scopes"] == token_data["scopes"]
        assert result["remaining_ttl"] > 0
        
        # Token expiré
        expired_token = token_data.copy()
        expired_token["expires_at"] = datetime.utcnow() - timedelta(hours=1)
        
        with patch.object(token_manager, '_get_token') as mock_get:
            mock_get.return_value = expired_token
            
            result = await token_manager.validate_token(
                token=expired_token["jwt_token"],
                required_scopes=["read"]
            )
        
        assert result["valid"] is False
        assert result["reason"] == "expired"
        
        # Token avec scopes insuffisants
        with patch.object(token_manager, '_get_token') as mock_get:
            mock_get.return_value = token_data
            
            result = await token_manager.validate_token(
                token=token_data["jwt_token"],
                required_scopes=["admin"]  # Scope non accordé
            )
        
        assert result["valid"] is False
        assert result["reason"] == "insufficient_scope"
    
    @pytest.mark.asyncio
    async def test_refresh_access_token(self, token_manager):
        """Test rafraîchissement de token d'accès"""
        refresh_token_data = TestDataFactory.create_refresh_token()
        
        with patch.object(token_manager, '_validate_refresh_token') as mock_validate:
            with patch.object(token_manager, '_generate_new_tokens') as mock_generate:
                mock_validate.return_value = {
                    "valid": True,
                    "token": refresh_token_data
                }
                
                new_access_token = TestDataFactory.create_access_token()
                new_refresh_token = TestDataFactory.create_refresh_token()
                
                mock_generate.return_value = {
                    "access_token": new_access_token,
                    "refresh_token": new_refresh_token
                }
                
                result = await token_manager.refresh_access_token(
                    refresh_token=refresh_token_data["token_value"]
                )
        
        assert result["success"] is True
        assert result["access_token"] is not None
        assert result["refresh_token"] is not None
        assert result["expires_in"] > 0
        
        # Le nouveau refresh token doit être différent
        assert result["refresh_token"] != refresh_token_data["token_value"]
    
    @pytest.mark.asyncio
    async def test_revoke_token(self, token_manager):
        """Test révocation de token"""
        token_data = TestDataFactory.create_access_token()
        
        with patch.object(token_manager, '_revoke_token_in_store') as mock_revoke:
            with patch.object(token_manager, '_blacklist_token') as mock_blacklist:
                mock_revoke.return_value = True
                mock_blacklist.return_value = True
                
                result = await token_manager.revoke_token(
                    token=token_data["jwt_token"],
                    reason="user_logout"
                )
        
        assert result["success"] is True
        assert result["revoked_at"] is not None
        assert result["reason"] == "user_logout"
        
        # Vérifier que le token est maintenant invalide
        with patch.object(token_manager, '_is_token_blacklisted') as mock_blacklisted:
            mock_blacklisted.return_value = True
            
            validation_result = await token_manager.validate_token(
                token=token_data["jwt_token"]
            )
        
        assert validation_result["valid"] is False
        assert validation_result["reason"] == "revoked"
    
    @pytest.mark.asyncio
    async def test_token_introspection(self, token_manager):
        """Test introspection de token (RFC 7662)"""
        token_data = TestDataFactory.create_access_token()
        
        with patch.object(token_manager, '_get_token_details') as mock_details:
            mock_details.return_value = {
                "active": True,
                "token_type": "access_token",
                "scope": " ".join(token_data["scopes"]),
                "client_id": token_data["client_id"],
                "username": "testuser",
                "sub": token_data["user_id"],
                "exp": int(token_data["expires_at"].timestamp()),
                "iat": int(token_data["created_at"].timestamp()),
                "jti": token_data["token_id"]
            }
            
            introspection = await token_manager.introspect_token(
                token=token_data["jwt_token"],
                client_id="authorized_client"
            )
        
        assert introspection["active"] is True
        assert introspection["token_type"] == "access_token"
        assert introspection["sub"] == token_data["user_id"]
        assert introspection["scope"] == " ".join(token_data["scopes"])
        assert introspection["exp"] > int(time.time())


@pytest_markers["unit"]
@pytest_markers["apikey"]
class TestAPIKeyManagement:
    """Tests pour la gestion des clés API"""
    
    @pytest.mark.asyncio
    async def test_create_api_key(self, token_manager):
        """Test création de clé API"""
        user_data = TestDataFactory.create_test_user()
        
        with patch.object(token_manager, '_store_api_key') as mock_store:
            mock_store.return_value = True
            
            api_key = await token_manager.create_api_key(
                user_id=user_data["user_id"],
                name="Production API Key",
                scopes=["api:read", "api:write"],
                expires_in_days=365,
                rate_limit=1000
            )
        
        assert api_key.key_id is not None
        assert api_key.user_id == user_data["user_id"]
        assert api_key.name == "Production API Key"
        assert api_key.scopes == ["api:read", "api:write"]
        assert api_key.rate_limit == 1000
        assert api_key.status == TokenStatus.ACTIVE
        assert api_key.expires_at > datetime.utcnow() + timedelta(days=364)
        
        # La clé doit être suffisamment longue et aléatoire
        assert len(api_key.key_value) >= 32
        assert api_key.key_value.startswith("spa_")  # Préfixe Spotify AI Agent
    
    @pytest.mark.asyncio
    async def test_validate_api_key(self, token_manager):
        """Test validation de clé API"""
        api_key_data = TestDataFactory.create_api_key()
        
        # Clé valide
        with patch.object(token_manager, '_get_api_key') as mock_get:
            mock_get.return_value = api_key_data
            
            result = await token_manager.validate_api_key(
                api_key=api_key_data["key_value"],
                required_scopes=["api:read"]
            )
        
        assert result["valid"] is True
        assert result["user_id"] == api_key_data["user_id"]
        assert result["scopes"] == api_key_data["scopes"]
        assert result["rate_limit"] == api_key_data["rate_limit"]
        
        # Clé expirée
        expired_key = api_key_data.copy()
        expired_key["expires_at"] = datetime.utcnow() - timedelta(days=1)
        expired_key["status"] = TokenStatus.EXPIRED
        
        with patch.object(token_manager, '_get_api_key') as mock_get:
            mock_get.return_value = expired_key
            
            result = await token_manager.validate_api_key(
                api_key=expired_key["key_value"]
            )
        
        assert result["valid"] is False
        assert result["reason"] == "expired"
    
    @pytest.mark.asyncio
    async def test_api_key_rotation(self, token_manager):
        """Test rotation de clé API"""
        api_key_data = TestDataFactory.create_api_key()
        
        with patch.object(token_manager, '_rotate_api_key') as mock_rotate:
            new_key_value = f"spa_{TestUtils.generate_random_string(32)}"
            mock_rotate.return_value = {
                "success": True,
                "new_key_value": new_key_value,
                "old_key_expires_at": datetime.utcnow() + timedelta(days=7),
                "rotated_at": datetime.utcnow()
            }
            
            result = await token_manager.rotate_api_key(
                key_id=api_key_data["key_id"],
                grace_period_days=7
            )
        
        assert result["success"] is True
        assert result["new_key_value"] != api_key_data["key_value"]
        assert result["old_key_expires_at"] > datetime.utcnow()
        
        # L'ancienne clé doit encore être valide pendant la période de grâce
        assert result["old_key_expires_at"] > datetime.utcnow() + timedelta(days=6)
    
    @pytest.mark.asyncio
    async def test_api_key_rate_limiting(self, token_manager):
        """Test limitation de taux pour les clés API"""
        api_key_data = TestDataFactory.create_api_key()
        api_key_data["rate_limit"] = 10  # 10 requêtes par minute
        
        # Vérifier usage normal
        with patch.object(token_manager, '_check_rate_limit') as mock_check:
            mock_check.return_value = {
                "allowed": True,
                "remaining": 9,
                "reset_at": datetime.utcnow() + timedelta(minutes=1),
                "current_usage": 1
            }
            
            result = await token_manager.check_api_key_rate_limit(
                api_key=api_key_data["key_value"]
            )
        
        assert result["allowed"] is True
        assert result["remaining"] == 9
        
        # Vérifier dépassement de limite
        with patch.object(token_manager, '_check_rate_limit') as mock_check:
            mock_check.return_value = {
                "allowed": False,
                "remaining": 0,
                "reset_at": datetime.utcnow() + timedelta(minutes=1),
                "current_usage": 10
            }
            
            result = await token_manager.check_api_key_rate_limit(
                api_key=api_key_data["key_value"]
            )
        
        assert result["allowed"] is False
        assert result["remaining"] == 0


@pytest_markers["unit"]
@pytest_markers["jwt"]
class TestJWTOperations:
    """Tests pour les opérations JWT"""
    
    @pytest.mark.asyncio
    async def test_jwt_creation_and_validation(self, token_manager):
        """Test création et validation JWT"""
        user_data = TestDataFactory.create_test_user()
        
        # Créer JWT
        claims = JWTClaims(
            sub=user_data["user_id"],
            iss="spotify-ai-agent.com",
            aud="api.spotify-ai-agent.com",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            jti=str(uuid.uuid4()),
            scope=["read", "write"],
            client_id="test_client"
        )
        
        with patch.object(token_manager, '_sign_jwt') as mock_sign:
            jwt_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.test.signature"
            mock_sign.return_value = jwt_token
            
            token = await token_manager.create_jwt(claims)
        
        assert token is not None
        assert "." in token  # Format JWT
        
        # Valider JWT
        with patch.object(token_manager, '_verify_jwt_signature') as mock_verify:
            with patch.object(token_manager, '_decode_jwt_claims') as mock_decode:
                mock_verify.return_value = True
                mock_decode.return_value = claims.__dict__
                
                validation = await token_manager.validate_jwt(token)
        
        assert validation["valid"] is True
        assert validation["claims"]["sub"] == user_data["user_id"]
        assert validation["claims"]["scope"] == ["read", "write"]
    
    @pytest.mark.asyncio
    async def test_jwt_signature_algorithms(self, token_manager):
        """Test algorithmes de signature JWT"""
        claims = {
            "sub": "test_user",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp())
        }
        
        # Test RS256 (recommandé)
        with patch.object(token_manager, '_sign_with_algorithm') as mock_sign:
            mock_sign.return_value = "rs256_signed_token"
            
            rs256_token = await token_manager.sign_jwt(
                claims=claims,
                algorithm="RS256",
                key_id="rsa_key_1"
            )
        
        assert rs256_token == "rs256_signed_token"
        
        # Test ES256 (ECDSA)
        with patch.object(token_manager, '_sign_with_algorithm') as mock_sign:
            mock_sign.return_value = "es256_signed_token"
            
            es256_token = await token_manager.sign_jwt(
                claims=claims,
                algorithm="ES256",
                key_id="ec_key_1"
            )
        
        assert es256_token == "es256_signed_token"
    
    @pytest.mark.asyncio
    async def test_jwt_key_rotation(self, token_manager):
        """Test rotation des clés de signature JWT"""
        # Générer nouvelle paire de clés
        with patch.object(token_manager, '_generate_key_pair') as mock_generate:
            mock_generate.return_value = {
                "key_id": "new_key_123",
                "private_key": "new_private_key_pem",
                "public_key": "new_public_key_pem",
                "algorithm": "RS256",
                "created_at": datetime.utcnow()
            }
            
            new_key = await token_manager.generate_signing_key(
                algorithm="RS256",
                key_size=2048
            )
        
        assert new_key["key_id"] is not None
        assert new_key["algorithm"] == "RS256"
        
        # Activer la nouvelle clé
        with patch.object(token_manager, '_activate_signing_key') as mock_activate:
            mock_activate.return_value = {
                "success": True,
                "active_key_id": new_key["key_id"],
                "previous_key_id": "old_key_456"
            }
            
            result = await token_manager.activate_signing_key(
                key_id=new_key["key_id"],
                deactivate_previous=False
            )
        
        assert result["success"] is True
        assert result["active_key_id"] == new_key["key_id"]


@pytest_markers["performance"]
@pytest_markers["token"]
class TestTokenPerformance:
    """Tests de performance pour les tokens"""
    
    @pytest.mark.asyncio
    async def test_token_generation_performance(self, token_manager):
        """Test performance génération de tokens"""
        user_data = TestDataFactory.create_test_user()
        
        with patch.object(token_manager, '_store_token', return_value=True):
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                token_manager.generate_access_token,
                user_id=user_data["user_id"],
                scopes=["read", "write"],
                expires_in=3600
            )
        
        # Génération de token doit être < 100ms
        assert execution_time < 0.1
        assert result.token_id is not None
        
        print(f"🎫 Temps génération token: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_token_validation_performance(self, token_manager):
        """Test performance validation de tokens"""
        token_data = TestDataFactory.create_access_token()
        
        with patch.object(token_manager, '_get_token', return_value=token_data):
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                token_manager.validate_token,
                token=token_data["jwt_token"],
                required_scopes=["read"]
            )
        
        # Validation de token doit être < 50ms
        assert execution_time < 0.05
        assert result["valid"] is True
        
        print(f"✅ Temps validation token: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_token_operations(self, token_manager):
        """Test opérations de tokens concurrentes"""
        tokens = [TestDataFactory.create_access_token() for i in range(50)]
        
        async def validate_token(token_data):
            with patch.object(token_manager, '_get_token', return_value=token_data):
                return await token_manager.validate_token(
                    token=token_data["jwt_token"]
                )
        
        stress_results = await PerformanceTestUtils.stress_test_function(
            validate_token,
            concurrent_calls=50,
            tokens[0]
        )
        
        assert stress_results["success_rate"] >= 0.98
        assert stress_results["average_time"] < 0.1
        
        print(f"📊 Taux réussite concurrent: {stress_results['success_rate']:.2%}")
        print(f"⏱️  Temps moyen validation: {stress_results['average_time']:.3f}s")


@pytest_markers["security"]
@pytest_markers["token"]
class TestTokenSecurity:
    """Tests de sécurité pour les tokens"""
    
    @pytest.mark.asyncio
    async def test_token_entropy(self, token_manager):
        """Test entropie des tokens"""
        user_data = TestDataFactory.create_test_user()
        
        tokens = []
        for i in range(100):
            with patch.object(token_manager, '_store_token', return_value=True):
                token = await token_manager.generate_access_token(
                    user_id=user_data["user_id"],
                    scopes=["read"]
                )
            tokens.append(token.token_id)
        
        # Tous les tokens doivent être uniques
        unique_tokens = set(tokens)
        assert len(unique_tokens) == len(tokens)
        
        # Calculer entropie approximative
        entropy = SecurityValidators.calculate_entropy("".join(tokens))
        assert entropy > 4.0  # Entropie minimale acceptable
        
        print(f"🔐 Entropie des tokens: {entropy:.2f} bits")
    
    @pytest.mark.asyncio
    async def test_jwt_tampering_detection(self, token_manager):
        """Test détection de falsification JWT"""
        valid_jwt = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.signature"
        
        # Token avec signature invalide
        tampered_jwt = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.invalid_signature"
        
        with patch.object(token_manager, '_verify_jwt_signature') as mock_verify:
            mock_verify.return_value = False
            
            result = await token_manager.validate_jwt(tampered_jwt)
        
        assert result["valid"] is False
        assert result["reason"] == "invalid_signature"
        
        # Token avec payload modifié
        modified_payload_jwt = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJoYWNrZXIifQ.signature"
        
        with patch.object(token_manager, '_verify_jwt_signature') as mock_verify:
            mock_verify.return_value = False  # Signature ne correspond plus
            
            result = await token_manager.validate_jwt(modified_payload_jwt)
        
        assert result["valid"] is False
        assert result["reason"] == "invalid_signature"
    
    @pytest.mark.asyncio
    async def test_token_scope_validation(self, token_manager):
        """Test validation stricte des scopes"""
        token_data = TestDataFactory.create_access_token()
        token_data["scopes"] = ["read", "profile"]
        
        # Scope valide
        with patch.object(token_manager, '_get_token', return_value=token_data):
            result = await token_manager.validate_token(
                token=token_data["jwt_token"],
                required_scopes=["read"]
            )
        
        assert result["valid"] is True
        
        # Scope requis mais non accordé
        with patch.object(token_manager, '_get_token', return_value=token_data):
            result = await token_manager.validate_token(
                token=token_data["jwt_token"],
                required_scopes=["admin"]
            )
        
        assert result["valid"] is False
        assert result["reason"] == "insufficient_scope"
        
        # Scopes multiples - un manquant
        with patch.object(token_manager, '_get_token', return_value=token_data):
            result = await token_manager.validate_token(
                token=token_data["jwt_token"],
                required_scopes=["read", "write"]  # write non accordé
            )
        
        assert result["valid"] is False
        assert result["reason"] == "insufficient_scope"


if __name__ == "__main__":
    print("🧪 Tests TokenManager Ultra-Avancés")
    print("📋 Modules testés:")
    print("  ✅ Génération et validation de tokens d'accès")
    print("  ✅ Génération et rafraîchissement de refresh tokens")
    print("  ✅ Révocation et introspection de tokens")
    print("  ✅ Gestion des clés API et rotation")
    print("  ✅ Opérations JWT et algorithmes de signature")
    print("  ✅ Rotation des clés de signature")
    print("  ✅ Tests de sécurité et entropie")
    print("  ✅ Tests de performance et concurrence")
    
    # Lancement des tests
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])
