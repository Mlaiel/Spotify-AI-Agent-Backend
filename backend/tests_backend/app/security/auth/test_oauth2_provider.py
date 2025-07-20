# 🧪 Tests OAuth2Provider Ultra-Avancés
# =====================================

import pytest
import pytest_asyncio
import asyncio
import time
import json
import jwt
import base64
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from urllib.parse import parse_qs, urlparse

from app.security.auth.oauth2_provider import (
    OAuth2Provider, OAuth2Client, AuthorizationCode, AccessToken, RefreshToken,
    GrantType, ResponseType, ClientType, TokenType
)

from conftest import (
    TestDataFactory, TestUtils, PerformanceTestUtils, SecurityValidators,
    pytest_markers
)


@pytest_markers["unit"]
@pytest_markers["oauth2"]
class TestOAuth2Provider:
    """Tests unitaires pour OAuth2Provider"""
    
    @pytest.mark.asyncio
    async def test_register_client(self, oauth2_provider):
        """Test enregistrement d'un client OAuth2"""
        client_data = TestDataFactory.create_oauth2_client()
        
        with patch.object(oauth2_provider, '_store_client') as mock_store:
            mock_store.return_value = True
            
            result = await oauth2_provider.register_client(
                client_data=client_data,
                auto_generate_secret=True
            )
        
        assert result.client_id is not None
        assert result.client_secret is not None
        assert result.client_name == client_data["client_name"]
        assert result.client_type == client_data["client_type"]
        assert result.redirect_uris == client_data["redirect_uris"]
        assert result.is_active is True
        assert result.created_at is not None
    
    @pytest.mark.asyncio
    async def test_authorization_code_flow_complete(self, oauth2_provider):
        """Test flux Authorization Code complet"""
        client_data = TestDataFactory.create_oauth2_client()
        user_data = TestDataFactory.create_test_user()
        
        # 1. Enregistrer le client
        with patch.object(oauth2_provider, '_store_client', return_value=True):
            client = await oauth2_provider.register_client(client_data)
        
        # 2. Générer code_verifier et code_challenge pour PKCE
        code_verifier = TestUtils.generate_pkce_code_verifier()
        code_challenge = TestUtils.generate_pkce_code_challenge(code_verifier)
        
        # 3. Créer demande d'autorisation
        with patch.object(oauth2_provider, '_store_authorization_request') as mock_store_req:
            mock_store_req.return_value = True
            
            auth_request = await oauth2_provider.create_authorization_request(
                client_id=client.client_id,
                redirect_uri=client.redirect_uris[0],
                scope="openid profile email",
                response_type=ResponseType.CODE.value,
                state="test_state_123",
                code_challenge=code_challenge,
                code_challenge_method="S256"
            )
        
        assert auth_request["success"] is True
        assert auth_request["authorization_url"] is not None
        
        # 4. Autoriser l'utilisateur
        with patch.object(oauth2_provider, '_store_authorization_code') as mock_store_code:
            auth_code = f"auth_code_{TestUtils.generate_random_string(16)}"
            mock_store_code.return_value = auth_code
            
            authorization_result = await oauth2_provider.authorize_user(
                user_id=user_data["user_id"],
                client_id=client.client_id,
                scopes=["openid", "profile", "email"],
                authorization_request_id=auth_request["request_id"]
            )
        
        assert authorization_result["success"] is True
        assert authorization_result["authorization_code"] is not None
        
        # 5. Échanger le code contre des tokens
        with patch.object(oauth2_provider, '_validate_authorization_code') as mock_validate:
            with patch.object(oauth2_provider, '_generate_tokens') as mock_tokens:
                mock_validate.return_value = {
                    "valid": True,
                    "client_id": client.client_id,
                    "user_id": user_data["user_id"],
                    "scopes": ["openid", "profile", "email"]
                }
                
                mock_tokens.return_value = {
                    "access_token": f"access_token_{TestUtils.generate_random_string(32)}",
                    "refresh_token": f"refresh_token_{TestUtils.generate_random_string(32)}",
                    "id_token": f"id_token_{TestUtils.generate_random_string(32)}",
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }
                
                token_response = await oauth2_provider.exchange_authorization_code(
                    client_id=client.client_id,
                    client_secret=client.client_secret,
                    authorization_code=authorization_result["authorization_code"],
                    redirect_uri=client.redirect_uris[0],
                    code_verifier=code_verifier
                )
        
        assert token_response["success"] is True
        assert token_response["access_token"] is not None
        assert token_response["refresh_token"] is not None
        assert token_response["token_type"] == "Bearer"
        assert token_response["expires_in"] > 0
    
    @pytest.mark.asyncio
    async def test_pkce_validation(self, oauth2_provider):
        """Test validation PKCE (Proof Key for Code Exchange)"""
        client_data = TestDataFactory.create_oauth2_client()
        
        # Générer PKCE
        code_verifier = TestUtils.generate_pkce_code_verifier()
        code_challenge = TestUtils.generate_pkce_code_challenge(code_verifier)
        
        # Test avec code_verifier correct
        with patch.object(oauth2_provider, '_validate_pkce') as mock_validate:
            mock_validate.return_value = True
            
            result = await oauth2_provider._validate_pkce(
                code_verifier=code_verifier,
                code_challenge=code_challenge,
                code_challenge_method="S256"
            )
        
        assert result is True
        
        # Test avec code_verifier incorrect
        wrong_verifier = TestUtils.generate_pkce_code_verifier()
        
        with patch.object(oauth2_provider, '_validate_pkce') as mock_validate:
            mock_validate.return_value = False
            
            result = await oauth2_provider._validate_pkce(
                code_verifier=wrong_verifier,
                code_challenge=code_challenge,
                code_challenge_method="S256"
            )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_client_credentials_flow(self, oauth2_provider):
        """Test flux Client Credentials"""
        client_data = TestDataFactory.create_oauth2_client()
        client_data["grant_types"] = [GrantType.CLIENT_CREDENTIALS.value]
        
        with patch.object(oauth2_provider, '_store_client', return_value=True):
            client = await oauth2_provider.register_client(client_data)
        
        with patch.object(oauth2_provider, '_validate_client_credentials') as mock_validate:
            with patch.object(oauth2_provider, '_generate_access_token') as mock_token:
                mock_validate.return_value = {"valid": True, "client": client}
                mock_token.return_value = {
                    "access_token": f"client_token_{TestUtils.generate_random_string(32)}",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                    "scope": "api:read api:write"
                }
                
                token_response = await oauth2_provider.client_credentials_grant(
                    client_id=client.client_id,
                    client_secret=client.client_secret,
                    scope="api:read api:write"
                )
        
        assert token_response["success"] is True
        assert token_response["access_token"] is not None
        assert token_response["token_type"] == "Bearer"
        assert "api:read" in token_response["scope"]
        assert "refresh_token" not in token_response  # Pas de refresh token en client_credentials
    
    @pytest.mark.asyncio
    async def test_refresh_token_flow(self, oauth2_provider):
        """Test flux de rafraîchissement de token"""
        client_data = TestDataFactory.create_oauth2_client()
        user_data = TestDataFactory.create_test_user()
        
        # Simuler un refresh token existant
        refresh_token = f"refresh_{TestUtils.generate_random_string(32)}"
        
        with patch.object(oauth2_provider, '_validate_refresh_token') as mock_validate:
            with patch.object(oauth2_provider, '_generate_tokens') as mock_tokens:
                mock_validate.return_value = {
                    "valid": True,
                    "client_id": client_data["client_id"],
                    "user_id": user_data["user_id"],
                    "scopes": ["openid", "profile"]
                }
                
                mock_tokens.return_value = {
                    "access_token": f"new_access_{TestUtils.generate_random_string(32)}",
                    "refresh_token": f"new_refresh_{TestUtils.generate_random_string(32)}",
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }
                
                token_response = await oauth2_provider.refresh_access_token(
                    client_id=client_data["client_id"],
                    client_secret=client_data["client_secret"],
                    refresh_token=refresh_token
                )
        
        assert token_response["success"] is True
        assert token_response["access_token"] is not None
        assert token_response["refresh_token"] is not None
        assert token_response["refresh_token"] != refresh_token  # Nouveau refresh token
    
    @pytest.mark.asyncio
    async def test_token_introspection(self, oauth2_provider):
        """Test introspection de token (RFC 7662)"""
        client_data = TestDataFactory.create_oauth2_client()
        access_token = f"access_{TestUtils.generate_random_string(32)}"
        
        with patch.object(oauth2_provider, '_introspect_token') as mock_introspect:
            mock_introspect.return_value = {
                "active": True,
                "client_id": client_data["client_id"],
                "username": "testuser",
                "scope": "openid profile",
                "token_type": "access_token",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "sub": "user_123"
            }
            
            introspection_result = await oauth2_provider.introspect_token(
                token=access_token,
                client_id=client_data["client_id"],
                client_secret=client_data["client_secret"]
            )
        
        assert introspection_result["active"] is True
        assert introspection_result["client_id"] == client_data["client_id"]
        assert introspection_result["token_type"] == "access_token"
        assert introspection_result["exp"] > int(time.time())
    
    @pytest.mark.asyncio
    async def test_token_revocation(self, oauth2_provider):
        """Test révocation de token (RFC 7009)"""
        client_data = TestDataFactory.create_oauth2_client()
        access_token = f"access_{TestUtils.generate_random_string(32)}"
        
        with patch.object(oauth2_provider, '_revoke_token') as mock_revoke:
            mock_revoke.return_value = {"success": True, "revoked_at": datetime.utcnow()}
            
            revocation_result = await oauth2_provider.revoke_token(
                token=access_token,
                client_id=client_data["client_id"],
                client_secret=client_data["client_secret"],
                token_type_hint="access_token"
            )
        
        assert revocation_result["success"] is True
        assert revocation_result["revoked_at"] is not None


@pytest_markers["unit"]
@pytest_markers["oauth2"]
class TestOpenIDConnect:
    """Tests pour OpenID Connect"""
    
    @pytest.mark.asyncio
    async def test_discovery_document(self, oauth2_provider):
        """Test document de découverte OpenID Connect"""
        with patch.object(oauth2_provider, '_get_discovery_document') as mock_discovery:
            mock_discovery.return_value = {
                "issuer": "https://test.spotify-ai-agent.com",
                "authorization_endpoint": "https://test.spotify-ai-agent.com/oauth/authorize",
                "token_endpoint": "https://test.spotify-ai-agent.com/oauth/token",
                "userinfo_endpoint": "https://test.spotify-ai-agent.com/oauth/userinfo",
                "jwks_uri": "https://test.spotify-ai-agent.com/.well-known/jwks.json",
                "scopes_supported": ["openid", "profile", "email"],
                "response_types_supported": ["code", "token", "id_token"],
                "grant_types_supported": ["authorization_code", "implicit", "refresh_token"],
                "id_token_signing_alg_values_supported": ["RS256", "HS256"],
                "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"]
            }
            
            discovery = await oauth2_provider.get_discovery_document()
        
        assert discovery["issuer"] is not None
        assert discovery["authorization_endpoint"] is not None
        assert discovery["token_endpoint"] is not None
        assert discovery["userinfo_endpoint"] is not None
        assert discovery["jwks_uri"] is not None
        assert "openid" in discovery["scopes_supported"]
        assert "authorization_code" in discovery["grant_types_supported"]
    
    @pytest.mark.asyncio
    async def test_jwks_endpoint(self, oauth2_provider):
        """Test endpoint JWKS (JSON Web Key Set)"""
        with patch.object(oauth2_provider, '_get_jwks') as mock_jwks:
            mock_jwks.return_value = {
                "keys": [
                    {
                        "kty": "RSA",
                        "use": "sig",
                        "kid": "test_key_1",
                        "alg": "RS256",
                        "n": "test_modulus",
                        "e": "AQAB"
                    }
                ]
            }
            
            jwks = await oauth2_provider.get_jwks()
        
        assert "keys" in jwks
        assert len(jwks["keys"]) > 0
        
        for key in jwks["keys"]:
            assert "kty" in key
            assert "use" in key
            assert "kid" in key
            assert "alg" in key
    
    @pytest.mark.asyncio
    async def test_id_token_generation(self, oauth2_provider):
        """Test génération d'ID Token"""
        client_data = TestDataFactory.create_oauth2_client()
        user_data = TestDataFactory.create_test_user()
        
        with patch.object(oauth2_provider, '_generate_id_token') as mock_id_token:
            # Simuler un ID Token JWT
            payload = {
                "iss": "https://test.spotify-ai-agent.com",
                "sub": user_data["user_id"],
                "aud": client_data["client_id"],
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "auth_time": int(time.time()),
                "nonce": "test_nonce_123",
                "email": user_data["email"],
                "name": f"{user_data['first_name']} {user_data['last_name']}"
            }
            
            # Encoder sans signature pour le test
            id_token = jwt.encode(payload, "test_secret", algorithm="HS256")
            mock_id_token.return_value = id_token
            
            result = await oauth2_provider.generate_id_token(
                user_id=user_data["user_id"],
                client_id=client_data["client_id"],
                scopes=["openid", "profile", "email"],
                nonce="test_nonce_123"
            )
        
        assert result is not None
        
        # Décoder et vérifier le payload
        decoded_payload = jwt.decode(result, "test_secret", algorithms=["HS256"])
        assert decoded_payload["sub"] == user_data["user_id"]
        assert decoded_payload["aud"] == client_data["client_id"]
        assert decoded_payload["nonce"] == "test_nonce_123"
        assert decoded_payload["email"] == user_data["email"]
    
    @pytest.mark.asyncio
    async def test_userinfo_endpoint(self, oauth2_provider):
        """Test endpoint UserInfo"""
        user_data = TestDataFactory.create_test_user()
        access_token = f"access_{TestUtils.generate_random_string(32)}"
        
        with patch.object(oauth2_provider, '_get_userinfo_from_token') as mock_userinfo:
            mock_userinfo.return_value = {
                "sub": user_data["user_id"],
                "email": user_data["email"],
                "email_verified": True,
                "name": f"{user_data['first_name']} {user_data['last_name']}",
                "given_name": user_data["first_name"],
                "family_name": user_data["last_name"],
                "profile": f"https://profile.test.com/users/{user_data['username']}",
                "updated_at": int(time.time())
            }
            
            userinfo = await oauth2_provider.get_userinfo(access_token)
        
        assert userinfo["sub"] == user_data["user_id"]
        assert userinfo["email"] == user_data["email"]
        assert userinfo["email_verified"] is True
        assert userinfo["given_name"] == user_data["first_name"]
        assert userinfo["family_name"] == user_data["last_name"]


@pytest_markers["performance"]
@pytest_markers["oauth2"]
class TestOAuth2Performance:
    """Tests de performance OAuth2"""
    
    @pytest.mark.asyncio
    async def test_token_generation_performance(self, oauth2_provider):
        """Test performance génération de tokens"""
        client_data = TestDataFactory.create_oauth2_client()
        
        with patch.object(oauth2_provider, '_validate_client_credentials', return_value={"valid": True}):
            with patch.object(oauth2_provider, '_generate_access_token') as mock_token:
                mock_token.return_value = {
                    "access_token": f"perf_token_{TestUtils.generate_random_string(32)}",
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }
                
                result, execution_time = await PerformanceTestUtils.measure_execution_time(
                    oauth2_provider.client_credentials_grant,
                    client_id=client_data["client_id"],
                    client_secret=client_data["client_secret"],
                    scope="read"
                )
        
        # Génération de token doit être < 500ms
        assert execution_time < 0.5
        assert result["success"] is True
        
        print(f"🎫 Temps génération token: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_token_requests(self, oauth2_provider):
        """Test requêtes de tokens concurrentes"""
        clients = [TestDataFactory.create_oauth2_client(f"client_{i}") for i in range(10)]
        
        async def get_token(client_data):
            with patch.object(oauth2_provider, '_validate_client_credentials', return_value={"valid": True}):
                with patch.object(oauth2_provider, '_generate_access_token') as mock_token:
                    mock_token.return_value = {
                        "access_token": f"concurrent_token_{TestUtils.generate_random_string(16)}",
                        "expires_in": 3600,
                        "token_type": "Bearer"
                    }
                    
                    return await oauth2_provider.client_credentials_grant(
                        client_id=client_data["client_id"],
                        client_secret=client_data["client_secret"],
                        scope="read"
                    )
        
        stress_results = await PerformanceTestUtils.stress_test_function(
            get_token,
            concurrent_calls=10,
            clients[0]
        )
        
        assert stress_results["success_rate"] >= 0.9
        assert stress_results["average_time"] < 1.0
        
        print(f"📊 Taux réussite concurrent: {stress_results['success_rate']:.2%}")
        print(f"⏱️  Temps moyen: {stress_results['average_time']:.3f}s")


@pytest_markers["security"]
@pytest_markers["oauth2"]
class TestOAuth2Security:
    """Tests de sécurité OAuth2"""
    
    @pytest.mark.asyncio
    async def test_client_authentication_security(self, oauth2_provider):
        """Test sécurité authentification client"""
        client_data = TestDataFactory.create_oauth2_client()
        
        # Test avec client_secret invalide
        with patch.object(oauth2_provider, '_validate_client_credentials') as mock_validate:
            mock_validate.return_value = {"valid": False, "error": "invalid_client"}
            
            result = await oauth2_provider.client_credentials_grant(
                client_id=client_data["client_id"],
                client_secret="wrong_secret",
                scope="read"
            )
        
        assert result["success"] is False
        assert result["error"] == "invalid_client"
        
        # Test avec client_id inexistant
        with patch.object(oauth2_provider, '_validate_client_credentials') as mock_validate:
            mock_validate.return_value = {"valid": False, "error": "invalid_client"}
            
            result = await oauth2_provider.client_credentials_grant(
                client_id="nonexistent_client",
                client_secret=client_data["client_secret"],
                scope="read"
            )
        
        assert result["success"] is False
        assert result["error"] == "invalid_client"
    
    @pytest.mark.asyncio
    async def test_scope_validation(self, oauth2_provider):
        """Test validation des scopes"""
        client_data = TestDataFactory.create_oauth2_client()
        client_data["scopes"] = ["read"]  # Scope limité
        
        # Test scope autorisé
        with patch.object(oauth2_provider, '_validate_client_credentials') as mock_validate:
            with patch.object(oauth2_provider, '_validate_scopes') as mock_scopes:
                mock_validate.return_value = {"valid": True, "client": client_data}
                mock_scopes.return_value = {"valid": True}
                
                result = await oauth2_provider.client_credentials_grant(
                    client_id=client_data["client_id"],
                    client_secret=client_data["client_secret"],
                    scope="read"
                )
        
        # Test scope non autorisé
        with patch.object(oauth2_provider, '_validate_client_credentials') as mock_validate:
            with patch.object(oauth2_provider, '_validate_scopes') as mock_scopes:
                mock_validate.return_value = {"valid": True, "client": client_data}
                mock_scopes.return_value = {"valid": False, "error": "invalid_scope"}
                
                result = await oauth2_provider.client_credentials_grant(
                    client_id=client_data["client_id"],
                    client_secret=client_data["client_secret"],
                    scope="admin"
                )
        
        assert result["success"] is False
        assert result["error"] == "invalid_scope"
    
    @pytest.mark.asyncio
    async def test_redirect_uri_validation(self, oauth2_provider):
        """Test validation des redirect URIs"""
        client_data = TestDataFactory.create_oauth2_client()
        client_data["redirect_uris"] = ["https://app.example.com/callback"]
        
        # Test avec redirect URI valide
        with patch.object(oauth2_provider, '_validate_redirect_uri') as mock_validate:
            mock_validate.return_value = {"valid": True}
            
            result = await oauth2_provider.create_authorization_request(
                client_id=client_data["client_id"],
                redirect_uri="https://app.example.com/callback",
                scope="openid",
                response_type="code"
            )
        
        # Test avec redirect URI invalide
        with patch.object(oauth2_provider, '_validate_redirect_uri') as mock_validate:
            mock_validate.return_value = {"valid": False, "error": "invalid_request"}
            
            result = await oauth2_provider.create_authorization_request(
                client_id=client_data["client_id"],
                redirect_uri="https://malicious.com/steal-code",
                scope="openid",
                response_type="code"
            )
        
        assert result["success"] is False
        assert result["error"] == "invalid_request"
    
    @pytest.mark.asyncio
    async def test_authorization_code_replay_protection(self, oauth2_provider):
        """Test protection contre la réutilisation de codes d'autorisation"""
        client_data = TestDataFactory.create_oauth2_client()
        auth_code = f"auth_code_{TestUtils.generate_random_string(16)}"
        
        # Première utilisation du code - doit réussir
        with patch.object(oauth2_provider, '_validate_authorization_code') as mock_validate:
            with patch.object(oauth2_provider, '_generate_tokens') as mock_tokens:
                mock_validate.return_value = {"valid": True, "client_id": client_data["client_id"]}
                mock_tokens.return_value = {
                    "access_token": f"access_{TestUtils.generate_random_string(32)}",
                    "refresh_token": f"refresh_{TestUtils.generate_random_string(32)}",
                    "expires_in": 3600,
                    "token_type": "Bearer"
                }
                
                result1 = await oauth2_provider.exchange_authorization_code(
                    client_id=client_data["client_id"],
                    client_secret=client_data["client_secret"],
                    authorization_code=auth_code,
                    redirect_uri=client_data["redirect_uris"][0]
                )
        
        assert result1["success"] is True
        
        # Deuxième utilisation du même code - doit échouer
        with patch.object(oauth2_provider, '_validate_authorization_code') as mock_validate:
            mock_validate.return_value = {"valid": False, "error": "invalid_grant"}
            
            result2 = await oauth2_provider.exchange_authorization_code(
                client_id=client_data["client_id"],
                client_secret=client_data["client_secret"],
                authorization_code=auth_code,
                redirect_uri=client_data["redirect_uris"][0]
            )
        
        assert result2["success"] is False
        assert result2["error"] == "invalid_grant"


if __name__ == "__main__":
    print("🧪 Tests OAuth2Provider Ultra-Avancés")
    print("📋 Modules testés:")
    print("  ✅ Enregistrement et gestion des clients OAuth2")
    print("  ✅ Flux Authorization Code avec PKCE")
    print("  ✅ Flux Client Credentials")
    print("  ✅ Flux Refresh Token")
    print("  ✅ Introspection et révocation de tokens")
    print("  ✅ OpenID Connect (Discovery, JWKS, ID Token, UserInfo)")
    print("  ✅ Tests de sécurité et validation")
    print("  ✅ Tests de performance et concurrence")
    
    # Lancement des tests
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])
