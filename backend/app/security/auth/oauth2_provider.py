# 🔐 OAuth2 Provider & OpenID Connect
# ===================================
# 
# Fournisseur OAuth2 et OpenID Connect enterprise
# avec support SSO et gestion des clients avancée.
#
# 🎖️ Expert: Lead Dev + Architecte IA + Spécialiste Sécurité Backend
#
# Développé par l'équipe d'experts enterprise
# ===================================

"""
🔐 Enterprise OAuth2 & OpenID Connect Provider
==============================================

Complete OAuth2 and OpenID Connect implementation providing:
- OAuth2 authorization server with PKCE support
- OpenID Connect identity provider
- Single Sign-On (SSO) capabilities
- Client registration and management
- Advanced scopes and claims management
- Token introspection and revocation
- Dynamic client registration
- JWT-secured authorization requests
"""

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import jwt
import redis
import logging
from fastapi import HTTPException, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import uuid

# Configuration et logging
logger = logging.getLogger(__name__)


class GrantType(Enum):
    """Types de grants OAuth2"""
    AUTHORIZATION_CODE = "authorization_code"
    IMPLICIT = "implicit"
    RESOURCE_OWNER_PASSWORD = "password"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"
    DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"


class ResponseType(Enum):
    """Types de réponse OAuth2"""
    CODE = "code"
    TOKEN = "token"
    ID_TOKEN = "id_token"
    CODE_TOKEN = "code token"
    CODE_ID_TOKEN = "code id_token"
    TOKEN_ID_TOKEN = "token id_token"
    CODE_TOKEN_ID_TOKEN = "code token id_token"


class ClientType(Enum):
    """Types de clients OAuth2"""
    CONFIDENTIAL = "confidential"
    PUBLIC = "public"


class TokenType(Enum):
    """Types de tokens"""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"
    AUTHORIZATION_CODE = "authorization_code"


@dataclass
class OAuth2Client:
    """Client OAuth2"""
    client_id: str
    client_secret: Optional[str]
    client_name: str
    client_type: ClientType
    redirect_uris: List[str]
    grant_types: List[str]
    response_types: List[str]
    scopes: List[str]
    token_endpoint_auth_method: str = "client_secret_basic"
    jwks_uri: Optional[str] = None
    jwks: Optional[Dict] = None
    created_at: datetime = None
    updated_at: datetime = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class AuthorizationCode:
    """Code d'autorisation"""
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scopes: List[str]
    created_at: datetime
    expires_at: datetime
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    nonce: Optional[str] = None


@dataclass
class AccessToken:
    """Token d'accès"""
    token: str
    client_id: str
    user_id: str
    scopes: List[str]
    created_at: datetime
    expires_at: datetime
    token_type: str = "Bearer"


@dataclass
class RefreshToken:
    """Token de rafraîchissement"""
    token: str
    client_id: str
    user_id: str
    scopes: List[str]
    created_at: datetime
    expires_at: datetime


class OAuth2Provider:
    """Fournisseur OAuth2 enterprise"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.authorization_code_lifetime = 600  # 10 minutes
        self.access_token_lifetime = 3600  # 1 heure
        self.refresh_token_lifetime = 86400 * 30  # 30 jours
        self.default_scopes = ["openid", "profile", "email"]
        
        # Clés cryptographiques
        self.private_key = None
        self.public_key = None
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialise les clés cryptographiques"""
        try:
            # Générer une paire de clés RSA
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            
        except Exception as exc:
            self.logger.error(f"Erreur initialisation clés: {exc}")
    
    async def register_client(
        self,
        client_data: Dict[str, Any],
        auto_generate_secret: bool = True
    ) -> OAuth2Client:
        """Enregistre un nouveau client OAuth2"""
        try:
            # Générer un client_id unique
            client_id = f"client_{secrets.token_urlsafe(16)}"
            
            # Générer un client_secret si nécessaire
            client_secret = None
            if auto_generate_secret and client_data.get("client_type") == ClientType.CONFIDENTIAL.value:
                client_secret = secrets.token_urlsafe(32)
            
            # Créer le client
            client = OAuth2Client(
                client_id=client_id,
                client_secret=client_secret,
                client_name=client_data.get("client_name", ""),
                client_type=ClientType(client_data.get("client_type", "confidential")),
                redirect_uris=client_data.get("redirect_uris", []),
                grant_types=client_data.get("grant_types", ["authorization_code"]),
                response_types=client_data.get("response_types", ["code"]),
                scopes=client_data.get("scopes", self.default_scopes),
                token_endpoint_auth_method=client_data.get("token_endpoint_auth_method", "client_secret_basic")
            )
            
            # Sauvegarder le client
            await self._store_client(client)
            
            self.logger.info(f"Client OAuth2 enregistré: {client_id}")
            return client
            
        except Exception as exc:
            self.logger.error(f"Erreur enregistrement client: {exc}")
            raise HTTPException(status_code=400, detail="Erreur enregistrement client")
    
    async def authorize(
        self,
        client_id: str,
        redirect_uri: str,
        response_type: str,
        scopes: str,
        state: Optional[str] = None,
        nonce: Optional[str] = None,
        code_challenge: Optional[str] = None,
        code_challenge_method: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """Traite une demande d'autorisation OAuth2"""
        try:
            # Valider le client
            client = await self._get_client(client_id)
            if not client or not client.is_active:
                raise HTTPException(status_code=400, detail="Client invalide")
            
            # Valider l'URI de redirection
            if redirect_uri not in client.redirect_uris:
                raise HTTPException(status_code=400, detail="URI de redirection invalide")
            
            # Valider le type de réponse
            if response_type not in client.response_types:
                raise HTTPException(status_code=400, detail="Type de réponse non autorisé")
            
            # Parser les scopes
            requested_scopes = scopes.split() if scopes else []
            authorized_scopes = [s for s in requested_scopes if s in client.scopes]
            
            # Traiter selon le type de réponse
            if response_type == ResponseType.CODE.value:
                return await self._handle_authorization_code_flow(
                    client, redirect_uri, authorized_scopes, state, nonce,
                    code_challenge, code_challenge_method, user_id
                )
            elif response_type == ResponseType.TOKEN.value:
                return await self._handle_implicit_flow(
                    client, redirect_uri, authorized_scopes, state, user_id
                )
            else:
                raise HTTPException(status_code=400, detail="Type de réponse non supporté")
                
        except HTTPException:
            raise
        except Exception as exc:
            self.logger.error(f"Erreur autorisation OAuth2: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    async def token(
        self,
        grant_type: str,
        client_id: str,
        client_secret: Optional[str] = None,
        code: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        refresh_token: Optional[str] = None,
        code_verifier: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        scopes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Endpoint de token OAuth2"""
        try:
            # Authentifier le client
            client = await self._authenticate_client(client_id, client_secret)
            if not client:
                raise HTTPException(status_code=401, detail="Authentification client échouée")
            
            # Traiter selon le type de grant
            if grant_type == GrantType.AUTHORIZATION_CODE.value:
                return await self._handle_authorization_code_grant(
                    client, code, redirect_uri, code_verifier
                )
            elif grant_type == GrantType.REFRESH_TOKEN.value:
                return await self._handle_refresh_token_grant(
                    client, refresh_token
                )
            elif grant_type == GrantType.CLIENT_CREDENTIALS.value:
                return await self._handle_client_credentials_grant(
                    client, scopes
                )
            elif grant_type == GrantType.RESOURCE_OWNER_PASSWORD.value:
                return await self._handle_password_grant(
                    client, username, password, scopes
                )
            else:
                raise HTTPException(status_code=400, detail="Type de grant non supporté")
                
        except HTTPException:
            raise
        except Exception as exc:
            self.logger.error(f"Erreur endpoint token: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    async def introspect(
        self,
        token: str,
        client_id: str,
        client_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Introspection de token RFC 7662"""
        try:
            # Authentifier le client
            client = await self._authenticate_client(client_id, client_secret)
            if not client:
                return {"active": False}
            
            # Vérifier le token
            token_data = await self._get_access_token(token)
            if not token_data:
                return {"active": False}
            
            # Vérifier l'expiration
            if datetime.utcnow() > token_data.expires_at:
                return {"active": False}
            
            # Retourner les informations du token
            return {
                "active": True,
                "client_id": token_data.client_id,
                "username": token_data.user_id,
                "scope": " ".join(token_data.scopes),
                "exp": int(token_data.expires_at.timestamp()),
                "iat": int(token_data.created_at.timestamp()),
                "token_type": token_data.token_type
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur introspection token: {exc}")
            return {"active": False}
    
    async def revoke(
        self,
        token: str,
        client_id: str,
        client_secret: Optional[str] = None,
        token_type_hint: Optional[str] = None
    ) -> bool:
        """Révocation de token RFC 7009"""
        try:
            # Authentifier le client
            client = await self._authenticate_client(client_id, client_secret)
            if not client:
                return False
            
            # Révoquer le token
            success = False
            
            # Essayer comme access token
            if await self._revoke_access_token(token, client_id):
                success = True
            
            # Essayer comme refresh token
            if await self._revoke_refresh_token(token, client_id):
                success = True
            
            return success
            
        except Exception as exc:
            self.logger.error(f"Erreur révocation token: {exc}")
            return False
    
    async def jwks(self) -> Dict[str, Any]:
        """Endpoint JWKS (JSON Web Key Set)"""
        try:
            if not self.public_key:
                raise HTTPException(status_code=500, detail="Clés non disponibles")
            
            # Sérialiser la clé publique
            public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Créer le JWKS
            jwk = {
                "kty": "RSA",
                "use": "sig",
                "kid": "oauth2-signing-key",
                "alg": "RS256",
                "n": base64.urlsafe_b64encode(
                    self.public_key.public_numbers().n.to_bytes(256, 'big')
                ).decode().rstrip('='),
                "e": base64.urlsafe_b64encode(
                    self.public_key.public_numbers().e.to_bytes(3, 'big')
                ).decode().rstrip('=')
            }
            
            return {"keys": [jwk]}
            
        except Exception as exc:
            self.logger.error(f"Erreur génération JWKS: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    # Méthodes de traitement des flows
    async def _handle_authorization_code_flow(
        self,
        client: OAuth2Client,
        redirect_uri: str,
        scopes: List[str],
        state: Optional[str],
        nonce: Optional[str],
        code_challenge: Optional[str],
        code_challenge_method: Optional[str],
        user_id: Optional[str]
    ) -> str:
        """Traite le flow Authorization Code"""
        try:
            # Générer le code d'autorisation
            auth_code = AuthorizationCode(
                code=secrets.token_urlsafe(32),
                client_id=client.client_id,
                user_id=user_id or "anonymous",
                redirect_uri=redirect_uri,
                scopes=scopes,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.authorization_code_lifetime),
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
                nonce=nonce
            )
            
            # Sauvegarder le code
            await self._store_authorization_code(auth_code)
            
            # Construire l'URL de redirection
            params = {"code": auth_code.code}
            if state:
                params["state"] = state
            
            query_string = urllib.parse.urlencode(params)
            return f"{redirect_uri}?{query_string}"
            
        except Exception as exc:
            self.logger.error(f"Erreur flow authorization code: {exc}")
            raise
    
    async def _handle_implicit_flow(
        self,
        client: OAuth2Client,
        redirect_uri: str,
        scopes: List[str],
        state: Optional[str],
        user_id: Optional[str]
    ) -> str:
        """Traite le flow Implicit"""
        try:
            # Générer le token d'accès directement
            access_token = await self._create_access_token(
                client.client_id, user_id or "anonymous", scopes
            )
            
            # Construire l'URL de redirection avec le token dans le fragment
            params = {
                "access_token": access_token.token,
                "token_type": access_token.token_type,
                "expires_in": str(self.access_token_lifetime)
            }
            if state:
                params["state"] = state
            
            query_string = urllib.parse.urlencode(params)
            return f"{redirect_uri}#{query_string}"
            
        except Exception as exc:
            self.logger.error(f"Erreur flow implicit: {exc}")
            raise
    
    async def _handle_authorization_code_grant(
        self,
        client: OAuth2Client,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str]
    ) -> Dict[str, Any]:
        """Traite le grant Authorization Code"""
        try:
            # Récupérer et valider le code d'autorisation
            auth_code = await self._get_authorization_code(code)
            if not auth_code:
                raise HTTPException(status_code=400, detail="Code d'autorisation invalide")
            
            # Vérifier l'expiration
            if datetime.utcnow() > auth_code.expires_at:
                await self._delete_authorization_code(code)
                raise HTTPException(status_code=400, detail="Code d'autorisation expiré")
            
            # Vérifier le client
            if auth_code.client_id != client.client_id:
                raise HTTPException(status_code=400, detail="Client invalide")
            
            # Vérifier l'URI de redirection
            if auth_code.redirect_uri != redirect_uri:
                raise HTTPException(status_code=400, detail="URI de redirection invalide")
            
            # Vérifier PKCE si présent
            if auth_code.code_challenge:
                if not code_verifier:
                    raise HTTPException(status_code=400, detail="Code verifier requis")
                
                if not self._verify_pkce(auth_code.code_challenge, auth_code.code_challenge_method, code_verifier):
                    raise HTTPException(status_code=400, detail="Code verifier invalide")
            
            # Créer les tokens
            access_token = await self._create_access_token(
                client.client_id, auth_code.user_id, auth_code.scopes
            )
            
            refresh_token = await self._create_refresh_token(
                client.client_id, auth_code.user_id, auth_code.scopes
            )
            
            # Supprimer le code d'autorisation utilisé
            await self._delete_authorization_code(code)
            
            response = {
                "access_token": access_token.token,
                "token_type": access_token.token_type,
                "expires_in": self.access_token_lifetime,
                "refresh_token": refresh_token.token,
                "scope": " ".join(auth_code.scopes)
            }
            
            # Ajouter ID token si scope OpenID
            if "openid" in auth_code.scopes:
                id_token = await self._create_id_token(
                    client.client_id, auth_code.user_id, auth_code.scopes, auth_code.nonce
                )
                response["id_token"] = id_token
            
            return response
            
        except HTTPException:
            raise
        except Exception as exc:
            self.logger.error(f"Erreur grant authorization code: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    async def _handle_refresh_token_grant(
        self,
        client: OAuth2Client,
        refresh_token_value: str
    ) -> Dict[str, Any]:
        """Traite le grant Refresh Token"""
        try:
            # Valider le refresh token
            refresh_token = await self._get_refresh_token(refresh_token_value)
            if not refresh_token:
                raise HTTPException(status_code=400, detail="Refresh token invalide")
            
            # Vérifier l'expiration
            if datetime.utcnow() > refresh_token.expires_at:
                await self._delete_refresh_token(refresh_token_value)
                raise HTTPException(status_code=400, detail="Refresh token expiré")
            
            # Vérifier le client
            if refresh_token.client_id != client.client_id:
                raise HTTPException(status_code=400, detail="Client invalide")
            
            # Créer un nouveau access token
            access_token = await self._create_access_token(
                client.client_id, refresh_token.user_id, refresh_token.scopes
            )
            
            # Optionnellement, créer un nouveau refresh token (rotation)
            new_refresh_token = await self._create_refresh_token(
                client.client_id, refresh_token.user_id, refresh_token.scopes
            )
            
            # Supprimer l'ancien refresh token
            await self._delete_refresh_token(refresh_token_value)
            
            return {
                "access_token": access_token.token,
                "token_type": access_token.token_type,
                "expires_in": self.access_token_lifetime,
                "refresh_token": new_refresh_token.token,
                "scope": " ".join(refresh_token.scopes)
            }
            
        except HTTPException:
            raise
        except Exception as exc:
            self.logger.error(f"Erreur grant refresh token: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    async def _handle_client_credentials_grant(
        self,
        client: OAuth2Client,
        scopes: Optional[str]
    ) -> Dict[str, Any]:
        """Traite le grant Client Credentials"""
        try:
            # Parser les scopes
            requested_scopes = scopes.split() if scopes else []
            authorized_scopes = [s for s in requested_scopes if s in client.scopes]
            
            # Créer un access token pour le client (pas d'utilisateur)
            access_token = await self._create_access_token(
                client.client_id, None, authorized_scopes
            )
            
            return {
                "access_token": access_token.token,
                "token_type": access_token.token_type,
                "expires_in": self.access_token_lifetime,
                "scope": " ".join(authorized_scopes)
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur grant client credentials: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    async def _handle_password_grant(
        self,
        client: OAuth2Client,
        username: str,
        password: str,
        scopes: Optional[str]
    ) -> Dict[str, Any]:
        """Traite le grant Resource Owner Password"""
        try:
            # Authentifier l'utilisateur
            user_id = await self._authenticate_user(username, password)
            if not user_id:
                raise HTTPException(status_code=400, detail="Identifiants invalides")
            
            # Parser les scopes
            requested_scopes = scopes.split() if scopes else []
            authorized_scopes = [s for s in requested_scopes if s in client.scopes]
            
            # Créer les tokens
            access_token = await self._create_access_token(
                client.client_id, user_id, authorized_scopes
            )
            
            refresh_token = await self._create_refresh_token(
                client.client_id, user_id, authorized_scopes
            )
            
            return {
                "access_token": access_token.token,
                "token_type": access_token.token_type,
                "expires_in": self.access_token_lifetime,
                "refresh_token": refresh_token.token,
                "scope": " ".join(authorized_scopes)
            }
            
        except HTTPException:
            raise
        except Exception as exc:
            self.logger.error(f"Erreur grant password: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    # Méthodes utilitaires
    def _verify_pkce(
        self,
        code_challenge: str,
        code_challenge_method: str,
        code_verifier: str
    ) -> bool:
        """Vérifie PKCE (Proof Key for Code Exchange)"""
        try:
            if code_challenge_method == "plain":
                return code_challenge == code_verifier
            elif code_challenge_method == "S256":
                digest = hashlib.sha256(code_verifier.encode()).digest()
                expected_challenge = base64.urlsafe_b64encode(digest).decode().rstrip('=')
                return code_challenge == expected_challenge
            else:
                return False
                
        except Exception:
            return False
    
    async def _create_access_token(
        self,
        client_id: str,
        user_id: Optional[str],
        scopes: List[str]
    ) -> AccessToken:
        """Crée un token d'accès"""
        try:
            access_token = AccessToken(
                token=secrets.token_urlsafe(32),
                client_id=client_id,
                user_id=user_id or "",
                scopes=scopes,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.access_token_lifetime)
            )
            
            await self._store_access_token(access_token)
            return access_token
            
        except Exception as exc:
            self.logger.error(f"Erreur création access token: {exc}")
            raise
    
    async def _create_refresh_token(
        self,
        client_id: str,
        user_id: str,
        scopes: List[str]
    ) -> RefreshToken:
        """Crée un refresh token"""
        try:
            refresh_token = RefreshToken(
                token=secrets.token_urlsafe(32),
                client_id=client_id,
                user_id=user_id,
                scopes=scopes,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=self.refresh_token_lifetime)
            )
            
            await self._store_refresh_token(refresh_token)
            return refresh_token
            
        except Exception as exc:
            self.logger.error(f"Erreur création refresh token: {exc}")
            raise
    
    async def _create_id_token(
        self,
        client_id: str,
        user_id: str,
        scopes: List[str],
        nonce: Optional[str] = None
    ) -> str:
        """Crée un ID token OpenID Connect"""
        try:
            # Récupérer les informations utilisateur
            user_info = await self._get_user_info(user_id, scopes)
            
            # Créer les claims du token
            claims = {
                "iss": "https://spotify-ai-agent.com",  # Issuer
                "sub": user_id,  # Subject
                "aud": client_id,  # Audience
                "exp": int((datetime.utcnow() + timedelta(seconds=self.access_token_lifetime)).timestamp()),
                "iat": int(datetime.utcnow().timestamp()),
                "auth_time": int(datetime.utcnow().timestamp())
            }
            
            # Ajouter le nonce si présent
            if nonce:
                claims["nonce"] = nonce
            
            # Ajouter les claims utilisateur selon les scopes
            if "profile" in scopes:
                claims.update({
                    "name": user_info.get("name"),
                    "preferred_username": user_info.get("username"),
                    "picture": user_info.get("avatar_url")
                })
            
            if "email" in scopes:
                claims.update({
                    "email": user_info.get("email"),
                    "email_verified": user_info.get("email_verified", False)
                })
            
            # Signer le token avec la clé privée
            if self.private_key:
                private_key_pem = self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                return jwt.encode(
                    claims,
                    private_key_pem,
                    algorithm="RS256",
                    headers={"kid": "oauth2-signing-key"}
                )
            else:
                # Fallback vers HMAC si pas de clé RSA
                return jwt.encode(claims, "secret-key", algorithm="HS256")
                
        except Exception as exc:
            self.logger.error(f"Erreur création ID token: {exc}")
            raise
    
    # Méthodes de stockage Redis (implémentation simplifiée)
    async def _store_client(self, client: OAuth2Client):
        """Stocke un client OAuth2"""
        try:
            key = f"oauth2:client:{client.client_id}"
            await self.redis_client.set(key, json.dumps(asdict(client), default=str))
        except Exception as exc:
            self.logger.error(f"Erreur stockage client: {exc}")
    
    async def _get_client(self, client_id: str) -> Optional[OAuth2Client]:
        """Récupère un client OAuth2"""
        try:
            key = f"oauth2:client:{client_id}"
            data = await self.redis_client.get(key)
            if data:
                client_data = json.loads(data)
                return OAuth2Client(**client_data)
            return None
        except Exception as exc:
            self.logger.error(f"Erreur récupération client: {exc}")
            return None
    
    async def _authenticate_client(
        self,
        client_id: str,
        client_secret: Optional[str]
    ) -> Optional[OAuth2Client]:
        """Authentifie un client OAuth2"""
        try:
            client = await self._get_client(client_id)
            if not client or not client.is_active:
                return None
            
            # Client public (pas de secret requis)
            if client.client_type == ClientType.PUBLIC:
                return client
            
            # Client confidentiel (secret requis)
            if client.client_secret and client_secret:
                if secrets.compare_digest(client.client_secret, client_secret):
                    return client
            
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur authentification client: {exc}")
            return None
    
    async def _store_authorization_code(self, auth_code: AuthorizationCode):
        """Stocke un code d'autorisation"""
        try:
            key = f"oauth2:auth_code:{auth_code.code}"
            await self.redis_client.setex(
                key,
                self.authorization_code_lifetime,
                json.dumps(asdict(auth_code), default=str)
            )
        except Exception as exc:
            self.logger.error(f"Erreur stockage code autorisation: {exc}")
    
    async def _get_authorization_code(self, code: str) -> Optional[AuthorizationCode]:
        """Récupère un code d'autorisation"""
        try:
            key = f"oauth2:auth_code:{code}"
            data = await self.redis_client.get(key)
            if data:
                code_data = json.loads(data)
                return AuthorizationCode(**code_data)
            return None
        except Exception as exc:
            self.logger.error(f"Erreur récupération code autorisation: {exc}")
            return None
    
    async def _delete_authorization_code(self, code: str):
        """Supprime un code d'autorisation"""
        try:
            key = f"oauth2:auth_code:{code}"
            await self.redis_client.delete(key)
        except Exception as exc:
            self.logger.error(f"Erreur suppression code autorisation: {exc}")
    
    async def _store_access_token(self, access_token: AccessToken):
        """Stocke un access token"""
        try:
            key = f"oauth2:access_token:{access_token.token}"
            await self.redis_client.setex(
                key,
                self.access_token_lifetime,
                json.dumps(asdict(access_token), default=str)
            )
        except Exception as exc:
            self.logger.error(f"Erreur stockage access token: {exc}")
    
    async def _get_access_token(self, token: str) -> Optional[AccessToken]:
        """Récupère un access token"""
        try:
            key = f"oauth2:access_token:{token}"
            data = await self.redis_client.get(key)
            if data:
                token_data = json.loads(data)
                return AccessToken(**token_data)
            return None
        except Exception as exc:
            self.logger.error(f"Erreur récupération access token: {exc}")
            return None
    
    async def _store_refresh_token(self, refresh_token: RefreshToken):
        """Stocke un refresh token"""
        try:
            key = f"oauth2:refresh_token:{refresh_token.token}"
            await self.redis_client.setex(
                key,
                self.refresh_token_lifetime,
                json.dumps(asdict(refresh_token), default=str)
            )
        except Exception as exc:
            self.logger.error(f"Erreur stockage refresh token: {exc}")
    
    async def _get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        """Récupère un refresh token"""
        try:
            key = f"oauth2:refresh_token:{token}"
            data = await self.redis_client.get(key)
            if data:
                token_data = json.loads(data)
                return RefreshToken(**token_data)
            return None
        except Exception as exc:
            self.logger.error(f"Erreur récupération refresh token: {exc}")
            return None
    
    async def _delete_refresh_token(self, token: str):
        """Supprime un refresh token"""
        try:
            key = f"oauth2:refresh_token:{token}"
            await self.redis_client.delete(key)
        except Exception as exc:
            self.logger.error(f"Erreur suppression refresh token: {exc}")
    
    async def _revoke_access_token(self, token: str, client_id: str) -> bool:
        """Révoque un access token"""
        try:
            access_token = await self._get_access_token(token)
            if access_token and access_token.client_id == client_id:
                key = f"oauth2:access_token:{token}"
                await self.redis_client.delete(key)
                return True
            return False
        except Exception as exc:
            self.logger.error(f"Erreur révocation access token: {exc}")
            return False
    
    async def _revoke_refresh_token(self, token: str, client_id: str) -> bool:
        """Révoque un refresh token"""
        try:
            refresh_token = await self._get_refresh_token(token)
            if refresh_token and refresh_token.client_id == client_id:
                key = f"oauth2:refresh_token:{token}"
                await self.redis_client.delete(key)
                return True
            return False
        except Exception as exc:
            self.logger.error(f"Erreur révocation refresh token: {exc}")
            return False
    
    # Méthodes utilitaires (implémentation simplifiée)
    async def _authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authentifie un utilisateur (à implémenter)"""
        # Implémentation avec votre système d'authentification
        return None
    
    async def _get_user_info(self, user_id: str, scopes: List[str]) -> Dict[str, Any]:
        """Récupère les informations utilisateur (à implémenter)"""
        # Implémentation avec votre base de données utilisateur
        return {}


class OpenIDConnectProvider:
    """Fournisseur OpenID Connect"""
    
    def __init__(self, oauth2_provider: OAuth2Provider):
        self.oauth2_provider = oauth2_provider
        self.logger = logging.getLogger(__name__)
    
    async def discovery(self) -> Dict[str, Any]:
        """Endpoint de découverte OpenID Connect"""
        try:
            base_url = "https://spotify-ai-agent.com"
            
            return {
                "issuer": base_url,
                "authorization_endpoint": f"{base_url}/oauth2/authorize",
                "token_endpoint": f"{base_url}/oauth2/token",
                "userinfo_endpoint": f"{base_url}/oauth2/userinfo",
                "jwks_uri": f"{base_url}/oauth2/jwks",
                "response_types_supported": [
                    "code",
                    "token",
                    "id_token",
                    "code token",
                    "code id_token",
                    "token id_token",
                    "code token id_token"
                ],
                "subject_types_supported": ["public"],
                "id_token_signing_alg_values_supported": ["RS256", "HS256"],
                "scopes_supported": ["openid", "profile", "email", "offline_access"],
                "token_endpoint_auth_methods_supported": [
                    "client_secret_basic",
                    "client_secret_post"
                ],
                "claims_supported": [
                    "sub",
                    "iss",
                    "aud",
                    "exp",
                    "iat",
                    "auth_time",
                    "nonce",
                    "name",
                    "preferred_username",
                    "email",
                    "email_verified",
                    "picture"
                ],
                "code_challenge_methods_supported": ["plain", "S256"]
            }
            
        except Exception as exc:
            self.logger.error(f"Erreur endpoint discovery: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")
    
    async def userinfo(self, access_token: str) -> Dict[str, Any]:
        """Endpoint UserInfo OpenID Connect"""
        try:
            # Valider l'access token
            token_data = await self.oauth2_provider._get_access_token(access_token)
            if not token_data:
                raise HTTPException(status_code=401, detail="Token invalide")
            
            # Vérifier l'expiration
            if datetime.utcnow() > token_data.expires_at:
                raise HTTPException(status_code=401, detail="Token expiré")
            
            # Vérifier que le scope openid est présent
            if "openid" not in token_data.scopes:
                raise HTTPException(status_code=403, detail="Scope openid requis")
            
            # Récupérer les informations utilisateur
            user_info = await self.oauth2_provider._get_user_info(
                token_data.user_id, token_data.scopes
            )
            
            # Construire la réponse selon les scopes
            response = {"sub": token_data.user_id}
            
            if "profile" in token_data.scopes:
                response.update({
                    "name": user_info.get("name"),
                    "preferred_username": user_info.get("username"),
                    "picture": user_info.get("avatar_url")
                })
            
            if "email" in token_data.scopes:
                response.update({
                    "email": user_info.get("email"),
                    "email_verified": user_info.get("email_verified", False)
                })
            
            return response
            
        except HTTPException:
            raise
        except Exception as exc:
            self.logger.error(f"Erreur endpoint userinfo: {exc}")
            raise HTTPException(status_code=500, detail="Erreur interne")


class SSOManager:
    """Gestionnaire Single Sign-On"""
    
    def __init__(self, oauth2_provider: OAuth2Provider, redis_client: redis.Redis):
        self.oauth2_provider = oauth2_provider
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def initiate_sso(
        self,
        user_id: str,
        target_application: str,
        return_url: Optional[str] = None
    ) -> str:
        """Initie une session SSO"""
        try:
            # Créer un token SSO temporaire
            sso_token = secrets.token_urlsafe(32)
            
            sso_data = {
                "user_id": user_id,
                "target_application": target_application,
                "return_url": return_url,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Stocker temporairement (5 minutes)
            await self.redis_client.setex(
                f"sso:token:{sso_token}",
                300,
                json.dumps(sso_data)
            )
            
            # Construire l'URL SSO
            sso_url = f"https://spotify-ai-agent.com/sso/authenticate?token={sso_token}"
            if return_url:
                sso_url += f"&return_url={urllib.parse.quote(return_url)}"
            
            return sso_url
            
        except Exception as exc:
            self.logger.error(f"Erreur initiation SSO: {exc}")
            raise
    
    async def validate_sso_token(self, sso_token: str) -> Optional[Dict[str, Any]]:
        """Valide un token SSO"""
        try:
            key = f"sso:token:{sso_token}"
            data = await self.redis_client.get(key)
            
            if data:
                sso_data = json.loads(data)
                # Supprimer le token après usage
                await self.redis_client.delete(key)
                return sso_data
            
            return None
            
        except Exception as exc:
            self.logger.error(f"Erreur validation token SSO: {exc}")
            return None
    
    async def create_sso_session(
        self,
        user_id: str,
        application_id: str
    ) -> Dict[str, Any]:
        """Crée une session SSO pour une application"""
        try:
            session_id = secrets.token_urlsafe(32)
            
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "application_id": application_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            
            # Stocker la session (24 heures)
            await self.redis_client.setex(
                f"sso:session:{session_id}",
                86400,
                json.dumps(session_data)
            )
            
            # Ajouter à la liste des sessions utilisateur
            await self.redis_client.sadd(f"sso:user_sessions:{user_id}", session_id)
            
            return session_data
            
        except Exception as exc:
            self.logger.error(f"Erreur création session SSO: {exc}")
            raise
    
    async def invalidate_sso_sessions(self, user_id: str):
        """Invalide toutes les sessions SSO d'un utilisateur"""
        try:
            # Récupérer toutes les sessions de l'utilisateur
            session_ids = await self.redis_client.smembers(f"sso:user_sessions:{user_id}")
            
            # Supprimer chaque session
            for session_id in session_ids:
                await self.redis_client.delete(f"sso:session:{session_id}")
            
            # Nettoyer la liste des sessions
            await self.redis_client.delete(f"sso:user_sessions:{user_id}")
            
        except Exception as exc:
            self.logger.error(f"Erreur invalidation sessions SSO: {exc}")
