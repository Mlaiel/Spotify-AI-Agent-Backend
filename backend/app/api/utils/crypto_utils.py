"""
🎵 Spotify AI Agent - Cryptographic Utilities
=============================================

Utilitaires enterprise pour la cryptographie et la sécurité
avec chiffrement AES, hashing sécurisé et gestion des clés.

Architecture:
- Chiffrement/déchiffrement AES
- Hashing sécurisé (bcrypt, scrypt, argon2)
- Génération de clés et tokens
- Signatures numériques
- Validation d'intégrité
- Chiffrement asymétrique

🎖️ Développé par l'équipe d'experts enterprise
"""

import hashlib
import secrets
import base64
import hmac
import json
from typing import Optional, Dict, Any, Union, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import bcrypt
import argon2


# =============================================================================
# CHIFFREMENT SYMÉTRIQUE
# =============================================================================

class SecureEncryption:
    """Gestionnaire de chiffrement symétrique enterprise"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            self.key = Fernet.generate_key()
            self.fernet = Fernet(self.key)
        else:
            self.key = key
            self.fernet = Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Chiffre des données
        
        Args:
            data: Données à chiffrer
            
        Returns:
            Données chiffrées en base64
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Déchiffre des données
        
        Args:
            encrypted_data: Données chiffrées en base64
            
        Returns:
            Données déchiffrées
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def encrypt_json(self, data: Dict[str, Any]) -> str:
        """
        Chiffre un objet JSON
        
        Args:
            data: Objet à chiffrer
            
        Returns:
            JSON chiffré
        """
        json_data = json.dumps(data, separators=(',', ':'))
        return self.encrypt(json_data)
    
    def decrypt_json(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Déchiffre un objet JSON
        
        Args:
            encrypted_data: JSON chiffré
            
        Returns:
            Objet déchiffré
        """
        decrypted_json = self.decrypt(encrypted_data)
        return json.loads(decrypted_json)


def generate_encryption_key(password: str, salt: Optional[bytes] = None) -> bytes:
    """
    Génère une clé de chiffrement à partir d'un mot de passe
    
    Args:
        password: Mot de passe
        salt: Sel (généré si None)
        
    Returns:
        Clé de chiffrement
    """
    if salt is None:
        salt = secrets.token_bytes(32)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
    return key


def aes_encrypt(data: bytes, key: bytes, mode: str = 'GCM') -> Tuple[bytes, bytes, bytes]:
    """
    Chiffrement AES avec différents modes
    
    Args:
        data: Données à chiffrer
        key: Clé de chiffrement (32 bytes pour AES-256)
        mode: Mode de chiffrement (GCM, CBC)
        
    Returns:
        Tuple (données_chiffrées, iv, tag)
    """
    if mode == 'GCM':
        iv = secrets.token_bytes(12)  # 96 bits pour GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return ciphertext, iv, encryptor.tag
    
    elif mode == 'CBC':
        iv = secrets.token_bytes(16)  # 128 bits pour CBC
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Padding PKCS7
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return ciphertext, iv, b''
    
    else:
        raise ValueError(f"Mode de chiffrement non supporté: {mode}")


def aes_decrypt(ciphertext: bytes, key: bytes, iv: bytes, 
               tag: Optional[bytes] = None, mode: str = 'GCM') -> bytes:
    """
    Déchiffrement AES
    
    Args:
        ciphertext: Données chiffrées
        key: Clé de déchiffrement
        iv: Vecteur d'initialisation
        tag: Tag d'authentification (pour GCM)
        mode: Mode de déchiffrement
        
    Returns:
        Données déchiffrées
    """
    if mode == 'GCM':
        if tag is None:
            raise ValueError("Tag requis pour le mode GCM")
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    elif mode == 'CBC':
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Supprimer le padding PKCS7
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    else:
        raise ValueError(f"Mode de déchiffrement non supporté: {mode}")


# =============================================================================
# HASHING SÉCURISÉ
# =============================================================================

def hash_password(password: str, algorithm: str = 'argon2') -> str:
    """
    Hash sécurisé d'un mot de passe
    
    Args:
        password: Mot de passe à hasher
        algorithm: Algorithme (argon2, bcrypt, scrypt)
        
    Returns:
        Hash du mot de passe
    """
    if algorithm == 'argon2':
        ph = argon2.PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=1,
            hash_len=32,
            salt_len=16
        )
        return ph.hash(password)
    
    elif algorithm == 'bcrypt':
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    elif algorithm == 'scrypt':
        salt = secrets.token_bytes(32)
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            n=2**14,
            r=8,
            p=1,
        )
        key = kdf.derive(password.encode('utf-8'))
        return base64.b64encode(salt + key).decode('utf-8')
    
    else:
        raise ValueError(f"Algorithme non supporté: {algorithm}")


def verify_password(password: str, hashed: str, algorithm: str = 'argon2') -> bool:
    """
    Vérifie un mot de passe contre son hash
    
    Args:
        password: Mot de passe en clair
        hashed: Hash stocké
        algorithm: Algorithme utilisé
        
    Returns:
        True si le mot de passe correspond
    """
    try:
        if algorithm == 'argon2':
            ph = argon2.PasswordHasher()
            ph.verify(hashed, password)
            return True
        
        elif algorithm == 'bcrypt':
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
        elif algorithm == 'scrypt':
            decoded = base64.b64decode(hashed.encode('utf-8'))
            salt = decoded[:32]
            stored_key = decoded[32:]
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,
                r=8,
                p=1,
            )
            
            kdf.verify(password.encode('utf-8'), stored_key)
            return True
        
        else:
            return False
    
    except Exception:
        return False


def secure_hash(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """
    Hash sécurisé de données
    
    Args:
        data: Données à hasher
        algorithm: Algorithme de hash
        
    Returns:
        Hash hexadécimal
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if algorithm == 'sha256':
        return hashlib.sha256(data).hexdigest()
    elif algorithm == 'sha512':
        return hashlib.sha512(data).hexdigest()
    elif algorithm == 'blake2b':
        return hashlib.blake2b(data).hexdigest()
    elif algorithm == 'sha3_256':
        return hashlib.sha3_256(data).hexdigest()
    else:
        raise ValueError(f"Algorithme non supporté: {algorithm}")


def hmac_sign(data: Union[str, bytes], key: Union[str, bytes], 
              algorithm: str = 'sha256') -> str:
    """
    Signature HMAC
    
    Args:
        data: Données à signer
        key: Clé de signature
        algorithm: Algorithme de hash
        
    Returns:
        Signature HMAC
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    if isinstance(key, str):
        key = key.encode('utf-8')
    
    if algorithm == 'sha256':
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    elif algorithm == 'sha512':
        return hmac.new(key, data, hashlib.sha512).hexdigest()
    else:
        raise ValueError(f"Algorithme non supporté: {algorithm}")


def verify_hmac(data: Union[str, bytes], signature: str, 
               key: Union[str, bytes], algorithm: str = 'sha256') -> bool:
    """
    Vérifie une signature HMAC
    
    Args:
        data: Données originales
        signature: Signature à vérifier
        key: Clé de signature
        algorithm: Algorithme utilisé
        
    Returns:
        True si la signature est valide
    """
    expected_signature = hmac_sign(data, key, algorithm)
    return hmac.compare_digest(signature, expected_signature)


# =============================================================================
# GÉNÉRATION DE TOKENS
# =============================================================================

def generate_secure_token(length: int = 32, url_safe: bool = True) -> str:
    """
    Génère un token sécurisé
    
    Args:
        length: Longueur en bytes
        url_safe: Token compatible URL
        
    Returns:
        Token généré
    """
    token_bytes = secrets.token_bytes(length)
    
    if url_safe:
        return base64.urlsafe_b64encode(token_bytes).decode('utf-8').rstrip('=')
    else:
        return base64.b64encode(token_bytes).decode('utf-8')


def generate_api_key(prefix: str = '', length: int = 32) -> str:
    """
    Génère une clé API
    
    Args:
        prefix: Préfixe de la clé
        length: Longueur de la partie aléatoire
        
    Returns:
        Clé API
    """
    random_part = generate_secure_token(length)
    
    if prefix:
        return f"{prefix}_{random_part}"
    else:
        return random_part


def generate_session_id() -> str:
    """
    Génère un identifiant de session sécurisé
    
    Returns:
        ID de session
    """
    return generate_secure_token(24, url_safe=True)


def generate_csrf_token() -> str:
    """
    Génère un token CSRF
    
    Returns:
        Token CSRF
    """
    return generate_secure_token(16, url_safe=True)


# =============================================================================
# CHIFFREMENT ASYMÉTRIQUE
# =============================================================================

def generate_rsa_keypair(key_size: int = 2048) -> Tuple[bytes, bytes]:
    """
    Génère une paire de clés RSA
    
    Args:
        key_size: Taille de la clé en bits
        
    Returns:
        Tuple (clé_privée, clé_publique)
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem


def rsa_encrypt(data: bytes, public_key_pem: bytes) -> bytes:
    """
    Chiffrement RSA
    
    Args:
        data: Données à chiffrer
        public_key_pem: Clé publique au format PEM
        
    Returns:
        Données chiffrées
    """
    public_key = serialization.load_pem_public_key(public_key_pem)
    
    encrypted = public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    return encrypted


def rsa_decrypt(encrypted_data: bytes, private_key_pem: bytes) -> bytes:
    """
    Déchiffrement RSA
    
    Args:
        encrypted_data: Données chiffrées
        private_key_pem: Clé privée au format PEM
        
    Returns:
        Données déchiffrées
    """
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    
    decrypted = private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    return decrypted


def rsa_sign(data: bytes, private_key_pem: bytes) -> bytes:
    """
    Signature RSA
    
    Args:
        data: Données à signer
        private_key_pem: Clé privée au format PEM
        
    Returns:
        Signature
    """
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    
    signature = private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    return signature


def rsa_verify_signature(data: bytes, signature: bytes, public_key_pem: bytes) -> bool:
    """
    Vérification de signature RSA
    
    Args:
        data: Données originales
        signature: Signature à vérifier
        public_key_pem: Clé publique au format PEM
        
    Returns:
        True si la signature est valide
    """
    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
        
        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return True
    
    except Exception:
        return False


# =============================================================================
# UTILITAIRES DE SÉCURITÉ
# =============================================================================

def constant_time_compare(a: str, b: str) -> bool:
    """
    Comparaison en temps constant pour éviter les attaques temporelles
    
    Args:
        a: Première chaîne
        b: Deuxième chaîne
        
    Returns:
        True si les chaînes sont identiques
    """
    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


def secure_random_choice(choices: list) -> Any:
    """
    Choix aléatoire cryptographiquement sécurisé
    
    Args:
        choices: Liste des choix possibles
        
    Returns:
        Élément choisi aléatoirement
    """
    if not choices:
        raise ValueError("La liste de choix ne peut pas être vide")
    
    return secrets.choice(choices)


def generate_salt(length: int = 32) -> bytes:
    """
    Génère un sel cryptographique
    
    Args:
        length: Longueur du sel en bytes
        
    Returns:
        Sel généré
    """
    return secrets.token_bytes(length)


def derive_key_from_password(password: str, salt: bytes, 
                           key_length: int = 32, iterations: int = 100000) -> bytes:
    """
    Dérive une clé à partir d'un mot de passe
    
    Args:
        password: Mot de passe
        salt: Sel
        key_length: Longueur de la clé
        iterations: Nombre d'itérations
        
    Returns:
        Clé dérivée
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        iterations=iterations,
    )
    
    return kdf.derive(password.encode('utf-8'))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SecureEncryption",
    "generate_encryption_key",
    "aes_encrypt",
    "aes_decrypt",
    "hash_password",
    "verify_password",
    "secure_hash",
    "hmac_sign",
    "verify_hmac",
    "generate_secure_token",
    "generate_api_key",
    "generate_session_id",
    "generate_csrf_token",
    "generate_rsa_keypair",
    "rsa_encrypt",
    "rsa_decrypt",
    "rsa_sign",
    "rsa_verify_signature",
    "constant_time_compare",
    "secure_random_choice",
    "generate_salt",
    "derive_key_from_password"
]
