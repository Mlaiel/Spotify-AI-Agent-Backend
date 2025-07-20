"""
Utilitaires de sécurité pour l'authentification
"""

import hashlib
import secrets
import bcrypt
import re
from typing import Optional, Dict, Any


class SecurityUtils:
    """Classe utilitaire pour les opérations de sécurité"""
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Hasher un token pour stockage sécurisé"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hasher un mot de passe avec bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Vérifier un mot de passe"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Générer un token sécurisé"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Nettoyer une chaîne d'entrée de manière sécurisée"""
        if not isinstance(input_str, str):
            return ""
        
        # Supprimer les balises HTML/XML complètes
        input_str = re.sub(r'<[^>]*>', '', input_str)
        
        # Supprimer les caractères dangereux
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
        for char in dangerous_chars:
            input_str = input_str.replace(char, '')
        
        # Supprimer les scripts JavaScript
        input_str = re.sub(r'javascript:', '', input_str, flags=re.IGNORECASE)
        input_str = re.sub(r'vbscript:', '', input_str, flags=re.IGNORECASE)
        input_str = re.sub(r'on\w+\s*=', '', input_str, flags=re.IGNORECASE)
        
        return input_str.strip()
    
    @staticmethod
    def is_safe_redirect_url(url: str) -> bool:
        """Vérifier si une URL de redirection est sûre"""
        if not url:
            return False
        
        # Rejeter les URLs avec des protocoles dangereux
        dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:', 'ftp:', 'mailto:']
        url_lower = url.lower()
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return False
        
        # Rejeter les URLs qui commencent par //
        if url.startswith('//'):
            return False
        
        # Accepter les URLs relatives (commençant par /)
        if url.startswith('/'):
            return True
        
        # Vérifier que l'URL commence par https:// et un domaine autorisé
        allowed_domains = ['localhost', '127.0.0.1']
        
        for domain in allowed_domains:
            if url.startswith(f'https://{domain}'):
                return True
            # Aussi accepter avec port
            if url.startswith(f'https://{domain}:'):
                return True
        
        return False
