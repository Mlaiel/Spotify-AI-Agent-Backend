#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests pour SecurityUtils - Utilitaires de sécurité
Created Date: July 14, 2025
Author: Spotify AI Agent Development Team
"""

import pytest
import hashlib
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request, HTTPException

# Import du module à tester
from app.core.security.security_utils import SecurityUtils


class TestSecurityUtils:
    """Tests pour la classe SecurityUtils"""

    @pytest.fixture
    def security_utils(self):
        """Fixture pour SecurityUtils"""
        return SecurityUtils()

    def test_security_utils_initialization(self, security_utils):
        """Test d'initialisation de SecurityUtils"""
        assert security_utils is not None
        assert hasattr(security_utils, 'generate_secure_token')
        assert hasattr(security_utils, 'hash_password')
        assert hasattr(security_utils, 'verify_password')
        assert hasattr(security_utils, 'hash_token')
        assert hasattr(security_utils, 'sanitize_input')
        assert hasattr(security_utils, 'is_safe_redirect_url')

    def test_generate_secure_token(self, security_utils):
        """Test de génération de token sécurisé"""
        # Test avec longueur par défaut
        token = security_utils.generate_secure_token()
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Test avec longueur spécifique
        token_32 = security_utils.generate_secure_token(32)
        assert len(token_32) == 43  # 32 bytes urlsafe = 43 caractères
        
        # Test unicité
        token1 = security_utils.generate_secure_token()
        token2 = security_utils.generate_secure_token()
        assert token1 != token2

    def test_hash_password(self, security_utils):
        """Test de hachage de mot de passe"""
        password = "test_password_123"
        
        # Test de hachage
        hashed = security_utils.hash_password(password)
        assert isinstance(hashed, str)
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith('$2b$')  # bcrypt format
        
        # Test que le même mot de passe produit des hachés différents (salt)
        hashed2 = security_utils.hash_password(password)
        assert hashed != hashed2

    def test_verify_password(self, security_utils):
        """Test de vérification de mot de passe"""
        password = "test_password_123"
        wrong_password = "wrong_password"
        
        # Hacher le mot de passe
        hashed = security_utils.hash_password(password)
        
        # Test vérification correcte
        assert security_utils.verify_password(password, hashed) is True
        
        # Test vérification incorrecte
        assert security_utils.verify_password(wrong_password, hashed) is False

    def test_hash_token(self, security_utils):
        """Test de hachage de token"""
        token = "test_token_12345"
        
        hashed = security_utils.hash_token(token)
        assert isinstance(hashed, str)
        assert hashed != token
        assert len(hashed) == 64  # SHA256 = 64 caractères hex
        
        # Test reproductibilité (même token = même hash)
        hashed2 = security_utils.hash_token(token)
        assert hashed == hashed2

    def test_sanitize_input(self, security_utils):
        """Test de nettoyage des entrées utilisateur"""
        # Test nettoyage caractères dangereux
        dirty_input = "<script>alert('xss')</script>Hello World"
        clean = security_utils.sanitize_input(dirty_input)
        assert "script" not in clean
        assert "Hello World" in clean
        
        # Test avec None
        assert security_utils.sanitize_input(None) == ""
        
        # Test avec chaîne vide
        assert security_utils.sanitize_input("") == ""
        
        # Test nettoyage caractères spéciaux
        special_chars = '<>"&\x00'
        clean_special = security_utils.sanitize_input(special_chars)
        assert clean_special == ""

    def test_is_safe_redirect_url(self, security_utils):
        """Test de validation d'URL de redirection sûre"""
        # URLs sûres
        assert security_utils.is_safe_redirect_url("/dashboard") is True
        assert security_utils.is_safe_redirect_url("/") is True
        assert security_utils.is_safe_redirect_url("https://localhost/page") is True
        assert security_utils.is_safe_redirect_url("https://127.0.0.1/page") is True
        
        # URLs non sûres
        assert security_utils.is_safe_redirect_url("") is False
        assert security_utils.is_safe_redirect_url(None) is False
        assert security_utils.is_safe_redirect_url("https://evil.com/phishing") is False
        assert security_utils.is_safe_redirect_url("javascript:alert('xss')") is False

    def test_security_methods_edge_cases(self, security_utils):
        """Test de cas limites"""
        # Test avec chaînes vides
        empty_password = ""
        hashed_empty = security_utils.hash_password(empty_password)
        assert security_utils.verify_password(empty_password, hashed_empty) is True
        
        # Test avec caractères spéciaux dans mot de passe
        special_password = "P@ssw0rd!#$%"
        hashed_special = security_utils.hash_password(special_password)
        assert security_utils.verify_password(special_password, hashed_special) is True
        
        # Test token avec longueur minimale
        min_token = security_utils.generate_secure_token(1)
        assert len(min_token) > 0

    def test_hash_consistency(self, security_utils):
        """Test de cohérence des fonctions de hachage"""
        # Token hashing doit être déterministe
        token = "consistent_token_123"
        hash1 = security_utils.hash_token(token)
        hash2 = security_utils.hash_token(token)
        assert hash1 == hash2
        
        # Password hashing doit être non-déterministe (salt)
        password = "consistent_password_123"
        password_hash1 = security_utils.hash_password(password)
        password_hash2 = security_utils.hash_password(password)
        assert password_hash1 != password_hash2
        
        # Mais les deux doivent être vérifiables
        assert security_utils.verify_password(password, password_hash1) is True
        assert security_utils.verify_password(password, password_hash2) is True

    def test_input_sanitization_comprehensive(self, security_utils):
        """Test complet de nettoyage d'entrées"""
        test_cases = [
            # (input, expected_result_should_contain)
            ("<script>", ""),  # Script complètement supprimé
            ("Hello<script>World", "HelloWorld"),  # Script supprimé, contenu conservé
            ('Test"quote', "Testquote"),  # Guillemets supprimés
            ("Test'apostrophe", "Testapostrophe"),  # Apostrophes supprimées
            ("Test&amp;", "Testamp;"),  # Caractère & supprimé
            ("  spaces  ", "spaces"),  # Espaces nettoyés
            ("multiple<>'\"&chars", "multiplechars"),  # Tous caractères dangereux supprimés
        ]
        
        for input_str, expected_result in test_cases:
            result = security_utils.sanitize_input(input_str)
            assert result == expected_result, f"Input: '{input_str}' -> Expected: '{expected_result}' -> Got: '{result}'"

    def test_url_validation_comprehensive(self, security_utils):
        """Test complet de validation d'URL"""
        safe_urls = [
            "/",
            "/dashboard",
            "/user/profile",
            "/api/v1/data",
            "https://localhost/callback",
            "https://127.0.0.1:8000/auth"
        ]
        
        unsafe_urls = [
            "http://evil.com",
            "https://phishing.site.com",
            "javascript:void(0)",
            "data:text/html,<script>",
            "ftp://files.com",
            "//evil.com",
            "mailto:admin@evil.com"
        ]
        
        for url in safe_urls:
            assert security_utils.is_safe_redirect_url(url) is True, f"URL should be safe: {url}"
        
        for url in unsafe_urls:
            assert security_utils.is_safe_redirect_url(url) is False, f"URL should be unsafe: {url}"


class TestSecurityUtilsIntegration:
    """Tests d'intégration pour SecurityUtils"""

    @pytest.fixture
    def security_utils(self):
        """Fixture pour SecurityUtils"""
        return SecurityUtils()

    def test_complete_authentication_flow(self, security_utils):
        """Test de flux d'authentification complet"""
        # 1. Générer un mot de passe et le hacher
        password = "TestPassword123!"
        hashed = security_utils.hash_password(password)
        
        # 2. Vérifier le mot de passe
        assert security_utils.verify_password(password, hashed) is True
        
        # 3. Générer un token
        token = security_utils.generate_secure_token()
        assert len(token) > 0
        
        # 4. Hacher le token pour stockage
        token_hash = security_utils.hash_token(token)
        assert len(token_hash) == 64

    def test_input_processing_flow(self, security_utils):
        """Test de flux de traitement d'entrées"""
        # 1. Nettoyer les entrées utilisateur
        user_input = "<script>alert('test')</script>Clean Content"
        clean_input = security_utils.sanitize_input(user_input)
        assert "Clean Content" in clean_input
        assert "<script>" not in clean_input
        
        # 2. Valider URL de redirection
        redirect_url = "/dashboard"
        assert security_utils.is_safe_redirect_url(redirect_url) is True
        
        # 3. Traiter URL dangereuse
        dangerous_url = "https://phishing.com"
        assert security_utils.is_safe_redirect_url(dangerous_url) is False

    def test_security_token_management(self, security_utils):
        """Test de gestion des tokens de sécurité"""
        # 1. Générer plusieurs tokens uniques
        tokens = [security_utils.generate_secure_token() for _ in range(5)]
        assert len(set(tokens)) == 5  # Tous uniques
        
        # 2. Hacher les tokens pour stockage
        token_hashes = [security_utils.hash_token(token) for token in tokens]
        assert len(set(token_hashes)) == 5  # Tous uniques
        
        # 3. Vérifier la reproductibilité des hashes
        for token, token_hash in zip(tokens, token_hashes):
            assert security_utils.hash_token(token) == token_hash

    def test_password_security_levels(self, security_utils):
        """Test de différents niveaux de sécurité des mots de passe"""
        passwords = [
            "simple",
            "Complex123",
            "VeryC0mpl3x!P@ssw0rd",
            "🔒Emoji123!",
            " spaces around ",
            ""
        ]
        
        for password in passwords:
            # Chaque mot de passe doit pouvoir être haché et vérifié
            hashed = security_utils.hash_password(password)
            assert security_utils.verify_password(password, hashed) is True
            
            # Les mots de passe différents doivent produire des hashes différents
            if password:  # Skip empty password for this test
                hashed2 = security_utils.hash_password(password)
                assert hashed != hashed2  # Different salt = different hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
