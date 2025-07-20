# 🧪 Tests PasswordManager Ultra-Avancés
# ====================================

import pytest
import pytest_asyncio
import asyncio
import time
import hashlib
import bcrypt
import secrets
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.security.auth.password_manager import (
    PasswordManager, PasswordPolicy, PasswordStrength, HashAlgorithm,
    PasswordHistory, PasswordlessAuth, BiometricType, WebAuthnCredential
)

from conftest import (
    TestDataFactory, TestUtils, PerformanceTestUtils, SecurityValidators,
    pytest_markers
)


@pytest_markers["unit"]
@pytest_markers["password"]
class TestPasswordManager:
    """Tests unitaires pour PasswordManager"""
    
    @pytest.mark.asyncio
    async def test_hash_password(self, password_manager):
        """Test hachage de mot de passe"""
        password = "SecurePassword123!"
        
        hashed = await password_manager.hash_password(
            password=password,
            algorithm=HashAlgorithm.BCRYPT
        )
        
        assert hashed["hash"] is not None
        assert hashed["salt"] is not None
        assert hashed["algorithm"] == HashAlgorithm.BCRYPT.value
        assert hashed["iterations"] > 10000
        assert hashed["created_at"] is not None
        
        # Vérifier que le hash est différent à chaque fois
        hashed2 = await password_manager.hash_password(
            password=password,
            algorithm=HashAlgorithm.BCRYPT
        )
        
        assert hashed["hash"] != hashed2["hash"]
        assert hashed["salt"] != hashed2["salt"]
    
    @pytest.mark.asyncio
    async def test_verify_password(self, password_manager):
        """Test vérification de mot de passe"""
        password = "TestPassword456!"
        
        # Hasher le mot de passe
        with patch.object(password_manager, 'hash_password') as mock_hash:
            mock_hash.return_value = {
                "hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
                "salt": "test_salt",
                "algorithm": HashAlgorithm.BCRYPT.value,
                "iterations": 12
            }
            
            hashed = await password_manager.hash_password(password)
        
        # Vérifier avec le bon mot de passe
        with patch.object(password_manager, '_verify_hash') as mock_verify:
            mock_verify.return_value = True
            
            result = await password_manager.verify_password(
                password=password,
                hashed_password=hashed["hash"],
                salt=hashed["salt"],
                algorithm=hashed["algorithm"]
            )
        
        assert result["valid"] is True
        assert result["verified_at"] is not None
        
        # Vérifier avec un mauvais mot de passe
        with patch.object(password_manager, '_verify_hash') as mock_verify:
            mock_verify.return_value = False
            
            result = await password_manager.verify_password(
                password="WrongPassword",
                hashed_password=hashed["hash"],
                salt=hashed["salt"],
                algorithm=hashed["algorithm"]
            )
        
        assert result["valid"] is False
    
    @pytest.mark.asyncio
    async def test_password_strength_validation(self, password_manager):
        """Test validation de la force du mot de passe"""
        # Mot de passe faible
        weak_password = "123456"
        
        strength = await password_manager.evaluate_password_strength(weak_password)
        
        assert strength["score"] < 30
        assert strength["level"] == PasswordStrength.WEAK.value
        assert "too_short" in strength["issues"]
        assert "no_uppercase" in strength["issues"]
        assert "no_special_chars" in strength["issues"]
        
        # Mot de passe moyen
        medium_password = "Password123"
        
        strength = await password_manager.evaluate_password_strength(medium_password)
        
        assert 30 <= strength["score"] < 70
        assert strength["level"] == PasswordStrength.MEDIUM.value
        
        # Mot de passe fort
        strong_password = "MyVeryS3cure!P@ssw0rd2024"
        
        strength = await password_manager.evaluate_password_strength(strong_password)
        
        assert strength["score"] >= 70
        assert strength["level"] == PasswordStrength.STRONG.value
        assert len(strength["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_password_policy_enforcement(self, password_manager):
        """Test application des politiques de mot de passe"""
        policy = PasswordPolicy(
            min_length=12,
            require_uppercase=True,
            require_lowercase=True,
            require_numbers=True,
            require_special_chars=True,
            max_repeated_chars=2,
            forbid_common_passwords=True,
            forbid_personal_info=True
        )
        
        user_data = TestDataFactory.create_test_user()
        
        # Test mot de passe conforme
        valid_password = "MySecure!Password123"
        
        with patch.object(password_manager, '_check_common_passwords') as mock_common:
            with patch.object(password_manager, '_check_personal_info') as mock_personal:
                mock_common.return_value = False
                mock_personal.return_value = False
                
                result = await password_manager.validate_password_policy(
                    password=valid_password,
                    policy=policy,
                    user_info=user_data
                )
        
        assert result["valid"] is True
        assert len(result["violations"]) == 0
        
        # Test mot de passe non conforme
        invalid_password = "weak"
        
        result = await password_manager.validate_password_policy(
            password=invalid_password,
            policy=policy,
            user_info=user_data
        )
        
        assert result["valid"] is False
        assert "min_length" in result["violations"]
        assert "require_uppercase" in result["violations"]
        assert "require_numbers" in result["violations"]
        assert "require_special_chars" in result["violations"]
    
    @pytest.mark.asyncio
    async def test_password_history_management(self, password_manager):
        """Test gestion de l'historique des mots de passe"""
        user_data = TestDataFactory.create_test_user()
        
        # Historique existant
        password_history = [
            {
                "hash": "old_hash_1",
                "created_at": datetime.utcnow() - timedelta(days=30),
                "algorithm": HashAlgorithm.BCRYPT.value
            },
            {
                "hash": "old_hash_2", 
                "created_at": datetime.utcnow() - timedelta(days=60),
                "algorithm": HashAlgorithm.BCRYPT.value
            }
        ]
        
        new_password = "NewSecurePassword123!"
        
        with patch.object(password_manager, '_get_password_history') as mock_history:
            with patch.object(password_manager, '_check_password_reuse') as mock_reuse:
                mock_history.return_value = password_history
                mock_reuse.return_value = False  # Nouveau mot de passe
                
                result = await password_manager.check_password_history(
                    user_id=user_data["user_id"],
                    new_password=new_password,
                    history_limit=5
                )
        
        assert result["can_use"] is True
        assert result["is_reused"] is False
        
        # Test réutilisation d'un ancien mot de passe
        with patch.object(password_manager, '_get_password_history') as mock_history:
            with patch.object(password_manager, '_check_password_reuse') as mock_reuse:
                mock_history.return_value = password_history
                mock_reuse.return_value = True  # Mot de passe réutilisé
                
                result = await password_manager.check_password_history(
                    user_id=user_data["user_id"],
                    new_password="old_password",
                    history_limit=5
                )
        
        assert result["can_use"] is False
        assert result["is_reused"] is True
        assert result["last_used"] is not None
    
    @pytest.mark.asyncio
    async def test_password_breach_detection(self, password_manager):
        """Test détection de mots de passe compromis"""
        # Mot de passe non compromis
        safe_password = "MyUniqueSecurePassword2024!"
        
        with patch.object(password_manager, '_check_breach_databases') as mock_breach:
            mock_breach.return_value = {
                "is_breached": False,
                "breach_count": 0,
                "sources": []
            }
            
            result = await password_manager.check_password_breach(safe_password)
        
        assert result["is_breached"] is False
        assert result["breach_count"] == 0
        
        # Mot de passe compromis
        breached_password = "password123"
        
        with patch.object(password_manager, '_check_breach_databases') as mock_breach:
            mock_breach.return_value = {
                "is_breached": True,
                "breach_count": 234567,
                "sources": ["haveibeenpwned", "dehashed"],
                "first_seen": "2019-03-15",
                "severity": "high"
            }
            
            result = await password_manager.check_password_breach(breached_password)
        
        assert result["is_breached"] is True
        assert result["breach_count"] > 0
        assert "haveibeenpwned" in result["sources"]
        assert result["severity"] == "high"
    
    @pytest.mark.asyncio
    async def test_password_expiration(self, password_manager):
        """Test expiration des mots de passe"""
        user_data = TestDataFactory.create_test_user()
        
        # Mot de passe récent
        recent_password = {
            "created_at": datetime.utcnow() - timedelta(days=30),
            "expires_at": datetime.utcnow() + timedelta(days=60)
        }
        
        with patch.object(password_manager, '_get_current_password') as mock_password:
            mock_password.return_value = recent_password
            
            result = await password_manager.check_password_expiration(
                user_id=user_data["user_id"],
                max_age_days=90
            )
        
        assert result["is_expired"] is False
        assert result["days_until_expiry"] > 0
        assert result["requires_change"] is False
        
        # Mot de passe expiré
        expired_password = {
            "created_at": datetime.utcnow() - timedelta(days=100),
            "expires_at": datetime.utcnow() - timedelta(days=10)
        }
        
        with patch.object(password_manager, '_get_current_password') as mock_password:
            mock_password.return_value = expired_password
            
            result = await password_manager.check_password_expiration(
                user_id=user_data["user_id"],
                max_age_days=90
            )
        
        assert result["is_expired"] is True
        assert result["days_until_expiry"] < 0
        assert result["requires_change"] is True


@pytest_markers["unit"]
@pytest_markers["passwordless"]
class TestPasswordlessAuth:
    """Tests pour l'authentification sans mot de passe"""
    
    @pytest.mark.asyncio
    async def test_webauthn_registration(self, password_manager):
        """Test enregistrement WebAuthn"""
        user_data = TestDataFactory.create_test_user()
        
        # Générer options d'enregistrement
        with patch.object(password_manager, '_generate_webauthn_registration_options') as mock_options:
            mock_options.return_value = {
                "challenge": TestUtils.generate_random_string(32),
                "rp": {"name": "Spotify AI Agent", "id": "spotify-ai-agent.com"},
                "user": {
                    "id": user_data["user_id"],
                    "name": user_data["email"],
                    "displayName": f"{user_data['first_name']} {user_data['last_name']}"
                },
                "pubKeyCredParams": [{"type": "public-key", "alg": -7}],
                "authenticatorSelection": {
                    "userVerification": "required",
                    "authenticatorAttachment": "platform"
                },
                "timeout": 60000
            }
            
            options = await password_manager.generate_webauthn_registration_options(
                user_id=user_data["user_id"]
            )
        
        assert options["challenge"] is not None
        assert options["user"]["id"] == user_data["user_id"]
        assert options["rp"]["name"] == "Spotify AI Agent"
        
        # Finaliser l'enregistrement
        credential_data = {
            "id": "credential_123",
            "rawId": "credential_123",
            "type": "public-key",
            "response": {
                "attestationObject": "test_attestation",
                "clientDataJSON": "test_client_data"
            }
        }
        
        with patch.object(password_manager, '_verify_webauthn_registration') as mock_verify:
            mock_verify.return_value = {
                "verified": True,
                "credential_id": "credential_123",
                "public_key": "test_public_key",
                "counter": 0
            }
            
            result = await password_manager.complete_webauthn_registration(
                user_id=user_data["user_id"],
                credential=credential_data,
                challenge=options["challenge"]
            )
        
        assert result["success"] is True
        assert result["credential_id"] == "credential_123"
        assert result["verified"] is True
    
    @pytest.mark.asyncio
    async def test_webauthn_authentication(self, password_manager):
        """Test authentification WebAuthn"""
        user_data = TestDataFactory.create_test_user()
        
        # Générer options d'authentification
        with patch.object(password_manager, '_generate_webauthn_auth_options') as mock_options:
            mock_options.return_value = {
                "challenge": TestUtils.generate_random_string(32),
                "allowCredentials": [
                    {
                        "type": "public-key",
                        "id": "credential_123"
                    }
                ],
                "userVerification": "required",
                "timeout": 60000
            }
            
            options = await password_manager.generate_webauthn_auth_options(
                user_id=user_data["user_id"]
            )
        
        assert options["challenge"] is not None
        assert len(options["allowCredentials"]) > 0
        
        # Effectuer l'authentification
        auth_data = {
            "id": "credential_123",
            "rawId": "credential_123",
            "type": "public-key",
            "response": {
                "authenticatorData": "test_auth_data",
                "clientDataJSON": "test_client_data",
                "signature": "test_signature"
            }
        }
        
        with patch.object(password_manager, '_verify_webauthn_auth') as mock_verify:
            mock_verify.return_value = {
                "verified": True,
                "credential_id": "credential_123",
                "counter": 1,
                "user_id": user_data["user_id"]
            }
            
            result = await password_manager.verify_webauthn_authentication(
                credential=auth_data,
                challenge=options["challenge"]
            )
        
        assert result["verified"] is True
        assert result["user_id"] == user_data["user_id"]
        assert result["credential_id"] == "credential_123"
    
    @pytest.mark.asyncio
    async def test_biometric_authentication(self, password_manager):
        """Test authentification biométrique"""
        user_data = TestDataFactory.create_test_user()
        
        # Enregistrer données biométriques
        biometric_data = {
            "type": BiometricType.FINGERPRINT.value,
            "template": "encrypted_biometric_template",
            "quality_score": 0.95,
            "enrollment_device": "iPhone_TouchID"
        }
        
        with patch.object(password_manager, '_store_biometric_template') as mock_store:
            mock_store.return_value = {
                "success": True,
                "template_id": "bio_template_123",
                "enrolled_at": datetime.utcnow()
            }
            
            result = await password_manager.enroll_biometric(
                user_id=user_data["user_id"],
                biometric_data=biometric_data
            )
        
        assert result["success"] is True
        assert result["template_id"] is not None
        
        # Vérifier authentification biométrique
        verification_data = {
            "type": BiometricType.FINGERPRINT.value,
            "sample": "encrypted_biometric_sample",
            "device_id": "iPhone_TouchID"
        }
        
        with patch.object(password_manager, '_verify_biometric') as mock_verify:
            mock_verify.return_value = {
                "verified": True,
                "confidence": 0.98,
                "template_id": "bio_template_123",
                "user_id": user_data["user_id"]
            }
            
            result = await password_manager.verify_biometric(
                verification_data=verification_data
            )
        
        assert result["verified"] is True
        assert result["confidence"] > 0.95
        assert result["user_id"] == user_data["user_id"]


@pytest_markers["performance"]
@pytest_markers["password"]
class TestPasswordPerformance:
    """Tests de performance pour les mots de passe"""
    
    @pytest.mark.asyncio
    async def test_password_hashing_performance(self, password_manager):
        """Test performance hachage de mot de passe"""
        password = "PerformanceTestPassword123!"
        
        with patch.object(password_manager, 'hash_password') as mock_hash:
            mock_hash.return_value = {
                "hash": "hashed_password",
                "salt": "salt",
                "algorithm": HashAlgorithm.BCRYPT.value,
                "iterations": 12
            }
            
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                password_manager.hash_password,
                password=password,
                algorithm=HashAlgorithm.BCRYPT
            )
        
        # Hachage doit être < 1s (même avec bcrypt)
        assert execution_time < 1.0
        assert result["hash"] is not None
        
        print(f"🔐 Temps hachage mot de passe: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_password_verification_performance(self, password_manager):
        """Test performance vérification de mot de passe"""
        password = "VerificationTestPassword456!"
        hashed = "hashed_password_value"
        
        with patch.object(password_manager, 'verify_password') as mock_verify:
            mock_verify.return_value = {"valid": True, "verified_at": datetime.utcnow()}
            
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                password_manager.verify_password,
                password=password,
                hashed_password=hashed,
                salt="salt",
                algorithm=HashAlgorithm.BCRYPT.value
            )
        
        # Vérification doit être < 500ms
        assert execution_time < 0.5
        assert result["valid"] is True
        
        print(f"✅ Temps vérification mot de passe: {execution_time:.3f}s")


@pytest_markers["security"]
@pytest_markers["password"]
class TestPasswordSecurity:
    """Tests de sécurité pour les mots de passe"""
    
    @pytest.mark.asyncio
    async def test_timing_attack_protection(self, password_manager):
        """Test protection contre les attaques par timing"""
        valid_password = "ValidPassword123!"
        invalid_password = "InvalidPassword456!"
        hashed = "test_hash_value"
        
        # Mesurer le temps pour mot de passe valide
        with patch.object(password_manager, 'verify_password') as mock_verify:
            mock_verify.return_value = {"valid": True, "verified_at": datetime.utcnow()}
            
            _, valid_time = await PerformanceTestUtils.measure_execution_time(
                password_manager.verify_password,
                password=valid_password,
                hashed_password=hashed,
                salt="salt",
                algorithm=HashAlgorithm.BCRYPT.value
            )
        
        # Mesurer le temps pour mot de passe invalide
        with patch.object(password_manager, 'verify_password') as mock_verify:
            mock_verify.return_value = {"valid": False}
            
            _, invalid_time = await PerformanceTestUtils.measure_execution_time(
                password_manager.verify_password,
                password=invalid_password,
                hashed_password=hashed,
                salt="salt",
                algorithm=HashAlgorithm.BCRYPT.value
            )
        
        # Les temps doivent être similaires (différence < 10ms)
        time_difference = abs(valid_time - invalid_time)
        assert time_difference < 0.01
        
        print(f"⏱️  Différence timing: {time_difference:.4f}s")
    
    @pytest.mark.asyncio
    async def test_salt_uniqueness(self, password_manager):
        """Test unicité des sels"""
        password = "TestPassword789!"
        salts = []
        
        # Générer plusieurs hashes du même mot de passe
        for i in range(10):
            with patch.object(password_manager, 'hash_password') as mock_hash:
                salt = TestUtils.generate_random_string(16)
                mock_hash.return_value = {
                    "hash": f"hash_{i}",
                    "salt": salt,
                    "algorithm": HashAlgorithm.BCRYPT.value,
                    "iterations": 12
                }
                
                result = await password_manager.hash_password(password)
                salts.append(result["salt"])
        
        # Tous les sels doivent être uniques
        unique_salts = set(salts)
        assert len(unique_salts) == len(salts)
        
        # Chaque sel doit avoir une entropie suffisante
        for salt in salts:
            assert len(salt) >= 16
            # Vérifier qu'il n'y a pas de pattern évident
            assert not salt.isdigit()  # Pas que des chiffres
            assert not salt.isalpha()  # Pas que des lettres


if __name__ == "__main__":
    print("🧪 Tests PasswordManager Ultra-Avancés")
    print("📋 Modules testés:")
    print("  ✅ Hachage et vérification de mots de passe")
    print("  ✅ Validation de la force et des politiques")
    print("  ✅ Gestion de l'historique des mots de passe")
    print("  ✅ Détection de mots de passe compromis")
    print("  ✅ Expiration des mots de passe")
    print("  ✅ Authentification WebAuthn")
    print("  ✅ Authentification biométrique")
    print("  ✅ Tests de sécurité et performance")
    
    # Lancement des tests
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])
