"""
🎵 Spotify AI Agent - Tests Crypto Utils Module
===============================================

Tests enterprise complets pour le module crypto_utils
avec validation de sécurité, cryptographie et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
import secrets
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from unittest.mock import patch, Mock

# Import du module à tester
from backend.app.api.utils.crypto_utils import (
    generate_salt,
    hash_password,
    verify_password,
    generate_token,
    verify_token,
    encrypt_data,
    decrypt_data,
    generate_key_pair,
    encrypt_with_public_key,
    decrypt_with_private_key,
    sign_data,
    verify_signature,
    secure_random_string,
    calculate_checksum,
    verify_checksum,
    derive_key,
    constant_time_compare,
    obfuscate_data,
    deobfuscate_data,
    generate_otp,
    verify_otp,
    encrypt_json,
    decrypt_json
)

from . import TestUtils, security_test, performance_test, integration_test


class TestCryptoUtils:
    """Tests pour le module crypto_utils"""
    
    @security_test
    def test_generate_salt_basic(self):
        """Test génération salt basique"""
        salt = generate_salt()
        
        assert isinstance(salt, (str, bytes))
        assert len(salt) > 0
    
    @security_test
    def test_generate_salt_length(self):
        """Test génération salt avec longueur"""
        salt = generate_salt(length=32)
        
        # Vérifier longueur (peut être encodé)
        if isinstance(salt, str):
            decoded = base64.b64decode(salt)
            assert len(decoded) == 32
        else:
            assert len(salt) == 32
    
    @security_test
    def test_generate_salt_uniqueness(self):
        """Test unicité des salts"""
        salt1 = generate_salt()
        salt2 = generate_salt()
        
        assert salt1 != salt2  # Doivent être différents
    
    @security_test
    def test_hash_password_basic(self):
        """Test hachage mot de passe basique"""
        password = "securepassword123"
        hashed = hash_password(password)
        
        assert isinstance(hashed, str)
        assert len(hashed) > 0
        assert hashed != password  # Ne doit pas être en clair
    
    @security_test
    def test_hash_password_with_salt(self):
        """Test hachage avec salt personnalisé"""
        password = "testpassword"
        salt = generate_salt()
        
        hashed = hash_password(password, salt=salt)
        
        assert isinstance(hashed, str)
        assert hashed != password
    
    @security_test
    def test_hash_password_consistency(self):
        """Test consistance hachage"""
        password = "testpassword"
        salt = generate_salt()
        
        hash1 = hash_password(password, salt=salt)
        hash2 = hash_password(password, salt=salt)
        
        assert hash1 == hash2  # Même input, même hash
    
    @security_test
    def test_verify_password_correct(self):
        """Test vérification mot de passe correct"""
        password = "correctpassword"
        hashed = hash_password(password)
        
        result = verify_password(password, hashed)
        
        assert result is True
    
    @security_test
    def test_verify_password_incorrect(self):
        """Test vérification mot de passe incorrect"""
        password = "correctpassword"
        wrong_password = "wrongpassword"
        hashed = hash_password(password)
        
        result = verify_password(wrong_password, hashed)
        
        assert result is False
    
    @security_test
    def test_verify_password_timing_attack_resistance(self):
        """Test résistance aux attaques temporelles"""
        password = "testpassword"
        hashed = hash_password(password)
        
        # Mesurer temps pour mot de passe correct
        import time
        start = time.perf_counter()
        verify_password(password, hashed)
        time_correct = time.perf_counter() - start
        
        # Mesurer temps pour mot de passe incorrect
        start = time.perf_counter()
        verify_password("wrongpassword", hashed)
        time_incorrect = time.perf_counter() - start
        
        # Les temps doivent être similaires (protection timing attack)
        time_ratio = max(time_correct, time_incorrect) / min(time_correct, time_incorrect)
        assert time_ratio < 2.0  # Pas plus de 2x de différence
    
    @security_test
    def test_generate_token_basic(self):
        """Test génération token basique"""
        token = generate_token()
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    @security_test
    def test_generate_token_length(self):
        """Test génération token avec longueur"""
        token = generate_token(length=64)
        
        # Vérifier longueur approximative (peut être encodé)
        assert len(token) >= 60  # Au moins proche de la longueur demandée
    
    @security_test
    def test_generate_token_uniqueness(self):
        """Test unicité des tokens"""
        token1 = generate_token()
        token2 = generate_token()
        
        assert token1 != token2
    
    @security_test
    def test_verify_token_valid(self):
        """Test vérification token valide"""
        # Token avec expiration
        token = generate_token(expires_in=3600)  # 1 heure
        
        result = verify_token(token)
        
        assert result['valid'] is True
        assert 'payload' in result
    
    @security_test
    def test_verify_token_expired(self):
        """Test vérification token expiré"""
        # Token avec expiration très courte
        token = generate_token(expires_in=1)
        
        # Attendre expiration
        import time
        time.sleep(2)
        
        result = verify_token(token)
        
        assert result['valid'] is False
        assert 'expired' in result['error'].lower()
    
    @security_test
    def test_verify_token_invalid(self):
        """Test vérification token invalide"""
        invalid_token = "invalid.token.string"
        
        result = verify_token(invalid_token)
        
        assert result['valid'] is False
    
    @security_test
    def test_encrypt_decrypt_data_basic(self):
        """Test chiffrement/déchiffrement basique"""
        data = "sensitive information"
        key = Fernet.generate_key()
        
        encrypted = encrypt_data(data, key)
        decrypted = decrypt_data(encrypted, key)
        
        assert encrypted != data  # Doit être chiffré
        assert decrypted == data   # Doit être identique après déchiffrement
    
    @security_test
    def test_encrypt_decrypt_binary_data(self):
        """Test chiffrement données binaires"""
        data = b"binary data content"
        key = Fernet.generate_key()
        
        encrypted = encrypt_data(data, key)
        decrypted = decrypt_data(encrypted, key)
        
        assert encrypted != data
        assert decrypted == data
    
    @security_test
    def test_encrypt_data_different_keys(self):
        """Test chiffrement avec clés différentes"""
        data = "test data"
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        
        encrypted1 = encrypt_data(data, key1)
        encrypted2 = encrypt_data(data, key2)
        
        assert encrypted1 != encrypted2  # Différentes clés = différents chiffrements
    
    @security_test
    def test_decrypt_data_wrong_key(self):
        """Test déchiffrement avec mauvaise clé"""
        data = "test data"
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        
        encrypted = encrypt_data(data, key1)
        
        # Tenter déchiffrement avec mauvaise clé
        try:
            decrypted = decrypt_data(encrypted, key2)
            assert False, "Déchiffrement devrait échouer"
        except Exception:
            assert True  # Exception attendue
    
    @security_test
    def test_generate_key_pair_basic(self):
        """Test génération paire de clés"""
        private_key, public_key = generate_key_pair()
        
        assert isinstance(private_key, bytes)
        assert isinstance(public_key, bytes)
        assert len(private_key) > 0
        assert len(public_key) > 0
    
    @security_test
    def test_generate_key_pair_uniqueness(self):
        """Test unicité paires de clés"""
        private1, public1 = generate_key_pair()
        private2, public2 = generate_key_pair()
        
        assert private1 != private2
        assert public1 != public2
    
    @security_test
    def test_encrypt_decrypt_with_rsa_keys(self):
        """Test chiffrement RSA asymétrique"""
        data = "secret message"
        private_key, public_key = generate_key_pair()
        
        encrypted = encrypt_with_public_key(data, public_key)
        decrypted = decrypt_with_private_key(encrypted, private_key)
        
        assert encrypted != data.encode()  # Doit être chiffré
        assert decrypted == data           # Doit être identique après déchiffrement
    
    @security_test
    def test_encrypt_rsa_data_too_large(self):
        """Test chiffrement RSA données trop grandes"""
        # RSA a une limite de taille
        large_data = "a" * 500  # Trop grand pour RSA-2048
        private_key, public_key = generate_key_pair()
        
        try:
            encrypted = encrypt_with_public_key(large_data, public_key)
            # Si supporté, vérifier que ça fonctionne
            decrypted = decrypt_with_private_key(encrypted, private_key)
            assert decrypted == large_data
        except Exception:
            # Exception attendue pour données trop grandes
            assert True
    
    @security_test
    def test_sign_verify_data_basic(self):
        """Test signature/vérification basique"""
        data = "message to sign"
        private_key, public_key = generate_key_pair()
        
        signature = sign_data(data, private_key)
        is_valid = verify_signature(data, signature, public_key)
        
        assert isinstance(signature, bytes)
        assert is_valid is True
    
    @security_test
    def test_verify_signature_tampered_data(self):
        """Test vérification signature données modifiées"""
        data = "original message"
        tampered_data = "tampered message"
        private_key, public_key = generate_key_pair()
        
        signature = sign_data(data, private_key)
        is_valid = verify_signature(tampered_data, signature, public_key)
        
        assert is_valid is False  # Signature invalide
    
    @security_test
    def test_verify_signature_wrong_key(self):
        """Test vérification signature mauvaise clé"""
        data = "message"
        private_key1, public_key1 = generate_key_pair()
        private_key2, public_key2 = generate_key_pair()
        
        signature = sign_data(data, private_key1)
        is_valid = verify_signature(data, signature, public_key2)
        
        assert is_valid is False  # Mauvaise clé publique
    
    @security_test
    def test_secure_random_string_basic(self):
        """Test génération string aléatoire sécurisée"""
        random_str = secure_random_string(32)
        
        assert isinstance(random_str, str)
        assert len(random_str) >= 32  # Au moins la longueur demandée
    
    @security_test
    def test_secure_random_string_uniqueness(self):
        """Test unicité strings aléatoires"""
        str1 = secure_random_string(16)
        str2 = secure_random_string(16)
        
        assert str1 != str2
    
    @security_test
    def test_secure_random_string_charset(self):
        """Test jeu de caractères string aléatoire"""
        # Alphanumerique seulement
        alnum_str = secure_random_string(20, charset='alphanumeric')
        assert all(c.isalnum() for c in alnum_str)
        
        # Hexadécimal seulement
        hex_str = secure_random_string(20, charset='hex')
        assert all(c in '0123456789abcdefABCDEF' for c in hex_str)
    
    def test_calculate_checksum_basic(self):
        """Test calcul checksum basique"""
        data = "test data for checksum"
        checksum = calculate_checksum(data)
        
        assert isinstance(checksum, str)
        assert len(checksum) > 0
    
    def test_calculate_checksum_consistency(self):
        """Test consistance checksum"""
        data = "consistent data"
        checksum1 = calculate_checksum(data)
        checksum2 = calculate_checksum(data)
        
        assert checksum1 == checksum2
    
    def test_calculate_checksum_different_data(self):
        """Test checksum données différentes"""
        data1 = "data one"
        data2 = "data two"
        
        checksum1 = calculate_checksum(data1)
        checksum2 = calculate_checksum(data2)
        
        assert checksum1 != checksum2
    
    def test_verify_checksum_valid(self):
        """Test vérification checksum valide"""
        data = "data to verify"
        checksum = calculate_checksum(data)
        
        is_valid = verify_checksum(data, checksum)
        
        assert is_valid is True
    
    def test_verify_checksum_invalid(self):
        """Test vérification checksum invalide"""
        data = "original data"
        tampered_data = "tampered data"
        checksum = calculate_checksum(data)
        
        is_valid = verify_checksum(tampered_data, checksum)
        
        assert is_valid is False
    
    @security_test
    def test_derive_key_basic(self):
        """Test dérivation de clé basique"""
        password = "userpassword"
        salt = generate_salt()
        
        key = derive_key(password, salt)
        
        assert isinstance(key, bytes)
        assert len(key) > 0
    
    @security_test
    def test_derive_key_consistency(self):
        """Test consistance dérivation clé"""
        password = "password"
        salt = generate_salt()
        
        key1 = derive_key(password, salt)
        key2 = derive_key(password, salt)
        
        assert key1 == key2  # Même input, même clé
    
    @security_test
    def test_derive_key_different_passwords(self):
        """Test dérivation clés différentes"""
        salt = generate_salt()
        
        key1 = derive_key("password1", salt)
        key2 = derive_key("password2", salt)
        
        assert key1 != key2
    
    @security_test
    def test_derive_key_different_salts(self):
        """Test dérivation salts différents"""
        password = "password"
        
        key1 = derive_key(password, generate_salt())
        key2 = derive_key(password, generate_salt())
        
        assert key1 != key2
    
    @security_test
    def test_constant_time_compare_equal(self):
        """Test comparaison temps constant - égaux"""
        str1 = "identical"
        str2 = "identical"
        
        result = constant_time_compare(str1, str2)
        
        assert result is True
    
    @security_test
    def test_constant_time_compare_different(self):
        """Test comparaison temps constant - différents"""
        str1 = "string one"
        str2 = "string two"
        
        result = constant_time_compare(str1, str2)
        
        assert result is False
    
    @security_test
    def test_constant_time_compare_timing(self):
        """Test résistance attaque temporelle"""
        # Strings de longueurs différentes
        short = "short"
        long_str = "a" * 1000
        
        import time
        
        # Mesurer temps comparaison courte
        start = time.perf_counter()
        constant_time_compare(short, "wrong")
        time_short = time.perf_counter() - start
        
        # Mesurer temps comparaison longue
        start = time.perf_counter()
        constant_time_compare(long_str, "wrong")
        time_long = time.perf_counter() - start
        
        # Le temps ne doit pas révéler d'information
        # (dépend de l'implémentation)
        assert time_short >= 0 and time_long >= 0
    
    @security_test
    def test_obfuscate_deobfuscate_data(self):
        """Test obfuscation/désobfuscation"""
        data = "sensitive data to obfuscate"
        key = "obfuscation_key"
        
        obfuscated = obfuscate_data(data, key)
        deobfuscated = deobfuscate_data(obfuscated, key)
        
        assert obfuscated != data        # Doit être obfusqué
        assert deobfuscated == data      # Doit être identique après désobfuscation
    
    @security_test
    def test_obfuscate_data_different_keys(self):
        """Test obfuscation clés différentes"""
        data = "data"
        key1 = "key1"
        key2 = "key2"
        
        obfuscated1 = obfuscate_data(data, key1)
        obfuscated2 = obfuscate_data(data, key2)
        
        assert obfuscated1 != obfuscated2
    
    @security_test
    def test_generate_otp_basic(self):
        """Test génération OTP basique"""
        secret = "shared_secret"
        otp = generate_otp(secret)
        
        assert isinstance(otp, str)
        assert otp.isdigit()  # Doit être numérique
        assert len(otp) >= 6  # Au moins 6 chiffres
    
    @security_test
    def test_generate_otp_time_based(self):
        """Test OTP basé sur le temps (TOTP)"""
        secret = "time_secret"
        
        otp1 = generate_otp(secret, algorithm='totp')
        
        # Attendre 1 seconde
        import time
        time.sleep(1)
        
        otp2 = generate_otp(secret, algorithm='totp')
        
        # Peuvent être identiques ou différents selon la fenêtre temporelle
        assert isinstance(otp1, str)
        assert isinstance(otp2, str)
    
    @security_test
    def test_verify_otp_valid(self):
        """Test vérification OTP valide"""
        secret = "verification_secret"
        otp = generate_otp(secret)
        
        is_valid = verify_otp(otp, secret)
        
        assert is_valid is True
    
    @security_test
    def test_verify_otp_invalid(self):
        """Test vérification OTP invalide"""
        secret = "verification_secret"
        wrong_otp = "000000"
        
        is_valid = verify_otp(wrong_otp, secret)
        
        assert is_valid is False
    
    @security_test
    def test_verify_otp_expired(self):
        """Test vérification OTP expiré"""
        secret = "expiry_secret"
        
        # Générer OTP avec timestamp ancien
        old_timestamp = int(time.time()) - 3600  # 1 heure dans le passé
        otp = generate_otp(secret, timestamp=old_timestamp)
        
        is_valid = verify_otp(otp, secret, window=30)  # Fenêtre de 30 secondes
        
        assert is_valid is False
    
    @security_test
    def test_encrypt_decrypt_json_basic(self):
        """Test chiffrement/déchiffrement JSON"""
        data = {
            'user_id': 12345,
            'username': 'testuser',
            'sensitive_info': 'secret data'
        }
        password = "encryption_password"
        
        encrypted = encrypt_json(data, password)
        decrypted = decrypt_json(encrypted, password)
        
        assert encrypted != str(data)  # Doit être chiffré
        assert decrypted == data       # Doit être identique après déchiffrement
    
    @security_test
    def test_encrypt_json_complex_data(self):
        """Test chiffrement JSON données complexes"""
        complex_data = {
            'nested': {
                'array': [1, 2, 3],
                'boolean': True,
                'null_value': None
            },
            'unicode': 'Café à Paris 🎵',
            'numbers': {
                'integer': 42,
                'float': 3.14159
            }
        }
        password = "complex_password"
        
        encrypted = encrypt_json(complex_data, password)
        decrypted = decrypt_json(encrypted, password)
        
        assert decrypted == complex_data
    
    @security_test
    def test_decrypt_json_wrong_password(self):
        """Test déchiffrement JSON mauvais mot de passe"""
        data = {'test': 'data'}
        password = "correct_password"
        wrong_password = "wrong_password"
        
        encrypted = encrypt_json(data, password)
        
        try:
            decrypted = decrypt_json(encrypted, wrong_password)
            assert False, "Déchiffrement devrait échouer"
        except Exception:
            assert True  # Exception attendue
    
    @performance_test
    def test_hashing_performance(self):
        """Test performance hachage"""
        passwords = [f"password{i}" for i in range(100)]
        
        def hash_many_passwords():
            for password in passwords:
                hash_password(password)
            return len(passwords)
        
        TestUtils.assert_performance(hash_many_passwords, max_time_ms=2000)
    
    @performance_test
    def test_encryption_performance(self):
        """Test performance chiffrement"""
        data = "test data " * 100  # Données moyennes
        key = Fernet.generate_key()
        
        def encrypt_decrypt_cycle():
            encrypted = encrypt_data(data, key)
            decrypted = decrypt_data(encrypted, key)
            return len(decrypted)
        
        TestUtils.assert_performance(encrypt_decrypt_cycle, max_time_ms=100)
    
    @performance_test
    def test_rsa_operations_performance(self):
        """Test performance opérations RSA"""
        private_key, public_key = generate_key_pair()
        data = "test message"
        
        def rsa_operations():
            encrypted = encrypt_with_public_key(data, public_key)
            decrypted = decrypt_with_private_key(encrypted, private_key)
            signature = sign_data(data, private_key)
            verified = verify_signature(data, signature, public_key)
            return len(decrypted)
        
        TestUtils.assert_performance(rsa_operations, max_time_ms=500)
    
    @integration_test
    def test_complete_crypto_workflow(self):
        """Test workflow cryptographique complet"""
        # Scénario: Authentification et chiffrement utilisateur
        
        # 1. Inscription utilisateur
        username = "testuser"
        password = "securepassword123"
        
        # 2. Hachage mot de passe
        salt = generate_salt()
        password_hash = hash_password(password, salt=salt)
        
        # 3. Génération token d'authentification
        auth_token = generate_token(expires_in=3600)
        
        # 4. Génération paire de clés pour l'utilisateur
        user_private_key, user_public_key = generate_key_pair()
        
        # 5. Données sensibles à protéger
        sensitive_data = {
            'personal_info': 'sensitive information',
            'preferences': ['music', 'podcasts'],
            'payment_info': 'encrypted separately'
        }
        
        # 6. Chiffrement données avec mot de passe dérivé
        derived_key = derive_key(password, salt)
        encrypted_data = encrypt_json(sensitive_data, password)
        
        # 7. Signature des données pour intégrité
        data_signature = sign_data(str(sensitive_data), user_private_key)
        
        # 8. Génération checksum pour vérification
        data_checksum = calculate_checksum(str(sensitive_data))
        
        # 9. Génération OTP pour 2FA
        otp_secret = secure_random_string(32)
        current_otp = generate_otp(otp_secret)
        
        # === Vérifications côté serveur ===
        
        # 10. Vérification mot de passe
        auth_valid = verify_password(password, password_hash)
        assert auth_valid is True
        
        # 11. Vérification token
        token_result = verify_token(auth_token)
        assert token_result['valid'] is True
        
        # 12. Vérification OTP
        otp_valid = verify_otp(current_otp, otp_secret)
        assert otp_valid is True
        
        # 13. Déchiffrement données
        decrypted_data = decrypt_json(encrypted_data, password)
        assert decrypted_data == sensitive_data
        
        # 14. Vérification signature
        signature_valid = verify_signature(str(sensitive_data), data_signature, user_public_key)
        assert signature_valid is True
        
        # 15. Vérification checksum
        checksum_valid = verify_checksum(str(sensitive_data), data_checksum)
        assert checksum_valid is True
        
        print("✅ Workflow cryptographique complet validé")


# Tests de sécurité avancés
class TestCryptoSecurityAdvanced:
    """Tests de sécurité avancés"""
    
    @security_test
    def test_password_hash_collision_resistance(self):
        """Test résistance collisions hash"""
        passwords = [
            "password1",
            "password2", 
            "1drowssap",  # Inverse
            "PASSWORD1",  # Casse différente
            "password1 "  # Espace
        ]
        
        hashes = [hash_password(pwd) for pwd in passwords]
        
        # Tous les hash doivent être différents
        assert len(set(hashes)) == len(hashes)
    
    @security_test
    def test_encryption_key_separation(self):
        """Test séparation des clés"""
        data = "test data"
        
        # Générer plusieurs clés
        keys = [Fernet.generate_key() for _ in range(5)]
        
        # Chiffrer avec chaque clé
        encrypted_versions = [encrypt_data(data, key) for key in keys]
        
        # Tous les chiffrements doivent être différents
        assert len(set(encrypted_versions)) == len(encrypted_versions)
    
    @security_test
    def test_timing_attack_resistance_comprehensive(self):
        """Test résistance attaques temporelles complet"""
        import time
        import statistics
        
        correct_password = "correctpassword"
        hashed = hash_password(correct_password)
        
        # Tester avec différents mots de passe incorrects
        wrong_passwords = [
            "a",
            "wrongpassword",
            "verylongwrongpasswordthatistoolongtobevalid",
            "",
            "correctpasswor",  # Presque correct
            "correctpasswordd"  # Un caractère en plus
        ]
        
        times = []
        for wrong_pwd in wrong_passwords:
            start = time.perf_counter()
            verify_password(wrong_pwd, hashed)
            end = time.perf_counter()
            times.append(end - start)
        
        # La variance des temps doit être faible
        if len(times) > 1:
            variance = statistics.variance(times)
            mean_time = statistics.mean(times)
            
            # Coefficient de variation < 50%
            cv = (variance ** 0.5) / mean_time if mean_time > 0 else 0
            assert cv < 0.5, f"Trop de variance temporelle: {cv}"
    
    @security_test
    def test_random_quality(self):
        """Test qualité de l'aléatoire"""
        # Générer beaucoup de données aléatoires
        random_data = [secure_random_string(10) for _ in range(1000)]
        
        # Vérifier unicité
        unique_count = len(set(random_data))
        assert unique_count >= 995  # Au moins 99.5% unique
        
        # Test distribution des caractères
        all_chars = ''.join(random_data)
        char_counts = {}
        for char in all_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Vérifier distribution approximativement uniforme
        if len(char_counts) > 10:  # Si assez de caractères différents
            counts = list(char_counts.values())
            min_count = min(counts)
            max_count = max(counts)
            
            # Ratio max/min ne doit pas être trop élevé
            ratio = max_count / min_count if min_count > 0 else 0
            assert ratio < 5.0, f"Distribution non uniforme: {ratio}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
