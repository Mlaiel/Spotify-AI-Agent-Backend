# 🧪 ML Analytics Utils Tests
# ===========================
# 
# Tests ultra-avancés pour les utilitaires ML Analytics
# Enterprise utilities testing
#
# 🎖️ Implementation par l'équipe d'experts:
# ✅ Lead Dev + Développeur Backend Senior + ML Engineer
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ===========================

"""
🔧 Utilities Test Suite
=======================

Comprehensive testing for utility functions:
- Data processing utilities
- Audio processing helpers
- Caching and optimization utilities
- Performance monitoring helpers
- Configuration utilities
- Security and validation helpers
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import hashlib
import base64

# Import modules to test
from app.ml_analytics.utils import (
    DataProcessor, AudioProcessor, CacheManager,
    PerformanceProfiler, ConfigHelper, SecurityUtils,
    ValidationUtils, DataConverter, FeatureExtractor,
    ModelUtils, FileUtils, NetworkUtils,
    process_audio_batch, extract_audio_features,
    cache_result, profile_performance, validate_data,
    convert_format, secure_hash, encrypt_data, decrypt_data,
    sanitize_input, normalize_features, calculate_similarity,
    batch_process, retry_with_backoff, measure_execution_time
)


class TestDataProcessor:
    """Tests pour le processeur de données"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.data_processor = DataProcessor()
    
    def test_data_processor_creation(self):
        """Test de création du processeur"""
        assert isinstance(self.data_processor, DataProcessor)
        assert self.data_processor.batch_size == 1000
        assert self.data_processor.max_workers == 4
    
    def test_process_dataframe(self):
        """Test de traitement de DataFrame"""
        # Données de test
        df = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3'],
            'popularity': [80, 65, 90],
            'duration_ms': [180000, 240000, 200000],
            'genre': ['rock', 'pop', 'jazz']
        })
        
        processed_df = self.data_processor.process_dataframe(
            df,
            operations=[
                {'type': 'normalize', 'columns': ['popularity']},
                {'type': 'scale', 'columns': ['duration_ms'], 'factor': 0.001},
                {'type': 'encode', 'columns': ['genre'], 'method': 'onehot'}
            ]
        )
        
        # Vérifications
        assert 'popularity_normalized' in processed_df.columns
        assert 'duration_ms_scaled' in processed_df.columns
        assert 'genre_rock' in processed_df.columns
        assert 'genre_pop' in processed_df.columns
        assert 'genre_jazz' in processed_df.columns
    
    def test_clean_data(self):
        """Test de nettoyage de données"""
        # Données avec problèmes
        dirty_df = pd.DataFrame({
            'track_id': ['track1', 'track2', None, 'track4'],
            'popularity': [80, None, 90, 65],
            'duration_ms': [180000, 240000, -1000, 200000],  # Valeur négative
            'artist': ['Artist A', 'Artist B', 'Artist C', '']  # Valeur vide
        })
        
        clean_df = self.data_processor.clean_data(
            dirty_df,
            strategies={
                'missing_values': 'drop',
                'negative_values': 'clip',
                'empty_strings': 'drop'
            }
        )
        
        # Vérifications
        assert clean_df.isnull().sum().sum() == 0  # Pas de valeurs manquantes
        assert (clean_df['duration_ms'] >= 0).all()  # Pas de valeurs négatives
        assert (clean_df['artist'] != '').all()  # Pas de chaînes vides
    
    def test_feature_engineering(self):
        """Test d'ingénierie des caractéristiques"""
        df = pd.DataFrame({
            'duration_ms': [180000, 240000, 200000],
            'popularity': [80, 65, 90],
            'release_date': ['2020-01-15', '2019-06-20', '2021-03-10']
        })
        
        engineered_df = self.data_processor.feature_engineering(
            df,
            features=[
                {'name': 'duration_minutes', 'expression': 'duration_ms / 60000'},
                {'name': 'popularity_tier', 'expression': 'popularity // 20'},
                {'name': 'release_year', 'expression': 'release_date.dt.year'}
            ]
        )
        
        assert 'duration_minutes' in engineered_df.columns
        assert 'popularity_tier' in engineered_df.columns
        assert 'release_year' in engineered_df.columns
        assert engineered_df['duration_minutes'].iloc[0] == 3.0
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test de traitement par lots asynchrone"""
        # Grande dataset
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.rand(10000)
        })
        
        async def process_batch(batch_df):
            # Simuler un traitement
            await asyncio.sleep(0.01)
            return batch_df['value'].sum()
        
        results = await self.data_processor.process_in_batches_async(
            large_df,
            process_func=process_batch,
            batch_size=1000
        )
        
        assert len(results) == 10  # 10 batches
        assert all(isinstance(r, float) for r in results)
    
    def test_data_validation(self):
        """Test de validation de données"""
        df = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3'],
            'popularity': [80, 65, 150],  # Valeur invalide
            'duration_ms': [180000, -5000, 200000]  # Valeur invalide
        })
        
        validation_rules = {
            'popularity': {'min': 0, 'max': 100, 'type': 'numeric'},
            'duration_ms': {'min': 1000, 'max': 600000, 'type': 'numeric'},
            'track_id': {'required': True, 'unique': True, 'type': 'string'}
        }
        
        validation_result = self.data_processor.validate_data(df, validation_rules)
        
        assert validation_result['is_valid'] is False
        assert validation_result['error_count'] == 2
        assert 'popularity' in validation_result['field_errors']
        assert 'duration_ms' in validation_result['field_errors']


class TestAudioProcessor:
    """Tests pour le processeur audio"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.audio_processor = AudioProcessor()
    
    def test_audio_processor_creation(self):
        """Test de création du processeur audio"""
        assert isinstance(self.audio_processor, AudioProcessor)
        assert self.audio_processor.sample_rate == 22050
        assert self.audio_processor.n_mfcc == 13
    
    def test_generate_test_audio(self):
        """Test de génération d'audio de test"""
        # Générer un signal audio de test
        audio_signal = self.audio_processor.generate_test_audio(
            frequency=440,  # A4
            duration=2.0,
            sample_rate=22050
        )
        
        expected_length = int(2.0 * 22050)
        assert len(audio_signal) == expected_length
        assert audio_signal.dtype == np.float32
        assert np.abs(audio_signal).max() <= 1.0  # Amplitude normalisée
    
    def test_extract_mfcc_features(self):
        """Test d'extraction des caractéristiques MFCC"""
        # Audio de test
        audio_signal = self.audio_processor.generate_test_audio(
            frequency=440,
            duration=1.0,
            sample_rate=22050
        )
        
        mfcc_features = self.audio_processor.extract_mfcc_features(
            audio_signal,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        assert mfcc_features.shape[0] == 13  # n_mfcc
        assert mfcc_features.shape[1] > 0    # Frames temporels
    
    def test_extract_spectral_features(self):
        """Test d'extraction des caractéristiques spectrales"""
        audio_signal = self.audio_processor.generate_test_audio(
            frequency=880,  # A5
            duration=1.0
        )
        
        spectral_features = self.audio_processor.extract_spectral_features(
            audio_signal
        )
        
        expected_features = [
            'spectral_centroid', 'spectral_bandwidth',
            'spectral_rolloff', 'zero_crossing_rate',
            'spectral_flatness'
        ]
        
        for feature in expected_features:
            assert feature in spectral_features
            assert isinstance(spectral_features[feature], (float, np.floating))
    
    def test_extract_tempo_features(self):
        """Test d'extraction des caractéristiques de tempo"""
        # Générer un audio avec rythme
        audio_signal = self.audio_processor.generate_rhythmic_audio(
            bpm=120,
            duration=4.0
        )
        
        tempo_features = self.audio_processor.extract_tempo_features(
            audio_signal
        )
        
        assert 'tempo' in tempo_features
        assert 'beat_frames' in tempo_features
        assert isinstance(tempo_features['tempo'], float)
        assert tempo_features['tempo'] > 0
    
    def test_audio_preprocessing(self):
        """Test de préprocessing audio"""
        # Audio avec bruit
        noisy_audio = np.random.normal(0, 0.1, 22050)  # 1 seconde de bruit
        
        preprocessed_audio = self.audio_processor.preprocess_audio(
            noisy_audio,
            operations=[
                {'type': 'normalize'},
                {'type': 'trim_silence', 'threshold': 0.01},
                {'type': 'apply_window', 'window_type': 'hann'}
            ]
        )
        
        assert len(preprocessed_audio) <= len(noisy_audio)
        assert np.abs(preprocessed_audio).max() <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_audio_processing(self):
        """Test de traitement audio par lots"""
        # Générer plusieurs signaux audio
        audio_signals = [
            self.audio_processor.generate_test_audio(freq, 1.0)
            for freq in [220, 440, 880]  # A3, A4, A5
        ]
        
        results = await self.audio_processor.process_audio_batch_async(
            audio_signals,
            extract_features=True
        )
        
        assert len(results) == 3
        for result in results:
            assert 'mfcc' in result
            assert 'spectral_features' in result
            assert result['mfcc'].shape[0] == 13


class TestCacheManager:
    """Tests pour le gestionnaire de cache"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.cache_manager = CacheManager()
    
    def test_cache_manager_creation(self):
        """Test de création du gestionnaire de cache"""
        assert isinstance(self.cache_manager, CacheManager)
        assert self.cache_manager.default_ttl == 3600
    
    def test_set_and_get_cache(self):
        """Test de définition et récupération de cache"""
        key = "test_key"
        value = {"data": "test_value", "timestamp": time.time()}
        
        # Définir la valeur en cache
        success = self.cache_manager.set(key, value, ttl=60)
        assert success is True
        
        # Récupérer la valeur
        cached_value = self.cache_manager.get(key)
        assert cached_value == value
    
    def test_cache_expiration(self):
        """Test d'expiration du cache"""
        key = "expiring_key"
        value = "expiring_value"
        
        # Définir avec TTL très court
        self.cache_manager.set(key, value, ttl=0.1)
        
        # Vérifier que la valeur existe immédiatement
        assert self.cache_manager.get(key) == value
        
        # Attendre l'expiration
        time.sleep(0.2)
        
        # La valeur devrait avoir expiré
        assert self.cache_manager.get(key) is None
    
    def test_cache_decorator(self):
        """Test du décorateur de cache"""
        call_count = 0
        
        @self.cache_manager.cached(ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # Premier appel
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Deuxième appel (devrait utiliser le cache)
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Pas d'appel supplémentaire
        
        # Appel avec paramètres différents
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2  # Nouvel appel
    
    @pytest.mark.asyncio
    async def test_async_cache_decorator(self):
        """Test du décorateur de cache asynchrone"""
        call_count = 0
        
        @self.cache_manager.cached_async(ttl=60)
        async def async_expensive_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simuler une opération asynchrone
            return x * x
        
        # Premier appel
        result1 = await async_expensive_function(5)
        assert result1 == 25
        assert call_count == 1
        
        # Deuxième appel (cache)
        result2 = await async_expensive_function(5)
        assert result2 == 25
        assert call_count == 1
    
    def test_cache_invalidation(self):
        """Test d'invalidation du cache"""
        keys = ["key1", "key2", "key3"]
        values = ["value1", "value2", "value3"]
        
        # Mettre des valeurs en cache
        for key, value in zip(keys, values):
            self.cache_manager.set(key, value)
        
        # Vérifier qu'elles existent
        for key, value in zip(keys, values):
            assert self.cache_manager.get(key) == value
        
        # Invalider une clé spécifique
        self.cache_manager.invalidate("key2")
        assert self.cache_manager.get("key2") is None
        assert self.cache_manager.get("key1") == "value1"
        
        # Invalider toutes les clés avec un pattern
        self.cache_manager.invalidate_pattern("key*")
        for key in keys:
            assert self.cache_manager.get(key) is None
    
    def test_cache_statistics(self):
        """Test des statistiques de cache"""
        # Générer quelques hits et misses
        self.cache_manager.set("hit_key", "value")
        
        # Hits
        self.cache_manager.get("hit_key")  # Hit
        self.cache_manager.get("hit_key")  # Hit
        
        # Misses
        self.cache_manager.get("miss_key1")  # Miss
        self.cache_manager.get("miss_key2")  # Miss
        
        stats = self.cache_manager.get_statistics()
        
        assert stats['hits'] >= 2
        assert stats['misses'] >= 2
        assert stats['hit_rate'] > 0.0


class TestPerformanceProfiler:
    """Tests pour le profileur de performance"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.profiler = PerformanceProfiler()
    
    def test_profiler_creation(self):
        """Test de création du profileur"""
        assert isinstance(self.profiler, PerformanceProfiler)
        assert len(self.profiler.measurements) == 0
    
    def test_time_measurement(self):
        """Test de mesure de temps"""
        with self.profiler.time_measurement("test_operation"):
            time.sleep(0.1)  # Simuler une opération
        
        measurements = self.profiler.get_measurements("test_operation")
        assert len(measurements) == 1
        assert measurements[0]['duration'] >= 0.1
    
    def test_memory_measurement(self):
        """Test de mesure de mémoire"""
        # Allouer de la mémoire
        with self.profiler.memory_measurement("memory_test"):
            large_list = [i for i in range(100000)]  # Allocation mémoire
            del large_list
        
        measurements = self.profiler.get_measurements("memory_test")
        assert len(measurements) == 1
        assert 'memory_before' in measurements[0]
        assert 'memory_after' in measurements[0]
        assert 'memory_delta' in measurements[0]
    
    def test_cpu_measurement(self):
        """Test de mesure CPU"""
        with self.profiler.cpu_measurement("cpu_test"):
            # Opération intensive CPU
            sum(i*i for i in range(100000))
        
        measurements = self.profiler.get_measurements("cpu_test")
        assert len(measurements) == 1
        assert 'cpu_percent' in measurements[0]
        assert measurements[0]['cpu_percent'] >= 0
    
    def test_performance_decorator(self):
        """Test du décorateur de performance"""
        @self.profiler.profile_performance
        def test_function(n):
            return sum(range(n))
        
        result = test_function(1000)
        assert result == sum(range(1000))
        
        measurements = self.profiler.get_measurements("test_function")
        assert len(measurements) == 1
        assert 'duration' in measurements[0]
    
    @pytest.mark.asyncio
    async def test_async_performance_decorator(self):
        """Test du décorateur de performance asynchrone"""
        @self.profiler.profile_performance_async
        async def async_test_function(delay):
            await asyncio.sleep(delay)
            return "completed"
        
        result = await async_test_function(0.1)
        assert result == "completed"
        
        measurements = self.profiler.get_measurements("async_test_function")
        assert len(measurements) == 1
        assert measurements[0]['duration'] >= 0.1
    
    def test_performance_report(self):
        """Test de génération de rapport de performance"""
        # Effectuer plusieurs mesures
        operations = ["op1", "op2", "op3"]
        
        for op in operations:
            with self.profiler.time_measurement(op):
                time.sleep(0.05)
        
        report = self.profiler.generate_performance_report()
        
        assert 'summary' in report
        assert 'operations' in report
        assert len(report['operations']) == 3
        
        for op in operations:
            assert op in report['operations']


class TestConfigHelper:
    """Tests pour l'assistant de configuration"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.config_helper = ConfigHelper()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_helper_creation(self):
        """Test de création de l'assistant"""
        assert isinstance(self.config_helper, ConfigHelper)
    
    def test_load_config_from_dict(self):
        """Test de chargement de configuration depuis dictionnaire"""
        config_dict = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "spotify_db"
            },
            "cache": {
                "type": "redis",
                "ttl": 3600
            }
        }
        
        config = self.config_helper.load_from_dict(config_dict)
        
        assert config.get("database.host") == "localhost"
        assert config.get("database.port") == 5432
        assert config.get("cache.type") == "redis"
    
    def test_load_config_from_file(self):
        """Test de chargement depuis fichier"""
        config_data = {
            "ml_models": {
                "recommendation": {
                    "type": "collaborative_filtering",
                    "parameters": {"n_factors": 50}
                }
            }
        }
        
        config_file = Path(self.temp_dir) / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = self.config_helper.load_from_file(str(config_file))
        
        assert config.get("ml_models.recommendation.type") == "collaborative_filtering"
        assert config.get("ml_models.recommendation.parameters.n_factors") == 50
    
    def test_config_validation(self):
        """Test de validation de configuration"""
        config_dict = {
            "database": {
                "host": "localhost",
                "port": 5432
            },
            "cache": {
                "ttl": 3600
            }
        }
        
        schema = {
            "database": {
                "host": {"type": "string", "required": True},
                "port": {"type": "integer", "required": True, "min": 1, "max": 65535}
            },
            "cache": {
                "ttl": {"type": "integer", "required": True, "min": 1}
            }
        }
        
        config = self.config_helper.load_from_dict(config_dict)
        is_valid, errors = self.config_helper.validate(config, schema)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_config_merging(self):
        """Test de fusion de configurations"""
        base_config = {
            "database": {"host": "localhost", "port": 5432},
            "debug": True
        }
        
        override_config = {
            "database": {"port": 5433, "name": "new_db"},
            "cache": {"ttl": 3600}
        }
        
        merged = self.config_helper.merge_configs(base_config, override_config)
        
        assert merged["database"]["host"] == "localhost"  # Conservé
        assert merged["database"]["port"] == 5433        # Remplacé
        assert merged["database"]["name"] == "new_db"    # Ajouté
        assert merged["debug"] is True                    # Conservé
        assert merged["cache"]["ttl"] == 3600            # Ajouté
    
    def test_environment_variable_substitution(self):
        """Test de substitution de variables d'environnement"""
        import os
        
        config_dict = {
            "database": {
                "host": "${DB_HOST:localhost}",
                "port": "${DB_PORT:5432}",
                "password": "${DB_PASSWORD}"
            }
        }
        
        # Définir des variables d'environnement
        os.environ["DB_HOST"] = "production-db"
        os.environ["DB_PASSWORD"] = "secret123"
        
        try:
            config = self.config_helper.load_from_dict(config_dict)
            resolved = self.config_helper.resolve_environment_variables(config)
            
            assert resolved.get("database.host") == "production-db"
            assert resolved.get("database.port") == "5432"  # Valeur par défaut
            assert resolved.get("database.password") == "secret123"
            
        finally:
            # Nettoyer les variables d'environnement
            os.environ.pop("DB_HOST", None)
            os.environ.pop("DB_PASSWORD", None)


class TestSecurityUtils:
    """Tests pour les utilitaires de sécurité"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.security_utils = SecurityUtils()
    
    def test_security_utils_creation(self):
        """Test de création des utilitaires"""
        assert isinstance(self.security_utils, SecurityUtils)
    
    def test_password_hashing(self):
        """Test de hachage de mot de passe"""
        password = "test_password_123"
        
        # Hacher le mot de passe
        hashed = self.security_utils.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        
        # Vérifier le mot de passe
        assert self.security_utils.verify_password(password, hashed) is True
        assert self.security_utils.verify_password("wrong_password", hashed) is False
    
    def test_data_encryption(self):
        """Test de chiffrement de données"""
        data = "sensitive_information_123"
        key = self.security_utils.generate_encryption_key()
        
        # Chiffrer les données
        encrypted = self.security_utils.encrypt_data(data, key)
        assert encrypted != data
        assert isinstance(encrypted, str)
        
        # Déchiffrer les données
        decrypted = self.security_utils.decrypt_data(encrypted, key)
        assert decrypted == data
    
    def test_secure_token_generation(self):
        """Test de génération de token sécurisé"""
        token1 = self.security_utils.generate_secure_token()
        token2 = self.security_utils.generate_secure_token()
        
        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2  # Tokens différents
        
        # Token avec longueur spécifique
        token_32 = self.security_utils.generate_secure_token(length=32)
        assert len(token_32) == 32
    
    def test_input_sanitization(self):
        """Test de désinfection d'entrée"""
        # Entrée malveillante
        malicious_input = "<script>alert('xss')</script>"
        
        sanitized = self.security_utils.sanitize_input(malicious_input)
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
    
    def test_sql_injection_prevention(self):
        """Test de prévention d'injection SQL"""
        # Requête avec tentative d'injection
        malicious_query = "SELECT * FROM users WHERE id = '1; DROP TABLE users; --'"
        
        sanitized_query = self.security_utils.sanitize_sql_input(malicious_query)
        
        assert "DROP TABLE" not in sanitized_query
        assert "--" not in sanitized_query
    
    def test_rate_limiting(self):
        """Test de limitation de taux"""
        identifier = "test_user"
        limit = 5
        window = 60  # 60 secondes
        
        # Effectuer des requêtes jusqu'à la limite
        for i in range(limit):
            allowed = self.security_utils.check_rate_limit(identifier, limit, window)
            assert allowed is True
        
        # La prochaine requête devrait être limitée
        limited = self.security_utils.check_rate_limit(identifier, limit, window)
        assert limited is False
    
    def test_jwt_token_operations(self):
        """Test des opérations de token JWT"""
        payload = {
            "user_id": 123,
            "username": "testuser",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        secret_key = "test_secret_key"
        
        # Générer un token
        token = self.security_utils.generate_jwt_token(payload, secret_key)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Valider le token
        decoded = self.security_utils.validate_jwt_token(token, secret_key)
        assert decoded["user_id"] == 123
        assert decoded["username"] == "testuser"
    
    def test_secure_file_operations(self):
        """Test d'opérations de fichier sécurisées"""
        content = "sensitive file content"
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Écriture sécurisée
            self.security_utils.write_secure_file(temp_path, content)
            
            # Lecture sécurisée
            read_content = self.security_utils.read_secure_file(temp_path)
            assert read_content == content
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestValidationUtils:
    """Tests pour les utilitaires de validation"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.validation_utils = ValidationUtils()
    
    def test_validation_utils_creation(self):
        """Test de création des utilitaires"""
        assert isinstance(self.validation_utils, ValidationUtils)
    
    def test_email_validation(self):
        """Test de validation d'email"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "test+label@gmail.com"
        ]
        
        invalid_emails = [
            "invalid_email",
            "@domain.com",
            "test@",
            "test..test@domain.com"
        ]
        
        for email in valid_emails:
            assert self.validation_utils.validate_email(email) is True
        
        for email in invalid_emails:
            assert self.validation_utils.validate_email(email) is False
    
    def test_url_validation(self):
        """Test de validation d'URL"""
        valid_urls = [
            "https://www.example.com",
            "http://api.spotify.com/v1/tracks",
            "ftp://files.example.com/data.csv"
        ]
        
        invalid_urls = [
            "not_a_url",
            "htp://invalid-protocol.com",
            "www.missing-protocol.com"
        ]
        
        for url in valid_urls:
            assert self.validation_utils.validate_url(url) is True
        
        for url in invalid_urls:
            assert self.validation_utils.validate_url(url) is False
    
    def test_data_type_validation(self):
        """Test de validation de type de données"""
        # Validation d'entier
        assert self.validation_utils.validate_integer("123") is True
        assert self.validation_utils.validate_integer("12.34") is False
        assert self.validation_utils.validate_integer("not_a_number") is False
        
        # Validation de flottant
        assert self.validation_utils.validate_float("12.34") is True
        assert self.validation_utils.validate_float("123") is True
        assert self.validation_utils.validate_float("not_a_number") is False
        
        # Validation de date
        assert self.validation_utils.validate_date("2023-12-25") is True
        assert self.validation_utils.validate_date("25/12/2023", format="%d/%m/%Y") is True
        assert self.validation_utils.validate_date("invalid_date") is False
    
    def test_range_validation(self):
        """Test de validation de plage"""
        # Plage numérique
        assert self.validation_utils.validate_range(50, min_val=0, max_val=100) is True
        assert self.validation_utils.validate_range(-10, min_val=0, max_val=100) is False
        assert self.validation_utils.validate_range(150, min_val=0, max_val=100) is False
        
        # Longueur de chaîne
        assert self.validation_utils.validate_string_length("test", min_len=1, max_len=10) is True
        assert self.validation_utils.validate_string_length("", min_len=1, max_len=10) is False
        assert self.validation_utils.validate_string_length("very_long_string", min_len=1, max_len=10) is False
    
    def test_pattern_validation(self):
        """Test de validation de motif"""
        # Pattern de numéro de téléphone
        phone_pattern = r'^\+?1?\d{9,15}$'
        assert self.validation_utils.validate_pattern("+1234567890", phone_pattern) is True
        assert self.validation_utils.validate_pattern("123-456-7890", phone_pattern) is False
        
        # Pattern de code postal
        zip_pattern = r'^\d{5}(-\d{4})?$'
        assert self.validation_utils.validate_pattern("12345", zip_pattern) is True
        assert self.validation_utils.validate_pattern("12345-6789", zip_pattern) is True
        assert self.validation_utils.validate_pattern("1234", zip_pattern) is False
    
    def test_schema_validation(self):
        """Test de validation de schéma"""
        schema = {
            "name": {"type": "string", "required": True, "max_length": 50},
            "age": {"type": "integer", "required": True, "min": 0, "max": 150},
            "email": {"type": "email", "required": True},
            "tags": {"type": "list", "required": False}
        }
        
        # Données valides
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "tags": ["music", "rock"]
        }
        
        is_valid, errors = self.validation_utils.validate_schema(valid_data, schema)
        assert is_valid is True
        assert len(errors) == 0
        
        # Données invalides
        invalid_data = {
            "name": "",  # Vide
            "age": -5,   # Négatif
            "email": "invalid_email",  # Email invalide
            # "name" manquant
        }
        
        is_valid, errors = self.validation_utils.validate_schema(invalid_data, schema)
        assert is_valid is False
        assert len(errors) > 0


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires"""
    
    @pytest.mark.asyncio
    async def test_process_audio_batch_function(self):
        """Test de la fonction process_audio_batch"""
        audio_files = [
            {"path": "/fake/audio1.mp3", "duration": 180},
            {"path": "/fake/audio2.mp3", "duration": 240},
            {"path": "/fake/audio3.mp3", "duration": 200}
        ]
        
        with patch('app.ml_analytics.utils.AudioProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance
            mock_instance.process_file.return_value = {"features": "mock_features"}
            
            results = await process_audio_batch(audio_files)
            
            assert len(results) == 3
            assert all("features" in result for result in results)
    
    def test_extract_audio_features_function(self):
        """Test de la fonction extract_audio_features"""
        # Audio de test
        audio_data = np.random.rand(22050)  # 1 seconde d'audio
        
        features = extract_audio_features(
            audio_data,
            sample_rate=22050,
            features=['mfcc', 'spectral', 'tempo']
        )
        
        assert 'mfcc' in features
        assert 'spectral' in features
        assert 'tempo' in features
    
    def test_cache_result_decorator(self):
        """Test du décorateur cache_result"""
        call_count = 0
        
        @cache_result(ttl=60)
        def expensive_calculation(x, y):
            nonlocal call_count
            call_count += 1
            return x * y + x ** y
        
        # Premier appel
        result1 = expensive_calculation(2, 3)
        assert call_count == 1
        
        # Deuxième appel (cache)
        result2 = expensive_calculation(2, 3)
        assert result2 == result1
        assert call_count == 1
    
    def test_profile_performance_decorator(self):
        """Test du décorateur profile_performance"""
        @profile_performance
        def test_function():
            time.sleep(0.1)
            return "completed"
        
        result = test_function()
        assert result == "completed"
        
        # Vérifier que les métriques ont été collectées
        # (Cette vérification dépend de l'implémentation spécifique)
    
    def test_validate_data_function(self):
        """Test de la fonction validate_data"""
        data = {
            "name": "Test User",
            "age": 25,
            "email": "test@example.com"
        }
        
        rules = {
            "name": {"required": True, "type": "string"},
            "age": {"required": True, "type": "integer", "min": 0},
            "email": {"required": True, "type": "email"}
        }
        
        is_valid, errors = validate_data(data, rules)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_convert_format_function(self):
        """Test de la fonction convert_format"""
        # Test de conversion JSON vers dict
        json_data = '{"name": "test", "value": 123}'
        converted = convert_format(json_data, from_format="json", to_format="dict")
        
        assert isinstance(converted, dict)
        assert converted["name"] == "test"
        assert converted["value"] == 123
        
        # Test de conversion dict vers JSON
        dict_data = {"name": "test", "value": 123}
        converted = convert_format(dict_data, from_format="dict", to_format="json")
        
        assert isinstance(converted, str)
        assert "test" in converted
        assert "123" in converted
    
    def test_secure_hash_function(self):
        """Test de la fonction secure_hash"""
        data = "sensitive_data_123"
        
        hash1 = secure_hash(data)
        hash2 = secure_hash(data)
        
        assert hash1 == hash2  # Même données = même hash
        assert len(hash1) > 0
        
        # Données différentes = hash différents
        hash3 = secure_hash("different_data")
        assert hash3 != hash1
    
    def test_encrypt_decrypt_functions(self):
        """Test des fonctions encrypt_data et decrypt_data"""
        original_data = "confidential_information"
        encryption_key = "secret_key_123"
        
        # Chiffrement
        encrypted = encrypt_data(original_data, encryption_key)
        assert encrypted != original_data
        
        # Déchiffrement
        decrypted = decrypt_data(encrypted, encryption_key)
        assert decrypted == original_data
    
    def test_sanitize_input_function(self):
        """Test de la fonction sanitize_input"""
        malicious_input = "<script>alert('xss')</script>"
        
        sanitized = sanitize_input(malicious_input)
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized
    
    def test_normalize_features_function(self):
        """Test de la fonction normalize_features"""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        normalized = normalize_features(features, method="minmax")
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        
        # Test avec standardisation
        standardized = normalize_features(features, method="standard")
        
        assert abs(standardized.mean()) < 1e-10  # Moyenne proche de 0
        assert abs(standardized.std() - 1) < 1e-10  # Écart-type proche de 1
    
    def test_calculate_similarity_function(self):
        """Test de la fonction calculate_similarity"""
        vector1 = np.array([1, 2, 3, 4])
        vector2 = np.array([2, 3, 4, 5])
        vector3 = np.array([-1, -2, -3, -4])
        
        # Similarité cosinus entre vecteurs similaires
        similarity_high = calculate_similarity(vector1, vector2, method="cosine")
        assert similarity_high > 0.9
        
        # Similarité cosinus entre vecteurs opposés
        similarity_low = calculate_similarity(vector1, vector3, method="cosine")
        assert similarity_low < -0.9
        
        # Distance euclidienne
        distance = calculate_similarity(vector1, vector2, method="euclidean")
        assert distance > 0
    
    @pytest.mark.asyncio
    async def test_batch_process_function(self):
        """Test de la fonction batch_process"""
        items = list(range(100))
        
        async def process_item(item):
            await asyncio.sleep(0.001)  # Simuler traitement
            return item * 2
        
        results = await batch_process(
            items,
            process_item,
            batch_size=10,
            max_workers=3
        )
        
        assert len(results) == 100
        assert results[0] == 0
        assert results[50] == 100
        assert results[99] == 198
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_function(self):
        """Test de la fonction retry_with_backoff"""
        attempt_count = 0
        
        async def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await retry_with_backoff(
            failing_function,
            max_retries=3,
            initial_delay=0.01
        )
        
        assert result == "success"
        assert attempt_count == 3
    
    def test_measure_execution_time_decorator(self):
        """Test du décorateur measure_execution_time"""
        @measure_execution_time
        def timed_function():
            time.sleep(0.1)
            return "completed"
        
        result = timed_function()
        assert result == "completed"
        
        # Vérifier que le temps a été mesuré
        # (La vérification exacte dépend de l'implémentation)


# Fixtures pour les tests
@pytest.fixture
def sample_dataframe():
    """DataFrame de test"""
    return pd.DataFrame({
        'track_id': ['track1', 'track2', 'track3'],
        'artist': ['Artist A', 'Artist B', 'Artist C'],
        'popularity': [80, 65, 90],
        'duration_ms': [180000, 240000, 200000],
        'genre': ['rock', 'pop', 'jazz']
    })


@pytest.fixture
def sample_audio_signal():
    """Signal audio de test"""
    return np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))


@pytest.fixture
def temp_workspace():
    """Espace de travail temporaire"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Tests d'intégration
@pytest.mark.integration
class TestUtilsIntegration:
    """Tests d'intégration pour les utilitaires"""
    
    @pytest.mark.asyncio
    async def test_full_data_processing_pipeline(self):
        """Test du pipeline complet de traitement de données"""
        # Données brutes
        raw_data = pd.DataFrame({
            'track_id': ['track1', 'track2', None, 'track4'],
            'popularity': [80, None, 90, 65],
            'duration_ms': [180000, 240000, -1000, 200000],
            'genre': ['rock', 'pop', '', 'jazz']
        })
        
        # Pipeline de traitement
        processor = DataProcessor()
        
        # 1. Nettoyage
        clean_data = processor.clean_data(raw_data, {
            'missing_values': 'drop',
            'negative_values': 'clip',
            'empty_strings': 'drop'
        })
        
        # 2. Validation
        validation_result = processor.validate_data(clean_data, {
            'track_id': {'required': True, 'unique': True},
            'popularity': {'min': 0, 'max': 100},
            'duration_ms': {'min': 1000, 'max': 600000}
        })
        
        # 3. Traitement
        processed_data = processor.process_dataframe(clean_data, [
            {'type': 'normalize', 'columns': ['popularity']},
            {'type': 'encode', 'columns': ['genre'], 'method': 'onehot'}
        ])
        
        # Vérifications
        assert validation_result['is_valid'] is True
        assert len(processed_data) > 0
        assert 'popularity_normalized' in processed_data.columns
    
    def test_security_validation_pipeline(self):
        """Test du pipeline de sécurité et validation"""
        security_utils = SecurityUtils()
        validation_utils = ValidationUtils()
        
        # Données utilisateur non sécurisées
        user_input = {
            "username": "<script>alert('xss')</script>",
            "email": "user@example.com",
            "password": "password123",
            "age": "25"
        }
        
        # 1. Sanitisation
        sanitized_input = {
            "username": security_utils.sanitize_input(user_input["username"]),
            "email": user_input["email"],
            "password": user_input["password"],
            "age": user_input["age"]
        }
        
        # 2. Validation
        schema = {
            "username": {"type": "string", "required": True, "max_length": 50},
            "email": {"type": "email", "required": True},
            "age": {"type": "integer", "required": True, "min": 0, "max": 150}
        }
        
        is_valid, errors = validation_utils.validate_schema(sanitized_input, schema)
        
        # 3. Sécurisation
        if is_valid:
            hashed_password = security_utils.hash_password(sanitized_input["password"])
            token = security_utils.generate_secure_token()
        
        # Vérifications
        assert "<script>" not in sanitized_input["username"]
        assert is_valid is True
        assert len(hashed_password) > 0
        assert len(token) > 0


# Tests de performance
@pytest.mark.performance
class TestUtilsPerformance:
    """Tests de performance pour les utilitaires"""
    
    def test_data_processing_performance(self):
        """Test de performance de traitement de données"""
        # Grande dataset
        large_df = pd.DataFrame({
            'id': range(100000),
            'value': np.random.rand(100000),
            'category': np.random.choice(['A', 'B', 'C'], 100000)
        })
        
        processor = DataProcessor()
        
        start_time = time.time()
        
        processed_df = processor.process_dataframe(large_df, [
            {'type': 'normalize', 'columns': ['value']},
            {'type': 'encode', 'columns': ['category'], 'method': 'onehot'}
        ])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Devrait être rapide pour 100k enregistrements
        assert duration < 5.0
        assert len(processed_df) == 100000
    
    def test_caching_performance(self):
        """Test de performance du cache"""
        cache_manager = CacheManager()
        
        # Test de beaucoup d'opérations de cache
        start_time = time.time()
        
        for i in range(10000):
            cache_manager.set(f"key_{i}", f"value_{i}")
        
        for i in range(10000):
            cache_manager.get(f"key_{i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Opérations de cache rapides
        assert duration < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
