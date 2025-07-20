"""
Tests pour le moteur principal ML Analytics
===========================================

Tests complets pour MLAnalyticsEngine avec couverture de:
- Initialisation et configuration
- Gestion des modèles
- Orchestration des pipelines
- Monitoring et santé du système
- Performance et optimisations
"""

import pytest
import asyncio
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ml_analytics.core import MLAnalyticsEngine
from ml_analytics.config import MLAnalyticsConfig
from ml_analytics.exceptions import MLAnalyticsError, ModelError, PipelineError


class TestMLAnalyticsEngine:
    """Tests pour la classe MLAnalyticsEngine."""
    
    @pytest.fixture
    async def engine(self):
        """Instance de test du moteur ML."""
        engine = MLAnalyticsEngine()
        test_config = {
            "environment": "testing",
            "models": {"path": "/tmp/test_models"},
            "monitoring": {"enabled": False}
        }
        await engine.initialize(config=test_config)
        yield engine
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test l'initialisation du moteur ML."""
        engine = MLAnalyticsEngine()
        assert not engine.is_initialized
        
        # Test initialisation réussie
        config = {"environment": "testing", "monitoring": {"enabled": False}}
        await engine.initialize(config=config)
        
        assert engine.is_initialized
        assert engine.config is not None
        assert engine.health_monitor is not None
        
        # Cleanup
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_engine_initialization_failure(self):
        """Test la gestion d'erreur lors de l'initialisation."""
        engine = MLAnalyticsEngine()
        
        # Configuration invalide
        with pytest.raises(MLAnalyticsError):
            await engine.initialize(config={"invalid": "config"})
    
    @pytest.mark.asyncio
    async def test_model_registration(self, engine):
        """Test l'enregistrement de modèles."""
        # Mock d'un modèle
        mock_model = AsyncMock()
        mock_model.name = "test_model"
        mock_model.version = "1.0.0"
        mock_model.initialize = AsyncMock()
        
        # Enregistrement du modèle
        await engine.register_model("test_model", mock_model)
        
        assert "test_model" in engine.models
        assert engine.models["test_model"] == mock_model
        mock_model.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_unregistration(self, engine):
        """Test la suppression de modèles."""
        # Enregistrer d'abord un modèle
        mock_model = AsyncMock()
        mock_model.name = "test_model"
        mock_model.cleanup = AsyncMock()
        
        await engine.register_model("test_model", mock_model)
        assert "test_model" in engine.models
        
        # Supprimer le modèle
        await engine.unregister_model("test_model")
        
        assert "test_model" not in engine.models
        mock_model.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, engine):
        """Test l'exécution de pipelines ML."""
        # Mock pipeline
        pipeline_config = {
            "name": "test_pipeline",
            "steps": [
                {"name": "data_loading", "type": "data_loader"},
                {"name": "preprocessing", "type": "preprocessor"},
                {"name": "model_inference", "type": "model"}
            ]
        }
        
        with patch.object(engine, '_execute_pipeline_step') as mock_step:
            mock_step.return_value = {"status": "success", "data": {"processed": True}}
            
            result = await engine.execute_pipeline(pipeline_config, {"input": "test_data"})
            
            assert result["status"] == "success"
            assert result["pipeline_name"] == "test_pipeline"
            assert mock_step.call_count == 3  # 3 étapes
    
    @pytest.mark.asyncio
    async def test_pipeline_execution_failure(self, engine):
        """Test la gestion d'erreur dans l'exécution de pipeline."""
        pipeline_config = {
            "name": "failing_pipeline",
            "steps": [{"name": "failing_step", "type": "faulty_processor"}]
        }
        
        with patch.object(engine, '_execute_pipeline_step') as mock_step:
            mock_step.side_effect = Exception("Pipeline step failed")
            
            with pytest.raises(PipelineError):
                await engine.execute_pipeline(pipeline_config, {})
    
    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        """Test le contrôle de santé du système."""
        # Mock des composants sains
        with patch.object(engine.health_monitor, 'check_system_health') as mock_health:
            mock_health.return_value = {
                "healthy": True,
                "components": {
                    "database": {"status": "healthy", "response_time": 0.01},
                    "cache": {"status": "healthy", "hit_ratio": 0.85},
                    "models": {"status": "healthy", "loaded": 3}
                },
                "timestamp": datetime.now().isoformat()
            }
            
            health = await engine.health_check()
            
            assert health["healthy"] is True
            assert "components" in health
            assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, engine):
        """Test le contrôle de santé avec composants défaillants."""
        with patch.object(engine.health_monitor, 'check_system_health') as mock_health:
            mock_health.return_value = {
                "healthy": False,
                "components": {
                    "database": {"status": "unhealthy", "error": "Connection timeout"},
                    "cache": {"status": "healthy", "hit_ratio": 0.85},
                    "models": {"status": "degraded", "loaded": 1, "failed": 2}
                },
                "timestamp": datetime.now().isoformat()
            }
            
            health = await engine.health_check()
            
            assert health["healthy"] is False
            assert health["components"]["database"]["status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, engine):
        """Test la collecte de métriques de performance."""
        with patch.object(engine, '_collect_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "requests_per_second": 150.5,
                "average_response_time": 0.234,
                "error_rate": 0.001,
                "memory_usage": 0.65,
                "cpu_usage": 0.45,
                "cache_hit_ratio": 0.89
            }
            
            metrics = await engine.get_performance_metrics()
            
            assert "requests_per_second" in metrics
            assert "average_response_time" in metrics
            assert "error_rate" in metrics
            assert metrics["cache_hit_ratio"] == 0.89
    
    @pytest.mark.asyncio
    async def test_model_lifecycle_management(self, engine):
        """Test la gestion du cycle de vie des modèles."""
        mock_model = AsyncMock()
        mock_model.name = "lifecycle_model"
        mock_model.version = "1.0.0"
        mock_model.train = AsyncMock()
        mock_model.save = AsyncMock()
        mock_model.load = AsyncMock()
        mock_model.validate = AsyncMock(return_value={"accuracy": 0.95})
        
        # Enregistrement
        await engine.register_model("lifecycle_model", mock_model)
        
        # Formation
        training_data = {"features": np.random.rand(100, 10), "labels": np.random.randint(0, 2, 100)}
        await engine.train_model("lifecycle_model", training_data)
        mock_model.train.assert_called_once()
        
        # Sauvegarde
        await engine.save_model("lifecycle_model", "/tmp/test_model")
        mock_model.save.assert_called_once()
        
        # Validation
        validation_result = await engine.validate_model("lifecycle_model", training_data)
        assert validation_result["accuracy"] == 0.95
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_execution(self, engine):
        """Test l'exécution concurrente de pipelines."""
        pipeline_configs = [
            {"name": f"pipeline_{i}", "steps": [{"name": "step_1", "type": "processor"}]}
            for i in range(5)
        ]
        
        with patch.object(engine, '_execute_pipeline_step') as mock_step:
            mock_step.return_value = {"status": "success", "data": {"processed": True}}
            
            # Exécution concurrente
            tasks = [
                engine.execute_pipeline(config, {"input": f"data_{i}"})
                for i, config in enumerate(pipeline_configs)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            for result in results:
                assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_resource_management(self, engine):
        """Test la gestion des ressources système."""
        # Simulation d'utilisation intensive
        with patch.object(engine, '_monitor_resource_usage') as mock_monitor:
            mock_monitor.return_value = {
                "memory_usage": 0.85,  # 85% d'utilisation
                "cpu_usage": 0.90,     # 90% d'utilisation
                "gpu_usage": 0.75      # 75% d'utilisation
            }
            
            # Vérifier que l'engine détecte la forte utilisation
            resource_status = await engine.check_resource_usage()
            
            assert resource_status["memory_usage"] > 0.8
            assert resource_status["cpu_usage"] > 0.8
            assert "gpu_usage" in resource_status
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, engine):
        """Test la gestion d'erreur et la récupération."""
        # Simuler une erreur de modèle
        mock_model = AsyncMock()
        mock_model.name = "error_model"
        mock_model.predict = AsyncMock(side_effect=Exception("Model prediction failed"))
        
        await engine.register_model("error_model", mock_model)
        
        # Tenter une prédiction qui échoue
        with pytest.raises(ModelError):
            await engine.predict("error_model", {"input": "test_data"})
        
        # Vérifier que l'erreur est enregistrée
        assert engine.error_count > 0
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, engine):
        """Test l'intégration du cache."""
        # Mock du cache
        with patch.object(engine, 'cache') as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            mock_cache.set = AsyncMock()
            
            # Première requête (cache miss)
            result1 = await engine.get_cached_prediction("test_model", {"input": "data"})
            mock_cache.get.assert_called_once()
            
            # Simuler un cache hit
            mock_cache.get.return_value = {"prediction": "cached_result"}
            result2 = await engine.get_cached_prediction("test_model", {"input": "data"})
            
            assert result2["prediction"] == "cached_result"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, engine):
        """Test le traitement par lot."""
        # Données de test par lot
        batch_data = [
            {"id": i, "features": np.random.rand(10).tolist()}
            for i in range(50)
        ]
        
        with patch.object(engine, '_process_batch') as mock_batch:
            mock_batch.return_value = [
                {"id": item["id"], "prediction": f"result_{item['id']}"}
                for item in batch_data
            ]
            
            results = await engine.process_batch(batch_data, batch_size=10)
            
            assert len(results) == 50
            assert all("prediction" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_model_versioning(self, engine):
        """Test la gestion des versions de modèles."""
        # Modèle v1
        model_v1 = AsyncMock()
        model_v1.name = "versioned_model"
        model_v1.version = "1.0.0"
        
        # Modèle v2
        model_v2 = AsyncMock()
        model_v2.name = "versioned_model"
        model_v2.version = "2.0.0"
        
        # Enregistrer v1
        await engine.register_model("versioned_model", model_v1)
        assert engine.get_model_version("versioned_model") == "1.0.0"
        
        # Mettre à jour vers v2
        await engine.update_model("versioned_model", model_v2)
        assert engine.get_model_version("versioned_model") == "2.0.0"
    
    @pytest.mark.asyncio
    async def test_cleanup_and_shutdown(self, engine):
        """Test le nettoyage et l'arrêt propre."""
        # Ajouter quelques modèles
        for i in range(3):
            mock_model = AsyncMock()
            mock_model.name = f"cleanup_model_{i}"
            mock_model.cleanup = AsyncMock()
            await engine.register_model(f"cleanup_model_{i}", mock_model)
        
        assert len(engine.models) == 3
        
        # Test cleanup
        await engine.cleanup()
        
        # Vérifier que tous les modèles ont été nettoyés
        for model in engine.models.values():
            model.cleanup.assert_called_once()
        
        assert not engine.is_initialized


class TestMLAnalyticsEnginePerformance:
    """Tests de performance pour MLAnalyticsEngine."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_load_performance(self):
        """Test les performances sous charge élevée."""
        engine = MLAnalyticsEngine()
        await engine.initialize({"environment": "testing", "monitoring": {"enabled": False}})
        
        # Simuler une charge élevée
        start_time = datetime.now()
        
        tasks = []
        for i in range(100):
            task = engine.health_check()
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Vérifier que toutes les requêtes ont réussi
        assert len(results) == 100
        
        # Performance: moins de 5 secondes pour 100 requêtes
        assert duration < 5.0
        
        await engine.cleanup()
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test l'utilisation mémoire sous charge."""
        import psutil
        import os
        
        engine = MLAnalyticsEngine()
        await engine.initialize({"environment": "testing", "monitoring": {"enabled": False}})
        
        # Mesurer la mémoire initiale
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Créer une charge de travail
        for i in range(50):
            mock_model = AsyncMock()
            mock_model.name = f"memory_test_model_{i}"
            await engine.register_model(f"memory_test_model_{i}", mock_model)
        
        # Mesurer la mémoire après charge
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # L'augmentation mémoire ne devrait pas dépasser 100MB
        assert memory_increase < 100
        
        await engine.cleanup()


@pytest.mark.integration
class TestMLAnalyticsEngineIntegration:
    """Tests d'intégration pour MLAnalyticsEngine."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """Test complet du workflow ML."""
        engine = MLAnalyticsEngine()
        await engine.initialize({"environment": "testing", "monitoring": {"enabled": False}})
        
        # 1. Enregistrer un modèle de test
        mock_model = AsyncMock()
        mock_model.name = "integration_model"
        mock_model.version = "1.0.0"
        mock_model.predict = AsyncMock(return_value={"prediction": "test_result"})
        
        await engine.register_model("integration_model", mock_model)
        
        # 2. Exécuter une prédiction
        result = await engine.predict("integration_model", {"input": "test_data"})
        assert result["prediction"] == "test_result"
        
        # 3. Vérifier la santé du système
        health = await engine.health_check()
        assert health["healthy"] is True
        
        # 4. Collecter les métriques
        metrics = await engine.get_performance_metrics()
        assert "requests_per_second" in metrics
        
        # 5. Cleanup
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_propagation_integration(self):
        """Test la propagation d'erreurs dans le système intégré."""
        engine = MLAnalyticsEngine()
        await engine.initialize({"environment": "testing", "monitoring": {"enabled": False}})
        
        # Modèle défaillant
        faulty_model = AsyncMock()
        faulty_model.name = "faulty_model"
        faulty_model.predict = AsyncMock(side_effect=Exception("Model failure"))
        
        await engine.register_model("faulty_model", faulty_model)
        
        # Vérifier que l'erreur est correctement gérée
        with pytest.raises(ModelError):
            await engine.predict("faulty_model", {"input": "test"})
        
        # Vérifier que le système reste opérationnel
        health = await engine.health_check()
        # Le système peut être dégradé mais pas complètement défaillant
        
        await engine.cleanup()
