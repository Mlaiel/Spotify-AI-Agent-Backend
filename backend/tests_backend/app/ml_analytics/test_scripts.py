# 🧪 ML Analytics Scripts Tests
# =============================
# 
# Tests ultra-avancés pour les scripts d'automatisation
# Enterprise automation scripts testing
#
# 🎖️ Implementation par l'équipe d'experts:
# ✅ DevOps Specialist + Lead Dev + Architecte Backend
#
# 👨‍💻 Développé par: Fahed Mlaiel
# =============================

"""
🔧 Automation Scripts Test Suite
================================

Comprehensive testing for automation scripts:
- Data pipeline automation
- Model deployment scripts
- Maintenance and cleanup scripts
- Monitoring and alerting automation
- Performance optimization scripts
"""

import pytest
import asyncio
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta
import subprocess
import logging
import shutil
import pandas as pd
import numpy as np

# Import modules to test
from app.ml_analytics.scripts import (
    DataPipelineScript, ModelDeploymentScript, MaintenanceScript,
    BackupScript, MigrationScript, OptimizationScript,
    MonitoringScript, AlertingScript, CleanupScript,
    ScriptManager, ScriptExecutor, ScriptScheduler,
    run_script_async, execute_pipeline, deploy_model,
    cleanup_old_data, backup_system, migrate_data
)


class TestScriptManager:
    """Tests pour le gestionnaire de scripts"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.script_manager = ScriptManager()
        self.temp_dir = tempfile.mkdtemp()
        self.script_dir = Path(self.temp_dir) / "scripts"
        self.script_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_script_manager_creation(self):
        """Test de création du gestionnaire"""
        assert isinstance(self.script_manager, ScriptManager)
        assert len(self.script_manager.registered_scripts) == 0
        assert self.script_manager.execution_history == []
    
    def test_register_script(self):
        """Test d'enregistrement de script"""
        script = DataPipelineScript(
            name="test_pipeline",
            description="Test data pipeline",
            script_path="/path/to/script.py"
        )
        
        self.script_manager.register_script(script)
        
        assert "test_pipeline" in self.script_manager.registered_scripts
        assert self.script_manager.registered_scripts["test_pipeline"] == script
    
    def test_get_script(self):
        """Test de récupération de script"""
        script = ModelDeploymentScript(
            name="deploy_model",
            description="Deploy ML model",
            model_path="/models/test_model.pkl"
        )
        
        self.script_manager.register_script(script)
        
        retrieved_script = self.script_manager.get_script("deploy_model")
        assert retrieved_script == script
        
        # Script inexistant
        non_existent = self.script_manager.get_script("non_existent")
        assert non_existent is None
    
    def test_list_scripts(self):
        """Test de listage des scripts"""
        scripts = [
            DataPipelineScript("pipeline1", "Pipeline 1", "/path1"),
            ModelDeploymentScript("deploy1", "Deploy 1", "/model1"),
            MaintenanceScript("maintenance1", "Maintenance 1", "/maint1")
        ]
        
        for script in scripts:
            self.script_manager.register_script(script)
        
        script_list = self.script_manager.list_scripts()
        
        assert len(script_list) == 3
        assert all(s.name in ["pipeline1", "deploy1", "maintenance1"] for s in script_list)
    
    @pytest.mark.asyncio
    async def test_execute_script(self):
        """Test d'exécution de script"""
        script = DataPipelineScript(
            name="test_execution",
            description="Test execution",
            script_path="/fake/path.py"
        )
        
        self.script_manager.register_script(script)
        
        with patch.object(script, 'execute', return_value={"status": "success"}):
            result = await self.script_manager.execute_script("test_execution")
            
            assert result["status"] == "success"
            assert len(self.script_manager.execution_history) == 1
    
    def test_get_execution_history(self):
        """Test de récupération de l'historique d'exécution"""
        # Simuler quelques exécutions
        executions = [
            {
                "script_name": "pipeline1",
                "timestamp": datetime.now(),
                "status": "success",
                "duration": 45.2
            },
            {
                "script_name": "deploy1", 
                "timestamp": datetime.now(),
                "status": "failed",
                "error": "Model not found"
            }
        ]
        
        self.script_manager.execution_history = executions
        
        history = self.script_manager.get_execution_history()
        assert len(history) == 2
        
        # Filtrer par script
        pipeline_history = self.script_manager.get_execution_history(
            script_name="pipeline1"
        )
        assert len(pipeline_history) == 1
        assert pipeline_history[0]["script_name"] == "pipeline1"


class TestDataPipelineScript:
    """Tests pour les scripts de pipeline de données"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = Path(self.temp_dir) / "input.csv"
        self.output_file = Path(self.temp_dir) / "output.csv"
        
        # Créer des données de test
        test_data = pd.DataFrame({
            'track_id': ['track1', 'track2', 'track3'],
            'artist': ['Artist A', 'Artist B', 'Artist C'],
            'popularity': [80, 65, 90]
        })
        test_data.to_csv(self.input_file, index=False)
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_pipeline_script_creation(self):
        """Test de création de script de pipeline"""
        script = DataPipelineScript(
            name="spotify_data_pipeline",
            description="Process Spotify track data",
            script_path="/scripts/process_spotify_data.py",
            input_path=str(self.input_file),
            output_path=str(self.output_file),
            config={
                "batch_size": 1000,
                "parallel_workers": 4
            }
        )
        
        assert script.name == "spotify_data_pipeline"
        assert script.script_type == "data_pipeline"
        assert script.input_path == str(self.input_file)
        assert script.output_path == str(self.output_file)
        assert script.config["batch_size"] == 1000
    
    @pytest.mark.asyncio
    async def test_execute_data_pipeline(self):
        """Test d'exécution de pipeline de données"""
        script = DataPipelineScript(
            name="test_pipeline",
            description="Test pipeline",
            script_path="/fake/script.py",
            input_path=str(self.input_file),
            output_path=str(self.output_file)
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Pipeline completed successfully",
                stderr=""
            )
            
            result = await script.execute()
            
            assert result["status"] == "success"
            assert "duration" in result
            mock_run.assert_called_once()
    
    def test_validate_pipeline_config(self):
        """Test de validation de configuration de pipeline"""
        valid_config = {
            "batch_size": 1000,
            "parallel_workers": 4,
            "timeout": 3600
        }
        
        script = DataPipelineScript(
            name="test_pipeline",
            description="Test",
            script_path="/fake/script.py",
            config=valid_config
        )
        
        assert script.validate_config() is True
        
        # Configuration invalide
        invalid_config = {
            "batch_size": -1,  # Invalide
            "parallel_workers": 0  # Invalide
        }
        
        script.config = invalid_config
        assert script.validate_config() is False
    
    def test_data_transformation(self):
        """Test de transformation de données"""
        script = DataPipelineScript(
            name="transform_pipeline",
            description="Transform data",
            script_path="/fake/script.py"
        )
        
        # Données d'entrée
        input_data = pd.DataFrame({
            'popularity': [10, 50, 90, 30],
            'duration_ms': [180000, 240000, 200000, 160000]
        })
        
        # Transformation : normalisation
        transformed_data = script.transform_data(
            input_data,
            transformations=[
                {"type": "normalize", "columns": ["popularity"]},
                {"type": "scale", "columns": ["duration_ms"], "factor": 0.001}
            ]
        )
        
        assert 'popularity_normalized' in transformed_data.columns
        assert 'duration_ms_scaled' in transformed_data.columns
        assert transformed_data['popularity_normalized'].max() <= 1.0
    
    def test_data_validation(self):
        """Test de validation de données"""
        script = DataPipelineScript(
            name="validation_pipeline", 
            description="Validate data",
            script_path="/fake/script.py"
        )
        
        # Données valides
        valid_data = pd.DataFrame({
            'track_id': ['track1', 'track2'],
            'popularity': [80, 65],
            'duration_ms': [180000, 240000]
        })
        
        validation_rules = {
            'track_id': {'required': True, 'unique': True},
            'popularity': {'min': 0, 'max': 100},
            'duration_ms': {'min': 1000, 'max': 600000}
        }
        
        validation_result = script.validate_data(valid_data, validation_rules)
        
        assert validation_result['is_valid'] is True
        assert validation_result['error_count'] == 0
        
        # Données invalides
        invalid_data = pd.DataFrame({
            'track_id': ['track1', 'track1'],  # Doublons
            'popularity': [150, -10],  # Hors limites
            'duration_ms': [100, 700000]  # Hors limites
        })
        
        validation_result = script.validate_data(invalid_data, validation_rules)
        
        assert validation_result['is_valid'] is False
        assert validation_result['error_count'] > 0


class TestModelDeploymentScript:
    """Tests pour les scripts de déploiement de modèle"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "model.pkl"
        
        # Créer un faux modèle
        with open(self.model_path, 'w') as f:
            f.write("fake_model_data")
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_deployment_script_creation(self):
        """Test de création de script de déploiement"""
        script = ModelDeploymentScript(
            name="deploy_recommendation_model",
            description="Deploy recommendation model to production",
            model_path=str(self.model_path),
            deployment_target="production",
            config={
                "replicas": 3,
                "memory_limit": "2Gi",
                "cpu_limit": "1000m"
            }
        )
        
        assert script.name == "deploy_recommendation_model"
        assert script.script_type == "model_deployment"
        assert script.model_path == str(self.model_path)
        assert script.deployment_target == "production"
        assert script.config["replicas"] == 3
    
    @pytest.mark.asyncio
    async def test_deploy_model(self):
        """Test de déploiement de modèle"""
        script = ModelDeploymentScript(
            name="test_deployment",
            description="Test deployment",
            model_path=str(self.model_path),
            deployment_target="staging"
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Model deployed successfully",
                stderr=""
            )
            
            result = await script.execute()
            
            assert result["status"] == "success"
            assert "deployment_id" in result
    
    def test_validate_model(self):
        """Test de validation de modèle"""
        script = ModelDeploymentScript(
            name="test_validation",
            description="Test validation",
            model_path=str(self.model_path)
        )
        
        # Validation basique (fichier existe)
        validation_result = script.validate_model()
        
        assert validation_result["is_valid"] is True
        
        # Modèle inexistant
        script.model_path = "/fake/nonexistent.pkl"
        validation_result = script.validate_model()
        
        assert validation_result["is_valid"] is False
        assert "not found" in validation_result["errors"][0].lower()
    
    def test_rollback_deployment(self):
        """Test de rollback de déploiement"""
        script = ModelDeploymentScript(
            name="test_rollback",
            description="Test rollback",
            model_path=str(self.model_path)
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Rollback completed",
                stderr=""
            )
            
            result = script.rollback_deployment("deployment_123")
            
            assert result["status"] == "success"
            mock_run.assert_called()
    
    def test_health_check_deployment(self):
        """Test de contrôle de santé du déploiement"""
        script = ModelDeploymentScript(
            name="test_health",
            description="Test health check",
            model_path=str(self.model_path)
        )
        
        with patch('requests.get') as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy", "version": "1.0.0"}
            )
            
            health_status = script.check_deployment_health("http://api.example.com/health")
            
            assert health_status["is_healthy"] is True
            assert health_status["version"] == "1.0.0"


class TestMaintenanceScript:
    """Tests pour les scripts de maintenance"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Créer quelques fichiers de log
        for i in range(5):
            log_file = self.log_dir / f"app_{i}.log"
            with open(log_file, 'w') as f:
                f.write(f"Log content {i}")
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_maintenance_script_creation(self):
        """Test de création de script de maintenance"""
        script = MaintenanceScript(
            name="system_maintenance",
            description="Perform system maintenance tasks",
            maintenance_type="cleanup",
            config={
                "log_retention_days": 30,
                "cleanup_temp_files": True,
                "optimize_database": True
            }
        )
        
        assert script.name == "system_maintenance"
        assert script.script_type == "maintenance"
        assert script.maintenance_type == "cleanup"
        assert script.config["log_retention_days"] == 30
    
    @pytest.mark.asyncio
    async def test_execute_maintenance(self):
        """Test d'exécution de maintenance"""
        script = MaintenanceScript(
            name="test_maintenance",
            description="Test maintenance",
            maintenance_type="cleanup"
        )
        
        with patch.object(script, '_cleanup_logs') as mock_cleanup:
            with patch.object(script, '_optimize_database') as mock_optimize:
                mock_cleanup.return_value = {"deleted_files": 10}
                mock_optimize.return_value = {"tables_optimized": 5}
                
                result = await script.execute()
                
                assert result["status"] == "success"
                assert "cleanup_results" in result
    
    def test_cleanup_old_logs(self):
        """Test de nettoyage des anciens logs"""
        script = MaintenanceScript(
            name="log_cleanup",
            description="Clean up old logs",
            maintenance_type="cleanup",
            config={"log_retention_days": 30}
        )
        
        # Créer des logs anciens et récents
        old_date = datetime.now() - timedelta(days=40)
        recent_date = datetime.now() - timedelta(days=10)
        
        old_log = self.log_dir / "old.log"
        recent_log = self.log_dir / "recent.log"
        
        old_log.touch()
        recent_log.touch()
        
        # Simuler les dates de modification
        os.utime(old_log, (old_date.timestamp(), old_date.timestamp()))
        os.utime(recent_log, (recent_date.timestamp(), recent_date.timestamp()))
        
        result = script.cleanup_old_logs(str(self.log_dir))
        
        assert result["deleted_count"] >= 1
        assert not old_log.exists() or recent_log.exists()
    
    def test_database_optimization(self):
        """Test d'optimisation de base de données"""
        script = MaintenanceScript(
            name="db_optimization",
            description="Optimize database",
            maintenance_type="optimization"
        )
        
        with patch('app.ml_analytics.scripts.asyncpg.connect') as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.return_value = None
            
            result = script.optimize_database()
            
            assert result["status"] == "success"
            assert "optimized_tables" in result
    
    def test_system_cleanup(self):
        """Test de nettoyage système"""
        script = MaintenanceScript(
            name="system_cleanup",
            description="System cleanup",
            maintenance_type="cleanup"
        )
        
        # Créer des fichiers temporaires
        temp_files = []
        for i in range(3):
            temp_file = Path(self.temp_dir) / f"temp_{i}.tmp"
            temp_file.touch()
            temp_files.append(temp_file)
        
        result = script.cleanup_temp_files(self.temp_dir, pattern="*.tmp")
        
        assert result["deleted_count"] == 3
        assert all(not f.exists() for f in temp_files)


class TestBackupScript:
    """Tests pour les scripts de sauvegarde"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = Path(self.temp_dir) / "source"
        self.backup_dir = Path(self.temp_dir) / "backup"
        
        self.source_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Créer des fichiers source
        for i in range(3):
            source_file = self.source_dir / f"file_{i}.txt"
            with open(source_file, 'w') as f:
                f.write(f"Content {i}")
    
    def teardown_method(self):
        """Nettoyage après chaque test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backup_script_creation(self):
        """Test de création de script de sauvegarde"""
        script = BackupScript(
            name="database_backup",
            description="Backup database and models",
            source_paths=["/data/database", "/models"],
            backup_destination="/backups",
            config={
                "compression": True,
                "encryption": True,
                "retention_days": 30
            }
        )
        
        assert script.name == "database_backup"
        assert script.script_type == "backup"
        assert "/data/database" in script.source_paths
        assert script.backup_destination == "/backups"
        assert script.config["compression"] is True
    
    @pytest.mark.asyncio
    async def test_execute_backup(self):
        """Test d'exécution de sauvegarde"""
        script = BackupScript(
            name="test_backup",
            description="Test backup",
            source_paths=[str(self.source_dir)],
            backup_destination=str(self.backup_dir)
        )
        
        result = await script.execute()
        
        assert result["status"] == "success"
        assert "backup_size" in result
        assert "backup_path" in result
    
    def test_create_compressed_backup(self):
        """Test de création de sauvegarde compressée"""
        script = BackupScript(
            name="compressed_backup",
            description="Compressed backup",
            source_paths=[str(self.source_dir)],
            backup_destination=str(self.backup_dir),
            config={"compression": True}
        )
        
        backup_path = script.create_backup()
        
        assert backup_path.exists()
        assert backup_path.suffix == ".tar.gz"
    
    def test_verify_backup(self):
        """Test de vérification de sauvegarde"""
        script = BackupScript(
            name="verify_backup",
            description="Verify backup",
            source_paths=[str(self.source_dir)],
            backup_destination=str(self.backup_dir)
        )
        
        # Créer une sauvegarde
        backup_path = script.create_backup()
        
        # Vérifier la sauvegarde
        verification_result = script.verify_backup(backup_path)
        
        assert verification_result["is_valid"] is True
        assert verification_result["file_count"] == 3
    
    def test_cleanup_old_backups(self):
        """Test de nettoyage des anciennes sauvegardes"""
        script = BackupScript(
            name="cleanup_backup",
            description="Cleanup old backups",
            source_paths=[str(self.source_dir)],
            backup_destination=str(self.backup_dir),
            config={"retention_days": 7}
        )
        
        # Créer des sauvegardes anciennes et récentes
        old_backup = self.backup_dir / "old_backup.tar.gz"
        recent_backup = self.backup_dir / "recent_backup.tar.gz"
        
        old_backup.touch()
        recent_backup.touch()
        
        # Simuler les dates
        old_date = datetime.now() - timedelta(days=10)
        recent_date = datetime.now() - timedelta(days=3)
        
        os.utime(old_backup, (old_date.timestamp(), old_date.timestamp()))
        os.utime(recent_backup, (recent_date.timestamp(), recent_date.timestamp()))
        
        result = script.cleanup_old_backups()
        
        assert result["deleted_count"] >= 1
        assert not old_backup.exists() or recent_backup.exists()


class TestScriptExecutor:
    """Tests pour l'exécuteur de scripts"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.executor = ScriptExecutor()
    
    def test_script_executor_creation(self):
        """Test de création de l'exécuteur"""
        assert isinstance(self.executor, ScriptExecutor)
        assert self.executor.max_concurrent_executions == 5
        assert len(self.executor.running_scripts) == 0
    
    @pytest.mark.asyncio
    async def test_execute_script_async(self):
        """Test d'exécution asynchrone de script"""
        script = DataPipelineScript(
            name="async_test",
            description="Async test",
            script_path="/fake/script.py"
        )
        
        with patch.object(script, 'execute', return_value={"status": "success"}):
            result = await self.executor.execute_script_async(script)
            
            assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_limit(self):
        """Test de limite d'exécution concurrente"""
        scripts = []
        for i in range(10):
            script = DataPipelineScript(
                name=f"concurrent_test_{i}",
                description=f"Concurrent test {i}",
                script_path="/fake/script.py"
            )
            scripts.append(script)
        
        # Mock d'exécution lente
        async def slow_execute():
            await asyncio.sleep(0.1)
            return {"status": "success"}
        
        for script in scripts:
            script.execute = slow_execute
        
        # Lancer toutes les exécutions
        tasks = [
            self.executor.execute_script_async(script)
            for script in scripts
        ]
        
        # Certaines devraient être en attente due à la limite
        assert len(self.executor.running_scripts) <= self.executor.max_concurrent_executions
        
        # Attendre la completion
        results = await asyncio.gather(*tasks)
        
        assert all(r["status"] == "success" for r in results)
    
    def test_get_execution_status(self):
        """Test de récupération du statut d'exécution"""
        script = DataPipelineScript(
            name="status_test",
            description="Status test",
            script_path="/fake/script.py"
        )
        
        # Simuler une exécution en cours
        execution_id = "exec_123"
        self.executor.running_scripts[execution_id] = {
            "script": script,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        status = self.executor.get_execution_status(execution_id)
        
        assert status["status"] == "running"
        assert status["script_name"] == "status_test"
        assert "start_time" in status
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self):
        """Test d'annulation d'exécution"""
        script = DataPipelineScript(
            name="cancel_test",
            description="Cancel test",
            script_path="/fake/script.py"
        )
        
        # Simuler une exécution longue
        async def long_execute():
            await asyncio.sleep(10)
            return {"status": "success"}
        
        script.execute = long_execute
        
        # Lancer l'exécution
        task = asyncio.create_task(self.executor.execute_script_async(script))
        
        # Attendre un peu puis annuler
        await asyncio.sleep(0.1)
        
        execution_id = list(self.executor.running_scripts.keys())[0]
        cancelled = await self.executor.cancel_execution(execution_id)
        
        assert cancelled is True
        
        # Nettoyer
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestScriptScheduler:
    """Tests pour le planificateur de scripts"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.scheduler = ScriptScheduler()
    
    def test_scheduler_creation(self):
        """Test de création du planificateur"""
        assert isinstance(self.scheduler, ScriptScheduler)
        assert len(self.scheduler.scheduled_scripts) == 0
    
    def test_schedule_script(self):
        """Test de planification de script"""
        script = MaintenanceScript(
            name="scheduled_maintenance",
            description="Scheduled maintenance",
            maintenance_type="cleanup"
        )
        
        # Planifier pour exécution quotidienne à 2h du matin
        self.scheduler.schedule_script(
            script,
            schedule_type="daily",
            schedule_time="02:00"
        )
        
        assert "scheduled_maintenance" in self.scheduler.scheduled_scripts
        
        scheduled = self.scheduler.scheduled_scripts["scheduled_maintenance"]
        assert scheduled["schedule_type"] == "daily"
        assert scheduled["schedule_time"] == "02:00"
    
    def test_schedule_cron_expression(self):
        """Test de planification avec expression cron"""
        script = BackupScript(
            name="weekly_backup",
            description="Weekly backup",
            source_paths=["/data"],
            backup_destination="/backups"
        )
        
        # Planifier avec cron (tous les dimanches à minuit)
        self.scheduler.schedule_script(
            script,
            schedule_type="cron",
            cron_expression="0 0 * * 0"
        )
        
        scheduled = self.scheduler.scheduled_scripts["weekly_backup"]
        assert scheduled["cron_expression"] == "0 0 * * 0"
    
    def test_get_next_execution_time(self):
        """Test de calcul du prochain temps d'exécution"""
        script = DataPipelineScript(
            name="hourly_pipeline",
            description="Hourly pipeline",
            script_path="/fake/script.py"
        )
        
        self.scheduler.schedule_script(
            script,
            schedule_type="hourly"
        )
        
        next_time = self.scheduler.get_next_execution_time("hourly_pipeline")
        
        assert isinstance(next_time, datetime)
        assert next_time > datetime.now()
    
    @pytest.mark.asyncio
    async def test_start_scheduler(self):
        """Test de démarrage du planificateur"""
        with patch.object(self.scheduler, '_scheduler_loop') as mock_loop:
            mock_loop.return_value = None
            
            await self.scheduler.start()
            
            assert self.scheduler.is_running is True
    
    def test_list_scheduled_scripts(self):
        """Test de listage des scripts planifiés"""
        scripts = [
            DataPipelineScript("pipeline1", "Pipeline 1", "/path1"),
            MaintenanceScript("maintenance1", "Maintenance 1", "cleanup"),
            BackupScript("backup1", "Backup 1", ["/data"], "/backup")
        ]
        
        for i, script in enumerate(scripts):
            self.scheduler.schedule_script(
                script,
                schedule_type="daily",
                schedule_time=f"0{i}:00"
            )
        
        scheduled_list = self.scheduler.list_scheduled_scripts()
        
        assert len(scheduled_list) == 3
        assert all(s["script_name"] in ["pipeline1", "maintenance1", "backup1"] 
                  for s in scheduled_list)


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires"""
    
    @pytest.mark.asyncio
    async def test_run_script_async_function(self):
        """Test de la fonction run_script_async"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Script completed",
                stderr=""
            )
            
            result = await run_script_async("/fake/script.py", ["--param", "value"])
            
            assert result["status"] == "success"
            assert result["stdout"] == "Script completed"
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_function(self):
        """Test de la fonction execute_pipeline"""
        pipeline_config = {
            "name": "test_pipeline",
            "steps": [
                {"name": "extract", "script": "/extract.py"},
                {"name": "transform", "script": "/transform.py"},
                {"name": "load", "script": "/load.py"}
            ]
        }
        
        with patch('app.ml_analytics.scripts.run_script_async') as mock_run:
            mock_run.return_value = {"status": "success"}
            
            result = await execute_pipeline(pipeline_config)
            
            assert result["status"] == "success"
            assert result["completed_steps"] == 3
            assert mock_run.call_count == 3
    
    @pytest.mark.asyncio
    async def test_deploy_model_function(self):
        """Test de la fonction deploy_model"""
        model_config = {
            "model_path": "/models/test_model.pkl",
            "deployment_target": "production",
            "replicas": 3
        }
        
        with patch('app.ml_analytics.scripts.ModelDeploymentScript') as mock_script:
            mock_instance = MagicMock()
            mock_script.return_value = mock_instance
            mock_instance.execute = AsyncMock(return_value={"status": "success"})
            
            result = await deploy_model(model_config)
            
            assert result["status"] == "success"
    
    def test_cleanup_old_data_function(self):
        """Test de la fonction cleanup_old_data"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Créer des fichiers anciens et récents
            old_file = Path(temp_dir) / "old_file.txt"
            recent_file = Path(temp_dir) / "recent_file.txt"
            
            old_file.touch()
            recent_file.touch()
            
            # Simuler les dates
            old_date = datetime.now() - timedelta(days=40)
            recent_date = datetime.now() - timedelta(days=5)
            
            os.utime(old_file, (old_date.timestamp(), old_date.timestamp()))
            os.utime(recent_file, (recent_date.timestamp(), recent_date.timestamp()))
            
            result = cleanup_old_data(temp_dir, retention_days=30)
            
            assert result["deleted_count"] >= 1
            assert not old_file.exists()
            assert recent_file.exists()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_backup_system_function(self):
        """Test de la fonction backup_system"""
        backup_config = {
            "source_paths": ["/data", "/models"],
            "destination": "/backups",
            "compression": True
        }
        
        with patch('app.ml_analytics.scripts.BackupScript') as mock_script:
            mock_instance = MagicMock()
            mock_script.return_value = mock_instance
            mock_instance.execute = AsyncMock(return_value={"status": "success"})
            
            result = await backup_system(backup_config)
            
            assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_migrate_data_function(self):
        """Test de la fonction migrate_data"""
        migration_config = {
            "source_db": "postgresql://old_db",
            "target_db": "postgresql://new_db",
            "tables": ["users", "tracks", "playlists"]
        }
        
        with patch('app.ml_analytics.scripts.MigrationScript') as mock_script:
            mock_instance = MagicMock()
            mock_script.return_value = mock_instance
            mock_instance.execute = AsyncMock(return_value={"status": "success"})
            
            result = await migrate_data(migration_config)
            
            assert result["status"] == "success"


# Fixtures pour les tests
@pytest.fixture
def sample_script():
    """Script de test"""
    return DataPipelineScript(
        name="test_script",
        description="Test script for unit tests",
        script_path="/fake/test_script.py"
    )


@pytest.fixture
def script_manager():
    """Gestionnaire de scripts de test"""
    return ScriptManager()


@pytest.fixture
def script_executor():
    """Exécuteur de scripts de test"""
    return ScriptExecutor()


@pytest.fixture
def script_scheduler():
    """Planificateur de scripts de test"""
    return ScriptScheduler()


@pytest.fixture
def temp_workspace():
    """Espace de travail temporaire"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Tests d'intégration
@pytest.mark.integration
class TestScriptsIntegration:
    """Tests d'intégration pour les scripts"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        """Test d'exécution complète de pipeline"""
        # Configuration des composants
        script_manager = ScriptManager()
        executor = ScriptExecutor()
        scheduler = ScriptScheduler()
        
        # Créer un pipeline complet
        extract_script = DataPipelineScript(
            name="extract_data",
            description="Extract Spotify data",
            script_path="/fake/extract.py"
        )
        
        transform_script = DataPipelineScript(
            name="transform_data",
            description="Transform extracted data",
            script_path="/fake/transform.py"
        )
        
        load_script = DataPipelineScript(
            name="load_data",
            description="Load transformed data",
            script_path="/fake/load.py"
        )
        
        # Enregistrer les scripts
        for script in [extract_script, transform_script, load_script]:
            script_manager.register_script(script)
        
        # Mock des exécutions
        for script in [extract_script, transform_script, load_script]:
            script.execute = AsyncMock(return_value={"status": "success"})
        
        # Exécuter le pipeline
        results = []
        for script_name in ["extract_data", "transform_data", "load_data"]:
            result = await script_manager.execute_script(script_name)
            results.append(result)
        
        # Vérifications
        assert all(r["status"] == "success" for r in results)
        assert len(script_manager.execution_history) == 3
    
    @pytest.mark.asyncio
    async def test_scheduled_backup_execution(self):
        """Test d'exécution de sauvegarde planifiée"""
        scheduler = ScriptScheduler()
        
        # Créer un script de sauvegarde
        backup_script = BackupScript(
            name="daily_backup",
            description="Daily system backup",
            source_paths=["/data"],
            backup_destination="/backups"
        )
        
        # Mock de l'exécution
        backup_script.execute = AsyncMock(return_value={"status": "success"})
        
        # Planifier pour exécution immédiate (test)
        scheduler.schedule_script(
            backup_script,
            schedule_type="immediate"
        )
        
        # Simuler l'exécution planifiée
        result = await backup_script.execute()
        
        assert result["status"] == "success"


# Tests de performance
@pytest.mark.performance
class TestScriptsPerformance:
    """Tests de performance pour les scripts"""
    
    @pytest.mark.asyncio
    async def test_concurrent_script_execution_performance(self):
        """Test de performance d'exécution concurrente"""
        import time
        
        executor = ScriptExecutor()
        
        # Créer plusieurs scripts
        scripts = []
        for i in range(20):
            script = DataPipelineScript(
                name=f"perf_test_{i}",
                description=f"Performance test {i}",
                script_path="/fake/script.py"
            )
            
            # Mock d'exécution rapide
            script.execute = AsyncMock(return_value={"status": "success"})
            scripts.append(script)
        
        start_time = time.time()
        
        # Exécuter tous les scripts de manière concurrente
        tasks = [
            executor.execute_script_async(script)
            for script in scripts
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Vérifications
        assert all(r["status"] == "success" for r in results)
        assert duration < 2.0  # Exécution rapide grâce à la concurrence
    
    def test_script_scheduling_performance(self):
        """Test de performance de planification"""
        import time
        
        scheduler = ScriptScheduler()
        
        start_time = time.time()
        
        # Planifier beaucoup de scripts
        for i in range(1000):
            script = DataPipelineScript(
                name=f"scheduled_script_{i}",
                description=f"Scheduled script {i}",
                script_path="/fake/script.py"
            )
            
            scheduler.schedule_script(
                script,
                schedule_type="daily",
                schedule_time=f"{i % 24:02d}:00"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Planification rapide
        assert duration < 1.0
        assert len(scheduler.scheduled_scripts) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
