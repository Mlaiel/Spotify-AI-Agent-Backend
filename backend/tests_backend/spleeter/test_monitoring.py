"""
Tests pour le module monitoring.py du système Spleeter
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import json
import pickle

from spleeter.monitoring import (
    PerformanceMetric, ProcessingStats, SystemHealth, MetricsCollector,
    PerformanceTimer, ResourceMonitor, get_global_collector,
    initialize_monitoring, shutdown_monitoring
)
from spleeter.exceptions import MonitoringError


class TestPerformanceMetric:
    """Tests pour la classe PerformanceMetric"""
    
    def test_performance_metric_creation(self):
        """Test de création d'une métrique"""
        metric = PerformanceMetric(
            name="processing_time",
            value=2.5,
            timestamp=datetime.now(),
            unit="seconds",
            tags={"model": "2stems", "gpu": "true"}
        )
        
        assert metric.name == "processing_time"
        assert metric.value == 2.5
        assert metric.unit == "seconds"
        assert metric.tags["model"] == "2stems"
        assert metric.tags["gpu"] == "true"
    
    def test_performance_metric_to_dict(self):
        """Test de conversion en dictionnaire"""
        timestamp = datetime.now()
        metric = PerformanceMetric(
            name="cache_hit_rate",
            value=85.5,
            timestamp=timestamp,
            unit="%",
            tags={"cache_type": "memory"}
        )
        
        metric_dict = metric.to_dict()
        
        assert metric_dict["name"] == "cache_hit_rate"
        assert metric_dict["value"] == 85.5
        assert metric_dict["timestamp"] == timestamp.isoformat()
        assert metric_dict["unit"] == "%"
        assert metric_dict["tags"]["cache_type"] == "memory"


class TestProcessingStats:
    """Tests pour la classe ProcessingStats"""
    
    def test_processing_stats_initialization(self):
        """Test d'initialisation des statistiques"""
        stats = ProcessingStats()
        
        assert stats.total_files == 0
        assert stats.successful_files == 0
        assert stats.failed_files == 0
        assert stats.total_duration_processed == 0.0
        assert stats.total_processing_time == 0.0
        assert stats.success_rate == 0.0
    
    def test_processing_stats_success_rate(self):
        """Test de calcul du taux de succès"""
        stats = ProcessingStats(
            total_files=10,
            successful_files=8,
            failed_files=2
        )
        
        assert stats.success_rate == 80.0
    
    def test_processing_stats_success_rate_no_files(self):
        """Test de taux de succès sans fichiers"""
        stats = ProcessingStats()
        assert stats.success_rate == 0.0
    
    def test_processing_stats_to_dict(self):
        """Test de conversion en dictionnaire"""
        stats = ProcessingStats(
            total_files=5,
            successful_files=4,
            failed_files=1,
            total_duration_processed=300.0,
            peak_memory_usage=1024.5
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["total_files"] == 5
        assert stats_dict["successful_files"] == 4
        assert stats_dict["failed_files"] == 1
        assert stats_dict["success_rate"] == 80.0
        assert stats_dict["total_duration_processed"] == 300.0
        assert stats_dict["peak_memory_usage"] == 1024.5


class TestSystemHealth:
    """Tests pour la classe SystemHealth"""
    
    def test_system_health_initialization(self):
        """Test d'initialisation de la santé système"""
        health = SystemHealth(
            cpu_usage=45.2,
            memory_usage=68.7,
            disk_usage=23.1,
            gpu_usage=12.5,
            load_average=1.8
        )
        
        assert health.cpu_usage == 45.2
        assert health.memory_usage == 68.7
        assert health.disk_usage == 23.1
        assert health.gpu_usage == 12.5
        assert health.load_average == 1.8
    
    def test_system_health_score_excellent(self):
        """Test de score de santé excellent"""
        health = SystemHealth(
            cpu_usage=10.0,
            memory_usage=20.0,
            disk_usage=15.0,
            gpu_usage=5.0
        )
        
        score = health.health_score
        assert score >= 80.0
        assert health.status == "excellent"
    
    def test_system_health_score_critical(self):
        """Test de score de santé critique"""
        health = SystemHealth(
            cpu_usage=95.0,
            memory_usage=90.0,
            disk_usage=85.0,
            gpu_usage=98.0
        )
        
        score = health.health_score
        assert score < 40.0
        assert health.status == "critical"
    
    def test_system_health_score_no_gpu(self):
        """Test de score sans GPU"""
        health = SystemHealth(
            cpu_usage=30.0,
            memory_usage=40.0,
            disk_usage=25.0
            # Pas de GPU
        )
        
        # Devrait calculer le score sans GPU
        score = health.health_score
        assert 0.0 <= score <= 100.0


class TestMetricsCollector:
    """Tests pour la classe MetricsCollector"""
    
    @pytest.fixture
    def collector(self):
        """Fixture pour créer un collecteur de métriques"""
        return MetricsCollector(
            buffer_size=100,
            retention_hours=1,
            enable_system_metrics=False  # Désactivé pour les tests
        )
    
    def test_metrics_collector_initialization(self, collector):
        """Test d'initialisation du collecteur"""
        assert collector.buffer_size == 100
        assert collector.retention_hours == 1
        assert len(collector.metrics_buffer) == 0
        assert len(collector.aggregated_metrics) == 0
    
    def test_record_metric_basic(self, collector):
        """Test d'enregistrement de métrique basique"""
        collector.record_metric("test_metric", 42.5, "units", {"tag": "value"})
        
        assert len(collector.metrics_buffer) == 1
        assert len(collector.aggregated_metrics["test_metric"]) == 1
        
        metric = collector.metrics_buffer[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.unit == "units"
        assert metric.tags["tag"] == "value"
    
    def test_record_multiple_metrics(self, collector):
        """Test d'enregistrement de plusieurs métriques"""
        for i in range(10):
            collector.record_metric(f"metric_{i}", i * 10, "count")
        
        assert len(collector.metrics_buffer) == 10
        assert len(collector.aggregated_metrics) == 10
    
    def test_record_processing_event_success(self, collector):
        """Test d'enregistrement d'événement de traitement réussi"""
        collector.record_processing_event(
            event_type="separation",
            audio_duration=120.0,
            processing_time=60.0,
            success=True,
            model_name="spleeter:2stems",
            file_size=10000000
        )
        
        stats = collector.processing_stats
        assert stats.total_files == 1
        assert stats.successful_files == 1
        assert stats.failed_files == 0
        assert stats.total_duration_processed == 120.0
        assert stats.total_processing_time == 60.0
        
        # Vérifier que des métriques ont été enregistrées
        assert len(collector.metrics_buffer) > 0
    
    def test_record_processing_event_failure(self, collector):
        """Test d'enregistrement d'événement de traitement échoué"""
        collector.record_processing_event(
            event_type="separation",
            success=False,
            model_name="spleeter:4stems"
        )
        
        stats = collector.processing_stats
        assert stats.total_files == 1
        assert stats.successful_files == 0
        assert stats.failed_files == 1
    
    def test_record_cache_event_hit(self, collector):
        """Test d'enregistrement d'événement de cache hit"""
        collector.record_cache_event(hit=True, cache_type="memory")
        
        assert collector._cache_stats['hits'] == 1
        assert collector._cache_stats['misses'] == 0
        assert collector.processing_stats.cache_hit_rate == 100.0
    
    def test_record_cache_event_miss(self, collector):
        """Test d'enregistrement d'événement de cache miss"""
        collector.record_cache_event(hit=False, cache_type="disk")
        
        assert collector._cache_stats['hits'] == 0
        assert collector._cache_stats['misses'] == 1
        assert collector.processing_stats.cache_hit_rate == 0.0
    
    def test_record_cache_event_mixed(self, collector):
        """Test d'enregistrement d'événements de cache mixtes"""
        # 3 hits, 1 miss
        for _ in range(3):
            collector.record_cache_event(hit=True)
        collector.record_cache_event(hit=False)
        
        assert collector.processing_stats.cache_hit_rate == 75.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_record_system_health(self, mock_disk, mock_memory, mock_cpu, collector):
        """Test d'enregistrement de santé système"""
        # Mock des valeurs système
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=30.0)
        
        health = SystemHealth(
            cpu_usage=45.0,
            memory_usage=60.0,
            disk_usage=30.0
        )
        
        collector.record_system_health(health)
        
        # Vérifier que les métriques ont été enregistrées
        metrics_names = [m.name for m in collector.metrics_buffer]
        assert "system_cpu_usage" in metrics_names
        assert "system_memory_usage" in metrics_names
        assert "system_disk_usage" in metrics_names
        assert "system_health_score" in metrics_names
    
    def test_get_stats_summary(self, collector):
        """Test de récupération du résumé de statistiques"""
        # Ajouter quelques métriques
        collector.record_metric("test_metric", 10.0)
        collector.record_metric("test_metric", 20.0)
        collector.record_processing_event("test", success=True)
        
        summary = collector.get_stats_summary()
        
        assert "processing_stats" in summary
        assert "recent_metrics" in summary
        assert "system_health" in summary
        assert "cache_stats" in summary
        assert "buffer_size" in summary
        assert "metrics_count" in summary
        
        # Vérifier quelques valeurs
        assert summary["processing_stats"]["total_files"] == 1
        assert summary["buffer_size"] > 0
    
    def test_get_metric_history(self, collector):
        """Test de récupération d'historique de métrique"""
        # Ajouter plusieurs valeurs de la même métrique
        for i in range(5):
            collector.record_metric("history_test", i * 10)
            time.sleep(0.01)  # Petit délai pour différencier les timestamps
        
        # Récupérer l'historique des 10 dernières minutes
        history = collector.get_metric_history("history_test", duration_minutes=10)
        
        assert len(history) == 5
        # Vérifier que c'est trié par timestamp
        timestamps = [m.timestamp for m in history]
        assert timestamps == sorted(timestamps)
        
        # Vérifier les valeurs
        values = [m.value for m in history]
        assert values == [0, 10, 20, 30, 40]
    
    def test_get_metric_history_time_filter(self, collector):
        """Test de filtrage par temps de l'historique"""
        # Ajouter une métrique ancienne (simulée)
        old_metric = PerformanceMetric(
            name="old_metric",
            value=100,
            timestamp=datetime.now() - timedelta(hours=2)
        )
        collector.metrics_buffer.append(old_metric)
        
        # Ajouter une métrique récente
        collector.record_metric("old_metric", 200)
        
        # Récupérer seulement les métriques de la dernière heure
        recent_history = collector.get_metric_history("old_metric", duration_minutes=60)
        
        # Ne devrait contenir que la métrique récente
        assert len(recent_history) == 1
        assert recent_history[0].value == 200
    
    def test_cleanup_old_metrics(self, collector):
        """Test de nettoyage des anciennes métriques"""
        # Créer un collecteur avec très courte rétention
        short_collector = MetricsCollector(retention_hours=0.001)  # ~3.6 secondes
        
        # Ajouter des métriques anciennes (simulées)
        old_metric = PerformanceMetric(
            name="old",
            value=1,
            timestamp=datetime.now() - timedelta(hours=1)
        )
        short_collector.metrics_buffer.append(old_metric)
        
        # Ajouter une métrique récente
        short_collector.record_metric("recent", 2)
        
        # Le nettoyage devrait être automatique lors de l'ajout
        assert len(short_collector.metrics_buffer) == 1
        assert short_collector.metrics_buffer[0].name == "recent"
    
    def test_add_callbacks(self, collector):
        """Test d'ajout de callbacks"""
        alert_called = []
        metric_called = []
        
        def alert_callback(alert):
            alert_called.append(alert)
        
        def metric_callback(metric):
            metric_called.append(metric)
        
        collector.add_alert_callback(alert_callback)
        collector.add_metric_callback("test_metric", metric_callback)
        
        # Déclencher une métrique
        collector.record_metric("test_metric", 42)
        
        # Le callback de métrique devrait avoir été appelé
        assert len(metric_called) == 1
        assert metric_called[0].name == "test_metric"
        assert metric_called[0].value == 42
    
    @patch('tempfile.NamedTemporaryFile')
    def test_export_metrics_json(self, mock_temp_file, collector):
        """Test d'export de métriques en JSON"""
        # Préparer des données de test
        collector.record_metric("export_test", 123.45)
        
        # Mock du fichier temporaire
        mock_file = Mock()
        mock_temp_file.return_value.__enter__.return_value = mock_file
        
        with tempfile.NamedTemporaryFile(suffix=".json", mode='w') as tmp_file:
            collector.export_metrics(tmp_file.name, format="json")
        
        # Vérifier qu'aucune exception n'a été levée
        # En réalité, on devrait tester le contenu du fichier
    
    def test_start_stop_system_monitoring(self, collector):
        """Test de démarrage/arrêt de surveillance système"""
        # Le collecteur de test a la surveillance désactivée
        collector.enable_system_metrics = True
        
        assert collector._system_thread is None
        
        collector.start_system_monitoring(interval=0.1)
        assert collector._system_thread is not None
        assert collector._system_thread.is_alive()
        
        collector.stop_system_monitoring()
        # Attendre un peu pour que le thread se termine
        time.sleep(0.2)
        assert not collector._system_thread.is_alive()


class TestPerformanceTimer:
    """Tests pour la classe PerformanceTimer"""
    
    def test_performance_timer_basic(self):
        """Test basique du timer de performance"""
        timer = PerformanceTimer("test_operation", auto_record=False)
        
        timer.start()
        time.sleep(0.1)  # Attendre 100ms
        duration = timer.stop()
        
        assert duration >= 0.1
        assert duration < 0.2  # Devrait être proche de 100ms
        assert timer.duration == duration
    
    def test_performance_timer_context_manager(self):
        """Test du timer comme context manager"""
        with PerformanceTimer("context_test", auto_record=False) as timer:
            time.sleep(0.05)  # Attendre 50ms
        
        assert timer.duration >= 0.05
        assert timer.duration < 0.1
    
    def test_performance_timer_not_started_error(self):
        """Test d'erreur si timer non démarré"""
        timer = PerformanceTimer("error_test", auto_record=False)
        
        with pytest.raises(RuntimeError):
            timer.stop()
    
    @patch('spleeter.monitoring.MetricsCollector')
    def test_performance_timer_auto_record(self, mock_collector_class):
        """Test d'enregistrement automatique"""
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        
        timer = PerformanceTimer("auto_test", collector=mock_collector, auto_record=True)
        
        timer.start()
        time.sleep(0.01)
        timer.stop()
        
        # Vérifier que record_metric a été appelé
        mock_collector.record_metric.assert_called_once()
        call_args = mock_collector.record_metric.call_args
        assert call_args[0][0] == "timer_auto_test"  # Nom de la métrique
        assert call_args[0][2] == "seconds"  # Unité
    
    def test_performance_timer_exception_handling(self):
        """Test de gestion d'exception dans le timer"""
        mock_collector = Mock()
        
        with pytest.raises(ValueError):
            with PerformanceTimer("exception_test", collector=mock_collector):
                raise ValueError("Test exception")
        
        # Vérifier qu'une métrique d'erreur a été enregistrée
        mock_collector.record_metric.assert_called()
        # Chercher l'appel avec _error dans le nom
        error_calls = [call for call in mock_collector.record_metric.call_args_list 
                      if "_error" in call[0][0]]
        assert len(error_calls) > 0


class TestResourceMonitor:
    """Tests pour la classe ResourceMonitor"""
    
    @pytest.fixture
    def resource_monitor(self):
        """Fixture pour créer un moniteur de ressources"""
        return ResourceMonitor(
            memory_threshold_mb=1024,
            gpu_threshold_percent=80
        )
    
    def test_resource_monitor_initialization(self, resource_monitor):
        """Test d'initialisation du moniteur"""
        assert resource_monitor.memory_threshold_mb == 1024
        assert resource_monitor.gpu_threshold_percent == 80
        assert not resource_monitor.monitoring
    
    @patch('psutil.Process')
    def test_resource_monitor_context_manager(self, mock_process, resource_monitor):
        """Test du moniteur comme context manager"""
        # Mock du processus
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = Mock(rss=500 * 1024 * 1024)  # 500MB
        mock_process.return_value = mock_process_instance
        
        with patch.object(resource_monitor, '_get_gpu_usage', return_value=(50.0, 60.0)):
            with resource_monitor.monitor_operation("test_op"):
                time.sleep(0.1)  # Opération simulée
        
        assert not resource_monitor.monitoring
    
    def test_resource_monitor_start_stop(self, resource_monitor):
        """Test de démarrage/arrêt manuel"""
        assert not resource_monitor.monitoring
        
        resource_monitor.start_monitoring("manual_test")
        assert resource_monitor.monitoring
        assert resource_monitor.monitor_thread is not None
        
        resource_monitor.stop_monitoring()
        time.sleep(0.1)  # Attendre que le thread se termine
        assert not resource_monitor.monitoring
    
    @patch('psutil.Process')
    def test_resource_monitor_threshold_warning(self, mock_process, resource_monitor):
        """Test d'avertissement de seuil dépassé"""
        # Simuler utilisation mémoire élevée
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = Mock(rss=2048 * 1024 * 1024)  # 2GB
        mock_process.return_value = mock_process_instance
        
        with patch.object(resource_monitor, '_get_gpu_usage', return_value=(95.0, 80.0)):
            with patch('spleeter.monitoring.logger') as mock_logger:
                resource_monitor.start_monitoring("threshold_test")
                time.sleep(0.6)  # Attendre quelques cycles de monitoring
                resource_monitor.stop_monitoring()
                
                # Vérifier qu'un avertissement a été loggé
                mock_logger.warning.assert_called()


class TestGlobalFunctions:
    """Tests pour les fonctions globales de monitoring"""
    
    def test_initialize_monitoring(self):
        """Test d'initialisation du monitoring global"""
        collector = initialize_monitoring(
            buffer_size=500,
            retention_hours=12,
            enable_system_metrics=False
        )
        
        assert collector is not None
        assert collector.buffer_size == 500
        assert collector.retention_hours == 12
        
        # Nettoyage
        shutdown_monitoring()
    
    def test_get_global_collector(self):
        """Test de récupération du collecteur global"""
        # Premier appel devrait créer l'instance
        collector1 = get_global_collector()
        assert collector1 is not None
        
        # Deuxième appel devrait retourner la même instance
        collector2 = get_global_collector()
        assert collector1 is collector2
        
        # Nettoyage
        shutdown_monitoring()
    
    def test_shutdown_monitoring(self):
        """Test d'arrêt du monitoring global"""
        # Initialiser
        get_global_collector()
        
        # Arrêter
        shutdown_monitoring()
        
        # Un nouvel appel devrait créer une nouvelle instance
        new_collector = get_global_collector()
        assert new_collector is not None
        
        # Nettoyage final
        shutdown_monitoring()
    
    @patch('spleeter.monitoring.get_global_collector')
    def test_record_metric_global(self, mock_get_collector):
        """Test d'enregistrement de métrique global"""
        from spleeter.monitoring import record_metric
        
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        
        record_metric("global_test", 42.0, "units", {"tag": "value"})
        
        mock_collector.record_metric.assert_called_once_with(
            "global_test", 42.0, "units", {"tag": "value"}
        )
    
    @patch('spleeter.monitoring.get_global_collector')
    def test_create_timer_global(self, mock_get_collector):
        """Test de création de timer global"""
        from spleeter.monitoring import create_timer
        
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        
        timer = create_timer("global_timer", auto_record=True)
        
        assert timer.name == "global_timer"
        assert timer.collector is mock_collector
        assert timer.auto_record is True
