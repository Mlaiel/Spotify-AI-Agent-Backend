# 🧪 ML Analytics Monitoring Tests
# ================================
# 
# Tests ultra-avancés pour le système de monitoring
# Enterprise monitoring system testing
#
# 🎖️ Implementation par l'équipe d'experts:
# ✅ Architecte Microservices + DBA & Data Engineer + DevOps
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ================================

"""
🔍 Monitoring System Test Suite
===============================

Comprehensive testing for monitoring system:
- Real-time performance monitoring
- Model drift detection
- Data quality monitoring
- Alert management and notifications
- Health checks and diagnostics
- Metrics collection and aggregation
"""

import pytest
import asyncio
import time
import json
import threading
from unittest.mock import patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import psutil

# Import modules to test
from app.ml_analytics.monitoring import (
    AlertSeverity, MetricType,
    Alert, HealthCheck, Metric,
    HealthMonitor, AlertManager, MetricsCollector,
    ModelDriftDetector, DataQualityMonitor, PerformanceMonitor,
    PrometheusExporter, NotificationService,
    create_alert, send_notification, calculate_drift_score,
    monitor_system_health, collect_performance_metrics
)


class TestAlertSeverity:
    """Tests pour l'énumération AlertSeverity"""
    
    def test_alert_severity_values(self):
        """Test des valeurs de sévérité d'alerte"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_severity_ordering(self):
        """Test de l'ordre des sévérités"""
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL
        ]
        
        # Vérifier l'ordre croissant de sévérité
        severity_values = [s.value for s in severities]
        assert severity_values == ["info", "warning", "error", "critical"]


class TestMetricType:
    """Tests pour l'énumération MetricType"""
    
    def test_metric_type_values(self):
        """Test des types de métriques"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"


class TestAlert:
    """Tests pour la classe Alert"""
    
    def test_alert_creation(self):
        """Test de création d'alerte"""
        alert = Alert(
            id="alert_001",
            severity=AlertSeverity.WARNING,
            title="High CPU Usage",
            message="CPU usage exceeded 80%",
            source="system_monitor"
        )
        
        assert alert.id == "alert_001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "High CPU Usage"
        assert alert.message == "CPU usage exceeded 80%"
        assert alert.source == "system_monitor"
        assert alert.resolved is False
        assert alert.resolved_at is None
        assert isinstance(alert.timestamp, datetime)
    
    def test_alert_resolution(self):
        """Test de résolution d'alerte"""
        alert = Alert(
            id="alert_002",
            severity=AlertSeverity.ERROR,
            title="Database Connection Lost",
            message="Cannot connect to database",
            source="db_monitor"
        )
        
        # Initialement non résolue
        assert alert.resolved is False
        assert alert.resolved_at is None
        
        # Résoudre l'alerte
        alert.resolve()
        
        assert alert.resolved is True
        assert isinstance(alert.resolved_at, datetime)
        assert alert.resolved_at > alert.timestamp
    
    def test_alert_to_dict(self):
        """Test de conversion d'alerte en dictionnaire"""
        metadata = {"cpu_usage": 85.5, "threshold": 80.0}
        
        alert = Alert(
            id="alert_003",
            severity=AlertSeverity.CRITICAL,
            title="Critical System Error",
            message="System overload detected",
            source="performance_monitor",
            metadata=metadata
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["id"] == "alert_003"
        assert alert_dict["severity"] == "critical"
        assert alert_dict["title"] == "Critical System Error"
        assert alert_dict["message"] == "System overload detected"
        assert alert_dict["source"] == "performance_monitor"
        assert alert_dict["resolved"] is False
        assert alert_dict["metadata"] == metadata
        assert "timestamp" in alert_dict
        assert alert_dict["resolved_at"] is None


class TestHealthCheck:
    """Tests pour la classe HealthCheck"""
    
    def test_health_check_creation(self):
        """Test de création de contrôle de santé"""
        metadata = {"response_time": 150, "status_code": 200}
        
        health_check = HealthCheck(
            name="api_health",
            status="healthy",
            message="API responding normally",
            duration_ms=45.2,
            metadata=metadata
        )
        
        assert health_check.name == "api_health"
        assert health_check.status == "healthy"
        assert health_check.message == "API responding normally"
        assert health_check.duration_ms == 45.2
        assert health_check.metadata == metadata
        assert isinstance(health_check.timestamp, datetime)
    
    def test_health_check_to_dict(self):
        """Test de conversion en dictionnaire"""
        health_check = HealthCheck(
            name="database_health",
            status="unhealthy",
            message="Connection timeout",
            duration_ms=5000.0
        )
        
        health_dict = health_check.to_dict()
        
        assert health_dict["name"] == "database_health"
        assert health_dict["status"] == "unhealthy"
        assert health_dict["message"] == "Connection timeout"
        assert health_dict["duration_ms"] == 5000.0
        assert "timestamp" in health_dict


class TestMetric:
    """Tests pour la classe Metric"""
    
    def test_metric_creation(self):
        """Test de création de métrique"""
        labels = {"service": "ml_analytics", "environment": "production"}
        
        metric = Metric(
            name="request_count",
            type=MetricType.COUNTER,
            value=42.0,
            labels=labels,
            help_text="Total number of requests"
        )
        
        assert metric.name == "request_count"
        assert metric.type == MetricType.COUNTER
        assert metric.value == 42.0
        assert metric.labels == labels
        assert metric.help_text == "Total number of requests"
        assert isinstance(metric.timestamp, datetime)
    
    def test_metric_increment(self):
        """Test d'incrémentation de métrique"""
        metric = Metric(
            name="error_count",
            type=MetricType.COUNTER,
            value=5.0
        )
        
        metric.increment(3.0)
        assert metric.value == 8.0
        
        metric.increment()  # Default increment by 1
        assert metric.value == 9.0
    
    def test_metric_update(self):
        """Test de mise à jour de métrique"""
        metric = Metric(
            name="cpu_usage",
            type=MetricType.GAUGE,
            value=45.5
        )
        
        metric.update(78.2)
        assert metric.value == 78.2
        assert isinstance(metric.timestamp, datetime)


class TestHealthMonitor:
    """Tests pour le moniteur de santé"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.health_monitor = HealthMonitor()
    
    def test_health_monitor_creation(self):
        """Test de création du moniteur"""
        assert isinstance(self.health_monitor, HealthMonitor)
        assert len(self.health_monitor.health_checks) == 0
        assert self.health_monitor.is_running is False
    
    @pytest.mark.asyncio
    async def test_check_system_health(self):
        """Test de contrôle de santé système"""
        health_status = await self.health_monitor.check_system_health()
        
        assert "healthy" in health_status
        assert "components" in health_status
        assert "timestamp" in health_status
        assert "overall_status" in health_status
        
        # Vérifier les composants de base
        components = health_status["components"]
        assert "cpu" in components
        assert "memory" in components
        assert "disk" in components
    
    @pytest.mark.asyncio
    async def test_check_database_health(self):
        """Test de contrôle de santé de base de données"""
        with patch('app.ml_analytics.monitoring.asyncpg.connect') as mock_connect:
            mock_conn = AsyncMock()
            mock_connect.return_value = mock_conn
            mock_conn.execute.return_value = None
            
            health_check = await self.health_monitor.check_database_health()
            
            assert health_check.name == "database"
            assert health_check.status in ["healthy", "unhealthy"]
            assert isinstance(health_check.duration_ms, float)
    
    @pytest.mark.asyncio
    async def test_check_cache_health(self):
        """Test de contrôle de santé du cache"""
        with patch('app.ml_analytics.monitoring.aioredis.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            health_check = await self.health_monitor.check_cache_health()
            
            assert health_check.name == "cache"
            assert health_check.status == "healthy"
    
    @pytest.mark.asyncio
    async def test_check_ml_models_health(self):
        """Test de contrôle de santé des modèles ML"""
        with patch.object(self.health_monitor, '_check_model_availability') as mock_check:
            mock_check.return_value = {"loaded": True, "last_updated": datetime.now()}
            
            health_check = await self.health_monitor.check_ml_models_health()
            
            assert health_check.name == "ml_models"
            assert health_check.status in ["healthy", "unhealthy"]
    
    def test_add_custom_health_check(self):
        """Test d'ajout de contrôle de santé personnalisé"""
        def custom_check():
            return HealthCheck(
                name="custom_service",
                status="healthy",
                message="Custom service is running"
            )
        
        self.health_monitor.add_custom_health_check("custom", custom_check)
        
        assert "custom" in self.health_monitor.custom_checks
        assert self.health_monitor.custom_checks["custom"] == custom_check
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self):
        """Test de démarrage du monitoring"""
        with patch.object(self.health_monitor, '_monitoring_loop') as mock_loop:
            mock_loop.return_value = None
            
            await self.health_monitor.start_monitoring(interval=1)
            
            assert self.health_monitor.is_running is True
            assert self.health_monitor.check_interval == 1
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self):
        """Test d'arrêt du monitoring"""
        # Démarrer le monitoring
        self.health_monitor.is_running = True
        
        await self.health_monitor.stop_monitoring()
        
        assert self.health_monitor.is_running is False


class TestAlertManager:
    """Tests pour le gestionnaire d'alertes"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.alert_manager = AlertManager()
    
    def test_alert_manager_creation(self):
        """Test de création du gestionnaire d'alertes"""
        assert isinstance(self.alert_manager, AlertManager)
        assert len(self.alert_manager.active_alerts) == 0
        assert len(self.alert_manager.resolved_alerts) == 0
    
    def test_create_alert(self):
        """Test de création d'alerte"""
        alert = self.alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="High Memory Usage",
            message="Memory usage exceeded threshold",
            source="memory_monitor",
            metadata={"usage": 85.5, "threshold": 80.0}
        )
        
        assert isinstance(alert, Alert)
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "High Memory Usage"
        assert alert.id in self.alert_manager.active_alerts
    
    def test_resolve_alert(self):
        """Test de résolution d'alerte"""
        alert = self.alert_manager.create_alert(
            severity=AlertSeverity.ERROR,
            title="Service Down",
            message="API service is not responding",
            source="api_monitor"
        )
        
        alert_id = alert.id
        
        # Résoudre l'alerte
        resolved = self.alert_manager.resolve_alert(alert_id)
        
        assert resolved is True
        assert alert_id not in self.alert_manager.active_alerts
        assert alert_id in self.alert_manager.resolved_alerts
        assert self.alert_manager.resolved_alerts[alert_id].resolved is True
    
    def test_get_alerts_by_severity(self):
        """Test de récupération d'alertes par sévérité"""
        # Créer des alertes de différentes sévérités
        self.alert_manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Info Alert",
            message="Information message",
            source="info_monitor"
        )
        
        critical_alert = self.alert_manager.create_alert(
            severity=AlertSeverity.CRITICAL,
            title="Critical Alert",
            message="Critical system failure",
            source="critical_monitor"
        )
        
        # Récupérer seulement les alertes critiques
        critical_alerts = self.alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
        
        assert len(critical_alerts) == 1
        assert critical_alerts[0].id == critical_alert.id
    
    def test_get_alerts_by_source(self):
        """Test de récupération d'alertes par source"""
        source = "database_monitor"
        
        for i in range(3):
            self.alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                title=f"DB Alert {i}",
                message=f"Database issue {i}",
                source=source
            )
        
        # Créer une alerte d'une autre source
        self.alert_manager.create_alert(
            severity=AlertSeverity.INFO,
            title="Other Alert",
            message="Different source",
            source="other_monitor"
        )
        
        db_alerts = self.alert_manager.get_alerts_by_source(source)
        
        assert len(db_alerts) == 3
        assert all(alert.source == source for alert in db_alerts)
    
    @pytest.mark.asyncio
    async def test_send_alert_notification(self):
        """Test d'envoi de notification d'alerte"""
        with patch.object(self.alert_manager, 'notification_service') as mock_service:
            mock_service.send_notification = AsyncMock()
            
            alert = self.alert_manager.create_alert(
                severity=AlertSeverity.CRITICAL,
                title="Critical System Error",
                message="System is down",
                source="system_monitor"
            )
            
            await self.alert_manager.send_notification(alert)
            
            mock_service.send_notification.assert_called_once_with(alert)
    
    def test_alert_aggregation(self):
        """Test d'agrégation d'alertes"""
        # Créer plusieurs alertes similaires
        for i in range(5):
            self.alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                title="High CPU",
                message=f"CPU spike {i}",
                source="cpu_monitor"
            )
        
        # Agréger les alertes similaires
        aggregated = self.alert_manager.aggregate_similar_alerts(
            time_window=timedelta(minutes=5)
        )
        
        assert len(aggregated) < 5  # Moins d'alertes après agrégation
        assert any("High CPU" in alert.title for alert in aggregated)


class TestMetricsCollector:
    """Tests pour le collecteur de métriques"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.metrics_collector = MetricsCollector()
    
    def test_metrics_collector_creation(self):
        """Test de création du collecteur"""
        assert isinstance(self.metrics_collector, MetricsCollector)
        assert len(self.metrics_collector.metrics) == 0
    
    def test_counter_metric(self):
        """Test de métrique compteur"""
        # Créer et incrémenter un compteur
        self.metrics_collector.increment_counter(
            "api_requests_total",
            labels={"method": "GET", "endpoint": "/health"}
        )
        
        self.metrics_collector.increment_counter(
            "api_requests_total",
            value=5,
            labels={"method": "POST", "endpoint": "/predict"}
        )
        
        # Vérifier les métriques
        metrics = self.metrics_collector.get_metrics()
        request_metrics = [m for m in metrics if m.name == "api_requests_total"]
        
        assert len(request_metrics) == 2
        assert any(m.value == 1.0 for m in request_metrics)
        assert any(m.value == 5.0 for m in request_metrics)
    
    def test_gauge_metric(self):
        """Test de métrique jauge"""
        # Mettre à jour une jauge
        self.metrics_collector.set_gauge(
            "memory_usage_bytes",
            value=1073741824,  # 1GB
            labels={"component": "ml_engine"}
        )
        
        self.metrics_collector.set_gauge(
            "cpu_usage_percent",
            value=75.5
        )
        
        # Vérifier les métriques
        metrics = self.metrics_collector.get_metrics()
        
        memory_metric = next(m for m in metrics if m.name == "memory_usage_bytes")
        cpu_metric = next(m for m in metrics if m.name == "cpu_usage_percent")
        
        assert memory_metric.value == 1073741824
        assert cpu_metric.value == 75.5
    
    def test_histogram_metric(self):
        """Test de métrique histogramme"""
        # Enregistrer des observations d'histogramme
        response_times = [0.1, 0.15, 0.25, 0.3, 0.45, 0.6, 1.2]
        
        for rt in response_times:
            self.metrics_collector.observe_histogram(
                "api_response_time_seconds",
                value=rt,
                labels={"endpoint": "/predict"}
            )
        
        # Vérifier l'histogramme
        histograms = self.metrics_collector.get_histograms()
        
        assert "api_response_time_seconds" in histograms
        histogram = histograms["api_response_time_seconds"]
        
        assert histogram["count"] == len(response_times)
        assert histogram["sum"] == sum(response_times)
        assert len(histogram["buckets"]) > 0
    
    def test_timer_metric(self):
        """Test de métrique timer"""
        import time
        
        # Utiliser le timer context manager
        with self.metrics_collector.timer(
            "operation_duration_seconds",
            labels={"operation": "model_inference"}
        ):
            time.sleep(0.1)  # Simuler une opération
        
        # Vérifier que la durée a été enregistrée
        timers = self.metrics_collector.get_timers()
        
        assert "operation_duration_seconds" in timers
        duration = timers["operation_duration_seconds"]["last_duration"]
        assert duration >= 0.1
    
    def test_metrics_export(self):
        """Test d'export de métriques"""
        # Créer quelques métriques
        self.metrics_collector.increment_counter("test_counter")
        self.metrics_collector.set_gauge("test_gauge", 42.0)
        
        # Exporter au format Prometheus
        prometheus_format = self.metrics_collector.export_prometheus_format()
        
        assert "test_counter" in prometheus_format
        assert "test_gauge" in prometheus_format
        assert "42.0" in prometheus_format
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test de performance de collection de métriques"""
        import time
        
        start_time = time.time()
        
        # Collecter beaucoup de métriques rapidement
        for i in range(1000):
            self.metrics_collector.increment_counter(
                "performance_test",
                labels={"iteration": str(i % 10)}
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Devrait être très rapide
        assert duration < 1.0
        assert len(self.metrics_collector.get_metrics()) == 10  # 10 labels uniques


class TestModelDriftDetector:
    """Tests pour le détecteur de dérive de modèle"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.drift_detector = ModelDriftDetector()
    
    def test_drift_detector_creation(self):
        """Test de création du détecteur"""
        assert isinstance(self.drift_detector, ModelDriftDetector)
        assert len(self.drift_detector.baseline_data) == 0
    
    def test_set_baseline(self):
        """Test d'établissement de ligne de base"""
        # Données de référence
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.exponential(1, 1000)
        })
        
        self.drift_detector.set_baseline(baseline_data)
        
        assert len(self.drift_detector.baseline_data) == 1000
        assert 'feature1' in self.drift_detector.baseline_stats
        assert 'feature2' in self.drift_detector.baseline_stats
        assert 'feature3' in self.drift_detector.baseline_stats
    
    def test_detect_drift_no_drift(self):
        """Test de détection sans dérive"""
        # Données de référence
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000)
        })
        
        self.drift_detector.set_baseline(baseline_data)
        
        # Nouvelles données similaires
        new_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 500),
            'feature2': np.random.normal(5, 2, 500)
        })
        
        drift_score = self.drift_detector.detect_drift(new_data)
        
        assert drift_score < 0.1  # Pas de dérive significative
    
    def test_detect_drift_with_drift(self):
        """Test de détection avec dérive"""
        # Données de référence
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000)
        })
        
        self.drift_detector.set_baseline(baseline_data)
        
        # Nouvelles données avec dérive
        new_data = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 500),  # Moyenne décalée
            'feature2': np.random.normal(8, 3, 500)   # Moyenne et variance différentes
        })
        
        drift_score = self.drift_detector.detect_drift(new_data)
        
        assert drift_score > 0.5  # Dérive significative détectée
    
    def test_kolmogorov_smirnov_test(self):
        """Test du test de Kolmogorov-Smirnov"""
        # Distributions identiques
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 1, 1000)
        
        ks_stat, p_value = self.drift_detector._kolmogorov_smirnov_test(data1, data2)
        
        assert p_value > 0.05  # Pas de différence significative
        
        # Distributions différentes
        data3 = np.random.normal(2, 1, 1000)
        
        ks_stat, p_value = self.drift_detector._kolmogorov_smirnov_test(data1, data3)
        
        assert p_value < 0.05  # Différence significative
    
    def test_population_stability_index(self):
        """Test de l'indice de stabilité de population"""
        # Distribution de référence
        baseline = np.random.normal(0, 1, 10000)
        
        # Distribution similaire
        current = np.random.normal(0, 1, 5000)
        
        psi = self.drift_detector._calculate_psi(baseline, current)
        
        assert psi < 0.1  # PSI faible = pas de dérive
        
        # Distribution différente
        current_different = np.random.normal(2, 1, 5000)
        
        psi_different = self.drift_detector._calculate_psi(baseline, current_different)
        
        assert psi_different > 0.2  # PSI élevé = dérive détectée
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self):
        """Test de monitoring continu"""
        # Établir la ligne de base
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000)
        })
        
        self.drift_detector.set_baseline(baseline_data)
        
        # Démarrer le monitoring continu
        with patch.object(self.drift_detector, '_monitoring_loop') as mock_loop:
            mock_loop.return_value = None
            
            await self.drift_detector.start_continuous_monitoring(
                interval=60  # Vérifier chaque minute
            )
            
            assert self.drift_detector.is_monitoring is True


class TestDataQualityMonitor:
    """Tests pour le moniteur de qualité des données"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.quality_monitor = DataQualityMonitor()
    
    def test_quality_monitor_creation(self):
        """Test de création du moniteur"""
        assert isinstance(self.quality_monitor, DataQualityMonitor)
    
    def test_check_data_completeness(self):
        """Test de vérification de complétude des données"""
        # Données avec valeurs manquantes
        data = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'feature2': [1, None, 3, None, 5],
            'feature3': [1, 2, 3, 4, 5]
        })
        
        completeness = self.quality_monitor.check_completeness(data)
        
        assert completeness['feature1'] == 0.8  # 4/5 = 80%
        assert completeness['feature2'] == 0.6  # 3/5 = 60%
        assert completeness['feature3'] == 1.0  # 5/5 = 100%
        assert completeness['overall'] == 0.8   # 12/15 = 80%
    
    def test_check_data_consistency(self):
        """Test de vérification de cohérence des données"""
        # Données avec incohérences
        data = pd.DataFrame({
            'age': [25, 30, -5, 150, 40],  # Âges invalides
            'score': [0.8, 1.2, 0.5, 0.9, -0.1]  # Scores hors limite
        })
        
        consistency_rules = {
            'age': {'min': 0, 'max': 120},
            'score': {'min': 0.0, 'max': 1.0}
        }
        
        consistency = self.quality_monitor.check_consistency(data, consistency_rules)
        
        assert consistency['age']['violations'] == 2  # -5 et 150
        assert consistency['score']['violations'] == 2  # 1.2 et -0.1
    
    def test_check_data_uniqueness(self):
        """Test de vérification d'unicité des données"""
        # Données avec doublons
        data = pd.DataFrame({
            'id': [1, 2, 3, 2, 5],  # Doublon : 2
            'name': ['A', 'B', 'C', 'D', 'B']  # Doublon : B
        })
        
        uniqueness = self.quality_monitor.check_uniqueness(data, ['id'])
        
        assert uniqueness['duplicate_count'] == 1
        assert uniqueness['unique_ratio'] == 0.8  # 4/5 = 80%
    
    def test_detect_outliers(self):
        """Test de détection d'outliers"""
        # Données avec outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = [10, -10, 15]  # Outliers évidents
        
        data = pd.DataFrame({
            'values': np.concatenate([normal_data, outliers])
        })
        
        outlier_info = self.quality_monitor.detect_outliers(
            data['values'],
            method='iqr'
        )
        
        assert len(outlier_info['outlier_indices']) >= 3
        assert outlier_info['outlier_ratio'] > 0.02
    
    def test_calculate_quality_score(self):
        """Test de calcul de score de qualité"""
        # Données de bonne qualité
        good_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        quality_score = self.quality_monitor.calculate_quality_score(good_data)
        
        assert quality_score > 0.9  # Bonne qualité
        
        # Données de mauvaise qualité
        bad_data = pd.DataFrame({
            'feature1': [1, None, None, None, None],
            'feature2': [0.1, None, None, None, None]
        })
        
        quality_score_bad = self.quality_monitor.calculate_quality_score(bad_data)
        
        assert quality_score_bad < 0.5  # Mauvaise qualité


class TestPerformanceMonitor:
    """Tests pour le moniteur de performance"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.performance_monitor = PerformanceMonitor()
    
    def test_performance_monitor_creation(self):
        """Test de création du moniteur"""
        assert isinstance(self.performance_monitor, PerformanceMonitor)
    
    def test_track_response_time(self):
        """Test de suivi du temps de réponse"""
        import time
        
        # Simuler une opération avec timer
        with self.performance_monitor.track_operation("api_call"):
            time.sleep(0.1)
        
        # Vérifier que le temps a été enregistré
        stats = self.performance_monitor.get_operation_stats("api_call")
        
        assert stats['count'] == 1
        assert stats['avg_duration'] >= 0.1
        assert stats['min_duration'] >= 0.1
        assert stats['max_duration'] >= 0.1
    
    def test_track_throughput(self):
        """Test de suivi du débit"""
        # Simuler plusieurs opérations
        for i in range(10):
            self.performance_monitor.record_operation_completed("data_processing")
        
        # Calculer le débit
        throughput = self.performance_monitor.get_throughput(
            "data_processing",
            time_window=timedelta(seconds=60)
        )
        
        assert throughput >= 0
    
    def test_resource_usage_monitoring(self):
        """Test de monitoring d'utilisation des ressources"""
        # Simuler l'usage des ressources
        resource_usage = self.performance_monitor.collect_resource_usage()
        
        assert 'cpu_percent' in resource_usage
        assert 'memory_percent' in resource_usage
        assert 'disk_usage' in resource_usage
        assert 'network_io' in resource_usage
        
        # Valeurs dans les bonnes plages
        assert 0 <= resource_usage['cpu_percent'] <= 100
        assert 0 <= resource_usage['memory_percent'] <= 100
    
    def test_performance_baseline(self):
        """Test d'établissement de ligne de base de performance"""
        # Enregistrer des performances de référence
        baseline_metrics = {
            'avg_response_time': 0.150,
            'throughput': 1000,
            'error_rate': 0.01
        }
        
        self.performance_monitor.set_performance_baseline(baseline_metrics)
        
        assert self.performance_monitor.baseline == baseline_metrics
    
    def test_performance_regression_detection(self):
        """Test de détection de régression de performance"""
        # Établir une ligne de base
        baseline = {
            'avg_response_time': 0.100,
            'throughput': 1000,
            'error_rate': 0.01
        }
        
        self.performance_monitor.set_performance_baseline(baseline)
        
        # Performances actuelles dégradées
        current_metrics = {
            'avg_response_time': 0.300,  # 3x plus lent
            'throughput': 500,           # 50% de débit en moins
            'error_rate': 0.05           # 5x plus d'erreurs
        }
        
        regression = self.performance_monitor.detect_performance_regression(
            current_metrics
        )
        
        assert regression['has_regression'] is True
        assert 'response_time' in regression['degraded_metrics']
        assert 'throughput' in regression['degraded_metrics']
        assert 'error_rate' in regression['degraded_metrics']


class TestPrometheusExporter:
    """Tests pour l'exporteur Prometheus"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.prometheus_exporter = PrometheusExporter()
    
    def test_prometheus_exporter_creation(self):
        """Test de création de l'exporteur"""
        assert isinstance(self.prometheus_exporter, PrometheusExporter)
    
    def test_register_metric(self):
        """Test d'enregistrement de métrique"""
        # Enregistrer une métrique compteur
        counter = self.prometheus_exporter.register_counter(
            name="test_requests_total",
            documentation="Total test requests",
            labels=["method", "endpoint"]
        )
        
        assert counter is not None
        assert "test_requests_total" in self.prometheus_exporter._metrics
    
    def test_export_metrics(self):
        """Test d'export de métriques"""
        # Créer et utiliser quelques métriques
        counter = self.prometheus_exporter.register_counter(
            "http_requests_total",
            "Total HTTP requests"
        )
        counter.inc()
        
        gauge = self.prometheus_exporter.register_gauge(
            "memory_usage_bytes",
            "Memory usage in bytes"
        )
        gauge.set(1073741824)  # 1GB
        
        # Exporter les métriques
        metrics_output = self.prometheus_exporter.export_metrics()
        
        assert "http_requests_total" in metrics_output
        assert "memory_usage_bytes" in metrics_output
        assert "1073741824" in metrics_output
    
    @pytest.mark.asyncio
    async def test_start_metrics_server(self):
        """Test de démarrage du serveur de métriques"""
        with patch('prometheus_client.start_http_server') as mock_server:
            await self.prometheus_exporter.start_metrics_server(port=8000)
            
            mock_server.assert_called_once_with(8000)


class TestNotificationService:
    """Tests pour le service de notifications"""
    
    def setup_method(self):
        """Configuration avant chaque test"""
        self.notification_service = NotificationService()
    
    def test_notification_service_creation(self):
        """Test de création du service"""
        assert isinstance(self.notification_service, NotificationService)
    
    @patch('smtplib.SMTP')
    def test_send_email_notification(self, mock_smtp):
        """Test d'envoi de notification email"""
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value = mock_smtp_instance
        
        alert = Alert(
            id="alert_001",
            severity=AlertSeverity.CRITICAL,
            title="Critical Alert",
            message="System is down",
            source="system_monitor"
        )
        
        success = self.notification_service.send_email_notification(
            alert,
            recipients=["admin@example.com"]
        )
        
        assert success is True
        mock_smtp.assert_called()
        mock_smtp_instance.send_message.assert_called()
    
    @patch('requests.post')
    def test_send_slack_notification(self, mock_post):
        """Test d'envoi de notification Slack"""
        mock_post.return_value.status_code = 200
        
        alert = Alert(
            id="alert_002",
            severity=AlertSeverity.WARNING,
            title="Warning Alert",
            message="High CPU usage",
            source="cpu_monitor"
        )
        
        success = self.notification_service.send_slack_notification(
            alert,
            webhook_url="https://hooks.slack.com/test"
        )
        
        assert success is True
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_send_webhook_notification(self, mock_post):
        """Test d'envoi de notification webhook"""
        mock_post.return_value.status_code = 200
        
        alert = Alert(
            id="alert_003",
            severity=AlertSeverity.ERROR,
            title="Error Alert",
            message="Database connection failed",
            source="db_monitor"
        )
        
        success = self.notification_service.send_webhook_notification(
            alert,
            webhook_url="https://example.com/webhook"
        )
        
        assert success is True
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_notification_async(self):
        """Test d'envoi de notification asynchrone"""
        with patch.object(self.notification_service, 'send_email_notification') as mock_email:
            mock_email.return_value = True
            
            alert = Alert(
                id="alert_004",
                severity=AlertSeverity.INFO,
                title="Info Alert",
                message="System update completed",
                source="update_monitor"
            )
            
            await self.notification_service.send_notification_async(alert)
            
            mock_email.assert_called_once()


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires"""
    
    def test_create_alert_function(self):
        """Test de la fonction create_alert"""
        alert = create_alert(
            severity="critical",
            title="Test Alert",
            message="Test message",
            source="test_source"
        )
        
        assert isinstance(alert, Alert)
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.title == "Test Alert"
    
    @pytest.mark.asyncio
    async def test_send_notification_function(self):
        """Test de la fonction send_notification"""
        with patch('app.ml_analytics.monitoring.NotificationService') as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.send_notification_async = AsyncMock(return_value=True)
            
            alert = Alert(
                id="test_alert",
                severity=AlertSeverity.WARNING,
                title="Test",
                message="Test message",
                source="test"
            )
            
            success = await send_notification(alert)
            
            assert success is True
    
    def test_calculate_drift_score_function(self):
        """Test de la fonction calculate_drift_score"""
        baseline_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(0.5, 1, 500)  # Légère dérive
        
        drift_score = calculate_drift_score(baseline_data, current_data)
        
        assert 0 <= drift_score <= 1
        assert drift_score > 0  # Devrait détecter une certaine dérive
    
    @pytest.mark.asyncio
    async def test_monitor_system_health_function(self):
        """Test de la fonction monitor_system_health"""
        with patch('app.ml_analytics.monitoring.HealthMonitor') as mock_monitor:
            mock_instance = MagicMock()
            mock_monitor.return_value = mock_instance
            mock_instance.check_system_health = AsyncMock(return_value={
                "healthy": True,
                "components": {"cpu": "healthy", "memory": "healthy"}
            })
            
            health_status = await monitor_system_health()
            
            assert health_status["healthy"] is True
            assert "components" in health_status
    
    def test_collect_performance_metrics_function(self):
        """Test de la fonction collect_performance_metrics"""
        with patch('psutil.cpu_percent', return_value=45.5):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 65.2
                
                metrics = collect_performance_metrics()
                
                assert "cpu_percent" in metrics
                assert "memory_percent" in metrics
                assert metrics["cpu_percent"] == 45.5
                assert metrics["memory_percent"] == 65.2


# Fixtures pour les tests
@pytest.fixture
def sample_alert():
    """Alerte de test"""
    return Alert(
        id="test_alert_001",
        severity=AlertSeverity.WARNING,
        title="Test Alert",
        message="This is a test alert",
        source="test_monitor"
    )


@pytest.fixture
def health_monitor():
    """Moniteur de santé de test"""
    return HealthMonitor()


@pytest.fixture
def alert_manager():
    """Gestionnaire d'alertes de test"""
    return AlertManager()


@pytest.fixture
def metrics_collector():
    """Collecteur de métriques de test"""
    return MetricsCollector()


@pytest.fixture
def sample_metrics_data():
    """Données de métriques de test"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
        'cpu_usage': np.random.uniform(20, 80, 100),
        'memory_usage': np.random.uniform(30, 90, 100),
        'request_count': np.random.poisson(50, 100)
    })


# Tests d'intégration
@pytest.mark.integration
class TestMonitoringIntegration:
    """Tests d'intégration pour le système de monitoring"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_pipeline(self):
        """Test du pipeline complet de monitoring"""
        # Configuration des composants
        health_monitor = HealthMonitor()
        alert_manager = AlertManager()
        metrics_collector = MetricsCollector()
        notification_service = NotificationService()
        
        # 1. Collecte de métriques
        metrics_collector.set_gauge("cpu_usage", 85.0)  # Usage élevé
        
        # 2. Contrôle de santé
        with patch.object(health_monitor, 'check_system_health') as mock_health:
            mock_health.return_value = {
                "healthy": False,
                "components": {"cpu": "unhealthy"}
            }
            
            health_status = await health_monitor.check_system_health()
        
        # 3. Création d'alerte basée sur les métriques
        if not health_status["healthy"]:
            alert = alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                title="High CPU Usage",
                message="CPU usage exceeded threshold",
                source="cpu_monitor"
            )
        
        # 4. Envoi de notification
        with patch.object(notification_service, 'send_notification_async') as mock_notify:
            mock_notify.return_value = True
            
            await notification_service.send_notification_async(alert)
            
            mock_notify.assert_called_once_with(alert)
        
        # Vérifications finales
        assert len(alert_manager.active_alerts) == 1
        assert alert.severity == AlertSeverity.WARNING


# Tests de performance
@pytest.mark.performance
class TestMonitoringPerformance:
    """Tests de performance pour le monitoring"""
    
    def test_metrics_collection_performance(self):
        """Test de performance de collection de métriques"""
        import time
        
        metrics_collector = MetricsCollector()
        
        start_time = time.time()
        
        # Collecter beaucoup de métriques
        for i in range(10000):
            metrics_collector.increment_counter(
                "performance_test",
                labels={"batch": str(i // 100)}
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Devrait être rapide
        assert duration < 2.0
    
    @pytest.mark.asyncio
    async def test_alert_processing_performance(self):
        """Test de performance de traitement d'alertes"""
        import time
        
        alert_manager = AlertManager()
        
        start_time = time.time()
        
        # Créer beaucoup d'alertes
        for i in range(1000):
            alert_manager.create_alert(
                severity=AlertSeverity.INFO,
                title=f"Performance Test Alert {i}",
                message="Test message",
                source="performance_test"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Traitement rapide
        assert duration < 1.0
        assert len(alert_manager.active_alerts) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
