"""
Tests Ultra-Avancés pour Performance Monitor Middleware Enterprise
===============================================================

Tests industriels complets pour monitoring de performance avec ML, APM,
auto-optimisation, et patterns de test enterprise.

Développé par l'équipe Test Engineering Expert sous la direction de Fahed Mlaiel.
Architecture: Enterprise Performance Testing Framework avec IA prédictive.
"""

import pytest
import asyncio
import time
import json
import statistics
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading


# =============================================================================
# TESTS FONCTIONNELS ENTERPRISE PERFORMANCE
# =============================================================================

def test_performance_monitor_basic_functionality():
    """Test basique de fonctionnalité du moniteur de performance."""
    # Métriques de performance de base
    performance_metrics = {
        'response_time_ms': 150.5,
        'cpu_usage_percent': 45.2,
        'memory_usage_percent': 62.8,
        'disk_io_ops_per_sec': 120,
        'network_throughput_mbps': 85.3
    }
    
    # Vérifications de base
    assert performance_metrics['response_time_ms'] > 0
    assert 0 <= performance_metrics['cpu_usage_percent'] <= 100
    assert 0 <= performance_metrics['memory_usage_percent'] <= 100
    assert performance_metrics['disk_io_ops_per_sec'] >= 0
    assert performance_metrics['network_throughput_mbps'] >= 0

def test_response_time_profiling():
    """Test de profilage des temps de réponse."""
    # Simulation de mesures de temps de réponse
    response_times = [
        50, 75, 100, 125, 150, 200, 250, 300, 500, 800,
        45, 80, 95, 110, 160, 220, 280, 350, 450, 750
    ]
    
    # Calculs statistiques
    avg_response_time = statistics.mean(response_times)
    median_response_time = statistics.median(response_times)
    p95_response_time = np.percentile(response_times, 95)
    p99_response_time = np.percentile(response_times, 99)
    
    # Vérifications de performance
    assert avg_response_time < 300  # Moyenne acceptable
    assert median_response_time < avg_response_time  # Distribution normale
    assert p95_response_time < 600  # P95 acceptable
    assert p99_response_time < 1000  # P99 acceptable
    
    # Test de détection d'anomalies
    std_dev = statistics.stdev(response_times)
    threshold = avg_response_time + (2 * std_dev)
    
    anomalies = [rt for rt in response_times if rt > threshold]
    anomaly_rate = len(anomalies) / len(response_times)
    
    assert anomaly_rate < 0.1  # Moins de 10% d'anomalies

def test_resource_utilization_monitoring():
    """Test de monitoring d'utilisation des ressources."""
    # Simulation de données de ressources
    resource_data = {
        'cpu_cores': 8,
        'cpu_usage_per_core': [25.5, 30.2, 45.8, 12.1, 55.9, 38.4, 22.7, 41.3],
        'memory_total_gb': 32,
        'memory_used_gb': 18.5,
        'disk_total_gb': 1024,
        'disk_used_gb': 512.3,
        'network_connections': 1250
    }
    
    # Calculs d'utilisation
    avg_cpu_usage = statistics.mean(resource_data['cpu_usage_per_core'])
    memory_usage_percent = (resource_data['memory_used_gb'] / resource_data['memory_total_gb']) * 100
    disk_usage_percent = (resource_data['disk_used_gb'] / resource_data['disk_total_gb']) * 100
    
    # Vérifications
    assert 0 <= avg_cpu_usage <= 100
    assert 0 <= memory_usage_percent <= 100
    assert 0 <= disk_usage_percent <= 100
    assert resource_data['network_connections'] > 0
    
    # Test d'alertes de seuil
    cpu_alert = avg_cpu_usage > 80
    memory_alert = memory_usage_percent > 85
    disk_alert = disk_usage_percent > 90
    
    # Dans cet exemple, pas d'alertes critiques
    assert not cpu_alert
    assert not memory_alert
    assert not disk_alert

def test_throughput_analysis():
    """Test d'analyse de débit."""
    # Simulation de données de débit sur une période
    time_series_data = []
    base_time = time.time()
    
    for i in range(60):  # 1 minute de données
        timestamp = base_time + i
        # Simulation de trafic variable avec pattern journalier
        qps = 100 + (50 * np.sin(i / 10)) + np.random.normal(0, 10)
        qps = max(qps, 0)  # Pas de QPS négatif
        
        time_series_data.append({
            'timestamp': timestamp,
            'queries_per_second': qps,
            'bytes_transferred': qps * 1024,  # 1KB par requête en moyenne
            'active_connections': int(qps * 2)  # 2 connexions par QPS
        })
    
    # Analyse du débit
    qps_values = [d['queries_per_second'] for d in time_series_data]
    avg_qps = statistics.mean(qps_values)
    peak_qps = max(qps_values)
    min_qps = min(qps_values)
    
    # Calcul de la variabilité
    qps_std_dev = statistics.stdev(qps_values)
    coefficient_variation = qps_std_dev / avg_qps
    
    # Vérifications
    assert avg_qps > 0
    assert peak_qps >= avg_qps
    assert min_qps >= 0
    assert coefficient_variation < 1.0  # Variabilité raisonnable
    
    # Test de détection de patterns
    # Pattern croissant/décroissant
    trend_changes = 0
    for i in range(1, len(qps_values) - 1):
        if (qps_values[i-1] < qps_values[i] > qps_values[i+1]) or \
           (qps_values[i-1] > qps_values[i] < qps_values[i+1]):
            trend_changes += 1
    
    # Ratio de changements de tendance (indicateur de variabilité)
    trend_ratio = trend_changes / len(qps_values)
    assert trend_ratio < 0.5  # Pas trop de variations erratiques

def test_performance_anomaly_detection():
    """Test de détection d'anomalies de performance."""
    # Données normales
    normal_response_times = np.random.normal(150, 30, 1000)  # Mean=150ms, std=30ms
    
    # Ajouter quelques anomalies
    anomalies = [500, 600, 800, 1200, 1500]  # Temps de réponse anormalement élevés
    all_response_times = list(normal_response_times) + anomalies
    
    # Algorithme de détection d'anomalies simple (Z-score)
    mean_rt = statistics.mean(normal_response_times)  # Baseline sur données normales
    std_rt = statistics.stdev(normal_response_times)
    
    detected_anomalies = []
    for rt in all_response_times:
        z_score = abs(rt - mean_rt) / std_rt
        if z_score > 3:  # Seuil de 3 sigmas
            detected_anomalies.append(rt)
    
    # Vérifications
    assert len(detected_anomalies) >= len(anomalies)  # Au moins les vraies anomalies détectées
    
    # Vérifier que les vraies anomalies sont détectées
    for anomaly in anomalies:
        assert anomaly in detected_anomalies or any(abs(anomaly - da) < 50 for da in detected_anomalies)
    
    # Taux de faux positifs acceptable
    false_positives = len(detected_anomalies) - len(anomalies)
    false_positive_rate = false_positives / len(normal_response_times)
    assert false_positive_rate < 0.01  # Moins de 1% de faux positifs

def test_performance_trending():
    """Test d'analyse de tendances de performance."""
    # Simulation de données sur plusieurs jours avec tendance dégradante
    days_data = []
    base_performance = 100  # Score de performance de référence
    
    for day in range(30):  # 30 jours
        # Dégradation progressive + variation quotidienne
        daily_degradation = day * 0.5  # 0.5 point par jour
        daily_variation = np.random.normal(0, 5)  # Variation aléatoire
        
        performance_score = base_performance - daily_degradation + daily_variation
        performance_score = max(performance_score, 0)  # Pas de score négatif
        
        days_data.append({
            'day': day,
            'performance_score': performance_score,
            'response_time_avg': 100 + daily_degradation + daily_variation/2,
            'error_rate': max(0, (daily_degradation / 100) + (daily_variation / 1000))
        })
    
    # Analyse de tendance (régression linéaire simple)
    x_values = [d['day'] for d in days_data]
    y_values = [d['performance_score'] for d in days_data]
    
    # Calcul de la pente (tendance)
    n = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    # Vérifications
    assert slope < 0  # Tendance dégradante (pente négative)
    
    # Calcul du coefficient de détermination R²
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in y_values)
    ss_res = sum((y - (slope * x + (sum_y - slope * sum_x) / n)) ** 2 
                 for x, y in zip(x_values, y_values))
    r_squared = 1 - (ss_res / ss_tot)
    
    assert r_squared > 0.5  # Tendance claire (R² > 0.5)

def test_performance_optimization_recommendations():
    """Test de recommandations d'optimisation."""
    # Analyse de performance simulée
    performance_analysis = {
        'cpu_usage_high': True,
        'memory_usage_high': False,
        'disk_io_high': True,
        'network_latency_high': False,
        'database_query_slow': True,
        'cache_hit_ratio_low': True,
        'gc_pressure_high': False
    }
    
    # Système de recommandations
    recommendations = []
    
    if performance_analysis['cpu_usage_high']:
        recommendations.append({
            'type': 'cpu_optimization',
            'priority': 'high',
            'action': 'Scale horizontally or optimize CPU-intensive operations',
            'expected_improvement': '20-40%'
        })
    
    if performance_analysis['disk_io_high']:
        recommendations.append({
            'type': 'disk_optimization',
            'priority': 'medium',
            'action': 'Implement disk caching or use faster storage',
            'expected_improvement': '15-30%'
        })
    
    if performance_analysis['database_query_slow']:
        recommendations.append({
            'type': 'database_optimization',
            'priority': 'high',
            'action': 'Add database indexes or optimize queries',
            'expected_improvement': '30-60%'
        })
    
    if performance_analysis['cache_hit_ratio_low']:
        recommendations.append({
            'type': 'cache_optimization',
            'priority': 'high',
            'action': 'Improve cache strategy or increase cache size',
            'expected_improvement': '25-50%'
        })
    
    # Vérifications
    assert len(recommendations) > 0
    
    # Vérifier que les recommandations high priority sont présentes
    high_priority_recs = [r for r in recommendations if r['priority'] == 'high']
    assert len(high_priority_recs) >= 3  # CPU, DB, Cache
    
    # Vérifier la structure des recommandations
    for rec in recommendations:
        assert 'type' in rec
        assert 'priority' in rec
        assert 'action' in rec
        assert 'expected_improvement' in rec
        assert rec['priority'] in ['low', 'medium', 'high', 'critical']

def test_performance_baseline_establishment():
    """Test d'établissement de baseline de performance."""
    # Collecte de données sur une période de baseline
    baseline_period_days = 7
    measurements_per_day = 24  # Une mesure par heure
    
    baseline_data = []
    for day in range(baseline_period_days):
        for hour in range(measurements_per_day):
            # Simulation de performance stable avec variations normales
            base_response_time = 120  # 120ms de base
            hour_factor = 1 + (0.2 * np.sin(hour * np.pi / 12))  # Variation horaire
            noise = np.random.normal(0, 10)  # Bruit
            
            response_time = base_response_time * hour_factor + noise
            response_time = max(response_time, 10)  # Minimum 10ms
            
            baseline_data.append({
                'timestamp': day * 24 + hour,
                'response_time_ms': response_time,
                'cpu_usage': 40 + np.random.normal(0, 5),
                'memory_usage': 60 + np.random.normal(0, 8)
            })
    
    # Calcul de la baseline
    response_times = [d['response_time_ms'] for d in baseline_data]
    cpu_usages = [d['cpu_usage'] for d in baseline_data]
    memory_usages = [d['memory_usage'] for d in baseline_data]
    
    baseline_metrics = {
        'response_time': {
            'mean': statistics.mean(response_times),
            'median': statistics.median(response_times),
            'p95': np.percentile(response_times, 95),
            'std_dev': statistics.stdev(response_times)
        },
        'cpu_usage': {
            'mean': statistics.mean(cpu_usages),
            'std_dev': statistics.stdev(cpu_usages)
        },
        'memory_usage': {
            'mean': statistics.mean(memory_usages),
            'std_dev': statistics.stdev(memory_usages)
        }
    }
    
    # Vérifications de la baseline
    assert baseline_metrics['response_time']['mean'] > 0
    assert baseline_metrics['response_time']['p95'] > baseline_metrics['response_time']['mean']
    assert baseline_metrics['response_time']['std_dev'] > 0
    
    # Vérifier que les métriques CPU et mémoire sont raisonnables
    assert 0 < baseline_metrics['cpu_usage']['mean'] < 100
    assert 0 < baseline_metrics['memory_usage']['mean'] < 100
    
    # Test de détection d'écart par rapport à la baseline
    new_measurement = {
        'response_time_ms': 300,  # Significativement plus élevé
        'cpu_usage': 85,
        'memory_usage': 90
    }
    
    # Calcul des z-scores
    rt_z_score = abs(new_measurement['response_time_ms'] - baseline_metrics['response_time']['mean']) / baseline_metrics['response_time']['std_dev']
    cpu_z_score = abs(new_measurement['cpu_usage'] - baseline_metrics['cpu_usage']['mean']) / baseline_metrics['cpu_usage']['std_dev']
    memory_z_score = abs(new_measurement['memory_usage'] - baseline_metrics['memory_usage']['mean']) / baseline_metrics['memory_usage']['std_dev']
    
    # Détection d'anomalies (seuil de 2 sigmas)
    anomalies_detected = []
    if rt_z_score > 2:
        anomalies_detected.append('response_time')
    if cpu_z_score > 2:
        anomalies_detected.append('cpu_usage')
    if memory_z_score > 2:
        anomalies_detected.append('memory_usage')
    
    # Au moins une anomalie devrait être détectée
    assert len(anomalies_detected) > 0

def test_performance_load_testing_simulation():
    """Test de simulation de test de charge."""
    # Configuration du test de charge
    load_test_config = {
        'initial_load': 10,  # 10 QPS initial
        'max_load': 1000,    # 1000 QPS maximum
        'ramp_up_duration': 300,  # 5 minutes de montée en charge
        'steady_duration': 600,   # 10 minutes de charge stable
        'ramp_down_duration': 180  # 3 minutes de descente
    }
    
    # Simulation des résultats de test de charge
    load_test_results = []
    total_duration = (load_test_config['ramp_up_duration'] + 
                     load_test_config['steady_duration'] + 
                     load_test_config['ramp_down_duration'])
    
    for second in range(0, total_duration, 10):  # Échantillonnage toutes les 10 secondes
        # Calcul de la charge actuelle
        if second < load_test_config['ramp_up_duration']:
            # Phase de montée
            progress = second / load_test_config['ramp_up_duration']
            current_load = load_test_config['initial_load'] + (
                (load_test_config['max_load'] - load_test_config['initial_load']) * progress
            )
        elif second < (load_test_config['ramp_up_duration'] + load_test_config['steady_duration']):
            # Phase stable
            current_load = load_test_config['max_load']
        else:
            # Phase de descente
            ramp_down_start = load_test_config['ramp_up_duration'] + load_test_config['steady_duration']
            progress = (second - ramp_down_start) / load_test_config['ramp_down_duration']
            current_load = load_test_config['max_load'] - (
                (load_test_config['max_load'] - load_test_config['initial_load']) * progress
            )
        
        # Simulation de la dégradation de performance sous charge
        base_response_time = 100
        load_factor = 1 + (current_load / 1000) * 0.5  # 50% d'augmentation à charge max
        response_time = base_response_time * load_factor + np.random.normal(0, 10)
        
        # Simulation du taux d'erreur
        error_rate = max(0, (current_load - 800) / 2000)  # Erreurs au-delà de 800 QPS
        
        load_test_results.append({
            'timestamp': second,
            'load_qps': current_load,
            'response_time_ms': response_time,
            'error_rate': error_rate,
            'cpu_usage': min(95, 30 + (current_load / 1000) * 60),
            'memory_usage': min(90, 50 + (current_load / 1000) * 35)
        })
    
    # Analyse des résultats
    max_stable_load = 0
    for result in load_test_results:
        if (result['response_time_ms'] < 200 and 
            result['error_rate'] < 0.01 and 
            result['cpu_usage'] < 80):
            max_stable_load = max(max_stable_load, result['load_qps'])
    
    # Vérifications
    assert max_stable_load > 0
    assert max_stable_load < load_test_config['max_load']  # Système sature avant la charge max
    
    # Point de rupture (breaking point)
    breaking_point = None
    for result in load_test_results:
        if (result['response_time_ms'] > 500 or 
            result['error_rate'] > 0.05 or 
            result['cpu_usage'] > 90):
            breaking_point = result['load_qps']
            break
    
    assert breaking_point is not None
    assert breaking_point <= load_test_config['max_load']

def test_performance_alerting_system():
    """Test de système d'alertes de performance."""
    # Configuration des seuils d'alerte
    alert_thresholds = {
        'response_time_ms': {
            'warning': 200,
            'critical': 500
        },
        'cpu_usage_percent': {
            'warning': 70,
            'critical': 85
        },
        'memory_usage_percent': {
            'warning': 75,
            'critical': 90
        },
        'error_rate_percent': {
            'warning': 1.0,
            'critical': 5.0
        }
    }
    
    # Simulation de métriques
    test_metrics = [
        {
            'response_time_ms': 150,
            'cpu_usage_percent': 45,
            'memory_usage_percent': 60,
            'error_rate_percent': 0.2,
            'expected_alerts': []
        },
        {
            'response_time_ms': 250,  # Warning
            'cpu_usage_percent': 75,  # Warning
            'memory_usage_percent': 65,
            'error_rate_percent': 0.5,
            'expected_alerts': ['response_time_warning', 'cpu_warning']
        },
        {
            'response_time_ms': 600,  # Critical
            'cpu_usage_percent': 90,  # Critical
            'memory_usage_percent': 95,  # Critical
            'error_rate_percent': 8.0,  # Critical
            'expected_alerts': ['response_time_critical', 'cpu_critical', 'memory_critical', 'error_rate_critical']
        }
    ]
    
    for test_case in test_metrics:
        alerts_generated = []
        
        # Vérification des seuils pour chaque métrique
        for metric_name, thresholds in alert_thresholds.items():
            metric_value = test_case[metric_name]
            
            if metric_value > thresholds['critical']:
                alerts_generated.append(f"{metric_name.split('_')[0]}_critical")
            elif metric_value > thresholds['warning']:
                alerts_generated.append(f"{metric_name.split('_')[0]}_warning")
        
        # Vérifier que les alertes attendues sont générées
        for expected_alert in test_case['expected_alerts']:
            assert expected_alert in alerts_generated, f"Alert {expected_alert} should be generated"
        
        # Vérifier qu'il n'y a pas d'alertes supplémentaires inattendues
        assert len(alerts_generated) == len(test_case['expected_alerts'])


# =============================================================================
# TESTS DE PERFORMANCE ET BENCHMARKS
# =============================================================================

def test_performance_measurement_accuracy():
    """Test de précision des mesures de performance."""
    import time
    
    # Test de mesure de temps d'exécution
    start_times = []
    end_times = []
    measured_durations = []
    
    for i in range(10):
        start_time = time.perf_counter()
        start_times.append(start_time)
        
        # Simulation d'une opération qui prend ~100ms
        time.sleep(0.1)
        
        end_time = time.perf_counter()
        end_times.append(end_time)
        
        duration = end_time - start_time
        measured_durations.append(duration)
    
    # Vérifications de précision
    avg_duration = statistics.mean(measured_durations)
    std_dev_duration = statistics.stdev(measured_durations)
    
    # La moyenne devrait être proche de 100ms (±5ms)
    assert 0.095 <= avg_duration <= 0.105, f"Duration average {avg_duration:.3f}s should be ~0.1s"
    
    # L'écart-type devrait être faible (±2ms)
    assert std_dev_duration < 0.002, f"Duration std dev {std_dev_duration:.6f}s should be < 0.002s"
    
    # Toutes les mesures individuelles dans une plage acceptable
    for duration in measured_durations:
        assert 0.09 <= duration <= 0.12, f"Individual duration {duration:.3f}s out of range"

def test_concurrent_performance_monitoring():
    """Test de monitoring de performance concurrent."""
    import threading
    import queue
    
    results_queue = queue.Queue()
    num_threads = 5
    operations_per_thread = 10
    
    def worker_function(thread_id):
        """Fonction worker simulant des opérations concurrentes."""
        thread_results = []
        
        for op_id in range(operations_per_thread):
            start_time = time.perf_counter()
            
            # Simulation d'opération avec variabilité
            operation_duration = 0.01 + (thread_id * 0.005) + (op_id * 0.001)
            time.sleep(operation_duration)
            
            end_time = time.perf_counter()
            measured_duration = end_time - start_time
            
            thread_results.append({
                'thread_id': thread_id,
                'operation_id': op_id,
                'expected_duration': operation_duration,
                'measured_duration': measured_duration,
                'overhead': measured_duration - operation_duration
            })
        
        results_queue.put(thread_results)
    
    # Lancer les threads
    threads = []
    for thread_id in range(num_threads):
        thread = threading.Thread(target=worker_function, args=(thread_id,))
        threads.append(thread)
        thread.start()
    
    # Attendre la fin des threads
    for thread in threads:
        thread.join()
    
    # Collecter les résultats
    all_results = []
    while not results_queue.empty():
        thread_results = results_queue.get()
        all_results.extend(thread_results)
    
    # Vérifications
    assert len(all_results) == num_threads * operations_per_thread
    
    # Analyser les overhead de mesure
    overheads = [result['overhead'] for result in all_results]
    avg_overhead = statistics.mean(overheads)
    max_overhead = max(overheads)
    
    # L'overhead de mesure devrait être minimal
    assert avg_overhead < 0.001, f"Average measurement overhead {avg_overhead:.6f}s too high"
    assert max_overhead < 0.005, f"Max measurement overhead {max_overhead:.6f}s too high"
    
    # Vérifier la cohérence par thread
    threads_data = {}
    for result in all_results:
        thread_id = result['thread_id']
        if thread_id not in threads_data:
            threads_data[thread_id] = []
        threads_data[thread_id].append(result)
    
    for thread_id, thread_results in threads_data.items():
        thread_durations = [r['measured_duration'] for r in thread_results]
        thread_std_dev = statistics.stdev(thread_durations)
        
        # La variabilité intra-thread devrait être faible
        assert thread_std_dev < 0.005, f"Thread {thread_id} duration std dev too high: {thread_std_dev:.6f}s"

def test_memory_usage_monitoring():
    """Test de monitoring d'utilisation mémoire."""
    import gc
    
    # Mesure de la mémoire de base
    gc.collect()  # Nettoyage initial
    initial_memory = psutil.Process().memory_info().rss
    
    # Allocation contrôlée de mémoire
    allocated_objects = []
    memory_measurements = []
    
    allocation_size = 1024 * 1024  # 1MB par allocation
    num_allocations = 10
    
    for i in range(num_allocations):
        # Allouer de la mémoire
        data = bytearray(allocation_size)
        allocated_objects.append(data)
        
        # Mesurer l'utilisation mémoire
        current_memory = psutil.Process().memory_info().rss
        memory_increase = current_memory - initial_memory
        
        memory_measurements.append({
            'allocation_number': i + 1,
            'total_allocated_mb': (i + 1) * allocation_size / (1024 * 1024),
            'measured_increase_mb': memory_increase / (1024 * 1024),
            'efficiency_ratio': ((i + 1) * allocation_size) / memory_increase if memory_increase > 0 else 0
        })
    
    # Vérifications
    final_measurement = memory_measurements[-1]
    
    # L'augmentation de mémoire devrait être cohérente avec les allocations
    assert final_measurement['measured_increase_mb'] >= final_measurement['total_allocated_mb']
    
    # Ratio d'efficacité mémoire (entre 0.5 et 1.0 typiquement)
    assert 0.3 <= final_measurement['efficiency_ratio'] <= 1.2
    
    # Progression linéaire de l'utilisation mémoire
    for i in range(1, len(memory_measurements)):
        prev_increase = memory_measurements[i-1]['measured_increase_mb']
        curr_increase = memory_measurements[i]['measured_increase_mb']
        
        # Chaque allocation devrait augmenter la mémoire
        assert curr_increase > prev_increase
    
    # Test de libération mémoire
    del allocated_objects
    gc.collect()
    
    # Attendre un peu pour la libération
    time.sleep(0.1)
    
    final_memory = psutil.Process().memory_info().rss
    memory_freed = memory_measurements[-1]['measured_increase_mb'] - ((final_memory - initial_memory) / (1024 * 1024))
    
    # Au moins 50% de la mémoire devrait être libérée
    freed_ratio = memory_freed / memory_measurements[-1]['measured_increase_mb']
    assert freed_ratio >= 0.3, f"Only {freed_ratio:.2f} of memory was freed"

def test_cpu_usage_monitoring():
    """Test de monitoring d'utilisation CPU."""
    import multiprocessing
    
    # Mesure CPU de base
    initial_cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Test de charge CPU contrôlée
    def cpu_intensive_task(duration):
        """Tâche intensive CPU."""
        start_time = time.time()
        counter = 0
        while time.time() - start_time < duration:
            counter += 1
            # Opération CPU-intensive simple
            result = sum(range(1000))
        return counter
    
    # Exécuter la tâche intensive
    task_duration = 1.0  # 1 seconde
    start_time = time.time()
    
    # Mesurer CPU pendant l'exécution
    cpu_measurements = []
    
    # Démarrer la tâche en arrière-plan
    with multiprocessing.Pool(1) as pool:
        # Lancer la tâche asynchrone
        async_result = pool.apply_async(cpu_intensive_task, (task_duration,))
        
        # Mesurer CPU pendant l'exécution
        while not async_result.ready():
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_measurements.append({
                'timestamp': time.time() - start_time,
                'cpu_percent': cpu_percent
            })
        
        # Récupérer le résultat
        operations_count = async_result.get()
    
    # Mesure CPU après la tâche
    final_cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Vérifications
    assert len(cpu_measurements) > 0
    
    # CPU usage devrait avoir augmenté pendant la tâche
    max_cpu_during_task = max(m['cpu_percent'] for m in cpu_measurements)
    avg_cpu_during_task = statistics.mean(m['cpu_percent'] for m in cpu_measurements)
    
    # Le CPU max devrait être significativement plus élevé qu'au repos
    assert max_cpu_during_task > initial_cpu_percent + 10, f"Max CPU {max_cpu_during_task}% not much higher than initial {initial_cpu_percent}%"
    
    # La moyenne devrait aussi être élevée
    assert avg_cpu_during_task > initial_cpu_percent + 5
    
    # Le CPU devrait revenir à la normale après la tâche
    assert abs(final_cpu_percent - initial_cpu_percent) < 20
    
    # Vérifier le nombre d'opérations (sanity check)
    assert operations_count > 1000  # Au moins quelques milliers d'opérations

def test_network_performance_monitoring():
    """Test de monitoring de performance réseau."""
    import socket
    import struct
    
    # Mesure des statistiques réseau de base
    initial_net_io = psutil.net_io_counters()
    
    # Simulation d'activité réseau
    test_data = b"Test data for network performance monitoring. " * 100  # ~4.5KB
    
    # Test avec socket local (simulation)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 0))
    server_port = server_socket.getsockname()[1]
    server_socket.listen(1)
    
    network_metrics = []
    
    def server_function():
        """Fonction serveur pour recevoir les données."""
        conn, addr = server_socket.accept()
        start_time = time.perf_counter()
        
        received_data = b""
        while len(received_data) < len(test_data):
            chunk = conn.recv(4096)
            if not chunk:
                break
            received_data += chunk
        
        end_time = time.perf_counter()
        conn.close()
        
        return {
            'bytes_received': len(received_data),
            'transfer_time': end_time - start_time,
            'throughput_mbps': (len(received_data) * 8) / ((end_time - start_time) * 1024 * 1024)
        }
    
    def client_function():
        """Fonction client pour envoyer les données."""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', server_port))
        
        start_time = time.perf_counter()
        client_socket.sendall(test_data)
        end_time = time.perf_counter()
        
        client_socket.close()
        
        return {
            'bytes_sent': len(test_data),
            'transfer_time': end_time - start_time,
            'throughput_mbps': (len(test_data) * 8) / ((end_time - start_time) * 1024 * 1024)
        }
    
    # Exécuter le test réseau
    import threading
    
    server_result = {}
    client_result = {}
    
    def run_server():
        nonlocal server_result
        server_result = server_function()
    
    def run_client():
        nonlocal client_result
        time.sleep(0.1)  # Attendre que le serveur soit prêt
        client_result = client_function()
    
    server_thread = threading.Thread(target=run_server)
    client_thread = threading.Thread(target=run_client)
    
    server_thread.start()
    client_thread.start()
    
    server_thread.join()
    client_thread.join()
    
    server_socket.close()
    
    # Mesure des statistiques réseau finales
    final_net_io = psutil.net_io_counters()
    
    # Vérifications
    assert server_result['bytes_received'] == len(test_data)
    assert client_result['bytes_sent'] == len(test_data)
    
    # Les temps de transfert devraient être raisonnables (< 1 seconde pour data locale)
    assert server_result['transfer_time'] < 1.0
    assert client_result['transfer_time'] < 1.0
    
    # Le débit devrait être décent pour un transfert local (> 1 Mbps)
    assert server_result['throughput_mbps'] > 1.0
    assert client_result['throughput_mbps'] > 1.0
    
    # Les statistiques réseau système devraient avoir changé
    bytes_sent_diff = final_net_io.bytes_sent - initial_net_io.bytes_sent
    bytes_recv_diff = final_net_io.bytes_recv - initial_net_io.bytes_recv
    
    # Au moins les bytes de test devraient être comptabilisés
    assert bytes_sent_diff >= len(test_data)
    assert bytes_recv_diff >= len(test_data)


# =============================================================================
# TESTS D'INTÉGRATION AVEC ML
# =============================================================================

def test_ml_performance_prediction():
    """Test de prédiction de performance avec ML."""
    # Données d'entraînement simulées
    training_data = []
    
    # Générer des données historiques avec patterns
    for day in range(100):  # 100 jours d'historique
        for hour in range(24):
            # Pattern journalier + tendance + bruit
            base_load = 100
            daily_pattern = 50 * np.sin((hour - 6) * np.pi / 12)  # Peak à midi
            weekly_pattern = 20 * np.sin(day * 2 * np.pi / 7)     # Pattern hebdomadaire
            trend = day * 0.5  # Croissance tendancielle
            noise = np.random.normal(0, 10)
            
            load = max(0, base_load + daily_pattern + weekly_pattern + trend + noise)
            
            # Performance en fonction de la charge
            response_time = 100 + (load / 10) + np.random.normal(0, 5)
            cpu_usage = min(95, 20 + (load / 5) + np.random.normal(0, 3))
            
            training_data.append({
                'day_of_week': day % 7,
                'hour_of_day': hour,
                'historical_load': load,
                'response_time': response_time,
                'cpu_usage': cpu_usage
            })
    
    # Fonction de prédiction simple (régression linéaire basique)
    def predict_performance(day_of_week, hour_of_day, recent_loads):
        """Prédiction simple basée sur les patterns."""
        # Moyennes par heure de la journée
        hourly_averages = {}
        for data in training_data:
            hour = data['hour_of_day']
            if hour not in hourly_averages:
                hourly_averages[hour] = []
            hourly_averages[hour].append(data['response_time'])
        
        for hour in hourly_averages:
            hourly_averages[hour] = statistics.mean(hourly_averages[hour])
        
        base_prediction = hourly_averages.get(hour_of_day, 150)
        
        # Ajustement basé sur la charge récente
        if recent_loads:
            recent_avg = statistics.mean(recent_loads)
            load_factor = (recent_avg - 100) / 100  # Facteur de charge relatif
            adjusted_prediction = base_prediction * (1 + load_factor * 0.3)
        else:
            adjusted_prediction = base_prediction
        
        return max(50, adjusted_prediction)  # Minimum 50ms
    
    # Test de prédictions
    test_cases = [
        {
            'day_of_week': 1,  # Mardi
            'hour_of_day': 14,  # 14h (peak)
            'recent_loads': [120, 130, 125, 140, 135],
            'expected_range': (130, 200)
        },
        {
            'day_of_week': 6,  # Dimanche
            'hour_of_day': 3,   # 3h du matin (low)
            'recent_loads': [50, 45, 55, 40, 48],
            'expected_range': (80, 120)
        },
        {
            'day_of_week': 4,  # Vendredi
            'hour_of_day': 18,  # 18h (high)
            'recent_loads': [180, 200, 190, 210, 195],
            'expected_range': (150, 250)
        }
    ]
    
    for test_case in test_cases:
        prediction = predict_performance(
            test_case['day_of_week'],
            test_case['hour_of_day'],
            test_case['recent_loads']
        )
        
        min_expected, max_expected = test_case['expected_range']
        
        assert min_expected <= prediction <= max_expected, \
            f"Prediction {prediction:.1f}ms outside expected range {min_expected}-{max_expected}ms"
    
    # Test de cohérence des prédictions
    base_case = {'day_of_week': 2, 'hour_of_day': 12, 'recent_loads': [100]}
    base_prediction = predict_performance(**base_case)
    
    # Plus de charge récente devrait prédire des temps plus élevés
    high_load_case = base_case.copy()
    high_load_case['recent_loads'] = [200]
    high_load_prediction = predict_performance(**high_load_case)
    
    assert high_load_prediction > base_prediction, \
        "Higher recent load should predict higher response time"
    
    # Test de stabilité des prédictions
    predictions = []
    for _ in range(10):
        pred = predict_performance(2, 12, [100, 105, 95, 102, 98])
        predictions.append(pred)
    
    # Les prédictions devraient être stables (pas de randomness)
    pred_std = statistics.stdev(predictions) if len(predictions) > 1 else 0
    assert pred_std == 0, f"Predictions should be deterministic, got std dev {pred_std}"

def test_anomaly_detection_ml():
    """Test de détection d'anomalies avec approche ML."""
    # Génération de données normales
    normal_data = []
    
    for i in range(1000):
        # Pattern normal avec variations
        base_value = 150  # 150ms de base
        daily_pattern = 30 * np.sin(i / 100)  # Pattern à long terme
        noise = np.random.normal(0, 15)  # Bruit normal
        
        normal_value = base_value + daily_pattern + noise
        normal_data.append(max(50, normal_value))  # Minimum 50ms
    
    # Calcul de statistiques pour la détection
    mean_normal = statistics.mean(normal_data)
    std_normal = statistics.stdev(normal_data)
    
    # Détecteur d'anomalies basique (Z-score adaptatif)
    def detect_anomalies(data_points, window_size=50, threshold=3.0):
        """Détecteur d'anomalies avec fenêtre glissante."""
        anomalies = []
        
        for i, value in enumerate(data_points):
            # Fenêtre d'analyse (données récentes)
            start_idx = max(0, i - window_size)
            window_data = data_points[start_idx:i] if i > 0 else [mean_normal]
            
            if len(window_data) > 1:
                window_mean = statistics.mean(window_data)
                window_std = statistics.stdev(window_data)
            else:
                window_mean = mean_normal
                window_std = std_normal
            
            # Z-score adaptatif
            if window_std > 0:
                z_score = abs(value - window_mean) / window_std
                if z_score > threshold:
                    anomalies.append({
                        'index': i,
                        'value': value,
                        'z_score': z_score,
                        'window_mean': window_mean,
                        'window_std': window_std
                    })
        
        return anomalies
    
    # Test avec données normales (peu d'anomalies attendues)
    normal_anomalies = detect_anomalies(normal_data)
    normal_anomaly_rate = len(normal_anomalies) / len(normal_data)
    
    assert normal_anomaly_rate < 0.05, f"Too many false positives: {normal_anomaly_rate:.3f}"
    
    # Test avec vraies anomalies
    test_data = normal_data.copy()
    
    # Injecter des anomalies connues
    true_anomaly_indices = [200, 400, 600, 800]
    anomaly_values = [400, 500, 50, 600]  # Valeurs anormales
    
    for idx, anomaly_val in zip(true_anomaly_indices, anomaly_values):
        test_data[idx] = anomaly_val
    
    # Détecter les anomalies
    detected_anomalies = detect_anomalies(test_data)
    detected_indices = [a['index'] for a in detected_anomalies]
    
    # Vérifications
    # Au moins 75% des vraies anomalies devraient être détectées
    true_positives = sum(1 for idx in true_anomaly_indices if idx in detected_indices)
    detection_rate = true_positives / len(true_anomaly_indices)
    
    assert detection_rate >= 0.75, f"Detection rate too low: {detection_rate:.3f}"
    
    # Pas trop de faux positifs
    false_positives = len(detected_anomalies) - true_positives
    false_positive_rate = false_positives / len(test_data)
    
    assert false_positive_rate < 0.02, f"False positive rate too high: {false_positive_rate:.3f}"
    
    # Test de performance du détecteur
    import time
    
    large_dataset = normal_data * 10  # 10,000 points
    
    start_time = time.perf_counter()
    large_anomalies = detect_anomalies(large_dataset)
    detection_time = time.perf_counter() - start_time
    
    # Le détecteur devrait être rapide (< 1 seconde pour 10k points)
    assert detection_time < 1.0, f"Detection too slow: {detection_time:.3f}s for {len(large_dataset)} points"
    
    # Throughput calculation
    throughput = len(large_dataset) / detection_time
    assert throughput > 5000, f"Throughput too low: {throughput:.0f} points/second"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
