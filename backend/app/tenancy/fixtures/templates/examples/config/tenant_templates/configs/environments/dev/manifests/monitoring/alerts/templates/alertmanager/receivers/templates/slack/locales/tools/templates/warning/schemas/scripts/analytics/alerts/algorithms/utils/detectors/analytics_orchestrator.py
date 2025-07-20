#!/usr/bin/env python3
"""
Script d'Analytics Avancé pour Détection d'Anomalies
===================================================

Auteur: Fahed Mlaiel
Rôles: Lead Dev + Architecte IA, DBA & Data Engineer (PostgreSQL/Redis/MongoDB)

Ce script orchestre l'ensemble des systèmes de détection et génère
des rapports d'analytics complets en temps réel.
"""

import asyncio
import argparse
import logging
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Ajout du chemin pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_detectors import MLAnomalyDetector, DetectorFactory
from threshold_detectors import AdaptiveThresholdDetector, ThresholdDetectorFactory
from pattern_detectors import SequenceAnalyzer, BehaviorAnalyzer, SecurityPatternDetector
from performance_analyzers import SystemMetricsCollector, PerformanceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalyticsOrchestrator:
    """Orchestrateur principal pour l'analytics et la détection d'anomalies"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.detectors = {}
        self.collectors = {}
        self.results_cache = {}
        
        # Statistiques globales
        self.analytics_stats = {
            'total_analyses': 0,
            'anomalies_detected': 0,
            'false_positives': 0,
            'last_run': None,
            'avg_processing_time': 0.0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Charge la configuration"""
        default_config = {
            'detection_modules': {
                'ml_detectors': True,
                'threshold_detectors': True,
                'pattern_detectors': True,
                'security_detectors': True,
                'performance_analyzers': True
            },
            'data_sources': {
                'redis_url': 'redis://localhost:6379',
                'database_url': 'postgresql://user:pass@localhost/monitoring',
                'metrics_endpoint': 'http://localhost:9090/metrics'
            },
            'analysis_intervals': {
                'real_time': 60,      # secondes
                'batch': 3600,        # 1 heure
                'deep_analysis': 86400 # 24 heures
            },
            'alert_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            },
            'output_formats': ['json', 'prometheus', 'grafana'],
            'retention_days': 30
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Erreur chargement config {config_path}: {e}")
        
        return default_config
    
    async def initialize(self):
        """Initialise tous les détecteurs et collecteurs"""
        logger.info("Initialisation de l'orchestrateur d'analytics...")
        
        try:
            # Initialisation des détecteurs ML
            if self.config['detection_modules']['ml_detectors']:
                self.detectors['music_anomaly'] = DetectorFactory.create_music_anomaly_detector()
                self.detectors['user_behavior'] = DetectorFactory.create_user_behavior_detector()
                self.detectors['performance'] = DetectorFactory.create_performance_detector()
                logger.info("Détecteurs ML initialisés")
            
            # Initialisation des détecteurs de seuils
            if self.config['detection_modules']['threshold_detectors']:
                self.detectors['cpu_threshold'] = ThresholdDetectorFactory.create_cpu_detector()
                self.detectors['memory_threshold'] = ThresholdDetectorFactory.create_memory_detector()
                self.detectors['latency_threshold'] = ThresholdDetectorFactory.create_latency_detector()
                logger.info("Détecteurs de seuils initialisés")
            
            # Initialisation des analyseurs de patterns
            if self.config['detection_modules']['pattern_detectors']:
                self.detectors['sequence_analyzer'] = SequenceAnalyzer()
                self.detectors['behavior_analyzer'] = BehaviorAnalyzer()
                logger.info("Analyseurs de patterns initialisés")
            
            # Initialisation des détecteurs de sécurité
            if self.config['detection_modules']['security_detectors']:
                self.detectors['security_detector'] = SecurityPatternDetector()
                logger.info("Détecteurs de sécurité initialisés")
            
            # Initialisation des collecteurs de performance
            if self.config['detection_modules']['performance_analyzers']:
                self.collectors['system_metrics'] = SystemMetricsCollector()
                await self.collectors['system_metrics'].initialize()
                await self.collectors['system_metrics'].start_collection()
                
                self.detectors['performance_analyzer'] = PerformanceAnalyzer(
                    self.collectors['system_metrics']
                )
                logger.info("Analyseurs de performance initialisés")
            
            logger.info("Initialisation terminée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def run_real_time_analysis(self):
        """Lance l'analyse en temps réel"""
        logger.info("Démarrage de l'analyse en temps réel")
        
        interval = self.config['analysis_intervals']['real_time']
        
        while True:
            try:
                start_time = datetime.now()
                
                # Collecte des données en temps réel
                current_data = await self._collect_real_time_data()
                
                # Analyse avec tous les détecteurs
                analysis_results = await self._run_detection_pipeline(current_data)
                
                # Traitement des résultats
                await self._process_analysis_results(analysis_results)
                
                # Mise à jour des statistiques
                processing_time = (datetime.now() - start_time).total_seconds()
                await self._update_analytics_stats(processing_time, analysis_results)
                
                # Attendre jusqu'au prochain cycle
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Arrêt de l'analyse en temps réel")
                break
            except Exception as e:
                logger.error(f"Erreur dans l'analyse temps réel: {e}")
                await asyncio.sleep(5)  # Attendre avant de retry
    
    async def run_batch_analysis(self, start_time: datetime = None, duration_hours: int = 1):
        """Lance une analyse par lot"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=duration_hours)
        
        logger.info(f"Démarrage de l'analyse par lot: {start_time} à {start_time + timedelta(hours=duration_hours)}")
        
        try:
            # Collecte des données historiques
            historical_data = await self._collect_historical_data(start_time, duration_hours)
            
            # Analyse approfondie
            deep_analysis_results = await self._run_deep_analysis(historical_data)
            
            # Génération de rapports
            reports = await self._generate_comprehensive_reports(deep_analysis_results)
            
            # Sauvegarde des résultats
            await self._save_analysis_results(reports, start_time)
            
            logger.info("Analyse par lot terminée avec succès")
            return reports
            
        except Exception as e:
            logger.error(f"Erreur dans l'analyse par lot: {e}")
            raise
    
    async def _collect_real_time_data(self) -> Dict[str, Any]:
        """Collecte les données en temps réel"""
        data = {
            'timestamp': datetime.now(),
            'system_metrics': {},
            'application_metrics': {},
            'user_activities': [],
            'security_events': [],
            'api_requests': []
        }
        
        try:
            # Métriques système
            if 'system_metrics' in self.collectors:
                # Les métriques sont collectées automatiquement
                data['system_metrics'] = {
                    'status': 'collecting',
                    'collector_active': True
                }
            
            # Données simulées pour la démonstration
            # Dans un vrai système, ces données viendraient de diverses sources
            data['application_metrics'] = {
                'response_time_ms': np.random.normal(200, 50),
                'requests_per_second': np.random.poisson(100),
                'error_rate_percent': np.random.exponential(0.5),
                'active_users': np.random.randint(1000, 5000)
            }
            
            # Activités utilisateur simulées
            for _ in range(np.random.poisson(10)):
                data['user_activities'].append({
                    'user_id': f"user_{np.random.randint(1, 1000)}",
                    'action': np.random.choice(['play', 'pause', 'skip', 'like', 'search']),
                    'timestamp': datetime.now(),
                    'session_duration': np.random.randint(30, 3600)
                })
            
            # Événements de sécurité simulés
            if np.random.random() < 0.1:  # 10% de chance
                data['security_events'].append({
                    'ip_address': f"{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                    'event_type': np.random.choice(['login_attempt', 'api_call', 'file_access']),
                    'status_code': np.random.choice([200, 401, 403, 500]),
                    'timestamp': datetime.now()
                })
        
        except Exception as e:
            logger.error(f"Erreur collecte données temps réel: {e}")
        
        return data
    
    async def _collect_historical_data(self, start_time: datetime, duration_hours: int) -> Dict[str, Any]:
        """Collecte les données historiques"""
        # Simulation de données historiques
        # Dans un vrai système, ceci interrogerait les bases de données
        
        end_time = start_time + timedelta(hours=duration_hours)
        logger.info(f"Collecte des données de {start_time} à {end_time}")
        
        # Génération de données simulées
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        historical_data = {
            'time_range': {
                'start': start_time,
                'end': end_time,
                'duration_hours': duration_hours
            },
            'metrics_time_series': {},
            'events': [],
            'user_behaviors': [],
            'system_performance': []
        }
        
        # Génération de séries temporelles de métriques
        for metric_name in ['cpu_usage', 'memory_usage', 'response_time', 'error_rate']:
            base_value = np.random.uniform(20, 80)
            trend = np.random.uniform(-0.1, 0.1)
            noise_level = np.random.uniform(5, 15)
            
            values = []
            for i, ts in enumerate(timestamps):
                value = base_value + trend * i + np.random.normal(0, noise_level)
                values.append(max(0, min(100, value)))  # Clamp entre 0 et 100
            
            historical_data['metrics_time_series'][metric_name] = {
                'timestamps': [ts.isoformat() for ts in timestamps],
                'values': values,
                'unit': '%' if 'usage' in metric_name or 'rate' in metric_name else 'ms'
            }
        
        return historical_data
    
    async def _run_detection_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute le pipeline de détection complet"""
        results = {
            'timestamp': data['timestamp'],
            'ml_detection_results': {},
            'threshold_detection_results': {},
            'pattern_detection_results': {},
            'security_detection_results': {},
            'performance_analysis_results': {},
            'consolidated_alerts': []
        }
        
        try:
            # Détection ML
            if 'music_anomaly' in self.detectors:
                # Préparer les données pour la détection ML
                ml_features = self._prepare_ml_features(data)
                if ml_features is not None:
                    ml_results = await self.detectors['music_anomaly'].detect_anomalies(ml_features)
                    results['ml_detection_results']['music_anomaly'] = [
                        {
                            'is_anomaly': r.is_anomaly,
                            'confidence': r.confidence_score,
                            'type': r.anomaly_type,
                            'timestamp': r.timestamp.isoformat(),
                            'recommendation': r.recommendation
                        } for r in ml_results
                    ]
            
            # Détection de seuils
            for detector_name, detector in self.detectors.items():
                if 'threshold' in detector_name and hasattr(detector, 'detect'):
                    metric_value = self._extract_metric_for_detector(data, detector_name)
                    if metric_value is not None:
                        threshold_result = await detector.detect(metric_value)
                        results['threshold_detection_results'][detector_name] = {
                            'is_anomaly': threshold_result.is_anomaly,
                            'confidence': threshold_result.confidence,
                            'method': threshold_result.method,
                            'threshold_used': threshold_result.threshold_used,
                            'statistic_value': threshold_result.statistic_value
                        }
            
            # Analyse de patterns
            if 'sequence_analyzer' in self.detectors:
                # Analyser les séquences d'activités utilisateur
                user_sequences = self._extract_user_sequences(data)
                for sequence in user_sequences:
                    pattern_result = self.detectors['sequence_analyzer'].detect_anomalous_sequence(sequence)
                    if pattern_result.pattern_detected:
                        results['pattern_detection_results'][f"sequence_{len(results['pattern_detection_results'])}"] = {
                            'detected': pattern_result.pattern_detected,
                            'confidence': pattern_result.confidence_score,
                            'description': pattern_result.pattern_description,
                            'indicators': pattern_result.anomaly_indicators
                        }
            
            # Détection de sécurité
            if 'security_detector' in self.detectors:
                for security_event in data.get('security_events', []):
                    security_result = await self.detectors['security_detector'].detect_security_patterns(security_event)
                    if security_result.pattern_detected:
                        results['security_detection_results'][f"security_{len(results['security_detection_results'])}"] = {
                            'detected': security_result.pattern_detected,
                            'confidence': security_result.confidence_score,
                            'risk_level': security_result.risk_level,
                            'indicators': security_result.anomaly_indicators,
                            'recommendations': security_result.recommended_actions
                        }
            
            # Analyse de performance
            if 'performance_analyzer' in self.detectors:
                perf_analysis = await self.detectors['performance_analyzer'].analyze_performance()
                results['performance_analysis_results'] = perf_analysis
            
            # Consolidation des alertes
            results['consolidated_alerts'] = self._consolidate_alerts(results)
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline de détection: {e}")
            results['error'] = str(e)
        
        return results
    
    def _prepare_ml_features(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prépare les features pour la détection ML"""
        try:
            app_metrics = data.get('application_metrics', {})
            
            features = [
                app_metrics.get('response_time_ms', 0),
                app_metrics.get('requests_per_second', 0),
                app_metrics.get('error_rate_percent', 0),
                app_metrics.get('active_users', 0),
                len(data.get('user_activities', [])),
                len(data.get('security_events', [])),
                datetime.now().hour,  # Feature temporelle
                datetime.now().weekday(),  # Feature jour de la semaine
            ]
            
            # Ajouter des features de bruit pour la démonstration
            features.extend(np.random.normal(0, 1, 7))  # 7 features supplémentaires
            
            return np.array([features])  # Reshape pour un seul échantillon
            
        except Exception as e:
            logger.error(f"Erreur préparation features ML: {e}")
            return None
    
    def _extract_metric_for_detector(self, data: Dict[str, Any], detector_name: str) -> Optional[float]:
        """Extrait la métrique appropriée pour un détecteur"""
        app_metrics = data.get('application_metrics', {})
        
        metric_mapping = {
            'cpu_threshold': app_metrics.get('cpu_usage_percent', np.random.uniform(20, 80)),
            'memory_threshold': app_metrics.get('memory_usage_percent', np.random.uniform(30, 90)),
            'latency_threshold': app_metrics.get('response_time_ms', np.random.uniform(100, 500))
        }
        
        return metric_mapping.get(detector_name)
    
    def _extract_user_sequences(self, data: Dict[str, Any]) -> List[List[str]]:
        """Extrait les séquences d'activités utilisateur"""
        sequences = []
        user_activities = data.get('user_activities', [])
        
        # Grouper par utilisateur
        user_actions = {}
        for activity in user_activities:
            user_id = activity['user_id']
            if user_id not in user_actions:
                user_actions[user_id] = []
            user_actions[user_id].append(activity['action'])
        
        # Convertir en séquences
        for user_id, actions in user_actions.items():
            if len(actions) >= 2:  # Au moins 2 actions pour une séquence
                sequences.append(actions)
        
        return sequences
    
    def _consolidate_alerts(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consolide toutes les alertes en une liste unifiée"""
        consolidated = []
        
        # Alertes ML
        for detector_name, detector_results in results.get('ml_detection_results', {}).items():
            if isinstance(detector_results, list):
                for result in detector_results:
                    if result.get('is_anomaly'):
                        consolidated.append({
                            'type': 'ml_anomaly',
                            'detector': detector_name,
                            'confidence': result.get('confidence', 0),
                            'severity': self._calculate_severity(result.get('confidence', 0)),
                            'message': f"Anomalie ML détectée par {detector_name}",
                            'recommendation': result.get('recommendation', ''),
                            'timestamp': result.get('timestamp')
                        })
        
        # Alertes de seuils
        for detector_name, result in results.get('threshold_detection_results', {}).items():
            if result.get('is_anomaly'):
                consolidated.append({
                    'type': 'threshold_violation',
                    'detector': detector_name,
                    'confidence': result.get('confidence', 0),
                    'severity': self._calculate_severity(result.get('confidence', 0)),
                    'message': f"Seuil dépassé pour {detector_name}",
                    'threshold': result.get('threshold_used'),
                    'value': result.get('statistic_value'),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Alertes de patterns
        for pattern_name, result in results.get('pattern_detection_results', {}).items():
            if result.get('detected'):
                consolidated.append({
                    'type': 'pattern_anomaly',
                    'detector': pattern_name,
                    'confidence': result.get('confidence', 0),
                    'severity': self._calculate_severity(result.get('confidence', 0)),
                    'message': result.get('description', 'Pattern anormal détecté'),
                    'indicators': result.get('indicators', []),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Alertes de sécurité
        for security_name, result in results.get('security_detection_results', {}).items():
            if result.get('detected'):
                consolidated.append({
                    'type': 'security_threat',
                    'detector': security_name,
                    'confidence': result.get('confidence', 0),
                    'severity': result.get('risk_level', 'medium'),
                    'message': 'Menace de sécurité détectée',
                    'indicators': result.get('indicators', []),
                    'recommendations': result.get('recommendations', []),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Alertes de performance
        perf_alerts = results.get('performance_analysis_results', {}).get('alerts', [])
        for alert in perf_alerts:
            consolidated.append({
                'type': 'performance_issue',
                'detector': 'performance_analyzer',
                'confidence': 1.0,  # Les alertes de performance sont déterministes
                'severity': getattr(alert, 'severity', 'medium'),
                'message': getattr(alert, 'message', 'Problème de performance détecté'),
                'metric': getattr(alert, 'metric_name', ''),
                'current_value': getattr(alert, 'current_value', 0),
                'threshold_value': getattr(alert, 'threshold_value', 0),
                'recommendations': getattr(alert, 'recommendations', []),
                'timestamp': getattr(alert, 'timestamp', datetime.now()).isoformat()
            })
        
        # Trier par sévérité et confiance
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
        consolidated.sort(
            key=lambda x: (severity_order.get(x['severity'], 0), x['confidence']), 
            reverse=True
        )
        
        return consolidated
    
    def _calculate_severity(self, confidence: float) -> str:
        """Calcule la sévérité basée sur la confiance"""
        thresholds = self.config['alert_thresholds']
        
        if confidence >= thresholds['critical']:
            return 'critical'
        elif confidence >= thresholds['high']:
            return 'high'
        elif confidence >= thresholds['medium']:
            return 'medium'
        elif confidence >= thresholds['low']:
            return 'low'
        else:
            return 'info'
    
    async def _process_analysis_results(self, results: Dict[str, Any]):
        """Traite et stocke les résultats d'analyse"""
        try:
            # Sauvegarder dans le cache
            timestamp = results['timestamp']
            cache_key = timestamp.strftime("%Y%m%d_%H%M%S")
            self.results_cache[cache_key] = results
            
            # Nettoyer le cache (garder seulement les 1000 derniers)
            if len(self.results_cache) > 1000:
                oldest_key = min(self.results_cache.keys())
                del self.results_cache[oldest_key]
            
            # Log des alertes importantes
            alerts = results.get('consolidated_alerts', [])
            critical_alerts = [a for a in alerts if a['severity'] == 'critical']
            
            if critical_alerts:
                logger.warning(f"{len(critical_alerts)} alertes critiques détectées")
                for alert in critical_alerts[:3]:  # Log des 3 premières
                    logger.warning(f"CRITIQUE: {alert['message']}")
            
            # Exportation vers différents formats
            await self._export_results(results)
            
        except Exception as e:
            logger.error(f"Erreur traitement résultats: {e}")
    
    async def _export_results(self, results: Dict[str, Any]):
        """Exporte les résultats vers différents formats"""
        timestamp = results['timestamp']
        
        try:
            # Export JSON
            if 'json' in self.config['output_formats']:
                json_path = f"/tmp/analytics_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.debug(f"Résultats exportés vers {json_path}")
            
            # Export Prometheus (simulation)
            if 'prometheus' in self.config['output_formats']:
                metrics_text = self._format_prometheus_metrics(results)
                # Dans un vrai système, ceci serait exposé via un endpoint HTTP
                logger.debug("Métriques Prometheus mises à jour")
            
            # Export Grafana (simulation)
            if 'grafana' in self.config['output_formats']:
                grafana_data = self._format_grafana_data(results)
                # Dans un vrai système, ceci serait envoyé à Grafana
                logger.debug("Données Grafana préparées")
                
        except Exception as e:
            logger.error(f"Erreur export résultats: {e}")
    
    def _format_prometheus_metrics(self, results: Dict[str, Any]) -> str:
        """Formate les résultats pour Prometheus"""
        metrics_lines = []
        
        # Métriques d'alertes
        total_alerts = len(results.get('consolidated_alerts', []))
        metrics_lines.append(f"spotify_ai_agent_total_alerts {total_alerts}")
        
        # Métriques par sévérité
        severity_counts = {}
        for alert in results.get('consolidated_alerts', []):
            severity = alert.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            metrics_lines.append(f'spotify_ai_agent_alerts_by_severity{{severity="{severity}"}} {count}')
        
        # Métriques de performance
        perf_results = results.get('performance_analysis_results', {})
        if 'overall_health' in perf_results:
            health_score = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4, 'critical': 0.2}.get(
                perf_results['overall_health'], 0.0
            )
            metrics_lines.append(f"spotify_ai_agent_system_health_score {health_score}")
        
        return '\n'.join(metrics_lines)
    
    def _format_grafana_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Formate les données pour Grafana"""
        timestamp = int(results['timestamp'].timestamp() * 1000)  # Milliseconds
        
        grafana_data = {
            'dashboard': 'spotify-ai-agent-monitoring',
            'timestamp': timestamp,
            'metrics': {
                'alerts': {
                    'total': len(results.get('consolidated_alerts', [])),
                    'by_severity': {},
                    'by_type': {}
                },
                'performance': {},
                'detection_rates': {}
            }
        }
        
        # Agrégation des alertes
        for alert in results.get('consolidated_alerts', []):
            severity = alert.get('severity', 'unknown')
            alert_type = alert.get('type', 'unknown')
            
            grafana_data['metrics']['alerts']['by_severity'][severity] = \
                grafana_data['metrics']['alerts']['by_severity'].get(severity, 0) + 1
            
            grafana_data['metrics']['alerts']['by_type'][alert_type] = \
                grafana_data['metrics']['alerts']['by_type'].get(alert_type, 0) + 1
        
        return grafana_data
    
    async def _update_analytics_stats(self, processing_time: float, results: Dict[str, Any]):
        """Met à jour les statistiques d'analytics"""
        self.analytics_stats['total_analyses'] += 1
        self.analytics_stats['last_run'] = datetime.now()
        
        # Mise à jour du temps de traitement moyen
        current_avg = self.analytics_stats['avg_processing_time']
        total_runs = self.analytics_stats['total_analyses']
        new_avg = (current_avg * (total_runs - 1) + processing_time) / total_runs
        self.analytics_stats['avg_processing_time'] = new_avg
        
        # Compter les anomalies
        anomalies_count = len(results.get('consolidated_alerts', []))
        self.analytics_stats['anomalies_detected'] += anomalies_count
        
        logger.debug(f"Stats mises à jour - Analyses: {total_runs}, Temps moyen: {new_avg:.2f}s")
    
    async def _run_deep_analysis(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute une analyse approfondie sur les données historiques"""
        logger.info("Démarrage de l'analyse approfondie")
        
        deep_results = {
            'analysis_type': 'deep_historical',
            'time_range': historical_data['time_range'],
            'trend_analysis': {},
            'correlation_analysis': {},
            'anomaly_patterns': {},
            'performance_insights': {},
            'recommendations': []
        }
        
        try:
            # Analyse des tendances
            metrics_ts = historical_data.get('metrics_time_series', {})
            for metric_name, metric_data in metrics_ts.items():
                values = metric_data['values']
                timestamps = pd.to_datetime(metric_data['timestamps'])
                
                # Analyse de tendance
                trend_analysis = self._analyze_metric_trend(values, timestamps)
                deep_results['trend_analysis'][metric_name] = trend_analysis
                
                # Détection de patterns cycliques
                cyclical_patterns = self._detect_cyclical_patterns(values, timestamps)
                deep_results['anomaly_patterns'][metric_name] = cyclical_patterns
            
            # Analyse de corrélation entre métriques
            if len(metrics_ts) > 1:
                correlation_matrix = self._calculate_metric_correlations(metrics_ts)
                deep_results['correlation_analysis'] = correlation_matrix
            
            # Insights de performance
            deep_results['performance_insights'] = self._generate_performance_insights(deep_results)
            
            # Recommandations basées sur l'analyse
            deep_results['recommendations'] = self._generate_deep_recommendations(deep_results)
            
        except Exception as e:
            logger.error(f"Erreur analyse approfondie: {e}")
            deep_results['error'] = str(e)
        
        return deep_results
    
    def _analyze_metric_trend(self, values: List[float], timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """Analyse la tendance d'une métrique"""
        try:
            # Régression linéaire
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # Direction de la tendance
            if slope > 0.1:
                direction = 'increasing'
            elif slope < -0.1:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Variabilité
            variability = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            # Détection de changements de niveau
            window_size = len(values) // 4
            if window_size > 1:
                first_quarter = np.mean(values[:window_size])
                last_quarter = np.mean(values[-window_size:])
                level_change = (last_quarter - first_quarter) / first_quarter if first_quarter != 0 else 0
            else:
                level_change = 0
            
            return {
                'direction': direction,
                'slope': slope,
                'variability': variability,
                'level_change_percent': level_change * 100,
                'min_value': min(values),
                'max_value': max(values),
                'mean_value': np.mean(values),
                'median_value': np.median(values)
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_cyclical_patterns(self, values: List[float], timestamps: pd.DatetimeIndex) -> Dict[str, Any]:
        """Détecte les patterns cycliques dans les données"""
        try:
            # FFT pour détecter les fréquences dominantes
            fft = np.fft.fft(values)
            freqs = np.fft.fftfreq(len(values))
            
            # Trouver les pics de fréquence
            power_spectrum = np.abs(fft)
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_frequency = abs(freqs[dominant_freq_idx])
            
            # Période estimée (en nombre de points)
            if dominant_frequency > 0:
                estimated_period = 1 / dominant_frequency
            else:
                estimated_period = len(values)
            
            # Autocorrélation pour confirmer la périodicité
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            return {
                'has_cyclical_pattern': dominant_frequency > 0,
                'dominant_frequency': dominant_frequency,
                'estimated_period_points': estimated_period,
                'max_autocorrelation': max(autocorr[1:]) if len(autocorr) > 1 else 0,
                'pattern_strength': power_spectrum[dominant_freq_idx] / sum(power_spectrum)
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_metric_correlations(self, metrics_ts: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les corrélations entre métriques"""
        try:
            # Créer un DataFrame avec toutes les métriques
            df_data = {}
            min_length = min(len(data['values']) for data in metrics_ts.values())
            
            for metric_name, metric_data in metrics_ts.items():
                df_data[metric_name] = metric_data['values'][:min_length]
            
            df = pd.DataFrame(df_data)
            correlation_matrix = df.corr()
            
            # Identifier les corrélations significatives
            significant_correlations = {}
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    metric1 = correlation_matrix.columns[i]
                    metric2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.7:  # Seuil de corrélation significative
                        significant_correlations[f"{metric1}_vs_{metric2}"] = {
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate',
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        }
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'significant_correlations': significant_correlations,
                'highly_correlated_pairs': len(significant_correlations)
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_performance_insights(self, deep_results: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des insights de performance"""
        insights = {
            'system_stability': 'unknown',
            'performance_trends': {},
            'risk_indicators': [],
            'optimization_opportunities': []
        }
        
        try:
            # Évaluer la stabilité du système
            trend_analysis = deep_results.get('trend_analysis', {})
            
            unstable_metrics = 0
            total_metrics = len(trend_analysis)
            
            for metric_name, analysis in trend_analysis.items():
                if isinstance(analysis, dict) and 'variability' in analysis:
                    variability = analysis['variability']
                    direction = analysis.get('direction', 'stable')
                    
                    # Considérer comme instable si haute variabilité ou tendance croissante
                    if variability > 0.3 or (direction == 'increasing' and 'usage' in metric_name):
                        unstable_metrics += 1
                        insights['risk_indicators'].append(f"{metric_name}: {direction} avec variabilité {variability:.2f}")
                    
                    # Identifier les opportunités d'optimisation
                    if direction == 'increasing' and analysis.get('level_change_percent', 0) > 20:
                        insights['optimization_opportunities'].append(
                            f"Optimiser {metric_name} - augmentation de {analysis['level_change_percent']:.1f}%"
                        )
            
            # Score de stabilité global
            if total_metrics > 0:
                stability_ratio = 1 - (unstable_metrics / total_metrics)
                if stability_ratio > 0.8:
                    insights['system_stability'] = 'excellent'
                elif stability_ratio > 0.6:
                    insights['system_stability'] = 'good'
                elif stability_ratio > 0.4:
                    insights['system_stability'] = 'fair'
                else:
                    insights['system_stability'] = 'poor'
            
            # Résumé des tendances de performance
            for metric_name, analysis in trend_analysis.items():
                if isinstance(analysis, dict):
                    insights['performance_trends'][metric_name] = {
                        'direction': analysis.get('direction', 'unknown'),
                        'change_magnitude': abs(analysis.get('level_change_percent', 0)),
                        'stability_score': 1 - analysis.get('variability', 0)
                    }
        
        except Exception as e:
            insights['error'] = str(e)
        
        return insights
    
    def _generate_deep_recommendations(self, deep_results: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur l'analyse approfondie"""
        recommendations = []
        
        try:
            # Recommandations basées sur les insights de performance
            performance_insights = deep_results.get('performance_insights', {})
            
            stability = performance_insights.get('system_stability', 'unknown')
            if stability in ['poor', 'fair']:
                recommendations.append(
                    f"Système {stability}: Investiguer les métriques instables et planifier des optimisations"
                )
            
            # Recommandations pour les risques identifiés
            risk_indicators = performance_insights.get('risk_indicators', [])
            if risk_indicators:
                recommendations.append(
                    f"Surveiller {len(risk_indicators)} indicateurs de risque identifiés"
                )
            
            # Recommandations pour les opportunités d'optimisation
            optimization_opportunities = performance_insights.get('optimization_opportunities', [])
            if optimization_opportunities:
                recommendations.extend(optimization_opportunities[:3])  # Top 3
            
            # Recommandations basées sur les corrélations
            correlation_analysis = deep_results.get('correlation_analysis', {})
            significant_correlations = correlation_analysis.get('significant_correlations', {})
            
            if significant_correlations:
                recommendations.append(
                    f"Analyser {len(significant_correlations)} corrélations significatives pour l'optimisation"
                )
            
            # Recommandations par défaut
            if not recommendations:
                recommendations = [
                    "Continuer la surveillance continue",
                    "Maintenir les seuils d'alerte actuels",
                    "Planifier une révision mensuelle des performances"
                ]
        
        except Exception as e:
            recommendations.append(f"Erreur génération recommandations: {e}")
        
        return recommendations
    
    async def _generate_comprehensive_reports(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des rapports complets"""
        reports = {
            'executive_summary': {},
            'technical_details': {},
            'trend_report': {},
            'recommendations_report': {},
            'generated_at': datetime.now().isoformat()
        }
        
        try:
            # Résumé exécutif
            reports['executive_summary'] = {
                'analysis_period': analysis_results.get('time_range', {}),
                'system_stability': analysis_results.get('performance_insights', {}).get('system_stability', 'unknown'),
                'total_risk_indicators': len(analysis_results.get('performance_insights', {}).get('risk_indicators', [])),
                'optimization_opportunities': len(analysis_results.get('performance_insights', {}).get('optimization_opportunities', [])),
                'key_findings': self._extract_key_findings(analysis_results),
                'action_required': self._determine_action_required(analysis_results)
            }
            
            # Détails techniques
            reports['technical_details'] = {
                'trend_analysis': analysis_results.get('trend_analysis', {}),
                'correlation_analysis': analysis_results.get('correlation_analysis', {}),
                'anomaly_patterns': analysis_results.get('anomaly_patterns', {}),
                'performance_metrics': analysis_results.get('performance_insights', {})
            }
            
            # Rapport de tendances
            reports['trend_report'] = self._generate_trend_report(analysis_results)
            
            # Rapport de recommandations
            reports['recommendations_report'] = {
                'immediate_actions': [],
                'short_term_improvements': [],
                'long_term_strategy': [],
                'all_recommendations': analysis_results.get('recommendations', [])
            }
            
            # Catégoriser les recommandations
            all_recommendations = analysis_results.get('recommendations', [])
            for rec in all_recommendations:
                if 'urgent' in rec.lower() or 'critique' in rec.lower():
                    reports['recommendations_report']['immediate_actions'].append(rec)
                elif 'optimiser' in rec.lower() or 'améliorer' in rec.lower():
                    reports['recommendations_report']['short_term_improvements'].append(rec)
                else:
                    reports['recommendations_report']['long_term_strategy'].append(rec)
        
        except Exception as e:
            reports['error'] = str(e)
        
        return reports
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Extrait les découvertes clés de l'analyse"""
        findings = []
        
        try:
            # Analyse des tendances
            trend_analysis = analysis_results.get('trend_analysis', {})
            increasing_metrics = [name for name, data in trend_analysis.items() 
                                if isinstance(data, dict) and data.get('direction') == 'increasing']
            
            if increasing_metrics:
                findings.append(f"{len(increasing_metrics)} métriques en augmentation: {', '.join(increasing_metrics[:3])}")
            
            # Corrélations significatives
            correlation_analysis = analysis_results.get('correlation_analysis', {})
            significant_correlations = correlation_analysis.get('significant_correlations', {})
            
            if significant_correlations:
                findings.append(f"{len(significant_correlations)} corrélations significatives détectées")
            
            # Patterns cycliques
            anomaly_patterns = analysis_results.get('anomaly_patterns', {})
            cyclical_metrics = [name for name, data in anomaly_patterns.items() 
                              if isinstance(data, dict) and data.get('has_cyclical_pattern')]
            
            if cyclical_metrics:
                findings.append(f"Patterns cycliques détectés dans: {', '.join(cyclical_metrics[:2])}")
            
            # Stabilité du système
            stability = analysis_results.get('performance_insights', {}).get('system_stability')
            if stability and stability != 'unknown':
                findings.append(f"Stabilité du système: {stability}")
        
        except Exception as e:
            findings.append(f"Erreur extraction findings: {e}")
        
        return findings or ["Aucune découverte significative"]
    
    def _determine_action_required(self, analysis_results: Dict[str, Any]) -> str:
        """Détermine le niveau d'action requis"""
        try:
            stability = analysis_results.get('performance_insights', {}).get('system_stability', 'unknown')
            risk_count = len(analysis_results.get('performance_insights', {}).get('risk_indicators', []))
            
            if stability == 'poor' or risk_count > 5:
                return 'immediate'
            elif stability == 'fair' or risk_count > 2:
                return 'urgent'
            elif stability == 'good':
                return 'routine'
            else:
                return 'monitoring'
        
        except:
            return 'unknown'
    
    def _generate_trend_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un rapport de tendances détaillé"""
        trend_report = {
            'summary': {},
            'by_metric': {},
            'predictions': {},
            'alerts': []
        }
        
        try:
            trend_analysis = analysis_results.get('trend_analysis', {})
            
            # Résumé des tendances
            directions = [data.get('direction', 'unknown') for data in trend_analysis.values() 
                         if isinstance(data, dict)]
            
            trend_report['summary'] = {
                'total_metrics': len(trend_analysis),
                'increasing': directions.count('increasing'),
                'decreasing': directions.count('decreasing'),
                'stable': directions.count('stable')
            }
            
            # Détails par métrique
            for metric_name, analysis in trend_analysis.items():
                if isinstance(analysis, dict):
                    trend_report['by_metric'][metric_name] = {
                        'direction': analysis.get('direction', 'unknown'),
                        'slope': analysis.get('slope', 0),
                        'variability': analysis.get('variability', 0),
                        'level_change': analysis.get('level_change_percent', 0),
                        'prediction_confidence': self._calculate_prediction_confidence(analysis)
                    }
                    
                    # Prédictions simples
                    current_mean = analysis.get('mean_value', 0)
                    slope = analysis.get('slope', 0)
                    
                    trend_report['predictions'][metric_name] = {
                        'next_hour': current_mean + slope * 60,
                        'next_day': current_mean + slope * 60 * 24,
                        'next_week': current_mean + slope * 60 * 24 * 7
                    }
                    
                    # Alertes de tendance
                    if abs(analysis.get('level_change_percent', 0)) > 50:
                        trend_report['alerts'].append(f"{metric_name}: Changement de niveau important ({analysis.get('level_change_percent', 0):.1f}%)")
        
        except Exception as e:
            trend_report['error'] = str(e)
        
        return trend_report
    
    def _calculate_prediction_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calcule la confiance dans les prédictions"""
        try:
            variability = analysis.get('variability', 1.0)
            slope_magnitude = abs(analysis.get('slope', 0))
            
            # Confiance plus élevée si faible variabilité et tendance claire
            base_confidence = 1 / (1 + variability)
            trend_confidence = min(slope_magnitude * 10, 1.0)
            
            return (base_confidence + trend_confidence) / 2
        
        except:
            return 0.5
    
    async def _save_analysis_results(self, reports: Dict[str, Any], start_time: datetime):
        """Sauvegarde les résultats d'analyse"""
        try:
            # Créer le répertoire de sauvegarde
            save_dir = Path("/tmp/analytics_reports")
            save_dir.mkdir(exist_ok=True)
            
            # Nom de fichier basé sur la date
            filename = f"analytics_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = save_dir / filename
            
            # Sauvegarder
            with open(filepath, 'w') as f:
                json.dump(reports, f, indent=2, default=str)
            
            logger.info(f"Rapport sauvegardé: {filepath}")
            
            # Nettoyage des anciens rapports (garder seulement les 30 derniers)
            report_files = sorted(save_dir.glob("analytics_report_*.json"))
            if len(report_files) > 30:
                for old_file in report_files[:-30]:
                    old_file.unlink()
                    logger.debug(f"Ancien rapport supprimé: {old_file}")
        
        except Exception as e:
            logger.error(f"Erreur sauvegarde rapport: {e}")
    
    async def get_current_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel du système d'analytics"""
        return {
            'orchestrator_stats': self.analytics_stats,
            'active_detectors': list(self.detectors.keys()),
            'active_collectors': list(self.collectors.keys()),
            'cache_size': len(self.results_cache),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        logger.info("Nettoyage des ressources...")
        
        try:
            # Arrêter les collecteurs
            for collector in self.collectors.values():
                if hasattr(collector, 'stop_collection'):
                    await collector.stop_collection()
            
            # Nettoyer le cache
            self.results_cache.clear()
            
            logger.info("Nettoyage terminé")
        
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")

async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Analytics Avancé pour Détection d\'Anomalies')
    parser.add_argument('--mode', choices=['realtime', 'batch', 'status'], default='realtime',
                       help='Mode d\'exécution')
    parser.add_argument('--config', help='Chemin vers le fichier de configuration')
    parser.add_argument('--duration', type=int, default=1, 
                       help='Durée en heures pour l\'analyse par lot')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mode verbeux')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialisation de l'orchestrateur
    orchestrator = AnalyticsOrchestrator(args.config)
    
    try:
        await orchestrator.initialize()
        
        if args.mode == 'realtime':
            logger.info("Démarrage en mode temps réel")
            await orchestrator.run_real_time_analysis()
        
        elif args.mode == 'batch':
            logger.info(f"Démarrage de l'analyse par lot ({args.duration}h)")
            reports = await orchestrator.run_batch_analysis(duration_hours=args.duration)
            print(json.dumps(reports, indent=2, default=str))
        
        elif args.mode == 'status':
            status = await orchestrator.get_current_status()
            print(json.dumps(status, indent=2, default=str))
    
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        return 1
    
    finally:
        await orchestrator.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
