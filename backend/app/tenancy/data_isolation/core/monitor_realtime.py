#!/usr/bin/env python3
"""
📊 Real-time Context Monitoring Dashboard Script
===============================================

Système de monitoring en temps réel pour le gestionnaire de contexte
avec métriques live, alertes et tableau de bord interactif.

Author: Lead Dev + Architecte IA - Fahed Mlaiel
"""

import asyncio
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import argparse
from dataclasses import dataclass, field

# Ajouter le chemin du module au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from app.tenancy.data_isolation.core import (
    ContextManager,
    PerformanceOptimizer,
    SecurityPolicyEngine,
    ComplianceEngine
)


@dataclass
class MetricPoint:
    """Point de métrique avec timestamp"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alerte système"""
    id: str
    level: str  # info, warning, error, critical
    message: str
    timestamp: datetime
    component: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """Collecteur de métriques en temps réel"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.alerts: List[Alert] = []
        self.thresholds = self._initialize_thresholds()
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialise les seuils d'alerte"""
        return {
            'context_switch_time_ms': {'warning': 5.0, 'error': 10.0, 'critical': 20.0},
            'memory_usage_mb': {'warning': 1024.0, 'error': 2048.0, 'critical': 4096.0},
            'cpu_usage_percent': {'warning': 70.0, 'error': 85.0, 'critical': 95.0},
            'cache_hit_ratio': {'warning': 0.7, 'error': 0.5, 'critical': 0.3},
            'active_contexts': {'warning': 100, 'error': 500, 'critical': 1000},
            'error_rate': {'warning': 0.01, 'error': 0.05, 'critical': 0.1}
        }
    
    def add_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Ajoute une métrique"""
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            tags=tags or {}
        )
        
        self.metrics[name].append(point)
        self._check_thresholds(name, value)
    
    def _check_thresholds(self, metric_name: str, value: float):
        """Vérifie les seuils et génère des alertes"""
        if metric_name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_name]
        
        # Détermination du niveau d'alerte
        alert_level = None
        if value >= thresholds.get('critical', float('inf')):
            alert_level = 'critical'
        elif value >= thresholds.get('error', float('inf')):
            alert_level = 'error'
        elif value >= thresholds.get('warning', float('inf')):
            alert_level = 'warning'
        
        # Cas spéciaux pour les métriques inversées (plus bas = pire)
        if metric_name == 'cache_hit_ratio':
            if value <= thresholds.get('critical', 0):
                alert_level = 'critical'
            elif value <= thresholds.get('error', 0):
                alert_level = 'error'
            elif value <= thresholds.get('warning', 0):
                alert_level = 'warning'
        
        if alert_level:
            self._create_alert(alert_level, metric_name, value)
    
    def _create_alert(self, level: str, metric_name: str, value: float):
        """Crée une nouvelle alerte"""
        alert_id = f"{metric_name}_{level}_{int(time.time())}"
        
        # Vérifier si une alerte similaire existe déjà
        for alert in self.alerts:
            if (alert.component == metric_name and 
                alert.level == level and 
                not alert.resolved and
                (datetime.now(timezone.utc) - alert.timestamp) < timedelta(minutes=5)):
                return  # Éviter le spam d'alertes
        
        alert = Alert(
            id=alert_id,
            level=level,
            message=f"{metric_name} {level}: {value}",
            timestamp=datetime.now(timezone.utc),
            component=metric_name
        )
        
        self.alerts.append(alert)
        print(f"🚨 {level.upper()}: {alert.message}")
    
    def get_recent_metrics(self, metric_name: str, duration_minutes: int = 5) -> List[MetricPoint]:
        """Récupère les métriques récentes"""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        return [point for point in self.metrics[metric_name] if point.timestamp >= cutoff_time]
    
    def get_metric_stats(self, metric_name: str, duration_minutes: int = 5) -> Dict[str, float]:
        """Calcule les statistiques d'une métrique"""
        recent_points = self.get_recent_metrics(metric_name, duration_minutes)
        
        if not recent_points:
            return {'count': 0}
        
        values = [point.value for point in recent_points]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0
        }


class ContextMonitor:
    """Moniteur principal du système de contexte"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.context_manager = None
        self.performance_optimizer = None
        self.security_engine = None
        self.compliance_engine = None
        
        # État du monitoring
        self.running = False
        self.monitoring_interval = 1.0  # secondes
        
        # Statistiques
        self.start_time = None
        self.total_collections = 0
    
    async def initialize_components(self):
        """Initialise les composants à surveiller"""
        try:
            self.context_manager = ContextManager()
            self.performance_optimizer = PerformanceOptimizer()
            self.security_engine = SecurityPolicyEngine()
            self.compliance_engine = ComplianceEngine()
            print("✅ All components initialized successfully")
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            raise
    
    async def collect_metrics(self):
        """Collecte toutes les métriques"""
        try:
            # Métriques du gestionnaire de contexte
            if self.context_manager:
                ctx_stats = self.context_manager.get_statistics()
                self.metrics_collector.add_metric('active_contexts', ctx_stats.get('active_contexts', 0))
                self.metrics_collector.add_metric('contexts_managed', ctx_stats.get('contexts_managed', 0))
                self.metrics_collector.add_metric('switcher_switches_total', 
                                                ctx_stats.get('switcher_stats', {}).get('switches_total', 0))
                
                # Temps de basculement moyen
                avg_switch_time = ctx_stats.get('switcher_stats', {}).get('avg_switch_time_ms', 0)
                self.metrics_collector.add_metric('context_switch_time_ms', avg_switch_time)
            
            # Métriques de performance
            if self.performance_optimizer:
                perf_stats = self.performance_optimizer.get_statistics()
                self.metrics_collector.add_metric('cache_hit_ratio', perf_stats.get('cache_hit_ratio', 0))
                self.metrics_collector.add_metric('cache_size_mb', perf_stats.get('cache_size_mb', 0))
                self.metrics_collector.add_metric('optimizations_applied', perf_stats.get('optimizations_applied', 0))
                self.metrics_collector.add_metric('performance_improvements', perf_stats.get('performance_improvements', 0))
            
            # Métriques de sécurité
            if self.security_engine:
                sec_stats = self.security_engine.get_statistics()
                self.metrics_collector.add_metric('policies_enforced', sec_stats.get('policies_enforced', 0))
                self.metrics_collector.add_metric('violations_blocked', sec_stats.get('violations_blocked', 0))
                self.metrics_collector.add_metric('threats_detected', sec_stats.get('threats_detected', 0))
                
                # Taux d'erreur de sécurité
                total_evals = sec_stats.get('evaluations_total', 1)
                violations = sec_stats.get('violations_blocked', 0)
                error_rate = violations / total_evals if total_evals > 0 else 0
                self.metrics_collector.add_metric('error_rate', error_rate)
            
            # Métriques de conformité
            if self.compliance_engine:
                comp_stats = self.compliance_engine.get_statistics()
                self.metrics_collector.add_metric('compliance_evaluations', comp_stats.get('evaluations_total', 0))
                self.metrics_collector.add_metric('compliance_violations', comp_stats.get('violations_detected', 0))
                self.metrics_collector.add_metric('audits_generated', comp_stats.get('audits_generated', 0))
            
            # Métriques système
            await self._collect_system_metrics()
            
            self.total_collections += 1
            
        except Exception as e:
            print(f"❌ Metrics collection failed: {e}")
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics_collector.add_metric('cpu_usage_percent', cpu_percent)
            
            # Mémoire
            memory = psutil.virtual_memory()
            memory_mb = (memory.total - memory.available) / (1024 * 1024)
            self.metrics_collector.add_metric('memory_usage_mb', memory_mb)
            self.metrics_collector.add_metric('memory_usage_percent', memory.percent)
            
            # Processus actuel
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)
            self.metrics_collector.add_metric('process_memory_mb', process_memory)
            
        except ImportError:
            pass  # psutil n'est pas disponible
        except Exception as e:
            print(f"⚠️ System metrics collection failed: {e}")
    
    def display_dashboard(self):
        """Affiche le tableau de bord en temps réel"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎛️  SPOTIFY AI AGENT - CONTEXT MONITORING DASHBOARD")
        print("=" * 80)
        print(f"🕐 Uptime: {self._get_uptime()}")
        print(f"📊 Collections: {self.total_collections}")
        print(f"🔄 Interval: {self.monitoring_interval}s")
        print("")
        
        # Métriques principales
        self._display_key_metrics()
        print("")
        
        # Alertes actives
        self._display_active_alerts()
        print("")
        
        # Tendances
        self._display_trends()
        print("")
        
        print("Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    def _display_key_metrics(self):
        """Affiche les métriques clés"""
        print("📊 KEY METRICS")
        print("-" * 40)
        
        metrics_to_show = [
            ('Context Switch Time', 'context_switch_time_ms', 'ms'),
            ('Active Contexts', 'active_contexts', ''),
            ('Cache Hit Ratio', 'cache_hit_ratio', '%'),
            ('CPU Usage', 'cpu_usage_percent', '%'),
            ('Memory Usage', 'memory_usage_mb', 'MB'),
            ('Error Rate', 'error_rate', '%'),
        ]
        
        for label, metric_name, unit in metrics_to_show:
            stats = self.metrics_collector.get_metric_stats(metric_name)
            if stats.get('count', 0) > 0:
                latest = stats['latest']
                if unit == '%' and metric_name != 'error_rate':
                    latest *= 100
                elif unit == '%' and metric_name == 'error_rate':
                    latest *= 100
                
                # Indicateur de statut
                status = self._get_metric_status(metric_name, latest)
                status_emoji = {'ok': '🟢', 'warning': '🟡', 'error': '🔴', 'critical': '🔴'}
                
                print(f"{status_emoji.get(status, '⚪')} {label:<20}: {latest:>8.2f} {unit}")
            else:
                print(f"⚪ {label:<20}: {'N/A':>8} {unit}")
    
    def _display_active_alerts(self):
        """Affiche les alertes actives"""
        active_alerts = [alert for alert in self.metrics_collector.alerts if not alert.resolved]
        
        print(f"🚨 ACTIVE ALERTS ({len(active_alerts)})")
        print("-" * 40)
        
        if not active_alerts:
            print("✅ No active alerts")
            return
        
        # Grouper par niveau
        by_level = defaultdict(list)
        for alert in active_alerts[-10:]:  # Dernières 10 alertes
            by_level[alert.level].append(alert)
        
        level_emojis = {'critical': '🔴', 'error': '🟠', 'warning': '🟡', 'info': '🔵'}
        level_order = ['critical', 'error', 'warning', 'info']
        
        for level in level_order:
            if level in by_level:
                alerts = by_level[level]
                print(f"{level_emojis[level]} {level.upper()} ({len(alerts)}):")
                for alert in alerts[-3:]:  # Max 3 par niveau
                    age = (datetime.now(timezone.utc) - alert.timestamp).total_seconds()
                    print(f"   • {alert.message} ({age:.0f}s ago)")
    
    def _display_trends(self):
        """Affiche les tendances"""
        print("📈 TRENDS (5min)")
        print("-" * 40)
        
        trend_metrics = [
            'context_switch_time_ms',
            'cache_hit_ratio', 
            'cpu_usage_percent',
            'active_contexts'
        ]
        
        for metric in trend_metrics:
            recent_points = self.metrics_collector.get_recent_metrics(metric, 5)
            if len(recent_points) >= 2:
                first_value = recent_points[0].value
                last_value = recent_points[-1].value
                
                if first_value != 0:
                    change_percent = ((last_value - first_value) / first_value) * 100
                    trend_arrow = "↗️" if change_percent > 5 else "↘️" if change_percent < -5 else "→"
                    print(f"{trend_arrow} {metric:<25}: {change_percent:>+6.1f}%")
                else:
                    print(f"→ {metric:<25}: {'N/A':>6}")
    
    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Détermine le statut d'une métrique"""
        if metric_name not in self.metrics_collector.thresholds:
            return 'ok'
        
        thresholds = self.metrics_collector.thresholds[metric_name]
        
        if metric_name == 'cache_hit_ratio':
            # Métrique inversée
            if value <= thresholds.get('critical', 0):
                return 'critical'
            elif value <= thresholds.get('error', 0):
                return 'error'
            elif value <= thresholds.get('warning', 0):
                return 'warning'
            else:
                return 'ok'
        else:
            # Métrique normale
            if value >= thresholds.get('critical', float('inf')):
                return 'critical'
            elif value >= thresholds.get('error', float('inf')):
                return 'error'
            elif value >= thresholds.get('warning', float('inf')):
                return 'warning'
            else:
                return 'ok'
    
    def _get_uptime(self) -> str:
        """Calcule le temps de fonctionnement"""
        if not self.start_time:
            return "00:00:00"
        
        uptime = datetime.now(timezone.utc) - self.start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    async def start_monitoring(self):
        """Démarre le monitoring en temps réel"""
        print("🚀 Starting real-time context monitoring...")
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        try:
            await self.initialize_components()
            
            while self.running:
                # Collecte des métriques
                await self.collect_metrics()
                
                # Affichage du tableau de bord
                self.display_dashboard()
                
                # Attente avant la prochaine collecte
                await asyncio.sleep(self.monitoring_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        finally:
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """Arrête le monitoring"""
        self.running = False
        
        # Nettoyage des composants
        if self.context_manager:
            await self.context_manager.shutdown()
        
        print("✅ Monitoring stopped and components cleaned up")
    
    def export_metrics(self, output_file: str):
        """Exporte les métriques"""
        export_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0,
            'total_collections': self.total_collections,
            'metrics': {},
            'alerts': [
                {
                    'id': alert.id,
                    'level': alert.level,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'component': alert.component,
                    'resolved': alert.resolved
                }
                for alert in self.metrics_collector.alerts
            ]
        }
        
        # Export des métriques récentes
        for metric_name, points in self.metrics_collector.metrics.items():
            recent_points = list(points)[-100:]  # Derniers 100 points
            export_data['metrics'][metric_name] = [
                {
                    'timestamp': point.timestamp.isoformat(),
                    'value': point.value,
                    'tags': point.tags
                }
                for point in recent_points
            ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Metrics exported to: {output_file}")


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Real-time Context Monitoring')
    parser.add_argument('--interval', '-i', type=float, default=1.0, help='Monitoring interval in seconds')
    parser.add_argument('--export', '-e', help='Export metrics to file on exit')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (less output)')
    
    args = parser.parse_args()
    
    monitor = ContextMonitor()
    monitor.monitoring_interval = args.interval
    
    try:
        await monitor.start_monitoring()
    finally:
        if args.export:
            monitor.export_metrics(args.export)


if __name__ == '__main__':
    asyncio.run(main())
