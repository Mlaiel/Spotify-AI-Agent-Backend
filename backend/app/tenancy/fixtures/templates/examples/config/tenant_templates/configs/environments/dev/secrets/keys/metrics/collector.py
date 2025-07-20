#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Metrics Collection Agent
=================================

Ultra-advanced real-time metrics collection agent with intelligent sampling,
adaptive collection rates, and distributed monitoring capabilities.

Expert Development Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- ML Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import asyncio
import json
import logging
import os
import sys
import signal
import time
import socket
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import argparse
import configparser
import yaml

# Ajout du chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import (
    EnterpriseMetricsSystem, MetricDataPoint, MetricType, 
    MetricCategory, MetricSeverity, get_metrics_system
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import kubernetes
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class CollectionMode(Enum):
    """Modes de collecte de métriques."""
    NORMAL = "normal"
    HIGH_FREQUENCY = "high_frequency"
    LOW_IMPACT = "low_impact"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


@dataclass
class CollectorConfig:
    """Configuration du collecteur de métriques."""
    
    # Intervalles de collecte (secondes)
    system_interval: int = 60
    security_interval: int = 300
    performance_interval: int = 30
    network_interval: int = 120
    storage_interval: int = 180
    
    # Limites et seuils
    max_cpu_threshold: float = 95.0
    max_memory_threshold: float = 90.0
    max_disk_threshold: float = 85.0
    max_load_threshold: float = 10.0
    
    # Configuration réseau
    network_interfaces: List[str] = field(default_factory=lambda: ["eth0", "wlan0", "lo"])
    monitor_ports: List[int] = field(default_factory=lambda: [22, 80, 443, 8080])
    
    # Configuration stockage
    monitored_disks: List[str] = field(default_factory=lambda: ["/", "/var", "/tmp"])
    
    # Mode de fonctionnement
    collection_mode: CollectionMode = CollectionMode.NORMAL
    adaptive_sampling: bool = True
    intelligent_batching: bool = True
    
    # Configuration externe
    collect_docker_metrics: bool = True
    collect_kubernetes_metrics: bool = False
    collect_application_metrics: bool = True
    
    # Alertes
    enable_real_time_alerts: bool = True
    alert_cooldown_seconds: int = 300
    
    # Performance
    max_concurrent_collectors: int = 10
    batch_size: int = 100
    compression_enabled: bool = True


class SystemMetricsCollector:
    """Collecteur de métriques système avancé."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.last_network_stats = {}
        self.last_disk_stats = {}
        self.collection_count = 0
        
    async def collect_cpu_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques CPU détaillées."""
        metrics = []
        timestamp = datetime.now()
        
        if not HAS_PSUTIL:
            return metrics
        
        try:
            # CPU global
            cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
            metrics.append(MetricDataPoint(
                metric_id="system.cpu.usage_total",
                timestamp=timestamp,
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                severity=self._get_cpu_severity(cpu_percent),
                tags={"type": "cpu", "scope": "total"},
                metadata={"collection_method": "psutil", "interval": 1}
            ))
            
            # CPU par core
            cpu_percpu = psutil.cpu_percent(interval=1, percpu=True)
            for i, cpu_core in enumerate(cpu_percpu):
                metrics.append(MetricDataPoint(
                    metric_id=f"system.cpu.usage_core",
                    timestamp=timestamp,
                    value=cpu_core,
                    metric_type=MetricType.GAUGE,
                    category=MetricCategory.SYSTEM,
                    severity=self._get_cpu_severity(cpu_core),
                    tags={"type": "cpu", "scope": "core", "core_id": str(i)},
                    metadata={"core_number": i, "total_cores": len(cpu_percpu)}
                ))
            
            # Statistiques CPU détaillées
            cpu_times = psutil.cpu_times()
            for stat_name in ['user', 'system', 'idle', 'iowait', 'irq', 'softirq']:
                if hasattr(cpu_times, stat_name):
                    value = getattr(cpu_times, stat_name)
                    metrics.append(MetricDataPoint(
                        metric_id=f"system.cpu.time_{stat_name}",
                        timestamp=timestamp,
                        value=value,
                        metric_type=MetricType.CUMULATIVE,
                        category=MetricCategory.SYSTEM,
                        tags={"type": "cpu_time", "stat": stat_name},
                        metadata={"unit": "seconds"}
                    ))
            
            # Load average
            if hasattr(os, 'getloadavg'):
                load1, load5, load15 = os.getloadavg()
                
                for interval, load_value in [("1min", load1), ("5min", load5), ("15min", load15)]:
                    metrics.append(MetricDataPoint(
                        metric_id=f"system.load_average",
                        timestamp=timestamp,
                        value=load_value,
                        metric_type=MetricType.GAUGE,
                        category=MetricCategory.SYSTEM,
                        severity=self._get_load_severity(load_value),
                        tags={"type": "load", "interval": interval},
                        metadata={"cpu_count": psutil.cpu_count()}
                    ))
            
            # Informations contextuelles
            cpu_count = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            
            metrics.append(MetricDataPoint(
                metric_id="system.cpu.count_logical",
                timestamp=timestamp,
                value=cpu_count,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                tags={"type": "cpu_info", "scope": "logical"}
            ))
            
            metrics.append(MetricDataPoint(
                metric_id="system.cpu.count_physical",
                timestamp=timestamp,
                value=cpu_count_physical,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                tags={"type": "cpu_info", "scope": "physical"}
            ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques CPU: {e}")
        
        return metrics
    
    async def collect_memory_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques mémoire détaillées."""
        metrics = []
        timestamp = datetime.now()
        
        if not HAS_PSUTIL:
            return metrics
        
        try:
            # Mémoire virtuelle
            vmem = psutil.virtual_memory()
            
            memory_metrics = {
                'total': vmem.total,
                'available': vmem.available,
                'used': vmem.used,
                'free': vmem.free,
                'percent': vmem.percent,
                'active': getattr(vmem, 'active', 0),
                'inactive': getattr(vmem, 'inactive', 0),
                'buffers': getattr(vmem, 'buffers', 0),
                'cached': getattr(vmem, 'cached', 0),
                'shared': getattr(vmem, 'shared', 0)
            }
            
            for metric_name, value in memory_metrics.items():
                if value is not None:
                    metrics.append(MetricDataPoint(
                        metric_id=f"system.memory.{metric_name}",
                        timestamp=timestamp,
                        value=value,
                        metric_type=MetricType.GAUGE,
                        category=MetricCategory.SYSTEM,
                        severity=self._get_memory_severity(metric_name, value, vmem.total),
                        tags={"type": "memory", "scope": "virtual", "metric": metric_name},
                        metadata={
                            "unit": "bytes" if metric_name != "percent" else "percent",
                            "total_memory": vmem.total
                        }
                    ))
            
            # Mémoire swap
            swap = psutil.swap_memory()
            
            swap_metrics = {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percent': swap.percent,
                'sin': swap.sin,
                'sout': swap.sout
            }
            
            for metric_name, value in swap_metrics.items():
                metrics.append(MetricDataPoint(
                    metric_id=f"system.swap.{metric_name}",
                    timestamp=timestamp,
                    value=value,
                    metric_type=MetricType.CUMULATIVE if metric_name in ['sin', 'sout'] else MetricType.GAUGE,
                    category=MetricCategory.SYSTEM,
                    severity=self._get_swap_severity(metric_name, value),
                    tags={"type": "swap", "metric": metric_name},
                    metadata={
                        "unit": "bytes" if metric_name not in ["percent"] else "percent",
                        "total_swap": swap.total
                    }
                ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques mémoire: {e}")
        
        return metrics
    
    async def collect_disk_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques disque détaillées."""
        metrics = []
        timestamp = datetime.now()
        
        if not HAS_PSUTIL:
            return metrics
        
        try:
            # Usage des disques
            for disk_path in self.config.monitored_disks:
                if os.path.exists(disk_path):
                    try:
                        usage = psutil.disk_usage(disk_path)
                        
                        disk_metrics = {
                            'total': usage.total,
                            'used': usage.used,
                            'free': usage.free,
                            'percent': (usage.used / usage.total) * 100
                        }
                        
                        for metric_name, value in disk_metrics.items():
                            metrics.append(MetricDataPoint(
                                metric_id=f"system.disk.{metric_name}",
                                timestamp=timestamp,
                                value=value,
                                metric_type=MetricType.GAUGE,
                                category=MetricCategory.STORAGE,
                                severity=self._get_disk_severity(metric_name, value),
                                tags={"type": "disk_usage", "mount_point": disk_path, "metric": metric_name},
                                metadata={
                                    "unit": "bytes" if metric_name != "percent" else "percent",
                                    "mount_point": disk_path
                                }
                            ))
                    except Exception as e:
                        logging.warning(f"Erreur collecte usage disque {disk_path}: {e}")
            
            # I/O disques
            disk_io = psutil.disk_io_counters(perdisk=True)
            if disk_io:
                for device, stats in disk_io.items():
                    # Calcul des rates si on a des données précédentes
                    current_time = time.time()
                    
                    io_metrics = {
                        'read_count': stats.read_count,
                        'write_count': stats.write_count,
                        'read_bytes': stats.read_bytes,
                        'write_bytes': stats.write_bytes,
                        'read_time': stats.read_time,
                        'write_time': stats.write_time
                    }
                    
                    # Stockage pour calcul des rates
                    if device not in self.last_disk_stats:
                        self.last_disk_stats[device] = {
                            'timestamp': current_time,
                            'stats': stats
                        }
                    
                    for metric_name, value in io_metrics.items():
                        metrics.append(MetricDataPoint(
                            metric_id=f"system.disk.io_{metric_name}",
                            timestamp=timestamp,
                            value=value,
                            metric_type=MetricType.CUMULATIVE,
                            category=MetricCategory.STORAGE,
                            tags={"type": "disk_io", "device": device, "metric": metric_name},
                            metadata={
                                "device": device,
                                "unit": "bytes" if "bytes" in metric_name else "milliseconds" if "time" in metric_name else "operations"
                            }
                        ))
                    
                    # Calcul des rates
                    if device in self.last_disk_stats:
                        last_data = self.last_disk_stats[device]
                        time_delta = current_time - last_data['timestamp']
                        
                        if time_delta > 0:
                            read_rate = (stats.read_bytes - last_data['stats'].read_bytes) / time_delta
                            write_rate = (stats.write_bytes - last_data['stats'].write_bytes) / time_delta
                            
                            metrics.extend([
                                MetricDataPoint(
                                    metric_id=f"system.disk.io_read_rate",
                                    timestamp=timestamp,
                                    value=read_rate,
                                    metric_type=MetricType.RATE,
                                    category=MetricCategory.STORAGE,
                                    tags={"type": "disk_rate", "device": device, "direction": "read"},
                                    metadata={"device": device, "unit": "bytes_per_second"}
                                ),
                                MetricDataPoint(
                                    metric_id=f"system.disk.io_write_rate",
                                    timestamp=timestamp,
                                    value=write_rate,
                                    metric_type=MetricType.RATE,
                                    category=MetricCategory.STORAGE,
                                    tags={"type": "disk_rate", "device": device, "direction": "write"},
                                    metadata={"device": device, "unit": "bytes_per_second"}
                                )
                            ])
                    
                    # Mise à jour des dernières stats
                    self.last_disk_stats[device] = {
                        'timestamp': current_time,
                        'stats': stats
                    }
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques disque: {e}")
        
        return metrics
    
    async def collect_network_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques réseau détaillées."""
        metrics = []
        timestamp = datetime.now()
        
        if not HAS_PSUTIL:
            return metrics
        
        try:
            # Statistiques réseau globales
            net_io = psutil.net_io_counters()
            if net_io:
                global_metrics = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                }
                
                for metric_name, value in global_metrics.items():
                    metrics.append(MetricDataPoint(
                        metric_id=f"system.network.{metric_name}",
                        timestamp=timestamp,
                        value=value,
                        metric_type=MetricType.CUMULATIVE,
                        category=MetricCategory.NETWORK,
                        severity=self._get_network_severity(metric_name, value),
                        tags={"type": "network", "scope": "global", "metric": metric_name},
                        metadata={
                            "unit": "bytes" if "bytes" in metric_name else "packets" if "packets" in metric_name else "errors"
                        }
                    ))
            
            # Statistiques par interface
            net_io_per_nic = psutil.net_io_counters(pernic=True)
            if net_io_per_nic:
                for interface, stats in net_io_per_nic.items():
                    if interface in self.config.network_interfaces or interface.startswith(('eth', 'wlan', 'en', 'wl')):
                        interface_metrics = {
                            'bytes_sent': stats.bytes_sent,
                            'bytes_recv': stats.bytes_recv,
                            'packets_sent': stats.packets_sent,
                            'packets_recv': stats.packets_recv,
                            'errin': stats.errin,
                            'errout': stats.errout,
                            'dropin': stats.dropin,
                            'dropout': stats.dropout
                        }
                        
                        for metric_name, value in interface_metrics.items():
                            metrics.append(MetricDataPoint(
                                metric_id=f"system.network.interface_{metric_name}",
                                timestamp=timestamp,
                                value=value,
                                metric_type=MetricType.CUMULATIVE,
                                category=MetricCategory.NETWORK,
                                tags={"type": "network_interface", "interface": interface, "metric": metric_name},
                                metadata={"interface": interface}
                            ))
            
            # Connexions réseau
            try:
                connections = psutil.net_connections()
                connection_stats = {
                    'ESTABLISHED': 0,
                    'LISTEN': 0,
                    'TIME_WAIT': 0,
                    'CLOSE_WAIT': 0,
                    'SYN_SENT': 0,
                    'SYN_RECV': 0
                }
                
                for conn in connections:
                    if conn.status in connection_stats:
                        connection_stats[conn.status] += 1
                
                for status, count in connection_stats.items():
                    metrics.append(MetricDataPoint(
                        metric_id=f"system.network.connections_{status.lower()}",
                        timestamp=timestamp,
                        value=count,
                        metric_type=MetricType.GAUGE,
                        category=MetricCategory.NETWORK,
                        tags={"type": "network_connections", "status": status},
                        metadata={"connection_status": status}
                    ))
                    
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass  # Permissions insuffisantes pour lister les connexions
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques réseau: {e}")
        
        return metrics
    
    async def collect_process_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques de processus."""
        metrics = []
        timestamp = datetime.now()
        
        if not HAS_PSUTIL:
            return metrics
        
        try:
            # Nombre total de processus
            process_count = len(psutil.pids())
            metrics.append(MetricDataPoint(
                metric_id="system.processes.total_count",
                timestamp=timestamp,
                value=process_count,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                tags={"type": "processes", "metric": "count"}
            ))
            
            # Processus par état
            process_states = {}
            top_processes_cpu = []
            top_processes_memory = []
            
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                try:
                    info = proc.info
                    status = info.get('status', 'unknown')
                    
                    # Comptage par état
                    process_states[status] = process_states.get(status, 0) + 1
                    
                    # Top processus par CPU/Mémoire
                    if info.get('cpu_percent', 0) > 0:
                        top_processes_cpu.append((info['name'], info['cpu_percent']))
                    if info.get('memory_percent', 0) > 0:
                        top_processes_memory.append((info['name'], info['memory_percent']))
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Métriques par état de processus
            for status, count in process_states.items():
                metrics.append(MetricDataPoint(
                    metric_id=f"system.processes.by_status",
                    timestamp=timestamp,
                    value=count,
                    metric_type=MetricType.GAUGE,
                    category=MetricCategory.SYSTEM,
                    tags={"type": "processes", "status": status},
                    metadata={"process_status": status}
                ))
            
            # Top 5 processus CPU
            top_processes_cpu.sort(key=lambda x: x[1], reverse=True)
            for i, (name, cpu_percent) in enumerate(top_processes_cpu[:5]):
                metrics.append(MetricDataPoint(
                    metric_id=f"system.processes.top_cpu",
                    timestamp=timestamp,
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    category=MetricCategory.PERFORMANCE,
                    tags={"type": "top_processes", "metric": "cpu", "rank": str(i+1), "process": name},
                    metadata={"process_name": name, "rank": i+1}
                ))
            
            # Top 5 processus Mémoire
            top_processes_memory.sort(key=lambda x: x[1], reverse=True)
            for i, (name, memory_percent) in enumerate(top_processes_memory[:5]):
                metrics.append(MetricDataPoint(
                    metric_id=f"system.processes.top_memory",
                    timestamp=timestamp,
                    value=memory_percent,
                    metric_type=MetricType.GAUGE,
                    category=MetricCategory.PERFORMANCE,
                    tags={"type": "top_processes", "metric": "memory", "rank": str(i+1), "process": name},
                    metadata={"process_name": name, "rank": i+1}
                ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques processus: {e}")
        
        return metrics
    
    def _get_cpu_severity(self, cpu_percent: float) -> MetricSeverity:
        """Détermine la sévérité basée sur l'usage CPU."""
        if cpu_percent >= self.config.max_cpu_threshold:
            return MetricSeverity.CRITICAL
        elif cpu_percent >= self.config.max_cpu_threshold * 0.8:
            return MetricSeverity.HIGH
        elif cpu_percent >= self.config.max_cpu_threshold * 0.6:
            return MetricSeverity.MEDIUM
        return MetricSeverity.LOW
    
    def _get_memory_severity(self, metric_name: str, value: float, total: float) -> MetricSeverity:
        """Détermine la sévérité basée sur l'usage mémoire."""
        if metric_name == "percent":
            percent = value
        else:
            percent = (value / total) * 100 if total > 0 else 0
        
        if percent >= self.config.max_memory_threshold:
            return MetricSeverity.CRITICAL
        elif percent >= self.config.max_memory_threshold * 0.8:
            return MetricSeverity.HIGH
        elif percent >= self.config.max_memory_threshold * 0.6:
            return MetricSeverity.MEDIUM
        return MetricSeverity.LOW
    
    def _get_disk_severity(self, metric_name: str, value: float) -> MetricSeverity:
        """Détermine la sévérité basée sur l'usage disque."""
        if metric_name == "percent":
            if value >= self.config.max_disk_threshold:
                return MetricSeverity.CRITICAL
            elif value >= self.config.max_disk_threshold * 0.8:
                return MetricSeverity.HIGH
            elif value >= self.config.max_disk_threshold * 0.6:
                return MetricSeverity.MEDIUM
        return MetricSeverity.LOW
    
    def _get_load_severity(self, load_value: float) -> MetricSeverity:
        """Détermine la sévérité basée sur la charge système."""
        cpu_count = psutil.cpu_count() if HAS_PSUTIL else 1
        load_ratio = load_value / cpu_count
        
        if load_ratio >= 2.0:
            return MetricSeverity.CRITICAL
        elif load_ratio >= 1.5:
            return MetricSeverity.HIGH
        elif load_ratio >= 1.0:
            return MetricSeverity.MEDIUM
        return MetricSeverity.LOW
    
    def _get_swap_severity(self, metric_name: str, value: float) -> MetricSeverity:
        """Détermine la sévérité basée sur l'usage swap."""
        if metric_name == "percent" and value > 50:
            return MetricSeverity.HIGH
        elif metric_name in ["sin", "sout"] and value > 0:
            return MetricSeverity.MEDIUM
        return MetricSeverity.LOW
    
    def _get_network_severity(self, metric_name: str, value: float) -> MetricSeverity:
        """Détermine la sévérité basée sur les métriques réseau."""
        if metric_name in ["errin", "errout", "dropin", "dropout"] and value > 0:
            return MetricSeverity.MEDIUM
        return MetricSeverity.LOW


class SecurityMetricsCollector:
    """Collecteur de métriques de sécurité avancé."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        
    async def collect_auth_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques d'authentification."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Simulation de métriques d'authentification
            # En production, ces données viendraient de vrais logs/systèmes
            
            # Tentatives d'authentification échouées (depuis les logs système)
            failed_attempts = await self._count_failed_auth_attempts()
            metrics.append(MetricDataPoint(
                metric_id="security.authentication.failed_attempts",
                timestamp=timestamp,
                value=failed_attempts,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.HIGH if failed_attempts > 10 else MetricSeverity.LOW,
                tags={"type": "authentication", "status": "failed"},
                metadata={"source": "system_logs", "time_window": "1_hour"}
            ))
            
            # Tentatives d'authentification réussies
            successful_attempts = await self._count_successful_auth_attempts()
            metrics.append(MetricDataPoint(
                metric_id="security.authentication.successful_attempts",
                timestamp=timestamp,
                value=successful_attempts,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                tags={"type": "authentication", "status": "successful"},
                metadata={"source": "system_logs", "time_window": "1_hour"}
            ))
            
            # Sessions actives
            active_sessions = await self._count_active_sessions()
            metrics.append(MetricDataPoint(
                metric_id="security.sessions.active_count",
                timestamp=timestamp,
                value=active_sessions,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.SECURITY,
                tags={"type": "sessions", "status": "active"}
            ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques authentification: {e}")
        
        return metrics
    
    async def collect_access_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques d'accès aux ressources."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Accès aux fichiers sensibles
            sensitive_file_access = await self._count_sensitive_file_access()
            metrics.append(MetricDataPoint(
                metric_id="security.file_access.sensitive_files",
                timestamp=timestamp,
                value=sensitive_file_access,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.MEDIUM if sensitive_file_access > 0 else MetricSeverity.LOW,
                tags={"type": "file_access", "classification": "sensitive"}
            ))
            
            # Accès aux clés cryptographiques
            key_access = await self._count_crypto_key_access()
            metrics.append(MetricDataPoint(
                metric_id="security.key_access.crypto_keys",
                timestamp=timestamp,
                value=key_access,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.HIGH if key_access > 100 else MetricSeverity.LOW,
                tags={"type": "key_access", "resource": "cryptographic_keys"}
            ))
            
            # Violations de permissions
            permission_violations = await self._count_permission_violations()
            metrics.append(MetricDataPoint(
                metric_id="security.violations.permissions",
                timestamp=timestamp,
                value=permission_violations,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.CRITICAL if permission_violations > 0 else MetricSeverity.LOW,
                tags={"type": "violations", "category": "permissions"}
            ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques accès: {e}")
        
        return metrics
    
    async def collect_threat_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques de menaces."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Tentatives d'intrusion détectées
            intrusion_attempts = await self._detect_intrusion_attempts()
            metrics.append(MetricDataPoint(
                metric_id="security.threats.intrusion_attempts",
                timestamp=timestamp,
                value=intrusion_attempts,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.CRITICAL if intrusion_attempts > 0 else MetricSeverity.LOW,
                tags={"type": "threats", "category": "intrusion"}
            ))
            
            # Scans de ports suspects
            port_scans = await self._detect_port_scans()
            metrics.append(MetricDataPoint(
                metric_id="security.threats.port_scans",
                timestamp=timestamp,
                value=port_scans,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.HIGH if port_scans > 5 else MetricSeverity.LOW,
                tags={"type": "threats", "category": "port_scan"}
            ))
            
            # Activité réseau suspecte
            suspicious_network = await self._detect_suspicious_network_activity()
            metrics.append(MetricDataPoint(
                metric_id="security.threats.suspicious_network",
                timestamp=timestamp,
                value=suspicious_network,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                severity=MetricSeverity.HIGH if suspicious_network > 0 else MetricSeverity.LOW,
                tags={"type": "threats", "category": "network_anomaly"}
            ))
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques menaces: {e}")
        
        return metrics
    
    async def _count_failed_auth_attempts(self) -> int:
        """Compte les tentatives d'authentification échouées."""
        try:
            # Lecture des logs d'authentification (exemple avec auth.log)
            auth_log_paths = ['/var/log/auth.log', '/var/log/secure']
            failed_count = 0
            
            for log_path in auth_log_paths:
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r') as f:
                            # Recherche des échecs d'authentification dans la dernière heure
                            for line in f:
                                if 'authentication failure' in line.lower() or 'failed login' in line.lower():
                                    # Parsing simple du timestamp
                                    # En production, utiliser un parsing plus robuste
                                    failed_count += 1
                        break
                    except PermissionError:
                        continue
            
            return failed_count
        except Exception as e:
            logging.warning(f"Erreur lecture logs authentification: {e}")
            return 0
    
    async def _count_successful_auth_attempts(self) -> int:
        """Compte les tentatives d'authentification réussies."""
        # Simulation - en production, lire les vrais logs
        return 45
    
    async def _count_active_sessions(self) -> int:
        """Compte les sessions actives."""
        try:
            if HAS_PSUTIL:
                # Compte les sessions utilisateur actives
                users = psutil.users()
                return len(users)
            return 0
        except Exception:
            return 0
    
    async def _count_sensitive_file_access(self) -> int:
        """Compte les accès aux fichiers sensibles."""
        # Simulation - en production, utiliser audit logs
        return 12
    
    async def _count_crypto_key_access(self) -> int:
        """Compte les accès aux clés cryptographiques."""
        # Simulation - en production, monitorer l'accès aux keystores
        return 8
    
    async def _count_permission_violations(self) -> int:
        """Compte les violations de permissions."""
        # Simulation - en production, analyser les logs de sécurité
        return 0
    
    async def _detect_intrusion_attempts(self) -> int:
        """Détecte les tentatives d'intrusion."""
        # Simulation - en production, utiliser des outils comme fail2ban
        return 0
    
    async def _detect_port_scans(self) -> int:
        """Détecte les scans de ports."""
        # Simulation - en production, analyser les logs réseau
        return 2
    
    async def _detect_suspicious_network_activity(self) -> int:
        """Détecte l'activité réseau suspecte."""
        # Simulation - en production, utiliser des outils de monitoring réseau
        return 0


class ApplicationMetricsCollector:
    """Collecteur de métriques applicatives."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        
    async def collect_performance_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques de performance applicative."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Métriques HTTP/API (simulation)
            response_times = await self._measure_api_response_times()
            for endpoint, response_time in response_times.items():
                metrics.append(MetricDataPoint(
                    metric_id="application.api.response_time",
                    timestamp=timestamp,
                    value=response_time,
                    metric_type=MetricType.TIMER,
                    category=MetricCategory.PERFORMANCE,
                    severity=MetricSeverity.HIGH if response_time > 1000 else MetricSeverity.LOW,
                    tags={"type": "api", "endpoint": endpoint},
                    metadata={"unit": "milliseconds", "endpoint": endpoint}
                ))
            
            # Débit de requêtes
            request_rate = await self._measure_request_rate()
            metrics.append(MetricDataPoint(
                metric_id="application.api.request_rate",
                timestamp=timestamp,
                value=request_rate,
                metric_type=MetricType.RATE,
                category=MetricCategory.PERFORMANCE,
                tags={"type": "api", "metric": "throughput"},
                metadata={"unit": "requests_per_second"}
            ))
            
            # Taux d'erreur
            error_rate = await self._calculate_error_rate()
            metrics.append(MetricDataPoint(
                metric_id="application.api.error_rate",
                timestamp=timestamp,
                value=error_rate,
                metric_type=MetricType.RATIO,
                category=MetricCategory.RELIABILITY,
                severity=MetricSeverity.CRITICAL if error_rate > 5 else MetricSeverity.LOW,
                tags={"type": "api", "metric": "reliability"},
                metadata={"unit": "percent"}
            ))
            
            # Métriques de base de données
            db_metrics = await self._collect_database_metrics()
            metrics.extend(db_metrics)
            
            # Métriques de cache
            cache_metrics = await self._collect_cache_metrics()
            metrics.extend(cache_metrics)
            
        except Exception as e:
            logging.error(f"Erreur collecte métriques application: {e}")
        
        return metrics
    
    async def _measure_api_response_times(self) -> Dict[str, float]:
        """Mesure les temps de réponse des APIs."""
        # Simulation - en production, utiliser des métriques réelles
        return {
            "/api/keys": 150.5,
            "/api/auth": 89.2,
            "/api/metrics": 234.1,
            "/api/health": 45.3
        }
    
    async def _measure_request_rate(self) -> float:
        """Mesure le débit de requêtes."""
        # Simulation - en production, calculer depuis les logs/métriques
        return 127.8
    
    async def _calculate_error_rate(self) -> float:
        """Calcule le taux d'erreur."""
        # Simulation - en production, calculer depuis les logs
        return 2.3
    
    async def _collect_database_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques de base de données."""
        metrics = []
        timestamp = datetime.now()
        
        # Simulation de métriques DB
        db_metrics = {
            "connection_count": 25,
            "query_time_avg": 45.2,
            "slow_queries": 3,
            "deadlocks": 0,
            "buffer_hit_ratio": 98.5
        }
        
        for metric_name, value in db_metrics.items():
            metrics.append(MetricDataPoint(
                metric_id=f"application.database.{metric_name}",
                timestamp=timestamp,
                value=value,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.PERFORMANCE,
                tags={"type": "database", "metric": metric_name},
                metadata={"database": "postgresql"}
            ))
        
        return metrics
    
    async def _collect_cache_metrics(self) -> List[MetricDataPoint]:
        """Collecte des métriques de cache."""
        metrics = []
        timestamp = datetime.now()
        
        # Simulation de métriques cache
        cache_metrics = {
            "hit_ratio": 89.3,
            "miss_ratio": 10.7,
            "evictions": 15,
            "memory_usage": 67.8
        }
        
        for metric_name, value in cache_metrics.items():
            metrics.append(MetricDataPoint(
                metric_id=f"application.cache.{metric_name}",
                timestamp=timestamp,
                value=value,
                metric_type=MetricType.GAUGE,
                category=MetricCategory.PERFORMANCE,
                tags={"type": "cache", "metric": metric_name},
                metadata={"cache_type": "redis"}
            ))
        
        return metrics


class MetricsCollectionAgent:
    """Agent principal de collecte de métriques."""
    
    def __init__(self, config: CollectorConfig, metrics_system: EnterpriseMetricsSystem):
        self.config = config
        self.metrics_system = metrics_system
        
        # Collecteurs spécialisés
        self.system_collector = SystemMetricsCollector(config)
        self.security_collector = SecurityMetricsCollector(config)
        self.application_collector = ApplicationMetricsCollector(config)
        
        # État de l'agent
        self.running = False
        self.collection_tasks = set()
        self.last_collection_times = {}
        
        # Statistiques
        self.metrics_collected = 0
        self.collection_errors = 0
        self.last_error_time = None
        
    async def start(self):
        """Démarre l'agent de collecte."""
        if self.running:
            logging.warning("Agent de collecte déjà en cours d'exécution")
            return
        
        self.running = True
        logging.info("Démarrage de l'agent de collecte de métriques")
        
        # Démarrage des tâches de collecte
        tasks = [
            asyncio.create_task(self._collection_loop("system", self._collect_system_metrics)),
            asyncio.create_task(self._collection_loop("security", self._collect_security_metrics)),
            asyncio.create_task(self._collection_loop("performance", self._collect_performance_metrics)),
            asyncio.create_task(self._collection_loop("network", self._collect_network_metrics))
        ]
        
        self.collection_tasks.update(tasks)
        
        # Tâche de monitoring de l'agent
        self.collection_tasks.add(
            asyncio.create_task(self._agent_monitoring_loop())
        )
        
        logging.info(f"Agent de collecte démarré avec {len(self.collection_tasks)} tâches")
    
    async def stop(self):
        """Arrête l'agent de collecte."""
        if not self.running:
            return
        
        self.running = False
        logging.info("Arrêt de l'agent de collecte de métriques")
        
        # Annulation des tâches
        for task in self.collection_tasks:
            task.cancel()
        
        # Attendre l'arrêt des tâches
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        self.collection_tasks.clear()
        
        logging.info("Agent de collecte arrêté")
    
    async def _collection_loop(self, collector_name: str, collector_func: Callable):
        """Boucle de collecte pour un collecteur spécifique."""
        interval = self._get_collection_interval(collector_name)
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collecte des métriques
                metrics = await collector_func()
                
                # Traitement et stockage
                for metric in metrics:
                    metric.source = f"agent_{collector_name}"
                    metric.metadata.update({
                        "agent_version": "1.0.0",
                        "collector": collector_name,
                        "collection_time": start_time
                    })
                    
                    # Stockage via le système de métriques
                    await self.metrics_system.storage.store_metric(metric)
                    self.metrics_collected += 1
                
                # Mise à jour du temps de dernière collecte
                self.last_collection_times[collector_name] = datetime.now()
                
                collection_time = time.time() - start_time
                logging.debug(f"Collecte {collector_name}: {len(metrics)} métriques en {collection_time:.2f}s")
                
                # Adaptation dynamique de l'intervalle
                if self.config.adaptive_sampling:
                    interval = self._adapt_collection_interval(collector_name, collection_time, len(metrics))
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.collection_errors += 1
                self.last_error_time = datetime.now()
                logging.error(f"Erreur dans collecte {collector_name}: {e}")
                await asyncio.sleep(min(interval, 30))  # Pause avant retry
    
    async def _collect_system_metrics(self) -> List[MetricDataPoint]:
        """Collecte toutes les métriques système."""
        all_metrics = []
        
        # Collecte en parallèle des différentes métriques système
        tasks = [
            self.system_collector.collect_cpu_metrics(),
            self.system_collector.collect_memory_metrics(),
            self.system_collector.collect_disk_metrics(),
            self.system_collector.collect_process_metrics()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_metrics.extend(result)
            elif isinstance(result, Exception):
                logging.error(f"Erreur collecte métrique système: {result}")
        
        return all_metrics
    
    async def _collect_security_metrics(self) -> List[MetricDataPoint]:
        """Collecte toutes les métriques de sécurité."""
        all_metrics = []
        
        tasks = [
            self.security_collector.collect_auth_metrics(),
            self.security_collector.collect_access_metrics(),
            self.security_collector.collect_threat_metrics()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_metrics.extend(result)
            elif isinstance(result, Exception):
                logging.error(f"Erreur collecte métrique sécurité: {result}")
        
        return all_metrics
    
    async def _collect_performance_metrics(self) -> List[MetricDataPoint]:
        """Collecte toutes les métriques de performance."""
        return await self.application_collector.collect_performance_metrics()
    
    async def _collect_network_metrics(self) -> List[MetricDataPoint]:
        """Collecte toutes les métriques réseau."""
        return await self.system_collector.collect_network_metrics()
    
    async def _agent_monitoring_loop(self):
        """Boucle de monitoring de l'agent lui-même."""
        while self.running:
            try:
                timestamp = datetime.now()
                
                # Métriques de l'agent
                agent_metrics = [
                    MetricDataPoint(
                        metric_id="agent.metrics.collected_total",
                        timestamp=timestamp,
                        value=self.metrics_collected,
                        metric_type=MetricType.CUMULATIVE,
                        category=MetricCategory.SYSTEM,
                        tags={"type": "agent", "metric": "collection_count"}
                    ),
                    MetricDataPoint(
                        metric_id="agent.errors.collection_errors",
                        timestamp=timestamp,
                        value=self.collection_errors,
                        metric_type=MetricType.CUMULATIVE,
                        category=MetricCategory.RELIABILITY,
                        severity=MetricSeverity.HIGH if self.collection_errors > 10 else MetricSeverity.LOW,
                        tags={"type": "agent", "metric": "error_count"}
                    ),
                    MetricDataPoint(
                        metric_id="agent.tasks.active_count",
                        timestamp=timestamp,
                        value=len(self.collection_tasks),
                        metric_type=MetricType.GAUGE,
                        category=MetricCategory.SYSTEM,
                        tags={"type": "agent", "metric": "task_count"}
                    )
                ]
                
                # Stockage des métriques de l'agent
                for metric in agent_metrics:
                    metric.source = "agent_self_monitoring"
                    await self.metrics_system.storage.store_metric(metric)
                
                await asyncio.sleep(60)  # Monitoring de l'agent toutes les minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Erreur monitoring agent: {e}")
                await asyncio.sleep(60)
    
    def _get_collection_interval(self, collector_name: str) -> int:
        """Obtient l'intervalle de collecte pour un collecteur."""
        intervals = {
            "system": self.config.system_interval,
            "security": self.config.security_interval,
            "performance": self.config.performance_interval,
            "network": self.config.network_interval
        }
        return intervals.get(collector_name, 60)
    
    def _adapt_collection_interval(self, collector_name: str, collection_time: float, metrics_count: int) -> int:
        """Adapte dynamiquement l'intervalle de collecte."""
        base_interval = self._get_collection_interval(collector_name)
        
        # Adaptation basée sur le temps de collecte
        if collection_time > 10:  # Si la collecte prend plus de 10s
            return min(base_interval * 2, 300)  # Double l'intervalle, max 5 min
        elif collection_time < 1 and metrics_count > 0:  # Collecte rapide et productive
            return max(base_interval // 2, 10)  # Réduit l'intervalle, min 10s
        
        return base_interval
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Retourne le statut de l'agent."""
        return {
            "running": self.running,
            "metrics_collected": self.metrics_collected,
            "collection_errors": self.collection_errors,
            "active_tasks": len(self.collection_tasks),
            "last_collection_times": {
                name: time.isoformat() for name, time in self.last_collection_times.items()
            },
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "config_mode": self.config.collection_mode.value
        }


def load_config_from_file(config_path: str) -> CollectorConfig:
    """Charge la configuration depuis un fichier."""
    config = CollectorConfig()
    
    if not os.path.exists(config_path):
        logging.warning(f"Fichier de configuration non trouvé: {config_path}")
        return config
    
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                data = json.load(f)
        elif config_path.endswith(('.yml', '.yaml')):
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            # Format INI
            parser = configparser.ConfigParser()
            parser.read(config_path)
            data = {section: dict(parser[section]) for section in parser.sections()}
        
        # Mise à jour de la configuration
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logging.info(f"Configuration chargée depuis {config_path}")
        
    except Exception as e:
        logging.error(f"Erreur chargement configuration: {e}")
    
    return config


async def main():
    """Fonction principale de l'agent de collecte."""
    parser = argparse.ArgumentParser(description="Agent de collecte de métriques")
    parser.add_argument("--config", "-c", help="Fichier de configuration", default="metrics_agent.json")
    parser.add_argument("--storage", "-s", help="Type de stockage (sqlite/redis)", default="sqlite")
    parser.add_argument("--db-path", help="Chemin de la base de données SQLite", default="metrics.db")
    parser.add_argument("--redis-url", help="URL Redis", default="redis://localhost:6379/0")
    parser.add_argument("--daemon", "-d", action="store_true", help="Mode daemon")
    parser.add_argument("--pid-file", help="Fichier PID pour mode daemon", default="metrics_agent.pid")
    parser.add_argument("--log-level", help="Niveau de log", default="INFO")
    parser.add_argument("--log-file", help="Fichier de log")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = getattr(logging, args.log_level.upper())
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.log_file:
        logging.basicConfig(level=log_level, format=log_format, filename=args.log_file)
    else:
        logging.basicConfig(level=log_level, format=log_format)
    
    # Chargement de la configuration
    config = load_config_from_file(args.config)
    
    # Configuration du stockage
    storage_config = {}
    if args.storage == "sqlite":
        storage_config['db_path'] = args.db_path
    elif args.storage == "redis":
        storage_config['redis_url'] = args.redis_url
    
    try:
        # Initialisation du système de métriques
        metrics_system = get_metrics_system(args.storage, storage_config)
        
        # Initialisation de l'agent
        agent = MetricsCollectionAgent(config, metrics_system)
        
        # Gestion des signaux pour arrêt propre
        def signal_handler(signum, frame):
            logging.info(f"Signal {signum} reçu, arrêt de l'agent")
            asyncio.create_task(agent.stop())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Mode daemon
        if args.daemon:
            with open(args.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logging.info(f"Agent démarré en mode daemon (PID: {os.getpid()})")
        
        # Démarrage du système de métriques et de l'agent
        await metrics_system.start()
        await agent.start()
        
        logging.info("Agent de collecte de métriques opérationnel")
        
        # Boucle principale
        try:
            while agent.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("Interruption clavier, arrêt de l'agent")
        
        # Arrêt propre
        await agent.stop()
        await metrics_system.stop()
        
        # Nettoyage du fichier PID
        if args.daemon and os.path.exists(args.pid_file):
            os.remove(args.pid_file)
        
        logging.info("Agent de collecte de métriques arrêté")
        
    except Exception as e:
        logging.error(f"Erreur fatale dans l'agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
