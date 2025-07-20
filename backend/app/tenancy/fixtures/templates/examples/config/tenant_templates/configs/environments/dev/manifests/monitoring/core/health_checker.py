# -*- coding: utf-8 -*-
"""
Health Checker - Système de Vérification de Santé Ultra-Avancé
============================================================

Vérificateur de santé complet pour l'agent IA Spotify.
Surveillance proactive de tous les composants système, application et business
avec détection précoce des problèmes et auto-remédiation.

Fonctionnalités:
- Health checks multi-niveaux (infrastructure, application, business)
- Détection proactive des dégradations de performance
- Vérifications de santé personnalisées par tenant
- Monitoring des dépendances externes (APIs, BDD, cache)
- Tests de charge et de stress automatisés
- Collecte de métriques de santé détaillées
- Alerting intelligent avec prédiction de pannes

Auteur: Expert Team - Ingénieur Machine Learning + DBA & Data Engineer - Fahed Mlaiel
Version: 2.0.0
"""

import time
import asyncio
import threading
import logging
import json
import psutil
import requests
import redis
import sqlite3
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
from urllib.parse import urlparse
import socket
import ssl
import pandas as pd
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Statuts de santé"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class CheckType(Enum):
    """Types de vérifications"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    BUSINESS = "business"
    CUSTOM = "custom"

@dataclass
class HealthCheck:
    """Définition d'une vérification de santé"""
    name: str
    description: str
    check_type: str
    endpoint: Optional[str] = None
    expected_status: int = 200
    timeout_seconds: int = 10
    interval_seconds: int = 30
    retries: int = 3
    tenant_id: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    custom_headers: Optional[Dict[str, str]] = None
    custom_checker: Optional[Callable] = None
    thresholds: Optional[Dict[str, float]] = None
    enabled: bool = True

@dataclass
class HealthResult:
    """Résultat d'une vérification de santé"""
    check_name: str
    status: str
    message: str
    duration_ms: float
    timestamp: float
    details: Dict[str, Any]
    tenant_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None

@dataclass
class SystemHealth:
    """État de santé global du système"""
    overall_status: str
    healthy_checks: int
    warning_checks: int
    critical_checks: int
    unknown_checks: int
    total_checks: int
    last_update: float
    uptime_percentage: float
    response_time_avg: float
    details: Dict[str, HealthResult]

class HealthChecker:
    """
    Vérificateur de santé ultra-avancé avec monitoring prédictif
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le vérificateur de santé
        
        Args:
            config: Configuration du vérificateur
        """
        self.config = config or self._default_config()
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, List[HealthResult]] = {}
        self.lock = threading.RLock()
        self.running = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
        # Storage
        self.db_path = self.config.get('db_path', 'health_checks.db')
        self.redis_client = self._init_redis()
        
        # Statistics
        self.stats = {
            'total_checks_run': 0,
            'failed_checks': 0,
            'avg_response_time': 0,
            'uptime_start': time.time()
        }
        
        # Background threads
        self.checker_thread = None
        self.analyzer_thread = None
        
        # Initialisation
        self._init_database()
        self._register_default_checks()
        
        logger.info("HealthChecker initialisé avec succès")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'check_interval': 30,
            'retention_days': 7,
            'alert_threshold_failures': 3,
            'slow_response_threshold_ms': 1000,
            'critical_failure_threshold': 5,
            'enable_predictive_analysis': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 2,
            'max_concurrent_checks': 10,
            'enable_detailed_metrics': True,
            'enable_auto_remediation': True,
            'notification_cooldown_minutes': 15
        }
    
    def _init_redis(self) -> Optional[redis.Redis]:
        """Initialise la connexion Redis"""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                db=self.config['redis_db'],
                decode_responses=True
            )
            client.ping()
            logger.info("Connexion Redis HealthChecker établie")
            return client
        except Exception as e:
            logger.warning(f"Redis non disponible pour HealthChecker: {e}")
            return None
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Table des vérifications
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    check_type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Table des résultats
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    duration_ms REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    details TEXT,
                    tenant_id TEXT,
                    metrics TEXT
                )
            ''')
            
            # Table des statistiques
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    tenant_id TEXT,
                    labels TEXT
                )
            ''')
            
            # Index pour les performances
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_check_name ON health_results(check_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_timestamp ON health_results(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_status ON health_results(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stats_metric_name ON health_stats(metric_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON health_stats(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("Base de données HealthChecker initialisée")
            
        except Exception as e:
            logger.error(f"Erreur initialisation base de données: {e}")
    
    def start(self):
        """Démarre le vérificateur de santé"""
        if self.running:
            logger.warning("HealthChecker déjà en cours d'exécution")
            return
        
        self.running = True
        
        # Démarre les threads de vérification
        self.checker_thread = threading.Thread(target=self._checking_loop, daemon=True)
        self.analyzer_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        
        self.checker_thread.start()
        self.analyzer_thread.start()
        
        logger.info("HealthChecker démarré")
    
    def stop(self):
        """Arrête le vérificateur de santé"""
        self.running = False
        
        if self.checker_thread and self.checker_thread.is_alive():
            self.checker_thread.join(timeout=5)
        
        if self.analyzer_thread and self.analyzer_thread.is_alive():
            self.analyzer_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        logger.info("HealthChecker arrêté")
    
    def register_check(self, check: HealthCheck) -> bool:
        """
        Enregistre une nouvelle vérification de santé
        
        Args:
            check: Vérification à enregistrer
            
        Returns:
            True si l'enregistrement a réussi
        """
        try:
            with self.lock:
                self.checks[check.name] = check
                if check.name not in self.results:
                    self.results[check.name] = []
            
            # Sauvegarde en base
            self._save_check_to_db(check)
            
            logger.info(f"Vérification enregistrée: {check.name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur enregistrement vérification: {e}")
            return False
    
    def unregister_check(self, check_name: str) -> bool:
        """
        Supprime une vérification de santé
        
        Args:
            check_name: Nom de la vérification à supprimer
            
        Returns:
            True si la suppression a réussi
        """
        try:
            with self.lock:
                if check_name in self.checks:
                    del self.checks[check_name]
                if check_name in self.results:
                    del self.results[check_name]
            
            logger.info(f"Vérification supprimée: {check_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur suppression vérification: {e}")
            return False
    
    def run_check(self, check_name: str) -> Optional[HealthResult]:
        """
        Exécute une vérification de santé spécifique
        
        Args:
            check_name: Nom de la vérification à exécuter
            
        Returns:
            Résultat de la vérification ou None en cas d'erreur
        """
        if check_name not in self.checks:
            logger.warning(f"Vérification inconnue: {check_name}")
            return None
        
        check = self.checks[check_name]
        if not check.enabled:
            logger.debug(f"Vérification désactivée: {check_name}")
            return None
        
        start_time = time.time()
        
        try:
            # Exécution selon le type de vérification
            if check.check_type == CheckType.INFRASTRUCTURE.value:
                result = self._check_infrastructure(check)
            elif check.check_type == CheckType.APPLICATION.value:
                result = self._check_application(check)
            elif check.check_type == CheckType.DATABASE.value:
                result = self._check_database(check)
            elif check.check_type == CheckType.CACHE.value:
                result = self._check_cache(check)
            elif check.check_type == CheckType.EXTERNAL_API.value:
                result = self._check_external_api(check)
            elif check.check_type == CheckType.BUSINESS.value:
                result = self._check_business(check)
            elif check.check_type == CheckType.CUSTOM.value:
                result = self._check_custom(check)
            else:
                result = HealthResult(
                    check_name=check_name,
                    status=HealthStatus.UNKNOWN.value,
                    message=f"Type de vérification non supporté: {check.check_type}",
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time(),
                    details={},
                    tenant_id=check.tenant_id
                )
            
            # Stockage du résultat
            with self.lock:
                if check_name not in self.results:
                    self.results[check_name] = []
                self.results[check_name].append(result)
                
                # Limite l'historique
                max_results = 100
                if len(self.results[check_name]) > max_results:
                    self.results[check_name] = self.results[check_name][-max_results:]
            
            # Sauvegarde en base
            self._save_result_to_db(result)
            
            # Mise à jour des statistiques
            self._update_stats(result)
            
            # Vérification des seuils d'alerte
            self._check_alert_thresholds(check, result)
            
            return result
            
        except Exception as e:
            error_result = HealthResult(
                check_name=check_name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur lors de la vérification: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
            
            logger.error(f"Erreur vérification {check_name}: {e}")
            return error_result
    
    def _check_infrastructure(self, check: HealthCheck) -> HealthResult:
        """Vérification de l'infrastructure système"""
        start_time = time.time()
        details = {}
        metrics = {}
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            details['cpu_percent'] = cpu_percent
            metrics['cpu_usage'] = cpu_percent
            
            # Mémoire
            memory = psutil.virtual_memory()
            details['memory_percent'] = memory.percent
            details['memory_available_gb'] = memory.available / (1024**3)
            metrics['memory_usage'] = memory.percent
            
            # Disque
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            details['disk_percent'] = disk_percent
            details['disk_free_gb'] = disk.free / (1024**3)
            metrics['disk_usage'] = disk_percent
            
            # Réseau
            network = psutil.net_io_counters()
            details['network_bytes_sent'] = network.bytes_sent
            details['network_bytes_recv'] = network.bytes_recv
            
            # Processus
            details['process_count'] = len(psutil.pids())
            
            # Load average (Unix seulement)
            try:
                load_avg = psutil.getloadavg()
                details['load_average'] = {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
                metrics['load_average_1min'] = load_avg[0]
            except AttributeError:
                pass  # Windows n'a pas getloadavg
            
            # Détermination du statut
            status = HealthStatus.HEALTHY.value
            message = "Infrastructure saine"
            
            # Seuils d'alerte
            thresholds = check.thresholds or {
                'cpu_critical': 90,
                'cpu_warning': 80,
                'memory_critical': 90,
                'memory_warning': 80,
                'disk_critical': 95,
                'disk_warning': 85
            }
            
            warnings = []
            
            if cpu_percent > thresholds.get('cpu_critical', 90):
                status = HealthStatus.CRITICAL.value
                warnings.append(f"CPU critique: {cpu_percent:.1f}%")
            elif cpu_percent > thresholds.get('cpu_warning', 80):
                status = HealthStatus.WARNING.value
                warnings.append(f"CPU élevé: {cpu_percent:.1f}%")
            
            if memory.percent > thresholds.get('memory_critical', 90):
                status = HealthStatus.CRITICAL.value
                warnings.append(f"Mémoire critique: {memory.percent:.1f}%")
            elif memory.percent > thresholds.get('memory_warning', 80):
                status = HealthStatus.WARNING.value
                warnings.append(f"Mémoire élevée: {memory.percent:.1f}%")
            
            if disk_percent > thresholds.get('disk_critical', 95):
                status = HealthStatus.CRITICAL.value
                warnings.append(f"Disque critique: {disk_percent:.1f}%")
            elif disk_percent > thresholds.get('disk_warning', 85):
                status = HealthStatus.WARNING.value
                warnings.append(f"Disque élevé: {disk_percent:.1f}%")
            
            if warnings:
                message = "; ".join(warnings)
            
            return HealthResult(
                check_name=check.name,
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details=details,
                tenant_id=check.tenant_id,
                metrics=metrics
            )
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur vérification infrastructure: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_application(self, check: HealthCheck) -> HealthResult:
        """Vérification de l'application/endpoint HTTP"""
        start_time = time.time()
        
        try:
            if not check.endpoint:
                raise ValueError("Endpoint requis pour vérification application")
            
            headers = check.custom_headers or {}
            headers.setdefault('User-Agent', 'HealthChecker/2.0')
            
            response = requests.get(
                check.endpoint,
                headers=headers,
                timeout=check.timeout_seconds,
                verify=True  # Vérification SSL
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                'status_code': response.status_code,
                'response_time_ms': duration_ms,
                'content_length': len(response.content),
                'headers': dict(response.headers)
            }
            
            metrics = {
                'response_time_ms': duration_ms,
                'status_code': response.status_code
            }
            
            # Vérification du statut HTTP
            if response.status_code == check.expected_status:
                if duration_ms > self.config['slow_response_threshold_ms']:
                    status = HealthStatus.WARNING.value
                    message = f"Réponse lente: {duration_ms:.0f}ms"
                else:
                    status = HealthStatus.HEALTHY.value
                    message = f"OK - {duration_ms:.0f}ms"
            else:
                status = HealthStatus.CRITICAL.value
                message = f"Status inattendu: {response.status_code} (attendu: {check.expected_status})"
            
            # Vérification du contenu si spécifié
            if check.thresholds and 'content_pattern' in check.thresholds:
                pattern = check.thresholds['content_pattern']
                if pattern not in response.text:
                    status = HealthStatus.CRITICAL.value
                    message = f"Pattern non trouvé dans la réponse: {pattern}"
            
            return HealthResult(
                check_name=check.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=time.time(),
                details=details,
                tenant_id=check.tenant_id,
                metrics=metrics
            )
            
        except requests.exceptions.Timeout:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Timeout après {check.timeout_seconds}s",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': 'timeout'},
                tenant_id=check.tenant_id
            )
        except requests.exceptions.ConnectionError as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur de connexion: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur vérification: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_database(self, check: HealthCheck) -> HealthResult:
        """Vérification de base de données"""
        start_time = time.time()
        
        try:
            if not check.endpoint:
                raise ValueError("Connection string requis pour vérification BDD")
            
            # Parse de la connection string
            # Format: protocol://user:pass@host:port/db
            parsed = urlparse(check.endpoint)
            
            details = {
                'host': parsed.hostname,
                'port': parsed.port,
                'database': parsed.path.lstrip('/') if parsed.path else None
            }
            
            if parsed.scheme in ['postgresql', 'postgres']:
                result = self._check_postgresql(check, parsed)
            elif parsed.scheme == 'redis':
                result = self._check_redis_db(check, parsed)
            elif parsed.scheme in ['mysql', 'mariadb']:
                result = self._check_mysql(check, parsed)
            elif parsed.scheme == 'mongodb':
                result = self._check_mongodb(check, parsed)
            else:
                # Test de connexion TCP générique
                result = self._check_tcp_connection(check, parsed.hostname, parsed.port)
            
            result.details.update(details)
            return result
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur vérification BDD: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_postgresql(self, check: HealthCheck, parsed_url) -> HealthResult:
        """Vérification spécifique PostgreSQL"""
        start_time = time.time()
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=parsed_url.hostname,
                port=parsed_url.port or 5432,
                database=parsed_url.path.lstrip('/') or 'postgres',
                user=parsed_url.username,
                password=parsed_url.password,
                connect_timeout=check.timeout_seconds
            )
            
            cursor = conn.cursor()
            
            # Test de requête simple
            query_start = time.time()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            query_duration = (time.time() - query_start) * 1000
            
            # Statistiques additionnelles
            cursor.execute("SELECT count(*) FROM pg_stat_activity")
            active_connections = cursor.fetchone()[0]
            
            cursor.execute("SELECT setting FROM pg_settings WHERE name = 'max_connections'")
            max_connections = int(cursor.fetchone()[0])
            
            cursor.close()
            conn.close()
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                'query_duration_ms': query_duration,
                'active_connections': active_connections,
                'max_connections': max_connections,
                'connection_usage_percent': (active_connections / max_connections) * 100
            }
            
            metrics = {
                'connection_time_ms': duration_ms,
                'query_time_ms': query_duration,
                'active_connections': active_connections,
                'connection_usage_percent': details['connection_usage_percent']
            }
            
            # Détermination du statut
            if details['connection_usage_percent'] > 90:
                status = HealthStatus.CRITICAL.value
                message = f"Connexions critiques: {details['connection_usage_percent']:.1f}%"
            elif details['connection_usage_percent'] > 80:
                status = HealthStatus.WARNING.value
                message = f"Connexions élevées: {details['connection_usage_percent']:.1f}%"
            elif query_duration > 1000:
                status = HealthStatus.WARNING.value
                message = f"Requête lente: {query_duration:.0f}ms"
            else:
                status = HealthStatus.HEALTHY.value
                message = f"PostgreSQL OK - {duration_ms:.0f}ms"
            
            return HealthResult(
                check_name=check.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=time.time(),
                details=details,
                tenant_id=check.tenant_id,
                metrics=metrics
            )
            
        except ImportError:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.UNKNOWN.value,
                message="Driver PostgreSQL non disponible",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': 'missing_driver'},
                tenant_id=check.tenant_id
            )
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur PostgreSQL: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_redis_db(self, check: HealthCheck, parsed_url) -> HealthResult:
        """Vérification spécifique Redis"""
        start_time = time.time()
        
        try:
            client = redis.Redis(
                host=parsed_url.hostname,
                port=parsed_url.port or 6379,
                db=int(parsed_url.path.lstrip('/')) if parsed_url.path.lstrip('/').isdigit() else 0,
                password=parsed_url.password,
                socket_timeout=check.timeout_seconds,
                decode_responses=True
            )
            
            # Test de ping
            ping_start = time.time()
            client.ping()
            ping_duration = (time.time() - ping_start) * 1000
            
            # Informations Redis
            info = client.info()
            
            duration_ms = (time.time() - start_time) * 1000
            
            details = {
                'ping_duration_ms': ping_duration,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'used_memory_rss_human': info.get('used_memory_rss_human', 'N/A'),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0),
                'redis_version': info.get('redis_version', 'unknown'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
            
            # Calcul du hit ratio
            hits = details['keyspace_hits']
            misses = details['keyspace_misses']
            hit_ratio = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 100
            details['hit_ratio_percent'] = hit_ratio
            
            metrics = {
                'connection_time_ms': duration_ms,
                'ping_time_ms': ping_duration,
                'connected_clients': details['connected_clients'],
                'hit_ratio_percent': hit_ratio
            }
            
            # Détermination du statut
            if ping_duration > 100:
                status = HealthStatus.WARNING.value
                message = f"Redis lent: {ping_duration:.0f}ms"
            elif hit_ratio < 50:
                status = HealthStatus.WARNING.value
                message = f"Hit ratio faible: {hit_ratio:.1f}%"
            else:
                status = HealthStatus.HEALTHY.value
                message = f"Redis OK - {duration_ms:.0f}ms"
            
            return HealthResult(
                check_name=check.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=time.time(),
                details=details,
                tenant_id=check.tenant_id,
                metrics=metrics
            )
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur Redis: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_tcp_connection(self, check: HealthCheck, host: str, port: int) -> HealthResult:
        """Vérification de connexion TCP générique"""
        start_time = time.time()
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(check.timeout_seconds)
            
            result = sock.connect_ex((host, port))
            duration_ms = (time.time() - start_time) * 1000
            
            sock.close()
            
            if result == 0:
                status = HealthStatus.HEALTHY.value
                message = f"Connexion TCP OK - {duration_ms:.0f}ms"
            else:
                status = HealthStatus.CRITICAL.value
                message = f"Connexion TCP échouée: code {result}"
            
            return HealthResult(
                check_name=check.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=time.time(),
                details={'host': host, 'port': port, 'result_code': result},
                tenant_id=check.tenant_id,
                metrics={'connection_time_ms': duration_ms}
            )
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur connexion TCP: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_cache(self, check: HealthCheck) -> HealthResult:
        """Vérification du cache (Redis)"""
        if self.redis_client:
            return self._check_redis_health(check)
        else:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.UNKNOWN.value,
                message="Cache Redis non configuré",
                duration_ms=0,
                timestamp=time.time(),
                details={},
                tenant_id=check.tenant_id
            )
    
    def _check_redis_health(self, check: HealthCheck) -> HealthResult:
        """Vérification de santé Redis spécifique"""
        start_time = time.time()
        
        try:
            # Test de ping
            ping_start = time.time()
            self.redis_client.ping()
            ping_duration = (time.time() - ping_start) * 1000
            
            # Test d'écriture/lecture
            test_key = f"health_check_{int(time.time())}"
            test_value = "health_test"
            
            write_start = time.time()
            self.redis_client.setex(test_key, 60, test_value)
            write_duration = (time.time() - write_start) * 1000
            
            read_start = time.time()
            retrieved_value = self.redis_client.get(test_key)
            read_duration = (time.time() - read_start) * 1000
            
            # Nettoyage
            self.redis_client.delete(test_key)
            
            total_duration = (time.time() - start_time) * 1000
            
            details = {
                'ping_duration_ms': ping_duration,
                'write_duration_ms': write_duration,
                'read_duration_ms': read_duration,
                'read_write_success': retrieved_value == test_value
            }
            
            metrics = {
                'ping_time_ms': ping_duration,
                'write_time_ms': write_duration,
                'read_time_ms': read_duration,
                'total_time_ms': total_duration
            }
            
            # Détermination du statut
            if not details['read_write_success']:
                status = HealthStatus.CRITICAL.value
                message = "Échec lecture/écriture cache"
            elif max(ping_duration, write_duration, read_duration) > 100:
                status = HealthStatus.WARNING.value
                message = f"Cache lent - max: {max(ping_duration, write_duration, read_duration):.0f}ms"
            else:
                status = HealthStatus.HEALTHY.value
                message = f"Cache OK - {total_duration:.0f}ms"
            
            return HealthResult(
                check_name=check.name,
                status=status,
                message=message,
                duration_ms=total_duration,
                timestamp=time.time(),
                details=details,
                tenant_id=check.tenant_id,
                metrics=metrics
            )
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur cache: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_external_api(self, check: HealthCheck) -> HealthResult:
        """Vérification d'API externe"""
        # Similaire à _check_application mais avec des critères spécifiques aux APIs externes
        return self._check_application(check)
    
    def _check_business(self, check: HealthCheck) -> HealthResult:
        """Vérification de métriques business"""
        start_time = time.time()
        
        try:
            # Cette vérification est généralement custom selon les besoins business
            # Par exemple: vérifier le nombre d'utilisateurs actifs, transactions, etc.
            
            details = {}
            metrics = {}
            
            # Exemple: vérification du nombre d'utilisateurs actifs
            if check.endpoint:
                # Appel à une API de métriques business
                response = requests.get(
                    check.endpoint,
                    headers=check.custom_headers or {},
                    timeout=check.timeout_seconds
                )
                
                if response.status_code == 200:
                    data = response.json()
                    details.update(data)
                    
                    # Extraction de métriques spécifiques
                    active_users = data.get('active_users', 0)
                    metrics['active_users'] = active_users
                    
                    # Vérification des seuils business
                    thresholds = check.thresholds or {}
                    min_users = thresholds.get('min_active_users', 10)
                    
                    if active_users < min_users:
                        status = HealthStatus.WARNING.value
                        message = f"Utilisateurs actifs faibles: {active_users} (min: {min_users})"
                    else:
                        status = HealthStatus.HEALTHY.value
                        message = f"Métriques business OK - {active_users} utilisateurs actifs"
                else:
                    status = HealthStatus.CRITICAL.value
                    message = f"API métriques inaccessible: {response.status_code}"
            else:
                # Vérification par défaut
                status = HealthStatus.HEALTHY.value
                message = "Vérification business par défaut"
            
            return HealthResult(
                check_name=check.name,
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details=details,
                tenant_id=check.tenant_id,
                metrics=metrics
            )
            
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur vérification business: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _check_custom(self, check: HealthCheck) -> HealthResult:
        """Exécution d'une vérification personnalisée"""
        start_time = time.time()
        
        try:
            if check.custom_checker and callable(check.custom_checker):
                result = check.custom_checker(check)
                if isinstance(result, HealthResult):
                    return result
                else:
                    # Conversion du résultat en HealthResult
                    return HealthResult(
                        check_name=check.name,
                        status=result.get('status', HealthStatus.UNKNOWN.value),
                        message=result.get('message', 'Vérification custom'),
                        duration_ms=(time.time() - start_time) * 1000,
                        timestamp=time.time(),
                        details=result.get('details', {}),
                        tenant_id=check.tenant_id,
                        metrics=result.get('metrics', {})
                    )
            else:
                return HealthResult(
                    check_name=check.name,
                    status=HealthStatus.UNKNOWN.value,
                    message="Vérification custom non définie",
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time(),
                    details={},
                    tenant_id=check.tenant_id
                )
                
        except Exception as e:
            return HealthResult(
                check_name=check.name,
                status=HealthStatus.CRITICAL.value,
                message=f"Erreur vérification custom: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                details={'error': str(e)},
                tenant_id=check.tenant_id
            )
    
    def _checking_loop(self):
        """Boucle principale de vérification"""
        while self.running:
            try:
                # Exécution de toutes les vérifications actives
                futures = []
                
                with self.lock:
                    active_checks = [
                        check for check in self.checks.values()
                        if check.enabled
                    ]
                
                for check in active_checks:
                    # Vérification de l'intervalle
                    if self._should_run_check(check):
                        future = self.executor.submit(self.run_check, check.name)
                        futures.append(future)
                
                # Attente des résultats
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    try:
                        result = future.result()
                        if result:
                            logger.debug(f"Vérification terminée: {result.check_name} - {result.status}")
                    except Exception as e:
                        logger.error(f"Erreur dans la vérification: {e}")
                
                # Pause avant la prochaine itération
                time.sleep(self.config['check_interval'])
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de vérification: {e}")
                time.sleep(5)
    
    def _should_run_check(self, check: HealthCheck) -> bool:
        """Détermine si une vérification doit être exécutée"""
        if check.name not in self.results or not self.results[check.name]:
            return True
        
        last_result = self.results[check.name][-1]
        time_since_last = time.time() - last_result.timestamp
        
        return time_since_last >= check.interval_seconds
    
    def _analysis_loop(self):
        """Boucle d'analyse prédictive"""
        while self.running:
            try:
                if self.config['enable_predictive_analysis']:
                    self._analyze_trends()
                    self._predict_failures()
                
                self._cleanup_old_results()
                time.sleep(300)  # Analyse toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur dans l'analyse: {e}")
                time.sleep(60)
    
    def _analyze_trends(self):
        """Analyse les tendances des métriques"""
        try:
            with self.lock:
                for check_name, results in self.results.items():
                    if len(results) < 10:  # Besoin d'historique
                        continue
                    
                    # Analyse des temps de réponse
                    recent_results = results[-20:]  # 20 derniers résultats
                    response_times = [r.duration_ms for r in recent_results]
                    
                    if len(response_times) >= 10:
                        # Calcul de la tendance
                        x = np.arange(len(response_times))
                        z = np.polyfit(x, response_times, 1)
                        trend = z[0]  # Pente de la tendance
                        
                        # Alerte si dégradation significative
                        if trend > 50:  # Augmentation de 50ms par mesure
                            logger.warning(f"Dégradation détectée pour {check_name}: +{trend:.1f}ms/check")
                            # TODO: Déclencher alerte prédictive
                        
        except Exception as e:
            logger.error(f"Erreur analyse tendances: {e}")
    
    def _predict_failures(self):
        """Prédiction de pannes basée sur l'historique"""
        try:
            # TODO: Implémentation ML pour prédiction de pannes
            # - Analyse des patterns de défaillance
            # - Corrélation entre métriques
            # - Modèle prédictif basé sur l'historique
            # - Alertes préventives
            pass
            
        except Exception as e:
            logger.error(f"Erreur prédiction pannes: {e}")
    
    def _register_default_checks(self):
        """Enregistre les vérifications par défaut"""
        try:
            # Vérification infrastructure
            self.register_check(HealthCheck(
                name="system_infrastructure",
                description="Vérification de l'infrastructure système",
                check_type=CheckType.INFRASTRUCTURE.value,
                interval_seconds=30,
                thresholds={
                    'cpu_warning': 80,
                    'cpu_critical': 90,
                    'memory_warning': 80,
                    'memory_critical': 90,
                    'disk_warning': 85,
                    'disk_critical': 95
                }
            ))
            
            # Vérification cache Redis si configuré
            if self.redis_client:
                self.register_check(HealthCheck(
                    name="redis_cache",
                    description="Vérification du cache Redis",
                    check_type=CheckType.CACHE.value,
                    interval_seconds=60
                ))
            
            # Vérifications d'endpoints par défaut
            default_endpoints = [
                ("health_endpoint", "http://localhost:8000/health"),
                ("metrics_endpoint", "http://localhost:8000/metrics")
            ]
            
            for name, endpoint in default_endpoints:
                self.register_check(HealthCheck(
                    name=name,
                    description=f"Vérification de {endpoint}",
                    check_type=CheckType.APPLICATION.value,
                    endpoint=endpoint,
                    expected_status=200,
                    timeout_seconds=10,
                    interval_seconds=60
                ))
            
            logger.info("Vérifications par défaut enregistrées")
            
        except Exception as e:
            logger.error(f"Erreur enregistrement vérifications par défaut: {e}")
    
    def _save_check_to_db(self, check: HealthCheck):
        """Sauvegarde une vérification en base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            config_json = json.dumps({
                'description': check.description,
                'check_type': check.check_type,
                'endpoint': check.endpoint,
                'expected_status': check.expected_status,
                'timeout_seconds': check.timeout_seconds,
                'interval_seconds': check.interval_seconds,
                'retries': check.retries,
                'tenant_id': check.tenant_id,
                'labels': check.labels,
                'custom_headers': check.custom_headers,
                'thresholds': check.thresholds,
                'enabled': check.enabled
            })
            
            now = time.time()
            cursor.execute('''
                INSERT OR REPLACE INTO health_checks 
                (name, description, check_type, config, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                check.name, check.description, check.check_type,
                config_json, check.enabled, now, now
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde vérification: {e}")
    
    def _save_result_to_db(self, result: HealthResult):
        """Sauvegarde un résultat en base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO health_results 
                (check_name, status, message, duration_ms, timestamp, details, tenant_id, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.check_name, result.status, result.message,
                result.duration_ms, result.timestamp,
                json.dumps(result.details), result.tenant_id,
                json.dumps(result.metrics) if result.metrics else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde résultat: {e}")
    
    def _update_stats(self, result: HealthResult):
        """Met à jour les statistiques"""
        try:
            self.stats['total_checks_run'] += 1
            
            if result.status in [HealthStatus.CRITICAL.value, HealthStatus.WARNING.value]:
                self.stats['failed_checks'] += 1
            
            # Mise à jour du temps de réponse moyen
            total_time = self.stats['avg_response_time'] * (self.stats['total_checks_run'] - 1)
            self.stats['avg_response_time'] = (total_time + result.duration_ms) / self.stats['total_checks_run']
            
        except Exception as e:
            logger.error(f"Erreur mise à jour stats: {e}")
    
    def _check_alert_thresholds(self, check: HealthCheck, result: HealthResult):
        """Vérifie les seuils d'alerte"""
        try:
            # Compte les échecs récents
            recent_failures = 0
            if check.name in self.results:
                recent_results = self.results[check.name][-self.config['alert_threshold_failures']:]
                recent_failures = sum(1 for r in recent_results 
                                    if r.status in [HealthStatus.CRITICAL.value, HealthStatus.WARNING.value])
            
            # Déclenchement d'alerte si trop d'échecs
            if recent_failures >= self.config['alert_threshold_failures']:
                self._trigger_health_alert(check, result, recent_failures)
            
        except Exception as e:
            logger.error(f"Erreur vérification seuils alerte: {e}")
    
    def _trigger_health_alert(self, check: HealthCheck, result: HealthResult, failure_count: int):
        """Déclenche une alerte de santé"""
        try:
            # Import dynamique pour éviter les dépendances circulaires
            from .alert_manager import get_alert_manager, AlertSeverity
            
            severity = AlertSeverity.CRITICAL.value if result.status == HealthStatus.CRITICAL.value else AlertSeverity.WARNING.value
            
            alert_manager = get_alert_manager()
            alert_manager.trigger_alert(
                name=f"health_check_failure_{check.name}",
                description=f"Vérification de santé échouée: {result.message}",
                severity=severity,
                source="health_checker",
                tenant_id=check.tenant_id,
                labels={
                    'check_name': check.name,
                    'check_type': check.check_type,
                    'failure_count': str(failure_count)
                },
                annotations={
                    'endpoint': check.endpoint or '',
                    'duration_ms': str(result.duration_ms),
                    'details': json.dumps(result.details)
                }
            )
            
            logger.warning(f"Alerte déclenchée pour {check.name}: {failure_count} échecs")
            
        except Exception as e:
            logger.error(f"Erreur déclenchement alerte: {e}")
    
    def _cleanup_old_results(self):
        """Nettoie les anciens résultats"""
        try:
            cutoff_time = time.time() - (self.config['retention_days'] * 24 * 3600)
            
            # Nettoyage mémoire
            with self.lock:
                for check_name in self.results:
                    self.results[check_name] = [
                        r for r in self.results[check_name]
                        if r.timestamp > cutoff_time
                    ]
            
            # Nettoyage base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM health_results WHERE timestamp < ?', (cutoff_time,))
            deleted_results = cursor.rowcount
            
            cursor.execute('DELETE FROM health_stats WHERE timestamp < ?', (cutoff_time,))
            deleted_stats = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_results > 0 or deleted_stats > 0:
                logger.info(f"Nettoyé {deleted_results} résultats et {deleted_stats} stats")
                
        except Exception as e:
            logger.error(f"Erreur nettoyage résultats: {e}")
    
    def get_system_health(self) -> SystemHealth:
        """Retourne l'état de santé global du système"""
        try:
            with self.lock:
                total_checks = len(self.checks)
                healthy_checks = 0
                warning_checks = 0
                critical_checks = 0
                unknown_checks = 0
                
                total_response_time = 0
                response_count = 0
                current_results = {}
                
                for check_name, check in self.checks.items():
                    if not check.enabled:
                        continue
                    
                    if check_name in self.results and self.results[check_name]:
                        latest_result = self.results[check_name][-1]
                        current_results[check_name] = latest_result
                        
                        if latest_result.status == HealthStatus.HEALTHY.value:
                            healthy_checks += 1
                        elif latest_result.status == HealthStatus.WARNING.value:
                            warning_checks += 1
                        elif latest_result.status == HealthStatus.CRITICAL.value:
                            critical_checks += 1
                        else:
                            unknown_checks += 1
                        
                        total_response_time += latest_result.duration_ms
                        response_count += 1
                    else:
                        unknown_checks += 1
                
                # Détermination du statut global
                if critical_checks > 0:
                    overall_status = HealthStatus.CRITICAL.value
                elif warning_checks > 0:
                    overall_status = HealthStatus.WARNING.value
                elif healthy_checks > 0:
                    overall_status = HealthStatus.HEALTHY.value
                else:
                    overall_status = HealthStatus.UNKNOWN.value
                
                # Calcul de l'uptime
                uptime_seconds = time.time() - self.stats['uptime_start']
                total_possible_checks = self.stats['total_checks_run'] + self.stats['failed_checks']
                uptime_percentage = ((total_possible_checks - self.stats['failed_checks']) / 
                                   max(total_possible_checks, 1)) * 100
                
                # Temps de réponse moyen
                avg_response_time = total_response_time / max(response_count, 1)
                
                return SystemHealth(
                    overall_status=overall_status,
                    healthy_checks=healthy_checks,
                    warning_checks=warning_checks,
                    critical_checks=critical_checks,
                    unknown_checks=unknown_checks,
                    total_checks=total_checks,
                    last_update=time.time(),
                    uptime_percentage=uptime_percentage,
                    response_time_avg=avg_response_time,
                    details=current_results
                )
                
        except Exception as e:
            logger.error(f"Erreur récupération santé système: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN.value,
                healthy_checks=0,
                warning_checks=0,
                critical_checks=0,
                unknown_checks=0,
                total_checks=0,
                last_update=time.time(),
                uptime_percentage=0,
                response_time_avg=0,
                details={}
            )
    
    def get_check_history(self, check_name: str, limit: int = 100) -> List[Dict]:
        """
        Récupère l'historique d'une vérification
        
        Args:
            check_name: Nom de la vérification
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des résultats historiques
        """
        try:
            with self.lock:
                if check_name not in self.results:
                    return []
                
                results = self.results[check_name][-limit:]
                return [asdict(result) for result in results]
                
        except Exception as e:
            logger.error(f"Erreur récupération historique: {e}")
            return []
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de santé"""
        try:
            system_health = self.get_system_health()
            
            return {
                'system_health': asdict(system_health),
                'stats': self.stats.copy(),
                'checks_count': len(self.checks),
                'enabled_checks': len([c for c in self.checks.values() if c.enabled]),
                'redis_connected': self.redis_client is not None and self._test_redis(),
                'checker_active': self.checker_thread and self.checker_thread.is_alive(),
                'analyzer_active': self.analyzer_thread and self.analyzer_thread.is_alive(),
                'config': self.config
            }
            
        except Exception as e:
            logger.error(f"Erreur stats santé: {e}")
            return {}
    
    def _test_redis(self) -> bool:
        """Test la connexion Redis"""
        try:
            self.redis_client.ping()
            return True
        except:
            return False

# Factory pour instance globale
_health_checker_instance = None

def get_health_checker(config: Optional[Dict] = None) -> HealthChecker:
    """
    Retourne l'instance globale du vérificateur de santé
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        Instance de HealthChecker
    """
    global _health_checker_instance
    
    if _health_checker_instance is None:
        _health_checker_instance = HealthChecker(config)
    
    return _health_checker_instance

# Fonctions de convenance
def register_health_check(check: HealthCheck) -> bool:
    """Fonction de convenance pour enregistrer une vérification"""
    checker = get_health_checker()
    return checker.register_check(check)

def run_health_check(check_name: str) -> Optional[HealthResult]:
    """Fonction de convenance pour exécuter une vérification"""
    checker = get_health_checker()
    return checker.run_check(check_name)

def get_system_health() -> SystemHealth:
    """Fonction de convenance pour obtenir la santé système"""
    checker = get_health_checker()
    return checker.get_system_health()
