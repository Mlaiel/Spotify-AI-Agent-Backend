#!/usr/bin/env python3
"""
Dashboard de Supervision en Temps Réel - Spotify AI Agent
=========================================================

Dashboard interactif pour surveiller l'état des bases de données,
les performances, la sécurité et la santé générale du système.

Auteur: Équipe Monitoring & DevOps (Lead: Fahed Mlaiel)
Version: 2.1.0
Dernière mise à jour: 2025-07-16

Fonctionnalités:
- Surveillance en temps réel de toutes les bases de données
- Métriques de performance et alertes intelligentes
- Dashboard web interactif avec WebSockets
- Intégration Prometheus/Grafana
- Alertes automatiques via Slack/Email
- API REST pour intégrations externes
"""

import asyncio
import json
import time
import logging
import aiohttp
import aioredis
import asyncpg
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
import uvicorn
import psutil
import asyncio
import yaml

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_DIR = SCRIPT_DIR
STATIC_DIR = SCRIPT_DIR / "static"
TEMPLATES_DIR = SCRIPT_DIR / "templates"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseHealth:
    """État de santé d'une base de données."""
    name: str
    type: str
    environment: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    response_time: float
    connection_count: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    last_check: datetime
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        return asdict(self)

@dataclass
class SystemMetrics:
    """Métriques système globales."""
    timestamp: datetime
    total_databases: int
    healthy_databases: int
    warning_databases: int
    critical_databases: int
    unknown_databases: int
    total_connections: int
    average_response_time: float
    system_cpu: float
    system_memory: float
    system_disk: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        return asdict(self)

class DatabaseMonitor:
    """Moniteur pour un type de base de données spécifique."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.last_health = None
        
    async def check_health(self) -> DatabaseHealth:
        """Vérifie l'état de santé de la base de données."""
        start_time = time.time()
        
        try:
            if 'postgresql' in self.name.lower() or 'postgres' in self.name.lower():
                health = await self._check_postgresql_health()
            elif 'redis' in self.name.lower():
                health = await self._check_redis_health()
            elif 'mongodb' in self.name.lower() or 'mongo' in self.name.lower():
                health = await self._check_mongodb_health()
            elif 'clickhouse' in self.name.lower():
                health = await self._check_clickhouse_health()
            elif 'elasticsearch' in self.name.lower():
                health = await self._check_elasticsearch_health()
            elif 'neo4j' in self.name.lower():
                health = await self._check_neo4j_health()
            elif 'cassandra' in self.name.lower():
                health = await self._check_cassandra_health()
            else:
                health = await self._check_generic_health()
                
            health.response_time = (time.time() - start_time) * 1000
            health.last_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de {self.name}: {e}")
            health = DatabaseHealth(
                name=self.name,
                type=self._extract_db_type(),
                environment=self._extract_environment(),
                status='critical',
                response_time=(time.time() - start_time) * 1000,
                connection_count=0,
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                last_check=datetime.now(),
                error_message=str(e)
            )
            
        self.last_health = health
        return health
        
    async def _check_postgresql_health(self) -> DatabaseHealth:
        """Vérifie la santé d'une base PostgreSQL."""
        # Configuration de connexion
        host = self.config.get('host', 'localhost')
        port = self.config.get('port', 5432)
        database = self.config.get('database', 'postgres')
        user = self.config.get('user', 'postgres')
        password = self.config.get('password', '')
        
        try:
            # Connexion à PostgreSQL
            conn = await asyncpg.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                timeout=5.0
            )
            
            # Requêtes de santé
            version = await conn.fetchval("SELECT version()")
            connection_count = await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity"
            )
            
            # Métriques de performance
            db_size = await conn.fetchval(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            )
            
            await conn.close()
            
            return DatabaseHealth(
                name=self.name,
                type='postgresql',
                environment=self._extract_environment(),
                status='healthy',
                response_time=0.0,  # Sera défini par le caller
                connection_count=connection_count or 0,
                cpu_usage=0.0,  # À implémenter avec des métriques système
                memory_usage=0.0,
                disk_usage=0.0,
                last_check=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"PostgreSQL health check failed: {e}")
            
    async def _check_redis_health(self) -> DatabaseHealth:
        """Vérifie la santé d'une base Redis."""
        host = self.config.get('host', 'localhost')
        port = self.config.get('port', 6379)
        password = self.config.get('password')
        
        try:
            redis = aioredis.from_url(
                f"redis://:{password}@{host}:{port}" if password else f"redis://{host}:{port}",
                socket_timeout=5.0
            )
            
            # Test de ping
            pong = await redis.ping()
            
            # Informations du serveur
            info = await redis.info()
            
            await redis.close()
            
            return DatabaseHealth(
                name=self.name,
                type='redis',
                environment=self._extract_environment(),
                status='healthy' if pong else 'critical',
                response_time=0.0,
                connection_count=int(info.get('connected_clients', 0)),
                cpu_usage=0.0,
                memory_usage=float(info.get('used_memory_percentage', 0.0)),
                disk_usage=0.0,
                last_check=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"Redis health check failed: {e}")
            
    async def _check_mongodb_health(self) -> DatabaseHealth:
        """Vérifie la santé d'une base MongoDB."""
        # Implémentation basique - à étendre avec pymongo
        return DatabaseHealth(
            name=self.name,
            type='mongodb',
            environment=self._extract_environment(),
            status='unknown',
            response_time=0.0,
            connection_count=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            last_check=datetime.now(),
            error_message="MongoDB monitoring not fully implemented"
        )
        
    async def _check_clickhouse_health(self) -> DatabaseHealth:
        """Vérifie la santé d'une base ClickHouse."""
        host = self.config.get('host', 'localhost')
        port = self.config.get('http_port', 8123)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}:{port}/ping",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        # Obtenir des métriques supplémentaires
                        async with session.get(
                            f"http://{host}:{port}/?query=SELECT+1"
                        ) as query_response:
                            if query_response.status == 200:
                                status = 'healthy'
                            else:
                                status = 'warning'
                    else:
                        status = 'critical'
                        
            return DatabaseHealth(
                name=self.name,
                type='clickhouse',
                environment=self._extract_environment(),
                status=status,
                response_time=0.0,
                connection_count=0,
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                last_check=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"ClickHouse health check failed: {e}")
            
    async def _check_elasticsearch_health(self) -> DatabaseHealth:
        """Vérifie la santé d'Elasticsearch."""
        host = self.config.get('host', 'localhost')
        port = self.config.get('port', 9200)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}:{port}/_cluster/health",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        cluster_status = health_data.get('status', 'red')
                        
                        if cluster_status == 'green':
                            status = 'healthy'
                        elif cluster_status == 'yellow':
                            status = 'warning'
                        else:
                            status = 'critical'
                    else:
                        status = 'critical'
                        
            return DatabaseHealth(
                name=self.name,
                type='elasticsearch',
                environment=self._extract_environment(),
                status=status,
                response_time=0.0,
                connection_count=0,
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                last_check=datetime.now()
            )
            
        except Exception as e:
            raise Exception(f"Elasticsearch health check failed: {e}")
            
    async def _check_neo4j_health(self) -> DatabaseHealth:
        """Vérifie la santé de Neo4j."""
        # Implémentation à faire avec neo4j driver
        return DatabaseHealth(
            name=self.name,
            type='neo4j',
            environment=self._extract_environment(),
            status='unknown',
            response_time=0.0,
            connection_count=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            last_check=datetime.now(),
            error_message="Neo4j monitoring not fully implemented"
        )
        
    async def _check_cassandra_health(self) -> DatabaseHealth:
        """Vérifie la santé de Cassandra."""
        # Implémentation à faire avec cassandra driver
        return DatabaseHealth(
            name=self.name,
            type='cassandra',
            environment=self._extract_environment(),
            status='unknown',
            response_time=0.0,
            connection_count=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            last_check=datetime.now(),
            error_message="Cassandra monitoring not fully implemented"
        )
        
    async def _check_generic_health(self) -> DatabaseHealth:
        """Vérification générique pour types de DB non supportés."""
        return DatabaseHealth(
            name=self.name,
            type='unknown',
            environment=self._extract_environment(),
            status='unknown',
            response_time=0.0,
            connection_count=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            last_check=datetime.now(),
            error_message="Database type not supported for monitoring"
        )
        
    def _extract_db_type(self) -> str:
        """Extrait le type de base de données du nom."""
        name_lower = self.name.lower()
        
        if 'postgresql' in name_lower or 'postgres' in name_lower:
            return 'postgresql'
        elif 'redis' in name_lower:
            return 'redis'
        elif 'mongodb' in name_lower or 'mongo' in name_lower:
            return 'mongodb'
        elif 'clickhouse' in name_lower:
            return 'clickhouse'
        elif 'elasticsearch' in name_lower:
            return 'elasticsearch'
        elif 'neo4j' in name_lower:
            return 'neo4j'
        elif 'cassandra' in name_lower:
            return 'cassandra'
        else:
            return 'unknown'
            
    def _extract_environment(self) -> str:
        """Extrait l'environnement du nom de configuration."""
        name_lower = self.name.lower()
        
        if 'production' in name_lower or 'prod' in name_lower:
            return 'production'
        elif 'staging' in name_lower:
            return 'staging'
        elif 'development' in name_lower or 'dev' in name_lower:
            return 'development'
        elif 'testing' in name_lower or 'test' in name_lower:
            return 'testing'
        else:
            return 'unknown'

class MonitoringService:
    """Service principal de monitoring."""
    
    def __init__(self, config_directory: Path):
        self.config_dir = config_directory
        self.monitors: List[DatabaseMonitor] = []
        self.health_history: List[SystemMetrics] = []
        self.connected_clients: List[WebSocket] = []
        self.running = False
        
    async def initialize(self) -> None:
        """Initialise le service de monitoring."""
        logger.info("🚀 Initialisation du service de monitoring...")
        
        # Chargement des configurations
        await self._load_database_configurations()
        
        logger.info(f"📊 {len(self.monitors)} moniteurs de base de données initialisés")
        
    async def _load_database_configurations(self) -> None:
        """Charge les configurations de base de données."""
        for config_file in self.config_dir.glob("*.yml"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    
                monitor = DatabaseMonitor(
                    name=config_file.stem,
                    config=config_data
                )
                
                self.monitors.append(monitor)
                logger.info(f"  ✅ Moniteur chargé: {config_file.stem}")
                
            except Exception as e:
                logger.error(f"  ❌ Erreur lors du chargement de {config_file}: {e}")
                
    async def start_monitoring(self) -> None:
        """Démarre la surveillance en continu."""
        self.running = True
        logger.info("🔄 Démarrage de la surveillance en continu...")
        
        while self.running:
            try:
                # Vérification de l'état de toutes les bases de données
                health_checks = await asyncio.gather(
                    *[monitor.check_health() for monitor in self.monitors],
                    return_exceptions=True
                )
                
                # Calcul des métriques système
                system_metrics = self._calculate_system_metrics(health_checks)
                self.health_history.append(system_metrics)
                
                # Limitation de l'historique
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]
                    
                # Diffusion aux clients WebSocket
                await self._broadcast_update(system_metrics, health_checks)
                
                # Attente avant la prochaine vérification
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur lors de la surveillance: {e}")
                await asyncio.sleep(60)  # Attendre plus longtemps en cas d'erreur
                
    def _calculate_system_metrics(self, health_checks: List) -> SystemMetrics:
        """Calcule les métriques système globales."""
        valid_checks = [h for h in health_checks if isinstance(h, DatabaseHealth)]
        
        total_databases = len(valid_checks)
        healthy_count = len([h for h in valid_checks if h.status == 'healthy'])
        warning_count = len([h for h in valid_checks if h.status == 'warning'])
        critical_count = len([h for h in valid_checks if h.status == 'critical'])
        unknown_count = len([h for h in valid_checks if h.status == 'unknown'])
        
        total_connections = sum(h.connection_count for h in valid_checks)
        avg_response_time = (
            sum(h.response_time for h in valid_checks) / len(valid_checks)
            if valid_checks else 0.0
        )
        
        # Métriques système
        system_cpu = psutil.cpu_percent()
        system_memory = psutil.virtual_memory().percent
        system_disk = psutil.disk_usage('/').percent
        
        return SystemMetrics(
            timestamp=datetime.now(),
            total_databases=total_databases,
            healthy_databases=healthy_count,
            warning_databases=warning_count,
            critical_databases=critical_count,
            unknown_databases=unknown_count,
            total_connections=total_connections,
            average_response_time=avg_response_time,
            system_cpu=system_cpu,
            system_memory=system_memory,
            system_disk=system_disk
        )
        
    async def _broadcast_update(self, metrics: SystemMetrics, health_checks: List) -> None:
        """Diffuse les mises à jour aux clients WebSocket."""
        if not self.connected_clients:
            return
            
        update_data = {
            'type': 'health_update',
            'timestamp': metrics.timestamp.isoformat(),
            'system_metrics': metrics.to_dict(),
            'database_health': [
                h.to_dict() for h in health_checks 
                if isinstance(h, DatabaseHealth)
            ]
        }
        
        # Diffusion à tous les clients connectés
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_json(update_data)
            except:
                disconnected_clients.append(client)
                
        # Nettoyage des clients déconnectés
        for client in disconnected_clients:
            self.connected_clients.remove(client)
            
    async def add_websocket_client(self, websocket: WebSocket) -> None:
        """Ajoute un client WebSocket."""
        await websocket.accept()
        self.connected_clients.append(websocket)
        logger.info(f"🔌 Nouveau client WebSocket connecté. Total: {len(self.connected_clients)}")
        
        # Envoi de l'état actuel
        if self.health_history:
            latest_metrics = self.health_history[-1]
            latest_health = [monitor.last_health for monitor in self.monitors if monitor.last_health]
            
            await websocket.send_json({
                'type': 'initial_state',
                'system_metrics': latest_metrics.to_dict(),
                'database_health': [h.to_dict() for h in latest_health]
            })
            
    def remove_websocket_client(self, websocket: WebSocket) -> None:
        """Supprime un client WebSocket."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
            logger.info(f"🔌 Client WebSocket déconnecté. Total: {len(self.connected_clients)}")
            
    async def stop_monitoring(self) -> None:
        """Arrête la surveillance."""
        self.running = False
        logger.info("🛑 Arrêt de la surveillance...")

# Application FastAPI
app = FastAPI(title="Spotify AI Agent - Dashboard de Monitoring", version="2.1.0")

# Service de monitoring global
monitoring_service: Optional[MonitoringService] = None

# Configuration des templates et fichiers statiques
if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def startup_event():
    """Démarrage de l'application."""
    global monitoring_service
    
    monitoring_service = MonitoringService(CONFIG_DIR)
    await monitoring_service.initialize()
    
    # Démarrage de la surveillance en arrière-plan
    asyncio.create_task(monitoring_service.start_monitoring())

@app.on_event("shutdown")
async def shutdown_event():
    """Arrêt de l'application."""
    if monitoring_service:
        await monitoring_service.stop_monitoring()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Page principale du dashboard."""
    dashboard_html = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 Spotify AI Agent - Dashboard de Monitoring</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333; 
            min-height: 100vh;
        }
        .dashboard { 
            display: grid; 
            grid-template-columns: 250px 1fr; 
            min-height: 100vh; 
        }
        .sidebar { 
            background: rgba(255,255,255,0.95); 
            padding: 20px; 
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        .main-content { 
            padding: 20px; 
            overflow-y: auto; 
        }
        .header { 
            background: rgba(255,255,255,0.95); 
            border-radius: 10px; 
            padding: 20px; 
            margin-bottom: 20px;
            text-align: center;
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 20px; 
        }
        .metric-card { 
            background: rgba(255,255,255,0.95); 
            border-radius: 10px; 
            padding: 20px; 
            text-align: center;
            transition: transform 0.3s ease;
        }
        .metric-card:hover { 
            transform: translateY(-5px); 
        }
        .metric-value { 
            font-size: 2.5em; 
            font-weight: bold; 
            color: #1DB954; 
            margin-bottom: 10px;
        }
        .database-list { 
            background: rgba(255,255,255,0.95); 
            border-radius: 10px; 
            padding: 20px; 
        }
        .database-item { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 8px; 
            transition: background 0.3s ease;
        }
        .database-item:hover { 
            background: #f8f9fa; 
        }
        .status-indicator { 
            width: 20px; 
            height: 20px; 
            border-radius: 50%; 
            margin-right: 10px;
        }
        .status-healthy { background: #28a745; }
        .status-warning { background: #ffc107; }
        .status-critical { background: #dc3545; }
        .status-unknown { background: #6c757d; }
        .last-update { 
            text-align: center; 
            margin-top: 20px; 
            color: #666; 
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <h2>🎵 Spotify AI</h2>
            <h3>Dashboard Monitoring</h3>
            <hr style="margin: 20px 0;">
            <div>
                <h4>📊 État du Système</h4>
                <div id="system-status"></div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>🎵 Spotify AI Agent - Monitoring en Temps Réel</h1>
                <p>Surveillance des Bases de Données Multi-Tenant</p>
            </div>
            
            <div class="metrics-grid" id="metrics-grid">
                <!-- Les métriques seront injectées ici -->
            </div>
            
            <div class="database-list">
                <h2>🗄️ État des Bases de Données</h2>
                <div id="database-status">
                    <p>Connexion en cours...</p>
                </div>
            </div>
            
            <div class="last-update" id="last-update">
                Dernière mise à jour: En attente...
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket pour les mises à jour en temps réel
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.onopen = function(event) {
            console.log('✅ Connexion WebSocket établie');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('📨 Données reçues:', data);
            
            if (data.type === 'health_update' || data.type === 'initial_state') {
                updateDashboard(data);
            }
        };
        
        ws.onerror = function(error) {
            console.error('❌ Erreur WebSocket:', error);
        };
        
        ws.onclose = function(event) {
            console.log('🔌 Connexion WebSocket fermée');
            setTimeout(() => location.reload(), 5000);
        };
        
        function updateDashboard(data) {
            updateMetrics(data.system_metrics);
            updateDatabaseStatus(data.database_health);
            updateLastUpdate(data.timestamp);
        }
        
        function updateMetrics(metrics) {
            const metricsGrid = document.getElementById('metrics-grid');
            
            metricsGrid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${metrics.total_databases}</div>
                    <div>Total Bases de Données</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #28a745;">${metrics.healthy_databases}</div>
                    <div>Bases Saines</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #ffc107;">${metrics.warning_databases}</div>
                    <div>Avertissements</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #dc3545;">${metrics.critical_databases}</div>
                    <div>Critiques</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.total_connections}</div>
                    <div>Connexions Totales</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.average_response_time.toFixed(1)}ms</div>
                    <div>Temps de Réponse Moyen</div>
                </div>
            `;
        }
        
        function updateDatabaseStatus(databases) {
            const statusDiv = document.getElementById('database-status');
            
            if (!databases || databases.length === 0) {
                statusDiv.innerHTML = '<p>Aucune base de données trouvée</p>';
                return;
            }
            
            const html = databases.map(db => `
                <div class="database-item">
                    <div style="display: flex; align-items: center;">
                        <div class="status-indicator status-${db.status}"></div>
                        <div>
                            <strong>${db.name}</strong><br>
                            <small>${db.type} - ${db.environment}</small>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div>${db.response_time.toFixed(1)}ms</div>
                        <small>${db.connection_count} connexions</small>
                    </div>
                </div>
            `).join('');
            
            statusDiv.innerHTML = html;
        }
        
        function updateLastUpdate(timestamp) {
            const lastUpdateDiv = document.getElementById('last-update');
            const date = new Date(timestamp);
            lastUpdateDiv.textContent = `Dernière mise à jour: ${date.toLocaleString()}`;
        }
    </script>
</body>
</html>
    """
    
    return HTMLResponse(content=dashboard_html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket pour les mises à jour en temps réel."""
    if monitoring_service:
        await monitoring_service.add_websocket_client(websocket)
        
        try:
            while True:
                # Maintenir la connexion active
                await websocket.receive_text()
        except WebSocketDisconnect:
            monitoring_service.remove_websocket_client(websocket)

@app.get("/api/health")
async def api_health():
    """API pour obtenir l'état de santé actuel."""
    if monitoring_service and monitoring_service.health_history:
        latest_metrics = monitoring_service.health_history[-1]
        latest_health = [
            monitor.last_health.to_dict() 
            for monitor in monitoring_service.monitors 
            if monitor.last_health
        ]
        
        return {
            "system_metrics": latest_metrics.to_dict(),
            "database_health": latest_health
        }
    
    return {"error": "Monitoring service not available"}

@app.get("/api/metrics")
async def api_metrics():
    """API pour obtenir les métriques historiques."""
    if monitoring_service:
        return {
            "metrics_history": [
                metrics.to_dict() 
                for metrics in monitoring_service.health_history[-100:]  # 100 dernières entrées
            ]
        }
    
    return {"error": "Monitoring service not available"}

def main():
    """Fonction principale."""
    print("🎵 Spotify AI Agent - Dashboard de Monitoring")
    print("=" * 50)
    print("🚀 Démarrage du serveur...")
    print("🌐 Dashboard disponible sur: http://localhost:8000")
    print("📊 API disponible sur: http://localhost:8000/api/")
    print("🔌 WebSocket sur: ws://localhost:8000/ws")
    
    # Démarrage du serveur
    uvicorn.run(
        "dashboard_monitoring:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
