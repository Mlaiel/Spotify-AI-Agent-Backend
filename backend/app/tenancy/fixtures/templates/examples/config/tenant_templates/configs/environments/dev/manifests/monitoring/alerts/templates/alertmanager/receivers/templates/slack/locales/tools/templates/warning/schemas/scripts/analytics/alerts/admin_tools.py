#!/usr/bin/env python3
"""
Utilitaires d'administration pour le système d'analytics d'alertes
Outils complets de gestion, monitoring et maintenance
"""

import asyncio
import json
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncpg
import aioredis
import pandas as pd
import tabulate

sys.path.append(str(Path(__file__).parent))

from config.analytics_config import get_analytics_config
from models.alert_models import AlertEvent, AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)

class AnalyticsAdminTool:
    """
    Outil d'administration pour le système d'analytics
    
    Fonctionnalités:
    - Gestion de la base de données
    - Monitoring et métriques
    - Maintenance et nettoyage
    - Import/Export de données
    - Tests de santé
    - Configuration et debugging
    """
    
    def __init__(self):
        self.config = get_analytics_config()
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialisation des connexions"""
        try:
            # Base de données
            self.db_pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=1,
                max_size=5
            )
            
            # Redis
            self.redis_client = await aioredis.from_url(
                self.config.redis_url,
                encoding='utf-8',
                decode_responses=True
            )
            
            logger.info("Connexions initialisées")
            
        except Exception as e:
            logger.error(f"Erreur initialisation: {e}")
            raise
    
    async def close(self):
        """Fermeture des connexions"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()
    
    # === Commandes de base de données ===
    
    async def init_database(self):
        """Initialisation de la base de données"""
        print("🔧 Initialisation de la base de données...")
        
        # Schémas SQL pour les tables principales
        schemas = [
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                service TEXT NOT NULL,
                component TEXT,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                labels JSONB DEFAULT '{}',
                metrics JSONB DEFAULT '{}',
                fingerprint TEXT,
                correlation_id TEXT,
                starts_at TIMESTAMPTZ,
                ends_at TIMESTAMPTZ,
                resolved_at TIMESTAMPTZ,
                acknowledged_at TIMESTAMPTZ,
                acknowledged_by TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS alert_analytics (
                id SERIAL PRIMARY KEY,
                alert_id TEXT REFERENCES alerts(id),
                analysis_timestamp TIMESTAMPTZ NOT NULL,
                anomaly_score FLOAT NOT NULL,
                predicted_impact TEXT,
                recommended_actions TEXT[],
                correlation_events TEXT[],
                risk_assessment JSONB,
                confidence_level FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS correlation_patterns (
                id SERIAL PRIMARY KEY,
                pattern_name TEXT NOT NULL,
                source_service TEXT NOT NULL,
                target_service TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                strength FLOAT NOT NULL,
                confidence FLOAT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMPTZ NOT NULL,
                examples TEXT[],
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(pattern_name, source_service, target_service)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS service_dependencies (
                id SERIAL PRIMARY KEY,
                source_service TEXT NOT NULL,
                target_service TEXT NOT NULL,
                dependency_type TEXT NOT NULL,
                strength FLOAT DEFAULT 1.0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(source_service, target_service, dependency_type)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ml_models (
                id SERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                version TEXT NOT NULL,
                training_data_hash TEXT,
                performance_metrics JSONB,
                model_data BYTEA,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                is_active BOOLEAN DEFAULT FALSE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                component TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value FLOAT NOT NULL,
                labels JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ]
        
        # Index pour les performances
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_alerts_service ON alerts(service);",
            "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);",
            "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);",
            "CREATE INDEX IF NOT EXISTS idx_analytics_alert_id ON alert_analytics(alert_id);",
            "CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON alert_analytics(analysis_timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_patterns_services ON correlation_patterns(source_service, target_service);",
            "CREATE INDEX IF NOT EXISTS idx_dependencies_source ON service_dependencies(source_service);",
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_metrics_component ON system_metrics(component);"
        ]
        
        try:
            async with self.db_pool.acquire() as conn:
                # Création des tables
                for schema in schemas:
                    await conn.execute(schema)
                    print("✅ Table créée/vérifiée")
                
                # Création des index
                for index in indexes:
                    await conn.execute(index)
                    print("✅ Index créé/vérifié")
                
                print("🎉 Base de données initialisée avec succès!")
                
        except Exception as e:
            print(f"❌ Erreur initialisation base de données: {e}")
            raise
    
    async def backup_database(self, output_file: str):
        """Sauvegarde de la base de données"""
        print(f"💾 Sauvegarde vers {output_file}...")
        
        # Tables à sauvegarder
        tables = [
            'alerts', 'alert_analytics', 'correlation_patterns',
            'service_dependencies', 'ml_models', 'system_metrics'
        ]
        
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.dict(),
            'tables': {}
        }
        
        try:
            async with self.db_pool.acquire() as conn:
                for table in tables:
                    rows = await conn.fetch(f"SELECT * FROM {table}")
                    backup_data['tables'][table] = [dict(row) for row in rows]
                    print(f"✅ Table {table}: {len(rows)} enregistrements")
            
            # Sauvegarde en JSON
            with open(output_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            print(f"🎉 Sauvegarde terminée: {output_file}")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            raise
    
    async def restore_database(self, backup_file: str):
        """Restauration de la base de données"""
        print(f"🔄 Restauration depuis {backup_file}...")
        
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            async with self.db_pool.acquire() as conn:
                # Nettoyage préalable (optionnel)
                response = input("⚠️  Vider les tables existantes? (y/N): ")
                if response.lower() == 'y':
                    tables = list(backup_data['tables'].keys())
                    for table in reversed(tables):  # Ordre inverse pour les FK
                        await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                        print(f"🗑️  Table {table} vidée")
                
                # Restauration des données
                for table, rows in backup_data['tables'].items():
                    if not rows:
                        continue
                    
                    # Construction de la requête d'insertion
                    columns = list(rows[0].keys())
                    placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
                    query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    # Insertion par batch
                    batch_size = 1000
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i+batch_size]
                        values = [tuple(row[col] for col in columns) for row in batch]
                        await conn.executemany(query, values)
                    
                    print(f"✅ Table {table}: {len(rows)} enregistrements restaurés")
            
            print("🎉 Restauration terminée avec succès!")
            
        except Exception as e:
            print(f"❌ Erreur restauration: {e}")
            raise
    
    # === Commandes de monitoring ===
    
    async def show_system_status(self):
        """Affichage du statut système"""
        print("📊 Statut du système d'analytics\n")
        
        try:
            # Statistiques base de données
            async with self.db_pool.acquire() as conn:
                # Compteurs d'alertes
                alert_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        COUNT(CASE WHEN status = 'firing' THEN 1 END) as active_alerts,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts,
                        COUNT(CASE WHEN timestamp > NOW() - INTERVAL '1 hour' THEN 1 END) as recent_alerts
                    FROM alerts
                """)
                
                # Statistiques analytics
                analytics_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_analyses,
                        AVG(anomaly_score) as avg_anomaly_score,
                        AVG(confidence_level) as avg_confidence,
                        COUNT(CASE WHEN analysis_timestamp > NOW() - INTERVAL '1 hour' THEN 1 END) as recent_analyses
                    FROM alert_analytics
                """)
                
                # Patterns de corrélation
                correlation_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_patterns,
                        AVG(strength) as avg_strength,
                        COUNT(CASE WHEN last_seen > NOW() - INTERVAL '1 day' THEN 1 END) as active_patterns
                    FROM correlation_patterns
                """)
            
            # Affichage formaté
            data = [
                ["Alertes totales", alert_stats['total_alerts']],
                ["Alertes actives", alert_stats['active_alerts']],
                ["Alertes critiques", alert_stats['critical_alerts']],
                ["Alertes récentes (1h)", alert_stats['recent_alerts']],
                ["", ""],
                ["Analyses totales", analytics_stats['total_analyses']],
                ["Score anomalie moyen", f"{analytics_stats['avg_anomaly_score']:.3f}" if analytics_stats['avg_anomaly_score'] else "N/A"],
                ["Confiance moyenne", f"{analytics_stats['avg_confidence']:.3f}" if analytics_stats['avg_confidence'] else "N/A"],
                ["Analyses récentes (1h)", analytics_stats['recent_analyses']],
                ["", ""],
                ["Patterns corrélation", correlation_stats['total_patterns']],
                ["Force moyenne", f"{correlation_stats['avg_strength']:.3f}" if correlation_stats['avg_strength'] else "N/A"],
                ["Patterns actifs (24h)", correlation_stats['active_patterns']]
            ]
            
            print(tabulate.tabulate(data, headers=["Métrique", "Valeur"], tablefmt="grid"))
            
            # Statut Redis
            print(f"\n🔴 Redis:")
            redis_info = await self.redis_client.info()
            print(f"   Connexions: {redis_info.get('connected_clients', 'N/A')}")
            print(f"   Mémoire utilisée: {redis_info.get('used_memory_human', 'N/A')}")
            print(f"   Clés: {await self.redis_client.dbsize()}")
            
        except Exception as e:
            print(f"❌ Erreur récupération statut: {e}")
    
    async def show_alert_trends(self, hours: int = 24):
        """Affichage des tendances d'alertes"""
        print(f"📈 Tendances d'alertes sur {hours}h\n")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Tendances par heure
                trends = await conn.fetch("""
                    SELECT 
                        date_trunc('hour', timestamp) as hour,
                        COUNT(*) as alert_count,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_count,
                        COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_count
                    FROM alerts 
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                    GROUP BY hour 
                    ORDER BY hour
                """, hours)
                
                if trends:
                    data = []
                    for trend in trends:
                        data.append([
                            trend['hour'].strftime('%Y-%m-%d %H:00'),
                            trend['alert_count'],
                            trend['critical_count'],
                            trend['high_count']
                        ])
                    
                    print(tabulate.tabulate(
                        data, 
                        headers=["Heure", "Total", "Critiques", "Élevées"],
                        tablefmt="grid"
                    ))
                else:
                    print("Aucune donnée de tendance disponible")
                
                # Top services
                print(f"\n🏆 Top services (alertes {hours}h):")
                top_services = await conn.fetch("""
                    SELECT service, COUNT(*) as alert_count
                    FROM alerts 
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                    GROUP BY service 
                    ORDER BY alert_count DESC 
                    LIMIT 10
                """, hours)
                
                if top_services:
                    service_data = [[s['service'], s['alert_count']] for s in top_services]
                    print(tabulate.tabulate(
                        service_data,
                        headers=["Service", "Alertes"],
                        tablefmt="simple"
                    ))
                
        except Exception as e:
            print(f"❌ Erreur récupération tendances: {e}")
    
    async def show_ml_performance(self):
        """Affichage des performances ML"""
        print("🤖 Performance des modèles ML\n")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Statistiques de détection d'anomalies
                anomaly_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_detections,
                        AVG(anomaly_score) as avg_score,
                        COUNT(CASE WHEN anomaly_score > 0.8 THEN 1 END) as high_anomalies,
                        MAX(analysis_timestamp) as last_analysis
                    FROM alert_analytics
                    WHERE analysis_timestamp > NOW() - INTERVAL '24 hours'
                """)
                
                # Distribution des scores d'anomalie
                score_distribution = await conn.fetch("""
                    SELECT 
                        CASE 
                            WHEN anomaly_score >= 0.9 THEN '0.9-1.0'
                            WHEN anomaly_score >= 0.8 THEN '0.8-0.9'
                            WHEN anomaly_score >= 0.7 THEN '0.7-0.8'
                            WHEN anomaly_score >= 0.6 THEN '0.6-0.7'
                            WHEN anomaly_score >= 0.5 THEN '0.5-0.6'
                            ELSE '0.0-0.5'
                        END as score_range,
                        COUNT(*) as count
                    FROM alert_analytics
                    WHERE analysis_timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY score_range
                    ORDER BY score_range DESC
                """)
                
                print("📊 Statistiques détection (24h):")
                ml_data = [
                    ["Détections totales", anomaly_stats['total_detections']],
                    ["Score moyen", f"{anomaly_stats['avg_score']:.3f}" if anomaly_stats['avg_score'] else "N/A"],
                    ["Anomalies élevées (>0.8)", anomaly_stats['high_anomalies']],
                    ["Dernière analyse", anomaly_stats['last_analysis'].strftime('%Y-%m-%d %H:%M:%S') if anomaly_stats['last_analysis'] else "N/A"]
                ]
                print(tabulate.tabulate(ml_data, tablefmt="simple"))
                
                if score_distribution:
                    print("\n📈 Distribution des scores:")
                    dist_data = [[s['score_range'], s['count']] for s in score_distribution]
                    print(tabulate.tabulate(
                        dist_data,
                        headers=["Plage Score", "Nombre"],
                        tablefmt="simple"
                    ))
                
                # Modèles actifs
                models = await conn.fetch("""
                    SELECT model_name, model_type, version, created_at, is_active
                    FROM ml_models 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                
                if models:
                    print("\n🔧 Modèles disponibles:")
                    model_data = []
                    for model in models:
                        status = "✅ Actif" if model['is_active'] else "⏸️  Inactif"
                        model_data.append([
                            model['model_name'],
                            model['model_type'],
                            model['version'],
                            status,
                            model['created_at'].strftime('%Y-%m-%d')
                        ])
                    
                    print(tabulate.tabulate(
                        model_data,
                        headers=["Nom", "Type", "Version", "Statut", "Créé"],
                        tablefmt="simple"
                    ))
                
        except Exception as e:
            print(f"❌ Erreur récupération performance ML: {e}")
    
    # === Commandes de maintenance ===
    
    async def cleanup_old_data(self, days: int = 90):
        """Nettoyage des anciennes données"""
        print(f"🧹 Nettoyage des données > {days} jours...")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            async with self.db_pool.acquire() as conn:
                # Nettoyage alertes anciennes
                deleted_alerts = await conn.fetchval("""
                    DELETE FROM alerts 
                    WHERE timestamp < $1 AND status = 'resolved'
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                # Nettoyage analytics
                deleted_analytics = await conn.fetchval("""
                    DELETE FROM alert_analytics 
                    WHERE analysis_timestamp < $1
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                # Nettoyage métriques système
                deleted_metrics = await conn.fetchval("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < $1
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                print(f"✅ Alertes supprimées: {deleted_alerts or 0}")
                print(f"✅ Analytics supprimées: {deleted_analytics or 0}")
                print(f"✅ Métriques supprimées: {deleted_metrics or 0}")
                
                # Nettoyage cache Redis
                pattern = f"{self.config.get_cache_key_prefix()}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    print(f"✅ Clés Redis supprimées: {len(keys)}")
                
                print("🎉 Nettoyage terminé!")
                
        except Exception as e:
            print(f"❌ Erreur nettoyage: {e}")
    
    async def optimize_database(self):
        """Optimisation de la base de données"""
        print("⚡ Optimisation de la base de données...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Analyse des tables
                tables = ['alerts', 'alert_analytics', 'correlation_patterns', 
                         'service_dependencies', 'system_metrics']
                
                for table in tables:
                    await conn.execute(f"ANALYZE {table}")
                    print(f"✅ Table {table} analysée")
                
                # Vacuum pour récupérer l'espace
                for table in tables:
                    await conn.execute(f"VACUUM {table}")
                    print(f"✅ Table {table} nettoyée")
                
                # Statistiques des index
                index_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                    ORDER BY idx_tup_read DESC
                """)
                
                if index_stats:
                    print("\n📊 Utilisation des index:")
                    index_data = []
                    for stat in index_stats[:10]:  # Top 10
                        index_data.append([
                            stat['tablename'],
                            stat['indexname'],
                            stat['idx_tup_read'],
                            stat['idx_tup_fetch']
                        ])
                    
                    print(tabulate.tabulate(
                        index_data,
                        headers=["Table", "Index", "Lectures", "Récupérations"],
                        tablefmt="simple"
                    ))
                
                print("🎉 Optimisation terminée!")
                
        except Exception as e:
            print(f"❌ Erreur optimisation: {e}")
    
    # === Commandes d'import/export ===
    
    async def export_analytics_report(self, output_file: str, hours: int = 24):
        """Export d'un rapport d'analytics"""
        print(f"📄 Export rapport analytics ({hours}h) vers {output_file}...")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Données d'alertes
                alerts_data = await conn.fetch("""
                    SELECT * FROM alerts 
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                """, hours)
                
                # Données d'analytics
                analytics_data = await conn.fetch("""
                    SELECT aa.*, a.service, a.severity, a.title
                    FROM alert_analytics aa
                    JOIN alerts a ON aa.alert_id = a.id
                    WHERE aa.analysis_timestamp > NOW() - INTERVAL '%s hours'
                    ORDER BY aa.analysis_timestamp DESC
                """, hours)
                
                # Patterns de corrélation
                patterns_data = await conn.fetch("""
                    SELECT * FROM correlation_patterns
                    WHERE last_seen > NOW() - INTERVAL '%s hours'
                    ORDER BY strength DESC
                """, hours)
            
            # Conversion en DataFrames pandas
            alerts_df = pd.DataFrame([dict(row) for row in alerts_data])
            analytics_df = pd.DataFrame([dict(row) for row in analytics_data])
            patterns_df = pd.DataFrame([dict(row) for row in patterns_data])
            
            # Export Excel avec multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                if not alerts_df.empty:
                    alerts_df.to_excel(writer, sheet_name='Alerts', index=False)
                if not analytics_df.empty:
                    analytics_df.to_excel(writer, sheet_name='Analytics', index=False)
                if not patterns_df.empty:
                    patterns_df.to_excel(writer, sheet_name='Correlations', index=False)
                
                # Sheet de résumé
                summary_data = {
                    'Métrique': [
                        'Période analysée',
                        'Nombre d\'alertes',
                        'Analyses ML',
                        'Patterns corrélation',
                        'Export généré le'
                    ],
                    'Valeur': [
                        f"{hours} heures",
                        len(alerts_df),
                        len(analytics_df),
                        len(patterns_df),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Résumé', index=False)
            
            print(f"🎉 Rapport exporté: {output_file}")
            
        except Exception as e:
            print(f"❌ Erreur export rapport: {e}")
    
    async def import_test_data(self, count: int = 100):
        """Import de données de test"""
        print(f"🧪 Import de {count} alertes de test...")
        
        try:
            import random
            from uuid import uuid4
            
            test_services = ['user-api', 'payment-service', 'recommendation-engine', 
                           'notification-service', 'analytics-api']
            test_severities = ['critical', 'high', 'medium', 'low', 'info']
            test_messages = [
                'High CPU usage detected',
                'Memory consumption exceeding threshold',
                'Database connection timeout',
                'API response time degraded',
                'Authentication service unavailable'
            ]
            
            alerts = []
            for i in range(count):
                alert = {
                    'id': str(uuid4()),
                    'timestamp': datetime.now() - timedelta(
                        minutes=random.randint(0, 1440)  # 24h
                    ),
                    'severity': random.choice(test_severities),
                    'status': random.choice(['firing', 'resolved']),
                    'source': 'test_import',
                    'service': random.choice(test_services),
                    'component': f'component-{random.randint(1, 5)}',
                    'title': f'Test Alert {i+1}',
                    'message': random.choice(test_messages),
                    'labels': json.dumps({
                        'instance': f'instance-{random.randint(1, 10)}',
                        'environment': 'test'
                    }),
                    'metrics': json.dumps({
                        'cpu_usage': random.uniform(0, 100),
                        'memory_usage': random.uniform(0, 100),
                        'response_time': random.uniform(10, 5000)
                    })
                }
                alerts.append(alert)
            
            # Insertion en base
            async with self.db_pool.acquire() as conn:
                columns = list(alerts[0].keys())
                placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
                query = f"""
                    INSERT INTO alerts ({', '.join(columns)}) 
                    VALUES ({placeholders})
                """
                
                values = [tuple(alert[col] for col in columns) for alert in alerts]
                await conn.executemany(query, values)
            
            print(f"✅ {count} alertes de test importées")
            
        except Exception as e:
            print(f"❌ Erreur import données test: {e}")
    
    # === Commandes de test ===
    
    async def run_health_checks(self):
        """Tests de santé du système"""
        print("🏥 Tests de santé du système\n")
        
        checks = [
            ("Base de données", self._check_database),
            ("Redis", self._check_redis),
            ("Configuration", self._check_config),
            ("Performances", self._check_performance)
        ]
        
        results = []
        for check_name, check_func in checks:
            try:
                result = await check_func()
                status = "✅ OK" if result['healthy'] else "❌ KO"
                results.append([check_name, status, result.get('message', '')])
            except Exception as e:
                results.append([check_name, "❌ ERREUR", str(e)])
        
        print(tabulate.tabulate(
            results,
            headers=["Test", "Statut", "Détails"],
            tablefmt="grid"
        ))
    
    async def _check_database(self) -> Dict[str, Any]:
        """Vérification base de données"""
        async with self.db_pool.acquire() as conn:
            # Test basique
            await conn.fetchval("SELECT 1")
            
            # Vérification tables
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            
            required_tables = {'alerts', 'alert_analytics', 'correlation_patterns'}
            existing_tables = {row['tablename'] for row in tables}
            
            if not required_tables.issubset(existing_tables):
                missing = required_tables - existing_tables
                return {
                    'healthy': False,
                    'message': f'Tables manquantes: {", ".join(missing)}'
                }
            
            return {'healthy': True, 'message': f'{len(tables)} tables trouvées'}
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Vérification Redis"""
        await self.redis_client.ping()
        info = await self.redis_client.info()
        
        # Vérification mémoire
        used_memory = info.get('used_memory', 0)
        max_memory = info.get('maxmemory', 0)
        
        if max_memory > 0 and used_memory / max_memory > 0.9:
            return {
                'healthy': False,
                'message': 'Mémoire Redis critique'
            }
        
        return {
            'healthy': True,
            'message': f'Mémoire: {info.get("used_memory_human", "N/A")}'
        }
    
    async def _check_config(self) -> Dict[str, Any]:
        """Vérification configuration"""
        issues = []
        
        # Vérification URLs
        if not self.config.database_url.startswith('postgresql'):
            issues.append('URL database invalide')
        
        if not self.config.redis_url.startswith('redis'):
            issues.append('URL Redis invalide')
        
        # Vérification seuils
        if self.config.critical_alert_threshold <= self.config.high_alert_threshold:
            issues.append('Seuils d\'alerte incohérents')
        
        return {
            'healthy': len(issues) == 0,
            'message': '; '.join(issues) if issues else 'Configuration valide'
        }
    
    async def _check_performance(self) -> Dict[str, Any]:
        """Vérification performances"""
        import time
        
        # Test performance base de données
        start = time.time()
        async with self.db_pool.acquire() as conn:
            await conn.fetchval("SELECT COUNT(*) FROM alerts LIMIT 1000")
        db_time = time.time() - start
        
        # Test performance Redis
        start = time.time()
        await self.redis_client.set('health_check', 'test')
        await self.redis_client.get('health_check')
        await self.redis_client.delete('health_check')
        redis_time = time.time() - start
        
        issues = []
        if db_time > 1.0:
            issues.append(f'DB lente: {db_time:.2f}s')
        if redis_time > 0.1:
            issues.append(f'Redis lent: {redis_time:.3f}s')
        
        return {
            'healthy': len(issues) == 0,
            'message': '; '.join(issues) if issues else f'DB: {db_time:.3f}s, Redis: {redis_time:.3f}s'
        }

# === CLI Interface ===

async def main():
    """Interface CLI principale"""
    parser = argparse.ArgumentParser(description='Spotify AI Agent - Outil d\'administration Analytics')
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Base de données
    db_parser = subparsers.add_parser('db', help='Gestion base de données')
    db_subparsers = db_parser.add_subparsers(dest='db_command')
    
    db_subparsers.add_parser('init', help='Initialiser la base de données')
    
    backup_parser = db_subparsers.add_parser('backup', help='Sauvegarder la base')
    backup_parser.add_argument('--output', required=True, help='Fichier de sortie')
    
    restore_parser = db_subparsers.add_parser('restore', help='Restaurer la base')
    restore_parser.add_argument('--input', required=True, help='Fichier de sauvegarde')
    
    # Monitoring
    status_parser = subparsers.add_parser('status', help='Statut du système')
    
    trends_parser = subparsers.add_parser('trends', help='Tendances d\'alertes')
    trends_parser.add_argument('--hours', type=int, default=24, help='Période en heures')
    
    subparsers.add_parser('ml-perf', help='Performance ML')
    
    # Maintenance
    maintenance_parser = subparsers.add_parser('maintenance', help='Maintenance')
    maintenance_subparsers = maintenance_parser.add_subparsers(dest='maintenance_command')
    
    cleanup_parser = maintenance_subparsers.add_parser('cleanup', help='Nettoyage')
    cleanup_parser.add_argument('--days', type=int, default=90, help='Âge max des données (jours)')
    
    maintenance_subparsers.add_parser('optimize', help='Optimiser la base')
    
    # Import/Export
    export_parser = subparsers.add_parser('export', help='Export rapport')
    export_parser.add_argument('--output', required=True, help='Fichier de sortie')
    export_parser.add_argument('--hours', type=int, default=24, help='Période en heures')
    
    import_parser = subparsers.add_parser('import-test', help='Import données test')
    import_parser.add_argument('--count', type=int, default=100, help='Nombre d\'alertes')
    
    # Tests
    subparsers.add_parser('health', help='Tests de santé')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialisation
    admin_tool = AnalyticsAdminTool()
    
    try:
        await admin_tool.initialize()
        
        # Exécution des commandes
        if args.command == 'db':
            if args.db_command == 'init':
                await admin_tool.init_database()
            elif args.db_command == 'backup':
                await admin_tool.backup_database(args.output)
            elif args.db_command == 'restore':
                await admin_tool.restore_database(args.input)
        
        elif args.command == 'status':
            await admin_tool.show_system_status()
        
        elif args.command == 'trends':
            await admin_tool.show_alert_trends(args.hours)
        
        elif args.command == 'ml-perf':
            await admin_tool.show_ml_performance()
        
        elif args.command == 'maintenance':
            if args.maintenance_command == 'cleanup':
                await admin_tool.cleanup_old_data(args.days)
            elif args.maintenance_command == 'optimize':
                await admin_tool.optimize_database()
        
        elif args.command == 'export':
            await admin_tool.export_analytics_report(args.output, args.hours)
        
        elif args.command == 'import-test':
            await admin_tool.import_test_data(args.count)
        
        elif args.command == 'health':
            await admin_tool.run_health_checks()
        
        else:
            print(f"Commande inconnue: {args.command}")
            parser.print_help()
    
    finally:
        await admin_tool.close()

if __name__ == '__main__':
    asyncio.run(main())
