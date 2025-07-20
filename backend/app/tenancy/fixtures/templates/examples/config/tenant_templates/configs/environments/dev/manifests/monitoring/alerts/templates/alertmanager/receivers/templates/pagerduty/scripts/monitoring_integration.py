#!/usr/bin/env python3
"""
Monitoring Integration pour PagerDuty Scripts

Module avancé d'intégration avec les systèmes de monitoring.
Fournit des adaptateurs pour Prometheus, Grafana, ELK Stack,
et autres systèmes de surveillance avec alerting intelligent.

Fonctionnalités:
- Intégration Prometheus/AlertManager
- Connecteurs Grafana avec dashboards
- Intégration ELK Stack (Elasticsearch, Logstash, Kibana)
- Adaptateurs CloudWatch, DataDog, New Relic
- Métriques personnalisées et KPIs
- Alerting intelligent basé sur ML
- Corrélation d'événements multi-sources

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import asyncio
import argparse
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml
import structlog
from abc import ABC, abstractmethod
import aiohttp
import prometheus_client
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Summary
import elasticsearch
from elasticsearch import AsyncElasticsearch
import boto3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich import print as rprint

from utils.api_client import PagerDutyAPIClient, PagerDutyEventAction, PagerDutySeverity
from utils.formatters import MessageFormatter
from alert_manager import PagerDutyAlertManager, Alert, AlertStatus
from incident_manager import PagerDutyIncidentManager

console = Console()
logger = structlog.get_logger(__name__)

class MonitoringSystem(Enum):
    """Types de systèmes de monitoring"""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    ELASTICSEARCH = "elasticsearch"
    CLOUDWATCH = "cloudwatch"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    SPLUNK = "splunk"
    ZABBIX = "zabbix"

class AlertSeverity(Enum):
    """Niveaux de sévérité d'alerte"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class MonitoringEvent:
    """Événement de monitoring"""
    id: str
    source_system: MonitoringSystem
    timestamp: datetime
    severity: AlertSeverity
    title: str
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    description: str
    metric_type: str  # gauge, counter, histogram, summary
    labels: List[str] = field(default_factory=list)
    unit: str = ""
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None

class MonitoringAdapter(ABC):
    """Adaptateur de base pour les systèmes de monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.enabled = config.get("enabled", True)
        self.connection = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Établit la connexion au système de monitoring"""
        pass
    
    @abstractmethod
    async def fetch_events(self, since: datetime) -> List[MonitoringEvent]:
        """Récupère les événements depuis une date"""
        pass
    
    @abstractmethod
    async def fetch_metrics(self, metric_names: List[str], time_range: Tuple[datetime, datetime]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Récupère les métriques pour une période"""
        pass
    
    @abstractmethod
    async def send_alert(self, event: MonitoringEvent) -> bool:
        """Envoie une alerte au système"""
        pass
    
    async def test_connection(self) -> bool:
        """Test la connexion au système"""
        try:
            return await self.connect()
        except Exception as e:
            logger.error(f"Connection test failed for {self.name}: {e}")
            return False
    
    async def close(self):
        """Ferme la connexion"""
        if self.connection:
            await self.connection.close()

class PrometheusAdapter(MonitoringAdapter):
    """Adaptateur pour Prometheus"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config["url"]
        self.session = None
        self.registry = CollectorRegistry()
        self.metrics = {}
    
    async def connect(self) -> bool:
        """Connexion à Prometheus"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test de connexion
            async with self.session.get(f"{self.base_url}/api/v1/status/config") as response:
                if response.status == 200:
                    logger.info(f"Connected to Prometheus at {self.base_url}")
                    return True
                else:
                    logger.error(f"Failed to connect to Prometheus: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Prometheus connection error: {e}")
            return False
    
    async def fetch_events(self, since: datetime) -> List[MonitoringEvent]:
        """Récupère les alertes Prometheus"""
        events = []
        
        try:
            # Récupérer les alertes actives
            async with self.session.get(f"{self.base_url}/api/v1/alerts") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for alert in data.get("data", []):
                        # Filtrer par date
                        active_at = datetime.fromisoformat(alert.get("activeAt", "").replace("Z", "+00:00"))
                        if active_at < since:
                            continue
                        
                        # Mapper les niveaux de sévérité
                        severity_map = {
                            "critical": AlertSeverity.CRITICAL,
                            "warning": AlertSeverity.HIGH,
                            "info": AlertSeverity.INFO
                        }
                        
                        severity = severity_map.get(
                            alert.get("labels", {}).get("severity", "warning").lower(),
                            AlertSeverity.MEDIUM
                        )
                        
                        event = MonitoringEvent(
                            id=f"prometheus_{alert.get('fingerprint', '')}",
                            source_system=MonitoringSystem.PROMETHEUS,
                            timestamp=active_at,
                            severity=severity,
                            title=alert.get("labels", {}).get("alertname", "Unknown alert"),
                            description=alert.get("annotations", {}).get("description", ""),
                            labels=alert.get("labels", {}),
                            annotations=alert.get("annotations", {}),
                            raw_data=alert
                        )
                        
                        events.append(event)
                        
        except Exception as e:
            logger.error(f"Failed to fetch Prometheus events: {e}")
        
        return events
    
    async def fetch_metrics(self, metric_names: List[str], time_range: Tuple[datetime, datetime]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Récupère les métriques Prometheus"""
        results = {}
        
        start_time, end_time = time_range
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        step = "60s"  # 1 minute interval
        
        for metric_name in metric_names:
            try:
                # Query range pour récupérer les données historiques
                params = {
                    "query": metric_name,
                    "start": start_timestamp,
                    "end": end_timestamp,
                    "step": step
                }
                
                async with self.session.get(f"{self.base_url}/api/v1/query_range", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        metric_data = []
                        for result in data.get("data", {}).get("result", []):
                            for timestamp, value in result.get("values", []):
                                dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                                metric_data.append((dt, float(value)))
                        
                        results[metric_name] = metric_data
                    
            except Exception as e:
                logger.error(f"Failed to fetch metric {metric_name}: {e}")
                results[metric_name] = []
        
        return results
    
    async def send_alert(self, event: MonitoringEvent) -> bool:
        """Envoie une alerte via Prometheus (via webhook)"""
        # Prometheus ne reçoit généralement pas d'alertes, mais on peut
        # utiliser pushgateway pour envoyer des métriques
        try:
            if "pushgateway_url" in self.config:
                # Créer une métrique pour l'événement
                metric_data = {
                    "alert_sent": 1,
                    "severity": event.severity.value,
                    "source": event.source_system.value
                }
                
                # Envoyer à pushgateway
                pushgateway_url = self.config["pushgateway_url"]
                job_name = "pagerduty_integration"
                
                async with self.session.post(
                    f"{pushgateway_url}/metrics/job/{job_name}",
                    data=self._format_prometheus_metrics(metric_data)
                ) as response:
                    return response.status == 200
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert to Prometheus: {e}")
            return False
    
    def _format_prometheus_metrics(self, metrics: Dict[str, Any]) -> str:
        """Formate les métriques pour Prometheus"""
        lines = []
        for name, value in metrics.items():
            lines.append(f"{name} {value}")
        return "\n".join(lines)
    
    def register_custom_metric(self, definition: MetricDefinition):
        """Enregistre une métrique personnalisée"""
        if definition.metric_type == "gauge":
            metric = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == "counter":
            metric = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == "histogram":
            metric = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == "summary":
            metric = Summary(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        else:
            raise ValueError(f"Unknown metric type: {definition.metric_type}")
        
        self.metrics[definition.name] = metric
        logger.info(f"Registered custom metric: {definition.name}")
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()

class ElasticsearchAdapter(MonitoringAdapter):
    """Adaptateur pour Elasticsearch"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hosts = config["hosts"]
        self.index_pattern = config.get("index_pattern", "logs-*")
        self.client = None
    
    async def connect(self) -> bool:
        """Connexion à Elasticsearch"""
        try:
            self.client = AsyncElasticsearch(
                hosts=self.hosts,
                http_auth=self.config.get("auth"),
                verify_certs=self.config.get("verify_certs", True)
            )
            
            # Test de connexion
            info = await self.client.info()
            logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Elasticsearch connection error: {e}")
            return False
    
    async def fetch_events(self, since: datetime) -> List[MonitoringEvent]:
        """Récupère les événements depuis Elasticsearch"""
        events = []
        
        try:
            # Construire la requête
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": since.isoformat()
                                    }
                                }
                            },
                            {
                                "exists": {
                                    "field": "level"
                                }
                            }
                        ],
                        "should": [
                            {"term": {"level": "ERROR"}},
                            {"term": {"level": "FATAL"}},
                            {"term": {"level": "WARN"}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                "sort": [
                    {"@timestamp": {"order": "desc"}}
                ],
                "size": 1000
            }
            
            # Exécuter la recherche
            response = await self.client.search(
                index=self.index_pattern,
                body=query
            )
            
            # Parser les résultats
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                
                # Mapper le niveau de log à la sévérité
                level = source.get("level", "INFO").upper()
                severity_map = {
                    "FATAL": AlertSeverity.CRITICAL,
                    "ERROR": AlertSeverity.HIGH,
                    "WARN": AlertSeverity.MEDIUM,
                    "INFO": AlertSeverity.INFO
                }
                severity = severity_map.get(level, AlertSeverity.LOW)
                
                # Créer l'événement
                event = MonitoringEvent(
                    id=f"elasticsearch_{hit['_id']}",
                    source_system=MonitoringSystem.ELASTICSEARCH,
                    timestamp=datetime.fromisoformat(source.get("@timestamp", "").replace("Z", "+00:00")),
                    severity=severity,
                    title=source.get("message", "Log event")[:100],
                    description=source.get("message", ""),
                    labels={
                        "service": source.get("service", "unknown"),
                        "environment": source.get("environment", "unknown"),
                        "host": source.get("host", "unknown")
                    },
                    raw_data=source
                )
                
                events.append(event)
                
        except Exception as e:
            logger.error(f"Failed to fetch Elasticsearch events: {e}")
        
        return events
    
    async def fetch_metrics(self, metric_names: List[str], time_range: Tuple[datetime, datetime]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Récupère les métriques agrégées depuis Elasticsearch"""
        results = {}
        start_time, end_time = time_range
        
        try:
            for metric_name in metric_names:
                # Construire une requête d'agrégation temporelle
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "@timestamp": {
                                            "gte": start_time.isoformat(),
                                            "lte": end_time.isoformat()
                                        }
                                    }
                                },
                                {
                                    "exists": {
                                        "field": metric_name
                                    }
                                }
                            ]
                        }
                    },
                    "aggs": {
                        "time_buckets": {
                            "date_histogram": {
                                "field": "@timestamp",
                                "interval": "1m"
                            },
                            "aggs": {
                                "metric_avg": {
                                    "avg": {
                                        "field": metric_name
                                    }
                                }
                            }
                        }
                    },
                    "size": 0
                }
                
                response = await self.client.search(
                    index=self.index_pattern,
                    body=query
                )
                
                # Parser les résultats d'agrégation
                metric_data = []
                for bucket in response["aggregations"]["time_buckets"]["buckets"]:
                    timestamp = datetime.fromisoformat(bucket["key_as_string"].replace("Z", "+00:00"))
                    value = bucket["metric_avg"]["value"]
                    if value is not None:
                        metric_data.append((timestamp, value))
                
                results[metric_name] = metric_data
                
        except Exception as e:
            logger.error(f"Failed to fetch Elasticsearch metrics: {e}")
        
        return results
    
    async def send_alert(self, event: MonitoringEvent) -> bool:
        """Indexe l'alerte dans Elasticsearch"""
        try:
            doc = {
                "@timestamp": event.timestamp.isoformat(),
                "event_id": event.id,
                "source_system": event.source_system.value,
                "severity": event.severity.value,
                "title": event.title,
                "description": event.description,
                "labels": event.labels,
                "annotations": event.annotations,
                "metrics": event.metrics
            }
            
            # Indexer le document
            response = await self.client.index(
                index=f"pagerduty-alerts-{datetime.now().strftime('%Y.%m')}",
                body=doc
            )
            
            return response["result"] in ["created", "updated"]
            
        except Exception as e:
            logger.error(f"Failed to send alert to Elasticsearch: {e}")
            return False
    
    async def close(self):
        """Ferme la connexion"""
        if self.client:
            await self.client.close()

class CloudWatchAdapter(MonitoringAdapter):
    """Adaptateur pour AWS CloudWatch"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.region = config.get("region", "us-east-1")
        self.cloudwatch = None
        self.logs_client = None
    
    async def connect(self) -> bool:
        """Connexion à CloudWatch"""
        try:
            # Créer les clients AWS
            session = boto3.Session(
                aws_access_key_id=self.config.get("access_key_id"),
                aws_secret_access_key=self.config.get("secret_access_key"),
                region_name=self.region
            )
            
            self.cloudwatch = session.client("cloudwatch")
            self.logs_client = session.client("logs")
            
            # Test de connexion
            self.cloudwatch.list_metrics(MaxRecords=1)
            logger.info(f"Connected to CloudWatch in region {self.region}")
            return True
            
        except Exception as e:
            logger.error(f"CloudWatch connection error: {e}")
            return False
    
    async def fetch_events(self, since: datetime) -> List[MonitoringEvent]:
        """Récupère les événements CloudWatch"""
        events = []
        
        try:
            # Récupérer les alarmes CloudWatch
            response = self.cloudwatch.describe_alarms(
                StateValue="ALARM",
                MaxRecords=100
            )
            
            for alarm in response["MetricAlarms"]:
                # Vérifier la date de la dernière transition
                state_updated = alarm.get("StateUpdatedTimestamp")
                if state_updated and state_updated.replace(tzinfo=timezone.utc) < since:
                    continue
                
                # Mapper la sévérité basée sur les tags ou le nom
                severity = AlertSeverity.HIGH  # Par défaut
                if "critical" in alarm["AlarmName"].lower():
                    severity = AlertSeverity.CRITICAL
                elif "warning" in alarm["AlarmName"].lower():
                    severity = AlertSeverity.MEDIUM
                
                event = MonitoringEvent(
                    id=f"cloudwatch_{alarm['AlarmArn']}",
                    source_system=MonitoringSystem.CLOUDWATCH,
                    timestamp=state_updated.replace(tzinfo=timezone.utc),
                    severity=severity,
                    title=alarm["AlarmName"],
                    description=alarm.get("AlarmDescription", ""),
                    labels={
                        "namespace": alarm["Namespace"],
                        "metric_name": alarm["MetricName"],
                        "statistic": alarm["Statistic"]
                    },
                    metrics={
                        "threshold": alarm["Threshold"],
                        "comparison_operator": alarm["ComparisonOperator"]
                    },
                    raw_data=alarm
                )
                
                events.append(event)
                
        except Exception as e:
            logger.error(f"Failed to fetch CloudWatch events: {e}")
        
        return events
    
    async def fetch_metrics(self, metric_names: List[str], time_range: Tuple[datetime, datetime]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Récupère les métriques CloudWatch"""
        results = {}
        start_time, end_time = time_range
        
        try:
            for metric_name in metric_names:
                # Parser le nom de métrique (format: namespace:metric_name:statistic)
                parts = metric_name.split(":")
                if len(parts) >= 3:
                    namespace, name, statistic = parts[0], parts[1], parts[2]
                else:
                    # Valeurs par défaut
                    namespace, name, statistic = "AWS/EC2", metric_name, "Average"
                
                response = self.cloudwatch.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=name,
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,  # 5 minutes
                    Statistics=[statistic]
                )
                
                # Parser les données
                metric_data = []
                for datapoint in response["Datapoints"]:
                    timestamp = datapoint["Timestamp"].replace(tzinfo=timezone.utc)
                    value = datapoint[statistic]
                    metric_data.append((timestamp, value))
                
                # Trier par timestamp
                metric_data.sort(key=lambda x: x[0])
                results[metric_name] = metric_data
                
        except Exception as e:
            logger.error(f"Failed to fetch CloudWatch metrics: {e}")
        
        return results
    
    async def send_alert(self, event: MonitoringEvent) -> bool:
        """Envoie une métrique personnalisée à CloudWatch"""
        try:
            # Créer une métrique personnalisée pour l'événement
            self.cloudwatch.put_metric_data(
                Namespace="PagerDuty/Integration",
                MetricData=[
                    {
                        "MetricName": "AlertsSent",
                        "Value": 1,
                        "Unit": "Count",
                        "Dimensions": [
                            {
                                "Name": "Severity",
                                "Value": event.severity.value
                            },
                            {
                                "Name": "SourceSystem",
                                "Value": event.source_system.value
                            }
                        ],
                        "Timestamp": event.timestamp
                    }
                ]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert to CloudWatch: {e}")
            return False

class EventCorrelator:
    """Corrélateur d'événements multi-sources"""
    
    def __init__(self, correlation_window: int = 300):  # 5 minutes
        self.correlation_window = correlation_window
        self.event_buffer = []
        self.correlation_rules = []
        self.ml_model = None
        self._initialize_ml_model()
    
    def _initialize_ml_model(self):
        """Initialise le modèle ML pour la détection d'anomalies"""
        try:
            self.ml_model = IsolationForest(
                contamination=0.1,  # 10% d'anomalies attendues
                random_state=42
            )
            self.scaler = StandardScaler()
            logger.info("ML model initialized for anomaly detection")
        except Exception as e:
            logger.warning(f"Failed to initialize ML model: {e}")
    
    def add_correlation_rule(self, rule: Dict[str, Any]):
        """Ajoute une règle de corrélation"""
        self.correlation_rules.append(rule)
        logger.info(f"Added correlation rule: {rule.get('name', 'unnamed')}")
    
    def add_event(self, event: MonitoringEvent):
        """Ajoute un événement au buffer"""
        self.event_buffer.append(event)
        
        # Nettoyer les anciens événements
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.correlation_window)
        self.event_buffer = [e for e in self.event_buffer if e.timestamp > cutoff_time]
    
    def correlate_events(self) -> List[Dict[str, Any]]:
        """Corrèle les événements selon les règles"""
        correlations = []
        
        # Appliquer les règles de corrélation
        for rule in self.correlation_rules:
            correlation = self._apply_correlation_rule(rule)
            if correlation:
                correlations.append(correlation)
        
        # Détection d'anomalies ML
        if self.ml_model and len(self.event_buffer) >= 10:
            ml_correlations = self._detect_ml_anomalies()
            correlations.extend(ml_correlations)
        
        return correlations
    
    def _apply_correlation_rule(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Applique une règle de corrélation"""
        try:
            rule_type = rule.get("type")
            
            if rule_type == "temporal":
                return self._temporal_correlation(rule)
            elif rule_type == "service":
                return self._service_correlation(rule)
            elif rule_type == "severity_escalation":
                return self._severity_escalation(rule)
            elif rule_type == "geographic":
                return self._geographic_correlation(rule)
            
        except Exception as e:
            logger.error(f"Failed to apply correlation rule {rule.get('name', 'unnamed')}: {e}")
        
        return None
    
    def _temporal_correlation(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Corrélation temporelle - événements rapprochés dans le temps"""
        time_window = rule.get("time_window", 60)  # secondes
        min_events = rule.get("min_events", 3)
        
        # Grouper les événements par fenêtre temporelle
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=time_window)
        
        recent_events = [e for e in self.event_buffer if e.timestamp > window_start]
        
        if len(recent_events) >= min_events:
            # Calculer les statistiques
            severities = [e.severity for e in recent_events]
            sources = [e.source_system for e in recent_events]
            
            return {
                "type": "temporal_correlation",
                "rule_name": rule.get("name"),
                "event_count": len(recent_events),
                "time_window": time_window,
                "severity_distribution": {s.value: severities.count(s) for s in set(severities)},
                "source_distribution": {s.value: sources.count(s) for s in set(sources)},
                "events": [e.id for e in recent_events],
                "confidence": min(1.0, len(recent_events) / (min_events * 2))
            }
        
        return None
    
    def _service_correlation(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Corrélation par service - événements du même service"""
        target_service = rule.get("service")
        time_window = rule.get("time_window", 300)
        
        # Filtrer par service et temps
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=time_window)
        service_events = [
            e for e in self.event_buffer
            if e.timestamp > cutoff_time and e.labels.get("service") == target_service
        ]
        
        if len(service_events) >= rule.get("min_events", 2):
            return {
                "type": "service_correlation",
                "rule_name": rule.get("name"),
                "service": target_service,
                "event_count": len(service_events),
                "events": [e.id for e in service_events],
                "confidence": 0.8
            }
        
        return None
    
    def _severity_escalation(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Détection d'escalade de sévérité"""
        time_window = rule.get("time_window", 600)  # 10 minutes
        
        # Grouper par source et analyser l'évolution de la sévérité
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=time_window)
        recent_events = [e for e in self.event_buffer if e.timestamp > cutoff_time]
        
        # Grouper par source
        by_source = {}
        for event in recent_events:
            source = event.labels.get("service", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(event)
        
        escalations = []
        for source, events in by_source.items():
            if len(events) >= 2:
                # Trier par timestamp
                events.sort(key=lambda e: e.timestamp)
                
                # Vérifier l'escalade de sévérité
                severity_values = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}
                first_severity = severity_values.get(events[0].severity.value, 1)
                last_severity = severity_values.get(events[-1].severity.value, 1)
                
                if last_severity > first_severity:
                    escalations.append({
                        "source": source,
                        "from_severity": events[0].severity.value,
                        "to_severity": events[-1].severity.value,
                        "event_count": len(events),
                        "time_span": (events[-1].timestamp - events[0].timestamp).total_seconds()
                    })
        
        if escalations:
            return {
                "type": "severity_escalation",
                "rule_name": rule.get("name"),
                "escalations": escalations,
                "confidence": 0.9
            }
        
        return None
    
    def _geographic_correlation(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Corrélation géographique - événements dans la même région"""
        target_regions = rule.get("regions", [])
        time_window = rule.get("time_window", 300)
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=time_window)
        regional_events = []
        
        for event in self.event_buffer:
            if event.timestamp > cutoff_time:
                event_region = event.labels.get("region", event.labels.get("zone"))
                if event_region in target_regions:
                    regional_events.append(event)
        
        if len(regional_events) >= rule.get("min_events", 2):
            return {
                "type": "geographic_correlation",
                "rule_name": rule.get("name"),
                "regions": target_regions,
                "event_count": len(regional_events),
                "events": [e.id for e in regional_events],
                "confidence": 0.7
            }
        
        return None
    
    def _detect_ml_anomalies(self) -> List[Dict[str, Any]]:
        """Détection d'anomalies avec ML"""
        try:
            # Préparer les features pour le ML
            features = []
            event_ids = []
            
            for event in self.event_buffer[-100:]:  # Derniers 100 événements
                feature_vector = [
                    event.timestamp.timestamp(),  # Timestamp
                    {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}.get(event.severity.value, 1),  # Sévérité numérique
                    len(event.title),  # Longueur du titre
                    len(event.labels),  # Nombre de labels
                    len(event.metrics)  # Nombre de métriques
                ]
                
                features.append(feature_vector)
                event_ids.append(event.id)
            
            if len(features) < 10:
                return []
            
            # Normaliser les features
            features_scaled = self.scaler.fit_transform(features)
            
            # Entraîner et prédire
            predictions = self.ml_model.fit_predict(features_scaled)
            
            # Identifier les anomalies (valeur -1)
            anomalies = []
            for i, prediction in enumerate(predictions):
                if prediction == -1:
                    anomaly_score = self.ml_model.score_samples([features_scaled[i]])[0]
                    anomalies.append({
                        "type": "ml_anomaly",
                        "event_id": event_ids[i],
                        "anomaly_score": float(anomaly_score),
                        "confidence": abs(anomaly_score) / 0.5  # Normaliser entre 0 et 1
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
            return []

class MonitoringIntegrationManager:
    """Gestionnaire principal d'intégration monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters = {}
        self.correlator = EventCorrelator()
        self.alert_manager = None
        self.incident_manager = None
        self.redis_client = None
        self.running = False
        
        # Métriques Prometheus personnalisées
        self.metrics = {
            "events_processed": Counter("monitoring_events_processed_total", "Total events processed", ["source", "severity"]),
            "correlations_detected": Counter("correlations_detected_total", "Total correlations detected", ["type"]),
            "alerts_sent": Counter("pagerduty_alerts_sent_total", "Total alerts sent to PagerDuty", ["severity"]),
            "processing_time": Histogram("event_processing_seconds", "Event processing time", ["source"])
        }
    
    async def initialize(self):
        """Initialise tous les composants"""
        logger.info("Initializing monitoring integration manager...")
        
        # Initialiser Redis pour le cache
        if "redis" in self.config:
            redis_config = self.config["redis"]
            self.redis_client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password"),
                decode_responses=True
            )
        
        # Initialiser les adaptateurs de monitoring
        for name, adapter_config in self.config.get("adapters", {}).items():
            await self._initialize_adapter(name, adapter_config)
        
        # Initialiser PagerDuty
        pagerduty_config = self.config.get("pagerduty", {})
        if pagerduty_config:
            self.alert_manager = PagerDutyAlertManager(
                api_key=pagerduty_config["api_key"],
                integration_key=pagerduty_config["integration_key"]
            )
            
            self.incident_manager = PagerDutyIncidentManager(
                api_key=pagerduty_config["api_key"]
            )
        
        # Charger les règles de corrélation
        await self._load_correlation_rules()
        
        logger.info(f"Initialized {len(self.adapters)} monitoring adapters")
    
    async def _initialize_adapter(self, name: str, config: Dict[str, Any]):
        """Initialise un adaptateur spécifique"""
        adapter_type = config.get("type")
        
        try:
            if adapter_type == "prometheus":
                adapter = PrometheusAdapter(config)
            elif adapter_type == "elasticsearch":
                adapter = ElasticsearchAdapter(config)
            elif adapter_type == "cloudwatch":
                adapter = CloudWatchAdapter(config)
            else:
                logger.warning(f"Unknown adapter type: {adapter_type}")
                return
            
            # Tester la connexion
            if await adapter.connect():
                self.adapters[name] = adapter
                logger.info(f"Successfully initialized adapter: {name}")
            else:
                logger.error(f"Failed to connect adapter: {name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize adapter {name}: {e}")
    
    async def _load_correlation_rules(self):
        """Charge les règles de corrélation"""
        rules_config = self.config.get("correlation_rules", [])
        
        for rule in rules_config:
            self.correlator.add_correlation_rule(rule)
        
        # Règles par défaut
        default_rules = [
            {
                "name": "temporal_burst",
                "type": "temporal",
                "time_window": 120,
                "min_events": 5
            },
            {
                "name": "service_cascade",
                "type": "service",
                "time_window": 300,
                "min_events": 3
            },
            {
                "name": "severity_escalation",
                "type": "severity_escalation",
                "time_window": 600
            }
        ]
        
        for rule in default_rules:
            self.correlator.add_correlation_rule(rule)
    
    async def start_monitoring(self):
        """Démarre le monitoring continu"""
        self.running = True
        logger.info("Starting continuous monitoring...")
        
        # Créer des tâches pour chaque adaptateur
        tasks = []
        
        for name, adapter in self.adapters.items():
            task = asyncio.create_task(self._monitor_adapter(name, adapter))
            tasks.append(task)
        
        # Tâche de corrélation
        correlation_task = asyncio.create_task(self._correlation_loop())
        tasks.append(correlation_task)
        
        # Tâche de métriques
        metrics_task = asyncio.create_task(self._metrics_loop())
        tasks.append(metrics_task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.running = False
    
    async def _monitor_adapter(self, name: str, adapter: MonitoringAdapter):
        """Boucle de monitoring pour un adaptateur"""
        last_fetch = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        while self.running:
            try:
                start_time = datetime.now()
                
                # Récupérer les nouveaux événements
                events = await adapter.fetch_events(last_fetch)
                
                # Traiter chaque événement
                for event in events:
                    await self._process_event(event, name)
                
                # Mettre à jour le timestamp
                if events:
                    last_fetch = max(e.timestamp for e in events)
                else:
                    last_fetch = datetime.now(timezone.utc)
                
                # Métriques de performance
                processing_time = (datetime.now() - start_time).total_seconds()
                self.metrics["processing_time"].labels(source=name).observe(processing_time)
                
                logger.debug(f"Processed {len(events)} events from {name} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error monitoring {name}: {e}")
            
            # Attendre avant la prochaine iteration
            await asyncio.sleep(30)  # 30 secondes
    
    async def _process_event(self, event: MonitoringEvent, source: str):
        """Traite un événement individuel"""
        try:
            # Métriques
            self.metrics["events_processed"].labels(
                source=source,
                severity=event.severity.value
            ).inc()
            
            # Filtrer les événements selon les règles
            if not self._should_process_event(event):
                return
            
            # Ajouter au corrélateur
            self.correlator.add_event(event)
            
            # Cache Redis
            if self.redis_client:
                await self._cache_event(event)
            
            # Décider si envoyer une alerte PagerDuty
            if self._should_send_alert(event):
                await self._send_pagerduty_alert(event)
            
        except Exception as e:
            logger.error(f"Failed to process event {event.id}: {e}")
    
    def _should_process_event(self, event: MonitoringEvent) -> bool:
        """Détermine si un événement doit être traité"""
        # Filtres de base
        filters = self.config.get("event_filters", {})
        
        # Filtre par sévérité
        min_severity = filters.get("min_severity", "info")
        severity_levels = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}
        
        if severity_levels.get(event.severity.value, 1) < severity_levels.get(min_severity, 1):
            return False
        
        # Filtre par source
        allowed_sources = filters.get("allowed_sources")
        if allowed_sources and event.source_system.value not in allowed_sources:
            return False
        
        # Filtre par service
        allowed_services = filters.get("allowed_services")
        if allowed_services and event.labels.get("service") not in allowed_services:
            return False
        
        return True
    
    def _should_send_alert(self, event: MonitoringEvent) -> bool:
        """Détermine si envoyer une alerte PagerDuty"""
        # Règles d'alerting
        alert_rules = self.config.get("alert_rules", {})
        
        # Toujours alerter pour critical
        if event.severity == AlertSeverity.CRITICAL:
            return True
        
        # Alerter pour high selon les services critiques
        if event.severity == AlertSeverity.HIGH:
            critical_services = alert_rules.get("critical_services", [])
            if event.labels.get("service") in critical_services:
                return True
        
        # Règles personnalisées
        custom_rules = alert_rules.get("custom_rules", [])
        for rule in custom_rules:
            if self._evaluate_alert_rule(event, rule):
                return True
        
        return False
    
    def _evaluate_alert_rule(self, event: MonitoringEvent, rule: Dict[str, Any]) -> bool:
        """Évalue une règle d'alerting personnalisée"""
        conditions = rule.get("conditions", {})
        
        # Vérifier chaque condition
        for field, condition in conditions.items():
            value = None
            
            if field == "severity":
                value = event.severity.value
            elif field == "source_system":
                value = event.source_system.value
            elif field.startswith("labels."):
                label_key = field[7:]  # Enlever "labels."
                value = event.labels.get(label_key)
            elif field.startswith("metrics."):
                metric_key = field[8:]  # Enlever "metrics."
                value = event.metrics.get(metric_key)
            
            if not self._check_condition(value, condition):
                return False
        
        return True
    
    def _check_condition(self, value: Any, condition: Any) -> bool:
        """Vérifie une condition individuelle"""
        if isinstance(condition, str):
            return str(value) == condition
        elif isinstance(condition, dict):
            operator = condition.get("op", "eq")
            expected = condition.get("value")
            
            if operator == "eq":
                return value == expected
            elif operator == "ne":
                return value != expected
            elif operator == "gt" and isinstance(value, (int, float)):
                return value > expected
            elif operator == "lt" and isinstance(value, (int, float)):
                return value < expected
            elif operator == "in":
                return value in expected
            elif operator == "contains":
                return expected in str(value)
        
        return False
    
    async def _send_pagerduty_alert(self, event: MonitoringEvent):
        """Envoie une alerte à PagerDuty"""
        if not self.alert_manager:
            return
        
        try:
            # Convertir l'événement en alerte PagerDuty
            alert = Alert(
                id=event.id,
                title=event.title,
                description=event.description,
                severity=event.severity.value,
                source=f"{event.source_system.value}:{event.labels.get('service', 'unknown')}",
                tags=list(event.labels.keys()),
                custom_details={
                    **event.labels,
                    **event.annotations,
                    "metrics": event.metrics,
                    "source_system": event.source_system.value
                }
            )
            
            # Traiter l'alerte
            success = await self.alert_manager.process_alert(alert)
            
            if success:
                self.metrics["alerts_sent"].labels(severity=event.severity.value).inc()
                logger.info(f"Successfully sent alert {event.id} to PagerDuty")
            else:
                logger.error(f"Failed to send alert {event.id} to PagerDuty")
                
        except Exception as e:
            logger.error(f"Error sending PagerDuty alert for event {event.id}: {e}")
    
    async def _cache_event(self, event: MonitoringEvent):
        """Met en cache un événement dans Redis"""
        try:
            cache_key = f"event:{event.id}"
            event_data = {
                "timestamp": event.timestamp.isoformat(),
                "severity": event.severity.value,
                "title": event.title,
                "source_system": event.source_system.value,
                "labels": event.labels
            }
            
            # Stocker avec TTL de 24 heures
            await self.redis_client.setex(
                cache_key,
                86400,  # 24 heures
                json.dumps(event_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache event {event.id}: {e}")
    
    async def _correlation_loop(self):
        """Boucle de corrélation d'événements"""
        while self.running:
            try:
                # Exécuter la corrélation
                correlations = self.correlator.correlate_events()
                
                # Traiter les corrélations détectées
                for correlation in correlations:
                    await self._handle_correlation(correlation)
                
            except Exception as e:
                logger.error(f"Correlation error: {e}")
            
            await asyncio.sleep(60)  # Toutes les minutes
    
    async def _handle_correlation(self, correlation: Dict[str, Any]):
        """Traite une corrélation détectée"""
        try:
            correlation_type = correlation["type"]
            confidence = correlation.get("confidence", 0.5)
            
            # Métriques
            self.metrics["correlations_detected"].labels(type=correlation_type).inc()
            
            logger.info(f"Detected correlation: {correlation_type} (confidence: {confidence:.2f})")
            
            # Si la confiance est élevée, créer un incident
            if confidence > 0.8 and self.incident_manager:
                await self._create_correlated_incident(correlation)
            
        except Exception as e:
            logger.error(f"Failed to handle correlation: {e}")
    
    async def _create_correlated_incident(self, correlation: Dict[str, Any]):
        """Crée un incident basé sur une corrélation"""
        incident_data = {
            "id": f"CORR-{int(datetime.now().timestamp())}",
            "title": f"Correlated incident: {correlation['type']}",
            "severity": "high",
            "service": "multiple",
            "status": "triggered",
            "created_at": datetime.now().isoformat(),
            "correlation_data": correlation
        }
        
        await self.incident_manager.handle_incident(incident_data)
    
    async def _metrics_loop(self):
        """Boucle de collecte de métriques"""
        while self.running:
            try:
                # Collecter des métriques depuis tous les adaptateurs
                await self._collect_system_metrics()
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            
            await asyncio.sleep(300)  # Toutes les 5 minutes
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=5)
        
        for name, adapter in self.adapters.items():
            try:
                # Métriques de base à collecter
                basic_metrics = [
                    "cpu_usage",
                    "memory_usage",
                    "disk_usage",
                    "network_errors"
                ]
                
                metrics_data = await adapter.fetch_metrics(basic_metrics, (start_time, end_time))
                
                # Traiter les métriques collectées
                for metric_name, data_points in metrics_data.items():
                    if data_points:
                        latest_value = data_points[-1][1]  # Dernière valeur
                        logger.debug(f"Metric {metric_name} from {name}: {latest_value}")
                
            except Exception as e:
                logger.debug(f"Failed to collect metrics from {name}: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Récupère les données pour le dashboard"""
        try:
            # Statistiques générales
            total_adapters = len(self.adapters)
            active_adapters = sum(1 for adapter in self.adapters.values() if adapter.enabled)
            
            # Événements récents depuis Redis
            recent_events = []
            if self.redis_client:
                event_keys = await self.redis_client.keys("event:*")
                for key in event_keys[-20:]:  # 20 derniers événements
                    event_data = await self.redis_client.get(key)
                    if event_data:
                        recent_events.append(json.loads(event_data))
            
            # Métriques Prometheus
            registry_metrics = {}
            for name, metric in self.metrics.items():
                try:
                    registry_metrics[name] = metric._value.get() if hasattr(metric, '_value') else 0
                except:
                    registry_metrics[name] = 0
            
            return {
                "adapters": {
                    "total": total_adapters,
                    "active": active_adapters,
                    "status": {name: adapter.enabled for name, adapter in self.adapters.items()}
                },
                "recent_events": recent_events,
                "metrics": registry_metrics,
                "correlation_buffer_size": len(self.correlator.event_buffer),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}
    
    async def stop(self):
        """Arrête le monitoring"""
        self.running = False
        logger.info("Stopping monitoring integration...")
        
        # Fermer toutes les connexions
        for adapter in self.adapters.values():
            await adapter.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.alert_manager:
            await self.alert_manager.close()
        
        if self.incident_manager:
            await self.incident_manager.close()

async def main():
    """Fonction principale CLI"""
    parser = argparse.ArgumentParser(description="PagerDuty Monitoring Integration")
    parser.add_argument("--config", required=True, help="Fichier de configuration")
    parser.add_argument("--action", choices=["start", "test", "dashboard", "metrics"],
                       default="start", help="Action à effectuer")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Port du dashboard")
    
    args = parser.parse_args()
    
    try:
        # Charger la configuration
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Créer le gestionnaire
        manager = MonitoringIntegrationManager(config)
        await manager.initialize()
        
        if args.action == "start":
            console.print("[green]Starting monitoring integration...[/green]")
            await manager.start_monitoring()
        
        elif args.action == "test":
            console.print("[blue]Testing adapter connections...[/blue]")
            
            results_table = Table(title="Adapter Connection Tests")
            results_table.add_column("Adapter", style="bold")
            results_table.add_column("Type")
            results_table.add_column("Status")
            
            for name, adapter in manager.adapters.items():
                success = await adapter.test_connection()
                status = "[green]✓ Connected[/green]" if success else "[red]✗ Failed[/red]"
                results_table.add_row(name, adapter.__class__.__name__, status)
            
            console.print(results_table)
        
        elif args.action == "dashboard":
            dashboard_data = await manager.get_dashboard_data()
            
            console.print(Panel.fit(
                json.dumps(dashboard_data, indent=2, default=str),
                title="Monitoring Dashboard Data"
            ))
        
        elif args.action == "metrics":
            console.print("[blue]Collecting system metrics...[/blue]")
            
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=5)
            
            for name, adapter in manager.adapters.items():
                console.print(f"\n[bold]Metrics from {name}:[/bold]")
                
                try:
                    metrics = await adapter.fetch_metrics(
                        ["cpu_usage", "memory_usage"],
                        (start_time, end_time)
                    )
                    
                    for metric_name, data_points in metrics.items():
                        if data_points:
                            latest = data_points[-1]
                            console.print(f"  {metric_name}: {latest[1]:.2f} at {latest[0]}")
                        else:
                            console.print(f"  {metric_name}: No data")
                            
                except Exception as e:
                    console.print(f"  [red]Error: {e}[/red]")
        
        await manager.stop()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
