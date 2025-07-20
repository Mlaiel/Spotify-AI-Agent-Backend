"""
Advanced Elasticsearch Multi-Tenant Metrics Exporter
===================================================

Exportateur haute performance pour Elasticsearch avec support multi-tenant,
indexation intelligente et recherche optimisée pour les métriques Spotify AI.

Fonctionnalités:
- Indexation multi-tenant sécurisée
- Templates d'index dynamiques
- Aggregations complexes
- Recherche en temps réel
- Archivage automatique
- Monitoring des performances
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, async_streaming_bulk
import structlog
from jinja2 import Template

logger = structlog.get_logger(__name__)


@dataclass
class ElasticsearchConfig:
    """Configuration pour Elasticsearch multi-tenant."""
    hosts: List[str] = field(default_factory=lambda: ['http://localhost:9200'])
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    ca_certs: Optional[str] = None
    verify_certs: bool = True
    timeout: int = 30
    max_retries: int = 3
    retry_on_timeout: bool = True


@dataclass
class TenantIndexConfig:
    """Configuration d'index pour un tenant."""
    tenant_id: str
    index_prefix: str = "spotify-ai-metrics"
    date_pattern: str = "%Y.%m.%d"
    number_of_shards: int = 3
    number_of_replicas: int = 1
    refresh_interval: str = "5s"
    max_age: str = "30d"
    enable_rollover: bool = True
    rollover_max_size: str = "50gb"
    rollover_max_docs: int = 1000000


@dataclass
class MetricDocument:
    """Document de métrique pour Elasticsearch."""
    tenant_id: str
    metric_name: str
    metric_value: float
    metric_type: str
    labels: Dict[str, str]
    timestamp: datetime
    source: str = "spotify-ai-agent"
    environment: str = "production"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ElasticsearchMetricsExporter:
    """
    Exportateur Elasticsearch avancé avec support multi-tenant.
    
    Fonctionnalités:
    - Isolation complète des tenants
    - Indexation optimisée
    - Templates dynamiques
    - Aggregations complexes
    - Archivage automatique
    """
    
    def __init__(
        self,
        config: ElasticsearchConfig,
        index_config: TenantIndexConfig
    ):
        self.config = config
        self.index_config = index_config
        self.client: Optional[AsyncElasticsearch] = None
        
        # Templates d'index
        self.index_templates = {}
        
        # Cache des mappings
        self.mapping_cache = {}
        
        # Métriques internes
        self.stats = {
            'documents_indexed': 0,
            'indices_created': 0,
            'search_queries': 0,
            'aggregations_executed': 0,
            'errors': 0
        }
        
    async def initialize(self):
        """Initialise l'exportateur Elasticsearch."""
        try:
            # Créer le client Elasticsearch
            self.client = AsyncElasticsearch(
                hosts=self.config.hosts,
                basic_auth=(self.config.username, self.config.password) if self.config.username else None,
                api_key=self.config.api_key,
                ca_certs=self.config.ca_certs,
                verify_certs=self.config.verify_certs,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
            # Vérifier la connexion
            await self._check_elasticsearch_connection()
            
            # Créer les templates d'index
            await self._setup_index_templates()
            
            # Créer l'index initial
            await self._ensure_current_index()
            
            logger.info(
                "ElasticsearchMetricsExporter initialized successfully",
                tenant_id=self.index_config.tenant_id
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch exporter: {e}")
            raise
            
    async def _check_elasticsearch_connection(self):
        """Vérifie la connexion à Elasticsearch."""
        info = await self.client.info()
        logger.info(
            "Connected to Elasticsearch",
            version=info['version']['number'],
            cluster_name=info['cluster_name']
        )
        
    async def _setup_index_templates(self):
        """Configure les templates d'index pour le tenant."""
        template_name = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}"
        
        # Template pour les métriques IA
        ai_metrics_template = {
            "index_patterns": [f"{template_name}-ai-*"],
            "template": {
                "settings": {
                    "number_of_shards": self.index_config.number_of_shards,
                    "number_of_replicas": self.index_config.number_of_replicas,
                    "refresh_interval": self.index_config.refresh_interval,
                    "index.lifecycle.name": f"{template_name}-policy",
                    "index.lifecycle.rollover_alias": f"{template_name}-ai-alias"
                },
                "mappings": {
                    "properties": {
                        "@timestamp": {"type": "date"},
                        "tenant_id": {"type": "keyword"},
                        "metric_name": {"type": "keyword"},
                        "metric_value": {"type": "double"},
                        "metric_type": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "environment": {"type": "keyword"},
                        "labels": {
                            "type": "object",
                            "dynamic": True
                        },
                        "ai_metrics": {
                            "properties": {
                                "model_name": {"type": "keyword"},
                                "model_version": {"type": "keyword"},
                                "inference_time": {"type": "double"},
                                "accuracy": {"type": "double"},
                                "tensor_size": {"type": "keyword"},
                                "algorithm": {"type": "keyword"}
                            }
                        },
                        "business_metrics": {
                            "properties": {
                                "tracks_generated": {"type": "long"},
                                "revenue_impact": {"type": "double"},
                                "artist_engagement": {"type": "double"},
                                "collaboration_type": {"type": "keyword"},
                                "genre": {"type": "keyword"},
                                "region": {"type": "keyword"}
                            }
                        },
                        "performance_metrics": {
                            "properties": {
                                "cpu_usage": {"type": "double"},
                                "memory_usage": {"type": "double"},
                                "disk_io": {"type": "double"},
                                "network_io": {"type": "double"},
                                "response_time": {"type": "double"}
                            }
                        },
                        "metadata": {
                            "type": "object",
                            "dynamic": True
                        }
                    }
                }
            }
        }
        
        # Créer le template
        await self.client.indices.put_index_template(
            name=f"{template_name}-ai-template",
            body=ai_metrics_template
        )
        
        # Template pour les métriques business
        business_metrics_template = {
            "index_patterns": [f"{template_name}-business-*"],
            "template": {
                "settings": ai_metrics_template["template"]["settings"],
                "mappings": {
                    "properties": {
                        **ai_metrics_template["template"]["mappings"]["properties"],
                        "collaboration_metrics": {
                            "properties": {
                                "collaboration_id": {"type": "keyword"},
                                "participants": {"type": "keyword"},
                                "success_rate": {"type": "double"},
                                "duration": {"type": "long"},
                                "outcome": {"type": "keyword"}
                            }
                        }
                    }
                }
            }
        }
        
        await self.client.indices.put_index_template(
            name=f"{template_name}-business-template",
            body=business_metrics_template
        )
        
        # Politique de cycle de vie des index
        if self.index_config.enable_rollover:
            await self._setup_index_lifecycle_policy(template_name)
            
        self.index_templates[self.index_config.tenant_id] = {
            'ai_template': f"{template_name}-ai-template",
            'business_template': f"{template_name}-business-template"
        }
        
    async def _setup_index_lifecycle_policy(self, template_name: str):
        """Configure la politique de cycle de vie des index."""
        policy = {
            "policy": {
                "phases": {
                    "hot": {
                        "actions": {
                            "rollover": {
                                "max_size": self.index_config.rollover_max_size,
                                "max_docs": self.index_config.rollover_max_docs,
                                "max_age": "7d"
                            }
                        }
                    },
                    "warm": {
                        "min_age": "7d",
                        "actions": {
                            "set_priority": {"priority": 50},
                            "allocate": {"number_of_replicas": 0}
                        }
                    },
                    "cold": {
                        "min_age": "14d",
                        "actions": {
                            "set_priority": {"priority": 0}
                        }
                    },
                    "delete": {
                        "min_age": self.index_config.max_age
                    }
                }
            }
        }
        
        await self.client.ilm.put_lifecycle(
            name=f"{template_name}-policy",
            body=policy
        )
        
    async def _ensure_current_index(self):
        """S'assure que l'index courant existe."""
        current_date = datetime.now().strftime(self.index_config.date_pattern)
        
        # Index pour les métriques IA
        ai_index_name = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-ai-{current_date}"
        if not await self.client.indices.exists(index=ai_index_name):
            await self.client.indices.create(
                index=ai_index_name,
                body={
                    "aliases": {
                        f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-ai-alias": {}
                    }
                }
            )
            self.stats['indices_created'] += 1
            
        # Index pour les métriques business
        business_index_name = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-business-{current_date}"
        if not await self.client.indices.exists(index=business_index_name):
            await self.client.indices.create(
                index=business_index_name,
                body={
                    "aliases": {
                        f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-business-alias": {}
                    }
                }
            )
            self.stats['indices_created'] += 1
            
    def _get_index_name(self, metric_type: str = "ai") -> str:
        """Retourne le nom de l'index pour un type de métrique."""
        current_date = datetime.now().strftime(self.index_config.date_pattern)
        return f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-{metric_type}-{current_date}"
        
    async def index_ai_metrics(self, metrics: List[Dict[str, Any]]):
        """
        Indexe des métriques IA dans Elasticsearch.
        
        Args:
            metrics: Liste des métriques IA à indexer
        """
        try:
            index_name = self._get_index_name("ai")
            documents = []
            
            for metric in metrics:
                doc = {
                    "_index": index_name,
                    "_source": {
                        "@timestamp": metric.get('timestamp', datetime.now().isoformat()),
                        "tenant_id": self.index_config.tenant_id,
                        "metric_name": metric['name'],
                        "metric_value": metric['value'],
                        "metric_type": "ai",
                        "source": "spotify-ai-agent",
                        "environment": metric.get('environment', 'production'),
                        "labels": metric.get('labels', {}),
                        "ai_metrics": {
                            "model_name": metric.get('model_name'),
                            "model_version": metric.get('model_version'),
                            "inference_time": metric.get('inference_time'),
                            "accuracy": metric.get('accuracy'),
                            "tensor_size": metric.get('tensor_size'),
                            "algorithm": metric.get('algorithm')
                        },
                        "metadata": metric.get('metadata', {})
                    }
                }
                documents.append(doc)
                
            # Indexation en lot
            success_count, failed_docs = await async_bulk(
                self.client,
                documents,
                refresh=True
            )
            
            self.stats['documents_indexed'] += success_count
            
            if failed_docs:
                logger.error(
                    f"Failed to index {len(failed_docs)} AI metrics",
                    tenant_id=self.index_config.tenant_id
                )
                self.stats['errors'] += len(failed_docs)
                
            logger.info(
                f"Indexed {success_count} AI metrics",
                tenant_id=self.index_config.tenant_id,
                index=index_name
            )
            
        except Exception as e:
            logger.error(
                f"Failed to index AI metrics: {e}",
                tenant_id=self.index_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    async def index_business_metrics(self, metrics: List[Dict[str, Any]]):
        """
        Indexe des métriques business dans Elasticsearch.
        
        Args:
            metrics: Liste des métriques business à indexer
        """
        try:
            index_name = self._get_index_name("business")
            documents = []
            
            for metric in metrics:
                doc = {
                    "_index": index_name,
                    "_source": {
                        "@timestamp": metric.get('timestamp', datetime.now().isoformat()),
                        "tenant_id": self.index_config.tenant_id,
                        "metric_name": metric['name'],
                        "metric_value": metric['value'],
                        "metric_type": "business",
                        "source": "spotify-ai-agent",
                        "environment": metric.get('environment', 'production'),
                        "labels": metric.get('labels', {}),
                        "business_metrics": {
                            "tracks_generated": metric.get('tracks_generated'),
                            "revenue_impact": metric.get('revenue_impact'),
                            "artist_engagement": metric.get('artist_engagement'),
                            "collaboration_type": metric.get('collaboration_type'),
                            "genre": metric.get('genre'),
                            "region": metric.get('region')
                        },
                        "collaboration_metrics": {
                            "collaboration_id": metric.get('collaboration_id'),
                            "participants": metric.get('participants', []),
                            "success_rate": metric.get('success_rate'),
                            "duration": metric.get('duration'),
                            "outcome": metric.get('outcome')
                        },
                        "metadata": metric.get('metadata', {})
                    }
                }
                documents.append(doc)
                
            # Indexation en lot
            success_count, failed_docs = await async_bulk(
                self.client,
                documents,
                refresh=True
            )
            
            self.stats['documents_indexed'] += success_count
            
            if failed_docs:
                logger.error(
                    f"Failed to index {len(failed_docs)} business metrics",
                    tenant_id=self.index_config.tenant_id
                )
                self.stats['errors'] += len(failed_docs)
                
            logger.info(
                f"Indexed {success_count} business metrics",
                tenant_id=self.index_config.tenant_id,
                index=index_name
            )
            
        except Exception as e:
            logger.error(
                f"Failed to index business metrics: {e}",
                tenant_id=self.index_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    async def search_metrics(
        self,
        query: Dict[str, Any],
        metric_type: str = "ai",
        size: int = 100,
        from_: int = 0,
        sort: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Recherche des métriques dans Elasticsearch.
        
        Args:
            query: Requête Elasticsearch
            metric_type: Type de métriques (ai, business)
            size: Nombre de résultats
            from_: Offset des résultats
            sort: Critères de tri
            
        Returns:
            Résultats de la recherche
        """
        try:
            alias_name = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-{metric_type}-alias"
            
            # Ajouter le filtre tenant
            tenant_filter = {
                "term": {"tenant_id": self.index_config.tenant_id}
            }
            
            if "bool" in query:
                if "filter" not in query["bool"]:
                    query["bool"]["filter"] = []
                query["bool"]["filter"].append(tenant_filter)
            else:
                query = {
                    "bool": {
                        "must": [query],
                        "filter": [tenant_filter]
                    }
                }
                
            search_body = {
                "query": query,
                "size": size,
                "from": from_
            }
            
            if sort:
                search_body["sort"] = sort
            else:
                search_body["sort"] = [{"@timestamp": {"order": "desc"}}]
                
            result = await self.client.search(
                index=alias_name,
                body=search_body
            )
            
            self.stats['search_queries'] += 1
            
            return result
            
        except Exception as e:
            logger.error(
                f"Failed to search metrics: {e}",
                tenant_id=self.index_config.tenant_id,
                metric_type=metric_type
            )
            self.stats['errors'] += 1
            raise
            
    async def aggregate_ai_performance(
        self,
        time_range: Dict[str, str],
        interval: str = "1h"
    ) -> Dict[str, Any]:
        """
        Agrège les performances des modèles IA.
        
        Args:
            time_range: Période d'analyse (gte, lte)
            interval: Intervalle d'agrégation
            
        Returns:
            Métriques agrégées
        """
        try:
            alias_name = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-ai-alias"
            
            aggregation_query = {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": self.index_config.tenant_id}},
                            {"range": {"@timestamp": time_range}}
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    "models_performance": {
                        "terms": {
                            "field": "ai_metrics.model_name",
                            "size": 50
                        },
                        "aggs": {
                            "avg_inference_time": {
                                "avg": {"field": "ai_metrics.inference_time"}
                            },
                            "avg_accuracy": {
                                "avg": {"field": "ai_metrics.accuracy"}
                            },
                            "inference_time_over_time": {
                                "date_histogram": {
                                    "field": "@timestamp",
                                    "fixed_interval": interval
                                },
                                "aggs": {
                                    "avg_inference": {
                                        "avg": {"field": "ai_metrics.inference_time"}
                                    }
                                }
                            }
                        }
                    },
                    "algorithm_comparison": {
                        "terms": {
                            "field": "ai_metrics.algorithm",
                            "size": 20
                        },
                        "aggs": {
                            "performance_metrics": {
                                "stats": {"field": "ai_metrics.accuracy"}
                            }
                        }
                    },
                    "tensor_size_impact": {
                        "terms": {
                            "field": "ai_metrics.tensor_size",
                            "size": 10
                        },
                        "aggs": {
                            "avg_inference_time": {
                                "avg": {"field": "ai_metrics.inference_time"}
                            }
                        }
                    }
                }
            }
            
            result = await self.client.search(
                index=alias_name,
                body=aggregation_query
            )
            
            self.stats['aggregations_executed'] += 1
            
            return self._format_ai_performance_results(result['aggregations'])
            
        except Exception as e:
            logger.error(
                f"Failed to aggregate AI performance: {e}",
                tenant_id=self.index_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    def _format_ai_performance_results(self, aggregations: Dict[str, Any]) -> Dict[str, Any]:
        """Formate les résultats d'agrégation pour les performances IA."""
        results = {
            "models": [],
            "algorithms": [],
            "tensor_sizes": []
        }
        
        # Performances par modèle
        for bucket in aggregations["models_performance"]["buckets"]:
            model_data = {
                "name": bucket["key"],
                "count": bucket["doc_count"],
                "avg_inference_time": bucket["avg_inference_time"]["value"],
                "avg_accuracy": bucket["avg_accuracy"]["value"],
                "timeline": []
            }
            
            for time_bucket in bucket["inference_time_over_time"]["buckets"]:
                model_data["timeline"].append({
                    "timestamp": time_bucket["key_as_string"],
                    "avg_inference": time_bucket["avg_inference"]["value"]
                })
                
            results["models"].append(model_data)
            
        # Comparaison des algorithmes
        for bucket in aggregations["algorithm_comparison"]["buckets"]:
            results["algorithms"].append({
                "name": bucket["key"],
                "count": bucket["doc_count"],
                "performance": bucket["performance_metrics"]
            })
            
        # Impact de la taille des tenseurs
        for bucket in aggregations["tensor_size_impact"]["buckets"]:
            results["tensor_sizes"].append({
                "size": bucket["key"],
                "count": bucket["doc_count"],
                "avg_inference_time": bucket["avg_inference_time"]["value"]
            })
            
        return results
        
    async def aggregate_business_impact(
        self,
        time_range: Dict[str, str],
        groupby: str = "genre"
    ) -> Dict[str, Any]:
        """
        Agrège l'impact business par critère.
        
        Args:
            time_range: Période d'analyse
            groupby: Critère de groupement (genre, region, collaboration_type)
            
        Returns:
            Métriques business agrégées
        """
        try:
            alias_name = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-business-alias"
            
            aggregation_query = {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": self.index_config.tenant_id}},
                            {"range": {"@timestamp": time_range}}
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    f"business_by_{groupby}": {
                        "terms": {
                            "field": f"business_metrics.{groupby}",
                            "size": 50
                        },
                        "aggs": {
                            "total_tracks": {
                                "sum": {"field": "business_metrics.tracks_generated"}
                            },
                            "total_revenue": {
                                "sum": {"field": "business_metrics.revenue_impact"}
                            },
                            "avg_engagement": {
                                "avg": {"field": "business_metrics.artist_engagement"}
                            },
                            "revenue_over_time": {
                                "date_histogram": {
                                    "field": "@timestamp",
                                    "fixed_interval": "1d"
                                },
                                "aggs": {
                                    "daily_revenue": {
                                        "sum": {"field": "business_metrics.revenue_impact"}
                                    }
                                }
                            }
                        }
                    },
                    "collaboration_success": {
                        "terms": {
                            "field": "business_metrics.collaboration_type",
                            "size": 20
                        },
                        "aggs": {
                            "avg_success_rate": {
                                "avg": {"field": "collaboration_metrics.success_rate"}
                            },
                            "total_collaborations": {
                                "value_count": {"field": "collaboration_metrics.collaboration_id"}
                            }
                        }
                    }
                }
            }
            
            result = await self.client.search(
                index=alias_name,
                body=aggregation_query
            )
            
            self.stats['aggregations_executed'] += 1
            
            return self._format_business_impact_results(result['aggregations'], groupby)
            
        except Exception as e:
            logger.error(
                f"Failed to aggregate business impact: {e}",
                tenant_id=self.index_config.tenant_id
            )
            self.stats['errors'] += 1
            raise
            
    def _format_business_impact_results(
        self, 
        aggregations: Dict[str, Any], 
        groupby: str
    ) -> Dict[str, Any]:
        """Formate les résultats d'agrégation business."""
        results = {
            f"by_{groupby}": [],
            "collaborations": [],
            "totals": {
                "tracks": 0,
                "revenue": 0,
                "avg_engagement": 0
            }
        }
        
        # Résultats par critère de groupement
        group_key = f"business_by_{groupby}"
        for bucket in aggregations[group_key]["buckets"]:
            group_data = {
                "name": bucket["key"],
                "tracks_generated": bucket["total_tracks"]["value"],
                "revenue_impact": bucket["total_revenue"]["value"],
                "avg_engagement": bucket["avg_engagement"]["value"],
                "revenue_timeline": []
            }
            
            for time_bucket in bucket["revenue_over_time"]["buckets"]:
                group_data["revenue_timeline"].append({
                    "date": time_bucket["key_as_string"],
                    "revenue": time_bucket["daily_revenue"]["value"]
                })
                
            results[f"by_{groupby}"].append(group_data)
            
            # Mise à jour des totaux
            results["totals"]["tracks"] += group_data["tracks_generated"]
            results["totals"]["revenue"] += group_data["revenue_impact"]
            
        # Calcul de l'engagement moyen
        if results[f"by_{groupby}"]:
            results["totals"]["avg_engagement"] = sum(
                item["avg_engagement"] for item in results[f"by_{groupby}"]
            ) / len(results[f"by_{groupby}"])
            
        # Succès des collaborations
        for bucket in aggregations["collaboration_success"]["buckets"]:
            results["collaborations"].append({
                "type": bucket["key"],
                "avg_success_rate": bucket["avg_success_rate"]["value"],
                "total_collaborations": bucket["total_collaborations"]["value"]
            })
            
        return results
        
    async def export_tenant_data(
        self,
        output_format: str = "json",
        time_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Exporte toutes les données d'un tenant.
        
        Args:
            output_format: Format d'export (json, csv)
            time_range: Période à exporter
            
        Returns:
            Données exportées
        """
        try:
            export_data = {
                "tenant_id": self.index_config.tenant_id,
                "export_timestamp": datetime.now().isoformat(),
                "ai_metrics": [],
                "business_metrics": []
            }
            
            # Query de base
            base_query = {
                "bool": {
                    "filter": [
                        {"term": {"tenant_id": self.index_config.tenant_id}}
                    ]
                }
            }
            
            if time_range:
                base_query["bool"]["filter"].append({
                    "range": {"@timestamp": time_range}
                })
                
            # Export des métriques IA
            ai_results = await self.search_metrics(
                query=base_query,
                metric_type="ai",
                size=10000
            )
            
            for hit in ai_results["hits"]["hits"]:
                export_data["ai_metrics"].append(hit["_source"])
                
            # Export des métriques business
            business_results = await self.search_metrics(
                query=base_query,
                metric_type="business",
                size=10000
            )
            
            for hit in business_results["hits"]["hits"]:
                export_data["business_metrics"].append(hit["_source"])
                
            return export_data
            
        except Exception as e:
            logger.error(
                f"Failed to export tenant data: {e}",
                tenant_id=self.index_config.tenant_id
            )
            raise
            
    async def get_tenant_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques du tenant."""
        try:
            summary = {
                "tenant_id": self.index_config.tenant_id,
                "stats": self.stats,
                "indices": {},
                "last_7_days": {}
            }
            
            # Informations sur les indices
            ai_alias = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-ai-alias"
            business_alias = f"{self.index_config.index_prefix}-{self.index_config.tenant_id}-business-alias"
            
            # Stats des indices AI
            ai_stats = await self.client.indices.stats(index=ai_alias)
            summary["indices"]["ai"] = {
                "total_docs": ai_stats["_all"]["primaries"]["docs"]["count"],
                "size_bytes": ai_stats["_all"]["primaries"]["store"]["size_in_bytes"]
            }
            
            # Stats des indices business
            business_stats = await self.client.indices.stats(index=business_alias)
            summary["indices"]["business"] = {
                "total_docs": business_stats["_all"]["primaries"]["docs"]["count"],
                "size_bytes": business_stats["_all"]["primaries"]["store"]["size_in_bytes"]
            }
            
            # Métriques des 7 derniers jours
            seven_days_ago = datetime.now() - timedelta(days=7)
            time_range = {
                "gte": seven_days_ago.isoformat(),
                "lte": datetime.now().isoformat()
            }
            
            # Agrégations rapides
            ai_performance = await self.aggregate_ai_performance(time_range, "1d")
            business_impact = await self.aggregate_business_impact(time_range, "genre")
            
            summary["last_7_days"] = {
                "ai_performance": ai_performance,
                "business_impact": business_impact
            }
            
            return summary
            
        except Exception as e:
            logger.error(
                f"Failed to get tenant summary: {e}",
                tenant_id=self.index_config.tenant_id
            )
            raise
            
    async def cleanup(self):
        """Nettoie les ressources."""
        if self.client:
            await self.client.close()
            
        logger.info(
            "ElasticsearchMetricsExporter cleaned up",
            tenant_id=self.index_config.tenant_id
        )


# Factory pour créer des exportateurs Elasticsearch
class ElasticsearchExporterFactory:
    """Factory pour créer des exportateurs Elasticsearch configurés."""
    
    @staticmethod
    def create_spotify_ai_exporter(
        tenant_id: str,
        hosts: List[str] = None,
        username: str = None,
        password: str = None
    ) -> ElasticsearchMetricsExporter:
        """Crée un exportateur configuré pour Spotify AI."""
        if hosts is None:
            hosts = ['http://localhost:9200']
            
        es_config = ElasticsearchConfig(
            hosts=hosts,
            username=username,
            password=password,
            timeout=30,
            max_retries=3
        )
        
        index_config = TenantIndexConfig(
            tenant_id=tenant_id,
            index_prefix="spotify-ai-metrics",
            number_of_shards=3,
            number_of_replicas=1,
            max_age="90d",
            enable_rollover=True
        )
        
        return ElasticsearchMetricsExporter(es_config, index_config)


# Usage example
if __name__ == "__main__":
    async def main():
        # Configuration pour un artiste Spotify
        exporter = ElasticsearchExporterFactory.create_spotify_ai_exporter(
            tenant_id="spotify_artist_daft_punk",
            hosts=["http://localhost:9200"],
            username="elastic",
            password="changeme"
        )
        
        await exporter.initialize()
        
        # Indexation de métriques IA
        ai_metrics = [
            {
                "name": "inference_time",
                "value": 0.045,
                "model_name": "collaborative_filter_v2",
                "model_version": "2.1.0",
                "accuracy": 0.94,
                "algorithm": "deep_learning",
                "tensor_size": "medium"
            },
            {
                "name": "recommendation_accuracy",
                "value": 0.89,
                "model_name": "content_based_v1",
                "model_version": "1.5.2",
                "accuracy": 0.89,
                "algorithm": "nlp",
                "tensor_size": "large"
            }
        ]
        
        await exporter.index_ai_metrics(ai_metrics)
        
        # Indexation de métriques business
        business_metrics = [
            {
                "name": "tracks_generated",
                "value": 25,
                "tracks_generated": 25,
                "revenue_impact": 15000.50,
                "collaboration_type": "ai_assisted",
                "genre": "electronic",
                "region": "europe"
            }
        ]
        
        await exporter.index_business_metrics(business_metrics)
        
        # Recherche de métriques
        search_query = {
            "match": {
                "ai_metrics.model_name": "collaborative_filter_v2"
            }
        }
        
        results = await exporter.search_metrics(search_query, "ai")
        print(f"Found {results['hits']['total']['value']} AI metrics")
        
        # Agrégations
        time_range = {
            "gte": "now-1d",
            "lte": "now"
        }
        
        ai_performance = await exporter.aggregate_ai_performance(time_range)
        print(f"AI Performance: {ai_performance}")
        
        # Résumé du tenant
        summary = await exporter.get_tenant_summary()
        print(f"Tenant Summary: {summary}")
        
        await exporter.cleanup()
        
    asyncio.run(main())
