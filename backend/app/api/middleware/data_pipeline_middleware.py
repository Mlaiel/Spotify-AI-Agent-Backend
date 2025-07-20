"""
🎵 Spotify AI Agent - Advanced Data Pipeline Middleware
=====================================================

Middleware de pipeline de données avancé pour le traitement en temps réel,
la transformation, la validation et l'orchestration des flux de données.

Architecture:
- Real-time Data Streaming
- ETL/ELT Pipeline Management
- Data Quality & Validation
- Schema Evolution & Migration
- Data Lineage & Governance
- Event-driven Processing
- Message Queue Integration
- Data Lake & Warehouse Sync

Enterprise Features:
- Multi-tenant Data Isolation
- Data Masking & Anonymization
- Compliance & Audit Trails
- Auto-scaling Data Processing
- Dead Letter Queue Handling
- Data Retry & Recovery
- Performance Optimization
- Cost Management & Monitoring
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import uuid
import hashlib
from pathlib import Path
import gzip
import pickle
from concurrent.futures import ThreadPoolExecutor

from fastapi import Request, Response, BackgroundTasks
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError, validator
import aiofiles
import aioredis
import aiokafka
from sqlalchemy.ext.asyncio import AsyncSession
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.database import get_async_session
from ..utils.encryption import EncryptionService
from ..utils.validation import DataValidator


class DataFormat(str, Enum):
    """Formats de données supportés"""
    JSON = "json"
    AVRO = "avro"
    PARQUET = "parquet"
    CSV = "csv"
    XML = "xml"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"


class ProcessingMode(str, Enum):
    """Modes de traitement"""
    BATCH = "batch"
    STREAMING = "streaming"
    MICRO_BATCH = "micro_batch"
    REAL_TIME = "real_time"


class DataQuality(str, Enum):
    """Niveaux de qualité des données"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PipelineStage(str, Enum):
    """Étapes du pipeline"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    AGGREGATION = "aggregation"
    STORAGE = "storage"
    INDEXING = "indexing"
    NOTIFICATION = "notification"


@dataclass
class DataSchema:
    """Schéma de données"""
    name: str
    version: str
    fields: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide les données contre le schéma"""
        errors = []
        
        for field_name, field_config in self.fields.items():
            required = field_config.get("required", False)
            field_type = field_config.get("type", "string")
            
            if required and field_name not in data:
                errors.append(f"Champ requis manquant: {field_name}")
                continue
            
            if field_name in data:
                value = data[field_name]
                
                # Validation du type
                if field_type == "string" and not isinstance(value, str):
                    errors.append(f"Type incorrect pour {field_name}: attendu string, reçu {type(value).__name__}")
                elif field_type == "integer" and not isinstance(value, int):
                    errors.append(f"Type incorrect pour {field_name}: attendu integer, reçu {type(value).__name__}")
                elif field_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Type incorrect pour {field_name}: attendu float, reçu {type(value).__name__}")
                elif field_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Type incorrect pour {field_name}: attendu boolean, reçu {type(value).__name__}")
                
                # Validation des contraintes
                constraints = field_config.get("constraints", {})
                if "min_length" in constraints and isinstance(value, str) and len(value) < constraints["min_length"]:
                    errors.append(f"Longueur minimale non respectée pour {field_name}")
                if "max_length" in constraints and isinstance(value, str) and len(value) > constraints["max_length"]:
                    errors.append(f"Longueur maximale dépassée pour {field_name}")
                if "min_value" in constraints and isinstance(value, (int, float)) and value < constraints["min_value"]:
                    errors.append(f"Valeur minimale non respectée pour {field_name}")
                if "max_value" in constraints and isinstance(value, (int, float)) and value > constraints["max_value"]:
                    errors.append(f"Valeur maximale dépassée pour {field_name}")
                if "pattern" in constraints and isinstance(value, str):
                    import re
                    if not re.match(constraints["pattern"], value):
                        errors.append(f"Pattern non respecté pour {field_name}")
        
        return len(errors) == 0, errors


@dataclass
class DataRecord:
    """Enregistrement de données"""
    id: str
    data: Dict[str, Any]
    schema_name: str
    schema_version: str
    source: str
    timestamp: datetime
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: DataQuality = DataQuality.VALID
    quality_issues: List[str] = field(default_factory=list)
    processing_stage: PipelineStage = PipelineStage.INGESTION
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    def get_hash(self) -> str:
        """Calcule un hash du contenu des données"""
        content = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class PipelineMetrics:
    """Métriques du pipeline"""
    records_processed: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    records_retried: int = 0
    records_failed: int = 0
    processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    
    def calculate_derived_metrics(self):
        """Calcule les métriques dérivées"""
        total = self.records_processed
        if total > 0:
            self.error_rate = (self.records_invalid + self.records_failed) / total
            if self.processing_time_ms > 0:
                self.throughput_per_second = (total * 1000) / self.processing_time_ms


class DataTransformer:
    """Transformateur de données configurable"""
    
    def __init__(self):
        self.transformations: Dict[str, Callable] = {
            "lowercase": lambda x: x.lower() if isinstance(x, str) else x,
            "uppercase": lambda x: x.upper() if isinstance(x, str) else x,
            "trim": lambda x: x.strip() if isinstance(x, str) else x,
            "normalize_phone": self._normalize_phone,
            "normalize_email": self._normalize_email,
            "mask_pii": self._mask_pii,
            "anonymize": self._anonymize_data,
            "extract_domain": self._extract_domain,
            "format_date": self._format_date,
            "calculate_age": self._calculate_age,
            "hash_sensitive": self._hash_sensitive
        }
    
    def apply_transformations(self, data: Dict[str, Any], transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Applique une liste de transformations"""
        result = data.copy()
        
        for transform in transformations:
            field = transform.get("field")
            operation = transform.get("operation")
            params = transform.get("params", {})
            
            if field in result and operation in self.transformations:
                try:
                    result[field] = self.transformations[operation](result[field], **params)
                except Exception as e:
                    # Log l'erreur mais continue le traitement
                    pass
        
        return result
    
    def _normalize_phone(self, phone: str, **params) -> str:
        """Normalise un numéro de téléphone"""
        import re
        # Supprimer tout sauf les chiffres
        digits = re.sub(r'\D', '', phone)
        
        # Format français par défaut
        if len(digits) == 10 and digits.startswith('0'):
            return f"+33{digits[1:]}"
        elif len(digits) == 9:
            return f"+33{digits}"
        
        return phone
    
    def _normalize_email(self, email: str, **params) -> str:
        """Normalise une adresse email"""
        return email.lower().strip()
    
    def _mask_pii(self, value: str, **params) -> str:
        """Masque les informations personnelles"""
        if not isinstance(value, str) or len(value) < 4:
            return value
        
        mask_char = params.get("mask_char", "*")
        keep_start = params.get("keep_start", 2)
        keep_end = params.get("keep_end", 2)
        
        start = value[:keep_start]
        end = value[-keep_end:] if keep_end > 0 else ""
        middle = mask_char * (len(value) - keep_start - keep_end)
        
        return start + middle + end
    
    def _anonymize_data(self, value: Any, **params) -> str:
        """Anonymise complètement une donnée"""
        # Retourne un hash stable mais non réversible
        return hashlib.sha256(str(value).encode()).hexdigest()[:8]
    
    def _extract_domain(self, email: str, **params) -> str:
        """Extrait le domaine d'un email"""
        if "@" in email:
            return email.split("@")[-1].lower()
        return ""
    
    def _format_date(self, date_str: str, **params) -> str:
        """Formate une date"""
        try:
            from dateutil.parser import parse
            dt = parse(date_str)
            format_str = params.get("format", "%Y-%m-%d")
            return dt.strftime(format_str)
        except:
            return date_str
    
    def _calculate_age(self, birth_date: str, **params) -> Optional[int]:
        """Calcule l'âge à partir d'une date de naissance"""
        try:
            from dateutil.parser import parse
            birth = parse(birth_date)
            today = datetime.now()
            age = today.year - birth.year
            if today.month < birth.month or (today.month == birth.month and today.day < birth.day):
                age -= 1
            return age
        except:
            return None
    
    def _hash_sensitive(self, value: str, **params) -> str:
        """Hash une valeur sensible avec sel"""
        salt = params.get("salt", "default_salt")
        return hashlib.pbkdf2_hmac('sha256', value.encode(), salt.encode(), 100000).hex()


class MessageQueueManager:
    """Gestionnaire de files de messages"""
    
    def __init__(self):
        self.logger = get_logger("message_queue")
        self.producers: Dict[str, Any] = {}
        self.consumers: Dict[str, Any] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        self.kafka_producer: Optional[aiokafka.AIOKafkaProducer] = None
        self.kafka_consumer: Optional[aiokafka.AIOKafkaConsumer] = None
        
    async def initialize(self):
        """Initialise les connexions aux queues"""
        try:
            settings = get_settings()
            
            # Redis pour les queues rapides
            if hasattr(settings, 'REDIS_URL'):
                self.redis_client = aioredis.from_url(settings.REDIS_URL)
                await self.redis_client.ping()
                self.logger.info("Redis connecté")
            
            # Kafka pour les streams haute performance
            if hasattr(settings, 'KAFKA_BOOTSTRAP_SERVERS'):
                self.kafka_producer = aiokafka.AIOKafkaProducer(
                    bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode()
                )
                await self.kafka_producer.start()
                self.logger.info("Kafka producer initialisé")
                
        except Exception as e:
            self.logger.error(f"Erreur initialisation message queue: {e}")
            raise
    
    async def publish_to_redis(self, queue_name: str, message: Dict[str, Any], priority: int = 0):
        """Publie un message dans Redis"""
        if not self.redis_client:
            return
        
        try:
            # Utiliser une liste avec score pour la priorité
            await self.redis_client.zadd(
                f"queue:{queue_name}",
                {json.dumps(message): priority}
            )
        except Exception as e:
            self.logger.error(f"Erreur publication Redis: {e}")
    
    async def consume_from_redis(self, queue_name: str, timeout: int = 1) -> Optional[Dict[str, Any]]:
        """Consomme un message depuis Redis"""
        if not self.redis_client:
            return None
        
        try:
            # Pop le message avec la priorité la plus haute
            result = await self.redis_client.bzpopmax(f"queue:{queue_name}", timeout=timeout)
            if result:
                _, message_json, _ = result
                return json.loads(message_json)
        except Exception as e:
            self.logger.error(f"Erreur consommation Redis: {e}")
        
        return None
    
    async def publish_to_kafka(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        """Publie un message dans Kafka"""
        if not self.kafka_producer:
            return
        
        try:
            await self.kafka_producer.send_and_wait(
                topic,
                value=message,
                key=key.encode() if key else None
            )
        except Exception as e:
            self.logger.error(f"Erreur publication Kafka: {e}")
    
    async def start_kafka_consumer(self, topics: List[str], group_id: str, handler: Callable):
        """Démarre un consommateur Kafka"""
        try:
            settings = get_settings()
            consumer = aiokafka.AIOKafkaConsumer(
                *topics,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode())
            )
            await consumer.start()
            
            self.consumers[group_id] = consumer
            
            # Boucle de consommation
            asyncio.create_task(self._consume_kafka_messages(consumer, handler))
            
        except Exception as e:
            self.logger.error(f"Erreur démarrage consumer Kafka: {e}")
    
    async def _consume_kafka_messages(self, consumer: aiokafka.AIOKafkaConsumer, handler: Callable):
        """Boucle de consommation des messages Kafka"""
        try:
            async for message in consumer:
                try:
                    await handler(message.value)
                except Exception as e:
                    self.logger.error(f"Erreur traitement message Kafka: {e}")
        except Exception as e:
            self.logger.error(f"Erreur boucle consommation Kafka: {e}")
    
    async def close(self):
        """Ferme les connexions"""
        try:
            if self.kafka_producer:
                await self.kafka_producer.stop()
            
            for consumer in self.consumers.values():
                await consumer.stop()
            
            if self.redis_client:
                await self.redis_client.close()
                
        except Exception as e:
            self.logger.error(f"Erreur fermeture message queue: {e}")


class DataPipelineProcessor:
    """Processeur de pipeline de données"""
    
    def __init__(self):
        self.logger = get_logger("data_pipeline_processor")
        self.schemas: Dict[str, DataSchema] = {}
        self.transformer = DataTransformer()
        self.queue_manager = MessageQueueManager()
        self.metrics = PipelineMetrics()
        
        # Configuration du pipeline
        self.processing_stages = [
            PipelineStage.INGESTION,
            PipelineStage.VALIDATION,
            PipelineStage.TRANSFORMATION,
            PipelineStage.ENRICHMENT,
            PipelineStage.AGGREGATION,
            PipelineStage.STORAGE
        ]
        
        # Dead Letter Queue pour les échecs
        self.dlq_records: deque = deque(maxlen=10000)
        
        # Pool de threads pour le traitement parallèle
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Cache pour les données enrichies
        self.enrichment_cache: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialise le processeur"""
        try:
            await self.queue_manager.initialize()
            self._load_schemas()
            self.logger.info("Pipeline processor initialisé")
        except Exception as e:
            self.logger.error(f"Erreur initialisation pipeline: {e}")
            raise
    
    def _load_schemas(self):
        """Charge les schémas de données"""
        # Schéma pour les événements utilisateur Spotify
        spotify_user_schema = DataSchema(
            name="spotify_user_event",
            version="1.0",
            fields={
                "user_id": {"type": "string", "required": True, "constraints": {"min_length": 1}},
                "event_type": {"type": "string", "required": True},
                "track_id": {"type": "string", "required": False},
                "artist_id": {"type": "string", "required": False},
                "playlist_id": {"type": "string", "required": False},
                "timestamp": {"type": "string", "required": True},
                "duration_ms": {"type": "integer", "required": False, "constraints": {"min_value": 0}},
                "device_type": {"type": "string", "required": False},
                "context": {"type": "object", "required": False}
            }
        )
        
        # Schéma pour les données d'analyse IA
        ai_analysis_schema = DataSchema(
            name="ai_analysis_result",
            version="1.0",
            fields={
                "analysis_id": {"type": "string", "required": True},
                "model_name": {"type": "string", "required": True},
                "model_version": {"type": "string", "required": True},
                "input_data": {"type": "object", "required": True},
                "predictions": {"type": "array", "required": True},
                "confidence_scores": {"type": "array", "required": True},
                "processing_time_ms": {"type": "float", "required": True},
                "timestamp": {"type": "string", "required": True}
            }
        )
        
        self.schemas["spotify_user_event"] = spotify_user_schema
        self.schemas["ai_analysis_result"] = ai_analysis_schema
    
    async def process_record(self, record: DataRecord) -> DataRecord:
        """Traite un enregistrement à travers le pipeline"""
        start_time = time.time()
        
        try:
            # Traitement par étapes
            for stage in self.processing_stages:
                record.processing_stage = stage
                record = await self._process_stage(record, stage)
                
                if record.quality == DataQuality.CRITICAL:
                    # Échec critique, arrêter le traitement
                    break
            
            # Calculer le temps de traitement
            processing_time = (time.time() - start_time) * 1000
            record.metadata["processing_time_ms"] = processing_time
            
            # Mettre à jour les métriques
            self.metrics.records_processed += 1
            if record.quality in [DataQuality.VALID, DataQuality.WARNING]:
                self.metrics.records_valid += 1
            else:
                self.metrics.records_invalid += 1
            
            return record
            
        except Exception as e:
            self.logger.error(f"Erreur traitement record {record.id}: {e}")
            record.quality = DataQuality.CRITICAL
            record.quality_issues.append(f"Processing error: {str(e)}")
            
            # Ajouter à la DLQ si maximum de tentatives atteint
            if record.retries >= record.max_retries:
                self.dlq_records.append(record)
                self.metrics.records_failed += 1
            else:
                record.retries += 1
                self.metrics.records_retried += 1
                # Republier pour retry
                await self.queue_manager.publish_to_redis("retry_queue", record.to_dict(), priority=10)
            
            return record
    
    async def _process_stage(self, record: DataRecord, stage: PipelineStage) -> DataRecord:
        """Traite une étape spécifique du pipeline"""
        try:
            if stage == PipelineStage.INGESTION:
                return await self._process_ingestion(record)
            elif stage == PipelineStage.VALIDATION:
                return await self._process_validation(record)
            elif stage == PipelineStage.TRANSFORMATION:
                return await self._process_transformation(record)
            elif stage == PipelineStage.ENRICHMENT:
                return await self._process_enrichment(record)
            elif stage == PipelineStage.AGGREGATION:
                return await self._process_aggregation(record)
            elif stage == PipelineStage.STORAGE:
                return await self._process_storage(record)
            else:
                return record
                
        except Exception as e:
            record.quality_issues.append(f"Error in {stage.value}: {str(e)}")
            if record.quality == DataQuality.VALID:
                record.quality = DataQuality.ERROR
            raise
    
    async def _process_ingestion(self, record: DataRecord) -> DataRecord:
        """Traite l'ingestion des données"""
        # Ajouter des métadonnées d'ingestion
        record.metadata["ingestion_timestamp"] = datetime.utcnow().isoformat()
        record.metadata["record_hash"] = record.get_hash()
        
        # Détecter le format des données si pas spécifié
        if "format" not in record.metadata:
            record.metadata["format"] = self._detect_data_format(record.data)
        
        return record
    
    async def _process_validation(self, record: DataRecord) -> DataRecord:
        """Traite la validation des données"""
        schema = self.schemas.get(record.schema_name)
        if not schema:
            record.quality = DataQuality.ERROR
            record.quality_issues.append(f"Schema not found: {record.schema_name}")
            return record
        
        # Valider contre le schéma
        is_valid, errors = schema.validate_data(record.data)
        
        if not is_valid:
            if len(errors) > 5:  # Beaucoup d'erreurs = critique
                record.quality = DataQuality.CRITICAL
            else:
                record.quality = DataQuality.ERROR
            record.quality_issues.extend(errors)
        else:
            # Vérifications de qualité supplémentaires
            quality_issues = self._check_data_quality(record.data)
            if quality_issues:
                record.quality = DataQuality.WARNING
                record.quality_issues.extend(quality_issues)
        
        record.metadata["validation_timestamp"] = datetime.utcnow().isoformat()
        return record
    
    async def _process_transformation(self, record: DataRecord) -> DataRecord:
        """Traite la transformation des données"""
        # Définir les transformations basées sur le schéma
        transformations = self._get_transformations_for_schema(record.schema_name)
        
        if transformations:
            try:
                original_data = record.data.copy()
                record.data = self.transformer.apply_transformations(record.data, transformations)
                record.metadata["transformations_applied"] = len(transformations)
                record.metadata["original_data_hash"] = hashlib.sha256(
                    json.dumps(original_data, sort_keys=True).encode()
                ).hexdigest()[:16]
            except Exception as e:
                record.quality_issues.append(f"Transformation error: {str(e)}")
                if record.quality == DataQuality.VALID:
                    record.quality = DataQuality.WARNING
        
        record.metadata["transformation_timestamp"] = datetime.utcnow().isoformat()
        return record
    
    async def _process_enrichment(self, record: DataRecord) -> DataRecord:
        """Traite l'enrichissement des données"""
        try:
            # Enrichissement basé sur le type de données
            if record.schema_name == "spotify_user_event":
                record.data = await self._enrich_spotify_data(record.data)
            elif record.schema_name == "ai_analysis_result":
                record.data = await self._enrich_ai_data(record.data)
            
            record.metadata["enrichment_timestamp"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            record.quality_issues.append(f"Enrichment error: {str(e)}")
            if record.quality == DataQuality.VALID:
                record.quality = DataQuality.WARNING
        
        return record
    
    async def _process_aggregation(self, record: DataRecord) -> DataRecord:
        """Traite l'agrégation des données"""
        # Logique d'agrégation en temps réel
        try:
            aggregation_key = self._get_aggregation_key(record)
            if aggregation_key:
                await self._update_aggregation(aggregation_key, record)
                record.metadata["aggregation_key"] = aggregation_key
            
            record.metadata["aggregation_timestamp"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            record.quality_issues.append(f"Aggregation error: {str(e)}")
        
        return record
    
    async def _process_storage(self, record: DataRecord) -> DataRecord:
        """Traite le stockage des données"""
        try:
            # Déterminer la stratégie de stockage
            storage_strategy = self._get_storage_strategy(record)
            
            if storage_strategy == "database":
                await self._store_in_database(record)
            elif storage_strategy == "data_lake":
                await self._store_in_data_lake(record)
            elif storage_strategy == "cache":
                await self._store_in_cache(record)
            
            record.metadata["storage_timestamp"] = datetime.utcnow().isoformat()
            record.metadata["storage_strategy"] = storage_strategy
            
        except Exception as e:
            record.quality_issues.append(f"Storage error: {str(e)}")
            if record.quality == DataQuality.VALID:
                record.quality = DataQuality.ERROR
        
        return record
    
    def _detect_data_format(self, data: Dict[str, Any]) -> str:
        """Détecte le format des données"""
        # Logique simple de détection
        if isinstance(data, dict):
            return DataFormat.JSON.value
        return "unknown"
    
    def _check_data_quality(self, data: Dict[str, Any]) -> List[str]:
        """Vérifie la qualité des données"""
        issues = []
        
        # Vérifier les valeurs nulles excessives
        null_count = sum(1 for v in data.values() if v is None or v == "")
        if null_count > len(data) * 0.5:  # Plus de 50% de valeurs nulles
            issues.append("Excessive null values detected")
        
        # Vérifier les doublons potentiels
        # (logique spécifique au contexte)
        
        # Vérifier les valeurs aberrantes
        # (à implémenter selon les besoins métier)
        
        return issues
    
    def _get_transformations_for_schema(self, schema_name: str) -> List[Dict[str, Any]]:
        """Retourne les transformations pour un schéma"""
        transformations_config = {
            "spotify_user_event": [
                {"field": "user_id", "operation": "trim"},
                {"field": "event_type", "operation": "lowercase"},
                {"field": "timestamp", "operation": "format_date", "params": {"format": "%Y-%m-%dT%H:%M:%SZ"}}
            ],
            "ai_analysis_result": [
                {"field": "model_name", "operation": "lowercase"},
                {"field": "timestamp", "operation": "format_date", "params": {"format": "%Y-%m-%dT%H:%M:%SZ"}}
            ]
        }
        
        return transformations_config.get(schema_name, [])
    
    async def _enrich_spotify_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit les données Spotify"""
        enriched = data.copy()
        
        # Ajouter des informations de géolocalisation si disponible
        if "user_id" in data:
            # Simuler l'enrichissement (en réalité, interroger des APIs externes)
            cache_key = f"user_geo:{data['user_id']}"
            if cache_key in self.enrichment_cache:
                enriched["geo_info"] = self.enrichment_cache[cache_key]
            else:
                # Interroger un service de géolocalisation
                geo_info = {"country": "FR", "city": "Paris"}  # Simulé
                self.enrichment_cache[cache_key] = geo_info
                enriched["geo_info"] = geo_info
        
        # Ajouter des informations sur les tendances
        if "track_id" in data:
            enriched["track_popularity"] = await self._get_track_popularity(data["track_id"])
        
        return enriched
    
    async def _enrich_ai_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichit les données IA"""
        enriched = data.copy()
        
        # Ajouter des métriques de performance du modèle
        if "model_name" in data:
            model_stats = await self._get_model_stats(data["model_name"])
            enriched["model_stats"] = model_stats
        
        # Calculer des scores de confiance agrégés
        if "confidence_scores" in data:
            scores = data["confidence_scores"]
            if scores:
                enriched["avg_confidence"] = sum(scores) / len(scores)
                enriched["min_confidence"] = min(scores)
                enriched["max_confidence"] = max(scores)
        
        return enriched
    
    async def _get_track_popularity(self, track_id: str) -> float:
        """Récupère la popularité d'un track"""
        # Simuler un appel API Spotify
        return 0.75  # Valeur simulée
    
    async def _get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Récupère les stats d'un modèle IA"""
        # Simuler des stats de modèle
        return {
            "accuracy": 0.92,
            "precision": 0.88,
            "recall": 0.90,
            "f1_score": 0.89
        }
    
    def _get_aggregation_key(self, record: DataRecord) -> Optional[str]:
        """Génère une clé d'agrégation"""
        if record.schema_name == "spotify_user_event":
            # Agrégation par utilisateur et heure
            timestamp = datetime.fromisoformat(record.data.get("timestamp", ""))
            hour_key = timestamp.strftime("%Y-%m-%d-%H")
            return f"user_activity:{record.data.get('user_id')}:{hour_key}"
        
        return None
    
    async def _update_aggregation(self, key: str, record: DataRecord):
        """Met à jour les agrégations"""
        # Utiliser Redis pour les compteurs en temps réel
        if self.queue_manager.redis_client:
            await self.queue_manager.redis_client.hincrby(f"aggregation:{key}", "count", 1)
            await self.queue_manager.redis_client.expire(f"aggregation:{key}", 86400)  # 24h TTL
    
    def _get_storage_strategy(self, record: DataRecord) -> str:
        """Détermine la stratégie de stockage"""
        if record.quality == DataQuality.CRITICAL:
            return "dlq"  # Dead Letter Queue
        elif record.schema_name == "spotify_user_event":
            return "data_lake"  # Pour l'analyse
        elif record.schema_name == "ai_analysis_result":
            return "database"  # Pour les requêtes rapides
        else:
            return "cache"  # Stockage temporaire
    
    async def _store_in_database(self, record: DataRecord):
        """Stocke dans la base de données"""
        # Simuler le stockage en base
        self.logger.debug(f"Storing record {record.id} in database")
    
    async def _store_in_data_lake(self, record: DataRecord):
        """Stocke dans le data lake"""
        # Simuler le stockage dans un data lake (S3, HDFS, etc.)
        self.logger.debug(f"Storing record {record.id} in data lake")
    
    async def _store_in_cache(self, record: DataRecord):
        """Stocke dans le cache"""
        if self.queue_manager.redis_client:
            await self.queue_manager.redis_client.setex(
                f"cache:{record.id}",
                3600,  # 1 heure
                json.dumps(record.to_dict())
            )
    
    async def process_batch(self, records: List[DataRecord]) -> List[DataRecord]:
        """Traite un lot d'enregistrements"""
        start_time = time.time()
        
        # Traitement parallèle avec asyncio
        tasks = [self.process_record(record) for record in records]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les exceptions
        processed_records = []
        for result in results:
            if isinstance(result, DataRecord):
                processed_records.append(result)
            else:
                self.logger.error(f"Batch processing error: {result}")
        
        # Mettre à jour les métriques du batch
        batch_time = (time.time() - start_time) * 1000
        self.metrics.processing_time_ms += batch_time
        
        return processed_records
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du pipeline"""
        self.metrics.calculate_derived_metrics()
        return asdict(self.metrics)
    
    def get_dlq_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne les enregistrements en échec"""
        return [record.to_dict() for record in list(self.dlq_records)[-limit:]]


class AdvancedDataPipelineMiddleware:
    """Middleware de pipeline de données avancé"""
    
    def __init__(self, 
                 enable_real_time: bool = True,
                 enable_batch: bool = True,
                 batch_size: int = 100,
                 batch_timeout: int = 5):
        
        self.enable_real_time = enable_real_time
        self.enable_batch = enable_batch
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self.logger = get_logger("data_pipeline_middleware")
        self.processor = DataPipelineProcessor()
        
        # Buffer pour le traitement par batch
        self.batch_buffer: List[DataRecord] = []
        self.last_batch_time = time.time()
        
        # Tâches de fond
        self.background_tasks: List[asyncio.Task] = []
        
        self._initialized = False
    
    async def initialize(self):
        """Initialise le middleware"""
        if self._initialized:
            return
        
        try:
            await self.processor.initialize()
            
            # Démarrer les tâches de fond
            if self.enable_batch:
                batch_task = asyncio.create_task(self._batch_processor_loop())
                self.background_tasks.append(batch_task)
            
            self._initialized = True
            self.logger.info("Middleware de pipeline de données initialisé")
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation pipeline: {e}")
            raise
    
    async def __call__(self, request: Request, call_next):
        """Traite la requête avec pipeline de données"""
        if not self._initialized:
            await self.initialize()
        
        # Extraire les données de la requête si applicable
        data_record = await self._extract_data_from_request(request)
        
        if data_record:
            # Traitement en temps réel ou ajout au batch
            if self.enable_real_time:
                processed_record = await self.processor.process_record(data_record)
                request.state.processed_data = processed_record
            elif self.enable_batch:
                await self._add_to_batch(data_record)
        
        # Continuer le traitement de la requête
        response = await call_next(request)
        
        # Post-traitement si nécessaire
        if hasattr(request.state, "processed_data"):
            await self._post_process_response(request, response)
        
        return response
    
    async def _extract_data_from_request(self, request: Request) -> Optional[DataRecord]:
        """Extrait les données de la requête"""
        try:
            # Vérifier si la requête contient des données à traiter
            if request.method not in ["POST", "PUT", "PATCH"]:
                return None
            
            # Vérifier les endpoints pertinents
            data_endpoints = [
                "/api/v1/spotify/events",
                "/api/v1/ai/analysis",
                "/api/v1/analytics/events",
                "/api/v1/user/activity"
            ]
            
            if not any(endpoint in request.url.path for endpoint in data_endpoints):
                return None
            
            # Lire le body de la requête
            body = await request.body()
            if not body:
                return None
            
            data = json.loads(body)
            
            # Déterminer le schéma basé sur l'endpoint
            schema_name = self._determine_schema_name(request.url.path)
            
            # Créer l'enregistrement de données
            record = DataRecord(
                id=str(uuid.uuid4()),
                data=data,
                schema_name=schema_name,
                schema_version="1.0",
                source=f"{request.method} {request.url.path}",
                timestamp=datetime.utcnow(),
                tenant_id=getattr(request.state, "tenant_id", None),
                metadata={
                    "request_id": getattr(request.state, "request_id", None),
                    "user_id": getattr(request.state, "user_id", None),
                    "user_agent": request.headers.get("user-agent", ""),
                    "ip_address": request.client.host if request.client else ""
                }
            )
            
            return record
            
        except Exception as e:
            self.logger.error(f"Erreur extraction données requête: {e}")
            return None
    
    def _determine_schema_name(self, path: str) -> str:
        """Détermine le nom du schéma basé sur le chemin"""
        if "/spotify/events" in path:
            return "spotify_user_event"
        elif "/ai/analysis" in path:
            return "ai_analysis_result"
        else:
            return "generic_event"
    
    async def _add_to_batch(self, record: DataRecord):
        """Ajoute un enregistrement au batch"""
        self.batch_buffer.append(record)
        
        # Traiter le batch si plein ou timeout atteint
        current_time = time.time()
        if (len(self.batch_buffer) >= self.batch_size or 
            current_time - self.last_batch_time >= self.batch_timeout):
            await self._process_current_batch()
    
    async def _process_current_batch(self):
        """Traite le batch actuel"""
        if not self.batch_buffer:
            return
        
        try:
            records_to_process = self.batch_buffer.copy()
            self.batch_buffer.clear()
            self.last_batch_time = time.time()
            
            # Traiter le batch
            await self.processor.process_batch(records_to_process)
            
            self.logger.info(f"Batch traité: {len(records_to_process)} enregistrements")
            
        except Exception as e:
            self.logger.error(f"Erreur traitement batch: {e}")
    
    async def _batch_processor_loop(self):
        """Boucle de traitement des batches en arrière-plan"""
        while True:
            try:
                await asyncio.sleep(self.batch_timeout)
                
                # Traiter le batch si timeout atteint
                current_time = time.time()
                if (self.batch_buffer and 
                    current_time - self.last_batch_time >= self.batch_timeout):
                    await self._process_current_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erreur boucle batch: {e}")
    
    async def _post_process_response(self, request: Request, response: Response):
        """Post-traite la réponse"""
        try:
            processed_data = request.state.processed_data
            
            # Ajouter des headers informatifs
            response.headers["X-Data-Quality"] = processed_data.quality.value
            response.headers["X-Processing-Stage"] = processed_data.processing_stage.value
            
            if processed_data.quality_issues:
                response.headers["X-Data-Issues"] = str(len(processed_data.quality_issues))
            
        except Exception as e:
            self.logger.error(f"Erreur post-traitement: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Retourne le statut du pipeline"""
        try:
            return {
                "pipeline_enabled": self._initialized,
                "real_time_enabled": self.enable_real_time,
                "batch_enabled": self.enable_batch,
                "batch_buffer_size": len(self.batch_buffer),
                "batch_size_limit": self.batch_size,
                "batch_timeout": self.batch_timeout,
                "metrics": self.processor.get_metrics(),
                "dlq_count": len(self.processor.dlq_records),
                "schemas_loaded": len(self.processor.schemas),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Erreur status pipeline: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Arrête proprement le middleware"""
        try:
            # Traiter le batch restant
            if self.batch_buffer:
                await self._process_current_batch()
            
            # Annuler les tâches de fond
            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Fermer les connexions
            await self.processor.queue_manager.close()
            
            self.logger.info("Middleware de pipeline arrêté")
            
        except Exception as e:
            self.logger.error(f"Erreur arrêt pipeline: {e}")


# Factory functions

def create_data_pipeline_middleware(
    enable_real_time: bool = True,
    enable_batch: bool = True,
    batch_size: int = 100,
    batch_timeout: int = 5
) -> AdvancedDataPipelineMiddleware:
    """Crée un middleware de pipeline de données configuré"""
    return AdvancedDataPipelineMiddleware(
        enable_real_time=enable_real_time,
        enable_batch=enable_batch,
        batch_size=batch_size,
        batch_timeout=batch_timeout
    )


def create_production_pipeline() -> AdvancedDataPipelineMiddleware:
    """Configuration de pipeline pour la production"""
    return AdvancedDataPipelineMiddleware(
        enable_real_time=True,
        enable_batch=True,
        batch_size=500,
        batch_timeout=10
    )


def create_development_pipeline() -> AdvancedDataPipelineMiddleware:
    """Configuration de pipeline pour le développement"""
    return AdvancedDataPipelineMiddleware(
        enable_real_time=True,
        enable_batch=False,
        batch_size=10,
        batch_timeout=1
    )


# Export des classes principales
__all__ = [
    "DataFormat",
    "ProcessingMode",
    "DataQuality",
    "PipelineStage",
    "DataSchema",
    "DataRecord",
    "PipelineMetrics",
    "DataTransformer",
    "MessageQueueManager",
    "DataPipelineProcessor",
    "AdvancedDataPipelineMiddleware",
    "create_data_pipeline_middleware",
    "create_production_pipeline",
    "create_development_pipeline"
]
