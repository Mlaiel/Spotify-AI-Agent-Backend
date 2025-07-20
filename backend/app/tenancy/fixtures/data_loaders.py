"""
Spotify AI Agent - Data Loaders
==============================

Enterprise data loading system for multi-tenant
Spotify AI Agent with specialized loaders for different data types.
"""

import asyncio
import csv
import json
import logging
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

import aiofiles
import aiohttp
from sqlalchemy import text, insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.tenancy.fixtures.base import BaseFixture, FixtureMetadata, FixtureType
from app.tenancy.fixtures.exceptions import (
    FixtureDataError,
    FixtureValidationError,
    FixtureTimeoutError
)
from app.tenancy.fixtures.constants import DEFAULT_BATCH_SIZE

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Data source type enumeration."""
    FILE = "file"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    CACHE = "cache"


class DataFormat(Enum):
    """Data format enumeration."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    YAML = "yaml"
    XML = "xml"
    AVRO = "avro"


@dataclass
class DataLoadMetrics:
    """Metrics for data loading operations."""
    total_records: int = 0
    loaded_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics."""
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 0:
                self.throughput_rps = self.loaded_records / duration
        
        if self.total_records > 0:
            self.error_rate = self.failed_records / self.total_records


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    Provides common functionality for:
    - Batch processing
    - Error handling
    - Progress tracking
    - Data validation
    """
    
    def __init__(
        self,
        tenant_id: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_errors: int = 100,
        session: Optional[AsyncSession] = None
    ):
        self.tenant_id = tenant_id
        self.batch_size = batch_size
        self.max_errors = max_errors
        self.session = session
        self.metrics = DataLoadMetrics()
        self.errors: List[str] = []
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def load_data(self) -> DataLoadMetrics:
        """Load data from source."""
        pass
    
    @abstractmethod
    async def validate_source(self) -> bool:
        """Validate data source."""
        pass
    
    async def process_batch(self, batch: List[Dict[str, Any]]) -> int:
        """Process a batch of records."""
        processed = 0
        
        for record in batch:
            try:
                if await self.validate_record(record):
                    await self.insert_record(record)
                    processed += 1
                    self.metrics.loaded_records += 1
                else:
                    self.metrics.skipped_records += 1
                    
            except Exception as e:
                self.metrics.failed_records += 1
                error_msg = f"Failed to process record: {str(e)}"
                self.errors.append(error_msg)
                self.logger.error(error_msg)
                
                if len(self.errors) > self.max_errors:
                    raise FixtureDataError(f"Too many errors: {len(self.errors)}")
        
        return processed
    
    async def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate individual record."""
        # Basic validation - override in subclasses
        return record is not None and isinstance(record, dict)
    
    @abstractmethod
    async def insert_record(self, record: Dict[str, Any]) -> None:
        """Insert record into database."""
        pass
    
    def add_error(self, error: str) -> None:
        """Add error to error list."""
        self.errors.append(error)
        self.logger.error(f"Data loader error: {error}")


class SpotifyDataLoader(DataLoader):
    """
    Specialized data loader for Spotify-related data.
    
    Handles:
    - Artist data
    - Track information
    - Playlist data
    - User listening history
    - Genre classifications
    """
    
    def __init__(
        self,
        tenant_id: str,
        spotify_client_id: str,
        spotify_client_secret: str,
        **kwargs
    ):
        super().__init__(tenant_id, **kwargs)
        self.spotify_client_id = spotify_client_id
        self.spotify_client_secret = spotify_client_secret
        self.access_token: Optional[str] = None
        self.base_url = "https://api.spotify.com/v1"
    
    async def validate_source(self) -> bool:
        """Validate Spotify API credentials and connectivity."""
        try:
            await self._get_access_token()
            return self.access_token is not None
        except Exception as e:
            self.add_error(f"Spotify API validation failed: {str(e)}")
            return False
    
    async def load_data(self) -> DataLoadMetrics:
        """Load Spotify data for tenant."""
        self.metrics.start_time = datetime.now(timezone.utc)
        
        try:
            # Get access token
            await self._get_access_token()
            
            # Load different types of Spotify data
            await self._load_genres()
            await self._load_featured_playlists()
            await self._load_new_releases()
            
            self.metrics.end_time = datetime.now(timezone.utc)
            self.metrics.calculate_metrics()
            
            self.logger.info(
                f"Spotify data loading completed for tenant {self.tenant_id}: "
                f"{self.metrics.loaded_records} records loaded"
            )
            
            return self.metrics
            
        except Exception as e:
            self.metrics.end_time = datetime.now(timezone.utc)
            error_msg = f"Spotify data loading failed: {str(e)}"
            self.add_error(error_msg)
            raise FixtureDataError(error_msg)
    
    async def _get_access_token(self) -> None:
        """Get Spotify API access token."""
        async with aiohttp.ClientSession() as session:
            auth_url = "https://accounts.spotify.com/api/token"
            
            data = {
                "grant_type": "client_credentials"
            }
            
            auth = aiohttp.BasicAuth(self.spotify_client_id, self.spotify_client_secret)
            
            async with session.post(auth_url, data=data, auth=auth) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                else:
                    raise FixtureDataError(f"Failed to get Spotify access token: {response.status}")
    
    async def _load_genres(self) -> None:
        """Load available genres from Spotify."""
        url = f"{self.base_url}/recommendations/available-genre-seeds"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    genres = data.get("genres", [])
                    
                    # Process genres in batches
                    batch = []
                    for genre in genres:
                        record = {
                            "tenant_id": self.tenant_id,
                            "genre_name": genre,
                            "is_active": True,
                            "created_at": datetime.now(timezone.utc)
                        }
                        batch.append(record)
                        
                        if len(batch) >= self.batch_size:
                            await self.process_batch(batch)
                            batch = []
                    
                    # Process remaining records
                    if batch:
                        await self.process_batch(batch)
                    
                    self.metrics.total_records += len(genres)
    
    async def _load_featured_playlists(self) -> None:
        """Load featured playlists from Spotify."""
        url = f"{self.base_url}/browse/featured-playlists"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"limit": 50}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    playlists = data.get("playlists", {}).get("items", [])
                    
                    batch = []
                    for playlist in playlists:
                        record = {
                            "tenant_id": self.tenant_id,
                            "spotify_playlist_id": playlist["id"],
                            "name": playlist["name"],
                            "description": playlist.get("description", ""),
                            "image_url": playlist["images"][0]["url"] if playlist["images"] else None,
                            "track_count": playlist["tracks"]["total"],
                            "is_public": playlist["public"],
                            "external_urls": json.dumps(playlist["external_urls"]),
                            "created_at": datetime.now(timezone.utc)
                        }
                        batch.append(record)
                        
                        if len(batch) >= self.batch_size:
                            await self.process_batch(batch)
                            batch = []
                    
                    if batch:
                        await self.process_batch(batch)
                    
                    self.metrics.total_records += len(playlists)
    
    async def _load_new_releases(self) -> None:
        """Load new album releases from Spotify."""
        url = f"{self.base_url}/browse/new-releases"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"limit": 50}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    albums = data.get("albums", {}).get("items", [])
                    
                    batch = []
                    for album in albums:
                        record = {
                            "tenant_id": self.tenant_id,
                            "spotify_album_id": album["id"],
                            "name": album["name"],
                            "album_type": album["album_type"],
                            "release_date": album["release_date"],
                            "total_tracks": album["total_tracks"],
                            "artists": json.dumps([artist["name"] for artist in album["artists"]]),
                            "image_url": album["images"][0]["url"] if album["images"] else None,
                            "external_urls": json.dumps(album["external_urls"]),
                            "created_at": datetime.now(timezone.utc)
                        }
                        batch.append(record)
                        
                        if len(batch) >= self.batch_size:
                            await self.process_batch(batch)
                            batch = []
                    
                    if batch:
                        await self.process_batch(batch)
                    
                    self.metrics.total_records += len(albums)
    
    async def insert_record(self, record: Dict[str, Any]) -> None:
        """Insert Spotify record into appropriate table."""
        # Determine table based on record content
        if "genre_name" in record:
            table = "spotify_genres"
        elif "spotify_playlist_id" in record:
            table = "spotify_playlists"
        elif "spotify_album_id" in record:
            table = "spotify_albums"
        else:
            raise ValueError("Unknown Spotify record type")
        
        # Get tenant schema
        schema = f"tenant_{self.tenant_id}"
        
        # Create insert statement
        columns = list(record.keys())
        placeholders = [f":{col}" for col in columns]
        
        sql = f"""
        INSERT INTO {schema}.{table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT DO NOTHING
        """
        
        await self.session.execute(text(sql), record)


class AIModelLoader(DataLoader):
    """
    Data loader for AI model configurations and parameters.
    
    Handles:
    - Model configurations
    - Training parameters
    - Feature definitions
    - Model metadata
    """
    
    def __init__(self, tenant_id: str, model_config_path: str, **kwargs):
        super().__init__(tenant_id, **kwargs)
        self.model_config_path = Path(model_config_path)
    
    async def validate_source(self) -> bool:
        """Validate model configuration files."""
        if not self.model_config_path.exists():
            self.add_error(f"Model config path not found: {self.model_config_path}")
            return False
        
        # Check for required configuration files
        required_files = ["models.json", "features.json", "parameters.json"]
        for file_name in required_files:
            file_path = self.model_config_path / file_name
            if not file_path.exists():
                self.add_error(f"Required model config file missing: {file_name}")
                return False
        
        return True
    
    async def load_data(self) -> DataLoadMetrics:
        """Load AI model configuration data."""
        self.metrics.start_time = datetime.now(timezone.utc)
        
        try:
            # Load model definitions
            await self._load_model_definitions()
            
            # Load feature configurations
            await self._load_feature_configurations()
            
            # Load training parameters
            await self._load_training_parameters()
            
            self.metrics.end_time = datetime.now(timezone.utc)
            self.metrics.calculate_metrics()
            
            self.logger.info(
                f"AI model data loading completed for tenant {self.tenant_id}: "
                f"{self.metrics.loaded_records} records loaded"
            )
            
            return self.metrics
            
        except Exception as e:
            self.metrics.end_time = datetime.now(timezone.utc)
            error_msg = f"AI model data loading failed: {str(e)}"
            self.add_error(error_msg)
            raise FixtureDataError(error_msg)
    
    async def _load_model_definitions(self) -> None:
        """Load AI model definitions."""
        config_file = self.model_config_path / "models.json"
        
        async with aiofiles.open(config_file, mode='r') as f:
            content = await f.read()
            models_data = json.loads(content)
        
        batch = []
        for model_id, model_config in models_data.items():
            record = {
                "tenant_id": self.tenant_id,
                "model_id": model_id,
                "model_name": model_config["name"],
                "model_type": model_config["type"],
                "version": model_config.get("version", "1.0.0"),
                "description": model_config.get("description", ""),
                "framework": model_config.get("framework", ""),
                "configuration": json.dumps(model_config.get("config", {})),
                "is_active": model_config.get("active", True),
                "created_at": datetime.now(timezone.utc)
            }
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                await self.process_batch(batch)
                batch = []
        
        if batch:
            await self.process_batch(batch)
        
        self.metrics.total_records += len(models_data)
    
    async def _load_feature_configurations(self) -> None:
        """Load feature configurations."""
        config_file = self.model_config_path / "features.json"
        
        async with aiofiles.open(config_file, mode='r') as f:
            content = await f.read()
            features_data = json.loads(content)
        
        batch = []
        for feature_id, feature_config in features_data.items():
            record = {
                "tenant_id": self.tenant_id,
                "feature_id": feature_id,
                "feature_name": feature_config["name"],
                "feature_type": feature_config["type"],
                "data_source": feature_config.get("source", ""),
                "transformation": json.dumps(feature_config.get("transformation", {})),
                "validation_rules": json.dumps(feature_config.get("validation", {})),
                "is_active": feature_config.get("active", True),
                "created_at": datetime.now(timezone.utc)
            }
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                await self.process_batch(batch)
                batch = []
        
        if batch:
            await self.process_batch(batch)
        
        self.metrics.total_records += len(features_data)
    
    async def _load_training_parameters(self) -> None:
        """Load training parameters."""
        config_file = self.model_config_path / "parameters.json"
        
        async with aiofiles.open(config_file, mode='r') as f:
            content = await f.read()
            params_data = json.loads(content)
        
        batch = []
        for param_set_id, param_config in params_data.items():
            record = {
                "tenant_id": self.tenant_id,
                "parameter_set_id": param_set_id,
                "model_type": param_config["model_type"],
                "parameters": json.dumps(param_config["parameters"]),
                "hyperparameters": json.dumps(param_config.get("hyperparameters", {})),
                "optimization_config": json.dumps(param_config.get("optimization", {})),
                "is_default": param_config.get("default", False),
                "created_at": datetime.now(timezone.utc)
            }
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                await self.process_batch(batch)
                batch = []
        
        if batch:
            await self.process_batch(batch)
        
        self.metrics.total_records += len(params_data)
    
    async def insert_record(self, record: Dict[str, Any]) -> None:
        """Insert AI model record into appropriate table."""
        schema = f"tenant_{self.tenant_id}"
        
        if "model_id" in record:
            table = "ai_models"
        elif "feature_id" in record:
            table = "ai_features"
        elif "parameter_set_id" in record:
            table = "ai_training_parameters"
        else:
            raise ValueError("Unknown AI model record type")
        
        columns = list(record.keys())
        placeholders = [f":{col}" for col in columns]
        
        sql = f"""
        INSERT INTO {schema}.{table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT DO NOTHING
        """
        
        await self.session.execute(text(sql), record)


class AnalyticsLoader(DataLoader):
    """
    Data loader for analytics and metrics data.
    
    Handles:
    - User behavior events
    - Performance metrics
    - Business metrics
    - System metrics
    """
    
    def __init__(self, tenant_id: str, analytics_data_path: str, **kwargs):
        super().__init__(tenant_id, **kwargs)
        self.analytics_data_path = Path(analytics_data_path)
    
    async def validate_source(self) -> bool:
        """Validate analytics data source."""
        if not self.analytics_data_path.exists():
            self.add_error(f"Analytics data path not found: {self.analytics_data_path}")
            return False
        
        return True
    
    async def load_data(self) -> DataLoadMetrics:
        """Load analytics data."""
        self.metrics.start_time = datetime.now(timezone.utc)
        
        try:
            # Load different types of analytics data
            await self._load_events_data()
            await self._load_metrics_data()
            await self._load_user_segments()
            
            self.metrics.end_time = datetime.now(timezone.utc)
            self.metrics.calculate_metrics()
            
            self.logger.info(
                f"Analytics data loading completed for tenant {self.tenant_id}: "
                f"{self.metrics.loaded_records} records loaded"
            )
            
            return self.metrics
            
        except Exception as e:
            self.metrics.end_time = datetime.now(timezone.utc)
            error_msg = f"Analytics data loading failed: {str(e)}"
            self.add_error(error_msg)
            raise FixtureDataError(error_msg)
    
    async def _load_events_data(self) -> None:
        """Load user events data."""
        events_file = self.analytics_data_path / "events.csv"
        
        if not events_file.exists():
            return
        
        # Read CSV file in chunks
        chunk_size = self.batch_size
        async for chunk in self._read_csv_chunks(events_file, chunk_size):
            batch = []
            for _, row in chunk.iterrows():
                record = {
                    "tenant_id": self.tenant_id,
                    "event_type": row["event_type"],
                    "event_name": row["event_name"],
                    "user_id": row.get("user_id"),
                    "session_id": row.get("session_id"),
                    "timestamp": pd.to_datetime(row["timestamp"]),
                    "properties": json.dumps(json.loads(row.get("properties", "{}"))),
                    "ip_address": row.get("ip_address"),
                    "user_agent": row.get("user_agent")
                }
                batch.append(record)
            
            await self.process_batch(batch)
            self.metrics.total_records += len(batch)
    
    async def _load_metrics_data(self) -> None:
        """Load metrics data."""
        metrics_file = self.analytics_data_path / "metrics.csv"
        
        if not metrics_file.exists():
            return
        
        async for chunk in self._read_csv_chunks(metrics_file, self.batch_size):
            batch = []
            for _, row in chunk.iterrows():
                record = {
                    "tenant_id": self.tenant_id,
                    "metric_name": row["metric_name"],
                    "metric_value": float(row["metric_value"]),
                    "dimensions": json.dumps(json.loads(row.get("dimensions", "{}"))),
                    "timestamp": pd.to_datetime(row["timestamp"]),
                    "aggregation_period": row.get("aggregation_period", "raw")
                }
                batch.append(record)
            
            await self.process_batch(batch)
            self.metrics.total_records += len(batch)
    
    async def _load_user_segments(self) -> None:
        """Load user segmentation data."""
        segments_file = self.analytics_data_path / "user_segments.json"
        
        if not segments_file.exists():
            return
        
        async with aiofiles.open(segments_file, mode='r') as f:
            content = await f.read()
            segments_data = json.loads(content)
        
        batch = []
        for segment_id, segment_config in segments_data.items():
            record = {
                "tenant_id": self.tenant_id,
                "segment_id": segment_id,
                "segment_name": segment_config["name"],
                "description": segment_config.get("description", ""),
                "criteria": json.dumps(segment_config["criteria"]),
                "is_active": segment_config.get("active", True),
                "created_at": datetime.now(timezone.utc)
            }
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                await self.process_batch(batch)
                batch = []
        
        if batch:
            await self.process_batch(batch)
        
        self.metrics.total_records += len(segments_data)
    
    async def _read_csv_chunks(self, file_path: Path, chunk_size: int) -> AsyncGenerator[pd.DataFrame, None]:
        """Read CSV file in chunks asynchronously."""
        # This is a simplified version - in production you'd want proper async CSV reading
        df_reader = pd.read_csv(file_path, chunksize=chunk_size)
        for chunk in df_reader:
            yield chunk
    
    async def insert_record(self, record: Dict[str, Any]) -> None:
        """Insert analytics record into appropriate table."""
        schema = f"analytics"  # Analytics data goes to shared analytics schema
        
        if "event_type" in record:
            table = "events"
        elif "metric_name" in record:
            table = "metrics"
        elif "segment_id" in record:
            table = "user_segments"
        else:
            raise ValueError("Unknown analytics record type")
        
        columns = list(record.keys())
        placeholders = [f":{col}" for col in columns]
        
        sql = f"""
        INSERT INTO {schema}.{table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT DO NOTHING
        """
        
        await self.session.execute(text(sql), record)


class CollaborationLoader(DataLoader):
    """
    Data loader for collaboration features and templates.
    
    Handles:
    - Collaboration templates
    - Default collaboration settings
    - Permission templates
    - Workflow definitions
    """
    
    def __init__(self, tenant_id: str, collaboration_config_path: str, **kwargs):
        super().__init__(tenant_id, **kwargs)
        self.collaboration_config_path = Path(collaboration_config_path)
    
    async def validate_source(self) -> bool:
        """Validate collaboration configuration source."""
        if not self.collaboration_config_path.exists():
            self.add_error(f"Collaboration config path not found: {self.collaboration_config_path}")
            return False
        
        return True
    
    async def load_data(self) -> DataLoadMetrics:
        """Load collaboration data."""
        self.metrics.start_time = datetime.now(timezone.utc)
        
        try:
            await self._load_collaboration_templates()
            await self._load_permission_templates()
            await self._load_workflow_definitions()
            
            self.metrics.end_time = datetime.now(timezone.utc)
            self.metrics.calculate_metrics()
            
            self.logger.info(
                f"Collaboration data loading completed for tenant {self.tenant_id}: "
                f"{self.metrics.loaded_records} records loaded"
            )
            
            return self.metrics
            
        except Exception as e:
            self.metrics.end_time = datetime.now(timezone.utc)
            error_msg = f"Collaboration data loading failed: {str(e)}"
            self.add_error(error_msg)
            raise FixtureDataError(error_msg)
    
    async def _load_collaboration_templates(self) -> None:
        """Load collaboration templates."""
        templates_file = self.collaboration_config_path / "templates.json"
        
        if not templates_file.exists():
            return
        
        async with aiofiles.open(templates_file, mode='r') as f:
            content = await f.read()
            templates_data = json.loads(content)
        
        batch = []
        for template_id, template_config in templates_data.items():
            record = {
                "tenant_id": self.tenant_id,
                "template_id": template_id,
                "template_name": template_config["name"],
                "description": template_config.get("description", ""),
                "collaboration_type": template_config["type"],
                "default_settings": json.dumps(template_config.get("settings", {})),
                "permissions": json.dumps(template_config.get("permissions", {})),
                "is_active": template_config.get("active", True),
                "created_at": datetime.now(timezone.utc)
            }
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                await self.process_batch(batch)
                batch = []
        
        if batch:
            await self.process_batch(batch)
        
        self.metrics.total_records += len(templates_data)
    
    async def _load_permission_templates(self) -> None:
        """Load permission templates."""
        permissions_file = self.collaboration_config_path / "permissions.json"
        
        if not permissions_file.exists():
            return
        
        async with aiofiles.open(permissions_file, mode='r') as f:
            content = await f.read()
            permissions_data = json.loads(content)
        
        batch = []
        for role_id, role_config in permissions_data.items():
            record = {
                "tenant_id": self.tenant_id,
                "role_id": role_id,
                "role_name": role_config["name"],
                "description": role_config.get("description", ""),
                "permissions": json.dumps(role_config["permissions"]),
                "is_default": role_config.get("default", False),
                "created_at": datetime.now(timezone.utc)
            }
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                await self.process_batch(batch)
                batch = []
        
        if batch:
            await self.process_batch(batch)
        
        self.metrics.total_records += len(permissions_data)
    
    async def _load_workflow_definitions(self) -> None:
        """Load workflow definitions."""
        workflows_file = self.collaboration_config_path / "workflows.json"
        
        if not workflows_file.exists():
            return
        
        async with aiofiles.open(workflows_file, mode='r') as f:
            content = await f.read()
            workflows_data = json.loads(content)
        
        batch = []
        for workflow_id, workflow_config in workflows_data.items():
            record = {
                "tenant_id": self.tenant_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow_config["name"],
                "description": workflow_config.get("description", ""),
                "steps": json.dumps(workflow_config["steps"]),
                "triggers": json.dumps(workflow_config.get("triggers", [])),
                "conditions": json.dumps(workflow_config.get("conditions", {})),
                "is_active": workflow_config.get("active", True),
                "created_at": datetime.now(timezone.utc)
            }
            batch.append(record)
            
            if len(batch) >= self.batch_size:
                await self.process_batch(batch)
                batch = []
        
        if batch:
            await self.process_batch(batch)
        
        self.metrics.total_records += len(workflows_data)
    
    async def insert_record(self, record: Dict[str, Any]) -> None:
        """Insert collaboration record into appropriate table."""
        schema = f"tenant_{self.tenant_id}"
        
        if "template_id" in record:
            table = "collaboration_templates"
        elif "role_id" in record:
            table = "collaboration_roles"
        elif "workflow_id" in record:
            table = "collaboration_workflows"
        else:
            raise ValueError("Unknown collaboration record type")
        
        columns = list(record.keys())
        placeholders = [f":{col}" for col in columns]
        
        sql = f"""
        INSERT INTO {schema}.{table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        ON CONFLICT DO NOTHING
        """
        
        await self.session.execute(text(sql), record)
