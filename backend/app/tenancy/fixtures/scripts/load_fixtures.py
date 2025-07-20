#!/usr/bin/env python3
"""
Spotify AI Agent - Fixture Data Loading Script
==============================================

Comprehensive data loading script that supports:
- Loading fixture data from various sources
- Batch processing with progress tracking
- Data validation and transformation
- Incremental and full data loads
- Error recovery and rollback

Usage:
    python -m app.tenancy.fixtures.scripts.load_fixtures --tenant-id mycompany --data-type spotify
    python load_fixtures.py --tenant-id startup --file data.json --batch-size 1000

Author: Expert Development Team (Fahed Mlaiel)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.cache import get_redis_client
from app.tenancy.fixtures.base import FixtureManager
from app.tenancy.fixtures.data_loaders import (
    SpotifyDataLoader, AIModelLoader, AnalyticsLoader, CollaborationLoader
)
from app.tenancy.fixtures.validators import DataValidator
from app.tenancy.fixtures.monitoring import FixtureMonitor
from app.tenancy.fixtures.utils import FixtureUtils, TenantUtils
from app.tenancy.fixtures.exceptions import FixtureError, FixtureDataError
from app.tenancy.fixtures.constants import SUPPORTED_DATA_TYPES, MAX_BATCH_SIZE

logger = logging.getLogger(__name__)


class FixtureLoader:
    """
    Comprehensive fixture data loader.
    
    Supports multiple data sources and formats:
    - JSON files
    - YAML files  
    - CSV files
    - API endpoints
    - Database dumps
    """
    
    def __init__(self, session: AsyncSession, redis_client=None):
        self.session = session
        self.redis_client = redis_client
        self.fixture_manager = FixtureManager(session, redis_client)
        self.monitor = FixtureMonitor(session, redis_client)
        
        # Initialize data loaders
        self.spotify_loader = SpotifyDataLoader(session, redis_client)
        self.ai_loader = AIModelLoader(session, redis_client)
        self.analytics_loader = AnalyticsLoader(session, redis_client)
        self.collaboration_loader = CollaborationLoader(session, redis_client)
        
        # Initialize validator
        self.validator = DataValidator(session)
    
    async def load_fixtures(
        self,
        tenant_id: str,
        data_type: str,
        source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        batch_size: int = 100,
        validate: bool = True,
        dry_run: bool = False,
        incremental: bool = False
    ) -> Dict[str, Any]:
        """
        Load fixture data for a tenant.
        
        Args:
            tenant_id: Target tenant identifier
            data_type: Type of data to load (spotify, ai_models, analytics, etc.)
            source: Data source (file path, URL, or data dict)
            batch_size: Number of records to process per batch
            validate: Whether to validate data before loading
            dry_run: Perform validation without loading
            incremental: Only load new/changed data
            
        Returns:
            Loading results and metrics
        """
        start_time = datetime.now(timezone.utc)
        load_result = {
            "tenant_id": tenant_id,
            "data_type": data_type,
            "source": str(source) if source else "default",
            "status": "started",
            "start_time": start_time.isoformat(),
            "batch_size": batch_size,
            "dry_run": dry_run,
            "incremental": incremental,
            "records_processed": 0,
            "records_loaded": 0,
            "records_skipped": 0,
            "records_failed": 0,
            "batches_processed": 0,
            "validation_errors": [],
            "warnings": []
        }
        
        try:
            # Validate inputs
            await self._validate_load_inputs(tenant_id, data_type, source)
            
            # Load data from source
            data = await self._load_data_from_source(source, data_type)
            if not data:
                raise FixtureDataError("No data loaded from source")
            
            load_result["records_total"] = len(data) if isinstance(data, list) else 1
            
            # Validate data if requested
            if validate:
                validation_result = await self._validate_data(tenant_id, data_type, data)
                load_result["validation"] = validation_result
                
                if validation_result["errors"]:
                    load_result["validation_errors"] = validation_result["errors"]
                    if not dry_run:
                        raise FixtureDataError(f"Data validation failed: {len(validation_result['errors'])} errors")
            
            # Process data in batches
            if not dry_run:
                batch_results = await self._process_data_batches(
                    tenant_id, data_type, data, batch_size, incremental
                )
                
                # Aggregate batch results
                for batch_result in batch_results:
                    load_result["records_processed"] += batch_result.get("processed", 0)
                    load_result["records_loaded"] += batch_result.get("loaded", 0)
                    load_result["records_skipped"] += batch_result.get("skipped", 0)
                    load_result["records_failed"] += batch_result.get("failed", 0)
                    load_result["batches_processed"] += 1
                    
                    if batch_result.get("warnings"):
                        load_result["warnings"].extend(batch_result["warnings"])
            else:
                load_result["records_processed"] = load_result["records_total"]
                logger.info(f"DRY RUN: Would load {load_result['records_total']} records")
            
            # Calculate final metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            load_result.update({
                "status": "completed",
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "records_per_second": load_result["records_processed"] / max(duration, 0.1),
                "success_rate": load_result["records_loaded"] / max(load_result["records_processed"], 1)
            })
            
            # Record metrics
            await self.monitor.record_load_operation(tenant_id, load_result)
            
            logger.info(
                f"Data loading completed for {tenant_id}/{data_type}: "
                f"{load_result['records_loaded']}/{load_result['records_processed']} records "
                f"in {FixtureUtils.format_duration(duration)}"
            )
            
        except Exception as e:
            load_result["status"] = "failed"
            load_result["error"] = str(e)
            logger.error(f"Data loading failed for {tenant_id}/{data_type}: {e}")
            raise
        
        return load_result
    
    async def _validate_load_inputs(
        self,
        tenant_id: str,
        data_type: str,
        source: Optional[Union[str, Path, Dict[str, Any]]]
    ) -> None:
        """Validate loading inputs."""
        if not TenantUtils.validate_tenant_id(tenant_id):
            raise FixtureError(f"Invalid tenant ID: {tenant_id}")
        
        if data_type not in SUPPORTED_DATA_TYPES:
            raise FixtureError(f"Unsupported data type: {data_type}")
        
        # Check if tenant exists
        tenant_exists = await self.fixture_manager.tenant_exists(tenant_id)
        if not tenant_exists:
            raise FixtureError(f"Tenant not found: {tenant_id}")
    
    async def _load_data_from_source(
        self,
        source: Optional[Union[str, Path, Dict[str, Any]]],
        data_type: str
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Load data from various sources."""
        if source is None:
            # Load default data for the data type
            return await self._load_default_data(data_type)
        
        elif isinstance(source, dict):
            # Data provided directly
            return source
        
        elif isinstance(source, (str, Path)):
            source_path = Path(source)
            
            if source_path.exists():
                # Load from file
                if source_path.suffix.lower() == '.json':
                    return FixtureUtils.load_json_file(source_path)
                elif source_path.suffix.lower() in ['.yaml', '.yml']:
                    return FixtureUtils.load_yaml_file(source_path)
                elif source_path.suffix.lower() == '.csv':
                    return await self._load_csv_data(source_path)
                else:
                    raise FixtureDataError(f"Unsupported file format: {source_path.suffix}")
            
            elif str(source).startswith(('http://', 'https://')):
                # Load from URL
                return await self._load_url_data(str(source))
            
            else:
                raise FixtureDataError(f"Source not found: {source}")
        
        else:
            raise FixtureDataError(f"Invalid source type: {type(source)}")
    
    async def _load_default_data(self, data_type: str) -> List[Dict[str, Any]]:
        """Load default data for data type."""
        default_data = {
            "spotify": [
                {
                    "track_id": "sample_track_001",
                    "name": "Sample Track",
                    "artist": "Sample Artist",
                    "duration_ms": 180000,
                    "genre": "Electronic",
                    "popularity": 75
                }
            ],
            "ai_models": [
                {
                    "model_id": "music_classifier_v1",
                    "name": "Music Genre Classifier",
                    "type": "classification",
                    "version": "1.0.0",
                    "accuracy": 0.92
                }
            ],
            "analytics": [
                {
                    "metric_name": "user_engagement",
                    "value": 0.85,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tags": {"category": "user_behavior"}
                }
            ],
            "collaborations": [
                {
                    "collab_id": "sample_collab_001",
                    "name": "Sample Collaboration",
                    "type": "music_production",
                    "status": "active",
                    "participant_count": 3
                }
            ]
        }
        
        return default_data.get(data_type, [])
    
    async def _load_csv_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        import csv
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(dict(row))
        
        return data
    
    async def _load_url_data(self, url: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Load data from URL."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        return await response.json()
                    else:
                        text_data = await response.text()
                        return json.loads(text_data)
                else:
                    raise FixtureDataError(f"Failed to load from URL: {response.status}")
    
    async def _validate_data(
        self,
        tenant_id: str,
        data_type: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate data before loading."""
        validation_result = {
            "data_type": data_type,
            "records_validated": 0,
            "errors": [],
            "warnings": []
        }
        
        # Normalize data to list
        data_list = data if isinstance(data, list) else [data]
        validation_result["records_validated"] = len(data_list)
        
        # Validate each record
        for i, record in enumerate(data_list):
            record_errors = await self.validator.validate_record(
                tenant_id, data_type, record
            )
            
            if record_errors:
                for error in record_errors:
                    validation_result["errors"].append(f"Record {i}: {error}")
        
        return validation_result
    
    async def _process_data_batches(
        self,
        tenant_id: str,
        data_type: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        batch_size: int,
        incremental: bool
    ) -> List[Dict[str, Any]]:
        """Process data in batches."""
        # Normalize data to list
        data_list = data if isinstance(data, list) else [data]
        
        # Create batches
        batches = [
            data_list[i:i + batch_size]
            for i in range(0, len(data_list), batch_size)
        ]
        
        batch_results = []
        
        for batch_index, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch)} records)")
            
            batch_result = await self._process_batch(
                tenant_id, data_type, batch, incremental
            )
            
            batch_results.append(batch_result)
            
            # Update progress
            await self.monitor.update_progress(
                tenant_id, f"load_{data_type}", batch_index + 1, len(batches)
            )
        
        return batch_results
    
    async def _process_batch(
        self,
        tenant_id: str,
        data_type: str,
        batch: List[Dict[str, Any]],
        incremental: bool
    ) -> Dict[str, Any]:
        """Process a single batch of data."""
        batch_result = {
            "processed": len(batch),
            "loaded": 0,
            "skipped": 0,
            "failed": 0,
            "warnings": []
        }
        
        try:
            # Route to appropriate loader
            if data_type == "spotify":
                result = await self.spotify_loader.load_spotify_data(
                    tenant_id, batch, incremental
                )
            elif data_type == "ai_models":
                result = await self.ai_loader.load_model_data(
                    tenant_id, batch, incremental
                )
            elif data_type == "analytics":
                result = await self.analytics_loader.load_analytics_data(
                    tenant_id, batch, incremental
                )
            elif data_type == "collaborations":
                result = await self.collaboration_loader.load_collaboration_data(
                    tenant_id, batch, incremental
                )
            else:
                # Generic loader
                result = await self.fixture_manager.load_fixtures(
                    tenant_id, {data_type: batch}
                )
            
            # Update batch result with loader results
            if isinstance(result, dict):
                batch_result.update({
                    "loaded": result.get("loaded", 0),
                    "skipped": result.get("skipped", 0),
                    "failed": result.get("failed", 0)
                })
                
                if result.get("warnings"):
                    batch_result["warnings"].extend(result["warnings"])
            
        except Exception as e:
            batch_result["failed"] = len(batch)
            batch_result["warnings"].append(f"Batch processing failed: {str(e)}")
            logger.error(f"Batch processing failed: {e}")
        
        return batch_result


async def load_tenant_data(
    tenant_id: str,
    data_type: str,
    source: Optional[str] = None,
    batch_size: int = 100,
    validate: bool = True,
    dry_run: bool = False,
    incremental: bool = False
) -> Dict[str, Any]:
    """
    Main function to load tenant fixture data.
    
    Args:
        tenant_id: Target tenant
        data_type: Type of data to load
        source: Data source (file path or URL)
        batch_size: Batch processing size
        validate: Enable data validation
        dry_run: Validation only mode
        incremental: Only load new data
        
    Returns:
        Loading results
    """
    # Validate batch size
    if batch_size > MAX_BATCH_SIZE:
        batch_size = MAX_BATCH_SIZE
        logger.warning(f"Batch size limited to {MAX_BATCH_SIZE}")
    
    # Get database session and Redis client
    async with get_async_session() as session:
        redis_client = await get_redis_client()
        
        try:
            loader = FixtureLoader(session, redis_client)
            result = await loader.load_fixtures(
                tenant_id=tenant_id,
                data_type=data_type,
                source=source,
                batch_size=batch_size,
                validate=validate,
                dry_run=dry_run,
                incremental=incremental
            )
            
            return result
            
        finally:
            if redis_client:
                await redis_client.close()


def main():
    """Command line interface for fixture data loading."""
    parser = argparse.ArgumentParser(
        description="Load fixture data for tenant"
    )
    
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="Target tenant identifier"
    )
    
    parser.add_argument(
        "--data-type",
        choices=SUPPORTED_DATA_TYPES,
        required=True,
        help="Type of data to load"
    )
    
    parser.add_argument(
        "--source",
        help="Data source (file path, URL, or 'default')"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch processing size"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data validation"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without loading"
    )
    
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only load new/changed data"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Run data loading
        result = asyncio.run(
            load_tenant_data(
                tenant_id=args.tenant_id,
                data_type=args.data_type,
                source=args.source,
                batch_size=args.batch_size,
                validate=not args.no_validate,
                dry_run=args.dry_run,
                incremental=args.incremental
            )
        )
        
        # Display results
        print(f"\nData Loading Results for '{args.tenant_id}/{args.data_type}':")
        print(f"Status: {result['status']}")
        print(f"Duration: {FixtureUtils.format_duration(result.get('duration_seconds', 0))}")
        print(f"Records Processed: {result['records_processed']}")
        print(f"Records Loaded: {result['records_loaded']}")
        
        if result['records_skipped'] > 0:
            print(f"Records Skipped: {result['records_skipped']}")
        
        if result['records_failed'] > 0:
            print(f"Records Failed: {result['records_failed']}")
        
        if result.get('success_rate'):
            print(f"Success Rate: {result['success_rate']:.1%}")
        
        if result.get('records_per_second'):
            print(f"Processing Rate: {result['records_per_second']:.1f} records/sec")
        
        if result['status'] == 'completed':
            print("\n✅ Data loading completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Data loading failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Loading interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
