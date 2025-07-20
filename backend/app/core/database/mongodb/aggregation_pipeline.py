"""
MongoDB Aggregation Pipeline Utilities
=====================================
- Dynamischer Pipeline-Builder für komplexe Analysen
- Vorlagen für Spotify-spezifische Use Cases (z.B. Top Artists, Audience Segmentation)
- Security & Performance Best Practices
"""

from pymongo.collection import Collection
from typing import List, Dict, Any
import logging

class AggregationPipelineBuilder:
    def __init__(self, collection: Collection):
        self.collection = collection

    def run_pipeline(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            result = list(self.collection.aggregate(pipeline))
            logging.info(f"Aggregation pipeline executed: {pipeline}")
            return result
        except Exception as e:
            logging.error(f"Aggregation failed: {e}")
            raise

    @staticmethod
    def top_artists_pipeline(limit=10):
        return [
            {"$match": {"deleted": False}},
            {"$group": {"_id": "$artist_id", "total_streams": {"$sum": "$streams"}}},
            {"$sort": {"total_streams": -1}},
            {"$limit": limit}
        ]

    @staticmethod
    def audience_segmentation_pipeline():
        return [
            {"$match": {"deleted": False}},
            {"$group": {"_id": "$audience_segment", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

# Beispiel für Nutzung:
# builder = AggregationPipelineBuilder(collection)
# result = builder.run_pipeline(AggregationPipelineBuilder.top_artists_pipeline()
