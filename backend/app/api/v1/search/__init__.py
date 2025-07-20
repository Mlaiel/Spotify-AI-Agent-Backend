"""
Module de recherche avancée IA pour artistes Spotify.
Expose : recherche fulltext, vectorielle, sémantique, facettes, analytics.
"""

from .elasticsearch_client import ElasticsearchClient
from .faceted_search import FacetedSearch
from .search_analytics import SearchAnalytics
from .semantic_search import SemanticSearch
from .vector_search import VectorSearch

__all__ = [
    "ElasticsearchClient",
    "FacetedSearch",
    "SearchAnalytics",
    "SemanticSearch",
    "VectorSearch"
]
