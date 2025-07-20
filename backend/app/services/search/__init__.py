"""
Spotify AI Agent – Search Module

Created by: Achiri AI Engineering Team
Roles: Lead Dev + Architecte IA, Développeur Backend Senior, Ingénieur ML, DBA/Data Engineer, Spécialiste Sécurité, Architecte Microservices
"""
from .search_service import SearchService
from .indexing_service import IndexingService
from .faceted_search_service import FacetedSearchService
from .semantic_search_service import SemanticSearchService

__version__ = "1.0.0"
__all__ = [
    "SearchService",
    "IndexingService",
    "FacetedSearchService",
    "SemanticSearchService",
]
