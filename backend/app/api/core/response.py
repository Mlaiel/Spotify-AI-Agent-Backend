"""
🎵 Spotify AI Agent - API Response Management
============================================

Système de gestion des réponses API enterprise avec formatage standardisé,
métadonnées enrichies, et patterns de réponse cohérents.

Architecture:
- Formats de réponse standardisés
- Métadonnées automatiques
- Pagination intégrée
- Sérialisation optimisée
- Headers de performance
- Validation de sortie

Développé par Fahed Mlaiel - Enterprise Response Management Expert
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from enum import Enum
from dataclasses import dataclass, asdict

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


T = TypeVar('T')


class ResponseStatus(str, Enum):
    """Statuts de réponse possibles"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


class ResponseFormat(str, Enum):
    """Formats de réponse supportés"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    MSGPACK = "msgpack"


@dataclass
class ResponseMetadata:
    """Métadonnées de réponse"""
    timestamp: str
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    version: str = "2.0.0"
    execution_time_ms: Optional[float] = None
    cache_hit: Optional[bool] = None
    cache_ttl: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PaginationInfo:
    """Informations de pagination"""
    page: int
    per_page: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool
    next_page: Optional[int] = None
    prev_page: Optional[int] = None
    
    @classmethod
    def create(
        cls,
        page: int,
        per_page: int,
        total_items: int
    ) -> 'PaginationInfo':
        """Crée une instance avec calculs automatiques"""
        total_pages = (total_items + per_page - 1) // per_page
        has_next = page < total_pages
        has_prev = page > 1
        
        return cls(
            page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
            next_page=page + 1 if has_next else None,
            prev_page=page - 1 if has_prev else None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)


class APIResponse(Generic[T]):
    """Réponse API standardisée"""
    
    def __init__(
        self,
        data: T = None,
        status: ResponseStatus = ResponseStatus.SUCCESS,
        message: str = None,
        metadata: ResponseMetadata = None,
        pagination: PaginationInfo = None,
        warnings: List[str] = None,
        debug_info: Dict[str, Any] = None
    ):
        self.data = data
        self.status = status
        self.message = message
        self.metadata = metadata or ResponseMetadata(
            timestamp=datetime.utcnow().isoformat()
        )
        self.pagination = pagination
        self.warnings = warnings or []
        self.debug_info = debug_info or {}
    
    def add_warning(self, warning: str):
        """Ajoute un avertissement"""
        self.warnings.append(warning)
        if self.status == ResponseStatus.SUCCESS:
            self.status = ResponseStatus.WARNING
    
    def add_debug_info(self, key: str, value: Any):
        """Ajoute des informations de debug"""
        self.debug_info[key] = value
    
    def set_metadata(self, **kwargs):
        """Met à jour les métadonnées"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
    
    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        result = {
            "status": self.status,
            "data": self.data,
            "metadata": self.metadata.to_dict()
        }
        
        if self.message:
            result["message"] = self.message
        
        if self.pagination:
            result["pagination"] = self.pagination.to_dict()
        
        if self.warnings:
            result["warnings"] = self.warnings
        
        if include_debug and self.debug_info:
            result["debug"] = self.debug_info
        
        return result
    
    def to_json_response(
        self,
        status_code: int = 200,
        headers: Dict[str, str] = None,
        include_debug: bool = False
    ) -> JSONResponse:
        """Convertit en JSONResponse FastAPI"""
        content = self.to_dict(include_debug)
        
        # Headers par défaut
        response_headers = {
            "X-Request-ID": self.metadata.request_id or "",
            "X-Correlation-ID": self.metadata.correlation_id or "",
            "X-Response-Time": str(self.metadata.execution_time_ms or 0),
            "X-Cache-Hit": str(self.metadata.cache_hit).lower() if self.metadata.cache_hit is not None else "false"
        }
        
        if headers:
            response_headers.update(headers)
        
        return JSONResponse(
            content=jsonable_encoder(content),
            status_code=status_code,
            headers=response_headers
        )


class SuccessResponse(APIResponse[T]):
    """Réponse de succès"""
    
    def __init__(self, data: T, message: str = None, **kwargs):
        super().__init__(
            data=data,
            status=ResponseStatus.SUCCESS,
            message=message or "Request completed successfully",
            **kwargs
        )


class ErrorResponse(APIResponse[None]):
    """Réponse d'erreur"""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        error_details: Dict[str, Any] = None,
        **kwargs
    ):
        data = {
            "error_code": error_code,
            "error_details": error_details or {}
        }
        
        super().__init__(
            data=data,
            status=ResponseStatus.ERROR,
            message=message,
            **kwargs
        )


class PaginatedResponse(APIResponse[List[T]]):
    """Réponse paginée"""
    
    def __init__(
        self,
        items: List[T],
        page: int,
        per_page: int,
        total_items: int,
        message: str = None,
        **kwargs
    ):
        pagination = PaginationInfo.create(page, per_page, total_items)
        
        super().__init__(
            data=items,
            status=ResponseStatus.SUCCESS,
            message=message or f"Retrieved {len(items)} items",
            pagination=pagination,
            **kwargs
        )


# =============================================================================
# FONCTIONS UTILITAIRES DE CRÉATION
# =============================================================================

def create_success_response(
    data: Any,
    message: str = None,
    **kwargs
) -> SuccessResponse:
    """Crée une réponse de succès"""
    return SuccessResponse(data, message, **kwargs)


def create_error_response(
    message: str,
    error_code: str = None,
    error_details: Dict[str, Any] = None,
    **kwargs
) -> ErrorResponse:
    """Crée une réponse d'erreur"""
    return ErrorResponse(message, error_code, error_details, **kwargs)


def create_paginated_response(
    items: List[Any],
    page: int,
    per_page: int,
    total_items: int,
    message: str = None,
    **kwargs
) -> PaginatedResponse:
    """Crée une réponse paginée"""
    return PaginatedResponse(items, page, per_page, total_items, message, **kwargs)


def create_empty_response(message: str = "No content") -> SuccessResponse:
    """Crée une réponse vide"""
    return SuccessResponse(None, message)


def create_created_response(data: Any, message: str = None) -> SuccessResponse:
    """Crée une réponse de création"""
    return SuccessResponse(
        data,
        message or "Resource created successfully"
    )


def create_updated_response(data: Any, message: str = None) -> SuccessResponse:
    """Crée une réponse de mise à jour"""
    return SuccessResponse(
        data,
        message or "Resource updated successfully"
    )


def create_deleted_response(message: str = None) -> SuccessResponse:
    """Crée une réponse de suppression"""
    return SuccessResponse(
        None,
        message or "Resource deleted successfully"
    )


# =============================================================================
# DÉCORATEURS POUR ENRICHISSEMENT AUTOMATIQUE
# =============================================================================

def enrich_response_metadata(func):
    """Décorateur pour enrichir automatiquement les métadonnées"""
    def wrapper(*args, **kwargs):
        from .context import get_request_context
        import time
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Enrichir les métadonnées si c'est une APIResponse
            if isinstance(result, APIResponse):
                context = get_request_context()
                if context:
                    result.set_metadata(
                        request_id=context.request_id,
                        correlation_id=context.correlation_id,
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
            
            return result
            
        except Exception as e:
            # En cas d'erreur, retourner une ErrorResponse enrichie
            context = get_request_context()
            error_response = create_error_response(
                message=str(e),
                error_code=type(e).__name__
            )
            
            if context:
                error_response.set_metadata(
                    request_id=context.request_id,
                    correlation_id=context.correlation_id,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            raise e
    
    return wrapper


# =============================================================================
# FORMATTERS POUR DIFFÉRENTS FORMATS
# =============================================================================

class ResponseFormatter:
    """Formateur de réponses pour différents formats"""
    
    @staticmethod
    def to_json(response: APIResponse) -> str:
        """Formate en JSON"""
        return json.dumps(response.to_dict(), default=str, indent=2)
    
    @staticmethod
    def to_xml(response: APIResponse) -> str:
        """Formate en XML (basique)"""
        # Implementation basique, peut être améliorée
        data = response.to_dict()
        xml = "<response>\n"
        for key, value in data.items():
            xml += f"  <{key}>{value}</{key}>\n"
        xml += "</response>"
        return xml
    
    @staticmethod
    def to_csv(response: APIResponse) -> str:
        """Formate en CSV (pour données tabulaires)"""
        if not isinstance(response.data, list):
            raise ValueError("CSV format requires list data")
        
        import csv
        import io
        
        output = io.StringIO()
        if response.data and len(response.data) > 0:
            if isinstance(response.data[0], dict):
                fieldnames = response.data[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(response.data)
        
        return output.getvalue()


# =============================================================================
# HELPERS POUR CONTEXTE DE REQUÊTE
# =============================================================================

def get_enriched_metadata() -> ResponseMetadata:
    """Retourne des métadonnées enrichies avec le contexte actuel"""
    from .context import get_request_context
    
    context = get_request_context()
    
    metadata = ResponseMetadata(
        timestamp=datetime.utcnow().isoformat()
    )
    
    if context:
        metadata.request_id = context.request_id
        metadata.correlation_id = context.correlation_id
        if context.performance.duration_ms:
            metadata.execution_time_ms = context.performance.duration_ms
    
    return metadata


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ResponseStatus",
    "ResponseFormat",
    "ResponseMetadata",
    "PaginationInfo",
    "APIResponse",
    "SuccessResponse",
    "ErrorResponse", 
    "PaginatedResponse",
    "create_success_response",
    "create_error_response",
    "create_paginated_response",
    "create_empty_response",
    "create_created_response",
    "create_updated_response",
    "create_deleted_response",
    "enrich_response_metadata",
    "ResponseFormatter",
    "get_enriched_metadata",
    "validate_response_data",
    "ResponseBuilder",
    "ResponseMimeType"
]


# =============================================================================
# FONCTIONS UTILITAIRES COMPLÉMENTAIRES  
# =============================================================================

def validate_response_data(data: Any, response_type: type = None) -> bool:
    """Valider les données de réponse"""
    if data is None:
        return True
    
    if response_type and isinstance(data, BaseModel):
        try:
            # Valider avec Pydantic
            response_type.model_validate(data.model_dump())
            return True
        except Exception:
            return False
    
    # Validation basique pour types primitifs
    try:
        jsonable_encoder(data)
        return True
    except Exception:
        return False


class ResponseBuilder:
    """Builder pattern pour créer des réponses API complexes"""
    
    def __init__(self):
        self.data = None
        self.status_code = 200
        self.headers = {}
        self.metadata = ResponseMetadata()
        self.error_details = None
    
    def with_data(self, data: Any):
        """Ajouter des données à la réponse"""
        self.data = data
        return self
    
    def with_status(self, status_code: int):
        """Définir le code de statut"""
        self.status_code = status_code
        return self
    
    def with_headers(self, headers: Dict[str, str]):
        """Ajouter des headers"""
        self.headers.update(headers)
        return self
    
    def with_metadata(self, metadata: ResponseMetadata):
        """Ajouter des métadonnées"""
        self.metadata = metadata
        return self
    
    def with_error(self, error_details: Dict[str, Any]):
        """Ajouter des détails d'erreur"""
        self.error_details = error_details
        return self
    
    def build(self) -> APIResponse:
        """Construire la réponse finale"""
        if self.error_details:
            return ErrorResponse(
                error=self.error_details,
                metadata=self.metadata
            )
        
        return SuccessResponse(
            data=self.data,
            metadata=self.metadata
        )


class ResponseMimeType(str, Enum):
    """Types MIME supportés pour les réponses"""
    JSON = "application/json"
    XML = "application/xml"
    CSV = "text/csv"
    HTML = "text/html"
    PLAIN = "text/plain"
    BINARY = "application/octet-stream"
