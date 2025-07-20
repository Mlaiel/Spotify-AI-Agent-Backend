"""
Storage Services Package

Zentraler Einstiegspunkt für alle Storage-bezogenen Services:
- Abstraktion für lokale, S3- und CDN-Speicher
- Siehe README für Details
"""
from .file_service import FileStorageService
from .local_storage_service import LocalStorageService
from .s3_service import S3StorageService
from .cdn_service import CDNService

__all__ = [
    "FileStorageService",
    "LocalStorageService",
    "S3StorageService",
    "CDNService",
]
