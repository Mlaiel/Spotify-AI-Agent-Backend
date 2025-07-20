"""
🎵 Spotify AI Agent - File Management Utilities
===============================================

Utilitaires enterprise pour la gestion avancée des fichiers
avec upload sécurisé, compression et métadonnées.

Architecture:
- Upload et validation de fichiers
- Compression et décompression
- Gestion des métadonnées
- Manipulation de chemins sécurisée
- Détection de types MIME
- Streaming et chunking

🎖️ Développé par l'équipe d'experts enterprise
"""

import os
import shutil
import mimetypes
import hashlib
import magic
import zipfile
import tarfile
import gzip
import bz2
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, BinaryIO, Iterator, Tuple
from PIL import Image, ExifTags
import mutagen
from mutagen.id3 import ID3NoHeaderError


# =============================================================================
# VALIDATION ET SÉCURITÉ
# =============================================================================

class FileValidator:
    """Validateur de fichiers avec contrôles de sécurité"""
    
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
        '.jar', '.ps1', '.sh', '.php', '.asp', '.aspx', '.jsp'
    }
    
    AUDIO_EXTENSIONS = {
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'
    }
    
    IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'
    }
    
    DOCUMENT_EXTENSIONS = {
        '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'
    }
    
    def __init__(self, max_size: int = 100 * 1024 * 1024):  # 100MB par défaut
        self.max_size = max_size
    
    def is_safe_filename(self, filename: str) -> bool:
        """
        Vérifie si un nom de fichier est sécurisé
        
        Args:
            filename: Nom du fichier
            
        Returns:
            True si sécurisé
        """
        # Caractères interdits
        forbidden_chars = ['/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', '.', ' ']
        if any(char in filename for char in forbidden_chars):
            return False
        
        # Extensions dangereuses
        file_ext = Path(filename).suffix.lower()
        if file_ext in self.DANGEROUS_EXTENSIONS:
            return False
        
        # Noms réservés Windows
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
            'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
            'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        if filename.upper() in reserved_names:
            return False
        
        return True
    
    def validate_file_size(self, file_path: Union[str, Path]) -> bool:
        """
        Valide la taille d'un fichier
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            True si la taille est acceptable
        """
        try:
            size = os.path.getsize(file_path)
            return size <= self.max_size
        except OSError:
            return False
    
    def get_file_type(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Détecte le type MIME d'un fichier
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Type MIME ou None
        """
        try:
            # Utiliser python-magic pour une détection précise
            mime_type = magic.from_file(str(file_path), mime=True)
            return mime_type
        except Exception:
            # Fallback sur mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type
    
    def is_audio_file(self, file_path: Union[str, Path]) -> bool:
        """
        Vérifie si un fichier est audio
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            True si audio
        """
        mime_type = self.get_file_type(file_path)
        if mime_type and mime_type.startswith('audio/'):
            return True
        
        extension = Path(file_path).suffix.lower()
        return extension in self.AUDIO_EXTENSIONS
    
    def is_image_file(self, file_path: Union[str, Path]) -> bool:
        """
        Vérifie si un fichier est une image
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            True si image
        """
        mime_type = self.get_file_type(file_path)
        if mime_type and mime_type.startswith('image/'):
            return True
        
        extension = Path(file_path).suffix.lower()
        return extension in self.IMAGE_EXTENSIONS


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """
    Assainit un nom de fichier
    
    Args:
        filename: Nom original
        replacement: Caractère de remplacement
        
    Returns:
        Nom assaini
    """
    # Supprimer les caractères dangereux
    forbidden_chars = ['/', '\\', '?', '%', '*', ':', '|', '"', '<', '>']
    sanitized = filename
    
    for char in forbidden_chars:
        sanitized = sanitized.replace(char, replacement)
    
    # Limiter la longueur
    name, ext = os.path.splitext(sanitized)
    if len(name) > 200:
        name = name[:200]
    
    sanitized = name + ext
    
    # Éviter les noms vides
    if not sanitized or sanitized == ext:
        sanitized = f"file{ext}"
    
    return sanitized


def generate_safe_path(base_dir: Union[str, Path], filename: str) -> Path:
    """
    Génère un chemin sécurisé pour un fichier
    
    Args:
        base_dir: Répertoire de base
        filename: Nom du fichier
        
    Returns:
        Chemin sécurisé
    """
    base_path = Path(base_dir)
    safe_filename = sanitize_filename(filename)
    
    # Éviter les collisions
    counter = 1
    original_name, ext = os.path.splitext(safe_filename)
    final_path = base_path / safe_filename
    
    while final_path.exists():
        new_filename = f"{original_name}_{counter}{ext}"
        final_path = base_path / new_filename
        counter += 1
    
    return final_path


# =============================================================================
# GESTION DES UPLOADS
# =============================================================================

class FileUploadManager:
    """Gestionnaire d'uploads de fichiers sécurisé"""
    
    def __init__(self, upload_dir: Union[str, Path], max_size: int = 100 * 1024 * 1024):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.validator = FileValidator(max_size)
    
    def save_uploaded_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Sauvegarde un fichier uploadé
        
        Args:
            file_data: Données du fichier
            filename: Nom du fichier
            
        Returns:
            Informations sur le fichier sauvé
        """
        # Validation du nom
        if not self.validator.is_safe_filename(filename):
            raise ValueError(f"Nom de fichier non sécurisé: {filename}")
        
        # Génération du chemin sécurisé
        file_path = generate_safe_path(self.upload_dir, filename)
        
        # Vérification de la taille
        if len(file_data) > self.validator.max_size:
            raise ValueError(f"Fichier trop volumineux: {len(file_data)} bytes")
        
        # Sauvegarde
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        # Validation post-sauvegarde
        if not self.validator.validate_file_size(file_path):
            file_path.unlink()  # Supprimer si invalide
            raise ValueError("Fichier invalide après sauvegarde")
        
        # Informations du fichier
        return {
            'path': str(file_path),
            'filename': file_path.name,
            'size': len(file_data),
            'mime_type': self.validator.get_file_type(file_path),
            'hash': calculate_file_hash(file_path),
            'is_audio': self.validator.is_audio_file(file_path),
            'is_image': self.validator.is_image_file(file_path)
        }
    
    def save_uploaded_stream(self, file_stream: BinaryIO, filename: str, 
                           chunk_size: int = 8192) -> Dict[str, Any]:
        """
        Sauvegarde un stream de fichier
        
        Args:
            file_stream: Stream du fichier
            filename: Nom du fichier
            chunk_size: Taille des chunks
            
        Returns:
            Informations sur le fichier sauvé
        """
        # Validation du nom
        if not self.validator.is_safe_filename(filename):
            raise ValueError(f"Nom de fichier non sécurisé: {filename}")
        
        # Génération du chemin sécurisé
        file_path = generate_safe_path(self.upload_dir, filename)
        
        # Sauvegarde par chunks
        total_size = 0
        hash_obj = hashlib.sha256()
        
        with open(file_path, 'wb') as f:
            while True:
                chunk = file_stream.read(chunk_size)
                if not chunk:
                    break
                
                total_size += len(chunk)
                if total_size > self.validator.max_size:
                    file_path.unlink()  # Supprimer si trop volumineux
                    raise ValueError(f"Fichier trop volumineux: {total_size} bytes")
                
                f.write(chunk)
                hash_obj.update(chunk)
        
        # Informations du fichier
        return {
            'path': str(file_path),
            'filename': file_path.name,
            'size': total_size,
            'mime_type': self.validator.get_file_type(file_path),
            'hash': hash_obj.hexdigest(),
            'is_audio': self.validator.is_audio_file(file_path),
            'is_image': self.validator.is_image_file(file_path)
        }


# =============================================================================
# MÉTADONNÉES ET HASH
# =============================================================================

def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Calcule le hash d'un fichier
    
    Args:
        file_path: Chemin du fichier
        algorithm: Algorithme de hash
        
    Returns:
        Hash hexadécimal
    """
    hash_obj = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extrait les métadonnées d'un fichier
    
    Args:
        file_path: Chemin du fichier
        
    Returns:
        Dictionnaire des métadonnées
    """
    path = Path(file_path)
    stat = path.stat()
    
    metadata = {
        'filename': path.name,
        'size': stat.st_size,
        'created': stat.st_ctime,
        'modified': stat.st_mtime,
        'extension': path.suffix,
        'mime_type': mimetypes.guess_type(str(path))[0],
        'hash': calculate_file_hash(path)
    }
    
    # Métadonnées spécifiques aux images
    validator = FileValidator()
    if validator.is_image_file(path):
        try:
            with Image.open(path) as img:
                metadata.update({
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode
                })
                
                # EXIF data
                if hasattr(img, '_getexif'):
                    exif = img._getexif()
                    if exif:
                        exif_data = {}
                        for tag_id, value in exif.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            exif_data[tag] = value
                        metadata['exif'] = exif_data
        except Exception:
            pass
    
    # Métadonnées spécifiques à l'audio
    elif validator.is_audio_file(path):
        try:
            audio_file = mutagen.File(str(path))
            if audio_file:
                metadata.update({
                    'duration': getattr(audio_file.info, 'length', 0),
                    'bitrate': getattr(audio_file.info, 'bitrate', 0),
                    'channels': getattr(audio_file.info, 'channels', 0),
                    'sample_rate': getattr(audio_file.info, 'sample_rate', 0)
                })
                
                # Tags audio
                if audio_file.tags:
                    audio_tags = {}
                    for key, value in audio_file.tags.items():
                        if isinstance(value, list) and len(value) == 1:
                            audio_tags[key] = value[0]
                        else:
                            audio_tags[key] = value
                    metadata['tags'] = audio_tags
        except (ID3NoHeaderError, Exception):
            pass
    
    return metadata


# =============================================================================
# COMPRESSION ET ARCHIVES
# =============================================================================

def compress_file(file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None,
                 compression: str = 'gzip') -> Path:
    """
    Compresse un fichier
    
    Args:
        file_path: Fichier à compresser
        output_path: Chemin de sortie
        compression: Type de compression (gzip, bz2)
        
    Returns:
        Chemin du fichier compressé
    """
    source_path = Path(file_path)
    
    if output_path is None:
        if compression == 'gzip':
            output_path = source_path.with_suffix(source_path.suffix + '.gz')
        elif compression == 'bz2':
            output_path = source_path.with_suffix(source_path.suffix + '.bz2')
        else:
            raise ValueError(f"Compression non supportée: {compression}")
    else:
        output_path = Path(output_path)
    
    if compression == 'gzip':
        with open(source_path, 'rb') as src, gzip.open(output_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    elif compression == 'bz2':
        with open(source_path, 'rb') as src, bz2.BZ2File(output_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    else:
        raise ValueError(f"Compression non supportée: {compression}")
    
    return output_path


def decompress_file(compressed_path: Union[str, Path], 
                   output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Décompresse un fichier
    
    Args:
        compressed_path: Fichier compressé
        output_path: Chemin de sortie
        
    Returns:
        Chemin du fichier décompressé
    """
    source_path = Path(compressed_path)
    
    if output_path is None:
        if source_path.suffix == '.gz':
            output_path = source_path.with_suffix('')
        elif source_path.suffix == '.bz2':
            output_path = source_path.with_suffix('')
        else:
            output_path = source_path.with_suffix('.decompressed')
    else:
        output_path = Path(output_path)
    
    if source_path.suffix == '.gz':
        with gzip.open(source_path, 'rb') as src, open(output_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    elif source_path.suffix == '.bz2':
        with bz2.BZ2File(source_path, 'rb') as src, open(output_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    else:
        raise ValueError(f"Format non supporté: {source_path.suffix}")
    
    return output_path


def create_archive(files: List[Union[str, Path]], archive_path: Union[str, Path],
                  format: str = 'zip') -> Path:
    """
    Crée une archive
    
    Args:
        files: Liste des fichiers à archiver
        archive_path: Chemin de l'archive
        format: Format (zip, tar, tar.gz, tar.bz2)
        
    Returns:
        Chemin de l'archive créée
    """
    archive_path = Path(archive_path)
    
    if format == 'zip':
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                file_path = Path(file_path)
                if file_path.is_file():
                    zf.write(file_path, file_path.name)
    
    elif format in ['tar', 'tar.gz', 'tar.bz2']:
        mode = 'w'
        if format == 'tar.gz':
            mode = 'w:gz'
        elif format == 'tar.bz2':
            mode = 'w:bz2'
        
        with tarfile.open(archive_path, mode) as tf:
            for file_path in files:
                file_path = Path(file_path)
                if file_path.is_file():
                    tf.add(file_path, file_path.name)
    
    else:
        raise ValueError(f"Format non supporté: {format}")
    
    return archive_path


def extract_archive(archive_path: Union[str, Path], 
                   extract_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Extrait une archive
    
    Args:
        archive_path: Chemin de l'archive
        extract_dir: Répertoire d'extraction
        
    Returns:
        Répertoire d'extraction
    """
    archive_path = Path(archive_path)
    
    if extract_dir is None:
        extract_dir = archive_path.parent / archive_path.stem
    else:
        extract_dir = Path(extract_dir)
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_dir)
    
    elif archive_path.suffix in ['.tar', '.gz', '.bz2'] or '.tar.' in archive_path.name:
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_dir)
    
    else:
        raise ValueError(f"Format non supporté: {archive_path.suffix}")
    
    return extract_dir


# =============================================================================
# STREAMING ET CHUNKING
# =============================================================================

def read_file_chunks(file_path: Union[str, Path], chunk_size: int = 8192) -> Iterator[bytes]:
    """
    Lit un fichier par chunks
    
    Args:
        file_path: Chemin du fichier
        chunk_size: Taille des chunks
        
    Yields:
        Chunks de données
    """
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk


def copy_file_chunked(source: Union[str, Path], destination: Union[str, Path],
                     chunk_size: int = 8192) -> None:
    """
    Copie un fichier par chunks
    
    Args:
        source: Fichier source
        destination: Fichier destination
        chunk_size: Taille des chunks
    """
    with open(source, 'rb') as src, open(destination, 'wb') as dst:
        while chunk := src.read(chunk_size):
            dst.write(chunk)


def split_file(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> List[Path]:
    """
    Divise un fichier en plusieurs parties
    
    Args:
        file_path: Fichier à diviser
        chunk_size: Taille de chaque partie
        
    Returns:
        Liste des fichiers créés
    """
    source_path = Path(file_path)
    chunks = []
    
    with open(source_path, 'rb') as f:
        chunk_number = 0
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            chunk_path = source_path.with_suffix(f'.part{chunk_number:03d}')
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk)
            
            chunks.append(chunk_path)
            chunk_number += 1
    
    return chunks


def join_file_parts(parts: List[Union[str, Path]], output_path: Union[str, Path]) -> Path:
    """
    Rejoint des parties de fichier
    
    Args:
        parts: Liste des parties
        output_path: Fichier de sortie
        
    Returns:
        Chemin du fichier reconstitué
    """
    output_path = Path(output_path)
    
    with open(output_path, 'wb') as output_file:
        for part_path in sorted(parts):
            with open(part_path, 'rb') as part_file:
                shutil.copyfileobj(part_file, output_file)
    
    return output_path


# =============================================================================
# UTILITAIRES DIVERS
# =============================================================================

def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Calcule la taille totale d'un répertoire
    
    Args:
        directory: Répertoire à analyser
        
    Returns:
        Taille en bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass
    return total_size


def clean_directory(directory: Union[str, Path], max_age_days: int = 30) -> int:
    """
    Nettoie un répertoire des fichiers anciens
    
    Args:
        directory: Répertoire à nettoyer
        max_age_days: Âge maximum en jours
        
    Returns:
        Nombre de fichiers supprimés
    """
    import time
    
    directory = Path(directory)
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    deleted_count = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
            except OSError:
                pass
    
    return deleted_count


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    S'assure qu'un répertoire existe
    
    Args:
        directory: Répertoire à créer
        
    Returns:
        Chemin du répertoire
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FileValidator",
    "sanitize_filename",
    "generate_safe_path",
    "FileUploadManager",
    "calculate_file_hash",
    "get_file_metadata",
    "compress_file",
    "decompress_file",
    "create_archive",
    "extract_archive",
    "read_file_chunks",
    "copy_file_chunked",
    "split_file",
    "join_file_parts",
    "get_directory_size",
    "clean_directory",
    "ensure_directory"
]
