"""
Advanced Asset Manager

This module provides sophisticated asset management for email templates including
image optimization, CDN integration, lazy loading, responsive images, and
automated asset processing.

Version: 3.0.0
Developed by Spotify AI Agent Team
"""

import re
import base64
import hashlib
import asyncio
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
import structlog
import aiofiles
import aiohttp
from PIL import Image, ImageOps, ImageFilter
import io
import json

logger = structlog.get_logger(__name__)

# ============================================================================
# Asset Configuration Classes
# ============================================================================

class AssetType(Enum):
    """Types d'assets supportés"""
    IMAGE = "image"
    ICON = "icon"
    FONT = "font"
    CSS = "css"
    JAVASCRIPT = "javascript"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"

class ImageFormat(Enum):
    """Formats d'images supportés"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    SVG = "svg"
    ICO = "ico"

class CompressionLevel(Enum):
    """Niveaux de compression"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

@dataclass
class AssetMetadata:
    """Métadonnées d'un asset"""
    filename: str
    asset_type: AssetType
    size: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    mime_type: Optional[str] = None
    hash: Optional[str] = None
    created_at: Optional[datetime] = None
    optimized: bool = False
    cdn_url: Optional[str] = None
    alt_text: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class ImageOptimizationConfig:
    """Configuration d'optimisation d'images"""
    max_width: int = 600
    max_height: int = 800
    quality: int = 85
    format: ImageFormat = ImageFormat.JPEG
    compression: CompressionLevel = CompressionLevel.MEDIUM
    progressive: bool = True
    preserve_metadata: bool = False
    auto_orient: bool = True

@dataclass
class ResponsiveImageConfig:
    """Configuration d'images responsive"""
    breakpoints: List[int] = field(default_factory=lambda: [320, 480, 768, 1024])
    formats: List[ImageFormat] = field(default_factory=lambda: [ImageFormat.WEBP, ImageFormat.JPEG])
    quality_levels: Dict[int, int] = field(default_factory=lambda: {320: 75, 480: 80, 768: 85, 1024: 90})

@dataclass
class CDNConfig:
    """Configuration CDN"""
    provider: str
    base_url: str
    api_key: Optional[str] = None
    transformation_endpoint: Optional[str] = None
    cache_ttl: int = 86400  # 24 heures
    enable_compression: bool = True
    enable_webp: bool = True

# ============================================================================
# Advanced Asset Manager
# ============================================================================

class AdvancedAssetManager:
    """Gestionnaire d'assets avancé pour emails"""
    
    def __init__(self,
                 assets_dir: str,
                 cdn_config: Optional[CDNConfig] = None,
                 enable_optimization: bool = True,
                 enable_responsive: bool = True):
        
        self.assets_dir = Path(assets_dir)
        self.cdn_config = cdn_config
        self.enable_optimization = enable_optimization
        self.enable_responsive = enable_responsive
        
        # Cache et métadonnées
        self.asset_cache: Dict[str, AssetMetadata] = {}
        self.metadata_file = self.assets_dir / "metadata.json"
        
        # Configuration par défaut
        self.optimization_config = ImageOptimizationConfig()
        self.responsive_config = ResponsiveImageConfig()
        
        # Formats supportés
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico'}
        self.supported_font_formats = {'.woff', '.woff2', '.ttf', '.otf', '.eot'}
        self.supported_video_formats = {'.mp4', '.webm', '.ogg'}
        
        # Session HTTP pour CDN
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Initialize
        asyncio.create_task(self._initialize())
        
        logger.info("Advanced Asset Manager initialized")
    
    async def _initialize(self):
        """Initialisation du gestionnaire"""
        
        # Création des répertoires
        await self._ensure_directories()
        
        # Chargement des métadonnées
        await self._load_metadata()
        
        # Initialisation de la session HTTP
        if self.cdn_config:
            self.http_session = aiohttp.ClientSession()
        
        # Scan initial des assets
        await self._scan_assets()
        
        logger.info("Asset Manager initialization completed")
    
    async def _ensure_directories(self):
        """Assure que les répertoires nécessaires existent"""
        
        directories = [
            self.assets_dir,
            self.assets_dir / "images",
            self.assets_dir / "images" / "original",
            self.assets_dir / "images" / "optimized",
            self.assets_dir / "images" / "responsive",
            self.assets_dir / "images" / "thumbnails",
            self.assets_dir / "icons",
            self.assets_dir / "fonts",
            self.assets_dir / "css",
            self.assets_dir / "js",
            self.assets_dir / "documents",
            self.assets_dir / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_metadata(self):
        """Charge les métadonnées des assets"""
        
        if self.metadata_file.exists():
            try:
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    data = json.loads(await f.read())
                    
                for filename, metadata_dict in data.items():
                    # Reconstruction des objets AssetMetadata
                    metadata_dict['asset_type'] = AssetType(metadata_dict['asset_type'])
                    if metadata_dict.get('created_at'):
                        metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                    
                    self.asset_cache[filename] = AssetMetadata(**metadata_dict)
                
                logger.info(f"Loaded metadata for {len(self.asset_cache)} assets")
                
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.asset_cache = {}
    
    async def _save_metadata(self):
        """Sauvegarde les métadonnées"""
        
        try:
            # Sérialisation des métadonnées
            data = {}
            for filename, metadata in self.asset_cache.items():
                metadata_dict = {
                    'filename': metadata.filename,
                    'asset_type': metadata.asset_type.value,
                    'size': metadata.size,
                    'width': metadata.width,
                    'height': metadata.height,
                    'format': metadata.format,
                    'mime_type': metadata.mime_type,
                    'hash': metadata.hash,
                    'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
                    'optimized': metadata.optimized,
                    'cdn_url': metadata.cdn_url,
                    'alt_text': metadata.alt_text,
                    'tags': metadata.tags
                }
                data[filename] = metadata_dict
            
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def _scan_assets(self):
        """Scan initial des assets"""
        
        # Scan de tous les répertoires
        for asset_type_dir in self.assets_dir.iterdir():
            if asset_type_dir.is_dir() and asset_type_dir.name != "cache":
                await self._scan_directory(asset_type_dir)
        
        # Sauvegarde des métadonnées
        await self._save_metadata()
    
    async def _scan_directory(self, directory: Path):
        """Scan un répertoire spécifique"""
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.assets_dir)
                
                if str(relative_path) not in self.asset_cache:
                    await self._analyze_asset(file_path)
    
    async def _analyze_asset(self, file_path: Path) -> AssetMetadata:
        """Analyse un asset et crée ses métadonnées"""
        
        try:
            # Informations de base
            stat = file_path.stat()
            relative_path = file_path.relative_to(self.assets_dir)
            
            # Calcul du hash
            file_hash = await self._calculate_file_hash(file_path)
            
            # Détermination du type d'asset
            asset_type = self._determine_asset_type(file_path)
            
            # Informations spécifiques aux images
            width, height, image_format = None, None, None
            if asset_type == AssetType.IMAGE:
                width, height, image_format = await self._get_image_info(file_path)
            
            # Type MIME
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Création des métadonnées
            metadata = AssetMetadata(
                filename=str(relative_path),
                asset_type=asset_type,
                size=stat.st_size,
                width=width,
                height=height,
                format=image_format,
                mime_type=mime_type,
                hash=file_hash,
                created_at=datetime.fromtimestamp(stat.st_mtime),
                optimized=False
            )
            
            self.asset_cache[str(relative_path)] = metadata
            
            logger.debug(f"Analyzed asset: {relative_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze asset {file_path}: {e}")
            raise
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcule le hash SHA-256 d'un fichier"""
        
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _determine_asset_type(self, file_path: Path) -> AssetType:
        """Détermine le type d'asset"""
        
        suffix = file_path.suffix.lower()
        
        if suffix in self.supported_image_formats:
            return AssetType.IMAGE
        elif suffix in self.supported_font_formats:
            return AssetType.FONT
        elif suffix == '.css':
            return AssetType.CSS
        elif suffix in {'.js', '.mjs'}:
            return AssetType.JAVASCRIPT
        elif suffix in self.supported_video_formats:
            return AssetType.VIDEO
        elif suffix in {'.mp3', '.wav', '.ogg'}:
            return AssetType.AUDIO
        elif suffix in {'.pdf', '.doc', '.docx', '.txt'}:
            return AssetType.DOCUMENT
        else:
            return AssetType.ICON if 'icon' in str(file_path) else AssetType.IMAGE
    
    async def _get_image_info(self, file_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """Obtient les informations d'une image"""
        
        try:
            if file_path.suffix.lower() == '.svg':
                # Pour SVG, parse le XML pour obtenir les dimensions
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    
                width_match = re.search(r'width=["\']([\d.]+)', content)
                height_match = re.search(r'height=["\']([\d.]+)', content)
                
                width = int(float(width_match.group(1))) if width_match else None
                height = int(float(height_match.group(1))) if height_match else None
                
                return width, height, "svg"
            
            else:
                # Pour les autres formats, utilise PIL
                with Image.open(file_path) as img:
                    return img.width, img.height, img.format.lower()
                    
        except Exception as e:
            logger.warning(f"Failed to get image info for {file_path}: {e}")
            return None, None, None
    
    async def add_asset(self,
                       file_content: bytes,
                       filename: str,
                       asset_type: Optional[AssetType] = None,
                       tags: Optional[List[str]] = None,
                       alt_text: Optional[str] = None) -> AssetMetadata:
        """Ajoute un nouvel asset"""
        
        file_path = self.assets_dir / filename
        
        # Création du répertoire parent si nécessaire
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Écriture du fichier
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        # Analyse de l'asset
        metadata = await self._analyze_asset(file_path)
        
        # Ajout d'informations supplémentaires
        if tags:
            metadata.tags = tags
        if alt_text:
            metadata.alt_text = alt_text
        
        # Optimisation automatique pour les images
        if metadata.asset_type == AssetType.IMAGE and self.enable_optimization:
            await self.optimize_image(filename)
        
        # Génération d'images responsive
        if metadata.asset_type == AssetType.IMAGE and self.enable_responsive:
            await self.generate_responsive_images(filename)
        
        # Upload vers CDN si configuré
        if self.cdn_config:
            await self.upload_to_cdn(filename)
        
        # Sauvegarde des métadonnées
        await self._save_metadata()
        
        logger.info(f"Added asset: {filename}")
        return metadata
    
    async def optimize_image(self,
                           filename: str,
                           config: Optional[ImageOptimizationConfig] = None) -> str:
        """Optimise une image"""
        
        config = config or self.optimization_config
        original_path = self.assets_dir / filename
        optimized_path = self.assets_dir / "images" / "optimized" / Path(filename).name
        
        if not original_path.exists():
            raise FileNotFoundError(f"Asset not found: {filename}")
        
        try:
            # Optimisation différente selon le format
            if original_path.suffix.lower() == '.svg':
                # Pour SVG, simple copie avec nettoyage
                await self._optimize_svg(original_path, optimized_path)
            else:
                # Pour les autres formats, utilise PIL
                await self._optimize_raster_image(original_path, optimized_path, config)
            
            # Mise à jour des métadonnées
            if filename in self.asset_cache:
                self.asset_cache[filename].optimized = True
            
            logger.info(f"Optimized image: {filename}")
            return str(optimized_path.relative_to(self.assets_dir))
            
        except Exception as e:
            logger.error(f"Failed to optimize image {filename}: {e}")
            raise
    
    async def _optimize_svg(self, input_path: Path, output_path: Path):
        """Optimise un fichier SVG"""
        
        async with aiofiles.open(input_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Nettoyage basique du SVG
        # Suppression des commentaires
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Suppression des espaces inutiles
        content = re.sub(r'>\s+<', '><', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Suppression des métadonnées
        content = re.sub(r'<metadata>.*?</metadata>', '', content, flags=re.DOTALL)
        content = re.sub(r'<title>.*?</title>', '', content, flags=re.DOTALL)
        content = re.sub(r'<desc>.*?</desc>', '', content, flags=re.DOTALL)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    async def _optimize_raster_image(self,
                                   input_path: Path,
                                   output_path: Path,
                                   config: ImageOptimizationConfig):
        """Optimise une image raster"""
        
        with Image.open(input_path) as img:
            # Auto-orientation
            if config.auto_orient:
                img = ImageOps.exif_transpose(img)
            
            # Redimensionnement si nécessaire
            if img.width > config.max_width or img.height > config.max_height:
                img.thumbnail((config.max_width, config.max_height), Image.Resampling.LANCZOS)
            
            # Conversion du mode si nécessaire
            if config.format == ImageFormat.JPEG and img.mode in ('RGBA', 'LA', 'P'):
                # Création d'un fond blanc pour JPEG
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            # Configuration de sauvegarde
            save_kwargs = {
                'format': config.format.value.upper(),
                'quality': config.quality,
                'optimize': True
            }
            
            if config.format == ImageFormat.JPEG:
                save_kwargs['progressive'] = config.progressive
            elif config.format == ImageFormat.PNG:
                save_kwargs['compress_level'] = 9
            elif config.format == ImageFormat.WEBP:
                save_kwargs['method'] = 6  # Meilleure compression
            
            # Suppression des métadonnées EXIF si demandé
            if not config.preserve_metadata:
                save_kwargs['exif'] = b''
            
            # Sauvegarde
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, **save_kwargs)
    
    async def generate_responsive_images(self,
                                       filename: str,
                                       config: Optional[ResponsiveImageConfig] = None) -> List[str]:
        """Génère des versions responsive d'une image"""
        
        config = config or self.responsive_config
        original_path = self.assets_dir / filename
        
        if not original_path.exists():
            raise FileNotFoundError(f"Asset not found: {filename}")
        
        if original_path.suffix.lower() == '.svg':
            # SVG est déjà responsive
            return [filename]
        
        responsive_images = []
        base_name = Path(filename).stem
        
        try:
            with Image.open(original_path) as img:
                for breakpoint in config.breakpoints:
                    if img.width <= breakpoint:
                        continue
                    
                    for format_type in config.formats:
                        # Nom du fichier responsive
                        responsive_filename = f"{base_name}_{breakpoint}w.{format_type.value}"
                        responsive_path = self.assets_dir / "images" / "responsive" / responsive_filename
                        
                        # Redimensionnement
                        responsive_img = img.copy()
                        aspect_ratio = img.height / img.width
                        new_height = int(breakpoint * aspect_ratio)
                        responsive_img = responsive_img.resize((breakpoint, new_height), Image.Resampling.LANCZOS)
                        
                        # Configuration de qualité
                        quality = config.quality_levels.get(breakpoint, 85)
                        
                        # Sauvegarde
                        save_kwargs = {
                            'format': format_type.value.upper(),
                            'quality': quality,
                            'optimize': True
                        }
                        
                        responsive_path.parent.mkdir(parents=True, exist_ok=True)
                        responsive_img.save(responsive_path, **save_kwargs)
                        
                        responsive_images.append(str(responsive_path.relative_to(self.assets_dir)))
            
            logger.info(f"Generated {len(responsive_images)} responsive images for {filename}")
            return responsive_images
            
        except Exception as e:
            logger.error(f"Failed to generate responsive images for {filename}: {e}")
            raise
    
    async def generate_thumbnail(self,
                               filename: str,
                               size: Tuple[int, int] = (150, 150)) -> str:
        """Génère une miniature"""
        
        original_path = self.assets_dir / filename
        
        if not original_path.exists():
            raise FileNotFoundError(f"Asset not found: {filename}")
        
        base_name = Path(filename).stem
        thumbnail_filename = f"{base_name}_thumb.jpg"
        thumbnail_path = self.assets_dir / "images" / "thumbnails" / thumbnail_filename
        
        try:
            with Image.open(original_path) as img:
                # Création de la miniature avec crop centré
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Conversion en RGB si nécessaire
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(thumbnail_path, 'JPEG', quality=80, optimize=True)
            
            logger.info(f"Generated thumbnail for {filename}")
            return str(thumbnail_path.relative_to(self.assets_dir))
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {filename}: {e}")
            raise
    
    async def upload_to_cdn(self, filename: str) -> Optional[str]:
        """Upload un asset vers le CDN"""
        
        if not self.cdn_config or not self.http_session:
            return None
        
        file_path = self.assets_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Asset not found: {filename}")
        
        try:
            # Lecture du fichier
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            # Détermination du type MIME
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Upload vers le CDN (exemple générique)
            upload_data = aiohttp.FormData()
            upload_data.add_field('file', file_content, filename=file_path.name, content_type=mime_type)
            
            if self.cdn_config.api_key:
                upload_data.add_field('api_key', self.cdn_config.api_key)
            
            async with self.http_session.post(
                f"{self.cdn_config.base_url}/upload",
                data=upload_data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    cdn_url = result.get('url')
                    
                    # Mise à jour des métadonnées
                    if filename in self.asset_cache:
                        self.asset_cache[filename].cdn_url = cdn_url
                    
                    logger.info(f"Uploaded to CDN: {filename} -> {cdn_url}")
                    return cdn_url
                else:
                    logger.error(f"CDN upload failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to upload {filename} to CDN: {e}")
            return None
    
    def get_asset_url(self, filename: str, use_cdn: bool = True) -> str:
        """Obtient l'URL d'un asset"""
        
        if filename in self.asset_cache:
            metadata = self.asset_cache[filename]
            
            # Utilisation du CDN si disponible
            if use_cdn and metadata.cdn_url:
                return metadata.cdn_url
        
        # URL locale par défaut
        return f"/assets/{filename}"
    
    async def get_responsive_image_srcset(self, filename: str) -> str:
        """Génère un srcset pour image responsive"""
        
        if filename not in self.asset_cache:
            return f"/assets/{filename}"
        
        base_name = Path(filename).stem
        responsive_dir = self.assets_dir / "images" / "responsive"
        
        srcset_parts = []
        
        # Recherche des versions responsive
        for responsive_file in responsive_dir.glob(f"{base_name}_*w.*"):
            # Extraction de la largeur
            width_match = re.search(r'(\d+)w\.', responsive_file.name)
            if width_match:
                width = width_match.group(1)
                url = self.get_asset_url(f"images/responsive/{responsive_file.name}")
                srcset_parts.append(f"{url} {width}w")
        
        # Image originale comme fallback
        original_url = self.get_asset_url(filename)
        if self.asset_cache[filename].width:
            srcset_parts.append(f"{original_url} {self.asset_cache[filename].width}w")
        
        return ", ".join(srcset_parts) if srcset_parts else original_url
    
    async def get_image_base64(self, filename: str, max_size: int = 50000) -> Optional[str]:
        """Convertit une image en base64 pour inlining"""
        
        file_path = self.assets_dir / filename
        
        if not file_path.exists():
            return None
        
        # Vérification de la taille
        if file_path.stat().st_size > max_size:
            logger.warning(f"Image {filename} too large for base64 inlining")
            return None
        
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            mime_type, _ = mimetypes.guess_type(str(file_path))
            encoded = base64.b64encode(file_content).decode('utf-8')
            
            return f"data:{mime_type};base64,{encoded}"
            
        except Exception as e:
            logger.error(f"Failed to convert {filename} to base64: {e}")
            return None
    
    async def clean_cache(self, max_age_days: int = 30):
        """Nettoie le cache des assets"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cache_dir = self.assets_dir / "cache"
        
        if cache_dir.exists():
            for cache_file in cache_dir.iterdir():
                if cache_file.is_file():
                    stat = cache_file.stat()
                    if datetime.fromtimestamp(stat.st_mtime) < cutoff_date:
                        cache_file.unlink()
                        logger.debug(f"Removed old cache file: {cache_file.name}")
    
    async def get_asset_stats(self) -> Dict[str, Any]:
        """Obtient les statistiques des assets"""
        
        stats = {
            "total_assets": len(self.asset_cache),
            "total_size": sum(metadata.size for metadata in self.asset_cache.values()),
            "by_type": {},
            "optimized_count": sum(1 for metadata in self.asset_cache.values() if metadata.optimized),
            "cdn_count": sum(1 for metadata in self.asset_cache.values() if metadata.cdn_url)
        }
        
        # Statistiques par type
        for metadata in self.asset_cache.values():
            asset_type = metadata.asset_type.value
            if asset_type not in stats["by_type"]:
                stats["by_type"][asset_type] = {"count": 0, "size": 0}
            
            stats["by_type"][asset_type]["count"] += 1
            stats["by_type"][asset_type]["size"] += metadata.size
        
        return stats
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        if self.http_session:
            await self.http_session.close()
        
        await self._save_metadata()

# ============================================================================
# Factory Functions
# ============================================================================

def create_asset_manager(
    assets_dir: str,
    cdn_provider: Optional[str] = None,
    enable_optimization: bool = True
) -> AdvancedAssetManager:
    """Factory pour créer un gestionnaire d'assets"""
    
    cdn_config = None
    if cdn_provider:
        # Configuration CDN par défaut selon le provider
        cdn_configs = {
            "cloudinary": CDNConfig(
                provider="cloudinary",
                base_url="https://api.cloudinary.com/v1_1",
                enable_compression=True,
                enable_webp=True
            ),
            "aws": CDNConfig(
                provider="aws",
                base_url="https://s3.amazonaws.com",
                enable_compression=True,
                enable_webp=False
            )
        }
        cdn_config = cdn_configs.get(cdn_provider)
    
    return AdvancedAssetManager(
        assets_dir=assets_dir,
        cdn_config=cdn_config,
        enable_optimization=enable_optimization
    )

def create_optimization_config(quality: int = 85, max_width: int = 600) -> ImageOptimizationConfig:
    """Crée une configuration d'optimisation"""
    
    return ImageOptimizationConfig(
        quality=quality,
        max_width=max_width
    )

# Export des classes principales
__all__ = [
    "AdvancedAssetManager",
    "AssetType",
    "ImageFormat",
    "AssetMetadata",
    "ImageOptimizationConfig",
    "ResponsiveImageConfig",
    "CDNConfig",
    "create_asset_manager",
    "create_optimization_config"
]
