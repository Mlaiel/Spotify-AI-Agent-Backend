"""
Chargeur de Locales Avancé pour Spotify AI Agent
Système de chargement dynamique et intelligent des locales
"""

import asyncio
import json
import yaml
import pickle
import gzip
import logging
from typing import Dict, List, Optional, Any, Set, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
import aiofiles
import aiofiles.os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
from collections import defaultdict
import hashlib
import mimetypes

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration du chargeur de locales"""
    formats: Set[str] = None
    compression: bool = True
    encryption: bool = False
    validation: bool = True
    cache_enabled: bool = True
    lazy_loading: bool = True
    preload_critical: bool = True
    batch_size: int = 50
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    timeout: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = {'json', 'yaml', 'yml', 'po', 'pot', 'xliff', 'csv'}


class LocaleFormat(ABC):
    """Interface pour les formats de locales"""
    
    @abstractmethod
    async def parse(self, content: bytes) -> Dict[str, Any]:
        """Parse le contenu du fichier"""
        pass
    
    @abstractmethod
    async def serialize(self, data: Dict[str, Any]) -> bytes:
        """Sérialise les données"""
        pass
    
    @property
    @abstractmethod
    def extensions(self) -> Set[str]:
        """Extensions de fichiers supportées"""
        pass


class JSONLocaleFormat(LocaleFormat):
    """Format JSON pour les locales"""
    
    async def parse(self, content: bytes) -> Dict[str, Any]:
        """Parse un fichier JSON"""
        try:
            text = content.decode('utf-8')
            return json.loads(text)
        except Exception as e:
            logger.error(f"Error parsing JSON locale: {e}")
            return {}
    
    async def serialize(self, data: Dict[str, Any]) -> bytes:
        """Sérialise en JSON"""
        try:
            text = json.dumps(data, ensure_ascii=False, indent=2)
            return text.encode('utf-8')
        except Exception as e:
            logger.error(f"Error serializing JSON locale: {e}")
            return b'{}'
    
    @property
    def extensions(self) -> Set[str]:
        return {'.json'}


class YAMLLocaleFormat(LocaleFormat):
    """Format YAML pour les locales"""
    
    async def parse(self, content: bytes) -> Dict[str, Any]:
        """Parse un fichier YAML"""
        try:
            text = content.decode('utf-8')
            return yaml.safe_load(text) or {}
        except Exception as e:
            logger.error(f"Error parsing YAML locale: {e}")
            return {}
    
    async def serialize(self, data: Dict[str, Any]) -> bytes:
        """Sérialise en YAML"""
        try:
            text = yaml.dump(data, allow_unicode=True, default_flow_style=False)
            return text.encode('utf-8')
        except Exception as e:
            logger.error(f"Error serializing YAML locale: {e}")
            return b''
    
    @property
    def extensions(self) -> Set[str]:
        return {'.yaml', '.yml'}


class POLocaleFormat(LocaleFormat):
    """Format PO (Gettext) pour les locales"""
    
    async def parse(self, content: bytes) -> Dict[str, Any]:
        """Parse un fichier PO"""
        try:
            text = content.decode('utf-8')
            translations = {}
            current_msgid = None
            current_msgstr = None
            
            for line in text.split('\n'):
                line = line.strip()
                
                if line.startswith('msgid '):
                    current_msgid = line[6:].strip('"')
                elif line.startswith('msgstr '):
                    current_msgstr = line[7:].strip('"')
                    if current_msgid and current_msgstr:
                        translations[current_msgid] = current_msgstr
                elif line.startswith('"') and current_msgstr is not None:
                    # Continuation de msgstr
                    current_msgstr += line.strip('"')
            
            return translations
            
        except Exception as e:
            logger.error(f"Error parsing PO locale: {e}")
            return {}
    
    async def serialize(self, data: Dict[str, Any]) -> bytes:
        """Sérialise en format PO"""
        try:
            lines = [
                '# Generated locale file',
                f'# Generated on {datetime.now().isoformat()}',
                '',
                'msgid ""',
                'msgstr ""',
                '"Content-Type: text/plain; charset=UTF-8\\n"',
                '"Language: \\n"',
                ''
            ]
            
            for msgid, msgstr in data.items():
                lines.extend([
                    f'msgid "{msgid}"',
                    f'msgstr "{msgstr}"',
                    ''
                ])
            
            text = '\n'.join(lines)
            return text.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error serializing PO locale: {e}")
            return b''
    
    @property
    def extensions(self) -> Set[str]:
        return {'.po', '.pot'}


class CSVLocaleFormat(LocaleFormat):
    """Format CSV pour les locales"""
    
    async def parse(self, content: bytes) -> Dict[str, Any]:
        """Parse un fichier CSV"""
        try:
            import csv
            import io
            
            text = content.decode('utf-8')
            reader = csv.DictReader(io.StringIO(text))
            translations = {}
            
            for row in reader:
                if 'key' in row and 'value' in row:
                    translations[row['key']] = row['value']
                elif len(row) >= 2:
                    # Fallback: première colonne = clé, deuxième = valeur
                    key, value = list(row.values())[:2]
                    translations[key] = value
            
            return translations
            
        except Exception as e:
            logger.error(f"Error parsing CSV locale: {e}")
            return {}
    
    async def serialize(self, data: Dict[str, Any]) -> bytes:
        """Sérialise en format CSV"""
        try:
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['key', 'value'])
            
            for key, value in data.items():
                writer.writerow([key, value])
            
            return output.getvalue().encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error serializing CSV locale: {e}")
            return b''
    
    @property
    def extensions(self) -> Set[str]:
        return {'.csv'}


class LocaleLoader:
    """Chargeur de locales avec support multi-format"""
    
    def __init__(self, config: LoaderConfig):
        self.config = config
        self._formats = {
            'json': JSONLocaleFormat(),
            'yaml': YAMLLocaleFormat(),
            'po': POLocaleFormat(),
            'csv': CSVLocaleFormat()
        }
        self._cache = {}
        self._file_cache = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._stats = defaultdict(int)
    
    async def load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Charge une locale depuis un fichier"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"Locale file not found: {file_path}")
                return {}
            
            # Vérifier la taille du fichier
            file_size = await aiofiles.os.path.getsize(file_path)
            if file_size > self.config.max_file_size:
                logger.error(f"File too large: {file_path} ({file_size} bytes)")
                return {}
            
            # Vérifier le cache
            if self.config.cache_enabled:
                cache_key = self._get_file_cache_key(file_path)
                cached_data = await self._get_from_cache(cache_key, file_path)
                if cached_data is not None:
                    self._stats['cache_hits'] += 1
                    return cached_data
            
            self._stats['cache_misses'] += 1
            
            # Charger le fichier
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # Décompresser si nécessaire
            if self._is_compressed(file_path):
                content = await self._decompress(content)
            
            # Décrypter si nécessaire
            if self.config.encryption and self._is_encrypted(file_path):
                content = await self._decrypt(content)
            
            # Parser selon le format
            format_handler = self._get_format_handler(file_path)
            if not format_handler:
                logger.error(f"Unsupported format for file: {file_path}")
                return {}
            
            data = await format_handler.parse(content)
            
            # Valider si activé
            if self.config.validation:
                data = await self._validate_data(data, file_path)
            
            # Mettre en cache
            if self.config.cache_enabled:
                await self._cache_data(cache_key, data, file_path)
            
            self._stats['files_loaded'] += 1
            return data
            
        except Exception as e:
            logger.error(f"Error loading locale file {file_path}: {e}")
            self._stats['load_errors'] += 1
            return {}
    
    async def save_to_file(self, file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """Sauvegarde une locale dans un fichier"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Obtenir le format handler
            format_handler = self._get_format_handler(file_path)
            if not format_handler:
                logger.error(f"Unsupported format for file: {file_path}")
                return False
            
            # Sérialiser les données
            content = await format_handler.serialize(data)
            
            # Crypter si nécessaire
            if self.config.encryption:
                content = await self._encrypt(content)
            
            # Compresser si nécessaire
            if self.config.compression:
                content = await self._compress(content)
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            # Sauvegarder
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Invalider le cache
            if self.config.cache_enabled:
                cache_key = self._get_file_cache_key(file_path)
                await self._invalidate_cache(cache_key)
            
            self._stats['files_saved'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error saving locale file {file_path}: {e}")
            self._stats['save_errors'] += 1
            return False
    
    async def load_from_directory(
        self, 
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Charge toutes les locales d'un répertoire"""
        try:
            directory = Path(directory)
            if not directory.exists():
                logger.warning(f"Directory not found: {directory}")
                return {}
            
            # Trouver tous les fichiers de locale
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            
            # Filtrer par extensions supportées
            locale_files = []
            for file_path in files:
                if self._is_locale_file(file_path):
                    locale_files.append(file_path)
            
            # Charger en parallèle par batches
            results = {}
            for i in range(0, len(locale_files), self.config.batch_size):
                batch = locale_files[i:i + self.config.batch_size]
                batch_tasks = []
                
                for file_path in batch:
                    task = self._load_file_with_timeout(file_path)
                    batch_tasks.append(task)
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for file_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error loading {file_path}: {result}")
                        continue
                    
                    # Utiliser le nom du fichier comme clé
                    locale_key = file_path.stem
                    results[locale_key] = result
            
            self._stats['directories_loaded'] += 1
            return results
            
        except Exception as e:
            logger.error(f"Error loading directory {directory}: {e}")
            return {}
    
    async def get_loader_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du chargeur"""
        return {
            'stats': dict(self._stats),
            'cache_size': len(self._cache),
            'supported_formats': list(self._formats.keys()),
            'config': {
                'formats': list(self.config.formats),
                'compression': self.config.compression,
                'encryption': self.config.encryption,
                'validation': self.config.validation,
                'cache_enabled': self.config.cache_enabled,
                'batch_size': self.config.batch_size
            }
        }
    
    def _get_format_handler(self, file_path: Path) -> Optional[LocaleFormat]:
        """Obtient le handler de format approprié"""
        extension = file_path.suffix.lower()
        
        for format_name, handler in self._formats.items():
            if extension in handler.extensions:
                return handler
        
        return None
    
    def _is_locale_file(self, file_path: Path) -> bool:
        """Vérifie si le fichier est un fichier de locale supporté"""
        extension = file_path.suffix.lower()
        
        for handler in self._formats.values():
            if extension in handler.extensions:
                return True
        
        return False
    
    def _is_compressed(self, file_path: Path) -> bool:
        """Vérifie si le fichier est compressé"""
        return file_path.suffix.lower() in {'.gz', '.bz2', '.xz'}
    
    def _is_encrypted(self, file_path: Path) -> bool:
        """Vérifie si le fichier est crypté"""
        return file_path.suffix.lower() in {'.enc', '.encrypted'}
    
    async def _decompress(self, content: bytes) -> bytes:
        """Décompresse le contenu"""
        try:
            return gzip.decompress(content)
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            return content
    
    async def _compress(self, content: bytes) -> bytes:
        """Compresse le contenu"""
        try:
            return gzip.compress(content)
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return content
    
    async def _encrypt(self, content: bytes) -> bytes:
        """Crypte le contenu (implémentation basique)"""
        # TODO: Implémenter le cryptage réel
        return content
    
    async def _decrypt(self, content: bytes) -> bytes:
        """Décrypte le contenu (implémentation basique)"""
        # TODO: Implémenter le décryptage réel
        return content
    
    async def _validate_data(self, data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Valide les données chargées"""
        try:
            if not isinstance(data, dict):
                logger.warning(f"Invalid data format in {file_path}")
                return {}
            
            # Validation basique des clés
            validated_data = {}
            for key, value in data.items():
                if isinstance(key, str) and key.strip():
                    validated_data[key] = value
                else:
                    logger.warning(f"Invalid key '{key}' in {file_path}")
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            return data
    
    def _get_file_cache_key(self, file_path: Path) -> str:
        """Génère une clé de cache pour un fichier"""
        return hashlib.md5(str(file_path).encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str, file_path: Path) -> Optional[Dict[str, Any]]:
        """Récupère depuis le cache avec vérification de fraîcheur"""
        try:
            with self._lock:
                if cache_key not in self._cache:
                    return None
                
                cached_data, cached_time, cached_mtime = self._cache[cache_key]
                
                # Vérifier si le fichier a été modifié
                current_mtime = await aiofiles.os.path.getmtime(file_path)
                if current_mtime > cached_mtime:
                    del self._cache[cache_key]
                    return None
                
                # Vérifier l'âge du cache
                if datetime.now() - cached_time > timedelta(hours=1):
                    del self._cache[cache_key]
                    return None
                
                return cached_data
                
        except Exception as e:
            logger.warning(f"Cache error: {e}")
            return None
    
    async def _cache_data(self, cache_key: str, data: Dict[str, Any], file_path: Path):
        """Met les données en cache"""
        try:
            mtime = await aiofiles.os.path.getmtime(file_path)
            
            with self._lock:
                self._cache[cache_key] = (data, datetime.now(), mtime)
                
                # Limiter la taille du cache
                if len(self._cache) > 1000:
                    # Supprimer les entrées les plus anciennes
                    oldest_keys = sorted(
                        self._cache.keys(),
                        key=lambda k: self._cache[k][1]
                    )[:100]
                    
                    for key in oldest_keys:
                        del self._cache[key]
                        
        except Exception as e:
            logger.warning(f"Cache error: {e}")
    
    async def _invalidate_cache(self, cache_key: str):
        """Invalide une entrée du cache"""
        with self._lock:
            self._cache.pop(cache_key, None)
    
    async def _load_file_with_timeout(self, file_path: Path) -> Dict[str, Any]:
        """Charge un fichier avec timeout"""
        try:
            return await asyncio.wait_for(
                self.load_from_file(file_path),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout loading file: {file_path}")
            return {}


class DynamicLocaleLoader:
    """Chargeur dynamique avec détection automatique et rechargement"""
    
    def __init__(self, locale_loader: LocaleLoader):
        self.locale_loader = locale_loader
        self._watchers = {}
        self._callbacks = defaultdict(list)
        self._running = False
    
    async def start_watching(self, directory: Path, callback: Callable):
        """Démarre la surveillance d'un répertoire"""
        try:
            self._callbacks[str(directory)].append(callback)
            
            if not self._running:
                self._running = True
                asyncio.create_task(self._watch_loop())
                
        except Exception as e:
            logger.error(f"Error starting directory watch: {e}")
    
    async def stop_watching(self, directory: Path):
        """Arrête la surveillance d'un répertoire"""
        directory_str = str(directory)
        self._callbacks.pop(directory_str, None)
        
        if not self._callbacks:
            self._running = False
    
    async def _watch_loop(self):
        """Boucle de surveillance des changements"""
        while self._running:
            try:
                # Surveiller les changements de fichiers
                # Implémentation basique - peut être améliorée avec inotify
                await asyncio.sleep(5)  # Vérifier toutes les 5 secondes
                
                for directory_str, callbacks in self._callbacks.items():
                    directory = Path(directory_str)
                    if directory.exists():
                        # Vérifier les modifications
                        for callback in callbacks:
                            try:
                                await callback(directory)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                
            except Exception as e:
                logger.error(f"Watch loop error: {e}")
                await asyncio.sleep(10)
