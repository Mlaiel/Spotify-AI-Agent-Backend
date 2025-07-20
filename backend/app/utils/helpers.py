

import json
import asyncio
import hashlib
import logging
import mimetypes
import os
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urlparse, urlencode
from pathlib import Path
import aiofiles
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processeur de données enterprise avec optimisations avancées.
    """
    
    @staticmethod
    def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
        """Fusion profonde de dictionnaires avec résolution de conflits."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = DataProcessor.deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def sanitize_data(data: Any, max_depth: int = 10) -> Any:
        """Sanitisation récursive des données avec protection XSS."""
        if max_depth <= 0:
            return None
            
        if isinstance(data, str):
            # Nettoyage XSS basique
            data = re.sub(r'<script[^>]*>.*?</script>', '', data, flags=re.IGNORECASE | re.DOTALL)
            data = re.sub(r'javascript:', '', data, flags=re.IGNORECASE)
            return data.strip()
        elif isinstance(data, dict):
            return {k: DataProcessor.sanitize_data(v, max_depth - 1) for k, v in data.items()}
        elif isinstance(data, list):
            return [DataProcessor.sanitize_data(item, max_depth - 1) for item in data]
        return data
    
    @staticmethod
    def extract_metadata(data: Dict) -> Dict[str, Any]:
        """Extraction de métadonnées enrichies."""
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "size_bytes": len(json.dumps(data, default=str)),
            "keys_count": len(data) if isinstance(data, dict) else 0,
            "data_types": {},
            "hash": hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
        }
        
        if isinstance(data, dict):
            for key, value in data.items():
                metadata["data_types"][key] = type(value).__name__
                
        return metadata

class ConfigManager:
    """
    Gestionnaire de configuration enterprise avec cache et reload à chaud.
    """
    
    def __init__(self):
        self._config_cache: Dict[str, Any] = {}
        self._file_timestamps: Dict[str, float] = {}
        
    def load_config(self, file_path: str, auto_reload: bool = True) -> Dict[str, Any]:
        """Chargement de configuration avec cache intelligent."""
        try:
            current_time = os.path.getmtime(file_path)
            
            if file_path in self._config_cache and auto_reload:
                if file_path in self._file_timestamps:
                    if current_time <= self._file_timestamps[file_path]:
                        return self._config_cache[file_path]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    config = json.load(f)
                else:
                    # Support YAML si besoin
                    config = {}
                    
            self._config_cache[file_path] = config
            self._file_timestamps[file_path] = current_time
            
            logger.info(f"Configuration loaded from {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return {}
    
    def get_env_config(self, prefix: str = "SPOTIFY_") -> Dict[str, str]:
        """Extraction des variables d'environnement par préfixe."""
        return {
            key[len(prefix):].lower(): value 
            for key, value in os.environ.items() 
            if key.startswith(prefix)
        }

class FileManager:
    """
    Gestionnaire de fichiers asynchrone avec sécurité avancée.
    """
    
    ALLOWED_EXTENSIONS = {
        'audio': {'.mp3', '.wav', '.flac', '.m4a', '.ogg'},
        'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp'},
        'document': {'.pdf', '.doc', '.docx', '.txt', '.md'},
        'data': {'.json', '.csv', '.xml', '.yaml', '.yml'}
    }
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    @staticmethod
    async def safe_read_file(file_path: str, max_size: Optional[int] = None) -> Optional[bytes]:
        """Lecture sécurisée de fichier avec limitations."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
                
            file_size = path.stat().st_size
            max_allowed = max_size or FileManager.MAX_FILE_SIZE
            
            if file_size > max_allowed:
                logger.warning(f"File {file_path} too large: {file_size} bytes")
                return None
                
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    @staticmethod
    async def safe_write_file(file_path: str, content: bytes, create_dirs: bool = True) -> bool:
        """Écriture sécurisée de fichier."""
        try:
            path = Path(file_path)
            
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
                
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
                
            logger.info(f"File written successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_file_type(filename: str, allowed_category: str) -> bool:
        """Validation du type de fichier par catégorie."""
        if allowed_category not in FileManager.ALLOWED_EXTENSIONS:
            return False
            
        file_ext = Path(filename).suffix.lower()
        return file_ext in FileManager.ALLOWED_EXTENSIONS[allowed_category]
    
    @staticmethod
    def generate_secure_filename(original_filename: str) -> str:
        """Génération de nom de fichier sécurisé."""
        # Nettoyage du nom
        clean_name = re.sub(r'[^\w\s-.]', '', original_filename)
        clean_name = re.sub(r'[-\s]+', '-', clean_name)
        
        # Ajout d'UUID pour unicité
        name_part = Path(clean_name).stem[:50]  # Limitation de longueur
        extension = Path(clean_name).suffix
        unique_id = uuid.uuid4().hex[:8]
        
        return f"{name_part}_{unique_id}{extension}"

class NetworkHelper:
    """
    Helper réseau avec gestion avancée des requêtes HTTP/HTTPS.
    """
    
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    
    @staticmethod
    async def make_request(
        url: str,
        method: str = 'GET',
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = MAX_RETRIES
    ) -> Optional[Dict]:
        """Requête HTTP avec retry et gestion d'erreurs avancée."""
        
        headers = headers or {}
        headers.setdefault('User-Agent', 'SpotifyAIAgent/2.0')
        
        for attempt in range(retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    async with session.request(method, url, headers=headers, json=data) as response:
                        if response.status < 400:
                            return {
                                'status': response.status,
                                'data': await response.json(),
                                'headers': dict(response.headers)
                            }
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Request error for {url}: {e}")
                
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)  # Backoff exponentiel
                
        return None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validation robuste d'URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except:
            return False
    
    @staticmethod
    def extract_domain(url: str) -> Optional[str]:
        """Extraction du domaine d'une URL."""
        try:
            return urlparse(url).netloc.lower()
        except:
            return None

class DateTimeHelper:
    """
    Helper avancé pour manipulation de dates et heures.
    """
    
    @staticmethod
    def utc_now() -> datetime:
        """Timestamp UTC actuel."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def parse_iso_datetime(iso_string: str) -> Optional[datetime]:
        """Parse d'une date ISO avec gestion d'erreurs."""
        try:
            return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        except:
            return None
    
    @staticmethod
    def format_duration(seconds: int) -> str:
        """Formatage de durée lisible."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def get_timezone_offset(timezone_name: str) -> Optional[timedelta]:
        """Calcul de décalage de timezone."""
        try:
            # Implémentation simplifiée
            common_timezones = {
                'UTC': timedelta(0),
                'EST': timedelta(hours=-5),
                'PST': timedelta(hours=-8),
                'CET': timedelta(hours=1)
            }
            return common_timezones.get(timezone_name)
        except:
            return None

class JsonHelper:
    """
    Helper JSON avancé avec sérialisation personnalisée.
    """
    
    @staticmethod
    def safe_serialize(obj: Any) -> str:
        """Sérialisation JSON sécurisée avec gestion des types complexes."""
        def default_serializer(o):
            if isinstance(o, datetime):
                return o.isoformat()
            elif isinstance(o, uuid.UUID):
                return str(o)
            elif hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)
        
        try:
            return json.dumps(obj, default=default_serializer, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"JSON serialization error: {e}")
            return "{}"
    
    @staticmethod
    def safe_parse(json_string: str) -> Optional[Dict]:
        """Parse JSON sécurisé avec validation."""
        try:
            if not json_string or not json_string.strip():
                return None
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None
    
    @staticmethod
    def flatten_dict(nested_dict: Dict, separator: str = '.') -> Dict:
        """Aplatissement de dictionnaire imbriqué."""
        def _flatten(obj, parent_key=''):
            items = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{separator}{k}" if parent_key else k
                    items.extend(_flatten(v, new_key).items())
            else:
                return {parent_key: obj}
            return dict(items)
        
        return _flatten(nested_dict)

class UrlHelper:
    """
    Helper URL avec construction et analyse avancées.
    """
    
    @staticmethod
    def build_url(base: str, path: str = '', params: Optional[Dict] = None) -> str:
        """Construction d'URL avec paramètres."""
        url = base.rstrip('/') + '/' + path.lstrip('/')
        if params:
            url += '?' + urlencode(params)
        return url
    
    @staticmethod
    def extract_params(url: str) -> Dict[str, str]:
        """Extraction des paramètres d'URL."""
        try:
            parsed = urlparse(url)
            from urllib.parse import parse_qs
            params = parse_qs(parsed.query)
            return {k: v[0] if v else '' for k, v in params.items()}
        except:
            return {}
    
    @staticmethod
    def clean_url(url: str) -> str:
        """Nettoyage et normalisation d'URL."""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url.lower()

class EmailHelper:
    """
    Helper email avec validation et construction avancées.
    """
    
    EMAIL_REGEX = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validation robuste d'email."""
        if not email or len(email) > 254:
            return False
        return bool(EmailHelper.EMAIL_REGEX.match(email.lower()))
    
    @staticmethod
    def extract_domain(email: str) -> Optional[str]:
        """Extraction du domaine d'un email."""
        try:
            return email.split('@')[1].lower()
        except:
            return None
    
    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalisation d'email."""
        return email.lower().strip()
    
    @staticmethod
    def build_mime_message(
        to_email: str,
        subject: str,
        text_content: str,
        html_content: Optional[str] = None,
        from_email: str = "noreply@spotify-ai.com"
    ) -> MIMEMultipart:
        """Construction de message MIME."""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        msg.attach(MIMEText(text_content, 'plain', 'utf-8'))
        if html_content:
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
        return msg

# Classes d'aide supplémentaires pour cache, métriques, audit...

class CacheHelper:
    """Helper de cache avancé avec TTL et invalidation."""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._default_ttl = timedelta(hours=1)
    
    def get(self, key: str) -> Optional[Any]:
        """Récupération avec vérification TTL."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.utcnow() < expiry:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Stockage avec TTL."""
        expiry = datetime.utcnow() + (ttl or self._default_ttl)
        self._cache[key] = (value, expiry)
    
    def invalidate(self, pattern: str = None) -> int:
        """Invalidation par pattern."""
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        
        keys_to_remove = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self._cache[key]
        return len(keys_to_remove)

class MetricsHelper:
    """Helper de métriques avec agrégation."""
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, List[float]] = {}
    
    def increment(self, metric: str, value: int = 1) -> None:
        """Incrémentation de compteur."""
        self._counters[metric] = self._counters.get(metric, 0) + value
    
    def record_time(self, metric: str, duration: float) -> None:
        """Enregistrement de durée."""
        if metric not in self._timers:
            self._timers[metric] = []
        self._timers[metric].append(duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques agrégées."""
        stats = {'counters': self._counters.copy()}
        
        for metric, times in self._timers.items():
            if times:
                stats[f"{metric}_avg"] = sum(times) / len(times)
                stats[f"{metric}_max"] = max(times)
                stats[f"{metric}_min"] = min(times)
                stats[f"{metric}_count"] = len(times)
        
        return stats

class AuditHelper:
    """Helper d'audit avec journalisation structurée."""
    
    @staticmethod
    def log_action(
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """Journalisation d'action utilisateur."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'ip_address': ip_address,
            'details': details or {}
        }
        
        # Log structuré pour agrégation
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    @staticmethod
    def log_security_event(
        event_type: str,
        severity: str,
        description: str,
        context: Optional[Dict] = None
    ) -> None:
        """Journalisation d'événement de sécurité."""
        security_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'context': context or {}
        }
        
        logger.warning(f"SECURITY: {json.dumps(security_entry)}")

# Export de toutes les classes
__all__ = [
    'DataProcessor', 'ConfigManager', 'FileManager', 'NetworkHelper',
    'DateTimeHelper', 'JsonHelper', 'UrlHelper', 'EmailHelper',
    'CacheHelper', 'MetricsHelper', 'AuditHelper'
]