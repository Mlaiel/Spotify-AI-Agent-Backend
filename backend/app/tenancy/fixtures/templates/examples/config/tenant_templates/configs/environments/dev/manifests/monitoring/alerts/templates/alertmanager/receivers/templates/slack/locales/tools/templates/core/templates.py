"""
Advanced Template Engine
========================

Moteur de templates avancé avec support Jinja2, cache intelligent,
hot-reload et fonctionnalités spécialisées pour le multi-tenancy.

Auteur: Fahed Mlaiel
"""

import asyncio
import logging
import os
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
import aiofiles
import jinja2
import yaml
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import weakref

logger = logging.getLogger(__name__)


@dataclass
class TemplateMetadata:
    """Métadonnées d'un template"""
    name: str
    path: Path
    last_modified: datetime
    size: int
    checksum: str
    variables: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tenant_specific: bool = False
    category: str = "general"
    description: str = ""


@dataclass
class TemplateCache:
    """Cache d'un template"""
    template: jinja2.Template
    metadata: TemplateMetadata
    last_accessed: datetime
    access_count: int = 0
    compiled_at: datetime = field(default_factory=datetime.utcnow)


class TemplateFileHandler(FileSystemEventHandler):
    """Gestionnaire des événements de fichiers templates"""
    
    def __init__(self, template_engine):
        self.template_engine = weakref.ref(template_engine)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.j2', '.jinja', '.template')):
            engine = self.template_engine()
            if engine:
                asyncio.create_task(engine._reload_template(Path(event.src_path)))
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.j2', '.jinja', '.template')):
            engine = self.template_engine()
            if engine:
                asyncio.create_task(engine._load_template(Path(event.src_path)))
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(('.j2', '.jinja', '.template')):
            engine = self.template_engine()
            if engine:
                asyncio.create_task(engine._unload_template(Path(event.src_path)))


class TemplateEngine:
    """
    Moteur de templates avancé
    
    Fonctionnalités:
    - Support Jinja2 avec extensions
    - Cache intelligent avec TTL
    - Hot-reload automatique
    - Templates tenant-spécifiques
    - Fonctions et filtres personnalisés
    - Validation et sécurité
    - Métriques et monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le moteur de templates
        
        Args:
            config: Configuration du moteur
        """
        self.config = config
        self.is_initialized = False
        
        # Environnement Jinja2
        self.jinja_env: Optional[jinja2.Environment] = None
        
        # Cache des templates
        self.template_cache: Dict[str, TemplateCache] = {}
        self.cache_max_size = config.get("cache_max_size", 1000)
        self.cache_ttl = config.get("cache_ttl", 3600)
        
        # Métadonnées des templates
        self.template_metadata: Dict[str, TemplateMetadata] = {}
        
        # Répertoires de templates
        self.template_dirs: List[Path] = []
        self.tenant_template_dirs: Dict[str, Path] = {}
        
        # Observateur de fichiers pour hot-reload
        self.file_observer: Optional[Observer] = None
        self.auto_reload = config.get("auto_reload", True)
        
        # Fonctions et filtres personnalisés
        self.custom_functions: Dict[str, Callable] = {}
        self.custom_filters: Dict[str, Callable] = {}
        
        # Métriques
        self.metrics = {
            "templates_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "render_count": 0,
            "render_errors": 0,
            "reload_count": 0
        }
        
        logger.info("TemplateEngine initialisé")
    
    async def initialize(self) -> None:
        """Initialise le moteur de templates"""
        if self.is_initialized:
            return
        
        logger.info("Initialisation du TemplateEngine...")
        
        try:
            # Configuration des répertoires
            await self._setup_template_directories()
            
            # Configuration de l'environnement Jinja2
            await self._setup_jinja_environment()
            
            # Chargement des templates
            await self._load_all_templates()
            
            # Configuration du hot-reload
            if self.auto_reload:
                await self._setup_file_watching()
            
            # Démarrage des tâches de maintenance
            await self._start_maintenance_tasks()
            
            self.is_initialized = True
            logger.info("TemplateEngine initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du TemplateEngine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrêt propre du moteur de templates"""
        if not self.is_initialized:
            return
        
        logger.info("Arrêt du TemplateEngine...")
        
        try:
            # Arrêt de l'observateur de fichiers
            if self.file_observer:
                self.file_observer.stop()
                self.file_observer.join()
            
            # Nettoyage du cache
            self.template_cache.clear()
            self.template_metadata.clear()
            
            self.is_initialized = False
            logger.info("TemplateEngine arrêté avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")
    
    async def _setup_template_directories(self) -> None:
        """Configure les répertoires de templates"""
        # Répertoire principal
        main_dir = Path(self.config.get("template_dir", "templates"))
        if main_dir.exists():
            self.template_dirs.append(main_dir)
        
        # Répertoires additionnels
        additional_dirs = self.config.get("additional_dirs", [])
        for dir_path in additional_dirs:
            path = Path(dir_path)
            if path.exists():
                self.template_dirs.append(path)
        
        # Répertoires tenant-spécifiques
        tenant_base_dir = Path(self.config.get("tenant_template_dir", "templates/tenants"))
        if tenant_base_dir.exists():
            for tenant_dir in tenant_base_dir.iterdir():
                if tenant_dir.is_dir():
                    self.tenant_template_dirs[tenant_dir.name] = tenant_dir
        
        logger.info(f"Configurés {len(self.template_dirs)} répertoires de templates")
        logger.info(f"Configurés {len(self.tenant_template_dirs)} répertoires tenant-spécifiques")
    
    async def _setup_jinja_environment(self) -> None:
        """Configure l'environnement Jinja2"""
        # Loader avec plusieurs répertoires
        loaders = [jinja2.FileSystemLoader(str(d)) for d in self.template_dirs]
        combined_loader = jinja2.ChoiceLoader(loaders)
        
        # Configuration de l'environnement
        self.jinja_env = jinja2.Environment(
            loader=combined_loader,
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=self.config.get("trim_blocks", True),
            lstrip_blocks=self.config.get("lstrip_blocks", True),
            extensions=self.config.get("extensions", [
                "jinja2.ext.do",
                "jinja2.ext.loopcontrols",
                "jinja2.ext.with_"
            ]),
            cache_size=self.config.get("jinja_cache_size", 400),
            auto_reload=False  # Nous gérons le reload manuellement
        )
        
        # Ajout des fonctions globales
        await self._setup_global_functions()
        
        # Ajout des filtres personnalisés
        await self._setup_custom_filters()
        
        logger.info("Environnement Jinja2 configuré")
    
    async def _setup_global_functions(self) -> None:
        """Configure les fonctions globales Jinja2"""
        # Fonctions de base
        self.jinja_env.globals.update({
            'now': datetime.utcnow,
            'today': datetime.utcnow().date,
            'range': range,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'enumerate': enumerate,
            'zip': zip,
        })
        
        # Fonctions personnalisées spécifiques au projet
        self.jinja_env.globals.update({
            'format_bytes': self._format_bytes,
            'format_duration': self._format_duration,
            'format_number': self._format_number,
            'generate_id': self._generate_id,
            'get_env': self._get_env_var,
            'tenant_config': self._get_tenant_config,
            'load_json': self._load_json_data,
            'load_yaml': self._load_yaml_data,
        })
        
        # Ajout des fonctions personnalisées configurées
        self.jinja_env.globals.update(self.custom_functions)
    
    async def _setup_custom_filters(self) -> None:
        """Configure les filtres personnalisés"""
        # Filtres de base
        self.jinja_env.filters.update({
            'datetime': self._filter_datetime,
            'date': self._filter_date,
            'time': self._filter_time,
            'ago': self._filter_ago,
            'bytes': self._format_bytes,
            'duration': self._format_duration,
            'number': self._format_number,
            'truncate_words': self._filter_truncate_words,
            'slugify': self._filter_slugify,
            'markdown': self._filter_markdown,
            'base64': self._filter_base64,
            'urlencode': self._filter_urlencode,
            'jsonify': self._filter_jsonify,
        })
        
        # Ajout des filtres personnalisés configurés
        self.jinja_env.filters.update(self.custom_filters)
    
    # Fonctions utilitaires pour les templates
    
    def _format_bytes(self, bytes_value: Union[int, float]) -> str:
        """Formate une taille en octets"""
        if bytes_value == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        unit_index = 0
        value = float(bytes_value)
        
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1
        
        return f"{value:.1f} {units[unit_index]}"
    
    def _format_duration(self, seconds: Union[int, float]) -> str:
        """Formate une durée en secondes"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
    
    def _format_number(self, number: Union[int, float], precision: int = 2) -> str:
        """Formate un nombre avec des séparateurs"""
        if isinstance(number, float):
            return f"{number:,.{precision}f}"
        return f"{number:,}"
    
    def _generate_id(self, prefix: str = "") -> str:
        """Génère un ID unique"""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}{unique_id}" if prefix else unique_id
    
    def _get_env_var(self, var_name: str, default: str = "") -> str:
        """Récupère une variable d'environnement"""
        return os.getenv(var_name, default)
    
    def _get_tenant_config(self, tenant_id: str, key: str, default: Any = None) -> Any:
        """Récupère une configuration tenant (à implémenter selon le contexte)"""
        # Placeholder - à connecter avec le système de configuration tenant
        return default
    
    def _load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Charge des données JSON"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                # Recherche dans les répertoires de templates
                for template_dir in self.template_dirs:
                    full_path = template_dir / path
                    if full_path.exists():
                        path = full_path
                        break
            
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement JSON {file_path}: {e}")
            return {}
    
    def _load_yaml_data(self, file_path: str) -> Dict[str, Any]:
        """Charge des données YAML"""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                # Recherche dans les répertoires de templates
                for template_dir in self.template_dirs:
                    full_path = template_dir / path
                    if full_path.exists():
                        path = full_path
                        break
            
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement YAML {file_path}: {e}")
            return {}
    
    # Filtres personnalisés
    
    def _filter_datetime(self, dt: datetime, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Filtre de formatage datetime"""
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        return dt.strftime(format)
    
    def _filter_date(self, dt: datetime, format: str = "%Y-%m-%d") -> str:
        """Filtre de formatage date"""
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        return dt.strftime(format)
    
    def _filter_time(self, dt: datetime, format: str = "%H:%M:%S") -> str:
        """Filtre de formatage time"""
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        return dt.strftime(format)
    
    def _filter_ago(self, dt: datetime) -> str:
        """Filtre 'il y a X temps'"""
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 0:
            return f"il y a {diff.days} jour{'s' if diff.days > 1 else ''}"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"il y a {hours} heure{'s' if hours > 1 else ''}"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
        else:
            return "à l'instant"
    
    def _filter_truncate_words(self, text: str, length: int = 20, end: str = "...") -> str:
        """Tronque un texte par nombre de mots"""
        words = text.split()
        if len(words) <= length:
            return text
        return " ".join(words[:length]) + end
    
    def _filter_slugify(self, text: str) -> str:
        """Convertit un texte en slug"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
    
    def _filter_markdown(self, text: str) -> str:
        """Convertit Markdown en HTML (basique)"""
        try:
            import markdown
            return markdown.markdown(text)
        except ImportError:
            logger.warning("Module markdown non disponible")
            return text
    
    def _filter_base64(self, text: str, encode: bool = True) -> str:
        """Encode/décode en base64"""
        import base64
        if encode:
            return base64.b64encode(text.encode()).decode()
        else:
            return base64.b64decode(text).decode()
    
    def _filter_urlencode(self, text: str) -> str:
        """Encode URL"""
        from urllib.parse import quote
        return quote(text)
    
    def _filter_jsonify(self, obj: Any) -> str:
        """Convertit un objet en JSON"""
        return json.dumps(obj, default=str, ensure_ascii=False)
    
    async def _load_all_templates(self) -> None:
        """Charge tous les templates"""
        for template_dir in self.template_dirs:
            await self._load_templates_from_directory(template_dir)
        
        # Chargement des templates tenant-spécifiques
        for tenant_id, tenant_dir in self.tenant_template_dirs.items():
            await self._load_templates_from_directory(tenant_dir, tenant_specific=True)
        
        logger.info(f"Chargés {len(self.template_metadata)} templates")
    
    async def _load_templates_from_directory(self, directory: Path, tenant_specific: bool = False) -> None:
        """Charge les templates d'un répertoire"""
        if not directory.exists():
            return
        
        extensions = self.config.get("allowed_extensions", [".j2", ".jinja", ".template"])
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and any(file_path.name.endswith(ext) for ext in extensions):
                await self._load_template(file_path, tenant_specific)
    
    async def _load_template(self, file_path: Path, tenant_specific: bool = False) -> None:
        """Charge un template spécifique"""
        try:
            # Calcul du nom relatif
            template_name = self._get_template_name(file_path)
            
            # Lecture du fichier
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Calcul du checksum
            checksum = hashlib.md5(content.encode()).hexdigest()
            
            # Extraction des variables du template
            variables = self._extract_template_variables(content)
            
            # Création des métadonnées
            stat = file_path.stat()
            metadata = TemplateMetadata(
                name=template_name,
                path=file_path,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                size=stat.st_size,
                checksum=checksum,
                variables=variables,
                tenant_specific=tenant_specific
            )
            
            # Compilation du template
            template = self.jinja_env.from_string(content)
            template.name = template_name
            
            # Mise en cache
            cache_entry = TemplateCache(
                template=template,
                metadata=metadata,
                last_accessed=datetime.utcnow()
            )
            
            self.template_cache[template_name] = cache_entry
            self.template_metadata[template_name] = metadata
            
            self.metrics["templates_loaded"] += 1
            
            logger.debug(f"Template chargé: {template_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du template {file_path}: {e}")
    
    def _get_template_name(self, file_path: Path) -> str:
        """Détermine le nom du template à partir du chemin"""
        # Recherche du répertoire parent dans template_dirs
        for template_dir in self.template_dirs:
            try:
                relative_path = file_path.relative_to(template_dir)
                return str(relative_path)
            except ValueError:
                continue
        
        # Recherche dans les répertoires tenant-spécifiques
        for tenant_id, tenant_dir in self.tenant_template_dirs.items():
            try:
                relative_path = file_path.relative_to(tenant_dir)
                return f"tenant/{tenant_id}/{relative_path}"
            except ValueError:
                continue
        
        # Par défaut, utiliser le nom du fichier
        return file_path.name
    
    def _extract_template_variables(self, content: str) -> List[str]:
        """Extrait les variables utilisées dans un template"""
        import re
        
        # Pattern pour les variables Jinja2
        variable_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*(?:\|[^}]*)?\}\}'
        
        variables = set()
        for match in re.finditer(variable_pattern, content):
            var_name = match.group(1).split('.')[0]  # Prendre seulement le nom de base
            variables.add(var_name)
        
        return sorted(list(variables))
    
    async def _setup_file_watching(self) -> None:
        """Configure la surveillance des fichiers pour le hot-reload"""
        if not self.auto_reload:
            return
        
        try:
            self.file_observer = Observer()
            handler = TemplateFileHandler(self)
            
            for template_dir in self.template_dirs:
                if template_dir.exists():
                    self.file_observer.schedule(handler, str(template_dir), recursive=True)
            
            for tenant_dir in self.tenant_template_dirs.values():
                if tenant_dir.exists():
                    self.file_observer.schedule(handler, str(tenant_dir), recursive=True)
            
            self.file_observer.start()
            logger.info("Surveillance des fichiers templates activée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration du file watching: {e}")
    
    async def _start_maintenance_tasks(self) -> None:
        """Démarre les tâches de maintenance"""
        # Nettoyage du cache
        asyncio.create_task(self._cache_cleanup_task())
        
        # Mise à jour des métriques
        asyncio.create_task(self._metrics_update_task())
    
    async def _cache_cleanup_task(self) -> None:
        """Tâche de nettoyage du cache"""
        while self.is_initialized:
            try:
                current_time = datetime.utcnow()
                expired_entries = []
                
                for name, cache_entry in self.template_cache.items():
                    # Vérification du TTL
                    if (current_time - cache_entry.last_accessed).total_seconds() > self.cache_ttl:
                        expired_entries.append(name)
                
                # Suppression des entrées expirées
                for name in expired_entries:
                    del self.template_cache[name]
                    logger.debug(f"Cache expiré pour le template: {name}")
                
                # Limitation de la taille du cache
                if len(self.template_cache) > self.cache_max_size:
                    # Suppression des moins récemment utilisés
                    sorted_entries = sorted(
                        self.template_cache.items(),
                        key=lambda x: x[1].last_accessed
                    )
                    
                    entries_to_remove = len(self.template_cache) - self.cache_max_size
                    for name, _ in sorted_entries[:entries_to_remove]:
                        del self.template_cache[name]
                        logger.debug(f"Cache LRU supprimé pour le template: {name}")
                
                await asyncio.sleep(300)  # Nettoyage toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"Erreur dans le nettoyage du cache: {e}")
                await asyncio.sleep(600)
    
    async def _metrics_update_task(self) -> None:
        """Tâche de mise à jour des métriques"""
        while self.is_initialized:
            try:
                self.metrics["cache_size"] = len(self.template_cache)
                self.metrics["metadata_count"] = len(self.template_metadata)
                
                await asyncio.sleep(60)  # Mise à jour toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur dans la mise à jour des métriques: {e}")
                await asyncio.sleep(120)
    
    async def _reload_template(self, file_path: Path) -> None:
        """Recharge un template modifié"""
        try:
            template_name = self._get_template_name(file_path)
            
            if template_name in self.template_cache:
                await self._load_template(file_path)
                self.metrics["reload_count"] += 1
                logger.info(f"Template rechargé: {template_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors du rechargement du template {file_path}: {e}")
    
    async def _unload_template(self, file_path: Path) -> None:
        """Décharge un template supprimé"""
        try:
            template_name = self._get_template_name(file_path)
            
            if template_name in self.template_cache:
                del self.template_cache[template_name]
            
            if template_name in self.template_metadata:
                del self.template_metadata[template_name]
            
            logger.info(f"Template déchargé: {template_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors du déchargement du template {file_path}: {e}")
    
    # API publique
    
    async def render_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Rend un template avec le contexte donné
        
        Args:
            template_name: Nom du template
            context: Contexte de rendu
            tenant_id: ID du tenant pour les templates spécifiques
            
        Returns:
            Contenu rendu
        """
        if not self.is_initialized:
            raise RuntimeError("TemplateEngine non initialisé")
        
        try:
            # Recherche du template (tenant-spécifique d'abord)
            template = await self._get_template(template_name, tenant_id)
            
            if not template:
                raise ValueError(f"Template non trouvé: {template_name}")
            
            # Mise à jour des métriques d'accès
            if template_name in self.template_cache:
                cache_entry = self.template_cache[template_name]
                cache_entry.last_accessed = datetime.utcnow()
                cache_entry.access_count += 1
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
            
            # Rendu du template
            rendered_content = template.render(**context)
            
            self.metrics["render_count"] += 1
            return rendered_content
            
        except Exception as e:
            self.metrics["render_errors"] += 1
            logger.error(f"Erreur lors du rendu du template {template_name}: {e}")
            raise
    
    async def _get_template(self, template_name: str, tenant_id: Optional[str] = None) -> Optional[jinja2.Template]:
        """Récupère un template du cache ou le charge"""
        # Recherche tenant-spécifique d'abord
        if tenant_id:
            tenant_template_name = f"tenant/{tenant_id}/{template_name}"
            if tenant_template_name in self.template_cache:
                return self.template_cache[tenant_template_name].template
        
        # Recherche du template général
        if template_name in self.template_cache:
            return self.template_cache[template_name].template
        
        # Tentative de chargement à la volée
        await self._try_load_template_on_demand(template_name, tenant_id)
        
        # Nouvelle tentative de récupération
        if tenant_id:
            tenant_template_name = f"tenant/{tenant_id}/{template_name}"
            if tenant_template_name in self.template_cache:
                return self.template_cache[tenant_template_name].template
        
        if template_name in self.template_cache:
            return self.template_cache[template_name].template
        
        return None
    
    async def _try_load_template_on_demand(self, template_name: str, tenant_id: Optional[str] = None) -> None:
        """Tente de charger un template à la demande"""
        # Recherche dans les répertoires tenant-spécifiques
        if tenant_id and tenant_id in self.tenant_template_dirs:
            tenant_dir = self.tenant_template_dirs[tenant_id]
            tenant_path = tenant_dir / template_name
            
            if tenant_path.exists():
                await self._load_template(tenant_path, tenant_specific=True)
                return
        
        # Recherche dans les répertoires généraux
        for template_dir in self.template_dirs:
            template_path = template_dir / template_name
            
            if template_path.exists():
                await self._load_template(template_path)
                return
    
    async def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """
        Rend une chaîne template
        
        Args:
            template_string: Chaîne template
            context: Contexte de rendu
            
        Returns:
            Contenu rendu
        """
        if not self.is_initialized:
            raise RuntimeError("TemplateEngine non initialisé")
        
        try:
            template = self.jinja_env.from_string(template_string)
            return template.render(**context)
            
        except Exception as e:
            logger.error(f"Erreur lors du rendu de la chaîne template: {e}")
            raise
    
    def add_global_function(self, name: str, function: Callable) -> None:
        """
        Ajoute une fonction globale
        
        Args:
            name: Nom de la fonction
            function: Fonction à ajouter
        """
        self.custom_functions[name] = function
        if self.jinja_env:
            self.jinja_env.globals[name] = function
    
    def add_filter(self, name: str, filter_func: Callable) -> None:
        """
        Ajoute un filtre personnalisé
        
        Args:
            name: Nom du filtre
            filter_func: Fonction filtre
        """
        self.custom_filters[name] = filter_func
        if self.jinja_env:
            self.jinja_env.filters[name] = filter_func
    
    def remove_global_function(self, name: str) -> None:
        """Supprime une fonction globale"""
        if name in self.custom_functions:
            del self.custom_functions[name]
        if self.jinja_env and name in self.jinja_env.globals:
            del self.jinja_env.globals[name]
    
    def remove_filter(self, name: str) -> None:
        """Supprime un filtre"""
        if name in self.custom_filters:
            del self.custom_filters[name]
        if self.jinja_env and name in self.jinja_env.filters:
            del self.jinja_env.filters[name]
    
    async def list_templates(self, tenant_id: Optional[str] = None) -> List[str]:
        """
        Liste les templates disponibles
        
        Args:
            tenant_id: ID du tenant pour filtrer
            
        Returns:
            Liste des noms de templates
        """
        templates = list(self.template_metadata.keys())
        
        if tenant_id:
            # Inclure les templates tenant-spécifiques
            tenant_prefix = f"tenant/{tenant_id}/"
            templates = [
                t for t in templates 
                if not t.startswith("tenant/") or t.startswith(tenant_prefix)
            ]
        else:
            # Exclure les templates tenant-spécifiques
            templates = [t for t in templates if not t.startswith("tenant/")]
        
        return sorted(templates)
    
    async def get_template_metadata(self, template_name: str) -> Optional[TemplateMetadata]:
        """
        Récupère les métadonnées d'un template
        
        Args:
            template_name: Nom du template
            
        Returns:
            Métadonnées ou None si non trouvé
        """
        return self.template_metadata.get(template_name)
    
    async def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        Valide un template
        
        Args:
            template_name: Nom du template
            
        Returns:
            Rapport de validation
        """
        try:
            template = await self._get_template(template_name)
            
            if not template:
                return {
                    "valid": False,
                    "error": "Template non trouvé"
                }
            
            # Test de rendu avec un contexte vide
            try:
                template.render()
                syntax_valid = True
                syntax_error = None
            except Exception as e:
                syntax_valid = False
                syntax_error = str(e)
            
            metadata = self.template_metadata.get(template_name)
            
            return {
                "valid": syntax_valid,
                "syntax_error": syntax_error,
                "variables": metadata.variables if metadata else [],
                "size": metadata.size if metadata else 0,
                "last_modified": metadata.last_modified.isoformat() if metadata else None
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du moteur
        
        Returns:
            Métriques
        """
        return {
            **self.metrics,
            "cache_size": len(self.template_cache),
            "templates_count": len(self.template_metadata),
            "template_dirs": len(self.template_dirs),
            "tenant_dirs": len(self.tenant_template_dirs)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérification de l'état de santé du moteur
        
        Returns:
            Rapport d'état
        """
        try:
            return {
                "status": "healthy",
                "is_initialized": self.is_initialized,
                "templates_loaded": len(self.template_metadata),
                "cache_size": len(self.template_cache),
                "auto_reload": self.auto_reload,
                "file_observer_active": self.file_observer and self.file_observer.is_alive() if self.file_observer else False
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_initialized": self.is_initialized
            }
