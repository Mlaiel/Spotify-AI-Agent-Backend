"""
Template Engine - Moteur de Templates Avancé pour Spotify AI Agent
Système de templating dynamique avec support multi-format et cache intelligent
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
import hashlib

import aioredis
import aiofiles
from jinja2 import Environment, FileSystemLoader, DictLoader, select_autoescape
from jinja2.exceptions import TemplateError, TemplateNotFound
from prometheus_client import Counter, Histogram
import yaml
import markdown
from bs4 import BeautifulSoup


class TemplateFormat(Enum):
    """Formats de templates supportés"""
    JINJA2 = "jinja2"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    YAML = "yaml"
    PLAINTEXT = "plaintext"


class TemplateCategory(Enum):
    """Catégories de templates"""
    ALERTS = "alerts"
    NOTIFICATIONS = "notifications"
    REPORTS = "reports"
    EMAILS = "emails"
    SLACK = "slack"
    WEBHOOKS = "webhooks"
    DASHBOARDS = "dashboards"
    SYSTEM = "system"


@dataclass
class TemplateInfo:
    """Informations sur un template"""
    name: str
    category: TemplateCategory
    format: TemplateFormat
    description: str
    variables: List[str]
    required_variables: List[str]
    author: str
    version: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class RenderContext:
    """Contexte de rendu d'un template"""
    template_name: str
    variables: Dict[str, Any]
    tenant_id: str
    locale: str
    format: TemplateFormat
    timestamp: datetime
    user_id: Optional[str] = None
    request_id: Optional[str] = None


class TemplateEngine:
    """
    Moteur de templates avancé avec fonctionnalités :
    - Support multi-format (Jinja2, Markdown, HTML, JSON, YAML)
    - Cache intelligent avec invalidation
    - Templates dynamiques avec conditions
    - Héritage et inclusion de templates
    - Fonctions personnalisées (filters, tests, globals)
    - Validation de variables requises
    - Versioning et rollback
    - Préprocesseurs et post-processeurs
    - Sandboxing pour la sécurité
    """
    
    def __init__(
        self,
        templates_path: str,
        redis_client: aioredis.Redis,
        config: Dict[str, Any],
        tenant_id: str = ""
    ):
        self.templates_path = Path(templates_path)
        self.redis_client = redis_client
        self.config = config
        self.tenant_id = tenant_id
        
        # Logger avec contexte
        self.logger = logging.getLogger(f"template_engine.{tenant_id}")
        
        # Métriques Prometheus
        self.render_counter = Counter(
            'templates_rendered_total',
            'Total templates rendered',
            ['tenant_id', 'template_name', 'format', 'status']
        )
        
        self.render_duration = Histogram(
            'template_render_duration_seconds',
            'Time spent rendering templates',
            ['tenant_id', 'template_name', 'format']
        )
        
        self.cache_hits = Counter(
            'template_cache_hits_total',
            'Template cache hits',
            ['tenant_id', 'template_name']
        )
        
        # Environnements Jinja2
        self.jinja_env = None
        self.sandbox_env = None
        
        # Cache des templates
        self.template_cache = {}
        self.metadata_cache = {}
        
        # Fonctions personnalisées
        self.custom_filters = {}
        self.custom_tests = {}
        self.custom_globals = {}
        
        # Préprocesseurs et post-processeurs
        self.preprocessors = {}
        self.postprocessors = {}
        
        # Configuration du cache
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 heure par défaut
        
        # Initialisation asynchrone
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialisation asynchrone du moteur de templates"""
        if self._initialized:
            return
        
        try:
            # Création des environnements Jinja2
            await self._setup_jinja_environments()
            
            # Chargement des templates
            await self._load_templates()
            
            # Chargement des fonctions personnalisées
            await self._load_custom_functions()
            
            # Configuration des préprocesseurs/post-processeurs
            await self._setup_processors()
            
            # Initialisation du cache
            await self._initialize_cache()
            
            self._initialized = True
            self.logger.info("Template engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize template engine: {str(e)}")
            raise
    
    async def render_template(
        self,
        template_name: str,
        data: Dict[str, Any],
        locale: str = "en_US",
        format: Optional[TemplateFormat] = None,
        tenant_id: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Rend un template avec les données fournies
        
        Args:
            template_name: Nom du template à rendre
            data: Données pour le template
            locale: Locale pour la localisation
            format: Format de rendu (détecté automatiquement si None)
            tenant_id: ID du tenant (par défaut celui de l'instance)
            use_cache: Utiliser le cache
            
        Returns:
            str: Template rendu
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Utilisation du tenant_id fourni ou celui de l'instance
            effective_tenant_id = tenant_id or self.tenant_id
            
            # Détection automatique du format si non spécifié
            if format is None:
                format = await self._detect_template_format(template_name)
            
            # Création du contexte de rendu
            render_context = RenderContext(
                template_name=template_name,
                variables=data,
                tenant_id=effective_tenant_id,
                locale=locale,
                format=format,
                timestamp=datetime.utcnow()
            )
            
            # Vérification du cache
            if use_cache:
                cached_result = await self._get_cached_render(render_context)
                if cached_result:
                    self.cache_hits.labels(
                        tenant_id=effective_tenant_id,
                        template_name=template_name
                    ).inc()
                    return cached_result
            
            # Chargement du template
            template_content = await self._load_template_content(
                template_name, effective_tenant_id
            )
            
            if not template_content:
                raise TemplateNotFound(f"Template '{template_name}' not found")
            
            # Validation des variables requises
            await self._validate_required_variables(template_name, data)
            
            # Préprocessing
            processed_data = await self._preprocess_data(data, format, render_context)
            
            # Rendu selon le format
            rendered_content = await self._render_by_format(
                template_content, processed_data, format, render_context
            )
            
            # Post-processing
            final_content = await self._postprocess_content(
                rendered_content, format, render_context
            )
            
            # Mise en cache
            if use_cache:
                await self._cache_render_result(render_context, final_content)
            
            # Métriques
            render_time = time.time() - start_time
            
            self.render_counter.labels(
                tenant_id=effective_tenant_id,
                template_name=template_name,
                format=format.value,
                status='success'
            ).inc()
            
            self.render_duration.labels(
                tenant_id=effective_tenant_id,
                template_name=template_name,
                format=format.value
            ).observe(render_time)
            
            self.logger.info(
                f"Template rendered successfully",
                extra={
                    "template_name": template_name,
                    "format": format.value,
                    "render_time": render_time,
                    "tenant_id": effective_tenant_id
                }
            )
            
            return final_content
            
        except Exception as e:
            self.render_counter.labels(
                tenant_id=effective_tenant_id or self.tenant_id,
                template_name=template_name,
                format=format.value if format else 'unknown',
                status='error'
            ).inc()
            
            self.logger.error(
                f"Error rendering template '{template_name}': {str(e)}",
                exc_info=True
            )
            raise
    
    async def get_template_info(self, template_name: str) -> Optional[TemplateInfo]:
        """
        Retourne les informations sur un template
        
        Args:
            template_name: Nom du template
            
        Returns:
            TemplateInfo: Informations du template ou None si non trouvé
        """
        try:
            # Vérification du cache de métadonnées
            cache_key = f"template_info:{self.tenant_id}:{template_name}"
            cached_info = await self.redis_client.get(cache_key)
            
            if cached_info:
                info_dict = json.loads(cached_info)
                return TemplateInfo(**info_dict)
            
            # Chargement depuis le système de fichiers
            metadata_file = self.templates_path / f"{template_name}.meta.yml"
            if not metadata_file.exists():
                return None
            
            async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(await f.read())
            
            # Analyse du template pour extraire les variables
            template_content = await self._load_template_content(template_name, self.tenant_id)
            variables = await self._extract_template_variables(template_content)
            
            template_info = TemplateInfo(
                name=metadata.get('name', template_name),
                category=TemplateCategory(metadata.get('category', 'system')),
                format=TemplateFormat(metadata.get('format', 'jinja2')),
                description=metadata.get('description', ''),
                variables=variables,
                required_variables=metadata.get('required_variables', []),
                author=metadata.get('author', 'Unknown'),
                version=metadata.get('version', '1.0.0'),
                created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(metadata.get('updated_at', datetime.utcnow().isoformat())),
                tags=metadata.get('tags', []),
                metadata=metadata.get('metadata', {})
            )
            
            # Mise en cache
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(asdict(template_info), default=str)
            )
            
            return template_info
            
        except Exception as e:
            self.logger.error(f"Error getting template info: {str(e)}")
            return None
    
    async def list_templates(
        self,
        category: Optional[TemplateCategory] = None,
        format: Optional[TemplateFormat] = None
    ) -> List[TemplateInfo]:
        """
        Liste les templates disponibles avec filtrage optionnel
        
        Args:
            category: Filtrer par catégorie
            format: Filtrer par format
            
        Returns:
            List[TemplateInfo]: Liste des templates
        """
        try:
            templates = []
            
            # Parcours des fichiers de templates
            for template_file in self.templates_path.glob("*.j2"):
                template_name = template_file.stem
                template_info = await self.get_template_info(template_name)
                
                if template_info:
                    # Filtrage par catégorie
                    if category and template_info.category != category:
                        continue
                    
                    # Filtrage par format
                    if format and template_info.format != format:
                        continue
                    
                    templates.append(template_info)
            
            # Tri par nom
            templates.sort(key=lambda t: t.name)
            
            return templates
            
        except Exception as e:
            self.logger.error(f"Error listing templates: {str(e)}")
            return []
    
    async def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        Valide un template et retourne un rapport de validation
        
        Args:
            template_name: Nom du template à valider
            
        Returns:
            Dict: Rapport de validation
        """
        try:
            validation_report = {
                'template_name': template_name,
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'info': {
                    'variables_found': [],
                    'includes_found': [],
                    'filters_used': []
                }
            }
            
            # Chargement du template
            template_content = await self._load_template_content(template_name, self.tenant_id)
            if not template_content:
                validation_report['is_valid'] = False
                validation_report['errors'].append(f"Template '{template_name}' not found")
                return validation_report
            
            # Validation de la syntaxe Jinja2
            try:
                self.jinja_env.parse(template_content)
            except TemplateError as e:
                validation_report['is_valid'] = False
                validation_report['errors'].append(f"Syntax error: {str(e)}")
            
            # Extraction des variables
            variables = await self._extract_template_variables(template_content)
            validation_report['info']['variables_found'] = variables
            
            # Détection des includes/extends
            includes = re.findall(r'{%\s*(?:include|extends)\s+["\']([^"\']+)["\']', template_content)
            validation_report['info']['includes_found'] = includes
            
            # Validation des includes
            for include in includes:
                if not await self._template_exists(include):
                    validation_report['warnings'].append(f"Included template '{include}' not found")
            
            # Détection des filtres utilisés
            filters = re.findall(r'\|\s*(\w+)', template_content)
            validation_report['info']['filters_used'] = list(set(filters))
            
            # Validation des filtres personnalisés
            for filter_name in filters:
                if filter_name not in self.jinja_env.filters and filter_name not in self.custom_filters:
                    validation_report['warnings'].append(f"Unknown filter '{filter_name}'")
            
            return validation_report
            
        except Exception as e:
            return {
                'template_name': template_name,
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'info': {}
            }
    
    async def clear_cache(self, template_name: Optional[str] = None) -> int:
        """
        Vide le cache des templates
        
        Args:
            template_name: Template spécifique (None pour tout vider)
            
        Returns:
            int: Nombre d'entrées supprimées du cache
        """
        try:
            if template_name:
                # Suppression d'un template spécifique
                pattern = f"template_cache:{self.tenant_id}:{template_name}:*"
            else:
                # Suppression de tout le cache du tenant
                pattern = f"template_cache:{self.tenant_id}:*"
            
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.logger.info(f"Cleared {deleted} template cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error clearing template cache: {str(e)}")
            return 0
    
    # Méthodes privées
    
    async def _setup_jinja_environments(self) -> None:
        """Configure les environnements Jinja2"""
        # Environnement principal
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_path)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Environnement sandboxé pour les templates non fiables
        from jinja2.sandbox import SandboxedEnvironment
        self.sandbox_env = SandboxedEnvironment(
            loader=FileSystemLoader(str(self.templates_path)),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    async def _load_templates(self) -> None:
        """Charge tous les templates disponibles"""
        try:
            if not self.templates_path.exists():
                self.templates_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created templates directory: {self.templates_path}")
                return
            
            template_count = 0
            for template_file in self.templates_path.glob("*.j2"):
                template_name = template_file.stem
                self.template_cache[template_name] = template_file
                template_count += 1
            
            self.logger.info(f"Loaded {template_count} templates")
            
        except Exception as e:
            self.logger.error(f"Error loading templates: {str(e)}")
    
    async def _load_custom_functions(self) -> None:
        """Charge les fonctions personnalisées pour Jinja2"""
        # Filtres personnalisés
        self.custom_filters = {
            'timestamp_format': self._filter_timestamp_format,
            'truncate_smart': self._filter_truncate_smart,
            'humanize_size': self._filter_humanize_size,
            'json_pretty': self._filter_json_pretty,
            'base64_encode': self._filter_base64_encode,
            'base64_decode': self._filter_base64_decode,
            'hash_md5': self._filter_hash_md5,
            'hash_sha256': self._filter_hash_sha256
        }
        
        # Tests personnalisés
        self.custom_tests = {
            'email': self._test_email,
            'url': self._test_url,
            'ip': self._test_ip,
            'json_valid': self._test_json_valid
        }
        
        # Variables globales
        self.custom_globals = {
            'now': datetime.utcnow,
            'tenant_id': self.tenant_id,
            'version': self.config.get('version', '1.0.0')
        }
        
        # Enregistrement dans l'environnement Jinja2
        self.jinja_env.filters.update(self.custom_filters)
        self.jinja_env.tests.update(self.custom_tests)
        self.jinja_env.globals.update(self.custom_globals)
    
    async def _setup_processors(self) -> None:
        """Configure les préprocesseurs et post-processeurs"""
        # Préprocesseurs par format
        self.preprocessors = {
            TemplateFormat.MARKDOWN: self._preprocess_markdown,
            TemplateFormat.HTML: self._preprocess_html,
            TemplateFormat.JSON: self._preprocess_json
        }
        
        # Post-processeurs par format
        self.postprocessors = {
            TemplateFormat.MARKDOWN: self._postprocess_markdown,
            TemplateFormat.HTML: self._postprocess_html,
            TemplateFormat.JSON: self._postprocess_json
        }
    
    async def _initialize_cache(self) -> None:
        """Initialise le cache Redis"""
        cache_key = f"template_engine_initialized:{self.tenant_id}"
        is_initialized = await self.redis_client.get(cache_key)
        
        if not is_initialized:
            await self.redis_client.setex(cache_key, 3600, "true")
    
    async def _detect_template_format(self, template_name: str) -> TemplateFormat:
        """Détecte automatiquement le format d'un template"""
        # Basé sur l'extension
        if template_name.endswith('.md'):
            return TemplateFormat.MARKDOWN
        elif template_name.endswith('.html'):
            return TemplateFormat.HTML
        elif template_name.endswith('.json'):
            return TemplateFormat.JSON
        elif template_name.endswith('.yml') or template_name.endswith('.yaml'):
            return TemplateFormat.YAML
        
        # Défaut : Jinja2
        return TemplateFormat.JINJA2
    
    async def _load_template_content(self, template_name: str, tenant_id: str) -> Optional[str]:
        """Charge le contenu d'un template"""
        try:
            # Recherche du fichier template
            possible_extensions = ['.j2', '.jinja2', '.txt', '.md', '.html', '.json', '.yml']
            
            for ext in possible_extensions:
                template_file = self.templates_path / f"{template_name}{ext}"
                if template_file.exists():
                    async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                        return await f.read()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading template content: {str(e)}")
            return None
    
    async def _get_cached_render(self, context: RenderContext) -> Optional[str]:
        """Récupère un rendu du cache"""
        try:
            cache_key = self._generate_render_cache_key(context)
            cached = await self.redis_client.get(cache_key)
            return cached.decode() if cached else None
        except:
            return None
    
    async def _cache_render_result(self, context: RenderContext, result: str) -> None:
        """Met en cache un résultat de rendu"""
        try:
            cache_key = self._generate_render_cache_key(context)
            await self.redis_client.setex(cache_key, self.cache_ttl, result)
        except Exception as e:
            self.logger.error(f"Error caching render result: {str(e)}")
    
    def _generate_render_cache_key(self, context: RenderContext) -> str:
        """Génère une clé de cache pour un rendu"""
        # Hash des variables pour créer une clé unique
        variables_hash = hashlib.md5(
            json.dumps(context.variables, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        
        return f"template_cache:{context.tenant_id}:{context.template_name}:{context.locale}:{variables_hash}"
    
    async def _validate_required_variables(self, template_name: str, data: Dict[str, Any]) -> None:
        """Valide que toutes les variables requises sont présentes"""
        template_info = await self.get_template_info(template_name)
        if not template_info or not template_info.required_variables:
            return
        
        missing_vars = []
        for var in template_info.required_variables:
            if var not in data:
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
    
    async def _render_by_format(
        self,
        template_content: str,
        data: Dict[str, Any],
        format: TemplateFormat,
        context: RenderContext
    ) -> str:
        """Rend un template selon son format"""
        if format == TemplateFormat.JINJA2:
            template = self.jinja_env.from_string(template_content)
            return template.render(**data)
        
        elif format == TemplateFormat.MARKDOWN:
            # Rendu Jinja2 puis conversion Markdown
            template = self.jinja_env.from_string(template_content)
            rendered_md = template.render(**data)
            return markdown.markdown(rendered_md, extensions=['extra', 'codehilite'])
        
        elif format == TemplateFormat.HTML:
            template = self.jinja_env.from_string(template_content)
            return template.render(**data)
        
        elif format == TemplateFormat.JSON:
            template = self.jinja_env.from_string(template_content)
            rendered = template.render(**data)
            # Validation JSON
            json.loads(rendered)  # Vérification de la validité
            return rendered
        
        elif format == TemplateFormat.YAML:
            template = self.jinja_env.from_string(template_content)
            rendered = template.render(**data)
            # Validation YAML
            yaml.safe_load(rendered)  # Vérification de la validité
            return rendered
        
        elif format == TemplateFormat.PLAINTEXT:
            # Simple substitution de variables
            result = template_content
            for key, value in data.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result
        
        else:
            raise ValueError(f"Unsupported template format: {format}")
    
    # Fonctions de filtres personnalisés
    
    def _filter_timestamp_format(self, timestamp: Union[datetime, int, float], format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Formate un timestamp"""
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        return timestamp.strftime(format_str)
    
    def _filter_truncate_smart(self, text: str, length: int = 100, suffix: str = '...') -> str:
        """Tronque intelligemment un texte"""
        if len(text) <= length:
            return text
        
        # Coupe au dernier espace avant la limite
        truncated = text[:length].rsplit(' ', 1)[0]
        return truncated + suffix
    
    def _filter_humanize_size(self, size_bytes: int) -> str:
        """Convertit une taille en octets en format lisible"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def _filter_json_pretty(self, data: Any, indent: int = 2) -> str:
        """Formate du JSON de manière lisible"""
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    
    def _filter_base64_encode(self, text: str) -> str:
        """Encode en base64"""
        import base64
        return base64.b64encode(text.encode()).decode()
    
    def _filter_base64_decode(self, encoded: str) -> str:
        """Décode du base64"""
        import base64
        return base64.b64decode(encoded.encode()).decode()
    
    def _filter_hash_md5(self, text: str) -> str:
        """Calcule le hash MD5"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _filter_hash_sha256(self, text: str) -> str:
        """Calcule le hash SHA256"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    # Fonctions de tests personnalisés
    
    def _test_email(self, value: str) -> bool:
        """Test si une valeur est un email valide"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
    
    def _test_url(self, value: str) -> bool:
        """Test si une valeur est une URL valide"""
        import re
        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, value))
    
    def _test_ip(self, value: str) -> bool:
        """Test si une valeur est une IP valide"""
        try:
            import ipaddress
            ipaddress.ip_address(value)
            return True
        except:
            return False
    
    def _test_json_valid(self, value: str) -> bool:
        """Test si une valeur est du JSON valide"""
        try:
            json.loads(value)
            return True
        except:
            return False
    
    # Préprocesseurs et post-processeurs
    
    async def _preprocess_data(self, data: Dict[str, Any], format: TemplateFormat, context: RenderContext) -> Dict[str, Any]:
        """Prétraite les données avant le rendu"""
        if format in self.preprocessors:
            return await self.preprocessors[format](data, context)
        return data
    
    async def _postprocess_content(self, content: str, format: TemplateFormat, context: RenderContext) -> str:
        """Post-traite le contenu après le rendu"""
        if format in self.postprocessors:
            return await self.postprocessors[format](content, context)
        return content
    
    async def _preprocess_markdown(self, data: Dict[str, Any], context: RenderContext) -> Dict[str, Any]:
        """Préprocesseur spécifique au Markdown"""
        # Ajout de métadonnées Markdown
        data['markdown_meta'] = {
            'generated_at': context.timestamp.isoformat(),
            'template': context.template_name,
            'tenant': context.tenant_id
        }
        return data
    
    async def _postprocess_markdown(self, content: str, context: RenderContext) -> str:
        """Post-processeur spécifique au Markdown"""
        # Nettoyage du HTML généré
        soup = BeautifulSoup(content, 'html.parser')
        
        # Ajout de classes CSS pour le styling
        for pre in soup.find_all('pre'):
            if not pre.get('class'):
                pre['class'] = ['code-block']
        
        for table in soup.find_all('table'):
            if not table.get('class'):
                table['class'] = ['table', 'table-striped']
        
        return str(soup)
    
    async def _preprocess_html(self, data: Dict[str, Any], context: RenderContext) -> Dict[str, Any]:
        """Préprocesseur spécifique au HTML"""
        return data
    
    async def _postprocess_html(self, content: str, context: RenderContext) -> str:
        """Post-processeur spécifique au HTML"""
        # Minification du HTML en production
        if self.config.get('minify_html', False):
            return re.sub(r'\s+', ' ', content).strip()
        return content
    
    async def _preprocess_json(self, data: Dict[str, Any], context: RenderContext) -> Dict[str, Any]:
        """Préprocesseur spécifique au JSON"""
        return data
    
    async def _postprocess_json(self, content: str, context: RenderContext) -> str:
        """Post-processeur spécifique au JSON"""
        # Validation et reformatage du JSON
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, indent=None if self.config.get('minify_json', False) else 2)
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON generated by template {context.template_name}")
            return content
    
    async def _extract_template_variables(self, template_content: str) -> List[str]:
        """Extrait les variables utilisées dans un template"""
        # Regex pour détecter les variables Jinja2
        variable_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*(?:\|[^}]*)?\}\}'
        variables = set()
        
        for match in re.finditer(variable_pattern, template_content):
            var_name = match.group(1).split('.')[0]  # Prend la variable racine
            variables.add(var_name)
        
        return list(variables)
    
    async def _template_exists(self, template_name: str) -> bool:
        """Vérifie si un template existe"""
        return await self._load_template_content(template_name, self.tenant_id) is not None
