"""
Slack Template Manager - Gestionnaire de templates multi-tenant
Gestion avancée des templates Slack avec isolation par tenant
"""

import json
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from dataclasses import dataclass, asdict
from enum import Enum
import redis
import yaml

from .template_engine import SlackTemplateEngine, AlertContext, AlertSeverity
from .template_validator import SlackTemplateValidator
from .locale_manager import LocaleManager


class TemplateType(Enum):
    """Types de templates disponibles"""
    ALERT = "alert"
    RECOVERY = "recovery" 
    SILENCE = "silence"
    MAINTENANCE = "maintenance"
    CUSTOM = "custom"


@dataclass
class TenantTemplate:
    """Représentation d'un template tenant"""
    template_id: str
    tenant_id: str
    name: str
    type: TemplateType
    content: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str
    is_active: bool = True
    locale_specific: bool = False
    priority: int = 0


@dataclass
class TemplateMetrics:
    """Métriques d'utilisation d'un template"""
    template_id: str
    usage_count: int
    last_used: datetime
    avg_response_time: float
    error_rate: float
    tenant_id: str


class SlackTemplateManager:
    """
    Gestionnaire avancé de templates Slack multi-tenant
    
    Fonctionnalités :
    - Gestion des templates par tenant
    - Versioning des templates
    - A/B testing de templates
    - Métriques et analytics
    - Import/Export de templates
    - Validation automatique
    """

    def __init__(
        self,
        templates_base_dir: str = "/templates",
        redis_client: Optional[redis.Redis] = None,
        engine: Optional[SlackTemplateEngine] = None,
        validator: Optional[SlackTemplateValidator] = None,
        locale_manager: Optional[LocaleManager] = None
    ):
        self.templates_base_dir = Path(templates_base_dir)
        self.redis_client = redis_client
        self.engine = engine or SlackTemplateEngine()
        self.validator = validator or SlackTemplateValidator()
        self.locale_manager = locale_manager or LocaleManager()
        
        # Pool d'exécution pour opérations asynchrones
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialisation des répertoires
        self._ensure_directories()

    def _ensure_directories(self):
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        directories = [
            self.templates_base_dir / "tenants",
            self.templates_base_dir / "shared",
            self.templates_base_dir / "custom",
            self.templates_base_dir / "backups",
            self.templates_base_dir / "metrics"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_tenant_template_dir(self, tenant_id: str) -> Path:
        """Retourne le répertoire des templates d'un tenant"""
        return self.templates_base_dir / "tenants" / tenant_id

    def _generate_template_id(self, tenant_id: str, name: str, type_: TemplateType) -> str:
        """Génère un ID unique pour un template"""
        content = f"{tenant_id}:{name}:{type_.value}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def create_tenant_template(
        self,
        tenant_id: str,
        name: str,
        template_type: TemplateType,
        content: Dict[str, Any],
        locale_specific: bool = False,
        priority: int = 0
    ) -> TenantTemplate:
        """
        Crée un nouveau template pour un tenant
        
        Args:
            tenant_id: ID du tenant
            name: Nom du template
            template_type: Type de template
            content: Contenu du template
            locale_specific: Si le template est spécifique à une locale
            priority: Priorité du template (plus élevé = plus prioritaire)
            
        Returns:
            TenantTemplate: Le template créé
        """
        
        # Validation du contenu
        if not self.validator.validate_template_content(content):
            raise ValueError("Le contenu du template n'est pas valide")
        
        # Génération de l'ID
        template_id = self._generate_template_id(tenant_id, name, template_type)
        
        # Création de l'objet template
        template = TenantTemplate(
            template_id=template_id,
            tenant_id=tenant_id,
            name=name,
            type=template_type,
            content=content,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="1.0.0",
            locale_specific=locale_specific,
            priority=priority
        )
        
        # Sauvegarde
        self._save_template(template)
        
        # Mise en cache
        self._cache_template(template)
        
        self.logger.info(f"Template créé: {template_id} pour tenant {tenant_id}")
        
        return template

    def _save_template(self, template: TenantTemplate):
        """Sauvegarde un template sur disque"""
        tenant_dir = self._get_tenant_template_dir(template.tenant_id)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = tenant_dir / f"{template.template_id}.yaml"
        
        template_data = asdict(template)
        template_data['created_at'] = template.created_at.isoformat()
        template_data['updated_at'] = template.updated_at.isoformat()
        template_data['type'] = template.type.value
        
        with open(template_file, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, default_flow_style=False, allow_unicode=True)

    def _cache_template(self, template: TenantTemplate):
        """Met en cache un template dans Redis"""
        if not self.redis_client:
            return
        
        cache_key = f"template:{template.tenant_id}:{template.template_id}"
        
        try:
            self.redis_client.setex(
                cache_key,
                3600,  # 1 heure
                json.dumps(asdict(template), default=str)
            )
        except Exception as e:
            self.logger.warning(f"Erreur lors de la mise en cache du template: {e}")

    def get_tenant_templates(
        self,
        tenant_id: str,
        template_type: Optional[TemplateType] = None,
        active_only: bool = True
    ) -> List[TenantTemplate]:
        """
        Récupère les templates d'un tenant
        
        Args:
            tenant_id: ID du tenant
            template_type: Type de template à filtrer (optionnel)
            active_only: Récupérer uniquement les templates actifs
            
        Returns:
            List[TenantTemplate]: Liste des templates
        """
        
        templates = []
        tenant_dir = self._get_tenant_template_dir(tenant_id)
        
        if not tenant_dir.exists():
            return templates
        
        for template_file in tenant_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                
                # Reconstruction de l'objet TenantTemplate
                template = TenantTemplate(
                    template_id=template_data['template_id'],
                    tenant_id=template_data['tenant_id'],
                    name=template_data['name'],
                    type=TemplateType(template_data['type']),
                    content=template_data['content'],
                    created_at=datetime.fromisoformat(template_data['created_at']),
                    updated_at=datetime.fromisoformat(template_data['updated_at']),
                    version=template_data['version'],
                    is_active=template_data.get('is_active', True),
                    locale_specific=template_data.get('locale_specific', False),
                    priority=template_data.get('priority', 0)
                )
                
                # Filtrage
                if active_only and not template.is_active:
                    continue
                
                if template_type and template.type != template_type:
                    continue
                
                templates.append(template)
                
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement du template {template_file}: {e}")
                continue
        
        # Tri par priorité (plus élevé en premier)
        templates.sort(key=lambda t: t.priority, reverse=True)
        
        return templates

    def update_template(
        self,
        template_id: str,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> Optional[TenantTemplate]:
        """
        Met à jour un template existant
        
        Args:
            template_id: ID du template
            tenant_id: ID du tenant
            updates: Dictionnaire des champs à mettre à jour
            
        Returns:
            TenantTemplate: Template mis à jour ou None si non trouvé
        """
        
        template = self.get_template_by_id(template_id, tenant_id)
        if not template:
            return None
        
        # Validation des mises à jour
        if 'content' in updates:
            if not self.validator.validate_template_content(updates['content']):
                raise ValueError("Le nouveau contenu du template n'est pas valide")
        
        # Application des mises à jour
        for key, value in updates.items():
            if hasattr(template, key):
                setattr(template, key, value)
        
        # Mise à jour du timestamp et version
        template.updated_at = datetime.now()
        template.version = self._increment_version(template.version)
        
        # Sauvegarde
        self._save_template(template)
        self._cache_template(template)
        
        self.logger.info(f"Template mis à jour: {template_id}")
        
        return template

    def get_template_by_id(
        self,
        template_id: str,
        tenant_id: str
    ) -> Optional[TenantTemplate]:
        """Récupère un template par son ID"""
        
        # Vérification du cache
        if self.redis_client:
            cache_key = f"template:{tenant_id}:{template_id}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                try:
                    template_data = json.loads(cached_data.decode('utf-8'))
                    return self._dict_to_template(template_data)
                except Exception as e:
                    self.logger.warning(f"Erreur lors de la désérialisation du cache: {e}")
        
        # Recherche sur disque
        templates = self.get_tenant_templates(tenant_id, active_only=False)
        for template in templates:
            if template.template_id == template_id:
                self._cache_template(template)
                return template
        
        return None

    def _dict_to_template(self, template_data: Dict[str, Any]) -> TenantTemplate:
        """Convertit un dictionnaire en objet TenantTemplate"""
        return TenantTemplate(
            template_id=template_data['template_id'],
            tenant_id=template_data['tenant_id'],
            name=template_data['name'],
            type=TemplateType(template_data['type']),
            content=template_data['content'],
            created_at=datetime.fromisoformat(template_data['created_at']),
            updated_at=datetime.fromisoformat(template_data['updated_at']),
            version=template_data['version'],
            is_active=template_data.get('is_active', True),
            locale_specific=template_data.get('locale_specific', False),
            priority=template_data.get('priority', 0)
        )

    def _increment_version(self, current_version: str) -> str:
        """Incrémente la version d'un template"""
        try:
            major, minor, patch = map(int, current_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except ValueError:
            return "1.0.1"

    def delete_template(self, template_id: str, tenant_id: str) -> bool:
        """
        Supprime un template (soft delete)
        
        Args:
            template_id: ID du template
            tenant_id: ID du tenant
            
        Returns:
            bool: True si supprimé avec succès
        """
        
        template = self.get_template_by_id(template_id, tenant_id)
        if not template:
            return False
        
        # Soft delete
        template.is_active = False
        template.updated_at = datetime.now()
        
        self._save_template(template)
        
        # Suppression du cache
        if self.redis_client:
            cache_key = f"template:{tenant_id}:{template_id}"
            self.redis_client.delete(cache_key)
        
        self.logger.info(f"Template supprimé (soft): {template_id}")
        
        return True

    def clone_template(
        self,
        source_template_id: str,
        source_tenant_id: str,
        target_tenant_id: str,
        new_name: Optional[str] = None
    ) -> Optional[TenantTemplate]:
        """Clone un template d'un tenant vers un autre"""
        
        source_template = self.get_template_by_id(source_template_id, source_tenant_id)
        if not source_template:
            return None
        
        name = new_name or f"{source_template.name}_clone"
        
        return self.create_tenant_template(
            tenant_id=target_tenant_id,
            name=name,
            template_type=source_template.type,
            content=source_template.content.copy(),
            locale_specific=source_template.locale_specific,
            priority=source_template.priority
        )

    def get_template_metrics(self, template_id: str, tenant_id: str) -> Optional[TemplateMetrics]:
        """Récupère les métriques d'utilisation d'un template"""
        
        metrics_file = (
            self.templates_base_dir / "metrics" / tenant_id / f"{template_id}_metrics.json"
        )
        
        if not metrics_file.exists():
            return None
        
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            return TemplateMetrics(
                template_id=metrics_data['template_id'],
                usage_count=metrics_data['usage_count'],
                last_used=datetime.fromisoformat(metrics_data['last_used']),
                avg_response_time=metrics_data['avg_response_time'],
                error_rate=metrics_data['error_rate'],
                tenant_id=metrics_data['tenant_id']
            )
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des métriques: {e}")
            return None

    def update_template_metrics(
        self,
        template_id: str,
        tenant_id: str,
        response_time: float,
        success: bool = True
    ):
        """Met à jour les métriques d'utilisation d'un template"""
        
        metrics = self.get_template_metrics(template_id, tenant_id)
        
        if metrics is None:
            # Création de nouvelles métriques
            metrics = TemplateMetrics(
                template_id=template_id,
                usage_count=1,
                last_used=datetime.now(),
                avg_response_time=response_time,
                error_rate=0.0 if success else 1.0,
                tenant_id=tenant_id
            )
        else:
            # Mise à jour des métriques existantes
            metrics.usage_count += 1
            metrics.last_used = datetime.now()
            
            # Calcul de la moyenne mobile pour le temps de réponse
            metrics.avg_response_time = (
                (metrics.avg_response_time * (metrics.usage_count - 1) + response_time)
                / metrics.usage_count
            )
            
            # Mise à jour du taux d'erreur
            if not success:
                error_count = metrics.error_rate * (metrics.usage_count - 1) + 1
                metrics.error_rate = error_count / metrics.usage_count
            else:
                error_count = metrics.error_rate * (metrics.usage_count - 1)
                metrics.error_rate = error_count / metrics.usage_count
        
        # Sauvegarde des métriques
        metrics_dir = self.templates_base_dir / "metrics" / tenant_id
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = metrics_dir / f"{template_id}_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, default=str, indent=2)

    async def batch_validate_templates(self, tenant_id: str) -> Dict[str, List[str]]:
        """Valide tous les templates d'un tenant en parallèle"""
        
        templates = self.get_tenant_templates(tenant_id, active_only=False)
        
        validation_tasks = []
        for template in templates:
            task = asyncio.create_task(
                self._validate_template_async(template)
            )
            validation_tasks.append((template.template_id, task))
        
        results = {"valid": [], "invalid": [], "errors": []}
        
        for template_id, task in validation_tasks:
            try:
                is_valid = await task
                if is_valid:
                    results["valid"].append(template_id)
                else:
                    results["invalid"].append(template_id)
            except Exception as e:
                results["errors"].append(f"{template_id}: {str(e)}")
        
        return results

    async def _validate_template_async(self, template: TenantTemplate) -> bool:
        """Valide un template de manière asynchrone"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.validator.validate_template_content,
            template.content
        )

    def export_tenant_templates(self, tenant_id: str, output_file: str):
        """Exporte tous les templates d'un tenant vers un fichier"""
        
        templates = self.get_tenant_templates(tenant_id, active_only=False)
        
        export_data = {
            "tenant_id": tenant_id,
            "export_date": datetime.now().isoformat(),
            "templates": [asdict(template) for template in templates]
        }
        
        # Conversion des dates et enums en chaînes
        for template_data in export_data["templates"]:
            template_data['created_at'] = template_data['created_at'].isoformat()
            template_data['updated_at'] = template_data['updated_at'].isoformat()
            template_data['type'] = template_data['type'].value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Templates exportés vers {output_file}")

    def import_tenant_templates(self, tenant_id: str, import_file: str) -> int:
        """Importe des templates depuis un fichier"""
        
        with open(import_file, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        imported_count = 0
        
        for template_data in import_data.get("templates", []):
            try:
                # Reconstruction du template
                template = TenantTemplate(
                    template_id=template_data['template_id'],
                    tenant_id=tenant_id,  # Forcer le tenant_id cible
                    name=template_data['name'],
                    type=TemplateType(template_data['type']),
                    content=template_data['content'],
                    created_at=datetime.fromisoformat(template_data['created_at']),
                    updated_at=datetime.now(),  # Mise à jour du timestamp
                    version=template_data['version'],
                    is_active=template_data.get('is_active', True),
                    locale_specific=template_data.get('locale_specific', False),
                    priority=template_data.get('priority', 0)
                )
                
                # Sauvegarde
                self._save_template(template)
                self._cache_template(template)
                
                imported_count += 1
                
            except Exception as e:
                self.logger.error(f"Erreur lors de l'import du template {template_data.get('template_id')}: {e}")
                continue
        
        self.logger.info(f"{imported_count} templates importés pour le tenant {tenant_id}")
        
        return imported_count
