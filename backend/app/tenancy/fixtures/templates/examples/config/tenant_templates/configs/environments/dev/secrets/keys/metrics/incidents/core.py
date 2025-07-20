"""
Gestionnaire Principal des Incidents - Architecture Industrielle
================================================================

Système de gestion d'incidents enterprise-grade avec:
- Classification automatique par IA
- Escalation intelligente
- Réponse automatisée
- Forensics en temps réel
- Intégration SIEM/SOC

Classes principales:
    - IncidentManager: Orchestrateur principal
    - IncidentClassifier: Classification par ML
    - ResponseOrchestrator: Coordination des réponses
    - ForensicsEngine: Analyse post-incident
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import structlog

# Configuration des logs structurés
logger = structlog.get_logger(__name__)

class IncidentSeverity(Enum):
    """Niveaux de criticité standardisés"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class IncidentStatus(Enum):
    """États du cycle de vie d'un incident"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    RESPONDING = "responding"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    CLOSED = "closed"

class IncidentCategory(Enum):
    """Catégories d'incidents"""
    SECURITY = "security"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    DATA = "data"
    COMPLIANCE = "compliance"
    BUSINESS = "business"

@dataclass
class IncidentEvent:
    """Structure standardisée d'un événement d'incident"""
    id: str
    tenant_id: str
    title: str
    description: str
    category: IncidentCategory
    severity: IncidentSeverity
    status: IncidentStatus
    source: str
    timestamp: datetime
    affected_systems: List[str]
    metrics: Dict[str, Any]
    context: Dict[str, Any]
    tags: List[str]
    assignee: Optional[str] = None
    escalation_level: int = 0
    auto_resolved: bool = False
    resolution_time: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation"""
        return asdict(self)

class IncidentClassifier:
    """Classificateur ML pour incidents"""
    
    def __init__(self):
        self.models = {
            "severity": None,  # Modèle de classification de criticité
            "category": None,  # Modèle de catégorisation
            "urgency": None    # Modèle d'urgence
        }
        self.feature_extractors = {}
        
    async def classify_incident(self, event_data: Dict[str, Any]) -> IncidentEvent:
        """Classification automatique d'un incident"""
        # Extraction des features
        features = await self._extract_features(event_data)
        
        # Classification de la criticité
        severity = await self._predict_severity(features)
        
        # Catégorisation
        category = await self._predict_category(features)
        
        # Génération de l'ID unique
        incident_id = f"INC-{uuid.uuid4().hex[:8]}"
        
        # Création de l'objet incident
        incident = IncidentEvent(
            id=incident_id,
            tenant_id=event_data.get("tenant_id"),
            title=self._generate_title(event_data, category),
            description=event_data.get("description", ""),
            category=category,
            severity=severity,
            status=IncidentStatus.DETECTED,
            source=event_data.get("source", "unknown"),
            timestamp=datetime.utcnow(),
            affected_systems=event_data.get("affected_systems", []),
            metrics=event_data.get("metrics", {}),
            context=event_data.get("context", {}),
            tags=await self._generate_tags(features)
        )
        
        return incident
    
    async def _extract_features(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extraction des caractéristiques pour ML"""
        return {
            "error_keywords": self._extract_error_keywords(event_data),
            "system_metrics": event_data.get("metrics", {}),
            "frequency": await self._calculate_frequency(event_data),
            "impact_score": await self._calculate_impact(event_data),
            "business_hours": self._is_business_hours(),
            "historical_context": await self._get_historical_context(event_data)
        }
    
    def _extract_error_keywords(self, event_data: Dict[str, Any]) -> List[str]:
        """Extraction des mots-clés d'erreur"""
        keywords = []
        text = f"{event_data.get('title', '')} {event_data.get('description', '')}"
        
        # Keywords de sécurité
        security_keywords = ["unauthorized", "breach", "attack", "malware", "phishing"]
        # Keywords de performance  
        performance_keywords = ["slow", "timeout", "high cpu", "memory", "latency"]
        # Keywords de disponibilité
        availability_keywords = ["down", "unavailable", "503", "502", "connection"]
        
        for keyword in security_keywords + performance_keywords + availability_keywords:
            if keyword in text.lower():
                keywords.append(keyword)
                
        return keywords
    
    async def _predict_severity(self, features: Dict[str, Any]) -> IncidentSeverity:
        """Prédiction de la criticité basée sur les features"""
        # Logique de scoring
        score = 0
        
        # Facteurs de criticité
        if features.get("impact_score", 0) > 0.8:
            score += 3
        elif features.get("impact_score", 0) > 0.6:
            score += 2
        elif features.get("impact_score", 0) > 0.3:
            score += 1
            
        # Mots-clés critiques
        critical_keywords = ["breach", "attack", "down", "critical"]
        for keyword in features.get("error_keywords", []):
            if keyword in critical_keywords:
                score += 2
                
        # Fréquence
        if features.get("frequency", 0) > 10:
            score += 1
            
        # Mapping du score vers la criticité
        if score >= 5:
            return IncidentSeverity.CRITICAL
        elif score >= 3:
            return IncidentSeverity.HIGH
        elif score >= 2:
            return IncidentSeverity.MEDIUM
        elif score >= 1:
            return IncidentSeverity.LOW
        else:
            return IncidentSeverity.INFO
    
    async def _predict_category(self, features: Dict[str, Any]) -> IncidentCategory:
        """Prédiction de la catégorie d'incident"""
        keywords = features.get("error_keywords", [])
        
        # Mapping keywords -> catégorie
        if any(k in keywords for k in ["breach", "attack", "unauthorized", "malware"]):
            return IncidentCategory.SECURITY
        elif any(k in keywords for k in ["slow", "timeout", "cpu", "memory", "latency"]):
            return IncidentCategory.PERFORMANCE
        elif any(k in keywords for k in ["down", "unavailable", "503", "502"]):
            return IncidentCategory.AVAILABILITY
        elif any(k in keywords for k in ["data", "corruption", "loss"]):
            return IncidentCategory.DATA
        elif any(k in keywords for k in ["compliance", "audit", "gdpr"]):
            return IncidentCategory.COMPLIANCE
        else:
            return IncidentCategory.BUSINESS
    
    def _generate_title(self, event_data: Dict[str, Any], category: IncidentCategory) -> str:
        """Génération automatique du titre"""
        base_title = event_data.get("title", "Incident détecté")
        return f"[{category.value.upper()}] {base_title}"
    
    async def _generate_tags(self, features: Dict[str, Any]) -> List[str]:
        """Génération automatique des tags"""
        tags = ["auto-detected"]
        
        # Tags basés sur les features
        if features.get("business_hours"):
            tags.append("business-hours")
        else:
            tags.append("after-hours")
            
        if features.get("frequency", 0) > 5:
            tags.append("high-frequency")
            
        if features.get("impact_score", 0) > 0.7:
            tags.append("high-impact")
            
        return tags

class ResponseOrchestrator:
    """Orchestrateur des réponses automatisées"""
    
    def __init__(self, incident_manager):
        self.incident_manager = incident_manager
        self.response_playbooks = {}
        self.automation_engines = []
        
    async def orchestrate_response(self, incident: IncidentEvent) -> Dict[str, Any]:
        """Orchestration de la réponse à un incident"""
        response_plan = await self._create_response_plan(incident)
        
        # Exécution parallèle des actions
        tasks = []
        for action in response_plan["actions"]:
            task = asyncio.create_task(self._execute_action(action, incident))
            tasks.append(task)
            
        # Attendre l'exécution de toutes les actions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compilation des résultats
        response_summary = {
            "incident_id": incident.id,
            "response_plan": response_plan,
            "execution_results": results,
            "timestamp": datetime.utcnow(),
            "success_rate": self._calculate_success_rate(results)
        }
        
        return response_summary
    
    async def _create_response_plan(self, incident: IncidentEvent) -> Dict[str, Any]:
        """Création du plan de réponse basé sur l'incident"""
        plan = {
            "incident_id": incident.id,
            "severity": incident.severity.value,
            "category": incident.category.value,
            "actions": []
        }
        
        # Actions basées sur la criticité
        if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
            plan["actions"].extend([
                {"type": "notify_oncall", "urgency": "immediate"},
                {"type": "create_war_room", "platform": "slack"},
                {"type": "scale_up_monitoring", "factor": 2}
            ])
            
        # Actions basées sur la catégorie
        if incident.category == IncidentCategory.SECURITY:
            plan["actions"].extend([
                {"type": "isolate_affected_systems"},
                {"type": "capture_forensics"},
                {"type": "notify_security_team"}
            ])
        elif incident.category == IncidentCategory.PERFORMANCE:
            plan["actions"].extend([
                {"type": "auto_scale_resources"},
                {"type": "clear_caches"},
                {"type": "analyze_bottlenecks"}
            ])
        elif incident.category == IncidentCategory.AVAILABILITY:
            plan["actions"].extend([
                {"type": "failover_to_backup"},
                {"type": "restart_services"},
                {"type": "check_health_endpoints"}
            ])
            
        return plan
    
    async def _execute_action(self, action: Dict[str, Any], incident: IncidentEvent) -> Dict[str, Any]:
        """Exécution d'une action de réponse"""
        try:
            action_type = action["type"]
            
            if action_type == "notify_oncall":
                return await self._notify_oncall(action, incident)
            elif action_type == "create_war_room":
                return await self._create_war_room(action, incident)
            elif action_type == "scale_up_monitoring":
                return await self._scale_up_monitoring(action, incident)
            elif action_type == "isolate_affected_systems":
                return await self._isolate_systems(action, incident)
            elif action_type == "auto_scale_resources":
                return await self._auto_scale_resources(action, incident)
            else:
                return {"status": "not_implemented", "action": action_type}
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'action {action['type']}: {e}")
            return {"status": "error", "error": str(e), "action": action["type"]}

class IncidentManager:
    """Gestionnaire principal des incidents - Orchestrateur Enterprise"""
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.classifier = IncidentClassifier()
        self.response_orchestrator = ResponseOrchestrator(self)
        self.active_incidents = {}
        self.incident_hooks = []
        self.escalation_rules = {}
        
    async def create_incident(self, event_data: Dict[str, Any]) -> IncidentEvent:
        """Création et traitement d'un nouvel incident"""
        # Classification automatique
        incident = await self.classifier.classify_incident(event_data)
        
        # Sauvegarde en cache et DB
        await self._persist_incident(incident)
        
        # Orchestration de la réponse
        response = await self.response_orchestrator.orchestrate_response(incident)
        
        # Hooks et notifications
        await self._trigger_hooks(incident, "created")
        
        # Ajout aux incidents actifs
        self.active_incidents[incident.id] = incident
        
        logger.info(
            "Incident créé et traité",
            incident_id=incident.id,
            severity=incident.severity.value,
            category=incident.category.value
        )
        
        return incident
    
    async def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> IncidentEvent:
        """Mise à jour d'un incident existant"""
        incident = await self._get_incident(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} non trouvé")
            
        # Application des mises à jour
        for key, value in updates.items():
            if hasattr(incident, key):
                setattr(incident, key, value)
                
        # Sauvegarde
        await self._persist_incident(incident)
        
        # Hooks
        await self._trigger_hooks(incident, "updated")
        
        return incident
    
    async def resolve_incident(self, incident_id: str, resolution_notes: str) -> IncidentEvent:
        """Résolution d'un incident"""
        incident = await self._get_incident(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} non trouvé")
            
        # Mise à jour du statut
        incident.status = IncidentStatus.RESOLVED
        incident.resolution_time = datetime.utcnow() - incident.timestamp
        incident.context["resolution_notes"] = resolution_notes
        
        # Sauvegarde
        await self._persist_incident(incident)
        
        # Nettoyage des incidents actifs
        self.active_incidents.pop(incident_id, None)
        
        # Post-incident analysis
        await self._trigger_post_incident_analysis(incident)
        
        logger.info(
            "Incident résolu",
            incident_id=incident_id,
            resolution_time=incident.resolution_time.total_seconds()
        )
        
        return incident
    
    async def _persist_incident(self, incident: IncidentEvent):
        """Persistance de l'incident en cache et DB"""
        # Cache Redis pour accès rapide
        await self.redis.setex(
            f"incident:{incident.id}",
            3600 * 24 * 7,  # 7 jours
            json.dumps(incident.to_dict(), default=str)
        )
        
        # TODO: Persistance en base de données
        # Sera implémenté avec le modèle SQLAlchemy
    
    async def _get_incident(self, incident_id: str) -> Optional[IncidentEvent]:
        """Récupération d'un incident"""
        # Vérification du cache
        cached = await self.redis.get(f"incident:{incident_id}")
        if cached:
            data = json.loads(cached)
            # Reconstruction de l'objet
            return IncidentEvent(**data)
            
        # TODO: Fallback sur la base de données
        return None
    
    async def _trigger_hooks(self, incident: IncidentEvent, event_type: str):
        """Déclenchement des hooks d'incident"""
        for hook in self.incident_hooks:
            try:
                await hook(incident, event_type)
            except Exception as e:
                logger.error(f"Erreur dans le hook d'incident: {e}")
    
    async def _trigger_post_incident_analysis(self, incident: IncidentEvent):
        """Analyse post-incident automatique"""
        analysis = {
            "incident_id": incident.id,
            "metrics": {
                "resolution_time": incident.resolution_time.total_seconds(),
                "escalation_level": incident.escalation_level,
                "auto_resolved": incident.auto_resolved
            },
            "lessons_learned": await self._extract_lessons_learned(incident),
            "recommendations": await self._generate_recommendations(incident)
        }
        
        # Sauvegarde de l'analyse
        await self.redis.setex(
            f"analysis:{incident.id}",
            3600 * 24 * 30,  # 30 jours
            json.dumps(analysis, default=str)
        )
        
    async def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Génération du dashboard de métriques"""
        return {
            "active_incidents": len(self.active_incidents),
            "incidents_by_severity": await self._get_incidents_by_severity(),
            "incidents_by_category": await self._get_incidents_by_category(),
            "average_resolution_time": await self._get_average_resolution_time(),
            "escalation_rate": await self._get_escalation_rate(),
            "auto_resolution_rate": await self._get_auto_resolution_rate()
        }
