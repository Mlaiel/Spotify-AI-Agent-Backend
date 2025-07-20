"""
🔐 Security Policy Engine - Moteur de Politiques de Sécurité
===========================================================

Système ultra-avancé de gestion des politiques de sécurité pour l'isolation
des données avec enforcement en temps réel et adaptation dynamique.

Author: Spécialiste Sécurité Backend - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Set, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
import hashlib
from abc import ABC, abstractmethod
import re

from .tenant_context import TenantContext, SecurityContext, IsolationLevel
from .compliance_engine import ComplianceRegulation, DataClassification
from ..exceptions import SecurityPolicyViolationError, PolicyEngineError
from ...core.config import settings
from ...utils.encryption import EncryptionManager
from ...monitoring.security_monitor import SecurityMonitor


class PolicyType(Enum):
    """Types de politiques de sécurité"""
    ACCESS_CONTROL = "access_control"
    DATA_ENCRYPTION = "data_encryption"
    DATA_MASKING = "data_masking"
    AUDIT_LOGGING = "audit_logging"
    RATE_LIMITING = "rate_limiting"
    IP_RESTRICTION = "ip_restriction"
    TIME_BASED = "time_based"
    GEOGRAPHIC = "geographic"
    MFA_ENFORCEMENT = "mfa_enforcement"
    SESSION_MANAGEMENT = "session_management"


class PolicyScope(Enum):
    """Portée des politiques"""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    RESOURCE = "resource"
    OPERATION = "operation"


class PolicyEnforcement(Enum):
    """Modes d'application des politiques"""
    BLOCK = "block"          # Bloque l'opération
    WARN = "warn"            # Avertissement seulement
    LOG = "log"              # Log seulement
    MODIFY = "modify"        # Modifie l'opération
    REDIRECT = "redirect"    # Redirige vers une action alternative


class ThreatLevel(Enum):
    """Niveaux de menace"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Politique de sécurité"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope
    enforcement: PolicyEnforcement
    
    # Conditions d'application
    conditions: Dict[str, Any] = field(default_factory=dict)
    target_resources: Set[str] = field(default_factory=set)
    target_operations: Set[str] = field(default_factory=set)
    target_users: Set[str] = field(default_factory=set)
    target_tenants: Set[str] = field(default_factory=set)
    
    # Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    exceptions: List[str] = field(default_factory=list)
    
    # Temporalité
    effective_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    effective_until: Optional[datetime] = None
    
    # Métadonnées
    priority: int = 100
    active: bool = True
    version: str = "1.0"
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PolicyEvaluation:
    """Résultat d'évaluation de politique"""
    policy_id: str
    applicable: bool
    result: str  # allow, deny, modify, warn
    enforcement_action: Optional[str] = None
    modifications: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ThreatContext:
    """Contexte de menace détecté"""
    threat_id: str
    threat_type: str
    level: ThreatLevel
    indicators: List[str]
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    patterns_matched: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PolicyEngine(ABC):
    """Interface pour les moteurs de politique"""
    
    @abstractmethod
    async def evaluate(
        self,
        policy: SecurityPolicy,
        context: TenantContext,
        operation: str,
        data: Dict[str, Any]
    ) -> PolicyEvaluation:
        """Évalue une politique"""
        pass


class AccessControlEngine(PolicyEngine):
    """Moteur de contrôle d'accès"""
    
    async def evaluate(
        self,
        policy: SecurityPolicy,
        context: TenantContext,
        operation: str,
        data: Dict[str, Any]
    ) -> PolicyEvaluation:
        """Évalue les politiques de contrôle d'accès"""
        
        # Vérification des permissions
        required_permissions = policy.parameters.get('required_permissions', [])
        user_permissions = context.security.permissions if context.security else set()
        
        has_permission = all(perm in user_permissions for perm in required_permissions)
        
        # Vérification des rôles
        required_roles = policy.parameters.get('required_roles', [])
        user_roles = context.security.roles if context.security else set()
        
        has_role = any(role in user_roles for role in required_roles) if required_roles else True
        
        # Vérification du niveau d'accès
        required_level = policy.parameters.get('required_access_level', 'basic')
        user_level = context.security.access_level if context.security else 'basic'
        
        access_levels = ['basic', 'standard', 'premium', 'admin', 'super_admin']
        has_level = access_levels.index(user_level) >= access_levels.index(required_level)
        
        allowed = has_permission and has_role and has_level
        
        return PolicyEvaluation(
            policy_id=policy.policy_id,
            applicable=True,
            result="allow" if allowed else "deny",
            context={
                'permissions_check': has_permission,
                'roles_check': has_role,
                'level_check': has_level
            }
        )


class DataEncryptionEngine(PolicyEngine):
    """Moteur de chiffrement des données"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
    
    async def evaluate(
        self,
        policy: SecurityPolicy,
        context: TenantContext,
        operation: str,
        data: Dict[str, Any]
    ) -> PolicyEvaluation:
        """Évalue les politiques de chiffrement"""
        
        encryption_fields = policy.parameters.get('fields', [])
        encryption_level = policy.parameters.get('level', 'standard')
        
        modifications = {}
        
        for field in encryption_fields:
            if field in data:
                encrypted_value = await self.encryption_manager.encrypt(
                    str(data[field]),
                    level=encryption_level
                )
                modifications[field] = encrypted_value
        
        return PolicyEvaluation(
            policy_id=policy.policy_id,
            applicable=len(modifications) > 0,
            result="modify" if modifications else "allow",
            modifications=modifications
        )


class DataMaskingEngine(PolicyEngine):
    """Moteur de masquage des données"""
    
    async def evaluate(
        self,
        policy: SecurityPolicy,
        context: TenantContext,
        operation: str,
        data: Dict[str, Any]
    ) -> PolicyEvaluation:
        """Évalue les politiques de masquage"""
        
        masking_rules = policy.parameters.get('rules', {})
        modifications = {}
        
        for field, rule in masking_rules.items():
            if field in data:
                masked_value = self._apply_masking_rule(data[field], rule)
                modifications[field] = masked_value
        
        return PolicyEvaluation(
            policy_id=policy.policy_id,
            applicable=len(modifications) > 0,
            result="modify" if modifications else "allow",
            modifications=modifications
        )
    
    def _apply_masking_rule(self, value: Any, rule: Dict[str, Any]) -> Any:
        """Applique une règle de masquage"""
        rule_type = rule.get('type', 'partial')
        
        if rule_type == 'full':
            return "*" * len(str(value))
        elif rule_type == 'partial':
            text = str(value)
            visible_chars = rule.get('visible_chars', 4)
            if len(text) <= visible_chars:
                return text
            return text[:visible_chars//2] + "*" * (len(text) - visible_chars) + text[-visible_chars//2:]
        elif rule_type == 'email':
            if '@' in str(value):
                local, domain = str(value).split('@', 1)
                return f"{local[:2]}***@{domain}"
        
        return value


class RateLimitingEngine(PolicyEngine):
    """Moteur de limitation de débit"""
    
    def __init__(self):
        self.rate_counters: Dict[str, Dict[str, Any]] = {}
    
    async def evaluate(
        self,
        policy: SecurityPolicy,
        context: TenantContext,
        operation: str,
        data: Dict[str, Any]
    ) -> PolicyEvaluation:
        """Évalue les politiques de limitation de débit"""
        
        rate_limit = policy.parameters.get('requests_per_minute', 100)
        window_size = timedelta(minutes=1)
        
        # Clé de limitation basée sur le tenant et l'opération
        rate_key = f"{context.tenant_id}:{operation}"
        current_time = datetime.now(timezone.utc)
        
        if rate_key not in self.rate_counters:
            self.rate_counters[rate_key] = {
                'count': 0,
                'window_start': current_time
            }
        
        counter = self.rate_counters[rate_key]
        
        # Réinitialiser si nouvelle fenêtre
        if current_time - counter['window_start'] > window_size:
            counter['count'] = 0
            counter['window_start'] = current_time
        
        counter['count'] += 1
        
        exceeded = counter['count'] > rate_limit
        
        return PolicyEvaluation(
            policy_id=policy.policy_id,
            applicable=True,
            result="deny" if exceeded else "allow",
            context={
                'current_count': counter['count'],
                'rate_limit': rate_limit,
                'window_start': counter['window_start'].isoformat()
            }
        )


class SecurityPolicyEngine:
    """
    Moteur de politiques de sécurité ultra-avancé
    
    Features:
    - Multi-policy evaluation
    - Real-time threat detection
    - Dynamic policy adaptation
    - Performance optimization
    - Audit trail integration
    - Context-aware enforcement
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_monitor = SecurityMonitor()
        
        # Politiques actives
        self.policies: Dict[str, SecurityPolicy] = {}
        
        # Moteurs spécialisés
        self.engines: Dict[PolicyType, PolicyEngine] = {
            PolicyType.ACCESS_CONTROL: AccessControlEngine(),
            PolicyType.DATA_ENCRYPTION: DataEncryptionEngine(),
            PolicyType.DATA_MASKING: DataMaskingEngine(),
            PolicyType.RATE_LIMITING: RateLimitingEngine()
        }
        
        # Cache des évaluations
        self._evaluation_cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # 1 minute
        
        # Détection de menaces
        self.threat_patterns = self._initialize_threat_patterns()
        self.active_threats: Dict[str, ThreatContext] = {}
        
        # Statistiques
        self.statistics = {
            'evaluations_total': 0,
            'policies_enforced': 0,
            'threats_detected': 0,
            'violations_blocked': 0
        }
        
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialise les politiques par défaut"""
        
        # Politique de contrôle d'accès admin
        admin_access = SecurityPolicy(
            policy_id="SEC_001",
            name="Admin Access Control",
            description="Contrôle d'accès pour les opérations administratives",
            policy_type=PolicyType.ACCESS_CONTROL,
            scope=PolicyScope.OPERATION,
            enforcement=PolicyEnforcement.BLOCK,
            target_operations={'admin_create', 'admin_delete', 'admin_modify'},
            parameters={
                'required_roles': ['admin', 'super_admin'],
                'required_access_level': 'admin'
            }
        )
        
        self.policies[admin_access.policy_id] = admin_access
        
        # Politique de chiffrement des données sensibles
        data_encryption = SecurityPolicy(
            policy_id="SEC_002",
            name="Sensitive Data Encryption",
            description="Chiffrement automatique des données sensibles",
            policy_type=PolicyType.DATA_ENCRYPTION,
            scope=PolicyScope.GLOBAL,
            enforcement=PolicyEnforcement.MODIFY,
            parameters={
                'fields': ['password', 'ssn', 'credit_card', 'api_key'],
                'level': 'high'
            }
        )
        
        self.policies[data_encryption.policy_id] = data_encryption
        
        # Politique de limitation de débit
        rate_limit = SecurityPolicy(
            policy_id="SEC_003",
            name="API Rate Limiting",
            description="Limitation du débit d'appels API",
            policy_type=PolicyType.RATE_LIMITING,
            scope=PolicyScope.TENANT,
            enforcement=PolicyEnforcement.BLOCK,
            parameters={
                'requests_per_minute': 1000
            }
        )
        
        self.policies[rate_limit.policy_id] = rate_limit
    
    def _initialize_threat_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les patterns de détection de menaces"""
        return {
            'sql_injection': {
                'patterns': [
                    r"(?i)(union|select|insert|delete|drop|create|alter)\s+",
                    r"(?i)';.*--",
                    r"(?i)\bor\b.*=.*\bor\b"
                ],
                'threat_level': ThreatLevel.HIGH
            },
            'xss_attack': {
                'patterns': [
                    r"<script[^>]*>.*</script>",
                    r"javascript:",
                    r"on\w+\s*="
                ],
                'threat_level': ThreatLevel.MEDIUM
            },
            'path_traversal': {
                'patterns': [
                    r"\.\./",
                    r"\.\.\\",
                    r"%2e%2e%2f"
                ],
                'threat_level': ThreatLevel.HIGH
            }
        }
    
    async def evaluate_policies(
        self,
        context: TenantContext,
        operation: str,
        data: Dict[str, Any],
        request_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Évalue toutes les politiques applicables
        
        Args:
            context: Contexte du tenant
            operation: Opération demandée
            data: Données de l'opération
            request_context: Contexte de la requête (IP, User-Agent, etc.)
            
        Returns:
            Résultat de l'évaluation des politiques
        """
        self.statistics['evaluations_total'] += 1
        
        # Détection de menaces
        threat_context = await self._detect_threats(data, request_context or {})
        
        # Obtenir les politiques applicables
        applicable_policies = self._get_applicable_policies(context, operation, data)
        
        evaluations = []
        final_result = "allow"
        modifications = {}
        warnings = []
        blocked_by = []
        
        # Évaluer chaque politique
        for policy in applicable_policies:
            if policy.policy_type in self.engines:
                engine = self.engines[policy.policy_type]
                evaluation = await engine.evaluate(policy, context, operation, data)
                evaluations.append(evaluation)
                
                # Traiter le résultat
                if evaluation.result == "deny":
                    final_result = "deny"
                    blocked_by.append(policy.policy_id)
                    
                    if policy.enforcement == PolicyEnforcement.BLOCK:
                        break  # Arrêt immédiat si blocage
                
                elif evaluation.result == "modify":
                    modifications.update(evaluation.modifications)
                
                elif evaluation.result == "warn":
                    warnings.extend(evaluation.warnings)
        
        # Application des modifications
        if modifications:
            data.update(modifications)
        
        # Gestion des menaces détectées
        if threat_context and threat_context.level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            final_result = "deny"
            blocked_by.append("threat_detection")
        
        result = {
            'result': final_result,
            'blocked_by': blocked_by,
            'modifications_applied': len(modifications) > 0,
            'modifications': modifications,
            'warnings': warnings,
            'threat_detected': threat_context is not None,
            'threat_context': threat_context.__dict__ if threat_context else None,
            'policies_evaluated': len(evaluations),
            'evaluations': [eval.__dict__ for eval in evaluations],
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Audit de l'évaluation
        await self._audit_policy_evaluation(context, operation, result)
        
        # Mise à jour des statistiques
        if final_result == "deny":
            self.statistics['violations_blocked'] += 1
        
        self.statistics['policies_enforced'] += len([e for e in evaluations if e.result != "allow"])
        
        return result
    
    async def add_policy(self, policy: SecurityPolicy):
        """Ajoute une nouvelle politique"""
        self.policies[policy.policy_id] = policy
        self.logger.info(f"Policy {policy.policy_id} added: {policy.name}")
    
    async def remove_policy(self, policy_id: str):
        """Supprime une politique"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            self.logger.info(f"Policy {policy_id} removed")
    
    async def update_policy(self, policy: SecurityPolicy):
        """Met à jour une politique existante"""
        policy.updated_at = datetime.now(timezone.utc)
        self.policies[policy.policy_id] = policy
        self.logger.info(f"Policy {policy.policy_id} updated")
    
    def _get_applicable_policies(
        self,
        context: TenantContext,
        operation: str,
        data: Dict[str, Any]
    ) -> List[SecurityPolicy]:
        """Obtient les politiques applicables"""
        applicable = []
        
        for policy in self.policies.values():
            if not policy.active:
                continue
            
            # Vérification de la période d'effectivité
            now = datetime.now(timezone.utc)
            if policy.effective_until and now > policy.effective_until:
                continue
            
            if now < policy.effective_from:
                continue
            
            # Vérification de la portée
            if policy.scope == PolicyScope.TENANT:
                if policy.target_tenants and context.tenant_id not in policy.target_tenants:
                    continue
            
            # Vérification des opérations cibles
            if policy.target_operations and operation not in policy.target_operations:
                continue
            
            # Vérification des ressources cibles
            if policy.target_resources:
                resource_type = data.get('resource_type')
                if resource_type and resource_type not in policy.target_resources:
                    continue
            
            applicable.append(policy)
        
        # Tri par priorité
        return sorted(applicable, key=lambda p: p.priority, reverse=True)
    
    async def _detect_threats(
        self,
        data: Dict[str, Any],
        request_context: Dict[str, Any]
    ) -> Optional[ThreatContext]:
        """Détecte les menaces potentielles"""
        
        # Analyse des données pour patterns malveillants
        for threat_type, config in self.threat_patterns.items():
            patterns = config['patterns']
            
            for key, value in data.items():
                if isinstance(value, str):
                    for pattern in patterns:
                        if re.search(pattern, value):
                            threat_id = hashlib.sha256(f"{threat_type}_{pattern}_{datetime.now()}".encode()).hexdigest()[:16]
                            
                            threat = ThreatContext(
                                threat_id=threat_id,
                                threat_type=threat_type,
                                level=config['threat_level'],
                                indicators=[f"Pattern match in field '{key}': {pattern}"],
                                source_ip=request_context.get('source_ip'),
                                user_agent=request_context.get('user_agent'),
                                patterns_matched=[pattern],
                                confidence_score=0.8
                            )
                            
                            self.active_threats[threat_id] = threat
                            self.statistics['threats_detected'] += 1
                            
                            return threat
        
        return None
    
    async def _audit_policy_evaluation(
        self,
        context: TenantContext,
        operation: str,
        result: Dict[str, Any]
    ):
        """Audit de l'évaluation des politiques"""
        # Implémentation de l'audit
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du moteur"""
        return {
            **self.statistics,
            'active_policies': len([p for p in self.policies.values() if p.active]),
            'total_policies': len(self.policies),
            'active_threats': len(self.active_threats),
            'engines_count': len(self.engines)
        }
    
    async def cleanup_threats(self, max_age_hours: int = 24):
        """Nettoie les anciennes menaces"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        expired_threats = [
            threat_id for threat_id, threat in self.active_threats.items()
            if threat.detected_at < cutoff_time
        ]
        
        for threat_id in expired_threats:
            del self.active_threats[threat_id]
        
        self.logger.info(f"Cleaned up {len(expired_threats)} expired threats")
