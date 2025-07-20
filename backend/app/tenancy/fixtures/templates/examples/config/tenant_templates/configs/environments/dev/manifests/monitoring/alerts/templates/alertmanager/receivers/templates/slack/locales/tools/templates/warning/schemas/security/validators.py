"""
Specialized Security Validators for Multi-Tenant Architecture
===========================================================

Ce module contient les validateurs spécialisés pour l'architecture de sécurité
multi-tenant du Spotify AI Agent.

Auteur: Fahed Mlaiel
"""

import asyncio
import json
import logging
import re
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import uuid
import hashlib
import secrets
from email_validator import validate_email, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
from .core import SecurityLevel, ThreatType, SecurityEvent, TenantSecurityConfig
from .schemas import (
    SecurityRuleSchema, PermissionSchema, AuditSchema,
    ComplianceStandard, PermissionType, SecurityAction
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Résultat d'une validation"""
    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    threat_score: float = 0.0
    suggested_action: Optional[SecurityAction] = None


@dataclass
class AccessContext:
    """Contexte d'accès pour validation"""
    user_id: str
    tenant_id: str
    resource_type: str
    resource_id: Optional[str] = None
    action: str = ""
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class TenantAccessValidator:
    """
    Validateur de contrôle d'accès tenant-specific
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.tenant_configs: Dict[str, TenantSecurityConfig] = {}
        self._cache_ttl = 300  # 5 minutes
        
    async def initialize(self):
        """Initialise le validateur"""
        await self._load_tenant_configs()
        logger.info("TenantAccessValidator initialized")
    
    async def _load_tenant_configs(self):
        """Charge les configurations des tenants"""
        try:
            # Chargement depuis cache Redis
            cached_configs = await self.redis.hgetall("tenant_access_configs")
            
            if cached_configs:
                for tenant_id, config_json in cached_configs.items():
                    config_data = json.loads(config_json)
                    self.tenant_configs[tenant_id.decode()] = TenantSecurityConfig(**config_data)
            else:
                await self._load_from_database()
                
        except Exception as e:
            logger.error(f"Error loading tenant configs: {e}")
            raise
    
    async def _load_from_database(self):
        """Charge depuis la base de données"""
        # Implémentation du chargement DB
        pass
    
    async def validate_tenant_access(self, context: AccessContext) -> ValidationResult:
        """Valide l'accès pour un tenant spécifique"""
        try:
            config = await self._get_tenant_config(context.tenant_id)
            if not config:
                return ValidationResult(
                    is_valid=False,
                    error_code="TENANT_NOT_FOUND",
                    error_message=f"Tenant {context.tenant_id} not found",
                    threat_score=0.8
                )
            
            # Validation IP
            ip_result = await self._validate_ip_access(context.source_ip, config)
            if not ip_result.is_valid:
                return ip_result
            
            # Validation géographique
            geo_result = await self._validate_geo_restrictions(context.source_ip, config)
            if not geo_result.is_valid:
                return geo_result
            
            # Validation des limites de taux
            rate_result = await self._validate_rate_limits(context, config)
            if not rate_result.is_valid:
                return rate_result
            
            # Validation de session
            session_result = await self._validate_session(context, config)
            if not session_result.is_valid:
                return session_result
            
            # Validation VPN si requis
            if config.vpn_required:
                vpn_result = await self._validate_vpn_requirement(context.source_ip)
                if not vpn_result.is_valid:
                    return vpn_result
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"Error validating tenant access: {e}")
            return ValidationResult(
                is_valid=False,
                error_code="VALIDATION_ERROR",
                error_message=str(e),
                threat_score=0.5
            )
    
    async def _get_tenant_config(self, tenant_id: str) -> Optional[TenantSecurityConfig]:
        """Récupère la configuration d'un tenant"""
        if tenant_id not in self.tenant_configs:
            await self._load_tenant_configs()
        return self.tenant_configs.get(tenant_id)
    
    async def _validate_ip_access(self, source_ip: str, config: TenantSecurityConfig) -> ValidationResult:
        """Valide l'accès par IP"""
        if not source_ip:
            return ValidationResult(is_valid=True)
        
        try:
            ip = ipaddress.ip_address(source_ip)
            
            # Vérification des IPs bloquées
            for blocked_range in config.blocked_ip_ranges:
                if ip in ipaddress.ip_network(blocked_range, strict=False):
                    return ValidationResult(
                        is_valid=False,
                        error_code="IP_BLOCKED",
                        error_message=f"IP {source_ip} is blocked",
                        threat_score=0.9,
                        suggested_action=SecurityAction.BLOCK
                    )
            
            # Vérification des IPs autorisées
            if config.allowed_ip_ranges:
                allowed = False
                for allowed_range in config.allowed_ip_ranges:
                    if ip in ipaddress.ip_network(allowed_range, strict=False):
                        allowed = True
                        break
                
                if not allowed:
                    return ValidationResult(
                        is_valid=False,
                        error_code="IP_NOT_ALLOWED",
                        error_message=f"IP {source_ip} is not in allowed ranges",
                        threat_score=0.7,
                        suggested_action=SecurityAction.DENY
                    )
            
            return ValidationResult(is_valid=True)
            
        except ValueError:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_IP",
                error_message=f"Invalid IP address: {source_ip}",
                threat_score=0.3
            )
    
    async def _validate_geo_restrictions(self, source_ip: str, config: TenantSecurityConfig) -> ValidationResult:
        """Valide les restrictions géographiques"""
        if not config.geo_restrictions or not source_ip:
            return ValidationResult(is_valid=True)
        
        try:
            # Récupération de la géolocalisation (simulation)
            country_code = await self._get_country_from_ip(source_ip)
            
            if country_code in config.geo_restrictions:
                return ValidationResult(
                    is_valid=False,
                    error_code="GEO_RESTRICTED",
                    error_message=f"Access from {country_code} is restricted",
                    threat_score=0.6,
                    suggested_action=SecurityAction.DENY
                )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.warning(f"Error validating geo restrictions: {e}")
            return ValidationResult(is_valid=True)  # Permettre l'accès en cas d'erreur geo
    
    async def _get_country_from_ip(self, ip: str) -> str:
        """Récupère le code pays depuis l'IP"""
        # Implémentation avec service de géolocalisation
        # Simulation pour l'exemple
        return "US"
    
    async def _validate_rate_limits(self, context: AccessContext, config: TenantSecurityConfig) -> ValidationResult:
        """Valide les limites de taux"""
        if not config.rate_limits:
            return ValidationResult(is_valid=True)
        
        try:
            action_limit = config.rate_limits.get(context.action)
            if not action_limit:
                return ValidationResult(is_valid=True)
            
            # Clé pour le rate limiting
            rate_key = f"rate_limit:{context.tenant_id}:{context.user_id}:{context.action}"
            
            # Vérification du compteur actuel
            current_count = await self.redis.get(rate_key)
            if current_count and int(current_count) >= action_limit:
                return ValidationResult(
                    is_valid=False,
                    error_code="RATE_LIMIT_EXCEEDED",
                    error_message=f"Rate limit exceeded for action {context.action}",
                    threat_score=0.4,
                    suggested_action=SecurityAction.DENY
                )
            
            # Incrémentation du compteur
            await self.redis.incr(rate_key)
            await self.redis.expire(rate_key, 3600)  # 1 heure
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"Error validating rate limits: {e}")
            return ValidationResult(is_valid=True)  # Permettre en cas d'erreur
    
    async def _validate_session(self, context: AccessContext, config: TenantSecurityConfig) -> ValidationResult:
        """Valide la session utilisateur"""
        if not context.session_id:
            return ValidationResult(is_valid=True)
        
        try:
            session_key = f"session:{context.tenant_id}:{context.user_id}:{context.session_id}"
            session_data = await self.redis.get(session_key)
            
            if not session_data:
                return ValidationResult(
                    is_valid=False,
                    error_code="INVALID_SESSION",
                    error_message="Session not found or expired",
                    threat_score=0.5
                )
            
            session_info = json.loads(session_data)
            
            # Vérification du timeout de session
            last_activity = datetime.fromisoformat(session_info.get('last_activity'))
            timeout_delta = timedelta(minutes=config.session_timeout_minutes)
            
            if datetime.utcnow() - last_activity > timeout_delta:
                await self.redis.delete(session_key)
                return ValidationResult(
                    is_valid=False,
                    error_code="SESSION_EXPIRED",
                    error_message="Session has expired",
                    threat_score=0.2
                )
            
            # Mise à jour de la dernière activité
            session_info['last_activity'] = datetime.utcnow().isoformat()
            await self.redis.set(session_key, json.dumps(session_info), ex=config.session_timeout_minutes * 60)
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return ValidationResult(is_valid=True)
    
    async def _validate_vpn_requirement(self, source_ip: str) -> ValidationResult:
        """Valide l'exigence VPN"""
        if not source_ip:
            return ValidationResult(is_valid=True)
        
        # Détection VPN (simulation)
        is_vpn = await self._detect_vpn(source_ip)
        
        if not is_vpn:
            return ValidationResult(
                is_valid=False,
                error_code="VPN_REQUIRED",
                error_message="VPN connection is required",
                threat_score=0.3,
                suggested_action=SecurityAction.DENY
            )
        
        return ValidationResult(is_valid=True)
    
    async def _detect_vpn(self, ip: str) -> bool:
        """Détecte si une IP utilise un VPN"""
        # Implémentation de détection VPN
        # Simulation pour l'exemple
        return True


class PermissionValidator:
    """
    Validateur de permissions RBAC/ABAC
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.permission_cache: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialise le validateur de permissions"""
        logger.info("PermissionValidator initialized")
    
    async def validate_permission(self, context: AccessContext, required_permission: PermissionType) -> ValidationResult:
        """Valide une permission spécifique"""
        try:
            # Récupération des permissions utilisateur
            user_permissions = await self._get_user_permissions(context.user_id, context.tenant_id)
            
            if not user_permissions:
                return ValidationResult(
                    is_valid=False,
                    error_code="NO_PERMISSIONS",
                    error_message="User has no permissions",
                    threat_score=0.3
                )
            
            # Vérification de la permission sur la ressource
            has_permission = await self._check_resource_permission(
                user_permissions, 
                context.resource_type, 
                context.resource_id, 
                required_permission
            )
            
            if not has_permission:
                return ValidationResult(
                    is_valid=False,
                    error_code="PERMISSION_DENIED",
                    error_message=f"Permission {required_permission.value} denied on {context.resource_type}",
                    threat_score=0.4,
                    suggested_action=SecurityAction.DENY
                )
            
            # Vérification des conditions contextuelles
            context_result = await self._validate_permission_context(user_permissions, context)
            if not context_result.is_valid:
                return context_result
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"Error validating permission: {e}")
            return ValidationResult(
                is_valid=False,
                error_code="PERMISSION_ERROR",
                error_message=str(e),
                threat_score=0.5
            )
    
    async def _get_user_permissions(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """Récupère les permissions d'un utilisateur"""
        cache_key = f"permissions:{tenant_id}:{user_id}"
        
        # Vérification cache
        cached_permissions = await self.redis.get(cache_key)
        if cached_permissions:
            return json.loads(cached_permissions)
        
        # Récupération depuis DB
        permissions = await self._load_user_permissions_from_db(user_id, tenant_id)
        
        # Mise en cache
        await self.redis.set(cache_key, json.dumps(permissions), ex=300)
        
        return permissions
    
    async def _load_user_permissions_from_db(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """Charge les permissions depuis la base de données"""
        # Implémentation de chargement des permissions
        return {
            "roles": ["user"],
            "permissions": {
                "audio_file": ["read"],
                "playlist": ["read", "write"]
            }
        }
    
    async def _check_resource_permission(self, user_permissions: Dict[str, Any], 
                                       resource_type: str, resource_id: Optional[str], 
                                       required_permission: PermissionType) -> bool:
        """Vérifie la permission sur une ressource"""
        resource_permissions = user_permissions.get("permissions", {}).get(resource_type, [])
        
        # Vérification permission directe
        if required_permission.value in resource_permissions:
            return True
        
        # Vérification permissions héritées des rôles
        for role in user_permissions.get("roles", []):
            role_permissions = await self._get_role_permissions(role, resource_type)
            if required_permission.value in role_permissions:
                return True
        
        return False
    
    async def _get_role_permissions(self, role: str, resource_type: str) -> List[str]:
        """Récupère les permissions d'un rôle"""
        cache_key = f"role_permissions:{role}:{resource_type}"
        
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Simulation des permissions de rôle
        role_permissions = {
            "admin": ["read", "write", "execute", "delete", "admin"],
            "user": ["read"],
            "premium_user": ["read", "write"]
        }
        
        permissions = role_permissions.get(role, [])
        await self.redis.set(cache_key, json.dumps(permissions), ex=600)
        
        return permissions
    
    async def _validate_permission_context(self, user_permissions: Dict[str, Any], 
                                         context: AccessContext) -> ValidationResult:
        """Valide le contexte de la permission"""
        # Vérification des conditions temporelles
        # Vérification des conditions géographiques
        # Vérification des conditions métier
        return ValidationResult(is_valid=True)


class SecurityRuleValidator:
    """
    Validateur de règles de sécurité personnalisées
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.rules_cache: Dict[str, List[SecurityRuleSchema]] = {}
        
    async def initialize(self):
        """Initialise le validateur de règles"""
        await self._load_security_rules()
        logger.info("SecurityRuleValidator initialized")
    
    async def _load_security_rules(self):
        """Charge les règles de sécurité"""
        # Chargement depuis DB et cache
        pass
    
    async def validate_against_rules(self, context: AccessContext) -> ValidationResult:
        """Valide contre les règles de sécurité"""
        try:
            tenant_rules = await self._get_tenant_rules(context.tenant_id)
            
            for rule in tenant_rules:
                if not rule.enabled:
                    continue
                
                if not await self._rule_applies_to_context(rule, context):
                    continue
                
                rule_result = await self._evaluate_rule(rule, context)
                if not rule_result.is_valid:
                    # Exécution des actions de la règle
                    await self._execute_rule_actions(rule, context)
                    return rule_result
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"Error validating security rules: {e}")
            return ValidationResult(
                is_valid=False,
                error_code="RULE_VALIDATION_ERROR",
                error_message=str(e),
                threat_score=0.5
            )
    
    async def _get_tenant_rules(self, tenant_id: str) -> List[SecurityRuleSchema]:
        """Récupère les règles d'un tenant"""
        if tenant_id not in self.rules_cache:
            rules = await self._load_tenant_rules_from_db(tenant_id)
            self.rules_cache[tenant_id] = rules
        
        return self.rules_cache.get(tenant_id, [])
    
    async def _load_tenant_rules_from_db(self, tenant_id: str) -> List[SecurityRuleSchema]:
        """Charge les règles depuis la base"""
        # Implémentation de chargement
        return []
    
    async def _rule_applies_to_context(self, rule: SecurityRuleSchema, context: AccessContext) -> bool:
        """Vérifie si une règle s'applique au contexte"""
        if rule.applies_to and context.resource_type not in rule.applies_to:
            return False
        
        # Vérification temporelle
        now = datetime.utcnow()
        if rule.effective_from and now < rule.effective_from:
            return False
        if rule.effective_until and now > rule.effective_until:
            return False
        
        return True
    
    async def _evaluate_rule(self, rule: SecurityRuleSchema, context: AccessContext) -> ValidationResult:
        """Évalue une règle de sécurité"""
        # Évaluation des conditions
        conditions_met = []
        
        for condition in rule.conditions:
            is_met = await self._evaluate_condition(condition, context)
            conditions_met.append(is_met)
        
        # Application de la logique (AND/OR)
        if rule.conditions_logic == "AND":
            rule_triggered = all(conditions_met)
        else:  # OR
            rule_triggered = any(conditions_met)
        
        if rule_triggered:
            # Mise à jour des métriques
            await self._update_rule_metrics(rule)
            
            return ValidationResult(
                is_valid=False,
                error_code="SECURITY_RULE_VIOLATED",
                error_message=f"Security rule '{rule.name}' violated",
                threat_score=0.7,
                metadata={"rule_id": rule.id, "rule_name": rule.name}
            )
        
        return ValidationResult(is_valid=True)
    
    async def _evaluate_condition(self, condition, context: AccessContext) -> bool:
        """Évalue une condition de règle"""
        # Récupération de la valeur du champ
        field_value = await self._get_field_value(condition.field, context)
        
        # Application de l'opérateur
        return await self._apply_operator(
            field_value, 
            condition.operator, 
            condition.value,
            condition.case_sensitive
        )
    
    async def _get_field_value(self, field: str, context: AccessContext) -> Any:
        """Récupère la valeur d'un champ depuis le contexte"""
        field_map = {
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
            "resource_type": context.resource_type,
            "resource_id": context.resource_id,
            "action": context.action,
            "source_ip": context.source_ip,
            "user_agent": context.user_agent,
            "timestamp": context.timestamp
        }
        
        return field_map.get(field)
    
    async def _apply_operator(self, field_value: Any, operator: str, expected_value: Any, case_sensitive: bool) -> bool:
        """Applique un opérateur de comparaison"""
        if not case_sensitive and isinstance(field_value, str) and isinstance(expected_value, str):
            field_value = field_value.lower()
            expected_value = expected_value.lower()
        
        operators = {
            "equals": lambda f, e: f == e,
            "not_equals": lambda f, e: f != e,
            "contains": lambda f, e: e in str(f) if f else False,
            "not_contains": lambda f, e: e not in str(f) if f else True,
            "starts_with": lambda f, e: str(f).startswith(str(e)) if f else False,
            "ends_with": lambda f, e: str(f).endswith(str(e)) if f else False,
            "regex": lambda f, e: bool(re.match(str(e), str(f))) if f else False,
            "greater_than": lambda f, e: f > e if f is not None else False,
            "less_than": lambda f, e: f < e if f is not None else False,
            "in": lambda f, e: f in e if isinstance(e, (list, tuple)) else False,
            "not_in": lambda f, e: f not in e if isinstance(e, (list, tuple)) else True
        }
        
        operator_func = operators.get(operator)
        if not operator_func:
            return False
        
        try:
            return operator_func(field_value, expected_value)
        except Exception:
            return False
    
    async def _execute_rule_actions(self, rule: SecurityRuleSchema, context: AccessContext):
        """Exécute les actions d'une règle"""
        for action in rule.actions:
            await self._execute_action(action, rule, context)
    
    async def _execute_action(self, action, rule: SecurityRuleSchema, context: AccessContext):
        """Exécute une action spécifique"""
        if action.type == SecurityAction.BLOCK:
            await self._block_user(context, action.parameters)
        elif action.type == SecurityAction.ALERT:
            await self._send_rule_alert(rule, context, action.parameters)
        elif action.type == SecurityAction.MONITOR:
            await self._increase_monitoring(context, action.parameters)
    
    async def _block_user(self, context: AccessContext, parameters: Dict[str, Any]):
        """Bloque un utilisateur"""
        duration = parameters.get("duration_minutes", 60)
        reason = parameters.get("reason", "Security rule violation")
        
        block_key = f"blocked:{context.tenant_id}:{context.user_id}"
        block_data = {
            "reason": reason,
            "blocked_at": datetime.utcnow().isoformat(),
            "blocked_until": (datetime.utcnow() + timedelta(minutes=duration)).isoformat()
        }
        
        await self.redis.set(block_key, json.dumps(block_data), ex=duration * 60)
    
    async def _send_rule_alert(self, rule: SecurityRuleSchema, context: AccessContext, parameters: Dict[str, Any]):
        """Envoie une alerte pour violation de règle"""
        # Implémentation d'envoi d'alerte
        pass
    
    async def _increase_monitoring(self, context: AccessContext, parameters: Dict[str, Any]):
        """Augmente le niveau de monitoring"""
        # Implémentation de monitoring renforcé
        pass
    
    async def _update_rule_metrics(self, rule: SecurityRuleSchema):
        """Met à jour les métriques de la règle"""
        metrics_key = f"rule_metrics:{rule.id}"
        
        # Incrément du compteur de déclenchement
        await self.redis.hincrby(metrics_key, "trigger_count", 1)
        await self.redis.hset(metrics_key, "last_triggered", datetime.utcnow().isoformat())


class ComplianceValidator:
    """
    Validateur de conformité réglementaire (RGPD, SOC2, ISO27001)
    """
    
    def __init__(self, redis_client: aioredis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.compliance_rules: Dict[ComplianceStandard, Dict] = {}
        
    async def initialize(self):
        """Initialise le validateur de conformité"""
        await self._load_compliance_rules()
        logger.info("ComplianceValidator initialized")
    
    async def _load_compliance_rules(self):
        """Charge les règles de conformité"""
        # Règles RGPD
        self.compliance_rules[ComplianceStandard.GDPR] = {
            "data_retention_max_days": 1095,  # 3 ans
            "consent_required": True,
            "right_to_be_forgotten": True,
            "data_portability": True,
            "privacy_by_design": True,
            "dpo_required": True
        }
        
        # Règles SOC2
        self.compliance_rules[ComplianceStandard.SOC2] = {
            "access_control_required": True,
            "encryption_required": True,
            "audit_logging_required": True,
            "incident_response_required": True,
            "change_management_required": True
        }
        
        # Règles ISO27001
        self.compliance_rules[ComplianceStandard.ISO27001] = {
            "risk_assessment_required": True,
            "security_policy_required": True,
            "access_control_required": True,
            "cryptography_required": True,
            "supplier_security_required": True
        }
    
    async def validate_compliance(self, context: AccessContext, 
                                standards: List[ComplianceStandard]) -> ValidationResult:
        """Valide la conformité pour des standards donnés"""
        try:
            violations = []
            
            for standard in standards:
                standard_violations = await self._validate_standard(standard, context)
                violations.extend(standard_violations)
            
            if violations:
                return ValidationResult(
                    is_valid=False,
                    error_code="COMPLIANCE_VIOLATION",
                    error_message=f"Compliance violations: {', '.join(violations)}",
                    threat_score=0.8,
                    metadata={"violations": violations},
                    suggested_action=SecurityAction.ALERT
                )
            
            return ValidationResult(is_valid=True)
            
        except Exception as e:
            logger.error(f"Error validating compliance: {e}")
            return ValidationResult(
                is_valid=False,
                error_code="COMPLIANCE_ERROR",
                error_message=str(e),
                threat_score=0.5
            )
    
    async def _validate_standard(self, standard: ComplianceStandard, 
                               context: AccessContext) -> List[str]:
        """Valide un standard de conformité spécifique"""
        violations = []
        rules = self.compliance_rules.get(standard, {})
        
        if standard == ComplianceStandard.GDPR:
            violations.extend(await self._validate_gdpr(context, rules))
        elif standard == ComplianceStandard.SOC2:
            violations.extend(await self._validate_soc2(context, rules))
        elif standard == ComplianceStandard.ISO27001:
            violations.extend(await self._validate_iso27001(context, rules))
        
        return violations
    
    async def _validate_gdpr(self, context: AccessContext, rules: Dict) -> List[str]:
        """Valide la conformité RGPD"""
        violations = []
        
        # Vérification du consentement
        if rules.get("consent_required"):
            has_consent = await self._check_user_consent(context.user_id, context.tenant_id)
            if not has_consent:
                violations.append("Missing user consent for data processing")
        
        # Vérification de la rétention des données
        if rules.get("data_retention_max_days"):
            max_days = rules["data_retention_max_days"]
            if await self._check_data_retention_violation(context, max_days):
                violations.append(f"Data retention exceeds {max_days} days")
        
        # Vérification de la minimisation des données
        if await self._check_data_minimization_violation(context):
            violations.append("Data minimization principle violated")
        
        return violations
    
    async def _validate_soc2(self, context: AccessContext, rules: Dict) -> List[str]:
        """Valide la conformité SOC2"""
        violations = []
        
        # Vérification du contrôle d'accès
        if rules.get("access_control_required"):
            if not await self._check_access_control_compliance(context):
                violations.append("Access control requirements not met")
        
        # Vérification du chiffrement
        if rules.get("encryption_required"):
            if not await self._check_encryption_compliance(context):
                violations.append("Encryption requirements not met")
        
        # Vérification de l'audit logging
        if rules.get("audit_logging_required"):
            if not await self._check_audit_logging_compliance(context):
                violations.append("Audit logging requirements not met")
        
        return violations
    
    async def _validate_iso27001(self, context: AccessContext, rules: Dict) -> List[str]:
        """Valide la conformité ISO27001"""
        violations = []
        
        # Vérification de l'évaluation des risques
        if rules.get("risk_assessment_required"):
            if not await self._check_risk_assessment_compliance(context):
                violations.append("Risk assessment requirements not met")
        
        # Vérification de la politique de sécurité
        if rules.get("security_policy_required"):
            if not await self._check_security_policy_compliance(context):
                violations.append("Security policy requirements not met")
        
        return violations
    
    async def _check_user_consent(self, user_id: str, tenant_id: str) -> bool:
        """Vérifie le consentement utilisateur"""
        consent_key = f"consent:{tenant_id}:{user_id}"
        consent_data = await self.redis.get(consent_key)
        
        if not consent_data:
            return False
        
        consent_info = json.loads(consent_data)
        return consent_info.get("given", False)
    
    async def _check_data_retention_violation(self, context: AccessContext, max_days: int) -> bool:
        """Vérifie les violations de rétention de données"""
        # Implémentation de vérification de rétention
        return False
    
    async def _check_data_minimization_violation(self, context: AccessContext) -> bool:
        """Vérifie les violations de minimisation des données"""
        # Implémentation de vérification de minimisation
        return False
    
    async def _check_access_control_compliance(self, context: AccessContext) -> bool:
        """Vérifie la conformité du contrôle d'accès"""
        # Implémentation de vérification du contrôle d'accès
        return True
    
    async def _check_encryption_compliance(self, context: AccessContext) -> bool:
        """Vérifie la conformité du chiffrement"""
        # Implémentation de vérification du chiffrement
        return True
    
    async def _check_audit_logging_compliance(self, context: AccessContext) -> bool:
        """Vérifie la conformité de l'audit logging"""
        # Implémentation de vérification de l'audit
        return True
    
    async def _check_risk_assessment_compliance(self, context: AccessContext) -> bool:
        """Vérifie la conformité de l'évaluation des risques"""
        # Implémentation de vérification des risques
        return True
    
    async def _check_security_policy_compliance(self, context: AccessContext) -> bool:
        """Vérifie la conformité de la politique de sécurité"""
        # Implémentation de vérification des politiques
        return True


# Validateurs utilitaires

class InputValidator:
    """Validateur d'entrées utilisateur"""
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Valide une adresse email"""
        try:
            validate_email(email)
            return ValidationResult(is_valid=True)
        except EmailNotValidError as e:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_EMAIL",
                error_message=str(e)
            )
    
    @staticmethod
    def validate_phone(phone: str, region: str = "US") -> ValidationResult:
        """Valide un numéro de téléphone"""
        try:
            parsed = phonenumbers.parse(phone, region)
            if phonenumbers.is_valid_number(parsed):
                return ValidationResult(is_valid=True)
            else:
                return ValidationResult(
                    is_valid=False,
                    error_code="INVALID_PHONE",
                    error_message="Invalid phone number"
                )
        except NumberParseException as e:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_PHONE",
                error_message=str(e)
            )
    
    @staticmethod
    def validate_password_strength(password: str, policy: Dict[str, Any] = None) -> ValidationResult:
        """Valide la force d'un mot de passe"""
        if not policy:
            policy = {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special_chars": True
            }
        
        errors = []
        
        if len(password) < policy.get("min_length", 8):
            errors.append(f"Password must be at least {policy['min_length']} characters")
        
        if policy.get("require_uppercase") and not re.search(r'[A-Z]', password):
            errors.append("Password must contain uppercase letters")
        
        if policy.get("require_lowercase") and not re.search(r'[a-z]', password):
            errors.append("Password must contain lowercase letters")
        
        if policy.get("require_numbers") and not re.search(r'\d', password):
            errors.append("Password must contain numbers")
        
        if policy.get("require_special_chars") and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain special characters")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                error_code="WEAK_PASSWORD",
                error_message="; ".join(errors)
            )
        
        return ValidationResult(is_valid=True)
    
    @staticmethod
    def sanitize_input(input_data: str) -> str:
        """Sanitise les entrées utilisateur"""
        if not input_data:
            return ""
        
        # Suppression des caractères dangereux
        sanitized = re.sub(r'[<>"\']', '', input_data)
        
        # Limitation de la longueur
        sanitized = sanitized[:1000]
        
        # Suppression des espaces en trop
        sanitized = sanitized.strip()
        
        return sanitized
