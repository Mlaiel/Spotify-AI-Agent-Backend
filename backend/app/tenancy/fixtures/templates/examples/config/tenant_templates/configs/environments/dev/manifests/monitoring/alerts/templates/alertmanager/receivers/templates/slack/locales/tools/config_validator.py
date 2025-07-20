"""
Moteur de validation de configuration multi-niveaux.

Ce module fournit un système complet de validation pour toutes les 
configurations avec support multi-tenant et compliance.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime
from pydantic import BaseModel, ValidationError
from ..schemas.validation_schemas import (
    ValidationResultSchema,
    ValidationErrorSchema,
    ValidationContextSchema,
    ValidationRuleSchema,
    ValidationType,
    ValidationSeverity
)
from ..schemas.tenant_schemas import TenantConfigSchema
from ..schemas.monitoring_schemas import MonitoringConfigSchema
from ..schemas.alert_schemas import AlertManagerConfigSchema
from ..schemas.slack_schemas import SlackConfigSchema


class ValidationEngine:
    """Moteur de validation avancé avec règles personnalisables."""
    
    def __init__(self):
        """Initialise le moteur de validation."""
        self.rules: Dict[str, List[ValidationRuleSchema]] = {}
        self.custom_validators: Dict[str, callable] = {}
        self._load_default_rules()
    
    def add_rule(self, rule: ValidationRuleSchema):
        """Ajoute une règle de validation."""
        if rule.type not in self.rules:
            self.rules[rule.type] = []
        self.rules[rule.type].append(rule)
    
    def add_custom_validator(self, name: str, validator_func: callable):
        """Ajoute un validateur personnalisé."""
        self.custom_validators[name] = validator_func
    
    def validate_data(
        self,
        data: Any,
        validation_type: ValidationType,
        context: ValidationContextSchema
    ) -> ValidationResultSchema:
        """Valide des données selon le type et le contexte."""
        start_time = datetime.now()
        errors = []
        warnings = []
        info = []
        
        # Récupération des règles applicables
        applicable_rules = self._get_applicable_rules(validation_type, context)
        
        # Application des règles
        for rule in applicable_rules:
            try:
                result = self._apply_rule(rule, data, context)
                if result:
                    if result.severity == ValidationSeverity.ERROR:
                        errors.append(result)
                    elif result.severity == ValidationSeverity.WARNING:
                        warnings.append(result)
                    else:
                        info.append(result)
            except Exception as e:
                errors.append(ValidationErrorSchema(
                    field="validation_engine",
                    message=f"Erreur lors de l'application de la règle {rule.name}: {e}",
                    severity=ValidationSeverity.ERROR,
                    code="RULE_EXECUTION_ERROR"
                ))
        
        # Calcul du temps d'exécution
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Détermination si la validation est réussie
        is_valid = len(errors) == 0 and (
            context.ignore_warnings or len(warnings) == 0
        )
        
        return ValidationResultSchema(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            validation_time=validation_time,
            context=context,
            recommendations=self._generate_recommendations(errors, warnings)
        )
    
    def _load_default_rules(self):
        """Charge les règles de validation par défaut."""
        # Règles de sécurité
        self.add_rule(ValidationRuleSchema(
            name="password_complexity",
            description="Vérification de la complexité des mots de passe",
            type=ValidationType.SECURITY,
            severity=ValidationSeverity.ERROR,
            pattern=r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$",
            parameters={"min_length": 12}
        ))
        
        self.add_rule(ValidationRuleSchema(
            name="encryption_required",
            description="Vérification que le chiffrement est activé",
            type=ValidationType.SECURITY,
            severity=ValidationSeverity.ERROR,
            expression="encryption_at_rest == True and encryption_in_transit == True"
        ))
        
        # Règles de performance
        self.add_rule(ValidationRuleSchema(
            name="resource_limits",
            description="Vérification des limites de ressources",
            type=ValidationType.PERFORMANCE,
            severity=ValidationSeverity.WARNING,
            expression="cpu_limit <= 4 and memory_limit <= 8192"
        ))
        
        # Règles de compliance
        self.add_rule(ValidationRuleSchema(
            name="gdpr_compliance",
            description="Vérification de conformité GDPR",
            type=ValidationType.COMPLIANCE,
            severity=ValidationSeverity.ERROR,
            conditions=["data_retention_policy", "user_consent", "data_encryption"]
        ))
    
    def _get_applicable_rules(
        self,
        validation_type: ValidationType,
        context: ValidationContextSchema
    ) -> List[ValidationRuleSchema]:
        """Récupère les règles applicables selon le type et le contexte."""
        rules = self.rules.get(validation_type, [])
        
        # Filtrage selon le contexte
        applicable_rules = []
        for rule in rules:
            if rule.enabled:
                # Vérification des conditions
                if self._check_rule_conditions(rule, context):
                    applicable_rules.append(rule)
        
        return applicable_rules
    
    def _check_rule_conditions(
        self,
        rule: ValidationRuleSchema,
        context: ValidationContextSchema
    ) -> bool:
        """Vérifie si les conditions d'une règle sont remplies."""
        if not rule.conditions:
            return True
        
        # Vérification des conditions selon le contexte
        for condition in rule.conditions:
            if condition == "strict_mode" and not context.strict_mode:
                return False
            elif condition == "production_only" and context.environment != "production":
                return False
            elif condition.startswith("tenant_tier:"):
                required_tier = condition.split(":")[1]
                # Cette vérification nécessiterait l'accès aux données du tenant
                pass
        
        return True
    
    def _apply_rule(
        self,
        rule: ValidationRuleSchema,
        data: Any,
        context: ValidationContextSchema
    ) -> Optional[ValidationErrorSchema]:
        """Applique une règle de validation sur les données."""
        try:
            # Validation par pattern regex
            if rule.pattern and isinstance(data, str):
                import re
                if not re.match(rule.pattern, data):
                    return ValidationErrorSchema(
                        field=rule.name,
                        message=f"Données ne respectant pas le pattern: {rule.pattern}",
                        severity=rule.severity,
                        code="PATTERN_MISMATCH"
                    )
            
            # Validation par expression
            if rule.expression:
                if not self._evaluate_expression(rule.expression, data):
                    return ValidationErrorSchema(
                        field=rule.name,
                        message=f"Expression de validation échouée: {rule.expression}",
                        severity=rule.severity,
                        code="EXPRESSION_FAILED"
                    )
            
            # Validation personnalisée
            if rule.name in self.custom_validators:
                validator = self.custom_validators[rule.name]
                result = validator(data, rule.parameters)
                if not result:
                    return ValidationErrorSchema(
                        field=rule.name,
                        message=f"Validation personnalisée échouée",
                        severity=rule.severity,
                        code="CUSTOM_VALIDATION_FAILED"
                    )
            
            return None
            
        except Exception as e:
            return ValidationErrorSchema(
                field=rule.name,
                message=f"Erreur lors de la validation: {e}",
                severity=ValidationSeverity.ERROR,
                code="VALIDATION_ERROR"
            )
    
    def _evaluate_expression(self, expression: str, data: Any) -> bool:
        """Évalue une expression de validation."""
        try:
            # Conversion des données en contexte d'évaluation
            if isinstance(data, dict):
                context = data
            else:
                context = {"value": data}
            
            # Évaluation sécurisée de l'expression
            # Note: En production, utiliser un parser plus sécurisé
            return eval(expression, {"__builtins__": {}}, context)
        except Exception:
            return False
    
    def _generate_recommendations(
        self,
        errors: List[ValidationErrorSchema],
        warnings: List[ValidationErrorSchema]
    ) -> List[str]:
        """Génère des recommandations basées sur les erreurs et avertissements."""
        recommendations = []
        
        # Recommandations basées sur les types d'erreurs
        error_types = [error.code for error in errors]
        warning_types = [warning.code for warning in warnings]
        
        if "PATTERN_MISMATCH" in error_types:
            recommendations.append(
                "Vérifiez que les valeurs respectent les formats requis (regex patterns)"
            )
        
        if "ENCRYPTION_REQUIRED" in error_types:
            recommendations.append(
                "Activez le chiffrement pour toutes les données sensibles"
            )
        
        if "RESOURCE_LIMITS" in warning_types:
            recommendations.append(
                "Optimisez l'utilisation des ressources pour améliorer les performances"
            )
        
        if "GDPR_COMPLIANCE" in error_types:
            recommendations.append(
                "Assurez-vous que toutes les exigences GDPR sont satisfaites"
            )
        
        return recommendations


class ConfigValidator:
    """Validateur de configuration principal."""
    
    def __init__(self, validation_engine: Optional[ValidationEngine] = None):
        """Initialise le validateur de configuration."""
        self.validation_engine = validation_engine or ValidationEngine()
        
        # Mapping des schémas de configuration
        self.schema_mapping = {
            'tenant': TenantConfigSchema,
            'monitoring': MonitoringConfigSchema,
            'alertmanager': AlertManagerConfigSchema,
            'slack': SlackConfigSchema
        }
    
    def validate_config_file(
        self,
        file_path: str,
        config_type: str,
        tenant_id: Optional[str] = None,
        environment: str = "dev"
    ) -> ValidationResultSchema:
        """Valide un fichier de configuration."""
        # Chargement du fichier
        config_data = self._load_config_file(file_path)
        
        # Validation
        return self.validate_config_data(
            config_data, config_type, tenant_id, environment
        )
    
    def validate_config_data(
        self,
        config_data: Dict[str, Any],
        config_type: str,
        tenant_id: Optional[str] = None,
        environment: str = "dev"
    ) -> ValidationResultSchema:
        """Valide des données de configuration."""
        # Création du contexte de validation
        context = ValidationContextSchema(
            tenant_id=tenant_id,
            environment=environment,
            validation_type=ValidationType.CONFIG
        )
        
        # Validation structurelle avec Pydantic
        structural_errors = self._validate_structure(config_data, config_type)
        
        # Validation métier avec le moteur de validation
        business_result = self.validation_engine.validate_data(
            config_data, ValidationType.CONFIG, context
        )
        
        # Fusion des résultats
        all_errors = structural_errors + business_result.errors
        
        return ValidationResultSchema(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=business_result.warnings,
            info=business_result.info,
            validation_time=business_result.validation_time,
            context=context,
            recommendations=business_result.recommendations
        )
    
    def validate_multi_config(
        self,
        configs: List[Dict[str, Any]]
    ) -> Dict[str, ValidationResultSchema]:
        """Valide plusieurs configurations."""
        results = {}
        
        for i, config_spec in enumerate(configs):
            try:
                result = self.validate_config_data(
                    config_spec['data'],
                    config_spec['type'],
                    config_spec.get('tenant_id'),
                    config_spec.get('environment', 'dev')
                )
                results[f"config_{i}"] = result
            except Exception as e:
                results[f"config_{i}"] = ValidationResultSchema(
                    is_valid=False,
                    errors=[ValidationErrorSchema(
                        field="general",
                        message=f"Erreur de validation: {e}",
                        severity=ValidationSeverity.ERROR,
                        code="VALIDATION_EXCEPTION"
                    )],
                    warnings=[],
                    info=[],
                    validation_time=0.0,
                    context=ValidationContextSchema(validation_type=ValidationType.CONFIG),
                    recommendations=[]
                )
        
        return results
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Charge un fichier de configuration."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Format de fichier non supporté: {path.suffix}")
    
    def _validate_structure(
        self,
        config_data: Dict[str, Any],
        config_type: str
    ) -> List[ValidationErrorSchema]:
        """Valide la structure avec les schémas Pydantic."""
        errors = []
        
        if config_type not in self.schema_mapping:
            errors.append(ValidationErrorSchema(
                field="config_type",
                message=f"Type de configuration non supporté: {config_type}",
                severity=ValidationSeverity.ERROR,
                code="UNSUPPORTED_CONFIG_TYPE"
            ))
            return errors
        
        schema_class = self.schema_mapping[config_type]
        
        try:
            schema_class(**config_data)
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(x) for x in error['loc'])
                errors.append(ValidationErrorSchema(
                    field=field,
                    message=error['msg'],
                    severity=ValidationSeverity.ERROR,
                    code="SCHEMA_VALIDATION_ERROR",
                    details={"type": error['type'], "input": error.get('input')}
                ))
        
        return errors
    
    def get_validation_summary(
        self,
        results: List[ValidationResultSchema]
    ) -> Dict[str, Any]:
        """Génère un résumé des validations."""
        total_configs = len(results)
        valid_configs = sum(1 for r in results if r.is_valid)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        avg_validation_time = sum(r.validation_time for r in results) / total_configs
        
        return {
            "total_configs": total_configs,
            "valid_configs": valid_configs,
            "invalid_configs": total_configs - valid_configs,
            "success_rate": valid_configs / total_configs * 100,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "average_validation_time": avg_validation_time,
            "recommendations": self._consolidate_recommendations(results)
        }
    
    def _consolidate_recommendations(
        self,
        results: List[ValidationResultSchema]
    ) -> List[str]:
        """Consolide les recommandations de plusieurs validations."""
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Déduplication et priorisation
        unique_recommendations = list(set(all_recommendations))
        return sorted(unique_recommendations)


# Factory function pour créer un validateur configuré
def create_config_validator(
    custom_rules: Optional[List[ValidationRuleSchema]] = None,
    custom_validators: Optional[Dict[str, callable]] = None
) -> ConfigValidator:
    """Factory pour créer un validateur de configuration."""
    engine = ValidationEngine()
    
    if custom_rules:
        for rule in custom_rules:
            engine.add_rule(rule)
    
    if custom_validators:
        for name, validator in custom_validators.items():
            engine.add_custom_validator(name, validator)
    
    return ConfigValidator(engine)


def get_default_validator() -> ConfigValidator:
    """Retourne un validateur avec la configuration par défaut."""
    return ConfigValidator()
