#!/usr/bin/env python3
"""
Schema Validation and Utility Scripts
====================================

Scripts utilitaires pour la validation, génération et gestion
des schémas tenancy avec automatisation complète et intégration CI/CD.
"""

import json
import yaml
import asyncio
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from pydantic import ValidationError

# Import des schémas
from .tenant_config_schema import TenantConfigSchema, TenantType
from .alert_schema import AlertSchema, TenantAlertSchema, AlertSeverity
from .warning_schema import WarningSchema, TenantWarningSchema, WarningSeverity
from .notification_schema import NotificationSchema, NotificationPriority
from .monitoring_schema import MonitoringConfigSchema
from .compliance_schema import ComplianceSchema, ComplianceStandard
from .performance_schema import PerformanceMetricsSchema


@dataclass
class ValidationResult:
    """Résultat de validation."""
    is_valid: bool
    schema_type: str
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class SchemaValidator:
    """Validateur de schémas avec analyses avancées."""
    
    def __init__(self):
        self.validators = {
            'tenant_config': TenantConfigSchema,
            'alert': AlertSchema,
            'tenant_alert': TenantAlertSchema,
            'warning': WarningSchema,
            'tenant_warning': TenantWarningSchema,
            'notification': NotificationSchema,
            'monitoring': MonitoringConfigSchema,
            'compliance': ComplianceSchema,
            'performance': PerformanceMetricsSchema
        }
    
    def validate_schema(self, data: Dict[str, Any], schema_type: str) -> ValidationResult:
        """Valide un schéma avec analyses détaillées."""
        result = ValidationResult(
            is_valid=False,
            schema_type=schema_type,
            errors=[],
            warnings=[],
            suggestions=[]
        )
        
        if schema_type not in self.validators:
            result.errors.append(f"Unknown schema type: {schema_type}")
            return result
        
        validator_class = self.validators[schema_type]
        
        try:
            # Validation principale
            instance = validator_class(**data)
            result.is_valid = True
            
            # Analyses supplémentaires
            result.warnings.extend(self._analyze_warnings(instance, schema_type))
            result.suggestions.extend(self._generate_suggestions(instance, schema_type))
            
        except ValidationError as e:
            result.errors = [str(error) for error in e.errors()]
        except Exception as e:
            result.errors.append(f"Unexpected error: {str(e)}")
        
        return result
    
    def _analyze_warnings(self, instance: Any, schema_type: str) -> List[str]:
        """Analyse et génère des avertissements."""
        warnings = []
        
        if schema_type == 'tenant_config':
            # Vérifications spécifiques tenant
            if instance.tenant_type == TenantType.TRIAL and instance.features.max_users and instance.features.max_users > 5:
                warnings.append("Trial tenants with >5 users may impact performance")
            
            if not instance.features.data_encryption and ComplianceStandard.GDPR in getattr(instance, 'compliance_levels', []):
                warnings.append("GDPR compliance recommended with data encryption")
        
        elif schema_type in ['alert', 'tenant_alert']:
            if instance.severity == AlertSeverity.CRITICAL and not instance.escalations:
                warnings.append("Critical alerts should have escalation configured")
            
            if len(instance.recipients) == 0:
                warnings.append("No recipients configured for alert")
        
        elif schema_type in ['warning', 'tenant_warning']:
            if instance.severity == WarningSeverity.URGENT and instance.escalation_threshold_hours > 2:
                warnings.append("Urgent warnings should escalate within 2 hours")
        
        return warnings
    
    def _generate_suggestions(self, instance: Any, schema_type: str) -> List[str]:
        """Génère des suggestions d'amélioration."""
        suggestions = []
        
        if schema_type == 'tenant_config':
            if instance.tenant_type == TenantType.ENTERPRISE:
                if not instance.features.custom_branding:
                    suggestions.append("Consider enabling custom branding for enterprise tenants")
                
                if not instance.features.sso_integration:
                    suggestions.append("SSO integration recommended for enterprise tenants")
        
        elif schema_type == 'monitoring':
            if instance.collection_interval_seconds > 60:
                suggestions.append("Consider shorter collection interval for better granularity")
            
            if not instance.anomaly_detection_enabled:
                suggestions.append("Enable anomaly detection for proactive monitoring")
        
        elif schema_type == 'performance':
            if instance.overall_performance_score < 80:
                suggestions.append("Performance score below 80% - review optimization recommendations")
        
        return suggestions
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """Valide un fichier de schéma."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Détermine le type de schéma depuis les métadonnées ou le nom de fichier
            schema_type = self._determine_schema_type(data, file_path)
            
            return self.validate_schema(data, schema_type)
        
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                schema_type="unknown",
                errors=[f"Failed to load file: {str(e)}"],
                warnings=[],
                suggestions=[]
            )
    
    def _determine_schema_type(self, data: Dict[str, Any], file_path: Path) -> str:
        """Détermine le type de schéma depuis les données ou le nom de fichier."""
        # Vérifier les métadonnées
        if 'schema_type' in data:
            return data['schema_type']
        
        # Vérifier le nom de fichier
        filename = file_path.stem.lower()
        for schema_name in self.validators.keys():
            if schema_name in filename:
                return schema_name
        
        # Vérifier la présence de champs distinctifs
        if 'tenant_id' in data and 'tenant_type' in data:
            return 'tenant_config'
        elif 'alert_id' in data or 'severity' in data and 'conditions' in data:
            return 'alert'
        elif 'warning_id' in data or 'warning_type' in data:
            return 'warning'
        elif 'notification_id' in data or 'notification_type' in data:
            return 'notification'
        elif 'metrics' in data and 'dashboards' in data:
            return 'monitoring'
        elif 'compliance_id' in data or 'applicable_standards' in data:
            return 'compliance'
        elif 'metrics_id' in data or 'performance_targets' in data:
            return 'performance'
        
        return 'unknown'


class SchemaGenerator:
    """Générateur de schémas avec templates et exemples."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Charge les templates de schémas."""
        return {
            'tenant_config': {
                'enterprise': {
                    'tenant_id': '{{tenant_id}}',
                    'tenant_name': '{{tenant_name}}',
                    'tenant_type': 'enterprise',
                    'admin_email': '{{admin_email}}',
                    'country_code': '{{country_code}}',
                    'features': {
                        'advanced_analytics': True,
                        'custom_alerts': True,
                        'real_time_monitoring': True,
                        'custom_branding': True,
                        'sso_integration': True,
                        'priority_support': True,
                        'max_users': 1000,
                        'max_storage_gb': 1000,
                        'max_api_calls_per_hour': 10000
                    },
                    'compliance_levels': ['gdpr', 'soc2', 'iso27001']
                },
                'professional': {
                    'tenant_id': '{{tenant_id}}',
                    'tenant_name': '{{tenant_name}}',
                    'tenant_type': 'professional',
                    'admin_email': '{{admin_email}}',
                    'country_code': '{{country_code}}',
                    'features': {
                        'advanced_analytics': True,
                        'custom_alerts': True,
                        'real_time_monitoring': True,
                        'max_users': 100,
                        'max_storage_gb': 100,
                        'max_api_calls_per_hour': 5000
                    },
                    'compliance_levels': ['gdpr']
                },
                'standard': {
                    'tenant_id': '{{tenant_id}}',
                    'tenant_name': '{{tenant_name}}',
                    'tenant_type': 'standard',
                    'admin_email': '{{admin_email}}',
                    'country_code': '{{country_code}}',
                    'features': {
                        'real_time_monitoring': True,
                        'max_users': 25,
                        'max_storage_gb': 25,
                        'max_api_calls_per_hour': 1000
                    }
                }
            },
            'alert': {
                'performance': {
                    'tenant_id': '{{tenant_id}}',
                    'name': 'high_{{metric_name}}',
                    'title': 'High {{metric_display_name}} Detected',
                    'description': '{{metric_display_name}} has exceeded {{threshold}} for more than {{duration}} minutes',
                    'severity': 'high',
                    'category': 'performance',
                    'conditions': [{
                        'metric_name': '{{metric_name}}',
                        'operator': 'gt',
                        'threshold': '{{threshold}}',
                        'duration_minutes': '{{duration}}'
                    }],
                    'notification_channels': ['email', 'slack'],
                    'recipients': ['{{admin_email}}', '#alerts']
                }
            }
        }
    
    def generate_schema(self, schema_type: str, template_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un schéma depuis un template."""
        if schema_type not in self.templates:
            raise ValueError(f"No templates available for schema type: {schema_type}")
        
        if template_name not in self.templates[schema_type]:
            raise ValueError(f"Template '{template_name}' not found for schema type: {schema_type}")
        
        template = self.templates[schema_type][template_name]
        return self._substitute_variables(template, variables)
    
    def _substitute_variables(self, template: Any, variables: Dict[str, Any]) -> Any:
        """Substitue les variables dans le template."""
        if isinstance(template, dict):
            return {k: self._substitute_variables(v, variables) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_variables(item, variables) for item in template]
        elif isinstance(template, str):
            for var, value in variables.items():
                template = template.replace(f'{{{{{var}}}}}', str(value))
            return template
        else:
            return template
    
    def list_templates(self) -> Dict[str, List[str]]:
        """Liste les templates disponibles."""
        return {schema_type: list(templates.keys()) 
                for schema_type, templates in self.templates.items()}


class SchemaManager:
    """Gestionnaire principal des schémas."""
    
    def __init__(self):
        self.validator = SchemaValidator()
        self.generator = SchemaGenerator()
    
    async def validate_directory(self, directory: Path, recursive: bool = True) -> Dict[str, ValidationResult]:
        """Valide tous les schémas dans un répertoire."""
        results = {}
        
        pattern = "**/*.{json,yml,yaml}" if recursive else "*.{json,yml,yaml}"
        
        for file_path in directory.glob(pattern.replace("{", "").replace("}", "")):
            if file_path.suffix.lower() in ['.json', '.yml', '.yaml']:
                result = self.validator.validate_file(file_path)
                results[str(file_path)] = result
        
        return results
    
    def generate_report(self, results: Dict[str, ValidationResult]) -> str:
        """Génère un rapport de validation."""
        report_lines = [
            "Schema Validation Report",
            "=" * 50,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            ""
        ]
        
        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.is_valid)
        invalid_files = total_files - valid_files
        
        report_lines.extend([
            f"Summary:",
            f"  Total files: {total_files}",
            f"  Valid: {valid_files}",
            f"  Invalid: {invalid_files}",
            f"  Success rate: {(valid_files/total_files*100):.1f}%" if total_files > 0 else "  Success rate: 0%",
            ""
        ])
        
        # Détails par fichier
        for file_path, result in results.items():
            status = "✅ VALID" if result.is_valid else "❌ INVALID"
            report_lines.append(f"{status} {file_path} ({result.schema_type})")
            
            if result.errors:
                for error in result.errors:
                    report_lines.append(f"  ERROR: {error}")
            
            if result.warnings:
                for warning in result.warnings:
                    report_lines.append(f"  WARNING: {warning}")
            
            if result.suggestions:
                for suggestion in result.suggestions:
                    report_lines.append(f"  SUGGESTION: {suggestion}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def export_schema_documentation(self, output_dir: Path) -> None:
        """Exporte la documentation des schémas."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for schema_name, schema_class in self.validator.validators.items():
            doc_content = self._generate_schema_doc(schema_name, schema_class)
            
            doc_file = output_dir / f"{schema_name}_schema.md"
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(doc_content)
    
    def _generate_schema_doc(self, schema_name: str, schema_class: Any) -> str:
        """Génère la documentation d'un schéma."""
        doc_lines = [
            f"# {schema_name.title().replace('_', ' ')} Schema",
            "",
            f"**Schema Class**: `{schema_class.__name__}`",
            "",
            "## Description",
            "",
            schema_class.__doc__ or "No description available.",
            "",
            "## Fields",
            ""
        ]
        
        # Extraire les champs du schéma
        if hasattr(schema_class, '__fields__'):
            for field_name, field_info in schema_class.__fields__.items():
                field_type = str(field_info.type_).replace('typing.', '')
                required = "Required" if field_info.required else "Optional"
                default = f" (default: {field_info.default})" if not field_info.required and field_info.default is not None else ""
                
                doc_lines.extend([
                    f"### {field_name}",
                    f"- **Type**: {field_type}",
                    f"- **Required**: {required}{default}",
                    f"- **Description**: {field_info.field_info.description or 'No description'}",
                    ""
                ])
        
        # Ajouter un exemple si disponible
        if hasattr(schema_class, 'Config') and hasattr(schema_class.Config, 'schema_extra'):
            example = schema_class.Config.schema_extra.get('example')
            if example:
                doc_lines.extend([
                    "## Example",
                    "",
                    "```json",
                    json.dumps(example, indent=2),
                    "```",
                    ""
                ])
        
        return "\n".join(doc_lines)


def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Schema validation and management tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Commande validate
    validate_parser = subparsers.add_parser('validate', help='Validate schemas')
    validate_parser.add_argument('path', type=Path, help='File or directory to validate')
    validate_parser.add_argument('--recursive', '-r', action='store_true', help='Recursive validation')
    validate_parser.add_argument('--output', '-o', type=Path, help='Output report file')
    
    # Commande generate
    generate_parser = subparsers.add_parser('generate', help='Generate schema from template')
    generate_parser.add_argument('schema_type', help='Schema type')
    generate_parser.add_argument('template_name', help='Template name')
    generate_parser.add_argument('--variables', '-v', type=str, help='Variables JSON string')
    generate_parser.add_argument('--output', '-o', type=Path, help='Output file')
    
    # Commande list-templates
    list_parser = subparsers.add_parser('list-templates', help='List available templates')
    
    # Commande docs
    docs_parser = subparsers.add_parser('docs', help='Generate documentation')
    docs_parser.add_argument('output_dir', type=Path, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = SchemaManager()
    
    if args.command == 'validate':
        if args.path.is_file():
            result = manager.validator.validate_file(args.path)
            results = {str(args.path): result}
        else:
            results = asyncio.run(manager.validate_directory(args.path, args.recursive))
        
        report = manager.generate_report(results)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
    
    elif args.command == 'generate':
        variables = json.loads(args.variables) if args.variables else {}
        
        try:
            schema = manager.generator.generate_schema(args.schema_type, args.template_name, variables)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    if args.output.suffix.lower() in ['.yml', '.yaml']:
                        yaml.dump(schema, f, default_flow_style=False)
                    else:
                        json.dump(schema, f, indent=2)
                print(f"Schema generated: {args.output}")
            else:
                print(json.dumps(schema, indent=2))
        
        except Exception as e:
            print(f"Error generating schema: {e}")
    
    elif args.command == 'list-templates':
        templates = manager.generator.list_templates()
        for schema_type, template_names in templates.items():
            print(f"{schema_type}:")
            for name in template_names:
                print(f"  - {name}")
            print()
    
    elif args.command == 'docs':
        manager.export_schema_documentation(args.output_dir)
        print(f"Documentation generated in {args.output_dir}")


if __name__ == '__main__':
    main()
