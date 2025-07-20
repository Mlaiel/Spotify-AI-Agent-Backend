#!/usr/bin/env python3
"""
🔧 Context Configuration Validator Script
=========================================

Script de validation et optimisation automatique de la configuration
du système de contexte tenant avec recommandations intelligentes.

Author: Lead Dev + Architecte IA - Fahed Mlaiel
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import argparse
from pathlib import Path

# Ajouter le chemin du module au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from app.tenancy.data_isolation.core import (
    ContextManager,
    TenantContext,
    TenantType,
    IsolationLevel,
    ValidationLevel,
    OptimizationStrategy
)


class ConfigurationValidator:
    """Validateur et optimiseur de configuration"""
    
    def __init__(self):
        self.validation_results = {}
        self.recommendations = []
        self.errors = []
        self.warnings = []
    
    async def validate_environment_variables(self) -> Dict[str, Any]:
        """Valide les variables d'environnement"""
        print("🔍 Validating environment variables...")
        
        required_vars = {
            'TENANT_ISOLATION_LEVEL': {
                'required': True,
                'valid_values': ['none', 'basic', 'strict', 'paranoid'],
                'default': 'strict'
            },
            'CACHE_SIZE_MB': {
                'required': False,
                'type': 'int',
                'min_value': 64,
                'max_value': 8192,
                'default': 2048
            },
            'CACHE_TTL_SECONDS': {
                'required': False,
                'type': 'int',
                'min_value': 60,
                'max_value': 3600,
                'default': 300
            },
            'SECURITY_PARANOID_MODE': {
                'required': False,
                'type': 'bool',
                'default': True
            },
            'PERFORMANCE_OPTIMIZATION': {
                'required': False,
                'valid_values': ['aggressive', 'balanced', 'conservative', 'adaptive'],
                'default': 'adaptive'
            },
            'COMPLIANCE_AUDIT_ENABLED': {
                'required': False,
                'type': 'bool',
                'default': True
            }
        }
        
        validation_results = {}
        
        for var_name, config in required_vars.items():
            value = os.getenv(var_name)
            var_result = {'status': 'ok', 'issues': []}
            
            # Vérification de la présence
            if config.get('required', False) and value is None:
                var_result['status'] = 'error'
                var_result['issues'].append(f"Required variable {var_name} is missing")
                self.errors.append(f"Missing required environment variable: {var_name}")
            
            if value is not None:
                # Validation du type
                if config.get('type') == 'int':
                    try:
                        int_value = int(value)
                        if 'min_value' in config and int_value < config['min_value']:
                            var_result['status'] = 'warning'
                            var_result['issues'].append(f"Value {int_value} is below minimum {config['min_value']}")
                        if 'max_value' in config and int_value > config['max_value']:
                            var_result['status'] = 'warning'
                            var_result['issues'].append(f"Value {int_value} exceeds maximum {config['max_value']}")
                    except ValueError:
                        var_result['status'] = 'error'
                        var_result['issues'].append(f"Invalid integer value: {value}")
                
                elif config.get('type') == 'bool':
                    if value.lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
                        var_result['status'] = 'warning'
                        var_result['issues'].append(f"Invalid boolean value: {value}")
                
                # Validation des valeurs autorisées
                if 'valid_values' in config and value not in config['valid_values']:
                    var_result['status'] = 'error'
                    var_result['issues'].append(f"Invalid value '{value}'. Valid values: {config['valid_values']}")
            
            # Valeur par défaut
            if value is None and 'default' in config:
                var_result['using_default'] = config['default']
                self.warnings.append(f"{var_name} not set, using default: {config['default']}")
            
            validation_results[var_name] = var_result
        
        print(f"   ✅ Environment validation completed")
        return validation_results
    
    async def validate_system_resources(self) -> Dict[str, Any]:
        """Valide les ressources système"""
        print("💻 Validating system resources...")
        
        import psutil
        
        # Mémoire
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Disque
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        
        results = {
            'memory': {
                'total_gb': round(memory_gb, 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'percent_used': memory.percent,
                'status': 'ok'
            },
            'cpu': {
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
                'status': 'ok'
            },
            'disk': {
                'free_gb': round(disk_free_gb, 2),
                'total_gb': round(disk.total / (1024**3), 2),
                'percent_used': round((disk.used / disk.total) * 100, 1),
                'status': 'ok'
            }
        }
        
        # Validation des seuils
        if memory_gb < 4:
            results['memory']['status'] = 'warning'
            self.warnings.append("Low memory: less than 4GB available")
        elif memory_gb < 2:
            results['memory']['status'] = 'error'
            self.errors.append("Insufficient memory: less than 2GB available")
        
        if cpu_count < 2:
            results['cpu']['status'] = 'warning'
            self.warnings.append("Low CPU count: less than 2 cores")
        
        if disk_free_gb < 5:
            results['disk']['status'] = 'warning'
            self.warnings.append("Low disk space: less than 5GB free")
        elif disk_free_gb < 1:
            results['disk']['status'] = 'error'
            self.errors.append("Critical disk space: less than 1GB free")
        
        print(f"   ✅ System resources validation completed")
        return results
    
    async def validate_context_configuration(self) -> Dict[str, Any]:
        """Valide la configuration du gestionnaire de contexte"""
        print("⚙️ Validating context configuration...")
        
        try:
            context_manager = ContextManager()
            
            # Test de création de contexte
            test_context = TenantContext(
                tenant_id="validation_test",
                tenant_type=TenantType.SPOTIFY_ARTIST,
                isolation_level=IsolationLevel.STRICT
            )
            
            # Test d'activation
            result = await context_manager.set_context(test_context)
            
            # Test de validation
            validation_result = await context_manager.validator.validate_context(test_context)
            
            # Statistiques du gestionnaire
            stats = context_manager.get_statistics()
            
            results = {
                'context_creation': 'ok' if result['success'] else 'error',
                'context_validation': 'ok' if validation_result['valid'] else 'warning',
                'validation_score': validation_result.get('score', 0),
                'validation_level': context_manager.validator.validation_level.value,
                'auto_optimization': context_manager.auto_optimization,
                'manager_stats': stats,
                'status': 'ok'
            }
            
            if not result['success']:
                self.errors.append(f"Context activation failed: {result.get('warnings', [])}")
                results['status'] = 'error'
            
            if not validation_result['valid']:
                self.warnings.append(f"Context validation issues: {validation_result.get('errors', [])}")
            
            # Nettoyage
            await context_manager.shutdown()
            
        except Exception as e:
            self.errors.append(f"Context configuration validation failed: {str(e)}")
            results = {'status': 'error', 'error': str(e)}
        
        print(f"   ✅ Context configuration validation completed")
        return results
    
    async def validate_performance_configuration(self) -> Dict[str, Any]:
        """Valide la configuration de performance"""
        print("⚡ Validating performance configuration...")
        
        try:
            from app.tenancy.data_isolation.core import PerformanceOptimizer
            
            optimizer = PerformanceOptimizer()
            
            # Test de configuration
            test_context = TenantContext(
                tenant_id="perf_test",
                tenant_type=TenantType.SPOTIFY_ARTIST,
                isolation_level=IsolationLevel.STRICT
            )
            
            # Test d'optimisation
            optimization_result = await optimizer.optimize_operation(
                operation_type="validation_test",
                context=test_context,
                data={"test": "data"}
            )
            
            stats = optimizer.get_statistics()
            
            results = {
                'optimizer_status': 'ok',
                'cache_size_mb': optimizer.cache.size_mb,
                'cache_max_size_mb': optimizer.cache.max_size_mb,
                'optimization_strategy': optimizer.current_strategy.value,
                'stats': stats,
                'test_optimization': 'ok' if optimization_result else 'warning'
            }
            
            # Validation des seuils
            if optimizer.cache.max_size_mb < 512:
                self.warnings.append("Cache size is quite small (< 512MB)")
            elif optimizer.cache.max_size_mb > 4096:
                self.warnings.append("Cache size is very large (> 4GB)")
            
        except Exception as e:
            self.errors.append(f"Performance configuration validation failed: {str(e)}")
            results = {'status': 'error', 'error': str(e)}
        
        print(f"   ✅ Performance configuration validation completed")
        return results
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        # Recommandations basées sur les erreurs
        if self.errors:
            recommendations.append({
                'priority': 'high',
                'category': 'critical_fixes',
                'title': 'Fix Critical Configuration Issues',
                'description': 'Resolve critical configuration errors that prevent proper operation',
                'actions': [f"Fix: {error}" for error in self.errors]
            })
        
        # Recommandations basées sur les avertissements
        if self.warnings:
            recommendations.append({
                'priority': 'medium',
                'category': 'optimizations',
                'title': 'Address Configuration Warnings',
                'description': 'Optimize configuration to improve performance and reliability',
                'actions': [f"Consider: {warning}" for warning in self.warnings]
            })
        
        # Recommandations générales
        recommendations.extend([
            {
                'priority': 'medium',
                'category': 'performance',
                'title': 'Enable Query Caching',
                'description': 'Enable query caching for better performance',
                'actions': [
                    'Set QUERY_CACHE_ENABLED=true',
                    'Configure appropriate cache TTL',
                    'Monitor cache hit ratios'
                ]
            },
            {
                'priority': 'low',
                'category': 'monitoring',
                'title': 'Setup Performance Monitoring',
                'description': 'Configure comprehensive monitoring for production use',
                'actions': [
                    'Enable metrics collection',
                    'Setup alerting for performance degradation',
                    'Configure log aggregation'
                ]
            },
            {
                'priority': 'low',
                'category': 'security',
                'title': 'Enhance Security Configuration',
                'description': 'Implement additional security measures',
                'actions': [
                    'Enable paranoid security mode for production',
                    'Configure audit logging',
                    'Setup compliance monitoring'
                ]
            }
        ])
        
        return recommendations
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Exécute la validation complète"""
        print("🔍 Starting comprehensive configuration validation...")
        print("=" * 60)
        
        validation_results = {}
        
        # Exécution des validations
        validation_results['environment'] = await self.validate_environment_variables()
        validation_results['system_resources'] = await self.validate_system_resources()
        validation_results['context_configuration'] = await self.validate_context_configuration()
        validation_results['performance_configuration'] = await self.validate_performance_configuration()
        
        # Génération des recommandations
        recommendations = self.generate_recommendations()
        
        # Résumé global
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        
        overall_status = 'ok'
        if total_errors > 0:
            overall_status = 'error'
        elif total_warnings > 0:
            overall_status = 'warning'
        
        summary = {
            'overall_status': overall_status,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': recommendations,
            'validation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        print("=" * 60)
        print(f"🏁 Validation completed!")
        print(f"   Status: {overall_status.upper()}")
        print(f"   Errors: {total_errors}")
        print(f"   Warnings: {total_warnings}")
        
        return {
            'summary': summary,
            'details': validation_results
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """Génère un rapport de validation"""
        summary = results['summary']
        details = results['details']
        
        status_emoji = {
            'ok': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        report_lines = [
            "# 🔧 Context Configuration Validation Report",
            "",
            f"**Generated:** {summary['validation_timestamp']}",
            f"**Overall Status:** {status_emoji.get(summary['overall_status'], '❓')} {summary['overall_status'].upper()}",
            f"**Errors:** {summary['total_errors']}",
            f"**Warnings:** {summary['total_warnings']}",
            "",
            "## 📋 Summary",
            ""
        ]
        
        # Erreurs
        if summary['errors']:
            report_lines.extend([
                "### ❌ Critical Issues",
                ""
            ])
            for error in summary['errors']:
                report_lines.append(f"- {error}")
            report_lines.append("")
        
        # Avertissements
        if summary['warnings']:
            report_lines.extend([
                "### ⚠️ Warnings",
                ""
            ])
            for warning in summary['warnings']:
                report_lines.append(f"- {warning}")
            report_lines.append("")
        
        # Détails par section
        sections = {
            'environment': '🌍 Environment Variables',
            'system_resources': '💻 System Resources',
            'context_configuration': '⚙️ Context Configuration',
            'performance_configuration': '⚡ Performance Configuration'
        }
        
        for section_key, section_title in sections.items():
            if section_key in details:
                report_lines.extend([
                    f"## {section_title}",
                    ""
                ])
                
                section_data = details[section_key]
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, dict) and 'status' in value:
                            status = value['status']
                            emoji = status_emoji.get(status, '❓')
                            report_lines.append(f"- **{key}:** {emoji} {status}")
                            
                            if 'issues' in value and value['issues']:
                                for issue in value['issues']:
                                    report_lines.append(f"  - {issue}")
                
                report_lines.append("")
        
        # Recommandations
        if summary['recommendations']:
            report_lines.extend([
                "## 💡 Recommendations",
                ""
            ])
            
            for rec in summary['recommendations']:
                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                emoji = priority_emoji.get(rec['priority'], '⚪')
                
                report_lines.extend([
                    f"### {emoji} {rec['title']} ({rec['priority']} priority)",
                    "",
                    rec['description'],
                    "",
                    "**Actions:**"
                ])
                
                for action in rec['actions']:
                    report_lines.append(f"- {action}")
                
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📝 Report saved to: {output_file}")
        
        return report


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Context Configuration Validator')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix configuration issues automatically')
    
    args = parser.parse_args()
    
    validator = ConfigurationValidator()
    
    try:
        # Exécution de la validation
        results = await validator.run_full_validation()
        
        # Génération du rapport
        if args.json:
            output = json.dumps(results, indent=2)
            print(output)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
        else:
            report = validator.generate_report(results, args.output)
            if not args.output:
                print(report)
        
        # Code de sortie basé sur le statut
        exit_code = 0
        if results['summary']['overall_status'] == 'error':
            exit_code = 1
        elif results['summary']['overall_status'] == 'warning':
            exit_code = 0  # Warnings n'empêchent pas l'exécution
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
