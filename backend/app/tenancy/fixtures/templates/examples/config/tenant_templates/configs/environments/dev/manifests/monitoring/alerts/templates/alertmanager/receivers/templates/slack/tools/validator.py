#!/usr/bin/env python3
"""
Validateur de templates Slack pour Alertmanager
Auteur: Fahed Mlaiel
Date: 2025-07-19

Ce script valide les templates Slack et vérifie leur conformité
avec les standards Slack et Alertmanager.
"""

import os
import re
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse
import requests


class SlackTemplateValidator:
    """Validateur de templates Slack pour Alertmanager."""
    
    def __init__(self, templates_dir: str):
        """
        Initialise le validateur.
        
        Args:
            templates_dir: Répertoire contenant les templates
        """
        self.templates_dir = Path(templates_dir)
        self.errors = []
        self.warnings = []
        
        # Règles de validation
        self.slack_limits = {
            'text_max_length': 4000,
            'attachment_text_max_length': 8000,
            'fields_max_count': 10,
            'attachments_max_count': 20,
            'field_title_max_length': 300,
            'field_value_max_length': 2000
        }
        
        # Couleurs Slack valides
        self.valid_colors = [
            'good', 'warning', 'danger',
            '#36a64f', '#ff9900', '#ff0000',
            '#439fe0', '#9c27b0', '#ff5722'
        ]
    
    def validate_slack_webhook_url(self, url: str) -> bool:
        """Valide une URL de webhook Slack."""
        if not url:
            return False
        
        parsed = urlparse(url)
        return (
            parsed.scheme in ['http', 'https'] and
            'hooks.slack.com' in parsed.netloc and
            '/services/' in parsed.path
        )
    
    def validate_slack_message_structure(self, message: Dict[str, Any]) -> List[str]:
        """Valide la structure d'un message Slack."""
        errors = []
        
        # Vérifier les champs requis
        if 'text' not in message and 'attachments' not in message:
            errors.append("Le message doit contenir 'text' ou 'attachments'")
        
        # Valider le texte principal
        if 'text' in message:
            text = message['text']
            if len(text) > self.slack_limits['text_max_length']:
                errors.append(f"Texte trop long: {len(text)} > {self.slack_limits['text_max_length']}")
        
        # Valider les attachements
        if 'attachments' in message:
            attachments = message['attachments']
            
            if not isinstance(attachments, list):
                errors.append("'attachments' doit être une liste")
            elif len(attachments) > self.slack_limits['attachments_max_count']:
                errors.append(f"Trop d'attachements: {len(attachments)} > {self.slack_limits['attachments_max_count']}")
            else:
                for i, attachment in enumerate(attachments):
                    errors.extend(self._validate_attachment(attachment, i))
        
        return errors
    
    def _validate_attachment(self, attachment: Dict[str, Any], index: int) -> List[str]:
        """Valide un attachement Slack."""
        errors = []
        prefix = f"Attachement {index}: "
        
        # Valider la couleur
        if 'color' in attachment:
            color = attachment['color']
            if color not in self.valid_colors and not re.match(r'^#[0-9a-fA-F]{6}$', color):
                errors.append(f"{prefix}Couleur invalide: {color}")
        
        # Valider le texte de l'attachement
        if 'text' in attachment:
            text = attachment['text']
            if len(text) > self.slack_limits['attachment_text_max_length']:
                errors.append(f"{prefix}Texte trop long: {len(text)} > {self.slack_limits['attachment_text_max_length']}")
        
        # Valider les champs
        if 'fields' in attachment:
            fields = attachment['fields']
            if not isinstance(fields, list):
                errors.append(f"{prefix}'fields' doit être une liste")
            elif len(fields) > self.slack_limits['fields_max_count']:
                errors.append(f"{prefix}Trop de champs: {len(fields)} > {self.slack_limits['fields_max_count']}")
            else:
                for j, field in enumerate(fields):
                    errors.extend(self._validate_field(field, index, j))
        
        # Valider les actions (boutons)
        if 'actions' in attachment:
            actions = attachment['actions']
            if not isinstance(actions, list):
                errors.append(f"{prefix}'actions' doit être une liste")
            elif len(actions) > 5:  # Limite Slack pour les actions
                errors.append(f"{prefix}Trop d'actions: {len(actions)} > 5")
        
        return errors
    
    def _validate_field(self, field: Dict[str, Any], attachment_index: int, field_index: int) -> List[str]:
        """Valide un champ d'attachement."""
        errors = []
        prefix = f"Attachement {attachment_index}, Champ {field_index}: "
        
        # Champs requis
        if 'title' not in field:
            errors.append(f"{prefix}'title' requis")
        elif len(field['title']) > self.slack_limits['field_title_max_length']:
            errors.append(f"{prefix}Titre trop long: {len(field['title'])} > {self.slack_limits['field_title_max_length']}")
        
        if 'value' not in field:
            errors.append(f"{prefix}'value' requis")
        elif len(str(field['value'])) > self.slack_limits['field_value_max_length']:
            errors.append(f"{prefix}Valeur trop longue: {len(str(field['value']))} > {self.slack_limits['field_value_max_length']}")
        
        # Valider le champ 'short'
        if 'short' in field and not isinstance(field['short'], bool):
            errors.append(f"{prefix}'short' doit être un booléen")
        
        return errors
    
    def validate_alertmanager_template(self, template_content: str) -> List[str]:
        """Valide un template Alertmanager."""
        errors = []
        
        # Vérifier les variables Alertmanager
        alertmanager_vars = [
            '{{ range .Alerts }}',
            '{{ .Status }}',
            '{{ .Labels.alertname }}',
            '{{ .Annotations.summary }}',
            '{{ .Annotations.description }}'
        ]
        
        for var in alertmanager_vars:
            if var in template_content:
                # C'est bien un template Alertmanager
                break
        else:
            errors.append("Aucune variable Alertmanager détectée")
        
        # Vérifier la syntaxe des templates Go
        go_template_errors = self._validate_go_template_syntax(template_content)
        errors.extend(go_template_errors)
        
        return errors
    
    def _validate_go_template_syntax(self, template_content: str) -> List[str]:
        """Valide la syntaxe des templates Go."""
        errors = []
        
        # Vérifier les accolades équilibrées
        open_braces = template_content.count('{{')
        close_braces = template_content.count('}}')
        
        if open_braces != close_braces:
            errors.append(f"Accolades déséquilibrées: {open_braces} ouvertes, {close_braces} fermées")
        
        # Vérifier les structures de contrôle
        range_opens = len(re.findall(r'{{\s*range\s+', template_content))
        range_closes = len(re.findall(r'{{\s*end\s*}}', template_content))
        
        if range_opens != range_closes:
            errors.append(f"Structures range déséquilibrées: {range_opens} ouvertures, {range_closes} fermetures")
        
        # Vérifier les pipes valides
        invalid_pipes = re.findall(r'{{\s*[^}]*\|\s*[^}]*\|\s*[^}]*}}', template_content)
        if invalid_pipes:
            errors.append(f"Pipes potentiellement invalides détectés: {invalid_pipes}")
        
        return errors
    
    def validate_template_file(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """Valide un fichier de template."""
        errors = []
        warnings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Essayer de parser comme YAML
            try:
                yaml_content = yaml.safe_load(content)
                
                # Valider la structure Slack
                if isinstance(yaml_content, dict):
                    slack_errors = self.validate_slack_message_structure(yaml_content)
                    errors.extend(slack_errors)
                
            except yaml.YAMLError as e:
                # Ce n'est pas du YAML valide, traiter comme template brut
                warnings.append(f"Pas du YAML valide (probablement un template): {e}")
            
            # Valider comme template Alertmanager
            alertmanager_errors = self.validate_alertmanager_template(content)
            errors.extend(alertmanager_errors)
            
            # Vérifications générales
            if len(content) == 0:
                errors.append("Fichier vide")
            
            if len(content) > 100000:  # 100KB
                warnings.append(f"Fichier très volumineux: {len(content)} caractères")
            
        except Exception as e:
            errors.append(f"Erreur lors de la lecture du fichier: {e}")
        
        return errors, warnings
    
    def validate_all_templates(self) -> Dict[str, Any]:
        """Valide tous les templates dans le répertoire."""
        results = {
            'total_files': 0,
            'valid_files': 0,
            'files_with_errors': 0,
            'files_with_warnings': 0,
            'details': {}
        }
        
        template_files = list(self.templates_dir.rglob('*.yaml')) + \
                        list(self.templates_dir.rglob('*.yml')) + \
                        list(self.templates_dir.rglob('*.j2')) + \
                        list(self.templates_dir.rglob('*.tmpl'))
        
        results['total_files'] = len(template_files)
        
        for file_path in template_files:
            relative_path = file_path.relative_to(self.templates_dir)
            
            errors, warnings = self.validate_template_file(file_path)
            
            results['details'][str(relative_path)] = {
                'errors': errors,
                'warnings': warnings,
                'is_valid': len(errors) == 0
            }
            
            if len(errors) == 0:
                results['valid_files'] += 1
            else:
                results['files_with_errors'] += 1
            
            if len(warnings) > 0:
                results['files_with_warnings'] += 1
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Génère un rapport de validation."""
        report = []
        report.append("RAPPORT DE VALIDATION DES TEMPLATES SLACK")
        report.append("=" * 50)
        report.append("")
        
        # Résumé
        report.append("📊 RÉSUMÉ:")
        report.append(f"  Fichiers totaux: {results['total_files']}")
        report.append(f"  Fichiers valides: {results['valid_files']}")
        report.append(f"  Fichiers avec erreurs: {results['files_with_errors']}")
        report.append(f"  Fichiers avec avertissements: {results['files_with_warnings']}")
        report.append("")
        
        # Taux de succès
        if results['total_files'] > 0:
            success_rate = (results['valid_files'] / results['total_files']) * 100
            report.append(f"✅ Taux de succès: {success_rate:.1f}%")
        else:
            report.append("❌ Aucun fichier trouvé")
        
        report.append("")
        
        # Détails par fichier
        report.append("📋 DÉTAILS PAR FICHIER:")
        report.append("-" * 30)
        
        for file_path, details in results['details'].items():
            status = "✅" if details['is_valid'] else "❌"
            report.append(f"{status} {file_path}")
            
            if details['errors']:
                report.append("  ERREURS:")
                for error in details['errors']:
                    report.append(f"    - {error}")
            
            if details['warnings']:
                report.append("  AVERTISSEMENTS:")
                for warning in details['warnings']:
                    report.append(f"    - {warning}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Validateur de templates Slack pour Alertmanager"
    )
    
    parser.add_argument(
        '--templates-dir',
        default='/workspaces/Achiri/spotify-ai-agent/backend/app/tenancy/fixtures/templates/examples/config/tenant_templates/configs/environments/dev/manifests/monitoring/alerts/templates/alertmanager/receivers/templates/slack',
        help='Répertoire contenant les templates'
    )
    
    parser.add_argument(
        '--output-report',
        help='Fichier de sortie pour le rapport'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Format du rapport'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Mode strict (échoue si des avertissements)'
    )
    
    args = parser.parse_args()
    
    # Initialiser le validateur
    validator = SlackTemplateValidator(args.templates_dir)
    
    print("🔍 Validation des templates Slack...")
    
    # Valider tous les templates
    results = validator.validate_all_templates()
    
    # Générer le rapport
    if args.format == 'json':
        report = json.dumps(results, indent=2, ensure_ascii=False)
    else:
        report = validator.generate_report(results)
    
    # Afficher ou sauvegarder le rapport
    if args.output_report:
        with open(args.output_report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Rapport sauvegardé: {args.output_report}")
    else:
        print(report)
    
    # Code de sortie
    exit_code = 0
    
    if results['files_with_errors'] > 0:
        exit_code = 1
        print("\n❌ Validation échouée: erreurs détectées")
    elif args.strict and results['files_with_warnings'] > 0:
        exit_code = 1
        print("\n⚠️ Validation échouée: avertissements en mode strict")
    else:
        print("\n✅ Validation réussie!")
    
    exit(exit_code)


if __name__ == '__main__':
    main()
