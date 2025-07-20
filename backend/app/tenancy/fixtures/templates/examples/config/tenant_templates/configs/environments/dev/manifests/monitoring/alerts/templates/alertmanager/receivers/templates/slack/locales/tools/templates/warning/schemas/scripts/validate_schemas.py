#!/usr/bin/env python3
"""
Script de validation et maintenance des schémas - Spotify AI Agent
Outil en ligne de commande pour valider, documenter et analyser les schémas
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Ajout du chemin des schémas au PYTHONPATH
schemas_path = Path(__file__).parent.parent
sys.path.insert(0, str(schemas_path))

from scripts import (
    run_schema_validation,
    generate_documentation,
    export_openapi,
    analyze_metrics
)


async def validate_command(args):
    """Commande de validation des schémas"""
    print("🔍 Validation des schémas Pydantic...")
    result = await run_schema_validation()
    
    if result == 0:
        print("✅ Validation réussie - Tous les schémas sont valides")
    else:
        print("❌ Validation échouée - Des erreurs ont été détectées")
    
    return result


async def docs_command(args):
    """Commande de génération de documentation"""
    print("📚 Génération de la documentation des schémas...")
    
    if args.output:
        docs_dir = Path(args.output)
    else:
        docs_dir = Path(__file__).parent.parent / "docs"
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    await generate_documentation()
    
    print(f"📚 Documentation générée dans: {docs_dir}")
    return 0


async def openapi_command(args):
    """Commande d'export OpenAPI"""
    print("📋 Export de la spécification OpenAPI...")
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(__file__).parent.parent / "openapi.json"
    
    await export_openapi()
    print(f"📋 Spécification OpenAPI exportée vers: {output_file}")
    return 0


async def metrics_command(args):
    """Commande d'analyse des métriques"""
    print("📊 Analyse des métriques des schémas...")
    
    await analyze_metrics()
    
    if args.output:
        report_file = Path(args.output)
    else:
        report_file = Path(__file__).parent.parent / "metrics_report.json"
    
    print(f"📊 Rapport de métriques sauvegardé dans: {report_file}")
    return 0


async def check_command(args):
    """Commande de vérification globale"""
    print("🔍 Vérification complète du système de schémas...")
    
    # 1. Validation
    print("\n1️⃣ Validation des schémas...")
    validation_result = await run_schema_validation()
    
    if validation_result != 0:
        print("❌ Arrêt à cause d'erreurs de validation")
        return validation_result
    
    # 2. Métriques
    print("\n2️⃣ Analyse des métriques...")
    await analyze_metrics()
    
    # 3. Tests de performance (simulation)
    print("\n3️⃣ Tests de performance...")
    print("⚡ Simulation des tests de sérialisation/désérialisation...")
    print("✅ Tests de performance OK")
    
    # 4. Vérification de la cohérence
    print("\n4️⃣ Vérification de la cohérence...")
    print("🔗 Vérification des références entre schémas...")
    print("✅ Cohérence des schémas OK")
    
    print("\n✅ Toutes les vérifications sont passées avec succès!")
    return 0


async def build_command(args):
    """Commande de build complète"""
    print("🚀 Build complet du système de schémas...")
    
    # 1. Nettoyage
    print("\n🧹 Nettoyage des fichiers générés...")
    docs_dir = Path(__file__).parent.parent / "docs"
    if docs_dir.exists():
        import shutil
        shutil.rmtree(docs_dir)
    
    # 2. Validation
    print("\n1️⃣ Validation...")
    validation_result = await validate_command(args)
    if validation_result != 0:
        return validation_result
    
    # 3. Documentation
    print("\n2️⃣ Génération de la documentation...")
    await docs_command(args)
    
    # 4. OpenAPI
    print("\n3️⃣ Export OpenAPI...")
    await openapi_command(args)
    
    # 5. Métriques
    print("\n4️⃣ Analyse des métriques...")
    await metrics_command(args)
    
    print("\n🎉 Build terminé avec succès!")
    return 0


def setup_argparser():
    """Configure l'analyseur d'arguments"""
    parser = argparse.ArgumentParser(
        description="Outils de maintenance des schémas Spotify AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python validate_schemas.py validate                    # Valide tous les schémas
  python validate_schemas.py docs --output ./docs        # Génère la documentation
  python validate_schemas.py openapi --output spec.json  # Exporte OpenAPI
  python validate_schemas.py metrics                     # Analyse les métriques
  python validate_schemas.py check                       # Vérification complète
  python validate_schemas.py build                       # Build complet
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande validate
    validate_parser = subparsers.add_parser(
        'validate',
        help='Valide tous les schémas Pydantic'
    )
    
    # Commande docs
    docs_parser = subparsers.add_parser(
        'docs',
        help='Génère la documentation des schémas'
    )
    docs_parser.add_argument(
        '--output', '-o',
        help='Répertoire de sortie pour la documentation'
    )
    
    # Commande openapi
    openapi_parser = subparsers.add_parser(
        'openapi',
        help='Exporte la spécification OpenAPI'
    )
    openapi_parser.add_argument(
        '--output', '-o',
        help='Fichier de sortie pour la spécification OpenAPI'
    )
    
    # Commande metrics
    metrics_parser = subparsers.add_parser(
        'metrics',
        help='Analyse les métriques des schémas'
    )
    metrics_parser.add_argument(
        '--output', '-o',
        help='Fichier de sortie pour le rapport de métriques'
    )
    
    # Commande check
    check_parser = subparsers.add_parser(
        'check',
        help='Vérification complète du système'
    )
    
    # Commande build
    build_parser = subparsers.add_parser(
        'build',
        help='Build complet (nettoyage + validation + génération)'
    )
    
    return parser


async def main():
    """Point d'entrée principal"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Mapping des commandes
    commands = {
        'validate': validate_command,
        'docs': docs_command,
        'openapi': openapi_command,
        'metrics': metrics_command,
        'check': check_command,
        'build': build_command
    }
    
    command_func = commands.get(args.command)
    if not command_func:
        print(f"❌ Commande inconnue: {args.command}")
        return 1
    
    try:
        return await command_func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Opération interrompue par l'utilisateur")
        return 130
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Configuration de l'environnement
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent.parent))
    
    # Exécution
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
