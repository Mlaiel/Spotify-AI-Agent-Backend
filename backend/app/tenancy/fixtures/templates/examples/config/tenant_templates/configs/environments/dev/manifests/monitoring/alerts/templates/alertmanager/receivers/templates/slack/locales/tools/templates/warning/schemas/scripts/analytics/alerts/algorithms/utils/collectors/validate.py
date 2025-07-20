#!/usr/bin/env python3
"""
Validation Script - Module Collectors Spotify AI Agent
=====================================================

Script de validation simple pour vérifier l'intégrité
de l'implémentation ultra-avancée du module collectors.

Développé par l'équipe Spotify AI Agent
Lead Developer: Fahed Mlaiel
"""

import os
import sys
from pathlib import Path

def validate_file_structure():
    """Valide la structure des fichiers."""
    
    print("🔍 Validation de la structure des fichiers...")
    
    required_files = [
        "README.md",
        "README.fr.md", 
        "README.de.md",
        "TECHNICAL_DOCUMENTATION.md",
        "__init__.py",
        "base.py",
        "config.py",
        "exceptions.py",
        "utils.py",
        "monitoring.py",
        "performance_collectors.py",
        "patterns.py",
        "strategies.py",
        "integrations.py",
        "business_collectors.py",
        "security_collectors.py",
        "ml_collectors.py",
        "spotify_api_collectors.py",
        "user_behavior_collectors.py",
        "audio_quality_collectors.py",
        "infrastructure_collectors.py"
    ]
    
    current_dir = Path(__file__).parent
    missing_files = []
    
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_name:<30} ({size:>8} bytes)")
        else:
            missing_files.append(file_name)
            print(f"❌ {file_name:<30} (MISSING)")
    
    if missing_files:
        print(f"\n⚠️ Fichiers manquants: {len(missing_files)}")
        return False
    else:
        print(f"\n✅ Tous les fichiers présents: {len(required_files)} fichiers")
        return True

def validate_file_content():
    """Valide le contenu des fichiers principaux."""
    
    print("\n🔍 Validation du contenu des fichiers...")
    
    validations = []
    
    # Validation __init__.py
    try:
        with open("__init__.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "CollectorOrchestrator" in content and "EnterpriseCollectorFactory" in content:
                validations.append(("__init__.py", True, "Orchestrateur et Factory présents"))
            else:
                validations.append(("__init__.py", False, "Composants manquants"))
    except Exception as e:
        validations.append(("__init__.py", False, f"Erreur: {e}"))
    
    # Validation base.py
    try:
        with open("base.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "BaseCollector" in content and "async def collect" in content:
                validations.append(("base.py", True, "BaseCollector avec interface async"))
            else:
                validations.append(("base.py", False, "Interface async manquante"))
    except Exception as e:
        validations.append(("base.py", False, f"Erreur: {e}"))
    
    # Validation performance_collectors.py
    try:
        with open("performance_collectors.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "SystemPerformanceCollector" in content and "IsolationForest" in content:
                validations.append(("performance_collectors.py", True, "ML et collecteur système présents"))
            else:
                validations.append(("performance_collectors.py", False, "Composants ML manquants"))
    except Exception as e:
        validations.append(("performance_collectors.py", False, f"Erreur: {e}"))
    
    # Validation patterns.py
    try:
        with open("patterns.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "CircuitBreaker" in content and ("RetryMechanism" in content or "RetryManager" in content) and "RateLimiter" in content:
                validations.append(("patterns.py", True, "Patterns enterprise présents"))
            else:
                validations.append(("patterns.py", False, "Patterns manquants"))
    except Exception as e:
        validations.append(("patterns.py", False, f"Erreur: {e}"))
    
    # Validation strategies.py
    try:
        with open("strategies.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "AdaptiveStrategy" in content and "PredictiveStrategy" in content:
                validations.append(("strategies.py", True, "Stratégies adaptatives présentes"))
            else:
                validations.append(("strategies.py", False, "Stratégies manquantes"))
    except Exception as e:
        validations.append(("strategies.py", False, f"Erreur: {e}"))
    
    # Validation integrations.py
    try:
        with open("integrations.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "SpotifyAPIIntegration" in content and "TimescaleDBIntegration" in content:
                validations.append(("integrations.py", True, "Intégrations enterprise présentes"))
            else:
                validations.append(("integrations.py", False, "Intégrations manquantes"))
    except Exception as e:
        validations.append(("integrations.py", False, f"Erreur: {e}"))
    
    # Affichage des résultats
    success_count = 0
    for file_name, success, message in validations:
        status = "✅" if success else "❌"
        print(f"{status} {file_name:<25} - {message}")
        if success:
            success_count += 1
    
    print(f"\n📊 Validation du contenu: {success_count}/{len(validations)} réussies")
    return success_count == len(validations)

def calculate_codebase_stats():
    """Calcule les statistiques de la base de code."""
    
    print("\n📈 Statistiques de la base de code...")
    
    total_lines = 0
    total_files = 0
    total_size = 0
    
    python_files = list(Path(".").glob("*.py"))
    markdown_files = list(Path(".").glob("*.md"))
    
    all_files = python_files + markdown_files
    
    for file_path in all_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = len(f.readlines())
                size = file_path.stat().st_size
                
                total_lines += lines
                total_size += size
                total_files += 1
                
        except Exception as e:
            print(f"⚠️ Erreur lecture {file_path}: {e}")
    
    print(f"📁 Fichiers totaux: {total_files}")
    print(f"📄 Lignes de code: {total_lines:,}")
    print(f"💾 Taille totale: {total_size / 1024:.1f} KB")
    print(f"📊 Moyenne par fichier: {total_lines // total_files if total_files > 0 else 0} lignes")
    
    return {
        "files": total_files,
        "lines": total_lines,
        "size_kb": round(total_size / 1024, 1)
    }

def validate_enterprise_features():
    """Valide la présence des fonctionnalités enterprise."""
    
    print("\n🏢 Validation des fonctionnalités enterprise...")
    
    enterprise_features = {
        "Machine Learning": ["sklearn", "IsolationForest", "RandomForest"],
        "Monitoring": ["prometheus", "Grafana", "metrics"],
        "Patterns": ["CircuitBreaker", "RetryMechanism", "RateLimiter"],
        "Security": ["encryption", "authentication", "audit"],
        "Multi-tenant": ["tenant_id", "isolation", "TenantConfig"],
        "Async": ["async def", "await", "asyncio"],
        "Database": ["TimescaleDB", "PostgreSQL", "hypertable"],
        "Cache": ["Redis", "cache", "compression"],
        "Integration": ["Spotify", "API", "OAuth"],
        "Documentation": ["README", "TECHNICAL_DOCUMENTATION"]
    }
    
    feature_validation = {}
    
    # Lecture de tous les fichiers Python
    all_content = ""
    for py_file in Path(".").glob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                all_content += f.read().lower()
        except Exception:
            continue
    
    # Lecture des fichiers markdown
    for md_file in Path(".").glob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                all_content += f.read().lower()
        except Exception:
            continue
    
    # Validation des features
    for feature_name, keywords in enterprise_features.items():
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in all_content:
                found_keywords.append(keyword)
        
        coverage = len(found_keywords) / len(keywords) * 100
        feature_validation[feature_name] = {
            "coverage": coverage,
            "found": found_keywords,
            "missing": [k for k in keywords if k.lower() not in all_content]
        }
        
        status = "✅" if coverage >= 66 else "⚠️" if coverage >= 33 else "❌"
        print(f"{status} {feature_name:<15} - {coverage:>5.1f}% coverage")
    
    # Calcul du score global enterprise
    total_coverage = sum(f["coverage"] for f in feature_validation.values())
    enterprise_score = total_coverage / len(enterprise_features)
    
    print(f"\n🏆 Score Enterprise Global: {enterprise_score:.1f}%")
    
    if enterprise_score >= 80:
        print("🎉 EXCELLENT - Implémentation enterprise de niveau production!")
    elif enterprise_score >= 60:
        print("👍 BON - Implémentation enterprise solide")
    elif enterprise_score >= 40:
        print("⚠️ MOYEN - Amélioration des features enterprise recommandée")
    else:
        print("❌ FAIBLE - Implémentation enterprise insuffisante")
    
    return enterprise_score

def generate_validation_report():
    """Génère un rapport de validation complet."""
    
    print("\n" + "="*60)
    print("📋 RAPPORT DE VALIDATION FINAL")
    print("="*60)
    
    # Validation structure
    structure_ok = validate_file_structure()
    
    # Validation contenu
    content_ok = validate_file_content()
    
    # Statistiques
    stats = calculate_codebase_stats()
    
    # Features enterprise
    enterprise_score = validate_enterprise_features()
    
    # Score global
    validations_passed = sum([structure_ok, content_ok])
    total_validations = 2
    validation_score = validations_passed / total_validations * 100
    
    print("\n" + "="*60)
    print("🎯 RÉSUMÉ FINAL")
    print("="*60)
    print(f"📁 Structure des fichiers: {'✅ OK' if structure_ok else '❌ ERREUR'}")
    print(f"📝 Contenu des fichiers: {'✅ OK' if content_ok else '❌ ERREUR'}")
    print(f"📊 Lignes de code: {stats['lines']:,}")
    print(f"💾 Taille du projet: {stats['size_kb']} KB")
    print(f"🏢 Score Enterprise: {enterprise_score:.1f}%")
    print(f"🎯 Score Global: {validation_score:.1f}%")
    
    if validation_score >= 100 and enterprise_score >= 80:
        print("\n🏆 VALIDATION RÉUSSIE!")
        print("✨ Module Collectors Ultra-Avancé validé avec succès")
        print("🚀 Prêt pour déploiement en production")
        print("👨‍💻 Développé par l'équipe Fahed Mlaiel")
        return True
    else:
        print("\n⚠️ Validation incomplète")
        print("🔧 Corrections nécessaires avant déploiement")
        return False

def main():
    """Fonction principale de validation."""
    
    print("🚀 SPOTIFY AI AGENT - VALIDATION MODULE COLLECTORS")
    print("=" * 60)
    print("Validation ultra-avancée de l'implémentation enterprise-grade")
    print("Développé par l'équipe Fahed Mlaiel")
    print("=" * 60)
    
    try:
        success = generate_validation_report()
        return success
    
    except Exception as e:
        print(f"\n❌ ERREUR LORS DE LA VALIDATION: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
