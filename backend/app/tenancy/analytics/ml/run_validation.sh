#!/bin/bash

# 🧠 Script de lancement rapide - Validation ML Ultra-Avancé
# Spotify AI Agent - Module ML Enterprise

echo "🧠 =========================================="
echo "   VALIDATION ML ULTRA-AVANCÉ"
echo "   Spotify AI Agent - Expert Team"
echo "=========================================="
echo ""

# Navigation vers le répertoire ML
cd "$(dirname "$0")"

echo "📁 Répertoire ML: $(pwd)"
echo "🔍 Vérification structure des fichiers..."

# Vérification présence des composants
components=(
    "__init__.py"
    "prediction_engine.py"
    "anomaly_detector.py"
    "neural_networks.py"
    "feature_engineer.py"
    "model_optimizer.py"
    "mlops_pipeline.py"
    "ensemble_methods.py"
    "data_preprocessor.py"
    "model_registry.py"
    "validate_ml_system.py"
)

missing_components=()

for component in "${components[@]}"; do
    if [[ -f "$component" ]]; then
        echo "✅ $component"
    else
        echo "❌ $component - MANQUANT"
        missing_components+=("$component")
    fi
done

echo ""

if [[ ${#missing_components[@]} -gt 0 ]]; then
    echo "🚨 ERREUR: Composants manquants détectés!"
    echo "   Composants manquants: ${missing_components[*]}"
    echo "   Impossible de continuer la validation."
    exit 1
fi

echo "✅ Tous les composants ML détectés!"
echo ""

# Vérification Python et dépendances
echo "🐍 Vérification environnement Python..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 non trouvé. Installation requise."
    exit 1
fi

python_version=$(python3 --version)
echo "✅ Python détecté: $python_version"

# Vérification des dépendances critiques
echo "📦 Vérification dépendances critiques..."

critical_deps=(
    "numpy"
    "pandas"
    "scikit-learn"
)

missing_deps=()

for dep in "${critical_deps[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "✅ $dep"
    else
        echo "❌ $dep - MANQUANT"
        missing_deps+=("$dep")
    fi
done

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    echo ""
    echo "⚠️  ATTENTION: Dépendances manquantes détectées!"
    echo "   Dépendances manquantes: ${missing_deps[*]}"
    echo "   Installation automatique..."
    
    for dep in "${missing_deps[@]}"; do
        echo "📦 Installation de $dep..."
        pip3 install "$dep" --quiet
        if [[ $? -eq 0 ]]; then
            echo "✅ $dep installé avec succès"
        else
            echo "❌ Échec installation $dep"
        fi
    done
fi

echo ""
echo "🚀 Lancement de la validation ML..."
echo "⏱️  Cela peut prendre quelques minutes..."
echo ""

# Lancement de la validation
python3 validate_ml_system.py

validation_exit_code=$?

echo ""
echo "🏁 Validation terminée avec code de sortie: $validation_exit_code"

case $validation_exit_code in
    0)
        echo "🎉 SUCCÈS COMPLET - Écosystème ML ultra-avancé opérationnel!"
        echo "🚀 Prêt pour production musicale industrielle"
        ;;
    1)
        echo "⚠️  SUCCÈS PARTIEL - Vérifications recommandées"
        echo "🔧 Certains composants nécessitent ajustements"
        ;;
    2)
        echo "❌ ÉCHEC - Intervention requise"
        echo "🚨 Problèmes critiques détectés"
        ;;
    3)
        echo "💥 ERREUR CRITIQUE - Échec catastrophique"
        echo "🆘 Support technique requis"
        ;;
    *)
        echo "❓ Code de sortie inattendu: $validation_exit_code"
        ;;
esac

echo ""
echo "📊 Résultats détaillés disponibles ci-dessus"
echo "🧠 Validation par équipe d'experts ML/AI"
echo "=========================================="

exit $validation_exit_code
