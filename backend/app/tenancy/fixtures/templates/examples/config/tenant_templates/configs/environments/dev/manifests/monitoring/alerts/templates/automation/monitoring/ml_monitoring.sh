#!/bin/bash

# ================================================================
# Script de Surveillance ML Ultra-Avancé - Spotify AI Agent
# ================================================================
# Auteur: Fahed Mlaiel  
# Équipe: Lead Dev + Architecte IA, Développeur Backend Senior,
#         Ingénieur Machine Learning, Spécialiste Sécurité Backend,
#         Architecte Microservices
# Version: 1.0.0
# Description: Surveillance intelligente des modèles ML avec drift detection
# ================================================================

set -euo pipefail

# Configuration globale
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/ml-monitoring.log"
ML_METRICS_DIR="/var/lib/ml-metrics"
MODEL_REGISTRY_DIR="/opt/ml-models"
DRIFT_THRESHOLD=0.1
PERFORMANCE_THRESHOLD=0.85

# Couleurs pour la sortie
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# ================================================================
# FONCTIONS UTILITAIRES
# ================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "ML")    echo -e "${PURPLE}[ML]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
        "DRIFT") echo -e "${CYAN}[DRIFT]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE" ;;
    esac
}

init_ml_monitoring() {
    log "INFO" "Initialisation du système de surveillance ML"
    
    # Création des répertoires
    mkdir -p "$ML_METRICS_DIR"/{models,predictions,drift,performance,alerts}
    mkdir -p "$MODEL_REGISTRY_DIR"/{production,staging,archive}
    
    # Vérification des dépendances Python ML
    local ml_deps=("numpy" "pandas" "scikit-learn" "tensorflow" "torch" "matplotlib" "seaborn" "scipy")
    local missing_deps=()
    
    for dep in "${ml_deps[@]}"; do
        if ! python3 -c "import $dep" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "WARN" "Dépendances ML manquantes: ${missing_deps[*]}"
        log "INFO" "Installation des dépendances manquantes..."
        pip3 install "${missing_deps[@]}" || log "ERROR" "Échec installation dépendances"
    fi
    
    log "INFO" "Système de surveillance ML initialisé"
}

# ================================================================
# SURVEILLANCE DES MODÈLES EN PRODUCTION
# ================================================================

monitor_production_models() {
    log "ML" "Surveillance des modèles ML en production"
    
    # Découverte des modèles actifs
    local active_models=$(discover_active_models)
    
    if [ -z "$active_models" ]; then
        log "WARN" "Aucun modèle ML actif détecté"
        return 1
    fi
    
    log "INFO" "Modèles actifs détectés: $active_models"
    
    # Surveillance de chaque modèle
    for model in $active_models; do
        log "ML" "Surveillance du modèle: $model"
        
        # Métriques de performance
        collect_model_performance_metrics "$model"
        
        # Détection de dérive (drift)
        detect_model_drift "$model"
        
        # Surveillance des prédictions
        monitor_prediction_quality "$model"
        
        # Analyse de la distribution des données
        analyze_data_distribution "$model"
        
        # Surveillance des ressources utilisées
        monitor_model_resources "$model"
    done
    
    # Génération du rapport de surveillance
    generate_monitoring_report
}

discover_active_models() {
    # Dans un vrai système, interroger l'API ou le registre de modèles
    # Simulation de modèles actifs
    echo "music_recommendation user_preference audio_classification sentiment_analysis"
}

collect_model_performance_metrics() {
    local model_name="$1"
    log "ML" "Collecte des métriques de performance pour: $model_name"
    
    # Script Python pour collecter les métriques détaillées
    python3 << EOF
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def simulate_model_metrics(model_name):
    """Simulation de métriques réelles - dans un vrai système, collecter depuis MLflow/TensorBoard"""
    
    # Métriques de base avec variation réaliste
    base_accuracy = {
        'music_recommendation': 0.87,
        'user_preference': 0.82,
        'audio_classification': 0.91,
        'sentiment_analysis': 0.79
    }.get(model_name, 0.80)
    
    # Ajout de variations réalistes
    current_accuracy = base_accuracy + random.uniform(-0.05, 0.02)
    
    metrics = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'performance': {
            'accuracy': round(current_accuracy, 4),
            'precision': round(current_accuracy + random.uniform(-0.03, 0.03), 4),
            'recall': round(current_accuracy + random.uniform(-0.02, 0.02), 4),
            'f1_score': round(current_accuracy + random.uniform(-0.02, 0.02), 4),
            'auc_roc': round(current_accuracy + random.uniform(-0.01, 0.04), 4)
        },
        'inference': {
            'latency_ms': random.randint(50, 200),
            'throughput_rps': random.randint(100, 1000),
            'memory_usage_mb': random.randint(512, 2048),
            'cpu_usage_percent': random.randint(20, 80)
        },
        'data_quality': {
            'missing_values_percent': round(random.uniform(0, 5), 2),
            'outliers_percent': round(random.uniform(0, 2), 2),
            'feature_correlation_change': round(random.uniform(-0.1, 0.1), 3)
        },
        'business_metrics': {
            'prediction_confidence_avg': round(random.uniform(0.7, 0.95), 3),
            'user_feedback_score': round(random.uniform(3.5, 4.8), 2),
            'conversion_rate': round(random.uniform(0.05, 0.15), 3)
        }
    }
    
    return metrics

def detect_performance_degradation(current_metrics, model_name):
    """Détection de dégradation des performances"""
    issues = []
    perf = current_metrics['performance']
    
    # Seuils critiques
    if perf['accuracy'] < 0.75:
        issues.append({
            'type': 'accuracy_degradation',
            'severity': 'critical',
            'value': perf['accuracy'],
            'threshold': 0.75,
            'description': f'Précision du modèle {model_name} critique'
        })
    elif perf['accuracy'] < $PERFORMANCE_THRESHOLD:
        issues.append({
            'type': 'accuracy_warning',
            'severity': 'warning',
            'value': perf['accuracy'],
            'threshold': $PERFORMANCE_THRESHOLD,
            'description': f'Précision du modèle {model_name} en baisse'
        })
    
    # Vérification latence
    if current_metrics['inference']['latency_ms'] > 500:
        issues.append({
            'type': 'latency_high',
            'severity': 'warning',
            'value': current_metrics['inference']['latency_ms'],
            'threshold': 500,
            'description': f'Latence élevée pour {model_name}'
        })
    
    # Vérification ressources
    if current_metrics['inference']['memory_usage_mb'] > 1500:
        issues.append({
            'type': 'memory_high',
            'severity': 'warning',
            'value': current_metrics['inference']['memory_usage_mb'],
            'threshold': 1500,
            'description': f'Utilisation mémoire élevée pour {model_name}'
        })
    
    return issues

# Collecte des métriques
model_metrics = simulate_model_metrics('$model_name')

# Détection des problèmes
performance_issues = detect_performance_degradation(model_metrics, '$model_name')

# Sauvegarde des métriques
metrics_file = f'$ML_METRICS_DIR/models/metrics_{model_metrics["model_name"]}_{int(datetime.now().timestamp())}.json'

# Ajout des issues détectées
model_metrics['performance_issues'] = performance_issues

with open(metrics_file, 'w') as f:
    json.dump(model_metrics, f, indent=2)

print(f"Métriques collectées: {metrics_file}")

# Alerte si problèmes critiques
critical_issues = [i for i in performance_issues if i['severity'] == 'critical']
if critical_issues:
    print(f"ALERTE CRITIQUE: {len(critical_issues)} problèmes détectés pour {model_metrics['model_name']}")
    for issue in critical_issues:
        print(f"  - {issue['description']}: {issue['value']}")

EOF
    
    log "ML" "Métriques de performance collectées pour: $model_name"
}

detect_model_drift() {
    local model_name="$1"
    log "DRIFT" "Détection de dérive pour le modèle: $model_name"
    
    # Analyse de dérive sophistiquée avec Python
    python3 << EOF
import json
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import random

def kolmogorov_smirnov_test(reference_data, current_data):
    """Test de Kolmogorov-Smirnov pour détecter la dérive"""
    try:
        statistic, p_value = stats.ks_2samp(reference_data, current_data)
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': p_value < 0.05,
            'test_name': 'kolmogorov_smirnov'
        }
    except Exception as e:
        return {'error': str(e)}

def population_stability_index(reference_data, current_data, bins=10):
    """Calcul du Population Stability Index (PSI)"""
    try:
        # Création des bins basés sur les données de référence
        _, bin_edges = np.histogram(reference_data, bins=bins)
        
        # Distribution des données de référence et actuelles
        ref_hist, _ = np.histogram(reference_data, bins=bin_edges)
        cur_hist, _ = np.histogram(current_data, bins=bin_edges)
        
        # Normalisation
        ref_pct = ref_hist / len(reference_data)
        cur_pct = cur_hist / len(current_data)
        
        # Éviter les divisions par zéro
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
        
        # Calcul PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        # Interprétation PSI
        if psi < 0.1:
            drift_level = 'no_drift'
        elif psi < 0.2:
            drift_level = 'minor_drift'
        else:
            drift_level = 'major_drift'
        
        return {
            'psi_score': float(psi),
            'drift_level': drift_level,
            'drift_detected': psi > $DRIFT_THRESHOLD,
            'test_name': 'population_stability_index'
        }
    except Exception as e:
        return {'error': str(e)}

def jensen_shannon_divergence(reference_data, current_data, bins=50):
    """Calcul de la divergence de Jensen-Shannon"""
    try:
        # Histogrammes
        ref_hist, bin_edges = np.histogram(reference_data, bins=bins, density=True)
        cur_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
        
        # Normalisation
        ref_hist = ref_hist / np.sum(ref_hist)
        cur_hist = cur_hist / np.sum(cur_hist)
        
        # Éviter les valeurs nulles
        ref_hist = np.where(ref_hist == 0, 1e-10, ref_hist)
        cur_hist = np.where(cur_hist == 0, 1e-10, cur_hist)
        
        # Distribution moyenne
        avg_hist = (ref_hist + cur_hist) / 2
        
        # Calcul divergence KL
        kl_ref = np.sum(ref_hist * np.log(ref_hist / avg_hist))
        kl_cur = np.sum(cur_hist * np.log(cur_hist / avg_hist))
        
        # Jensen-Shannon divergence
        js_divergence = (kl_ref + kl_cur) / 2
        
        return {
            'js_divergence': float(js_divergence),
            'drift_detected': js_divergence > 0.1,
            'test_name': 'jensen_shannon_divergence'
        }
    except Exception as e:
        return {'error': str(e)}

def generate_synthetic_data(model_name, data_type='reference'):
    """Génération de données synthétiques pour simulation"""
    
    # Paramètres par modèle
    model_params = {
        'music_recommendation': {'mean': 0.5, 'std': 0.2, 'size': 1000},
        'user_preference': {'mean': 0.3, 'std': 0.15, 'size': 800},
        'audio_classification': {'mean': 0.7, 'std': 0.25, 'size': 1200},
        'sentiment_analysis': {'mean': 0.4, 'std': 0.18, 'size': 900}
    }
    
    params = model_params.get(model_name, {'mean': 0.5, 'std': 0.2, 'size': 1000})
    
    if data_type == 'current':
        # Simulation de dérive en modifiant légèrement les paramètres
        drift_factor = random.uniform(0.8, 1.3)
        params['mean'] *= drift_factor
        params['std'] *= random.uniform(0.9, 1.2)
    
    # Génération des données
    data = np.random.normal(params['mean'], params['std'], params['size'])
    return np.clip(data, 0, 1)  # Limitation entre 0 et 1

def analyze_feature_drift(model_name):
    """Analyse complète de la dérive des features"""
    
    # Génération des données de référence et actuelles
    reference_data = generate_synthetic_data(model_name, 'reference')
    current_data = generate_synthetic_data(model_name, 'current')
    
    drift_results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'data_stats': {
            'reference': {
                'mean': float(np.mean(reference_data)),
                'std': float(np.std(reference_data)),
                'min': float(np.min(reference_data)),
                'max': float(np.max(reference_data)),
                'samples': len(reference_data)
            },
            'current': {
                'mean': float(np.mean(current_data)),
                'std': float(np.std(current_data)),
                'min': float(np.min(current_data)),
                'max': float(np.max(current_data)),
                'samples': len(current_data)
            }
        },
        'drift_tests': {}
    }
    
    # Tests de dérive
    drift_results['drift_tests']['ks_test'] = kolmogorov_smirnov_test(reference_data, current_data)
    drift_results['drift_tests']['psi_test'] = population_stability_index(reference_data, current_data)
    drift_results['drift_tests']['js_divergence'] = jensen_shannon_divergence(reference_data, current_data)
    
    # Analyse globale de la dérive
    drift_detected_count = sum(1 for test in drift_results['drift_tests'].values() 
                              if isinstance(test, dict) and test.get('drift_detected', False))
    
    drift_results['overall_drift'] = {
        'drift_detected': drift_detected_count >= 2,
        'confidence_level': drift_detected_count / 3,
        'recommendation': 'retrain_model' if drift_detected_count >= 2 else 'continue_monitoring'
    }
    
    return drift_results

# Analyse de dérive
drift_analysis = analyze_feature_drift('$model_name')

# Sauvegarde des résultats
drift_file = f'$ML_METRICS_DIR/drift/drift_analysis_{drift_analysis["model_name"]}_{int(datetime.now().timestamp())}.json'

with open(drift_file, 'w') as f:
    json.dump(drift_analysis, f, indent=2)

print(f"Analyse de dérive terminée: {drift_file}")

# Alertes de dérive
if drift_analysis['overall_drift']['drift_detected']:
    print(f"ALERTE DÉRIVE: Dérive détectée pour {drift_analysis['model_name']}")
    print(f"Confiance: {drift_analysis['overall_drift']['confidence_level']:.2f}")
    print(f"Recommandation: {drift_analysis['overall_drift']['recommendation']}")
    
    # Tests ayant détecté une dérive
    for test_name, result in drift_analysis['drift_tests'].items():
        if isinstance(result, dict) and result.get('drift_detected', False):
            print(f"  - {test_name}: Dérive confirmée")

EOF
    
    log "DRIFT" "Analyse de dérive terminée pour: $model_name"
}

monitor_prediction_quality() {
    local model_name="$1"
    log "ML" "Surveillance de la qualité des prédictions: $model_name"
    
    python3 << EOF
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def analyze_prediction_quality(model_name):
    """Analyse de la qualité des prédictions en temps réel"""
    
    # Simulation des prédictions récentes (dernières 24h)
    num_predictions = random.randint(1000, 5000)
    
    predictions = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'analysis_period': '24h',
        'total_predictions': num_predictions,
        'quality_metrics': {}
    }
    
    # Simulation des scores de confiance
    confidence_scores = np.random.beta(8, 2, num_predictions)  # Distribution biaisée vers les hautes valeurs
    
    predictions['quality_metrics']['confidence'] = {
        'mean': float(np.mean(confidence_scores)),
        'median': float(np.median(confidence_scores)),
        'std': float(np.std(confidence_scores)),
        'min': float(np.min(confidence_scores)),
        'max': float(np.max(confidence_scores)),
        'low_confidence_percent': float(np.sum(confidence_scores < 0.5) / len(confidence_scores) * 100)
    }
    
    # Simulation de la distribution des prédictions
    prediction_values = np.random.dirichlet(np.ones(5), num_predictions)  # 5 classes
    
    predictions['quality_metrics']['distribution'] = {
        'class_balance': [float(np.mean(prediction_values[:, i])) for i in range(5)],
        'entropy': float(np.mean([-np.sum(p * np.log(p + 1e-10)) for p in prediction_values])),
        'uniformity_score': float(1 - np.std([np.mean(prediction_values[:, i]) for i in range(5)]))
    }
    
    # Détection d'anomalies dans les prédictions
    anomalies = []
    
    # Anomalie: trop de prédictions avec faible confiance
    if predictions['quality_metrics']['confidence']['low_confidence_percent'] > 20:
        anomalies.append({
            'type': 'low_confidence_spike',
            'severity': 'warning',
            'value': predictions['quality_metrics']['confidence']['low_confidence_percent'],
            'threshold': 20,
            'description': f'Trop de prédictions à faible confiance pour {model_name}'
        })
    
    # Anomalie: distribution déséquilibrée
    max_class_prob = max(predictions['quality_metrics']['distribution']['class_balance'])
    if max_class_prob > 0.7:
        anomalies.append({
            'type': 'class_imbalance',
            'severity': 'warning',
            'value': max_class_prob,
            'threshold': 0.7,
            'description': f'Distribution des classes déséquilibrée pour {model_name}'
        })
    
    # Anomalie: entropie trop faible (prédictions trop certaines)
    if predictions['quality_metrics']['distribution']['entropy'] < 0.5:
        anomalies.append({
            'type': 'low_entropy',
            'severity': 'info',
            'value': predictions['quality_metrics']['distribution']['entropy'],
            'threshold': 0.5,
            'description': f'Entropie faible - modèle trop confiant pour {model_name}'
        })
    
    predictions['anomalies'] = anomalies
    
    # Score de qualité global
    quality_score = (
        predictions['quality_metrics']['confidence']['mean'] * 0.4 +
        predictions['quality_metrics']['distribution']['uniformity_score'] * 0.3 +
        min(predictions['quality_metrics']['distribution']['entropy'] / 1.6, 1.0) * 0.3
    )
    
    predictions['overall_quality'] = {
        'quality_score': float(quality_score),
        'quality_grade': 'excellent' if quality_score > 0.8 else
                        'good' if quality_score > 0.6 else
                        'poor' if quality_score > 0.4 else 'critical',
        'requires_attention': quality_score < 0.6 or len(anomalies) > 0
    }
    
    return predictions

# Analyse de la qualité des prédictions
quality_analysis = analyze_prediction_quality('$model_name')

# Sauvegarde
quality_file = f'$ML_METRICS_DIR/predictions/quality_{quality_analysis["model_name"]}_{int(datetime.now().timestamp())}.json'

with open(quality_file, 'w') as f:
    json.dump(quality_analysis, f, indent=2)

print(f"Analyse de qualité terminée: {quality_file}")

# Alertes qualité
if quality_analysis['overall_quality']['requires_attention']:
    print(f"ATTENTION: Qualité des prédictions nécessite surveillance pour {quality_analysis['model_name']}")
    print(f"Score de qualité: {quality_analysis['overall_quality']['quality_score']:.3f} ({quality_analysis['overall_quality']['quality_grade']})")
    
    for anomaly in quality_analysis['anomalies']:
        print(f"  - {anomaly['description']}: {anomaly['value']:.2f}")

EOF
    
    log "ML" "Surveillance qualité prédictions terminée: $model_name"
}

analyze_data_distribution() {
    local model_name="$1"
    log "ML" "Analyse de la distribution des données: $model_name"
    
    python3 << EOF
import json
import numpy as np
import pandas as pd
from datetime import datetime
import random

def analyze_input_data_distribution(model_name):
    """Analyse de la distribution des données d'entrée"""
    
    # Simulation des caractéristiques des données d'entrée
    num_features = random.randint(10, 50)
    num_samples = random.randint(1000, 10000)
    
    analysis = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'num_features': num_features,
            'num_samples': num_samples,
            'timeframe': '24h'
        },
        'feature_analysis': {},
        'data_quality': {}
    }
    
    # Analyse par feature
    feature_stats = []
    for i in range(min(5, num_features)):  # Analyser les 5 premières features
        feature_name = f'feature_{i}'
        
        # Génération de données simulées pour cette feature
        if random.random() > 0.8:  # 20% de chance d'avoir des données problématiques
            # Feature avec problèmes
            data = np.concatenate([
                np.random.normal(0.5, 0.2, int(num_samples * 0.8)),
                np.random.normal(2.0, 0.5, int(num_samples * 0.2))  # Outliers
            ])
        else:
            # Feature normale
            data = np.random.normal(0.5, 0.2, num_samples)
        
        feature_stat = {
            'name': feature_name,
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'missing_count': random.randint(0, int(num_samples * 0.05)),
            'outlier_count': int(np.sum(np.abs(data - np.mean(data)) > 3 * np.std(data))),
            'skewness': float(pd.Series(data).skew()),
            'kurtosis': float(pd.Series(data).kurtosis())
        }
        
        feature_stats.append(feature_stat)
    
    analysis['feature_analysis']['features'] = feature_stats
    
    # Métriques de qualité globales
    total_missing = sum(f['missing_count'] for f in feature_stats)
    total_outliers = sum(f['outlier_count'] for f in feature_stats)
    
    analysis['data_quality'] = {
        'missing_data_percent': round(total_missing / (num_samples * len(feature_stats)) * 100, 2),
        'outlier_percent': round(total_outliers / (num_samples * len(feature_stats)) * 100, 2),
        'avg_skewness': round(np.mean([f['skewness'] for f in feature_stats]), 3),
        'avg_kurtosis': round(np.mean([f['kurtosis'] for f in feature_stats]), 3)
    }
    
    # Détection de problèmes de qualité
    quality_issues = []
    
    if analysis['data_quality']['missing_data_percent'] > 5:
        quality_issues.append({
            'type': 'high_missing_data',
            'severity': 'warning',
            'value': analysis['data_quality']['missing_data_percent'],
            'threshold': 5,
            'description': f'Taux élevé de données manquantes: {analysis["data_quality"]["missing_data_percent"]}%'
        })
    
    if analysis['data_quality']['outlier_percent'] > 10:
        quality_issues.append({
            'type': 'high_outliers',
            'severity': 'warning',
            'value': analysis['data_quality']['outlier_percent'],
            'threshold': 10,
            'description': f'Taux élevé d\'outliers: {analysis["data_quality"]["outlier_percent"]}%'
        })
    
    if abs(analysis['data_quality']['avg_skewness']) > 2:
        quality_issues.append({
            'type': 'high_skewness',
            'severity': 'info',
            'value': abs(analysis['data_quality']['avg_skewness']),
            'threshold': 2,
            'description': f'Distribution très asymétrique: skewness = {analysis["data_quality"]["avg_skewness"]:.2f}'
        })
    
    analysis['quality_issues'] = quality_issues
    
    # Score de qualité des données
    quality_score = max(0, 1 - (
        analysis['data_quality']['missing_data_percent'] / 100 * 2 +
        analysis['data_quality']['outlier_percent'] / 100 * 1.5 +
        min(abs(analysis['data_quality']['avg_skewness']) / 5, 0.2)
    ))
    
    analysis['overall_data_quality'] = {
        'quality_score': round(quality_score, 3),
        'quality_grade': 'excellent' if quality_score > 0.9 else
                        'good' if quality_score > 0.7 else
                        'poor' if quality_score > 0.5 else 'critical',
        'requires_cleaning': quality_score < 0.7
    }
    
    return analysis

# Analyse de distribution
distribution_analysis = analyze_input_data_distribution('$model_name')

# Sauvegarde
distribution_file = f'$ML_METRICS_DIR/performance/data_distribution_{distribution_analysis["model_name"]}_{int(datetime.now().timestamp())}.json'

with open(distribution_file, 'w') as f:
    json.dump(distribution_analysis, f, indent=2)

print(f"Analyse de distribution terminée: {distribution_file}")

# Alertes qualité données
if distribution_analysis['overall_data_quality']['requires_cleaning']:
    print(f"ATTENTION: Qualité des données nécessite nettoyage pour {distribution_analysis['model_name']}")
    print(f"Score qualité: {distribution_analysis['overall_data_quality']['quality_score']:.3f}")
    
    for issue in distribution_analysis['quality_issues']:
        print(f"  - {issue['description']}")

EOF
    
    log "ML" "Analyse distribution données terminée: $model_name"
}

monitor_model_resources() {
    local model_name="$1"
    log "ML" "Surveillance des ressources du modèle: $model_name"
    
    # Collecte des métriques de ressources
    local timestamp=$(date +%s)
    local resources_file="$ML_METRICS_DIR/performance/resources_${model_name}_${timestamp}.json"
    
    # Simulation des métriques de ressources
    cat > "$resources_file" << EOF
{
    "model_name": "$model_name",
    "timestamp": $timestamp,
    "resource_usage": {
        "cpu": {
            "usage_percent": $(shuf -i 20-80 -n 1),
            "cores_allocated": $(shuf -i 2-8 -n 1),
            "cpu_time_seconds": $(shuf -i 3600-86400 -n 1)
        },
        "memory": {
            "usage_mb": $(shuf -i 512-4096 -n 1),
            "allocated_mb": $(shuf -i 1024-8192 -n 1),
            "peak_usage_mb": $(shuf -i 1024-6144 -n 1)
        },
        "gpu": {
            "usage_percent": $(shuf -i 0-95 -n 1),
            "memory_usage_mb": $(shuf -i 0-11000 -n 1),
            "temperature_celsius": $(shuf -i 45-85 -n 1)
        },
        "disk": {
            "model_size_mb": $(shuf -i 100-2048 -n 1),
            "cache_size_mb": $(shuf -i 50-1024 -n 1),
            "io_operations": $(shuf -i 1000-50000 -n 1)
        },
        "network": {
            "requests_per_second": $(shuf -i 10-1000 -n 1),
            "bandwidth_mbps": $(shuf -i 1-100 -n 1),
            "latency_ms": $(shuf -i 10-200 -n 1)
        }
    },
    "efficiency_metrics": {
        "predictions_per_cpu_hour": $(shuf -i 10000-100000 -n 1),
        "energy_efficiency_score": $(echo "scale=2; 0.$(shuf -i 60-95 -n 1)" | bc),
        "cost_per_prediction": $(echo "scale=4; 0.$(shuf -i 1-50 -n 1)" | bc)
    }
}
EOF
    
    log "ML" "Métriques de ressources collectées: $resources_file"
}

# ================================================================
# GÉNÉRATION DE RAPPORTS ET ALERTES
# ================================================================

generate_monitoring_report() {
    log "INFO" "Génération du rapport de surveillance ML"
    
    python3 << EOF
import json
import glob
import os
from datetime import datetime, timedelta
import numpy as np

def aggregate_monitoring_data():
    """Agrégation de toutes les données de surveillance"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'report_period': '24h',
        'models_monitored': [],
        'global_alerts': [],
        'summary': {}
    }
    
    # Collecte des données par modèle
    metrics_dir = '$ML_METRICS_DIR'
    
    # Recherche des fichiers récents (dernières 24h)
    cutoff_time = datetime.now() - timedelta(hours=24)
    cutoff_timestamp = int(cutoff_time.timestamp())
    
    models_data = {}
    
    # Collecte des métriques de performance
    for metrics_file in glob.glob(f'{metrics_dir}/models/metrics_*.json'):
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Vérifier si le fichier est récent
            file_timestamp = int(data.get('timestamp', '0').split('.')[0]) if '.' in data.get('timestamp', '0') else int(data.get('timestamp', '0'))
            if file_timestamp < cutoff_timestamp:
                continue
                
            model_name = data['model_name']
            if model_name not in models_data:
                models_data[model_name] = {'metrics': [], 'drift': [], 'quality': [], 'resources': []}
            
            models_data[model_name]['metrics'].append(data)
        except Exception as e:
            print(f"Erreur lecture fichier métriques {metrics_file}: {e}")
    
    # Collecte des analyses de dérive
    for drift_file in glob.glob(f'{metrics_dir}/drift/drift_analysis_*.json'):
        try:
            with open(drift_file, 'r') as f:
                data = json.load(f)
            
            model_name = data['model_name']
            if model_name in models_data:
                models_data[model_name]['drift'].append(data)
        except Exception as e:
            print(f"Erreur lecture fichier dérive {drift_file}: {e}")
    
    # Collecte des analyses de qualité
    for quality_file in glob.glob(f'{metrics_dir}/predictions/quality_*.json'):
        try:
            with open(quality_file, 'r') as f:
                data = json.load(f)
            
            model_name = data['model_name']
            if model_name in models_data:
                models_data[model_name]['quality'].append(data)
        except Exception as e:
            print(f"Erreur lecture fichier qualité {quality_file}: {e}")
    
    # Génération du rapport par modèle
    for model_name, data in models_data.items():
        model_report = generate_model_report(model_name, data)
        report['models_monitored'].append(model_report)
        
        # Collecte des alertes globales
        if model_report.get('critical_alerts'):
            report['global_alerts'].extend(model_report['critical_alerts'])
    
    # Statistiques globales
    report['summary'] = {
        'total_models': len(models_data),
        'models_with_issues': len([m for m in report['models_monitored'] if m.get('requires_attention', False)]),
        'total_alerts': len(report['global_alerts']),
        'overall_health': calculate_overall_health(report['models_monitored'])
    }
    
    return report

def generate_model_report(model_name, model_data):
    """Génération du rapport pour un modèle spécifique"""
    
    model_report = {
        'model_name': model_name,
        'performance': {},
        'drift_status': {},
        'quality_status': {},
        'resource_usage': {},
        'alerts': [],
        'recommendations': [],
        'requires_attention': False
    }
    
    # Analyse des performances
    if model_data['metrics']:
        latest_metrics = model_data['metrics'][-1]
        perf = latest_metrics['performance']
        
        model_report['performance'] = {
            'accuracy': perf['accuracy'],
            'trend': calculate_trend([m['performance']['accuracy'] for m in model_data['metrics']]),
            'status': 'good' if perf['accuracy'] > 0.8 else 'warning' if perf['accuracy'] > 0.7 else 'critical'
        }
        
        # Alertes de performance
        if latest_metrics.get('performance_issues'):
            for issue in latest_metrics['performance_issues']:
                model_report['alerts'].append(issue)
                if issue['severity'] == 'critical':
                    model_report['requires_attention'] = True
    
    # Analyse de dérive
    if model_data['drift']:
        latest_drift = model_data['drift'][-1]
        
        model_report['drift_status'] = {
            'drift_detected': latest_drift['overall_drift']['drift_detected'],
            'confidence': latest_drift['overall_drift']['confidence_level'],
            'recommendation': latest_drift['overall_drift']['recommendation']
        }
        
        if latest_drift['overall_drift']['drift_detected']:
            model_report['alerts'].append({
                'type': 'model_drift',
                'severity': 'warning',
                'description': f'Dérive détectée pour {model_name}',
                'confidence': latest_drift['overall_drift']['confidence_level']
            })
            model_report['requires_attention'] = True
    
    # Analyse de qualité
    if model_data['quality']:
        latest_quality = model_data['quality'][-1]
        
        model_report['quality_status'] = {
            'quality_score': latest_quality['overall_quality']['quality_score'],
            'quality_grade': latest_quality['overall_quality']['quality_grade'],
            'requires_attention': latest_quality['overall_quality']['requires_attention']
        }
        
        if latest_quality['overall_quality']['requires_attention']:
            model_report['requires_attention'] = True
    
    # Génération de recommandations
    model_report['recommendations'] = generate_recommendations(model_report)
    
    return model_report

def calculate_trend(values):
    """Calcul de la tendance (positive, negative, stable)"""
    if len(values) < 2:
        return 'stable'
    
    # Calcul de la pente de régression linéaire simple
    n = len(values)
    x = list(range(n))
    
    slope = (n * sum(x[i] * values[i] for i in range(n)) - sum(x) * sum(values)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
    
    if slope > 0.01:
        return 'improving'
    elif slope < -0.01:
        return 'degrading'
    else:
        return 'stable'

def generate_recommendations(model_report):
    """Génération de recommandations basées sur l'analyse"""
    recommendations = []
    
    # Recommandations basées sur les performances
    if model_report['performance'].get('status') == 'critical':
        recommendations.append({
            'priority': 'high',
            'action': 'immediate_retraining',
            'description': 'Performances critiques - Re-entraînement immédiat requis'
        })
    elif model_report['performance'].get('trend') == 'degrading':
        recommendations.append({
            'priority': 'medium',
            'action': 'schedule_retraining',
            'description': 'Performances en baisse - Planifier un re-entraînement'
        })
    
    # Recommandations basées sur la dérive
    if model_report['drift_status'].get('drift_detected'):
        recommendations.append({
            'priority': 'high',
            'action': 'investigate_drift',
            'description': 'Dérive détectée - Analyser les changements de données'
        })
    
    # Recommandations basées sur la qualité
    if model_report['quality_status'].get('quality_grade') in ['poor', 'critical']:
        recommendations.append({
            'priority': 'medium',
            'action': 'improve_data_quality',
            'description': 'Qualité des prédictions faible - Améliorer la qualité des données'
        })
    
    return recommendations

def calculate_overall_health(models_reports):
    """Calcul de la santé globale du système ML"""
    if not models_reports:
        return 'unknown'
    
    critical_count = len([m for m in models_reports if any(a.get('severity') == 'critical' for a in m.get('alerts', []))])
    warning_count = len([m for m in models_reports if m.get('requires_attention', False)])
    
    total_models = len(models_reports)
    
    if critical_count > 0:
        return 'critical'
    elif warning_count > total_models * 0.5:
        return 'degraded'
    elif warning_count > 0:
        return 'warning'
    else:
        return 'healthy'

# Génération du rapport
monitoring_report = aggregate_monitoring_data()

# Sauvegarde du rapport
report_file = f'$ML_METRICS_DIR/reports/monitoring_report_{int(datetime.now().timestamp())}.json'
os.makedirs(os.path.dirname(report_file), exist_ok=True)

with open(report_file, 'w') as f:
    json.dump(monitoring_report, f, indent=2)

print(f"Rapport de surveillance généré: {report_file}")

# Affichage du résumé
print(f"\n=== RÉSUMÉ DE SURVEILLANCE ML ===")
print(f"Modèles surveillés: {monitoring_report['summary']['total_models']}")
print(f"Modèles avec problèmes: {monitoring_report['summary']['models_with_issues']}")
print(f"Alertes totales: {monitoring_report['summary']['total_alerts']}")
print(f"Santé globale: {monitoring_report['summary']['overall_health']}")

# Alertes critiques
critical_alerts = [a for a in monitoring_report['global_alerts'] if a.get('severity') == 'critical']
if critical_alerts:
    print(f"\n🚨 ALERTES CRITIQUES:")
    for alert in critical_alerts:
        print(f"  - {alert.get('description', 'Alerte critique')}")

EOF
    
    log "INFO" "Rapport de surveillance ML généré"
}

# ================================================================
# ACTIONS AUTOMATISÉES
# ================================================================

trigger_model_retraining() {
    local model_name="$1"
    local reason="$2"
    
    log "ML" "Déclenchement du re-entraînement pour: $model_name (Raison: $reason)"
    
    # Dans un vrai système, déclencher le pipeline MLOps
    if [ "${DRY_RUN:-false}" = "true" ]; then
        log "INFO" "[DRY-RUN] Re-entraînement simulé pour $model_name"
        return 0
    fi
    
    # Simulation du déclenchement
    python3 << EOF
import json
from datetime import datetime
import subprocess

def trigger_retraining_pipeline(model_name, reason):
    """Déclenchement du pipeline de re-entraînement"""
    
    retraining_config = {
        'model_name': model_name,
        'trigger_reason': reason,
        'timestamp': datetime.now().isoformat(),
        'pipeline_config': {
            'data_version': 'latest',
            'hyperparameter_tuning': True,
            'validation_split': 0.2,
            'early_stopping': True,
            'model_comparison': True
        },
        'notifications': {
            'slack_channel': '#ml-ops',
            'email_recipients': ['ml-team@company.com'],
            'pagerduty_integration': True
        }
    }
    
    # Sauvegarde de la configuration de re-entraînement
    config_file = f'$ML_METRICS_DIR/retraining/config_{model_name}_{int(datetime.now().timestamp())}.json'
    
    with open(config_file, 'w') as f:
        json.dump(retraining_config, f, indent=2)
    
    print(f"Configuration de re-entraînement sauvegardée: {config_file}")
    
    # Dans un vrai système:
    # - Déclencher Kubeflow Pipelines
    # - Appeler MLflow pour le tracking
    # - Notifier les équipes
    
    return config_file

config_file = trigger_retraining_pipeline('$model_name', '$reason')
print(f"Pipeline de re-entraînement configuré pour $model_name")

EOF
    
    log "ML" "Pipeline de re-entraînement configuré pour: $model_name"
}

# ================================================================
# FONCTION PRINCIPALE
# ================================================================

main() {
    log "INFO" "=== Surveillance ML Ultra-Avancée Spotify AI Agent ==="
    log "INFO" "Version: 1.0.0 | Auteur: Fahed Mlaiel"
    log "INFO" "Équipe: Lead Dev + Architecte IA, ML Engineer, DevOps"
    
    # Initialisation
    init_ml_monitoring
    
    local action="${1:-monitor}"
    local model_name="${2:-all}"
    
    case "$action" in
        "monitor")
            log "INFO" "Mode: Surveillance complète des modèles ML"
            monitor_production_models
            ;;
        "drift")
            log "INFO" "Mode: Détection de dérive pour $model_name"
            if [ "$model_name" = "all" ]; then
                for model in $(discover_active_models); do
                    detect_model_drift "$model"
                done
            else
                detect_model_drift "$model_name"
            fi
            ;;
        "performance")
            log "INFO" "Mode: Analyse des performances pour $model_name"
            if [ "$model_name" = "all" ]; then
                for model in $(discover_active_models); do
                    collect_model_performance_metrics "$model"
                done
            else
                collect_model_performance_metrics "$model_name"
            fi
            ;;
        "quality")
            log "INFO" "Mode: Surveillance qualité prédictions pour $model_name"
            if [ "$model_name" = "all" ]; then
                for model in $(discover_active_models); do
                    monitor_prediction_quality "$model"
                done
            else
                monitor_prediction_quality "$model_name"
            fi
            ;;
        "report")
            log "INFO" "Mode: Génération de rapport de surveillance"
            generate_monitoring_report
            ;;
        "retrain")
            log "INFO" "Mode: Déclenchement re-entraînement pour $model_name"
            local reason="${3:-manual_trigger}"
            trigger_model_retraining "$model_name" "$reason"
            ;;
        *)
            echo "Usage: $0 {monitor|drift|performance|quality|report|retrain} [model_name] [reason]"
            echo ""
            echo "Actions disponibles:"
            echo "  monitor      - Surveillance complète de tous les modèles"
            echo "  drift        - Détection de dérive des modèles"
            echo "  performance  - Analyse des performances des modèles"
            echo "  quality      - Surveillance de la qualité des prédictions"
            echo "  report       - Génération du rapport de surveillance"
            echo "  retrain      - Déclenchement du re-entraînement d'un modèle"
            echo ""
            echo "Paramètres:"
            echo "  model_name   - Nom du modèle (ou 'all' pour tous)"
            echo "  reason       - Raison du re-entraînement (pour action retrain)"
            echo ""
            echo "Variables d'environnement:"
            echo "  DRIFT_THRESHOLD=0.1        - Seuil de détection de dérive"
            echo "  PERFORMANCE_THRESHOLD=0.85 - Seuil minimal de performance"
            echo "  DRY_RUN=true              - Mode simulation"
            exit 1
            ;;
    esac
    
    log "INFO" "Surveillance ML terminée avec succès"
}

# Gestion des signaux
trap 'log "INFO" "Arrêt de la surveillance ML"; exit 0' SIGTERM SIGINT

# Exécution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
