"""
Advanced Model Drift Detection Configuration
===========================================

Configuration ultra-avancée pour la détection de dérive des modèles ML
avec algorithmes sophistiqués et seuils adaptatifs par tenant.
"""

import yaml
from typing import Dict, Any, List
from datetime import datetime, timedelta

# ================================================================
# CONFIGURATION GLOBALE DE DÉTECTION DE DÉRIVE
# ================================================================

DRIFT_DETECTION_CONFIG = {
    "version": "1.0.0",
    "last_updated": datetime.utcnow().isoformat(),
    
    # Algorithmes de détection activés
    "enabled_detectors": [
        "kolmogorov_smirnov",
        "population_stability_index", 
        "jensen_shannon_divergence",
        "wasserstein_distance",
        "adversarial_validation",
        "feature_correlation_drift",
        "prediction_distribution_drift"
    ],
    
    # Seuils de détection par défaut
    "default_thresholds": {
        "statistical_significance": 0.05,
        "psi_warning": 0.1,
        "psi_critical": 0.2,
        "js_divergence": 0.1,
        "wasserstein_threshold": 0.15,
        "correlation_drift": 0.1,
        "prediction_drift": 0.1,
        "data_quality_minimum": 0.95
    },
    
    # Configuration des fenêtres temporelles
    "time_windows": {
        "reference_window_days": 30,
        "detection_window_hours": 24,
        "trend_analysis_hours": 168,  # 1 semaine
        "baseline_update_days": 7
    },
    
    # Configuration des alertes
    "alerting": {
        "enabled": True,
        "alert_levels": {
            "low": {"threshold": 0.1, "frequency": "hourly"},
            "medium": {"threshold": 0.2, "frequency": "every_30min"},
            "high": {"threshold": 0.3, "frequency": "every_15min"},
            "critical": {"threshold": 0.5, "frequency": "immediate"}
        },
        "escalation_policy": {
            "l1_timeout_minutes": 15,
            "l2_timeout_minutes": 30,
            "l3_timeout_minutes": 60
        }
    },
    
    # Auto-remédiation
    "auto_remediation": {
        "enabled": True,
        "actions": {
            "retrain_model": {"threshold": 0.3, "auto_approve": False},
            "fallback_to_baseline": {"threshold": 0.5, "auto_approve": True},
            "scale_up_resources": {"threshold": 0.2, "auto_approve": True},
            "alert_data_team": {"threshold": 0.1, "auto_approve": True}
        }
    }
}

# ================================================================
# CONFIGURATIONS SPÉCIFIQUES PAR TYPE DE MODÈLE
# ================================================================

MODEL_TYPE_CONFIGS = {
    "recommendation_engine": {
        "name": "Moteur de Recommandation",
        "description": "Détection de dérive pour les modèles de recommandation musicale",
        "specific_thresholds": {
            "user_behavior_drift": 0.15,
            "content_drift": 0.2,
            "interaction_pattern_drift": 0.1
        },
        "key_features": [
            "user_listening_history",
            "track_features",
            "user_demographics",
            "session_context",
            "temporal_patterns"
        ],
        "monitoring_metrics": [
            "recommendation_precision",
            "recommendation_recall",
            "user_engagement_rate",
            "click_through_rate",
            "diversity_score"
        ],
        "drift_indicators": [
            "seasonal_pattern_shift",
            "genre_preference_evolution",
            "user_discovery_behavior"
        ]
    },
    
    "sentiment_analysis": {
        "name": "Analyse de Sentiment",
        "description": "Détection de dérive pour l'analyse de sentiment des lyrics",
        "specific_thresholds": {
            "vocabulary_drift": 0.1,
            "sentiment_distribution_drift": 0.15,
            "language_pattern_drift": 0.12
        },
        "key_features": [
            "word_embeddings",
            "sentiment_scores",
            "linguistic_features",
            "contextual_embeddings"
        ],
        "monitoring_metrics": [
            "sentiment_accuracy",
            "emotion_classification_f1",
            "confidence_scores",
            "prediction_consistency"
        ]
    },
    
    "music_generation": {
        "name": "Génération Musicale",
        "description": "Détection de dérive pour les modèles génératifs musicaux",
        "specific_thresholds": {
            "audio_feature_drift": 0.2,
            "style_consistency_drift": 0.15,
            "quality_metrics_drift": 0.1
        },
        "key_features": [
            "spectral_features",
            "rhythmic_patterns",
            "harmonic_progressions",
            "timbral_characteristics"
        ],
        "monitoring_metrics": [
            "audio_quality_score",
            "style_consistency",
            "user_acceptance_rate",
            "generation_diversity"
        ]
    },
    
    "user_segmentation": {
        "name": "Segmentation Utilisateur", 
        "description": "Détection de dérive pour la segmentation et clustering d'utilisateurs",
        "specific_thresholds": {
            "cluster_stability": 0.1,
            "feature_importance_drift": 0.15,
            "segment_distribution_drift": 0.2
        },
        "key_features": [
            "behavioral_metrics",
            "demographic_features",
            "engagement_patterns",
            "preference_vectors"
        ],
        "monitoring_metrics": [
            "silhouette_score",
            "cluster_purity",
            "segment_stability",
            "business_value_score"
        ]
    }
}

# ================================================================
# CONFIGURATIONS SPÉCIFIQUES PAR TENANT
# ================================================================

TENANT_SPECIFIC_CONFIGS = {
    "enterprise_tier": {
        "enhanced_monitoring": True,
        "custom_thresholds": {
            "statistical_significance": 0.01,  # Plus strict
            "psi_warning": 0.05,
            "psi_critical": 0.1
        },
        "real_time_monitoring": True,
        "advanced_analytics": True,
        "dedicated_resources": True,
        "sla_requirements": {
            "detection_latency_max_minutes": 5,
            "alert_response_time_max_minutes": 2,
            "false_positive_rate_max": 0.05
        }
    },
    
    "premium_tier": {
        "enhanced_monitoring": True,
        "custom_thresholds": {
            "statistical_significance": 0.03,
            "psi_warning": 0.08,
            "psi_critical": 0.15
        },
        "real_time_monitoring": False,
        "advanced_analytics": True,
        "sla_requirements": {
            "detection_latency_max_minutes": 15,
            "alert_response_time_max_minutes": 10,
            "false_positive_rate_max": 0.1
        }
    },
    
    "standard_tier": {
        "enhanced_monitoring": False,
        "custom_thresholds": "default",
        "real_time_monitoring": False,
        "advanced_analytics": False,
        "sla_requirements": {
            "detection_latency_max_minutes": 60,
            "alert_response_time_max_minutes": 30,
            "false_positive_rate_max": 0.15
        }
    }
}

# ================================================================
# CONFIGURATION DES DONNÉES ET FEATURES
# ================================================================

DATA_MONITORING_CONFIG = {
    "data_quality_checks": {
        "missing_values": {
            "threshold": 0.05,
            "critical_threshold": 0.2,
            "action": "alert_data_team"
        },
        "data_types": {
            "enforce_schema": True,
            "allow_type_evolution": False,
            "action": "reject_batch"
        },
        "value_ranges": {
            "check_bounds": True,
            "outlier_detection": True,
            "z_score_threshold": 3.0
        },
        "freshness": {
            "max_age_hours": 24,
            "staleness_threshold_hours": 48,
            "action": "alert_and_fallback"
        }
    },
    
    "feature_monitoring": {
        "importance_tracking": True,
        "correlation_monitoring": True,
        "distribution_tracking": True,
        "interaction_analysis": True,
        "feature_evolution": {
            "new_features": "auto_detect",
            "deprecated_features": "gradual_removal",
            "feature_versioning": True
        }
    },
    
    "data_sources": {
        "streaming_data": {
            "kafka_topics": ["user_interactions", "track_plays", "user_feedback"],
            "real_time_processing": True,
            "batch_processing": True
        },
        "batch_data": {
            "data_warehouse": "postgresql",
            "update_frequency": "daily",
            "historical_depth_days": 365
        },
        "external_data": {
            "music_databases": ["spotify_api", "musicbrainz"],
            "social_media": ["twitter_api", "instagram_api"],
            "weather_data": "openweather_api"
        }
    }
}

# ================================================================
# CONFIGURATION PROMETHEUS POUR ML MONITORING
# ================================================================

PROMETHEUS_ML_METRICS = {
    "ml_model_drift_score": {
        "type": "gauge",
        "help": "Score de dérive du modèle ML (0-1, plus élevé = plus de dérive)",
        "labels": ["tenant_id", "model_id", "model_type", "drift_algorithm"]
    },
    
    "ml_data_quality_score": {
        "type": "gauge",
        "help": "Score de qualité des données (0-1, plus élevé = meilleure qualité)",
        "labels": ["tenant_id", "model_id", "data_source", "quality_dimension"]
    },
    
    "ml_prediction_accuracy": {
        "type": "gauge",
        "help": "Précision/performance actuelle du modèle",
        "labels": ["tenant_id", "model_id", "metric_type", "evaluation_set"]
    },
    
    "ml_feature_importance_score": {
        "type": "gauge",
        "help": "Score d'importance des features",
        "labels": ["tenant_id", "model_id", "feature_name", "importance_method"]
    },
    
    "ml_inference_latency_seconds": {
        "type": "histogram",
        "help": "Latence des inférences ML en secondes",
        "labels": ["tenant_id", "model_id", "model_type"],
        "buckets": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    },
    
    "ml_prediction_confidence": {
        "type": "histogram",
        "help": "Distribution de la confiance des prédictions",
        "labels": ["tenant_id", "model_id"],
        "buckets": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    },
    
    "ml_training_duration_seconds": {
        "type": "gauge",
        "help": "Durée du dernier entraînement en secondes",
        "labels": ["tenant_id", "model_id", "training_type"]
    },
    
    "ml_model_version": {
        "type": "gauge",
        "help": "Version actuelle du modèle en production",
        "labels": ["tenant_id", "model_id", "version", "deployment_timestamp"]
    },
    
    "ml_data_drift_alerts_total": {
        "type": "counter",
        "help": "Nombre total d'alertes de dérive de données",
        "labels": ["tenant_id", "model_id", "alert_level", "drift_type"]
    },
    
    "ml_auto_remediation_actions_total": {
        "type": "counter",
        "help": "Nombre total d'actions d'auto-remédiation déclenchées",
        "labels": ["tenant_id", "model_id", "action_type", "success"]
    }
}

# ================================================================
# CONFIGURATION DES ALERTES GRAFANA
# ================================================================

GRAFANA_ALERT_RULES = {
    "ml_critical_drift": {
        "condition": "ml_model_drift_score > 0.5",
        "frequency": "1m",
        "notifications": ["slack-ml-alerts", "pagerduty-ml"],
        "message": "Dérive critique détectée pour le modèle {{ $labels.model_id }} (tenant: {{ $labels.tenant_id }})"
    },
    
    "ml_data_quality_degradation": {
        "condition": "ml_data_quality_score < 0.9",
        "frequency": "5m", 
        "notifications": ["slack-data-quality"],
        "message": "Dégradation de la qualité des données pour {{ $labels.model_id }}"
    },
    
    "ml_performance_drop": {
        "condition": "ml_prediction_accuracy < 0.8",
        "frequency": "10m",
        "notifications": ["email-ml-team"],
        "message": "Performance du modèle {{ $labels.model_id }} en dessous du seuil acceptable"
    },
    
    "ml_inference_latency_high": {
        "condition": "histogram_quantile(0.95, ml_inference_latency_seconds_bucket) > 1.0",
        "frequency": "2m",
        "notifications": ["slack-performance"],
        "message": "Latence d'inférence élevée pour {{ $labels.model_id }}"
    }
}

def generate_tenant_drift_config(tenant_id: str, tier: str = "standard") -> Dict[str, Any]:
    """
    Génère une configuration de détection de dérive personnalisée pour un tenant
    
    Args:
        tenant_id: Identifiant du tenant
        tier: Niveau de service (standard, premium, enterprise)
        
    Returns:
        Configuration personnalisée
    """
    base_config = DRIFT_DETECTION_CONFIG.copy()
    
    # Application des configurations spécifiques au tier
    if tier in TENANT_SPECIFIC_CONFIGS:
        tier_config = TENANT_SPECIFIC_CONFIGS[tier]
        
        # Fusion des seuils personnalisés
        if "custom_thresholds" in tier_config and tier_config["custom_thresholds"] != "default":
            base_config["default_thresholds"].update(tier_config["custom_thresholds"])
        
        # Ajout des configurations spécifiques
        base_config["tenant_specific"] = tier_config
    
    # Ajout des métadonnées du tenant
    base_config["tenant_metadata"] = {
        "tenant_id": tenant_id,
        "tier": tier,
        "created_at": datetime.utcnow().isoformat(),
        "config_version": "1.0.0"
    }
    
    return base_config

def export_prometheus_config() -> str:
    """Exporte la configuration Prometheus au format YAML"""
    return yaml.dump(PROMETHEUS_ML_METRICS, default_flow_style=False)

def export_grafana_alerts() -> str:
    """Exporte les règles d'alertes Grafana au format YAML"""
    return yaml.dump(GRAFANA_ALERT_RULES, default_flow_style=False)
