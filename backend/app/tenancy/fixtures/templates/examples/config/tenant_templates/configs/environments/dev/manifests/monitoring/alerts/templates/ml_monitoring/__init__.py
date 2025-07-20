"""
ML Model Drift Detection Ultra-Avancé
=====================================

Système de détection de dérive des modèles ML avec algorithmes sophistiqués,
alertes prédictives et auto-remédiation pour l'architecture multi-tenant.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import warnings

logger = logging.getLogger(__name__)

class MLModelDriftDetector:
    """Détecteur ultra-avancé de dérive des modèles ML"""
    
    def __init__(self, tenant_id: str, model_id: str):
        self.tenant_id = tenant_id
        self.model_id = model_id
        self.baseline_data = None
        self.drift_thresholds = {
            "statistical": 0.05,
            "population_stability": 0.1,
            "data_quality": 0.95,
            "feature_drift": 0.15,
            "prediction_drift": 0.1
        }
        self.drift_history = []
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialise les détecteurs de dérive sophistiqués"""
        logger.info(f"Initialisation détecteurs de dérive pour {self.tenant_id}:{self.model_id}")
        
        self.detectors = {
            "kolmogorov_smirnov": self._ks_test,
            "population_stability_index": self._psi_calculation,
            "jensen_shannon_divergence": self._js_divergence,
            "wasserstein_distance": self._wasserstein_distance,
            "mutual_information": self._mutual_info_drift,
            "adversarial_validation": self._adversarial_validation
        }
    
    def detect_drift(self, current_data: pd.DataFrame, 
                    reference_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Détection complète de dérive avec multiple algorithmes
        
        Args:
            current_data: Données actuelles à analyser
            reference_data: Données de référence (optionnel)
            
        Returns:
            Rapport complet de détection de dérive
        """
        if reference_data is None:
            reference_data = self.baseline_data
        
        if reference_data is None:
            logger.warning("Pas de données de référence disponibles")
            return {"status": "no_baseline", "drift_detected": False}
        
        drift_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": self.tenant_id,
            "model_id": self.model_id,
            "drift_detected": False,
            "overall_drift_score": 0.0,
            "detailed_analysis": {},
            "recommendations": [],
            "alerts": []
        }
        
        # Analyse statistique de base
        drift_report["detailed_analysis"]["statistical"] = self._statistical_drift_analysis(
            current_data, reference_data
        )
        
        # Analyse de stabilité de population
        drift_report["detailed_analysis"]["population_stability"] = self._population_stability_analysis(
            current_data, reference_data
        )
        
        # Analyse de qualité des données
        drift_report["detailed_analysis"]["data_quality"] = self._data_quality_analysis(
            current_data, reference_data
        )
        
        # Analyse de dérive des features
        drift_report["detailed_analysis"]["feature_drift"] = self._feature_drift_analysis(
            current_data, reference_data
        )
        
        # Analyse de dérive des prédictions
        if "predictions" in current_data.columns and "predictions" in reference_data.columns:
            drift_report["detailed_analysis"]["prediction_drift"] = self._prediction_drift_analysis(
                current_data, reference_data
            )
        
        # Calcul du score global de dérive
        drift_report["overall_drift_score"] = self._calculate_overall_drift_score(
            drift_report["detailed_analysis"]
        )
        
        # Détermination si dérive détectée
        drift_report["drift_detected"] = drift_report["overall_drift_score"] > self.drift_thresholds["statistical"]
        
        # Génération des recommandations
        drift_report["recommendations"] = self._generate_recommendations(drift_report)
        
        # Génération des alertes
        drift_report["alerts"] = self._generate_alerts(drift_report)
        
        # Sauvegarde dans l'historique
        self.drift_history.append(drift_report)
        
        return drift_report
    
    def _statistical_drift_analysis(self, current_data: pd.DataFrame, 
                                   reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse statistique sophistiquée de la dérive"""
        analysis = {
            "ks_tests": {},
            "chi2_tests": {},
            "mann_whitney_tests": {},
            "overall_p_value": 1.0
        }
        
        p_values = []
        
        for column in current_data.select_dtypes(include=[np.number]).columns:
            if column in reference_data.columns:
                # Test de Kolmogorov-Smirnov
                ks_stat, ks_pvalue = stats.ks_2samp(
                    reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                analysis["ks_tests"][column] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_pvalue),
                    "drift_detected": ks_pvalue < self.drift_thresholds["statistical"]
                }
                p_values.append(ks_pvalue)
                
                # Test de Mann-Whitney U
                mw_stat, mw_pvalue = stats.mannwhitneyu(
                    reference_data[column].dropna(),
                    current_data[column].dropna(),
                    alternative='two-sided'
                )
                analysis["mann_whitney_tests"][column] = {
                    "statistic": float(mw_stat),
                    "p_value": float(mw_pvalue),
                    "drift_detected": mw_pvalue < self.drift_thresholds["statistical"]
                }
        
        # Correction de Bonferroni pour tests multiples
        if p_values:
            analysis["overall_p_value"] = min(1.0, min(p_values) * len(p_values))
        
        return analysis
    
    def _population_stability_analysis(self, current_data: pd.DataFrame,
                                     reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse de stabilité de population (PSI)"""
        analysis = {
            "psi_scores": {},
            "overall_psi": 0.0,
            "stability_status": "stable"
        }
        
        psi_scores = []
        
        for column in current_data.select_dtypes(include=[np.number]).columns:
            if column in reference_data.columns:
                psi_score = self._calculate_psi(
                    reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                analysis["psi_scores"][column] = {
                    "score": float(psi_score),
                    "status": self._interpret_psi(psi_score)
                }
                psi_scores.append(psi_score)
        
        if psi_scores:
            analysis["overall_psi"] = np.mean(psi_scores)
            analysis["stability_status"] = self._interpret_psi(analysis["overall_psi"])
        
        return analysis
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      buckets: int = 10) -> float:
        """Calcul du Population Stability Index (PSI)"""
        try:
            # Création des bins basés sur les données de référence
            bins = np.linspace(reference.min(), reference.max(), buckets + 1)
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Distribution de référence
            ref_counts, _ = np.histogram(reference, bins=bins)
            ref_pct = ref_counts / len(reference)
            ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)  # Éviter division par zéro
            
            # Distribution actuelle
            cur_counts, _ = np.histogram(current, bins=bins)
            cur_pct = cur_counts / len(current)
            cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
            
            # Calcul PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return psi
            
        except Exception as e:
            logger.error(f"Erreur calcul PSI: {e}")
            return 0.0
    
    def _interpret_psi(self, psi_score: float) -> str:
        """Interprétation du score PSI"""
        if psi_score < 0.1:
            return "stable"
        elif psi_score < 0.2:
            return "moderate_drift"
        else:
            return "significant_drift"
    
    def _data_quality_analysis(self, current_data: pd.DataFrame,
                              reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse de qualité des données"""
        analysis = {
            "missing_values": {},
            "data_types": {},
            "outliers": {},
            "quality_score": 1.0
        }
        
        quality_issues = 0
        total_checks = 0
        
        for column in current_data.columns:
            if column in reference_data.columns:
                # Analyse des valeurs manquantes
                ref_missing = reference_data[column].isna().mean()
                cur_missing = current_data[column].isna().mean()
                missing_change = abs(cur_missing - ref_missing)
                
                analysis["missing_values"][column] = {
                    "reference_pct": float(ref_missing),
                    "current_pct": float(cur_missing),
                    "change": float(missing_change),
                    "significant_change": missing_change > 0.05
                }
                
                if missing_change > 0.05:
                    quality_issues += 1
                total_checks += 1
                
                # Analyse des types de données
                ref_dtype = str(reference_data[column].dtype)
                cur_dtype = str(current_data[column].dtype)
                type_changed = ref_dtype != cur_dtype
                
                analysis["data_types"][column] = {
                    "reference_type": ref_dtype,
                    "current_type": cur_dtype,
                    "type_changed": type_changed
                }
                
                if type_changed:
                    quality_issues += 1
                total_checks += 1
        
        if total_checks > 0:
            analysis["quality_score"] = 1.0 - (quality_issues / total_checks)
        
        return analysis
    
    def _feature_drift_analysis(self, current_data: pd.DataFrame,
                               reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse de dérive des features"""
        analysis = {
            "feature_importance_changes": {},
            "correlation_changes": {},
            "distribution_changes": {}
        }
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = list(set(numeric_cols) & set(reference_data.columns))
        
        if len(common_cols) > 1:
            # Analyse des corrélations
            ref_corr = reference_data[common_cols].corr()
            cur_corr = current_data[common_cols].corr()
            corr_diff = np.abs(ref_corr - cur_corr).mean().mean()
            
            analysis["correlation_changes"] = {
                "mean_correlation_change": float(corr_diff),
                "significant_change": corr_diff > 0.1
            }
        
        return analysis
    
    def _prediction_drift_analysis(self, current_data: pd.DataFrame,
                                  reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse de dérive des prédictions"""
        analysis = {
            "prediction_distribution": {},
            "accuracy_metrics": {}
        }
        
        if "predictions" in current_data.columns and "predictions" in reference_data.columns:
            # Distribution des prédictions
            ks_stat, ks_pvalue = stats.ks_2samp(
                reference_data["predictions"].dropna(),
                current_data["predictions"].dropna()
            )
            
            analysis["prediction_distribution"] = {
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_pvalue),
                "drift_detected": ks_pvalue < self.drift_thresholds["prediction_drift"]
            }
        
        return analysis
    
    def _calculate_overall_drift_score(self, detailed_analysis: Dict[str, Any]) -> float:
        """Calcul du score global de dérive"""
        scores = []
        
        # Score statistique
        if "statistical" in detailed_analysis and "overall_p_value" in detailed_analysis["statistical"]:
            stat_score = 1.0 - detailed_analysis["statistical"]["overall_p_value"]
            scores.append(stat_score)
        
        # Score PSI
        if "population_stability" in detailed_analysis and "overall_psi" in detailed_analysis["population_stability"]:
            psi_score = min(1.0, detailed_analysis["population_stability"]["overall_psi"] / 0.2)
            scores.append(psi_score)
        
        # Score qualité données
        if "data_quality" in detailed_analysis and "quality_score" in detailed_analysis["data_quality"]:
            quality_score = 1.0 - detailed_analysis["data_quality"]["quality_score"]
            scores.append(quality_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, drift_report: Dict[str, Any]) -> List[str]:
        """Génération de recommandations basées sur l'analyse"""
        recommendations = []
        
        if drift_report["drift_detected"]:
            recommendations.append("Dérive détectée - Réentraînement du modèle recommandé")
            
            if drift_report["overall_drift_score"] > 0.5:
                recommendations.append("Dérive significative - Réentraînement urgent requis")
                recommendations.append("Analyse approfondie des causes racines nécessaire")
            
            # Recommandations spécifiques
            if "population_stability" in drift_report["detailed_analysis"]:
                psi = drift_report["detailed_analysis"]["population_stability"]["overall_psi"]
                if psi > 0.2:
                    recommendations.append("Population très instable - Vérifier la qualité des données d'entrée")
            
            if "data_quality" in drift_report["detailed_analysis"]:
                quality = drift_report["detailed_analysis"]["data_quality"]["quality_score"]
                if quality < 0.9:
                    recommendations.append("Problèmes de qualité des données détectés - Nettoyage requis")
        
        return recommendations
    
    def _generate_alerts(self, drift_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Génération d'alertes structurées"""
        alerts = []
        
        if drift_report["drift_detected"]:
            severity = "warning"
            if drift_report["overall_drift_score"] > 0.5:
                severity = "critical"
            elif drift_report["overall_drift_score"] > 0.3:
                severity = "high"
            
            alerts.append({
                "type": "model_drift",
                "severity": severity,
                "message": f"Dérive détectée pour le modèle {self.model_id} (tenant: {self.tenant_id})",
                "score": drift_report["overall_drift_score"],
                "recommendations": drift_report["recommendations"],
                "timestamp": drift_report["timestamp"]
            })
        
        return alerts
    
    def set_baseline(self, baseline_data: pd.DataFrame):
        """Définit les données de référence pour la détection de dérive"""
        self.baseline_data = baseline_data.copy()
        logger.info(f"Baseline définie pour {self.tenant_id}:{self.model_id}")
    
    def get_drift_trend(self, window_hours: int = 24) -> Dict[str, Any]:
        """Analyse de tendance de dérive sur une fenêtre temporelle"""
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        
        recent_history = [
            report for report in self.drift_history
            if datetime.fromisoformat(report["timestamp"]) > cutoff_time
        ]
        
        if not recent_history:
            return {"trend": "insufficient_data", "score_trend": []}
        
        scores = [report["overall_drift_score"] for report in recent_history]
        timestamps = [report["timestamp"] for report in recent_history]
        
        # Calcul de tendance simple
        if len(scores) > 1:
            trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
            if trend_slope > 0.01:
                trend = "increasing"
            elif trend_slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "score_trend": list(zip(timestamps, scores)),
            "average_score": np.mean(scores),
            "max_score": max(scores),
            "drift_frequency": sum(1 for report in recent_history if report["drift_detected"]) / len(recent_history)
        }

# Utilitaires pour monitoring ML
class MLMonitoringUtils:
    """Utilitaires pour le monitoring ML avancé"""
    
    @staticmethod
    def generate_drift_metrics_config() -> Dict[str, Any]:
        """Génère la configuration des métriques de dérive pour Prometheus"""
        return {
            "ml_model_drift_score": {
                "type": "gauge",
                "help": "Score de dérive du modèle ML (0-1)",
                "labels": ["tenant_id", "model_id", "model_type"]
            },
            "ml_data_quality_score": {
                "type": "gauge", 
                "help": "Score de qualité des données (0-1)",
                "labels": ["tenant_id", "model_id", "data_source"]
            },
            "ml_prediction_accuracy": {
                "type": "gauge",
                "help": "Précision des prédictions du modèle",
                "labels": ["tenant_id", "model_id", "metric_type"]
            },
            "ml_feature_importance": {
                "type": "gauge",
                "help": "Importance des features",
                "labels": ["tenant_id", "model_id", "feature_name"]
            }
        }

# Instance globale pour l'exportation
ml_monitoring_utils = MLMonitoringUtils()
