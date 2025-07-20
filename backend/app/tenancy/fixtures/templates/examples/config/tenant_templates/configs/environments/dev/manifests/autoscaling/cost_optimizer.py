"""
Optimiseur de coûts intelligent pour l'infrastructure cloud
Analyse des patterns d'usage et optimisation financière automatique
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class CostOptimizationStrategy(Enum):
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    SPOT_INSTANCES = "spot_instances"
    RESERVED_INSTANCES = "reserved_instances"

class InstanceType(Enum):
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    SAVINGS_PLAN = "savings_plan"

@dataclass
class CostMetrics:
    """Métriques de coût pour une période donnée"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    compute_cost: float
    storage_cost: float
    network_cost: float
    management_cost: float
    currency: str = "USD"

@dataclass
class ResourceCostProfile:
    """Profil de coût d'une ressource"""
    resource_id: str
    resource_type: str
    tenant_id: str
    instance_type: InstanceType
    hourly_cost: float
    monthly_estimate: float
    usage_pattern: Dict[str, float]  # heures -> utilisation
    efficiency_score: float
    cost_per_request: Optional[float] = None
    cost_per_gb_processed: Optional[float] = None

@dataclass
class CostOptimizationRecommendation:
    """Recommandation d'optimisation des coûts"""
    resource_id: str
    current_cost: float
    optimized_cost: float
    savings_amount: float
    savings_percentage: float
    strategy: CostOptimizationStrategy
    confidence_score: float
    implementation_complexity: str  # low, medium, high
    risk_level: str  # low, medium, high
    description: str
    action_required: str

class CostOptimizer:
    """Optimiseur de coûts intelligent"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
        # Configuration
        self.analysis_window_days = 30
        self.prediction_window_days = 7
        self.min_savings_threshold = 5.0  # % minimum pour recommander
        self.spot_instance_savings = 0.70  # 70% d'économies avec les instances spot
        self.reserved_instance_savings = 0.40  # 40% d'économies avec les instances réservées
        
        # État du système
        self.cost_history: List[CostMetrics] = []
        self.resource_profiles: Dict[str, ResourceCostProfile] = {}
        self.optimization_recommendations: List[CostOptimizationRecommendation] = []
        self.savings_achieved: float = 0.0
        
        # Cache pour les calculs coûteux
        self._usage_patterns_cache: Dict[str, Dict[str, float]] = {}
        self._cost_predictions_cache: Dict[str, float] = {}
    
    async def analyze_cost_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns de coût et d'utilisation"""
        
        logger.info("Starting comprehensive cost pattern analysis")
        
        # Collecte des données de coût
        await self._collect_cost_data()
        
        # Analyse des patterns d'utilisation
        usage_patterns = await self._analyze_usage_patterns()
        
        # Identification des opportunités d'optimisation
        optimization_opportunities = await self._identify_optimization_opportunities()
        
        # Calcul des prédictions de coût
        cost_predictions = await self._calculate_cost_predictions()
        
        # Analyse des tendances
        cost_trends = self._analyze_cost_trends()
        
        analysis_result = {
            'analysis_period': {
                'start': datetime.utcnow() - timedelta(days=self.analysis_window_days),
                'end': datetime.utcnow(),
                'duration_days': self.analysis_window_days
            },
            'current_costs': self._get_current_cost_summary(),
            'usage_patterns': usage_patterns,
            'optimization_opportunities': optimization_opportunities,
            'cost_predictions': cost_predictions,
            'cost_trends': cost_trends,
            'potential_savings': self._calculate_total_potential_savings()
        }
        
        logger.info(f"Cost analysis completed. Potential savings: ${analysis_result['potential_savings']:.2f}")
        
        return analysis_result
    
    async def _collect_cost_data(self):
        """Collecte les données de coût depuis diverses sources"""
        
        # Simulation de données de coût
        # TODO: Intégration réelle avec AWS Cost Explorer, Azure Cost Management, etc.
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.analysis_window_days)
        
        # Génération de données simulées
        self.cost_history.clear()
        for i in range(self.analysis_window_days):
            date = start_date + timedelta(days=i)
            
            # Simulation de variation des coûts avec pattern hebdomadaire
            base_cost = 1000.0
            weekly_pattern = math.sin(2 * math.pi * i / 7) * 0.2 + 1.0
            random_variation = np.random.normal(1.0, 0.1)
            daily_cost = base_cost * weekly_pattern * random_variation
            
            self.cost_history.append(CostMetrics(
                period_start=date,
                period_end=date + timedelta(days=1),
                total_cost=daily_cost,
                compute_cost=daily_cost * 0.6,
                storage_cost=daily_cost * 0.2,
                network_cost=daily_cost * 0.1,
                management_cost=daily_cost * 0.1
            ))
        
        # Génération de profils de ressources
        self.resource_profiles.clear()
        
        services = ["api-service", "ml-service", "audio-processor", "cache-service", "database"]
        tenants = ["enterprise_001", "premium_001", "basic_001"]
        
        for tenant in tenants:
            for service in services:
                resource_id = f"{tenant}:{service}"
                
                # Coûts variables selon le service
                if service == "ml-service":
                    hourly_cost = np.random.uniform(2.5, 4.0)
                    instance_type = InstanceType.ON_DEMAND
                elif service == "database":
                    hourly_cost = np.random.uniform(1.8, 3.2)
                    instance_type = InstanceType.RESERVED
                elif service == "cache-service":
                    hourly_cost = np.random.uniform(0.8, 1.5)
                    instance_type = InstanceType.ON_DEMAND
                else:
                    hourly_cost = np.random.uniform(0.5, 2.0)
                    instance_type = InstanceType.ON_DEMAND
                
                # Pattern d'utilisation (24h)
                usage_pattern = {}
                for hour in range(24):
                    if service == "api-service":
                        # Pattern de trafic typique avec pics aux heures ouvrables
                        if 9 <= hour <= 17:
                            usage = np.random.uniform(0.7, 0.95)
                        elif 6 <= hour <= 9 or 17 <= hour <= 22:
                            usage = np.random.uniform(0.4, 0.7)
                        else:
                            usage = np.random.uniform(0.1, 0.3)
                    elif service == "ml-service":
                        # Utilisation plus constante pour les tâches ML
                        usage = np.random.uniform(0.6, 0.9)
                    else:
                        # Pattern standard
                        usage = np.random.uniform(0.3, 0.8)
                    
                    usage_pattern[str(hour)] = usage
                
                efficiency_score = np.mean(list(usage_pattern.values()))
                
                self.resource_profiles[resource_id] = ResourceCostProfile(
                    resource_id=resource_id,
                    resource_type=service,
                    tenant_id=tenant,
                    instance_type=instance_type,
                    hourly_cost=hourly_cost,
                    monthly_estimate=hourly_cost * 24 * 30,
                    usage_pattern=usage_pattern,
                    efficiency_score=efficiency_score
                )
    
    async def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns d'utilisation des ressources"""
        
        patterns = {
            'peak_hours': [],
            'low_usage_hours': [],
            'weekly_patterns': {},
            'resource_efficiency': {},
            'underutilized_resources': [],
            'overprovisioned_resources': []
        }
        
        # Analyse des heures de pic et creuses
        hourly_usage = defaultdict(list)
        
        for resource_id, profile in self.resource_profiles.items():
            for hour_str, usage in profile.usage_pattern.items():
                hourly_usage[int(hour_str)].append(usage)
        
        # Calcul des moyennes par heure
        avg_hourly_usage = {}
        for hour, usages in hourly_usage.items():
            avg_hourly_usage[hour] = np.mean(usages)
        
        # Identification des heures de pic (> 80% utilisation moyenne)
        patterns['peak_hours'] = [hour for hour, usage in avg_hourly_usage.items() if usage > 0.8]
        
        # Identification des heures creuses (< 40% utilisation moyenne)
        patterns['low_usage_hours'] = [hour for hour, usage in avg_hourly_usage.items() if usage < 0.4]
        
        # Analyse de l'efficacité des ressources
        for resource_id, profile in self.resource_profiles.items():
            patterns['resource_efficiency'][resource_id] = {
                'efficiency_score': profile.efficiency_score,
                'peak_usage': max(profile.usage_pattern.values()),
                'min_usage': min(profile.usage_pattern.values()),
                'usage_variance': np.var(list(profile.usage_pattern.values()))
            }
            
            # Identification des ressources sous-utilisées
            if profile.efficiency_score < 0.3:
                patterns['underutilized_resources'].append({
                    'resource_id': resource_id,
                    'efficiency_score': profile.efficiency_score,
                    'monthly_waste': profile.monthly_estimate * (1 - profile.efficiency_score)
                })
            
            # Identification des ressources sur-provisionnées
            peak_usage = max(profile.usage_pattern.values())
            if peak_usage < 0.6:
                patterns['overprovisioned_resources'].append({
                    'resource_id': resource_id,
                    'peak_usage': peak_usage,
                    'potential_downsizing': 1 - peak_usage
                })
        
        return patterns
    
    async def _identify_optimization_opportunities(self) -> List[CostOptimizationRecommendation]:
        """Identifie les opportunités d'optimisation des coûts"""
        
        recommendations = []
        
        for resource_id, profile in self.resource_profiles.items():
            
            # Recommandation 1: Migration vers instances spot
            if (profile.instance_type == InstanceType.ON_DEMAND and 
                profile.resource_type in ["ml-service", "audio-processor"]):
                
                current_cost = profile.monthly_estimate
                optimized_cost = current_cost * (1 - self.spot_instance_savings)
                savings = current_cost - optimized_cost
                
                if savings / current_cost * 100 >= self.min_savings_threshold:
                    recommendations.append(CostOptimizationRecommendation(
                        resource_id=resource_id,
                        current_cost=current_cost,
                        optimized_cost=optimized_cost,
                        savings_amount=savings,
                        savings_percentage=savings / current_cost * 100,
                        strategy=CostOptimizationStrategy.SPOT_INSTANCES,
                        confidence_score=0.8,
                        implementation_complexity="medium",
                        risk_level="medium",
                        description=f"Migration vers instances spot pour {resource_id}",
                        action_required="Configurer la tolérance aux interruptions et migrer vers spot instances"
                    ))
            
            # Recommandation 2: Instances réservées pour usage stable
            if (profile.instance_type == InstanceType.ON_DEMAND and 
                profile.efficiency_score > 0.7 and
                np.var(list(profile.usage_pattern.values())) < 0.1):
                
                current_cost = profile.monthly_estimate
                optimized_cost = current_cost * (1 - self.reserved_instance_savings)
                savings = current_cost - optimized_cost
                
                if savings / current_cost * 100 >= self.min_savings_threshold:
                    recommendations.append(CostOptimizationRecommendation(
                        resource_id=resource_id,
                        current_cost=current_cost,
                        optimized_cost=optimized_cost,
                        savings_amount=savings,
                        savings_percentage=savings / current_cost * 100,
                        strategy=CostOptimizationStrategy.RESERVED_INSTANCES,
                        confidence_score=0.9,
                        implementation_complexity="low",
                        risk_level="low",
                        description=f"Achat d'instances réservées pour {resource_id}",
                        action_required="Acheter des instances réservées 1 ou 3 ans"
                    ))
            
            # Recommandation 3: Redimensionnement pour ressources sous-utilisées
            if profile.efficiency_score < 0.4:
                current_cost = profile.monthly_estimate
                right_sizing_factor = max(profile.efficiency_score + 0.2, 0.5)  # Garde une marge
                optimized_cost = current_cost * right_sizing_factor
                savings = current_cost - optimized_cost
                
                if savings / current_cost * 100 >= self.min_savings_threshold:
                    recommendations.append(CostOptimizationRecommendation(
                        resource_id=resource_id,
                        current_cost=current_cost,
                        optimized_cost=optimized_cost,
                        savings_amount=savings,
                        savings_percentage=savings / current_cost * 100,
                        strategy=CostOptimizationStrategy.MODERATE,
                        confidence_score=0.7,
                        implementation_complexity="medium",
                        risk_level="low",
                        description=f"Redimensionnement de {resource_id} (sous-utilisé)",
                        action_required="Réduire les ressources allouées selon l'utilisation réelle"
                    ))
            
            # Recommandation 4: Planification d'arrêt pour heures creuses
            usage_values = list(profile.usage_pattern.values())
            min_usage = min(usage_values)
            low_usage_hours = sum(1 for usage in usage_values if usage < 0.2)
            
            if low_usage_hours >= 8:  # Plus de 8h de faible utilisation
                shutdown_savings = profile.hourly_cost * low_usage_hours * 30  # par mois
                current_cost = profile.monthly_estimate
                
                if shutdown_savings / current_cost * 100 >= self.min_savings_threshold:
                    recommendations.append(CostOptimizationRecommendation(
                        resource_id=resource_id,
                        current_cost=current_cost,
                        optimized_cost=current_cost - shutdown_savings,
                        savings_amount=shutdown_savings,
                        savings_percentage=shutdown_savings / current_cost * 100,
                        strategy=CostOptimizationStrategy.AGGRESSIVE,
                        confidence_score=0.6,
                        implementation_complexity="high",
                        risk_level="medium",
                        description=f"Arrêt programmé pendant les heures creuses pour {resource_id}",
                        action_required="Implémenter un système d'arrêt/démarrage automatique"
                    ))
        
        # Tri par économies potentielles
        recommendations.sort(key=lambda r: r.savings_amount, reverse=True)
        
        self.optimization_recommendations = recommendations
        return recommendations
    
    async def _calculate_cost_predictions(self) -> Dict[str, Any]:
        """Calcule les prédictions de coût"""
        
        if len(self.cost_history) < 7:
            return {'error': 'Insufficient historical data'}
        
        # Extraction des coûts quotidiens
        daily_costs = [metrics.total_cost for metrics in self.cost_history]
        
        # Calcul de la tendance linéaire
        x = np.arange(len(daily_costs))
        coefficients = np.polyfit(x, daily_costs, 1)
        trend_slope = coefficients[0]
        
        # Prédiction pour les prochains jours
        last_cost = daily_costs[-1]
        predictions = []
        
        for i in range(1, self.prediction_window_days + 1):
            predicted_cost = last_cost + (trend_slope * i)
            
            # Ajout de la variabilité hebdomadaire
            day_of_week = (len(daily_costs) + i) % 7
            weekly_factor = 1.0 + 0.1 * math.sin(2 * math.pi * day_of_week / 7)
            predicted_cost *= weekly_factor
            
            predictions.append({
                'day': i,
                'predicted_cost': max(0, predicted_cost),
                'confidence': max(0.3, 0.9 - (i * 0.1))  # Confiance décroissante
            })
        
        # Calcul du coût mensuel projeté
        avg_daily_cost = np.mean(daily_costs[-7:])  # Moyenne des 7 derniers jours
        monthly_projection = avg_daily_cost * 30
        
        # Calcul de la variance pour l'estimation d'incertitude
        cost_variance = np.var(daily_costs[-14:]) if len(daily_costs) >= 14 else 0
        uncertainty_range = math.sqrt(cost_variance) * 1.96  # 95% intervalle de confiance
        
        return {
            'daily_predictions': predictions,
            'monthly_projection': {
                'base_estimate': monthly_projection,
                'lower_bound': max(0, monthly_projection - uncertainty_range * 30),
                'upper_bound': monthly_projection + uncertainty_range * 30,
                'confidence_level': 0.8
            },
            'trend_analysis': {
                'slope': trend_slope,
                'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'significance': abs(trend_slope) > (np.std(daily_costs) * 0.1)
            }
        }
    
    def _analyze_cost_trends(self) -> Dict[str, Any]:
        """Analyse les tendances de coût"""
        
        if len(self.cost_history) < 14:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Séparation en deux périodes
        mid_point = len(self.cost_history) // 2
        first_half = self.cost_history[:mid_point]
        second_half = self.cost_history[mid_point:]
        
        # Calcul des moyennes
        first_half_avg = np.mean([m.total_cost for m in first_half])
        second_half_avg = np.mean([m.total_cost for m in second_half])
        
        # Calcul du changement
        cost_change = second_half_avg - first_half_avg
        cost_change_percent = (cost_change / first_half_avg * 100) if first_half_avg > 0 else 0
        
        # Analyse par catégorie
        categories = ['compute_cost', 'storage_cost', 'network_cost', 'management_cost']
        category_trends = {}
        
        for category in categories:
            first_avg = np.mean([getattr(m, category) for m in first_half])
            second_avg = np.mean([getattr(m, category) for m in second_half])
            change = second_avg - first_avg
            change_percent = (change / first_avg * 100) if first_avg > 0 else 0
            
            category_trends[category] = {
                'change_amount': change,
                'change_percentage': change_percent,
                'trend': 'increasing' if change > 0 else 'decreasing'
            }
        
        # Détection d'anomalies (coûts > 2 écarts-types de la moyenne)
        all_costs = [m.total_cost for m in self.cost_history]
        mean_cost = np.mean(all_costs)
        std_cost = np.std(all_costs)
        threshold = mean_cost + 2 * std_cost
        
        anomalies = []
        for i, metrics in enumerate(self.cost_history):
            if metrics.total_cost > threshold:
                anomalies.append({
                    'date': metrics.period_start,
                    'cost': metrics.total_cost,
                    'deviation': metrics.total_cost - mean_cost
                })
        
        return {
            'overall_trend': {
                'change_amount': cost_change,
                'change_percentage': cost_change_percent,
                'direction': 'increasing' if cost_change > 0 else 'decreasing'
            },
            'category_trends': category_trends,
            'volatility': {
                'coefficient_of_variation': (std_cost / mean_cost) if mean_cost > 0 else 0,
                'stability': 'high' if (std_cost / mean_cost) < 0.1 else 'medium' if (std_cost / mean_cost) < 0.3 else 'low'
            },
            'anomalies': anomalies,
            'analysis_period': {
                'start': self.cost_history[0].period_start,
                'end': self.cost_history[-1].period_end
            }
        }
    
    def _get_current_cost_summary(self) -> Dict[str, float]:
        """Résumé des coûts actuels"""
        
        if not self.cost_history:
            return {}
        
        latest_metrics = self.cost_history[-1]
        
        return {
            'daily_total': latest_metrics.total_cost,
            'monthly_estimate': latest_metrics.total_cost * 30,
            'breakdown': {
                'compute': latest_metrics.compute_cost,
                'storage': latest_metrics.storage_cost,
                'network': latest_metrics.network_cost,
                'management': latest_metrics.management_cost
            }
        }
    
    def _calculate_total_potential_savings(self) -> float:
        """Calcule les économies totales potentielles"""
        
        return sum(rec.savings_amount for rec in self.optimization_recommendations)
    
    async def implement_cost_optimizations(self, selected_recommendations: List[str] = None) -> Dict[str, Any]:
        """Implémente les optimisations de coût sélectionnées"""
        
        if selected_recommendations is None:
            # Sélectionne automatiquement les recommandations à faible risque
            selected_recommendations = [
                rec.resource_id for rec in self.optimization_recommendations
                if rec.risk_level == "low" and rec.confidence_score >= 0.8
            ]
        
        implementation_results = {
            'implemented': 0,
            'failed': 0,
            'total_savings': 0.0,
            'details': []
        }
        
        for resource_id in selected_recommendations:
            # Trouve la recommandation correspondante
            recommendation = next(
                (rec for rec in self.optimization_recommendations if rec.resource_id == resource_id),
                None
            )
            
            if not recommendation:
                continue
            
            try:
                # Simulation d'implémentation
                success = await self._implement_single_optimization(recommendation)
                
                if success:
                    implementation_results['implemented'] += 1
                    implementation_results['total_savings'] += recommendation.savings_amount
                    self.savings_achieved += recommendation.savings_amount
                    
                    implementation_results['details'].append({
                        'resource_id': resource_id,
                        'status': 'success',
                        'savings': recommendation.savings_amount,
                        'strategy': recommendation.strategy.value
                    })
                else:
                    implementation_results['failed'] += 1
                    implementation_results['details'].append({
                        'resource_id': resource_id,
                        'status': 'failed',
                        'error': 'Implementation failed'
                    })
                    
            except Exception as e:
                logger.error(f"Failed to implement optimization for {resource_id}: {e}")
                implementation_results['failed'] += 1
                implementation_results['details'].append({
                    'resource_id': resource_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        logger.info(f"Implemented {implementation_results['implemented']} optimizations, "
                   f"saved ${implementation_results['total_savings']:.2f}")
        
        return implementation_results
    
    async def _implement_single_optimization(self, recommendation: CostOptimizationRecommendation) -> bool:
        """Implémente une optimisation unique"""
        
        # Simulation d'implémentation
        await asyncio.sleep(0.1)
        
        # TODO: Implémentation réelle selon la stratégie
        if recommendation.strategy == CostOptimizationStrategy.SPOT_INSTANCES:
            # Migrer vers instances spot
            pass
        elif recommendation.strategy == CostOptimizationStrategy.RESERVED_INSTANCES:
            # Acheter des instances réservées
            pass
        elif recommendation.strategy == CostOptimizationStrategy.MODERATE:
            # Redimensionner les ressources
            pass
        elif recommendation.strategy == CostOptimizationStrategy.AGGRESSIVE:
            # Implémenter arrêt/démarrage automatique
            pass
        
        # Simulation: 90% de succès
        return np.random.random() < 0.9
    
    def generate_cost_optimization_report(self) -> Dict[str, Any]:
        """Génère un rapport complet d'optimisation des coûts"""
        
        # Calcul des statistiques générales
        total_monthly_cost = sum(profile.monthly_estimate for profile in self.resource_profiles.values())
        total_potential_savings = self._calculate_total_potential_savings()
        savings_percentage = (total_potential_savings / total_monthly_cost * 100) if total_monthly_cost > 0 else 0
        
        # Analyse par stratégie
        strategy_breakdown = defaultdict(lambda: {'count': 0, 'savings': 0.0})
        for rec in self.optimization_recommendations:
            strategy_breakdown[rec.strategy.value]['count'] += 1
            strategy_breakdown[rec.strategy.value]['savings'] += rec.savings_amount
        
        # Top recommandations
        top_recommendations = sorted(
            self.optimization_recommendations,
            key=lambda r: r.savings_amount,
            reverse=True
        )[:5]
        
        # Analyse de risque
        risk_analysis = {
            'low_risk_savings': sum(rec.savings_amount for rec in self.optimization_recommendations if rec.risk_level == "low"),
            'medium_risk_savings': sum(rec.savings_amount for rec in self.optimization_recommendations if rec.risk_level == "medium"),
            'high_risk_savings': sum(rec.savings_amount for rec in self.optimization_recommendations if rec.risk_level == "high")
        }
        
        return {
            'executive_summary': {
                'total_monthly_cost': total_monthly_cost,
                'potential_monthly_savings': total_potential_savings,
                'savings_percentage': savings_percentage,
                'recommendations_count': len(self.optimization_recommendations),
                'already_saved': self.savings_achieved
            },
            'strategy_breakdown': dict(strategy_breakdown),
            'top_recommendations': [
                {
                    'resource_id': rec.resource_id,
                    'savings_amount': rec.savings_amount,
                    'savings_percentage': rec.savings_percentage,
                    'strategy': rec.strategy.value,
                    'risk_level': rec.risk_level
                }
                for rec in top_recommendations
            ],
            'risk_analysis': risk_analysis,
            'implementation_priority': {
                'immediate': [rec.resource_id for rec in self.optimization_recommendations 
                            if rec.risk_level == "low" and rec.savings_percentage >= 20],
                'short_term': [rec.resource_id for rec in self.optimization_recommendations 
                             if rec.risk_level == "medium" and rec.savings_percentage >= 15],
                'long_term': [rec.resource_id for rec in self.optimization_recommendations 
                            if rec.risk_level == "high" or rec.implementation_complexity == "high"]
            },
            'generated_at': datetime.utcnow(),
            'analysis_confidence': np.mean([rec.confidence_score for rec in self.optimization_recommendations]) if self.optimization_recommendations else 0.0
        }
