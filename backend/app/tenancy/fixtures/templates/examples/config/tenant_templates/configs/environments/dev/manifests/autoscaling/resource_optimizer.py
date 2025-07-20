"""
Optimiseur de ressources intelligent avec algorithmes génétiques
Optimisation automatique du placement et du dimensionnement des workloads
Développé par l'équipe d'experts dirigée par Fahed Mlaiel
"""

import asyncio
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    COST = "cost"
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    EFFICIENCY = "efficiency"
    SUSTAINABILITY = "sustainability"

@dataclass
class ResourceProfile:
    """Profil de ressources d'un workload"""
    service_name: str
    tenant_id: str
    cpu_request: float
    cpu_limit: float
    memory_request: float  # en GB
    memory_limit: float    # en GB
    storage_request: float # en GB
    network_bandwidth: float # en Mbps
    gpu_requirement: int = 0
    
@dataclass
class NodeProfile:
    """Profil d'un nœud"""
    node_id: str
    cpu_capacity: float
    memory_capacity: float
    storage_capacity: float
    cpu_available: float
    memory_available: float
    storage_available: float
    cost_per_hour: float
    performance_score: float
    reliability_score: float
    location: str = ""
    node_type: str = "compute"  # compute, memory, storage, gpu

@dataclass
class PlacementSolution:
    """Solution de placement de workloads"""
    placements: Dict[str, str]  # workload_id -> node_id
    fitness_score: float
    cost_score: float
    performance_score: float
    efficiency_score: float
    constraint_violations: int

class ResourceOptimizer:
    """Optimiseur de ressources avec algorithmes génétiques"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
        # Configuration de l'algorithme génétique
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
        # État du système
        self.workloads: Dict[str, ResourceProfile] = {}
        self.nodes: Dict[str, NodeProfile] = {}
        self.current_placements: Dict[str, str] = {}
        self.optimization_history: List[PlacementSolution] = []
        
        # Cache pour les calculs coûteux
        self._fitness_cache: Dict[str, float] = {}
        self._constraint_cache: Dict[str, bool] = {}
    
    async def optimize_resource_placement(self, objective: OptimizationObjective = OptimizationObjective.BALANCED) -> PlacementSolution:
        """Lance l'optimisation du placement des ressources"""
        
        logger.info(f"Starting resource optimization with objective: {objective.value}")
        
        # Mise à jour de l'état du système
        await self._update_system_state()
        
        if not self.workloads or not self.nodes:
            logger.warning("No workloads or nodes available for optimization")
            return PlacementSolution({}, 0.0, 0.0, 0.0, 0.0, 0)
        
        # Génération de la population initiale
        population = self._generate_initial_population()
        
        # Évolution génétique
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Évaluation de la fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, objective)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            # Logging du progrès
            if generation % 20 == 0:
                avg_fitness = np.mean(fitness_scores)
                logger.info(f"Generation {generation}: avg_fitness={avg_fitness:.3f}, best_fitness={best_fitness:.3f}")
            
            # Sélection et reproduction
            population = self._evolve_population(population, fitness_scores, objective)
        
        # Création de la solution finale
        final_solution = self._create_solution_from_placement(best_solution, objective)
        
        # Sauvegarde dans l'historique
        self.optimization_history.append(final_solution)
        
        # Nettoyage de l'historique (garde 7 jours)
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        # Note: PlacementSolution devrait avoir un timestamp pour un vrai nettoyage
        
        logger.info(f"Optimization completed. Final fitness: {final_solution.fitness_score:.3f}")
        
        return final_solution
    
    async def _update_system_state(self):
        """Met à jour l'état du système (workloads et nœuds)"""
        
        # TODO: Récupération réelle des workloads depuis Kubernetes
        # Pour l'instant, simulation avec des données d'exemple
        
        # Simulation de workloads
        tenant_ids = ["enterprise_001", "premium_001", "basic_001"]
        services = ["api-service", "ml-service", "audio-processor", "cache-service"]
        
        self.workloads.clear()
        for i, (tenant_id, service) in enumerate([(t, s) for t in tenant_ids for s in services]):
            workload_id = f"{tenant_id}:{service}"
            
            # Profils de ressources variables selon le service
            if service == "ml-service":
                cpu_req, cpu_lim = 2.0, 4.0
                mem_req, mem_lim = 4.0, 8.0
                storage = 50.0
            elif service == "audio-processor":
                cpu_req, cpu_lim = 1.5, 3.0
                mem_req, mem_lim = 2.0, 4.0
                storage = 100.0
            elif service == "cache-service":
                cpu_req, cpu_lim = 0.5, 1.0
                mem_req, mem_lim = 8.0, 16.0
                storage = 20.0
            else:  # api-service
                cpu_req, cpu_lim = 1.0, 2.0
                mem_req, mem_lim = 1.0, 2.0
                storage = 10.0
            
            self.workloads[workload_id] = ResourceProfile(
                service_name=service,
                tenant_id=tenant_id,
                cpu_request=cpu_req,
                cpu_limit=cpu_lim,
                memory_request=mem_req,
                memory_limit=mem_lim,
                storage_request=storage,
                network_bandwidth=1000.0  # 1 Gbps
            )
        
        # Simulation de nœuds
        self.nodes.clear()
        node_types = [
            ("compute", 16.0, 32.0, 500.0, 0.20, 8.5),
            ("memory", 8.0, 128.0, 200.0, 0.35, 7.8),
            ("storage", 4.0, 16.0, 2000.0, 0.15, 9.2),
            ("gpu", 32.0, 64.0, 1000.0, 0.80, 8.8)
        ]
        
        for i in range(10):  # 10 nœuds
            node_type, cpu_cap, mem_cap, stor_cap, cost, perf = random.choice(node_types)
            node_id = f"node-{i:02d}"
            
            # Disponibilité simulée (70-90% de la capacité)
            availability_factor = random.uniform(0.7, 0.9)
            
            self.nodes[node_id] = NodeProfile(
                node_id=node_id,
                cpu_capacity=cpu_cap,
                memory_capacity=mem_cap,
                storage_capacity=stor_cap,
                cpu_available=cpu_cap * availability_factor,
                memory_available=mem_cap * availability_factor,
                storage_available=stor_cap * availability_factor,
                cost_per_hour=cost,
                performance_score=perf,
                reliability_score=random.uniform(7.0, 9.5),
                node_type=node_type
            )
    
    def _generate_initial_population(self) -> List[Dict[str, str]]:
        """Génère la population initiale pour l'algorithme génétique"""
        
        population = []
        workload_ids = list(self.workloads.keys())
        node_ids = list(self.nodes.keys())
        
        # Solution actuelle comme seed
        if self.current_placements:
            population.append(self.current_placements.copy())
        
        # Solutions aléatoires
        for _ in range(self.population_size - 1):
            placement = {}
            for workload_id in workload_ids:
                # Placement aléatoire avec biais vers les nœuds compatibles
                compatible_nodes = self._get_compatible_nodes(workload_id)
                if compatible_nodes:
                    placement[workload_id] = random.choice(compatible_nodes)
                else:
                    placement[workload_id] = random.choice(node_ids)
            population.append(placement)
        
        return population
    
    def _get_compatible_nodes(self, workload_id: str) -> List[str]:
        """Trouve les nœuds compatibles avec un workload"""
        
        workload = self.workloads[workload_id]
        compatible = []
        
        for node_id, node in self.nodes.items():
            # Vérification des ressources
            if (node.cpu_available >= workload.cpu_request and
                node.memory_available >= workload.memory_request and
                node.storage_available >= workload.storage_request):
                
                # Vérification de la compatibilité type
                if workload.service_name == "ml-service" and node.node_type in ["compute", "gpu"]:
                    compatible.append(node_id)
                elif workload.service_name == "cache-service" and node.node_type in ["memory", "compute"]:
                    compatible.append(node_id)
                elif workload.service_name == "audio-processor" and node.node_type in ["compute", "storage"]:
                    compatible.append(node_id)
                elif workload.service_name == "api-service":
                    compatible.append(node_id)  # Peut aller partout
        
        return compatible
    
    def _evaluate_fitness(self, placement: Dict[str, str], objective: OptimizationObjective) -> float:
        """Évalue la fitness d'une solution de placement"""
        
        # Cache pour éviter les recalculs
        placement_key = json.dumps(placement, sort_keys=True)
        if placement_key in self._fitness_cache:
            return self._fitness_cache[placement_key]
        
        # Calcul des scores individuels
        cost_score = self._calculate_cost_score(placement)
        performance_score = self._calculate_performance_score(placement)
        efficiency_score = self._calculate_efficiency_score(placement)
        constraint_penalty = self._calculate_constraint_penalty(placement)
        
        # Agrégation selon l'objectif
        if objective == OptimizationObjective.COST:
            fitness = 0.7 * (1 - cost_score) + 0.2 * efficiency_score + 0.1 * performance_score
        elif objective == OptimizationObjective.PERFORMANCE:
            fitness = 0.7 * performance_score + 0.2 * efficiency_score + 0.1 * (1 - cost_score)
        elif objective == OptimizationObjective.EFFICIENCY:
            fitness = 0.8 * efficiency_score + 0.1 * performance_score + 0.1 * (1 - cost_score)
        else:  # BALANCED
            fitness = 0.4 * (1 - cost_score) + 0.4 * performance_score + 0.2 * efficiency_score
        
        # Application de la pénalité pour les contraintes violées
        fitness *= (1 - constraint_penalty)
        
        # Cache du résultat
        self._fitness_cache[placement_key] = fitness
        
        return fitness
    
    def _calculate_cost_score(self, placement: Dict[str, str]) -> float:
        """Calcule le score de coût (normalisé 0-1)"""
        
        total_cost = 0.0
        max_possible_cost = 0.0
        
        for workload_id, node_id in placement.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                total_cost += node.cost_per_hour
                
                # Coût maximum = nœud le plus cher
                max_cost_node = max(self.nodes.values(), key=lambda n: n.cost_per_hour)
                max_possible_cost += max_cost_node.cost_per_hour
        
        return total_cost / max_possible_cost if max_possible_cost > 0 else 0.0
    
    def _calculate_performance_score(self, placement: Dict[str, str]) -> float:
        """Calcule le score de performance (normalisé 0-1)"""
        
        total_performance = 0.0
        max_possible_performance = 0.0
        
        for workload_id, node_id in placement.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                workload = self.workloads[workload_id]
                
                # Score basé sur l'adéquation du nœud au workload
                performance_factor = node.performance_score / 10.0  # Normalisation
                
                # Bonus pour l'adéquation type
                if workload.service_name == "ml-service" and node.node_type == "gpu":
                    performance_factor *= 1.3
                elif workload.service_name == "cache-service" and node.node_type == "memory":
                    performance_factor *= 1.2
                elif workload.service_name == "audio-processor" and node.node_type == "compute":
                    performance_factor *= 1.1
                
                total_performance += performance_factor
                max_possible_performance += 1.3  # Score max possible
        
        return total_performance / max_possible_performance if max_possible_performance > 0 else 0.0
    
    def _calculate_efficiency_score(self, placement: Dict[str, str]) -> float:
        """Calcule le score d'efficacité (utilisation des ressources)"""
        
        node_utilization = {}
        
        # Calcul de l'utilisation par nœud
        for node_id in self.nodes:
            node_utilization[node_id] = {
                'cpu_used': 0.0,
                'memory_used': 0.0,
                'storage_used': 0.0
            }
        
        for workload_id, node_id in placement.items():
            if node_id in self.nodes and workload_id in self.workloads:
                workload = self.workloads[workload_id]
                node_utilization[node_id]['cpu_used'] += workload.cpu_request
                node_utilization[node_id]['memory_used'] += workload.memory_request
                node_utilization[node_id]['storage_used'] += workload.storage_request
        
        # Calcul de l'efficacité globale
        efficiency_scores = []
        for node_id, utilization in node_utilization.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                cpu_efficiency = utilization['cpu_used'] / node.cpu_capacity if node.cpu_capacity > 0 else 0
                memory_efficiency = utilization['memory_used'] / node.memory_capacity if node.memory_capacity > 0 else 0
                storage_efficiency = utilization['storage_used'] / node.storage_capacity if node.storage_capacity > 0 else 0
                
                # Efficacité globale du nœud (évite la sur-utilisation)
                node_efficiency = (cpu_efficiency + memory_efficiency + storage_efficiency) / 3
                
                # Pénalité pour sous-utilisation et sur-utilisation
                if node_efficiency > 0.9:  # Sur-utilisation
                    node_efficiency *= 0.5
                elif node_efficiency < 0.3:  # Sous-utilisation
                    node_efficiency *= 0.7
                
                efficiency_scores.append(node_efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    def _calculate_constraint_penalty(self, placement: Dict[str, str]) -> float:
        """Calcule la pénalité pour les contraintes violées"""
        
        violations = 0
        total_constraints = 0
        
        # Vérification des contraintes de ressources
        node_usage = {node_id: {'cpu': 0, 'memory': 0, 'storage': 0} for node_id in self.nodes}
        
        for workload_id, node_id in placement.items():
            if workload_id in self.workloads and node_id in self.nodes:
                workload = self.workloads[workload_id]
                node = self.nodes[node_id]
                
                node_usage[node_id]['cpu'] += workload.cpu_request
                node_usage[node_id]['memory'] += workload.memory_request
                node_usage[node_id]['storage'] += workload.storage_request
                
                total_constraints += 3  # 3 types de ressources
        
        # Comptage des violations
        for node_id, usage in node_usage.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                if usage['cpu'] > node.cpu_capacity:
                    violations += 1
                if usage['memory'] > node.memory_capacity:
                    violations += 1
                if usage['storage'] > node.storage_capacity:
                    violations += 1
        
        return violations / total_constraints if total_constraints > 0 else 0.0
    
    def _evolve_population(self, population: List[Dict[str, str]], fitness_scores: List[float],
                          objective: OptimizationObjective) -> List[Dict[str, str]]:
        """Fait évoluer la population via sélection, croisement et mutation"""
        
        # Sélection des élites
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        new_population = [population[i].copy() for i in elite_indices]
        
        # Génération du reste de la population
        while len(new_population) < self.population_size:
            # Sélection par tournoi
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Croisement
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Limitation à la taille de population
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[Dict[str, str]], fitness_scores: List[float],
                            tournament_size: int = 3) -> Dict[str, str]:
        """Sélection par tournoi"""
        
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index].copy()
    
    def _crossover(self, parent1: Dict[str, str], parent2: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Croisement uniforme entre deux parents"""
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        for workload_id in parent1.keys():
            if random.random() < 0.5:
                child1[workload_id] = parent2[workload_id]
                child2[workload_id] = parent1[workload_id]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, str]) -> Dict[str, str]:
        """Mutation d'un individu"""
        
        mutated = individual.copy()
        workload_ids = list(mutated.keys())
        node_ids = list(self.nodes.keys())
        
        # Mutation de quelques gènes aléatoires
        num_mutations = max(1, int(len(workload_ids) * 0.1))
        
        for _ in range(num_mutations):
            workload_id = random.choice(workload_ids)
            
            # Essaie de trouver un nœud compatible, sinon aléatoire
            compatible_nodes = self._get_compatible_nodes(workload_id)
            if compatible_nodes:
                mutated[workload_id] = random.choice(compatible_nodes)
            else:
                mutated[workload_id] = random.choice(node_ids)
        
        return mutated
    
    def _create_solution_from_placement(self, placement: Dict[str, str],
                                      objective: OptimizationObjective) -> PlacementSolution:
        """Crée un objet PlacementSolution à partir d'un placement"""
        
        fitness_score = self._evaluate_fitness(placement, objective)
        cost_score = self._calculate_cost_score(placement)
        performance_score = self._calculate_performance_score(placement)
        efficiency_score = self._calculate_efficiency_score(placement)
        constraint_violations = int(self._calculate_constraint_penalty(placement) * len(placement) * 3)
        
        return PlacementSolution(
            placements=placement.copy(),
            fitness_score=fitness_score,
            cost_score=cost_score,
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            constraint_violations=constraint_violations
        )
    
    async def apply_optimization_solution(self, solution: PlacementSolution) -> Dict[str, Any]:
        """Applique une solution d'optimisation au cluster"""
        
        logger.info("Applying resource optimization solution")
        
        results = {
            'migrations_planned': 0,
            'migrations_successful': 0,
            'migrations_failed': 0,
            'cost_reduction_estimated': 0.0,
            'performance_improvement_estimated': 0.0
        }
        
        # Comparaison avec les placements actuels
        migrations_needed = []
        for workload_id, new_node in solution.placements.items():
            current_node = self.current_placements.get(workload_id)
            if current_node != new_node:
                migrations_needed.append((workload_id, current_node, new_node))
        
        results['migrations_planned'] = len(migrations_needed)
        
        # Application des migrations (simulation)
        for workload_id, old_node, new_node in migrations_needed:
            try:
                # TODO: Implémentation réelle de la migration via Kubernetes
                success = await self._migrate_workload(workload_id, old_node, new_node)
                if success:
                    results['migrations_successful'] += 1
                    self.current_placements[workload_id] = new_node
                else:
                    results['migrations_failed'] += 1
                    
            except Exception as e:
                logger.error(f"Migration failed for {workload_id}: {e}")
                results['migrations_failed'] += 1
        
        # Estimation des améliorations
        if results['migrations_successful'] > 0:
            old_cost = self._calculate_cost_score(self.current_placements)
            new_cost = solution.cost_score
            results['cost_reduction_estimated'] = max(0, (old_cost - new_cost) / old_cost * 100)
            
            results['performance_improvement_estimated'] = solution.performance_score * 100
        
        logger.info(f"Applied {results['migrations_successful']}/{results['migrations_planned']} migrations")
        
        return results
    
    async def _migrate_workload(self, workload_id: str, old_node: str, new_node: str) -> bool:
        """Migre un workload vers un nouveau nœud"""
        
        # Simulation de migration
        await asyncio.sleep(0.1)  # Simule le temps de migration
        
        # TODO: Implémentation réelle:
        # 1. Créer une nouvelle instance sur le nouveau nœud
        # 2. Attendre qu'elle soit prête
        # 3. Rediriger le trafic
        # 4. Supprimer l'ancienne instance
        
        # Simulation: 95% de succès
        return random.random() < 0.95
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Génère des recommandations d'optimisation"""
        
        recommendations = []
        
        if not self.optimization_history:
            return recommendations
        
        latest_solution = self.optimization_history[-1]
        
        # Recommandation basée sur l'efficacité
        if latest_solution.efficiency_score < 0.6:
            recommendations.append({
                'type': 'efficiency',
                'title': 'Améliorer l\'utilisation des ressources',
                'description': 'L\'efficacité actuelle est faible. Considérez une redistribution des workloads.',
                'priority': 'high',
                'estimated_improvement': f"{(0.8 - latest_solution.efficiency_score) * 100:.1f}%"
            })
        
        # Recommandation basée sur les coûts
        if latest_solution.cost_score > 0.7:
            recommendations.append({
                'type': 'cost',
                'title': 'Optimisation des coûts',
                'description': 'Les coûts sont élevés. Migration vers des nœuds moins chers recommandée.',
                'priority': 'medium',
                'estimated_savings': f"{(latest_solution.cost_score - 0.5) * 100:.1f}%"
            })
        
        # Recommandation basée sur les contraintes
        if latest_solution.constraint_violations > 0:
            recommendations.append({
                'type': 'constraints',
                'title': 'Résoudre les violations de contraintes',
                'description': f'{latest_solution.constraint_violations} contraintes violées détectées.',
                'priority': 'critical'
            })
        
        return recommendations
    
    def get_system_efficiency_report(self) -> Dict[str, Any]:
        """Génère un rapport d'efficacité du système"""
        
        if not self.nodes or not self.workloads:
            return {'error': 'Insufficient data'}
        
        # Calcul de l'utilisation actuelle
        node_utilization = {}
        for node_id in self.nodes:
            node_utilization[node_id] = {'cpu': 0, 'memory': 0, 'storage': 0, 'workloads': 0}
        
        for workload_id, node_id in self.current_placements.items():
            if workload_id in self.workloads and node_id in self.nodes:
                workload = self.workloads[workload_id]
                node_utilization[node_id]['cpu'] += workload.cpu_request
                node_utilization[node_id]['memory'] += workload.memory_request
                node_utilization[node_id]['storage'] += workload.storage_request
                node_utilization[node_id]['workloads'] += 1
        
        # Statistiques globales
        total_cpu_capacity = sum(node.cpu_capacity for node in self.nodes.values())
        total_memory_capacity = sum(node.memory_capacity for node in self.nodes.values())
        total_cpu_used = sum(util['cpu'] for util in node_utilization.values())
        total_memory_used = sum(util['memory'] for util in node_utilization.values())
        
        cpu_utilization_percent = (total_cpu_used / total_cpu_capacity * 100) if total_cpu_capacity > 0 else 0
        memory_utilization_percent = (total_memory_used / total_memory_capacity * 100) if total_memory_capacity > 0 else 0
        
        # Nœuds sous-utilisés et sur-utilisés
        underutilized_nodes = []
        overutilized_nodes = []
        
        for node_id, utilization in node_utilization.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                cpu_percent = (utilization['cpu'] / node.cpu_capacity * 100) if node.cpu_capacity > 0 else 0
                memory_percent = (utilization['memory'] / node.memory_capacity * 100) if node.memory_capacity > 0 else 0
                
                avg_utilization = (cpu_percent + memory_percent) / 2
                
                if avg_utilization < 30:
                    underutilized_nodes.append({
                        'node_id': node_id,
                        'utilization_percent': avg_utilization,
                        'workloads': utilization['workloads']
                    })
                elif avg_utilization > 85:
                    overutilized_nodes.append({
                        'node_id': node_id,
                        'utilization_percent': avg_utilization,
                        'workloads': utilization['workloads']
                    })
        
        return {
            'global_statistics': {
                'total_nodes': len(self.nodes),
                'total_workloads': len(self.workloads),
                'cpu_utilization_percent': cpu_utilization_percent,
                'memory_utilization_percent': memory_utilization_percent,
                'average_workloads_per_node': len(self.workloads) / len(self.nodes) if self.nodes else 0
            },
            'efficiency_score': latest_solution.efficiency_score if self.optimization_history else 0.0,
            'underutilized_nodes': underutilized_nodes,
            'overutilized_nodes': overutilized_nodes,
            'optimization_opportunities': len(underutilized_nodes) + len(overutilized_nodes),
            'last_optimization': self.optimization_history[-1].fitness_score if self.optimization_history else None
        }
