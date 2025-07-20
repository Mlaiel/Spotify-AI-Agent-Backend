"""
Détecteurs de Patterns et Comportements Avancés
===============================================

Auteur: Fahed Mlaiel
Rôles: Lead Dev + Architecte IA, Spécialiste Sécurité Backend

Ce module implémente des détecteurs sophistiqués pour l'analyse de patterns,
la détection de comportements anormaux et l'analyse de sécurité en temps réel.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from collections import defaultdict, deque, Counter
import json
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy import signal
from scipy.stats import entropy
import ipaddress
import geoip2.database
import whois

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types de patterns détectables"""
    TEMPORAL = "temporal"          # Patterns temporels
    SEQUENTIAL = "sequential"      # Séquences d'événements
    FREQUENCY = "frequency"        # Patterns de fréquence
    BEHAVIORAL = "behavioral"      # Comportements utilisateur
    NETWORK = "network"           # Patterns réseau
    SECURITY = "security"         # Patterns de sécurité
    CONTENT = "content"           # Patterns de contenu
    GEOGRAPHIC = "geographic"     # Patterns géographiques

class BehaviorType(Enum):
    """Types de comportements"""
    USER_ACTIVITY = "user_activity"
    API_USAGE = "api_usage"
    LOGIN_PATTERN = "login_pattern"
    CONTENT_ACCESS = "content_access"
    SEARCH_BEHAVIOR = "search_behavior"
    PLAYLIST_CREATION = "playlist_creation"
    SOCIAL_INTERACTION = "social_interaction"
    PAYMENT_BEHAVIOR = "payment_behavior"

@dataclass
class PatternResult:
    """Résultat de détection de pattern"""
    pattern_detected: bool
    pattern_type: str
    confidence_score: float
    pattern_description: str
    anomaly_indicators: List[str]
    risk_level: str
    recommended_actions: List[str]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BehaviorProfile:
    """Profil comportemental d'une entité"""
    entity_id: str
    entity_type: str
    behavior_patterns: Dict[str, Any]
    baseline_metrics: Dict[str, float]
    risk_score: float
    last_updated: datetime
    anomaly_history: List[Dict[str, Any]]

class SequenceAnalyzer:
    """Analyseur de séquences d'événements"""
    
    def __init__(self, max_sequence_length: int = 50, min_frequency: int = 3):
        self.max_sequence_length = max_sequence_length
        self.min_frequency = min_frequency
        self.sequence_patterns = defaultdict(int)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.event_frequencies = defaultdict(int)
        
    def learn_sequences(self, sequences: List[List[str]]):
        """Apprend les patterns de séquences normales"""
        for sequence in sequences:
            # Apprentissage des n-grammes
            for n in range(2, min(len(sequence) + 1, self.max_sequence_length + 1)):
                for i in range(len(sequence) - n + 1):
                    pattern = tuple(sequence[i:i+n])
                    self.sequence_patterns[pattern] += 1
            
            # Matrice de transition
            for i in range(len(sequence) - 1):
                self.transition_matrix[sequence[i]][sequence[i+1]] += 1
            
            # Fréquences des événements
            for event in sequence:
                self.event_frequencies[event] += 1
    
    def detect_anomalous_sequence(self, sequence: List[str]) -> PatternResult:
        """Détecte les séquences anormales"""
        anomaly_indicators = []
        total_score = 0.0
        pattern_scores = []
        
        # Vérifier les n-grammes
        for n in range(2, min(len(sequence) + 1, self.max_sequence_length + 1)):
            for i in range(len(sequence) - n + 1):
                pattern = tuple(sequence[i:i+n])
                frequency = self.sequence_patterns.get(pattern, 0)
                
                if frequency == 0:
                    anomaly_indicators.append(f"Séquence inconnue: {' -> '.join(pattern)}")
                    pattern_scores.append(1.0)
                elif frequency < self.min_frequency:
                    anomaly_indicators.append(f"Séquence rare: {' -> '.join(pattern)} (freq: {frequency})")
                    pattern_scores.append(0.8)
                else:
                    pattern_scores.append(0.0)
        
        # Vérifier les transitions
        for i in range(len(sequence) - 1):
            current, next_event = sequence[i], sequence[i+1]
            transition_count = self.transition_matrix[current].get(next_event, 0)
            total_transitions = sum(self.transition_matrix[current].values())
            
            if total_transitions > 0:
                transition_prob = transition_count / total_transitions
                if transition_prob < 0.1:  # Transition rare
                    anomaly_indicators.append(f"Transition rare: {current} -> {next_event}")
                    pattern_scores.append(0.7)
        
        # Score global
        if pattern_scores:
            total_score = np.mean(pattern_scores)
        
        is_anomalous = total_score > 0.3 or len(anomaly_indicators) > 0
        
        return PatternResult(
            pattern_detected=is_anomalous,
            pattern_type=PatternType.SEQUENTIAL.value,
            confidence_score=total_score,
            pattern_description=f"Analyse de séquence de {len(sequence)} événements",
            anomaly_indicators=anomaly_indicators,
            risk_level=self._calculate_risk_level(total_score),
            recommended_actions=self._get_sequence_recommendations(anomaly_indicators),
            context={
                'sequence_length': len(sequence),
                'unique_events': len(set(sequence)),
                'pattern_scores': pattern_scores
            }
        )
    
    def _calculate_risk_level(self, score: float) -> str:
        """Calcule le niveau de risque"""
        if score > 0.8:
            return "critical"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_sequence_recommendations(self, indicators: List[str]) -> List[str]:
        """Génère des recommandations basées sur les indicateurs"""
        recommendations = []
        
        if any("inconnue" in ind for ind in indicators):
            recommendations.append("Investiguer les nouvelles séquences d'événements")
        
        if any("rare" in ind for ind in indicators):
            recommendations.append("Analyser les patterns rares pour détection d'attaque")
        
        if any("transition" in ind for ind in indicators):
            recommendations.append("Vérifier la logique métier des transitions")
        
        return recommendations or ["Continuer la surveillance"]

class BehaviorAnalyzer:
    """Analyseur de comportements utilisateur avancé"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=2)
        self.behavior_profiles = {}
        self.anomaly_thresholds = {
            'activity_volume': 3.0,      # Z-score
            'temporal_deviation': 2.5,
            'geographic_anomaly': 4.0,
            'content_diversity': 2.0,
            'interaction_pattern': 2.5
        }
        
        # Modèles pour clustering
        self.user_clusterer = KMeans(n_clusters=10, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    async def create_behavior_profile(self, entity_id: str, entity_type: str, 
                                    historical_data: List[Dict[str, Any]]) -> BehaviorProfile:
        """Crée un profil comportemental pour une entité"""
        
        # Extraction des features comportementales
        features = await self._extract_behavioral_features(historical_data)
        
        # Calcul des métriques de base
        baseline_metrics = self._calculate_baseline_metrics(features)
        
        # Score de risque initial
        risk_score = self._calculate_initial_risk_score(features, baseline_metrics)
        
        profile = BehaviorProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            behavior_patterns=features,
            baseline_metrics=baseline_metrics,
            risk_score=risk_score,
            last_updated=datetime.now(),
            anomaly_history=[]
        )
        
        # Stocker le profil
        await self._store_profile(profile)
        
        return profile
    
    async def _extract_behavioral_features(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extrait les features comportementales des données"""
        features = {
            'activity_volume': [],
            'temporal_patterns': {},
            'geographic_locations': [],
            'content_preferences': {},
            'interaction_types': {},
            'session_durations': [],
            'device_patterns': {},
            'api_usage_patterns': {}
        }
        
        for record in data:
            timestamp = pd.to_datetime(record.get('timestamp', datetime.now()))
            
            # Volume d'activité par heure
            hour = timestamp.hour
            features['temporal_patterns'][hour] = features['temporal_patterns'].get(hour, 0) + 1
            
            # Localisation géographique
            if 'location' in record:
                features['geographic_locations'].append(record['location'])
            
            # Préférences de contenu
            if 'content_type' in record:
                content_type = record['content_type']
                features['content_preferences'][content_type] = features['content_preferences'].get(content_type, 0) + 1
            
            # Types d'interaction
            if 'action' in record:
                action = record['action']
                features['interaction_types'][action] = features['interaction_types'].get(action, 0) + 1
            
            # Durée de session
            if 'session_duration' in record:
                features['session_durations'].append(record['session_duration'])
            
            # Patterns d'appareils
            if 'device_type' in record:
                device = record['device_type']
                features['device_patterns'][device] = features['device_patterns'].get(device, 0) + 1
            
            # Usage API
            if 'api_endpoint' in record:
                endpoint = record['api_endpoint']
                features['api_usage_patterns'][endpoint] = features['api_usage_patterns'].get(endpoint, 0) + 1
        
        return features
    
    def _calculate_baseline_metrics(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calcule les métriques de base du comportement"""
        metrics = {}
        
        # Volume d'activité moyen
        temporal_values = list(features['temporal_patterns'].values())
        metrics['avg_hourly_activity'] = np.mean(temporal_values) if temporal_values else 0
        metrics['activity_std'] = np.std(temporal_values) if temporal_values else 0
        
        # Diversité de contenu (entropie)
        content_values = list(features['content_preferences'].values())
        if content_values:
            total = sum(content_values)
            probs = [v/total for v in content_values]
            metrics['content_diversity'] = entropy(probs)
        else:
            metrics['content_diversity'] = 0
        
        # Durée de session moyenne
        session_durations = features['session_durations']
        metrics['avg_session_duration'] = np.mean(session_durations) if session_durations else 0
        metrics['session_duration_std'] = np.std(session_durations) if session_durations else 0
        
        # Nombre de localisations uniques
        unique_locations = len(set(features['geographic_locations']))
        metrics['location_diversity'] = unique_locations
        
        # Diversité des interactions
        interaction_values = list(features['interaction_types'].values())
        if interaction_values:
            total = sum(interaction_values)
            probs = [v/total for v in interaction_values]
            metrics['interaction_diversity'] = entropy(probs)
        else:
            metrics['interaction_diversity'] = 0
        
        return metrics
    
    def _calculate_initial_risk_score(self, features: Dict[str, Any], 
                                    baseline_metrics: Dict[str, float]) -> float:
        """Calcule le score de risque initial"""
        risk_factors = []
        
        # Facteur de volume d'activité
        avg_activity = baseline_metrics.get('avg_hourly_activity', 0)
        if avg_activity > 100:  # Activité très élevée
            risk_factors.append(0.3)
        
        # Facteur de diversité géographique
        location_diversity = baseline_metrics.get('location_diversity', 0)
        if location_diversity > 10:  # Trop de localisations
            risk_factors.append(0.4)
        
        # Facteur de diversité de contenu
        content_diversity = baseline_metrics.get('content_diversity', 0)
        if content_diversity < 0.5:  # Contenu peu diversifié (comportement de bot)
            risk_factors.append(0.2)
        
        # Score final
        return min(sum(risk_factors), 1.0)
    
    async def detect_behavioral_anomaly(self, entity_id: str, 
                                      current_data: Dict[str, Any]) -> PatternResult:
        """Détecte les anomalies comportementales"""
        
        # Récupérer le profil existant
        profile = await self._get_profile(entity_id)
        if not profile:
            return PatternResult(
                pattern_detected=False,
                pattern_type=BehaviorType.USER_ACTIVITY.value,
                confidence_score=0.0,
                pattern_description="Profil non trouvé",
                anomaly_indicators=[],
                risk_level="unknown",
                recommended_actions=["Créer un profil comportemental"],
                context={'entity_id': entity_id}
            )
        
        anomaly_indicators = []
        anomaly_scores = []
        
        # Analyser le volume d'activité
        current_activity = current_data.get('activity_count', 0)
        expected_activity = profile.baseline_metrics.get('avg_hourly_activity', 0)
        activity_std = profile.baseline_metrics.get('activity_std', 1)
        
        if activity_std > 0:
            activity_zscore = abs(current_activity - expected_activity) / activity_std
            if activity_zscore > self.anomaly_thresholds['activity_volume']:
                anomaly_indicators.append(f"Volume d'activité anormal: {current_activity} vs {expected_activity}")
                anomaly_scores.append(min(activity_zscore / self.anomaly_thresholds['activity_volume'], 1.0))
        
        # Analyser les patterns temporels
        current_hour = datetime.now().hour
        expected_hourly_activity = profile.behavior_patterns['temporal_patterns'].get(current_hour, 0)
        actual_hourly_activity = current_data.get('hourly_activity', 0)
        
        if expected_hourly_activity > 0:
            temporal_deviation = abs(actual_hourly_activity - expected_hourly_activity) / expected_hourly_activity
            if temporal_deviation > 2.0:  # 200% de déviation
                anomaly_indicators.append(f"Pattern temporel inhabituel à {current_hour}h")
                anomaly_scores.append(min(temporal_deviation / 2.0, 1.0))
        
        # Analyser la localisation
        current_location = current_data.get('location')
        if current_location:
            known_locations = set(profile.behavior_patterns['geographic_locations'])
            if current_location not in known_locations:
                anomaly_indicators.append(f"Nouvelle localisation: {current_location}")
                anomaly_scores.append(0.7)
        
        # Analyser les types de contenu
        current_content_type = current_data.get('content_type')
        if current_content_type:
            known_content = profile.behavior_patterns['content_preferences']
            if current_content_type not in known_content:
                anomaly_indicators.append(f"Nouveau type de contenu: {current_content_type}")
                anomaly_scores.append(0.5)
        
        # Score global d'anomalie
        overall_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        is_anomalous = overall_score > 0.3 or len(anomaly_indicators) > 2
        
        # Mettre à jour le profil
        if is_anomalous:
            await self._update_profile_with_anomaly(profile, current_data, anomaly_indicators)
        
        return PatternResult(
            pattern_detected=is_anomalous,
            pattern_type=BehaviorType.USER_ACTIVITY.value,
            confidence_score=overall_score,
            pattern_description=f"Analyse comportementale pour {entity_id}",
            anomaly_indicators=anomaly_indicators,
            risk_level=self._calculate_risk_level_from_score(overall_score),
            recommended_actions=self._generate_behavioral_recommendations(anomaly_indicators),
            context={
                'entity_id': entity_id,
                'baseline_risk_score': profile.risk_score,
                'current_risk_score': profile.risk_score + overall_score * 0.1
            }
        )
    
    async def _store_profile(self, profile: BehaviorProfile):
        """Stocke un profil comportemental"""
        profile_data = {
            'entity_id': profile.entity_id,
            'entity_type': profile.entity_type,
            'behavior_patterns': json.dumps(profile.behavior_patterns),
            'baseline_metrics': json.dumps(profile.baseline_metrics),
            'risk_score': profile.risk_score,
            'last_updated': profile.last_updated.isoformat(),
            'anomaly_history': json.dumps(profile.anomaly_history)
        }
        
        key = f"behavior_profile:{profile.entity_id}"
        await asyncio.get_event_loop().run_in_executor(
            None, self.redis_client.hmset, key, profile_data
        )
        
        # TTL de 30 jours
        await asyncio.get_event_loop().run_in_executor(
            None, self.redis_client.expire, key, 30 * 24 * 3600
        )
    
    async def _get_profile(self, entity_id: str) -> Optional[BehaviorProfile]:
        """Récupère un profil comportemental"""
        key = f"behavior_profile:{entity_id}"
        
        try:
            profile_data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.hgetall, key
            )
            
            if not profile_data:
                return None
            
            # Décoder les données
            behavior_patterns = json.loads(profile_data[b'behavior_patterns'].decode())
            baseline_metrics = json.loads(profile_data[b'baseline_metrics'].decode())
            anomaly_history = json.loads(profile_data[b'anomaly_history'].decode())
            
            return BehaviorProfile(
                entity_id=profile_data[b'entity_id'].decode(),
                entity_type=profile_data[b'entity_type'].decode(),
                behavior_patterns=behavior_patterns,
                baseline_metrics=baseline_metrics,
                risk_score=float(profile_data[b'risk_score']),
                last_updated=pd.to_datetime(profile_data[b'last_updated'].decode()),
                anomaly_history=anomaly_history
            )
        
        except Exception as e:
            logger.error(f"Erreur récupération profil {entity_id}: {e}")
            return None
    
    async def _update_profile_with_anomaly(self, profile: BehaviorProfile, 
                                         current_data: Dict[str, Any], 
                                         anomaly_indicators: List[str]):
        """Met à jour le profil avec une nouvelle anomalie"""
        anomaly_record = {
            'timestamp': datetime.now().isoformat(),
            'indicators': anomaly_indicators,
            'data': current_data
        }
        
        profile.anomaly_history.append(anomaly_record)
        
        # Garder seulement les 100 dernières anomalies
        if len(profile.anomaly_history) > 100:
            profile.anomaly_history = profile.anomaly_history[-100:]
        
        # Augmenter légèrement le score de risque
        profile.risk_score = min(profile.risk_score + 0.05, 1.0)
        profile.last_updated = datetime.now()
        
        # Sauvegarder
        await self._store_profile(profile)
    
    def _calculate_risk_level_from_score(self, score: float) -> str:
        """Calcule le niveau de risque à partir du score"""
        if score > 0.8:
            return "critical"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        elif score > 0.2:
            return "low"
        else:
            return "info"
    
    def _generate_behavioral_recommendations(self, indicators: List[str]) -> List[str]:
        """Génère des recommandations comportementales"""
        recommendations = []
        
        if any("volume" in ind.lower() for ind in indicators):
            recommendations.append("Analyser la cause du pic d'activité")
        
        if any("temporel" in ind.lower() for ind in indicators):
            recommendations.append("Vérifier les patterns d'utilisation inhabituels")
        
        if any("localisation" in ind.lower() for ind in indicators):
            recommendations.append("Confirmer l'authenticité de la nouvelle localisation")
        
        if any("contenu" in ind.lower() for ind in indicators):
            recommendations.append("Surveiller les changements de préférences")
        
        return recommendations or ["Continuer la surveillance comportementale"]

class SecurityPatternDetector:
    """Détecteur de patterns de sécurité avancé"""
    
    def __init__(self):
        self.suspicious_patterns = {
            'brute_force': {
                'max_attempts': 10,
                'time_window': 300,  # 5 minutes
                'pattern': r'(failed|error|invalid).*(login|auth|password)'
            },
            'sql_injection': {
                'pattern': r'(union|select|insert|update|delete|drop|exec|script)',
                'context': ['query', 'parameter', 'input']
            },
            'xss_attack': {
                'pattern': r'(<script|javascript:|vbscript:|onload=|onerror=)',
                'context': ['input', 'form', 'parameter']
            },
            'directory_traversal': {
                'pattern': r'(\.\./|\.\.\\|%2e%2e)',
                'context': ['path', 'file', 'url']
            },
            'rate_limiting': {
                'max_requests': 1000,
                'time_window': 3600,  # 1 heure
                'burst_threshold': 100  # requêtes en 1 minute
            }
        }
        
        self.attack_counters = defaultdict(lambda: defaultdict(int))
        self.ip_reputation = {}
        self.blocked_ips = set()
    
    async def detect_security_patterns(self, event_data: Dict[str, Any]) -> PatternResult:
        """Détecte les patterns de sécurité suspects"""
        anomaly_indicators = []
        threat_scores = []
        threat_types = []
        
        # Extraction des informations de base
        ip_address = event_data.get('ip_address', '')
        user_agent = event_data.get('user_agent', '')
        request_path = event_data.get('path', '')
        request_data = str(event_data.get('data', ''))
        timestamp = event_data.get('timestamp', datetime.now())
        
        # Détection de brute force
        brute_force_result = await self._detect_brute_force(ip_address, event_data, timestamp)
        if brute_force_result['detected']:
            anomaly_indicators.extend(brute_force_result['indicators'])
            threat_scores.append(brute_force_result['score'])
            threat_types.append('brute_force')
        
        # Détection d'injection SQL
        sql_injection_result = self._detect_sql_injection(request_data, request_path)
        if sql_injection_result['detected']:
            anomaly_indicators.extend(sql_injection_result['indicators'])
            threat_scores.append(sql_injection_result['score'])
            threat_types.append('sql_injection')
        
        # Détection XSS
        xss_result = self._detect_xss(request_data, request_path)
        if xss_result['detected']:
            anomaly_indicators.extend(xss_result['indicators'])
            threat_scores.append(xss_result['score'])
            threat_types.append('xss_attack')
        
        # Détection de directory traversal
        dir_traversal_result = self._detect_directory_traversal(request_path, request_data)
        if dir_traversal_result['detected']:
            anomaly_indicators.extend(dir_traversal_result['indicators'])
            threat_scores.append(dir_traversal_result['score'])
            threat_types.append('directory_traversal')
        
        # Détection de rate limiting
        rate_limit_result = await self._detect_rate_limiting(ip_address, timestamp)
        if rate_limit_result['detected']:
            anomaly_indicators.extend(rate_limit_result['indicators'])
            threat_scores.append(rate_limit_result['score'])
            threat_types.append('rate_limiting')
        
        # Analyse de la réputation IP
        ip_reputation_result = await self._analyze_ip_reputation(ip_address)
        if ip_reputation_result['suspicious']:
            anomaly_indicators.extend(ip_reputation_result['indicators'])
            threat_scores.append(ip_reputation_result['score'])
            threat_types.append('ip_reputation')
        
        # Score global de menace
        overall_threat_score = max(threat_scores) if threat_scores else 0.0
        is_threat_detected = overall_threat_score > 0.3 or len(anomaly_indicators) > 0
        
        return PatternResult(
            pattern_detected=is_threat_detected,
            pattern_type=PatternType.SECURITY.value,
            confidence_score=overall_threat_score,
            pattern_description=f"Analyse de sécurité pour {ip_address}",
            anomaly_indicators=anomaly_indicators,
            risk_level=self._calculate_security_risk_level(overall_threat_score, threat_types),
            recommended_actions=self._generate_security_recommendations(threat_types, ip_address),
            context={
                'ip_address': ip_address,
                'threat_types': threat_types,
                'user_agent': user_agent,
                'request_path': request_path
            }
        )
    
    async def _detect_brute_force(self, ip_address: str, event_data: Dict[str, Any], 
                                timestamp: datetime) -> Dict[str, Any]:
        """Détecte les attaques par force brute"""
        result = {'detected': False, 'indicators': [], 'score': 0.0}
        
        # Vérifier si c'est un échec d'authentification
        log_message = str(event_data.get('message', '')).lower()
        status_code = event_data.get('status_code', 200)
        
        is_auth_failure = (
            status_code in [401, 403] or
            any(keyword in log_message for keyword in ['failed', 'invalid', 'denied', 'unauthorized'])
        )
        
        if is_auth_failure:
            # Compter les tentatives
            time_key = timestamp.replace(second=0, microsecond=0)
            self.attack_counters[ip_address][time_key] += 1
            
            # Nettoyer les anciens compteurs (plus de 5 minutes)
            cutoff_time = timestamp - timedelta(minutes=5)
            keys_to_remove = [k for k in self.attack_counters[ip_address] if k < cutoff_time]
            for key in keys_to_remove:
                del self.attack_counters[ip_address][key]
            
            # Vérifier le seuil
            recent_attempts = sum(self.attack_counters[ip_address].values())
            max_attempts = self.suspicious_patterns['brute_force']['max_attempts']
            
            if recent_attempts >= max_attempts:
                result['detected'] = True
                result['indicators'].append(f"Brute force détecté: {recent_attempts} tentatives en 5 minutes")
                result['score'] = min(recent_attempts / max_attempts, 1.0)
                
                # Bloquer l'IP
                self.blocked_ips.add(ip_address)
        
        return result
    
    def _detect_sql_injection(self, request_data: str, request_path: str) -> Dict[str, Any]:
        """Détecte les tentatives d'injection SQL"""
        result = {'detected': False, 'indicators': [], 'score': 0.0}
        
        pattern = self.suspicious_patterns['sql_injection']['pattern']
        text_to_check = f"{request_data} {request_path}".lower()
        
        matches = re.findall(pattern, text_to_check, re.IGNORECASE)
        if matches:
            result['detected'] = True
            result['indicators'].append(f"Tentative d'injection SQL détectée: {matches}")
            result['score'] = min(len(matches) * 0.3, 1.0)
        
        # Vérifications supplémentaires
        suspicious_combinations = [
            'union select',
            'or 1=1',
            'drop table',
            'exec xp_',
            'script>'
        ]
        
        for combo in suspicious_combinations:
            if combo in text_to_check:
                result['detected'] = True
                result['indicators'].append(f"Pattern SQL malveillant: {combo}")
                result['score'] = max(result['score'], 0.8)
        
        return result
    
    def _detect_xss(self, request_data: str, request_path: str) -> Dict[str, Any]:
        """Détecte les tentatives d'attaque XSS"""
        result = {'detected': False, 'indicators': [], 'score': 0.0}
        
        pattern = self.suspicious_patterns['xss_attack']['pattern']
        text_to_check = f"{request_data} {request_path}".lower()
        
        matches = re.findall(pattern, text_to_check, re.IGNORECASE)
        if matches:
            result['detected'] = True
            result['indicators'].append(f"Tentative XSS détectée: {matches}")
            result['score'] = min(len(matches) * 0.4, 1.0)
        
        return result
    
    def _detect_directory_traversal(self, request_path: str, request_data: str) -> Dict[str, Any]:
        """Détecte les tentatives de directory traversal"""
        result = {'detected': False, 'indicators': [], 'score': 0.0}
        
        pattern = self.suspicious_patterns['directory_traversal']['pattern']
        text_to_check = f"{request_path} {request_data}"
        
        matches = re.findall(pattern, text_to_check, re.IGNORECASE)
        if matches:
            result['detected'] = True
            result['indicators'].append(f"Tentative de directory traversal: {matches}")
            result['score'] = min(len(matches) * 0.5, 1.0)
        
        return result
    
    async def _detect_rate_limiting(self, ip_address: str, timestamp: datetime) -> Dict[str, Any]:
        """Détecte les violations de rate limiting"""
        result = {'detected': False, 'indicators': [], 'score': 0.0}
        
        # Compter les requêtes par minute
        minute_key = timestamp.replace(second=0, microsecond=0)
        minute_counter_key = f"minute_{minute_key}"
        
        if minute_counter_key not in self.attack_counters[ip_address]:
            self.attack_counters[ip_address][minute_counter_key] = 0
        
        self.attack_counters[ip_address][minute_counter_key] += 1
        
        # Vérifier le burst threshold
        minute_requests = self.attack_counters[ip_address][minute_counter_key]
        burst_threshold = self.suspicious_patterns['rate_limiting']['burst_threshold']
        
        if minute_requests > burst_threshold:
            result['detected'] = True
            result['indicators'].append(f"Burst de requêtes: {minute_requests}/minute")
            result['score'] = min(minute_requests / burst_threshold, 1.0)
        
        return result
    
    async def _analyze_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Analyse la réputation d'une adresse IP"""
        result = {'suspicious': False, 'indicators': [], 'score': 0.0}
        
        try:
            # Vérifier si c'est une IP privée
            ip_obj = ipaddress.ip_address(ip_address)
            if ip_obj.is_private:
                return result
            
            # Vérifier la liste noire locale
            if ip_address in self.blocked_ips:
                result['suspicious'] = True
                result['indicators'].append(f"IP bloquée localement: {ip_address}")
                result['score'] = 1.0
                return result
            
            # Vérifications géographiques (simulation)
            # Dans un vrai système, utiliser une vraie base de données GeoIP
            suspicious_countries = ['CN', 'RU', 'KP', 'IR']  # Exemple
            # country_code = get_country_code(ip_address)  # À implémenter
            
            # Ici, on simule une vérification
            import random
            if random.random() < 0.1:  # 10% de chance d'être suspect
                result['suspicious'] = True
                result['indicators'].append(f"IP géographiquement suspecte: {ip_address}")
                result['score'] = 0.6
        
        except Exception as e:
            logger.warning(f"Erreur analyse IP {ip_address}: {e}")
        
        return result
    
    def _calculate_security_risk_level(self, score: float, threat_types: List[str]) -> str:
        """Calcule le niveau de risque de sécurité"""
        critical_threats = ['brute_force', 'sql_injection']
        
        if any(threat in critical_threats for threat in threat_types):
            return "critical"
        elif score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        elif score > 0.2:
            return "low"
        else:
            return "info"
    
    def _generate_security_recommendations(self, threat_types: List[str], 
                                         ip_address: str) -> List[str]:
        """Génère des recommandations de sécurité"""
        recommendations = []
        
        if 'brute_force' in threat_types:
            recommendations.append(f"Bloquer immédiatement l'IP {ip_address}")
            recommendations.append("Activer le captcha pour les connexions")
        
        if 'sql_injection' in threat_types:
            recommendations.append("Vérifier la validation des entrées")
            recommendations.append("Auditer les requêtes SQL")
        
        if 'xss_attack' in threat_types:
            recommendations.append("Implémenter l'échappement HTML")
            recommendations.append("Configurer Content Security Policy")
        
        if 'rate_limiting' in threat_types:
            recommendations.append("Renforcer les limites de taux")
            recommendations.append("Implémenter un système de quotas")
        
        return recommendations or ["Continuer la surveillance de sécurité"]
