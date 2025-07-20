"""
Utilitaires Avancés pour le Système de Cache
===========================================

Collection d'utilitaires essentiels pour la gestion optimisée du cache,
incluant génération de clés, calcul de TTL, sécurité, validation et compression.

Fonctionnalités:
- Génération de clés hiérarchiques et sécurisées
- Calcul de TTL adaptatif basé sur l'usage
- Utilitaires de sécurité et validation
- Compression intelligente et analyse de données
- Outils de monitoring et debugging
- Helper pour optimisation des performances

Auteurs: Équipe Spotify AI Agent - Direction technique Fahed Mlaiel
"""

import hashlib
import hmac
import base64
import time
import json
import zlib
import threading
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import sys
import gzip
import pickle
from collections import defaultdict, deque
import secrets
import uuid

from .exceptions import CacheSecurityError, CacheValidationError


@dataclass
class CacheKeyInfo:
    """Informations sur une clé de cache"""
    original_key: str
    hashed_key: str
    namespace: str
    tenant_id: Optional[str] = None
    created_at: datetime = None
    key_type: str = "standard"
    security_level: str = "normal"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class CacheKeyGenerator:
    """Générateur de clés de cache sécurisées et optimisées"""
    
    def __init__(self, secret_key: str = None, max_key_length: int = 250):
        self.secret_key = secret_key or self._generate_secret_key()
        self.max_key_length = max_key_length
        self.namespace_separator = "::"
        self.key_history = deque(maxlen=10000)
        self.key_stats = defaultdict(int)
        
    def _generate_secret_key(self) -> str:
        """Génère une clé secrète sécurisée"""
        return secrets.token_urlsafe(32)
    
    def generate_key(self, base_key: str, namespace: str = "default",
                    tenant_id: str = None, key_type: str = "standard",
                    include_timestamp: bool = False,
                    security_level: str = "normal") -> CacheKeyInfo:
        """Génère une clé de cache sécurisée et optimisée"""
        
        # Nettoyage et validation de la clé de base
        cleaned_key = self._clean_key(base_key)
        
        # Construction de la clé hiérarchique
        key_parts = [namespace]
        
        if tenant_id:
            key_parts.append(f"tenant:{tenant_id}")
        
        if key_type != "standard":
            key_parts.append(f"type:{key_type}")
        
        if include_timestamp:
            timestamp = int(time.time() * 1000)  # Milliseconds
            key_parts.append(f"ts:{timestamp}")
        
        key_parts.append(cleaned_key)
        
        # Assemblage de la clé
        full_key = self.namespace_separator.join(key_parts)
        
        # Hachage sécurisé si nécessaire
        if len(full_key) > self.max_key_length or security_level == "high":
            hashed_key = self._hash_key(full_key, security_level)
        else:
            hashed_key = full_key
        
        # Création de l'objet CacheKeyInfo
        key_info = CacheKeyInfo(
            original_key=base_key,
            hashed_key=hashed_key,
            namespace=namespace,
            tenant_id=tenant_id,
            key_type=key_type,
            security_level=security_level,
            metadata={
                "full_key": full_key,
                "key_length": len(hashed_key),
                "hashed": len(full_key) > self.max_key_length
            }
        )
        
        # Enregistrement pour statistiques
        self._record_key_usage(key_info)
        
        return key_info
    
    def _clean_key(self, key: str) -> str:
        """Nettoie et normalise une clé"""
        # Suppression des caractères dangereux
        cleaned = re.sub(r'[^\w\-_.:/]', '_', key)
        
        # Normalisation des séparateurs
        cleaned = re.sub(r'[/\\]+', '/', cleaned)
        
        # Suppression des doubles underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # Limitation de la longueur
        if len(cleaned) > 200:
            cleaned = cleaned[:200]
        
        return cleaned.strip('_')
    
    def _hash_key(self, key: str, security_level: str = "normal") -> str:
        """Hache une clé selon le niveau de sécurité"""
        key_bytes = key.encode('utf-8')
        
        if security_level == "high":
            # HMAC-SHA256 pour haute sécurité
            mac = hmac.new(
                self.secret_key.encode('utf-8'),
                key_bytes,
                hashlib.sha256
            )
            hash_value = mac.hexdigest()
        else:
            # SHA-256 standard
            hash_value = hashlib.sha256(key_bytes).hexdigest()
        
        # Préfixe pour identifier le type de hash
        prefix = "hs:" if security_level == "high" else "s:"
        
        return f"{prefix}{hash_value[:32]}"
    
    def _record_key_usage(self, key_info: CacheKeyInfo):
        """Enregistre l'utilisation d'une clé pour statistiques"""
        self.key_history.append({
            "timestamp": datetime.now(),
            "namespace": key_info.namespace,
            "tenant_id": key_info.tenant_id,
            "key_type": key_info.key_type,
            "security_level": key_info.security_level
        })
        
        # Mise à jour des statistiques
        self.key_stats[key_info.namespace] += 1
        if key_info.tenant_id:
            self.key_stats[f"tenant:{key_info.tenant_id}"] += 1
    
    def generate_batch_keys(self, base_keys: List[str], **kwargs) -> List[CacheKeyInfo]:
        """Génère un lot de clés de manière optimisée"""
        return [self.generate_key(key, **kwargs) for key in base_keys]
    
    def get_key_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation des clés"""
        return {
            "total_keys_generated": len(self.key_history),
            "namespace_usage": dict(self.key_stats),
            "recent_activity": list(self.key_history)[-10:] if self.key_history else []
        }


class TTLCalculator:
    """Calculateur de TTL adaptatif basé sur l'usage et les patterns"""
    
    def __init__(self, default_ttl: int = 3600, min_ttl: int = 60, max_ttl: int = 86400):
        self.default_ttl = default_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.access_patterns = defaultdict(list)
        self.ttl_performance = defaultdict(dict)
        
    def calculate_ttl(self, key: str, data_size: int = 0, 
                     access_frequency: float = 0.0, 
                     data_type: str = "generic",
                     tenant_id: str = None) -> int:
        """Calcule un TTL optimal basé sur plusieurs facteurs"""
        
        # TTL de base selon le type de données
        base_ttl = self._get_base_ttl_by_type(data_type)
        
        # Ajustement selon la taille des données
        size_factor = self._calculate_size_factor(data_size)
        
        # Ajustement selon la fréquence d'accès
        frequency_factor = self._calculate_frequency_factor(access_frequency)
        
        # Ajustement selon l'historique de la clé
        history_factor = self._calculate_history_factor(key)
        
        # Ajustement selon le tenant (quota, priorité)
        tenant_factor = self._calculate_tenant_factor(tenant_id)
        
        # Calcul final
        calculated_ttl = int(
            base_ttl * size_factor * frequency_factor * history_factor * tenant_factor
        )
        
        # Application des limites
        final_ttl = max(self.min_ttl, min(calculated_ttl, self.max_ttl))
        
        # Enregistrement pour apprentissage
        self._record_ttl_decision(key, final_ttl, {
            "base_ttl": base_ttl,
            "size_factor": size_factor,
            "frequency_factor": frequency_factor,
            "history_factor": history_factor,
            "tenant_factor": tenant_factor
        })
        
        return final_ttl
    
    def _get_base_ttl_by_type(self, data_type: str) -> int:
        """Retourne le TTL de base selon le type de données"""
        type_mapping = {
            "session": 1800,      # 30 minutes
            "user_profile": 3600, # 1 heure
            "music_metadata": 7200, # 2 heures
            "analytics": 900,     # 15 minutes
            "ml_model": 14400,    # 4 heures
            "configuration": 86400, # 24 heures
            "temporary": 300,     # 5 minutes
            "generic": self.default_ttl
        }
        return type_mapping.get(data_type, self.default_ttl)
    
    def _calculate_size_factor(self, data_size: int) -> float:
        """Calcule un facteur basé sur la taille des données"""
        if data_size == 0:
            return 1.0
        
        # Plus les données sont grandes, plus le TTL est long (éviter la recomputation)
        if data_size < 1024:  # < 1KB
            return 0.5
        elif data_size < 10240:  # < 10KB
            return 1.0
        elif data_size < 102400:  # < 100KB
            return 1.5
        else:  # >= 100KB
            return 2.0
    
    def _calculate_frequency_factor(self, access_frequency: float) -> float:
        """Calcule un facteur basé sur la fréquence d'accès"""
        if access_frequency == 0.0:
            return 1.0
        
        # Plus la fréquence est élevée, plus le TTL est long
        if access_frequency < 0.1:  # Rarement accédé
            return 0.5
        elif access_frequency < 1.0:  # Accès normal
            return 1.0
        elif access_frequency < 10.0:  # Accès fréquent
            return 1.5
        else:  # Très fréquent
            return 2.0
    
    def _calculate_history_factor(self, key: str) -> float:
        """Calcule un facteur basé sur l'historique de la clé"""
        if key not in self.access_patterns:
            return 1.0
        
        patterns = self.access_patterns[key]
        if len(patterns) < 2:
            return 1.0
        
        # Analyse des patterns d'accès
        recent_accesses = [p for p in patterns if (datetime.now() - p).seconds < 3600]
        
        if len(recent_accesses) > 5:
            return 1.5  # Très populaire récemment
        elif len(recent_accesses) > 2:
            return 1.2  # Populaire
        else:
            return 0.8  # Moins populaire
    
    def _calculate_tenant_factor(self, tenant_id: str) -> float:
        """Calcule un facteur basé sur le tenant"""
        if not tenant_id:
            return 1.0
        
        # Ici, on pourrait intégrer des informations sur le plan du tenant,
        # ses quotas, sa priorité, etc.
        # Pour l'exemple, on retourne 1.0
        return 1.0
    
    def _record_ttl_decision(self, key: str, ttl: int, factors: Dict[str, float]):
        """Enregistre une décision de TTL pour apprentissage"""
        self.ttl_performance[key] = {
            "ttl": ttl,
            "factors": factors,
            "timestamp": datetime.now(),
            "hits": 0,
            "effective_duration": None
        }
    
    def record_access(self, key: str):
        """Enregistre un accès à une clé"""
        self.access_patterns[key].append(datetime.now())
        
        # Nettoyage des anciens accès (garde seulement les 24 dernières heures)
        cutoff = datetime.now() - timedelta(hours=24)
        self.access_patterns[key] = [
            access for access in self.access_patterns[key] if access > cutoff
        ]
        
        # Mise à jour des performances TTL
        if key in self.ttl_performance:
            self.ttl_performance[key]["hits"] += 1
    
    def record_expiration(self, key: str):
        """Enregistre l'expiration d'une clé"""
        if key in self.ttl_performance:
            perf = self.ttl_performance[key]
            if perf["timestamp"]:
                perf["effective_duration"] = (datetime.now() - perf["timestamp"]).total_seconds()
    
    def get_optimal_ttl_suggestions(self) -> Dict[str, int]:
        """Retourne des suggestions d'optimisation TTL"""
        suggestions = {}
        
        for key, perf in self.ttl_performance.items():
            if perf["effective_duration"] and perf["hits"] > 1:
                # Si la durée effective est beaucoup plus courte que le TTL,
                # suggérer une réduction
                if perf["effective_duration"] < perf["ttl"] * 0.5:
                    suggestions[key] = int(perf["effective_duration"] * 1.2)
                # Si beaucoup d'accès et pas d'expiration, suggérer une augmentation
                elif perf["hits"] > 10 and not perf["effective_duration"]:
                    suggestions[key] = int(perf["ttl"] * 1.5)
        
        return suggestions


class SecurityUtils:
    """Utilitaires de sécurité pour le cache"""
    
    def __init__(self, master_key: str = None):
        self.master_key = master_key or self._generate_master_key()
        self.security_policies = self._load_default_policies()
        
    def _generate_master_key(self) -> str:
        """Génère une clé maître sécurisée"""
        return secrets.token_urlsafe(64)
    
    def _load_default_policies(self) -> Dict[str, Any]:
        """Charge les politiques de sécurité par défaut"""
        return {
            "encryption_required_patterns": [
                r".*password.*",
                r".*secret.*",
                r".*token.*",
                r".*credential.*"
            ],
            "sensitive_namespaces": [
                "user_secrets",
                "api_keys",
                "oauth_tokens"
            ],
            "max_value_size": 1024 * 1024,  # 1MB
            "allowed_tenant_operations": {
                "read": True,
                "write": True,
                "delete": True,
                "admin": False
            }
        }
    
    def is_sensitive_key(self, key: str, namespace: str = None) -> bool:
        """Détermine si une clé contient des données sensibles"""
        # Vérification par pattern
        for pattern in self.security_policies["encryption_required_patterns"]:
            if re.search(pattern, key, re.IGNORECASE):
                return True
        
        # Vérification par namespace
        if namespace in self.security_policies["sensitive_namespaces"]:
            return True
        
        return False
    
    def validate_tenant_access(self, tenant_id: str, operation: str, 
                             resource: str = None) -> bool:
        """Valide l'accès d'un tenant à une opération"""
        if not tenant_id:
            return False
        
        # Vérification des opérations autorisées
        allowed_ops = self.security_policies["allowed_tenant_operations"]
        if operation not in allowed_ops or not allowed_ops[operation]:
            return False
        
        # Ici, on pourrait ajouter des vérifications plus complexes
        # comme les ACL, les rôles, etc.
        
        return True
    
    def validate_value_size(self, value: Any) -> bool:
        """Valide la taille d'une valeur"""
        try:
            size = len(pickle.dumps(value))
            return size <= self.security_policies["max_value_size"]
        except Exception:
            return False
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Génère un token sécurisé"""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str, salt: str = None) -> Tuple[str, str]:
        """Hache des données sensibles avec salt"""
        if salt is None:
            salt = secrets.token_urlsafe(16)
        
        combined = f"{salt}{data}{self.master_key}"
        hash_value = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        return hash_value, salt
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Vérifie un hash de données sensibles"""
        expected_hash, _ = self.hash_sensitive_data(data, salt)
        return hmac.compare_digest(hash_value, expected_hash)


class ValidationUtils:
    """Utilitaires de validation pour le cache"""
    
    @staticmethod
    def validate_key(key: str) -> bool:
        """Valide une clé de cache"""
        if not key or not isinstance(key, str):
            return False
        
        if len(key) > 500:  # Clé trop longue
            return False
        
        # Caractères interdits
        forbidden_chars = ['\n', '\r', '\t', '\0']
        if any(char in key for char in forbidden_chars):
            return False
        
        return True
    
    @staticmethod
    def validate_ttl(ttl: int) -> bool:
        """Valide une valeur TTL"""
        if not isinstance(ttl, int):
            return False
        
        return 0 < ttl <= 31536000  # Max 1 an
    
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """Valide un ID de tenant"""
        if not tenant_id or not isinstance(tenant_id, str):
            return False
        
        # Format UUID ou alphanumeric
        pattern = r'^[a-zA-Z0-9\-_]{1,64}$'
        return bool(re.match(pattern, tenant_id))
    
    @staticmethod
    def validate_namespace(namespace: str) -> bool:
        """Valide un namespace"""
        if not namespace or not isinstance(namespace, str):
            return False
        
        pattern = r'^[a-zA-Z0-9\-_.]{1,100}$'
        return bool(re.match(pattern, namespace))
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Nettoie et sécurise une chaîne"""
        if not isinstance(value, str):
            return str(value)
        
        # Suppression des caractères de contrôle
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # Limitation de la longueur
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()


class CompressionUtils:
    """Utilitaires de compression avancés"""
    
    @staticmethod
    def analyze_data_compressibility(data: bytes) -> Dict[str, Any]:
        """Analyse la compressibilité des données"""
        original_size = len(data)
        
        if original_size == 0:
            return {"compressible": False, "analysis": "empty_data"}
        
        # Test avec zlib (rapide)
        try:
            compressed = zlib.compress(data, level=1)
            compression_ratio = len(compressed) / original_size
            
            analysis = {
                "compressible": compression_ratio < 0.9,
                "compression_ratio": compression_ratio,
                "original_size": original_size,
                "compressed_size": len(compressed),
                "recommended": compression_ratio < 0.7,
                "algorithm_suggestion": "zstd" if compression_ratio < 0.5 else "lz4"
            }
            
            return analysis
        except Exception:
            return {"compressible": False, "analysis": "compression_error"}
    
    @staticmethod
    def estimate_optimal_compression_level(data: bytes, algorithm: str = "zlib") -> int:
        """Estime le niveau de compression optimal"""
        if len(data) < 1024:  # Petites données
            return 1
        elif len(data) < 10240:  # Données moyennes
            return 3
        else:  # Grandes données
            return 6


class PerformanceUtils:
    """Utilitaires d'optimisation des performances"""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.lock = threading.Lock()
    
    def time_operation(self, operation_name: str):
        """Décorateur pour mesurer le temps d'opération"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    with self.lock:
                        self.timing_data[operation_name].append(duration)
                        # Garde seulement les 1000 dernières mesures
                        if len(self.timing_data[operation_name]) > 1000:
                            self.timing_data[operation_name] = self.timing_data[operation_name][-1000:]
            return wrapper
        return decorator
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Retourne les statistiques de performance"""
        stats = {}
        
        with self.lock:
            for operation, timings in self.timing_data.items():
                if timings:
                    stats[operation] = {
                        "count": len(timings),
                        "avg": sum(timings) / len(timings),
                        "min": min(timings),
                        "max": max(timings),
                        "p95": self._percentile(timings, 95),
                        "p99": self._percentile(timings, 99)
                    }
        
        return stats
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class DebugUtils:
    """Utilitaires de debugging pour le cache"""
    
    @staticmethod
    def format_cache_entry(entry: Any) -> str:
        """Formate une entrée de cache pour l'affichage"""
        try:
            if hasattr(entry, '__dict__'):
                return json.dumps(entry.__dict__, default=str, indent=2)
            else:
                return str(entry)
        except Exception:
            return f"<{type(entry).__name__} object>"
    
    @staticmethod
    def estimate_memory_usage(obj: Any) -> int:
        """Estime l'utilisation mémoire d'un objet"""
        try:
            return sys.getsizeof(pickle.dumps(obj))
        except Exception:
            return sys.getsizeof(obj)
    
    @staticmethod
    def analyze_cache_key_distribution(keys: List[str]) -> Dict[str, Any]:
        """Analyse la distribution des clés de cache"""
        if not keys:
            return {"total": 0}
        
        namespaces = defaultdict(int)
        key_lengths = []
        
        for key in keys:
            parts = key.split("::")
            if len(parts) > 1:
                namespaces[parts[0]] += 1
            key_lengths.append(len(key))
        
        return {
            "total": len(keys),
            "unique_namespaces": len(namespaces),
            "namespace_distribution": dict(namespaces),
            "avg_key_length": sum(key_lengths) / len(key_lengths),
            "max_key_length": max(key_lengths),
            "min_key_length": min(key_lengths)
        }
