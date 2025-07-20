"""
Utilitaires pour le module Warning - Spotify AI Agent
Fonctions d'aide, sécurité, performance et analyse pour le système d'alerting
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import re
import base64
from urllib.parse import urlparse
import ipaddress

import aioredis
import aiohttp
from cryptography.fernet import Fernet
from prometheus_client import Counter, Histogram, Gauge
import psutil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SecurityUtils:
    """Utilitaires de sécurité pour le module d'alerting"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.logger = logging.getLogger("security_utils")
    
    def encrypt_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """
        Chiffre des données sensibles
        
        Args:
            data: Données à chiffrer (string ou dict)
            
        Returns:
            str: Données chiffrées en base64
        """
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            
            encrypted = self.cipher.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Déchiffre des données
        
        Args:
            encrypted_data: Données chiffrées en base64
            
        Returns:
            str: Données déchiffrées
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
            
        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            raise
    
    def generate_api_key(self, prefix: str = "spa") -> str:
        """
        Génère une clé API sécurisée
        
        Args:
            prefix: Préfixe pour la clé
            
        Returns:
            str: Clé API générée
        """
        random_part = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode().rstrip('=')
        timestamp = str(int(time.time()))
        return f"{prefix}_{timestamp}_{random_part}"
    
    def validate_webhook_signature(
        self,
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """
        Valide la signature d'un webhook
        
        Args:
            payload: Charge utile du webhook
            signature: Signature reçue
            secret: Secret partagé
            
        Returns:
            bool: True si valide
        """
        try:
            expected_signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(f"sha256={expected_signature}", signature)
            
        except Exception as e:
            self.logger.error(f"Signature validation error: {str(e)}")
            return False
    
    def sanitize_input(self, input_text: str, max_length: int = 1000) -> str:
        """
        Nettoie et sécurise une entrée utilisateur
        
        Args:
            input_text: Texte à nettoyer
            max_length: Longueur maximale
            
        Returns:
            str: Texte nettoyé
        """
        if not input_text:
            return ""
        
        # Suppression des caractères dangereux
        sanitized = re.sub(r'[<>"\';\\]', '', input_text)
        
        # Limitation de la longueur
        sanitized = sanitized[:max_length]
        
        # Suppression des espaces en trop
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def is_safe_url(self, url: str) -> bool:
        """
        Vérifie si une URL est sûre
        
        Args:
            url: URL à vérifier
            
        Returns:
            bool: True si sûre
        """
        try:
            parsed = urlparse(url)
            
            # Vérification du schéma
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Vérification de l'hôte
            if not parsed.netloc:
                return False
            
            # Vérification contre les IPs privées
            try:
                ip = ipaddress.ip_address(parsed.hostname)
                if ip.is_private or ip.is_loopback:
                    return False
            except:
                pass  # Pas une IP, c'est OK
            
            return True
            
        except Exception:
            return False
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash des données sensibles avec salt
        
        Args:
            data: Données à hasher
            salt: Salt (généré si None)
            
        Returns:
            Tuple[str, str]: (hash, salt)
        """
        if salt is None:
            salt = base64.b64encode(uuid.uuid4().bytes).decode()
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        hashed = base64.b64encode(hash_obj).decode()
        
        return hashed, salt


class PerformanceMonitor:
    """Moniteur de performance pour optimisation"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client
        self.logger = logging.getLogger("performance_monitor")
        
        # Métriques Prometheus
        self.operation_duration = Histogram(
            'operation_duration_seconds',
            'Duration of operations',
            ['operation_name', 'status']
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['component']
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['component']
        )
    
    async def measure_async_operation(self, operation_name: str, func, *args, **kwargs):
        """
        Mesure la performance d'une opération asynchrone
        
        Args:
            operation_name: Nom de l'opération
            func: Fonction à mesurer
            *args, **kwargs: Arguments de la fonction
            
        Returns:
            Résultat de la fonction
        """
        start_time = time.time()
        status = "success"
        
        try:
            result = await func(*args, **kwargs)
            return result
            
        except Exception as e:
            status = "error"
            self.logger.error(f"Operation {operation_name} failed: {str(e)}")
            raise
            
        finally:
            duration = time.time() - start_time
            self.operation_duration.labels(
                operation_name=operation_name,
                status=status
            ).observe(duration)
            
            # Log si l'opération est lente
            if duration > 1.0:
                self.logger.warning(
                    f"Slow operation detected: {operation_name} took {duration:.2f}s"
                )
    
    def get_system_metrics(self) -> Dict[str, float]:
        """
        Récupère les métriques système
        
        Returns:
            Dict: Métriques système
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024)
            }
            
            # Mise à jour des métriques Prometheus
            self.cpu_usage.labels(component='system').set(cpu_percent)
            self.memory_usage.labels(component='system').set(memory.used)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    async def cache_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Met en cache les métriques de performance
        
        Args:
            metrics: Métriques à mettre en cache
        """
        if not self.redis_client:
            return
        
        try:
            cache_key = f"performance_metrics:{int(time.time())}"
            await self.redis_client.setex(
                cache_key,
                3600,  # 1 heure
                json.dumps(metrics, default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Error caching performance metrics: {str(e)}")
    
    async def get_performance_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """
        Récupère les tendances de performance
        
        Args:
            hours: Nombre d'heures à analyser
            
        Returns:
            Dict: Tendances par métrique
        """
        if not self.redis_client:
            return {}
        
        try:
            end_time = int(time.time())
            start_time = end_time - (hours * 3600)
            
            trends = {}
            
            # Récupération des métriques cachées
            for timestamp in range(start_time, end_time, 3600):  # Par heure
                cache_key = f"performance_metrics:{timestamp}"
                cached_data = await self.redis_client.get(cache_key)
                
                if cached_data:
                    metrics = json.loads(cached_data)
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if metric_name not in trends:
                                trends[metric_name] = []
                            trends[metric_name].append(value)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {str(e)}")
            return {}


class TextAnalyzer:
    """Analyseur de texte avancé pour les messages d'alerte"""
    
    def __init__(self):
        self.logger = logging.getLogger("text_analyzer")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._fitted = False
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extrait les mots-clés d'un texte
        
        Args:
            text: Texte à analyser
            max_keywords: Nombre maximum de mots-clés
            
        Returns:
            List[str]: Mots-clés extraits
        """
        try:
            # Nettoyage du texte
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = cleaned_text.split()
            
            # Suppression des mots courts et communs
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            keywords = [word for word in words if len(word) > 2 and word not in stop_words]
            
            # Comptage des fréquences
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Tri par fréquence
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in sorted_words[:max_keywords]]
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité entre deux textes
        
        Args:
            text1: Premier texte
            text2: Deuxième texte
            
        Returns:
            float: Score de similarité (0-1)
        """
        try:
            if not self._fitted:
                # Fitting initial avec les deux textes
                self.vectorizer.fit([text1, text2])
                self._fitted = True
            
            # Vectorisation
            vectors = self.vectorizer.transform([text1, text2])
            
            # Calcul de la similarité cosinus
            similarity_matrix = cosine_similarity(vectors)
            
            return float(similarity_matrix[0][1])
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def detect_urgency_indicators(self, text: str) -> Dict[str, Any]:
        """
        Détecte les indicateurs d'urgence dans un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dict: Indicateurs d'urgence détectés
        """
        urgency_patterns = {
            'critical': r'\b(critical|emergency|urgent|immediate|fatal|disaster)\b',
            'high': r'\b(error|failed|failure|down|crashed|broken)\b',
            'medium': r'\b(warning|alert|issue|problem|slow)\b',
            'time_sensitive': r'\b(now|asap|immediately|urgent|quickly)\b',
            'impact_words': r'\b(users|customers|production|system|service)\b'
        }
        
        indicators = {}
        text_lower = text.lower()
        
        for category, pattern in urgency_patterns.items():
            matches = re.findall(pattern, text_lower)
            indicators[category] = {
                'count': len(matches),
                'words': list(set(matches)),
                'present': len(matches) > 0
            }
        
        # Calcul d'un score d'urgence global
        urgency_score = 0.0
        urgency_score += indicators['critical']['count'] * 0.4
        urgency_score += indicators['high']['count'] * 0.3
        urgency_score += indicators['medium']['count'] * 0.2
        urgency_score += indicators['time_sensitive']['count'] * 0.3
        urgency_score += indicators['impact_words']['count'] * 0.1
        
        indicators['urgency_score'] = min(urgency_score, 1.0)
        
        return indicators
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait des entités nommées du texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dict: Entités extraites par type
        """
        entities = {
            'urls': [],
            'emails': [],
            'ips': [],
            'numbers': [],
            'timestamps': []
        }
        
        try:
            # URLs
            url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
            entities['urls'] = re.findall(url_pattern, text)
            
            # Emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities['emails'] = re.findall(email_pattern, text)
            
            # IPs
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            entities['ips'] = re.findall(ip_pattern, text)
            
            # Nombres (avec unités)
            number_pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:ms|s|mb|gb|kb|%|rpm|rps))\b'
            entities['numbers'] = re.findall(number_pattern, text, re.IGNORECASE)
            
            # Timestamps
            timestamp_pattern = r'\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}\b'
            entities['timestamps'] = re.findall(timestamp_pattern, text)
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
        
        return entities


class MLUtils:
    """Utilitaires Machine Learning pour l'analyse des alertes"""
    
    def __init__(self):
        self.logger = logging.getLogger("ml_utils")
    
    def calculate_anomaly_score(self, values: List[float], window_size: int = 10) -> float:
        """
        Calcule un score d'anomalie pour une série de valeurs
        
        Args:
            values: Valeurs à analyser
            window_size: Taille de la fenêtre d'analyse
            
        Returns:
            float: Score d'anomalie (0-1)
        """
        try:
            if len(values) < window_size:
                return 0.0
            
            # Prendre les dernières valeurs
            recent_values = values[-window_size:]
            
            # Calcul des statistiques
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)
            
            if std_val == 0:
                return 0.0
            
            # Score basé sur l'écart à la moyenne
            latest_value = recent_values[-1]
            z_score = abs((latest_value - mean_val) / std_val)
            
            # Normalisation du z-score vers [0, 1]
            anomaly_score = min(z_score / 3.0, 1.0)  # 3 sigma = score max
            
            return anomaly_score
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly score: {str(e)}")
            return 0.0
    
    def predict_alert_escalation(self, alert_history: List[Dict[str, Any]]) -> float:
        """
        Prédit la probabilité qu'une alerte nécessite une escalade
        
        Args:
            alert_history: Historique des alertes similaires
            
        Returns:
            float: Probabilité d'escalade (0-1)
        """
        try:
            if not alert_history:
                return 0.5  # Probabilité neutre
            
            escalation_factors = {
                'severity_score': 0.3,
                'recurrence': 0.2,
                'resolution_time': 0.2,
                'user_impact': 0.2,
                'system_load': 0.1
            }
            
            escalation_score = 0.0
            
            for alert in alert_history:
                # Score de sévérité
                severity = alert.get('severity_score', 0.5)
                escalation_score += severity * escalation_factors['severity_score']
                
                # Récurrence (plus d'alertes = plus de chance d'escalade)
                escalation_score += 0.1 * escalation_factors['recurrence']
                
                # Temps de résolution (plus long = plus de chance d'escalade)
                resolution_time = alert.get('resolution_time_minutes', 30)
                if resolution_time > 60:
                    escalation_score += 0.5 * escalation_factors['resolution_time']
                
                # Impact utilisateur
                affected_users = alert.get('affected_users', 0)
                if affected_users > 100:
                    escalation_score += 0.7 * escalation_factors['user_impact']
                elif affected_users > 10:
                    escalation_score += 0.4 * escalation_factors['user_impact']
            
            # Moyenne pondérée
            escalation_probability = escalation_score / len(alert_history)
            
            return min(escalation_probability, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error predicting escalation: {str(e)}")
            return 0.5
    
    def cluster_similar_alerts(self, alerts: List[str], max_clusters: int = 5) -> Dict[str, List[str]]:
        """
        Groupe les alertes similaires en clusters
        
        Args:
            alerts: Liste des messages d'alerte
            max_clusters: Nombre maximum de clusters
            
        Returns:
            Dict: Clusters d'alertes similaires
        """
        try:
            if len(alerts) < 2:
                return {"cluster_0": alerts}
            
            # Vectorisation TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            vectors = vectorizer.fit_transform(alerts)
            
            # Clustering avec K-means ou DBSCAN selon la taille
            if len(alerts) <= max_clusters:
                # Trop peu d'alertes pour du clustering
                return {f"cluster_{i}": [alert] for i, alert in enumerate(alerts)}
            
            from sklearn.cluster import KMeans
            n_clusters = min(max_clusters, len(alerts) // 2)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(vectors)
            
            # Organisation en clusters
            clusters = {}
            for i, label in enumerate(labels):
                cluster_name = f"cluster_{label}"
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(alerts[i])
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering alerts: {str(e)}")
            return {"cluster_0": alerts}


class AttachmentBuilder:
    """Constructeur d'attachments pour les notifications enrichies"""
    
    def __init__(self):
        self.logger = logging.getLogger("attachment_builder")
    
    def build_slack_attachment(
        self,
        alert_data: Dict[str, Any],
        color: str = "#ff0000",
        include_actions: bool = True
    ) -> Dict[str, Any]:
        """
        Construit un attachment Slack enrichi
        
        Args:
            alert_data: Données de l'alerte
            color: Couleur de l'attachment
            include_actions: Inclure les boutons d'action
            
        Returns:
            Dict: Attachment Slack
        """
        try:
            attachment = {
                "color": color,
                "title": f"🚨 {alert_data.get('level', 'ALERT')} Alert",
                "text": alert_data.get('message', 'No message provided'),
                "fields": [],
                "footer": "Spotify AI Agent Alert System",
                "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
                "ts": int(time.time())
            }
            
            # Champs d'information
            fields = [
                {
                    "title": "Service",
                    "value": alert_data.get('service_name', 'Unknown'),
                    "short": True
                },
                {
                    "title": "Environment",
                    "value": alert_data.get('environment', 'Unknown'),
                    "short": True
                },
                {
                    "title": "Severity Score",
                    "value": f"{alert_data.get('severity_score', 0):.2f}",
                    "short": True
                },
                {
                    "title": "Tenant",
                    "value": alert_data.get('tenant_id', 'Unknown'),
                    "short": True
                }
            ]
            
            # Ajout des champs contextuels
            context = alert_data.get('context', {})
            for key, value in context.items():
                if len(fields) < 10:  # Limite Slack
                    fields.append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value)[:100],  # Limitation de longueur
                        "short": True
                    })
            
            attachment["fields"] = fields
            
            # Actions interactives
            if include_actions:
                attachment["actions"] = [
                    {
                        "type": "button",
                        "text": "Acknowledge",
                        "style": "primary",
                        "value": f"ack_{alert_data.get('id', 'unknown')}"
                    },
                    {
                        "type": "button",
                        "text": "Resolve",
                        "style": "good",
                        "value": f"resolve_{alert_data.get('id', 'unknown')}"
                    },
                    {
                        "type": "button",
                        "text": "Escalate",
                        "style": "danger",
                        "value": f"escalate_{alert_data.get('id', 'unknown')}"
                    }
                ]
            
            return attachment
            
        except Exception as e:
            self.logger.error(f"Error building Slack attachment: {str(e)}")
            return {
                "color": "#ff0000",
                "title": "Error Building Alert",
                "text": "An error occurred while building the alert attachment"
            }
    
    def build_email_html(self, alert_data: Dict[str, Any]) -> str:
        """
        Construit un email HTML enrichi
        
        Args:
            alert_data: Données de l'alerte
            
        Returns:
            str: HTML de l'email
        """
        try:
            level = alert_data.get('level', 'ALERT')
            message = alert_data.get('message', 'No message provided')
            service = alert_data.get('service_name', 'Unknown')
            environment = alert_data.get('environment', 'Unknown')
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            
            # Couleur basée sur le niveau
            level_colors = {
                'CRITICAL': '#ff0000',
                'HIGH': '#ff8c00',
                'WARNING': '#ffd700',
                'INFO': '#00ced1',
                'DEBUG': '#808080'
            }
            color = level_colors.get(level, '#ff0000')
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Spotify AI Agent Alert</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .field {{ margin-bottom: 15px; padding: 10px; background-color: #f9f9f9; border-radius: 4px; }}
                    .field-label {{ font-weight: bold; color: #333; }}
                    .field-value {{ color: #666; margin-top: 5px; }}
                    .footer {{ background-color: #333; color: white; padding: 15px; text-align: center; font-size: 12px; }}
                    .button {{ display: inline-block; padding: 10px 20px; background-color: {color}; color: white; text-decoration: none; border-radius: 4px; margin: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚨 {level} Alert</h1>
                        <p>{message}</p>
                    </div>
                    <div class="content">
                        <div class="field">
                            <div class="field-label">Service</div>
                            <div class="field-value">{service}</div>
                        </div>
                        <div class="field">
                            <div class="field-label">Environment</div>
                            <div class="field-value">{environment}</div>
                        </div>
                        <div class="field">
                            <div class="field-label">Timestamp</div>
                            <div class="field-value">{timestamp}</div>
                        </div>
            """
            
            # Ajout des champs contextuels
            context = alert_data.get('context', {})
            for key, value in context.items():
                html += f"""
                        <div class="field">
                            <div class="field-label">{key.replace('_', ' ').title()}</div>
                            <div class="field-value">{str(value)}</div>
                        </div>
                """
            
            html += """
                        <div style="text-align: center; margin-top: 30px;">
                            <a href="#" class="button">View in Dashboard</a>
                            <a href="#" class="button">Acknowledge</a>
                        </div>
                    </div>
                    <div class="footer">
                        <p>Spotify AI Agent Alert System</p>
                        <p>This is an automated message. Please do not reply to this email.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error building email HTML: {str(e)}")
            return "<html><body><h1>Error building alert email</h1></body></html>"


class CacheUtils:
    """Utilitaires pour la gestion du cache Redis"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger("cache_utils")
    
    async def get_or_set(
        self,
        key: str,
        fetch_func,
        ttl: int = 3600,
        *args,
        **kwargs
    ) -> Any:
        """
        Récupère une valeur du cache ou l'exécute et la met en cache
        
        Args:
            key: Clé de cache
            fetch_func: Fonction à exécuter si pas en cache
            ttl: Durée de vie en secondes
            *args, **kwargs: Arguments pour fetch_func
            
        Returns:
            Any: Valeur du cache ou calculée
        """
        try:
            # Tentative de récupération du cache
            cached_value = await self.redis_client.get(key)
            
            if cached_value:
                return json.loads(cached_value)
            
            # Exécution de la fonction
            if asyncio.iscoroutinefunction(fetch_func):
                result = await fetch_func(*args, **kwargs)
            else:
                result = fetch_func(*args, **kwargs)
            
            # Mise en cache
            await self.redis_client.setex(
                key,
                ttl,
                json.dumps(result, default=str)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in get_or_set for key {key}: {str(e)}")
            # Fallback : exécution directe
            if asyncio.iscoroutinefunction(fetch_func):
                return await fetch_func(*args, **kwargs)
            else:
                return fetch_func(*args, **kwargs)
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalide toutes les clés correspondant à un pattern
        
        Args:
            pattern: Pattern de clés (avec wildcards)
            
        Returns:
            int: Nombre de clés supprimées
        """
        try:
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.logger.info(f"Invalidated {deleted} cache keys with pattern {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error invalidating cache pattern {pattern}: {str(e)}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache
        
        Returns:
            Dict: Statistiques du cache
        """
        try:
            info = await self.redis_client.info()
            
            return {
                'used_memory': info.get('used_memory_human', 'Unknown'),
                'used_memory_peak': info.get('used_memory_peak_human', 'Unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calcule le taux de hit du cache"""
        total = hits + misses
        return (hits / total) if total > 0 else 0.0


# Fonctions utilitaires globales

def generate_correlation_id() -> str:
    """Génère un ID de corrélation unique"""
    return f"corr_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def format_duration(seconds: float) -> str:
    """Formate une durée en secondes en format lisible"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Tronque un texte intelligemment"""
    if len(text) <= max_length:
        return text
    
    # Coupe au dernier espace avant la limite
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return truncated + suffix


def validate_json(json_string: str) -> Tuple[bool, Optional[str]]:
    """Valide une chaîne JSON"""
    try:
        json.loads(json_string)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)


def mask_sensitive_data(data: str, patterns: List[str] = None) -> str:
    """Masque les données sensibles dans un texte"""
    if not patterns:
        patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Numéros de carte
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\bkey[_-]?[a-zA-Z0-9]{10,}\b',  # API keys
            r'\btoken[_-]?[a-zA-Z0-9]{10,}\b'  # Tokens
        ]
    
    masked_data = data
    for pattern in patterns:
        masked_data = re.sub(pattern, '[MASKED]', masked_data, flags=re.IGNORECASE)
    
    return masked_data
