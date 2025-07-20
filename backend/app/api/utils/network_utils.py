"""
🎵 Spotify AI Agent - Network Utilities
=======================================

Utilitaires enterprise pour les communications réseau
avec gestion avancée des requêtes HTTP, WebSockets et API.

Architecture:
- Client HTTP asynchrone sécurisé
- Gestion des timeouts et retry
- Validation d'URLs et domaines
- Rate limiting distribué
- Health checks automatiques
- Monitoring de connectivité

🎖️ Développé par l'équipe d'experts enterprise
"""

import asyncio
import aiohttp
import socket
import ssl
import dns.resolver
import validators
import ipaddress
from typing import Optional, Dict, Any, List, Union, Tuple
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
import time
import json
import logging
from contextlib import asynccontextmanager


# =============================================================================
# CONFIGURATION RÉSEAU
# =============================================================================

@dataclass
class NetworkConfig:
    """Configuration réseau enterprise"""
    
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_redirects: int = 10
    user_agent: str = "Spotify-AI-Agent/1.0"
    verify_ssl: bool = True
    max_connections: int = 100
    max_connections_per_host: int = 30


@dataclass
class RequestMetrics:
    """Métriques de requête réseau"""
    
    url: str
    method: str
    status_code: Optional[int] = None
    response_time: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    dns_time: float = 0.0
    connect_time: float = 0.0
    ssl_time: float = 0.0
    success: bool = False
    error: Optional[str] = None
    timestamp: float = 0.0


# =============================================================================
# CLIENT HTTP ENTERPRISE
# =============================================================================

class EnterpriseHttpClient:
    """Client HTTP enterprise avec fonctionnalités avancées"""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics: List[RequestMetrics] = []
        self._rate_limiter: Dict[str, List[float]] = {}
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def start(self) -> None:
        """Initialise le client HTTP"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_connections_per_host,
                verify_ssl=self.config.verify_ssl
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': self.config.user_agent}
            )
    
    async def close(self) -> None:
        """Ferme le client HTTP"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def request(self, method: str, url: str, 
                     retry_count: int = 0, **kwargs) -> aiohttp.ClientResponse:
        """
        Effectue une requête HTTP avec retry automatique
        
        Args:
            method: Méthode HTTP
            url: URL de destination
            retry_count: Nombre de tentatives effectuées
            **kwargs: Arguments supplémentaires
            
        Returns:
            Réponse HTTP
        """
        if not self.session:
            await self.start()
        
        # Vérifier le rate limiting
        if not self._check_rate_limit(url):
            raise Exception(f"Rate limit dépassé pour {url}")
        
        # Métriques de début
        start_time = time.perf_counter()
        metrics = RequestMetrics(
            url=url,
            method=method.upper(),
            timestamp=time.time()
        )
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                # Métriques de fin
                end_time = time.perf_counter()
                metrics.response_time = end_time - start_time
                metrics.status_code = response.status
                metrics.bytes_received = len(await response.read())
                metrics.success = 200 <= response.status < 400
                
                self.metrics.append(metrics)
                
                # Retry si nécessaire
                if (response.status >= 500 and 
                    retry_count < self.config.max_retries):
                    await asyncio.sleep(self.config.retry_delay * (2 ** retry_count))
                    return await self.request(method, url, retry_count + 1, **kwargs)
                
                return response
        
        except Exception as e:
            metrics.error = str(e)
            metrics.response_time = time.perf_counter() - start_time
            self.metrics.append(metrics)
            
            # Retry en cas d'erreur
            if retry_count < self.config.max_retries:
                await asyncio.sleep(self.config.retry_delay * (2 ** retry_count))
                return await self.request(method, url, retry_count + 1, **kwargs)
            
            raise
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request"""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request"""
        return await self.request('POST', url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """PUT request"""
        return await self.request('PUT', url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """DELETE request"""
        return await self.request('DELETE', url, **kwargs)
    
    async def get_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        GET request avec parsing JSON automatique
        
        Args:
            url: URL de l'API
            **kwargs: Arguments supplémentaires
            
        Returns:
            Données JSON parsées
        """
        async with await self.get(url, **kwargs) as response:
            if response.content_type != 'application/json':
                raise ValueError(f"Réponse non-JSON: {response.content_type}")
            return await response.json()
    
    async def post_json(self, url: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        POST request avec données JSON
        
        Args:
            url: URL de l'API
            data: Données à envoyer
            **kwargs: Arguments supplémentaires
            
        Returns:
            Réponse JSON
        """
        kwargs['json'] = data
        async with await self.post(url, **kwargs) as response:
            return await response.json()
    
    def _check_rate_limit(self, url: str, max_requests: int = 100, 
                         window_seconds: int = 60) -> bool:
        """
        Vérifie le rate limiting pour une URL
        
        Args:
            url: URL à vérifier
            max_requests: Nombre maximum de requêtes
            window_seconds: Fenêtre temporelle en secondes
            
        Returns:
            True si autorisé
        """
        domain = urlparse(url).netloc
        current_time = time.time()
        
        if domain not in self._rate_limiter:
            self._rate_limiter[domain] = []
        
        # Nettoyer les anciennes entrées
        cutoff_time = current_time - window_seconds
        self._rate_limiter[domain] = [
            timestamp for timestamp in self._rate_limiter[domain]
            if timestamp > cutoff_time
        ]
        
        # Vérifier la limite
        if len(self._rate_limiter[domain]) >= max_requests:
            return False
        
        # Ajouter la nouvelle requête
        self._rate_limiter[domain].append(current_time)
        return True
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtient un résumé des métriques
        
        Returns:
            Résumé des métriques réseau
        """
        if not self.metrics:
            return {}
        
        total_requests = len(self.metrics)
        successful_requests = sum(1 for m in self.metrics if m.success)
        
        response_times = [m.response_time for m in self.metrics]
        avg_response_time = sum(response_times) / len(response_times)
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': (successful_requests / total_requests) * 100,
            'avg_response_time': avg_response_time,
            'min_response_time': min(response_times),
            'max_response_time': max(response_times)
        }


# =============================================================================
# VALIDATION RÉSEAU
# =============================================================================

def is_valid_url(url: str, schemes: Optional[List[str]] = None) -> bool:
    """
    Valide une URL
    
    Args:
        url: URL à valider
        schemes: Schémas autorisés (http, https par défaut)
        
    Returns:
        True si URL valide
    """
    if schemes is None:
        schemes = ['http', 'https']
    
    try:
        parsed = urlparse(url)
        return (parsed.scheme in schemes and 
                bool(parsed.netloc) and 
                validators.url(url))
    except Exception:
        return False


def is_valid_domain(domain: str) -> bool:
    """
    Valide un nom de domaine
    
    Args:
        domain: Domaine à valider
        
    Returns:
        True si domaine valide
    """
    try:
        return validators.domain(domain)
    except Exception:
        return False


def is_valid_ip(ip: str, version: Optional[int] = None) -> bool:
    """
    Valide une adresse IP
    
    Args:
        ip: Adresse IP à valider
        version: Version IP (4 ou 6, auto-détection si None)
        
    Returns:
        True si IP valide
    """
    try:
        if version == 4:
            ipaddress.IPv4Address(ip)
        elif version == 6:
            ipaddress.IPv6Address(ip)
        else:
            ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def is_private_ip(ip: str) -> bool:
    """
    Vérifie si une IP est privée
    
    Args:
        ip: Adresse IP
        
    Returns:
        True si IP privée
    """
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private
    except ValueError:
        return False


def extract_domain(url: str) -> Optional[str]:
    """
    Extrait le domaine d'une URL
    
    Args:
        url: URL source
        
    Returns:
        Nom de domaine ou None
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None


def normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """
    Normalise une URL
    
    Args:
        url: URL à normaliser
        base_url: URL de base pour les URLs relatives
        
    Returns:
        URL normalisée
    """
    # Joindre avec base_url si nécessaire
    if base_url and not urlparse(url).scheme:
        url = urljoin(base_url, url)
    
    # Parser et reconstruire pour normaliser
    parsed = urlparse(url)
    
    # Normaliser le chemin
    path = parsed.path
    if not path:
        path = '/'
    
    # Reconstruire l'URL
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    
    if parsed.query:
        normalized += f"?{parsed.query}"
    
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"
    
    return normalized


# =============================================================================
# RÉSOLUTION DNS
# =============================================================================

def resolve_dns(domain: str, record_type: str = 'A') -> List[str]:
    """
    Résout un enregistrement DNS
    
    Args:
        domain: Domaine à résoudre
        record_type: Type d'enregistrement (A, AAAA, MX, etc.)
        
    Returns:
        Liste des enregistrements
    """
    try:
        answers = dns.resolver.resolve(domain, record_type)
        return [str(answer) for answer in answers]
    except Exception:
        return []


def get_mx_records(domain: str) -> List[Tuple[int, str]]:
    """
    Obtient les enregistrements MX d'un domaine
    
    Args:
        domain: Domaine à vérifier
        
    Returns:
        Liste des (priorité, serveur_mail)
    """
    try:
        answers = dns.resolver.resolve(domain, 'MX')
        return [(answer.preference, str(answer.exchange)) for answer in answers]
    except Exception:
        return []


def check_domain_exists(domain: str) -> bool:
    """
    Vérifie si un domaine existe
    
    Args:
        domain: Domaine à vérifier
        
    Returns:
        True si le domaine existe
    """
    try:
        dns.resolver.resolve(domain, 'A')
        return True
    except Exception:
        try:
            dns.resolver.resolve(domain, 'AAAA')
            return True
        except Exception:
            return False


# =============================================================================
# HEALTH CHECKS
# =============================================================================

async def check_http_health(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Vérifie la santé d'un endpoint HTTP
    
    Args:
        url: URL à vérifier
        timeout: Timeout en secondes
        
    Returns:
        Rapport de santé
    """
    start_time = time.perf_counter()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as response:
                end_time = time.perf_counter()
                
                return {
                    'url': url,
                    'status': 'healthy' if 200 <= response.status < 400 else 'unhealthy',
                    'status_code': response.status,
                    'response_time': end_time - start_time,
                    'timestamp': time.time(),
                    'error': None
                }
    
    except Exception as e:
        end_time = time.perf_counter()
        return {
            'url': url,
            'status': 'unhealthy',
            'status_code': None,
            'response_time': end_time - start_time,
            'timestamp': time.time(),
            'error': str(e)
        }


def check_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    """
    Vérifie si un port est ouvert
    
    Args:
        host: Adresse de l'hôte
        port: Numéro de port
        timeout: Timeout en secondes
        
    Returns:
        True si le port est ouvert
    """
    try:
        sock = socket.create_connection((host, port), timeout)
        sock.close()
        return True
    except Exception:
        return False


def check_ssl_certificate(hostname: str, port: int = 443) -> Dict[str, Any]:
    """
    Vérifie un certificat SSL
    
    Args:
        hostname: Nom d'hôte
        port: Port SSL
        
    Returns:
        Informations sur le certificat
    """
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                
                return {
                    'hostname': hostname,
                    'valid': True,
                    'subject': dict(x[0] for x in cert['subject']),
                    'issuer': dict(x[0] for x in cert['issuer']),
                    'version': cert['version'],
                    'serialNumber': cert['serialNumber'],
                    'notBefore': cert['notBefore'],
                    'notAfter': cert['notAfter'],
                    'subjectAltName': cert.get('subjectAltName', [])
                }
    
    except Exception as e:
        return {
            'hostname': hostname,
            'valid': False,
            'error': str(e)
        }


# =============================================================================
# MONITORING DE CONNECTIVITÉ
# =============================================================================

class ConnectivityMonitor:
    """Moniteur de connectivité réseau"""
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.endpoints: List[str] = []
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def add_endpoint(self, url: str) -> None:
        """
        Ajoute un endpoint à monitorer
        
        Args:
            url: URL de l'endpoint
        """
        if url not in self.endpoints:
            self.endpoints.append(url)
            self.results[url] = []
    
    def remove_endpoint(self, url: str) -> None:
        """
        Supprime un endpoint du monitoring
        
        Args:
            url: URL à supprimer
        """
        if url in self.endpoints:
            self.endpoints.remove(url)
            if url in self.results:
                del self.results[url]
    
    async def start_monitoring(self) -> None:
        """Démarre le monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self) -> None:
        """Arrête le monitoring"""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self) -> None:
        """Boucle de monitoring"""
        while self.is_monitoring:
            try:
                # Vérifier tous les endpoints
                tasks = [check_http_health(url) for url in self.endpoints]
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for url, result in zip(self.endpoints, results):
                        if not isinstance(result, Exception):
                            self.results[url].append(result)
                            
                            # Limiter l'historique
                            if len(self.results[url]) > 1000:
                                self.results[url] = self.results[url][-500:]
                
                await asyncio.sleep(self.check_interval)
            
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    def get_endpoint_status(self, url: str) -> Dict[str, Any]:
        """
        Obtient le statut d'un endpoint
        
        Args:
            url: URL de l'endpoint
            
        Returns:
            Statut de l'endpoint
        """
        if url not in self.results or not self.results[url]:
            return {'status': 'unknown'}
        
        recent_results = self.results[url][-10:]  # 10 derniers résultats
        healthy_count = sum(1 for r in recent_results if r['status'] == 'healthy')
        
        return {
            'url': url,
            'current_status': recent_results[-1]['status'],
            'uptime_percentage': (healthy_count / len(recent_results)) * 100,
            'last_check': recent_results[-1]['timestamp'],
            'avg_response_time': sum(r['response_time'] for r in recent_results) / len(recent_results)
        }
    
    def get_overall_status(self) -> Dict[str, Any]:
        """
        Obtient le statut global de tous les endpoints
        
        Returns:
            Statut global
        """
        if not self.endpoints:
            return {'status': 'no_endpoints'}
        
        endpoint_statuses = [self.get_endpoint_status(url) for url in self.endpoints]
        healthy_endpoints = sum(1 for s in endpoint_statuses if s.get('current_status') == 'healthy')
        
        return {
            'total_endpoints': len(self.endpoints),
            'healthy_endpoints': healthy_endpoints,
            'overall_health': (healthy_endpoints / len(self.endpoints)) * 100,
            'endpoints': endpoint_statuses
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NetworkConfig",
    "RequestMetrics",
    "EnterpriseHttpClient",
    "is_valid_url",
    "is_valid_domain",
    "is_valid_ip",
    "is_private_ip",
    "extract_domain",
    "normalize_url",
    "resolve_dns",
    "get_mx_records",
    "check_domain_exists",
    "check_http_health",
    "check_port_open",
    "check_ssl_certificate",
    "ConnectivityMonitor"
]
