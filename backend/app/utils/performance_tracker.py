"""
Performance Tracker Utility - Spotify AI Agent
==============================================
Outil de tracking des performances pour les middlewares

Auteur: Équipe Lead Dev + Architecte IA
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Métriques de performance"""
    response_time: float
    cpu_usage: float
    memory_usage: float
    timestamp: datetime
    endpoint: Optional[str] = None
    status_code: Optional[int] = None


class PerformanceTracker:
    """Tracker de performance pour middleware"""
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._metrics: Dict[str, Any] = {}
    
    def start_tracking(self) -> None:
        """Démarre le tracking"""
        self._start_time = time.time()
    
    def stop_tracking(self) -> PerformanceMetrics:
        """Arrête le tracking et retourne les métriques"""
        end_time = time.time()
        response_time = end_time - (self._start_time or end_time)
        
        # Récupération des métriques système
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        return PerformanceMetrics(
            response_time=response_time,
            cpu_usage=cpu_percent,
            memory_usage=memory_info.percent,
            timestamp=datetime.utcnow()
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques actuelles"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': datetime.utcnow().isoformat()
        }
