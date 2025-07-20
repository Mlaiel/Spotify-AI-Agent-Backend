#!/usr/bin/env python3
"""
Health Checker for PagerDuty Integration

Système avancé de vérification de santé pour les intégrations PagerDuty.
Effectue des contrôles complets de connectivité, performance, configuration,
et fonctionnalités avec reporting détaillé et alertes automatiques.

Fonctionnalités:
- Health checks multi-niveaux (basic, full, deep)
- Vérification de connectivité API
- Tests de performance et latence
- Validation de configuration en temps réel
- Surveillance des quotas et limites
- Tests d'intégration end-to-end
- Reporting automatisé
- Alertes proactives

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aioredis
import yaml
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich import print as rprint

console = Console()
logger = structlog.get_logger(__name__)

class HealthStatus(Enum):
    """Statuts de santé possibles"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class CheckLevel(Enum):
    """Niveaux de vérification"""
    BASIC = "basic"
    FULL = "full"
    DEEP = "deep"

@dataclass
class HealthCheck:
    """Résultat d'une vérification de santé"""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class HealthReport:
    """Rapport de santé complet"""
    overall_status: HealthStatus
    timestamp: datetime
    duration_ms: float
    checks: List[HealthCheck]
    summary: Dict[str, int]
    recommendations: List[str]

class PagerDutyAPIChecker:
    """Vérificateur d'API PagerDuty"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.pagerduty.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Token token={self.api_key}",
                "Accept": "application/vnd.pagerduty+json;version=2",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_api_connectivity(self) -> HealthCheck:
        """Vérifie la connectivité de base à l'API"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/abilities") as response:
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    abilities = data.get("abilities", [])
                    
                    return HealthCheck(
                        name="API Connectivity",
                        status=HealthStatus.HEALTHY,
                        message="API accessible and responding",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        details={"abilities_count": len(abilities), "response_time_ms": duration_ms}
                    )
                else:
                    return HealthCheck(
                        name="API Connectivity",
                        status=HealthStatus.CRITICAL,
                        message=f"API returned status {response.status}",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="API Connectivity",
                status=HealthStatus.CRITICAL,
                message="Failed to connect to API",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )
    
    async def check_authentication(self) -> HealthCheck:
        """Vérifie l'authentification"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/users/me") as response:
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    user = data.get("user", {})
                    
                    return HealthCheck(
                        name="Authentication",
                        status=HealthStatus.HEALTHY,
                        message="Authentication successful",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        details={
                            "user_id": user.get("id"),
                            "user_name": user.get("name"),
                            "user_email": user.get("email"),
                            "role": user.get("role")
                        }
                    )
                elif response.status == 401:
                    return HealthCheck(
                        name="Authentication",
                        status=HealthStatus.CRITICAL,
                        message="Authentication failed - invalid API key",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        error="Invalid API key"
                    )
                else:
                    return HealthCheck(
                        name="Authentication",
                        status=HealthStatus.WARNING,
                        message=f"Unexpected response status {response.status}",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="Authentication",
                status=HealthStatus.CRITICAL,
                message="Authentication check failed",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )
    
    async def check_rate_limits(self) -> HealthCheck:
        """Vérifie les limites de taux d'appels"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/users?limit=1") as response:
                duration_ms = (time.time() - start_time) * 1000
                
                headers = response.headers
                rate_limit = headers.get("X-Rate-Limit-Limit")
                rate_remaining = headers.get("X-Rate-Limit-Remaining")
                rate_reset = headers.get("X-Rate-Limit-Reset")
                
                if rate_limit and rate_remaining:
                    remaining_percent = (int(rate_remaining) / int(rate_limit)) * 100
                    
                    if remaining_percent > 20:
                        status = HealthStatus.HEALTHY
                        message = f"Rate limit OK ({remaining_percent:.1f}% remaining)"
                    elif remaining_percent > 5:
                        status = HealthStatus.WARNING
                        message = f"Rate limit warning ({remaining_percent:.1f}% remaining)"
                    else:
                        status = HealthStatus.CRITICAL
                        message = f"Rate limit critical ({remaining_percent:.1f}% remaining)"
                    
                    return HealthCheck(
                        name="Rate Limits",
                        status=status,
                        message=message,
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        details={
                            "limit": rate_limit,
                            "remaining": rate_remaining,
                            "reset_time": rate_reset,
                            "remaining_percent": remaining_percent
                        }
                    )
                else:
                    return HealthCheck(
                        name="Rate Limits",
                        status=HealthStatus.WARNING,
                        message="Rate limit headers not found",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="Rate Limits",
                status=HealthStatus.WARNING,
                message="Could not check rate limits",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )
    
    async def check_services(self) -> HealthCheck:
        """Vérifie les services configurés"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/services") as response:
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    services = data.get("services", [])
                    
                    active_services = [s for s in services if s.get("status") == "active"]
                    disabled_services = [s for s in services if s.get("status") == "disabled"]
                    
                    if len(active_services) > 0:
                        status = HealthStatus.HEALTHY
                        message = f"{len(active_services)} active services found"
                    else:
                        status = HealthStatus.WARNING
                        message = "No active services found"
                    
                    return HealthCheck(
                        name="Services",
                        status=status,
                        message=message,
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        details={
                            "total_services": len(services),
                            "active_services": len(active_services),
                            "disabled_services": len(disabled_services)
                        }
                    )
                else:
                    return HealthCheck(
                        name="Services",
                        status=HealthStatus.WARNING,
                        message=f"Could not retrieve services (HTTP {response.status})",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="Services",
                status=HealthStatus.WARNING,
                message="Failed to check services",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )
    
    async def check_escalation_policies(self) -> HealthCheck:
        """Vérifie les politiques d'escalade"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/escalation_policies") as response:
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    policies = data.get("escalation_policies", [])
                    
                    valid_policies = []
                    invalid_policies = []
                    
                    for policy in policies:
                        rules = policy.get("escalation_rules", [])
                        if len(rules) > 0:
                            valid_policies.append(policy)
                        else:
                            invalid_policies.append(policy)
                    
                    if len(valid_policies) > 0:
                        status = HealthStatus.HEALTHY
                        message = f"{len(valid_policies)} escalation policies configured"
                    else:
                        status = HealthStatus.WARNING
                        message = "No valid escalation policies found"
                    
                    return HealthCheck(
                        name="Escalation Policies",
                        status=status,
                        message=message,
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        details={
                            "total_policies": len(policies),
                            "valid_policies": len(valid_policies),
                            "invalid_policies": len(invalid_policies)
                        }
                    )
                else:
                    return HealthCheck(
                        name="Escalation Policies",
                        status=HealthStatus.WARNING,
                        message=f"Could not retrieve escalation policies (HTTP {response.status})",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="Escalation Policies",
                status=HealthStatus.WARNING,
                message="Failed to check escalation policies",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )

class IntegrationChecker:
    """Vérificateur d'intégration end-to-end"""
    
    def __init__(self, integration_key: str, api_checker: PagerDutyAPIChecker):
        self.integration_key = integration_key
        self.api_checker = api_checker
    
    async def check_event_sending(self) -> HealthCheck:
        """Test d'envoi d'événement de test"""
        start_time = time.time()
        
        try:
            event_data = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": f"health-check-{int(time.time())}",
                "payload": {
                    "summary": "Health Check Test Event",
                    "source": "health-checker",
                    "severity": "info",
                    "custom_details": {
                        "test": True,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            }
            
            async with self.api_checker.session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=event_data
            ) as response:
                duration_ms = (time.time() - start_time) * 1000
                
                if response.status == 202:
                    data = await response.json()
                    dedup_key = data.get("dedup_key")
                    
                    # Résoudre immédiatement l'incident de test
                    resolve_data = {
                        "routing_key": self.integration_key,
                        "event_action": "resolve",
                        "dedup_key": dedup_key
                    }
                    
                    async with self.api_checker.session.post(
                        "https://events.pagerduty.com/v2/enqueue",
                        json=resolve_data
                    ) as resolve_response:
                        if resolve_response.status == 202:
                            return HealthCheck(
                                name="Event Sending",
                                status=HealthStatus.HEALTHY,
                                message="Test event sent and resolved successfully",
                                duration_ms=duration_ms,
                                timestamp=datetime.now(timezone.utc),
                                details={
                                    "dedup_key": dedup_key,
                                    "event_action": "trigger_and_resolve"
                                }
                            )
                        else:
                            return HealthCheck(
                                name="Event Sending",
                                status=HealthStatus.WARNING,
                                message="Event sent but could not resolve",
                                duration_ms=duration_ms,
                                timestamp=datetime.now(timezone.utc),
                                details={"dedup_key": dedup_key}
                            )
                else:
                    return HealthCheck(
                        name="Event Sending",
                        status=HealthStatus.CRITICAL,
                        message=f"Failed to send test event (HTTP {response.status})",
                        duration_ms=duration_ms,
                        timestamp=datetime.now(timezone.utc),
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="Event Sending",
                status=HealthStatus.CRITICAL,
                message="Failed to send test event",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )

class PerformanceChecker:
    """Vérificateur de performance"""
    
    def __init__(self, api_checker: PagerDutyAPIChecker):
        self.api_checker = api_checker
    
    async def check_response_times(self, iterations: int = 5) -> HealthCheck:
        """Mesure les temps de réponse de l'API"""
        start_time = time.time()
        response_times = []
        
        try:
            for _ in range(iterations):
                iteration_start = time.time()
                async with self.api_checker.session.get(f"{self.api_checker.base_url}/abilities") as response:
                    if response.status == 200:
                        response_times.append((time.time() - iteration_start) * 1000)
                    await asyncio.sleep(0.1)  # Petit délai entre les appels
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                min_time = min(response_times)
                max_time = max(response_times)
                
                if avg_time < 500:
                    status = HealthStatus.HEALTHY
                    message = f"Response times OK (avg: {avg_time:.1f}ms)"
                elif avg_time < 1000:
                    status = HealthStatus.WARNING
                    message = f"Response times slow (avg: {avg_time:.1f}ms)"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"Response times critical (avg: {avg_time:.1f}ms)"
                
                duration_ms = (time.time() - start_time) * 1000
                
                return HealthCheck(
                    name="Response Times",
                    status=status,
                    message=message,
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "iterations": iterations,
                        "avg_time_ms": avg_time,
                        "min_time_ms": min_time,
                        "max_time_ms": max_time,
                        "all_times": response_times
                    }
                )
            else:
                duration_ms = (time.time() - start_time) * 1000
                return HealthCheck(
                    name="Response Times",
                    status=HealthStatus.CRITICAL,
                    message="No successful responses",
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc),
                    error="All requests failed"
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="Response Times",
                status=HealthStatus.CRITICAL,
                message="Performance check failed",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )

class RedisChecker:
    """Vérificateur Redis pour la mise en cache"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
    
    async def check_connectivity(self) -> HealthCheck:
        """Vérifie la connectivité Redis"""
        start_time = time.time()
        
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Test de lecture/écriture
            test_key = f"health_check_{int(time.time())}"
            test_value = "test_value"
            
            await self.redis.set(test_key, test_value, ex=60)
            retrieved_value = await self.redis.get(test_key)
            await self.redis.delete(test_key)
            
            if retrieved_value and retrieved_value.decode() == test_value:
                return HealthCheck(
                    name="Redis Connectivity",
                    status=HealthStatus.HEALTHY,
                    message="Redis connected and functional",
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc),
                    details={"ping_successful": True, "read_write_test": True}
                )
            else:
                return HealthCheck(
                    name="Redis Connectivity",
                    status=HealthStatus.WARNING,
                    message="Redis connected but read/write test failed",
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc),
                    error="Read/write test failed"
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name="Redis Connectivity",
                status=HealthStatus.CRITICAL,
                message="Redis connection failed",
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            )
        finally:
            if self.redis:
                await self.redis.close()

class PagerDutyHealthChecker:
    """Vérificateur de santé principal pour PagerDuty"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.integration_key = config.get("integration_key")
        self.redis_url = config.get("redis_url", "redis://localhost:6379")
        
    async def run_checks(self, level: CheckLevel = CheckLevel.BASIC) -> HealthReport:
        """Lance les vérifications de santé selon le niveau spécifié"""
        start_time = time.time()
        checks = []
        recommendations = []
        
        try:
            async with PagerDutyAPIChecker(self.api_key) as api_checker:
                # Vérifications de base
                checks.append(await api_checker.check_api_connectivity())
                checks.append(await api_checker.check_authentication())
                
                if level in [CheckLevel.FULL, CheckLevel.DEEP]:
                    # Vérifications complètes
                    checks.append(await api_checker.check_rate_limits())
                    checks.append(await api_checker.check_services())
                    checks.append(await api_checker.check_escalation_policies())
                    
                    # Vérifications Redis
                    redis_checker = RedisChecker(self.redis_url)
                    checks.append(await redis_checker.check_connectivity())
                    
                    if level == CheckLevel.DEEP:
                        # Vérifications approfondies
                        performance_checker = PerformanceChecker(api_checker)
                        checks.append(await performance_checker.check_response_times())
                        
                        if self.integration_key:
                            integration_checker = IntegrationChecker(self.integration_key, api_checker)
                            checks.append(await integration_checker.check_event_sending())
        
        except Exception as e:
            checks.append(HealthCheck(
                name="System Error",
                status=HealthStatus.CRITICAL,
                message="Failed to complete health checks",
                duration_ms=0,
                timestamp=datetime.now(timezone.utc),
                error=str(e)
            ))
        
        # Analyser les résultats
        duration_ms = (time.time() - start_time) * 1000
        summary = self._analyze_checks(checks)
        overall_status = self._determine_overall_status(checks)
        recommendations = self._generate_recommendations(checks)
        
        return HealthReport(
            overall_status=overall_status,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            checks=checks,
            summary=summary,
            recommendations=recommendations
        )
    
    def _analyze_checks(self, checks: List[HealthCheck]) -> Dict[str, int]:
        """Analyse les résultats des vérifications"""
        summary = {
            "total": len(checks),
            "healthy": 0,
            "warning": 0,
            "critical": 0,
            "unknown": 0
        }
        
        for check in checks:
            summary[check.status.value] += 1
        
        return summary
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Détermine le statut global"""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        elif all(check.status == HealthStatus.HEALTHY for check in checks):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _generate_recommendations(self, checks: List[HealthCheck]) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        recommendations = []
        
        for check in checks:
            if check.status == HealthStatus.CRITICAL:
                if "API Connectivity" in check.name:
                    recommendations.append("Vérifier la connectivité réseau vers PagerDuty")
                elif "Authentication" in check.name:
                    recommendations.append("Vérifier la validité de la clé API PagerDuty")
                elif "Redis" in check.name:
                    recommendations.append("Vérifier la connectivité et configuration Redis")
                elif "Event Sending" in check.name:
                    recommendations.append("Vérifier la clé d'intégration PagerDuty")
            
            elif check.status == HealthStatus.WARNING:
                if "Rate Limits" in check.name:
                    recommendations.append("Réduire la fréquence des appels API ou demander une augmentation de quota")
                elif "Response Times" in check.name:
                    recommendations.append("Optimiser les performances ou vérifier la charge système")
                elif "Services" in check.name:
                    recommendations.append("Configurer des services actifs dans PagerDuty")
        
        # Recommandations générales
        if not any("Performance" in check.name for check in checks):
            recommendations.append("Considérer l'ajout de tests de performance réguliers")
        
        return list(set(recommendations))  # Supprimer les doublons

def format_health_report(report: HealthReport) -> None:
    """Formate et affiche le rapport de santé"""
    
    # Statut global
    status_color = {
        HealthStatus.HEALTHY: "green",
        HealthStatus.WARNING: "yellow", 
        HealthStatus.CRITICAL: "red",
        HealthStatus.UNKNOWN: "white"
    }
    
    console.print(Panel.fit(
        f"Status: [{status_color[report.overall_status]}]{report.overall_status.value.upper()}[/]\n"
        f"Duration: {report.duration_ms:.1f}ms\n"
        f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        title="🏥 PagerDuty Health Check",
        border_style=status_color[report.overall_status]
    ))
    
    # Résumé
    summary_table = Table(title="Summary")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Percentage", justify="right")
    
    total = report.summary["total"]
    for status in ["healthy", "warning", "critical", "unknown"]:
        count = report.summary[status]
        percentage = (count / total * 100) if total > 0 else 0
        summary_table.add_row(
            status.title(),
            str(count),
            f"{percentage:.1f}%",
            style=status_color.get(HealthStatus(status), "white")
        )
    
    console.print(summary_table)
    
    # Détails des vérifications
    checks_table = Table(title="Health Checks Details")
    checks_table.add_column("Check", style="bold")
    checks_table.add_column("Status")
    checks_table.add_column("Message")
    checks_table.add_column("Duration", justify="right")
    
    for check in report.checks:
        checks_table.add_row(
            check.name,
            check.status.value.upper(),
            check.message,
            f"{check.duration_ms:.1f}ms",
            style=status_color[check.status]
        )
    
    console.print(checks_table)
    
    # Erreurs détaillées
    errors = [check for check in report.checks if check.error]
    if errors:
        error_tree = Tree("🚨 Errors")
        for check in errors:
            error_node = error_tree.add(f"{check.name}: {check.message}")
            error_node.add(f"Error: {check.error}")
        console.print(error_tree)
    
    # Recommandations
    if report.recommendations:
        recommendations_panel = Panel(
            "\n".join(f"• {rec}" for rec in report.recommendations),
            title="💡 Recommendations",
            border_style="blue"
        )
        console.print(recommendations_panel)

async def main():
    """Fonction principale CLI"""
    parser = argparse.ArgumentParser(description="PagerDuty Health Checker")
    parser.add_argument("--config-file", required=True, help="Fichier de configuration")
    parser.add_argument("--level", choices=["basic", "full", "deep"], default="basic",
                       help="Niveau de vérification")
    parser.add_argument("--output", choices=["console", "json", "yaml"], default="console",
                       help="Format de sortie")
    parser.add_argument("--output-file", help="Fichier de sortie pour JSON/YAML")
    parser.add_argument("--watch", type=int, help="Mode surveillance (intervalle en secondes)")
    
    args = parser.parse_args()
    
    # Charger la configuration
    try:
        with open(args.config_file, 'r') as f:
            if args.config_file.endswith('.json'):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        return 1
    
    checker = PagerDutyHealthChecker(config)
    level = CheckLevel(args.level)
    
    async def run_single_check():
        report = await checker.run_checks(level)
        
        if args.output == "console":
            format_health_report(report)
        elif args.output in ["json", "yaml"]:
            data = asdict(report)
            # Convertir les enums et datetime en strings
            def convert_for_serialization(obj):
                if isinstance(obj, dict):
                    return {k: convert_for_serialization(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_serialization(item) for item in obj]
                elif isinstance(obj, (HealthStatus, CheckLevel)):
                    return obj.value
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                else:
                    return obj
            
            data = convert_for_serialization(data)
            
            if args.output == "json":
                output = json.dumps(data, indent=2)
            else:
                output = yaml.dump(data, default_flow_style=False)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(output)
                console.print(f"Report saved to {args.output_file}")
            else:
                console.print(output)
        
        return report.overall_status
    
    if args.watch:
        # Mode surveillance
        console.print(f"Starting health monitoring (checking every {args.watch} seconds)")
        console.print("Press Ctrl+C to stop")
        
        try:
            while True:
                status = await run_single_check()
                console.print(f"Next check in {args.watch} seconds...")
                await asyncio.sleep(args.watch)
        except KeyboardInterrupt:
            console.print("\nMonitoring stopped")
            return 0
    else:
        # Vérification unique
        status = await run_single_check()
        return 0 if status == HealthStatus.HEALTHY else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
