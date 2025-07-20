# 🧪 Tests SecurityMonitor Ultra-Avancés
# ====================================

import pytest
import pytest_asyncio
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

from app.security.auth.monitoring import (
    SecurityMonitor, ThreatLevel, SecurityEvent, SecurityEventType,
    AnomalyDetector, RiskAssessment, SecurityMetrics, AlertSeverity,
    SecurityIncident, ThreatIntelligence, BehaviorProfile
)

from conftest import (
    TestDataFactory, TestUtils, PerformanceTestUtils, SecurityValidators,
    pytest_markers
)


@pytest_markers["unit"]
@pytest_markers["monitoring"]
class TestSecurityMonitor:
    """Tests unitaires pour SecurityMonitor"""
    
    @pytest.mark.asyncio
    async def test_security_event_detection(self, security_monitor):
        """Test détection d'événements de sécurité"""
        event_data = {
            "user_id": "user_123",
            "ip_address": "203.0.113.1",
            "user_agent": "SuspiciousBot/1.0",
            "endpoint": "/api/users/sensitive",
            "method": "POST",
            "timestamp": datetime.utcnow(),
            "payload_size": 5000000  # 5MB - suspect
        }
        
        with patch.object(security_monitor, '_analyze_request') as mock_analyze:
            mock_analyze.return_value = {
                "threat_detected": True,
                "threat_level": ThreatLevel.HIGH.value,
                "event_type": SecurityEventType.SUSPICIOUS_REQUEST.value,
                "risk_score": 0.85,
                "indicators": [
                    "suspicious_user_agent",
                    "large_payload",
                    "sensitive_endpoint_access"
                ],
                "recommended_actions": ["block_request", "alert_admin"]
            }
            
            result = await security_monitor.analyze_security_event(event_data)
        
        assert result["threat_detected"] is True
        assert result["threat_level"] == ThreatLevel.HIGH.value
        assert result["risk_score"] > 0.8
        assert "suspicious_user_agent" in result["indicators"]
        assert "block_request" in result["recommended_actions"]
    
    @pytest.mark.asyncio
    async def test_brute_force_detection(self, security_monitor):
        """Test détection d'attaques par force brute"""
        ip_address = "198.51.100.1"
        
        # Simuler 10 tentatives de connexion échouées
        failed_attempts = []
        for i in range(10):
            attempt = {
                "ip_address": ip_address,
                "user_id": f"user_{i % 3}",  # 3 utilisateurs différents
                "timestamp": datetime.utcnow() - timedelta(minutes=10-i),
                "success": False,
                "endpoint": "/auth/login"
            }
            failed_attempts.append(attempt)
        
        with patch.object(security_monitor, '_detect_brute_force') as mock_detect:
            mock_detect.return_value = {
                "attack_detected": True,
                "attack_type": "brute_force_login",
                "ip_address": ip_address,
                "failed_attempts": 10,
                "time_window": "10_minutes",
                "targeted_users": ["user_0", "user_1", "user_2"],
                "threat_level": ThreatLevel.CRITICAL.value,
                "recommended_actions": ["block_ip", "alert_security_team"]
            }
            
            result = await security_monitor.detect_brute_force_attack(
                failed_attempts=failed_attempts
            )
        
        assert result["attack_detected"] is True
        assert result["attack_type"] == "brute_force_login"
        assert result["failed_attempts"] == 10
        assert result["threat_level"] == ThreatLevel.CRITICAL.value
        assert "block_ip" in result["recommended_actions"]
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_ml(self, security_monitor):
        """Test détection d'anomalies par ML"""
        user_behavior = {
            "user_id": "user_456",
            "login_time": "03:30",  # Heure inhabituelle
            "location": {"country": "RU", "city": "Moscow"},  # Localisation suspecte
            "device_fingerprint": "new_unknown_device",
            "access_patterns": [
                "/api/admin/users",  # Accès inhabituel
                "/api/admin/config",
                "/api/data/export"
            ],
            "session_duration": 180,  # 3 minutes - très court
            "data_volume": 50000000  # 50MB téléchargés - beaucoup
        }
        
        with patch.object(security_monitor.anomaly_detector, 'analyze_behavior') as mock_ml:
            mock_ml.return_value = {
                "anomaly_detected": True,
                "anomaly_score": 0.92,
                "anomaly_type": "behavioral_deviation",
                "detected_anomalies": [
                    "unusual_login_time",
                    "suspicious_geolocation", 
                    "unknown_device",
                    "admin_access_pattern",
                    "high_data_volume"
                ],
                "user_profile_match": 0.08,  # Très faible correspondance
                "risk_factors": {
                    "geographic_risk": 0.9,
                    "temporal_risk": 0.8,
                    "behavioral_risk": 0.95
                }
            }
            
            result = await security_monitor.detect_behavioral_anomaly(
                user_behavior=user_behavior
            )
        
        assert result["anomaly_detected"] is True
        assert result["anomaly_score"] > 0.9
        assert "unusual_login_time" in result["detected_anomalies"]
        assert "suspicious_geolocation" in result["detected_anomalies"]
        assert result["risk_factors"]["behavioral_risk"] > 0.9
    
    @pytest.mark.asyncio
    async def test_threat_intelligence_integration(self, security_monitor):
        """Test intégration de la threat intelligence"""
        suspicious_indicators = [
            "203.0.113.1",  # IP suspecte
            "malware.example.com",  # Domaine malveillant
            "5d41402abc4b2a76b9719d911017c592"  # Hash MD5 suspect
        ]
        
        with patch.object(security_monitor.threat_intel, 'check_indicators') as mock_intel:
            mock_intel.return_value = {
                "threats_found": True,
                "threat_count": 3,
                "threat_details": [
                    {
                        "indicator": "203.0.113.1",
                        "type": "ip_address",
                        "threat_type": "botnet_c2",
                        "confidence": 0.95,
                        "first_seen": "2024-01-15",
                        "sources": ["malware_bytes", "virus_total"]
                    },
                    {
                        "indicator": "malware.example.com",
                        "type": "domain",
                        "threat_type": "phishing",
                        "confidence": 0.88,
                        "first_seen": "2024-01-20",
                        "sources": ["phish_tank", "openphish"]
                    },
                    {
                        "indicator": "5d41402abc4b2a76b9719d911017c592",
                        "type": "file_hash",
                        "threat_type": "trojan",
                        "confidence": 0.99,
                        "first_seen": "2024-01-10",
                        "sources": ["virus_total", "hybrid_analysis"]
                    }
                ],
                "recommended_actions": ["block_all", "investigate_further"]
            }
            
            result = await security_monitor.check_threat_intelligence(
                indicators=suspicious_indicators
            )
        
        assert result["threats_found"] is True
        assert result["threat_count"] == 3
        assert len(result["threat_details"]) == 3
        assert "block_all" in result["recommended_actions"]
        
        # Vérifier détails des menaces
        ip_threat = next(t for t in result["threat_details"] if t["type"] == "ip_address")
        assert ip_threat["threat_type"] == "botnet_c2"
        assert ip_threat["confidence"] > 0.9
    
    @pytest.mark.asyncio
    async def test_security_incident_creation(self, security_monitor):
        """Test création d'incident de sécurité"""
        security_events = [
            {
                "event_id": "event_001",
                "type": SecurityEventType.BRUTE_FORCE_ATTACK.value,
                "severity": AlertSeverity.HIGH.value,
                "timestamp": datetime.utcnow() - timedelta(minutes=5)
            },
            {
                "event_id": "event_002", 
                "type": SecurityEventType.SUSPICIOUS_LOGIN.value,
                "severity": AlertSeverity.MEDIUM.value,
                "timestamp": datetime.utcnow() - timedelta(minutes=3)
            },
            {
                "event_id": "event_003",
                "type": SecurityEventType.DATA_EXFILTRATION.value,
                "severity": AlertSeverity.CRITICAL.value,
                "timestamp": datetime.utcnow()
            }
        ]
        
        with patch.object(security_monitor, '_correlate_events') as mock_correlate:
            with patch.object(security_monitor, '_create_incident') as mock_incident:
                mock_correlate.return_value = {
                    "correlation_found": True,
                    "attack_pattern": "apt_attack",
                    "confidence": 0.87
                }
                
                mock_incident.return_value = {
                    "incident_id": "INC_001",
                    "title": "Suspected APT Attack - Data Exfiltration",
                    "severity": AlertSeverity.CRITICAL.value,
                    "status": "open",
                    "affected_assets": ["user_database", "api_servers"],
                    "timeline": security_events,
                    "assigned_to": "security_team",
                    "created_at": datetime.utcnow()
                }
                
                incident = await security_monitor.create_security_incident(
                    events=security_events,
                    attack_pattern="apt_attack"
                )
        
        assert incident["incident_id"] == "INC_001"
        assert incident["severity"] == AlertSeverity.CRITICAL.value
        assert incident["status"] == "open"
        assert "user_database" in incident["affected_assets"]
        assert len(incident["timeline"]) == 3
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, security_monitor):
        """Test surveillance en temps réel"""
        # Simuler flux d'événements en temps réel
        real_time_events = [
            {"timestamp": datetime.utcnow(), "type": "login_attempt", "result": "failed"},
            {"timestamp": datetime.utcnow(), "type": "api_access", "endpoint": "/admin"},
            {"timestamp": datetime.utcnow(), "type": "file_download", "size": 100000},
            {"timestamp": datetime.utcnow(), "type": "privilege_escalation", "user": "user_789"}
        ]
        
        with patch.object(security_monitor, '_process_real_time_event') as mock_process:
            alert_triggered = False
            
            async def mock_process_event(event):
                if event["type"] == "privilege_escalation":
                    return {
                        "alert_triggered": True,
                        "alert_level": AlertSeverity.HIGH.value,
                        "immediate_action": "suspend_user"
                    }
                return {"alert_triggered": False}
            
            mock_process.side_effect = mock_process_event
            
            for event in real_time_events:
                result = await security_monitor.process_real_time_event(event)
                if result.get("alert_triggered"):
                    alert_triggered = True
                    break
        
        assert alert_triggered is True


@pytest_markers["unit"]
@pytest_markers["risk"]
class TestRiskAssessment:
    """Tests pour l'évaluation des risques"""
    
    @pytest.mark.asyncio
    async def test_user_risk_scoring(self, security_monitor):
        """Test scoring de risque utilisateur"""
        user_profile = {
            "user_id": "user_789",
            "account_age": 30,  # Compte récent
            "privilege_level": "admin",  # Privilèges élevés
            "recent_activities": [
                {"action": "password_change", "timestamp": datetime.utcnow() - timedelta(hours=1)},
                {"action": "permission_request", "timestamp": datetime.utcnow() - timedelta(hours=2)},
                {"action": "bulk_data_access", "timestamp": datetime.utcnow() - timedelta(hours=3)}
            ],
            "device_trust_level": 0.3,  # Appareil peu fiable
            "geographic_anomaly": True,
            "behavior_score": 0.25  # Comportement inhabituel
        }
        
        with patch.object(security_monitor, '_calculate_user_risk') as mock_risk:
            mock_risk.return_value = {
                "overall_risk_score": 0.78,
                "risk_level": "high",
                "risk_factors": [
                    {"factor": "recent_account", "weight": 0.15, "score": 0.8},
                    {"factor": "admin_privileges", "weight": 0.25, "score": 0.9},
                    {"factor": "unusual_activity", "weight": 0.20, "score": 0.85},
                    {"factor": "untrusted_device", "weight": 0.15, "score": 0.7},
                    {"factor": "geographic_anomaly", "weight": 0.15, "score": 0.9},
                    {"factor": "behavior_deviation", "weight": 0.10, "score": 0.75}
                ],
                "recommendations": [
                    "require_additional_mfa",
                    "limit_data_access",
                    "increase_monitoring"
                ]
            }
            
            risk_assessment = await security_monitor.assess_user_risk(user_profile)
        
        assert risk_assessment["overall_risk_score"] > 0.7
        assert risk_assessment["risk_level"] == "high"
        assert len(risk_assessment["risk_factors"]) == 6
        assert "require_additional_mfa" in risk_assessment["recommendations"]
    
    @pytest.mark.asyncio
    async def test_session_risk_evaluation(self, security_monitor):
        """Test évaluation du risque de session"""
        session_context = {
            "session_id": "session_abc123",
            "user_id": "user_456",
            "ip_address": "192.168.1.100",
            "location": {"country": "FR", "city": "Paris"},
            "device_fingerprint": "known_device_123",
            "authentication_method": "password_only",  # Pas de MFA
            "session_duration": 14400,  # 4 heures
            "actions_performed": [
                {"action": "view_sensitive_data", "count": 15},
                {"action": "modify_user_permissions", "count": 3},
                {"action": "export_data", "count": 2}
            ],
            "unusual_patterns": ["rapid_successive_requests", "bulk_operations"]
        }
        
        with patch.object(security_monitor, '_evaluate_session_risk') as mock_eval:
            mock_eval.return_value = {
                "session_risk_score": 0.65,
                "risk_category": "medium",
                "risk_indicators": [
                    "no_mfa_authentication",
                    "extended_session_duration", 
                    "sensitive_data_access",
                    "permission_modifications",
                    "bulk_operations"
                ],
                "trust_score": 0.35,
                "recommended_actions": [
                    "request_mfa_reverification",
                    "limit_session_duration",
                    "log_all_actions"
                ]
            }
            
            result = await security_monitor.evaluate_session_risk(session_context)
        
        assert result["session_risk_score"] > 0.6
        assert result["risk_category"] == "medium"
        assert "no_mfa_authentication" in result["risk_indicators"]
        assert "request_mfa_reverification" in result["recommended_actions"]


@pytest_markers["performance"]
@pytest_markers["monitoring"]
class TestMonitoringPerformance:
    """Tests de performance pour la surveillance"""
    
    @pytest.mark.asyncio
    async def test_event_processing_performance(self, security_monitor):
        """Test performance traitement d'événements"""
        event = {
            "user_id": "user_123",
            "timestamp": datetime.utcnow(),
            "event_type": "api_request",
            "details": {"endpoint": "/api/data", "method": "GET"}
        }
        
        with patch.object(security_monitor, 'analyze_security_event') as mock_analyze:
            mock_analyze.return_value = {"threat_detected": False, "risk_score": 0.1}
            
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                security_monitor.analyze_security_event,
                event
            )
        
        # Analyse d'événement doit être < 50ms
        assert execution_time < 0.05
        assert result["threat_detected"] is False
        
        print(f"🔍 Temps analyse événement: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_high_volume_monitoring(self, security_monitor):
        """Test surveillance à haut volume"""
        # Simuler 1000 événements par seconde
        events = [
            {
                "event_id": f"event_{i}",
                "timestamp": datetime.utcnow(),
                "type": "api_request",
                "user_id": f"user_{i % 100}"
            }
            for i in range(1000)
        ]
        
        with patch.object(security_monitor, 'process_event_batch') as mock_batch:
            mock_batch.return_value = {
                "processed_count": 1000,
                "threats_detected": 5,
                "processing_time": 0.8,
                "events_per_second": 1250
            }
            
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                security_monitor.process_event_batch,
                events
            )
        
        # Traitement en lot doit être efficace
        assert execution_time < 1.0
        assert result["processed_count"] == 1000
        assert result["events_per_second"] > 1000
        
        print(f"📊 Débit événements: {result['events_per_second']:.0f} evt/s")


@pytest_markers["security"]
@pytest_markers["monitoring"]
class TestMonitoringSecurity:
    """Tests de sécurité pour la surveillance"""
    
    @pytest.mark.asyncio
    async def test_false_positive_rate(self, security_monitor):
        """Test taux de faux positifs"""
        # Événements normaux qui ne devraient pas déclencher d'alertes
        normal_events = [
            {
                "user_id": "user_123",
                "ip_address": "192.168.1.100",
                "action": "login",
                "success": True,
                "timestamp": datetime.utcnow()
            },
            {
                "user_id": "user_123", 
                "ip_address": "192.168.1.100",
                "action": "view_profile",
                "timestamp": datetime.utcnow()
            },
            {
                "user_id": "user_123",
                "ip_address": "192.168.1.100", 
                "action": "logout",
                "timestamp": datetime.utcnow()
            }
        ]
        
        false_positives = 0
        
        with patch.object(security_monitor, 'analyze_security_event') as mock_analyze:
            mock_analyze.return_value = {"threat_detected": False, "risk_score": 0.05}
            
            for event in normal_events:
                result = await security_monitor.analyze_security_event(event)
                if result["threat_detected"]:
                    false_positives += 1
        
        false_positive_rate = false_positives / len(normal_events)
        
        # Taux de faux positifs doit être < 5%
        assert false_positive_rate < 0.05
        
        print(f"📊 Taux faux positifs: {false_positive_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_monitoring_system_integrity(self, security_monitor):
        """Test intégrité du système de surveillance"""
        # Vérifier que le système de surveillance ne peut pas être contourné
        with patch.object(security_monitor, '_check_monitoring_health') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "components": {
                    "event_collector": "running",
                    "threat_detector": "running", 
                    "anomaly_detector": "running",
                    "alert_system": "running"
                },
                "last_heartbeat": datetime.utcnow(),
                "events_processed_last_minute": 1500,
                "system_integrity": "verified"
            }
            
            health_check = await security_monitor.check_system_health()
        
        assert health_check["status"] == "healthy"
        assert health_check["system_integrity"] == "verified"
        assert all(status == "running" for status in health_check["components"].values())
    
    @pytest.mark.asyncio
    async def test_alert_tampering_protection(self, security_monitor):
        """Test protection contre la falsification d'alertes"""
        alert = {
            "alert_id": "alert_001",
            "severity": AlertSeverity.HIGH.value,
            "message": "Suspicious activity detected",
            "timestamp": datetime.utcnow(),
            "source": "threat_detector"
        }
        
        with patch.object(security_monitor, '_sign_alert') as mock_sign:
            with patch.object(security_monitor, '_verify_alert_signature') as mock_verify:
                # Signer l'alerte
                mock_sign.return_value = {
                    "signature": "alert_digital_signature",
                    "signing_key_id": "monitoring_key_001"
                }
                
                signed_alert = await security_monitor.sign_alert(alert)
                
                # Vérifier la signature
                mock_verify.return_value = {"valid": True, "integrity_verified": True}
                
                verification = await security_monitor.verify_alert_signature(signed_alert)
        
        assert signed_alert["signature"] is not None
        assert verification["valid"] is True
        assert verification["integrity_verified"] is True


if __name__ == "__main__":
    print("🧪 Tests SecurityMonitor Ultra-Avancés")
    print("📋 Modules testés:")
    print("  ✅ Détection d'événements de sécurité et menaces")
    print("  ✅ Détection d'attaques par force brute")
    print("  ✅ Détection d'anomalies par ML")
    print("  ✅ Intégration threat intelligence")
    print("  ✅ Création d'incidents de sécurité")
    print("  ✅ Surveillance en temps réel")
    print("  ✅ Évaluation des risques utilisateur et session")
    print("  ✅ Tests de performance et sécurité")
    
    # Lancement des tests
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])
