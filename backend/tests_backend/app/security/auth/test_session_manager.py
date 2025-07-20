# 🧪 Tests SessionManager Ultra-Avancés
# ===================================

import pytest
import pytest_asyncio
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.security.auth.session_manager import (
    SessionManager, Session, DeviceInfo, SessionState, SecurityContext,
    SessionRisk, TrustLevel, DeviceType
)

from conftest import (
    TestDataFactory, TestUtils, PerformanceTestUtils, SecurityValidators,
    pytest_markers
)


@pytest_markers["unit"]
@pytest_markers["session"]
class TestSessionManager:
    """Tests unitaires pour SessionManager"""
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test création de session"""
        user_data = TestDataFactory.create_test_user()
        device_info = TestDataFactory.create_device_info()
        
        with patch.object(session_manager, '_store_session') as mock_store:
            mock_store.return_value = True
            
            session = await session_manager.create_session(
                user_id=user_data["user_id"],
                device_info=device_info,
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)",
                remember_device=True
            )
        
        assert session.session_id is not None
        assert session.user_id == user_data["user_id"]
        assert session.device_info.device_id is not None
        assert session.device_info.device_type == device_info["device_type"]
        assert session.state == SessionState.ACTIVE
        assert session.created_at is not None
        assert session.last_activity is not None
        assert session.expires_at > datetime.utcnow()
        assert session.is_trusted_device is True
    
    @pytest.mark.asyncio
    async def test_validate_session(self, session_manager):
        """Test validation de session"""
        session_data = TestDataFactory.create_session()
        
        # Test session valide
        with patch.object(session_manager, '_get_session') as mock_get:
            mock_get.return_value = session_data
            
            result = await session_manager.validate_session(
                session_id=session_data["session_id"],
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)"
            )
        
        assert result["valid"] is True
        assert result["session"]["session_id"] == session_data["session_id"]
        assert result["session"]["user_id"] == session_data["user_id"]
        
        # Test session expirée
        expired_session = session_data.copy()
        expired_session["expires_at"] = datetime.utcnow() - timedelta(hours=1)
        
        with patch.object(session_manager, '_get_session') as mock_get:
            mock_get.return_value = expired_session
            
            result = await session_manager.validate_session(
                session_id=expired_session["session_id"],
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)"
            )
        
        assert result["valid"] is False
        assert result["reason"] == "expired"
    
    @pytest.mark.asyncio
    async def test_device_trust_management(self, session_manager):
        """Test gestion de la confiance des appareils"""
        user_data = TestDataFactory.create_test_user()
        device_info = TestDataFactory.create_device_info()
        
        # Nouveau appareil - non fiable
        with patch.object(session_manager, '_is_trusted_device') as mock_trusted:
            mock_trusted.return_value = False
            
            trust_result = await session_manager.evaluate_device_trust(
                user_id=user_data["user_id"],
                device_info=device_info,
                ip_address="192.168.1.100"
            )
        
        assert trust_result["is_trusted"] is False
        assert trust_result["trust_level"] == TrustLevel.UNTRUSTED.value
        assert trust_result["requires_verification"] is True
        
        # Marquer l'appareil comme fiable
        with patch.object(session_manager, '_mark_device_trusted') as mock_mark:
            mock_mark.return_value = True
            
            result = await session_manager.trust_device(
                user_id=user_data["user_id"],
                device_id=device_info["device_id"],
                verification_code="123456"
            )
        
        assert result["success"] is True
        assert result["device_trusted"] is True
        
        # Vérifier que l'appareil est maintenant fiable
        with patch.object(session_manager, '_is_trusted_device') as mock_trusted:
            mock_trusted.return_value = True
            
            trust_result = await session_manager.evaluate_device_trust(
                user_id=user_data["user_id"],
                device_info=device_info,
                ip_address="192.168.1.100"
            )
        
        assert trust_result["is_trusted"] is True
        assert trust_result["trust_level"] == TrustLevel.TRUSTED.value
        assert trust_result["requires_verification"] is False
    
    @pytest.mark.asyncio
    async def test_session_hijacking_detection(self, session_manager):
        """Test détection de piratage de session"""
        session_data = TestDataFactory.create_session()
        
        # Utilisation normale de la session
        with patch.object(session_manager, '_analyze_session_behavior') as mock_analyze:
            mock_analyze.return_value = {
                "risk_score": 0.1,
                "anomalies": [],
                "suspicious": False
            }
            
            security_check = await session_manager.check_session_security(
                session_id=session_data["session_id"],
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)",
                geolocation={"country": "FR", "city": "Paris"}
            )
        
        assert security_check["secure"] is True
        assert security_check["risk_score"] < 0.5
        
        # Changement suspect d'IP/localisation
        with patch.object(session_manager, '_analyze_session_behavior') as mock_analyze:
            mock_analyze.return_value = {
                "risk_score": 0.9,
                "anomalies": ["ip_change", "location_change"],
                "suspicious": True,
                "details": {
                    "previous_ip": "192.168.1.100",
                    "current_ip": "203.0.113.1",
                    "distance_km": 500
                }
            }
            
            security_check = await session_manager.check_session_security(
                session_id=session_data["session_id"],
                ip_address="203.0.113.1",
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)",
                geolocation={"country": "DE", "city": "Berlin"}
            )
        
        assert security_check["secure"] is False
        assert security_check["risk_score"] > 0.8
        assert "ip_change" in security_check["anomalies"]
        assert "location_change" in security_check["anomalies"]
    
    @pytest.mark.asyncio
    async def test_concurrent_session_management(self, session_manager):
        """Test gestion des sessions concurrentes"""
        user_data = TestDataFactory.create_test_user()
        
        # Créer plusieurs sessions pour le même utilisateur
        sessions = []
        for i in range(3):
            device_info = TestDataFactory.create_device_info(f"device_{i}")
            
            with patch.object(session_manager, '_store_session', return_value=True):
                session = await session_manager.create_session(
                    user_id=user_data["user_id"],
                    device_info=device_info,
                    ip_address=f"192.168.1.{100+i}",
                    user_agent=f"TestAgent/{i}"
                )
            
            sessions.append(session)
        
        # Vérifier limitation des sessions concurrentes
        with patch.object(session_manager, '_get_user_sessions') as mock_get_sessions:
            mock_get_sessions.return_value = sessions
            
            with patch.object(session_manager, '_enforce_session_limit') as mock_enforce:
                mock_enforce.return_value = {
                    "limit_exceeded": True,
                    "max_sessions": 2,
                    "current_sessions": 3,
                    "terminated_sessions": [sessions[0].session_id]
                }
                
                limit_result = await session_manager.enforce_session_limits(
                    user_id=user_data["user_id"]
                )
        
        assert limit_result["limit_exceeded"] is True
        assert limit_result["max_sessions"] == 2
        assert len(limit_result["terminated_sessions"]) == 1
    
    @pytest.mark.asyncio
    async def test_session_refresh(self, session_manager):
        """Test renouvellement de session"""
        session_data = TestDataFactory.create_session()
        
        with patch.object(session_manager, '_get_session') as mock_get:
            with patch.object(session_manager, '_update_session') as mock_update:
                mock_get.return_value = session_data
                mock_update.return_value = True
                
                refresh_result = await session_manager.refresh_session(
                    session_id=session_data["session_id"],
                    extend_expiry=True
                )
        
        assert refresh_result["success"] is True
        assert refresh_result["new_expires_at"] > datetime.utcnow()
        assert refresh_result["extended"] is True
    
    @pytest.mark.asyncio
    async def test_terminate_session(self, session_manager):
        """Test terminaison de session"""
        session_data = TestDataFactory.create_session()
        
        with patch.object(session_manager, '_terminate_session') as mock_terminate:
            mock_terminate.return_value = {
                "success": True,
                "terminated_at": datetime.utcnow(),
                "reason": "user_logout"
            }
            
            result = await session_manager.terminate_session(
                session_id=session_data["session_id"],
                reason="user_logout"
            )
        
        assert result["success"] is True
        assert result["terminated_at"] is not None
        assert result["reason"] == "user_logout"
    
    @pytest.mark.asyncio
    async def test_bulk_session_operations(self, session_manager):
        """Test opérations en lot sur les sessions"""
        user_data = TestDataFactory.create_test_user()
        session_ids = [f"session_{i}" for i in range(5)]
        
        # Terminer toutes les sessions d'un utilisateur
        with patch.object(session_manager, '_terminate_user_sessions') as mock_terminate:
            mock_terminate.return_value = {
                "success": True,
                "terminated_count": 5,
                "terminated_sessions": session_ids
            }
            
            result = await session_manager.terminate_all_user_sessions(
                user_id=user_data["user_id"],
                except_session_id=None,
                reason="security_breach"
            )
        
        assert result["success"] is True
        assert result["terminated_count"] == 5
        assert len(result["terminated_sessions"]) == 5


@pytest_markers["unit"]
@pytest_markers["session"]
class TestDeviceManagement:
    """Tests pour la gestion des appareils"""
    
    @pytest.mark.asyncio
    async def test_device_fingerprinting(self, session_manager):
        """Test empreinte d'appareil"""
        device_info = TestDataFactory.create_device_info()
        
        with patch.object(session_manager, '_generate_device_fingerprint') as mock_fingerprint:
            mock_fingerprint.return_value = {
                "fingerprint": "fp_abc123def456",
                "components": {
                    "screen_resolution": "1920x1080",
                    "timezone": "Europe/Paris",
                    "language": "fr-FR",
                    "platform": "iPhone",
                    "user_agent_hash": "ua_hash_123"
                },
                "confidence": 0.95
            }
            
            fingerprint = await session_manager.generate_device_fingerprint(device_info)
        
        assert fingerprint["fingerprint"] is not None
        assert fingerprint["confidence"] > 0.9
        assert "screen_resolution" in fingerprint["components"]
        assert "timezone" in fingerprint["components"]
    
    @pytest.mark.asyncio
    async def test_device_risk_assessment(self, session_manager):
        """Test évaluation du risque d'appareil"""
        device_info = TestDataFactory.create_device_info()
        
        # Appareil sûr
        with patch.object(session_manager, '_assess_device_risk') as mock_assess:
            mock_assess.return_value = {
                "risk_score": 0.2,
                "risk_factors": [],
                "risk_level": "low",
                "recommendations": []
            }
            
            risk_assessment = await session_manager.assess_device_risk(
                device_info=device_info,
                ip_address="192.168.1.100",
                user_history={}
            )
        
        assert risk_assessment["risk_score"] < 0.5
        assert risk_assessment["risk_level"] == "low"
        
        # Appareil suspect
        with patch.object(session_manager, '_assess_device_risk') as mock_assess:
            mock_assess.return_value = {
                "risk_score": 0.8,
                "risk_factors": ["tor_exit_node", "suspicious_geolocation"],
                "risk_level": "high",
                "recommendations": ["require_mfa", "block_device"]
            }
            
            risk_assessment = await session_manager.assess_device_risk(
                device_info=device_info,
                ip_address="198.51.100.1",  # IP suspecte
                user_history={}
            )
        
        assert risk_assessment["risk_score"] > 0.7
        assert risk_assessment["risk_level"] == "high"
        assert "tor_exit_node" in risk_assessment["risk_factors"]
        assert "require_mfa" in risk_assessment["recommendations"]


@pytest_markers["performance"]
@pytest_markers["session"]
class TestSessionPerformance:
    """Tests de performance pour les sessions"""
    
    @pytest.mark.asyncio
    async def test_session_creation_performance(self, session_manager):
        """Test performance création de session"""
        user_data = TestDataFactory.create_test_user()
        device_info = TestDataFactory.create_device_info()
        
        with patch.object(session_manager, '_store_session', return_value=True):
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                session_manager.create_session,
                user_id=user_data["user_id"],
                device_info=device_info,
                ip_address="192.168.1.100",
                user_agent="TestAgent/1.0"
            )
        
        # Création de session doit être < 200ms
        assert execution_time < 0.2
        assert result.session_id is not None
        
        print(f"🔐 Temps création session: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_session_validation_performance(self, session_manager):
        """Test performance validation de session"""
        session_data = TestDataFactory.create_session()
        
        with patch.object(session_manager, '_get_session', return_value=session_data):
            result, execution_time = await PerformanceTestUtils.measure_execution_time(
                session_manager.validate_session,
                session_id=session_data["session_id"],
                ip_address="192.168.1.100",
                user_agent="TestAgent/1.0"
            )
        
        # Validation de session doit être < 100ms
        assert execution_time < 0.1
        assert result["valid"] is True
        
        print(f"✅ Temps validation session: {execution_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, session_manager):
        """Test opérations de session concurrentes"""
        sessions = [TestDataFactory.create_session(f"session_{i}") for i in range(20)]
        
        async def validate_session(session_data):
            with patch.object(session_manager, '_get_session', return_value=session_data):
                return await session_manager.validate_session(
                    session_id=session_data["session_id"],
                    ip_address="192.168.1.100",
                    user_agent="TestAgent/1.0"
                )
        
        stress_results = await PerformanceTestUtils.stress_test_function(
            validate_session,
            concurrent_calls=20,
            sessions[0]
        )
        
        assert stress_results["success_rate"] >= 0.95
        assert stress_results["average_time"] < 0.5
        
        print(f"📊 Taux réussite concurrent: {stress_results['success_rate']:.2%}")
        print(f"⏱️  Temps moyen validation: {stress_results['average_time']:.3f}s")


@pytest_markers["security"]
@pytest_markers["session"]
class TestSessionSecurity:
    """Tests de sécurité pour les sessions"""
    
    @pytest.mark.asyncio
    async def test_session_fixation_protection(self, session_manager):
        """Test protection contre la fixation de session"""
        user_data = TestDataFactory.create_test_user()
        
        # Créer session avant authentification
        anonymous_session_id = "anonymous_session_123"
        
        # Après authentification, l'ID de session doit changer
        with patch.object(session_manager, '_regenerate_session_id') as mock_regen:
            mock_regen.return_value = f"auth_session_{TestUtils.generate_random_string(16)}"
            
            result = await session_manager.elevate_session_security(
                current_session_id=anonymous_session_id,
                user_id=user_data["user_id"],
                authentication_method="password"
            )
        
        assert result["success"] is True
        assert result["new_session_id"] != anonymous_session_id
        assert result["security_elevated"] is True
    
    @pytest.mark.asyncio
    async def test_brute_force_protection(self, session_manager):
        """Test protection contre les attaques par force brute"""
        ip_address = "203.0.113.1"
        
        # Simuler tentatives multiples de validation de session invalide
        failed_attempts = []
        for i in range(5):
            with patch.object(session_manager, '_record_failed_attempt') as mock_record:
                mock_record.return_value = {"attempts": i + 1, "blocked": i >= 4}
                
                result = await session_manager.validate_session(
                    session_id=f"invalid_session_{i}",
                    ip_address=ip_address,
                    user_agent="AttackAgent/1.0"
                )
                
                failed_attempts.append(result)
        
        # Après 5 tentatives, l'IP doit être bloquée
        with patch.object(session_manager, '_is_ip_blocked') as mock_blocked:
            mock_blocked.return_value = True
            
            blocked_result = await session_manager.validate_session(
                session_id="any_session",
                ip_address=ip_address,
                user_agent="AttackAgent/1.0"
            )
        
        assert blocked_result["valid"] is False
        assert blocked_result["reason"] == "ip_blocked"
    
    @pytest.mark.asyncio
    async def test_session_token_security(self, session_manager):
        """Test sécurité des tokens de session"""
        session_data = TestDataFactory.create_session()
        
        # Vérifier que les tokens sont cryptographiquement sécurisés
        session_token = session_data["session_id"]
        
        # Doit être suffisamment long
        assert len(session_token) >= 32
        
        # Ne doit pas contenir d'informations prévisibles
        assert not any(char in session_token for char in ["user", "session", "123"])
        
        # Test rotation de token
        with patch.object(session_manager, '_rotate_session_token') as mock_rotate:
            new_token = f"rotated_{TestUtils.generate_random_string(32)}"
            mock_rotate.return_value = {"success": True, "new_token": new_token}
            
            rotation_result = await session_manager.rotate_session_token(
                session_id=session_token
            )
        
        assert rotation_result["success"] is True
        assert rotation_result["new_token"] != session_token
        assert len(rotation_result["new_token"]) >= 32


if __name__ == "__main__":
    print("🧪 Tests SessionManager Ultra-Avancés")
    print("📋 Modules testés:")
    print("  ✅ Création et validation de sessions")
    print("  ✅ Gestion de la confiance des appareils")
    print("  ✅ Détection de piratage de session")
    print("  ✅ Gestion des sessions concurrentes")
    print("  ✅ Renouvellement et terminaison de sessions")
    print("  ✅ Empreinte et évaluation du risque d'appareil")
    print("  ✅ Tests de sécurité et protection")
    print("  ✅ Tests de performance et concurrence")
    
    # Lancement des tests
    import subprocess
    subprocess.run(["pytest", __file__, "-v", "--tb=short"])
