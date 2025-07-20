"""
📋 Compliance Tests - GDPR, SOC2, HIPAA, PCI DSS Validation
==========================================================

Comprehensive compliance testing suite for enterprise regulatory requirements
including data protection, security controls, and audit capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import re
from unittest.mock import patch, Mock

from app.tenancy import EnterpriseTenantManager
from app.tenancy.models import TenantCreate, TenantUser, DataSubject
from app.tenancy.advanced_managers import ComplianceManager, DataGovernanceManager
from app.tenancy.services import AuditService, DataProtectionService
from tests_backend.app.tenancy.fixtures.tenant_factories import create_sample_tenant_data

pytestmark = pytest.mark.asyncio


class TestGDPRCompliance:
    """🇪🇺 GDPR (General Data Protection Regulation) Compliance Tests"""
    
    @pytest.fixture
    async def compliance_manager(self):
        """Create compliance manager for GDPR testing"""
        manager = ComplianceManager()
        await manager.initialize_gdpr_framework()
        yield manager
        await manager.cleanup()
    
    async def test_right_to_be_informed(self, compliance_manager):
        """Test GDPR Article 13 & 14 - Right to be informed"""
        tenant_id = "gdpr_informed_test"
        
        # Test privacy notice compliance
        privacy_notice = await compliance_manager.get_privacy_notice(tenant_id)
        
        required_elements = [
            "data_controller_identity",
            "processing_purposes",
            "legal_basis",
            "data_categories",
            "recipients",
            "retention_period",
            "data_subject_rights",
            "right_to_withdraw_consent",
            "complaint_procedure",
            "data_source"
        ]
        
        for element in required_elements:
            assert element in privacy_notice, f"Privacy notice missing: {element}"
            assert privacy_notice[element] is not None
            assert len(str(privacy_notice[element])) > 10
        
        # Test consent mechanism
        consent_test = await compliance_manager.test_consent_mechanism(
            tenant_id, "marketing_emails"
        )
        
        assert consent_test["freely_given"] is True
        assert consent_test["specific"] is True
        assert consent_test["informed"] is True
        assert consent_test["unambiguous"] is True
        assert consent_test["withdrawable"] is True
    
    async def test_right_of_access(self, compliance_manager):
        """Test GDPR Article 15 - Right of access"""
        tenant_id = "gdpr_access_test"
        data_subject_id = "test_subject_123"
        
        # Create test personal data
        personal_data = {
            "profile": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123"
            },
            "preferences": {
                "language": "en",
                "notifications": True,
                "marketing_consent": True
            },
            "activity_logs": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "login",
                    "ip_address": "192.168.1.100"
                }
            ]
        }
        
        # Store personal data
        await compliance_manager.store_personal_data(
            tenant_id, data_subject_id, personal_data
        )
        
        # Test subject access request (SAR)
        sar_result = await compliance_manager.process_subject_access_request(
            tenant_id, data_subject_id
        )
        
        assert sar_result["status"] == "completed"
        assert sar_result["data_provided"] is True
        assert "personal_data" in sar_result
        assert "processing_purposes" in sar_result
        assert "categories_of_data" in sar_result
        assert "recipients" in sar_result
        assert "retention_period" in sar_result
        assert "third_country_transfers" in sar_result
        
        # Verify completeness of data export
        exported_data = sar_result["personal_data"]
        assert exported_data["profile"]["name"] == "John Doe"
        assert exported_data["profile"]["email"] == "john.doe@example.com"
        assert len(exported_data["activity_logs"]) > 0
        
        # Test response time compliance (within 30 days)
        response_time = sar_result["response_time_days"]
        assert response_time <= 30, f"SAR response time {response_time} days exceeds GDPR limit"
    
    async def test_right_to_rectification(self, compliance_manager):
        """Test GDPR Article 16 - Right to rectification"""
        tenant_id = "gdpr_rectification_test"
        data_subject_id = "rectification_subject"
        
        # Create personal data with errors
        incorrect_data = {
            "name": "John Smith",  # Incorrect
            "email": "wrong.email@example.com",  # Incorrect
            "address": "123 Wrong St, Error City"  # Incorrect
        }
        
        await compliance_manager.store_personal_data(
            tenant_id, data_subject_id, incorrect_data
        )
        
        # Submit rectification request
        rectification_request = {
            "requested_changes": {
                "name": "John Doe",  # Correct name
                "email": "john.doe@example.com",  # Correct email
                "address": "456 Correct Ave, Right City"  # Correct address
            },
            "justification": "Name change due to marriage, email update for primary contact"
        }
        
        rectification_result = await compliance_manager.process_rectification_request(
            tenant_id, data_subject_id, rectification_request
        )
        
        assert rectification_result["status"] == "completed"
        assert rectification_result["changes_applied"] is True
        
        # Verify changes were applied
        updated_data = await compliance_manager.get_personal_data(
            tenant_id, data_subject_id
        )
        
        assert updated_data["name"] == "John Doe"
        assert updated_data["email"] == "john.doe@example.com"
        assert updated_data["address"] == "456 Correct Ave, Right City"
        
        # Test audit trail for rectification
        audit_trail = rectification_result["audit_trail"]
        assert audit_trail["original_values"] == incorrect_data
        assert audit_trail["new_values"] == rectification_request["requested_changes"]
        assert audit_trail["timestamp"] is not None
        assert audit_trail["justification"] == rectification_request["justification"]
    
    async def test_right_to_erasure(self, compliance_manager):
        """Test GDPR Article 17 - Right to erasure (Right to be forgotten)"""
        tenant_id = "gdpr_erasure_test"
        data_subject_id = "erasure_subject"
        
        # Create comprehensive personal data
        personal_data = {
            "profile": {"name": "Jane Doe", "email": "jane@example.com"},
            "sensitive_data": {"health_info": "allergic to peanuts"},
            "transactional_data": [
                {"transaction_id": "tx_001", "amount": 50.00},
                {"transaction_id": "tx_002", "amount": 25.00}
            ],
            "backups": ["backup_1", "backup_2", "backup_3"]
        }
        
        await compliance_manager.store_personal_data(
            tenant_id, data_subject_id, personal_data
        )
        
        # Submit erasure request
        erasure_request = {
            "grounds": "withdrawal_of_consent",
            "scope": "complete_erasure",
            "include_backups": True
        }
        
        erasure_result = await compliance_manager.process_erasure_request(
            tenant_id, data_subject_id, erasure_request
        )
        
        assert erasure_result["status"] == "completed"
        assert erasure_result["data_erased"] is True
        assert erasure_result["backups_erased"] is True
        
        # Verify complete erasure
        verification_result = await compliance_manager.verify_data_erasure(
            tenant_id, data_subject_id
        )
        
        assert verification_result["personal_data_found"] is False
        assert verification_result["backup_data_found"] is False
        assert verification_result["log_entries_anonymized"] is True
        
        # Test exceptions to erasure (legal basis)
        legal_obligations_data = {
            "financial_records": {"tax_year": 2023, "amount": 1000.00},
            "legal_basis": "legal_obligation"
        }
        
        await compliance_manager.store_personal_data(
            tenant_id, "legal_subject", legal_obligations_data
        )
        
        legal_erasure_result = await compliance_manager.process_erasure_request(
            tenant_id, "legal_subject", {"grounds": "withdrawal_of_consent"}
        )
        
        assert legal_erasure_result["status"] == "partially_completed"
        assert legal_erasure_result["exceptions"]["legal_obligation"] is True
    
    async def test_right_to_data_portability(self, compliance_manager):
        """Test GDPR Article 20 - Right to data portability"""
        tenant_id = "gdpr_portability_test"
        data_subject_id = "portability_subject"
        
        # Create structured personal data
        portable_data = {
            "user_profile": {
                "first_name": "Alice",
                "last_name": "Johnson",
                "email": "alice.johnson@example.com",
                "preferences": {
                    "language": "en",
                    "timezone": "UTC",
                    "notifications": True
                }
            },
            "user_content": [
                {
                    "type": "playlist",
                    "name": "My Favorites",
                    "tracks": ["track_1", "track_2", "track_3"]
                },
                {
                    "type": "review",
                    "content": "Great song!",
                    "rating": 5
                }
            ],
            "interaction_data": [
                {"action": "play", "item": "track_1", "timestamp": "2024-01-01T10:00:00Z"},
                {"action": "like", "item": "track_2", "timestamp": "2024-01-01T10:05:00Z"}
            ]
        }
        
        await compliance_manager.store_personal_data(
            tenant_id, data_subject_id, portable_data
        )
        
        # Test data portability request
        portability_result = await compliance_manager.process_portability_request(
            tenant_id, data_subject_id, {"format": "json", "include_metadata": True}
        )
        
        assert portability_result["status"] == "completed"
        assert portability_result["format"] == "json"
        assert portability_result["machine_readable"] is True
        
        # Verify exported data structure
        exported_data = portability_result["exported_data"]
        assert "user_profile" in exported_data
        assert "user_content" in exported_data
        assert "interaction_data" in exported_data
        
        # Verify data integrity
        assert exported_data["user_profile"]["email"] == "alice.johnson@example.com"
        assert len(exported_data["user_content"]) == 2
        assert len(exported_data["interaction_data"]) == 2
        
        # Test alternative formats
        csv_result = await compliance_manager.process_portability_request(
            tenant_id, data_subject_id, {"format": "csv"}
        )
        assert csv_result["format"] == "csv"
        assert csv_result["status"] == "completed"
    
    async def test_right_to_restrict_processing(self, compliance_manager):
        """Test GDPR Article 18 - Right to restriction of processing"""
        tenant_id = "gdpr_restriction_test"
        data_subject_id = "restriction_subject"
        
        # Create personal data for processing
        processing_data = {
            "marketing_profile": {"interests": ["music", "technology"]},
            "analytics_data": {"behavior_patterns": ["active_user", "premium_subscriber"]},
            "automated_decisions": {"recommendation_engine": True}
        }
        
        await compliance_manager.store_personal_data(
            tenant_id, data_subject_id, processing_data
        )
        
        # Submit restriction request
        restriction_request = {
            "grounds": "accuracy_contested",
            "processing_activities": ["marketing", "analytics"],
            "restriction_period": "pending_verification"
        }
        
        restriction_result = await compliance_manager.process_restriction_request(
            tenant_id, data_subject_id, restriction_request
        )
        
        assert restriction_result["status"] == "applied"
        assert restriction_result["restricted_activities"] == ["marketing", "analytics"]
        
        # Verify processing is restricted
        processing_status = await compliance_manager.check_processing_status(
            tenant_id, data_subject_id
        )
        
        assert processing_status["marketing"]["restricted"] is True
        assert processing_status["analytics"]["restricted"] is True
        assert processing_status["storage"]["restricted"] is False  # Storage still allowed
        
        # Test lifting of restriction
        lift_result = await compliance_manager.lift_processing_restriction(
            tenant_id, data_subject_id, {"reason": "accuracy_verified"}
        )
        
        assert lift_result["status"] == "lifted"
        
        updated_status = await compliance_manager.check_processing_status(
            tenant_id, data_subject_id
        )
        
        assert updated_status["marketing"]["restricted"] is False
        assert updated_status["analytics"]["restricted"] is False
    
    async def test_data_protection_impact_assessment(self, compliance_manager):
        """Test GDPR Article 35 - Data Protection Impact Assessment (DPIA)"""
        tenant_id = "gdpr_dpia_test"
        
        # High-risk processing scenario
        processing_scenario = {
            "processing_type": "automated_decision_making",
            "data_categories": ["special_categories", "biometric_data"],
            "data_subjects": "large_scale",
            "technology": ["ai_profiling", "facial_recognition"],
            "purposes": ["targeted_advertising", "credit_scoring"]
        }
        
        # Conduct DPIA
        dpia_result = await compliance_manager.conduct_dpia(
            tenant_id, processing_scenario
        )
        
        assert dpia_result["dpia_required"] is True
        assert dpia_result["risk_level"] in ["high", "very_high"]
        
        # Verify DPIA components
        dpia_components = dpia_result["dpia_assessment"]
        required_components = [
            "processing_description",
            "necessity_assessment",
            "proportionality_assessment",
            "risk_identification",
            "risk_mitigation_measures",
            "consultation_records",
            "monitoring_measures"
        ]
        
        for component in required_components:
            assert component in dpia_components
            assert dpia_components[component] is not None
        
        # Test risk mitigation
        mitigation_measures = dpia_components["risk_mitigation_measures"]
        assert len(mitigation_measures) > 0
        assert any("encryption" in measure.lower() for measure in mitigation_measures)
        assert any("access_control" in measure.lower() for measure in mitigation_measures)
    
    async def test_consent_management(self, compliance_manager):
        """Test GDPR Article 7 - Conditions for consent"""
        tenant_id = "gdpr_consent_test"
        data_subject_id = "consent_subject"
        
        # Test valid consent collection
        consent_request = {
            "purposes": ["marketing_emails", "analytics", "personalization"],
            "consent_text": "I agree to receive marketing emails and data analysis for personalization",
            "granular_consent": True,
            "easy_withdrawal": True
        }
        
        consent_result = await compliance_manager.collect_consent(
            tenant_id, data_subject_id, consent_request
        )
        
        assert consent_result["consent_valid"] is True
        assert consent_result["consent_id"] is not None
        assert consent_result["timestamp"] is not None
        
        # Test consent verification
        verification_result = await compliance_manager.verify_consent(
            tenant_id, data_subject_id, "marketing_emails"
        )
        
        assert verification_result["has_consent"] is True
        assert verification_result["consent_details"]["freely_given"] is True
        assert verification_result["consent_details"]["specific"] is True
        assert verification_result["consent_details"]["informed"] is True
        
        # Test consent withdrawal
        withdrawal_result = await compliance_manager.withdraw_consent(
            tenant_id, data_subject_id, ["marketing_emails"]
        )
        
        assert withdrawal_result["status"] == "withdrawn"
        assert withdrawal_result["processing_stopped"] is True
        
        # Verify withdrawal
        post_withdrawal_verification = await compliance_manager.verify_consent(
            tenant_id, data_subject_id, "marketing_emails"
        )
        
        assert post_withdrawal_verification["has_consent"] is False
        assert post_withdrawal_verification["withdrawal_date"] is not None


class TestSOC2Compliance:
    """🛡️ SOC 2 Type II Compliance Tests"""
    
    @pytest.fixture
    async def soc2_manager(self):
        """Create SOC2 compliance manager"""
        manager = ComplianceManager()
        await manager.initialize_soc2_framework()
        yield manager
        await manager.cleanup()
    
    async def test_security_principle_controls(self, soc2_manager):
        """Test SOC 2 Security Principle controls"""
        tenant_id = "soc2_security_test"
        
        # Test CC6.1 - Logical and Physical Access Controls
        access_controls_result = await soc2_manager.test_access_controls(tenant_id)
        
        assert access_controls_result["logical_access"]["implemented"] is True
        assert access_controls_result["physical_access"]["implemented"] is True
        assert access_controls_result["privileged_access"]["monitored"] is True
        
        # Test CC6.2 - Authentication Systems
        auth_systems_result = await soc2_manager.test_authentication_systems(tenant_id)
        
        assert auth_systems_result["multi_factor_auth"]["required"] is True
        assert auth_systems_result["password_complexity"]["enforced"] is True
        assert auth_systems_result["session_management"]["secure"] is True
        
        # Test CC6.3 - Authorization Systems  
        authz_result = await soc2_manager.test_authorization_systems(tenant_id)
        
        assert authz_result["role_based_access"]["implemented"] is True
        assert authz_result["least_privilege"]["enforced"] is True
        assert authz_result["segregation_of_duties"]["implemented"] is True
        
        # Test CC7.1 - System Operations
        operations_result = await soc2_manager.test_system_operations(tenant_id)
        
        assert operations_result["monitoring"]["continuous"] is True
        assert operations_result["incident_response"]["documented"] is True
        assert operations_result["capacity_planning"]["performed"] is True
    
    async def test_availability_principle_controls(self, soc2_manager):
        """Test SOC 2 Availability Principle controls"""
        tenant_id = "soc2_availability_test"
        
        # Test A1.1 - Performance Monitoring
        performance_result = await soc2_manager.test_performance_monitoring(tenant_id)
        
        assert performance_result["availability_target"] >= 99.9
        assert performance_result["monitoring_frequency"] == "real_time"
        assert performance_result["alerting"]["configured"] is True
        
        # Test A1.2 - Backup and Recovery
        backup_result = await soc2_manager.test_backup_recovery(tenant_id)
        
        assert backup_result["backup_frequency"] == "daily"
        assert backup_result["recovery_testing"]["performed"] is True
        assert backup_result["rto"]["minutes"] <= 60  # Recovery Time Objective
        assert backup_result["rpo"]["minutes"] <= 15  # Recovery Point Objective
        
        # Test A1.3 - Business Continuity
        continuity_result = await soc2_manager.test_business_continuity(tenant_id)
        
        assert continuity_result["disaster_recovery_plan"]["documented"] is True
        assert continuity_result["failover_procedures"]["tested"] is True
        assert continuity_result["redundancy"]["implemented"] is True
    
    async def test_processing_integrity_controls(self, soc2_manager):
        """Test SOC 2 Processing Integrity controls"""
        tenant_id = "soc2_integrity_test"
        
        # Test PI1.1 - Data Processing Controls
        processing_result = await soc2_manager.test_data_processing_controls(tenant_id)
        
        assert processing_result["input_validation"]["implemented"] is True
        assert processing_result["error_handling"]["comprehensive"] is True
        assert processing_result["data_integrity_checks"]["automated"] is True
        
        # Test PI1.2 - Quality Assurance
        qa_result = await soc2_manager.test_quality_assurance(tenant_id)
        
        assert qa_result["code_reviews"]["mandatory"] is True
        assert qa_result["testing_coverage"]["percentage"] >= 90
        assert qa_result["deployment_controls"]["implemented"] is True
    
    async def test_confidentiality_controls(self, soc2_manager):
        """Test SOC 2 Confidentiality controls"""
        tenant_id = "soc2_confidentiality_test"
        
        # Test C1.1 - Data Classification
        classification_result = await soc2_manager.test_data_classification(tenant_id)
        
        assert classification_result["classification_scheme"]["implemented"] is True
        assert classification_result["sensitive_data"]["identified"] is True
        assert classification_result["handling_procedures"]["documented"] is True
        
        # Test C1.2 - Encryption Controls
        encryption_result = await soc2_manager.test_encryption_controls(tenant_id)
        
        assert encryption_result["data_at_rest"]["encrypted"] is True
        assert encryption_result["data_in_transit"]["encrypted"] is True
        assert encryption_result["key_management"]["secure"] is True
    
    async def test_privacy_controls(self, soc2_manager):
        """Test SOC 2 Privacy controls"""
        tenant_id = "soc2_privacy_test"
        
        # Test P1.1 - Privacy Notice
        privacy_notice_result = await soc2_manager.test_privacy_notice(tenant_id)
        
        assert privacy_notice_result["notice_provided"] is True
        assert privacy_notice_result["collection_purposes"]["disclosed"] is True
        assert privacy_notice_result["retention_periods"]["specified"] is True
        
        # Test P2.1 - Choice and Consent
        consent_result = await soc2_manager.test_choice_consent(tenant_id)
        
        assert consent_result["opt_in_mechanism"]["provided"] is True
        assert consent_result["granular_choices"]["available"] is True
        assert consent_result["withdrawal_option"]["easy"] is True
    
    async def test_control_effectiveness_over_time(self, soc2_manager):
        """Test SOC 2 Type II control effectiveness over time"""
        tenant_id = "soc2_effectiveness_test"
        
        # Simulate testing over a period (Type II requirement)
        test_period_months = 12
        
        effectiveness_result = await soc2_manager.test_control_effectiveness_over_time(
            tenant_id, test_period_months
        )
        
        assert effectiveness_result["test_period_months"] == test_period_months
        assert effectiveness_result["controls_tested"] >= 50
        assert effectiveness_result["effectiveness_rate"] >= 95.0
        
        # Test for any control deficiencies
        deficiencies = effectiveness_result.get("deficiencies", [])
        significant_deficiencies = [d for d in deficiencies if d["severity"] == "significant"]
        material_weaknesses = [d for d in deficiencies if d["severity"] == "material"]
        
        assert len(material_weaknesses) == 0, f"Material weaknesses found: {material_weaknesses}"
        assert len(significant_deficiencies) <= 2, f"Too many significant deficiencies: {significant_deficiencies}"


class TestHIPAACompliance:
    """🏥 HIPAA Compliance Tests"""
    
    @pytest.fixture
    async def hipaa_manager(self):
        """Create HIPAA compliance manager"""
        manager = ComplianceManager()
        await manager.initialize_hipaa_framework()
        yield manager
        await manager.cleanup()
    
    async def test_administrative_safeguards(self, hipaa_manager):
        """Test HIPAA Administrative Safeguards"""
        tenant_id = "hipaa_admin_test"
        
        # Test Security Officer Assignment (§164.308(a)(2))
        security_officer_result = await hipaa_manager.test_security_officer_assignment(tenant_id)
        
        assert security_officer_result["security_officer"]["assigned"] is True
        assert security_officer_result["security_officer"]["responsibilities"]["documented"] is True
        assert security_officer_result["security_officer"]["training"]["completed"] is True
        
        # Test Workforce Training (§164.308(a)(5))
        training_result = await hipaa_manager.test_workforce_training(tenant_id)
        
        assert training_result["initial_training"]["completed"] is True
        assert training_result["ongoing_training"]["scheduled"] is True
        assert training_result["training_records"]["maintained"] is True
        
        # Test Access Management (§164.308(a)(4))
        access_mgmt_result = await hipaa_manager.test_access_management(tenant_id)
        
        assert access_mgmt_result["access_authorization"]["documented"] is True
        assert access_mgmt_result["minimum_necessary"]["enforced"] is True
        assert access_mgmt_result["access_reviews"]["regular"] is True
    
    async def test_physical_safeguards(self, hipaa_manager):
        """Test HIPAA Physical Safeguards"""
        tenant_id = "hipaa_physical_test"
        
        # Test Facility Access Controls (§164.310(a)(1))
        facility_result = await hipaa_manager.test_facility_access_controls(tenant_id)
        
        assert facility_result["physical_access"]["controlled"] is True
        assert facility_result["visitor_logs"]["maintained"] is True
        assert facility_result["emergency_procedures"]["documented"] is True
        
        # Test Workstation Controls (§164.310(b))
        workstation_result = await hipaa_manager.test_workstation_controls(tenant_id)
        
        assert workstation_result["workstation_security"]["implemented"] is True
        assert workstation_result["screen_locks"]["automatic"] is True
        assert workstation_result["authorized_users"]["controlled"] is True
        
        # Test Device and Media Controls (§164.310(d)(1))
        media_result = await hipaa_manager.test_device_media_controls(tenant_id)
        
        assert media_result["media_disposal"]["secure"] is True
        assert media_result["device_tracking"]["implemented"] is True
        assert media_result["data_backup"]["protected"] is True
    
    async def test_technical_safeguards(self, hipaa_manager):
        """Test HIPAA Technical Safeguards"""
        tenant_id = "hipaa_technical_test"
        
        # Test Access Control (§164.312(a)(1))
        access_control_result = await hipaa_manager.test_technical_access_control(tenant_id)
        
        assert access_control_result["unique_user_identification"]["enforced"] is True
        assert access_control_result["authentication"]["strong"] is True
        assert access_control_result["session_timeout"]["configured"] is True
        
        # Test Audit Controls (§164.312(b))
        audit_result = await hipaa_manager.test_audit_controls(tenant_id)
        
        assert audit_result["audit_logging"]["comprehensive"] is True
        assert audit_result["log_review"]["regular"] is True
        assert audit_result["log_protection"]["implemented"] is True
        
        # Test Integrity (§164.312(c)(1))
        integrity_result = await hipaa_manager.test_data_integrity(tenant_id)
        
        assert integrity_result["data_alteration"]["protected"] is True
        assert integrity_result["data_destruction"]["controlled"] is True
        assert integrity_result["integrity_monitoring"]["automated"] is True
        
        # Test Transmission Security (§164.312(e)(1))
        transmission_result = await hipaa_manager.test_transmission_security(tenant_id)
        
        assert transmission_result["encryption_in_transit"]["implemented"] is True
        assert transmission_result["network_security"]["monitored"] is True
        assert transmission_result["secure_protocols"]["enforced"] is True
    
    async def test_phi_protection(self, hipaa_manager):
        """Test Protected Health Information (PHI) protection"""
        tenant_id = "hipaa_phi_test"
        
        # Test PHI identification and classification
        phi_data = {
            "patient_id": "PHI123456",
            "name": "John Patient",
            "dob": "1980-01-01",
            "ssn": "123-45-6789",
            "medical_record": "Patient has diabetes",
            "treatment_history": ["insulin", "metformin"],
            "insurance_info": "Blue Cross Blue Shield"
        }
        
        classification_result = await hipaa_manager.classify_phi_data(tenant_id, phi_data)
        
        assert classification_result["contains_phi"] is True
        assert "patient_id" in classification_result["phi_elements"]
        assert "ssn" in classification_result["phi_elements"]
        assert "medical_record" in classification_result["phi_elements"]
        
        # Test PHI encryption
        encryption_result = await hipaa_manager.encrypt_phi_data(tenant_id, phi_data)
        
        assert encryption_result["encrypted"] is True
        assert encryption_result["encryption_standard"] == "AES-256"
        assert "123-45-6789" not in str(encryption_result["encrypted_data"])
        
        # Test PHI access logging
        access_result = await hipaa_manager.log_phi_access(
            tenant_id, "user123", "view_patient_record", phi_data["patient_id"]
        )
        
        assert access_result["logged"] is True
        assert access_result["audit_entry"]["user_id"] == "user123"
        assert access_result["audit_entry"]["action"] == "view_patient_record"
        assert access_result["audit_entry"]["phi_accessed"] == phi_data["patient_id"]
    
    async def test_business_associate_agreements(self, hipaa_manager):
        """Test Business Associate Agreement (BAA) compliance"""
        tenant_id = "hipaa_baa_test"
        
        # Test BAA requirements
        baa_assessment = await hipaa_manager.assess_business_associate_requirements(tenant_id)
        
        required_baa_elements = [
            "permitted_uses_disclosures",
            "safeguard_requirements",
            "prohibited_uses_disclosures",
            "breach_notification",
            "subcontractor_agreements",
            "termination_procedures",
            "return_destruction_phi"
        ]
        
        for element in required_baa_elements:
            assert element in baa_assessment["required_elements"]
            assert baa_assessment["compliance_status"][element] is True
        
        # Test subcontractor compliance
        subcontractor_result = await hipaa_manager.test_subcontractor_compliance(tenant_id)
        
        assert subcontractor_result["subcontractor_baas"]["required"] is True
        assert subcontractor_result["compliance_monitoring"]["implemented"] is True
        assert subcontractor_result["due_diligence"]["performed"] is True


class TestPCIDSSCompliance:
    """💳 PCI DSS Compliance Tests"""
    
    @pytest.fixture
    async def pci_manager(self):
        """Create PCI DSS compliance manager"""
        manager = ComplianceManager()
        await manager.initialize_pci_dss_framework()
        yield manager
        await manager.cleanup()
    
    async def test_cardholder_data_environment(self, pci_manager):
        """Test PCI DSS Requirement 1 & 2 - Network Security"""
        tenant_id = "pci_network_test"
        
        # Test firewall configuration
        firewall_result = await pci_manager.test_firewall_configuration(tenant_id)
        
        assert firewall_result["firewall_installed"] is True
        assert firewall_result["default_deny_policy"] is True
        assert firewall_result["rule_review"]["documented"] is True
        
        # Test system hardening
        hardening_result = await pci_manager.test_system_hardening(tenant_id)
        
        assert hardening_result["default_passwords"]["changed"] is True
        assert hardening_result["unnecessary_services"]["disabled"] is True
        assert hardening_result["security_patches"]["current"] is True
    
    async def test_cardholder_data_protection(self, pci_manager):
        """Test PCI DSS Requirement 3 & 4 - Data Protection"""
        tenant_id = "pci_data_test"
        
        # Test data encryption
        encryption_result = await pci_manager.test_cardholder_data_encryption(tenant_id)
        
        assert encryption_result["stored_data"]["encrypted"] is True
        assert encryption_result["transmission"]["encrypted"] is True
        assert encryption_result["key_management"]["secure"] is True
        
        # Test data retention
        retention_result = await pci_manager.test_data_retention_policies(tenant_id)
        
        assert retention_result["retention_policy"]["documented"] is True
        assert retention_result["secure_deletion"]["implemented"] is True
        assert retention_result["data_purging"]["automated"] is True
        
        # Test PAN (Primary Account Number) protection
        pan_test_data = {
            "card_number": "4111111111111111",  # Test card number
            "expiry_date": "12/25",
            "cvv": "123"
        }
        
        pan_protection_result = await pci_manager.test_pan_protection(tenant_id, pan_test_data)
        
        assert pan_protection_result["pan_masked"] is True
        assert pan_protection_result["storage_encrypted"] is True
        assert "4111111111111111" not in str(pan_protection_result["protected_data"])
    
    async def test_access_control_requirements(self, pci_manager):
        """Test PCI DSS Requirement 7 & 8 - Access Control"""
        tenant_id = "pci_access_test"
        
        # Test access control systems
        access_control_result = await pci_manager.test_access_control_systems(tenant_id)
        
        assert access_control_result["role_based_access"]["implemented"] is True
        assert access_control_result["need_to_know"]["enforced"] is True
        assert access_control_result["access_reviews"]["regular"] is True
        
        # Test user authentication
        auth_result = await pci_manager.test_user_authentication(tenant_id)
        
        assert auth_result["unique_user_ids"]["enforced"] is True
        assert auth_result["strong_passwords"]["required"] is True
        assert auth_result["multi_factor_auth"]["implemented"] is True
        assert auth_result["session_timeout"]["configured"] is True
    
    async def test_vulnerability_management(self, pci_manager):
        """Test PCI DSS Requirement 6 & 11 - Vulnerability Management"""
        tenant_id = "pci_vulnerability_test"
        
        # Test secure development practices
        development_result = await pci_manager.test_secure_development(tenant_id)
        
        assert development_result["secure_coding"]["implemented"] is True
        assert development_result["code_reviews"]["mandatory"] is True
        assert development_result["vulnerability_testing"]["automated"] is True
        
        # Test vulnerability scanning
        scanning_result = await pci_manager.test_vulnerability_scanning(tenant_id)
        
        assert scanning_result["internal_scans"]["quarterly"] is True
        assert scanning_result["external_scans"]["quarterly"] is True
        assert scanning_result["remediation"]["timely"] is True
        
        # Test penetration testing
        pentest_result = await pci_manager.test_penetration_testing(tenant_id)
        
        assert pentest_result["annual_testing"]["performed"] is True
        assert pentest_result["scope"]["comprehensive"] is True
        assert pentest_result["remediation"]["verified"] is True
    
    async def test_monitoring_and_logging(self, pci_manager):
        """Test PCI DSS Requirement 10 - Monitoring"""
        tenant_id = "pci_monitoring_test"
        
        # Test audit logging
        logging_result = await pci_manager.test_audit_logging(tenant_id)
        
        assert logging_result["access_logs"]["comprehensive"] is True
        assert logging_result["authentication_logs"]["detailed"] is True
        assert logging_result["administrative_actions"]["logged"] is True
        
        # Test log monitoring
        monitoring_result = await pci_manager.test_log_monitoring(tenant_id)
        
        assert monitoring_result["real_time_monitoring"]["implemented"] is True
        assert monitoring_result["anomaly_detection"]["automated"] is True
        assert monitoring_result["incident_response"]["documented"] is True
        
        # Test log protection
        protection_result = await pci_manager.test_log_protection(tenant_id)
        
        assert protection_result["log_integrity"]["protected"] is True
        assert protection_result["unauthorized_access"]["prevented"] is True
        assert protection_result["retention_period"]["compliant"] is True


# Compliance test utilities
@pytest.fixture
async def compliance_test_environment():
    """Setup compliance testing environment"""
    test_config = {
        "gdpr_enabled": True,
        "soc2_enabled": True,
        "hipaa_enabled": True,
        "pci_dss_enabled": True,
        "audit_mode": True,
        "compliance_reporting": True
    }
    
    with patch.dict('os.environ', {
        'COMPLIANCE_TEST_MODE': 'true',
        'AUDIT_LEVEL': 'comprehensive',
        'COMPLIANCE_STANDARDS': 'gdpr,soc2,hipaa,pci_dss'
    }):
        yield test_config


@pytest.fixture(autouse=True)
async def compliance_audit_logger(request):
    """Log compliance test execution for audit purposes"""
    test_name = request.node.name
    audit_service = AuditService()
    
    # Log test start
    await audit_service.log_compliance_test_start(test_name)
    
    yield
    
    # Log test completion
    await audit_service.log_compliance_test_completion(test_name)
    await audit_service.cleanup()
