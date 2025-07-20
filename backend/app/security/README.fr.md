# 🛡️ Security Module - Guide d'Utilisation Complet
# ===============================================

![Security](https://img.shields.io/badge/Security-Production%20Ready-success)
![Enterprise](https://img.shields.io/badge/Enterprise-Grade-blue)
![Compliance](https://img.shields.io/badge/Compliance-Multi%20Standard-orange)

## 🎯 Introduction

Bienvenue dans le **Guide d'Utilisation Complet** du Module de Sécurité Enterprise de Spotify AI Agent ! 🚀

Ce guide vous accompagnera dans l'utilisation de tous les composants de sécurité, des cas d'usage basiques aux configurations enterprise les plus avancées.

---

## 📋 Table des Matières

1. [🚀 Démarrage Rapide](#-démarrage-rapide)
2. [🔐 Authentification](#-authentification)
3. [🎫 Gestion des Tokens](#-gestion-des-tokens)
4. [💾 Sessions & Devices](#-sessions--devices)
5. [🔒 Chiffrement](#-chiffrement)
6. [📊 Monitoring](#-monitoring)
7. [⚖️ Conformité](#️-conformité)
8. [🧪 Tests & Debug](#-tests--debug)
9. [🛠️ Configuration Avancée](#️-configuration-avancée)
10. [🚨 Troubleshooting](#-troubleshooting)

---

## 🚀 Démarrage Rapide

### **Installation et Configuration Minimale**

```bash
# 1. Installation des dépendances
pip install -r requirements.txt

# 2. Démarrage Redis
sudo systemctl start redis-server

# 3. Variables d'environnement
export REDIS_URL=redis://localhost:6379/0
export JWT_SECRET_KEY=your-super-secret-key
export ENCRYPTION_KEY=your-encryption-key
```

### **Configuration Basique**

```python
import redis
from app.security import SecurityManager, get_security_config

# Configuration Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# Initialisation du Security Manager
config = get_security_config()
security_manager = SecurityManager(
    redis_client=redis_client,
    config=config
)

print("✅ Module de sécurité initialisé avec succès !")
```

### **Premier Test**

```python
# Test d'authentification simple
from app.security.auth import create_authentication_manager

auth_manager = create_authentication_manager(
    redis_client=redis_client,
    config=config['authentication']
)

# Créer un utilisateur de test
test_user = await auth_manager.create_test_user(
    username="test@example.com",
    password="SecurePassword123!",
    email="test@example.com"
)

# Test d'authentification
result = await auth_manager.authenticate_user(
    username="test@example.com",
    password="SecurePassword123!",
    ip_address="127.0.0.1",
    user_agent="Test Agent"
)

if result.success:
    print("🎉 Authentification réussie !")
else:
    print("❌ Échec de l'authentification")
```

---

## 🔐 Authentification

### **Authentification Simple**

```python
from app.security.auth.authenticator import AuthenticationManager

# Authentification par mot de passe
async def login_user(username: str, password: str, request):
    result = await auth_manager.authenticate_user(
        username=username,
        password=password,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )
    
    if result.success:
        # Créer une session
        session = await session_manager.create_session(
            user_id=result.user_id,
            request=request,
            authentication_methods=result.methods_used
        )
        
        # Générer les tokens
        access_token = await token_manager.create_access_token(
            user_id=result.user_id,
            scopes=["read", "write"]
        )
        
        return {
            "success": True,
            "access_token": access_token,
            "session_id": session.session_id,
            "user_id": result.user_id
        }
    else:
        return {
            "success": False,
            "error": result.error_message,
            "requires_mfa": result.requires_mfa
        }
```

### **Multi-Factor Authentication (MFA)**

#### **Configuration TOTP (Google Authenticator)**

```python
from app.security.auth.authenticator import MultiFactorAuthenticator

# Setup TOTP pour un utilisateur
async def setup_totp(user_id: str):
    totp_secret = await mfa_manager.setup_totp(
        user_id=user_id,
        issuer="Spotify AI Agent",
        account_name="user@example.com"
    )
    
    # Générer le QR code pour l'app
    qr_code_url = await mfa_manager.generate_totp_qr_code(
        user_id=user_id,
        secret=totp_secret
    )
    
    return {
        "secret": totp_secret,
        "qr_code_url": qr_code_url,
        "backup_codes": await mfa_manager.generate_backup_codes(user_id)
    }

# Vérification TOTP
async def verify_totp(user_id: str, code: str):
    is_valid = await mfa_manager.verify_totp(
        user_id=user_id,
        code=code,
        allow_reuse=False  # Empêcher la réutilisation du code
    )
    
    if is_valid:
        # Marquer MFA comme complété
        await auth_manager.complete_mfa_challenge(
            user_id=user_id,
            method="totp"
        )
        return {"success": True}
    else:
        return {"success": False, "error": "Code TOTP invalide"}
```

#### **MFA par SMS**

```python
# Envoyer un code SMS
async def send_sms_code(user_id: str, phone_number: str):
    sms_code = await mfa_manager.send_sms_code(
        user_id=user_id,
        phone_number=phone_number,
        expires_in=300  # 5 minutes
    )
    
    return {"message": "Code SMS envoyé", "expires_in": 300}

# Vérifier le code SMS
async def verify_sms_code(user_id: str, code: str):
    is_valid = await mfa_manager.verify_sms_code(
        user_id=user_id,
        code=code
    )
    
    return {"success": is_valid}
```

#### **Authentification Biométrique**

```python
from app.security.auth.authenticator import BiometricAuthenticator

# Enregistrement d'empreinte digitale
async def register_fingerprint(user_id: str, fingerprint_data: bytes):
    registration_result = await biometric_auth.register_fingerprint(
        user_id=user_id,
        fingerprint_template=fingerprint_data,
        device_id="device_123"
    )
    
    return {
        "success": registration_result.success,
        "fingerprint_id": registration_result.fingerprint_id
    }

# Vérification d'empreinte
async def verify_fingerprint(user_id: str, fingerprint_data: bytes):
    verification_result = await biometric_auth.verify_fingerprint(
        user_id=user_id,
        fingerprint_data=fingerprint_data,
        confidence_threshold=0.95
    )
    
    return {
        "success": verification_result.success,
        "confidence": verification_result.confidence_score
    }

# Reconnaissance faciale
async def verify_face(user_id: str, face_image: bytes):
    face_result = await biometric_auth.verify_face(
        user_id=user_id,
        face_image=face_image,
        confidence_threshold=0.95
    )
    
    return {
        "success": face_result.success,
        "confidence": face_result.confidence_score
    }
```

### **Risk-Based Authentication**

```python
from app.security.auth.authenticator import RiskBasedAuthenticator

async def analyze_login_risk(user_id: str, request):
    # Analyser le risque de la tentative de connexion
    risk_analysis = await risk_auth.analyze_authentication_risk(
        user_id=user_id,
        context={
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent"),
            "geolocation": await geoip_service.get_location(request.client.host),
            "device_fingerprint": await device_manager.get_device_fingerprint(request),
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday()
        }
    )
    
    # Décision basée sur le score de risque
    if risk_analysis.risk_score > 0.8:
        # Risque très élevé - bloquer
        return {
            "action": "block",
            "reason": "Risk score too high",
            "risk_score": risk_analysis.risk_score
        }
    elif risk_analysis.risk_score > 0.5:
        # Risque élevé - MFA renforcé
        return {
            "action": "require_enhanced_mfa",
            "required_methods": ["totp", "sms", "email"],
            "risk_score": risk_analysis.risk_score
        }
    elif risk_analysis.risk_score > 0.2:
        # Risque modéré - MFA simple
        return {
            "action": "require_mfa",
            "required_methods": ["totp"],
            "risk_score": risk_analysis.risk_score
        }
    else:
        # Risque faible - authentification normale
        return {
            "action": "allow",
            "risk_score": risk_analysis.risk_score
        }
```

---

## 🎫 Gestion des Tokens

### **JWT Tokens**

```python
from app.security.auth.token_manager import AdvancedTokenManager

# Création de tokens
async def create_user_tokens(user_id: str, scopes: list):
    # Access token (courte durée)
    access_token = await token_manager.create_access_token(
        user_id=user_id,
        scopes=scopes,
        expires_in=900,  # 15 minutes
        token_type="Bearer"
    )
    
    # Refresh token (longue durée)
    refresh_token = await token_manager.create_refresh_token(
        user_id=user_id,
        expires_in=2592000,  # 30 jours
        refresh_token_family=f"family_{user_id}"
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": 900
    }

# Validation de token
async def validate_token(token: str):
    validation_result = await token_manager.validate_token(
        token=token,
        check_expiration=True,
        check_revocation=True
    )
    
    if validation_result.valid:
        return {
            "valid": True,
            "user_id": validation_result.user_id,
            "scopes": validation_result.scopes,
            "expires_at": validation_result.expires_at
        }
    else:
        return {
            "valid": False,
            "error": validation_result.error_message
        }
```

### **Refresh Token Rotation**

```python
async def refresh_access_token(refresh_token: str):
    # Rotation sécurisée du refresh token
    rotation_result = await token_manager.rotate_refresh_token(
        refresh_token=refresh_token,
        invalidate_family=False  # Garder la famille active
    )
    
    if rotation_result.success:
        return {
            "access_token": rotation_result.new_access_token,
            "refresh_token": rotation_result.new_refresh_token,
            "token_type": "Bearer",
            "expires_in": 900
        }
    else:
        # En cas d'erreur, invalider toute la famille
        await token_manager.invalidate_refresh_token_family(
            refresh_token=refresh_token
        )
        
        return {
            "error": "Invalid refresh token",
            "require_reauth": True
        }
```

### **API Keys**

```python
from app.security.auth.token_manager import APIKeyManager

# Créer une API key
async def create_api_key(user_id: str, name: str, scopes: list):
    api_key = await api_key_manager.create_api_key(
        user_id=user_id,
        name=name,
        scopes=scopes,
        expires_in=31536000,  # 1 an
        rate_limit={
            "requests_per_minute": 1000,
            "requests_per_hour": 10000,
            "requests_per_day": 100000
        },
        ip_whitelist=["192.168.1.0/24", "10.0.0.0/8"]
    )
    
    return {
        "api_key": api_key.key,
        "api_key_id": api_key.id,
        "scopes": api_key.scopes,
        "rate_limits": api_key.rate_limit,
        "expires_at": api_key.expires_at
    }

# Validation d'API key
async def validate_api_key(api_key: str, required_scopes: list, request):
    validation_result = await api_key_manager.validate_api_key(
        api_key=api_key,
        required_scopes=required_scopes,
        ip_address=request.client.host,
        check_rate_limit=True
    )
    
    return {
        "valid": validation_result.valid,
        "user_id": validation_result.user_id,
        "scopes": validation_result.scopes,
        "rate_limit_remaining": validation_result.rate_limit_remaining
    }
```

---

## 💾 Sessions & Devices

### **Gestion des Sessions**

```python
from app.security.auth.session_manager import SecureSessionManager

# Créer une session
async def create_user_session(user_id: str, request, auth_methods: list):
    session = await session_manager.create_session(
        user_id=user_id,
        request=request,
        authentication_methods=auth_methods,
        remember_device=True,
        session_lifetime=3600  # 1 heure
    )
    
    return {
        "session_id": session.session_id,
        "expires_at": session.expires_at,
        "device_id": session.device_id
    }

# Valider une session
async def validate_session(session_id: str, request):
    validation_result = await session_manager.validate_session(
        session_id=session_id,
        request=request,
        check_hijacking=True
    )
    
    if validation_result.valid:
        # Mettre à jour l'activité
        await session_manager.update_session_activity(
            session_id=session_id,
            request=request
        )
        
        return {
            "valid": True,
            "user_id": validation_result.user_id,
            "expires_at": validation_result.expires_at
        }
    else:
        return {
            "valid": False,
            "error": validation_result.error_message
        }
```

### **Device Management**

```python
from app.security.auth.session_manager import DeviceManager

# Enregistrer un nouveau device
async def register_device(user_id: str, request, device_info: dict):
    device = await device_manager.register_device(
        user_id=user_id,
        device_info={
            "name": device_info.get("name", "Unknown Device"),
            "type": device_info.get("type", "unknown"),
            "os": device_info.get("os"),
            "browser": device_info.get("browser"),
            "fingerprint": await device_manager.generate_device_fingerprint(request)
        },
        trust_level="medium",
        require_approval=True
    )
    
    # Envoyer notification de nouveau device
    await notification_service.send_new_device_notification(
        user_id=user_id,
        device_name=device.name,
        location=await geoip_service.get_location(request.client.host)
    )
    
    return {
        "device_id": device.device_id,
        "requires_approval": device.requires_approval,
        "trust_level": device.trust_level
    }

# Obtenir les devices d'un utilisateur
async def get_user_devices(user_id: str):
    devices = await device_manager.get_user_devices(user_id)
    
    return {
        "devices": [
            {
                "device_id": device.device_id,
                "name": device.name,
                "type": device.type,
                "last_seen": device.last_seen,
                "trust_level": device.trust_level,
                "is_current": device.is_current_device
            }
            for device in devices
        ]
    }
```

### **Détection de Session Hijacking**

```python
async def check_session_security(session_id: str, request):
    # Vérifier les signes de hijacking
    hijacking_detected = await session_manager.detect_session_hijacking(
        session_id=session_id,
        current_request=request
    )
    
    if hijacking_detected:
        # Invalider immédiatement la session
        await session_manager.invalidate_session(
            session_id=session_id,
            reason="suspected_hijacking"
        )
        
        # Créer un incident de sécurité
        await threat_detector.create_security_incident(
            title="Session Hijacking Detected",
            description=f"Suspected session hijacking for session {session_id}",
            severity="HIGH",
            user_id=session.user_id,
            ip_address=request.client.host
        )
        
        return {
            "hijacking_detected": True,
            "session_invalidated": True,
            "require_reauth": True
        }
    
    return {"hijacking_detected": False}
```

---

## 🔒 Chiffrement

### **Chiffrement de Données**

```python
from app.security.encryption import EnterpriseEncryptionManager, EncryptionContext, DataClassification

# Chiffrer des données sensibles
async def encrypt_user_data(user_data: dict, user_id: str):
    context = EncryptionContext(
        data_classification=DataClassification.CONFIDENTIAL,
        compliance_requirements=["GDPR", "HIPAA"],
        user_id=user_id,
        purpose="user_data_storage"
    )
    
    # Chiffrer chaque champ sensible
    encrypted_data = {}
    for field, value in user_data.items():
        if field in ["email", "phone", "address", "ssn"]:
            encrypted_data[field] = await encryption_manager.encrypt_data(
                plaintext=str(value),
                context=context
            )
        else:
            encrypted_data[field] = value
    
    return encrypted_data

# Déchiffrer des données
async def decrypt_user_data(encrypted_data: dict, user_id: str):
    context = EncryptionContext(
        data_classification=DataClassification.CONFIDENTIAL,
        user_id=user_id,
        purpose="user_data_retrieval"
    )
    
    decrypted_data = {}
    for field, value in encrypted_data.items():
        if isinstance(value, dict) and "encrypted_data" in value:
            decrypted_data[field] = await encryption_manager.decrypt_data(
                encrypted_data=value,
                context=context
            )
        else:
            decrypted_data[field] = value
    
    return decrypted_data
```

### **Chiffrement Field-Level**

```python
# Chiffrer des champs spécifiques
async def encrypt_database_fields(record: dict):
    # Configuration du chiffrement par champ
    field_encryption_config = {
        "email": DataClassification.INTERNAL,
        "phone": DataClassification.CONFIDENTIAL,
        "ssn": DataClassification.RESTRICTED,
        "credit_card": DataClassification.TOP_SECRET
    }
    
    encrypted_record = record.copy()
    
    for field, classification in field_encryption_config.items():
        if field in record:
            context = EncryptionContext(
                data_classification=classification,
                purpose=f"field_encryption_{field}"
            )
            
            encrypted_record[field] = await encryption_manager.encrypt_field(
                field_name=field,
                field_value=record[field],
                context=context
            )
    
    return encrypted_record
```

### **Rotation des Clés**

```python
# Rotation automatique des clés de chiffrement
async def rotate_encryption_keys():
    rotation_result = await encryption_manager.rotate_encryption_keys(
        force_rotation=False,  # Seulement si nécessaire
        background_reencryption=True  # Re-chiffrer en arrière-plan
    )
    
    return {
        "keys_rotated": rotation_result.keys_rotated,
        "reencryption_tasks": rotation_result.reencryption_tasks,
        "estimated_completion": rotation_result.estimated_completion
    }

# Vérifier le statut de rotation
async def check_key_rotation_status():
    status = await encryption_manager.get_key_rotation_status()
    
    return {
        "last_rotation": status.last_rotation,
        "next_rotation": status.next_rotation,
        "reencryption_progress": status.reencryption_progress,
        "pending_tasks": status.pending_tasks
    }
```

---

## 📊 Monitoring

### **Surveillance des Événements**

```python
from app.security.monitoring import AdvancedThreatDetector, EventType, ThreatLevel

# Traiter un événement de sécurité
async def log_security_event(event_type: str, user_id: str, request, details: dict):
    event = await threat_detector.process_security_event(
        event_type=EventType[event_type.upper()],
        user_id=user_id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
        details=details,
        source="web_application"
    )
    
    # Analyser les anomalies
    anomalies = await threat_detector.detect_anomalies(
        user_id=user_id,
        event_type=event_type,
        time_window=3600  # 1 heure
    )
    
    if anomalies:
        # Créer une alerte
        await threat_detector.create_security_alert(
            title=f"Anomaly detected for user {user_id}",
            description=f"Unusual {event_type} pattern detected",
            severity=ThreatLevel.MEDIUM,
            user_id=user_id,
            anomalies=anomalies
        )
    
    return {"event_id": event.event_id, "anomalies_detected": len(anomalies)}
```

### **Dashboard de Sécurité**

```python
# Obtenir les métriques du dashboard
async def get_security_dashboard():
    dashboard_data = await threat_detector.get_security_dashboard_data(
        time_range=86400  # 24 heures
    )
    
    return {
        "metrics": {
            "total_events": dashboard_data["metrics"]["total_events"],
            "security_incidents": dashboard_data["metrics"]["security_incidents"],
            "blocked_attempts": dashboard_data["metrics"]["blocked_attempts"],
            "anomalies_detected": dashboard_data["metrics"]["anomalies_detected"]
        },
        "threats": {
            "suspicious_ips": dashboard_data["threats"]["suspicious_ips"],
            "failed_logins": dashboard_data["threats"]["failed_logins"],
            "brute_force_attempts": dashboard_data["threats"]["brute_force_attempts"]
        },
        "compliance": {
            "audit_events": dashboard_data["compliance"]["audit_events"],
            "policy_violations": dashboard_data["compliance"]["policy_violations"]
        },
        "charts": dashboard_data["charts"]
    }
```

### **Alertes et Notifications**

```python
# Configuration des alertes
async def configure_security_alerts(user_id: str, preferences: dict):
    alert_config = {
        "email_alerts": preferences.get("email_alerts", True),
        "sms_alerts": preferences.get("sms_alerts", False),
        "push_notifications": preferences.get("push_notifications", True),
        "severity_threshold": preferences.get("severity_threshold", "MEDIUM"),
        "alert_types": preferences.get("alert_types", [
            "login_anomaly",
            "new_device",
            "suspicious_activity",
            "security_incident"
        ])
    }
    
    await threat_detector.configure_user_alerts(
        user_id=user_id,
        config=alert_config
    )
    
    return {"message": "Alert configuration updated"}

# Envoyer une alerte de sécurité
async def send_security_alert(user_id: str, alert_type: str, details: dict):
    alert = await threat_detector.send_security_alert(
        user_id=user_id,
        alert_type=alert_type,
        title=details["title"],
        message=details["message"],
        severity=details.get("severity", "MEDIUM"),
        action_required=details.get("action_required", False)
    )
    
    return {"alert_id": alert.alert_id, "sent_at": alert.sent_at}
```

---

## ⚖️ Conformité

### **Audit et Compliance**

```python
from app.security.monitoring import ComplianceMonitor

# Générer un rapport de conformité
async def generate_compliance_report(standard: str, time_range: int):
    compliance_report = await compliance_monitor.generate_compliance_report(
        standard=standard,  # "GDPR", "HIPAA", "PCI_DSS", "SOX"
        time_range=time_range,
        include_evidence=True
    )
    
    return {
        "standard": compliance_report.standard,
        "compliance_score": compliance_report.compliance_score,
        "requirements_met": compliance_report.requirements_met,
        "requirements_total": compliance_report.requirements_total,
        "violations": compliance_report.violations,
        "recommendations": compliance_report.recommendations,
        "evidence": compliance_report.evidence
    }

# Vérifier la conformité en temps réel
async def check_real_time_compliance(action: str, data: dict):
    compliance_check = await compliance_monitor.check_compliance(
        action=action,
        data=data,
        standards=["GDPR", "HIPAA"]
    )
    
    if not compliance_check.compliant:
        # Bloquer l'action si non conforme
        await compliance_monitor.log_compliance_violation(
            action=action,
            violations=compliance_check.violations,
            data_hash=hash(str(data))
        )
        
        return {
            "allowed": False,
            "violations": compliance_check.violations,
            "message": "Action blocked due to compliance violations"
        }
    
    return {"allowed": True}
```

### **Gestion du Consentement (GDPR)**

```python
# Enregistrer le consentement
async def record_user_consent(user_id: str, consent_data: dict):
    consent_record = await compliance_monitor.record_consent(
        user_id=user_id,
        consent_type=consent_data["type"],
        consent_text=consent_data["text"],
        consent_version=consent_data["version"],
        collection_method=consent_data.get("method", "web_form"),
        ip_address=consent_data.get("ip_address"),
        user_agent=consent_data.get("user_agent")
    )
    
    return {
        "consent_id": consent_record.consent_id,
        "recorded_at": consent_record.recorded_at,
        "valid_until": consent_record.valid_until
    }

# Retrait du consentement
async def withdraw_consent(user_id: str, consent_type: str):
    withdrawal_result = await compliance_monitor.withdraw_consent(
        user_id=user_id,
        consent_type=consent_type,
        withdrawal_reason="user_request"
    )
    
    # Démarrer le processus de suppression des données
    if withdrawal_result.requires_data_deletion:
        await compliance_monitor.schedule_data_deletion(
            user_id=user_id,
            consent_type=consent_type,
            deletion_delay=2592000  # 30 jours
        )
    
    return {
        "withdrawn": True,
        "data_deletion_scheduled": withdrawal_result.requires_data_deletion,
        "deletion_date": withdrawal_result.deletion_date
    }
```

---

## 🧪 Tests & Debug

### **Tests de Sécurité**

```python
import pytest
from app.security.auth.authenticator import AuthenticationManager

# Test d'authentification
@pytest.mark.asyncio
async def test_user_authentication():
    """Test complet d'authentification utilisateur"""
    
    # Setup
    auth_manager = AuthenticationManager(redis_client=mock_redis)
    
    # Test authentification réussie
    result = await auth_manager.authenticate_user(
        username="test@example.com",
        password="ValidPassword123!",
        ip_address="127.0.0.1",
        user_agent="Test Agent"
    )
    
    assert result.success is True
    assert result.user_id == "test_user_id"
    assert "password" in result.methods_used
    
    # Test authentification échouée
    fail_result = await auth_manager.authenticate_user(
        username="test@example.com",
        password="WrongPassword",
        ip_address="127.0.0.1",
        user_agent="Test Agent"
    )
    
    assert fail_result.success is False
    assert fail_result.error_code == "INVALID_CREDENTIALS"

# Test MFA
@pytest.mark.asyncio
async def test_mfa_flow():
    """Test du flow MFA complet"""
    
    # Setup TOTP
    secret = await mfa_manager.setup_totp("test_user", "Test App")
    assert secret is not None
    
    # Générer et vérifier un code valide
    import pyotp
    totp = pyotp.TOTP(secret)
    valid_code = totp.now()
    
    result = await mfa_manager.verify_totp("test_user", valid_code)
    assert result is True
    
    # Test code invalide
    invalid_result = await mfa_manager.verify_totp("test_user", "000000")
    assert invalid_result is False
```

### **Tests de Charge**

```python
# Script de test de charge avec locust
from locust import HttpUser, task, between

class SecurityLoadTest(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup avant les tests"""
        # Créer un utilisateur de test
        self.login_data = {
            "username": f"test_{self.client.id}@example.com",
            "password": "TestPassword123!"
        }
        
        # Créer le compte
        self.client.post("/auth/register", json=self.login_data)
    
    @task(3)
    def test_login(self):
        """Test de connexion"""
        response = self.client.post("/auth/login", json=self.login_data)
        if response.status_code == 200:
            self.access_token = response.json()["access_token"]
    
    @task(2)
    def test_protected_endpoint(self):
        """Test d'endpoint protégé"""
        if hasattr(self, 'access_token'):
            headers = {"Authorization": f"Bearer {self.access_token}"}
            self.client.get("/auth/profile", headers=headers)
    
    @task(1)
    def test_token_refresh(self):
        """Test de refresh token"""
        if hasattr(self, 'refresh_token'):
            self.client.post("/auth/refresh", json={
                "refresh_token": self.refresh_token
            })
```

### **Debug et Troubleshooting**

```python
import logging

# Configuration des logs pour debug
def setup_security_logging():
    """Configuration des logs de debug pour la sécurité"""
    
    # Logger principal
    security_logger = logging.getLogger('app.security')
    security_logger.setLevel(logging.DEBUG)
    
    # Handler pour fichier
    file_handler = logging.FileHandler('security_debug.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format des logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    security_logger.addHandler(file_handler)
    security_logger.addHandler(console_handler)

# Diagnostic de sécurité
async def run_security_diagnostics():
    """Exécuter un diagnostic complet de sécurité"""
    
    diagnostics = {}
    
    # Test de connectivité Redis
    try:
        await redis_client.ping()
        diagnostics["redis"] = "OK"
    except Exception as e:
        diagnostics["redis"] = f"ERROR: {e}"
    
    # Test de configuration JWT
    try:
        test_token = await token_manager.create_access_token("test_user")
        await token_manager.validate_token(test_token)
        diagnostics["jwt"] = "OK"
    except Exception as e:
        diagnostics["jwt"] = f"ERROR: {e}"
    
    # Test de chiffrement
    try:
        test_data = "Test encryption"
        encrypted = await encryption_manager.encrypt_data(
            test_data, 
            EncryptionContext(data_classification=DataClassification.INTERNAL)
        )
        decrypted = await encryption_manager.decrypt_data(encrypted)
        if decrypted == test_data:
            diagnostics["encryption"] = "OK"
        else:
            diagnostics["encryption"] = "ERROR: Decryption mismatch"
    except Exception as e:
        diagnostics["encryption"] = f"ERROR: {e}"
    
    # Test de monitoring
    try:
        await threat_detector.process_security_event(
            event_type=EventType.SYSTEM_TEST,
            user_id="test_user",
            ip_address="127.0.0.1",
            details={"test": True}
        )
        diagnostics["monitoring"] = "OK"
    except Exception as e:
        diagnostics["monitoring"] = f"ERROR: {e}"
    
    return diagnostics
```

---

## 🛠️ Configuration Avancée

### **Configuration Environment**

```bash
# .env file pour production
# ===========================

# Redis Configuration
REDIS_URL=redis://redis-cluster:6379/0
REDIS_PASSWORD=your-redis-password
REDIS_SSL=true

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-256-bits-minimum
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# Encryption Configuration
ENCRYPTION_KEY=your-encryption-key-32-bytes-exactly
HSM_ENABLED=true
HSM_PROVIDER=SoftHSM
HSM_SLOT=0
HSM_PIN=your-hsm-pin

# Security Settings
SECURITY_RATE_LIMIT_ENABLED=true
SECURITY_MAX_LOGIN_ATTEMPTS=5
SECURITY_LOCKOUT_DURATION=900
SECURITY_PASSWORD_MIN_LENGTH=12
SECURITY_MFA_REQUIRED=true

# Monitoring Settings
SECURITY_THREAT_DETECTION_ENABLED=true
SECURITY_AUDIT_LOG_ENABLED=true
SECURITY_REAL_TIME_MONITORING=true

# Compliance Settings
COMPLIANCE_GDPR_ENABLED=true
COMPLIANCE_HIPAA_ENABLED=true
COMPLIANCE_PCI_DSS_ENABLED=false
COMPLIANCE_AUDIT_RETENTION_DAYS=2555

# External Services
GEOIP_DATABASE_PATH=/usr/share/GeoIP/GeoLite2-City.mmdb
SMTP_SERVER=smtp.example.com
SMS_PROVIDER=twilio
PUSH_NOTIFICATION_SERVICE=firebase

# Development Settings (disable in production)
SECURITY_DEBUG_MODE=false
SECURITY_ALLOW_HTTP=false
SECURITY_CSRF_ENABLED=true
```

### **Configuration Programmatique**

```python
from app.security import SecurityConfig

# Configuration enterprise complète
enterprise_config = SecurityConfig(
    # Authentication settings
    authentication={
        "max_login_attempts": 3,
        "lockout_duration": 1800,  # 30 minutes
        "enable_risk_analysis": True,
        "risk_threshold": 0.7,
        "biometric_enabled": True,
        "face_recognition_confidence": 0.95,
        "fingerprint_confidence": 0.98,
        "mfa_required_roles": ["admin", "finance"],
        "password_policy": {
            "min_length": 14,
            "max_length": 128,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special_chars": True,
            "disallow_common_passwords": True,
            "disallow_personal_info": True,
            "password_history_count": 24,
            "max_age_days": 60,
            "breach_detection": True
        }
    },
    
    # OAuth2 settings
    oauth2={
        "enable_pkce": True,
        "require_pkce": True,
        "authorization_code_lifetime": 600,
        "access_token_lifetime": 900,
        "refresh_token_lifetime": 1800000,
        "enable_token_introspection": True,
        "enable_token_revocation": True,
        "client_secret_basic": True,
        "client_secret_post": True,
        "private_key_jwt": True
    },
    
    # Session settings
    session={
        "session_lifetime": 3600,
        "idle_timeout": 1800,
        "enable_device_tracking": True,
        "max_sessions_per_user": 3,
        "enable_concurrent_sessions": False,
        "session_hijacking_detection": True,
        "device_fingerprinting": True,
        "geolocation_tracking": True
    },
    
    # Token settings
    token={
        "algorithm": "RS256",
        "issuer": "spotify-ai-agent",
        "audience": "spotify-ai-agent-api",
        "access_token_expire_minutes": 15,
        "refresh_token_expire_days": 7,
        "enable_token_rotation": True,
        "token_family_enabled": True,
        "api_key_enabled": True,
        "api_key_prefix": "spa_"
    },
    
    # Encryption settings
    encryption={
        "default_algorithm": "AES_256_GCM",
        "key_rotation_days": 90,
        "enable_hsm": True,
        "hsm_provider": "SoftHSM",
        "compression_enabled": True,
        "field_level_encryption": True,
        "data_classification_required": True
    },
    
    # Monitoring settings
    monitoring={
        "enable_real_time_detection": True,
        "threat_intelligence_enabled": True,
        "behavioral_analytics": True,
        "ml_anomaly_detection": True,
        "incident_auto_response": True,
        "alert_severity_threshold": "MEDIUM",
        "audit_log_retention_days": 2555
    },
    
    # Compliance settings
    compliance={
        "gdpr_enabled": True,
        "hipaa_enabled": True,
        "pci_dss_enabled": False,
        "sox_enabled": True,
        "iso27001_enabled": True,
        "consent_management": True,
        "data_retention_policies": True,
        "right_to_be_forgotten": True
    }
)
```

---

## 🚨 Troubleshooting

### **Problèmes Fréquents et Solutions**

#### **1. Erreur Redis Connection**

```bash
# Problème
ConnectionError: Error connecting to Redis

# Diagnostic
redis-cli ping
systemctl status redis-server

# Solutions
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Vérifier la configuration
grep -E "^(bind|port|requirepass)" /etc/redis/redis.conf
```

#### **2. Erreur JWT Invalid Signature**

```python
# Problème
JWTError: Invalid signature

# Diagnostic
async def diagnose_jwt_issue():
    config = token_manager.config
    
    # Vérifier les clés
    if not config.get('private_key'):
        print("❌ Clé privée JWT manquante")
        return False
    
    if not config.get('public_key'):
        print("❌ Clé publique JWT manquante")
        return False
    
    # Test de création/validation
    try:
        test_token = await token_manager.create_access_token("test_user")
        validation_result = await token_manager.validate_token(test_token)
        if validation_result.valid:
            print("✅ JWT configuration OK")
            return True
        else:
            print(f"❌ JWT validation failed: {validation_result.error}")
            return False
    except Exception as e:
        print(f"❌ JWT error: {e}")
        return False

# Solution
await diagnose_jwt_issue()
```

#### **3. MFA TOTP Code Always Invalid**

```python
# Diagnostic TOTP
async def diagnose_totp_issue(user_id: str):
    # Vérifier le secret
    secret = await mfa_manager.get_totp_secret(user_id)
    if not secret:
        print("❌ Secret TOTP non configuré")
        return
    
    # Vérifier la synchronisation temporelle
    import time
    import pyotp
    
    current_time = int(time.time())
    totp = pyotp.TOTP(secret)
    
    print(f"Temps actuel: {current_time}")
    print(f"Fenêtre temporelle: {current_time // 30}")
    print(f"Code attendu: {totp.now()}")
    
    # Test avec fenêtre élargie
    for window in range(-2, 3):
        test_time = current_time + (window * 30)
        test_code = pyotp.TOTP(secret).at(test_time)
        print(f"Fenêtre {window}: {test_code}")

await diagnose_totp_issue("user123")
```

#### **4. Erreur de Chiffrement**

```python
# Diagnostic du chiffrement
async def diagnose_encryption_issue():
    try:
        # Test simple
        test_data = "Test encryption data"
        context = EncryptionContext(
            data_classification=DataClassification.INTERNAL,
            purpose="diagnostic_test"
        )
        
        # Chiffrement
        encrypted = await encryption_manager.encrypt_data(test_data, context)
        print(f"✅ Chiffrement réussi: {encrypted['algorithm']}")
        
        # Déchiffrement
        decrypted = await encryption_manager.decrypt_data(encrypted, context)
        
        if decrypted == test_data:
            print("✅ Déchiffrement réussi")
        else:
            print("❌ Données déchiffrées incorrectes")
            
    except Exception as e:
        print(f"❌ Erreur de chiffrement: {e}")
        
        # Vérifier la configuration
        config = encryption_manager.config
        if not config.get('encryption_key'):
            print("❌ Clé de chiffrement manquante")
        
        if config.get('hsm_enabled') and not config.get('hsm_config'):
            print("❌ Configuration HSM manquante")

await diagnose_encryption_issue()
```

### **Logs et Monitoring**

```python
# Activer les logs détaillés pour debug
import logging

def enable_debug_logging():
    """Activer tous les logs de debug"""
    
    loggers = [
        'app.security',
        'app.security.auth',
        'app.security.auth.authenticator',
        'app.security.auth.oauth2',
        'app.security.auth.session',
        'app.security.auth.password',
        'app.security.auth.token',
        'app.security.encryption',
        'app.security.monitoring'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Handler console
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)

# Analyser les logs d'erreur
async def analyze_security_errors():
    """Analyser les erreurs récentes"""
    
    # Récupérer les erreurs des dernières 24h
    errors = await security_manager.get_error_logs(
        time_range=86400,
        severity=["ERROR", "CRITICAL"]
    )
    
    # Grouper par type d'erreur
    error_types = {}
    for error in errors:
        error_type = error.get('error_type', 'UNKNOWN')
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    # Afficher le résumé
    for error_type, error_list in error_types.items():
        print(f"{error_type}: {len(error_list)} occurrences")
        if len(error_list) > 0:
            print(f"  Dernière occurrence: {error_list[-1]['timestamp']}")
            print(f"  Message: {error_list[-1]['message']}")
```

### **Health Checks**

```python
async def security_health_check():
    """Vérification complète de la santé du système de sécurité"""
    
    health_status = {
        "overall": "UNKNOWN",
        "components": {}
    }
    
    # Check Redis
    try:
        await redis_client.ping()
        health_status["components"]["redis"] = "HEALTHY"
    except Exception as e:
        health_status["components"]["redis"] = f"UNHEALTHY: {e}"
    
    # Check Authentication
    try:
        await auth_manager.health_check()
        health_status["components"]["authentication"] = "HEALTHY"
    except Exception as e:
        health_status["components"]["authentication"] = f"UNHEALTHY: {e}"
    
    # Check Encryption
    try:
        await encryption_manager.health_check()
        health_status["components"]["encryption"] = "HEALTHY"
    except Exception as e:
        health_status["components"]["encryption"] = f"UNHEALTHY: {e}"
    
    # Check Monitoring
    try:
        await threat_detector.health_check()
        health_status["components"]["monitoring"] = "HEALTHY"
    except Exception as e:
        health_status["components"]["monitoring"] = f"UNHEALTHY: {e}"
    
    # Déterminer le statut global
    unhealthy_components = [
        comp for comp, status in health_status["components"].items()
        if not status.startswith("HEALTHY")
    ]
    
    if len(unhealthy_components) == 0:
        health_status["overall"] = "HEALTHY"
    elif len(unhealthy_components) < len(health_status["components"]) / 2:
        health_status["overall"] = "DEGRADED"
    else:
        health_status["overall"] = "UNHEALTHY"
    
    return health_status
```

---

## 📚 Ressources et Références

### **Documentation Officielle**
- 📖 **[OWASP Security Guidelines](https://owasp.org/)**
- 📖 **[NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)**
- 📖 **[OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)**
- 📖 **[OpenID Connect](https://openid.net/connect/)**
- 📖 **[JWT RFC 7519](https://tools.ietf.org/html/rfc7519)**

### **Standards de Conformité**
- ⚖️ **[GDPR](https://gdpr.eu/)** - General Data Protection Regulation
- 🏥 **[HIPAA](https://www.hhs.gov/hipaa/)** - Health Insurance Portability and Accountability Act
- 💳 **[PCI DSS](https://www.pcisecuritystandards.org/)** - Payment Card Industry Data Security Standard
- 📊 **[SOX](https://www.sox-online.com/)** - Sarbanes-Oxley Act
- 🛡️ **[ISO 27001](https://www.iso.org/isoiec-27001-information-security.html)** - Information Security Management

### **Outils de Sécurité**
- 🔍 **[Bandit](https://bandit.readthedocs.io/)** - Security linter for Python
- 🛡️ **[Safety](https://pyup.io/safety/)** - Dependency vulnerability scanner
- 🔐 **[HashiCorp Vault](https://www.vaultproject.io/)** - Secrets management
- 📊 **[ELK Stack](https://www.elastic.co/elk-stack)** - Log analysis and SIEM

---

## 🤝 Support et Communauté

### **Obtenir de l'Aide**

1. **📧 Contact Support**: security@spotify-ai-agent.com
2. **💬 Discord**: [Rejoindre notre serveur](https://discord.gg/spotify-ai-agent)
3. **📱 Slack**: [Workspace de développement](https://spotify-ai-agent.slack.com)
4. **📋 GitHub Issues**: [Rapporter un bug](https://github.com/spotify-ai-agent/security/issues)

### **Contribuer**

```bash
# Fork le repository
git clone https://github.com/your-username/spotify-ai-agent.git

# Créer une branch feature
git checkout -b feature/security-improvement

# Développer et tester
pytest tests/security/

# Commit et push
git commit -am "Add security feature"
git push origin feature/security-improvement

# Créer une Pull Request
```

---

## 👨‍💻 Équipe de Développement

🎖️ **Développé par l'Équipe d'Experts Enterprise Security**

- **Lead Developer + AI Architect** - Architecture et conception
- **Senior Backend Developer** - Implémentation core
- **ML Engineer** - Analyse comportementale et détection d'anomalies
- **Security Specialist** - Audit et conformité
- **DBA** - Optimisation et chiffrement des données
- **Microservices Architect** - Intégration et scalabilité

---

## 📄 License et Legal

Ce module est sous licence MIT. Voir le fichier [LICENSE](../../../LICENSE) pour plus de détails.

**⚠️ Note Légale**: Ce guide est fourni à des fins éducatives. L'implémentation en production nécessite une validation par des experts en sécurité et une adaptation aux exigences spécifiques de votre organisation.

---

*Guide d'Utilisation Complet - Module de Sécurité Enterprise*  
*Spotify AI Agent v1.0.0 | Dernière mise à jour: Juillet 2025*

---

**🎉 Félicitations ! Vous êtes maintenant prêt à utiliser le Module de Sécurité Enterprise le plus avancé du marché ! 🚀**
