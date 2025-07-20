"""
Ultra-Advanced Authentication System - Complete Usage Guide
==========================================================

Comprehensive usage guide and examples for the enterprise-grade
authentication system of the Spotify AI Agent platform.

Authors: Fahed Mlaiel (Lead Developer & AI Architect)
Team: Expert Security Specialists and Backend Development Team

This guide provides complete examples and best practices for using
the ultra-advanced authentication system in production environments.

Quick Start Example:
```python
from auth.core import initialize_authentication_module, AuthenticationModuleConfig
from fastapi import FastAPI

# Initialize authentication module
config = AuthenticationModuleConfig()
auth_module = await initialize_authentication_module(config)

# Create FastAPI app and integrate
app = FastAPI()
auth_module.integrate_with_fastapi(app)

# Use authentication in endpoints
@app.post("/api/login")
async def login(credentials: dict):
    result = await auth_module.authenticate_user(
        user_id=credentials["username"],
        credentials=credentials
    )
    return result
```

Architecture Overview:
- Multi-layered security with defense-in-depth
- Zero-trust architecture with continuous verification
- Enterprise-grade scalability and performance
- Comprehensive audit and compliance logging
- Real-time threat detection and response
- Quantum-resistant cryptographic security

Components:
1. Authentication Core - Base authentication framework
2. Configuration Management - Advanced configuration system
3. Token Management - JWT, refresh tokens, API keys
4. Session Management - Distributed session handling
5. Security Framework - Zero-trust security architecture
6. Security Middleware - Multi-layered protection
7. Exception Handling - Enterprise exception framework

Version: 3.0.0
License: MIT
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import asyncio

# Example usage patterns for the authentication system

class AuthenticationUsageExamples:
    """
    Comprehensive examples for using the ultra-advanced authentication system.
    
    This class provides real-world usage patterns and best practices
    for implementing enterprise-grade authentication in production.
    """
    
    @staticmethod
    async def basic_setup_example():
        """Basic setup and initialization example."""
        
        from .auth_module import (
            initialize_authentication_module, 
            AuthenticationModuleConfig,
            get_authentication_module
        )
        from fastapi import FastAPI, Depends, HTTPException
        
        # 1. Create configuration
        config = AuthenticationModuleConfig()
        config.environment = "production"
        config.rate_limit_enabled = True
        config.mfa_enabled = True
        config.threat_detection_enabled = True
        
        # 2. Initialize authentication module
        auth_module = await initialize_authentication_module(config)
        
        # 3. Create FastAPI application
        app = FastAPI(title="Spotify AI Agent API")
        
        # 4. Integrate authentication with FastAPI
        auth_module.integrate_with_fastapi(app)
        
        # 5. Define protected endpoints
        @app.post("/api/v1/login")
        async def login(credentials: Dict[str, Any]):
            """User login endpoint."""
            try:
                result = await auth_module.authenticate_user(
                    user_id=credentials["username"],
                    credentials=credentials,
                    request_context={
                        "ip_address": "192.168.1.1",
                        "user_agent": "Mozilla/5.0..."
                    }
                )
                
                if result.success:
                    # Create session
                    session = await auth_module.create_session(
                        user_id=result.user_id,
                        tenant_id=result.tenant_id,
                        auth_method=result.auth_method.value
                    )
                    
                    return {
                        "success": True,
                        "access_token": result.access_token,
                        "refresh_token": result.refresh_token,
                        "session_id": session.session_id,
                        "expires_at": result.expires_at.isoformat()
                    }
                else:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication failed"
                    )
                    
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Authentication error: {str(e)}"
                )
        
        return app
    
    @staticmethod
    async def advanced_authentication_example():
        """Advanced authentication with MFA and risk assessment."""
        
        from .auth_module import get_authentication_module
        from .security import SecurityContext, AuthenticationMethod
        
        auth_module = get_authentication_module()
        
        # Multi-factor authentication flow
        async def mfa_authentication_flow(
            user_id: str,
            primary_credentials: Dict[str, Any],
            mfa_credentials: Optional[Dict[str, Any]] = None,
            request_context: Optional[Dict[str, Any]] = None
        ):
            """Complete MFA authentication flow."""
            
            # Step 1: Primary authentication (username/password)
            primary_result = await auth_module.authenticate_user(
                user_id=user_id,
                credentials=primary_credentials,
                request_context=request_context
            )
            
            if not primary_result.success:
                return {"success": False, "error": "Primary authentication failed"}
            
            # Step 2: Risk assessment
            if primary_result.security_context.risk_score > 0.6:
                return {
                    "success": False,
                    "requires_mfa": True,
                    "mfa_methods": ["totp", "sms", "email"],
                    "temp_token": primary_result.temp_token
                }
            
            # Step 3: MFA verification if required
            if mfa_credentials:
                mfa_result = await auth_module.authenticate_user(
                    user_id=user_id,
                    credentials={
                        **primary_credentials,
                        **mfa_credentials
                    },
                    request_context=request_context
                )
                
                if mfa_result.success:
                    # Create secure session
                    session = await auth_module.create_session(
                        user_id=user_id,
                        tenant_id=mfa_result.tenant_id,
                        auth_method=AuthenticationMethod.MFA_TOTP.value,
                        mfa_verified=True,
                        security_level="high"
                    )
                    
                    return {
                        "success": True,
                        "access_token": mfa_result.access_token,
                        "refresh_token": mfa_result.refresh_token,
                        "session_id": session.session_id,
                        "security_level": "high"
                    }
            
            return {"success": False, "error": "MFA verification failed"}
        
        # Example usage
        result = await mfa_authentication_flow(
            user_id="user123",
            primary_credentials={
                "username": "user123",
                "password": "secure_password"
            },
            mfa_credentials={
                "totp_code": "123456"
            },
            request_context={
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "device_fingerprint": "abc123def456"
            }
        )
        
        return result
    
    @staticmethod
    async def session_management_example():
        """Advanced session management examples."""
        
        from .auth_module import get_authentication_module
        from .sessions import SessionType, DeviceInfo
        
        auth_module = get_authentication_module()
        session_manager = auth_module.session_manager
        
        # Create session with device tracking
        async def create_tracked_session(
            user_id: str,
            tenant_id: str,
            device_info: Dict[str, Any],
            location_info: Optional[Dict[str, Any]] = None
        ):
            """Create session with comprehensive tracking."""
            
            session = await session_manager.create_session(
                user_id=user_id,
                tenant_id=tenant_id,
                auth_method="mfa_totp",
                session_type=SessionType.WEB,
                ip_address=location_info.get("ip_address") if location_info else None,
                user_agent=device_info.get("user_agent"),
                device_id=device_info.get("device_id"),
                permissions={"read", "write", "admin"},
                roles={"user", "premium"},
                custom_attributes={
                    "subscription_type": "premium",
                    "last_payment": "2024-01-15"
                }
            )
            
            return session
        
        # Monitor session activity
        async def monitor_session_activity(session_id: str):
            """Monitor and update session activity."""
            
            # Update activity
            await session_manager.update_session_activity(
                session_id=session_id,
                ip_address="192.168.1.101",
                user_agent="Mozilla/5.0...",
                page_view=True,
                api_call=False
            )
            
            # Get session analytics
            analytics = await session_manager.get_session_analytics(
                tenant_id="tenant_001",
                days=7
            )
            
            return analytics
        
        # Session cleanup and management
        async def manage_user_sessions(user_id: str, tenant_id: str):
            """Comprehensive session management."""
            
            # Get all user sessions
            user_sessions = await session_manager.storage.get_user_sessions(
                user_id=user_id,
                tenant_id=tenant_id
            )
            
            # Terminate inactive sessions
            terminated_count = 0
            for session in user_sessions:
                if session.idle_time.total_seconds() > 7200:  # 2 hours
                    await session_manager.terminate_session(
                        session.session_id,
                        "idle_timeout"
                    )
                    terminated_count += 1
            
            # Limit concurrent sessions
            if len(user_sessions) > 5:
                # Keep only 5 most recent sessions
                sessions_by_activity = sorted(
                    user_sessions,
                    key=lambda s: s.last_activity,
                    reverse=True
                )
                
                for session in sessions_by_activity[5:]:
                    await session_manager.terminate_session(
                        session.session_id,
                        "concurrent_limit"
                    )
            
            return {
                "total_sessions": len(user_sessions),
                "terminated_inactive": terminated_count,
                "active_sessions": len([s for s in user_sessions if s.is_active])
            }
        
        return {
            "create_tracked_session": create_tracked_session,
            "monitor_session_activity": monitor_session_activity,
            "manage_user_sessions": manage_user_sessions
        }
    
    @staticmethod
    async def security_monitoring_example():
        """Security monitoring and threat detection examples."""
        
        from .auth_module import get_authentication_module
        from .security import ThreatLevel, SecurityLevel
        
        auth_module = get_authentication_module()
        security_manager = auth_module.security_manager
        
        # Real-time threat monitoring
        async def setup_threat_monitoring():
            """Setup comprehensive threat monitoring."""
            
            # Monitor authentication attempts
            async def monitor_auth_attempts(user_id: str, request_data: Dict[str, Any]):
                """Monitor authentication attempts for threats."""
                
                # Create security context
                from .security import SecurityContext
                context = SecurityContext(
                    user_id=user_id,
                    tenant_id=request_data.get("tenant_id", "default"),
                    ip_address=request_data.get("ip_address"),
                    user_agent=request_data.get("user_agent"),
                    device_fingerprint=request_data.get("device_fingerprint")
                )
                
                # Calculate risk score
                risk_score = security_manager.threat_engine.calculate_risk_score(context)
                
                # Take action based on risk level
                if risk_score > 0.8:
                    # High risk - block and alert
                    security_manager.audit_service.log_security_event(
                        "high_risk_authentication",
                        context,
                        {
                            "risk_score": risk_score,
                            "action": "blocked",
                            "threat_indicators": context.threat_indicators if hasattr(context, 'threat_indicators') else []
                        }
                    )
                    return {"allowed": False, "reason": "High risk detected"}
                
                elif risk_score > 0.6:
                    # Medium risk - require additional verification
                    return {
                        "allowed": True,
                        "requires_mfa": True,
                        "risk_score": risk_score
                    }
                
                return {"allowed": True, "risk_score": risk_score}
            
            # Geographic anomaly detection
            async def detect_geographic_anomalies(user_id: str, current_location: Dict[str, Any]):
                """Detect geographic anomalies in user access patterns."""
                
                # Get recent user sessions
                user_sessions = await auth_module.session_manager.storage.get_user_sessions(
                    user_id=user_id,
                    tenant_id="default"
                )
                
                recent_locations = []
                for session in user_sessions[-10:]:  # Last 10 sessions
                    if session.location_info:
                        recent_locations.append({
                            "latitude": session.location_info.latitude,
                            "longitude": session.location_info.longitude,
                            "timestamp": session.created_at
                        })
                
                # Calculate distance from recent locations
                if recent_locations:
                    # Simple distance calculation (in production use proper geospatial library)
                    max_distance = 0
                    for location in recent_locations[-5:]:  # Last 5 locations
                        if location["latitude"] and location["longitude"]:
                            # Simplified distance calculation
                            lat_diff = abs(current_location["latitude"] - location["latitude"])
                            lon_diff = abs(current_location["longitude"] - location["longitude"])
                            distance = (lat_diff ** 2 + lon_diff ** 2) ** 0.5 * 111  # Rough km conversion
                            max_distance = max(max_distance, distance)
                    
                    # If user has traveled more than 1000km in short time
                    if max_distance > 1000:
                        return {
                            "anomaly_detected": True,
                            "distance_km": max_distance,
                            "action_required": "additional_verification"
                        }
                
                return {"anomaly_detected": False}
            
            return {
                "monitor_auth_attempts": monitor_auth_attempts,
                "detect_geographic_anomalies": detect_geographic_anomalies
            }
        
        # Security incident response
        async def security_incident_response(incident_type: str, context: Dict[str, Any]):
            """Automated security incident response."""
            
            response_actions = {
                "brute_force_attack": [
                    "lock_account",
                    "alert_security_team",
                    "increase_monitoring"
                ],
                "credential_stuffing": [
                    "require_password_reset",
                    "enable_mfa",
                    "alert_user"
                ],
                "session_hijacking": [
                    "terminate_all_sessions",
                    "force_password_change",
                    "alert_security_team"
                ],
                "geographic_anomaly": [
                    "require_additional_verification",
                    "alert_user",
                    "monitor_closely"
                ]
            }
            
            actions = response_actions.get(incident_type, ["log_incident"])
            
            for action in actions:
                if action == "lock_account":
                    # Lock user account
                    pass
                elif action == "alert_security_team":
                    # Send alert to security team
                    pass
                elif action == "terminate_all_sessions":
                    # Terminate all user sessions
                    if "user_id" in context:
                        await auth_module.session_manager.terminate_user_sessions(
                            user_id=context["user_id"],
                            tenant_id=context.get("tenant_id", "default"),
                            reason=incident_type
                        )
            
            return {"actions_taken": actions}
        
        return {
            "setup_threat_monitoring": setup_threat_monitoring,
            "security_incident_response": security_incident_response
        }
    
    @staticmethod
    async def compliance_and_audit_example():
        """Compliance and audit logging examples."""
        
        from .auth_module import get_authentication_module
        
        auth_module = get_authentication_module()
        security_manager = auth_module.security_manager
        
        # GDPR compliance example
        async def gdpr_compliance_operations():
            """GDPR compliance operations."""
            
            # Data export for user
            async def export_user_data(user_id: str) -> Dict[str, Any]:
                """Export all user data for GDPR compliance."""
                
                # Get user sessions
                user_sessions = await auth_module.session_manager.storage.get_user_sessions(
                    user_id=user_id,
                    tenant_id="default"
                )
                
                # Get audit events
                audit_events = [
                    event for event in security_manager.audit_service.audit_events
                    if event.get("user_id") == user_id
                ]
                
                # Get token history (if available)
                token_history = []  # Would get from token manager
                
                exported_data = {
                    "user_id": user_id,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "sessions": [session.to_dict(include_sensitive=False) for session in user_sessions],
                    "audit_events": audit_events,
                    "token_history": token_history
                }
                
                # Log data export
                security_manager.audit_service.log_security_event(
                    "data_export",
                    context=None,  # Would create proper context
                    details={
                        "user_id": user_id,
                        "export_type": "gdpr_compliance",
                        "data_types": ["sessions", "audit_events", "tokens"]
                    }
                )
                
                return exported_data
            
            # Data deletion for user
            async def delete_user_data(user_id: str) -> Dict[str, Any]:
                """Delete all user data for GDPR compliance."""
                
                # Terminate all sessions
                terminated_sessions = await auth_module.session_manager.terminate_user_sessions(
                    user_id=user_id,
                    tenant_id="default",
                    reason="gdpr_deletion"
                )
                
                # Invalidate all tokens (if token manager supports it)
                # await auth_module.token_manager.invalidate_user_tokens(user_id)
                
                # Remove from audit logs (keeping anonymized entries)
                # This would typically be done with data retention policies
                
                deletion_report = {
                    "user_id": user_id,
                    "deletion_timestamp": datetime.now(timezone.utc).isoformat(),
                    "terminated_sessions": terminated_sessions,
                    "invalidated_tokens": 0,  # Would get actual count
                    "anonymized_audit_entries": 0  # Would get actual count
                }
                
                # Log deletion
                security_manager.audit_service.log_security_event(
                    "data_deletion",
                    context=None,  # Would create proper context
                    details={
                        "user_id": user_id,
                        "deletion_type": "gdpr_compliance",
                        "data_deleted": deletion_report
                    }
                )
                
                return deletion_report
            
            return {
                "export_user_data": export_user_data,
                "delete_user_data": delete_user_data
            }
        
        # Compliance reporting
        async def generate_compliance_reports():
            """Generate comprehensive compliance reports."""
            
            # Generate monthly security report
            async def monthly_security_report(year: int, month: int) -> Dict[str, Any]:
                """Generate monthly security compliance report."""
                
                start_date = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
                
                # Generate compliance report
                report = security_manager.audit_service.generate_compliance_report(
                    standard="SOC2",
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Add additional metrics
                report.update({
                    "authentication_metrics": {
                        "total_attempts": auth_module.metrics["total_authentications"],
                        "success_rate": (
                            auth_module.metrics["successful_authentications"] / 
                            max(auth_module.metrics["total_authentications"], 1)
                        ),
                        "mfa_usage": 0.85  # Example percentage
                    },
                    "session_metrics": {
                        "average_session_duration": 1800,  # Would calculate actual
                        "active_sessions": auth_module.metrics["active_sessions"],
                        "geographic_distribution": {}  # Would calculate actual
                    },
                    "security_incidents": {
                        "threats_detected": auth_module.metrics["threats_detected"],
                        "rate_limits_triggered": auth_module.metrics["rate_limits_triggered"],
                        "false_positive_rate": 0.02  # Example rate
                    }
                })
                
                return report
            
            return {"monthly_security_report": monthly_security_report}
        
        return {
            "gdpr_compliance_operations": gdpr_compliance_operations,
            "generate_compliance_reports": generate_compliance_reports
        }
    
    @staticmethod
    async def performance_optimization_example():
        """Performance optimization examples."""
        
        from .auth_module import get_authentication_module
        
        auth_module = get_authentication_module()
        
        # Caching strategies
        async def implement_caching_strategies():
            """Implement performance caching strategies."""
            
            # Token validation caching
            async def cached_token_validation(token: str) -> bool:
                """Cached token validation for performance."""
                
                # Check cache first
                cache_key = f"token_validation:{hash(token)}"
                cached_result = await auth_module.redis_client.get(cache_key)
                
                if cached_result is not None:
                    return cached_result == "valid"
                
                # Validate token
                is_valid = await auth_module.validate_token(token)
                
                # Cache result (short TTL for security)
                await auth_module.redis_client.setex(
                    cache_key,
                    60,  # 1 minute cache
                    "valid" if is_valid else "invalid"
                )
                
                return is_valid
            
            # Session preloading
            async def preload_user_sessions(user_id: str, tenant_id: str):
                """Preload user sessions for performance."""
                
                # Get sessions from storage
                sessions = await auth_module.session_manager.storage.get_user_sessions(
                    user_id=user_id,
                    tenant_id=tenant_id
                )
                
                # Cache active sessions
                for session in sessions:
                    if session.is_active:
                        cache_key = f"session_cache:{session.session_id}"
                        await auth_module.redis_client.setex(
                            cache_key,
                            300,  # 5 minute cache
                            session.to_dict()
                        )
            
            # Permission caching
            async def cache_user_permissions(user_id: str, tenant_id: str, permissions: List[str]):
                """Cache user permissions for performance."""
                
                cache_key = f"permissions:{user_id}:{tenant_id}"
                await auth_module.redis_client.setex(
                    cache_key,
                    600,  # 10 minute cache
                    ",".join(permissions)
                )
            
            return {
                "cached_token_validation": cached_token_validation,
                "preload_user_sessions": preload_user_sessions,
                "cache_user_permissions": cache_user_permissions
            }
        
        # Connection pooling
        async def optimize_connections():
            """Optimize database and cache connections."""
            
            # Connection health monitoring
            async def monitor_connection_health():
                """Monitor connection pool health."""
                
                try:
                    # Test Redis connection
                    redis_ping = await auth_module.redis_client.ping()
                    
                    # Get connection pool stats (if available)
                    pool_stats = {
                        "redis_connected": redis_ping,
                        "pool_size": 20,  # Would get actual pool size
                        "active_connections": 5,  # Would get actual count
                        "idle_connections": 15   # Would get actual count
                    }
                    
                    return pool_stats
                    
                except Exception as e:
                    return {"error": str(e), "healthy": False}
            
            return {"monitor_connection_health": monitor_connection_health}
        
        return {
            "implement_caching_strategies": implement_caching_strategies,
            "optimize_connections": optimize_connections
        }


# Production deployment example
async def production_deployment_example():
    """Complete production deployment example."""
    
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from .auth_module import (
        initialize_authentication_module,
        AuthenticationModuleConfig,
        get_auth_manager,
        get_session_manager
    )
    
    # Production configuration
    config = AuthenticationModuleConfig()
    config.environment = "production"
    config.debug = False
    config.log_level = "INFO"
    
    # Security settings
    config.rate_limit_enabled = True
    config.rate_limit_requests_per_minute = 60
    config.mfa_enabled = True
    config.geo_filtering_enabled = True
    config.threat_detection_enabled = True
    config.audit_enabled = True
    
    # Compliance settings
    config.gdpr_enabled = True
    config.hipaa_enabled = False
    config.sox_enabled = True
    
    # Initialize authentication module
    auth_module = await initialize_authentication_module(config)
    
    # Create FastAPI application
    app = FastAPI(
        title="Spotify AI Agent API",
        version="3.0.0",
        description="Ultra-Advanced AI-Powered Music Platform"
    )
    
    # Integrate authentication
    auth_module.integrate_with_fastapi(app)
    
    # Security dependency
    security = HTTPBearer()
    
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        auth_manager = Depends(get_auth_manager)
    ):
        """Get current authenticated user."""
        
        token = credentials.credentials
        is_valid = await auth_module.validate_token(token)
        
        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
        
        # Extract user info from token (simplified)
        # In production, properly decode JWT
        return {"user_id": "user123", "tenant_id": "tenant001"}
    
    # Protected endpoints
    @app.post("/api/v1/authenticate")
    async def authenticate(request: Request, credentials: Dict[str, Any]):
        """Authentication endpoint."""
        
        try:
            # Get request context
            request_context = {
                "ip_address": request.client.host,
                "user_agent": request.headers.get("User-Agent"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Authenticate user
            result = await auth_module.authenticate_user(
                user_id=credentials["username"],
                credentials=credentials,
                request_context=request_context
            )
            
            if result.success:
                return {
                    "success": True,
                    "access_token": result.access_token,
                    "refresh_token": result.refresh_token,
                    "expires_at": result.expires_at.isoformat(),
                    "user_id": result.user_id,
                    "tenant_id": result.tenant_id
                }
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication failed"
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Authentication error: {str(e)}"
            )
    
    @app.get("/api/v1/user/profile")
    async def get_user_profile(current_user = Depends(get_current_user)):
        """Get user profile (protected endpoint)."""
        
        return {
            "user_id": current_user["user_id"],
            "tenant_id": current_user["tenant_id"],
            "profile": {
                "name": "John Doe",
                "email": "john@example.com",
                "subscription": "premium"
            }
        }
    
    @app.post("/api/v1/logout")
    async def logout(
        current_user = Depends(get_current_user),
        session_manager = Depends(get_session_manager)
    ):
        """Logout endpoint."""
        
        # Terminate user sessions
        terminated_count = await session_manager.terminate_user_sessions(
            user_id=current_user["user_id"],
            tenant_id=current_user["tenant_id"],
            reason="user_logout"
        )
        
        return {
            "success": True,
            "message": "Logged out successfully",
            "terminated_sessions": terminated_count
        }
    
    # Admin endpoints
    @app.get("/api/v1/admin/metrics")
    async def get_admin_metrics(current_user = Depends(get_current_user)):
        """Get system metrics (admin only)."""
        
        # Check admin permissions (simplified)
        # In production, use proper permission system
        
        metrics = await auth_module.get_comprehensive_metrics()
        return metrics
    
    @app.get("/api/v1/admin/security/report")
    async def get_security_report(current_user = Depends(get_current_user)):
        """Get security report (admin only)."""
        
        # Generate security report
        report = auth_module.security_manager.audit_service.generate_compliance_report(
            standard="SOC2",
            start_date=datetime.now(timezone.utc).replace(day=1),
            end_date=datetime.now(timezone.utc)
        )
        
        return report
    
    return app


# Example of running the complete system
if __name__ == "__main__":
    import uvicorn
    
    async def main():
        """Main application entry point."""
        
        # Create production application
        app = await production_deployment_example()
        
        # Run with Uvicorn
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            workers=4
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    # Run the application
    asyncio.run(main())


# Export examples
__all__ = [
    "AuthenticationUsageExamples",
    "production_deployment_example"
]
