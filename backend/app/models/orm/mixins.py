"""
Mixins for ORM models
- Provide reusable features: Versioning, Traceability, UserAttribution, Explainability, DataLineage, Compliance, Logging, Monitoring.
- Combine with BaseModel for advanced, production-ready, business-aligned ORM models.
"""
from sqlalchemy import Column, String, Integer, DateTime, JSON, Boolean
from datetime import datetime, timezone

class VersioningMixin:
    version = Column(String, default="1.0")

class TraceabilityMixin:
    trace_id = Column(String, nullable=True, index=True)

class UserAttributionMixin:
    created_by = Column(Integer, nullable=True, index=True)
    updated_by = Column(Integer, nullable=True, index=True)

class ExplainabilityMixin:
    explainability = Column(JSON, nullable=True)

class DataLineageMixin:
    data_lineage = Column(JSON, nullable=True)

class ComplianceMixin:
    compliance_flags = Column(JSON, nullable=True)

class LoggingMixin:
    audit_log = Column(JSON, nullable=True)

class MonitoringMixin:
    monitoring_data = Column(JSON, nullable=True)

class SoftDeleteMixin:
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        if hasattr(self, 'audit_log'):
            self.audit_log = (self.audit_log or []) + [{
                "action": "soft_delete", "user_id": user_id, "timestamp": datetime.now(timezone.utc).isoformat()
            }]
