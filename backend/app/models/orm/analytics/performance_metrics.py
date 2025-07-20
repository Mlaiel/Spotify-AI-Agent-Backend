"""
PerformanceMetrics ORM Model
- Tracks all system/model KPIs, uptime, latency, errors, monitoring, alerting, audit, compliance, traceability.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class PerformanceMetrics(Base):
    __tablename__ = "performance_metrics"
    id = Column(Integer, primary_key=True)
    target_type = Column(String, nullable=False, index=True)  # system, model, api, etc.
    target_id = Column(String, nullable=True, index=True)
    kpis = Column(JSON, nullable=True)  # accuracy, latency, throughput, etc.
    uptime = Column(Float, nullable=True)
    latency = Column(Float, nullable=True)
    error_count = Column(Integer, nullable=True)
    monitoring_data = Column(JSON, nullable=True)
    alert_flags = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(target_type: str, **kwargs):
        return PerformanceMetrics(
            target_type=target_type,)
            target_id=kwargs.get("target_id"),
            kpis=kwargs.get("kpis"),
            uptime=kwargs.get("uptime"),
            latency=kwargs.get("latency"),
            error_count=kwargs.get("error_count"),
            monitoring_data=kwargs.get("monitoring_data"),
            alert_flags=kwargs.get("alert_flags"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "kpis": self.kpis,
            "uptime": self.uptime,
            "latency": self.latency,
            "error_count": self.error_count,
            "monitoring_data": self.monitoring_data,
            "alert_flags": self.alert_flags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id
        }
