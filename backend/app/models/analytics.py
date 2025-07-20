"""
Analytics Business Model
- Tracks all analytics data (engagement, reach, KPIs, revenue, trend, user analytics), audit, compliance, traceability, multi-tenancy.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Analytics(Base):
    __tablename__ = "analytics"
    id = Column(Integer, primary_key=True)
    target_type = Column(String, nullable=False, index=True)  # content, user, revenue, etc.
    target_id = Column(String, nullable=True, index=True)
    metrics = Column(JSON, nullable=True)
    kpis = Column(JSON, nullable=True)
    revenue = Column(Float, nullable=True)
    trend_data = Column(JSON, nullable=True)
    user_analytics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    audit_log = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    trace_id = Column(String, nullable=True, index=True)
    meta_data = Column(JSON, nullable=True)  # renommé de 'metadata' à 'meta_data'

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.now(datetime.timezone.utc)
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        }]

    @staticmethod
    def create(target_type: str, **kwargs):
        return Analytics(
            target_type=target_type,
            target_id=kwargs.get("target_id"),
            metrics=kwargs.get("metrics"),
            kpis=kwargs.get("kpis"),
            revenue=kwargs.get("revenue"),
            trend_data=kwargs.get("trend_data"),
            user_analytics=kwargs.get("user_analytics"),
            tenant_id=kwargs.get("tenant_id"),
            audit_log=kwargs.get("audit_log"),
            compliance_flags=kwargs.get("compliance_flags"),
            trace_id=kwargs.get("trace_id"),
            meta_data=kwargs.get("meta_data")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "metrics": self.metrics,
            "kpis": self.kpis,
            "revenue": self.revenue,
            "trend_data": self.trend_data,
            "user_analytics": self.user_analytics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "audit_log": self.audit_log,
            "compliance_flags": self.compliance_flags,
            "trace_id": self.trace_id,
            "meta_data": self.meta_data
        }
