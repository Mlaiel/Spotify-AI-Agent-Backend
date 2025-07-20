"""
TrendData ORM Model
- Tracks all time series, trend detection, anomaly detection, forecasting, data lineage, audit, compliance, traceability.
- Supports advanced analytics, soft-delete, GDPR/DSGVO, security, logging, multi-tenancy.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TrendData(Base):
    __tablename__ = "trend_data"
    id = Column(Integer, primary_key=True)
    target_type = Column(String, nullable=False, index=True)  # content, user, revenue, etc.
    target_id = Column(String, nullable=True, index=True)
    time_series = Column(JSON, nullable=True)
    trend_metrics = Column(JSON, nullable=True)
    anomaly_flags = Column(JSON, nullable=True)
    forecast = Column(JSON, nullable=True)
    data_lineage = Column(JSON, nullable=True)
    compliance_flags = Column(JSON, nullable=True)
    audit_log = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted = Column(Boolean, default=False, index=True)
    deleted_at = Column(DateTime, nullable=True)
    tenant_id = Column(String, nullable=True, index=True)
    trace_id = Column(String, nullable=True, index=True)

    def soft_delete(self, user_id: int):
        self.deleted = True
        self.deleted_at = datetime.utcnow()
        self.audit_log = (self.audit_log or []) + [{
            "action": "soft_delete", "user_id": user_id, "timestamp": datetime.utcnow().isoformat()
        }]

    @staticmethod
    def create(target_type: str, **kwargs):
        return TrendData(
            target_type=target_type,)
            target_id=kwargs.get("target_id"),
            time_series=kwargs.get("time_series"),
            trend_metrics=kwargs.get("trend_metrics"),
            anomaly_flags=kwargs.get("anomaly_flags"),
            forecast=kwargs.get("forecast"),
            data_lineage=kwargs.get("data_lineage"),
            compliance_flags=kwargs.get("compliance_flags"),
            audit_log=kwargs.get("audit_log"),
            tenant_id=kwargs.get("tenant_id"),
            trace_id=kwargs.get("trace_id")
        )

    def to_dict(self):
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "time_series": self.time_series,
            "trend_metrics": self.trend_metrics,
            "anomaly_flags": self.anomaly_flags,
            "forecast": self.forecast,
            "data_lineage": self.data_lineage,
            "compliance_flags": self.compliance_flags,
            "audit_log": self.audit_log,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": self.deleted,
            "deleted_at": self.deleted_at,
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id
        }
